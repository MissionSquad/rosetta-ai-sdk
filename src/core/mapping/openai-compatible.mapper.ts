/**
 * Mapper for Custom Providers Adhering to the OpenAI API Specification.
 *
 * This mapper leverages the official OpenAI SDK and common mapping utilities
 * to interact with providers that expose an OpenAI-compatible API endpoint.
 * It simplifies the process of adding custom providers like Novita, LM Studio, etc.
 */
import OpenAI from 'openai'
import { Stream } from 'openai/streaming'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  CustomProviderConfig,
  Provider, // Import Provider enum for common functions
  ProviderKey,
  RosettaMessage,
  RosettaTool,
  EmbedParams,
  EmbedResult
} from '../../types'
import { BaseCustomMapper } from './base.custom.mapper'
import {
  mapFromOpenAIResponse,
  mapOpenAIStream,
  mapContentForOpenAIRole,
  mapRoleToOpenAI,
  wrapOpenAIError
} from './openai.common' // Reuse common OpenAI mapping logic
import { mapBaseToolChoice, mapToOpenAIResponseFormat } from '../mapping/common.utils'
import { MappingError, RosettaAIError, ConfigurationError, UnsupportedFeatureError } from '../../errors'
import { mapToOpenAIEmbedParams, mapFromOpenAIEmbedResponse } from './openai.embed.mapper'

export class OpenAICompatibleMapper extends BaseCustomMapper {
  private openaiClient: OpenAI

  constructor(config: CustomProviderConfig) {
    super(config) // Pass config to base class (sets this.provider and this.config)

    // Initialize the OpenAI client using the custom provider's specifics
    this.openaiClient = new OpenAI({
      apiKey: config.apiKey || 'required-but-not-used', // Pass dummy key if none provided (some compatible APIs don't need one)
      baseURL: config.baseURL, // Use the custom provider's base URL
      maxRetries: config.defaultMaxRetries ?? 2,
      timeout: config.defaultTimeoutMs ?? 60 * 1000,
      dangerouslyAllowBrowser: false // Ensure this is false for backend usage
    })
  }

  // --- Helper Methods (Copied from previous examples for direct use) ---

  // Helper to map RosettaMessages to OpenAI format
  private mapMessagesToOpenAI(messages: RosettaMessage[]): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    return messages.map(msg => {
      const role = mapRoleToOpenAI(msg.role)
      const content = mapContentForOpenAIRole(msg.content, role)

      // Construct the message param based on role, ensuring required fields
      switch (role) {
        case 'system':
          if (typeof content !== 'string' || content === '') {
            throw new MappingError('System message content must be a non-empty string.', this.provider)
          }
          return { role, content }
        case 'user':
          if (content === null || (Array.isArray(content) && content.length === 0)) {
            throw new MappingError('User message content cannot be empty.', this.provider)
          }
          return { role, content: content as string | OpenAI.Chat.Completions.ChatCompletionContentPart[] }
        case 'assistant':
          const assistantMsg: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam = {
            role,
            content: content as string | null
          }
          if (msg.toolCalls && msg.toolCalls.length > 0) {
            assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
              id: tc.id,
              type: tc.type,
              function: { name: tc.function.name, arguments: tc.function.arguments }
            }))
            // Compatibility: Set content to empty string instead of null when tool calls exist for some providers.
            if (content === null || content === '') {
              assistantMsg.content = '' // Use empty string
            }
          } else if (assistantMsg.content === null) {
            // If no tool calls, content cannot be null for OpenAI assistant message (standard behavior)
            assistantMsg.content = '' // Default to empty string if no tool calls and content was null
          }
          return assistantMsg
        case 'tool':
          if (!msg.toolCallId) {
            throw new MappingError('Tool message requires toolCallId.', this.provider)
          }
          if (typeof content !== 'string') {
            // Should ideally not happen if mapContentForOpenAIRole works correctly
            throw new MappingError('Tool message content must map to a string.', this.provider)
          }
          // Allow empty string for tool content
          return { role, tool_call_id: msg.toolCallId, content: content }
        default:
          // Should be unreachable if mapRoleToOpenAI is correct
          throw new MappingError(`Unhandled role: ${role}`, this.provider)
      }
    })
  }

  // Helper to map RosettaTools to OpenAI format
  private mapToolsToOpenAI(tools?: RosettaTool<any>[]): OpenAI.Chat.Completions.ChatCompletionTool[] | undefined {
    return tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Unsupported tool type: ${tool.type}`, this.provider)
      }
      return {
        type: tool.type,
        function: {
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters as OpenAI.FunctionDefinition['parameters'] // Basic cast
        }
      }
    })
  }

  // Helper to map Rosetta toolChoice to OpenAI format
  private mapToolChoiceToOpenAI(
    toolChoice?: GenerateParams['toolChoice']
  ): OpenAI.Chat.Completions.ChatCompletionToolChoiceOption | undefined {
    const baseChoice = mapBaseToolChoice(toolChoice)
    if (baseChoice === 'auto' || baseChoice === 'none' || baseChoice === 'required') {
      return baseChoice
    } else if (typeof baseChoice === 'object' && baseChoice.type === 'function') {
      return { type: 'function', function: { name: baseChoice.function.name } }
    }
    return undefined
  }

  // --- Overridden Execution Methods ---
  // eslint-disable-next-line prettier/prettier
  override async executeGenerate(
    _mappedParams: any, // Not used, we map from originalParams
    _apiKey: string | undefined, // Already configured in internal client
    _providerConfig: CustomProviderConfig, // Use config passed to constructor (this.config)
    originalParams: GenerateParams
  ): Promise<GenerateResult> {
    // Feature Check
    if (!this.config.supportedFeatures.includes('generate')) {
      throw new UnsupportedFeatureError(this.provider, 'generate')
    }

    const model = originalParams.model ?? this.config.defaultModel // Use this.config
    if (!model) {
      throw new ConfigurationError(`Model must be specified for ${this.provider} (or set a default).`)
    }

    // Map Rosetta messages/tools to OpenAI format using the helpers
    const messages = this.mapMessagesToOpenAI(originalParams.messages)
    const tools = this.mapToolsToOpenAI(originalParams.tools)
    const tool_choice = this.mapToolChoiceToOpenAI(originalParams.toolChoice)

    // Precedence and sentinel handling for token limits
    const hasMct = originalParams.maxCompletionTokens != null
    const hasMt = originalParams.maxTokens != null
    const normalizedMct =
      hasMct && originalParams.maxCompletionTokens !== -1 ? originalParams.maxCompletionTokens : undefined
    const normalizedMt = hasMt && originalParams.maxTokens !== -1 ? originalParams.maxTokens : undefined
    const effectiveMaxTokens = normalizedMct ?? normalizedMt

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
      ...(originalParams.extraParams ?? {}),
      model: model,
      messages: messages,
      ...(hasMct ? { max_completion_tokens: effectiveMaxTokens } : {}), // only use max_completion_tokens if sent
      ...(hasMt && !hasMct ? { max_tokens: effectiveMaxTokens } : {}), // only use max_tokens if sent and max_completion_tokens is not sent
      temperature: originalParams.temperature,
      stream: false,
      tools,
      tool_choice,
      top_p: originalParams.topP,
      ...(originalParams.stop ? { stop: originalParams.stop } : {})
    }

    try {
      const response = await this.openaiClient.chat.completions.create(openAIParams)

      // Map the OpenAI SDK response back to GenerateResult using common helper
      // Pass original tools for validation within mapFromOpenAIResponse
      return mapFromOpenAIResponse(response, model, originalParams.tools)
    } catch (error) {
      // Wrap potential OpenAI SDK errors using the common wrapper
      throw this.wrapProviderError(error, this.provider)
    }
  }

  override async *executeStream(
    _mappedParams: any, // Not used
    _apiKey: string | undefined, // Already configured
    _providerConfig: CustomProviderConfig, // Use this.config
    originalParams: GenerateParams,
    abortSignal?: AbortSignal
  ): AsyncIterable<StreamChunk> {
    // Feature Check
    if (!this.config.supportedFeatures.includes('stream')) {
      throw new UnsupportedFeatureError(this.provider, 'stream')
    }

    const model = originalParams.model ?? this.config.defaultModel // Use this.config
    if (!model) {
      yield {
        type: 'error',
        data: { error: new ConfigurationError(`Model must be specified for ${this.provider} (or set a default).`) }
      }
      return
    }

    // Map Rosetta messages/tools to OpenAI format using the helpers
    let messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]
    let tools: OpenAI.Chat.Completions.ChatCompletionTool[] | undefined
    let tool_choice: OpenAI.Chat.Completions.ChatCompletionToolChoiceOption | undefined
    try {
      messages = this.mapMessagesToOpenAI(originalParams.messages)
      tools = this.mapToolsToOpenAI(originalParams.tools)
      tool_choice = this.mapToolChoiceToOpenAI(originalParams.toolChoice)
    } catch (mappingError) {
      yield { type: 'error', data: { error: mappingError as Error } }
      return
    }

    // Precedence and sentinel handling for token limits
    const hasMct = originalParams.maxCompletionTokens != null
    const hasMt = originalParams.maxTokens != null
    const normalizedMct =
      hasMct && originalParams.maxCompletionTokens !== -1 ? originalParams.maxCompletionTokens : undefined
    const normalizedMt = hasMt && originalParams.maxTokens !== -1 ? originalParams.maxTokens : undefined
    const effectiveMaxTokens = normalizedMct ?? normalizedMt

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
      ...(originalParams.extraParams ?? {}),
      model: model,
      messages: messages,
      ...(hasMct ? { max_completion_tokens: effectiveMaxTokens } : {}), // only use max_completion_tokens if sent
      ...(hasMt && !hasMct ? { max_tokens: effectiveMaxTokens } : {}), // only use max_tokens if sent and max_completion_tokens is not sent
      temperature: originalParams.temperature,
      stream: true,
      stream_options: { include_usage: true }, // Request usage data
      tools,
      tool_choice,
      top_p: originalParams.topP,
      ...(originalParams.stop != null ? { stop: originalParams.stop } : {}),
      ...(originalParams.responseFormat != null
        ? { response_format: mapToOpenAIResponseFormat(originalParams.responseFormat) }
        : {})
    }

    try {
      if (abortSignal?.aborted) return

      const stream = await this.openaiClient.chat.completions.create(openAIParams, { signal: abortSignal })

      // Reuse the common OpenAI stream mapping logic
      // Pass Provider.OpenAI as the provider type expected by the common helper
      yield* mapOpenAIStream(
        stream as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
        Provider.OpenAI, // Use the base enum type for the common helper
        model,
        originalParams.tools
      )
    } catch (error) {
      // Wrap potential OpenAI SDK errors using the common wrapper
      yield { type: 'error', data: { error: this.wrapProviderError(error, this.provider) } }
    }
  }

  // --- NEW: Implement executeEmbed ---
  override async executeEmbed(
    _mappedParams: any, // Not used, we map from originalParams
    _apiKey: string | undefined, // Already configured in internal client
    providerConfig: CustomProviderConfig, // Use config passed to constructor (this.config)
    originalParams: EmbedParams
  ): Promise<EmbedResult> {
    // Feature Check
    if (!this.config.supportedFeatures.includes('embed')) {
      throw new UnsupportedFeatureError(this.provider, 'embed')
    }

    // Model Determination
    const modelId =
      originalParams.model ?? providerConfig.defaultEmbeddingModel ?? 'nomic-embed-text-v1.5' // Use documented default

    if (!modelId) {
      // This case is less likely now with the default, but keep for robustness
      throw new ConfigurationError(
        `Embedding model must be specified for provider ${this.provider} (or set a defaultEmbeddingModel in config).`
      )
    }

    try {
      // Parameter Mapping (using helper from openai.embed.mapper)
      const openAIEmbedParams = mapToOpenAIEmbedParams({ ...originalParams, model: modelId })

      // API Call
      const response = await this.openaiClient.embeddings.create(openAIEmbedParams)

      // Response Mapping (using helper from openai.embed.mapper)
      return mapFromOpenAIEmbedResponse(response, modelId)
    } catch (error) {
      // Error Handling
      throw this.wrapProviderError(error, this.provider)
    }
  }
  // --- End NEW ---

  // Override wrapProviderError to delegate to the common OpenAI error wrapper
  override wrapProviderError(error: unknown, _provider: ProviderKey): RosettaAIError {
    // Delegate to the common wrapper, passing Provider.OpenAI as the expected type
    return wrapOpenAIError(error, Provider.OpenAI)
  }

  // Note: executeGenerateSpeech, executeStreamSpeech, executeTranscribe, executeTranslate
  // are not overridden here. If an OpenAI-compatible provider supports these,
  // this mapper would need to implement the corresponding `execute*` methods,
  // likely by calling the relevant methods on `this.openaiClient` and mapping
  // parameters/results using helpers from `openai.embed.mapper.ts` or `openai.audio.mapper.ts`.
}
