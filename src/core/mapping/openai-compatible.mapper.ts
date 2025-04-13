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
  RosettaTool
} from '../../types' // Adjust path as needed
import { BaseCustomMapper } from './base.custom.mapper' // Adjust path
import {
  mapFromOpenAIResponse,
  mapOpenAIStream,
  mapContentForOpenAIRole,
  mapRoleToOpenAI,
  wrapOpenAIError
} from './openai.common' // Reuse common OpenAI mapping logic
import { mapBaseToolChoice } from '../mapping/common.utils' // Import tool choice mapper
import { MappingError, RosettaAIError } from '../../errors'

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
    const model = originalParams.model ?? this.config.defaultModel // Use this.config
    if (!model) {
      throw new Error(`Model must be specified for ${this.provider} (or set a default).`)
    }

    // Map Rosetta messages/tools to OpenAI format using the helpers
    const messages = this.mapMessagesToOpenAI(originalParams.messages)
    const tools = this.mapToolsToOpenAI(originalParams.tools)
    const tool_choice = this.mapToolChoiceToOpenAI(originalParams.toolChoice)

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
      model: model,
      messages: messages,
      max_tokens: originalParams.maxTokens,
      temperature: originalParams.temperature,
      stream: false,
      tools,
      tool_choice,
      top_p: originalParams.topP
      // Add other compatible parameters if needed (e.g., stop, seed)
    }

    try {
      console.log(`[${this.provider}] Calling API (via OpenAI SDK)... Params:`, JSON.stringify(openAIParams))
      const response = await this.openaiClient.chat.completions.create(openAIParams)
      console.log(`[${this.provider}] Received response.`)

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
    originalParams: GenerateParams
  ): AsyncIterable<StreamChunk> {
    const model = originalParams.model ?? this.config.defaultModel // Use this.config
    if (!model) {
      yield {
        type: 'error',
        data: { error: new Error(`Model must be specified for ${this.provider} (or set a default).`) }
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

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
      model: model,
      messages: messages,
      max_tokens: originalParams.maxTokens,
      temperature: originalParams.temperature,
      stream: true,
      stream_options: { include_usage: true }, // Request usage data
      tools,
      tool_choice,
      top_p: originalParams.topP
      // Add other compatible parameters if needed
    }

    try {
      console.log(`[${this.provider}] Calling API for stream (via OpenAI SDK)... Params:`, JSON.stringify(openAIParams))
      const stream = await this.openaiClient.chat.completions.create(openAIParams)
      console.log(`[${this.provider}] Received stream.`)

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

  // Override wrapProviderError to delegate to the common OpenAI error wrapper
  override wrapProviderError(error: unknown, _provider: ProviderKey): RosettaAIError {
    // Delegate to the common wrapper, passing Provider.OpenAI as the expected type
    return wrapOpenAIError(error, Provider.OpenAI)
  }

  // Note: executeEmbed, executeGenerateSpeech, executeStreamSpeech, executeTranscribe, executeTranslate
  // are not overridden here. If an OpenAI-compatible provider supports these,
  // this mapper would need to implement the corresponding `execute*` methods,
  // likely by calling the relevant methods on `this.openaiClient` and mapping
  // parameters/results using helpers from `openai.embed.mapper.ts` or `openai.audio.mapper.ts`.
}