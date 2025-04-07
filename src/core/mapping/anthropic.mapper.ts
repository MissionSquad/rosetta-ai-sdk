import Anthropic, { APIError } from '@anthropic-ai/sdk'
import {
  RawMessageStreamEvent,
  MessageParam as AnthropicMessageParam,
  Tool as AnthropicTool,
  ThinkingConfigParam as AnthropicThinkingConfig,
  ContentBlockParam as AnthropicContentBlockParam,
  TextBlockParam as AnthropicTextBlockParam,
  ImageBlockParam as AnthropicImageBlockParam,
  RawContentBlockStopEvent,
  ToolUseBlockParam // INPUT type for tool use
} from '@anthropic-ai/sdk/resources/messages'
import { Tool as AnthropicToolType } from '@anthropic-ai/sdk/resources'
import {
  Message as AnthropicMessage,
  ContentBlock as AnthropicResponseContentBlock,
  ToolUseBlock as AnthropicToolUseBlock
} from '@anthropic-ai/sdk/resources/messages'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult
} from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError, UnsupportedFeatureError } from '../../errors'
import { safeGet } from '../utils'
import { IProviderMapper } from './base.mapper'
import { mapTokenUsage, mapBaseParams, mapBaseToolChoice } from './common.utils'

// Type alias for the stream type from Anthropic SDK
type AnthropicMessageStream = AsyncIterable<RawMessageStreamEvent>

export class AnthropicMapper implements IProviderMapper {
  readonly provider = Provider.Anthropic

  // --- Parameter Mapping ---

  private mapRoleToAnthropic(role: RosettaMessage['role']): AnthropicMessageParam['role'] {
    switch (role) {
      case 'user':
        return 'user'
      case 'assistant':
        return 'assistant'
      case 'system':
      case 'tool':
        throw new MappingError(
          `Role '${role}' should be handled structurally for Anthropic.`,
          this.provider,
          'mapRoleToAnthropic'
        )
      default:
        // Ensure exhaustive check works with `never`
        const _exhaustiveCheck: never = role
        throw new MappingError(`Unsupported role: ${_exhaustiveCheck}`, this.provider)
    }
  }

  private mapContentToAnthropic(content: RosettaMessage['content']): string | Array<AnthropicContentBlockParam> {
    if (content === null) {
      console.warn('Mapping null content to empty string for Anthropic input.')
      return ''
    }
    if (typeof content === 'string') {
      return content
    }
    // Handle empty array case - return empty string as Anthropic content cannot be empty array
    if (Array.isArray(content) && content.length === 0) {
      console.warn('Mapping empty content array to empty string for Anthropic input.')
      return ''
    }

    const parts: AnthropicContentBlockParam[] = content.map(part => {
      if (part.type === 'text') {
        const textParam: AnthropicTextBlockParam = { type: 'text', text: part.text }
        return textParam
      } else if (part.type === 'image') {
        const imageParam: AnthropicImageBlockParam = {
          type: 'image',
          source: { type: 'base64', media_type: part.image.mimeType, data: part.image.base64Data }
        }
        return imageParam
      } else {
        // Ensure exhaustive check works with `never`
        const _exhaustiveCheck: never = part
        throw new MappingError(`Unsupported content part type: ${(_exhaustiveCheck as any).type}`, this.provider)
      }
    })
    return parts
  }

  mapToProviderParams(
    params: GenerateParams
  ): Anthropic.Messages.MessageCreateParamsNonStreaming | Anthropic.Messages.MessageCreateParamsStreaming {
    let systemPrompt: string | undefined = undefined
    const messages: AnthropicMessageParam[] = []

    for (const msg of params.messages) {
      if (msg.role === 'system') {
        if (systemPrompt) throw new MappingError('Multiple system messages not supported by Anthropic.', this.provider)
        if (typeof msg.content !== 'string')
          throw new MappingError('Anthropic system prompt must be string.', this.provider)
        systemPrompt = msg.content
        continue
      }
      if (msg.role === 'tool') {
        if (!msg.toolCallId || typeof msg.content !== 'string') {
          throw new MappingError(
            'Invalid tool result message format for Anthropic. Requires toolCallId and string content.',
            this.provider
          )
        }
        messages.push({
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: msg.toolCallId, content: msg.content }
          ] as AnthropicContentBlockParam[] // Explicit cast
        })
      } else if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
        const assistantContent = this.mapContentToAnthropic(msg.content)
        const contentBlocks: AnthropicContentBlockParam[] = []

        // FIX: Only add text block if assistantContent is a non-empty string or a non-empty array containing text
        if (typeof assistantContent === 'string' && assistantContent.length > 0) {
          contentBlocks.push({ type: 'text', text: assistantContent })
        } else if (Array.isArray(assistantContent)) {
          // Add text/image blocks from assistant message
          assistantContent.forEach(block => {
            if (block.type === 'text' || block.type === 'image') {
              // Include empty text blocks if they were explicitly provided
              // @ts-ignore
              if (block.type === 'text' || (block.type === 'image' && block.source.data)) {
                contentBlocks.push(block)
              }
            } else {
              console.warn(
                `Ignoring unexpected content block type '${block.type}' in assistant message with tool calls.`
              )
            }
          })
        }

        // Add tool_use blocks
        msg.toolCalls.forEach(toolCall => {
          try {
            const toolUseBlock: ToolUseBlockParam = {
              type: 'tool_use',
              id: toolCall.id,
              name: toolCall.function.name,
              input: JSON.parse(toolCall.function.arguments || '{}')
            }
            contentBlocks.push(toolUseBlock)
          } catch (e) {
            throw new MappingError(
              `Failed to parse arguments for tool_use block ${toolCall.id}`,
              this.provider,
              'mapToProviderParams toolCall mapping',
              e
            )
          }
        })

        // Ensure contentBlocks is not empty if assistant message exists
        // If only tool_use blocks exist, that's valid.
        if (contentBlocks.length === 0) {
          // This case should only happen if toolCalls were present but msg.content was null/empty
          // and resulted in an empty contentBlocks array.
          console.warn('Assistant message with tool calls resulted in empty content blocks; this might be invalid.')
          // Anthropic requires at least one block if the role is assistant.
          // If only tool_use blocks exist, that's fine. If somehow both content and tool_calls
          // resulted in nothing, this is an error state.
          if (msg.toolCalls.length === 0) {
            throw new MappingError(
              'Assistant message resulted in empty content blocks without tool calls.',
              this.provider
            )
          }
        }

        messages.push({ role: 'assistant', content: contentBlocks })
      } else {
        // Handle regular user/assistant messages
        const mappedContent = this.mapContentToAnthropic(msg.content)
        // Ensure content is not empty array for user/assistant roles
        if (Array.isArray(mappedContent) && mappedContent.length === 0) {
          // If content was originally empty array or null, mapContentToAnthropic returns "".
          // If it was non-empty but resulted in empty (e.g., only unsupported types), throw.
          throw new MappingError(
            `Role '${msg.role}' requires non-empty content for Anthropic. Received empty array after mapping.`,
            this.provider
          )
        }
        messages.push({
          role: this.mapRoleToAnthropic(msg.role as 'user' | 'assistant'),
          content: mappedContent
        })
      }
    }

    // Ensure messages array is not empty (Anthropic requires at least one message)
    if (messages.length === 0 && !systemPrompt) {
      // Allow system prompt only requests? Check Anthropic docs. Assuming not for now.
      throw new MappingError('No user or assistant messages provided for Anthropic.', this.provider)
    }

    const tools: AnthropicTool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Unsupported tool type for Anthropic: ${tool.type}`, this.provider)
      }
      const inputSchemaSource = tool.function.parameters
      if (
        typeof inputSchemaSource !== 'object' ||
        inputSchemaSource === null ||
        Array.isArray(inputSchemaSource) ||
        inputSchemaSource.type !== 'object'
      ) {
        throw new MappingError(
          `Invalid parameters schema for tool '${
            tool.function.name
          }'. Anthropic requires a JSON Schema object with top-level 'type: "object"'. Received: ${JSON.stringify(
            inputSchemaSource
          )}`,
          this.provider
        )
      }
      const inputSchema: AnthropicToolType.InputSchema = inputSchemaSource as AnthropicToolType.InputSchema
      return { name: tool.function.name, description: tool.function.description, input_schema: inputSchema }
    })

    // Use common utility for base tool choice mapping
    const baseToolChoice = mapBaseToolChoice(params.toolChoice)
    let anthropicToolChoice: Anthropic.Messages.ToolChoice | undefined = undefined
    if (baseToolChoice) {
      if (baseToolChoice === 'auto' || baseToolChoice === 'none') {
        anthropicToolChoice = { type: baseToolChoice }
      } else if (baseToolChoice === 'required') {
        anthropicToolChoice = { type: 'any' } // Map 'required' to Anthropic's 'any'
      } else if (typeof baseToolChoice === 'object' && baseToolChoice.type === 'function') {
        anthropicToolChoice = { type: 'tool', name: baseToolChoice.function.name }
      } else {
        console.warn(`Unhandled baseToolChoice format: ${JSON.stringify(baseToolChoice)}`)
      }
    }

    let thinkingParam: AnthropicThinkingConfig | undefined = undefined
    if (params.thinking) {
      // Corrected: budget_tokens is part of the thinking object
      thinkingParam = { type: 'enabled', budget_tokens: 1024 }
    }

    let systemParam: string | AnthropicTextBlockParam[] | undefined
    if (typeof systemPrompt === 'string') {
      systemParam = systemPrompt
    } else {
      systemParam = undefined
    }

    // Use common utility for base parameters
    const baseMappedParams = mapBaseParams(params)

    const basePayload = {
      model: params.model!,
      messages: messages,
      system: systemParam,
      max_tokens: baseMappedParams.maxTokens ?? 4096, // Use mapped value or default
      temperature: baseMappedParams.temperature, // Use mapped value
      top_p: baseMappedParams.topP, // Use mapped value
      stop_sequences: baseMappedParams.stopSequences, // Use mapped value
      tools: tools,
      tool_choice: anthropicToolChoice,
      ...(thinkingParam && { thinking: thinkingParam }) // Correctly add thinking param
    }

    if (params.stream) {
      const streamPayload: Anthropic.Messages.MessageCreateParamsStreaming = { ...basePayload, stream: true }
      return streamPayload
    } else {
      const nonStreamPayload: Anthropic.Messages.MessageCreateParamsNonStreaming = basePayload
      return nonStreamPayload
    }
  }

  // --- Result Mapping ---

  private mapToolCallsFromAnthropic(
    contentBlocks: AnthropicResponseContentBlock[] | undefined
  ): RosettaToolCallRequest[] | undefined {
    if (!Array.isArray(contentBlocks)) return undefined
    const toolCalls: RosettaToolCallRequest[] = contentBlocks
      .filter((block): block is AnthropicToolUseBlock => block.type === 'tool_use')
      .map(block => ({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) }
      }))
    return toolCalls.length > 0 ? toolCalls : undefined
  }

  mapFromProviderResponse(response: AnthropicMessage, model: string): GenerateResult {
    let combinedTextContent: string | null = null
    let thinkingText: string | null = null
    const responseContent = response.content as AnthropicResponseContentBlock[]

    if (Array.isArray(responseContent)) {
      const textParts: string[] = []
      responseContent.forEach(block => {
        if (block.type === 'text') {
          textParts.push(block.text)
        } else if (block.type === 'thinking' && typeof block.thinking === 'string') {
          thinkingText = block.thinking
        }
        // Ignore tool_use blocks for combined text content
      })
      if (textParts.length > 0) {
        combinedTextContent = textParts.join('')
      }
    }

    const toolCalls = this.mapToolCallsFromAnthropic(responseContent)
    const finishReason =
      response.stop_reason === 'tool_use'
        ? 'tool_calls'
        : response.stop_reason === 'max_tokens'
        ? 'length'
        : response.stop_reason === 'stop_sequence'
        ? 'stop'
        : response.stop_reason === 'end_turn'
        ? 'stop'
        : response.stop_reason ?? 'unknown'

    // Use common utility for usage mapping
    const usage = mapTokenUsage(response.usage)

    return {
      content: combinedTextContent,
      toolCalls: toolCalls,
      finishReason: finishReason,
      usage: usage, // Use mapped usage
      thinkingSteps: thinkingText,
      citations: undefined,
      parsedContent: null,
      model: response.model ?? model,
      rawResponse: response
    }
  }

  // --- Stream Mapping ---

  async *mapProviderStream(stream: AnthropicMessageStream): AsyncIterable<StreamChunk> {
    let currentUsage: TokenUsage | undefined
    let finalFinishReason: string | null = null
    let thinkingStarted = false
    let model = ''
    const toolCallArgAccumulators: Record<string, { id: string; name: string; args: string; index: number }> = {}
    const toolCallIdByIndex: Record<number, string> = {}
    let aggregatedResult: GenerateResult | null = null

    try {
      for await (const event of stream) {
        if (typeof event !== 'object' || !event || !('type' in event)) {
          console.warn('Received unexpected event format from Anthropic stream:', event)
          continue
        }

        switch (event.type) {
          case 'message_start':
            model = safeGet<string>(event.message, 'model') ?? ''
            yield { type: 'message_start', data: { provider: this.provider, model: model } }
            // Use common utility for usage mapping
            currentUsage = mapTokenUsage(safeGet<Anthropic.Usage>(event.message, 'usage'))
            aggregatedResult = {
              content: '',
              toolCalls: [],
              finishReason: null,
              usage: undefined,
              model: model,
              thinkingSteps: null,
              citations: undefined,
              parsedContent: null,
              rawResponse: undefined
            }
            break
          case 'content_block_start':
            if (event.content_block.type === 'thinking') {
              yield { type: 'thinking_start' }
              thinkingStarted = true
            } else if (event.content_block.type === 'tool_use') {
              const toolUse = event.content_block
              const index = event.index
              toolCallArgAccumulators[toolUse.id] = { id: toolUse.id, name: toolUse.name, args: '', index }
              toolCallIdByIndex[index] = toolUse.id
              yield {
                type: 'tool_call_start',
                data: { index, toolCall: { id: toolUse.id, type: 'function', function: { name: toolUse.name } } }
              }
            }
            break
          case 'content_block_delta':
            if (event.delta.type === 'text_delta') {
              yield { type: 'content_delta', data: { delta: event.delta.text } }
              if (aggregatedResult) aggregatedResult.content = (aggregatedResult.content ?? '') + event.delta.text
            } else if (event.delta.type === 'thinking_delta') {
              if (!thinkingStarted) {
                yield { type: 'thinking_start' }
                thinkingStarted = true
              }
              yield { type: 'thinking_delta', data: { delta: event.delta.thinking } }
              if (aggregatedResult)
                aggregatedResult.thinkingSteps = (aggregatedResult.thinkingSteps ?? '') + event.delta.thinking
            } else if (event.delta.type === 'input_json_delta') {
              const index = event.index
              const currentToolCallId = toolCallIdByIndex[index]
              if (currentToolCallId && toolCallArgAccumulators[currentToolCallId]) {
                toolCallArgAccumulators[currentToolCallId].args += event.delta.partial_json
                yield {
                  type: 'tool_call_delta',
                  data: { index, id: currentToolCallId, functionArgumentChunk: event.delta.partial_json }
                }
              } else {
                console.warn(`Received input_json_delta for unknown tool index: ${index}`)
              }
            }
            break
          case 'content_block_stop':
            const stoppedEvent = event as RawContentBlockStopEvent
            const stoppedBlockIndex = stoppedEvent.index
            const finishedToolCallId = toolCallIdByIndex[stoppedBlockIndex]

            if (finishedToolCallId && toolCallArgAccumulators[finishedToolCallId]) {
              const toolData = toolCallArgAccumulators[finishedToolCallId]
              yield { type: 'tool_call_done', data: { index: stoppedBlockIndex, id: finishedToolCallId } }
              if (aggregatedResult) {
                aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
                aggregatedResult.toolCalls.push({
                  id: toolData.id,
                  type: 'function',
                  function: { name: toolData.name, arguments: toolData.args }
                })
              }
              delete toolCallArgAccumulators[finishedToolCallId]
              delete toolCallIdByIndex[stoppedBlockIndex]
            }
            if (thinkingStarted) {
              yield { type: 'thinking_stop' }
              thinkingStarted = false
            }
            break
          case 'message_delta':
            // Use common utility for usage mapping
            const deltaUsage = mapTokenUsage(event.usage)
            if (deltaUsage?.completionTokens !== undefined) {
              // Update currentUsage, preserving prompt tokens if already set
              currentUsage = {
                promptTokens: currentUsage?.promptTokens, // Keep existing prompt tokens
                completionTokens: deltaUsage.completionTokens,
                // FIX: Calculate totalTokens correctly
                totalTokens:
                  currentUsage?.promptTokens !== undefined
                    ? currentUsage.promptTokens + deltaUsage.completionTokens
                    : deltaUsage.completionTokens // If prompt tokens unknown, total is just completion
              }
            }
            if (event.delta.stop_reason) {
              finalFinishReason =
                event.delta.stop_reason === 'tool_use'
                  ? 'tool_calls'
                  : event.delta.stop_reason === 'max_tokens'
                  ? 'length'
                  : event.delta.stop_reason === 'stop_sequence'
                  ? 'stop'
                  : event.delta.stop_reason === 'end_turn'
                  ? 'stop'
                  : event.delta.stop_reason ?? 'unknown'
            }
            break
          case 'message_stop':
            // Ensure finalFinishReason is set if it was null
            finalFinishReason = finalFinishReason ?? 'stop'
            yield { type: 'message_stop', data: { finishReason: finalFinishReason } }
            if (currentUsage) {
              yield { type: 'final_usage', data: { usage: currentUsage } }
              if (aggregatedResult) aggregatedResult.usage = currentUsage
            }
            if (aggregatedResult) {
              aggregatedResult.finishReason = finalFinishReason
              if (aggregatedResult.content === '') aggregatedResult.content = null
              if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
              yield { type: 'final_result', data: { result: aggregatedResult } }
            } else {
              console.warn('Message stop received but no aggregated result was built.')
            }
            break
        }
      }
    } catch (error) {
      const mappedError = this.wrapProviderError(error, this.provider)
      yield { type: 'error', data: { error: mappedError } }
    }
  }

  // --- Embedding Mapping ---
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToEmbedParams(_params: EmbedParams): any {
    throw new UnsupportedFeatureError(this.provider, 'Embeddings')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromEmbedResponse(_response: any, _modelId: string): EmbedResult {
    throw new UnsupportedFeatureError(this.provider, 'Embeddings')
  }

  // --- Audio Mapping ---
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranscribeParams(_params: TranscribeParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranscribeResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranslateParams(_params: TranslateParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranslateResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    if (error instanceof RosettaAIError) {
      return error
    }

    // Add structural check for Anthropic APIError-like objects
    const isAnthropicAPIErrorLike = (e: any): e is APIError =>
      typeof e === 'object' &&
      e !== null &&
      typeof e.status === 'number' &&
      typeof e.message === 'string' &&
      'error' in e // Check for the nested 'error' property

    if (error instanceof Anthropic.APIError || isAnthropicAPIErrorLike(error)) {
      const anthropicError = error as APIError // Cast after check
      // FIX: Correctly access the nested error type using safeGet
      const nestedErrorType = safeGet<string>(anthropicError, 'error', 'error', 'type')

      // Use the nested type for both code and type if available
      const finalCode = nestedErrorType
      const finalType = nestedErrorType

      return new ProviderAPIError(anthropicError.message, provider, anthropicError.status, finalCode, finalType, error)
    }

    if (error instanceof Error) {
      return new ProviderAPIError(error.message, provider, undefined, undefined, undefined, error)
    }
    return new ProviderAPIError(
      String(error ?? 'Unknown error occurred'),
      provider,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
