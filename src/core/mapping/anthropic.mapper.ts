import Anthropic from '@anthropic-ai/sdk'
import {
  RawMessageStreamEvent,
  MessageParam as AnthropicMessageParam,
  Tool as AnthropicTool,
  ThinkingConfigParam as AnthropicThinkingConfig,
  ContentBlockParam as AnthropicContentBlockParam,
  TextBlockParam as AnthropicTextBlockParam,
  ImageBlockParam as AnthropicImageBlockParam,
  RawContentBlockStopEvent,
  ToolUseBlockParam // Import the INPUT type for tool use
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
  Provider
} from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError } from '../../errors' // Import error types
import { safeGet } from '../utils'

// Type alias for the stream type from Anthropic SDK
type AnthropicMessageStream = AsyncIterable<RawMessageStreamEvent>

// --- Parameter Mapping ---

function mapRoleToAnthropic(role: RosettaMessage['role']): AnthropicMessageParam['role'] {
  switch (role) {
    case 'user':
      return 'user'
    case 'assistant':
      return 'assistant'
    case 'system':
    case 'tool':
      throw new MappingError(
        `Role '${role}' should be handled structurally for Anthropic.`,
        Provider.Anthropic,
        'mapRoleToAnthropic'
      )
    default:
      const _exhaustiveCheck: never = role
      throw new MappingError(`Unsupported role: ${_exhaustiveCheck}`, Provider.Anthropic)
  }
}

// FIX: This function maps to the INPUT type array: AnthropicContentBlockParam[]
function mapContentToAnthropic(content: RosettaMessage['content']): string | Array<AnthropicContentBlockParam> {
  if (content === null) {
    // Anthropic allows string content, returning empty string seems reasonable for null
    console.warn('Mapping null content to empty string for Anthropic input.')
    return ''
  }
  if (typeof content === 'string') {
    // Single string is valid input for AnthropicMessageParam.content
    return content
  }

  // Map RosettaContentPart[] to AnthropicContentBlockParam[]
  // FIX: Change type annotation of 'parts' to the correct INPUT param type
  const parts: AnthropicContentBlockParam[] = content.map(part => {
    if (part.type === 'text') {
      // Create an object matching AnthropicTextBlockParam
      const textParam: AnthropicTextBlockParam = {
        type: 'text',
        text: part.text
        // citations: null // Citations aren't part of TextBlockParam input
      }
      return textParam
    } else if (part.type === 'image') {
      // Create an object matching AnthropicImageBlockParam
      const imageParam: AnthropicImageBlockParam = {
        type: 'image',
        source: {
          type: 'base64', // Source type is base64
          media_type: part.image.mimeType,
          data: part.image.base64Data
        }
        // cache_control: undefined // Optional cache control
      }
      return imageParam
    } else {
      const _exhaustiveCheck: never = part
      throw new MappingError(`Unsupported content part type: ${(_exhaustiveCheck as any).type}`, Provider.Anthropic)
    }
  })

  return parts
}

export function mapToAnthropicParams(
  params: GenerateParams
): Anthropic.Messages.MessageCreateParamsNonStreaming | Anthropic.Messages.MessageCreateParamsStreaming {
  let systemPrompt: string | undefined = undefined
  const messages: AnthropicMessageParam[] = []

  for (const msg of params.messages) {
    if (msg.role === 'system') {
      if (systemPrompt)
        throw new MappingError('Multiple system messages not supported by Anthropic.', Provider.Anthropic)
      if (typeof msg.content !== 'string')
        throw new MappingError('Anthropic system prompt must be string.', Provider.Anthropic)
      systemPrompt = msg.content
      continue
    }
    if (msg.role === 'tool') {
      if (!msg.toolCallId || typeof msg.content !== 'string') {
        throw new MappingError(
          'Invalid tool result message format for Anthropic. Requires toolCallId and string content.',
          Provider.Anthropic
        )
      }
      messages.push({
        role: 'user', // Tool results are sent back as user messages
        content: [
          {
            // Must be ContentBlockParam array
            type: 'tool_result',
            tool_use_id: msg.toolCallId,
            content: msg.content // Tool result content is string or array of Text/Image BlockParams
            // is_error: false, // Optional
          }
        ]
      })
    } else if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
      // FIX: DO NOT skip the assistant message. Map it correctly including tool_use blocks.
      const assistantContent = mapContentToAnthropic(msg.content)
      const contentBlocks: AnthropicContentBlockParam[] = []

      // Add text content if present
      if (typeof assistantContent === 'string' && assistantContent.length > 0) {
        contentBlocks.push({ type: 'text', text: assistantContent })
      } else if (Array.isArray(assistantContent)) {
        // If content was already an array (e.g., multimodal), filter/add text parts
        assistantContent.forEach(block => {
          if (block.type === 'text') {
            contentBlocks.push(block)
          } else {
            console.warn(`Ignoring non-text content block type '${block.type}' in assistant message with tool calls.`)
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
            input: JSON.parse(toolCall.function.arguments || '{}') // Parse arguments
          }
          contentBlocks.push(toolUseBlock)
        } catch (e) {
          throw new MappingError(
            `Failed to parse arguments for tool_use block ${toolCall.id}`,
            Provider.Anthropic,
            'mapToAnthropicParams toolCall mapping',
            e
          )
        }
      })

      // Add the assistant message with text and tool_use blocks
      messages.push({
        role: 'assistant',
        content: contentBlocks
      })
    } else {
      // Handle regular user/assistant messages without tool calls
      messages.push({
        role: mapRoleToAnthropic(msg.role as 'user' | 'assistant'),
        content: mapContentToAnthropic(msg.content) // Returns string | ContentBlockParam[]
      })
    }
  }

  // FIX: Validate input_schema requires type: 'object'
  const tools: AnthropicTool[] | undefined = params.tools?.map(tool => {
    if (tool.type !== 'function') {
      throw new MappingError(`Unsupported tool type for Anthropic: ${tool.type}`, Provider.Anthropic)
    }
    const inputSchemaSource = tool.function.parameters

    // Validate the schema structure required by Anthropic's Tool.InputSchema
    if (
      typeof inputSchemaSource !== 'object' ||
      inputSchemaSource === null ||
      Array.isArray(inputSchemaSource) ||
      // Crucially check for the required 'type' property
      inputSchemaSource.type !== 'object'
    ) {
      throw new MappingError(
        `Invalid parameters schema for tool '${
          tool.function.name
        }'. Anthropic requires a JSON Schema object with top-level 'type: "object"'. Received: ${JSON.stringify(
          inputSchemaSource
        )}`,
        Provider.Anthropic
      )
    }

    // Cast to the specific InputSchema type AFTER validation
    // Note: AnthropicToolType.Tool.InputSchema is the correct path
    const inputSchema: AnthropicToolType.InputSchema = inputSchemaSource as AnthropicToolType.InputSchema

    return {
      name: tool.function.name,
      description: tool.function.description,
      input_schema: inputSchema
    }
  })

  let toolChoice: Anthropic.Messages.ToolChoice | undefined = undefined
  if (params.toolChoice) {
    if (params.toolChoice === 'auto' || params.toolChoice === 'none') {
      toolChoice = { type: params.toolChoice }
    } else if (params.toolChoice === 'required') {
      toolChoice = { type: 'any' }
    } else if (typeof params.toolChoice === 'object' && params.toolChoice.type === 'function') {
      toolChoice = { type: 'tool', name: params.toolChoice.function.name }
    } else {
      throw new MappingError(
        `Unsupported tool_choice format for Anthropic: ${JSON.stringify(params.toolChoice)}`,
        Provider.Anthropic
      )
    }
  }

  let thinkingParam: AnthropicThinkingConfig | undefined = undefined
  if (params.thinking) {
    // FIX: Update budget_tokens to the minimum required value
    thinkingParam = { type: 'enabled', budget_tokens: 1024 }
  }

  // System prompt for Anthropic should be string | TextBlockParam[] according to types
  let systemParam: string | AnthropicTextBlockParam[] | undefined
  if (typeof systemPrompt === 'string') {
    systemParam = systemPrompt
  } else {
    systemParam = undefined // Handle case where systemPrompt might be mapped differently if needed
  }

  const basePayload = {
    model: params.model!,
    messages: messages,
    system: systemParam, // Use the potentially string system prompt
    max_tokens: params.maxTokens ?? 4096,
    temperature: params.temperature,
    top_p: params.topP,
    stop_sequences: Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined,
    tools: tools,
    tool_choice: toolChoice,
    ...(thinkingParam && { thinking: thinkingParam })
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

function mapUsageFromAnthropic(usage: Anthropic.Usage | undefined): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.input_tokens,
    completionTokens: usage.output_tokens,
    totalTokens: usage.input_tokens + usage.output_tokens
  }
}

// FIX: Use the specific AnthropicResponseContentBlock type for mapping results
function mapToolCallsFromAnthropic(
  contentBlocks: AnthropicResponseContentBlock[] | undefined
): RosettaToolCallRequest[] | undefined {
  if (!Array.isArray(contentBlocks)) return undefined

  // Filter specifically for ToolUseBlock which is part of the RESPONSE ContentBlock union
  const toolCalls: RosettaToolCallRequest[] = contentBlocks
    .filter((block): block is AnthropicToolUseBlock => block.type === 'tool_use')
    .map(block => ({
      id: block.id,
      type: 'function',
      function: {
        name: block.name,
        // Input is 'unknown', safely stringify
        arguments: JSON.stringify(block.input ?? {})
      }
    }))

  return toolCalls.length > 0 ? toolCalls : undefined
}

// FIX: Use the specific AnthropicMessage response type
export function mapFromAnthropicResponse(response: AnthropicMessage, model: string): GenerateResult {
  let combinedTextContent: string | null = null
  let thinkingText: string | null = null

  // Ensure content is treated as the RESPONSE ContentBlock array
  const responseContent = response.content as AnthropicResponseContentBlock[]

  if (Array.isArray(responseContent)) {
    const textParts: string[] = []
    responseContent.forEach(block => {
      // Check against RESPONSE block types
      if (block.type === 'text') {
        textParts.push(block.text)
      } else if (block.type === 'thinking' && typeof block.thinking === 'string') {
        thinkingText = block.thinking
      }
      // Ignore ToolUseBlock for combinedTextContent
    })
    if (textParts.length > 0) {
      combinedTextContent = textParts.join('')
    }
  }

  // Map tool calls from the response content
  const toolCalls = mapToolCallsFromAnthropic(responseContent)

  // Standardize finish reason
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

  return {
    content: combinedTextContent,
    toolCalls: toolCalls,
    finishReason: finishReason,
    usage: mapUsageFromAnthropic(response.usage),
    thinkingSteps: thinkingText,
    citations: undefined, // Citations would be in TextBlock.citations if enabled/supported
    parsedContent: null,
    model: response.model ?? model,
    rawResponse: response
  }
}

// --- Stream Mapping ---

export async function* mapAnthropicStream(stream: AnthropicMessageStream): AsyncIterable<StreamChunk> {
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
          yield { type: 'message_start', data: { provider: Provider.Anthropic, model: model } }
          currentUsage = mapUsageFromAnthropic(safeGet<Anthropic.Usage>(event.message, 'usage'))
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
          // Check against RESPONSE block types here
          if (event.content_block.type === 'thinking') {
            yield { type: 'thinking_start' }
            thinkingStarted = true
          } else if (event.content_block.type === 'tool_use') {
            const toolUse = event.content_block // Type is ToolUseBlock here
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
          // FIX: Correctly handle event type RawContentBlockStopEvent
          const stoppedEvent = event as RawContentBlockStopEvent // Cast for clarity
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
          // Yield thinking_stop if flag is set when *any* block stops
          if (thinkingStarted) {
            yield { type: 'thinking_stop' }
            thinkingStarted = false // Reset flag
          }
          break
        case 'message_delta':
          if (event.usage?.output_tokens) {
            currentUsage = {
              promptTokens: currentUsage?.promptTokens ?? 0,
              completionTokens: event.usage.output_tokens,
              totalTokens: (currentUsage?.promptTokens ?? 0) + event.usage.output_tokens
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
    const mappedError =
      error instanceof RosettaAIError
        ? error
        : error instanceof Anthropic.APIError
        ? new ProviderAPIError(
            error.message,
            Provider.Anthropic,
            error.status,
            safeGet<string>(error, 'error', 'type'),
            safeGet<string>(error, 'error', 'type'),
            error
          )
        : new ProviderAPIError(String(error), Provider.Anthropic, undefined, undefined, undefined, error)
    yield { type: 'error', data: { error: mappedError } }
  }
}
