/**
 * OpenAI Responses API Mapper
 *
 * Handles transformation between RosettaAI's Responses API types and OpenAI's native Responses API.
 * This is a separate interface from Chat Completions, designed for stateful, agent-ready interactions.
 */

import {
  CreateResponseParams,
  ResponseResult,
  ResponsesStreamChunk,
  ResponsesTool,
  ResponsesOutputItem,
  ResponsesToolCall,
  ResponsesInputItem
} from '../../types/responses.types'
import { Provider } from '../../types/common.types'
import {
  ComputerUseDecision,
  computerUseDecisionSchema,
  COMPUTER_USE_VIEWPORT_HEIGHT,
  COMPUTER_USE_VIEWPORT_WIDTH
} from '../../types'
import {
  ComputerUseMappingError,
  MappingError,
  InvalidToolDefinitionError,
  ToolArgumentValidationError
} from '../../errors'
import {
  Response,
  ResponseComputerToolCall,
  ResponseCreateParamsNonStreaming,
  ResponseCreateParamsStreaming,
  ResponseInputContent,
  ResponseInputItem,
  Tool
} from 'openai/resources/responses/responses'
import { z } from 'zod'
import { mapOpenAIComputerAction } from './openai.computer-use'
import { JSONSchema7 } from 'json-schema'

type MappedCreateResponseParams = ResponseCreateParamsNonStreaming | ResponseCreateParamsStreaming

function asOpenAIJsonSchema(schema: JSONSchema7): Record<string, unknown> {
  // `json-schema` and OpenAI independently declare the same JSON object boundary;
  // OpenAI's index signature is the only structural difference.
  return (schema as unknown) as Record<string, unknown>
}

function mapComputerScreenshot(
  output: Extract<ResponsesInputItem, { type: 'computer_call_output' }>['output']
): ResponseInputItem.ComputerCallOutput['output'] {
  if ('image' in output) {
    return {
      type: 'computer_screenshot',
      image_url: `data:${output.image.mimeType};base64,${output.image.base64Data}`
    }
  }
  if ('image_url' in output) return { type: 'computer_screenshot', image_url: output.image_url }
  return { type: 'computer_screenshot', file_id: output.file_id }
}

function mapComputerCallOutput(
  item: Extract<ResponsesInputItem, { type: 'computer_call_output' }>
): ResponseInputItem.ComputerCallOutput {
  return {
    type: 'computer_call_output',
    call_id: item.call_id,
    output: mapComputerScreenshot(item.output),
    ...(item.acknowledged_safety_checks
      ? { acknowledged_safety_checks: item.acknowledged_safety_checks.map(check => ({ ...check })) }
      : {})
  }
}

function mapResponsesInput(items: ResponsesInputItem[]): ResponseInputItem[] {
  const content: ResponseInputContent[] = []
  const topLevelItems: ResponseInputItem[] = []

  for (const item of items) {
    if (item.type === 'input_text') {
      content.push({ type: 'input_text', text: item.text })
    } else if (item.type === 'input_image') {
      const imageUrl =
        'image_url' in item ? item.image_url : `data:${item.image.mimeType};base64,${item.image.base64Data}`
      content.push({ type: 'input_image', image_url: imageUrl, detail: 'auto' })
    } else {
      topLevelItems.push(mapComputerCallOutput(item))
    }
  }

  return content.length > 0 ? [{ type: 'message', role: 'user', content }, ...topLevelItems] : topLevelItems
}

function mapResponsesTool(tool: ResponsesTool): Tool {
  if (tool.type === 'function') {
    if (!tool.name) throw new InvalidToolDefinitionError('Function tool missing name', 'unknown')
    if (!tool.parameters) throw new InvalidToolDefinitionError('Function tool missing parameters', tool.name)
    return {
      type: 'function',
      name: tool.name,
      description: tool.description,
      parameters: asOpenAIJsonSchema(tool.parameters),
      strict: tool.strict ?? null
    }
  }
  if (tool.type === 'computer') return { type: 'computer' }
  if (tool.type === 'web_search') return { type: 'web_search' }
  if (tool.type === 'file_search') return { type: 'file_search', vector_store_ids: tool.vector_store_ids ?? [] }

  // This legacy Rosetta image-generation shape predates the installed OpenAI declaration.
  // Preserve its existing wire mapping without allowing it into the new computer-use types.
  if (tool.type === 'image_generation') {
    return ({
      type: 'image_generation',
      ...(tool.options && { options: tool.options })
    } as unknown) as Tool
  }
  return { type: 'code_interpreter', container: { type: 'auto' } }
}

function mapResponsesToolChoice(
  choice: NonNullable<CreateResponseParams['tool_choice']>
): ResponseCreateParamsNonStreaming['tool_choice'] {
  if (typeof choice === 'string') return choice
  if (choice.type === 'function') return { type: 'function', name: choice.name }
  if (choice.type === 'web_search') {
    throw new MappingError(
      'Installed OpenAI Responses declarations do not support forcing the GA web_search tool',
      Provider.OpenAI
    )
  }
  return { type: choice.type }
}

/**
 * Maps CreateResponseParams to OpenAI Responses API parameters.
 */
export function mapToOpenAIResponsesParams(
  params: CreateResponseParams & { stream: true }
): ResponseCreateParamsStreaming
export function mapToOpenAIResponsesParams(
  params: CreateResponseParams & { stream?: false }
): ResponseCreateParamsNonStreaming
export function mapToOpenAIResponsesParams(params: CreateResponseParams): MappedCreateResponseParams {
  if (!params.model) throw new MappingError('Responses API model is required', Provider.OpenAI)
  if (params.stop !== undefined) {
    throw new MappingError('OpenAI Responses API does not declare a stop parameter', Provider.OpenAI)
  }

  const payload = {
    model: params.model,
    ...(params.instructions !== undefined ? { instructions: params.instructions } : {}),
    ...(params.input !== undefined
      ? { input: typeof params.input === 'string' ? params.input : mapResponsesInput(params.input) }
      : {}),
    ...(params.tools && params.tools.length > 0 ? { tools: params.tools.map(mapResponsesTool) } : {}),
    ...(params.tool_choice !== undefined ? { tool_choice: mapResponsesToolChoice(params.tool_choice) } : {}),
    ...(params.response_format
      ? {
          text: {
            format: {
              type: 'json_schema' as const,
              name: params.response_format.json_schema.name,
              strict: params.response_format.json_schema.strict,
              schema: asOpenAIJsonSchema(params.response_format.json_schema.schema)
            }
          }
        }
      : {}),
    ...(params.previous_response_id !== undefined ? { previous_response_id: params.previous_response_id } : {}),
    ...(params.max_tokens !== undefined ? { max_output_tokens: params.max_tokens } : {}),
    ...(params.temperature !== undefined ? { temperature: params.temperature } : {}),
    ...(params.top_p !== undefined ? { top_p: params.top_p } : {}),
    ...(params.metadata !== undefined ? { metadata: params.metadata } : {})
  }

  return params.stream ? { ...payload, stream: true } : { ...payload, stream: false }
}

/**
 * Maps one GA native call only after its shape is proven truthfully acknowledgeable.
 *
 * @throws {ComputerUseMappingError} If the call shape, batch size, or canonical decision is invalid.
 */
export function mapOpenAIComputerCallToDecision(
  call: ResponseComputerToolCall,
  responseId: string
): ComputerUseDecision {
  const hasSingularAction = Object.prototype.hasOwnProperty.call(call, 'action')
  const hasActions = Object.prototype.hasOwnProperty.call(call, 'actions')
  if (hasSingularAction || !hasActions) {
    throw new ComputerUseMappingError(
      'PROVIDER_ACTION_SHAPE_UNSUPPORTED',
      'OpenAI V1 requires actions[] and rejects singular or mixed action shapes',
      Provider.OpenAI
    )
  }
  if (!Array.isArray(call.actions) || call.actions.length !== 1) {
    throw new ComputerUseMappingError(
      'PROVIDER_ACTION_BATCH_UNSUPPORTED',
      'OpenAI V1 requires exactly one action in actions[]',
      Provider.OpenAI
    )
  }

  const action = call.actions[0]
  if (!action) {
    throw new ComputerUseMappingError(
      'PROVIDER_ACTION_BATCH_UNSUPPORTED',
      'OpenAI V1 requires exactly one action in actions[]',
      Provider.OpenAI
    )
  }

  const decision = {
    schemaVersion: '1' as const,
    actionId: call.call_id,
    actions: [
      mapOpenAIComputerAction(action, 'pixels', {
        width: COMPUTER_USE_VIEWPORT_WIDTH,
        height: COMPUTER_USE_VIEWPORT_HEIGHT
      })
    ] as const,
    providerTraceId: call.id,
    responseId,
    pendingSafetyChecks: call.pending_safety_checks.map(check => ({
      id: check.id,
      code: check.code === undefined ? null : check.code,
      message: check.message === undefined ? null : check.message
    }))
  }

  const parsed = computerUseDecisionSchema.safeParse(decision)
  if (!parsed.success) {
    throw new ComputerUseMappingError(
      'PROVIDER_ACTION_INVALID',
      'OpenAI computer call failed canonical decision validation',
      Provider.OpenAI,
      parsed.error
    )
  }
  return parsed.data
}

function validateFunctionToolCall(toolCall: ResponsesToolCall, originalTools: ResponsesTool[] | undefined): void {
  const matchingTool = originalTools?.find(tool => tool.type === 'function' && tool.name === toolCall.function.name)
  if (!matchingTool || matchingTool.type !== 'function' || !matchingTool.zodSchema) return

  let parsedArgs: unknown
  try {
    parsedArgs = JSON.parse(toolCall.function.arguments)
  } catch (error) {
    throw new MappingError(
      `Failed to parse tool arguments JSON for '${toolCall.function.name}'`,
      Provider.OpenAI,
      'function_call',
      error
    )
  }

  const validation = matchingTool.zodSchema.safeParse(parsedArgs)
  if (!validation.success) {
    throw new ToolArgumentValidationError(
      `Tool '${toolCall.function.name}' arguments validation failed`,
      validation.error.issues,
      toolCall.function.name,
      toolCall.id,
      parsedArgs
    )
  }
}

/**
 * Maps OpenAI Responses API response to ResponseResult.
 */
export function mapFromOpenAIResponsesResponse(response: Response, originalTools?: ResponsesTool[]): ResponseResult {
  const output: ResponsesOutputItem[] = []
  const toolCalls: ResponsesToolCall[] = []

  for (const item of response.output) {
    if (item.type === 'message') {
      for (const content of item.content) {
        if (content.type === 'output_text') output.push({ type: 'output_text', text: content.text })
      }
    } else if (item.type === 'function_call') {
      const toolCall: ResponsesToolCall = {
        ...(item.id !== undefined ? { id: item.id } : {}),
        call_id: item.call_id,
        type: 'function',
        function: { name: item.name, arguments: item.arguments }
      }
      validateFunctionToolCall(toolCall, originalTools)
      toolCalls.push(toolCall)
      output.push({
        type: 'function_call',
        ...(item.id !== undefined ? { id: item.id } : {}),
        call_id: item.call_id,
        name: item.name,
        arguments: item.arguments
      })
    } else if (item.type === 'computer_call') {
      output.push({
        type: 'computer_call',
        status: item.status,
        decision: mapOpenAIComputerCallToDecision(item, response.id)
      })
    } else if (item.type === 'image_generation_call' && item.result) {
      output.push({ type: 'image', image_url: `data:image/png;base64,${item.result}` })
    }
  }

  return {
    id: response.id,
    output,
    output_text: response.output_text,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
    ...(response.usage
      ? {
          usage: {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            total_tokens: response.usage.total_tokens
          }
        }
      : {}),
    model: response.model,
    rawResponse: response
  }
}

/**
 * Maps OpenAI Responses API streaming events to ResponsesStreamChunk.
 */
export async function* mapOpenAIResponsesStream(
  stream: AsyncIterable<any>,
  originalTools?: ResponsesTool[]
): AsyncIterable<ResponsesStreamChunk> {
  const accumulatedResult: Partial<ResponseResult> = {
    output: [],
    output_text: ''
  }
  let currentToolCall: { id: string; name: string; arguments: string } | null = null
  let reasoningSummary = ''
  let sawReasoningSummaryDelta = false
  let sawReasoningTextDelta = false
  let reasoningSummaryFlushed = false

  const flushReasoningSummary = (): ResponsesStreamChunk[] => {
    if (sawReasoningTextDelta || reasoningSummaryFlushed || reasoningSummary.length === 0) return []
    reasoningSummaryFlushed = true
    return [{ type: 'thinking_delta', data: { delta: reasoningSummary } }]
  }

  try {
    for await (const event of stream) {
      // OpenAI Responses API uses event.type for semantic events
      const eventType = event.type || event.event

      if (!eventType) {
        // Skip events without a type
        continue
      }

      // Map to our semantic event types
      if (eventType === 'response.created') {
        accumulatedResult.id = event.response?.id || event.id
        accumulatedResult.model = event.response?.model || event.model
        yield {
          type: 'response.created',
          data: {
            id: accumulatedResult.id!,
            model: accumulatedResult.model!
          }
        }
      } else if (eventType === 'response.reasoning_summary_text.delta') {
        if (!sawReasoningTextDelta && typeof event.delta === 'string') {
          sawReasoningSummaryDelta = true
          reasoningSummary += event.delta
        }
      } else if (eventType === 'response.reasoning_summary_text.done') {
        if (!sawReasoningTextDelta && !sawReasoningSummaryDelta && typeof event.text === 'string') {
          reasoningSummary = event.text
        }
      } else if (eventType === 'response.reasoning_text.delta') {
        if (typeof event.delta === 'string' && event.delta.length > 0) {
          if (!sawReasoningTextDelta) {
            sawReasoningTextDelta = true
            reasoningSummary = ''
          }
          yield { type: 'thinking_delta', data: { delta: event.delta } }
        }
      } else if (eventType === 'response.reasoning_text.done') {
        if (!sawReasoningTextDelta && typeof event.text === 'string' && event.text.length > 0) {
          sawReasoningTextDelta = true
          reasoningSummary = ''
          yield { type: 'thinking_delta', data: { delta: event.text } }
        }
      } else if (eventType === 'response.output_text.delta' || eventType === 'content.delta') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const delta = event.delta || event.text || ''
        accumulatedResult.output_text = (accumulatedResult.output_text || '') + delta
        yield {
          type: 'response.output_text.delta',
          data: { delta }
        }
      } else if (eventType === 'response.output_text.done' || eventType === 'content.done') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const text = event.text || accumulatedResult.output_text || ''
        yield {
          type: 'response.output_text.done',
          data: { text }
        }
      } else if (eventType === 'response.tool_call.start' || eventType === 'tool_call.start') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const toolCall = event.tool_call || event
        currentToolCall = {
          id: toolCall.id,
          name: toolCall.name || toolCall.function?.name,
          arguments: ''
        }
        yield {
          type: 'response.tool_call.start',
          data: {
            id: currentToolCall.id,
            name: currentToolCall.name
          }
        }
      } else if (eventType === 'response.tool_call.delta' || eventType === 'tool_call.delta') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const delta = event.delta || event.arguments || ''
        if (currentToolCall) {
          currentToolCall.arguments += delta
        }
        yield {
          type: 'response.tool_call.delta',
          data: {
            id: event.tool_call?.id || event.id || currentToolCall?.id || '',
            delta
          }
        }
      } else if (eventType === 'response.tool_call.done' || eventType === 'tool_call.done') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const toolCall = event.tool_call || currentToolCall
        if (toolCall) {
          // Validate tool arguments if original tools were provided
          if (originalTools && originalTools.length > 0) {
            const matchingTool = originalTools.find(t => t.type === 'function' && t.name === toolCall.name)
            if (matchingTool && matchingTool.type === 'function' && matchingTool.zodSchema) {
              try {
                const parsedArgs = JSON.parse(toolCall.arguments)
                matchingTool.zodSchema.parse(parsedArgs)
              } catch (error) {
                if (error instanceof z.ZodError) {
                  const validationError = new ToolArgumentValidationError(
                    `Tool '${toolCall.name}' arguments validation failed`,
                    error.issues,
                    toolCall.name,
                    toolCall.id
                  )
                  yield {
                    type: 'error',
                    data: { error: validationError }
                  }
                  return
                }
              }
            }
          }

          yield {
            type: 'response.tool_call.done',
            data: {
              id: toolCall.id,
              name: toolCall.name,
              arguments: toolCall.arguments
            }
          }
          currentToolCall = null
        }
      } else if (eventType === 'response.completed' || eventType === 'done') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const finalResponse = event.response || event
        const result = mapFromOpenAIResponsesResponse(finalResponse, originalTools)
        yield {
          type: 'response.completed',
          data: result
        }
      } else if (eventType === 'response.failed' || eventType === 'error') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        const error = event.error || event
        yield {
          type: 'response.failed',
          data: {
            error: {
              message: error.message || 'Unknown error',
              code: error.code
            }
          }
        }
      } else if (eventType === 'response.cancelled') {
        for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
        yield {
          type: 'response.cancelled',
          data: {
            reason: event.reason
          }
        }
      }
      // Ignore other event types
    }
    for (const reasoningChunk of flushReasoningSummary()) yield reasoningChunk
  } catch (error) {
    yield {
      type: 'error',
      data: {
        error: error instanceof Error ? error : new Error(String(error))
      }
    }
  }
}
