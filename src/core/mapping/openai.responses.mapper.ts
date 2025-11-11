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
  ResponsesToolCall
} from '../../types/responses.types'
import { Provider } from '../../types/common.types'
import { MappingError, InvalidToolDefinitionError, ToolArgumentValidationError } from '../../errors'
import { z } from 'zod'

/**
 * Maps CreateResponseParams to OpenAI Responses API parameters.
 */
export function mapToOpenAIResponsesParams(params: CreateResponseParams): any {
  // Map input
  let mappedInput: any
  if (typeof params.input === 'string') {
    mappedInput = params.input
  } else if (Array.isArray(params.input)) {
    mappedInput = params.input.map(item => {
      if (item.type === 'input_text') {
        return { type: 'input_text', text: item.text }
      } else if (item.type === 'input_image') {
        if ('image_url' in item) {
          return { type: 'input_image', image_url: item.image_url }
        } else if ('image' in item) {
          // Convert RosettaImageData to OpenAI format
          const dataUri = `data:${item.image.mimeType};base64,${item.image.base64Data}`
          return { type: 'input_image', image_url: dataUri }
        }
      }
      throw new MappingError(`Unsupported input item type: ${(item as any).type}`, Provider.OpenAI)
    })
  }

  // Map tools
  let mappedTools: any[] | undefined
  if (params.tools && params.tools.length > 0) {
    mappedTools = params.tools.map(tool => {
      if (tool.type === 'function') {
        // Validate function tool has required fields
        if (!tool.name) {
          throw new InvalidToolDefinitionError('Function tool missing name', 'unknown')
        }
        if (!tool.parameters) {
          throw new InvalidToolDefinitionError('Function tool missing parameters', tool.name)
        }
        return {
          type: 'function',
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters
        }
      } else if (tool.type === 'web_search') {
        return { type: 'web_search' }
      } else if (tool.type === 'file_search') {
        return {
          type: 'file_search',
          vector_store_ids: tool.vector_store_ids
        }
      } else if (tool.type === 'image_generation') {
        return {
          type: 'image_generation',
          ...(tool.options && { options: tool.options })
        }
      } else if (tool.type === 'code_interpreter') {
        return { type: 'code_interpreter' }
      }
      throw new InvalidToolDefinitionError(`Unsupported tool type: ${(tool as any).type}`, 'unknown')
    })
  }

  // Map tool_choice
  let mappedToolChoice: any
  if (params.tool_choice) {
    if (typeof params.tool_choice === 'string') {
      mappedToolChoice = params.tool_choice
    } else if (typeof params.tool_choice === 'object') {
      mappedToolChoice = {
        type: params.tool_choice.type,
        ...(params.tool_choice.name && { name: params.tool_choice.name })
      }
    }
  }

  // Map response_format
  let mappedResponseFormat: any
  if (params.response_format) {
    mappedResponseFormat = {
      type: 'json_schema',
      json_schema: {
        name: params.response_format.json_schema.name,
        strict: params.response_format.json_schema.strict,
        schema: params.response_format.json_schema.schema
      }
    }
  }

  // Build the request payload
  const payload: any = {
    model: params.model,
    ...(params.instructions && { instructions: params.instructions }),
    ...(mappedInput && { input: mappedInput }),
    ...(mappedTools && { tools: mappedTools }),
    ...(mappedToolChoice && { tool_choice: mappedToolChoice }),
    ...(mappedResponseFormat && { response_format: mappedResponseFormat }),
    ...(params.previous_response_id && { previous_response_id: params.previous_response_id }),
    ...(params.max_tokens && { max_tokens: params.max_tokens }),
    ...(params.temperature !== undefined && { temperature: params.temperature }),
    ...(params.top_p !== undefined && { top_p: params.top_p }),
    ...(params.stop && { stop: params.stop }),
    ...(params.metadata && { metadata: params.metadata }),
    stream: params.stream ?? false
  }

  return payload
}

/**
 * Maps OpenAI Responses API response to ResponseResult.
 */
export function mapFromOpenAIResponsesResponse(
  response: any,
  originalTools?: ResponsesTool[]
): ResponseResult {
  // Validate response structure
  if (!response || typeof response !== 'object') {
    throw new MappingError('Invalid response from OpenAI Responses API', Provider.OpenAI)
  }

  // Extract output items
  const output: ResponsesOutputItem[] = []
  let outputText = ''

  if (Array.isArray(response.output)) {
    for (const item of response.output) {
      if (item.type === 'text' || item.type === 'output_text') {
        const text = item.text || item.content || ''
        output.push({ type: 'output_text', text })
        outputText += text
      } else if (item.type === 'function_call') {
        output.push({
          type: 'function_call',
          id: item.id,
          name: item.name,
          arguments: item.arguments
        })
      } else if (item.type === 'image') {
        output.push({
          type: 'image',
          image_url: item.image_url || item.url
        })
      }
    }
  }

  // Handle convenience accessor if available
  if (response.output_text) {
    outputText = response.output_text
  }

  // Extract tool calls
  let toolCalls: ResponsesToolCall[] | undefined
  if (response.tool_calls && Array.isArray(response.tool_calls)) {
    toolCalls = response.tool_calls.map((tc: any) => ({
      id: tc.id,
      type: 'function',
      function: {
        name: tc.function?.name || tc.name,
        arguments: tc.function?.arguments || tc.arguments || '{}'
      }
    }))

    // Validate tool call arguments if original tools were provided
    if (originalTools && originalTools.length > 0 && toolCalls) {
      for (const toolCall of toolCalls) {
        const matchingTool = originalTools.find(
          t => t.type === 'function' && t.name === toolCall.function.name
        )
        if (matchingTool && matchingTool.type === 'function' && matchingTool.zodSchema) {
          try {
            const parsedArgs = JSON.parse(toolCall.function.arguments)
            matchingTool.zodSchema.parse(parsedArgs)
          } catch (error) {
            if (error instanceof z.ZodError) {
              throw new ToolArgumentValidationError(
                `Tool '${toolCall.function.name}' arguments validation failed`,
                error.issues,
                toolCall.function.name,
                toolCall.id
              )
            } else if (error instanceof SyntaxError) {
              throw new MappingError(
                `Failed to parse tool arguments JSON for '${toolCall.function.name}': ${error.message}`,
                Provider.OpenAI
              )
            }
            throw error
          }
        }
      }
    }
  }

  // Extract usage
  let usage: ResponseResult['usage']
  if (response.usage) {
    usage = {
      input_tokens: response.usage.input_tokens || response.usage.prompt_tokens || 0,
      output_tokens: response.usage.output_tokens || response.usage.completion_tokens || 0,
      total_tokens: response.usage.total_tokens || 0
    }
  }

  return {
    id: response.id,
    output,
    output_text: outputText,
    tool_calls: toolCalls,
    usage,
    model: response.model,
    finish_reason: response.finish_reason || response.stop_reason,
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
  let accumulatedResult: Partial<ResponseResult> = {
    output: [],
    output_text: ''
  }
  let currentToolCall: { id: string; name: string; arguments: string } | null = null

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
      } else if (eventType === 'response.output_text.delta' || eventType === 'content.delta') {
        const delta = event.delta || event.text || ''
        accumulatedResult.output_text = (accumulatedResult.output_text || '') + delta
        yield {
          type: 'response.output_text.delta',
          data: { delta }
        }
      } else if (eventType === 'response.output_text.done' || eventType === 'content.done') {
        const text = event.text || accumulatedResult.output_text || ''
        yield {
          type: 'response.output_text.done',
          data: { text }
        }
      } else if (eventType === 'response.tool_call.start' || eventType === 'tool_call.start') {
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
        const toolCall = event.tool_call || currentToolCall
        if (toolCall) {
          // Validate tool arguments if original tools were provided
          if (originalTools && originalTools.length > 0) {
            const matchingTool = originalTools.find(
              t => t.type === 'function' && t.name === toolCall.name
            )
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
        const finalResponse = event.response || event
        const result = mapFromOpenAIResponsesResponse(finalResponse, originalTools)
        yield {
          type: 'response.completed',
          data: result
        }
      } else if (eventType === 'response.failed' || eventType === 'error') {
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
        yield {
          type: 'response.cancelled',
          data: {
            reason: event.reason
          }
        }
      }
      // Ignore other event types
    }
  } catch (error) {
    yield {
      type: 'error',
      data: {
        error: error instanceof Error ? error : new Error(String(error))
      }
    }
  }
}
