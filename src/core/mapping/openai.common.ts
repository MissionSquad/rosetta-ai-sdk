import OpenAI from 'openai'
import {
  ChatCompletionContentPart as OpenAIContentPart,
  ChatCompletionContentPartText,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartRefusal,
  ChatCompletionRole as OpenAIRole,
  ChatCompletionMessageToolCall as OpenAIToolCall,
  ChatCompletion,
  ChatCompletionChunk
} from 'openai/resources/chat/completions'
import { Stream } from 'openai/streaming'
import {
  GenerateResult,
  Provider,
  RosettaMessage,
  RosettaToolCallRequest,
  StreamChunk,
  TokenUsage,
  RosettaTool // Import RosettaTool
} from '../../types'
import {
  InvalidToolDefinitionError,
  MappingError,
  ProviderAPIError,
  RosettaAIError,
  ToolArgumentValidationError
} from '../../errors'
import { safeGet } from '../utils'
import { mapTokenUsage } from './common.utils'

export function mapRoleToOpenAI(role: RosettaMessage['role']): OpenAIRole {
  switch (role) {
    case 'system':
      return 'system'
    case 'user':
      return 'user'
    case 'assistant':
      return 'assistant'
    case 'tool':
      return 'tool'
    default:
      // Ensure exhaustive check works with `never`
      const _e: never = role
      throw new MappingError(`Unsupported role: ${_e}`, Provider.OpenAI)
  }
}

export function mapContentForOpenAIRole(
  content: RosettaMessage['content'],
  role: OpenAIRole
): string | OpenAIContentPart[] | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal> | null {
  // Perform role-specific null/empty checks FIRST
  if (content === null) {
    if (role === 'assistant' || role === 'tool') return null // Allowed for assistant/tool
    throw new MappingError(`Role '${role}' requires non-null content.`, Provider.OpenAI)
  }
  if (content === '') {
    // Allow empty string for user, assistant, and tool roles
    if (role === 'system') {
      throw new MappingError(`Role '${role}' requires non-empty string content.`, Provider.OpenAI)
    }
    // Allow empty string for user, assistant, and tool roles
  }
  if (Array.isArray(content) && content.length === 0) {
    if (role === 'system' || role === 'tool') {
      throw new MappingError(`Role '${role}' requires non-empty content array. Received empty array.`, Provider.OpenAI)
    }
    // Allow empty array for user/assistant (maps to [] or null below)
  }

  if (typeof content === 'string') {
    // Allow empty string for roles that permit string content (user, assistant, tool)
    return content
  }

  const mappedParts: Array<ChatCompletionContentPartText | ChatCompletionContentPartImage> = content.map(part => {
    if (part.type === 'text') {
      return { type: 'text', text: part.text } as ChatCompletionContentPartText
    } else if (part.type === 'image') {
      if (role !== 'user') {
        throw new MappingError(`Image content parts only allowed for 'user' role, not '${role}'.`, Provider.OpenAI)
      }
      return {
        type: 'image_url',
        image_url: { url: `data:${part.image.mimeType};base64,${part.image.base64Data}` }
      } as ChatCompletionContentPartImage
    } else {
      // Ensure exhaustive check works with `never`
      const _e: never = part
      throw new MappingError(`Unsupported content part type: ${(_e as any).type}`, Provider.OpenAI)
    }
  })

  if (role === 'user') {
    return mappedParts
  } else if (role === 'assistant') {
    const assistantParts = mappedParts.filter((p): p is ChatCompletionContentPartText => p.type === 'text')
    if (assistantParts.length !== mappedParts.length) {
      console.warn(`Non-text content parts filtered out for assistant message.`)
    }
    // Return null if no text parts remain (e.g., input was [])
    return assistantParts.length > 0 ? assistantParts : null
  } else if (role === 'system' || role === 'developer') {
    const textParts = mappedParts.filter((p): p is ChatCompletionContentPartText => p.type === 'text')
    if (textParts.length !== mappedParts.length) {
      throw new MappingError(`Role '${role}' content array can only contain text parts.`, Provider.OpenAI)
    }
    // If input was [], textParts will be [], which is handled by the check at the start.
    return textParts
  } else if (role === 'tool') {
    const textParts = mappedParts.filter(p => p.type === 'text').map(p => (p as ChatCompletionContentPartText).text)
    if (textParts.length !== mappedParts.length) {
      console.warn(`Tool message content contained non-text parts. Stringifying.`)
      try {
        return JSON.stringify(mappedParts)
      } catch {
        throw new MappingError(`Could not stringify complex tool content.`, Provider.OpenAI)
      }
    }
    // If input was [], textParts will be [], join results in "". Handled by check at start.
    const joinedText = textParts.join('')
    // Empty string check moved to the beginning
    // if (joinedText === '') {
    //   throw new MappingError(`Role 'tool' requires non-empty string content.`, Provider.OpenAI)
    // }
    return joinedText
  } else {
    throw new MappingError(`Cannot map content parts for unhandled role '${role}'.`, Provider.OpenAI)
  }
}

function mapAndValidateToolCallsFromOpenAI(
  toolCalls: OpenAIToolCall[] | undefined,
  originalTools?: RosettaTool<any>[]
): RosettaToolCallRequest[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined

  const mappedCalls: RosettaToolCallRequest[] = []
  for (const tc of toolCalls) {
    if (tc.type !== 'function' || !tc.function?.name || !tc.id) {
      console.warn(`Skipping invalid tool call structure from OpenAI: ${JSON.stringify(tc)}`)
      continue
    }

    const toolDefinition = originalTools?.find(t => t.function.name === tc.function.name)
    const rawArguments = tc.function.arguments ?? '{}'

    if (!toolDefinition) {
      console.warn(`Received tool call for unknown tool '${tc.function.name}'. Skipping validation.`)
    } else {
      // Validate arguments using Zod schema
      let parsedArgs: any
      try {
        parsedArgs = JSON.parse(rawArguments)
      } catch (parseError) {
        throw new MappingError(
          `Failed to parse arguments for tool '${tc.function.name}' (ID: ${tc.id})`,
          Provider.OpenAI,
          'mapAndValidateToolCallsFromOpenAI validation',
          parseError
        )
      }

      const validationResult = toolDefinition.function.zodSchema.safeParse(parsedArgs)
      if (!validationResult.success) {
        throw new ToolArgumentValidationError(
          `Arguments failed validation for tool '${tc.function.name}'.`,
          validationResult.error.issues,
          tc.function.name,
          tc.id
        )
      }
    }

    // Arguments are valid (or validation skipped), add the raw tool call request
    mappedCalls.push({
      id: tc.id,
      type: tc.type,
      function: { name: tc.function.name, arguments: rawArguments } // Return raw string args
    })
  }
  return mappedCalls.length > 0 ? mappedCalls : undefined
}

export function mapFromOpenAIResponse(
  response: ChatCompletion,
  modelUsed: string,
  originalTools?: RosettaTool<any>[] // Accept original tools for validation
): GenerateResult {
  // Add null/undefined check for response.choices before accessing index 0
  const choice = response?.choices?.[0]
  if (!choice) {
    console.warn('OpenAI response missing choices.')
    const finishReason =
      safeGet<string>(response, 'choices', 0, 'finish_reason') ??
      safeGet<string>(response, 'prompt_annotations', 0, 'content_filter', 'reason') ??
      'error'
    return {
      content: null,
      toolCalls: undefined,
      finishReason,
      usage: mapTokenUsage(response?.usage), // Use common utility, handle potential null response
      citations: undefined,
      parsedContent: null,
      thinkingSteps: undefined,
      model: response?.model ?? modelUsed,
      rawResponse: response
    }
  }
  let parsedJson: any = null
  const textContent = choice.message?.content ?? null
  if (textContent && choice.finish_reason !== 'tool_calls') {
    const isJsonLike = textContent.trim().startsWith('{') || textContent.trim().startsWith('[')
    if (isJsonLike)
      try {
        parsedJson = JSON.parse(textContent)
      } catch (e) {
        console.warn('Failed to auto-parse potential JSON from OpenAI:', e)
      }
  }

  // Map and validate tool calls
  const mappedToolCalls = mapAndValidateToolCallsFromOpenAI(choice.message?.tool_calls, originalTools)
  return {
    content: textContent,
    toolCalls: mappedToolCalls, // Use validated (but still raw) tool calls
    finishReason: choice.finish_reason,
    usage: mapTokenUsage(response.usage), // Use common utility
    citations: undefined,
    parsedContent: parsedJson,
    thinkingSteps: undefined,
    model: response.model ?? modelUsed,
    rawResponse: response
  }
}

export function wrapOpenAIError(error: unknown, provider: Provider): RosettaAIError {
  // Handle specific validation errors first
  if (error instanceof ToolArgumentValidationError || error instanceof InvalidToolDefinitionError) {
    return error
  }
  if (error instanceof RosettaAIError) {
    return error
  }
  if (error instanceof OpenAI.APIError) {
    let message = 'Unknown OpenAI API Error'
    const nestedErrorObj = error.error as any
    const nestedMessage = nestedErrorObj?.message
    if (nestedMessage && typeof nestedMessage === 'string' && nestedMessage.trim()) {
      message = nestedMessage.trim()
    } else if (error.message && typeof error.message === 'string' && error.message.trim()) {
      message = error.message.trim()
    } else if (nestedErrorObj) {
      try {
        const stringifiedBody = JSON.stringify(nestedErrorObj)
        if (stringifiedBody !== '{}') message = stringifiedBody
      } catch {}
    }
    return new ProviderAPIError(message, provider, error.status, error.code, error.type, error)
  }
  // Handle non-Error objects better in this specific wrapper
  if (error instanceof Error) {
    return new ProviderAPIError(error.message, provider, undefined, undefined, undefined, error)
  }
  // Fallback for non-Error types, attempt JSON.stringify
  let errorMessage = 'Unknown error occurred'
  if (error !== null && typeof error === 'object') {
    try {
      errorMessage = JSON.stringify(error)
    } catch {
      errorMessage = String(error) // Fallback to String() if stringify fails
    }
  } else {
    errorMessage = String(error ?? errorMessage)
  }
  return new ProviderAPIError(errorMessage, provider, undefined, undefined, undefined, error)
}

export async function* mapOpenAIStream(
  stream: Stream<ChatCompletionChunk>,
  provider: Provider,
  modelId: string, // Accept modelId as argument
  originalTools?: RosettaTool<any>[] // Accept original tools for validation
): AsyncIterable<StreamChunk> {
  let accumulatedContent = ''
  const accumulatedToolCalls: Record<
    number,
    Partial<RosettaToolCallRequest & { index: number; function: { name: string; arguments: string } }>
  > = {}
  let finalUsage: TokenUsage | undefined
  let finalFinishReason: string | null = null
  let isJsonMode = false
  let messageStartYielded = false // Track if message_start has been yielded

  // Initialize aggregatedResult immediately using the passed modelId
  const aggregatedResult: GenerateResult | null = {
    content: '',
    toolCalls: [],
    finishReason: null,
    usage: undefined,
    model: modelId, // Use passed modelId initially
    thinkingSteps: null,
    citations: undefined,
    parsedContent: null,
    rawResponse: undefined // Raw response isn't available in stream aggregation
  }

  try {
    for await (const chunk of stream) {
      // Yield message_start only once, when the first relevant chunk arrives
      if (!messageStartYielded && (chunk.choices[0]?.delta || chunk.model)) {
        // Update model in aggregatedResult if a more specific one arrives in the stream
        if (aggregatedResult && chunk.model && aggregatedResult.model !== chunk.model) {
          aggregatedResult.model = chunk.model
        }
        yield { type: 'message_start', data: { provider, model: aggregatedResult?.model ?? modelId } }
        messageStartYielded = true
      }

      // Update usage if present and aggregatedResult exists
      if (aggregatedResult && chunk.usage) {
        finalUsage = mapTokenUsage(chunk.usage) // Use common utility
        aggregatedResult.usage = finalUsage
        // Allow loop to continue to check for other fields like delta.content
        // in the same chunk, as some providers (like Novita) send them together.
        // Removed: continue
      }

      // Check for choices AFTER processing potential usage in the same chunk
      const choice = chunk.choices[0]
      if (!choice) continue

      // --- Content Processing ---
      if (choice.delta?.content) {
        const delta = choice.delta.content
        accumulatedContent += delta
        if (aggregatedResult) aggregatedResult.content = accumulatedContent // Update aggregated content

        // Detect JSON mode on the fly
        if (!isJsonMode && accumulatedContent.trim().match(/^[{[]/)) {
          isJsonMode = true
        }

        if (isJsonMode) {
          let partialParsed = undefined
          try {
            partialParsed = JSON.parse(accumulatedContent)
          } catch {}
          yield { type: 'json_delta', data: { delta, parsed: partialParsed, snapshot: accumulatedContent } }
        } else {
          yield { type: 'content_delta', data: { delta } }
        }
      }

      // --- Tool Call Processing ---
      if (choice.delta?.tool_calls) {
        for (const tcChunk of choice.delta.tool_calls) {
          const index = tcChunk.index
          if (typeof index !== 'number') continue
          if (!accumulatedToolCalls[index])
            accumulatedToolCalls[index] = { index, function: { name: '', arguments: '' } }
          const currentTool = accumulatedToolCalls[index]
          if (tcChunk.id) currentTool.id = tcChunk.id
          if (tcChunk.type) currentTool.type = tcChunk.type as 'function'
          if (tcChunk.function?.name) currentTool.function!.name = tcChunk.function.name
          if (tcChunk.function?.arguments) currentTool.function!.arguments += tcChunk.function.arguments
          if (currentTool.id && currentTool.function?.name && !(currentTool as any).yieldedStart) {
            ;(currentTool as any).yieldedStart = true
            yield {
              type: 'tool_call_start',
              data: {
                index,
                toolCall: { id: currentTool.id, type: 'function', function: { name: currentTool.function.name } }
              }
            }
          }
          if (tcChunk.function?.arguments) {
            yield {
              type: 'tool_call_delta',
              data: {
                index,
                id: currentTool.id ?? `unk_openai_${index}`,
                functionArgumentChunk: tcChunk.function.arguments
              }
            }
          }
        }
      }

      // --- Finish Reason Processing ---
      if (choice.finish_reason) {
        const reason = choice.finish_reason
        finalFinishReason = reason
        if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason // Update aggregated reason

        // Validate completed tool calls
        for (const tc of Object.values(accumulatedToolCalls)) {
          if (!(tc as any).yieldedDone && tc.id && tc.index !== undefined && tc.function?.name) {
            const toolDefinition = originalTools?.find(t => t.function.name === tc.function!.name)
            const rawArguments = tc.function.arguments ?? '{}'

            if (toolDefinition) {
              let parsedArgs: any
              try {
                parsedArgs = JSON.parse(rawArguments)
              } catch (parseError) {
                throw new MappingError(
                  `Failed to parse arguments for tool '${tc.function!.name}' (ID: ${tc.id})`,
                  provider,
                  'mapOpenAIStream validation',
                  parseError
                )
              }
              const validationResult = toolDefinition.function.zodSchema.safeParse(parsedArgs)
              if (!validationResult.success) {
                throw new ToolArgumentValidationError(
                  `Streamed arguments failed validation for tool '${tc.function!.name}'.`,
                  validationResult.error.issues,
                  tc.function!.name,
                  tc.id
                )
              }
            } else {
              console.warn(`Skipping validation for unknown streamed tool '${tc.function!.name}'.`)
            }

            // Yield done chunk after validation (or skipping)
            yield { type: 'tool_call_done', data: { index: tc.index, id: tc.id } }
            ;(tc as any).yieldedDone = true

            // Add raw tool call to aggregated result
            if (aggregatedResult) {
              aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
              aggregatedResult.toolCalls.push({
                id: tc.id,
                type: tc.type as 'function',
                function: { name: tc.function.name, arguments: rawArguments } // Store raw args
              })
            }
          }
        }

        if (isJsonMode) {
          let finalParsedJson = null
          try {
            finalParsedJson = JSON.parse(accumulatedContent ?? '')
          } catch {}
          yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: accumulatedContent ?? '' } }
          if (aggregatedResult) aggregatedResult.parsedContent = finalParsedJson // Update aggregated parsed content
        }
      }
    } // End for await loop

    // --- Stream End Logic ---
    finalFinishReason = finalFinishReason ?? 'stop' // Default finish reason
    if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason // Ensure final reason is set

    // Yield message_stop
    yield { type: 'message_stop', data: { finishReason: finalFinishReason } }

    // Yield final_usage if available
    if (finalUsage) {
      yield { type: 'final_usage', data: { usage: finalUsage } }
    }

    // Yield final_result if aggregation was successful
    if (aggregatedResult) {
      // Final cleanup before yielding result
      if (!isJsonMode && aggregatedResult.content === '') aggregatedResult.content = null
      if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
      yield { type: 'final_result', data: { result: aggregatedResult } }
    } else {
      // This case should be less likely now, but keep the warning
      console.warn('OpenAI stream finished but no aggregated result was built.')
    }
  } catch (error) {
    // Wrap and yield errors, including ToolArgumentValidationError
    const mappedError = wrapOpenAIError(error, provider)
    yield { type: 'error', data: { error: mappedError } }
  }
}
