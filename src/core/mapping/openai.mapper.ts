import OpenAI from 'openai'
import { CompletionUsage } from 'openai/resources/completions'
import {
  ChatCompletionToolChoiceOption as OpenAIToolChoiceOption,
  ChatCompletionContentPart as OpenAIContentPart, // General part type
  ChatCompletionContentPartText, // Specific text part
  ChatCompletionContentPartImage, // Specific image part (for user messages)
  ChatCompletionContentPartRefusal, // Specific refusal part (for assistant messages)
  ChatCompletionTool as OpenAITool,
  ChatCompletionRole as OpenAIRole,
  ChatCompletionMessageParam as OpenAIMessageParam, // Union type
  ChatCompletionSystemMessageParam, // Specific types for construction
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionMessageToolCall as OpenAIToolCall,
  ChatCompletion, // Non-streaming response
  ChatCompletionChunk, // Streaming chunk
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming
} from 'openai/resources/chat/completions'
import { FunctionDefinition as OpenAIFunctionDef } from 'openai/resources/shared'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider
} from '../../types'
import { Stream } from 'openai/streaming'
import { MappingError, UnsupportedFeatureError, ProviderAPIError, RosettaAIError } from '../../errors' // Import errors
import { safeGet } from '../utils'

// Re-export sub-mappers
export * from './openai.embed.mapper'
export * from './openai.audio.mapper'

// --- Parameter Mapping (Chat/Completion) ---
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
      const _e: never = role
      throw new MappingError(`Unsupported role: ${_e}`, Provider.OpenAI)
  }
}

export function mapContentForOpenAIRole(
  content: RosettaMessage['content'],
  role: OpenAIRole
): string | OpenAIContentPart[] | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal> | null {
  if (content === null) {
    if (role === 'assistant' || role === 'tool') return null
    throw new MappingError(`Role '${role}' requires non-null content.`, Provider.OpenAI)
  }
  if (typeof content === 'string') {
    // Empty string is valid content for user/assistant/tool/system
    return content
  }

  // --- FIX START: Handle empty array explicitly for relevant roles ---
  if (Array.isArray(content) && content.length === 0) {
    if (role === 'system' || role === 'tool') {
      throw new MappingError(`Role '${role}' requires non-empty content array. Received empty array.`, Provider.OpenAI)
    }
    // For 'user', an empty array is technically possible but unusual. Let it pass for now.
    // For 'assistant', an empty array will result in `null` below, which is correct if tool calls exist.
  }
  // --- FIX END ---

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
    // Return null if no text parts remain (e.g., input was [] or only images)
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
      // This case should ideally not be reached if only text parts are allowed for tool role mapping.
      // If it does, it means non-text parts were somehow passed.
      console.warn(`Tool message content contained non-text parts. Stringifying.`)
      try {
        return JSON.stringify(mappedParts)
      } catch {
        throw new MappingError(`Could not stringify complex tool content.`, Provider.OpenAI)
      }
    }
    // If input was [], textParts will be [], join results in "". Handled by check at start.
    return textParts.join('')
  } else {
    throw new MappingError(`Cannot map content parts for unhandled role '${role}'.`, Provider.OpenAI)
  }
}

export function mapToOpenAIParams(
  params: GenerateParams
): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
  const messages: OpenAIMessageParam[] = params.messages.map(msg => {
    const role = mapRoleToOpenAI(msg.role)
    // Map content specifically for the target role
    const content = mapContentForOpenAIRole(msg.content, role)

    switch (role) {
      case 'system':
        let systemContentString: string
        if (typeof content === 'string') {
          systemContentString = content
        } else if (Array.isArray(content)) {
          // Content array for system should only contain text parts after mapping
          systemContentString = content.map(p => (p as ChatCompletionContentPartText).text).join('')
        } else {
          // Should not happen if mapContentForOpenAIRole throws for invalid types
          throw new MappingError('System message content could not be resolved to string.', Provider.OpenAI)
        }
        return { role: 'system', content: systemContentString } as ChatCompletionSystemMessageParam
      case 'user':
        // Content mapping ensures string or ContentPart[]
        return { role: 'user', content: content as string | OpenAIContentPart[] } as ChatCompletionUserMessageParam
      case 'assistant':
        // Content mapping ensures string | TextPart[] | null
        const assistantMsg: ChatCompletionAssistantMessageParam = {
          role: 'assistant',
          content: content as string | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal> | null
        }
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
            id: tc.id,
            type: tc.type as 'function',
            function: { name: tc.function.name, arguments: tc.function.arguments }
          }))
          // Ensure content is null if tool calls exist and content was originally null/empty
          if (assistantMsg.content === '' || content === null) {
            assistantMsg.content = null
          }
        } else if (assistantMsg.content === null) {
          // If no tool calls, content cannot be null, default to empty string
          // This handles the case where mapContentForOpenAIRole returned null for an empty array input
          assistantMsg.content = ''
        }
        return assistantMsg
      case 'tool':
        if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', Provider.OpenAI)
        // Content mapping ensures string for tool role
        if (typeof content !== 'string') {
          // Should not happen if mapContentForOpenAIRole throws for invalid types
          throw new MappingError(
            `Tool message content must resolve to a string. Got: ${typeof content}`,
            Provider.OpenAI
          )
        }
        return {
          role: 'tool',
          tool_call_id: msg.toolCallId,
          content: content // content is string here
        } as ChatCompletionToolMessageParam
      // case 'developer': // Add if needed
      //     if (typeof content !== 'string' && !(Array.isArray(content) && content.every(p => p.type === 'text'))) {
      //         throw new MappingError(`Developer message content must be string or text parts array.`, Provider.OpenAI);
      //     }
      //     return { role: 'developer', content: content as string | ChatCompletionContentPartText[] };

      default:
        throw new MappingError(`Unhandled role type during message construction: ${role}`, Provider.OpenAI)
    }
  })

  // Map tools
  const tools: OpenAITool[] | undefined = params.tools?.map(tool => {
    if (tool.type !== 'function') {
      throw new MappingError(`Unsupported tool type for OpenAI: ${tool.type}`, Provider.OpenAI)
    }
    const parameters = tool.function.parameters as OpenAIFunctionDef['parameters']
    if (typeof parameters !== 'object' || parameters === null) {
      throw new MappingError(
        `Invalid parameters schema for tool ${tool.function.name}. Expected JSON Schema object.`,
        Provider.OpenAI
      )
    }
    return {
      type: tool.type,
      function: { name: tool.function.name, description: tool.function.description, parameters: parameters }
    }
  })

  // Map tool choice
  let toolChoice: OpenAIToolChoiceOption | undefined
  if (typeof params.toolChoice === 'string' && ['none', 'auto', 'required'].includes(params.toolChoice)) {
    toolChoice = params.toolChoice as 'none' | 'auto' | 'required'
  } else if (typeof params.toolChoice === 'object' && params.toolChoice.type === 'function') {
    toolChoice = { type: 'function', function: { name: params.toolChoice.function.name } }
  } else if (params.toolChoice) {
    console.warn(`Unsupported tool_choice format for OpenAI: ${JSON.stringify(params.toolChoice)}. Using 'auto'.`)
    toolChoice = 'auto'
  }

  // Map Response Format
  let responseFormat: { type: 'text' | 'json_object' } | undefined
  if (params.responseFormat?.type === 'json_object') {
    responseFormat = { type: 'json_object' }
    if (params.responseFormat.schema) {
      console.warn(
        'OpenAI JSON mode: schema parameter provided in responseFormat is ignored. Describe the desired schema in the prompt.'
      )
    }
  } else if (params.responseFormat?.type === 'text') {
    responseFormat = { type: 'text' }
  }

  // Check for unsupported features
  if (params.thinking) {
    throw new UnsupportedFeatureError(Provider.OpenAI, 'Thinking steps')
  }
  if (params.grounding) {
    throw new UnsupportedFeatureError(Provider.OpenAI, 'Grounding/Citations')
  }

  // Construct the base payload
  const basePayload = {
    model: params.model!,
    messages,
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    stop: params.stop,
    tools,
    tool_choice: toolChoice,
    response_format: responseFormat
  }

  // Return the correct streaming/non-streaming type
  if (params.stream) {
    return {
      ...basePayload,
      stream: true,
      stream_options: { include_usage: true }
    } as ChatCompletionCreateParamsStreaming
  } else {
    return {
      ...basePayload,
      stream: false
    } as ChatCompletionCreateParamsNonStreaming
  }
}

// --- Result Mapping (Chat/Completion) ---
export function mapUsageFromOpenAI(usage: CompletionUsage | undefined | null): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens
  }
}

export function mapToolCallsFromOpenAI(toolCalls: OpenAIToolCall[] | undefined): RosettaToolCallRequest[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined
  return toolCalls
    .filter(tc => tc.type === 'function' && tc.function?.name && tc.id)
    .map(tc => ({
      id: tc.id!,
      type: tc.type,
      function: { name: tc.function!.name!, arguments: tc.function!.arguments ?? '{}' }
    }))
}

export function mapFromOpenAIResponse(response: ChatCompletion, modelUsed: string): GenerateResult {
  const choice = response.choices[0]
  if (!choice) {
    console.warn('OpenAI response missing choices.')
    const finishReason =
      safeGet<string>(response, 'choices', 0, 'finish_reason') ??
      safeGet<string>(response, 'prompt_annotations', 0, 'content_filter', 'reason') ??
      'error'
    return {
      content: null,
      toolCalls: undefined,
      finishReason: finishReason,
      usage: mapUsageFromOpenAI(response.usage),
      citations: undefined,
      parsedContent: null,
      thinkingSteps: undefined,
      model: response.model ?? modelUsed,
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
  const mappedToolCalls = mapToolCallsFromOpenAI(choice.message?.tool_calls)
  return {
    content: textContent,
    toolCalls: mappedToolCalls,
    finishReason: choice.finish_reason,
    usage: mapUsageFromOpenAI(response.usage),
    citations: undefined,
    parsedContent: parsedJson,
    thinkingSteps: undefined,
    model: response.model ?? modelUsed,
    rawResponse: response
  }
}

// --- Stream Mapping (Chat/Completion) ---
export async function* mapOpenAIStream(stream: Stream<ChatCompletionChunk>): AsyncIterable<StreamChunk> {
  let accumulatedContent = ''
  const accumulatedToolCalls: Record<
    number,
    Partial<RosettaToolCallRequest & { index: number; function: { name: string; arguments: string } }>
  > = {}
  let finalUsage: TokenUsage | undefined
  let finalFinishReason: string | null = null
  let model = ''
  let isJsonMode = false
  let aggregatedResult: GenerateResult | null = null

  try {
    for await (const chunk of stream) {
      if (chunk.usage) {
        finalUsage = mapUsageFromOpenAI(chunk.usage)
        if (aggregatedResult) aggregatedResult.usage = finalUsage
        continue // Usage chunk doesn't have choices, skip to next iteration
      }
      if (!model && chunk.model) {
        model = chunk.model
        yield { type: 'message_start', data: { provider: Provider.OpenAI, model: model } }
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
      }
      const choice = chunk.choices[0]
      if (!choice) continue
      if (!isJsonMode && accumulatedContent === '' && choice.delta?.content?.trim().match(/^[{[]/)) {
        isJsonMode = true
      }
      if (choice.delta?.content) {
        const delta = choice.delta.content
        accumulatedContent += delta
        if (aggregatedResult) aggregatedResult.content = accumulatedContent
        if (isJsonMode) {
          let partialParsed = undefined
          try {
            partialParsed = JSON.parse(accumulatedContent)
          } catch {}
          yield { type: 'json_delta', data: { delta, parsed: partialParsed, snapshot: accumulatedContent } }
        } else {
          yield { type: 'content_delta', data: { delta: delta } }
        }
      }
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
      if (choice.finish_reason) {
        const reason = choice.finish_reason
        finalFinishReason = reason
        if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

        for (const tc of Object.values(accumulatedToolCalls)) {
          if (!(tc as any).yieldedDone && tc.id && tc.index !== undefined) {
            // Yield is now in the correct scope
            yield { type: 'tool_call_done', data: { index: tc.index, id: tc.id } }
            ;(tc as any).yieldedDone = true
            if (
              aggregatedResult &&
              tc.type === 'function' &&
              tc.function?.name &&
              tc.function?.arguments !== undefined
            ) {
              aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
              aggregatedResult.toolCalls.push({
                id: tc.id,
                type: tc.type,
                function: { name: tc.function.name, arguments: tc.function.arguments }
              })
            }
          }
        } // End for...of loop

        if (isJsonMode) {
          let finalParsedJson = null
          try {
            finalParsedJson = JSON.parse(accumulatedContent ?? '')
          } catch {}
          yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: accumulatedContent ?? '' } }
          if (aggregatedResult) aggregatedResult.parsedContent = finalParsedJson
        }
      }
    }
    finalFinishReason = finalFinishReason ?? 'stop'
    if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason
    yield { type: 'message_stop', data: { finishReason: finalFinishReason } }
    if (finalUsage) {
      yield { type: 'final_usage', data: { usage: finalUsage } }
    }
    if (aggregatedResult) {
      if (!isJsonMode && aggregatedResult.content === '') aggregatedResult.content = null
      if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
      yield { type: 'final_result', data: { result: aggregatedResult } }
    } else {
      console.warn('OpenAI stream finished but no aggregated result was built.')
    }
  } catch (error) {
    // FIX: Use direct access logic from wrapProviderError for consistency
    let mappedError: RosettaAIError
    if (error instanceof RosettaAIError) {
      mappedError = error
    } else if (error instanceof OpenAI.APIError) {
      let message = 'Unknown OpenAI API Error' // Default fallback
      const nestedErrorObj = error.error as any
      const nestedMessage = nestedErrorObj?.message

      if (nestedMessage && typeof nestedMessage === 'string' && nestedMessage.trim()) {
        message = nestedMessage.trim()
      } else if (error.message && typeof error.message === 'string' && error.message.trim()) {
        message = error.message.trim()
      } else if (nestedErrorObj) {
        try {
          const stringifiedBody = JSON.stringify(nestedErrorObj)
          if (stringifiedBody !== '{}') {
            message = stringifiedBody
          }
        } catch {}
      }
      mappedError = new ProviderAPIError(
        message,
        Provider.OpenAI,
        error.status,
        error.code, // Use error.code directly
        error.type, // Use error.type directly
        error
      )
    } else {
      mappedError = new ProviderAPIError(String(error), Provider.OpenAI, undefined, undefined, undefined, error)
    }
    yield { type: 'error', data: { error: mappedError } }
  }
}
