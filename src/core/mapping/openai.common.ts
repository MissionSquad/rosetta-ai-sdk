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
import { GenerateResult, Provider, RosettaMessage, RosettaToolCallRequest, StreamChunk, TokenUsage } from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError } from '../../errors'
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
  // --- FIX: Perform role-specific null/empty checks FIRST ---
  if (content === null) {
    if (role === 'assistant' || role === 'tool') return null // Allowed for assistant/tool
    throw new MappingError(`Role '${role}' requires non-null content.`, Provider.OpenAI)
  }
  if (content === '') {
    // FIX: Allow empty string for 'tool' role, only throw for 'system'
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
  // --- End FIX ---

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

function mapToolCallsFromOpenAI(toolCalls: OpenAIToolCall[] | undefined): RosettaToolCallRequest[] | undefined {
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
  // FIX: Add null/undefined check for response.choices before accessing index 0
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
  const mappedToolCalls = mapToolCallsFromOpenAI(choice.message?.tool_calls)
  return {
    content: textContent,
    toolCalls: mappedToolCalls,
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
  // --- FIX: Handle non-Error objects better in this specific wrapper ---
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
  // --- End FIX ---
}

export async function* mapOpenAIStream(
  stream: Stream<ChatCompletionChunk>,
  provider: Provider
): AsyncIterable<StreamChunk> {
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
        finalUsage = mapTokenUsage(chunk.usage) // Use common utility
        if (aggregatedResult) aggregatedResult.usage = finalUsage
        continue
      }
      if (!model && chunk.model) {
        model = chunk.model
        yield { type: 'message_start', data: { provider, model } }
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
          yield { type: 'content_delta', data: { delta } }
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
        }

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
    const mappedError = wrapOpenAIError(error, provider)
    yield { type: 'error', data: { error: mappedError } }
  }
}
