import Groq from 'groq-sdk'
import {
  ChatCompletionMessageParam, // Union type for array
  ChatCompletionSystemMessageParam, // Specific type
  ChatCompletionUserMessageParam, // Specific type
  ChatCompletionAssistantMessageParam, // Specific type
  ChatCompletionToolMessageParam, // Specific type
  ChatCompletionContentPart,
  ChatCompletionTool,
  ChatCompletionRole,
  ChatCompletionToolChoiceOption,
  ChatCompletionCreateParams, // Import the correct union type for parameters
  ChatCompletion, // Non-streaming response type
  ChatCompletionChunk, // Streaming chunk type
  ChatCompletionMessage // Message structure in response
} from 'groq-sdk/resources/chat/completions'
import { CompletionUsage } from 'groq-sdk/resources/completions'
// FIX: Change import from GroqStreamType to standard AsyncIterable
// import { Stream as GroqStreamType } from 'groq-sdk/lib/streaming'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider
} from '../../types'
import { MappingError, UnsupportedFeatureError, ProviderAPIError, RosettaAIError } from '../../errors'
import { safeGet } from '../utils'
// Removed duplicate CompletionUsage import

// Import mappers for sub-features
export * from './groq.embed.mapper'
export * from './groq.audio.mapper'

// --- Parameter Mapping ---

function mapRoleToGroq(role: RosettaMessage['role']): ChatCompletionRole {
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
      throw new MappingError(`Unsupported role: ${_e}`, Provider.Groq)
  }
}

// Adjusted content mapping - aligns with previous OpenAI fix logic
export function mapContentForGroqRole(
  content: RosettaMessage['content'],
  role: ChatCompletionRole
): string | Array<ChatCompletionContentPart> | null {
  if (content === null) {
    // Groq allows null content for assistant (with tool calls) or tool roles
    if (role === 'assistant' || role === 'tool') return null
    throw new MappingError(`Role '${role}' requires non-null content for Groq.`, Provider.Groq)
  }
  if (typeof content === 'string') {
    return content
  }

  // Handle array of parts
  const mappedParts: ChatCompletionContentPart[] = content.map(part => {
    if (part.type === 'text') {
      return { type: 'text', text: part.text }
    } else if (part.type === 'image') {
      // User role allows image parts
      if (role !== 'user') {
        throw new MappingError(
          `Image content parts are only allowed for the 'user' role in Groq, not '${role}'.`,
          Provider.Groq
        )
      }
      return {
        type: 'image_url',
        image_url: { url: `data:${part.image.mimeType};base64,${part.image.base64Data}` }
      } // Matches ChatCompletionContentPartImage structure
    } else {
      const _e: never = part
      throw new MappingError(`Unsupported content part type: ${(_e as any).type}`, Provider.Groq)
    }
  })

  // Check role constraints for array content
  if (role === 'user') {
    return mappedParts // User allows text/image parts array
  } else if (role === 'assistant') {
    // Assistant content must be string | null. Filter/stringify non-string parts?
    // If Groq Assistant content MUST be string or null, we need to adapt.
    // Let's assume it behaves like OpenAI and stringify for now, adding a warning.
    const textParts = mappedParts
      .filter(p => p.type === 'text')
      .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
    if (textParts.length === mappedParts.length) {
      return textParts.join('') // Combine multiple text parts if needed
    } else {
      // Throw error if non-text parts are present for assistant
      throw new MappingError('Assistant message content for Groq must resolve to a string or be null.', Provider.Groq)
    }
  } else if (role === 'system') {
    // System content must be string
    const textParts = mappedParts
      .filter(p => p.type === 'text')
      .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
    if (textParts.length !== mappedParts.length) {
      throw new MappingError(`System message content for Groq must resolve to a string.`, Provider.Groq)
    }
    return textParts.join('')
  } else if (role === 'tool') {
    // Tool content must be string
    const textParts = mappedParts
      .filter(p => p.type === 'text')
      .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
    if (textParts.length !== mappedParts.length) {
      // Throw error if non-text parts are present for tool
      throw new MappingError('Tool message content for Groq must resolve to a string.', Provider.Groq)
    }
    return textParts.join('') // Return combined string if only text parts
  } else {
    throw new MappingError(`Cannot map content parts for unhandled Groq role '${role}'.`, Provider.Groq)
  }
}

export function mapToGroqParams(params: GenerateParams): ChatCompletionCreateParams {
  // Construct specific message types based on role

  const messages: ChatCompletionMessageParam[] = params.messages.map(msg => {
    const role = mapRoleToGroq(msg.role)
    // Map content based on the role's constraints
    const content = mapContentForGroqRole(msg.content, role)

    switch (role) {
      case 'system':
        if (typeof content !== 'string') throw new MappingError('System content mismatch.', Provider.Groq)
        return { role: 'system', content } as ChatCompletionSystemMessageParam
      case 'user':
        if (content === null) throw new MappingError('User content cannot be null.', Provider.Groq)
        // Content is string | ChatCompletionContentPart[]
        return { role: 'user', content } as ChatCompletionUserMessageParam
      case 'assistant':
        // Content should be string | null for assistant
        const assistantMsg: ChatCompletionAssistantMessageParam = {
          role: 'assistant',
          content: content as string | null // Cast based on mapping logic
        }
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
            id: tc.id,
            type: tc.type as 'function', // Assert type
            function: { name: tc.function.name, arguments: tc.function.arguments }
          }))
          // Allow content to be null if tool calls are present
          if (assistantMsg.content === '') assistantMsg.content = null
        } else if (assistantMsg.content === null) {
          // Content must not be null if no tool calls
          assistantMsg.content = ''
        }
        return assistantMsg
      case 'tool':
        if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', Provider.Groq)
        // Content must be string for tool message
        if (typeof content !== 'string') {
          throw new MappingError(
            `Tool message content must be a string for Groq. Received: ${typeof content}`,
            Provider.Groq
          )
        }
        return {
          role: 'tool',
          tool_call_id: msg.toolCallId,
          content: content // Content is already validated as string
        } as ChatCompletionToolMessageParam
      case 'function': // Deprecated role, map to tool? Groq might not support.
        throw new MappingError("Deprecated 'function' role not supported for Groq.", Provider.Groq)
      default:
        throw new MappingError(`Unhandled role type in message construction: ${role}`, Provider.Groq)
    }
  })

  // Map tools
  const tools: ChatCompletionTool[] | undefined = params.tools?.map(tool => {
    if (tool.type !== 'function') {
      throw new MappingError(`Unsupported tool type for Groq: ${tool.type}`, Provider.Groq)
    }
    // Groq expects parameters to be a JSON schema object
    const parameters = tool.function.parameters // Already Record<string, unknown>
    // Groq's FunctionParameters type is structurally compatible with JSON Schema object
    if (typeof parameters !== 'object' || parameters === null || Array.isArray(parameters)) {
      throw new MappingError(
        `Invalid parameters schema for tool ${tool.function.name}. Expected JSON Schema object.`,
        Provider.Groq
      )
    }
    // FIX: Ensure parameters has type: 'object' as required by Groq
    if (parameters.type !== 'object') {
      throw new MappingError(
        `Invalid parameters schema for tool ${tool.function.name}. Groq requires top-level 'type: \"object\"'.`,
        Provider.Groq
      )
    }
    return {
      type: tool.type,
      function: {
        name: tool.function.name,
        description: tool.function.description,
        // Cast to GroqFunctionParameters, assuming structural compatibility
        parameters: parameters
      }
    }
  })

  // Map tool choice (logic remains the same)
  let toolChoice: ChatCompletionToolChoiceOption | undefined
  if (typeof params.toolChoice === 'string' && ['none', 'auto'].includes(params.toolChoice)) {
    toolChoice = params.toolChoice as 'none' | 'auto'
  } else if (params.toolChoice === 'required') {
    console.warn("'required' tool_choice mapped to 'auto' for Groq.")
    toolChoice = 'auto'
  } else if (typeof params.toolChoice === 'object' && params.toolChoice.type === 'function') {
    toolChoice = { type: 'function', function: { name: params.toolChoice.function.name } }
  } else if (params.toolChoice) {
    console.warn(`Unsupported tool_choice format for Groq: ${JSON.stringify(params.toolChoice)}. Using 'auto'.`)
    toolChoice = 'auto'
  }

  // Check for unsupported features (logic remains the same)
  if (params.responseFormat?.type === 'json_object') {
    console.warn('JSON response format requested, but Groq support is unconfirmed via standard parameters.')
  }
  if (params.thinking) {
    throw new UnsupportedFeatureError(Provider.Groq, 'Thinking steps')
  }
  if (params.grounding) {
    throw new UnsupportedFeatureError(Provider.Groq, 'Grounding/Citations')
  }

  // Construct the final payload which fits the ChatCompletionCreateParams union
  const payload: ChatCompletionCreateParams = {
    model: params.model!,
    messages: messages,
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    stop: params.stop,
    tools: tools,
    tool_choice: toolChoice,
    stream: params.stream // Let stream flag determine the variant

    // Add other Base params if needed: n, presence_penalty, frequency_penalty, seed, response_format(if mapped)
  }

  return payload
}

export function mapUsageFromGroq(usage: CompletionUsage | undefined | null): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens
  }
}

type GroqToolCall = NonNullable<ChatCompletionMessage['tool_calls']>[number]

export function mapToolCallsFromGroq(
  toolCalls: ReadonlyArray<GroqToolCall> | undefined | null
): RosettaToolCallRequest[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined

  return toolCalls
    .filter(tc => tc.type === 'function' && tc.function?.name && tc.id)
    .map(tc => {
      return {
        id: tc.id!,
        type: tc.type,
        function: {
          name: tc.function!.name!,
          arguments: tc.function!.arguments ?? '{}'
        }
      }
    })
}

export function mapFromGroqResponse(response: ChatCompletion, modelUsed: string): GenerateResult {
  const choice = response.choices[0]
  if (!choice) {
    console.warn('Groq response missing choices.')
    const finishReason = safeGet<string>(response, 'choices', 0, 'finish_reason') ?? 'error'
    return {
      content: null,
      toolCalls: undefined,
      finishReason: finishReason,
      usage: mapUsageFromGroq(response.usage),
      citations: undefined,
      parsedContent: null,
      thinkingSteps: undefined,
      model: response.model ?? modelUsed,
      rawResponse: response
    }
  }
  let parsedJson: any = null
  const textContent = choice.message?.content ?? null
  if (textContent && (textContent.trim().startsWith('{') || textContent.trim().startsWith('['))) {
    try {
      parsedJson = JSON.parse(textContent)
    } catch {}
  }
  const groqToolCalls = safeGet<ReadonlyArray<GroqToolCall>>(choice.message, 'tool_calls')
  const mappedToolCalls = mapToolCallsFromGroq(groqToolCalls)
  const fr = choice.finish_reason
  const standardizedFR = fr === 'tool_calls' ? 'tool_calls' : fr === 'stop' ? 'stop' : fr === 'length' ? 'length' : fr
  return {
    content: textContent,
    toolCalls: mappedToolCalls,
    finishReason: standardizedFR,
    usage: mapUsageFromGroq(response.usage),
    citations: undefined,
    parsedContent: parsedJson,
    thinkingSteps: undefined,
    model: response.model ?? modelUsed,
    rawResponse: response
  }
}

// FIX: Correct the input type signature
export async function* mapGroqStream(stream: AsyncIterable<ChatCompletionChunk>): AsyncIterable<StreamChunk> {
  let accumulatedContent = ''
  const accumulatedToolCalls: Record<
    number,
    Partial<RosettaToolCallRequest & { index: number; function: { name: string; arguments: string } }>
  > = {}
  let finalUsage: TokenUsage | undefined
  let finalFinishReason: string | null = null
  let model = ''
  let aggregatedResult: GenerateResult | null = null

  try {
    for await (const chunk of stream) {
      if (!model && chunk.model) {
        model = chunk.model
        yield { type: 'message_start', data: { provider: Provider.Groq, model: model } }
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
      if (choice.delta?.content) {
        accumulatedContent += choice.delta.content
        if (aggregatedResult) aggregatedResult.content = accumulatedContent
        yield { type: 'content_delta', data: { delta: choice.delta.content } }
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
                id: currentTool.id ?? `unk_groq_${index}`,
                functionArgumentChunk: tcChunk.function.arguments
              }
            }
          }
        }
      }
      if (choice.finish_reason) {
        const reason = choice.finish_reason
        finalFinishReason =
          reason === 'tool_calls' ? 'tool_calls' : reason === 'stop' ? 'stop' : reason === 'length' ? 'length' : reason // Keep other reasons

        if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

        // Use a for...of loop instead of forEach to allow yield
        for (const tc of Object.values(accumulatedToolCalls)) {
          // Check if 'done' hasn't been yielded yet for this tool
          if (!(tc as any).yieldedDone && tc.id && tc.index !== undefined) {
            // Yield is now directly inside the generator function scope
            yield { type: 'tool_call_done', data: { index: tc.index, id: tc.id } }
            ;(tc as any).yieldedDone = true // Mark as done yielded

            // Add completed tool call to aggregated result
            if (
              aggregatedResult &&
              tc.type === 'function' &&
              tc.function?.name &&
              tc.function?.arguments !== undefined
            ) {
              aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
              // Ensure we are pushing the fully formed tool call request
              aggregatedResult.toolCalls.push({
                id: tc.id,
                type: tc.type, // Should be 'function'
                function: { name: tc.function.name, arguments: tc.function.arguments }
              })
            }
          }
        } // End of for...of loop
      }
      // FIX: Correctly process usage from x_groq using safeGet
      const usageFromChunk = safeGet<Groq.CompletionUsage>(chunk, 'x_groq', 'usage')
      if (usageFromChunk) {
        finalUsage = mapUsageFromGroq(usageFromChunk)
        if (aggregatedResult) aggregatedResult.usage = finalUsage
      }
    }
    finalFinishReason = finalFinishReason ?? 'stop'
    if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

    // FIX: Yield message_stop *before* final_usage
    yield { type: 'message_stop', data: { finishReason: finalFinishReason } }

    // FIX: Yield final_usage *after* message_stop if usage exists
    if (finalUsage) {
      yield { type: 'final_usage', data: { usage: finalUsage } }
    }

    if (aggregatedResult) {
      let finalParsed = null
      if (
        aggregatedResult.content &&
        (aggregatedResult.content.trim().startsWith('{') || aggregatedResult.content.trim().startsWith('['))
      ) {
        try {
          finalParsed = JSON.parse(aggregatedResult.content)
        } catch {}
      }
      aggregatedResult.parsedContent = finalParsed
      if (aggregatedResult.content === '') aggregatedResult.content = null
      if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
      yield { type: 'final_result', data: { result: aggregatedResult } }
    } else {
      console.warn('Groq stream finished but no aggregated result was built.')
    }
  } catch (error) {
    const mappedError =
      error instanceof RosettaAIError
        ? error
        : error instanceof Groq.APIError
        ? new ProviderAPIError(
            error.message,
            Provider.Groq,
            error.status,
            safeGet<string>(error, 'code'),
            safeGet<string>(error, 'type'),
            error
          )
        : new ProviderAPIError(String(error), Provider.Groq, undefined, undefined, undefined, error)
    yield { type: 'error', data: { error: mappedError } }
  }
}
