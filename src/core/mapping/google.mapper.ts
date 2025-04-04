import {
  Content as GoogleContent,
  FunctionCall,
  Tool as GoogleTool,
  GenerateContentRequest,
  GenerateContentResponse,
  StartChatParams,
  CitationMetadata,
  SchemaType as GoogleSchemaType,
  FunctionDeclarationSchema,
  FunctionCallPart,
  TextPart
} from '@google/generative-ai'
import {
  // Import Part type alias
  Part
} from '@google/generative-ai'
import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider,
  Citation
} from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError } from '../../errors' // Import necessary errors
import { safeGet } from '../utils'

// Re-export sub-mappers
export * from './google.embed.mapper'

// Alias Google Part type for clarity in this file
type GooglePart = Part

// --- Parameter Mapping ---
function mapRoleToGoogle(role: RosettaMessage['role']): 'user' | 'model' | 'function' | 'system' {
  switch (role) {
    case 'user':
      return 'user'
    case 'assistant':
      return 'model'
    case 'tool':
      return 'function'
    case 'system':
      return 'system'
    default:
      const _e: never = role
      throw new MappingError(`Unsupported role: ${_e}`, Provider.Google)
  }
}

function mapContentToGoogleParts(content: RosettaMessage['content']): GooglePart[] {
  if (content === null) {
    // Google generally expects non-null content for user/model. Return empty array or handle error.
    console.warn('Mapping null content to empty parts array for Google.')
    return []
  }
  if (typeof content === 'string') {
    return [{ text: content }]
  }
  return content.map(part => {
    if (part.type === 'text') return { text: part.text }
    if (part.type === 'image') return { inlineData: { mimeType: part.image.mimeType, data: part.image.base64Data } }
    const _e: never = part
    throw new MappingError(`Unsupported content part: ${(_e as any).type}`, Provider.Google)
  })
}

function isFunctionDeclarationSchema(schema: any): schema is FunctionDeclarationSchema {
  // Basic check, might need refinement based on actual GoogleSchema structure
  return (
    typeof schema === 'object' &&
    schema !== null &&
    'type' in schema &&
    Object.values(GoogleSchemaType).includes(schema.type)
  )
}

/**
 * Finds the name of the function called in the most recent model message
 * that precedes a 'function' role message needing its name.
 * NOTE: This is a heuristic. A more robust solution might involve tracking IDs if available.
 */
function findLastToolCallName(history: GoogleContent[], _toolCallId: string): string | undefined {
  // Google doesn't use toolCallId in the same way as OpenAI/Anthropic.
  // We need to find the function name from the preceding 'model' turn's FunctionCallPart.
  for (let i = history.length - 1; i >= 0; i--) {
    const prevMsg = history[i]
    if (prevMsg?.role === 'model' && Array.isArray(prevMsg.parts)) {
      for (const part of prevMsg.parts) {
        if ('functionCall' in part && part.functionCall?.name) {
          // Found the most recent function call name from the model
          return part.functionCall.name
        }
      }
    }
  }
  console.warn(`Could not determine preceding function name for tool result (ID: ${_toolCallId}) from history.`)
  return undefined
}

export function mapToGoogleParams(
  params: GenerateParams
): { googleMappedParams: GenerateContentRequest | (StartChatParams & { contents: GooglePart[] }); isChat: boolean } {
  let systemInstruction: GoogleContent | undefined = undefined
  const history: GoogleContent[] = []
  let lastUserMessageParts: GooglePart[] = []

  params.messages.forEach((msg, index) => {
    const isLastMessage = index === params.messages.length - 1
    const googleRole = mapRoleToGoogle(msg.role)

    if (googleRole === 'system') {
      if (systemInstruction)
        throw new MappingError('Multiple system messages not supported by Google.', Provider.Google)
      if (typeof msg.content !== 'string')
        throw new MappingError('Google system instruction must be string.', Provider.Google)
      systemInstruction = { role: 'system', parts: [{ text: msg.content }] }
      return
    }

    const parts = mapContentToGoogleParts(msg.content)

    // Handle model's function calls
    if (googleRole === 'model' && msg.toolCalls && msg.toolCalls.length > 0) {
      const functionCallParts: FunctionCallPart[] = msg.toolCalls.map(tc => {
        try {
          return { functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) } }
        } catch (e) {
          throw new MappingError(
            `Failed to parse arguments for tool ${tc.function.name}`,
            Provider.Google,
            'mapToGoogleParams toolCall mapping',
            e
          )
        }
      })
      // Combine with potential text content from the assistant
      const existingTextParts = parts.filter((p): p is TextPart => 'text' in p)
      history.push({ role: googleRole, parts: [...existingTextParts, ...functionCallParts] })
    }
    // Handle user's function responses
    else if (googleRole === 'function') {
      if (!msg.toolCallId || typeof msg.content !== 'string') {
        throw new MappingError(
          'Invalid tool result message for Google. Requires toolCallId and string content.',
          Provider.Google
        )
      }
      // Find the name associated with this response from previous model turn
      const funcName = findLastToolCallName(history, msg.toolCallId)
      if (!funcName) {
        throw new MappingError(
          `Cannot find function name for tool result (ID: ${msg.toolCallId}). Ensure model message with FunctionCall precedes this tool message.`,
          Provider.Google
        )
      }
      let respContent: any
      try {
        // Google expects the 'response' field to be an object
        respContent = JSON.parse(msg.content)
      } catch {
        // If parsing fails, wrap the string content in a standard object structure
        respContent = { content: msg.content }
        console.warn(`Tool result content for ${funcName} was not valid JSON. Wrapping as { content: "..." }`)
      }
      history.push({ role: googleRole, parts: [{ functionResponse: { name: funcName, response: respContent } }] })
    }
    // Handle the last user message separately for chat vs. non-chat
    else if (isLastMessage && googleRole === 'user') {
      lastUserMessageParts = parts
    }
    // Handle regular user/model messages in history
    else {
      // This condition ensures googleRole is 'user' or 'model' here
      history.push({ role: googleRole, parts })
    }
  })

  // Map Tools and Grounding
  const googleTools: GoogleTool[] | undefined = params.tools?.map(tool => {
    if (tool.type !== 'function') {
      throw new MappingError(`Only 'function' tools are currently supported for Google.`, Provider.Google)
    }
    const schema = tool.function.parameters
    if (!isFunctionDeclarationSchema(schema)) {
      throw new MappingError(
        `Invalid parameters schema for tool ${tool.function.name}. Expected FunctionDeclarationSchema.`,
        Provider.Google
      )
    }
    return {
      functionDeclarations: [{ name: tool.function.name, description: tool.function.description, parameters: schema }]
    }
  })

  let finalTools = googleTools
  if (params.grounding?.enabled) {
    // Currently only supports GoogleSearchRetrieval
    const searchTool: GoogleTool = { googleSearchRetrieval: {} } // Default empty config
    if (params.grounding.source && params.grounding.source !== 'web') {
      console.warn(
        `Only 'web' grounding source currently mapped for Google Search Retrieval. Ignoring source: ${params.grounding.source}`
      )
    }
    finalTools = finalTools ? [...finalTools, searchTool] : [searchTool]
  }

  // Map Response Format
  let responseMimeType: string | undefined
  // const responseSchemaForConfig: GoogleSchema | undefined = undefined // Not directly usable in GenerationConfig. no unused vars!
  if (params.responseFormat?.type === 'json_object') {
    responseMimeType = 'application/json'
    if (params.responseFormat.schema) {
      // Note: Google's `responseSchema` in GenerationConfig is for function calling, not general JSON mode.
      // We set responseMimeType, but schema needs to be handled via prompting.
      console.warn(
        'Google JSON mode requested via responseFormat. Ensure schema is described in the prompt. `schema` parameter is ignored for Google GenerationConfig.'
      )
      // if (isFunctionDeclarationSchema(params.responseFormat.schema)) {
      // responseSchemaForConfig = params.responseFormat.schema; // This is incorrect mapping for Google
      // }
    }
  }

  // Build GenerationConfig
  const generationConfig = {
    maxOutputTokens: params.maxTokens,
    temperature: params.temperature,
    topP: params.topP,
    stopSequences: Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined,
    responseMimeType: responseMimeType
    // responseSchema: responseSchemaForConfig // Cannot map schema here directly
  }

  // Determine if it's a chat or single-turn request
  const isChat = history.length > 0 || !!systemInstruction

  if (isChat) {
    // For chat, use StartChatParams structure, add last user message to `contents`
    const chatParams: StartChatParams = {
      history,
      generationConfig,
      tools: finalTools,
      systemInstruction
    }
    // The google SDK's startChat doesn't take initial contents directly, sendMessage does.
    // Return structure indicating chat and include the parts for the first sendMessage call.
    return { googleMappedParams: { ...chatParams, contents: lastUserMessageParts }, isChat: true }
  } else {
    // For single-turn, use GenerateContentRequest
    const request: GenerateContentRequest = {
      contents: [{ role: 'user', parts: lastUserMessageParts }],
      generationConfig,
      tools: finalTools,
      systemInstruction
    }
    return { googleMappedParams: request, isChat: false }
  }
}

// --- Result Mapping ---

function mapUsageFromGoogle(usage: GenerateContentResponse['usageMetadata'] | undefined): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.promptTokenCount,
    completionTokens: usage.candidatesTokenCount,
    totalTokens: usage.totalTokenCount,
    cachedContentTokenCount: usage.cachedContentTokenCount
  }
}

function mapToolCallsFromGoogle(calls: FunctionCall[] | undefined): RosettaToolCallRequest[] | undefined {
  if (!calls || calls.length === 0) return undefined

  // Filter out calls missing essential info (name) and map
  return calls
    .filter(call => call?.name) // Ensure call and name exist
    .map((call, index) => {
      return {
        // Generate a relatively stable ID if Google doesn't provide one easily accessible
        id: `google_func_${call.name}_${index}_${Date.now()}`,
        type: 'function',
        function: {
          name: call.name!, // We filtered for name existence
          arguments: JSON.stringify(call.args ?? {}) // Ensure args is at least an empty object stringified
        }
      }
    })
}

function mapCitationsFromGoogle(metadata: CitationMetadata | undefined): Citation[] | undefined {
  if (!metadata?.citationSources || metadata.citationSources.length === 0) return undefined

  return metadata.citationSources.map((s, index) => ({
    // Use URI if available, otherwise generate an ID
    sourceId: s.uri ?? `google_cite_idx_${index}`,
    startIndex: s.startIndex,
    endIndex: s.endIndex,
    text: undefined // Google citations often don't include the text snippet directly here
  }))
}

export function mapFromGoogleResponse(response: GenerateContentResponse | undefined, model: string): GenerateResult {
  // Handle blocked prompts first
  const promptFeedbackReason = safeGet<string>(response, 'promptFeedback', 'blockReason')
  const promptFeedbackSafetyRatings = safeGet<any[]>(response, 'promptFeedback', 'safetyRatings')

  if (promptFeedbackReason) {
    const fr =
      promptFeedbackReason === 'SAFETY'
        ? 'content_filter'
        : promptFeedbackReason === 'OTHER'
        ? 'error'
        : promptFeedbackReason.toLowerCase() // Default to lowercased reason
    console.warn(`Google prompt blocked. Reason: ${fr}. Ratings: ${JSON.stringify(promptFeedbackSafetyRatings)}`)
    return {
      content: null,
      toolCalls: undefined,
      finishReason: fr,
      usage: mapUsageFromGoogle(response?.usageMetadata), // Usage might still be present
      citations: undefined,
      parsedContent: null,
      thinkingSteps: undefined,
      model: model,
      rawResponse: response
    }
  }

  // Process successful responses (or responses blocked at candidate level)
  const candidate = response?.candidates?.[0]
  const candidateFinishReason = candidate?.finishReason
  const candidateSafetyRatings = candidate?.safetyRatings

  if (!response || !candidate) {
    // Should not happen if prompt wasn't blocked, but handle defensively
    console.warn('Google response or candidate is missing despite no prompt block.')
    return {
      content: null,
      toolCalls: undefined,
      finishReason: 'error', // Indicate an unexpected error state
      usage: mapUsageFromGoogle(response?.usageMetadata),
      citations: undefined,
      parsedContent: null,
      thinkingSteps: undefined,
      model: model,
      rawResponse: response
    }
  }

  let textContent: string | null = null
  let toolCalls: RosettaToolCallRequest[] | undefined
  let parsedJson: any = null
  let finishReason = candidateFinishReason ?? 'unknown' // Start with candidate reason

  // Extract content parts
  if (candidate.content?.parts) {
    // FIX: Safely handle null/undefined parts in the filter
    const textParts = candidate.content.parts.filter((p): p is TextPart => p && 'text' in p)
    if (textParts.length > 0) {
      textContent = textParts.map(p => p.text).join('')
      // Attempt to parse if it looks like JSON *and* not blocked for safety
      const isJsonLike = textContent?.trim().startsWith('{') || textContent?.trim().startsWith('[')
      const isBlocked = candidateFinishReason === 'SAFETY' || !!promptFeedbackReason
      if (isJsonLike && !isBlocked) {
        try {
          parsedJson = JSON.parse(textContent)
        } catch (e) {
          console.warn('Failed to auto-parse potential JSON from Google:', e)
        }
      }
    }

    // Extract function calls
    const functionCallParts = candidate.content.parts.filter((p): p is FunctionCallPart => p && 'functionCall' in p)
    if (functionCallParts.length > 0) {
      const mappedCalls = mapToolCallsFromGoogle(functionCallParts.map(p => p.functionCall))
      if (mappedCalls && mappedCalls.length > 0) {
        toolCalls = mappedCalls
        // If the primary reason wasn't safety/recitation/max_tokens, set to tool_calls
        if (!['SAFETY', 'RECITATION', 'MAX_TOKENS'].includes(candidateFinishReason ?? '')) {
          finishReason = 'tool_calls'
        }
      }
    }
  }

  // Standardize finish reason strings
  if (candidateFinishReason === 'SAFETY') {
    finishReason = 'content_filter'
    console.warn(`Google candidate blocked due to safety. Ratings: ${JSON.stringify(candidateSafetyRatings)}`)
  } else if (candidateFinishReason === 'RECITATION') {
    finishReason = 'recitation_filter'
  } else if (candidateFinishReason === 'MAX_TOKENS') {
    finishReason = 'length'
  } else if (candidateFinishReason === 'STOP' && !toolCalls) {
    finishReason = 'stop'
  } else if (finishReason === 'unknown' && candidate.content) {
    // If reason is unknown but content exists, assume it stopped normally
    finishReason = 'stop'
  }

  const citations: Citation[] | undefined = mapCitationsFromGoogle(candidate.citationMetadata)

  return {
    content: textContent,
    toolCalls: toolCalls,
    finishReason: finishReason,
    usage: mapUsageFromGoogle(response.usageMetadata),
    citations: citations,
    parsedContent: parsedJson,
    thinkingSteps: undefined,
    model: model,
    rawResponse: response
  }
}

// --- Stream Mapping ---

// --- Stream Mapping ---

export async function* mapGoogleStream(stream: AsyncIterable<GenerateContentResponse>): AsyncIterable<StreamChunk> {
  let currentUsage: TokenUsage | undefined
  let finalFinishReason: string | null = null
  let aggregatedText = ''
  const aggregatedCitations: Citation[] = []
  const aggregatedToolCalls: RosettaToolCallRequest[] = []
  const model = '' // Model name isn't directly in the stream chunks, needs to be passed or inferred
  let isPotentiallyJson = false
  let aggregatedResult: GenerateResult | null = null // Store parts needed for final result

  try {
    // Yield message_start immediately (model unknown initially)
    yield { type: 'message_start', data: { provider: Provider.Google, model: model } } // Parenthesized

    for await (const chunk of stream) {
      // Aggregate usage metadata if present
      if (chunk.usageMetadata) {
        currentUsage = mapUsageFromGoogle(chunk.usageMetadata)
        if (aggregatedResult) aggregatedResult.usage = currentUsage
      }

      // Process candidate data
      const candidate = chunk.candidates?.[0]
      if (!candidate) continue // Skip chunks without candidates

      // Initialize aggregated result on first valid candidate
      if (!aggregatedResult) {
        aggregatedResult = {
          content: '',
          toolCalls: [],
          finishReason: null,
          usage: currentUsage,
          model: model,
          thinkingSteps: null,
          citations: [],
          parsedContent: null,
          rawResponse: undefined
        }
      }

      // --- Text Delta ---
      // FIX: Safely handle null/undefined parts in the filter
      const textDelta =
        safeGet<GooglePart[]>(candidate, 'content', 'parts')
          ?.filter((p): p is TextPart => p && 'text' in p)
          .map(p => p.text)
          .join('') ?? ''

      if (textDelta) {
        if (!isPotentiallyJson && aggregatedText === '' && textDelta.trim().match(/^[{[]/)) {
          isPotentiallyJson = true
        }
        aggregatedText += textDelta
        if (aggregatedResult) aggregatedResult.content = aggregatedText

        if (isPotentiallyJson) {
          let partialParsed = undefined
          try {
            partialParsed = JSON.parse(aggregatedText)
          } catch {}
          yield { type: 'json_delta', data: { delta: textDelta, parsed: partialParsed, snapshot: aggregatedText } } // Parenthesized
        } else {
          yield { type: 'content_delta', data: { delta: textDelta } } // Parenthesized
        }
      }

      // --- Function Call Delta ---
      // FIX: Safely handle null/undefined parts in the filter
      const functionCallParts = safeGet<GooglePart[]>(candidate, 'content', 'parts')?.filter(
        (p): p is FunctionCallPart => p && 'functionCall' in p
      )

      if (functionCallParts && functionCallParts.length > 0) {
        const newCalls = mapToolCallsFromGoogle(functionCallParts.map(p => p.functionCall))
        if (newCalls) {
          for (const tc of newCalls) {
            if (!aggregatedToolCalls.some(existing => existing.id === tc.id)) {
              const overallIndex = aggregatedToolCalls.length
              aggregatedToolCalls.push(tc)
              if (aggregatedResult && aggregatedResult.toolCalls) aggregatedResult.toolCalls.push(tc)

              // Yield is now valid within the generator scope
              yield {
                type: 'tool_call_start',
                data: {
                  index: overallIndex,
                  toolCall: { id: tc.id, type: 'function', function: { name: tc.function.name } }
                }
              } // Parenthesized
              yield {
                type: 'tool_call_delta',
                data: { index: overallIndex, id: tc.id, functionArgumentChunk: tc.function.arguments }
              } // Parenthesized
              yield { type: 'tool_call_done', data: { index: overallIndex, id: tc.id } } // Parenthesized
              finalFinishReason = 'tool_calls'
            }
          } // End for...of loop
        }
      }

      // --- Citation Delta ---
      const citationsChunk = mapCitationsFromGoogle(candidate.citationMetadata)
      if (citationsChunk) {
        for (const citation of citationsChunk) {
          if (
            !aggregatedCitations.some(
              existing => existing.sourceId === citation.sourceId && existing.startIndex === citation.startIndex
            )
          ) {
            const overallIndex = aggregatedCitations.length
            aggregatedCitations.push(citation)
            if (aggregatedResult && aggregatedResult.citations) aggregatedResult.citations.push(citation)

            // Yield is now valid within the generator scope
            yield { type: 'citation_delta', data: { index: overallIndex, citation } } // Parenthesized
            yield { type: 'citation_done', data: { index: overallIndex, citation } } // Parenthesized
          }
        } // End for...of loop
      }

      // --- Finish Reason ---
      const reason = candidate.finishReason
      if (reason && reason !== 'FINISH_REASON_UNSPECIFIED' && finalFinishReason !== 'tool_calls') {
        if (reason === 'SAFETY') finalFinishReason = 'content_filter'
        else if (reason === 'RECITATION') finalFinishReason = 'recitation_filter'
        else if (reason === 'MAX_TOKENS') finalFinishReason = 'length'
        else if (reason === 'STOP') finalFinishReason = 'stop'
        else finalFinishReason = reason.toLowerCase()
        if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason
      }
    } // End main stream loop (for await...)

    // --- End of Stream ---
    if (
      aggregatedResult &&
      (aggregatedResult.finishReason === null || aggregatedResult.finishReason === 'tool_calls') &&
      aggregatedText
    ) {
      aggregatedResult.finishReason = 'stop'
      finalFinishReason = 'stop'
    }
    finalFinishReason = finalFinishReason ?? 'stop'
    if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

    if (isPotentiallyJson) {
      let finalParsedJson = null
      try {
        finalParsedJson = JSON.parse(aggregatedText)
      } catch {}
      yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: aggregatedText } } // Parenthesized
      if (aggregatedResult) aggregatedResult.parsedContent = finalParsedJson
    }

    yield { type: 'message_stop', data: { finishReason: finalFinishReason } } // Parenthesized

    if (currentUsage) {
      yield { type: 'final_usage', data: { usage: currentUsage } } // Parenthesized
    }

    if (aggregatedResult) {
      if (!isPotentiallyJson && aggregatedResult.content === '') aggregatedResult.content = null
      if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
      if (aggregatedResult.citations?.length === 0) aggregatedResult.citations = undefined
      yield { type: 'final_result', data: { result: aggregatedResult } } // Parenthesized
    } else {
      console.warn('Google stream finished, but no aggregated result was built.')
    }
  } catch (error) {
    // Use unknown for catch block errors
    const mappedError =
      error instanceof RosettaAIError
        ? error
        : new ProviderAPIError(String(error), Provider.Google, undefined, undefined, undefined, error)
    yield { type: 'error', data: { error: mappedError } } // Parenthesized
  }
}
