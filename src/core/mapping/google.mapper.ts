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
  TextPart,
  Part as GooglePart,
  EmbedContentRequest,
  BatchEmbedContentsRequest
} from '@google/generative-ai'
import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider,
  Citation,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult
} from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError, UnsupportedFeatureError } from '../../errors'
import { safeGet } from '../utils'
import { IProviderMapper } from './base.mapper'
import { mapTokenUsage, mapBaseParams } from './common.utils'
import * as GoogleEmbedMapper from './google.embed.mapper'

export class GoogleMapper implements IProviderMapper {
  readonly provider = Provider.Google

  // --- Parameter Mapping ---
  private mapRoleToGoogle(role: RosettaMessage['role']): 'user' | 'model' | 'function' | 'system' {
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
        // Ensure exhaustive check works with `never`
        const _e: never = role
        throw new MappingError(`Unsupported role: ${_e}`, this.provider)
    }
  }

  private mapContentToGoogleParts(content: RosettaMessage['content']): GooglePart[] {
    if (content === null) {
      console.warn('Mapping null content to empty parts array for Google history.')
      return []
    }
    if (typeof content === 'string') {
      // Handle empty string case - return empty array as Google requires non-empty parts for user messages
      if (content === '') {
        console.warn('Mapping empty string content to empty parts array for Google history.')
        return []
      }
      return [{ text: content }]
    }
    // Handle empty array case
    if (Array.isArray(content) && content.length === 0) {
      console.warn('Mapping empty content array to empty parts array for Google history.')
      return []
    }
    return content.map(part => {
      if (part.type === 'text') return { text: part.text }
      if (part.type === 'image') return { inlineData: { mimeType: part.image.mimeType, data: part.image.base64Data } }
      // Ensure exhaustive check works with `never`
      const _e: never = part
      throw new MappingError(`Unsupported content part: ${(_e as any).type}`, this.provider)
    })
  }

  private isFunctionDeclarationSchema(schema: any): schema is FunctionDeclarationSchema {
    return (
      typeof schema === 'object' &&
      schema !== null &&
      'type' in schema &&
      Object.values(GoogleSchemaType).includes(schema.type)
    )
  }

  private findLastToolCallName(history: GoogleContent[], _toolCallId: string): string | undefined {
    for (let i = history.length - 1; i >= 0; i--) {
      const prevMsg = history[i]
      if (prevMsg?.role === 'model' && Array.isArray(prevMsg.parts)) {
        for (const part of prevMsg.parts) {
          if ('functionCall' in part && part.functionCall?.name) {
            return part.functionCall.name
          }
        }
      }
    }
    console.warn(`Could not determine preceding function name for tool result (ID: ${_toolCallId}) from history.`)
    return undefined
  }

  mapToProviderParams(
    params: GenerateParams
  ): { googleMappedParams: GenerateContentRequest | (StartChatParams & { contents: GooglePart[] }); isChat: boolean } {
    let systemInstruction: GoogleContent | undefined = undefined
    const history: GoogleContent[] = []
    const messagesToProcess = [...params.messages]

    const lastMessage = messagesToProcess.pop()
    if (!lastMessage) {
      throw new MappingError('No messages provided to map for Google.', this.provider)
    }

    messagesToProcess.forEach(msg => {
      const googleRole = this.mapRoleToGoogle(msg.role)

      if (googleRole === 'system') {
        if (systemInstruction)
          throw new MappingError('Multiple system messages not supported by Google.', this.provider)
        if (typeof msg.content !== 'string')
          throw new MappingError('Google system instruction must be string.', this.provider)
        systemInstruction = { role: 'system', parts: [{ text: msg.content }] }
        return
      }

      const parts = this.mapContentToGoogleParts(msg.content)
      // Skip adding history entries if parts array is empty (from null/empty string/array content)
      if (parts.length === 0 && googleRole !== 'model') {
        // Allow empty parts for model role if tool calls are present
        // @ts-ignore
        if (!(googleRole === 'model' && msg.toolCalls && msg.toolCalls.length > 0)) {
          console.warn(`Skipping history message with role '${googleRole}' due to empty content parts.`)
          return
        }
      }

      if (googleRole === 'model' && msg.toolCalls && msg.toolCalls.length > 0) {
        const functionCallParts: FunctionCallPart[] = msg.toolCalls.map(tc => {
          try {
            return { functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) } }
          } catch (e) {
            throw new MappingError(
              `Failed to parse arguments for tool ${tc.function.name}`,
              this.provider,
              'mapToProviderParams toolCall mapping',
              e
            )
          }
        })
        const existingTextParts = parts.filter((p): p is TextPart => 'text' in p)
        // Ensure parts array is not empty if only function calls exist
        const finalParts = [...existingTextParts, ...functionCallParts]
        if (finalParts.length === 0) {
          // This case should be rare, but handle defensively
          console.warn(`Model message with tool calls resulted in empty parts array.`)
          return // Skip adding empty message
        }
        history.push({ role: googleRole, parts: finalParts })
      } else if (googleRole === 'function') {
        if (!msg.toolCallId || typeof msg.content !== 'string') {
          throw new MappingError(
            'Invalid tool result message for Google history. Requires toolCallId and string content.',
            this.provider
          )
        }
        const funcName = this.findLastToolCallName(history, msg.toolCallId)
        if (!funcName) {
          throw new MappingError(
            `Cannot find function name for tool result (ID: ${msg.toolCallId}). Ensure model message with FunctionCall precedes this tool message.`,
            this.provider
          )
        }
        let respContent: any
        try {
          respContent = JSON.parse(msg.content)
        } catch {
          respContent = { content: msg.content } // Wrap non-JSON string content
          console.warn(`Tool result content for ${funcName} was not valid JSON. Wrapping as { content: "..." }`)
        }
        history.push({ role: googleRole, parts: [{ functionResponse: { name: funcName, response: respContent } }] })
      } else {
        // Only add if parts is not empty
        if (parts.length > 0) {
          history.push({ role: googleRole, parts })
        } else {
          console.warn(`Skipping history message with role '${googleRole}' due to empty content parts.`)
        }
      }
    })

    let currentTurnParts: GooglePart[]
    const lastMessageRole = this.mapRoleToGoogle(lastMessage.role)

    if (lastMessageRole === 'function') {
      if (!lastMessage.toolCallId || typeof lastMessage.content !== 'string') {
        throw new MappingError(
          'Invalid last message: Tool result requires toolCallId and string content.',
          this.provider
        )
      }
      const funcName = this.findLastToolCallName(history, lastMessage.toolCallId) // Check history before last message
      if (!funcName) {
        throw new MappingError(
          `Cannot find function name for final tool result (ID: ${lastMessage.toolCallId}).`,
          this.provider
        )
      }
      let respContent: any
      try {
        respContent = JSON.parse(lastMessage.content)
      } catch {
        respContent = { content: lastMessage.content } // Wrap non-JSON string content
        console.warn(`Final tool result content for ${funcName} was not valid JSON. Wrapping as { content: "..." }`)
      }
      currentTurnParts = [{ functionResponse: { name: funcName, response: respContent } }]
    } else if (lastMessageRole === 'user') {
      currentTurnParts = this.mapContentToGoogleParts(lastMessage.content)
      if (currentTurnParts.length === 0) {
        // Google requires the final user message to have content parts
        throw new MappingError('Final user message content cannot be null or empty.', this.provider)
      }
    } else {
      throw new MappingError(
        `Invalid role for the final message in a Google chat turn: '${lastMessageRole}'. Expected 'user' or 'tool'.`,
        this.provider
      )
    }

    const googleTools: GoogleTool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Only 'function' tools are currently supported for Google.`, this.provider)
      }
      const schema = tool.function.parameters
      if (!this.isFunctionDeclarationSchema(schema)) {
        throw new MappingError(
          `Invalid parameters schema for tool ${tool.function.name}. Expected FunctionDeclarationSchema.`,
          this.provider
        )
      }
      return {
        functionDeclarations: [{ name: tool.function.name, description: tool.function.description, parameters: schema }]
      }
    })

    let finalTools = googleTools
    if (params.grounding?.enabled) {
      const searchTool: GoogleTool = { googleSearchRetrieval: {} }
      if (params.grounding.source && params.grounding.source !== 'web') {
        console.warn(
          `Only 'web' grounding source currently mapped for Google Search Retrieval. Ignoring source: ${params.grounding.source}`
        )
      }
      finalTools = finalTools ? [...finalTools, searchTool] : [searchTool]
    }

    let responseMimeType: string | undefined
    if (params.responseFormat?.type === 'json_object') {
      responseMimeType = 'application/json'
      if (params.responseFormat.schema) {
        console.warn(
          'Google JSON mode requested via responseFormat. Ensure schema is described in the prompt. `schema` parameter is ignored for Google GenerationConfig.'
        )
      }
    }

    // Use common utility for base parameters
    const baseMappedParams = mapBaseParams(params)

    const generationConfig = {
      maxOutputTokens: baseMappedParams.maxTokens,
      temperature: baseMappedParams.temperature,
      topP: baseMappedParams.topP,
      stopSequences: baseMappedParams.stopSequences,
      responseMimeType: responseMimeType
    }

    // Determine if it's a chat or single-turn request
    const isChat = history.length > 0 || !!systemInstruction

    if (isChat) {
      const chatParams: StartChatParams = {
        history,
        generationConfig,
        tools: finalTools,
        systemInstruction
      }
      return { googleMappedParams: { ...chatParams, contents: currentTurnParts }, isChat: true }
    } else {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: currentTurnParts }],
        generationConfig,
        tools: finalTools,
        systemInstruction
      }
      return { googleMappedParams: request, isChat: false }
    }
  }

  // --- Result Mapping ---

  private mapToolCallsFromGoogle(calls: FunctionCall[] | undefined): RosettaToolCallRequest[] | undefined {
    if (!calls || calls.length === 0) return undefined
    return calls
      .filter(call => call?.name)
      .map((call, index) => {
        return {
          id: `google_func_${call.name}_${index}_${Date.now()}`,
          type: 'function',
          function: { name: call.name!, arguments: JSON.stringify(call.args ?? {}) }
        }
      })
  }

  private mapCitationsFromGoogle(metadata: CitationMetadata | undefined): Citation[] | undefined {
    if (!metadata?.citationSources || metadata.citationSources.length === 0) return undefined
    return metadata.citationSources.map((s, index) => ({
      sourceId: s.uri ?? `google_cite_idx_${index}`,
      startIndex: s.startIndex,
      endIndex: s.endIndex,
      text: undefined
    }))
  }

  mapFromProviderResponse(response: GenerateContentResponse | undefined, model: string): GenerateResult {
    const promptFeedbackReason = safeGet<string>(response, 'promptFeedback', 'blockReason')
    const promptFeedbackSafetyRatings = safeGet<any[]>(response, 'promptFeedback', 'safetyRatings')

    if (promptFeedbackReason) {
      const fr =
        promptFeedbackReason === 'SAFETY'
          ? 'content_filter'
          : promptFeedbackReason === 'OTHER'
          ? 'error'
          : promptFeedbackReason.toLowerCase()
      console.warn(`Google prompt blocked. Reason: ${fr}. Ratings: ${JSON.stringify(promptFeedbackSafetyRatings)}`)
      return {
        content: null,
        toolCalls: undefined,
        finishReason: fr,
        usage: mapTokenUsage(response?.usageMetadata), // Use common utility
        citations: undefined,
        parsedContent: null,
        thinkingSteps: undefined,
        model: model,
        rawResponse: response
      }
    }

    const candidate = response?.candidates?.[0]
    const candidateFinishReason = candidate?.finishReason
    const candidateSafetyRatings = candidate?.safetyRatings

    if (!response || !candidate) {
      console.warn('Google response or candidate is missing despite no prompt block.')
      return {
        content: null,
        toolCalls: undefined,
        finishReason: 'error',
        usage: mapTokenUsage(response?.usageMetadata), // Use common utility
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
    let finishReason = candidateFinishReason ?? 'unknown'

    if (candidate.content?.parts) {
      const textParts = candidate.content.parts.filter((p): p is TextPart => p && 'text' in p)
      if (textParts.length > 0) {
        textContent = textParts.map(p => p.text).join('')
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

      const functionCallParts = candidate.content.parts.filter((p): p is FunctionCallPart => p && 'functionCall' in p)
      if (functionCallParts.length > 0) {
        const mappedCalls = this.mapToolCallsFromGoogle(functionCallParts.map(p => p.functionCall))
        if (mappedCalls && mappedCalls.length > 0) {
          toolCalls = mappedCalls
          if (!['SAFETY', 'RECITATION', 'MAX_TOKENS'].includes(candidateFinishReason ?? '')) {
            finishReason = 'tool_calls'
          }
        }
      }
    }

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
      finishReason = 'stop'
    }

    const citations: Citation[] | undefined = this.mapCitationsFromGoogle(candidate.citationMetadata)

    return {
      content: textContent,
      toolCalls: toolCalls,
      finishReason: finishReason,
      usage: mapTokenUsage(response.usageMetadata), // Use common utility
      citations: citations,
      parsedContent: parsedJson,
      thinkingSteps: undefined,
      model: model,
      rawResponse: response
    }
  }

  // --- Stream Mapping ---

  async *mapProviderStream(stream: AsyncIterable<GenerateContentResponse>): AsyncIterable<StreamChunk> {
    let currentUsage: TokenUsage | undefined
    let finalFinishReason: string | null = null
    let aggregatedText = ''
    const aggregatedCitations: Citation[] = []
    const aggregatedToolCalls: RosettaToolCallRequest[] = []
    const model = '' // Model name isn't directly in the stream chunks
    let isPotentiallyJson = false
    let aggregatedResult: GenerateResult | null = null

    try {
      // Yield message_start immediately (model unknown initially)
      yield { type: 'message_start', data: { provider: this.provider, model: model } }

      for await (const chunk of stream) {
        // Aggregate usage metadata if present
        if (chunk.usageMetadata) {
          currentUsage = mapTokenUsage(chunk.usageMetadata) // Use common utility
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
            model: model, // Will be empty initially, maybe update later if possible?
            thinkingSteps: null,
            citations: [],
            parsedContent: null,
            rawResponse: undefined
          }
        }

        // --- Text Delta ---
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
            yield { type: 'json_delta', data: { delta: textDelta, parsed: partialParsed, snapshot: aggregatedText } }
          } else {
            yield { type: 'content_delta', data: { delta: textDelta } }
          }
        }

        // --- Function Call Delta ---
        const functionCallParts = safeGet<GooglePart[]>(candidate, 'content', 'parts')?.filter(
          (p): p is FunctionCallPart => p && 'functionCall' in p
        )

        if (functionCallParts && functionCallParts.length > 0) {
          const newCalls = this.mapToolCallsFromGoogle(functionCallParts.map(p => p.functionCall))
          if (newCalls) {
            for (const tc of newCalls) {
              // Check if this specific tool call ID has already been fully processed and added
              if (!aggregatedToolCalls.some(existing => existing.id === tc.id)) {
                const overallIndex = aggregatedToolCalls.length
                aggregatedToolCalls.push(tc) // Add the fully formed call
                if (aggregatedResult && aggregatedResult.toolCalls) aggregatedResult.toolCalls.push(tc)

                // Yield start, delta (full args), and done for this new call
                yield {
                  type: 'tool_call_start',
                  data: {
                    index: overallIndex,
                    toolCall: { id: tc.id, type: 'function', function: { name: tc.function.name } }
                  }
                }
                yield {
                  type: 'tool_call_delta',
                  data: { index: overallIndex, id: tc.id, functionArgumentChunk: tc.function.arguments }
                }
                yield { type: 'tool_call_done', data: { index: overallIndex, id: tc.id } }
                finalFinishReason = 'tool_calls' // Set finish reason if a tool call occurred
              }
            }
          }
        }

        // --- Citation Delta ---
        const citationsChunk = this.mapCitationsFromGoogle(candidate.citationMetadata)
        if (citationsChunk) {
          for (const citation of citationsChunk) {
            // Check if this citation has already been processed
            if (
              !aggregatedCitations.some(
                existing => existing.sourceId === citation.sourceId && existing.startIndex === citation.startIndex
              )
            ) {
              const overallIndex = aggregatedCitations.length
              aggregatedCitations.push(citation)
              if (aggregatedResult && aggregatedResult.citations) aggregatedResult.citations.push(citation)

              // Yield delta and done for this new citation
              yield { type: 'citation_delta', data: { index: overallIndex, citation } }
              yield { type: 'citation_done', data: { index: overallIndex, citation } }
            }
          }
        }

        // --- Finish Reason ---
        const reason = candidate.finishReason
        // Only update finalFinishReason if it's not already 'tool_calls'
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
      // Determine final reason if still null
      if (finalFinishReason === null) {
        if (aggregatedText || aggregatedToolCalls.length > 0 || aggregatedCitations.length > 0) {
          finalFinishReason = 'stop' // Assume normal stop if content/tools/citations were generated
        } else {
          finalFinishReason = 'unknown' // Otherwise, unknown reason
        }
      }
      if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

      if (isPotentiallyJson) {
        let finalParsedJson = null
        try {
          finalParsedJson = JSON.parse(aggregatedText)
        } catch {}
        yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: aggregatedText } }
        if (aggregatedResult) aggregatedResult.parsedContent = finalParsedJson
      }

      yield { type: 'message_stop', data: { finishReason: finalFinishReason } }

      if (currentUsage) {
        yield { type: 'final_usage', data: { usage: currentUsage } }
      }

      if (aggregatedResult) {
        if (!isPotentiallyJson && aggregatedResult.content === '') aggregatedResult.content = null
        if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
        if (aggregatedResult.citations?.length === 0) aggregatedResult.citations = undefined
        yield { type: 'final_result', data: { result: aggregatedResult } }
      } else {
        console.warn('Google stream finished, but no aggregated result was built.')
      }
    } catch (error) {
      const mappedError = this.wrapProviderError(error, this.provider)
      yield { type: 'error', data: { error: mappedError } }
    }
  }

  // --- Embedding Mapping ---
  mapToEmbedParams(params: EmbedParams): EmbedContentRequest | BatchEmbedContentsRequest {
    // Google Embeddings API uses different structures for single vs batch
    if (Array.isArray(params.input) && params.input.length > 1) {
      // Batch request
      const requests: EmbedContentRequest[] = params.input.map(text => ({
        model: `models/${params.model!}`, // Model needs prefix for batch
        content: { parts: [{ text }], role: 'user' }
      }))
      return { requests }
    } else {
      // Single request
      const inputText = Array.isArray(params.input) ? params.input[0] : params.input
      if (typeof inputText !== 'string' || inputText === '') {
        throw new MappingError('Input text for Google embedding cannot be empty.', this.provider)
      }
      return {
        // @ts-ignore
        model: `models/${params.model!}`, // Model needs prefix. should this be in google.embed.mapper.ts?
        content: { parts: [{ text: inputText }], role: 'user' }
      }
    }
  }

  mapFromEmbedResponse(response: any, modelId: string): EmbedResult {
    // Determine if it's a batch or single response based on structure
    if ('embeddings' in response && Array.isArray(response.embeddings)) {
      return GoogleEmbedMapper.mapFromGoogleEmbedBatchResponse(response, modelId)
    } else if ('embedding' in response) {
      return GoogleEmbedMapper.mapFromGoogleEmbedResponse(response, modelId)
    } else {
      throw new MappingError('Unknown Google embedding response structure.', this.provider)
    }
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
    // Check for specific Google error structures
    if (
      typeof error === 'object' &&
      error !== null &&
      'message' in error &&
      (error.constructor?.name?.includes('Google') ||
        (error as any).code ||
        (error as any).status ||
        (error as any).httpStatus ||
        (error as any).errorDetails)
    ) {
      const gError = error as any
      const statusCode =
        gError.httpStatus ?? gError.status ?? safeGet<number>(gError, 'response', 'status') ?? undefined
      const errorCode = safeGet<string>(gError, 'errorDetails', 0, 'reason') ?? gError.code ?? undefined
      const errorType = gError.name ?? safeGet<string>(gError, 'errorDetails', 0, 'type') ?? undefined
      const message = (error as Error).message || 'Unknown Google API Error'
      return new ProviderAPIError(message, provider, statusCode, errorCode, errorType, error)
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
