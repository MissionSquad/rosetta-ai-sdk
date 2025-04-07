import Groq from 'groq-sdk'
import {
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionContentPart,
  ChatCompletionTool,
  ChatCompletionRole,
  ChatCompletionToolChoiceOption,
  ChatCompletionCreateParams,
  ChatCompletion,
  ChatCompletionChunk
} from 'groq-sdk/resources/chat/completions'
import { Uploadable as GroqUploadable } from 'groq-sdk/core'

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
import { MappingError, UnsupportedFeatureError, ProviderAPIError, RosettaAIError } from '../../errors'
import { safeGet } from '../utils'
import { IProviderMapper } from './base.mapper'
import { mapTokenUsage, mapBaseParams, mapBaseToolChoice } from './common.utils'
import * as GroqEmbedMapper from './groq.embed.mapper'
import * as GroqAudioMapper from './groq.audio.mapper'

export class GroqMapper implements IProviderMapper {
  readonly provider = Provider.Groq

  // --- Parameter Mapping (Chat/Completion) ---

  private mapRoleToGroq(role: RosettaMessage['role']): ChatCompletionRole {
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
        throw new MappingError(`Unsupported role: ${_e}`, this.provider)
    }
  }

  private mapContentForGroqRole(
    content: RosettaMessage['content'],
    role: ChatCompletionRole
  ): string | Array<ChatCompletionContentPart> | null {
    if (content === null) {
      if (role === 'assistant' || role === 'tool') return null
      throw new MappingError(`Role '${role}' requires non-null content for Groq.`, this.provider)
    }
    if (typeof content === 'string') {
      // Handle empty string - Groq might require non-empty string for some roles
      // FIX: Include 'tool' role in the empty string check
      if (content === '' && (role === 'system' || role === 'user' || role === 'tool')) {
        throw new MappingError(`Role '${role}' requires non-empty string content for Groq.`, this.provider)
      }
      return content
    }
    // Handle empty array case
    if (Array.isArray(content) && content.length === 0) {
      if (role === 'system' || role === 'tool' || role === 'user') {
        throw new MappingError(`Role '${role}' requires non-empty content array. Received empty array.`, this.provider)
      }
      // For assistant, empty array maps to null below
    }

    const mappedParts: ChatCompletionContentPart[] = content.map(part => {
      if (part.type === 'text') {
        return { type: 'text', text: part.text }
      } else if (part.type === 'image') {
        if (role !== 'user') {
          throw new MappingError(
            `Image content parts are only allowed for the 'user' role in Groq, not '${role}'.`,
            this.provider
          )
        }
        return {
          type: 'image_url',
          image_url: { url: `data:${part.image.mimeType};base64,${part.image.base64Data}` }
        }
      } else {
        // Ensure exhaustive check works with `never`
        const _e: never = part
        throw new MappingError(`Unsupported content part type: ${(_e as any).type}`, this.provider)
      }
    })

    if (role === 'user') {
      return mappedParts
    } else if (role === 'assistant') {
      const textParts = mappedParts
        .filter(p => p.type === 'text')
        .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
      if (textParts.length !== mappedParts.length) {
        throw new MappingError('Assistant message content for Groq must resolve to a string or be null.', this.provider)
      }
      // Return null if no text parts remain (e.g., input was [])
      return textParts.length > 0 ? textParts.join('') : null
    } else if (role === 'system') {
      const textParts = mappedParts
        .filter(p => p.type === 'text')
        .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
      if (textParts.length !== mappedParts.length) {
        throw new MappingError(`System message content for Groq must resolve to a string.`, this.provider)
      }
      // Ensure non-empty string for system role
      const joinedText = textParts.join('')
      if (joinedText === '') {
        throw new MappingError(`Role 'system' requires non-empty string content for Groq.`, this.provider)
      }
      return joinedText
    } else if (role === 'tool') {
      const textParts = mappedParts
        .filter(p => p.type === 'text')
        .map(p => (p as Groq.Chat.Completions.ChatCompletionContentPartText).text)
      if (textParts.length !== mappedParts.length) {
        throw new MappingError('Tool message content for Groq must resolve to a string.', this.provider)
      }
      // Ensure non-empty string for tool role
      const joinedText = textParts.join('')
      // FIX: The check for empty string is now done earlier for the 'tool' role
      // if (joinedText === '') {
      //   throw new MappingError(`Role 'tool' requires non-empty string content for Groq.`, this.provider)
      // }
      return joinedText
    } else {
      throw new MappingError(`Cannot map content parts for unhandled Groq role '${role}'.`, this.provider)
    }
  }

  mapToProviderParams(params: GenerateParams): ChatCompletionCreateParams {
    // FIX: Add checks for invalid starting messages
    if (!params.messages || params.messages.length === 0) {
      throw new MappingError('Message list cannot be empty.', this.provider)
    }
    // @ts-ignore
    const firstRole = params.messages[0].role
    if (firstRole === 'assistant') {
      throw new MappingError('Conversation cannot start with an assistant message.', this.provider)
    }
    // Allow system message first, but ensure there's more than just system
    if (firstRole === 'system' && params.messages.length === 1) {
      throw new MappingError('Conversation cannot consist only of a system message.', this.provider)
    }
    // End FIX

    const messages: ChatCompletionMessageParam[] = params.messages.map(msg => {
      const role = this.mapRoleToGroq(msg.role)
      const content = this.mapContentForGroqRole(msg.content, role)

      switch (role) {
        case 'system':
          if (typeof content !== 'string') throw new MappingError('System content mismatch.', this.provider)
          return { role: 'system', content } as ChatCompletionSystemMessageParam
        case 'user':
          if (content === null) throw new MappingError('User content cannot be null.', this.provider)
          return { role: 'user', content } as ChatCompletionUserMessageParam
        case 'assistant':
          const assistantMsg: ChatCompletionAssistantMessageParam = {
            role: 'assistant',
            content: content as string | null
          }
          if (msg.toolCalls && msg.toolCalls.length > 0) {
            assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
              id: tc.id,
              type: tc.type as 'function',
              function: { name: tc.function.name, arguments: tc.function.arguments }
            }))
            // FIX: Ensure content is null if it was originally null/empty and tool calls exist
            if (content === null || content === '') assistantMsg.content = null
          } else if (assistantMsg.content === null) {
            // Groq requires assistant content to be non-null if no tool calls
            throw new MappingError(
              'Assistant message content cannot be null if no tool calls are present.',
              this.provider
            )
          }
          return assistantMsg
        case 'tool':
          if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', this.provider)
          if (typeof content !== 'string') {
            throw new MappingError(
              `Tool message content must be a string for Groq. Received: ${typeof content}`,
              this.provider
            )
          }
          return {
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content: content
          } as ChatCompletionToolMessageParam
        default:
          throw new MappingError(`Unhandled role type in message construction: ${role}`, this.provider)
      }
    })

    const tools: ChatCompletionTool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Unsupported tool type for Groq: ${tool.type}`, this.provider)
      }
      const parameters = tool.function.parameters
      if (typeof parameters !== 'object' || parameters === null || Array.isArray(parameters)) {
        throw new MappingError(
          `Invalid parameters schema for tool ${tool.function.name}. Expected JSON Schema object.`,
          this.provider
        )
      }
      if (parameters.type !== 'object') {
        throw new MappingError(
          `Invalid parameters schema for tool ${tool.function.name}. Groq requires top-level 'type: \"object\"'.`,
          this.provider
        )
      }
      return {
        type: tool.type,
        function: {
          name: tool.function.name,
          description: tool.function.description,
          parameters: parameters
        }
      }
    })

    // Use common utility for base tool choice mapping
    const baseToolChoice = mapBaseToolChoice(params.toolChoice)
    let groqToolChoice: ChatCompletionToolChoiceOption | undefined
    if (baseToolChoice === 'auto' || baseToolChoice === 'none') {
      groqToolChoice = baseToolChoice
    } else if (baseToolChoice === 'required') {
      console.warn("'required' tool_choice mapped to 'auto' for Groq.")
      groqToolChoice = 'auto'
    } else if (typeof baseToolChoice === 'object' && baseToolChoice.type === 'function') {
      groqToolChoice = { type: 'function', function: { name: baseToolChoice.function.name } }
    } else if (baseToolChoice) {
      // FIX: Default to undefined instead of 'auto' if format is invalid
      console.warn(`Unsupported tool_choice format for Groq: ${JSON.stringify(baseToolChoice)}. Ignoring.`)
      groqToolChoice = undefined
    }

    if (params.responseFormat?.type === 'json_object') {
      console.warn('JSON response format requested, but Groq support is unconfirmed via standard parameters.')
    }
    if (params.thinking) {
      throw new UnsupportedFeatureError(this.provider, 'Thinking steps')
    }
    if (params.grounding) {
      throw new UnsupportedFeatureError(this.provider, 'Grounding/Citations')
    }

    // Use common utility for base parameters
    const baseMappedParams = mapBaseParams(params)

    const payload: ChatCompletionCreateParams = {
      model: params.model!,
      messages: messages,
      max_tokens: baseMappedParams.maxTokens,
      temperature: baseMappedParams.temperature,
      top_p: baseMappedParams.topP,
      stop: baseMappedParams.stopSequences, // Use mapped stopSequences
      tools: tools,
      tool_choice: groqToolChoice,
      stream: params.stream
    }

    return payload
  }

  // --- Result Mapping (Chat/Completion) ---

  private mapToolCallsFromGroq(
    toolCalls: ReadonlyArray<Groq.Chat.Completions.ChatCompletionMessageToolCall> | undefined | null
  ): RosettaToolCallRequest[] | undefined {
    if (!toolCalls || toolCalls.length === 0) return undefined
    return toolCalls
      .filter(tc => tc.type === 'function' && tc.function?.name && tc.id)
      .map(tc => {
        return {
          id: tc.id!,
          type: tc.type,
          function: { name: tc.function!.name!, arguments: tc.function!.arguments ?? '{}' }
        }
      })
  }

  mapFromProviderResponse(response: ChatCompletion, modelUsed: string): GenerateResult {
    const choice = response.choices[0]
    if (!choice) {
      console.warn('Groq response missing choices.')
      const finishReason = safeGet<string>(response, 'choices', 0, 'finish_reason') ?? 'error'
      return {
        content: null,
        toolCalls: undefined,
        finishReason: finishReason,
        usage: mapTokenUsage(response.usage), // Use common utility
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
    const groqToolCalls = safeGet<ReadonlyArray<Groq.Chat.Completions.ChatCompletionMessageToolCall>>(
      choice.message,
      'tool_calls'
    )
    const mappedToolCalls = this.mapToolCallsFromGroq(groqToolCalls)
    const fr = choice.finish_reason
    const standardizedFR = fr === 'tool_calls' ? 'tool_calls' : fr === 'stop' ? 'stop' : fr === 'length' ? 'length' : fr
    return {
      content: textContent,
      toolCalls: mappedToolCalls,
      finishReason: standardizedFR,
      usage: mapTokenUsage(response.usage), // Use common utility
      citations: undefined,
      parsedContent: parsedJson,
      thinkingSteps: undefined,
      model: response.model ?? modelUsed,
      rawResponse: response
    }
  }

  // --- Stream Mapping (Chat/Completion) ---

  async *mapProviderStream(stream: AsyncIterable<ChatCompletionChunk>): AsyncIterable<StreamChunk> {
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
        // FIX: Add safety check for chunk structure
        if (typeof chunk !== 'object' || chunk === null) {
          console.warn('Received unexpected non-object chunk from Groq stream:', chunk)
          continue
        }

        if (!model && chunk.model) {
          model = chunk.model
          yield { type: 'message_start', data: { provider: this.provider, model: model } }
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

        // FIX: Add safety check for choices array
        const choice = chunk.choices?.[0]
        if (!choice) {
          // Handle chunks without choices (like usage-only chunks)
          const usageFromChunk = safeGet<Groq.CompletionUsage>(chunk, 'x_groq', 'usage')
          if (usageFromChunk) {
            finalUsage = mapTokenUsage(usageFromChunk) // Use common utility
            if (aggregatedResult) aggregatedResult.usage = finalUsage
          }
          continue // Skip to next chunk if no choice
        }

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
            reason === 'tool_calls'
              ? 'tool_calls'
              : reason === 'stop'
              ? 'stop'
              : reason === 'length'
              ? 'length'
              : reason

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
        }
        // FIX: Capture usage even if it arrives in the same chunk as finish_reason or delta
        const usageFromChunk = safeGet<Groq.CompletionUsage>(chunk, 'x_groq', 'usage')
        if (usageFromChunk) {
          finalUsage = mapTokenUsage(usageFromChunk) // Use common utility
          if (aggregatedResult) aggregatedResult.usage = finalUsage
        }
      } // End for await loop

      // --- FIX: Ensure final lifecycle events are always yielded ---
      finalFinishReason = finalFinishReason ?? 'stop' // Default to stop if null
      if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

      yield { type: 'message_stop', data: { finishReason: finalFinishReason } }

      // Yield final usage, even if undefined
      // @ts-ignore
      yield { type: 'final_usage', data: { usage: finalUsage } }

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
        // Still yield a final result even if aggregation failed (e.g., only usage chunk received)
        console.warn('Groq stream finished but no aggregated result was built. Yielding empty final result.')
        yield {
          type: 'final_result',
          data: {
            result: {
              content: null,
              toolCalls: undefined,
              finishReason: finalFinishReason,
              usage: finalUsage,
              model: model,
              thinkingSteps: null,
              citations: undefined,
              parsedContent: null,
              rawResponse: undefined
            }
          }
        }
      }
      // --- End FIX ---
    } catch (error) {
      const mappedError = this.wrapProviderError(error, this.provider)
      yield { type: 'error', data: { error: mappedError } }
    }
  }

  // --- Embedding Mapping ---
  mapToEmbedParams(params: EmbedParams): Groq.Embeddings.EmbeddingCreateParams {
    return GroqEmbedMapper.mapToGroqEmbedParams(params)
  }

  mapFromEmbedResponse(response: Groq.Embeddings.CreateEmbeddingResponse, modelId: string): EmbedResult {
    return GroqEmbedMapper.mapFromGroqEmbedResponse(response, modelId)
  }

  // --- Audio Mapping ---
  mapToTranscribeParams(params: TranscribeParams, file: GroqUploadable): Groq.Audio.TranscriptionCreateParams {
    return GroqAudioMapper.mapToGroqSttParams(params, file)
  }

  mapFromTranscribeResponse(response: Groq.Audio.Transcription, modelId: string): TranscriptionResult {
    return GroqAudioMapper.mapFromGroqTranscriptionResponse(response, modelId)
  }

  mapToTranslateParams(params: TranslateParams, file: GroqUploadable): Groq.Audio.TranslationCreateParams {
    return GroqAudioMapper.mapToGroqTranslateParams(params, file)
  }

  mapFromTranslateResponse(response: Groq.Audio.Translation, modelId: string): TranscriptionResult {
    return GroqAudioMapper.mapFromGroqTranslationResponse(response, modelId)
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    if (error instanceof RosettaAIError) {
      return error
    }
    if (error instanceof Groq.APIError) {
      // FIX: Correctly extract nested error details and handle missing message
      const nestedMessage = safeGet<string>(error.error, 'message')
      const message =
        nestedMessage && nestedMessage.trim() ? nestedMessage.trim() : error.message ?? 'Unknown Groq API Error' // Fallback logic
      const code = safeGet<string>(error.error, 'code') // Correct path
      const type = safeGet<string>(error.error, 'type') // Correct path
      return new ProviderAPIError(message, provider, error.status, code, type, error)
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
