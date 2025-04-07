import OpenAI from 'openai'
import {
  ChatCompletionToolChoiceOption as OpenAIToolChoiceOption,
  ChatCompletionContentPart as OpenAIContentPart,
  ChatCompletionContentPartText,
  ChatCompletionContentPartRefusal,
  ChatCompletionTool as OpenAITool,
  ChatCompletionMessageParam as OpenAIMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming
} from 'openai/resources/chat/completions'
import { FunctionDefinition as OpenAIFunctionDef } from 'openai/resources/shared'
import { Uploadable as OpenAIUploadable } from 'openai/uploads'
import { Stream } from 'openai/streaming'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  Provider,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult
} from '../../types'
import { MappingError, UnsupportedFeatureError, RosettaAIError } from '../../errors'
import { IProviderMapper } from './base.mapper'
import { mapBaseParams, mapBaseToolChoice } from './common.utils'
import * as OpenAIEmbedMapper from './openai.embed.mapper'
import * as OpenAIAudioMapper from './openai.audio.mapper'
import {
  mapContentForOpenAIRole,
  mapFromOpenAIResponse,
  mapOpenAIStream,
  mapRoleToOpenAI,
  wrapOpenAIError
} from './openai.common'

export class OpenAIMapper implements IProviderMapper {
  readonly provider = Provider.OpenAI

  // --- Chat/Completion Mapping ---

  mapToProviderParams(
    params: GenerateParams
  ): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
    const messages: OpenAIMessageParam[] = params.messages.map(msg => {
      const role = mapRoleToOpenAI(msg.role)
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
            throw new MappingError('System message content could not be resolved to string.', this.provider)
          }
          // Ensure non-empty string for system role
          if (systemContentString === '') {
            throw new MappingError(`Role 'system' requires non-empty string content.`, this.provider)
          }
          return { role: 'system', content: systemContentString } as ChatCompletionSystemMessageParam
        case 'user':
          // Ensure content is not null or empty array for user role
          if (content === null || (Array.isArray(content) && content.length === 0)) {
            throw new MappingError(`Role 'user' requires non-empty content.`, this.provider)
          }
          return { role: 'user', content: content as string | OpenAIContentPart[] } as ChatCompletionUserMessageParam
        case 'assistant':
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
          if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', this.provider)
          // Content mapping ensures string for tool role, and non-empty
          if (typeof content !== 'string') {
            // Should not happen if mapContentForOpenAIRole throws for invalid types
            throw new MappingError(
              `Tool message content must resolve to a string. Got: ${typeof content}`,
              this.provider
            )
          }
          return {
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content: content // content is non-empty string here
          } as ChatCompletionToolMessageParam
        default:
          throw new MappingError(`Unhandled role type during message construction: ${role}`, this.provider)
      }
    })

    const tools: OpenAITool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Unsupported tool type for OpenAI: ${tool.type}`, this.provider)
      }
      const parameters = tool.function.parameters as OpenAIFunctionDef['parameters']
      if (typeof parameters !== 'object' || parameters === null) {
        throw new MappingError(
          `Invalid parameters schema for tool ${tool.function.name}. Expected JSON Schema object.`,
          this.provider
        )
      }
      return {
        type: tool.type,
        function: { name: tool.function.name, description: tool.function.description, parameters: parameters }
      }
    })

    // Use common utility for base tool choice mapping
    const baseToolChoice = mapBaseToolChoice(params.toolChoice)
    let openAIToolChoice: OpenAIToolChoiceOption | undefined
    if (baseToolChoice === 'auto' || baseToolChoice === 'none' || baseToolChoice === 'required') {
      openAIToolChoice = baseToolChoice
    } else if (typeof baseToolChoice === 'object' && baseToolChoice.type === 'function') {
      openAIToolChoice = { type: 'function', function: { name: baseToolChoice.function.name } }
    }

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

    if (params.thinking) {
      throw new UnsupportedFeatureError(this.provider, 'Thinking steps')
    }
    if (params.grounding) {
      throw new UnsupportedFeatureError(this.provider, 'Grounding/Citations')
    }

    // Use common utility for base parameters
    const baseMappedParams = mapBaseParams(params)

    const basePayload = {
      model: params.model!,
      messages,
      max_tokens: baseMappedParams.maxTokens,
      temperature: baseMappedParams.temperature,
      top_p: baseMappedParams.topP,
      stop: baseMappedParams.stopSequences, // Use mapped stopSequences
      tools,
      tool_choice: openAIToolChoice,
      response_format: responseFormat
    }

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

  mapFromProviderResponse(response: ChatCompletion, modelUsed: string): GenerateResult {
    // Delegate to the base OpenAI mapper function
    return mapFromOpenAIResponse(response, modelUsed)
  }

  async *mapProviderStream(stream: Stream<ChatCompletionChunk>): AsyncIterable<StreamChunk> {
    // Delegate to the base OpenAI stream mapper function
    yield* mapOpenAIStream(stream, this.provider)
  }

  // --- Embedding Mapping ---
  mapToEmbedParams(params: EmbedParams): OpenAI.Embeddings.EmbeddingCreateParams {
    // Delegate to the specific embed mapper function
    return OpenAIEmbedMapper.mapToOpenAIEmbedParams(params)
  }

  mapFromEmbedResponse(response: OpenAI.Embeddings.CreateEmbeddingResponse, modelId: string): EmbedResult {
    // Delegate to the specific embed mapper function
    return OpenAIEmbedMapper.mapFromOpenAIEmbedResponse(response, modelId)
  }

  // --- Audio Mapping ---
  mapToTranscribeParams(params: TranscribeParams, file: OpenAIUploadable): OpenAI.Audio.TranscriptionCreateParams {
    // Delegate to the specific audio mapper function
    return OpenAIAudioMapper.mapToOpenAITranscribeParams(params, file)
  }

  mapFromTranscribeResponse(response: OpenAI.Audio.Transcriptions.Transcription, modelId: string): TranscriptionResult {
    // Delegate to the specific audio mapper function
    return OpenAIAudioMapper.mapFromOpenAITranscriptionResponse(response, modelId)
  }

  mapToTranslateParams(params: TranslateParams, file: OpenAIUploadable): OpenAI.Audio.TranslationCreateParams {
    // Delegate to the specific audio mapper function
    return OpenAIAudioMapper.mapToOpenAITranslateParams(params, file)
  }

  mapFromTranslateResponse(response: OpenAI.Audio.Translations.Translation, modelId: string): TranscriptionResult {
    // Delegate to the specific audio mapper function
    return OpenAIAudioMapper.mapFromOpenAITranslationResponse(response, modelId)
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    return wrapOpenAIError(error, provider)
  }
}
