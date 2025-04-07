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
  ChatCompletionCreateParamsStreaming,
  ChatCompletionCreateParamsNonStreaming
} from 'openai/resources/chat/completions'
import { FunctionDefinition as OpenAIFunctionDef } from 'openai/resources/shared'
import {
  EmbeddingCreateParams,
  CreateEmbeddingResponse as OpenAICreateEmbeddingResponse
} from 'openai/resources/embeddings'
import { TranscriptionCreateParams } from 'openai/resources/audio/transcriptions'
import { TranslationCreateParams } from 'openai/resources/audio/translations'
import { Uploadable as OpenAIUploadable } from 'openai/uploads'
import { Stream } from 'openai/streaming'

import {
  GenerateParams,
  GenerateResult,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  StreamChunk,
  Provider,
  RosettaAIConfig
} from '../../types'
import { MappingError, UnsupportedFeatureError, ConfigurationError, RosettaAIError } from '../../errors'
import { IProviderMapper } from './base.mapper'
import { mapBaseParams, mapBaseToolChoice } from './common.utils'
import {
  mapContentForOpenAIRole,
  mapFromOpenAIResponse,
  mapOpenAIStream,
  mapRoleToOpenAI,
  wrapOpenAIError
} from './openai.common'
import { mapFromOpenAIEmbedResponse as mapFromOpenAIBaseEmbedResponse } from './openai.embed.mapper'
import {
  mapToOpenAITranscribeParams as mapToOpenAIBaseTranscribeParams,
  mapFromOpenAITranscriptionResponse as mapFromOpenAIBaseTranscriptionResponse,
  mapToOpenAITranslateParams as mapToOpenAIBaseTranslateParams,
  mapFromOpenAITranslationResponse as mapFromOpenAIBaseTranslationResponse
} from './openai.audio.mapper'

export class AzureOpenAIMapper implements IProviderMapper {
  readonly provider = Provider.OpenAI // Azure uses the OpenAI provider key

  // Store config for deployment ID lookups
  private config: RosettaAIConfig

  constructor(config: RosettaAIConfig) {
    this.config = config
  }

  // --- Chat/Completion Mapping ---

  mapToProviderParams(
    params: GenerateParams
  ): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
    const deploymentId =
      params.providerOptions?.azureChatDeploymentId ??
      this.config.providerOptions?.[Provider.OpenAI]?.azureChatDeploymentId ??
      this.config.azureOpenAIDefaultChatDeploymentName

    if (!deploymentId) {
      throw new ConfigurationError('Azure chat deployment ID/name must be configured.')
    }

    const messages: OpenAIMessageParam[] = params.messages.map(msg => {
      const role = mapRoleToOpenAI(msg.role)
      const content = mapContentForOpenAIRole(msg.content, role)

      switch (role) {
        case 'system':
          let systemContentString: string
          if (typeof content === 'string') {
            systemContentString = content
          } else if (Array.isArray(content)) {
            systemContentString = content.map(p => (p as ChatCompletionContentPartText).text).join('')
          } else {
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
            if (assistantMsg.content === '' || content === null) {
              assistantMsg.content = null
            }
          } else if (assistantMsg.content === null) {
            assistantMsg.content = ''
          }
          return assistantMsg
        case 'tool':
          if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', this.provider)
          if (typeof content !== 'string') {
            throw new MappingError(
              `Tool message content must resolve to a string. Got: ${typeof content}`,
              this.provider
            )
          }
          // Ensure non-empty string for tool role
          if (content === '') {
            throw new MappingError(`Role 'tool' requires non-empty string content.`, this.provider)
          }
          return {
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content
          } as ChatCompletionToolMessageParam
        default:
          throw new MappingError(`Unhandled role type during message construction: ${role}`, this.provider)
      }
    })

    const tools: OpenAITool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function')
        throw new MappingError(`Unsupported tool type for Azure OpenAI: ${tool.type}`, this.provider)
      const parameters = tool.function.parameters as OpenAIFunctionDef['parameters']
      if (typeof parameters !== 'object' || parameters === null)
        throw new MappingError(`Invalid parameters schema for tool ${tool.function.name}.`, this.provider)
      return {
        type: tool.type,
        function: { name: tool.function.name, description: tool.function.description, parameters }
      }
    })

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
          'Azure OpenAI JSON mode: schema parameter provided in responseFormat is ignored. Describe the desired schema in the prompt.'
        )
      }
    } else if (params.responseFormat?.type === 'text') {
      responseFormat = { type: 'text' }
    }

    if (params.thinking) throw new UnsupportedFeatureError(this.provider, 'Thinking steps')
    if (params.grounding) throw new UnsupportedFeatureError(this.provider, 'Grounding/Citations')

    const baseMappedParams = mapBaseParams(params)

    const basePayload = {
      model: deploymentId, // Use deployment ID as model for Azure
      messages,
      max_tokens: baseMappedParams.maxTokens,
      temperature: baseMappedParams.temperature,
      top_p: baseMappedParams.topP,
      stop: baseMappedParams.stopSequences,
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

  mapToEmbedParams(params: EmbedParams): EmbeddingCreateParams {
    const deploymentId =
      params.providerOptions?.azureEmbeddingDeploymentId ??
      this.config.providerOptions?.[Provider.OpenAI]?.azureEmbeddingDeploymentId ??
      this.config.azureOpenAIDefaultEmbeddingDeploymentName
    if (!deploymentId) {
      throw new ConfigurationError('Azure embedding deployment ID/name must be configured.')
    }
    let inputData: EmbeddingCreateParams['input']
    if (typeof params.input === 'string' || Array.isArray(params.input)) {
      inputData = params.input
    } else {
      throw new MappingError('Invalid input type for Azure OpenAI embeddings.', this.provider)
    }
    return {
      model: deploymentId, // Use deployment ID as model
      input: inputData,
      encoding_format: params.encodingFormat,
      dimensions: params.dimensions
    }
  }

  mapFromEmbedResponse(response: OpenAICreateEmbeddingResponse, modelUsed: string): EmbedResult {
    // Delegate to the base OpenAI embed mapper function
    return mapFromOpenAIBaseEmbedResponse(response, modelUsed)
  }

  // --- Audio Mapping ---

  mapToTranscribeParams(params: TranscribeParams, file: OpenAIUploadable): TranscriptionCreateParams {
    // Delegate to the base OpenAI audio mapper function
    return mapToOpenAIBaseTranscribeParams(params, file)
  }

  mapFromTranscribeResponse(response: OpenAI.Audio.Transcriptions.Transcription, modelId: string): TranscriptionResult {
    // Delegate to the base OpenAI audio mapper function
    return mapFromOpenAIBaseTranscriptionResponse(response, modelId)
  }

  mapToTranslateParams(params: TranslateParams, file: OpenAIUploadable): TranslationCreateParams {
    // Delegate to the base OpenAI audio mapper function
    return mapToOpenAIBaseTranslateParams(params, file)
  }

  mapFromTranslateResponse(response: OpenAI.Audio.Translations.Translation, modelId: string): TranscriptionResult {
    // Delegate to the base OpenAI audio mapper function
    return mapFromOpenAIBaseTranslationResponse(response, modelId)
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    // Reuse the same logic as standard OpenAI error wrapping
    return wrapOpenAIError(error, provider)
  }
}
