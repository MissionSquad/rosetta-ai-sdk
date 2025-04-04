import Anthropic from '@anthropic-ai/sdk'
import {
  GoogleGenerativeAI,
  GenerativeModel,
  HarmCategory,
  HarmBlockThreshold,
  StartChatParams,
  GenerateContentRequest,
  EmbedContentRequest,
  Part as GooglePart // Import Part type alias
} from '@google/generative-ai'
import Groq from 'groq-sdk'
import OpenAI, { AzureOpenAI } from 'openai'
import { Stream } from 'openai/streaming'
import {
  ChatCompletionCreateParams as GroqChatCompletionCreateParams,
  ChatCompletionChunk as GroqChatCompletionChunk,
  ChatCompletion as GroqChatCompletion
} from 'groq-sdk/resources/chat/completions'
import { Stream as GroqStreamType } from 'groq-sdk/lib/streaming' // Use type import for Groq Stream
import { Uploadable as OpenAIUploadable } from 'openai/uploads'
import { Uploadable as GroqUploadable } from 'groq-sdk/core'

import { config as dotenvConfig } from 'dotenv'

import {
  Provider,
  RosettaAIConfig,
  GenerateParams,
  GenerateResult,
  EmbedParams,
  EmbedResult,
  SpeechParams,
  AudioStreamChunk,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  StreamChunk,
  ProviderOptions
} from '../types'
import { ConfigurationError, ProviderAPIError, UnsupportedFeatureError, RosettaAIError, MappingError } from '../errors'

// Import mapping functions
import * as AnthropicMapper from './mapping/anthropic.mapper'
import * as GoogleMapper from './mapping/google.mapper'
import * as GroqMapper from './mapping/groq.mapper'
import * as OpenAIMapper from './mapping/openai.mapper'
import * as AzureOpenAIMapper from './mapping/azure.openai.mapper'
import * as GoogleEmbedMapper from './mapping/google.embed.mapper'
import { prepareAudioUpload, safeGet } from './utils'

dotenvConfig()

/**
 * RosettaAI: Unified SDK for Interacting with Multiple AI Providers.
 */
export class RosettaAI {
  /** @internal The configuration used by the client instance. */
  readonly config: RosettaAIConfig
  private anthropicClient?: Anthropic
  private googleClient?: GoogleGenerativeAI
  private groqClient?: Groq
  private openAIClient?: OpenAI
  private azureOpenAIClient?: AzureOpenAI

  /** Creates an instance of the RosettaAI client. */
  constructor(config: RosettaAIConfig = {}) {
    const loadEnv = (key: string): string | undefined => process.env[key]

    // Load configuration, prioritizing constructor args > env vars
    this.config = {
      anthropicApiKey: config.anthropicApiKey ?? loadEnv('ANTHROPIC_API_KEY'),
      googleApiKey: config.googleApiKey ?? loadEnv('GOOGLE_API_KEY'),
      groqApiKey: config.groqApiKey ?? loadEnv('GROQ_API_KEY'),
      openaiApiKey: config.openaiApiKey ?? loadEnv('OPENAI_API_KEY'),
      azureOpenAIApiKey: config.azureOpenAIApiKey ?? loadEnv('AZURE_OPENAI_API_KEY'),
      azureOpenAIEndpoint: config.azureOpenAIEndpoint ?? loadEnv('AZURE_OPENAI_ENDPOINT'),
      azureOpenAIDefaultChatDeploymentName:
        config.azureOpenAIDefaultChatDeploymentName ?? loadEnv('AZURE_OPENAI_DEPLOYMENT_NAME'),
      azureOpenAIDefaultEmbeddingDeploymentName:
        config.azureOpenAIDefaultEmbeddingDeploymentName ?? loadEnv('ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
      azureOpenAIApiVersion: config.azureOpenAIApiVersion ?? loadEnv('AZURE_OPENAI_API_VERSION'),
      // Default models with fallbacks from env
      defaultModels: {
        [Provider.Anthropic]: config.defaultModels?.[Provider.Anthropic] ?? loadEnv('ROSETTA_DEFAULT_ANTHROPIC_MODEL'),
        [Provider.Google]: config.defaultModels?.[Provider.Google] ?? loadEnv('ROSETTA_DEFAULT_GOOGLE_MODEL'),
        [Provider.Groq]: config.defaultModels?.[Provider.Groq] ?? loadEnv('ROSETTA_DEFAULT_GROQ_MODEL'),
        [Provider.OpenAI]: config.defaultModels?.[Provider.OpenAI] ?? loadEnv('ROSETTA_DEFAULT_OPENAI_MODEL'),
        ...config.defaultModels // Allow overriding with constructor args
      },
      defaultEmbeddingModels: {
        [Provider.Google]:
          config.defaultEmbeddingModels?.[Provider.Google] ?? loadEnv('ROSETTA_DEFAULT_EMBEDDING_GOOGLE_MODEL'),
        [Provider.OpenAI]:
          config.defaultEmbeddingModels?.[Provider.OpenAI] ?? loadEnv('ROSETTA_DEFAULT_EMBEDDING_OPENAI_MODEL'),
        [Provider.Groq]:
          config.defaultEmbeddingModels?.[Provider.Groq] ?? loadEnv('ROSETTA_DEFAULT_EMBEDDING_GROQ_MODEL'),
        ...config.defaultEmbeddingModels
      },
      defaultTtsModels: {
        [Provider.OpenAI]: config.defaultTtsModels?.[Provider.OpenAI] ?? loadEnv('ROSETTA_DEFAULT_TTS_OPENAI_MODEL'),
        ...config.defaultTtsModels
      },
      defaultSttModels: {
        [Provider.OpenAI]: config.defaultSttModels?.[Provider.OpenAI] ?? loadEnv('ROSETTA_DEFAULT_STT_OPENAI_MODEL'),
        [Provider.Groq]: config.defaultSttModels?.[Provider.Groq] ?? loadEnv('ROSETTA_DEFAULT_STT_GROQ_MODEL'),
        ...config.defaultSttModels
      },
      providerOptions: config.providerOptions,
      defaultMaxRetries: config.defaultMaxRetries ?? 2,
      defaultTimeoutMs: config.defaultTimeoutMs ?? 60 * 1000
    }

    this.initializeClients()
    this.validateConfiguration()
  }

  /** @internal Initializes clients based on configuration. */
  private initializeClients(): void {
    // Anthropic
    if (this.config.anthropicApiKey) {
      this.anthropicClient = new Anthropic({
        apiKey: this.config.anthropicApiKey,
        baseURL: this.config.providerOptions?.[Provider.Anthropic]?.baseURL,
        maxRetries: this.config.defaultMaxRetries,
        timeout: this.config.defaultTimeoutMs
      })
    }

    // Google
    if (this.config.googleApiKey) {
      // Google SDK doesn't take baseURL/retries/timeout in constructor
      this.googleClient = new GoogleGenerativeAI(this.config.googleApiKey)
    }

    // Groq
    if (this.config.groqApiKey) {
      try {
        this.groqClient = new Groq({
          apiKey: this.config.groqApiKey,
          baseURL: this.config.providerOptions?.[Provider.Groq]?.baseURL,
          maxRetries: this.config.defaultMaxRetries,
          timeout: this.config.defaultTimeoutMs
        })
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e)
        console.warn(`RosettaAI: Groq init failed: ${message}. Provider unavailable.`)
      }
    }

    // OpenAI / Azure OpenAI
    // Prioritize Azure if endpoint and key are provided
    if (this.config.azureOpenAIEndpoint && this.config.azureOpenAIApiKey && this.config.azureOpenAIApiVersion) {
      try {
        this.azureOpenAIClient = new AzureOpenAI({
          apiKey: this.config.azureOpenAIApiKey,
          endpoint: this.config.azureOpenAIEndpoint,
          apiVersion: this.config.azureOpenAIApiVersion,
          maxRetries: this.config.defaultMaxRetries,
          timeout: this.config.defaultTimeoutMs
          // deployment: this.config.azureOpenAIDefaultChatDeploymentName, // Deployment applied per-request if needed
        })
        console.log(
          `RosettaAI: Initialized Azure OpenAI client (Endpoint: ${this.config.azureOpenAIEndpoint}, API Version: ${this.config.azureOpenAIApiVersion}).`
        )
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e)
        console.warn(`RosettaAI: Azure OpenAI init failed: ${message}. Ensure endpoint and apiVersion are correct.`)
        // Fallback to standard OpenAI if key exists? Or just fail Azure? Let's just warn.
      }
    } else if (this.config.openaiApiKey) {
      // Initialize standard OpenAI if Azure isn't fully configured but OpenAI key exists
      this.openAIClient = new OpenAI({
        apiKey: this.config.openaiApiKey,
        baseURL: this.config.providerOptions?.[Provider.OpenAI]?.baseURL,
        maxRetries: this.config.defaultMaxRetries,
        timeout: this.config.defaultTimeoutMs
      })
      console.log('RosettaAI: Initialized standard OpenAI client.')
    }
  }

  /** @internal Validates necessary configuration is present. */
  private validateConfiguration(): void {
    const configured = this.getConfiguredProviders()
    if (configured.length === 0) {
      throw new ConfigurationError(
        'No AI providers configured. Please provide API keys via constructor or environment variables.'
      )
    }
    console.log(`RosettaAI: Active providers: ${configured.join(', ')}`)

    // Specific Azure warnings
    if (this.config.azureOpenAIEndpoint && !this.config.azureOpenAIApiKey && !this.azureOpenAIClient) {
      console.warn(
        'RosettaAI Warning: Azure OpenAI endpoint provided, but API key is missing or invalid. Azure OpenAI client not initialized.'
      )
    }
    if (!this.config.azureOpenAIEndpoint && this.config.azureOpenAIApiKey && !this.azureOpenAIClient) {
      console.warn(
        'RosettaAI Warning: Azure OpenAI API key provided, but endpoint is missing. Azure OpenAI client not initialized.'
      )
    }
    if (this.config.azureOpenAIEndpoint && this.config.azureOpenAIApiKey && !this.config.azureOpenAIApiVersion) {
      console.warn(
        'RosettaAI Warning: Azure OpenAI endpoint and key provided, but API version is missing. Azure OpenAI client not initialized.'
      )
    }
    if (this.azureOpenAIClient && this.openAIClient) {
      console.warn(
        "RosettaAI Warning: Both Azure and standard OpenAI clients are configured. Azure OpenAI will be prioritized for provider 'openai'."
      )
    }
  }

  /** Gets a list of successfully configured providers for this client instance. */
  public getConfiguredProviders(): Provider[] {
    const p: Provider[] = []
    if (this.anthropicClient) p.push(Provider.Anthropic)
    if (this.googleClient) p.push(Provider.Google)
    if (this.groqClient) p.push(Provider.Groq)
    // Provider.OpenAI is active if either Azure or standard client is initialized
    if (this.openAIClient || this.azureOpenAIClient) p.push(Provider.OpenAI)
    return p
  }

  /** Generates a chat completion (non-streaming). */
  public async generate(params: GenerateParams): Promise<GenerateResult> {
    const model = params.model ?? this.config.defaultModels?.[params.provider]
    if (!model) {
      throw new ConfigurationError(`Model must be specified for provider ${params.provider} (or set a default).`)
    }
    const effectiveParams = { ...params, model, stream: false }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, !!this.azureOpenAIClient)

    try {
      switch (params.provider) {
        case Provider.Anthropic:
          if (!this.anthropicClient) throw this.providerNotConfigured(Provider.Anthropic)
          const anthropicP = AnthropicMapper.mapToAnthropicParams(effectiveParams)
          const anthropicR = await this.anthropicClient.messages.create(
            anthropicP as Anthropic.Messages.MessageCreateParamsNonStreaming
          )
          if (Symbol.asyncIterator in anthropicR) {
            throw new MappingError('Anthropic returned a stream for a non-streaming request.', Provider.Anthropic)
          }
          return AnthropicMapper.mapFromAnthropicResponse(anthropicR, model)

        case Provider.Google:
          if (!this.googleClient) throw this.providerNotConfigured(Provider.Google)
          const googleM = this.getGoogleModel(model, params.providerOptions)
          const { googleMappedParams: googleP, isChat } = GoogleMapper.mapToGoogleParams(effectiveParams)

          if (isChat) {
            const chatP = googleP as StartChatParams & { contents: GooglePart[] }
            const chat = googleM.startChat(chatP)
            const googleCR = await chat.sendMessage(chatP.contents)
            return GoogleMapper.mapFromGoogleResponse(googleCR.response, model)
          } else {
            const googleR = await googleM.generateContent(googleP as GenerateContentRequest)
            return GoogleMapper.mapFromGoogleResponse(googleR.response, model)
          }

        case Provider.Groq:
          if (!this.groqClient) throw this.providerNotConfigured(Provider.Groq)
          const groqP = GroqMapper.mapToGroqParams(effectiveParams) as GroqChatCompletionCreateParams
          const groqR = await this.groqClient.chat.completions.create(groqP)
          if ('iterator' in groqR || Symbol.asyncIterator in groqR) {
            // Check both possible stream indicators
            throw new MappingError('Groq returned a stream for a non-streaming request.', Provider.Groq)
          }
          // Cast result to non-streaming type for the mapper
          return GroqMapper.mapFromGroqResponse(groqR as GroqChatCompletion, model)

        case Provider.OpenAI:
          if (this.azureOpenAIClient) {
            const azureP = AzureOpenAIMapper.mapToAzureOpenAIParams(effectiveParams, this.config)
            const azureR = await this.azureOpenAIClient.chat.completions.create(
              azureP as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
            )
            if (typeof (azureR as any)[Symbol.asyncIterator] === 'function')
              throw new MappingError('Azure returned stream for non-stream request.', Provider.OpenAI)
            return AzureOpenAIMapper.mapFromAzureOpenAIResponse(azureR as OpenAI.Chat.ChatCompletion, azureP.model)
          } else if (this.openAIClient) {
            const openaiP = OpenAIMapper.mapToOpenAIParams(effectiveParams)
            const openaiR = await this.openAIClient.chat.completions.create(
              openaiP as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
            )
            if (typeof (openaiR as any)[Symbol.asyncIterator] === 'function')
              throw new MappingError('OpenAI returned stream for non-stream request.', Provider.OpenAI)
            return OpenAIMapper.mapFromOpenAIResponse(openaiR as OpenAI.Chat.ChatCompletion, model)
          } else {
            throw this.providerNotConfigured(Provider.OpenAI)
          }

        default:
          const _e: never = params.provider
          throw new RosettaAIError(`Unsupported provider: ${_e}`)
      }
    } catch (error) {
      throw this.wrapProviderError(error, params.provider)
    }
  }

  /** Generates a streaming response. */
  public async *stream(params: GenerateParams): AsyncIterable<StreamChunk> {
    const model = params.model ?? this.config.defaultModels?.[params.provider]
    if (!model) {
      const ce = new ConfigurationError(`Model must be specified for provider ${params.provider} (or set a default).`)
      yield { type: 'error', data: { error: ce } } // Parenthesized
      throw ce
    }
    const effectiveParams = { ...params, model, stream: true }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, !!this.azureOpenAIClient)

    try {
      switch (params.provider) {
        case Provider.Anthropic:
          if (!this.anthropicClient) throw this.providerNotConfigured(Provider.Anthropic)
          const anthropicP = AnthropicMapper.mapToAnthropicParams(effectiveParams)
          const anthropicSR = await this.anthropicClient.messages.create(
            anthropicP as Anthropic.Messages.MessageCreateParamsStreaming
          )
          if (!(Symbol.asyncIterator in anthropicSR)) {
            throw new MappingError('Anthropic did not return a stream for a streaming request.', Provider.Anthropic)
          }
          yield* AnthropicMapper.mapAnthropicStream(anthropicSR)
          break

        case Provider.Google:
          if (!this.googleClient) throw this.providerNotConfigured(Provider.Google)
          const googleM = this.getGoogleModel(model, params.providerOptions)
          const { googleMappedParams: googleP, isChat } = GoogleMapper.mapToGoogleParams(effectiveParams)

          if (isChat) {
            const chatP = googleP as StartChatParams & { contents: GooglePart[] }
            const chat = googleM.startChat(chatP)
            const googleSR = await chat.sendMessageStream(chatP.contents)
            yield* GoogleMapper.mapGoogleStream(googleSR.stream)
          } else {
            const googleSR = await googleM.generateContentStream(googleP as GenerateContentRequest)
            yield* GoogleMapper.mapGoogleStream(googleSR.stream)
          }
          break

        case Provider.Groq:
          if (!this.groqClient) throw this.providerNotConfigured(Provider.Groq)
          const groqP = GroqMapper.mapToGroqParams(effectiveParams) as GroqChatCompletionCreateParams
          const groqSR = await this.groqClient.chat.completions.create(groqP)
          if (!(typeof (groqSR as any)[Symbol.asyncIterator] === 'function')) {
            throw new MappingError(
              'Groq did not return a stream (async iterator) for a streaming request.',
              Provider.Groq
            )
          }
          yield* GroqMapper.mapGroqStream(groqSR as GroqStreamType<GroqChatCompletionChunk>)
          break

        case Provider.OpenAI:
          if (this.azureOpenAIClient) {
            const azureP = AzureOpenAIMapper.mapToAzureOpenAIParams(effectiveParams, this.config)
            const azureSR = await this.azureOpenAIClient.chat.completions.create(
              azureP as OpenAI.Chat.ChatCompletionCreateParamsStreaming
            )
            if (!(typeof (azureSR as any)[Symbol.asyncIterator] === 'function'))
              throw new MappingError('Azure did not return a stream for a streaming request.', Provider.OpenAI)
            yield* AzureOpenAIMapper.mapAzureOpenAIStream(azureSR as Stream<OpenAI.Chat.ChatCompletionChunk>)
          } else if (this.openAIClient) {
            const openaiP = OpenAIMapper.mapToOpenAIParams(effectiveParams)
            const openaiSR = await this.openAIClient.chat.completions.create(
              openaiP as OpenAI.Chat.ChatCompletionCreateParamsStreaming
            )
            if (!(typeof (openaiSR as any)[Symbol.asyncIterator] === 'function'))
              throw new MappingError('OpenAI did not return a stream for a streaming request.', Provider.OpenAI)
            yield* OpenAIMapper.mapOpenAIStream(openaiSR as Stream<OpenAI.Chat.ChatCompletionChunk>)
          } else {
            throw this.providerNotConfigured(Provider.OpenAI)
          }
          break

        default:
          const _e: never = params.provider
          throw new RosettaAIError(`Unsupported provider: ${_e}`)
      }
    } catch (error) {
      const wrappedError = this.wrapProviderError(error, params.provider)
      yield { type: 'error', data: { error: wrappedError } } // Parenthesized
    }
  }

  /** Generates embedding vectors. */
  public async embed(params: EmbedParams): Promise<EmbedResult> {
    const provider = params.provider
    const model = params.model ?? this.config.defaultEmbeddingModels?.[provider]
    if (!model) {
      throw new ConfigurationError(`Embedding model must be specified for provider ${provider} (or set a default).`)
    }
    const effectiveParams = { ...params, model }
    const eP = effectiveParams // Alias for readability inside switch

    this.checkUnsupportedFeatures(provider, effectiveParams, !!this.azureOpenAIClient)

    try {
      switch (provider) {
        case Provider.Google:
          if (!this.googleClient) throw this.providerNotConfigured(provider)
          const googleEM = this.googleClient.getGenerativeModel({ model })
          if (Array.isArray(eP.input) && eP.input.length > 1) {
            const requests: EmbedContentRequest[] = eP.input.map(text => ({
              content: { parts: [{ text }], role: 'user' }
            }))
            const batchRequest = { requests }
            const gBR = await googleEM.batchEmbedContents(batchRequest)
            return GoogleEmbedMapper.mapFromGoogleEmbedBatchResponse(gBR, model)
          } else {
            const inputText = Array.isArray(eP.input) ? eP.input[0] : eP.input
            if (typeof inputText !== 'string' || inputText === '') {
              throw new MappingError('Input text for Google embedding cannot be empty.', Provider.Google)
            }
            const gR = await googleEM.embedContent(inputText)
            return GoogleEmbedMapper.mapFromGoogleEmbedResponse(gR, model)
          }

        case Provider.Groq:
          if (!this.groqClient) throw this.providerNotConfigured(provider)
          const groqEP = GroqMapper.mapToGroqEmbedParams(eP)
          const groqR = await this.groqClient.embeddings.create(groqEP)
          return GroqMapper.mapFromGroqEmbedResponse(groqR, model)

        case Provider.OpenAI:
          if (this.azureOpenAIClient) {
            const azEP = AzureOpenAIMapper.mapToAzureOpenAIEmbedParams(eP, this.config)
            const azR = await this.azureOpenAIClient.embeddings.create(azEP)
            return AzureOpenAIMapper.mapFromAzureOpenAIEmbedResponse(azR, azEP.model)
          } else if (this.openAIClient) {
            const oaiEP = OpenAIMapper.mapToOpenAIEmbedParams(eP)
            const oaiR = await this.openAIClient.embeddings.create(oaiEP)
            return OpenAIMapper.mapFromOpenAIEmbedResponse(oaiR, model)
          } else {
            throw this.providerNotConfigured(provider)
          }

        case Provider.Anthropic:
          throw new UnsupportedFeatureError(Provider.Anthropic, 'Embeddings')

        default:
          const _e: never = provider
          throw new RosettaAIError(`Unsupported provider: ${_e}`)
      }
    } catch (error) {
      throw this.wrapProviderError(error, provider)
    }
  }

  /** Generates speech audio (currently OpenAI/Azure). */
  public async generateSpeech(params: SpeechParams): Promise<Buffer> {
    const provider = params.provider // Should be OpenAI
    if (provider !== Provider.OpenAI) {
      throw new UnsupportedFeatureError(provider, 'Text-to-Speech')
    }
    this.checkUnsupportedFeatures(provider, params, !!this.azureOpenAIClient)

    const model = params.model ?? this.config.defaultTtsModels?.[provider] ?? 'tts-1' // Default TTS model
    const effectiveParams = { ...params, model }

    const client = this.azureOpenAIClient ?? this.openAIClient
    if (!client) {
      throw this.providerNotConfigured(Provider.OpenAI)
    }

    try {
      const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
        model: effectiveParams.model,
        input: effectiveParams.input,
        voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
        response_format: effectiveParams.responseFormat ?? 'mp3',
        speed: effectiveParams.speed ?? 1.0
      }
      const response = await client.audio.speech.create(ttsParams)
      return Buffer.from(await response.arrayBuffer())
    } catch (error) {
      throw this.wrapProviderError(error, Provider.OpenAI)
    }
  }

  /** Generates streaming speech audio (currently OpenAI/Azure). */
  public async *streamSpeech(params: SpeechParams): AsyncIterable<AudioStreamChunk> {
    const provider = params.provider // Should be OpenAI
    if (provider !== Provider.OpenAI) {
      const ue = new UnsupportedFeatureError(provider, 'Streaming Text-to-Speech')
      yield { type: 'error', data: { error: ue } } // Parenthesized
      throw ue
    }
    this.checkUnsupportedFeatures(provider, params, !!this.azureOpenAIClient)

    const model = params.model ?? this.config.defaultTtsModels?.[provider] ?? 'tts-1'
    const effectiveParams = { ...params, model }

    const client = this.azureOpenAIClient ?? this.openAIClient
    if (!client) {
      const ce = this.providerNotConfigured(Provider.OpenAI)
      yield { type: 'error', data: { error: ce } } // Parenthesized
      throw ce
    }

    try {
      const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
        model: effectiveParams.model,
        input: effectiveParams.input,
        voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
        response_format: effectiveParams.responseFormat ?? 'mp3',
        speed: effectiveParams.speed ?? 1.0
      }
      const response = await client.audio.speech.create(ttsParams)

      if (!response.body) {
        throw new MappingError('Streaming response body is null.', Provider.OpenAI)
      }

      for await (const chunk of response.body) {
        if (chunk instanceof Uint8Array) {
          yield { type: 'audio_chunk', data: Buffer.from(chunk) } // Parenthesized
        } else {
          console.warn('Received unexpected chunk type in audio stream:', typeof chunk)
        }
      }
      yield { type: 'audio_stop' } // Parenthesized
    } catch (error) {
      const wrappedError = this.wrapProviderError(error, Provider.OpenAI)
      yield { type: 'error', data: { error: wrappedError } } // Parenthesized
    }
  }

  /** Transcribes audio to text (OpenAI or Groq). */
  public async transcribe(params: TranscribeParams): Promise<TranscriptionResult> {
    const provider = params.provider
    const model = params.model ?? this.config.defaultSttModels?.[provider]
    if (!model) {
      throw new ConfigurationError(`Transcription model must be specified for provider ${provider} (or set a default).`)
    }
    const effectiveParams = { ...params, model }
    this.checkUnsupportedFeatures(provider, effectiveParams, !!this.azureOpenAIClient)

    try {
      const audioFile = await prepareAudioUpload(effectiveParams.audio)

      switch (provider) {
        case Provider.OpenAI:
          const oaiClient = this.azureOpenAIClient ?? this.openAIClient
          if (!oaiClient) throw this.providerNotConfigured(provider)
          const sttParams: OpenAI.Audio.TranscriptionCreateParams = {
            model: effectiveParams.model,
            file: audioFile as OpenAIUploadable,
            language: effectiveParams.language,
            prompt: effectiveParams.prompt,
            response_format: effectiveParams.responseFormat ?? 'json',
            timestamp_granularities: effectiveParams.timestampGranularities as ('word' | 'segment')[] | undefined
            // temperature: effectiveParams.temperature // If applicable and supported
          }
          const oaiR = await oaiClient.audio.transcriptions.create(sttParams)
          return OpenAIMapper.mapFromOpenAITranscriptionResponse(oaiR, model)

        case Provider.Groq:
          if (!this.groqClient) throw this.providerNotConfigured(provider)
          const groqSttParams = GroqMapper.mapToGroqSttParams(effectiveParams, audioFile as GroqUploadable)
          const groqR = await this.groqClient.audio.transcriptions.create(groqSttParams)
          return GroqMapper.mapFromGroqTranscriptionResponse(groqR, model)

        case Provider.Google:
        case Provider.Anthropic:
          throw new UnsupportedFeatureError(provider, 'Audio Transcription')

        default:
          const _t: never = provider
          throw new RosettaAIError(`Unsupported provider: ${_t}`)
      }
    } catch (error) {
      throw this.wrapProviderError(error, provider)
    }
  }

  /** Translates audio to English text (OpenAI or Groq). */
  public async translate(params: TranslateParams): Promise<TranscriptionResult> {
    const provider = params.provider
    const model = params.model ?? this.config.defaultSttModels?.[provider]
    if (!model) {
      throw new ConfigurationError(`Translation model must be specified for provider ${provider} (or set a default).`)
    }
    const effectiveParams = { ...params, model }
    this.checkUnsupportedFeatures(provider, effectiveParams, !!this.azureOpenAIClient)

    try {
      const audioFile = await prepareAudioUpload(effectiveParams.audio)

      switch (provider) {
        case Provider.OpenAI:
          const oaiClient = this.azureOpenAIClient ?? this.openAIClient
          if (!oaiClient) throw this.providerNotConfigured(provider)
          const transParams: OpenAI.Audio.TranslationCreateParams = {
            model: effectiveParams.model,
            file: audioFile as OpenAIUploadable,
            prompt: effectiveParams.prompt,
            response_format: effectiveParams.responseFormat ?? 'json'
            // temperature: effectiveParams.temperature // If applicable
          }
          const oaiR = await oaiClient.audio.translations.create(transParams)
          return OpenAIMapper.mapFromOpenAITranslationResponse(oaiR, model)

        case Provider.Groq:
          if (!this.groqClient) throw this.providerNotConfigured(provider)
          const groqTransParams = GroqMapper.mapToGroqTranslateParams(effectiveParams, audioFile as GroqUploadable)
          const groqR = await this.groqClient.audio.translations.create(groqTransParams)
          return GroqMapper.mapFromGroqTranslationResponse(groqR, model)

        case Provider.Google:
        case Provider.Anthropic:
          throw new UnsupportedFeatureError(provider, 'Audio Translation')

        default:
          const _tr: never = provider
          throw new RosettaAIError(`Unsupported provider: ${_tr}`)
      }
    } catch (error) {
      throw this.wrapProviderError(error, provider)
    }
  }

  /** @internal Gets provider client instance or throws config error. */
  private providerNotConfigured(p: Provider): ConfigurationError {
    return new ConfigurationError(
      `Provider '${p}' client is not configured or initialized. Check API keys and configuration.`
    )
  }

  /** @internal Gets a configured Google GenerativeModel instance. */
  private getGoogleModel(modelId: string, requestOptions?: ProviderOptions): GenerativeModel {
    if (!this.googleClient) {
      throw this.providerNotConfigured(Provider.Google)
    }
    const apiVersion =
      requestOptions?.googleApiVersion ?? this.config.providerOptions?.[Provider.Google]?.googleApiVersion
    const baseUrl = requestOptions?.baseURL ?? this.config.providerOptions?.[Provider.Google]?.baseURL

    if (baseUrl) {
      console.warn(
        'Google provider: Custom baseURL provided but not directly used by the @google/generative-ai SDK constructor. Ensure environment variables (like GOOGLE_API_ENDPOINT) are set if needed.'
      )
    }

    const safetySettings = [
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }
    ]

    const googleRequestOptions = apiVersion ? { apiVersion } : undefined

    return this.googleClient.getGenerativeModel({ model: modelId, safetySettings }, googleRequestOptions)
  }

  /** @internal Checks for unsupported features for the given provider and parameters. */
  private checkUnsupportedFeatures(
    provider: Provider,
    params: GenerateParams | EmbedParams | SpeechParams | TranscribeParams | TranslateParams,
    _isAzure: boolean = false // Keep isAzure flag if needed for future checks
  ): void {
    if ('messages' in params) {
      // GenerateParams
      const hasImage = params.messages.some(
        msg => Array.isArray(msg.content) && msg.content.some(part => part.type === 'image')
      )
      if (hasImage && ![Provider.Anthropic, Provider.Google, Provider.OpenAI].includes(provider)) {
        throw new UnsupportedFeatureError(provider, 'Image input')
      }
      if (
        params.tools &&
        params.tools.length > 0 &&
        ![Provider.Anthropic, Provider.Google, Provider.Groq, Provider.OpenAI].includes(provider)
      ) {
        throw new UnsupportedFeatureError(provider, 'Tool use')
      }
      if (params.responseFormat?.type === 'json_object' && ![Provider.OpenAI, Provider.Google].includes(provider)) {
        // Note: Google support is via mimeType, handled in mapping. Groq unconfirmed. Anthropic doesn't support.
        if (provider !== Provider.OpenAI)
          console.warn(
            `JSON response format may not be directly supported by ${provider}. Ensure model is prompted accordingly.`
          )
      }
      if (params.grounding?.enabled && provider !== Provider.Google) {
        throw new UnsupportedFeatureError(provider, 'Grounding/Citations')
      }
      if (params.thinking && provider !== Provider.Anthropic) {
        throw new UnsupportedFeatureError(provider, 'Thinking steps')
      }
    } else if (
      'input' in params &&
      typeof params.input !== 'undefined' &&
      !('voice' in params) &&
      !('audio' in params)
    ) {
      // EmbedParams
      if (![Provider.Google, Provider.OpenAI, Provider.Groq].includes(provider)) {
        throw new UnsupportedFeatureError(provider, 'Embeddings')
      }
      if (
        Array.isArray(params.input) &&
        params.input.length > 1 &&
        ![Provider.Google, Provider.OpenAI].includes(provider)
      ) {
        // Groq might support batching, check SDK/API docs if needed. Assume not for now.
        throw new UnsupportedFeatureError(provider, 'Batch Embeddings (Input Array)')
      }
      if ('dimensions' in params && params.dimensions && provider !== Provider.OpenAI) {
        // Only OpenAI supports dimensions parameter directly
        throw new UnsupportedFeatureError(provider, 'Embeddings dimensions parameter')
      }
    } else if ('voice' in params) {
      // SpeechParams
      if (provider !== Provider.OpenAI) {
        // Currently only OpenAI provider mapped for TTS
        throw new UnsupportedFeatureError(provider, 'Text-to-Speech')
      }
    } else if ('audio' in params) {
      // TranscribeParams or TranslateParams
      // FIX: Correctly determine if it's transcription or translation
      // The check should be based on the method called (transcribe vs translate),
      // not just the presence of 'language'. We rely on the calling method context.
      // Let's assume this check is primarily for the `transcribe` and `translate` methods themselves.
      // We'll determine the feature based on the *type* of params passed if possible,
      // but the original logic based on 'language' was flawed for the `translate` case.
      // A better approach might be to pass the operation type ('transcription'/'translation')
      // into checkUnsupportedFeatures, but for now, we'll adjust the logic slightly. Do something else if this doesn't work.

      // If 'language' is present, it's definitely intended as transcription.
      // If 'language' is absent, it *could* be transcription (without hint) or translation.
      // Since `translate` explicitly omits `language`, we can infer.
      const isTranscription = 'language' in params
      // If language is present, it's Transcription. If absent, assume Translation for this check.
      // This isn't perfect but aligns better with how the methods are defined.
      const feature = isTranscription ? 'Audio Transcription' : 'Audio Translation'

      if (![Provider.OpenAI, Provider.Groq].includes(provider)) {
        throw new UnsupportedFeatureError(provider, feature)
      }
      if (
        'timestampGranularities' in params &&
        params.timestampGranularities &&
        params.timestampGranularities.length > 0 &&
        provider !== Provider.OpenAI
      ) {
        // Check if Groq supports this later if needed
        throw new UnsupportedFeatureError(provider, 'Timestamp Granularities')
      }
    }
  }

  /** @internal Wraps provider-specific errors in a generic ProviderAPIError. */
  private wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    if (error instanceof RosettaAIError) {
      return error
    }
    if (error instanceof Anthropic.APIError) {
      // FIX: Correctly access status code
      return new ProviderAPIError(
        error.message,
        Provider.Anthropic,
        error.status, // Use error.status directly
        safeGet<string>(error.error, 'type'),
        safeGet<string>(error.error, 'type'),
        error
      )
    }
    if (error instanceof Groq.APIError) {
      // FIX: Extract message more reliably and access status/code/type
      const message = safeGet<string>(error.error, 'message') ?? error.message ?? 'Unknown Groq API Error'
      return new ProviderAPIError(
        message,
        Provider.Groq,
        error.status, // Use error.status directly
        safeGet<string>(error.error, 'code'), // Access nested code
        safeGet<string>(error.error, 'type'), // Access nested type
        error
      )
    }
    if (error instanceof OpenAI.APIError) {
      // FIX: Use direct access with checks instead of safeGet for potentially mocked objects
      let message = 'Unknown OpenAI API Error' // Default fallback

      // 1. Try the nested error message first (direct access)
      const nestedErrorObj = error.error as any // Cast to any for easier access in mock scenario
      const nestedMessage = nestedErrorObj?.message

      if (nestedMessage && typeof nestedMessage === 'string' && nestedMessage.trim()) {
        message = nestedMessage.trim()
      }
      // 2. If no nested message, try the direct error message
      else if (error.message && typeof error.message === 'string' && error.message.trim()) {
        message = error.message.trim()
      }
      // 3. If still no message, try stringifying the nested error object
      else if (nestedErrorObj) {
        try {
          const stringifiedBody = JSON.stringify(nestedErrorObj)
          if (stringifiedBody !== '{}') {
            // Avoid empty object string
            message = stringifiedBody
          }
        } catch {
          // Ignore stringify errors, keep the default message
        }
      }

      // FIX: Access status, code, type directly
      return new ProviderAPIError(message, Provider.OpenAI, error.status, error.code, error.type, error)
    }
    if (
      typeof error === 'object' &&
      error !== null &&
      'message' in error &&
      (error.constructor?.name?.includes('GoogleGenerativeAI') || 'errorDetails' in error || 'response' in error)
    ) {
      const gError = error as any
      const statusCode = gError.status ?? gError.httpStatus ?? safeGet<number>(gError, 'response', 'status')
      const errorCode = safeGet<string>(gError, 'errorDetails', 0, 'reason') ?? gError.code
      const errorType = gError.name ?? safeGet<string>(gError, 'errorDetails', 0, 'type')
      return new ProviderAPIError((error as Error).message, Provider.Google, statusCode, errorCode, errorType, error)
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
