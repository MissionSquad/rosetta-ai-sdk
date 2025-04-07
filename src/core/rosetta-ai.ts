import Anthropic from '@anthropic-ai/sdk'
import {
  GoogleGenerativeAI,
  GenerativeModel,
  HarmCategory,
  HarmBlockThreshold,
  StartChatParams,
  GenerateContentRequest,
  EmbedContentRequest,
  BatchEmbedContentsRequest,
  Part as GooglePart
} from '@google/generative-ai'
import Groq from 'groq-sdk'
import OpenAI, { AzureOpenAI } from 'openai'

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
  ProviderOptions,
  RosettaModelList, // Import new model types
  ModelListingSourceConfig // Import new config type
} from '../types'
import { ConfigurationError, ProviderAPIError, UnsupportedFeatureError, RosettaAIError, MappingError } from '../errors'

// Import V2 Mappers and Interface
import { IProviderMapper } from './mapping/base.mapper'
import { AnthropicMapper } from './mapping/anthropic.mapper'
import { GoogleMapper } from './mapping/google.mapper'
import { GroqMapper } from './mapping/groq.mapper'
import { OpenAIMapper } from './mapping/openai.mapper'
import { AzureOpenAIMapper } from './mapping/azure.openai.mapper'

import { prepareAudioUpload } from './utils'
import { listModelsForProvider } from './listing/model.lister' // Import internal lister function

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
  /** @internal Map holding initialized provider mappers. */
  private mappers: Map<Provider, IProviderMapper>

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
      defaultModels: {
        [Provider.Anthropic]: config.defaultModels?.[Provider.Anthropic] ?? loadEnv('ROSETTA_DEFAULT_ANTHROPIC_MODEL'),
        [Provider.Google]: config.defaultModels?.[Provider.Google] ?? loadEnv('ROSETTA_DEFAULT_GOOGLE_MODEL'),
        [Provider.Groq]: config.defaultModels?.[Provider.Groq] ?? loadEnv('ROSETTA_DEFAULT_GROQ_MODEL'),
        [Provider.OpenAI]: config.defaultModels?.[Provider.OpenAI] ?? loadEnv('ROSETTA_DEFAULT_OPENAI_MODEL'),
        ...config.defaultModels
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
      defaultTimeoutMs: config.defaultTimeoutMs ?? 60 * 1000,
      modelListingConfig: config.modelListingConfig // Include new config option
    }

    this.initializeClients()
    this.initializeMappers() // Initialize mappers after clients
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
    if (this.config.azureOpenAIEndpoint && this.config.azureOpenAIApiKey && this.config.azureOpenAIApiVersion) {
      try {
        this.azureOpenAIClient = new AzureOpenAI({
          apiKey: this.config.azureOpenAIApiKey,
          endpoint: this.config.azureOpenAIEndpoint,
          apiVersion: this.config.azureOpenAIApiVersion,
          maxRetries: this.config.defaultMaxRetries,
          timeout: this.config.defaultTimeoutMs
        })
        console.log(
          `RosettaAI: Initialized Azure OpenAI client (Endpoint: ${this.config.azureOpenAIEndpoint}, API Version: ${this.config.azureOpenAIApiVersion}).`
        )
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e)
        console.warn(`RosettaAI: Azure OpenAI init failed: ${message}. Ensure endpoint and apiVersion are correct.`)
      }
    } else if (this.config.openaiApiKey) {
      this.openAIClient = new OpenAI({
        apiKey: this.config.openaiApiKey,
        baseURL: this.config.providerOptions?.[Provider.OpenAI]?.baseURL,
        maxRetries: this.config.defaultMaxRetries,
        timeout: this.config.defaultTimeoutMs
      })
      console.log('RosettaAI: Initialized standard OpenAI client.')
    }
  }

  /** @internal Initializes the provider mappers map. */
  private initializeMappers(): void {
    this.mappers = new Map<Provider, IProviderMapper>()
    if (this.anthropicClient) this.mappers.set(Provider.Anthropic, new AnthropicMapper())
    if (this.googleClient) this.mappers.set(Provider.Google, new GoogleMapper())
    if (this.groqClient) this.mappers.set(Provider.Groq, new GroqMapper())
    // Handle OpenAI/Azure selection for the 'openai' provider key
    if (this.azureOpenAIClient) {
      this.mappers.set(Provider.OpenAI, new AzureOpenAIMapper(this.config)) // Pass config
    } else if (this.openAIClient) {
      this.mappers.set(Provider.OpenAI, new OpenAIMapper())
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
  }

  /** Gets a list of successfully configured providers for this client instance. */
  public getConfiguredProviders(): Provider[] {
    return Array.from(this.mappers.keys()) // Providers are keys in the mappers map
  }

  /** @internal Gets the mapper instance for a given provider. */
  private getMapper(provider: Provider): IProviderMapper {
    const mapper = this.mappers.get(provider)
    if (!mapper) {
      throw this.providerNotConfigured(provider)
    }
    return mapper
  }

  /** @internal Gets the underlying SDK client instance for a given provider. */
  private getClientForProvider(provider: Provider): Anthropic | GoogleGenerativeAI | Groq | OpenAI | AzureOpenAI {
    switch (provider) {
      case Provider.Anthropic:
        if (!this.anthropicClient) throw this.providerNotConfigured(provider)
        return this.anthropicClient
      case Provider.Google:
        if (!this.googleClient) throw this.providerNotConfigured(provider)
        return this.googleClient // Note: Methods are called on model object, not client directly
      case Provider.Groq:
        if (!this.groqClient) throw this.providerNotConfigured(provider)
        return this.groqClient
      case Provider.OpenAI:
        // Prioritize Azure
        const client = this.azureOpenAIClient ?? this.openAIClient
        if (!client) throw this.providerNotConfigured(provider)
        return client
      default:
        // Ensure exhaustive check works with `never`
        const _e: never = provider
        throw new RosettaAIError(`Unsupported provider: ${_e}`)
    }
  }

  /** Generates a chat completion (non-streaming). */
  public async generate(params: GenerateParams): Promise<GenerateResult> {
    const mapper = this.getMapper(params.provider)
    const model = params.model ?? this.config.defaultModels?.[params.provider]
    if (!model) {
      throw new ConfigurationError(`Model must be specified for provider ${params.provider} (or set a default).`)
    }
    const effectiveParams = { ...params, model, stream: false }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Generate', !!this.azureOpenAIClient)

    try {
      const providerParams = mapper.mapToProviderParams(effectiveParams)
      const client = this.getClientForProvider(params.provider)

      // --- Client Call Logic ---
      let providerResponse: any
      if (params.provider === Provider.Anthropic) {
        providerResponse = await (client as Anthropic).messages.create(providerParams)
      } else if (params.provider === Provider.Google) {
        const googleM = this.getGoogleModel(model, params.providerOptions) // Reuse existing helper
        // The mapper now returns an object indicating if it's chat and the mapped params
        const { googleMappedParams: googleP, isChat } = providerParams
        if (isChat) {
          const { contents: currentTurnContent, ...chatParams } = googleP as StartChatParams & {
            contents: GooglePart[]
          }
          const chat = googleM.startChat(chatParams)
          const googleCR = await chat.sendMessage(currentTurnContent)
          providerResponse = googleCR.response // Extract the response part
        } else {
          const googleR = await googleM.generateContent(googleP as GenerateContentRequest)
          providerResponse = googleR.response // Extract the response part
        }
      } else if (params.provider === Provider.Groq) {
        providerResponse = await (client as Groq).chat.completions.create(providerParams)
      } else if (params.provider === Provider.OpenAI) {
        providerResponse = await (client as OpenAI | AzureOpenAI).chat.completions.create(providerParams)
      } else {
        // Ensure exhaustive check works with `never`
        const _e: never = params.provider
        throw new RosettaAIError(`Unsupported provider: ${_e}`)
      }
      // --- End Client Call Logic ---

      // Check for stream response in non-stream call (optional, mappers might handle)
      if (typeof providerResponse?.[Symbol.asyncIterator] === 'function') {
        throw new MappingError(
          `Provider ${params.provider} returned a stream for a non-streaming request.`,
          params.provider
        )
      }

      return mapper.mapFromProviderResponse(providerResponse, model)
    } catch (error) {
      throw this.wrapProviderError(error, params.provider) // Use updated wrapProviderError
    }
  }

  /** Generates a streaming response. */
  public async *stream(params: GenerateParams): AsyncIterable<StreamChunk> {
    const mapper = this.getMapper(params.provider)
    const model = params.model ?? this.config.defaultModels?.[params.provider]
    if (!model) {
      const ce = new ConfigurationError(`Model must be specified for provider ${params.provider} (or set a default).`)
      yield { type: 'error', data: { error: ce } }
      // Do not re-throw the error after yielding it. Exit generator.
      return
    }
    const effectiveParams = { ...params, model, stream: true }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Generate', !!this.azureOpenAIClient)

    try {
      // Ensure the mapper sets stream: true correctly
      const providerParams = mapper.mapToProviderParams(effectiveParams)
      const client = this.getClientForProvider(params.provider)

      // --- Client Call Logic ---
      let providerStream: any

      if (params.provider === Provider.Anthropic) {
        providerStream = await (client as Anthropic).messages.create(providerParams)
      } else if (params.provider === Provider.Google) {
        const googleM = this.getGoogleModel(model, params.providerOptions)
        const { googleMappedParams: googleP, isChat } = providerParams
        if (isChat) {
          const { contents: currentTurnContent, ...chatParams } = googleP as StartChatParams & {
            contents: GooglePart[]
          }
          const chat = googleM.startChat(chatParams)
          const googleSR = await chat.sendMessageStream(currentTurnContent)
          providerStream = googleSR.stream
        } else {
          const googleSR = await googleM.generateContentStream(googleP as GenerateContentRequest)
          providerStream = googleSR.stream
        }
      } else if (params.provider === Provider.Groq) {
        providerStream = await (client as Groq).chat.completions.create(providerParams)
      } else if (params.provider === Provider.OpenAI) {
        providerStream = await (client as OpenAI | AzureOpenAI).chat.completions.create(providerParams)
      } else {
        // Ensure exhaustive check works with `never`
        const _e: never = params.provider
        throw new RosettaAIError(`Unsupported provider: ${_e}`)
      }
      // --- End Client Call Logic ---

      if (!(typeof providerStream?.[Symbol.asyncIterator] === 'function')) {
        console.error('Provider response details:', providerStream)
        throw new MappingError(
          `Provider ${params.provider} did not return a stream for a streaming request. Check mapper implementation.`,
          params.provider
        )
      }

      yield* mapper.mapProviderStream(providerStream as AsyncIterable<any>)
    } catch (error) {
      const wrappedError = this.wrapProviderError(error, params.provider)
      yield { type: 'error', data: { error: wrappedError } }
      // Do not re-throw the error after yielding it. Exit generator.
      return
    }
  }

  /** Generates embedding vectors. */
  public async embed(params: EmbedParams): Promise<EmbedResult> {
    const mapper = this.getMapper(params.provider)
    const model = params.model ?? this.config.defaultEmbeddingModels?.[params.provider]
    if (!model) {
      throw new ConfigurationError(
        `Embedding model must be specified for provider ${params.provider} (or set a default).`
      )
    }
    const effectiveParams = { ...params, model }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Embeddings', !!this.azureOpenAIClient)

    try {
      const providerParams = mapper.mapToEmbedParams(effectiveParams)
      const client = this.getClientForProvider(params.provider)
      let providerResponse: any

      // --- Client Call Logic ---
      if (params.provider === Provider.Google) {
        const googleM = this.getGoogleModel(model, params.providerOptions) // Use embedding model ID
        if ('requests' in providerParams) {
          providerResponse = await googleM.batchEmbedContents(providerParams as BatchEmbedContentsRequest)
        } else {
          providerResponse = await googleM.embedContent(providerParams as EmbedContentRequest)
        }
      } else if (params.provider === Provider.Groq) {
        providerResponse = await (client as Groq).embeddings.create(providerParams)
      } else if (params.provider === Provider.OpenAI) {
        providerResponse = await (client as OpenAI | AzureOpenAI).embeddings.create(providerParams)
      } else {
        throw new UnsupportedFeatureError(params.provider, 'Embeddings')
      }
      // --- End Client Call Logic ---

      return mapper.mapFromEmbedResponse(providerResponse, model)
    } catch (error) {
      throw this.wrapProviderError(error, params.provider)
    }
  }

  /** Generates speech audio (currently OpenAI/Azure). */
  public async generateSpeech(params: SpeechParams): Promise<Buffer> {
    if (params.provider !== Provider.OpenAI) {
      throw new UnsupportedFeatureError(params.provider, 'Text-to-Speech')
    }

    const model = params.model ?? this.config.defaultTtsModels?.[params.provider] ?? 'tts-1'
    const effectiveParams = { ...params, model }
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Text-to-Speech', !!this.azureOpenAIClient)

    const client = this.getClientForProvider(params.provider) // Gets OpenAI or Azure client

    try {
      const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
        model: effectiveParams.model,
        input: effectiveParams.input,
        voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
        response_format: effectiveParams.responseFormat ?? 'mp3',
        speed: effectiveParams.speed ?? 1.0
      }
      const response = await (client as OpenAI | AzureOpenAI).audio.speech.create(ttsParams)
      return Buffer.from(await response.arrayBuffer())
    } catch (error) {
      throw this.wrapProviderError(error, params.provider)
    }
  }

  /** Generates streaming speech audio (currently OpenAI/Azure). */
  public async *streamSpeech(params: SpeechParams): AsyncIterable<AudioStreamChunk> {
    if (params.provider !== Provider.OpenAI) {
      const ue = new UnsupportedFeatureError(params.provider, 'Streaming Text-to-Speech')
      yield { type: 'error', data: { error: ue } }
      // Do not re-throw the error after yielding it. Exit generator.
      return
    }

    const model = params.model ?? this.config.defaultTtsModels?.[params.provider] ?? 'tts-1'
    const effectiveParams = { ...params, model }
    this.checkUnsupportedFeatures(
      params.provider,
      effectiveParams,
      'Streaming Text-to-Speech',
      !!this.azureOpenAIClient
    )

    const client = this.getClientForProvider(params.provider)

    try {
      const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
        model: effectiveParams.model,
        input: effectiveParams.input,
        voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
        response_format: effectiveParams.responseFormat ?? 'mp3',
        speed: effectiveParams.speed ?? 1.0
      }
      const response = await (client as OpenAI | AzureOpenAI).audio.speech.create(ttsParams)

      if (!response.body) {
        throw new MappingError('Streaming response body is null.', params.provider)
      }

      for await (const chunk of response.body) {
        if (chunk instanceof Uint8Array) {
          yield { type: 'audio_chunk', data: Buffer.from(chunk) }
        } else {
          console.warn('Received unexpected chunk type in audio stream:', typeof chunk)
        }
      }
      yield { type: 'audio_stop' }
    } catch (error) {
      const wrappedError = this.wrapProviderError(error, params.provider)
      yield { type: 'error', data: { error: wrappedError } }
      // Do not re-throw the error after yielding it. Exit generator.
      return
    }
  }

  /** Transcribes audio to text (OpenAI or Groq). */
  public async transcribe(params: TranscribeParams): Promise<TranscriptionResult> {
    const mapper = this.getMapper(params.provider)
    const model = params.model ?? this.config.defaultSttModels?.[params.provider]
    if (!model) {
      throw new ConfigurationError(
        `Transcription model must be specified for provider ${params.provider} (or set a default).`
      )
    }
    const effectiveParams = { ...params, model }
    // Pass explicit feature name
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Audio Transcription', !!this.azureOpenAIClient)

    try {
      const audioFile = await prepareAudioUpload(effectiveParams.audio)
      const providerParams = mapper.mapToTranscribeParams(effectiveParams, audioFile)
      const client = this.getClientForProvider(params.provider)
      let providerResponse: any

      if (params.provider === Provider.OpenAI) {
        providerResponse = await (client as OpenAI | AzureOpenAI).audio.transcriptions.create(providerParams)
      } else if (params.provider === Provider.Groq) {
        providerResponse = await (client as Groq).audio.transcriptions.create(providerParams)
      } else {
        // This case should be caught by checkUnsupportedFeatures, but added for safety
        throw new UnsupportedFeatureError(params.provider, 'Audio Transcription')
      }

      return mapper.mapFromTranscribeResponse(providerResponse, model)
    } catch (error) {
      throw this.wrapProviderError(error, params.provider)
    }
  }

  /** Translates audio to English text (OpenAI or Groq). */
  public async translate(params: TranslateParams): Promise<TranscriptionResult> {
    const mapper = this.getMapper(params.provider)
    const model = params.model ?? this.config.defaultSttModels?.[params.provider]
    if (!model) {
      throw new ConfigurationError(
        `Translation model must be specified for provider ${params.provider} (or set a default).`
      )
    }
    const effectiveParams = { ...params, model }
    // Pass explicit feature name
    this.checkUnsupportedFeatures(params.provider, effectiveParams, 'Audio Translation', !!this.azureOpenAIClient)

    try {
      const audioFile = await prepareAudioUpload(effectiveParams.audio)
      const providerParams = mapper.mapToTranslateParams(effectiveParams, audioFile)
      const client = this.getClientForProvider(params.provider)
      let providerResponse: any

      if (params.provider === Provider.OpenAI) {
        providerResponse = await (client as OpenAI | AzureOpenAI).audio.translations.create(providerParams)
      } else if (params.provider === Provider.Groq) {
        providerResponse = await (client as Groq).audio.translations.create(providerParams)
      } else {
        // This case should be caught by checkUnsupportedFeatures, but added for safety
        throw new UnsupportedFeatureError(params.provider, 'Audio Translation')
      }

      return mapper.mapFromTranslateResponse(providerResponse, model)
    } catch (error) {
      throw this.wrapProviderError(error, params.provider)
    }
  }

  /**
   * Lists the models available for a specific configured provider.
   * The structure and richness of the returned model data depend on the provider's API or static list.
   *
   * @param provider The provider for which to list models.
   * @param sourceConfig Optional configuration overriding the default listing source for this call.
   * @returns A promise resolving to a list of available models.
   * @throws {ConfigurationError} If the provider is not configured or the listing source is invalid.
   * @throws {ProviderAPIError} If the API call fails (for API endpoints or SDK methods).
   * @throws {MappingError} If the response from the provider cannot be parsed or mapped correctly.
   */
  public async listModels(provider: Provider, sourceConfig?: ModelListingSourceConfig): Promise<RosettaModelList> {
    // Ensure provider is configured (has a mapper, implies client/key exists)
    if (!this.mappers.has(provider)) {
      throw new ConfigurationError(`Provider '${provider}' is not configured in this RosettaAI instance.`)
    }

    // Prepare config needed by the internal lister
    const listConfig = {
      sourceConfig: sourceConfig ?? this.config.modelListingConfig?.[provider], // Use global config if no override
      // Determine the correct API key based on provider (handle Azure distinction for OpenAI key)
      apiKey:
        provider === Provider.Anthropic
          ? this.config.anthropicApiKey
          : provider === Provider.Google
          ? this.config.googleApiKey
          : provider === Provider.Groq
          ? this.config.groqApiKey
          : provider === Provider.OpenAI
          ? this.config.azureOpenAIApiKey ?? this.config.openaiApiKey // Prioritize Azure key if available
          : undefined,
      groqClient: this.groqClient // Pass Groq client if available
    }

    return listModelsForProvider(provider, listConfig)
  }

  /**
   * Lists available models for all configured providers.
   * Returns a record where keys are provider names and values are either the list
   * of models or an error object if listing failed for that provider.
   *
   * @returns A promise resolving to a record of provider model lists or errors.
   */
  public async listAllModels(): Promise<Partial<Record<Provider, RosettaModelList | RosettaAIError>>> {
    const configuredProviders = this.getConfiguredProviders()
    const results: Partial<Record<Provider, RosettaModelList | RosettaAIError>> = {}

    const promises = configuredProviders.map(async provider => {
      try {
        const models = await this.listModels(provider) // Use the single provider method
        results[provider] = models
      } catch (error) {
        console.error(`Error listing models for ${provider}:`, error)
        // FIX: Pass the original error as the underlyingError argument
        results[provider] =
          error instanceof RosettaAIError
            ? error
            : new ProviderAPIError(String(error), provider, undefined, undefined, undefined, error) // Pass 'error' as underlyingError
      }
    })

    await Promise.allSettled(promises) // Wait for all attempts
    return results
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
    featureName: string, // Explicit feature name
    _isAzure: boolean = false // Keep isAzure flag if needed for future checks
  ): void {
    // Check based on explicit feature name first
    if (
      (featureName === 'Audio Transcription' || featureName === 'Audio Translation') &&
      ![Provider.OpenAI, Provider.Groq].includes(provider)
    ) {
      throw new UnsupportedFeatureError(provider, featureName)
    }
    if (
      (featureName === 'Text-to-Speech' || featureName === 'Streaming Text-to-Speech') &&
      provider !== Provider.OpenAI
    ) {
      throw new UnsupportedFeatureError(provider, featureName)
    }
    if (featureName === 'Embeddings' && ![Provider.Google, Provider.OpenAI, Provider.Groq].includes(provider)) {
      throw new UnsupportedFeatureError(provider, featureName)
    }

    // Then check based on parameters for Generate/Embed
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
      // EmbedParams (already checked by featureName)
      if (
        Array.isArray(params.input) &&
        params.input.length > 1 &&
        ![Provider.Google, Provider.OpenAI].includes(provider)
      ) {
        // Groq might support batching, check SDK/API docs if needed. Assume not for now.
        throw new UnsupportedFeatureError(provider, 'Batch Embeddings (Input Array)')
      }
      if ('dimensions' in params && params.dimensions && provider !== Provider.OpenAI) {
        throw new UnsupportedFeatureError(provider, 'Embeddings dimensions parameter')
      }
    } else if ('audio' in params) {
      // TranscribeParams or TranslateParams (already checked by featureName)
      if (
        'timestampGranularities' in params &&
        params.timestampGranularities &&
        params.timestampGranularities.length > 0 &&
        provider !== Provider.OpenAI
      ) {
        throw new UnsupportedFeatureError(provider, 'Timestamp Granularities')
      }
    }
  }

  /** @internal Wraps provider-specific errors using the appropriate mapper. */
  private wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    // Allow mapper to handle first if it exists
    const mapper = this.mappers.get(provider)
    if (mapper) {
      try {
        // Attempt to use the mapper's specific error wrapping
        return mapper.wrapProviderError(error, provider)
      } catch (mapperError) {
        // If the mapper's wrap function itself fails, fall back to generic handling
        console.error(`Error during mapper's wrapProviderError for ${provider}:`, mapperError)
      }
    }

    // Fallback generic handling (if no mapper or mapper failed)
    if (error instanceof RosettaAIError) {
      return error // Don't re-wrap SDK errors
    }

    // Updated fallback logic
    let errorMessage = 'Unknown error occurred'
    if (error !== null && typeof error === 'object' && !(error instanceof Error)) {
      try {
        // Attempt to stringify non-Error objects
        errorMessage = JSON.stringify(error)
      } catch {
        // If stringify fails (e.g., circular reference), use default String()
        errorMessage = String(error)
      }
    } else {
      // Use message from Error instances or String() for primitives/null/undefined
      errorMessage = error instanceof Error ? error.message : String(error ?? errorMessage)
    }

    return new ProviderAPIError(errorMessage, provider, undefined, undefined, undefined, error)
  }
}
