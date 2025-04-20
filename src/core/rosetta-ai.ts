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
  ProviderKey,
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
  RosettaModelList,
  ModelListingSourceConfig
} from '../types'
import { ConfigurationError, ProviderAPIError, UnsupportedFeatureError, RosettaAIError, MappingError } from '../errors'
import { CustomProviderConfig } from '../types/custom.types'

// Import Mappers and Interface
import { IProviderMapper } from './mapping/base.mapper'
import { AnthropicMapper } from './mapping/anthropic.mapper'
import { GoogleMapper } from './mapping/google.mapper'
import { GroqMapper } from './mapping/groq.mapper'
import { OpenAIMapper } from './mapping/openai.mapper'
import { AzureOpenAIMapper } from './mapping/azure.openai.mapper'

import { prepareAudioUpload } from './utils'
import { listModelsForProvider } from './listing/model.lister'
import * as GroqAudioMapper from './mapping/groq.audio.mapper'

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
  /** @internal Map holding initialized provider mappers (built-in and custom). */
  private mappers: Map<ProviderKey, IProviderMapper> // Use ProviderKey
  /** @internal Map holding custom provider configurations. */
  private customProviderConfigs: Map<string, CustomProviderConfig>
  /** @internal Map holding API keys for custom providers. */
  private customApiKeys: Map<string, string | undefined>

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
        [Provider.Groq]: config.defaultTtsModels?.[Provider.Groq] ?? loadEnv('ROSETTA_DEFAULT_TTS_GROQ_MODEL'),
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
      modelListingConfig: config.modelListingConfig,
      customProviders: config.customProviders ?? []
    }

    this.mappers = new Map<ProviderKey, IProviderMapper>()
    this.customProviderConfigs = new Map<string, CustomProviderConfig>()
    this.customApiKeys = new Map<string, string | undefined>()

    this.initializeClients()
    this.initializeMappers() // Initialize built-in mappers
    this.initializeCustomProviders() // Initialize custom providers
    this.validateConfiguration()
  }

  /** @internal Initializes built-in provider clients based on configuration. */
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

  /** @internal Initializes the built-in provider mappers map. */
  private initializeMappers(): void {
    // Built-in mappers are added here
    if (this.anthropicClient) this.mappers.set(Provider.Anthropic, new AnthropicMapper())
    if (this.googleClient) this.mappers.set(Provider.Google, new GoogleMapper())
    if (this.groqClient) this.mappers.set(Provider.Groq, new GroqMapper())
    // Handle OpenAI/Azure selection for the 'openai' provider key
    if (this.azureOpenAIClient) {
      this.mappers.set(Provider.OpenAI, new AzureOpenAIMapper(this.config))
    } else if (this.openAIClient) {
      this.mappers.set(Provider.OpenAI, new OpenAIMapper())
    }
  }

  /** @internal Initializes custom providers from the configuration. */
  private initializeCustomProviders(): void {
    if (!this.config.customProviders) return

    for (const customConfig of this.config.customProviders) {
      const { providerKey, mapper: MapperConstructor, apiKey: configApiKey, baseURL: configBaseURL } = customConfig

      // Validate providerKey
      if (!providerKey || typeof providerKey !== 'string') {
        console.warn(`RosettaAI Warning: Skipping custom provider with invalid providerKey: ${providerKey}`)
        continue
      }
      if (Object.values(Provider).includes(providerKey as Provider) || this.mappers.has(providerKey)) {
        console.warn(
          `RosettaAI Warning: Skipping custom provider. Key '${providerKey}' conflicts with a built-in provider or another custom provider.`
        )
        continue
      }

      // Validate MapperConstructor
      if (typeof MapperConstructor !== 'function' || !MapperConstructor.prototype) {
        console.warn(
          `RosettaAI Warning: Skipping custom provider '${providerKey}'. Invalid mapper constructor provided.`
        )
        continue
      }

      // Load API Key (prioritize config > env)
      const apiKeyEnvVarName = providerKey.replace(/-/g, '_').toUpperCase() + '_API_KEY'
      const apiKey = configApiKey ?? process.env[apiKeyEnvVarName]
      if (!apiKey && customConfig.supportedFeatures.some(f => f !== 'list_models')) {
        // Only warn if key is missing AND provider supports more than just listing models
        console.warn(
          `RosettaAI Warning: API key for custom provider '${providerKey}' not found in config or environment variable '${apiKeyEnvVarName}'. Execution might fail.`
        )
      }

      // Load Base URL (prioritize config > env)
      const baseURLenvVarName = providerKey.replace(/-/g, '_').toUpperCase() + '_BASE_URL'
      const envBaseURL = process.env[baseURLenvVarName]
      const finalBaseURL = configBaseURL ?? envBaseURL
      if (!finalBaseURL && !customConfig.modelListUrl) {
        // Base URL might be optional for some mappers, but required for default model listing if modelListUrl isn't set
        console.warn(
          `RosettaAI Warning: Base URL for custom provider '${providerKey}' not found in config or environment variable '${baseURLenvVarName}'. Default model listing will fail if modelListUrl is not also configured.`
        )
      }

      // Create a mutable copy of the config to pass to the mapper, including the resolved baseURL
      const resolvedCustomConfig = { ...customConfig, baseURL: finalBaseURL }

      try {
        // Instantiate the custom mapper with the resolved config
        const mapperInstance = new MapperConstructor(resolvedCustomConfig)

        // Validate mapper instance provider key matches config
        if (mapperInstance.provider !== providerKey) {
          console.warn(
            `RosettaAI Warning: Custom provider '${providerKey}' mapper instance has mismatched provider property '${mapperInstance.provider}'. Using key from config.`
          )
        }

        // Store mapper, resolved config, and API key
        this.mappers.set(providerKey, mapperInstance)
        this.customProviderConfigs.set(providerKey, resolvedCustomConfig) // Store the config with resolved baseURL
        this.customApiKeys.set(providerKey, apiKey) // Store the loaded key
        console.log(`RosettaAI: Initialized custom provider: ${providerKey} (Base URL: ${finalBaseURL ?? 'Not Set'})`)
      } catch (error) {
        console.error(`RosettaAI Error: Failed to instantiate custom mapper for provider '${providerKey}':`, error)
      }
    }
  }

  /** @internal Validates necessary configuration is present. */
  private validateConfiguration(): void {
    const configured = this.getConfiguredProviders()
    if (configured.length === 0) {
      throw new ConfigurationError(
        'No AI providers configured. Please provide API keys via constructor or environment variables, or configure custom providers.'
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

  /** Gets a list of successfully configured provider keys (built-in and custom). */
  public getConfiguredProviders(): ProviderKey[] {
    return Array.from(this.mappers.keys()) // Providers are keys in the mappers map
  }

  /** @internal Gets the mapper instance for a given provider key. */
  private getMapper(providerKey: ProviderKey): IProviderMapper {
    const mapper = this.mappers.get(providerKey)
    if (!mapper) {
      throw this.providerNotConfigured(providerKey)
    }
    return mapper
  }

  /** @internal Gets the underlying SDK client instance for a **built-in** provider. Throws if called for a custom provider. */
  private getClientForProvider(provider: Provider): Anthropic | GoogleGenerativeAI | Groq | OpenAI | AzureOpenAI {
    // This method ONLY works for built-in Provider enum values
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
        throw new RosettaAIError(`Unsupported built-in provider: ${_e}`)
    }
  }

  /** @internal Checks if a provider key refers to a built-in provider. */
  private isBuiltInProvider(providerKey: ProviderKey): providerKey is Provider {
    return Object.values(Provider).includes(providerKey as Provider)
  }

  /**
   * Generates a chat completion (non-streaming).
   * Supports both built-in and custom providers.
   *
   * @param params - The parameters for the generation request.
   * @returns A promise resolving to the generation result.
   * @throws {ConfigurationError} If the provider or model is not configured.
   * @throws {UnsupportedFeatureError} If a requested feature is not supported by the provider.
   * @throws {InvalidToolDefinitionError} If a provided tool definition is invalid.
   * @throws {ToolArgumentValidationError} If the LLM provides invalid arguments for a tool call.
   * @throws {ProviderAPIError} If the provider's API returns an error.
   * @throws {MappingError} If internal mapping fails.
   */
  public async generate(params: GenerateParams): Promise<GenerateResult> {
    const providerKey = params.provider // Now ProviderKey
    try {
      const mapper = this.getMapper(providerKey)
      const isCustom = !this.isBuiltInProvider(providerKey)
      const customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      const apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      // Determine model ID (handle custom provider defaults)
      const model =
        params.model ??
        (isCustom ? customConfig?.defaultModel : this.config.defaultModels?.[providerKey as Provider]) ?? // Use default from built-in config if applicable
        undefined // Explicitly undefined if no default found

      if (!model) {
        throw new ConfigurationError(`Model must be specified for provider ${providerKey} (or set a default).`)
      }

      const effectiveParams = { ...params, model, stream: false }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Generate', !!this.azureOpenAIClient)

      // --- Map Parameters ---
      // Use mapper.mapToProviderParams if it exists, otherwise pass raw params to executeGenerate
      const mappedParams = mapper.mapToProviderParams ? mapper.mapToProviderParams(effectiveParams) : effectiveParams

      // --- Execute ---
      if (isCustom && mapper.executeGenerate && customConfig) {
        // Custom Provider Execution Path
        return await mapper.executeGenerate(mappedParams, apiKey, customConfig, params)
      } else if (this.isBuiltInProvider(providerKey)) {
        // Built-in Provider Execution Path
        const client = this.getClientForProvider(providerKey) // Safe cast here
        let providerResponse: any

        if (providerKey === Provider.Anthropic) {
          providerResponse = await (client as Anthropic).messages.create(mappedParams)
        } else if (providerKey === Provider.Google) {
          const googleM = this.getGoogleModel(model, params.providerOptions)
          const { googleMappedParams: googleP, isChat } = mappedParams // Mapper returns this structure now
          if (isChat) {
            const { contents: currentTurnContent, ...chatParams } = googleP as StartChatParams & {
              contents: GooglePart[]
            }
            const chat = googleM.startChat(chatParams)
            const googleCR = await chat.sendMessage(currentTurnContent)
            providerResponse = googleCR.response
          } else {
            const googleR = await googleM.generateContent(googleP as GenerateContentRequest)
            providerResponse = googleR.response
          }
        } else if (providerKey === Provider.Groq) {
          providerResponse = await (client as Groq).chat.completions.create(mappedParams)
        } else if (providerKey === Provider.OpenAI) {
          providerResponse = await (client as OpenAI | AzureOpenAI).chat.completions.create(mappedParams)
        } else {
          const _e: never = providerKey
          throw new RosettaAIError(`Unsupported built-in provider: ${_e}`)
        }

        // --- Map Response ---
        if (!mapper.mapFromProviderResponse) {
          throw new MappingError(
            `mapFromProviderResponse is required for built-in provider '${providerKey}' but not implemented.`,
            providerKey
          )
        }
        return mapper.mapFromProviderResponse(providerResponse, model, params.tools)
      } else {
        // Custom provider without executeGenerate or built-in provider without mapFromProviderResponse
        throw new ConfigurationError(
          `Provider '${providerKey}' is not configured correctly for non-streaming generation. Missing required mapper methods.`
        )
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      if (error instanceof RosettaAIError) {
        throw error
      }
      throw this.wrapProviderError(error, providerKey)
    }
  }

  /**
   * Generates a streaming response.
   * Supports both built-in and custom providers.
   *
   * @param params - The parameters for the streaming generation request.
   * @returns An async iterable yielding stream chunks.
   * @throws {ConfigurationError} If the provider or model is not configured (yielded as error chunk).
   * @throws {UnsupportedFeatureError} If a requested feature is not supported (yielded as error chunk).
   * @throws {InvalidToolDefinitionError} If a provided tool definition is invalid (yielded as error chunk).
   * @throws {ToolArgumentValidationError} If the LLM provides invalid arguments for a tool call (yielded as error chunk).
   * @throws {ProviderAPIError} If the provider's API returns an error during setup or streaming (yielded as error chunk).
   * @throws {MappingError} If internal mapping fails during setup or streaming (yielded as error chunk).
   */
  public async *stream(params: GenerateParams): AsyncIterable<StreamChunk> {
    const providerKey = params.provider // Now ProviderKey
    let mapper: IProviderMapper
    let isCustom: boolean
    let customConfig: CustomProviderConfig | undefined
    let apiKey: string | undefined
    let model: string | undefined

    try {
      mapper = this.getMapper(providerKey)
      isCustom = !this.isBuiltInProvider(providerKey)
      customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      // Determine model ID
      model =
        params.model ??
        (isCustom ? customConfig?.defaultModel : this.config.defaultModels?.[providerKey as Provider]) ??
        undefined

      if (!model) {
        throw new ConfigurationError(`Model must be specified for provider ${providerKey} (or set a default).`)
      }

      const effectiveParams = { ...params, model, stream: true }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Generate', !!this.azureOpenAIClient)

      // --- Map Parameters ---
      const mappedParams = mapper.mapToProviderParams ? mapper.mapToProviderParams(effectiveParams) : effectiveParams

      // --- Execute ---
      if (isCustom && mapper.executeStream && customConfig) {
        // Custom Provider Execution Path
        yield* mapper.executeStream(mappedParams, apiKey, customConfig, params)
      } else if (this.isBuiltInProvider(providerKey)) {
        // Built-in Provider Execution Path
        const client = this.getClientForProvider(providerKey)
        let providerStream: any

        if (providerKey === Provider.Anthropic) {
          providerStream = await (client as Anthropic).messages.create(mappedParams)
        } else if (providerKey === Provider.Google) {
          const googleM = this.getGoogleModel(model, params.providerOptions)
          const { googleMappedParams: googleP, isChat } = mappedParams
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
        } else if (providerKey === Provider.Groq) {
          providerStream = await (client as Groq).chat.completions.create(mappedParams)
        } else if (providerKey === Provider.OpenAI) {
          providerStream = await (client as OpenAI | AzureOpenAI).chat.completions.create(mappedParams)
        } else {
          const _e: never = providerKey
          throw new RosettaAIError(`Unsupported built-in provider: ${_e}`)
        }

        // --- Map Stream ---
        if (!mapper.mapProviderStream) {
          throw new MappingError(
            `mapProviderStream is required for built-in provider '${providerKey}' but not implemented.`,
            providerKey
          )
        }
        if (!(typeof providerStream?.[Symbol.asyncIterator] === 'function')) {
          console.error('Provider response details:', providerStream)
          throw new MappingError(
            `Provider ${providerKey} did not return a stream for a streaming request. Check mapper implementation.`,
            providerKey
          )
        }
        // Pass the full params object to mapProviderStream
        yield* mapper.mapProviderStream(providerStream as AsyncIterable<any>, params)
      } else {
        // Custom provider without executeStream or built-in provider without mapProviderStream
        throw new ConfigurationError(
          `Provider '${providerKey}' is not configured correctly for streaming generation. Missing required mapper methods.`
        )
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      const wrappedError = error instanceof RosettaAIError ? error : this.wrapProviderError(error, providerKey)
      yield { type: 'error', data: { error: wrappedError } }
      return // Exit generator after yielding error
    }
  }

  /** Generates embedding vectors. Supports built-in and custom providers. */
  public async embed(params: EmbedParams): Promise<EmbedResult> {
    const providerKey = params.provider
    try {
      const mapper = this.getMapper(providerKey)
      const isCustom = !this.isBuiltInProvider(providerKey)
      const customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      const apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      const model =
        params.model ??
        (isCustom
          ? customConfig?.defaultEmbeddingModel
          : this.config.defaultEmbeddingModels?.[providerKey as Provider]) ??
        undefined

      if (!model) {
        throw new ConfigurationError(
          `Embedding model must be specified for provider ${providerKey} (or set a default).`
        )
      }

      const effectiveParams = { ...params, model }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Embeddings', !!this.azureOpenAIClient)

      // --- Map Parameters ---
      const mappedParams = mapper.mapToEmbedParams ? mapper.mapToEmbedParams(effectiveParams) : effectiveParams

      // --- Execute ---
      if (isCustom && mapper.executeEmbed && customConfig) {
        // Custom Provider Execution Path
        return await mapper.executeEmbed(mappedParams, apiKey, customConfig, params)
      } else if (this.isBuiltInProvider(providerKey)) {
        // Built-in Provider Execution Path
        const client = this.getClientForProvider(providerKey)
        let providerResponse: any
        // --- Client Call Logic ---
        if (providerKey === Provider.Google) {
          const googleM = this.getGoogleModel(model, params.providerOptions)
          if ('requests' in mappedParams) {
            providerResponse = await googleM.batchEmbedContents(mappedParams as BatchEmbedContentsRequest)
          } else {
            providerResponse = await googleM.embedContent(mappedParams as EmbedContentRequest)
          }
        } else if (providerKey === Provider.Groq) {
          providerResponse = await (client as Groq).embeddings.create(mappedParams)
        } else if (providerKey === Provider.OpenAI) {
          providerResponse = await (client as OpenAI | AzureOpenAI).embeddings.create(mappedParams)
        } else {
          throw new UnsupportedFeatureError(providerKey, 'Embeddings')
        }
        // --- End Client Call Logic ---

        // --- Map Response ---
        if (!mapper.mapFromEmbedResponse) {
          throw new MappingError(
            `mapFromEmbedResponse is required for built-in provider '${providerKey}' but not implemented.`,
            providerKey
          )
        }
        return mapper.mapFromEmbedResponse(providerResponse, model)
      } else {
        throw new ConfigurationError(
          `Provider '${providerKey}' is not configured correctly for embeddings. Missing required mapper methods.`
        )
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      if (error instanceof RosettaAIError) {
        throw error
      }
      throw this.wrapProviderError(error, providerKey)
    }
  }

  /** Generates speech audio. Supports built-in (OpenAI) and custom providers. */
  public async generateSpeech(params: SpeechParams): Promise<Buffer> {
    const providerKey = params.provider

    // Check for known unsupported built-in providers first
    if (this.isBuiltInProvider(providerKey)) {
      const provider = providerKey as Provider
      if (![Provider.OpenAI, Provider.Groq].includes(provider)) {
        throw new UnsupportedFeatureError(provider, 'Text-to-Speech')
      }
    }

    try {
      const mapper = this.getMapper(providerKey)
      const isCustom = !this.isBuiltInProvider(providerKey)
      const customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      const apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      const model =
        params.model ??
        (isCustom ? customConfig?.defaultTtsModel : this.config.defaultTtsModels?.[providerKey as Provider.OpenAI]) ??
        (providerKey === Provider.OpenAI ? 'tts-1' : undefined) // Default for OpenAI

      if (!model && providerKey === Provider.OpenAI) {
        // This case should technically be covered by the default 'tts-1' above, but added for clarity
        throw new ConfigurationError(`TTS model must be specified for provider ${providerKey} (or set a default).`)
      }
      // For custom providers, model might be optional depending on the implementation

      const effectiveParams = { ...params, model }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Text-to-Speech', !!this.azureOpenAIClient)

      // --- Map Parameters (Optional for TTS, often raw params are fine) ---
      // Custom mappers might not need specific mapping for TTS params
      const mappedParams = effectiveParams // Pass effective params directly for now

      // --- Execute ---
      if (isCustom && mapper.executeGenerateSpeech && customConfig) {
        // Custom Provider Execution Path
        return await mapper.executeGenerateSpeech(mappedParams, apiKey, customConfig, params)
      } else if (providerKey === Provider.OpenAI) {
        // Built-in OpenAI Execution Path
        const client = this.getClientForProvider(Provider.OpenAI)
        const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
          model: effectiveParams.model!, // Model is guaranteed by checks above for OpenAI
          input: effectiveParams.input,
          voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
          response_format: effectiveParams.responseFormat ?? 'mp3',
          speed: effectiveParams.speed ?? 1.0
        }
        const response = await (client as OpenAI | AzureOpenAI).audio.speech.create(ttsParams)
        return Buffer.from(await response.arrayBuffer())
      } else if (providerKey === Provider.Groq) {
        // Built-in Groq Execution Path
        const client = this.getClientForProvider(Provider.Groq)
        const ttsParams = GroqAudioMapper.mapToGroqTtsParams(effectiveParams)
        const response = await (client as Groq).audio.speech.create(ttsParams)
        return Buffer.from(await response.arrayBuffer())
      } else {
        throw new UnsupportedFeatureError(providerKey, 'Text-to-Speech')
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      if (error instanceof RosettaAIError) {
        throw error
      }
      throw this.wrapProviderError(error, providerKey)
    }
  }

  /** Generates streaming speech audio. Supports built-in (OpenAI) and custom providers. */
  public async *streamSpeech(params: SpeechParams): AsyncIterable<AudioStreamChunk> {
    const providerKey = params.provider
    let mapper: IProviderMapper
    let isCustom: boolean
    let customConfig: CustomProviderConfig | undefined
    let apiKey: string | undefined
    let model: string | undefined

    try {
      mapper = this.getMapper(providerKey)
      isCustom = !this.isBuiltInProvider(providerKey)
      customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      model =
        params.model ??
        (isCustom ? customConfig?.defaultTtsModel : this.config.defaultTtsModels?.[providerKey as Provider.OpenAI]) ??
        (providerKey === Provider.OpenAI ? 'tts-1' : undefined)

      if (!model && providerKey === Provider.OpenAI) {
        throw new ConfigurationError(`TTS model must be specified for provider ${providerKey} (or set a default).`)
      }

      const effectiveParams = { ...params, model }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Streaming Text-to-Speech', !!this.azureOpenAIClient)

      // --- Map Parameters ---
      const mappedParams = effectiveParams // Pass effective params directly

      // --- Execute ---
      if (isCustom && mapper.executeStreamSpeech && customConfig) {
        // Custom Provider Execution Path
        yield* mapper.executeStreamSpeech(mappedParams, apiKey, customConfig, params)
      } else if (providerKey === Provider.OpenAI) {
        // Built-in OpenAI Execution Path
        const client = this.getClientForProvider(Provider.OpenAI)
        const ttsParams: OpenAI.Audio.Speech.SpeechCreateParams = {
          model: effectiveParams.model!,
          input: effectiveParams.input,
          voice: effectiveParams.voice as OpenAI.Audio.Speech.SpeechCreateParams['voice'],
          response_format: effectiveParams.responseFormat ?? 'mp3',
          speed: effectiveParams.speed ?? 1.0
        }
        const response = await (client as OpenAI | AzureOpenAI).audio.speech.create(ttsParams)

        if (!response.body) {
          throw new MappingError('Streaming response body is null.', providerKey)
        }

        for await (const chunk of response.body) {
          if (chunk instanceof Uint8Array) {
            yield { type: 'audio_chunk', data: Buffer.from(chunk) }
          } else {
            console.warn('Received unexpected chunk type in audio stream:', typeof chunk)
          }
        }
        yield { type: 'audio_stop' }
      } else {
        throw new UnsupportedFeatureError(providerKey, 'Streaming Text-to-Speech')
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      const wrappedError = error instanceof RosettaAIError ? error : this.wrapProviderError(error, providerKey)
      yield { type: 'error', data: { error: wrappedError } }
      return // Exit generator after yielding error. Do not throw.
    }
  }

  /** Transcribes audio to text. Supports built-in (OpenAI, Groq) and custom providers. */
  public async transcribe(params: TranscribeParams): Promise<TranscriptionResult> {
    const providerKey = params.provider
    try {
      const mapper = this.getMapper(providerKey)
      const isCustom = !this.isBuiltInProvider(providerKey)
      const customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      const apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      const model =
        params.model ??
        (isCustom ? customConfig?.defaultSttModel : this.config.defaultSttModels?.[providerKey as Provider]) ??
        undefined

      if (!model && (providerKey === Provider.OpenAI || providerKey === Provider.Groq)) {
        throw new ConfigurationError(
          `Transcription model must be specified for provider ${providerKey} (or set a default).`
        )
      }

      const effectiveParams = { ...params, model }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Audio Transcription', !!this.azureOpenAIClient)

      const audioFile = await prepareAudioUpload(effectiveParams.audio)
      // --- Map Parameters ---
      const mappedParams = mapper.mapToTranscribeParams
        ? mapper.mapToTranscribeParams(effectiveParams, audioFile)
        : effectiveParams

      // --- Execute ---
      if (isCustom && mapper.executeTranscribe && customConfig) {
        // Custom Provider Execution Path
        return await mapper.executeTranscribe(mappedParams, apiKey, customConfig, params)
      } else if (this.isBuiltInProvider(providerKey)) {
        // Built-in Provider Execution Path
        const client = this.getClientForProvider(providerKey)
        let providerResponse: any

        if (providerKey === Provider.OpenAI) {
          providerResponse = await (client as OpenAI | AzureOpenAI).audio.transcriptions.create(mappedParams)
        } else if (providerKey === Provider.Groq) {
          providerResponse = await (client as Groq).audio.transcriptions.create(mappedParams)
        } else {
          throw new UnsupportedFeatureError(providerKey, 'Audio Transcription')
        }

        // --- Map Response ---
        if (!mapper.mapFromTranscribeResponse) {
          throw new MappingError(
            `mapFromTranscribeResponse is required for built-in provider '${providerKey}' but not implemented.`,
            providerKey
          )
        }
        return mapper.mapFromTranscribeResponse(providerResponse, model!) // Model is guaranteed by checks above
      } else {
        throw new ConfigurationError(
          `Provider '${providerKey}' is not configured correctly for transcription. Missing required mapper methods.`
        )
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      if (error instanceof RosettaAIError) {
        throw error
      }
      throw this.wrapProviderError(error, providerKey)
    }
  }

  /** Translates audio to English text. Supports built-in (OpenAI, Groq) and custom providers. */
  public async translate(params: TranslateParams): Promise<TranscriptionResult> {
    const providerKey = params.provider
    try {
      const mapper = this.getMapper(providerKey)
      const isCustom = !this.isBuiltInProvider(providerKey)
      const customConfig = isCustom ? this.customProviderConfigs.get(providerKey) : undefined
      const apiKey = isCustom ? this.customApiKeys.get(providerKey) : undefined

      const model =
        params.model ??
        (isCustom ? customConfig?.defaultSttModel : this.config.defaultSttModels?.[providerKey as Provider]) ??
        undefined

      if (!model && (providerKey === Provider.OpenAI || providerKey === Provider.Groq)) {
        throw new ConfigurationError(
          `Translation model must be specified for provider ${providerKey} (or set a default).`
        )
      }

      const effectiveParams = { ...params, model }
      this.checkUnsupportedFeatures(providerKey, effectiveParams, 'Audio Translation', !!this.azureOpenAIClient)

      const audioFile = await prepareAudioUpload(effectiveParams.audio)
      // --- Map Parameters ---
      const mappedParams = mapper.mapToTranslateParams
        ? mapper.mapToTranslateParams(effectiveParams, audioFile)
        : effectiveParams

      // --- Execute ---
      if (isCustom && mapper.executeTranslate && customConfig) {
        // Custom Provider Execution Path
        return await mapper.executeTranslate(mappedParams, apiKey, customConfig, params)
      } else if (this.isBuiltInProvider(providerKey)) {
        // Built-in Provider Execution Path
        const client = this.getClientForProvider(providerKey)
        let providerResponse: any

        if (providerKey === Provider.OpenAI) {
          providerResponse = await (client as OpenAI | AzureOpenAI).audio.translations.create(mappedParams)
        } else if (providerKey === Provider.Groq) {
          providerResponse = await (client as Groq).audio.translations.create(mappedParams)
        } else {
          throw new UnsupportedFeatureError(providerKey, 'Audio Translation')
        }

        // --- Map Response ---
        if (!mapper.mapFromTranslateResponse) {
          throw new MappingError(
            `mapFromTranslateResponse is required for built-in provider '${providerKey}' but not implemented.`,
            providerKey
          )
        }
        return mapper.mapFromTranslateResponse(providerResponse, model!) // Model is guaranteed by checks above
      } else {
        throw new ConfigurationError(
          `Provider '${providerKey}' is not configured correctly for translation. Missing required mapper methods.`
        )
      }
    } catch (error) {
      // Check if error is already RosettaAIError before wrapping
      if (error instanceof RosettaAIError) {
        throw error
      }
      throw this.wrapProviderError(error, providerKey)
    }
  }

  /**
   * Lists the models available for a specific configured provider (built-in or custom).
   *
   * @param providerKey The provider key (built-in or custom) for which to list models.
   * @param sourceConfig Optional configuration overriding the default listing source for this call.
   * @returns A promise resolving to a list of available models.
   * @throws {ConfigurationError} If the provider is not configured or the listing source is invalid.
   * @throws {ProviderAPIError} If the API call fails (for API endpoints or SDK methods).
   * @throws {MappingError} If the response from the provider cannot be parsed or mapped correctly.
   * @throws {UnsupportedFeatureError} If model listing is not supported for the provider.
   */
  public async listModels(
    providerKey: ProviderKey,
    sourceConfig?: ModelListingSourceConfig
  ): Promise<RosettaModelList> {
    // Ensure provider is configured
    if (!this.mappers.has(providerKey)) {
      throw new ConfigurationError(`Provider '${providerKey}' is not configured in this RosettaAI instance.`)
    }

    // Handle built-in providers
    if (this.isBuiltInProvider(providerKey)) {
      // Determine if we're using Azure OpenAI for the OpenAI provider
      const isAzureOpenAI = providerKey === Provider.OpenAI && !!this.azureOpenAIClient

      // Determine the appropriate API key based on provider and client type
      let apiKey: string | undefined
      if (providerKey === Provider.Anthropic) {
        apiKey = this.config.anthropicApiKey
      } else if (providerKey === Provider.Google) {
        apiKey = this.config.googleApiKey
      } else if (providerKey === Provider.Groq) {
        apiKey = this.config.groqApiKey
      } else if (providerKey === Provider.OpenAI) {
        // For OpenAI, use the appropriate key based on which client is active
        apiKey = isAzureOpenAI ? this.config.azureOpenAIApiKey : this.config.openaiApiKey
      }

      // Create a custom source config for Azure OpenAI if needed
      let effectiveSourceConfig = sourceConfig ?? this.config.modelListingConfig?.[providerKey]

      // If using Azure OpenAI but no custom source config is provided, create one
      if (isAzureOpenAI && !effectiveSourceConfig && this.config.azureOpenAIEndpoint) {
        // Create a source config for Azure OpenAI deployments endpoint
        const azureEndpoint = this.config.azureOpenAIEndpoint
        const apiVersion = this.config.azureOpenAIApiVersion
        if (azureEndpoint && apiVersion) {
          const baseUrl = azureEndpoint.endsWith('/') ? azureEndpoint.slice(0, -1) : azureEndpoint
          effectiveSourceConfig = {
            type: 'apiEndpoint',
            url: `${baseUrl}/openai/deployments?api-version=${apiVersion}`
          }
          console.log(
            `RosettaAI: Using Azure OpenAI deployments endpoint for model listing: ${effectiveSourceConfig.url}`
          )
        }
      }

      const listConfig = {
        sourceConfig: effectiveSourceConfig,
        apiKey,
        groqClient: this.groqClient,
        isAzureOpenAI // Pass this flag to listModelsForProvider
      }

      return listModelsForProvider(providerKey, listConfig)
    } else {
      // --- Custom Provider Model Listing ---
      const customConfig = this.customProviderConfigs.get(providerKey)
      const apiKey = this.customApiKeys.get(providerKey)

      if (!customConfig) {
        // Should not happen if mapper exists, but handle defensively
        throw new ConfigurationError(`Configuration for custom provider '${providerKey}' not found.`)
      }

      // Check if the custom provider supports listing models
      if (!customConfig.supportedFeatures.includes('list_models')) {
        throw new UnsupportedFeatureError(providerKey, 'Model Listing')
      }

      // Pass custom config to the internal lister
      const listConfig = {
        sourceConfig: sourceConfig, // Pass override if provided
        apiKey: apiKey,
        customConfig: customConfig // Pass the full custom config
      }
      // Call the internal lister, which now needs to handle customConfig
      return listModelsForProvider(providerKey, listConfig)
    }
  }

  /**
   * Lists available models for all configured providers (built-in and custom).
   * Returns a record where keys are provider names and values are either the list
   * of models or an error object if listing failed for that provider.
   *
   * @returns A promise resolving to a record of provider model lists or errors.
   */
  public async listAllModels(): Promise<Partial<Record<ProviderKey, RosettaModelList | RosettaAIError>>> {
    const configuredProviders = this.getConfiguredProviders()
    console.log(`RosettaAI: Listing models for all configured providers: ${configuredProviders.join(', ')}`)
    const results: Partial<Record<ProviderKey, RosettaModelList | RosettaAIError>> = {}

    // Process providers sequentially to avoid potential race conditions
    // This is more reliable than parallel processing when multiple providers are configured
    for (const providerKey of configuredProviders) {
      try {
        console.log(`RosettaAI: Listing models for provider: ${providerKey}`)
        const models = await this.listModels(providerKey)
        console.log(`RosettaAI: Successfully listed ${models.data.length} models for ${providerKey}`)
        results[providerKey] = models
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error)
        console.error(`RosettaAI: Error listing models for ${providerKey}: ${errorMessage}`)
        // Add detailed error information
        if (error instanceof RosettaAIError) {
          console.error(`RosettaAI: Error type: ${error.constructor.name}`)
          if (error instanceof ProviderAPIError && error.statusCode) {
            console.error(`RosettaAI: Status code: ${error.statusCode}`)
          }
        }
        // Store the error in the results
        results[providerKey] =
          error instanceof RosettaAIError
            ? error
            : new ProviderAPIError(String(error), providerKey, undefined, undefined, undefined, error)
      }
    }

    return results
  }

  /** @internal Gets provider client instance or throws config error. */
  private providerNotConfigured(p: ProviderKey): ConfigurationError {
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
    providerKey: ProviderKey,
    params: GenerateParams | EmbedParams | SpeechParams | TranscribeParams | TranslateParams,
    featureName: string,
    _isAzure: boolean = false // Keep isAzure flag if needed for future checks
  ): void {
    // Check custom provider features first if applicable
    if (!this.isBuiltInProvider(providerKey)) {
      const customConfig = this.customProviderConfigs.get(providerKey)
      if (!customConfig) {
        // Should not happen if getMapper succeeded, but handle defensively
        throw new ConfigurationError(`Custom provider '${providerKey}' configuration not found.`)
      }
      // Map featureName to the keys in supportedFeatures array
      let featureKey: CustomProviderConfig['supportedFeatures'][number] | undefined
      switch (featureName.toLowerCase()) {
        case 'generate':
          featureKey = 'generate'
          break
        case 'streaming generation': // Handle stream feature name if needed
          featureKey = 'stream'
          break
        case 'embeddings':
          featureKey = 'embed'
          break
        case 'text-to-speech':
          featureKey = 'tts'
          break
        case 'streaming text-to-speech':
          featureKey = 'tts' // Assume same feature flag for now
          break
        case 'audio transcription':
          featureKey = 'stt'
          break
        case 'audio translation':
          featureKey = 'translate'
          break
        case 'model listing': // Check for model listing
          featureKey = 'list_models'
          break
        // Add mappings for other features like tool_use, image_input, json_mode
      }

      if (featureKey && !customConfig.supportedFeatures.includes(featureKey)) {
        throw new UnsupportedFeatureError(providerKey, featureName)
      }
      // Add checks based on parameters for custom providers if needed
      if ('messages' in params) {
        // GenerateParams
        const hasImage = params.messages.some(
          msg => Array.isArray(msg.content) && msg.content.some(part => part.type === 'image')
        )
        if (hasImage && !customConfig.supportedFeatures.includes('image_input')) {
          throw new UnsupportedFeatureError(providerKey, 'Image input')
        }
        if (params.tools && params.tools.length > 0 && !customConfig.supportedFeatures.includes('tool_use')) {
          throw new UnsupportedFeatureError(providerKey, 'Tool use')
        }
        if (params.responseFormat?.type === 'json_object' && !customConfig.supportedFeatures.includes('json_mode')) {
          throw new UnsupportedFeatureError(providerKey, 'JSON Mode')
        }
      }
      // Add checks for other param types (Embed, Speech, etc.) against supportedFeatures

      return // Skip built-in checks for custom providers
    }

    // --- Built-in Provider Checks ---
    const provider = providerKey as Provider // Safe cast after isBuiltInProvider check

    if (
      featureName === 'Model Listing' &&
      ![Provider.OpenAI, Provider.Groq, Provider.Google, Provider.Anthropic].includes(provider)
    ) {
      // Currently all built-in providers support some form of listing
      // This check might be redundant unless a new built-in provider is added without listing support
      throw new UnsupportedFeatureError(provider, featureName)
    }
    if (
      (featureName === 'Audio Transcription' || featureName === 'Audio Translation') &&
      ![Provider.OpenAI, Provider.Groq].includes(provider)
    ) {
      throw new UnsupportedFeatureError(provider, featureName)
    }
    if (featureName === 'Text-to-Speech' && ![Provider.OpenAI, Provider.Groq].includes(provider)) {
      throw new UnsupportedFeatureError(provider, featureName)
    }
    if (featureName === 'Streaming Text-to-Speech' && provider !== Provider.OpenAI) {
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
      // Check for tools usage
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
      // EmbedParams
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
      // TranscribeParams or TranslateParams
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
  private wrapProviderError(error: unknown, providerKey: ProviderKey): RosettaAIError {
    // Check if error is already RosettaAIError before delegating to mapper
    if (error instanceof RosettaAIError) {
      return error
    }

    // Allow mapper to handle first if it exists
    const mapper = this.mappers.get(providerKey)
    if (mapper) {
      try {
        // Attempt to use the mapper's specific error wrapping
        return mapper.wrapProviderError(error, providerKey)
      } catch (mapperError) {
        // If the mapper's wrap function itself fails, fall back to generic handling
        console.error(`Error during mapper's wrapProviderError for ${providerKey}:`, mapperError)
      }
    }

    // Fallback generic handling (if no mapper or mapper failed)
    // Check if it's already an SDK error first in the fallback (redundant due to check above, but safe)
    if (error instanceof RosettaAIError) {
      return error
    }

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

    // Use ProviderAPIError as the default fallback wrapper
    return new ProviderAPIError(errorMessage, providerKey, undefined, undefined, undefined, error)
  }
}
