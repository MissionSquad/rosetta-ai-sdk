import Anthropic from '@anthropic-ai/sdk'
import Groq from 'groq-sdk'
import OpenAI, { AzureOpenAI } from 'openai'
import { GoogleGenAI } from '@google/genai' // Import Google client
import { z } from 'zod' // Import Zod
import {
  RosettaAI,
  Provider,
  ConfigurationError,
  UnsupportedFeatureError,
  ProviderAPIError,
  RosettaAudioData,
  StreamChunk,
  GenerateParams,
  SpeechParams,
  TranscribeParams,
  EmbedParams,
  TranslateParams,
  GenerateResult,
  EmbedResult,
  TranscriptionResult,
  RosettaAIError,
  RosettaModelList,
  ModelListingSourceConfig,
  RosettaTool,
  ToolArgumentValidationError,
  CustomProviderConfig
} from '../../../src'
import { BaseCustomMapper } from '../../../src/core/mapping/base.custom.mapper'

// Mock the mapper classes
jest.mock('../../../src/core/mapping/anthropic.mapper')
jest.mock('../../../src/core/mapping/google.mapper')
jest.mock('../../../src/core/mapping/groq.mapper')
jest.mock('../../../src/core/mapping/openai.mapper')
jest.mock('../../../src/core/mapping/azure.openai.mapper')

// Import the mocked classes
import { AnthropicMapper } from '../../../src/core/mapping/anthropic.mapper'
import { GoogleMapper } from '../../../src/core/mapping/google.mapper'
import { GroqMapper } from '../../../src/core/mapping/groq.mapper'
import { OpenAIMapper } from '../../../src/core/mapping/openai.mapper'
import { AzureOpenAIMapper } from '../../../src/core/mapping/azure.openai.mapper'

// Mock the underlying SDK clients
jest.mock('@anthropic-ai/sdk')
jest.mock('@google/genai')
jest.mock('groq-sdk')
jest.mock('openai') // Mocks both OpenAI and AzureOpenAI constructors

// Mock utility functions
jest.mock('../../../src/core/utils', () => ({
  ...jest.requireActual('../../../src/core/utils'),
  prepareAudioUpload: jest.fn()
}))
import { prepareAudioUpload } from '../../../src/core/utils'

// Mock the internal model lister function
jest.mock('../../../src/core/listing/model.lister')
import { listModelsForProvider } from '../../../src/core/listing/model.lister'

// Mock the Groq audio mapper for TTS tests
jest.mock('../../../src/core/mapping/groq.audio.mapper')
import * as GroqAudioMapper from '../../../src/core/mapping/groq.audio.mapper'

// Mock implementation for prepareAudioUpload
const mockPrepareAudioUpload = prepareAudioUpload as jest.Mock
const mockAudioFile = { name: 'mock.mp3', type: 'audio/mpeg', [Symbol.toStringTag]: 'File' }

// Mock implementation for listModelsForProvider
const mockListModelsForProvider = listModelsForProvider as jest.Mock

import { AudioStreamChunk } from '../../../src' // Import AudioStreamChunk

// Helper async generator for stream tests
async function* mockStreamGenerator<T>(chunks: T[]): AsyncIterable<T> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield chunk
  }
}

// Helper async generator that throws an error during iteration
async function* mockErrorStreamGenerator<T>(chunks: T[], errorToThrow: Error): AsyncIterable<T> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield chunk
  }
  throw errorToThrow
}

// Generic helper to collect stream chunks
// Define a base chunk type and the specific error chunk structure
type BaseChunk = { type: string; data?: any }
type ErrorChunk = { type: 'error'; data: { error: RosettaAIError } }
// Use a union type for the collectable chunks
type CollectableChunk = StreamChunk | AudioStreamChunk | ErrorChunk
async function collectStreamChunks<T extends CollectableChunk>(stream: AsyncIterable<T>): Promise<CollectableChunk[]> {
  const chunks: CollectableChunk[] = [] // Use the union type here
  try {
    for await (const chunk of stream) {
      chunks.push(chunk)
    }
  } catch (error) {
    const wrappedError =
      error instanceof RosettaAIError
        ? error
        : error instanceof Error
        ? new RosettaAIError(error.message) // Wrap generic errors
        : new RosettaAIError(String(error ?? 'Unknown stream error during collection'))

    const lastChunk = chunks[chunks.length - 1]
    // Check if the last chunk is already an error chunk
    if (!(lastChunk?.type === 'error' && (lastChunk as ErrorChunk).data?.error)) {
      // Add the correctly typed ErrorChunk
      const errorChunk: ErrorChunk = {
        type: 'error',
        data: { error: wrappedError }
      }
      chunks.push(errorChunk) // No cast needed as chunks array type includes ErrorChunk
    } else {
      console.warn('Caught error during stream collection, but an error chunk was already present.')
    }
  }
  return chunks
}

// Define a mock tool for testing
const mockTool: RosettaTool<any> = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Gets the weather',
    parameters: { type: 'object', properties: { location: { type: 'string' } } },
    zodSchema: z.object({ location: z.string() }) // Add zod schema
  }
}

// --- Mock Custom Mapper ---
class MockCustomMapper extends BaseCustomMapper {
  // Override executeGenerate for testing delegation
  async executeGenerate(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): Promise<GenerateResult> {
    // Simulate API call using apiKey and config
    if (!apiKey) throw new Error('Custom provider API key missing')
    return {
      content: `Custom response for ${originalParams.messages[0]?.content} using ${apiKey.substring(0, 3)}...`,
      finishReason: 'stop',
      model: providerConfig.defaultModel ?? 'custom-default',
      usage: { promptTokens: 1, completionTokens: 10, totalTokens: 11 },
      rawResponse: { customData: 'xyz' }
    }
  }

  // Override executeStream for testing delegation
  async *executeStream(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): AsyncIterable<StreamChunk> {
    if (!apiKey) throw new Error('Custom provider API key missing')
    yield {
      type: 'message_start',
      data: { provider: this.provider, model: providerConfig.defaultModel ?? 'custom-stream' }
    }
    yield { type: 'content_delta', data: { delta: 'Custom ' } }
    yield { type: 'content_delta', data: { delta: 'Stream ' } }
    yield { type: 'message_stop', data: { finishReason: 'stop' } }
    yield { type: 'final_usage', data: { usage: { promptTokens: 1, completionTokens: 2, totalTokens: 3 } } }
  }

  // Override executeEmbed for testing delegation
  async executeEmbed(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: EmbedParams
  ): Promise<EmbedResult> {
    if (!apiKey) throw new Error('Custom provider API key missing')
    return {
      embeddings: [[0.9, 0.8]],
      model: providerConfig.defaultEmbeddingModel ?? 'custom-embed-default',
      usage: { totalTokens: 5 },
      rawResponse: { customEmbedData: 'abc' }
    }
  }
}
// --- End Mock Custom Mapper ---

describe('RosettaAI Core (with V2 Mappers & Custom Providers)', () => {
  let originalEnv: NodeJS.ProcessEnv
  let warnSpy: jest.SpyInstance

  // --- Mock Mapper Implementations ---
  // Use jest.fn() for methods to allow tracking calls and setting return values
  const mockAnthropicMapperInstance = {
    provider: Provider.Anthropic,
    mapToProviderParams: jest.fn().mockReturnValue({ mapped: 'anthropic_params' }),
    mapFromProviderResponse: jest.fn().mockReturnValue(({ mapped: 'anthropic_result' } as unknown) as GenerateResult),
    mapProviderStream: jest.fn().mockImplementation(() => mockStreamGenerator([])),
    mapToEmbedParams: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Embeddings')
    }),
    mapFromEmbedResponse: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Embeddings')
    }),
    mapToTranscribeParams: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Audio Transcription')
    }),
    mapFromTranscribeResponse: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Audio Transcription')
    }),
    mapToTranslateParams: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Audio Translation')
    }),
    mapFromTranslateResponse: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Anthropic, 'Audio Translation')
    }),
    wrapProviderError: jest.fn(err => err) // Simple pass-through for testing
  }
  const mockGoogleMapperInstance = {
    provider: Provider.Google,
    mapToProviderParams: jest.fn().mockReturnValue({ googleMappedParams: { mapped: 'google_params' }, isChat: false }),
    mapFromProviderResponse: jest.fn().mockReturnValue(({ mapped: 'google_result' } as unknown) as GenerateResult),
    mapProviderStream: jest.fn().mockImplementation(() => mockStreamGenerator([])),
    mapToEmbedParams: jest.fn().mockReturnValue({ mapped: 'google_embed_params' }),
    mapFromEmbedResponse: jest.fn().mockReturnValue(({ mapped: 'google_embed_result' } as unknown) as EmbedResult),
    mapToTranscribeParams: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Google, 'Audio Transcription')
    }),
    mapFromTranscribeResponse: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Google, 'Audio Transcription')
    }),
    mapToTranslateParams: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Google, 'Audio Translation')
    }),
    mapFromTranslateResponse: jest.fn(() => {
      throw new UnsupportedFeatureError(Provider.Google, 'Audio Translation')
    }),
    wrapProviderError: jest.fn(err => err)
  }
  const mockGroqMapperInstance = {
    provider: Provider.Groq,
    mapToProviderParams: jest.fn().mockReturnValue({ mapped: 'groq_params' }),
    mapFromProviderResponse: jest.fn().mockReturnValue(({ mapped: 'groq_result' } as unknown) as GenerateResult),
    mapProviderStream: jest.fn().mockImplementation(() => mockStreamGenerator([])),
    mapToEmbedParams: jest.fn().mockReturnValue({ mapped: 'groq_embed_params' }),
    mapFromEmbedResponse: jest.fn().mockReturnValue(({ mapped: 'groq_embed_result' } as unknown) as EmbedResult),
    mapToTranscribeParams: jest.fn().mockReturnValue({ mapped: 'groq_stt_params' }),
    mapFromTranscribeResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'groq_stt_result' } as unknown) as TranscriptionResult),
    mapToTranslateParams: jest.fn().mockReturnValue({ mapped: 'groq_translate_params' }),
    mapFromTranslateResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'groq_translate_result' } as unknown) as TranscriptionResult),
    wrapProviderError: jest.fn(err => err)
  }
  const mockOpenAIMapperInstance = {
    provider: Provider.OpenAI,
    mapToProviderParams: jest.fn().mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' }),
    mapFromProviderResponse: jest.fn().mockReturnValue(({ mapped: 'openai_result' } as unknown) as GenerateResult),
    mapProviderStream: jest.fn().mockImplementation(() => mockStreamGenerator([])),
    mapToEmbedParams: jest.fn().mockReturnValue({ mapped: 'openai_embed_params', model: 'text-embedding-ada-002' }),
    mapFromEmbedResponse: jest.fn().mockReturnValue(({ mapped: 'openai_embed_result' } as unknown) as EmbedResult),
    mapToTranscribeParams: jest.fn().mockReturnValue({ mapped: 'openai_stt_params' }),
    mapFromTranscribeResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'openai_stt_result' } as unknown) as TranscriptionResult),
    mapToTranslateParams: jest.fn().mockReturnValue({ mapped: 'openai_translate_params' }),
    mapFromTranslateResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'openai_translate_result' } as unknown) as TranscriptionResult),
    wrapProviderError: jest.fn(err => err)
  }
  const mockAzureMapperInstance = {
    provider: Provider.OpenAI,
    mapToProviderParams: jest.fn().mockReturnValue({ mapped: 'azure_params', model: 'azure_deployment' }),
    mapFromProviderResponse: jest.fn().mockReturnValue(({ mapped: 'azure_result' } as unknown) as GenerateResult),
    mapProviderStream: jest.fn().mockImplementation(() => mockStreamGenerator([])),
    mapToEmbedParams: jest.fn().mockReturnValue({ mapped: 'azure_embed_params', model: 'azure_embed_deployment' }),
    mapFromEmbedResponse: jest.fn().mockReturnValue(({ mapped: 'azure_embed_result' } as unknown) as EmbedResult),
    mapToTranscribeParams: jest.fn().mockReturnValue({ mapped: 'azure_stt_params' }),
    mapFromTranscribeResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'azure_stt_result' } as unknown) as TranscriptionResult),
    mapToTranslateParams: jest.fn().mockReturnValue({ mapped: 'azure_translate_params' }),
    mapFromTranslateResponse: jest
      .fn()
      .mockReturnValue(({ mapped: 'azure_translate_result' } as unknown) as TranscriptionResult),
    wrapProviderError: jest.fn(err => err)
  }
  // --- End Mock Mapper Implementations ---

  // --- Mock Client Instances ---
  let mockAnthropicClientInstance: any
  let mockGroqClientInstance: any // Add mock for Groq client
  let mockGoogleClientInstance: any // Add mock for Google client
  // Add mocks for other clients if needed in specific tests
  // --- End Mock Client Instances ---

  beforeEach(() => {
    originalEnv = { ...process.env }
    jest.clearAllMocks()
    process.env = {}
    mockPrepareAudioUpload.mockResolvedValue(mockAudioFile)
    warnSpy = jest.spyOn(console, 'warn').mockImplementation() // Mock console.warn

    // Setup mock implementations for the mapper classes
    ;(AnthropicMapper as jest.Mock).mockImplementation(() => mockAnthropicMapperInstance)
    ;(GoogleMapper as jest.Mock).mockImplementation(() => mockGoogleMapperInstance)
    ;(GroqMapper as jest.Mock).mockImplementation(() => mockGroqMapperInstance)
    ;(OpenAIMapper as jest.Mock).mockImplementation(() => mockOpenAIMapperInstance)
    ;(AzureOpenAIMapper as jest.Mock).mockImplementation(() => mockAzureMapperInstance)

    // Setup mock client instances
    mockAnthropicClientInstance = {
      messages: {
        create: jest.fn().mockResolvedValue({ mapped: 'anthropic_mock_response' })
      }
    }
    mockGroqClientInstance = {
      // Add mock for Groq client methods if needed by tests
      models: { list: jest.fn() }, // Mock models.list for listModels test
      chat: { completions: { create: jest.fn().mockResolvedValue({ mapped: 'groq_mock_response' }) } },
      embeddings: { create: jest.fn().mockResolvedValue({ mapped: 'groq_embed_mock_response' }) },
      audio: {
        transcriptions: { create: jest.fn().mockResolvedValue({ mapped: 'groq_stt_mock_response' }) },
        translations: { create: jest.fn().mockResolvedValue({ mapped: 'groq_translate_mock_response' }) }
      }
    }
    mockGoogleClientInstance = {
      getGenerativeModel: jest.fn().mockReturnValue({
        // Mock methods used by generate/stream
        generateContent: jest.fn().mockResolvedValue({ response: { mapped: 'google_mock_response' } }),
        generateContentStream: jest.fn().mockResolvedValue({
          stream: {
            async *[Symbol.asyncIterator]() {
              yield
            }
          }
        }), // Fix ESLint error
        startChat: jest.fn().mockReturnValue({
          sendMessage: jest.fn().mockResolvedValue({ response: { mapped: 'google_chat_response' } }),
          sendMessageStream: jest.fn().mockResolvedValue({
            stream: {
              async *[Symbol.asyncIterator]() {
                yield
              }
            }
          }) // Fix ESLint error
        }),
        // Mock methods used by embed
        embedContent: jest.fn().mockResolvedValue({ mapped: 'google_embed_mock_response' }),
        batchEmbedContents: jest.fn().mockResolvedValue({ mapped: 'google_batch_embed_mock_response' })
      })
    }
    ;(Anthropic as jest.Mock).mockReturnValue(mockAnthropicClientInstance)
    ;(Groq as jest.Mock).mockReturnValue(mockGroqClientInstance) // Mock Groq constructor
    ;(GoogleGenAI as jest.Mock).mockReturnValue(mockGoogleClientInstance) // Mock Google constructor
    // Mock other clients as needed
  })

  afterEach(() => {
    process.env = originalEnv
    warnSpy.mockRestore() // Restore console.warn
  })

  describe('Constructor & Configuration', () => {
    it('should throw ConfigurationError if no keys are provided', () => {
      expect(() => new RosettaAI()).toThrow(ConfigurationError)
    })

    it('should initialize clients and mappers based on env vars', () => {
      process.env.ANTHROPIC_API_KEY = 'env-anthropic-key'
      process.env.OPENAI_API_KEY = 'env-openai-key'

      const rosetta = new RosettaAI()

      expect(rosetta.getConfiguredProviders()).toEqual([Provider.Anthropic, Provider.OpenAI])
      expect(Anthropic).toHaveBeenCalled()
      expect(OpenAI).toHaveBeenCalled()
      expect(AzureOpenAI).not.toHaveBeenCalled()
      expect(AnthropicMapper).toHaveBeenCalled()
      expect(OpenAIMapper).toHaveBeenCalled()
      expect(AzureOpenAIMapper).not.toHaveBeenCalled()
      expect((rosetta as any).mappers.size).toBe(2)
    })

    it('should initialize Azure client and mapper when Azure config is provided', () => {
      const config = {
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.getConfiguredProviders()).toEqual([Provider.OpenAI]) // Only OpenAI provider key
      expect(AzureOpenAI).toHaveBeenCalled()
      expect(OpenAI).not.toHaveBeenCalled()
      expect(AzureOpenAIMapper).toHaveBeenCalledWith(expect.objectContaining(config)) // Check config passed
      expect(OpenAIMapper).not.toHaveBeenCalled()

      // Check that the correct *mock constructor* was called, not instanceof
      // This verifies that the logic inside RosettaAI correctly chose to instantiate AzureOpenAIMapper
      expect(AzureOpenAIMapper).toHaveBeenCalledTimes(1)
      expect(OpenAIMapper).not.toHaveBeenCalled()
      // Optional: Check the instance stored is the one returned by the mock constructor
      expect((rosetta as any).mappers.get(Provider.OpenAI)).toBe(mockAzureMapperInstance)
    })

    it('should prioritize Azure mapper over standard OpenAI mapper', () => {
      const config = {
        openaiApiKey: 'standard-key', // Standard key also present
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.getConfiguredProviders()).toEqual([Provider.OpenAI])
      expect(AzureOpenAI).toHaveBeenCalled()
      expect(OpenAI).not.toHaveBeenCalled() // Standard client not initialized
      expect(AzureOpenAIMapper).toHaveBeenCalled()
      expect(OpenAIMapper).not.toHaveBeenCalled() // Standard mapper not initialized

      // Check that the correct *mock constructor* was called
      expect(AzureOpenAIMapper).toHaveBeenCalledTimes(1)
      expect(OpenAIMapper).not.toHaveBeenCalled()
      // Optional: Check the instance stored is the one returned by the mock constructor
      expect((rosetta as any).mappers.get(Provider.OpenAI)).toBe(mockAzureMapperInstance)
    })

    it('[Easy] should warn if Azure endpoint provided without key', () => {
      // Add another valid provider config to prevent initial error
      new RosettaAI({
        azureOpenAIEndpoint: 'ep',
        azureOpenAIApiVersion: 'v1',
        groqApiKey: 'dummy-key' // Add another provider
      })
      expect(warnSpy).toHaveBeenCalledWith(
        'RosettaAI Warning: Azure OpenAI endpoint provided, but API key is missing or invalid. Azure OpenAI client not initialized.'
      )
    })

    it('[Easy] should warn if Azure key provided without endpoint', () => {
      // Add another valid provider config
      new RosettaAI({
        azureOpenAIApiKey: 'key',
        azureOpenAIApiVersion: 'v1',
        groqApiKey: 'dummy-key' // Add another provider
      })
      expect(warnSpy).toHaveBeenCalledWith(
        'RosettaAI Warning: Azure OpenAI API key provided, but endpoint is missing. Azure OpenAI client not initialized.'
      )
    })

    it('[Easy] should warn if Azure key/endpoint provided without version', () => {
      // Add another valid provider config
      new RosettaAI({
        azureOpenAIApiKey: 'key',
        azureOpenAIEndpoint: 'ep',
        groqApiKey: 'dummy-key' // Add another provider
      })
      expect(warnSpy).toHaveBeenCalledWith(
        'RosettaAI Warning: Azure OpenAI endpoint and key provided, but API version is missing. Azure OpenAI client not initialized.'
      )
    })

    it('should initialize custom providers correctly', () => {
      process.env.MY_CUSTOM_API_KEY = 'env-custom-key'
      const customProviderConfig: CustomProviderConfig[] = [
        {
          providerKey: 'my-custom',
          mapper: MockCustomMapper, // Pass the constructor
          supportedFeatures: ['generate', 'stream'],
          defaultModel: 'custom-model-1',
          // apiKey not provided, should load from env.
          someOtherConfig: 'value1'
        },
        {
          providerKey: 'another-custom',
          mapper: MockCustomMapper,
          supportedFeatures: ['embed'],
          apiKey: 'direct-key', // API key provided directly
          defaultEmbeddingModel: 'custom-embed-1'
        }
      ]
      const rosetta = new RosettaAI({
        openaiApiKey: 'openai-key', // Include a built-in provider
        customProviders: customProviderConfig
      })

      expect(rosetta.getConfiguredProviders()).toEqual(['openai', 'my-custom', 'another-custom'])
      expect((rosetta as any).mappers.get('my-custom')).toBeInstanceOf(MockCustomMapper)
      expect((rosetta as any).mappers.get('another-custom')).toBeInstanceOf(MockCustomMapper)
      // expect((rosetta as any).customProviderConfigs.get('my-custom')).toBe(customProviderConfig[0])
      // expect((rosetta as any).customProviderConfigs.get('another-custom')).toBe(customProviderConfig[1])
      expect((rosetta as any).customApiKeys.get('my-custom')).toBe('env-custom-key')
      expect((rosetta as any).customApiKeys.get('another-custom')).toBe('direct-key')
    })

    it('should warn and skip custom provider with conflicting key', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: Provider.OpenAI, mapper: MockCustomMapper, supportedFeatures: [] } // Conflict with built-in
      ]
      new RosettaAI({
        openaiApiKey: 'openai-key',
        customProviders: customProviderConfig
      })
      expect(warnSpy).toHaveBeenCalledWith(
        "RosettaAI Warning: Skipping custom provider. Key 'openai' conflicts with a built-in provider or another custom provider."
      )
    })

    it('should warn and skip custom provider with invalid mapper constructor', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: 'bad-mapper', mapper: {} as any, supportedFeatures: [] } // Invalid mapper
      ]
      new RosettaAI({
        openaiApiKey: 'openai-key',
        customProviders: customProviderConfig
      })
      expect(warnSpy).toHaveBeenCalledWith(
        "RosettaAI Warning: Skipping custom provider 'bad-mapper'. Invalid mapper constructor provided."
      )
    })

    it('should warn if custom provider API key is missing', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: 'no-key-custom', mapper: MockCustomMapper, supportedFeatures: [] } // No key in config or env
      ]
      new RosettaAI({
        openaiApiKey: 'openai-key',
        customProviders: customProviderConfig
      })
      expect(warnSpy).toHaveBeenCalledWith(
        "RosettaAI Warning: Base URL for custom provider 'no-key-custom' not found in config or environment variable 'NO_KEY_CUSTOM_BASE_URL'. Default model listing will fail if modelListUrl is not also configured."
      )
    })
  })

  describe('getConfiguredProviders', () => {
    it('should return both built-in and custom provider keys', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: 'custom1', mapper: MockCustomMapper, supportedFeatures: [] }
      ]
      const rosetta = new RosettaAI({
        groqApiKey: 'key-groq',
        customProviders: customProviderConfig
      })
      expect(rosetta.getConfiguredProviders()).toEqual([Provider.Groq, 'custom1'])
    })
  })

  describe('getMapper', () => {
    it('should return mapper for custom provider key', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: 'custom1', mapper: MockCustomMapper, supportedFeatures: [] }
      ]
      const rosetta = new RosettaAI({ customProviders: customProviderConfig })
      const mapper = (rosetta as any).getMapper('custom1')
      expect(mapper).toBeInstanceOf(MockCustomMapper)
    })
  })

  describe('getClientForProvider', () => {
    it('should throw error if called with a custom provider key', () => {
      const customProviderConfig: CustomProviderConfig[] = [
        { providerKey: 'custom1', mapper: MockCustomMapper, supportedFeatures: [] }
      ]
      const rosetta = new RosettaAI({ customProviders: customProviderConfig })
      // Expect getClientForProvider to throw when called with a non-Provider enum value
      expect(() => (rosetta as any).getClientForProvider('custom1')).toThrow(RosettaAIError)
      expect(() => (rosetta as any).getClientForProvider('custom1')).toThrow('Unsupported built-in provider: custom1')
    })

    it('should return client for built-in provider', () => {
      const rosetta = new RosettaAI({ groqApiKey: 'key-groq' })
      const client = (rosetta as any).getClientForProvider(Provider.Groq)
      // Check if it's the mocked Groq client instance
      expect(client).toBeDefined()
      expect(Groq).toHaveBeenCalled()
    })
  })

  describe('generate', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any

    beforeEach(() => {
      // Mock the OpenAI client instance specifically for generate tests
      mockOpenAIClientInstance = {
        chat: {
          completions: {
            create: jest.fn().mockResolvedValue({ mapped: 'openai_raw_response' })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      rosetta = new RosettaAI({ openaiApiKey: 'key' })
      // Reset mock calls on the mapper instance
      mockOpenAIMapperInstance.mapToProviderParams.mockClear()
      mockOpenAIMapperInstance.mapFromProviderResponse.mockClear()
      mockOpenAIMapperInstance.wrapProviderError.mockClear()
    })

    it('should pass tools to mapFromProviderResponse', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [mockTool] // Pass the mock tool
      }
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      mockOpenAIMapperInstance.mapFromProviderResponse.mockReturnValue({ mapped: 'openai_result' })

      await rosetta.generate(params)

      expect(mockOpenAIMapperInstance.mapFromProviderResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_response' },
        'gpt-4o-mini',
        [mockTool] // Expect tools to be passed
      )
    })

    it('should parse JSON into parsedContent when structured output is requested', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Return JSON' }],
        responseFormat: {
          type: 'json_schema',
          json_schema: { schema: { type: 'object', properties: { a: { type: 'number' } } } }
        }
      }

      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      mockOpenAIMapperInstance.mapFromProviderResponse.mockReturnValue({
        content: '{"a":1}',
        parsedContent: null,
        finishReason: 'stop',
        model: 'gpt-4o-mini'
      } as unknown as GenerateResult)

      const result = await rosetta.generate(params)
      expect(result.parsedContent).toEqual({ a: 1 })
    })

    it('should throw errors from mapFromProviderResponse (e.g., ToolArgumentValidationError)', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [mockTool]
      }
      ;(Anthropic as jest.Mock).mockImplementation(() => mockAnthropicClientInstance)
      ;(Groq as jest.Mock).mockImplementation(() => mockGroqClientInstance) // Mock Groq constructor
      ;(GoogleGenAI as jest.Mock).mockImplementation(() => mockGoogleClientInstance) // Mock Google constructor
      // Mock other clients as needed
      const validationError = new ToolArgumentValidationError('Invalid args', [], 'get_weather', 'call_1')
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      // Simulate mapFromProviderResponse throwing the validation error
      mockOpenAIMapperInstance.mapFromProviderResponse.mockImplementation(() => {
        throw validationError
      })

      // Wrap the async call in a function for rejects.toThrow
      await expect(async () => rosetta.generate(params)).rejects.toThrow(ToolArgumentValidationError)
      await expect(async () => rosetta.generate(params)).rejects.toThrow('Invalid args')
      expect(mockOpenAIMapperInstance.mapFromProviderResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_response' },
        'gpt-4o-mini',
        [mockTool]
      )
      // Ensure wrapProviderError is NOT called for SDK-specific errors like ToolArgumentValidationError
      // Check the mock on the *mapper instance*
      expect(mockOpenAIMapperInstance.wrapProviderError).not.toHaveBeenCalled()
    })

    it('should get mapper, map params, call client, and map response', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Hi' }]
      }
      const getMapperSpy = jest.spyOn(rosetta as any, 'getMapper')
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')

      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      mockOpenAIMapperInstance.mapFromProviderResponse.mockReturnValue({ mapped: 'openai_result' })

      const result = await rosetta.generate(params)

      expect(getMapperSpy).toHaveBeenCalledWith(Provider.OpenAI)
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI, stream: false }),
        'Generate', // Feature name
        false // isAzure
      )
      expect(mockOpenAIMapperInstance.mapToProviderParams).toHaveBeenCalledWith(
        expect.objectContaining({ provider: Provider.OpenAI, stream: false })
      )
      expect(mockOpenAIClientInstance.chat.completions.create).toHaveBeenCalledWith({
        mapped: 'openai_params',
        model: 'gpt-4o-mini'
      })
      expect(mockOpenAIMapperInstance.mapFromProviderResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_response' },
        'gpt-4o-mini',
        undefined // No tools passed in this specific test
      )
      expect(result).toEqual({
        mapped: 'openai_result',
        rawResponse: { mapped: 'openai_raw_response' }
      }) // From mock mapper return
    })

    it('should use default model if not provided in params', async () => {
      const rosettaWithDefault = new RosettaAI({
        openaiApiKey: 'key',
        defaultModels: { [Provider.OpenAI]: 'default-gpt' }
      })
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        messages: [{ role: 'user', content: 'Hi' }]
        // No model specified
      }
      // Need to get the client instance associated with rosettaWithDefault
      const clientInstance = (OpenAI as jest.Mock).mock.results[1].value // Assuming this is the second instance created
      clientInstance.chat.completions.create.mockResolvedValue({ mapped: 'openai_raw_response' })

      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'default-gpt' })
      mockOpenAIMapperInstance.mapFromProviderResponse.mockReturnValue({ mapped: 'openai_result' })

      await rosettaWithDefault.generate(params)

      expect(mockOpenAIMapperInstance.mapToProviderParams).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'default-gpt' }) // Check effective params passed to mapper
      )
      // Check that the client was called with the mapped params (which include the model from the mapper)
      expect(clientInstance.chat.completions.create).toHaveBeenCalledWith({
        mapped: 'openai_params',
        model: 'default-gpt' // Model comes from the mapped params
      })
    })

    it('should throw ConfigurationError if model is missing (no default)', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        messages: [{ role: 'user', content: 'Hi' }]
        // No model, and assume no default configured
      }
      await expect(rosetta.generate(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        'Model must be specified for provider openai (or set a default).'
      )
    })

    it('should throw error wrapped by the mapper', async () => {
      const apiError = new OpenAI.APIError(400, { message: 'Bad Request' }, 'Error', new Headers())
      mockOpenAIClientInstance.chat.completions.create.mockRejectedValue(apiError)
      const wrappedError = new ProviderAPIError('Wrapped by mapper', Provider.OpenAI, 400)
      mockOpenAIMapperInstance.wrapProviderError.mockReturnValue(wrappedError) // Mock the mapper's wrap function

      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Hi' }]
      }

      await expect(rosetta.generate(params)).rejects.toThrow(ProviderAPIError)
      await expect(rosetta.generate(params)).rejects.toThrow('Wrapped by mapper')
      expect(mockOpenAIMapperInstance.wrapProviderError).toHaveBeenCalledWith(apiError, Provider.OpenAI)
    })

    // --- Tests for checkUnsupportedFeatures ---
    it('[Medium] should NOT throw for image input with Groq', async () => {
      const rosettaGroq = new RosettaAI({ groqApiKey: 'key' })
      const params: GenerateParams = {
        provider: Provider.Groq,
        model: 'meta-llama/llama-4-scout-17b-16e-instruct',
        messages: [{ role: 'user', content: [{ type: 'image', image: {} as any }] }]
      }
      // The generate call should now resolve successfully as image support is enabled
      await expect(rosettaGroq.generate(params)).resolves.toBeDefined()
      await expect(rosettaGroq.generate(params)).resolves.toEqual({
        mapped: 'groq_result',
        rawResponse: { mapped: 'groq_mock_response' }
      })
    })

    it('[Medium] should NOT throw UnsupportedFeatureError for tool use with supported provider (Anthropic)', async () => {
      const rosettaAnt = new RosettaAI({ anthropicApiKey: 'key' })
      const checkUnsupportedSpy = jest.spyOn(rosettaAnt as any, 'checkUnsupportedFeatures')
      const params: GenerateParams = {
        provider: Provider.Anthropic,
        model: 'claude-3',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [mockTool] // Use mockTool
      }

      // Expect the generate call to proceed (and potentially fail later if client mock is incomplete, but not at checkUnsupportedFeatures)
      await expect(rosettaAnt.generate(params)).resolves.toBeDefined() // Or check for the mocked response

      // Verify checkUnsupportedFeatures was called correctly and didn't throw
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.Anthropic,
        expect.objectContaining({ tools: params.tools }),
        'Generate', // Feature name
        false // isAzure
      )
      // Verify the client mock was called (assuming the generate call resolves)
      expect(mockAnthropicClientInstance.messages.create).toHaveBeenCalled()
    })
  })

  describe('stream', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any

    beforeEach(() => {
      mockOpenAIClientInstance = {
        chat: {
          completions: {
            create: jest.fn().mockResolvedValue({
              // Simulate a stream object
              async *[Symbol.asyncIterator]() {
                yield { choices: [{ delta: { content: 'Streamed ' } }] }
                yield { choices: [{ delta: { content: 'Data' } }] }
              }
            })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      rosetta = new RosettaAI({ openaiApiKey: 'key' })
      // Reset mock calls on the mapper instance
      mockOpenAIMapperInstance.mapToProviderParams.mockClear()
      mockOpenAIMapperInstance.mapProviderStream.mockClear()
      mockOpenAIMapperInstance.wrapProviderError.mockClear()
    })

    it('should pass tools to mapProviderStream', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Stream Hi' }],
        tools: [mockTool] // Pass tools
      }
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      mockOpenAIMapperInstance.mapProviderStream.mockImplementation(() =>
        mockStreamGenerator([
          { type: 'message_start', data: { provider: Provider.OpenAI, model: 'gpt-4o-mini' } },
          { type: 'content_delta', data: { delta: 'Mapped Stream' } },
          { type: 'message_stop', data: { finishReason: 'stop' } }
        ])
      )

      const stream = rosetta.stream(params)
      await collectStreamChunks(stream) // Consume the stream

      expect(mockOpenAIMapperInstance.mapProviderStream).toHaveBeenCalledWith(
        expect.any(Object), // The raw provider stream
        params // Expect params to be passed
      )
    })

    it('should yield error chunk if mapProviderStream throws ToolArgumentValidationError', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Stream Hi' }],
        tools: [mockTool]
      }
      const validationError = new ToolArgumentValidationError('Invalid streamed args', [], 'get_weather', 'call_2')
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      // Simulate mapProviderStream throwing the validation error
      mockOpenAIMapperInstance.mapProviderStream.mockImplementation(async function*() {
        yield { type: 'message_start', data: { provider: Provider.OpenAI, model: 'gpt-4o-mini' } }
        yield {
          type: 'tool_call_start',
          data: { index: 0, toolCall: { id: 'call_2', type: 'function', function: { name: 'get_weather' } } }
        }
        // Simulate error during argument processing
        throw validationError
      })
      // Mock wrapProviderError to handle the validation error correctly
      // This mock is on the *RosettaAI* instance's internal method, not the mapper's mock
      // const wrapSpy = jest.spyOn(rosetta as any, 'wrapProviderError').mockImplementation(err => {
      //   if (err instanceof ToolArgumentValidationError) return err // Pass validation error through
      //   return new ProviderAPIError('Wrapped', Provider.OpenAI) // Default wrapping
      // })

      const stream = rosetta.stream(params)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, tool_start, error
      expect(results[2].type).toBe('error')
      // Type guard for error chunk
      if (results[2].type === 'error') {
        expect(results[2].data.error).toBeInstanceOf(ToolArgumentValidationError)
        expect(results[2].data.error.message).toBe(
          "Tool Argument Validation Error for 'get_weather': Invalid streamed args"
        )
      } else {
        fail('Expected error chunk')
      }
      // Ensure RosettaAI.wrapProviderError was called and returned the original validation error
      // expect(wrapSpy).toHaveBeenCalledWith(validationError, Provider.OpenAI)
      // Ensure the *mapper's* wrapProviderError was NOT called because the SDK handled the error directly
      expect(mockOpenAIMapperInstance.wrapProviderError).not.toHaveBeenCalled()

      // wrapSpy.mockRestore() // Clean up the spy
    })

    it('should get mapper, map params, call client stream, and map stream response', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Stream Hi' }]
      }
      const getMapperSpy = jest.spyOn(rosetta as any, 'getMapper')
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      mockOpenAIMapperInstance.mapProviderStream.mockImplementation(() =>
        mockStreamGenerator([
          { type: 'message_start', data: { provider: Provider.OpenAI, model: 'gpt-4o-mini' } },
          { type: 'content_delta', data: { delta: 'Mapped Stream' } },
          { type: 'message_stop', data: { finishReason: 'stop' } }
        ])
      )

      const stream = rosetta.stream(params)
      const results = await collectStreamChunks(stream) // Consume the stream

      expect(getMapperSpy).toHaveBeenCalledWith(Provider.OpenAI)
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI, stream: true }),
        'Generate', // Feature name
        false // isAzure
      )
      expect(mockOpenAIMapperInstance.mapToProviderParams).toHaveBeenCalledWith(
        expect.objectContaining({ provider: Provider.OpenAI, stream: true })
      )
      expect(mockOpenAIClientInstance.chat.completions.create).toHaveBeenCalledWith(
        {
          mapped: 'openai_params',
          model: 'gpt-4o-mini'
        },
        { signal: expect.any(AbortSignal) }
      ) // Check client call args and abort support
      // Check mock calls - should receive the full params object now
      expect(mockOpenAIMapperInstance.mapProviderStream).toHaveBeenCalledWith(
        expect.any(Object), // Raw stream
        params // Expect the full params object
      )
      expect(results).toHaveLength(3)
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Mapped Stream' } })
    })

    it('should yield error chunk if stream setup fails (e.g., missing model)', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        messages: [{ role: 'user', content: 'Fail setup' }]
        // Missing model intentionally
      }

      const stream = rosetta.stream(params)
      const results = await collectStreamChunks(stream) // collectStreamChunks handles setup errors

      expect(results).toHaveLength(1) // Should yield only the error chunk
      expect(results[0].type).toBe('error')
      // Check error type is ConfigurationError
      expect(results[0].data.error).toBeInstanceOf(ConfigurationError)
      expect(results[0].data.error.message).toContain('Model must be specified')
    })

    it('should yield error chunk if client call fails', async () => {
      const apiError = new OpenAI.APIError(500, { message: 'Server Error' }, 'Error', new Headers())
      mockOpenAIClientInstance.chat.completions.create.mockRejectedValue(apiError)
      const wrappedError = new ProviderAPIError('Wrapped', Provider.OpenAI)
      // Mock the wrapProviderError on the *mapper instance*
      mockOpenAIMapperInstance.wrapProviderError.mockReturnValue(wrappedError)

      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Fail client' }]
      }

      const stream = rosetta.stream(params)
      const results = await collectStreamChunks(stream) // collectStreamChunks handles setup errors

      expect(results).toHaveLength(1) // Should yield only the error chunk
      expect(results[0].type).toBe('error')
      expect(results[0].data.error).toBeInstanceOf(ProviderAPIError)
      expect(results[0].data.error.message).toBe('[openai] API Error : Wrapped')
      expect(mockOpenAIMapperInstance.wrapProviderError).toHaveBeenCalledWith(apiError, Provider.OpenAI)
    })

    // --- New Test for Stream Error During Iteration ---
    it('[Hard] should yield error chunk if stream fails during iteration', async () => {
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Stream Hi' }]
      }
      const iterationError = new Error('Stream processing failed')
      const wrappedError = new ProviderAPIError('Wrapped Iteration Error', Provider.OpenAI)
      mockOpenAIMapperInstance.mapToProviderParams.mockReturnValue({ mapped: 'openai_params', model: 'gpt-4o-mini' })
      // Mock the mapper's stream function to throw an error after yielding some chunks
      mockOpenAIMapperInstance.mapProviderStream.mockImplementation(() =>
        mockErrorStreamGenerator(
          [
            { type: 'message_start', data: { provider: Provider.OpenAI, model: 'gpt-4o-mini' } },
            { type: 'content_delta', data: { delta: 'Partial ' } }
          ],
          iterationError // Error to throw
        )
      )
      // Mock the error wrapper to return our specific wrapped error
      mockOpenAIMapperInstance.wrapProviderError.mockReturnValue(wrappedError)

      const stream = rosetta.stream(params)
      const results = await collectStreamChunks(stream) // Consume the stream

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      // Type guard for error chunk
      if (results[2].type === 'error') {
        expect(results[2].data.error).toBe(wrappedError) // Check it's the wrapped error
      } else {
        fail('Expected error chunk')
      }
      expect(mockOpenAIMapperInstance.wrapProviderError).toHaveBeenCalledWith(iterationError, Provider.OpenAI)
    })
    // --- End New Test ---
  })

  describe('embed', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any

    beforeEach(() => {
      mockOpenAIClientInstance = {
        embeddings: {
          create: jest.fn().mockResolvedValue({ mapped: 'openai_raw_embed_response' })
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      rosetta = new RosettaAI({ openaiApiKey: 'key' })
      // Reset mock calls on the mapper instance
      mockOpenAIMapperInstance.mapToEmbedParams.mockClear()
      mockOpenAIMapperInstance.mapFromEmbedResponse.mockClear()
    })

    it('should get mapper, map params, call client, and map response', async () => {
      const params: EmbedParams = {
        provider: Provider.OpenAI,
        model: 'text-embedding-ada-002',
        input: 'Embed me'
      }
      const getMapperSpy = jest.spyOn(rosetta as any, 'getMapper')
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToEmbedParams.mockReturnValue({
        mapped: 'openai_embed_params',
        model: 'text-embedding-ada-002'
      })
      mockOpenAIMapperInstance.mapFromEmbedResponse.mockReturnValue({ mapped: 'openai_embed_result' })

      const result = await rosetta.embed(params)

      expect(getMapperSpy).toHaveBeenCalledWith(Provider.OpenAI)
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI }),
        'Embeddings', // Feature name
        false // isAzure
      )
      expect(mockOpenAIMapperInstance.mapToEmbedParams).toHaveBeenCalledWith(
        expect.objectContaining({ provider: Provider.OpenAI })
      )
      expect(mockOpenAIClientInstance.embeddings.create).toHaveBeenCalledWith({
        mapped: 'openai_embed_params',
        model: 'text-embedding-ada-002'
      })
      expect(mockOpenAIMapperInstance.mapFromEmbedResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_embed_response' },
        'text-embedding-ada-002'
      )
      expect(result).toEqual({ mapped: 'openai_embed_result' }) // From mock mapper return
    })

    it('[Medium] should throw UnsupportedFeatureError for embeddings with Anthropic', async () => {
      // Ensure Anthropic is configured for this test
      const rosettaAnt = new RosettaAI({ anthropicApiKey: 'key' })
      const params: EmbedParams = {
        provider: Provider.Anthropic,
        model: 'some-model', // Model doesn't matter here
        input: 'Embed me'
      }
      await expect(rosettaAnt.embed(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosettaAnt.embed(params)).rejects.toThrow(
        "Provider 'anthropic' does not support the requested feature: Embeddings"
      )
    })
  })

  describe('generateSpeech', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any
    let mockGroqClientInstance: any

    beforeEach(() => {
      mockOpenAIClientInstance = {
        audio: {
          speech: {
            create: jest.fn().mockResolvedValue({ arrayBuffer: async () => Buffer.from('speech') })
          }
        }
      }
      mockGroqClientInstance = {
        audio: {
          speech: {
            create: jest.fn().mockResolvedValue({ arrayBuffer: async () => Buffer.from('groq-speech') })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      ;(Groq as jest.Mock).mockReturnValue(mockGroqClientInstance)

      // Mock the mapToGroqTtsParams function
      ;(GroqAudioMapper.mapToGroqTtsParams as jest.Mock).mockReturnValue({
        model: 'playai-tts',
        input: 'Speak this with Groq',
        voice: 'Fritz-PlayAI',
        response_format: 'wav'
      })

      // Ensure Groq is configured for the tests
      rosetta = new RosettaAI({ openaiApiKey: 'key', groqApiKey: 'groq-key' })
    })

    it('should call client speech create and return buffer for OpenAI', async () => {
      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Speak this',
        voice: 'alloy'
      }
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      const result = await rosetta.generateSpeech(params)

      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI }),
        'Text-to-Speech', // Feature name
        false // isAzure
      )
      expect(mockOpenAIClientInstance.audio.speech.create).toHaveBeenCalledWith({
        model: 'tts-1', // Default model
        input: 'Speak this',
        voice: 'alloy',
        response_format: 'mp3', // Default format
        speed: 1.0 // Default speed
      })
      expect(result).toBeInstanceOf(Buffer)
      expect(result.toString()).toBe('speech')
    })

    it('should call Groq client speech create and return buffer', async () => {
      const params: SpeechParams = {
        provider: Provider.Groq,
        model: 'playai-tts',
        input: 'Speak this with Groq',
        voice: 'Fritz-PlayAI'
      }
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')

      const result = await rosetta.generateSpeech(params)

      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.Groq,
        expect.objectContaining({ provider: Provider.Groq }),
        'Text-to-Speech', // Feature name
        false // isAzure
      )
      expect(GroqAudioMapper.mapToGroqTtsParams).toHaveBeenCalledWith(params)
      expect(mockGroqClientInstance.audio.speech.create).toHaveBeenCalledWith({
        model: 'playai-tts',
        input: 'Speak this with Groq',
        voice: 'Fritz-PlayAI',
        response_format: 'wav'
      })
      expect(result).toBeInstanceOf(Buffer)
      expect(result.toString()).toBe('groq-speech')
    })

    it('[Easy] should use default TTS model for Groq if configured', async () => {
      const rosettaWithDefault = new RosettaAI({
        groqApiKey: 'key-groq',
        defaultTtsModels: { [Provider.Groq]: 'custom-playai-tts' }
      })
      const params: SpeechParams = {
        provider: Provider.Groq,
        input: 'Speak this with Groq default model',
        voice: 'Fritz-PlayAI'
      }

      // Need to get the client instance associated with rosettaWithDefault
      const clientInstance = (Groq as jest.Mock).mock.results[1].value
      clientInstance.audio.speech.create.mockResolvedValue({
        arrayBuffer: async () => Buffer.from('groq-speech-custom')
      })

      // Mock the mapToGroqTtsParams function for this test
      ;(GroqAudioMapper.mapToGroqTtsParams as jest.Mock).mockReturnValue({
        model: 'custom-playai-tts',
        input: 'Speak this with Groq default model',
        voice: 'Fritz-PlayAI',
        response_format: 'wav'
      })

      await rosettaWithDefault.generateSpeech(params)

      expect(GroqAudioMapper.mapToGroqTtsParams).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'custom-playai-tts' })
      )
      expect(clientInstance.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'custom-playai-tts' })
      )
    })

    it('[Easy] should use default TTS model for OpenAI if configured', async () => {
      const rosettaWithDefault = new RosettaAI({
        openaiApiKey: 'key',
        defaultTtsModels: { [Provider.OpenAI]: 'tts-1-hd' }
      })
      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Speak this',
        voice: 'echo'
      }
      // Need to get the client instance associated with rosettaWithDefault
      const clientInstance = (OpenAI as jest.Mock).mock.results[1].value
      clientInstance.audio.speech.create.mockResolvedValue({ arrayBuffer: async () => Buffer.from('speech-hd') })

      await rosettaWithDefault.generateSpeech(params)

      expect(clientInstance.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'tts-1-hd' }) // Check default model used
      )
    })

    it('should throw UnsupportedFeatureError for unsupported provider', async () => {
      const params = { provider: Provider.Anthropic, input: 'Hi', voice: 'a' } as any
      // Wrap async call
      await expect(async () => rosetta.generateSpeech(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(async () => rosetta.generateSpeech(params)).rejects.toThrow(
        "Provider 'anthropic' does not support the requested feature: Text-to-Speech"
      )
    })
  })

  describe('streamSpeech', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any
    let mockStreamBody: any

    beforeEach(() => {
      mockStreamBody = {
        async *[Symbol.asyncIterator]() {
          yield Buffer.from('chunk1')
          yield Buffer.from('chunk2')
        }
      }
      mockOpenAIClientInstance = {
        audio: {
          speech: {
            create: jest.fn().mockResolvedValue({ body: mockStreamBody })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      // Ensure Google is configured for the unsupported test
      rosetta = new RosettaAI({ openaiApiKey: 'key', googleApiKey: 'google-key' })
    })

    it('[Medium] should yield audio chunks for streamSpeech', async () => {
      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Stream audio',
        voice: 'fable'
      }
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      const stream = rosetta.streamSpeech(params)
      // Use the generic collector
      const chunks = await collectStreamChunks(stream)

      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI }),
        'Streaming Text-to-Speech', // Feature name
        false // isAzure
      )
      expect(mockOpenAIClientInstance.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({ input: 'Stream audio', voice: 'fable' }),
        { signal: expect.any(AbortSignal) }
      )
      expect(chunks).toHaveLength(3) // chunk1, chunk2, stop
      expect(chunks[0].type).toBe('audio_chunk')
      // Type guard before accessing data
      if (chunks[0].type === 'audio_chunk') {
        expect(chunks[0].data).toEqual(Buffer.from('chunk1'))
      } else {
        fail('Expected audio_chunk')
      }
      expect(chunks[1].type).toBe('audio_chunk')
      // Type guard before accessing data
      if (chunks[1].type === 'audio_chunk') {
        expect(chunks[1].data).toEqual(Buffer.from('chunk2'))
      } else {
        fail('Expected audio_chunk')
      }
      expect(chunks[2].type).toBe('audio_stop')
    })

    it('[Medium] should yield error chunk if streamSpeech called for unsupported provider', async () => {
      const params = { provider: Provider.Google, input: 'Hi', voice: 'a' } as any
      const stream = rosetta.streamSpeech(params)
      // Use the generic collector
      const chunks = await collectStreamChunks(stream)

      expect(chunks).toHaveLength(1)
      expect(chunks[0].type).toBe('error')
      // Check error type is UnsupportedFeatureError
      // Type guard before accessing data
      if (chunks[0].type === 'error') {
        expect(chunks[0].data.error).toBeInstanceOf(UnsupportedFeatureError)
        expect(chunks[0].data.error.message).toContain('Streaming Text-to-Speech')
      } else {
        fail('Expected error chunk')
      }
    })

    it('[Medium] should yield error chunk if client call fails for streamSpeech', async () => {
      const apiError = new OpenAI.APIError(500, {}, '', new Headers())
      mockOpenAIClientInstance.audio.speech.create.mockRejectedValue(apiError)
      const wrappedError = new ProviderAPIError('Wrapped TTS Error', Provider.OpenAI)
      // Mock the wrapProviderError on the *mapper instance*
      mockOpenAIMapperInstance.wrapProviderError.mockReturnValue(wrappedError)

      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Fail this',
        voice: 'onyx'
      }
      const stream = rosetta.streamSpeech(params)
      // Use the generic collector
      const chunks = await collectStreamChunks(stream)

      expect(chunks).toHaveLength(1)
      expect(chunks[0].type).toBe('error')
      // Type guard before accessing data
      if (chunks[0].type === 'error') {
        expect(chunks[0].data.error).toBe(wrappedError)
      } else {
        fail('Expected error chunk')
      }
      expect(mockOpenAIMapperInstance.wrapProviderError).toHaveBeenCalledWith(apiError, Provider.OpenAI)
    })

    it('[Medium] should yield error chunk if streamSpeech called for Groq', async () => {
      // Create RosettaAI instance with Groq
      const rosettaGroq = new RosettaAI({ groqApiKey: 'key-groq' })

      // Setup test parameters
      const params: SpeechParams = {
        provider: Provider.Groq,
        input: 'Stream this with Groq',
        voice: 'Fritz-PlayAI',
        model: 'playai-tts'
      }

      // Call the method and collect chunks
      const stream = rosettaGroq.streamSpeech(params)
      const chunks = await collectStreamChunks(stream)

      // Verify error chunk
      expect(chunks).toHaveLength(1)
      expect(chunks[0].type).toBe('error')
      // Type guard before accessing data
      if (chunks[0].type === 'error') {
        expect(chunks[0].data.error).toBeInstanceOf(UnsupportedFeatureError)
        expect(chunks[0].data.error.message).toContain('Streaming Text-to-Speech')
      } else {
        fail('Expected error chunk')
      }
    })
  })

  describe('transcribe', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any

    beforeEach(() => {
      mockOpenAIClientInstance = {
        audio: {
          transcriptions: {
            create: jest.fn().mockResolvedValue({ mapped: 'openai_raw_stt_response' })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      // Configure Anthropic for unsupported test
      rosetta = new RosettaAI({ openaiApiKey: 'key', anthropicApiKey: 'ant-key' })
      // Reset mock calls on the mapper instance
      mockOpenAIMapperInstance.mapToTranscribeParams.mockClear()
      mockOpenAIMapperInstance.mapFromTranscribeResponse.mockClear()
    })

    it('should get mapper, prepare upload, map params, call client, map response', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('a'), filename: 'a.mp3', mimeType: 'audio/mpeg' }
      const params: TranscribeParams = {
        provider: Provider.OpenAI,
        model: 'whisper-1',
        audio: audioData
      }
      const getMapperSpy = jest.spyOn(rosetta as any, 'getMapper')
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToTranscribeParams.mockReturnValue({ mapped: 'openai_stt_params' })
      mockOpenAIMapperInstance.mapFromTranscribeResponse.mockReturnValue({ mapped: 'openai_stt_result' })

      const result = await rosetta.transcribe(params)

      expect(getMapperSpy).toHaveBeenCalledWith(Provider.OpenAI)
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI }),
        'Audio Transcription', // Feature name
        false // isAzure
      )
      expect(mockPrepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockOpenAIMapperInstance.mapToTranscribeParams).toHaveBeenCalledWith(
        expect.objectContaining({ provider: Provider.OpenAI }),
        mockAudioFile
      )
      expect(mockOpenAIClientInstance.audio.transcriptions.create).toHaveBeenCalledWith({ mapped: 'openai_stt_params' })
      expect(mockOpenAIMapperInstance.mapFromTranscribeResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_stt_response' },
        'whisper-1'
      )
      expect(result).toEqual({ mapped: 'openai_stt_result' }) // From mock mapper return
    })

    it('[Medium] should throw UnsupportedFeatureError for transcribe with unsupported provider', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('a'), filename: 'a.mp3', mimeType: 'audio/mpeg' }
      const params: TranscribeParams = {
        provider: Provider.Anthropic,
        model: 'model',
        audio: audioData
      }
      await expect(rosetta.transcribe(params)).rejects.toThrow(UnsupportedFeatureError)
      // Check the correct feature name is passed
      await expect(rosetta.transcribe(params)).rejects.toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Transcription"
      )
    })

    it('[Medium] should throw ConfigurationError if transcribe model is missing (no default)', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('a'), filename: 'a.mp3', mimeType: 'audio/mpeg' }
      const params: TranscribeParams = {
        provider: Provider.OpenAI,
        audio: audioData
        // Missing model
      }
      await expect(rosetta.transcribe(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.transcribe(params)).rejects.toThrow(
        'Transcription model must be specified for provider openai (or set a default).'
      )
    })
  })

  describe('translate', () => {
    let rosetta: RosettaAI
    let mockOpenAIClientInstance: any

    beforeEach(() => {
      mockOpenAIClientInstance = {
        audio: {
          translations: {
            create: jest.fn().mockResolvedValue({ mapped: 'openai_raw_translate_response' })
          }
        }
      }
      ;(OpenAI as jest.Mock).mockReturnValue(mockOpenAIClientInstance)
      // Configure Anthropic for unsupported test
      rosetta = new RosettaAI({ openaiApiKey: 'key', anthropicApiKey: 'ant-key' })
      // Reset mock calls on the mapper instance
      mockOpenAIMapperInstance.mapToTranslateParams.mockClear()
      mockOpenAIMapperInstance.mapFromTranslateResponse.mockClear()
    })

    it('should get mapper, prepare upload, map params, call client, map response', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('b'), filename: 'b.wav', mimeType: 'audio/wav' }
      const params: TranslateParams = {
        provider: Provider.OpenAI,
        model: 'whisper-1',
        audio: audioData
      }
      const getMapperSpy = jest.spyOn(rosetta as any, 'getMapper')
      const checkUnsupportedSpy = jest.spyOn(rosetta as any, 'checkUnsupportedFeatures')
      // Setup mock return value for the specific mapper instance used
      mockOpenAIMapperInstance.mapToTranslateParams.mockReturnValue({ mapped: 'openai_translate_params' })
      mockOpenAIMapperInstance.mapFromTranslateResponse.mockReturnValue({ mapped: 'openai_translate_result' })

      const result = await rosetta.translate(params)

      expect(getMapperSpy).toHaveBeenCalledWith(Provider.OpenAI)
      expect(checkUnsupportedSpy).toHaveBeenCalledWith(
        Provider.OpenAI,
        expect.objectContaining({ provider: Provider.OpenAI }),
        'Audio Translation', // Feature name
        false // isAzure
      )
      expect(mockPrepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockOpenAIMapperInstance.mapToTranslateParams).toHaveBeenCalledWith(
        expect.objectContaining({ provider: Provider.OpenAI }),
        mockAudioFile
      )
      expect(mockOpenAIClientInstance.audio.translations.create).toHaveBeenCalledWith({
        mapped: 'openai_translate_params'
      })
      expect(mockOpenAIMapperInstance.mapFromTranslateResponse).toHaveBeenCalledWith(
        { mapped: 'openai_raw_translate_response' },
        'whisper-1'
      )
      expect(result).toEqual({ mapped: 'openai_translate_result' }) // From mock mapper return
    })

    it('[Medium] should throw UnsupportedFeatureError for translate with unsupported provider', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('a'), filename: 'a.mp3', mimeType: 'audio/mpeg' }
      const params: TranslateParams = {
        provider: Provider.Anthropic,
        model: 'model',
        audio: audioData
      }
      await expect(rosetta.translate(params)).rejects.toThrow(UnsupportedFeatureError)
      // Check the correct feature name is passed
      await expect(rosetta.translate(params)).rejects.toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Translation"
      )
    })

    it('[Medium] should throw ConfigurationError if translate model is missing (no default)', async () => {
      const audioData: RosettaAudioData = { data: Buffer.from('a'), filename: 'a.mp3', mimeType: 'audio/mpeg' }
      const params: TranslateParams = {
        provider: Provider.OpenAI,
        audio: audioData
        // Missing model
      }
      await expect(rosetta.translate(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.translate(params)).rejects.toThrow(
        'Translation model must be specified for provider openai (or set a default).'
      )
    })
  })

  describe('wrapProviderError', () => {
    it('should delegate error wrapping to the correct mapper', () => {
      const rosetta = new RosettaAI({ openaiApiKey: 'key', groqApiKey: 'key' })
      const openaiError = new OpenAI.APIError(400, {}, '', new Headers())
      const groqError = new Groq.APIError(401, {}, '', new Headers())

      // Mock the wrapProviderError method on the instances retrieved from the map
      const openaiMapperInstance = (rosetta as any).mappers.get(Provider.OpenAI)
      const groqMapperInstance = (rosetta as any).mappers.get(Provider.Groq)
      ;(rosetta as any).wrapProviderError(openaiError, Provider.OpenAI)
      expect(openaiMapperInstance.wrapProviderError).toHaveBeenCalledWith(openaiError, Provider.OpenAI)
      expect(groqMapperInstance.wrapProviderError).not.toHaveBeenCalled()

      jest.clearAllMocks()
      ;(rosetta as any).wrapProviderError(groqError, Provider.Groq)
      expect(groqMapperInstance.wrapProviderError).toHaveBeenCalledWith(groqError, Provider.Groq)
      expect(openaiMapperInstance.wrapProviderError).not.toHaveBeenCalled()
    })

    it('should handle generic errors if mapper fails or is missing', () => {
      const rosetta = new RosettaAI({ openaiApiKey: 'key' })
      const genericError = new Error('Something failed')
      const mapperInstance = (rosetta as any).mappers.get(Provider.OpenAI)
      // Simulate mapper's wrap function throwing an error
      mapperInstance.wrapProviderError.mockImplementation(() => {
        throw new Error('Mapper wrap failed')
      })

      const wrappedError = (rosetta as any).wrapProviderError(genericError, Provider.OpenAI)

      expect(wrappedError).toBeInstanceOf(ProviderAPIError)
      expect(wrappedError.message).toBe('[openai] API Error : Something failed') // Falls back to generic handling
      // Check provider property after constructor fix
      expect(wrappedError.provider).toBe(Provider.OpenAI)
      expect(wrappedError.underlyingError).toBe(genericError)
    })

    it('[Hard] should handle non-Error object in fallback wrapProviderError', () => {
      const rosetta = new RosettaAI({ openaiApiKey: 'key' })
      const nonError = { detail: 'Failed object' }
      const mapperInstance = (rosetta as any).mappers.get(Provider.OpenAI)
      // Simulate mapper's wrap function throwing an error
      mapperInstance.wrapProviderError.mockImplementation(() => {
        throw new Error('Mapper wrap failed')
      })

      const wrappedError = (rosetta as any).wrapProviderError(nonError, Provider.OpenAI)

      expect(wrappedError).toBeInstanceOf(ProviderAPIError)
      // Expect JSON stringified output due to fix in wrapProviderError
      expect(wrappedError.message).toBe('[openai] API Error : {"detail":"Failed object"}')
      // Check provider property after constructor fix
      expect(wrappedError.provider).toBe(Provider.OpenAI)
      expect(wrappedError.underlyingError).toBe(nonError)
    })
  })

  // Note: getGoogleModel tests skipped - method removed in @google/genai migration
  // New SDK uses centralized API (models.generateContent) instead of model instances
  describe.skip('getGoogleModel (internal)', () => {
    let rosetta: RosettaAI
    let mockGoogleClientInstance: any

    beforeEach(() => {
      mockGoogleClientInstance = {
        getGenerativeModel: jest.fn().mockReturnValue({ mocked: 'google-model-instance' })
      }
      ;(GoogleGenAI as jest.Mock).mockReturnValue(mockGoogleClientInstance)
      rosetta = new RosettaAI({ googleApiKey: 'key' })
    })

    it('[Easy] should call getGenerativeModel with modelId and safety settings', () => {
      const modelInstance = (rosetta as any).getGoogleModel('gemini-pro', undefined)
      expect(mockGoogleClientInstance.getGenerativeModel).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'gemini-pro',
          safetySettings: expect.any(Array)
        }),
        undefined // No request options
      )
      expect(modelInstance).toEqual({ mocked: 'google-model-instance' })
    })

    it('[Medium] should pass apiVersion from providerOptions', () => {
      const providerOptions = { googleApiVersion: 'v1beta' as const }
      ;(rosetta as any).getGoogleModel('gemini-pro', providerOptions)
      expect(mockGoogleClientInstance.getGenerativeModel).toHaveBeenCalledWith(
        expect.any(Object), // model/safety
        { apiVersion: 'v1beta' } // request options
      )
    })

    it('[Medium] should pass apiVersion from global config if not in request options', () => {
      const rosettaWithGlobalOpts = new RosettaAI({
        googleApiKey: 'key',
        providerOptions: { [Provider.Google]: { googleApiVersion: 'v1alpha' as const } }
      })
      ;(rosettaWithGlobalOpts as any).getGoogleModel('gemini-pro', undefined) // No request options passed
      // Need to get the client instance associated with rosettaWithGlobalOpts
      const clientInstance = (GoogleGenAI as jest.Mock).mock.results[1].value
      expect(clientInstance.getGenerativeModel).toHaveBeenCalledWith(
        expect.any(Object), // model/safety
        { apiVersion: 'v1alpha' } // request options from global config
      )
    })

    it('[Medium] should warn if baseURL is provided (not used by SDK)', () => {
      const providerOptions = { baseURL: 'http://custom.google' }
      ;(rosetta as any).getGoogleModel('gemini-pro', providerOptions)
      expect(warnSpy).toHaveBeenCalledWith(
        'Google provider: Custom baseURL provided but not directly used by the @google/generative-ai SDK constructor. Ensure environment variables (like GOOGLE_API_ENDPOINT) are set if needed.'
      )
    })

    it('[Medium] should throw ConfigurationError if Google client not configured', () => {
      const rosettaNoGoogle = new RosettaAI({ openaiApiKey: 'key' }) // No Google key
      expect(() => (rosettaNoGoogle as any).getGoogleModel('gemini-pro', undefined)).toThrow(ConfigurationError)
      expect(() => (rosettaNoGoogle as any).getGoogleModel('gemini-pro', undefined)).toThrow(
        "Provider 'google' client is not configured or initialized."
      )
    })
  })

  // --- Tests for listModels & listAllModels ---
  describe('listModels & listAllModels (with ProviderKey)', () => {
    const mockOpenAIModelList: RosettaModelList = {
      object: 'list',
      data: [{ id: 'gpt-4o-mini', object: 'model', owned_by: 'openai', provider: Provider.OpenAI }]
    }
    const mockGroqModelList: RosettaModelList = {
      object: 'list',
      data: [{ id: 'llama-3.1-8b-instant', object: 'model', owned_by: 'meta', provider: Provider.Groq }]
    }
    const mockAnthropicModelList: RosettaModelList = {
      object: 'list',
      data: [{ id: 'claude-haiku-4-5', object: 'model', owned_by: 'anthropic', provider: Provider.Anthropic }]
    }
    const mockCustomModelList: RosettaModelList = {
      object: 'list',
      data: [{ id: 'custom-model-x', object: 'model', owned_by: 'custom-org', provider: 'my-custom' }]
    }

    const customProviderConfig: CustomProviderConfig[] = [
      {
        providerKey: 'my-custom',
        mapper: MockCustomMapper,
        supportedFeatures: ['generate', 'list_models'], // Supports listing
        modelListUrl: 'https://custom.api/models' // Define a source
      }
    ]

    beforeEach(() => {
      // Reset the mock implementation for listModelsForProvider
      mockListModelsForProvider.mockImplementation(async (provider, config) => {
        if (provider === Provider.OpenAI) return mockOpenAIModelList
        if (provider === Provider.Groq) return mockGroqModelList
        if (provider === Provider.Anthropic) return mockAnthropicModelList
        if (provider === 'my-custom') {
          // Check if customConfig was passed correctly
          expect(config.customConfig).toBeDefined()
          expect(config.customConfig?.providerKey).toBe('my-custom')
          return mockCustomModelList
        }
        throw new ConfigurationError(`Mock: Provider ${provider} not mocked for listModels`)
      })
    })

    describe('listModels', () => {
      it('[Easy] should call internal lister with correct provider and config', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai', groqApiKey: 'key-groq' })
        const result = await rosetta.listModels(Provider.OpenAI)

        expect(mockListModelsForProvider).toHaveBeenCalledTimes(1)
        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          Provider.OpenAI,
          expect.objectContaining({
            apiKey: 'key-openai',
            groqClient: expect.any(Object) // Groq client is initialized even if not used for OpenAI listing
          })
        )
        expect(result).toEqual(mockOpenAIModelList)
      })

      it('[Easy] should pass sourceConfig override to internal lister', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai' })
        const sourceConfig: ModelListingSourceConfig = { type: 'apiEndpoint', url: 'http://custom.url' }
        await rosetta.listModels(Provider.OpenAI, sourceConfig)

        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          Provider.OpenAI,
          expect.objectContaining({
            sourceConfig: sourceConfig,
            apiKey: 'key-openai'
          })
        )
      })

      it('[Easy] should throw ConfigurationError if provider not configured', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai' }) // Only OpenAI configured
        await expect(rosetta.listModels(Provider.Groq)).rejects.toThrow(ConfigurationError)
        await expect(rosetta.listModels(Provider.Groq)).rejects.toThrow(
          "Provider 'groq' is not configured in this RosettaAI instance."
        )
      })

      it('[Medium] should correctly determine API key (prioritize Azure)', async () => {
        const rosettaAzure = new RosettaAI({
          azureOpenAIApiKey: 'key-azure',
          azureOpenAIEndpoint: 'ep',
          azureOpenAIApiVersion: 'v1'
        })
        await rosettaAzure.listModels(Provider.OpenAI)
        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          Provider.OpenAI,
          expect.objectContaining({ apiKey: 'key-azure' }) // Should use Azure key
        )

        const rosettaBoth = new RosettaAI({
          openaiApiKey: 'key-std',
          azureOpenAIApiKey: 'key-azure-2',
          azureOpenAIEndpoint: 'ep2',
          azureOpenAIApiVersion: 'v2'
        })
        await rosettaBoth.listModels(Provider.OpenAI)
        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          Provider.OpenAI,
          expect.objectContaining({ apiKey: 'key-azure-2' }) // Should prioritize Azure key
        )
      })

      it('[Medium] should pass Groq client to internal lister', async () => {
        const rosetta = new RosettaAI({ groqApiKey: 'key-groq' })
        await rosetta.listModels(Provider.Groq)
        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          Provider.Groq,
          expect.objectContaining({
            apiKey: 'key-groq',
            groqClient: expect.any(Object) // Check that the client is passed
          })
        )
      })

      // --- New Custom Provider Tests for listModels ---
      it('[Medium] should call internal lister for custom provider with custom config', async () => {
        process.env.MY_CUSTOM_API_KEY = 'custom-env-key' // Set env key for this test
        const rosetta = new RosettaAI({ customProviders: customProviderConfig })
        const result = await rosetta.listModels('my-custom')

        expect(mockListModelsForProvider).toHaveBeenCalledTimes(1)
        expect(mockListModelsForProvider).toHaveBeenCalledWith(
          'my-custom',
          expect.objectContaining({
            apiKey: 'custom-env-key', // Check correct API key resolution
            customConfig: expect.objectContaining({
              providerKey: 'my-custom',
              modelListUrl: 'https://custom.api/models'
            })
          })
        )
        expect(result).toEqual(mockCustomModelList)
      })

      it('[Medium] should throw UnsupportedFeatureError if custom provider does not support list_models', async () => {
        const customNoList: CustomProviderConfig[] = [
          {
            providerKey: 'custom-no-list',
            mapper: MockCustomMapper,
            supportedFeatures: ['generate'] // Does NOT include list_models
          }
        ]
        const rosetta = new RosettaAI({ customProviders: customNoList })
        await expect(rosetta.listModels('custom-no-list')).rejects.toThrow(UnsupportedFeatureError)
        await expect(rosetta.listModels('custom-no-list')).rejects.toThrow(
          "Provider 'custom-no-list' does not support the requested feature: Model Listing"
        )
      })
      // --- End New Custom Provider Tests ---
    })

    describe('listAllModels', () => {
      it('[Easy] should call listModels for all configured providers', async () => {
        const rosetta = new RosettaAI({
          openaiApiKey: 'key-openai',
          groqApiKey: 'key-groq',
          anthropicApiKey: 'key-ant'
        })
        const listModelsSpy = jest.spyOn(rosetta, 'listModels')

        await rosetta.listAllModels()

        expect(listModelsSpy).toHaveBeenCalledTimes(3)
        expect(listModelsSpy).toHaveBeenCalledWith(Provider.Anthropic)
        expect(listModelsSpy).toHaveBeenCalledWith(Provider.Groq)
        expect(listModelsSpy).toHaveBeenCalledWith(Provider.OpenAI)
      })

      it('[Medium] should return results for successful calls', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai', groqApiKey: 'key-groq' })
        const results = await rosetta.listAllModels()

        expect(results[Provider.OpenAI]).toEqual(mockOpenAIModelList)
        expect(results[Provider.Groq]).toEqual(mockGroqModelList)
        expect(results[Provider.Anthropic]).toBeUndefined() // Not configured
      })

      it('[Medium] should return error objects for failed calls', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai', groqApiKey: 'key-groq' })
        const apiError = new ProviderAPIError('Groq failed', Provider.Groq)
        // Mock listModels to throw for Groq
        mockListModelsForProvider.mockImplementation(async (provider, _config) => {
          if (provider === Provider.OpenAI) return mockOpenAIModelList
          if (provider === Provider.Groq) throw apiError
          throw new Error('Unexpected provider in mock')
        })

        const results = await rosetta.listAllModels()

        expect(results[Provider.OpenAI]).toEqual(mockOpenAIModelList)
        expect(results[Provider.Groq]).toBeInstanceOf(ProviderAPIError)
        expect(results[Provider.Groq]).toEqual(apiError)
      })

      it('[Hard] should handle a mix of success and different error types', async () => {
        const rosetta = new RosettaAI({
          openaiApiKey: 'key-openai',
          groqApiKey: 'key-groq',
          anthropicApiKey: 'key-ant'
        })
        const groqApiError = new ProviderAPIError('Groq API failed', Provider.Groq, 500)
        const antConfigError = new ConfigurationError('Anthropic config issue')

        mockListModelsForProvider.mockImplementation(async (provider, _config) => {
          if (provider === Provider.OpenAI) return mockOpenAIModelList
          if (provider === Provider.Groq) throw groqApiError
          if (provider === Provider.Anthropic) throw antConfigError
          throw new Error('Unexpected provider')
        })

        const results = await rosetta.listAllModels()

        expect(results[Provider.OpenAI]).toEqual(mockOpenAIModelList)
        expect(results[Provider.Groq]).toBe(groqApiError)
        expect(results[Provider.Anthropic]).toBe(antConfigError)
      })

      it('[Hard] should wrap non-RosettaAIError errors', async () => {
        const rosetta = new RosettaAI({ openaiApiKey: 'key-openai', groqApiKey: 'key-groq' })
        const genericError = new Error('Generic failure')

        mockListModelsForProvider.mockImplementation(async (provider, _config) => {
          if (provider === Provider.OpenAI) return mockOpenAIModelList
          if (provider === Provider.Groq) throw genericError // Throw generic error
          throw new Error('Unexpected provider')
        })

        const results = await rosetta.listAllModels()

        expect(results[Provider.OpenAI]).toEqual(mockOpenAIModelList)
        expect(results[Provider.Groq]).toBeInstanceOf(ProviderAPIError)
        expect((results[Provider.Groq] as ProviderAPIError).message).toBe('[groq] API Error : Error: Generic failure')
        expect((results[Provider.Groq] as ProviderAPIError).underlyingError).toBe(genericError)
      })

      // --- New Custom Provider Test for listAllModels ---
      it('[Medium] should include custom provider results/errors in listAllModels', async () => {
        process.env.MY_CUSTOM_API_KEY = 'custom-key'
        const rosetta = new RosettaAI({
          openaiApiKey: 'key-openai',
          customProviders: customProviderConfig // Use the config that supports listing
        })
        const listModelsSpy = jest.spyOn(rosetta, 'listModels')

        const results = await rosetta.listAllModels()

        expect(listModelsSpy).toHaveBeenCalledTimes(2)
        expect(listModelsSpy).toHaveBeenCalledWith(Provider.OpenAI)
        expect(listModelsSpy).toHaveBeenCalledWith('my-custom')

        expect(results[Provider.OpenAI]).toEqual(mockOpenAIModelList)
        expect(results['my-custom']).toEqual(mockCustomModelList)
      })
      // --- End New Custom Provider Test ---
    })
  })
})
