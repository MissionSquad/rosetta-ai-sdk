import {
  RosettaAI,
  Provider,
  ConfigurationError,
  UnsupportedFeatureError,
  ProviderAPIError,
  RosettaAudioData,
  StreamChunk,
  GenerateParams, // Import GenerateParams
  SpeechParams, // Import SpeechParams
  TranscribeParams, // Import TranscribeParams
  EmbedParams, // Import EmbedParams
  // RosettaAIError, // Removed unused import
  MappingError, // Import MappingError
  TranslateParams
} from '../../../src'
import Anthropic from '@anthropic-ai/sdk'
import { GoogleGenerativeAI } from '@google/generative-ai'
import Groq from 'groq-sdk'
import OpenAI, { AzureOpenAI } from 'openai'
import { ChatCompletionChunk } from 'openai/resources/chat/completions' // Import Chunk type

// Mock the underlying SDK clients
jest.mock('@anthropic-ai/sdk')
jest.mock('@google/generative-ai')
jest.mock('groq-sdk')
jest.mock('openai') // Mocks both OpenAI and AzureOpenAI constructors

// Mock the mappers selectively
jest.mock('../../../src/core/mapping/anthropic.mapper')
jest.mock('../../../src/core/mapping/google.mapper')
// jest.mock('../../../src/core/mapping/groq.mapper'); // Keep actual Groq mapper
jest.mock('../../../src/core/mapping/openai.mapper', () => ({
  ...jest.requireActual('../../../src/core/mapping/openai.mapper'), // Keep original parts
  mapOpenAIStream: jest.requireActual('../../../src/core/mapping/openai.mapper').mapOpenAIStream // Use actual stream mapper
}))
jest.mock('../../../src/core/mapping/azure.openai.mapper')

// Import the actual Groq mapper functions needed for the test
// import * as GroqMapper from '../../../src/core/mapping/groq.mapper'; // No longer needed for direct calls

// Mock utility functions
jest.mock('../../../src/core/utils', () => ({
  ...jest.requireActual('../../../src/core/utils'), // Keep original parts if needed
  prepareAudioUpload: jest.fn()
}))
import { prepareAudioUpload } from '../../../src/core/utils'

// Mock implementation for prepareAudioUpload
const mockPrepareAudioUpload = prepareAudioUpload as jest.Mock
const mockAudioFile = { name: 'mock.mp3', type: 'audio/mpeg', [Symbol.toStringTag]: 'File' } // Simulate FileLike structure

describe('RosettaAI Core', () => {
  let originalEnv: NodeJS.ProcessEnv

  beforeEach(() => {
    // Store original environment variables
    originalEnv = { ...process.env }
    // Reset mocks for each test
    jest.clearAllMocks()
    // Reset environment variables for each test
    process.env = {}
    // Reset mock implementations
    mockPrepareAudioUpload.mockResolvedValue(mockAudioFile)
  })

  afterEach(() => {
    // Restore original environment variables
    process.env = originalEnv
  })

  describe('Constructor & Configuration', () => {
    it('should throw ConfigurationError if no keys are provided', () => {
      expect(() => new RosettaAI()).toThrow(ConfigurationError)
      expect(() => new RosettaAI({})).toThrow(ConfigurationError)
    })

    it('should load configuration from environment variables', () => {
      process.env.ANTHROPIC_API_KEY = 'env-anthropic-key'
      process.env.GOOGLE_API_KEY = 'env-google-key'
      process.env.GROQ_API_KEY = 'env-groq-key'
      process.env.OPENAI_API_KEY = 'env-openai-key'
      process.env.ROSETTA_DEFAULT_OPENAI_MODEL = 'env-gpt-model'

      const rosetta = new RosettaAI()

      expect(rosetta.config.anthropicApiKey).toBe('env-anthropic-key')
      expect(rosetta.config.googleApiKey).toBe('env-google-key')
      expect(rosetta.config.groqApiKey).toBe('env-groq-key')
      expect(rosetta.config.openaiApiKey).toBe('env-openai-key')
      expect(rosetta.config.defaultModels?.[Provider.OpenAI]).toBe('env-gpt-model')

      expect(Anthropic).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'env-anthropic-key' }))
      expect(GoogleGenerativeAI).toHaveBeenCalledWith('env-google-key')
      expect(Groq).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'env-groq-key' }))
      expect(OpenAI).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'env-openai-key' }))
      expect(AzureOpenAI).not.toHaveBeenCalled()
    })

    it('should load configuration from constructor arguments', () => {
      const config = {
        anthropicApiKey: 'ctor-anthropic-key',
        googleApiKey: 'ctor-google-key',
        groqApiKey: 'ctor-groq-key',
        openaiApiKey: 'ctor-openai-key',
        defaultModels: {
          [Provider.Google]: 'ctor-gemini-model'
        }
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.config.anthropicApiKey).toBe('ctor-anthropic-key')
      expect(rosetta.config.googleApiKey).toBe('ctor-google-key')
      expect(rosetta.config.groqApiKey).toBe('ctor-groq-key')
      expect(rosetta.config.openaiApiKey).toBe('ctor-openai-key')
      expect(rosetta.config.defaultModels?.[Provider.Google]).toBe('ctor-gemini-model')

      expect(Anthropic).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'ctor-anthropic-key' }))
      expect(GoogleGenerativeAI).toHaveBeenCalledWith('ctor-google-key')
      expect(Groq).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'ctor-groq-key' }))
      expect(OpenAI).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'ctor-openai-key' }))
      expect(AzureOpenAI).not.toHaveBeenCalled()
    })

    it('should prioritize constructor arguments over environment variables', () => {
      process.env.ANTHROPIC_API_KEY = 'env-anthropic-key'
      process.env.ROSETTA_DEFAULT_ANTHROPIC_MODEL = 'env-claude-model'

      const config = {
        anthropicApiKey: 'ctor-anthropic-key',
        defaultModels: {
          [Provider.Anthropic]: 'ctor-claude-model'
        }
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.config.anthropicApiKey).toBe('ctor-anthropic-key')
      expect(rosetta.config.defaultModels?.[Provider.Anthropic]).toBe('ctor-claude-model')
      expect(Anthropic).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'ctor-anthropic-key' }))
    })

    it('should initialize Azure OpenAI client when Azure config is provided', () => {
      const config = {
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview',
        azureOpenAIDefaultChatDeploymentName: 'azure-chat-deploy'
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.config.azureOpenAIApiKey).toBe('azure-key')
      expect(rosetta.config.azureOpenAIEndpoint).toBe('https://azure.endpoint')
      expect(rosetta.config.azureOpenAIApiVersion).toBe('2024-05-01-preview')
      expect(rosetta.config.azureOpenAIDefaultChatDeploymentName).toBe('azure-chat-deploy')

      expect(AzureOpenAI).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'azure-key',
          endpoint: 'https://azure.endpoint',
          apiVersion: '2024-05-01-preview'
        })
      )
      expect(OpenAI).not.toHaveBeenCalled() // Standard OpenAI should not be initialized
    })

    it('should prioritize Azure OpenAI over standard OpenAI if both configured', () => {
      process.env.OPENAI_API_KEY = 'env-openai-key' // Standard key in env
      const config = {
        azureOpenAIApiKey: 'azure-key', // Azure config in constructor
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
      }
      const rosetta = new RosettaAI(config)

      expect(rosetta.config.openaiApiKey).toBe('env-openai-key') // Standard key is still stored
      expect(rosetta.config.azureOpenAIApiKey).toBe('azure-key')

      expect(AzureOpenAI).toHaveBeenCalled() // Azure should be initialized
      expect(OpenAI).not.toHaveBeenCalled() // Standard should NOT be initialized
    })

    it('should warn about partial Azure configuration (missing key)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const config = {
        googleApiKey: 'dummy-google-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
        // Missing azureOpenAIApiKey
      }
      new RosettaAI(config) // Should not throw ConfigurationError now

      expect(AzureOpenAI).not.toHaveBeenCalled()
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Azure OpenAI endpoint provided, but API key is missing')
      )
      warnSpy.mockRestore()
    })

    it('should warn about partial Azure configuration (missing endpoint)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const config = {
        googleApiKey: 'dummy-google-key',
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIApiVersion: '2024-05-01-preview'
        // Missing azureOpenAIEndpoint
      }
      new RosettaAI(config) // Should not throw ConfigurationError now

      expect(AzureOpenAI).not.toHaveBeenCalled()
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Azure OpenAI API key provided, but endpoint is missing')
      )
      warnSpy.mockRestore()
    })

    it('should warn about partial Azure configuration (missing version)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const config = {
        googleApiKey: 'dummy-google-key',
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint'
        // Missing azureOpenAIApiVersion
      }
      new RosettaAI(config) // Should not throw ConfigurationError now

      expect(AzureOpenAI).not.toHaveBeenCalled()
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Azure OpenAI endpoint and key provided, but API version is missing')
      )
      warnSpy.mockRestore()
    })

    // --- NEW TESTS ---
    it('[Easy] should store providerOptions, defaultMaxRetries, defaultTimeoutMs', () => {
      const config = {
        openaiApiKey: 'key',
        providerOptions: {
          [Provider.OpenAI]: { baseURL: 'https://custom.openai.com' }
        },
        defaultMaxRetries: 5,
        defaultTimeoutMs: 90000
      }
      const rosetta = new RosettaAI(config)
      expect(rosetta.config.providerOptions?.[Provider.OpenAI]?.baseURL).toBe('https://custom.openai.com')
      expect(rosetta.config.defaultMaxRetries).toBe(5)
      expect(rosetta.config.defaultTimeoutMs).toBe(90000)
    })

    it('[Easy] should pass baseURL, maxRetries, timeout to client constructors', () => {
      const config = {
        openaiApiKey: 'key',
        anthropicApiKey: 'key',
        groqApiKey: 'key',
        providerOptions: {
          [Provider.OpenAI]: { baseURL: 'https://custom.openai.com' },
          [Provider.Anthropic]: { baseURL: 'https://custom.anthropic.com' },
          [Provider.Groq]: { baseURL: 'https://custom.groq.com' }
        },
        defaultMaxRetries: 3,
        defaultTimeoutMs: 120000
      }
      new RosettaAI(config)
      expect(OpenAI).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://custom.openai.com',
          maxRetries: 3,
          timeout: 120000
        })
      )
      expect(Anthropic).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://custom.anthropic.com',
          maxRetries: 3,
          timeout: 120000
        })
      )
      expect(Groq).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://custom.groq.com',
          maxRetries: 3,
          timeout: 120000
        })
      )
      // Google SDK doesn't take these in constructor
    })
    // --- END NEW TESTS ---
  })

  describe('getConfiguredProviders', () => {
    // Test cases remain the same as before, ensuring they still pass
    it('should return an empty array if no clients are initialized', () => {
      // This scenario is hard to test directly now because the constructor throws
      // if no keys are provided. We rely on the constructor tests.
      // If a test requires bypassing the constructor check, manual instantiation is needed.
      expect(() => new RosettaAI()).toThrow(ConfigurationError)
    })

    it('should return providers for which clients were initialized (standard OpenAI)', () => {
      const rosetta = new RosettaAI({
        anthropicApiKey: 'key1',
        openaiApiKey: 'key2'
      })
      expect(rosetta.getConfiguredProviders()).toEqual([Provider.Anthropic, Provider.OpenAI])
    })

    it('should return providers for which clients were initialized (Azure OpenAI)', () => {
      const rosetta = new RosettaAI({
        googleApiKey: 'key1',
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
      })
      // Provider.OpenAI represents both standard and Azure
      expect(rosetta.getConfiguredProviders()).toEqual([Provider.Google, Provider.OpenAI])
    })

    it('should return all providers if all configured', () => {
      const rosetta = new RosettaAI({
        anthropicApiKey: 'key1',
        googleApiKey: 'key2',
        groqApiKey: 'key3',
        openaiApiKey: 'key4' // Standard OpenAI
      })
      expect(rosetta.getConfiguredProviders()).toEqual([
        Provider.Anthropic,
        Provider.Google,
        Provider.Groq,
        Provider.OpenAI
      ])
    })
  })

  describe('Error Handling', () => {
    it('should throw ConfigurationError if generate called for unconfigured provider', async () => {
      const rosetta = new RosettaAI({ googleApiKey: 'google-key' }) // Only Google configured
      const params: GenerateParams = {
        provider: Provider.OpenAI, // Requesting OpenAI
        model: 'gpt-4o-mini',
        messages: [{ role: 'user' as const, content: 'Hello' }]
      }

      // Expect the internal providerNotConfigured check to throw ConfigurationError
      await expect(rosetta.generate(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        "Provider 'openai' client is not configured or initialized. Check API keys and configuration."
      )
    })

    it('should throw ConfigurationError if generate called without a model (and no default)', async () => {
      const rosetta = new RosettaAI({ openaiApiKey: 'openai-key' }) // OpenAI configured, but no default model
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        // No model specified
        messages: [{ role: 'user' as const, content: 'Hello' }]
      }

      await expect(rosetta.generate(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        'Model must be specified for provider openai (or set a default).'
      )
    })

    it('should throw UnsupportedFeatureError if generate called with unsupported feature', async () => {
      const rosetta = new RosettaAI({ openaiApiKey: 'openai-key' }) // OpenAI configured
      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o-mini',
        messages: [{ role: 'user' as const, content: 'Hello' }],
        thinking: true // Requesting 'thinking' which is Anthropic-specific
      }

      await expect(rosetta.generate(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        "Provider 'openai' does not support the requested feature: Thinking steps"
      )
    })

    it('should throw UnsupportedFeatureError for image input on unsupported provider', async () => {
      const rosetta = new RosettaAI({ groqApiKey: 'groq-key' }) // Groq configured
      const params: GenerateParams = {
        provider: Provider.Groq,
        model: 'llama3-8b-8192',
        messages: [
          {
            role: 'user' as const,
            content: [
              { type: 'text' as const, text: 'What is this?' },
              { type: 'image' as const, image: { mimeType: 'image/png', base64Data: 'abc' } }
            ]
          }
        ]
      }
      await expect(rosetta.generate(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        "Provider 'groq' does not support the requested feature: Image input"
      )
    })

    it('should throw UnsupportedFeatureError for embeddings on unsupported provider', async () => {
      const rosetta = new RosettaAI({ anthropicApiKey: 'anthropic-key' }) // Anthropic configured
      const params: EmbedParams = {
        provider: Provider.Anthropic,
        model: 'claude-3-haiku-20240307', // A valid model, but provider doesn't support embed
        input: 'Embed this text'
      }
      await expect(rosetta.embed(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.embed(params)).rejects.toThrow(
        "Provider 'anthropic' does not support the requested feature: Embeddings"
      )
    })

    // Add similar tests for stream, embed, audio methods checking for config/unsupported errors
  })

  // --- NEW TESTS ---
  describe('generateSpeech', () => {
    // Removed unused mockOpenAIClient variable
    let rosetta: RosettaAI

    beforeEach(() => {
      // Mock the constructor return value for OpenAI
      const mockSpeechCreate = jest.fn()
      ;(OpenAI as jest.Mock).mockImplementation(() => ({
        audio: {
          speech: {
            create: mockSpeechCreate
          }
        }
      }))
      // Mock the constructor return value for AzureOpenAI
      const mockAzureSpeechCreate = jest.fn()
      ;(AzureOpenAI as jest.Mock).mockImplementation(() => ({
        audio: {
          speech: {
            create: mockAzureSpeechCreate
          }
        }
      }))

      rosetta = new RosettaAI({ openaiApiKey: 'fake-key' })
      // Retrieve the instance created by the constructor - NO LONGER NEEDED HERE
    })

    it('should call OpenAI audio speech create with correct params', async () => {
      // Get the mock client instance created by the rosetta instance in beforeEach
      const mockClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value
      const mockResponse = {
        arrayBuffer: jest.fn().mockResolvedValue(Buffer.from('fake audio data')),
        body: null // Simulate non-streaming response body
      }
      mockClient.audio.speech.create.mockResolvedValue(mockResponse)

      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Hello TTS',
        voice: 'nova' as const,
        model: 'tts-1-hd',
        responseFormat: 'mp3' as const,
        speed: 1.1
      }
      const result = await rosetta.generateSpeech(params)

      expect(mockClient.audio.speech.create).toHaveBeenCalledWith({
        model: 'tts-1-hd',
        input: 'Hello TTS',
        voice: 'nova',
        response_format: 'mp3',
        speed: 1.1
      })
      expect(result).toBeInstanceOf(Buffer)
      expect(result.toString()).toBe('fake audio data')
    })

    it('should use default TTS model if not provided', async () => {
      const mockResponse = { arrayBuffer: jest.fn().mockResolvedValue(Buffer.from('d')) }
      const rosettaWithDefault = new RosettaAI({
        openaiApiKey: 'fake-key',
        defaultTtsModels: { [Provider.OpenAI]: 'tts-default' }
      })

      // Get the mock client instance created by this specific RosettaAI instance
      const specificMockClient = ((OpenAI as any) as jest.Mock).mock.results[1]?.value
      specificMockClient.audio.speech.create.mockResolvedValue(mockResponse)

      await rosettaWithDefault.generateSpeech({
        provider: Provider.OpenAI,
        input: 'Test',
        voice: 'echo'
      })

      expect(specificMockClient.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'tts-default' })
      )
    })

    it('should throw UnsupportedFeatureError for non-OpenAI provider', async () => {
      const params = {
        provider: Provider.Groq, // Invalid provider for TTS
        input: 'Hello',
        voice: 'alloy'
      } as any // Cast to bypass type check for test
      await expect(rosetta.generateSpeech(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.generateSpeech(params)).rejects.toThrow(
        "Provider 'groq' does not support the requested feature: Text-to-Speech"
      )
    })

    it('should throw ConfigurationError if OpenAI client is not configured', async () => {
      const rosettaNoOpenAI = new RosettaAI({ groqApiKey: 'fake-groq-key' }) // No OpenAI key
      const params: SpeechParams = {
        provider: Provider.OpenAI,
        input: 'Hello',
        voice: 'alloy'
      }
      await expect(rosettaNoOpenAI.generateSpeech(params)).rejects.toThrow(ConfigurationError)
      await expect(rosettaNoOpenAI.generateSpeech(params)).rejects.toThrow(
        "Provider 'openai' client is not configured or initialized."
      )
    })

    // FIX: Update error expectation
    it('should wrap OpenAI API errors', async () => {
      // Get the mock client instance created by the rosetta instance in beforeEach
      const mockClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value
      const errorPayload = { message: 'Nested bad request', type: 'invalid_request_error', code: 'invalid_input' }
      const apiError = new OpenAI.APIError(400, errorPayload, 'Bad request message from headers?', {})
      mockClient.audio.speech.create.mockRejectedValue(apiError)

      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Test', voice: 'fable' as const }

      await expect(rosetta.generateSpeech(params)).rejects.toThrow(ProviderAPIError)
      // FIX: Check against the actual wrapped message (corrected by wrapProviderError fix)
      await expect(rosetta.generateSpeech(params)).rejects.toThrow(
        '[openai] API Error (Status 400) [Code: invalid_input] : Nested bad request'
      )
    })

    // --- NEW TESTS ---
    it('[Easy] should call Azure audio speech create when Azure is configured', async () => {
      const azureConfig = {
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview'
      }
      const rosettaAzure = new RosettaAI(azureConfig)
      // Get the mock Azure client instance created by this specific RosettaAI instance
      const mockAzureClient = ((AzureOpenAI as any) as jest.Mock).mock.results[0]?.value
      const mockResponse = { arrayBuffer: jest.fn().mockResolvedValue(Buffer.from('azure audio')) }
      mockAzureClient.audio.speech.create.mockResolvedValue(mockResponse)

      const params: SpeechParams = {
        provider: Provider.OpenAI, // Still use OpenAI provider type
        input: 'Hello Azure TTS',
        voice: 'echo' as const
      }
      await rosettaAzure.generateSpeech(params)

      expect(mockAzureClient.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({
          input: 'Hello Azure TTS',
          voice: 'echo',
          model: 'tts-1' // Default model
        })
      )
      // FIX: Removed the problematic assertion expect(mockOpenAIClient).toBeUndefined()
    })
    // --- END NEW TESTS ---
  })

  describe('transcribe', () => {
    let rosetta: RosettaAI
    let mockOpenAIClient: any
    let mockGroqClient: any
    const audioData: RosettaAudioData = {
      data: Buffer.from('fake'),
      filename: 'audio.mp3',
      mimeType: 'audio/mpeg'
    }

    beforeEach(() => {
      // Mock constructors
      const mockOpenAICreate = jest.fn()
      ;(OpenAI as jest.Mock).mockImplementation(() => ({
        audio: { transcriptions: { create: mockOpenAICreate } }
      }))
      const mockGroqCreate = jest.fn()
      ;(Groq as jest.Mock).mockImplementation(() => ({
        audio: { transcriptions: { create: mockGroqCreate } }
      }))
      const mockAzureCreate = jest.fn()
      ;(AzureOpenAI as jest.Mock).mockImplementation(() => ({
        audio: { transcriptions: { create: mockAzureCreate } }
      }))

      rosetta = new RosettaAI({ openaiApiKey: 'fake-openai', groqApiKey: 'fake-groq' })
      // Retrieve instances
      mockOpenAIClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value
      mockGroqClient = ((Groq as any) as jest.Mock).mock.results[0]?.value
    })

    it('should call OpenAI transcriptions create for OpenAI provider', async () => {
      const mockTranscription = { text: 'OpenAI transcription' }
      mockOpenAIClient.audio.transcriptions.create.mockResolvedValue(mockTranscription)
      const params: TranscribeParams = { provider: Provider.OpenAI, audio: audioData, model: 'whisper-1' }

      await rosetta.transcribe(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockOpenAIClient.audio.transcriptions.create).toHaveBeenCalledWith({
        model: 'whisper-1',
        file: mockAudioFile, // Result from mocked prepareAudioUpload
        language: undefined,
        prompt: undefined,
        response_format: 'json',
        timestamp_granularities: undefined
      })
    })

    it('should call Groq transcriptions create for Groq provider', async () => {
      const mockTranscription = { text: 'Groq transcription' }
      mockGroqClient.audio.transcriptions.create.mockResolvedValue(mockTranscription)
      const params: TranscribeParams = { provider: Provider.Groq, audio: audioData, model: 'whisper-large-v3' }
      // Use the actual mapper function to get expected args - REMOVED direct mapper call

      await rosetta.transcribe(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockGroqClient.audio.transcriptions.create).toHaveBeenCalledTimes(1)
      // FIX: Check arguments passed to the mock client method
      expect(mockGroqClient.audio.transcriptions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'whisper-large-v3',
          file: mockAudioFile,
          response_format: 'json' // Default format
        })
      )
    })

    it('should throw UnsupportedFeatureError for Google', async () => {
      const params: TranscribeParams = {
        provider: Provider.Google,
        audio: audioData,
        model: 'some-model',
        language: 'en'
      }
      await expect(rosetta.transcribe(params)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.transcribe(params)).rejects.toThrow(
        "Provider 'google' does not support the requested feature: Audio Transcription"
      )
    })

    it('should throw ConfigurationError if required client is missing (OpenAI)', async () => {
      const rosettaNoOpenAI = new RosettaAI({ groqApiKey: 'fake-groq' })
      const params: TranscribeParams = { provider: Provider.OpenAI, audio: audioData, model: 'whisper-1' }
      await expect(rosettaNoOpenAI.transcribe(params)).rejects.toThrow(ConfigurationError)
    })

    it('should throw ConfigurationError if required client is missing (Groq)', async () => {
      const rosettaNoGroq = new RosettaAI({ openaiApiKey: 'fake-openai' })
      const params: TranscribeParams = { provider: Provider.Groq, audio: audioData, model: 'whisper-large-v3' }
      await expect(rosettaNoGroq.transcribe(params)).rejects.toThrow(ConfigurationError)
    })

    it('should throw ConfigurationError if model is missing', async () => {
      const params = { provider: Provider.OpenAI, audio: audioData } as TranscribeParams // No model
      await expect(rosetta.transcribe(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.transcribe(params)).rejects.toThrow(
        'Transcription model must be specified for provider openai (or set a default).'
      )
    })

    // --- NEW TESTS ---
    it('[Easy] should call Azure transcriptions create when Azure is configured', async () => {
      const azureConfig = {
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview',
        defaultSttModels: { [Provider.OpenAI]: 'azure-whisper' } // Use default for Azure
      }
      const rosettaAzure = new RosettaAI(azureConfig)
      // Get the mock Azure client instance created by this specific RosettaAI instance
      const mockAzureClient = ((AzureOpenAI as any) as jest.Mock).mock.results[0]?.value
      const mockTranscription = { text: 'Azure transcription' }
      mockAzureClient.audio.transcriptions.create.mockResolvedValue(mockTranscription)

      const params: TranscribeParams = {
        provider: Provider.OpenAI, // Still use OpenAI provider type
        audio: audioData
        // Model uses default 'azure-whisper'
      }
      await rosettaAzure.transcribe(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockAzureClient.audio.transcriptions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'azure-whisper',
          file: mockAudioFile
        })
      )
    })

    it('[Medium] should wrap Groq API errors during transcription', async () => {
      const apiError = new Groq.APIError(401, { error: { message: 'Invalid API Key' } }, 'Auth Error', {})
      mockGroqClient.audio.transcriptions.create.mockRejectedValue(apiError)
      const params: TranscribeParams = { provider: Provider.Groq, audio: audioData, model: 'whisper-large-v3' }

      await expect(rosetta.transcribe(params)).rejects.toThrow(ProviderAPIError)
      // FIX: Check against the actual wrapped message (corrected by wrapProviderError fix)
      await expect(rosetta.transcribe(params)).rejects.toThrow('[groq] API Error (Status 401) : Invalid API Key')
    })
    // --- END NEW TESTS ---
  })

  // --- NEW TESTS ---
  describe('translate', () => {
    let rosetta: RosettaAI
    let mockOpenAIClient: any
    let mockGroqClient: any
    const audioData: RosettaAudioData = {
      data: Buffer.from('fake'),
      filename: 'audio.mp3',
      mimeType: 'audio/mpeg'
    }

    beforeEach(() => {
      // Mock constructors
      const mockOpenAICreate = jest.fn()
      ;(OpenAI as jest.Mock).mockImplementation(() => ({
        audio: { translations: { create: mockOpenAICreate } }
      }))
      const mockGroqCreate = jest.fn()
      ;(Groq as jest.Mock).mockImplementation(() => ({
        audio: { translations: { create: mockGroqCreate } }
      }))
      const mockAzureCreate = jest.fn()
      ;(AzureOpenAI as jest.Mock).mockImplementation(() => ({
        audio: { translations: { create: mockAzureCreate } }
      }))

      rosetta = new RosettaAI({ openaiApiKey: 'fake-openai', groqApiKey: 'fake-groq' })
      // Retrieve instances
      mockOpenAIClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value
      mockGroqClient = ((Groq as any) as jest.Mock).mock.results[0]?.value
    })

    it('[Easy] should call OpenAI translations create for OpenAI provider', async () => {
      const mockTranslation = { text: 'OpenAI translation' }
      mockOpenAIClient.audio.translations.create.mockResolvedValue(mockTranslation)
      const params: TranslateParams = { provider: Provider.OpenAI, audio: audioData, model: 'whisper-1' }

      await rosetta.translate(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockOpenAIClient.audio.translations.create).toHaveBeenCalledWith({
        model: 'whisper-1',
        file: mockAudioFile,
        prompt: undefined,
        response_format: 'json'
      })
    })

    it('[Easy] should call Groq translations create for Groq provider', async () => {
      const mockTranslation = { text: 'Groq translation' }
      mockGroqClient.audio.translations.create.mockResolvedValue(mockTranslation)
      const params: TranslateParams = { provider: Provider.Groq, audio: audioData, model: 'whisper-large-v3' }
      // Use the actual mapper function to get expected args - REMOVED direct mapper call

      await rosetta.translate(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      // FIX: Check arguments passed to the mock client method
      expect(mockGroqClient.audio.translations.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'whisper-large-v3',
          file: mockAudioFile,
          response_format: 'json' // Default format
        })
      )
    })

    it('[Easy] should call Azure translations create when Azure is configured', async () => {
      const azureConfig = {
        azureOpenAIApiKey: 'azure-key',
        azureOpenAIEndpoint: 'https://azure.endpoint',
        azureOpenAIApiVersion: '2024-05-01-preview',
        defaultSttModels: { [Provider.OpenAI]: 'azure-whisper-trans' }
      }
      const rosettaAzure = new RosettaAI(azureConfig)
      // Get the mock Azure client instance created by this specific RosettaAI instance
      const mockAzureClient = ((AzureOpenAI as any) as jest.Mock).mock.results[0]?.value
      const mockTranslation = { text: 'Azure translation' }
      mockAzureClient.audio.translations.create.mockResolvedValue(mockTranslation)

      const params: TranslateParams = {
        provider: Provider.OpenAI,
        audio: audioData
        // Model uses default 'azure-whisper-trans'
      }
      await rosettaAzure.translate(params)

      expect(prepareAudioUpload).toHaveBeenCalledWith(audioData)
      expect(mockAzureClient.audio.translations.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'azure-whisper-trans',
          file: mockAudioFile
        })
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for Google/Anthropic', async () => {
      const paramsGoogle: TranslateParams = { provider: Provider.Google, audio: audioData, model: 'some-model' }
      const paramsAnthropic: TranslateParams = { provider: Provider.Anthropic, audio: audioData, model: 'some-model' }
      await expect(rosetta.translate(paramsGoogle)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.translate(paramsAnthropic)).rejects.toThrow(UnsupportedFeatureError)
      await expect(rosetta.translate(paramsGoogle)).rejects.toThrow(
        "Provider 'google' does not support the requested feature: Audio Translation"
      )
    })

    it('[Easy] should throw ConfigurationError if required client is missing', async () => {
      const rosettaNoClients = new RosettaAI({ googleApiKey: 'g-key' }) // No OpenAI or Groq
      const paramsOpenAI: TranslateParams = { provider: Provider.OpenAI, audio: audioData, model: 'whisper-1' }
      const paramsGroq: TranslateParams = { provider: Provider.Groq, audio: audioData, model: 'whisper-large-v3' }
      await expect(rosettaNoClients.translate(paramsOpenAI)).rejects.toThrow(ConfigurationError)
      await expect(rosettaNoClients.translate(paramsGroq)).rejects.toThrow(ConfigurationError)
    })

    it('[Easy] should throw ConfigurationError if model is missing', async () => {
      const params = { provider: Provider.OpenAI, audio: audioData } as TranslateParams // No model
      await expect(rosetta.translate(params)).rejects.toThrow(ConfigurationError)
      await expect(rosetta.translate(params)).rejects.toThrow(
        'Translation model must be specified for provider openai (or set a default).'
      )
    })

    it('[Medium] should wrap OpenAI API errors during translation', async () => {
      const apiError = new OpenAI.APIError(429, { message: 'Rate limited' }, 'Rate limit', {})
      mockOpenAIClient.audio.translations.create.mockRejectedValue(apiError)
      const params: TranslateParams = { provider: Provider.OpenAI, audio: audioData, model: 'whisper-1' }

      await expect(rosetta.translate(params)).rejects.toThrow(ProviderAPIError)
      // FIX: Check against the actual wrapped message (corrected by wrapProviderError fix)
      await expect(rosetta.translate(params)).rejects.toThrow('[openai] API Error (Status 429) : Rate limited')
    })
  })
  // --- END NEW TESTS ---

  // --- NEW TESTS ---
  describe('streamSpeech', () => {
    let rosetta: RosettaAI
    let mockAzureClient: any

    // Helper to create a mock ReadableStream for audio chunks
    async function* createMockAudioStream(dataChunks: Buffer[]): AsyncGenerator<Uint8Array, void, undefined> {
      for (const chunk of dataChunks) {
        await new Promise(resolve => setTimeout(resolve, 1))
        yield new Uint8Array(chunk)
      }
    }

    // Helper to collect stream chunks
    async function collectAudioStreamChunks(stream: AsyncIterable<any>): Promise<any[]> {
      const chunks: any[] = []
      try {
        for await (const chunk of stream) {
          chunks.push(chunk)
        }
      } catch (error) {
        // If the stream setup itself throws, catch it here for tests that expect setup errors
        if (error instanceof Error) {
          chunks.push({ type: 'error', data: { error } }) // Push an error chunk representation
        } else {
          chunks.push({ type: 'error', data: { error: new Error(String(error)) } })
        }
      }
      return chunks
    }

    beforeEach(() => {
      // Mock constructors to return instances with mocked methods
      const mockSpeechCreate = jest.fn()
      ;(OpenAI as jest.Mock).mockImplementation(() => ({
        audio: { speech: { create: mockSpeechCreate } }
      }))
      const mockAzureSpeechCreate = jest.fn()
      ;(AzureOpenAI as jest.Mock).mockImplementation(() => ({
        audio: { speech: { create: mockAzureSpeechCreate } }
      }))

      // Configure both standard and Azure for different tests
      rosetta = new RosettaAI({
        openaiApiKey: 'fake-openai-key',
        azureOpenAIApiKey: 'fake-azure-key',
        azureOpenAIEndpoint: 'https://fake-azure.openai.azure.com/',
        azureOpenAIApiVersion: '2024-05-01-preview'
      })
      // Retrieve the mock instances created by the constructor
      // mockOpenAIClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value // Standard client (if created)
      mockAzureClient = ((AzureOpenAI as any) as jest.Mock).mock.results[0]?.value // Azure client (prioritized)
    })

    it('[Easy] should stream audio chunks from OpenAI', async () => {
      // Re-initialize RosettaAI with only standard OpenAI for this test
      const rosettaStd = new RosettaAI({ openaiApiKey: 'fake-openai-key' })
      // FIX: Get the correct mock client instance associated with rosettaStd
      // The mock results array grows with each `new RosettaAI` call.
      // Assuming the previous tests created 2 instances (one in outer beforeEach, one in azure test), this is the 3rd.
      const mockStdClient = (OpenAI as jest.Mock).mock.results[1]?.value // Get the new instance

      // FIX: Ensure mockStdClient is defined before accessing properties
      if (!mockStdClient?.audio?.speech?.create) {
        throw new Error('Mock OpenAI client for rosettaStd was not initialized correctly in the test.')
      }

      const audioChunks = [Buffer.from('chunk1'), Buffer.from('chunk2')]
      const mockResponse = {
        body: createMockAudioStream(audioChunks),
        arrayBuffer: jest.fn() // Mock arrayBuffer even if not used by stream
      }
      // FIX: Use the correctly captured mockStdClient
      mockStdClient.audio.speech.create.mockResolvedValue(mockResponse)

      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Stream me', voice: 'alloy' }
      const stream = rosettaStd.streamSpeech(params)
      const results = await collectAudioStreamChunks(stream)

      expect(mockStdClient.audio.speech.create).toHaveBeenCalledWith(expect.objectContaining({ input: 'Stream me' }))
      expect(results).toHaveLength(3) // chunk, chunk, stop
      expect(results[0]).toEqual({ type: 'audio_chunk', data: audioChunks[0] })
      expect(results[1]).toEqual({ type: 'audio_chunk', data: audioChunks[1] })
      expect(results[2]).toEqual({ type: 'audio_stop' })
    })

    it('[Easy] should stream audio chunks from Azure', async () => {
      const audioChunks = [Buffer.from('azure1'), Buffer.from('azure2')]
      const mockResponse = {
        body: createMockAudioStream(audioChunks),
        arrayBuffer: jest.fn()
      }
      mockAzureClient.audio.speech.create.mockResolvedValue(mockResponse)

      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Stream Azure', voice: 'fable' }
      const stream = rosetta.streamSpeech(params) // Uses Azure client due to constructor config
      const results = await collectAudioStreamChunks(stream)

      expect(mockAzureClient.audio.speech.create).toHaveBeenCalledWith(
        expect.objectContaining({ input: 'Stream Azure' })
      )
      expect(results).toHaveLength(3) // chunk, chunk, stop
      expect(results[0]).toEqual({ type: 'audio_chunk', data: audioChunks[0] })
      expect(results[1]).toEqual({ type: 'audio_chunk', data: audioChunks[1] })
      expect(results[2]).toEqual({ type: 'audio_stop' })
    })

    it('[Easy] should yield error chunk for unsupported provider', async () => {
      const params = { provider: Provider.Groq, input: 'Test', voice: 'echo' } as any
      const stream = rosetta.streamSpeech(params)
      // FIX: Use try/catch to check the yielded error
      const iterator = stream[Symbol.asyncIterator]()
      const firstChunk = await iterator.next()

      expect(firstChunk.done).toBe(false)
      expect(firstChunk.value.type).toBe('error')
      expect(firstChunk.value.data.error).toBeInstanceOf(UnsupportedFeatureError)
      expect(firstChunk.value.data.error.message).toContain('Streaming Text-to-Speech')

      // Check that the underlying error was also thrown (as per stream contract)
      await expect(iterator.next()).rejects.toThrow(UnsupportedFeatureError)
    })

    it('[Easy] should yield error chunk if client not configured', async () => {
      const rosettaNoAudio = new RosettaAI({ googleApiKey: 'g-key' }) // No OpenAI/Azure
      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Test', voice: 'echo' }
      const stream = rosettaNoAudio.streamSpeech(params)
      // FIX: Use try/catch to check the yielded error
      const iterator = stream[Symbol.asyncIterator]()
      const firstChunk = await iterator.next()

      expect(firstChunk.done).toBe(false)
      expect(firstChunk.value.type).toBe('error')
      expect(firstChunk.value.data.error).toBeInstanceOf(ConfigurationError)

      // Check that the underlying error was also thrown
      await expect(iterator.next()).rejects.toThrow(ConfigurationError)
    })

    it('[Medium] should yield error chunk if stream setup fails', async () => {
      const apiError = new OpenAI.APIError(400, { message: 'Invalid voice' }, 'Bad Request', {})
      mockAzureClient.audio.speech.create.mockRejectedValue(apiError) // Error during setup

      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Fail setup', voice: 'invalid' }
      const stream = rosetta.streamSpeech(params)
      // FIX: Use try/catch to check the yielded error
      const iterator = stream[Symbol.asyncIterator]()
      const firstChunk = await iterator.next()

      expect(firstChunk.done).toBe(false)
      expect(firstChunk.value.type).toBe('error')
      expect(firstChunk.value.data.error).toBeInstanceOf(ProviderAPIError)
      // FIX: Check against the actual wrapped message (corrected by wrapProviderError fix)
      expect(firstChunk.value.data.error.message).toContain('[openai] API Error (Status 400) : Invalid voice')

      // Check that the underlying error was also thrown
      await expect(iterator.next()).rejects.toThrow(ProviderAPIError)
    })

    it('[Medium] should yield error chunk if error occurs mid-stream', async () => {
      const streamError = new Error('Network connection lost')
      async function* errorMidStream(): AsyncGenerator<Uint8Array, void, undefined> {
        yield new Uint8Array(Buffer.from('part1'))
        await new Promise(resolve => setTimeout(resolve, 1))
        throw streamError // Error after first chunk
      }
      const mockResponse = {
        body: errorMidStream(),
        arrayBuffer: jest.fn()
      }
      mockAzureClient.audio.speech.create.mockResolvedValue(mockResponse)

      const params: SpeechParams = { provider: Provider.OpenAI, input: 'Fail mid stream', voice: 'nova' }
      const stream = rosetta.streamSpeech(params)
      const results = await collectAudioStreamChunks(stream) // collect helper handles mid-stream errors

      expect(results).toHaveLength(2) // chunk, error
      expect(results[0]).toEqual({ type: 'audio_chunk', data: Buffer.from('part1') })
      expect(results[1].type).toBe('error')
      expect(results[1].data.error).toBeInstanceOf(ProviderAPIError) // Wrapped error
      expect(results[1].data.error.message).toContain('Network connection lost')
    })
  })
  // --- END NEW TESTS ---

  // --- NEW TESTS ---
  describe('wrapProviderError', () => {
    let rosetta: RosettaAI

    beforeEach(() => {
      // Need an instance to call the private method via a public one
      rosetta = new RosettaAI({ googleApiKey: 'dummy' })
    })

    // Helper to access the private method
    const callWrapError = (error: unknown, provider: Provider) => {
      try {
        // Simulate calling it internally
        throw (rosetta as any).wrapProviderError(error, provider)
      } catch (e) {
        return e
      }
    }

    it('[Easy] should wrap Anthropic.APIError', () => {
      const underlying = new Anthropic.APIError(
        429,
        { error: { type: 'rate_limit_error', message: 'Limit exceeded' } },
        'Rate Limit',
        {}
      )
      const wrapped = callWrapError(underlying, Provider.Anthropic)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Anthropic)
      // FIX: Check status code after fix
      expect(wrapped.statusCode).toBe(429)
      expect(wrapped.errorCode).toBe('rate_limit_error')
      expect(wrapped.errorType).toBe('rate_limit_error')
      expect(wrapped.message).toContain('Limit exceeded')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap Groq.APIError', () => {
      const underlying = new Groq.APIError(
        401,
        { error: { message: 'Bad key', code: 'auth_failed', type: 'invalid_request' } }, // Added type
        'Auth Error',
        {}
      )
      const wrapped = callWrapError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      // FIX: Check status code, code, type after fix
      expect(wrapped.statusCode).toBe(401)
      expect(wrapped.errorCode).toBe('auth_failed')
      expect(wrapped.errorType).toBe('invalid_request')
      expect(wrapped.message).toContain('Bad key')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap OpenAI.APIError (with nested message)', () => {
      const underlying = new OpenAI.APIError(
        400,
        { message: 'Invalid input param', code: 'invalid_input', type: 'validation_error' }, // Added code/type
        'Bad Request',
        {}
      )
      const wrapped = callWrapError(underlying, Provider.OpenAI)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.OpenAI)
      // FIX: Check status, code, type after fix
      expect(wrapped.statusCode).toBe(400)
      expect(wrapped.errorCode).toBe('invalid_input')
      expect(wrapped.errorType).toBe('validation_error')
      expect(wrapped.message).toContain('Invalid input param') // Uses nested message
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap OpenAI.APIError (with direct message)', () => {
      // Simulate error where nested message is missing/empty
      const underlying = new OpenAI.APIError(503, undefined, 'Service Unavailable', {}) // Use undefined instead of null
      const wrapped = callWrapError(underlying, Provider.OpenAI)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.OpenAI)
      // FIX: Check status code after fix
      expect(wrapped.statusCode).toBe(503)
      expect(wrapped.message).toContain('Service Unavailable') // Uses direct message
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap simulated Google Error', () => {
      const underlying = {
        message: 'Permission denied',
        status: 403, // Simulate status
        errorDetails: [{ reason: 'PERMISSION_DENIED' }]
      }
      const wrapped = callWrapError(underlying, Provider.Google)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Google)
      expect(wrapped.statusCode).toBe(403)
      expect(wrapped.errorCode).toBe('PERMISSION_DENIED')
      expect(wrapped.message).toContain('Permission denied')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap generic Error', () => {
      const underlying = new Error('Generic network failure')
      const wrapped = callWrapError(underlying, Provider.OpenAI)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.OpenAI)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Generic network failure')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap unknown/string error', () => {
      const underlying = 'Something went wrong'
      const wrapped = callWrapError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Something went wrong')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should not re-wrap RosettaAIError', () => {
      const underlying = new MappingError('Already mapped', Provider.Anthropic)
      const wrapped = callWrapError(underlying, Provider.Anthropic)
      expect(wrapped).toBe(underlying) // Should return the original error instance
      expect(wrapped).toBeInstanceOf(MappingError)
      expect(wrapped).not.toBeInstanceOf(ProviderAPIError)
    })
  })
  // --- END NEW TESTS ---

  // --- NEW TESTS ---
  describe('getGoogleModel', () => {
    let rosetta: RosettaAI
    let mockGoogleClient: any

    beforeEach(() => {
      // Mock constructor
      const mockGetModel = jest.fn().mockReturnValue({})
      ;(GoogleGenerativeAI as jest.Mock).mockImplementation(() => ({
        getGenerativeModel: mockGetModel
      }))

      rosetta = new RosettaAI({ googleApiKey: 'dummy-key' })
      // Retrieve instance
      mockGoogleClient = ((GoogleGenerativeAI as any) as jest.Mock).mock.results[0]?.value
    })

    it('[Easy] should call getGenerativeModel with modelId and default safety', () => {
      ;(rosetta as any).getGoogleModel('gemini-pro')
      expect(mockGoogleClient.getGenerativeModel).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'gemini-pro', safetySettings: expect.any(Array) }),
        undefined // No request options
      )
      const safetySettings = mockGoogleClient.getGenerativeModel.mock.calls[0][0].safetySettings
      expect(safetySettings).toHaveLength(4)
    })

    it('[Easy] should pass apiVersion from providerOptions', () => {
      ;(rosetta as any).getGoogleModel('gemini-pro', { googleApiVersion: 'v1beta' })
      expect(mockGoogleClient.getGenerativeModel).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'gemini-pro' }),
        { apiVersion: 'v1beta' } // Request options with apiVersion
      )
    })

    it('[Easy] should pass apiVersion from global config', () => {
      const rosettaWithVersion = new RosettaAI({
        googleApiKey: 'dummy-key',
        providerOptions: { [Provider.Google]: { googleApiVersion: 'v1alpha' } }
      })
      // Get the new mock instance
      const mockClientInstance = ((GoogleGenerativeAI as any) as jest.Mock).mock.results[1]?.value
      ;(rosettaWithVersion as any).getGoogleModel('gemini-pro')
      expect(mockClientInstance.getGenerativeModel).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'gemini-pro' }),
        { apiVersion: 'v1alpha' }
      )
    })

    it('[Easy] should warn when baseURL is provided in options', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      ;(rosetta as any).getGoogleModel('gemini-pro', { baseURL: 'https://custom.google.com' })
      expect(mockGoogleClient.getGenerativeModel).toHaveBeenCalled() // Still calls the method
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Custom baseURL provided but not directly used by the @google/generative-ai SDK')
      )
      warnSpy.mockRestore()
    })
  })
  // --- END NEW TESTS ---

  // --- NEW STREAMING TEST SUITE ---
  describe('stream', () => {
    let rosetta: RosettaAI
    let mockOpenAIClient: any

    beforeEach(() => {
      // Mock constructor
      const mockCreate = jest.fn()
      ;(OpenAI as jest.Mock).mockImplementation(() => ({
        chat: { completions: { create: mockCreate } }
      }))

      rosetta = new RosettaAI({ openaiApiKey: 'fake-key' })
      // Retrieve instance
      mockOpenAIClient = ((OpenAI as any) as jest.Mock).mock.results[0]?.value
    })

    // Helper to create a mock OpenAI stream
    async function* createMockOpenAIStreamGenerator(
      chunks: ChatCompletionChunk[]
    ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
      for (const chunk of chunks) {
        // Simulate network delay slightly
        await new Promise(resolve => setTimeout(resolve, 1))
        yield chunk
      }
    }

    it('should process and map OpenAI stream chunks correctly', async () => {
      const modelId = 'gpt-4o-mini-stream'
      const mockStreamChunks: ChatCompletionChunk[] = [
        {
          id: 'chatcmpl-stream-1',
          choices: [{ delta: { role: 'assistant', content: '' }, finish_reason: null, index: 0, logprobs: null }],
          created: 1700000000,
          model: modelId,
          object: 'chat.completion.chunk',
          system_fingerprint: 'fp_abc'
        },
        {
          id: 'chatcmpl-stream-1',
          choices: [{ delta: { content: 'Hello' }, finish_reason: null, index: 0, logprobs: null }],
          created: 1700000001,
          model: modelId,
          object: 'chat.completion.chunk',
          system_fingerprint: 'fp_abc'
        },
        {
          id: 'chatcmpl-stream-1',
          choices: [{ delta: { content: ' World' }, finish_reason: null, index: 0, logprobs: null }],
          created: 1700000002,
          model: modelId,
          object: 'chat.completion.chunk',
          system_fingerprint: 'fp_abc'
        },
        {
          id: 'chatcmpl-stream-1',
          choices: [{ delta: {}, finish_reason: 'stop', index: 0, logprobs: null }], // Stop chunk
          created: 1700000003,
          model: modelId,
          object: 'chat.completion.chunk',
          system_fingerprint: 'fp_abc'
        },
        // Usage chunk (added via stream_options)
        {
          id: 'chatcmpl-stream-1',
          choices: [],
          created: 1700000004,
          model: modelId,
          object: 'chat.completion.chunk',
          usage: { completion_tokens: 2, prompt_tokens: 5, total_tokens: 7 }
        }
      ]

      const mockStream = {
        async *[Symbol.asyncIterator]() {
          yield* createMockOpenAIStreamGenerator(mockStreamChunks)
        }
      }
      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockStream)

      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: modelId,
        messages: [{ role: 'user' as const, content: 'Say hello' }]
      }

      const receivedChunks: StreamChunk[] = []
      const stream = rosetta.stream(params)

      for await (const chunk of stream) {
        receivedChunks.push(chunk)
      }

      // Assertions
      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          stream: true,
          stream_options: { include_usage: true }
        })
      )

      expect(receivedChunks).toHaveLength(6) // Start, Delta, Delta, Stop, Usage, FinalResult

      // Check specific chunk types and data
      expect(receivedChunks[0]).toEqual({ type: 'message_start', data: { provider: Provider.OpenAI, model: modelId } })
      expect(receivedChunks[1]).toEqual({ type: 'content_delta', data: { delta: 'Hello' } })
      expect(receivedChunks[2]).toEqual({ type: 'content_delta', data: { delta: ' World' } })
      expect(receivedChunks[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(receivedChunks[4]).toEqual({
        type: 'final_usage',
        data: { usage: { completionTokens: 2, promptTokens: 5, totalTokens: 7 } }
      })
      // Check the final aggregated result
      expect(receivedChunks[5].type).toBe('final_result')
      expect((receivedChunks[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Hello World',
          finishReason: 'stop',
          model: modelId,
          usage: { completionTokens: 2, promptTokens: 5, totalTokens: 7 }
        })
      )
    })

    // FIX: Update error expectation
    it('should handle stream errors correctly', async () => {
      const errorPayload = { message: 'Server error details' } // Nested error object
      const errorMessage = 'Server error' // Direct message property
      const apiError = new OpenAI.APIError(500, errorPayload, errorMessage, {})

      // Mock the SDK stream to throw an error
      async function* createErrorStreamGenerator(): AsyncGenerator<ChatCompletionChunk, void, undefined> {
        yield {
          id: 'chatcmpl-err-1',
          choices: [{ delta: { role: 'assistant', content: '' }, finish_reason: null, index: 0, logprobs: null }],
          created: 1700000000,
          model: 'gpt-err',
          object: 'chat.completion.chunk'
        }
        await new Promise(resolve => setTimeout(resolve, 1))
        throw apiError // Throw error during stream
      }

      const mockErrorStream = {
        async *[Symbol.asyncIterator]() {
          yield* createErrorStreamGenerator()
        }
      }
      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockErrorStream)

      const params: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'gpt-err',
        messages: [{ role: 'user' as const, content: 'Trigger error' }]
      }

      const receivedChunks: StreamChunk[] = []
      try {
        const stream = rosetta.stream(params)
        for await (const chunk of stream) {
          receivedChunks.push(chunk)
        }
      } catch (e) {
        // Errors during stream setup might be caught here, but stream processing errors yield 'error' chunk
      }

      // Assertions
      expect(receivedChunks).toHaveLength(2) // Start, Error
      expect(receivedChunks[0].type).toBe('message_start')
      expect(receivedChunks[1].type).toBe('error')
      const receivedError = (receivedChunks[1] as any).data.error
      expect(receivedError).toBeInstanceOf(ProviderAPIError)

      // FIX: Check against the actual wrapped message (corrected by wrapProviderError fix)
      expect(receivedError.message).toBe('[openai] API Error (Status 500) : Server error details')
      expect(receivedError.provider).toBe(Provider.OpenAI)
      expect(receivedError.statusCode).toBe(500)
    })

    // Add more stream tests: tool calls, JSON mode, etc.
  })

  // Add similar tests for `translate` and `streamSpeech` if needed
})
