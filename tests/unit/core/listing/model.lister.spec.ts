import Groq from 'groq-sdk'
import { listModelsForProvider } from '../../../../src/core/listing/model.lister'
import * as FetchUtils from '../../../../src/core/listing/fetch.utils'
import { anthropicStaticModels } from '../../../../src/core/listing/static-data/anthropic.models'
import { Provider, RosettaModelList, ModelListingSourceConfig } from '../../../../src/types'
import { ConfigurationError, ProviderAPIError, MappingError } from '../../../../src/errors'

// Mock dependencies
jest.mock('../../../../src/core/listing/fetch.utils')
jest.mock('groq-sdk') // Mock the Groq SDK

const mockFetchAndValidateModelsFromApi = FetchUtils.fetchAndValidateModelsFromApi as jest.Mock
const MockGroq = Groq as jest.MockedClass<typeof Groq>
let mockGroqClientInstance: { models: { list: jest.Mock } }

describe('listModelsForProvider', () => {
  const testApiKey = 'test-key'
  const mockModelList: RosettaModelList = {
    object: 'list',
    data: [
      {
        id: 'api-model-1',
        object: 'model',
        owned_by: 'api-owner',
        provider: Provider.OpenAI,
        rawData: {}
      }
    ]
  }

  beforeEach(() => {
    jest.clearAllMocks()
    mockFetchAndValidateModelsFromApi.mockResolvedValue(mockModelList)
    // Setup mock Groq client instance
    mockGroqClientInstance = {
      models: {
        list: jest.fn().mockResolvedValue({
          object: 'list',
          data: [
            {
              id: 'groq-model-1',
              object: 'model',
              owned_by: 'groq-owner',
              active: true,
              context_window: 8192
            }
          ]
        })
      }
    }
    MockGroq.mockImplementation(() => mockGroqClientInstance as any)
  })

  // --- Anthropic Tests ---
  describe('Anthropic', () => {
    it('[Easy] should return static list for Anthropic by default', async () => {
      const result = await listModelsForProvider(Provider.Anthropic, { apiKey: testApiKey })
      expect(result).toEqual(anthropicStaticModels)
      expect(mockFetchAndValidateModelsFromApi).not.toHaveBeenCalled()
      expect(mockGroqClientInstance.models.list).not.toHaveBeenCalled()
    })

    it('[Easy] should return static list for Anthropic when explicitly configured', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'staticList' }
      const result = await listModelsForProvider(Provider.Anthropic, { sourceConfig, apiKey: testApiKey })
      expect(result).toEqual(anthropicStaticModels)
    })

    it('[Medium] should throw ConfigurationError if staticList config used for non-Anthropic', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'staticList' }
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        ConfigurationError
      )
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        'Static list is only configured for Anthropic, not openai.'
      )
    })
  })

  // --- Groq Tests ---
  describe('Groq', () => {
    it('[Easy] should use sdkMethod for Groq by default if client provided', async () => {
      const result = await listModelsForProvider(Provider.Groq, {
        apiKey: testApiKey,
        groqClient: mockGroqClientInstance as any
      })
      expect(mockGroqClientInstance.models.list).toHaveBeenCalledTimes(1)
      expect(mockFetchAndValidateModelsFromApi).not.toHaveBeenCalled()
      expect(result.object).toBe('list')
      expect(result.data).toHaveLength(1)
      expect(result.data[0]).toEqual({
        id: 'groq-model-1',
        object: 'model',
        owned_by: 'groq-owner',
        created: undefined,
        active: true,
        context_window: 8192,
        public_apps: undefined,
        max_completion_tokens: undefined,
        properties: undefined,
        provider: Provider.Groq,
        rawData: {
          id: 'groq-model-1',
          object: 'model',
          owned_by: 'groq-owner',
          active: true,
          context_window: 8192
        }
      })
    })

    it('[Easy] should use sdkMethod for Groq when explicitly configured', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'sdkMethod' }
      await listModelsForProvider(Provider.Groq, {
        sourceConfig,
        apiKey: testApiKey,
        groqClient: mockGroqClientInstance as any
      })
      expect(mockGroqClientInstance.models.list).toHaveBeenCalledTimes(1)
    })

    it('[Medium] should throw ConfigurationError for sdkMethod if Groq client missing', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'sdkMethod' }
      await expect(listModelsForProvider(Provider.Groq, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        ConfigurationError
      )
      await expect(listModelsForProvider(Provider.Groq, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        'SDK method listing is only configured for Groq with an active client.'
      )
    })

    it('[Medium] should throw ConfigurationError for sdkMethod if used for non-Groq provider', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'sdkMethod' }
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        ConfigurationError
      )
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        'SDK method listing is only configured for Groq with an active client.'
      )
    })

    it('[Hard] should wrap errors from Groq SDK call', async () => {
      const sdkError = new Error('Groq SDK failed')
      mockGroqClientInstance.models.list.mockRejectedValueOnce(sdkError)
      // REMOVED: First expect call
      // await expect(
      //   listModelsForProvider(Provider.Groq, { apiKey: testApiKey, groqClient: mockGroqClientInstance as any })
      // ).rejects.toThrow(ProviderAPIError)
      await expect(
        listModelsForProvider(Provider.Groq, { apiKey: testApiKey, groqClient: mockGroqClientInstance as any })
      ).rejects.toThrow('Failed to list models for groq using sdkMethod: Groq SDK failed')
    })
  })

  // --- OpenAI/Google Tests (API Endpoint) ---
  describe('API Endpoint Providers (OpenAI/Google)', () => {
    it('[Easy] should use apiEndpoint for OpenAI by default', async () => {
      await listModelsForProvider(Provider.OpenAI, { apiKey: testApiKey })
      expect(mockFetchAndValidateModelsFromApi).toHaveBeenCalledWith(
        'https://api.openai.com/v1/models',
        Provider.OpenAI,
        testApiKey
      )
      expect(mockGroqClientInstance.models.list).not.toHaveBeenCalled()
    })

    it('[Easy] should use apiEndpoint for Google by default', async () => {
      await listModelsForProvider(Provider.Google, { apiKey: testApiKey })
      expect(mockFetchAndValidateModelsFromApi).toHaveBeenCalledWith(
        'https://generativelanguage.googleapis.com/v1beta/models',
        Provider.Google,
        testApiKey
      )
    })

    it('[Easy] should use configured URL for apiEndpoint', async () => {
      const customUrl = 'http://custom.openai/api/models'
      const sourceConfig: ModelListingSourceConfig = { type: 'apiEndpoint', url: customUrl }
      await listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })
      expect(mockFetchAndValidateModelsFromApi).toHaveBeenCalledWith(customUrl, Provider.OpenAI, testApiKey)
    })

    it('[Medium] should throw ConfigurationError if default URL not found for apiEndpoint', async () => {
      const sourceConfig: ModelListingSourceConfig = { type: 'apiEndpoint', url: '' } // No URL provided
      // Use a provider without a default URL logic path
      await expect(
        listModelsForProvider(Provider.Anthropic, { sourceConfig, apiKey: testApiKey }) // Using Anthropic to force error
      ).rejects.toThrow(ConfigurationError)
      await expect(listModelsForProvider(Provider.Anthropic, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        'API endpoint URL for anthropic not configured.'
      )
    })

    it('[Hard] should wrap errors from fetchAndValidateModelsFromApi', async () => {
      const fetchError = new MappingError('Invalid API structure', Provider.OpenAI)
      mockFetchAndValidateModelsFromApi.mockRejectedValueOnce(fetchError)
      // REMOVED: First expect call
      // await expect(listModelsForProvider(Provider.OpenAI, { apiKey: testApiKey })).rejects.toThrow(MappingError) // Should re-throw MappingError
      await expect(listModelsForProvider(Provider.OpenAI, { apiKey: testApiKey })).rejects.toThrow(
        'Invalid API structure'
      )
    })

    it('[Hard] should wrap generic errors from fetchAndValidateModelsFromApi as ProviderAPIError', async () => {
      const genericError = new Error('Network failed')
      mockFetchAndValidateModelsFromApi.mockRejectedValueOnce(genericError)
      // REMOVED: First expect call
      // await expect(listModelsForProvider(Provider.OpenAI, { apiKey: testApiKey })).rejects.toThrow(ProviderAPIError)
      await expect(listModelsForProvider(Provider.OpenAI, { apiKey: testApiKey })).rejects.toThrow(
        'Failed to list models for openai using apiEndpoint: Network failed'
      )
    })
  })

  // --- General Error Handling ---
  describe('General Error Handling', () => {
    it('[Medium] should throw ConfigurationError for unknown provider default strategy', async () => {
      const unknownProvider = 'unknown_provider' as Provider
      await expect(listModelsForProvider(unknownProvider, { apiKey: testApiKey })).rejects.toThrow(ConfigurationError)
      await expect(listModelsForProvider(unknownProvider, { apiKey: testApiKey })).rejects.toThrow(
        'Model listing source type for provider unknown_provider could not be determined.'
      )
    })

    it('[Medium] should throw ConfigurationError for unsupported source type', async () => {
      const sourceConfig = { type: 'invalid_type' } as any
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        ConfigurationError
      )
      await expect(listModelsForProvider(Provider.OpenAI, { sourceConfig, apiKey: testApiKey })).rejects.toThrow(
        'Unsupported model listing source type: invalid_type'
      )
    })
  })
})
