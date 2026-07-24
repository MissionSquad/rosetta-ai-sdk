import { fetchAndValidateModelsFromApi } from '../../../../src/core/listing/fetch.utils'
import { Provider, RosettaModelList } from '../../../../src/types'
import { ProviderAPIError, MappingError } from '../../../../src/errors'
import { z } from 'zod'

// Mock the global fetch function
global.fetch = jest.fn()

const mockFetch = fetch as jest.Mock

describe('fetchAndValidateModelsFromApi', () => {
  const testUrl = 'http://test.api/models'
  const testApiKey = 'test-key'
  const testProvider = Provider.OpenAI

  beforeEach(() => {
    // Reset the mock before each test to ensure a clean state
    mockFetch.mockReset()
  })

  it('[Easy] should fetch, validate, and map models successfully', async () => {
    const mockApiResponse = {
      object: 'list',
      data: [
        {
          id: 'model-1',
          object: 'model',
          owned_by: 'org1',
          created: 1677652288,
          active: true,
          context_window: 4096,
          public_apps: null,
          max_completion_tokens: 2048,
          properties: { description: 'Model 1 Desc' }
        },
        {
          id: 'model-2',
          object: 'model',
          owned_by: 'org2',
          created: null, // Test null created
          active: false, // Test optional active
          context_window: 8192
          // Missing other optional fields
        }
      ]
    }
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => mockApiResponse,
      status: 200
    })

    const result: RosettaModelList = await fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)

    expect(mockFetch).toHaveBeenCalledWith(testUrl, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${testApiKey}`,
        Accept: 'application/json'
      }
    })
    expect(result.object).toBe('list')
    expect(result.data).toHaveLength(2)

    // Check model 1 mapping
    expect(result.data[0]).toEqual({
      id: 'model-1',
      object: 'model',
      owned_by: 'org1',
      created: 1677652288,
      active: true,
      context_window: 4096,
      public_apps: undefined, // Mapped from null
      max_completion_tokens: 2048,
      properties: {
        description: 'Model 1 Desc',
        strengths: undefined,
        multilingual: undefined,
        vision: undefined
      },
      provider: testProvider,
      rawData: mockApiResponse.data[0]
    })

    // Check model 2 mapping (optional fields)
    expect(result.data[1]).toEqual({
      id: 'model-2',
      object: 'model',
      owned_by: 'org2',
      created: undefined, // Mapped from null
      active: false,
      context_window: 8192,
      public_apps: undefined,
      max_completion_tokens: undefined,
      properties: undefined,
      provider: testProvider,
      rawData: mockApiResponse.data[1]
    })
  })

  it('[Easy] should throw ProviderAPIError for missing API key', async () => {
    // No fetch mock needed as it should throw before fetch
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, undefined)).rejects.toThrow(ProviderAPIError)
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, undefined)).rejects.toThrow(
      `API key for ${testProvider} is required but missing for model listing.`
    )
  })

  it('[Easy] should throw ProviderAPIError for non-OK HTTP response', async () => {
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      text: async () => 'Not Found'
    })

    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(ProviderAPIError)
    // Check the specific message thrown by the !response.ok block
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(
      `Failed to fetch models from ${testProvider} API: Not Found`
    )
  })

  it('[Medium] should throw MappingError for invalid JSON response', async () => {
    const jsonError = new SyntaxError('Invalid JSON')
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => {
        throw jsonError
      }, // Simulate JSON parsing error
      status: 200
    })

    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(ProviderAPIError) // The catch block wraps it
    // Check the specific message thrown by the catch block when json() fails
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(
      `Network or parsing error fetching models for ${testProvider}: Invalid JSON`
    )
  })

  it('[Medium] should throw MappingError for Zod validation failure (missing required fields)', async () => {
    const invalidApiResponse = {
      // Missing 'object' and 'data'
      items: [{ id: 'model-1', object: 'model', owned_by: 'org1' }]
    }
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => invalidApiResponse,
      status: 200
    })

    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(MappingError)
    // Check the specific message thrown by the Zod validation failure
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(
      `Invalid API response structure received from ${testProvider}.`
    )
  })

  it('[Medium] should throw MappingError for Zod validation failure (incorrect types)', async () => {
    const invalidApiResponse = {
      object: 'list',
      data: [
        {
          id: 123, // Incorrect type
          object: 'model',
          owned_by: 'org1'
        }
      ]
    }
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => invalidApiResponse,
      status: 200
    })

    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(MappingError)
    // Check the specific message thrown by the Zod validation failure
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(
      `Invalid API response structure received from ${testProvider}.`
    )
  })

  it('[Hard] should handle fetch throwing an error', async () => {
    const fetchError = new Error('Network connection failed')
    // Use mockRejectedValue to make it persistent for the test
    mockFetch.mockRejectedValue(fetchError)

    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(ProviderAPIError)
    // Check the specific message thrown by the catch block when fetch rejects
    await expect(fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)).rejects.toThrow(
      `Network or parsing error fetching models for ${testProvider}: Network connection failed`
    )
  })

  describe('OpenRouter-style responses (no object/owned_by markers)', () => {
    const openRouterUrl = 'https://openrouter.ai/api/v1/models'
    const openRouterProvider = 'openrouter'
    // Shape taken from https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties
    const openRouterModel = {
      architecture: {
        input_modalities: ['text'],
        instruct_type: 'chatml',
        modality: 'text->text',
        output_modalities: ['text'],
        tokenizer: 'GPT'
      },
      canonical_slug: 'openai/gpt-4',
      context_length: 8192,
      created: 1692901234,
      default_parameters: null,
      description: 'GPT-4 is a large multimodal model that can solve difficult problems with greater accuracy.',
      expiration_date: null,
      id: 'openai/gpt-4',
      knowledge_cutoff: null,
      links: { details: '/api/v1/models/openai/gpt-4/endpoints' },
      name: 'GPT-4',
      per_request_limits: null,
      pricing: { completion: '0.00006', image: '0', prompt: '0.00003', request: '0' },
      supported_parameters: ['temperature', 'top_p', 'max_tokens'],
      supported_voices: null,
      top_provider: { context_length: 8192, is_moderated: true, max_completion_tokens: 4096 }
    }

    it('[Medium] should validate and normalize the OpenRouter model list response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [openRouterModel] }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      expect(result.object).toBe('list')
      expect(result.data).toHaveLength(1)
      const model = result.data[0]
      expect(model.id).toBe('openai/gpt-4')
      expect(model.object).toBe('model')
      // Derived from the vendor prefix of the id
      expect(model.owned_by).toBe('openai')
      expect(model.created).toBe(1692901234)
      // Mapped from OpenRouter's context_length
      expect(model.context_window).toBe(8192)
      // Mapped from top_provider.max_completion_tokens
      expect(model.max_completion_tokens).toBe(4096)
      expect(model.properties).toEqual({
        description: 'GPT-4 is a large multimodal model that can solve difficult problems with greater accuracy.',
        strengths: undefined,
        multilingual: undefined,
        vision: false
      })
      expect(model.provider).toBe(openRouterProvider)
      // Original payload preserved for downstream use
      expect(model.rawData).toEqual(openRouterModel)
    })

    it('[Medium] should derive vision capability from architecture.input_modalities', async () => {
      const visionModel = {
        ...openRouterModel,
        id: 'google/gemini-2.5-pro',
        architecture: { ...openRouterModel.architecture, input_modalities: ['text', 'image'] }
      }
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [visionModel] }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      expect(result.data[0].owned_by).toBe('google')
      expect(result.data[0].properties?.vision).toBe(true)
    })

    it('[Medium] should ignore non-boolean vision and non-string description values during normalization', async () => {
      const badPropsModel = {
        ...openRouterModel,
        id: 'openai/gpt-4o',
        properties: { vision: 'yes', description: 123, strengths: 42, multilingual: 'oui' },
        architecture: { ...openRouterModel.architecture, input_modalities: ['text', 'image'] }
      }
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [badPropsModel] }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      // Non-boolean properties.vision is discarded; modality inference wins
      expect(result.data[0].properties?.vision).toBe(true)
      // Non-string properties.description falls back to the top-level description
      expect(result.data[0].properties?.description).toBe(openRouterModel.description)
      // Invalid strengths/multilingual types are dropped rather than propagated
      expect(result.data[0].properties?.strengths).toBeUndefined()
      expect(result.data[0].properties?.multilingual).toBeUndefined()
    })

    it('[Medium] should fall back to the provider key for owned_by when the id has no vendor prefix', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [{ id: 'my-local-model', context_length: 4096 }] }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi('http://localhost:1234/v1/models', 'lmstudio', undefined)

      expect(result.data[0].owned_by).toBe('lmstudio')
      expect(result.data[0].context_window).toBe(4096)
    })

    it('[Medium] should treat an empty owned_by string as absent', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [
            { id: 'acme/widget', owned_by: '' },
            { id: 'plain-model', owned_by: '' }
          ]
        }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      // Empty string falls through to the vendor prefix, then the provider key
      expect(result.data[0].owned_by).toBe('acme')
      expect(result.data[1].owned_by).toBe(openRouterProvider)
    })

    it('[Medium] should tolerate extra top-level fields in the response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [{ id: 'a/b' }], extra_field: 'ignored' }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      expect(result.data[0].id).toBe('a/b')
    })

    it('[Medium] should accept a bare array response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => [{ id: 'bare/model', owned_by: 'someone' }],
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      expect(result.object).toBe('list')
      expect(result.data[0].id).toBe('bare/model')
      expect(result.data[0].owned_by).toBe('someone')
    })

    it('[Medium] should leave vision undefined when no modality signal is present', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: [{ id: 'mystery/model' }] }),
        status: 200
      })

      const result = await fetchAndValidateModelsFromApi(openRouterUrl, openRouterProvider, testApiKey)

      // No properties, no architecture: no capability signal at all
      expect(result.data[0].properties).toBeUndefined()
      expect(result.data[0].context_window).toBeUndefined()
    })
  })

  it('[Hard] should handle optional properties correctly during mapping', async () => {
    const mockApiResponse = {
      object: 'list',
      data: [
        {
          id: 'full-model',
          object: 'model',
          owned_by: 'org',
          created: 12345,
          active: true,
          context_window: 1000,
          public_apps: 'yes', // Test string value
          max_completion_tokens: 500,
          properties: {
            description: 'Desc',
            strengths: 'Strength',
            multilingual: true,
            vision: false
          }
        },
        {
          id: 'minimal-model',
          object: 'model',
          owned_by: 'org'
          // All optional fields missing
        }
      ]
    }
    // Use mockResolvedValue to make it persistent for the test
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => mockApiResponse,
      status: 200
    })

    const result = await fetchAndValidateModelsFromApi(testUrl, testProvider, testApiKey)
    expect(result.data).toHaveLength(2)

    // Full model
    expect(result.data[0].created).toBe(12345)
    expect(result.data[0].active).toBe(true)
    expect(result.data[0].context_window).toBe(1000)
    expect(result.data[0].public_apps).toBe('yes')
    expect(result.data[0].max_completion_tokens).toBe(500)
    expect(result.data[0].properties).toEqual({
      description: 'Desc',
      strengths: 'Strength',
      multilingual: true,
      vision: false
    })

    // Minimal model
    expect(result.data[1].created).toBeUndefined()
    expect(result.data[1].active).toBeUndefined()
    expect(result.data[1].context_window).toBeUndefined()
    expect(result.data[1].public_apps).toBeUndefined()
    expect(result.data[1].max_completion_tokens).toBeUndefined()
    expect(result.data[1].properties).toBeUndefined()
  })
})
