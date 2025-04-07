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
