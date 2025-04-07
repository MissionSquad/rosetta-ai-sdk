import OpenAI from 'openai'
import { mapToOpenAIEmbedParams, mapFromOpenAIEmbedResponse } from '../../../../src/core/mapping/openai.embed.mapper'
import { EmbedParams, Provider, EmbedResult } from '../../../../src/types'
import { MappingError } from '../../../../src/errors'

describe('OpenAI Embed Mapper', () => {
  const model = 'text-embedding-3-small'
  const baseParams: EmbedParams = {
    provider: Provider.OpenAI,
    model: model,
    input: ''
  }

  describe('mapToOpenAIEmbedParams', () => {
    it('[Easy] should map string input', () => {
      const params: EmbedParams = { ...baseParams, input: 'Embed this' }
      const result = mapToOpenAIEmbedParams(params)
      expect(result.model).toBe(model)
      expect(result.input).toBe('Embed this')
      expect(result.encoding_format).toBeUndefined()
      expect(result.dimensions).toBeUndefined()
    })

    it('[Easy] should map string array input', () => {
      const params: EmbedParams = { ...baseParams, input: ['Embed this', 'And this'] }
      const result = mapToOpenAIEmbedParams(params)
      expect(result.model).toBe(model)
      expect(result.input).toEqual(['Embed this', 'And this'])
    })

    it('[Easy] should map encodingFormat', () => {
      const params: EmbedParams = { ...baseParams, input: 'Test', encodingFormat: 'base64' }
      const result = mapToOpenAIEmbedParams(params)
      expect(result.encoding_format).toBe('base64')
    })

    it('[Easy] should map dimensions', () => {
      const params: EmbedParams = { ...baseParams, input: 'Test', dimensions: 256 }
      const result = mapToOpenAIEmbedParams(params)
      expect(result.dimensions).toBe(256)
    })

    it('[Medium] should throw MappingError for invalid input type', () => {
      const params = { ...baseParams, input: 123 } as any // Invalid input type
      expect(() => mapToOpenAIEmbedParams(params)).toThrow(MappingError)
      expect(() => mapToOpenAIEmbedParams(params)).toThrow(
        'Invalid input type for OpenAI embeddings. Expected string or string[].'
      )
    })

    it('[Medium] should map a combination of encodingFormat and dimensions', () => {
      const params: EmbedParams = { ...baseParams, input: 'Combo', encodingFormat: 'base64', dimensions: 1024 }
      const result = mapToOpenAIEmbedParams(params)
      expect(result.encoding_format).toBe('base64')
      expect(result.dimensions).toBe(1024)
    })
  })

  describe('mapFromOpenAIEmbedResponse', () => {
    const modelUsed = 'text-embedding-3-small-test'

    it('[Easy] should map a valid embedding response', () => {
      const response: OpenAI.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [
          { object: 'embedding', index: 0, embedding: [0.1, 0.2, 0.3] },
          { object: 'embedding', index: 1, embedding: [0.4, 0.5, 0.6] }
        ],
        model: 'text-embedding-3-small-id',
        usage: { prompt_tokens: 10, total_tokens: 10 }
      }
      const result: EmbedResult = mapFromOpenAIEmbedResponse(response, modelUsed)

      expect(result.embeddings).toEqual([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
      ])
      expect(result.model).toBe('text-embedding-3-small-id')
      expect(result.usage).toEqual({ promptTokens: 10, totalTokens: 10, completionTokens: undefined })
      expect(result.rawResponse).toBe(response)
    })

    it('[Easy] should throw MappingError if data array is missing or empty', () => {
      const invalidResponse1 = { object: 'list', data: null } as any
      const invalidResponse2 = { object: 'list', data: [] } as any
      expect(() => mapFromOpenAIEmbedResponse(invalidResponse1, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAIEmbedResponse(invalidResponse2, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAIEmbedResponse(invalidResponse1, modelUsed)).toThrow(
        'Invalid or empty embedding data structure from OpenAI.'
      )
    })

    it('[Medium] should throw MappingError if an embedding vector is missing or invalid', () => {
      const invalidResponse: OpenAI.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [
          { object: 'embedding', index: 0, embedding: [0.1, 0.2] },
          { object: 'embedding', index: 1, embedding: null } as any
        ],
        model: modelUsed,
        usage: { prompt_tokens: 5, total_tokens: 5 }
      }
      expect(() => mapFromOpenAIEmbedResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAIEmbedResponse(invalidResponse, modelUsed)).toThrow(
        'Missing or invalid embedding vector at index 1 in OpenAI response.'
      )
    })

    it('[Medium] should handle missing usage gracefully', () => {
      const response: OpenAI.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.7, 0.8] }],
        model: modelUsed,
        usage: null as any
      }
      const result: EmbedResult = mapFromOpenAIEmbedResponse(response, modelUsed)
      expect(result.usage).toBeUndefined()
    })

    it('[Medium] should use model from response when available', () => {
      const response: OpenAI.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.9] }],
        model: 'model-from-response', // Different model ID
        usage: { prompt_tokens: 2, total_tokens: 2 }
      }
      const result = mapFromOpenAIEmbedResponse(response, modelUsed) // modelUsed is 'text-embedding-3-small-test'
      expect(result.model).toBe('model-from-response') // Should prioritize response model
    })
  })
})
