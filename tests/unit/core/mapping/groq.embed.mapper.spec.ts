import {
  mapToGroqEmbedParams,
  mapFromGroqEmbedResponse,
  mapUsageFromGroqEmbed // Import internal function
} from '../../../../src/core/mapping/groq.embed.mapper'
import { EmbedParams, Provider, EmbedResult, TokenUsage } from '../../../../src/types'
import { MappingError } from '../../../../src/errors'
import Groq from 'groq-sdk'

describe('Groq Embed Mapper', () => {
  const model = 'nomic-embed-text-v1.5'
  const baseParams: EmbedParams = {
    provider: Provider.Groq,
    model: model,
    input: ''
  }

  // --- NEW: Tests for mapUsageFromGroqEmbed ---
  describe('mapUsageFromGroqEmbed', () => {
    it('[Easy] should map a valid usage object', () => {
      const usage: Groq.Embeddings.CreateEmbeddingResponse['usage'] = { prompt_tokens: 15, total_tokens: 15 }
      const expected: TokenUsage = { promptTokens: 15, totalTokens: 15, completionTokens: undefined }
      expect(mapUsageFromGroqEmbed(usage)).toEqual(expected)
    })

    it('[Easy] should return undefined for null/undefined input', () => {
      expect(mapUsageFromGroqEmbed(null as any)).toBeUndefined()
      expect(mapUsageFromGroqEmbed(undefined)).toBeUndefined()
    })

    it('[Easy] should handle missing optional fields (prompt_tokens)', () => {
      const usage = { total_tokens: 10 } as Groq.Embeddings.CreateEmbeddingResponse['usage'] // Missing prompt_tokens
      const expected: TokenUsage = { promptTokens: undefined, totalTokens: 10, completionTokens: undefined }
      expect(mapUsageFromGroqEmbed(usage)).toEqual(expected)
    })
  })
  // --- END: Tests for mapUsageFromGroqEmbed ---

  describe('mapToGroqEmbedParams', () => {
    it('should map string input', () => {
      const params: EmbedParams = { ...baseParams, input: 'Embed this text' }
      const result = mapToGroqEmbedParams(params)
      expect(result.model).toBe(model)
      expect(result.input).toBe('Embed this text')
      expect(result.encoding_format).toBeUndefined()
    })

    // --- NEW: Tests for mapToGroqEmbedParams ---
    it('[Easy] should map encodingFormat', () => {
      const params: EmbedParams = { ...baseParams, input: 'Test', encodingFormat: 'base64' }
      const result = mapToGroqEmbedParams(params)
      expect(result.encoding_format).toBe('base64')
    })

    it('[Medium] should warn when dimensions is provided', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: EmbedParams = { ...baseParams, input: 'Test', dimensions: 512 }
      const result = mapToGroqEmbedParams(params)
      expect(result.dimensions).toBeUndefined() // Should be ignored
      expect(warnSpy).toHaveBeenCalledWith(
        "Groq provider does not support 'dimensions' parameter for embeddings. Parameter ignored."
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should warn when input is an array', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: EmbedParams = { ...baseParams, input: ['Text 1', 'Text 2'] }
      const result = mapToGroqEmbedParams(params)
      expect(result.input).toEqual(['Text 1', 'Text 2']) // Array is passed through
      expect(warnSpy).toHaveBeenCalledWith(
        'Mapping array input for Groq embeddings. Ensure the specific model supports batching.'
      )
      warnSpy.mockRestore()
    })
    // --- END: Tests for mapToGroqEmbedParams ---
  })

  describe('mapFromGroqEmbedResponse', () => {
    const modelUsed = 'nomic-embed-text-v1.5-test'

    it('should map a valid embedding response', () => {
      const response: Groq.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1, -0.2, 0.3] }],
        model: 'nomic-embed-text-v1.5-id',
        usage: { prompt_tokens: 5, total_tokens: 5 }
      }
      const result = mapFromGroqEmbedResponse(response, modelUsed)
      expect(result.embeddings).toEqual([[0.1, -0.2, 0.3]])
      expect(result.model).toBe('nomic-embed-text-v1.5-id') // Use model from response
      expect(result.usage).toEqual({ promptTokens: 5, totalTokens: 5, completionTokens: undefined })
    })

    it('should throw MappingError for invalid data structure', () => {
      const invalidResponse = { object: 'list', data: null } as any
      expect(() => mapFromGroqEmbedResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGroqEmbedResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid or empty embedding data structure from Groq.'
      )
    })

    it('should throw MappingError for missing embedding vector', () => {
      const invalidDataResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: null }], // Missing embedding
        model: modelUsed
      } as any
      expect(() => mapFromGroqEmbedResponse(invalidDataResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGroqEmbedResponse(invalidDataResponse, modelUsed)).toThrow(
        'Missing or invalid embedding vector at index 0 in Groq response.'
      )
    })

    // --- NEW: Tests for mapFromGroqEmbedResponse ---
    it('[Easy] should map response with usage correctly', () => {
      const response: Groq.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1] }],
        model: modelUsed,
        usage: { prompt_tokens: 7, total_tokens: 7 } // Valid usage
      }
      const result = mapFromGroqEmbedResponse(response, modelUsed)
      expect(result.usage).toEqual({ promptTokens: 7, totalTokens: 7, completionTokens: undefined })
    })

    it('[Easy] should use model from response when available', () => {
      const response: Groq.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1] }],
        model: 'groq-model-from-response', // Different model ID
        usage: { prompt_tokens: 5, total_tokens: 5 }
      }
      const result = mapFromGroqEmbedResponse(response, modelUsed) // modelUsed is 'nomic-embed-text-v1.5-test'
      expect(result.model).toBe('groq-model-from-response') // Should prioritize response model
    })

    it('[Medium] should handle multiple embeddings in the data array', () => {
      const response: Groq.Embeddings.CreateEmbeddingResponse = {
        object: 'list',
        data: [
          { object: 'embedding', index: 0, embedding: [0.1, 0.2] },
          { object: 'embedding', index: 1, embedding: [0.3, 0.4] },
          { object: 'embedding', index: 2, embedding: [0.5, 0.6] }
        ],
        model: modelUsed,
        usage: { prompt_tokens: 15, total_tokens: 15 }
      }
      const result = mapFromGroqEmbedResponse(response, modelUsed)
      expect(result.embeddings).toHaveLength(3)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
      ])
    })
    // --- END: Tests for mapFromGroqEmbedResponse ---
  })
})
