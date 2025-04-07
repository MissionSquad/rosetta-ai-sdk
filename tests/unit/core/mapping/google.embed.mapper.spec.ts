import { EmbedContentResponse, BatchEmbedContentsResponse } from '@google/generative-ai'
import {
  mapFromGoogleEmbedResponse,
  mapFromGoogleEmbedBatchResponse
} from '../../../../src/core/mapping/google.embed.mapper'
import { TokenUsage } from '../../../../src/types'
import { MappingError } from '../../../../src/errors'

describe('Google Embed Mapper', () => {
  const modelUsed = 'embedding-001-test'

  describe('mapFromGoogleEmbedResponse', () => {
    it('[Easy] should map a single embedding response', () => {
      const response: EmbedContentResponse = {
        embedding: { values: [0.1, 0.2, 0.3] }
      }
      const result = mapFromGoogleEmbedResponse(response, modelUsed)
      expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toBeUndefined() // No usage metadata in mock
      expect(result.rawResponse).toBe(response)
    })

    it('[Easy] should map a single embedding response with usage metadata', () => {
      const response = {
        embedding: { values: [0.1, 0.2, 0.3] },
        usageMetadata: { totalTokenCount: 10, cachedContentTokenCount: 2 }
      } as EmbedContentResponse // Cast as SDK type might not include usage yet
      const expectedUsage: TokenUsage = {
        promptTokens: undefined,
        completionTokens: undefined,
        totalTokens: 10,
        cachedContentTokenCount: 2
      }
      const result = mapFromGoogleEmbedResponse(response, modelUsed)
      expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toEqual(expectedUsage)
      expect(result.rawResponse).toBe(response)
    })

    it('[Medium] should throw MappingError if embedding structure is invalid (null embedding)', () => {
      const invalidResponse = { embedding: null } as any
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid single embedding response structure from Google.'
      )
    })

    it('[Medium] should throw MappingError if embedding structure is invalid (missing values)', () => {
      const invalidResponse = { embedding: {} } as any
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid single embedding response structure from Google.'
      )
    })
  })

  describe('mapFromGoogleEmbedBatchResponse', () => {
    it('[Easy] should map a batch embedding response', () => {
      const response: BatchEmbedContentsResponse = {
        embeddings: [{ values: [0.1, 0.2] }, { values: [0.3, 0.4] }]
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.3, 0.4]
      ])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toBeUndefined() // No usage metadata in mock
      expect(result.rawResponse).toBe(response)
    })

    it('[Easy] should map a batch embedding response with usage metadata', () => {
      const response = {
        embeddings: [{ values: [0.1, 0.2] }, { values: [0.3, 0.4] }],
        usageMetadata: { totalTokenCount: 25 }
      } as BatchEmbedContentsResponse // Cast as SDK type might not include usage yet
      const expectedUsage: TokenUsage = {
        promptTokens: undefined,
        completionTokens: undefined,
        totalTokens: 25,
        cachedContentTokenCount: undefined
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.3, 0.4]
      ])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toEqual(expectedUsage)
      expect(result.rawResponse).toBe(response)
    })

    it('[Medium] should handle missing embeddings in batch response and warn', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: BatchEmbedContentsResponse = {
        embeddings: [
          { values: [0.1, 0.2] },
          null as any, // Simulate a missing embedding object
          { values: [0.5, 0.6] }
        ]
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.5, 0.6]
      ])
      expect(warnSpy).toHaveBeenCalledWith('Some embeddings were missing values in Google batch response.')
      warnSpy.mockRestore()
    })

    it('[Medium] should throw MappingError if batch structure is invalid (null embeddings)', () => {
      const invalidResponse = { embeddings: null } as any
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid batch embedding response structure from Google.'
      )
    })

    it('[Medium] should throw MappingError if batch structure is invalid (not array)', () => {
      const invalidResponse = { embeddings: {} } as any
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid batch embedding response structure from Google.'
      )
    })

    it('[Hard] should throw MappingError if all embeddings are missing values', () => {
      const response = { embeddings: [null, undefined, { values: undefined }] } as any
      expect(() => mapFromGoogleEmbedBatchResponse(response, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedBatchResponse(response, modelUsed)).toThrow(
        'All embeddings were missing values in Google batch response.'
      )
    })

    it('[Hard] should handle empty embeddings array gracefully', () => {
      const response: BatchEmbedContentsResponse = {
        embeddings: [] // Empty array
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toBeUndefined()
    })
  })
})
