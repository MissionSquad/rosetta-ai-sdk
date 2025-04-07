// Dedicated mapper for Google Embeddings

import { EmbedContentResponse, BatchEmbedContentsResponse } from '@google/generative-ai'
import { EmbedResult, Provider } from '../../types'
import { MappingError } from '../../errors'
import { mapTokenUsage } from './common.utils'

// Removed mapUsageFromGoogleEmbed as mapTokenUsage handles it

// Map single embedding response
export function mapFromGoogleEmbedResponse(response: EmbedContentResponse, model: string): EmbedResult {
  if (!response?.embedding?.values) {
    throw new MappingError(
      'Invalid single embedding response structure from Google.',
      Provider.Google,
      'mapFromGoogleEmbedResponse'
    )
  }
  return {
    embeddings: [response.embedding.values],
    // Use common utility for usage mapping
    // Google Embeddings API v1beta includes usageMetadata
    usage: mapTokenUsage((response as any).usageMetadata), // Access usage if present
    model: model,
    rawResponse: response
  }
}

// Map batch embedding response
export function mapFromGoogleEmbedBatchResponse(response: BatchEmbedContentsResponse, model: string): EmbedResult {
  if (!response?.embeddings || !Array.isArray(response.embeddings)) {
    throw new MappingError(
      'Invalid batch embedding response structure from Google.',
      Provider.Google,
      'mapFromGoogleEmbedBatchResponse'
    )
  }
  // FIX: Correctly map and filter potentially null embeddings/values
  const embeddings = response.embeddings
    .map(e => e?.values) // Get values array, might be undefined
    .filter((v): v is number[] => v !== undefined && Array.isArray(v)) // Filter out undefined/null and ensure it's number[]

  if (embeddings.length !== response.embeddings.length) {
    // Check if any were filtered out
    console.warn('Some embeddings were missing values in Google batch response.')
  }
  if (embeddings.length === 0 && response.embeddings.length > 0) {
    throw new MappingError('All embeddings were missing values in Google batch response.', Provider.Google)
  }

  return {
    embeddings: embeddings,
    // Use common utility for usage mapping
    // Batch responses in v1beta also include usageMetadata
    usage: mapTokenUsage((response as any).usageMetadata),
    model: model,
    rawResponse: response
  }
}
