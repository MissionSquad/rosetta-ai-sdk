// Dedicated mapper for Google Embeddings

import type { EmbedResult } from '../../types'
import { Provider } from '../../types'
import { MappingError } from '../../errors'
import { mapTokenUsage } from './common.utils'

// Type for new SDK embedding response
type EmbedContentResponse = {
  embeddings?: Array<{
    values?: number[]
    statistics?: any
  }>
  metadata?: {
    billableCharacterCount?: number
  }
  usageMetadata?: any
}

// Map embedding response (handles both single and batch)
export function mapFromGoogleEmbedResponse(response: EmbedContentResponse, model: string): EmbedResult {
  if (!response?.embeddings || !Array.isArray(response.embeddings)) {
    throw new MappingError(
      'Invalid embedding response structure from Google.',
      Provider.Google,
      'mapFromGoogleEmbedResponse'
    )
  }

  // Extract all embedding values
  const embeddings = response.embeddings
    .map(e => e?.values) // Get values array, might be undefined
    .filter((v): v is number[] => v !== undefined && Array.isArray(v)) // Filter out undefined/null and ensure it's number[]

  if (embeddings.length === 0 && response.embeddings.length > 0) {
    throw new MappingError('All embeddings were missing values in Google response.', Provider.Google)
  }
  if (embeddings.length !== response.embeddings.length) {
    console.warn('Some embeddings were missing values in Google response.')
  }

  return {
    embeddings: embeddings,
    // Use common utility for usage mapping
    usage: mapTokenUsage((response as any).usageMetadata),
    model: model,
    rawResponse: response
  }
}

// Keep the batch function for backwards compatibility, but it now calls the unified function
export function mapFromGoogleEmbedBatchResponse(response: EmbedContentResponse, model: string): EmbedResult {
  return mapFromGoogleEmbedResponse(response, model)
}
