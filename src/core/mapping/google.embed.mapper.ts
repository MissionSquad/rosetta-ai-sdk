// Dedicated mapper for Google Embeddings

import { EmbedContentResponse, BatchEmbedContentsResponse, UsageMetadata } from '@google/generative-ai' // Import UsageMetadata
import { EmbedResult, Provider, TokenUsage } from '../../types'
import { MappingError } from '../../errors'

function mapUsageFromGoogleEmbed(usageMetadata: UsageMetadata | undefined): TokenUsage | undefined {
  // Google Embeddings API responses *do* include token usage in v1beta+
  if (!usageMetadata) return undefined
  return {
    // Note: Google Embeddings usage only reports total tokens, not prompt/completion breakdown.
    promptTokens: undefined,
    completionTokens: undefined,
    totalTokens: usageMetadata.totalTokenCount ?? undefined, // Prefer totalTokenCount if available
    cachedContentTokenCount: usageMetadata.cachedContentTokenCount
  }
}

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
    // Assuming UsageMetadata might be available at the top level or within a specific field.
    // Adjust path if necessary based on actual response structure.
    usage: mapUsageFromGoogleEmbed((response as any).usageMetadata), // Access usage if present
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
    // Batch responses might also have overall usage info. Adjust if needed.
    usage: mapUsageFromGoogleEmbed((response as any).usageMetadata),
    model: model,
    rawResponse: response
  }
}
