// Mappers for Groq Embeddings
import {
  EmbeddingCreateParams,
  CreateEmbeddingResponse as GroqCreateEmbeddingResponse
} from 'groq-sdk/resources/embeddings'
import { EmbedParams, EmbedResult, Provider, TokenUsage } from '../../types'
import { MappingError } from '../../errors'

// --- Parameter Mapping ---

export function mapToGroqEmbedParams(params: EmbedParams): EmbeddingCreateParams {
  if (Array.isArray(params.input)) {
    // Groq API might support array input even if SDK types were initially restrictive
    console.warn('Mapping array input for Groq embeddings. Ensure the specific model supports batching.')
  }
  // Allow passing array input through
  const inputData = params.input

  if (params.dimensions) {
    console.warn("Groq provider does not support 'dimensions' parameter for embeddings. Parameter ignored.")
  }

  return {
    model: params.model!,
    input: inputData, // Pass string or array
    encoding_format: params.encodingFormat
  }
}

// Groq's usage type from the response
type GroqEmbeddingUsage = GroqCreateEmbeddingResponse['usage']

export function mapUsageFromGroqEmbed(usage: GroqEmbeddingUsage): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.prompt_tokens ?? undefined,
    totalTokens: usage.total_tokens ?? undefined,
    completionTokens: undefined // Embeddings don't have completion tokens
  }
}

// --- Result Mapping ---

export function mapFromGroqEmbedResponse(response: GroqCreateEmbeddingResponse, modelUsed: string): EmbedResult {
  // Use the imported GroqCreateEmbeddingResponse type
  if (!response?.data || !Array.isArray(response.data) || response.data.length === 0) {
    throw new MappingError(
      'Invalid or empty embedding data structure from Groq.',
      Provider.Groq,
      'mapFromGroqEmbedResponse'
    )
  }

  const embeddings = response.data.map(d => d?.embedding)

  if (embeddings.some(e => !e || !Array.isArray(e))) {
    // Find the first invalid embedding for a better error message
    const invalidIndex = embeddings.findIndex(e => !e || !Array.isArray(e))
    throw new MappingError(
      `Missing or invalid embedding vector at index ${invalidIndex} in Groq response.`,
      Provider.Groq
    )
  }

  return {
    embeddings: embeddings as number[][], // Cast after validation
    usage: mapUsageFromGroqEmbed(response.usage),
    model: response.model ?? modelUsed, // Use model from response if available
    rawResponse: response
  }
}
