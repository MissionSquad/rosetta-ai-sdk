// Mappers for OpenAI Embeddings

import OpenAI from 'openai'
import { Embedding, EmbeddingCreateParams } from 'openai/resources/embeddings' // Import CreateEmbeddingResponse correctly
import { EmbedParams, EmbedResult, Provider, TokenUsage } from '../../types'
import { MappingError } from '../../errors'

// --- Parameter Mapping ---

export function mapToOpenAIEmbedParams(params: EmbedParams): EmbeddingCreateParams {
  // Input can be string | string[] | number[] | number[][]
  // Ensure input matches expected type
  let inputData: EmbeddingCreateParams['input']
  if (typeof params.input === 'string' || Array.isArray(params.input)) {
    inputData = params.input // string or string[] is fine
  } else {
    // Handle potential number arrays if Rosetta ever supports token inputs directly
    throw new MappingError('Invalid input type for OpenAI embeddings. Expected string or string[].', Provider.OpenAI)
  }

  return {
    model: params.model!,
    input: inputData,
    encoding_format: params.encodingFormat,
    dimensions: params.dimensions
  }
}

// Define specific usage type for Embedding response based on OpenAI's type
type EmbeddingUsage = OpenAI.Embeddings.CreateEmbeddingResponse['usage']

function mapUsageFromOpenAIEmbed(usage: EmbeddingUsage | undefined | null): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.prompt_tokens,
    totalTokens: usage.total_tokens,
    completionTokens: undefined // Embeddings don't have completion tokens
  }
}

// --- Result Mapping ---

export function mapFromOpenAIEmbedResponse(
  response: OpenAI.Embeddings.CreateEmbeddingResponse,
  modelUsed: string
): EmbedResult {
  // Validate the response structure
  if (!Array.isArray(response?.data) || response.data.length === 0) {
    throw new MappingError(
      'Invalid or empty embedding data structure from OpenAI.',
      Provider.OpenAI,
      'mapFromOpenAIEmbedResponse'
    )
  }

  // Map and validate embeddings
  const embeddings = response.data.map((d: Embedding) => d?.embedding) // Access embedding property
  if (embeddings.some(e => !e || !Array.isArray(e))) {
    const invalidIndex = embeddings.findIndex(e => !e || !Array.isArray(e))
    throw new MappingError(
      `Missing or invalid embedding vector at index ${invalidIndex} in OpenAI response.`,
      Provider.OpenAI
    )
  }

  // Use correct usage type for embeddings
  return {
    embeddings: embeddings as number[][], // Cast after validation
    usage: mapUsageFromOpenAIEmbed(response.usage), // Use specific embed usage mapper
    model: response.model ?? modelUsed, // Use model from response if available
    rawResponse: response
  }
}
