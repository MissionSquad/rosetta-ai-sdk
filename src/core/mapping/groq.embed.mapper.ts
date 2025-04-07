// Mappers for Groq Embeddings
import {
  EmbeddingCreateParams,
  CreateEmbeddingResponse as GroqCreateEmbeddingResponse
} from 'groq-sdk/resources/embeddings'
import { EmbedParams, EmbedResult, Provider } from '../../types'
import { MappingError, UnsupportedFeatureError } from '../../errors'
import { mapTokenUsage } from './common.utils'

// --- Parameter Mapping ---

export function mapToGroqEmbedParams(params: EmbedParams): EmbeddingCreateParams {
  if (Array.isArray(params.input)) {
    // Groq API might support array input even if SDK types were initially restrictive
    console.warn('Mapping array input for Groq embeddings. Ensure the specific model supports batching.')
  }
  // Allow passing array input through
  const inputData = params.input

  if (params.dimensions) {
    // Throw error as this is explicitly unsupported
    throw new UnsupportedFeatureError(Provider.Groq, 'Embeddings dimensions parameter')
  }
  if (params.encodingFormat && params.encodingFormat !== 'float') {
    // Groq currently only supports float
    throw new UnsupportedFeatureError(Provider.Groq, `Embeddings encodingFormat: ${params.encodingFormat}`)
  }

  return {
    model: params.model!,
    input: inputData, // Pass string or array
    encoding_format: params.encodingFormat // Pass float or undefined
  }
}

// --- Result Mapping ---

export function mapFromGroqEmbedResponse(response: GroqCreateEmbeddingResponse, modelUsed: string): EmbedResult {
  if (!response?.data || !Array.isArray(response.data) || response.data.length === 0) {
    throw new MappingError(
      'Invalid or empty embedding data structure from Groq.',
      Provider.Groq,
      'mapFromGroqEmbedResponse'
    )
  }

  const embeddings = response.data.map(d => d?.embedding)

  if (embeddings.some(e => !e || !Array.isArray(e))) {
    const invalidIndex = embeddings.findIndex(e => !e || !Array.isArray(e))
    throw new MappingError(
      `Missing or invalid embedding vector at index ${invalidIndex} in Groq response.`,
      Provider.Groq
    )
  }

  return {
    embeddings: embeddings as number[][], // Cast after validation
    usage: mapTokenUsage(response.usage), // Use common utility
    model: response.model ?? modelUsed, // Use model from response if available
    rawResponse: response
  }
}
