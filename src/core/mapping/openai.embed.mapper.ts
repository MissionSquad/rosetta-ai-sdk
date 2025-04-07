import OpenAI from 'openai'
import { Embedding, EmbeddingCreateParams } from 'openai/resources/embeddings'
import { EmbedParams, EmbedResult, Provider } from '../../types'
import { MappingError } from '../../errors'
import { mapTokenUsage } from './common.utils'

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

  // Use common utility for usage mapping
  return {
    embeddings: embeddings as number[][], // Cast after validation
    usage: mapTokenUsage(response.usage), // Use specific embed usage mapper
    model: response.model ?? modelUsed, // Use model from response if available
    rawResponse: response
  }
}
