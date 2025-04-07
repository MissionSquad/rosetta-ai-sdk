import {
  GenerateParams,
  GenerateResult,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  StreamChunk,
  Provider
} from '../../types'
import { RosettaAIError } from '../../errors'

/**
 * Defines the common interface for provider-specific mapping logic.
 * Each provider implementation will handle the transformation between
 * RosettaAI's unified types and the provider's specific request/response formats.
 */
export interface IProviderMapper {
  /** The provider this mapper handles. */
  readonly provider: Provider

  // --- Chat/Completion Mapping ---

  /**
   * Maps RosettaAI GenerateParams to the provider's specific parameters for chat completion.
   * @param params - The unified RosettaAI generation parameters.
   * @returns The provider-specific parameters object (type `any` for flexibility).
   * @throws {MappingError} If required parameters are missing or invalid for the provider.
   */
  mapToProviderParams(params: GenerateParams): any // Provider-specific params type

  /**
   * Maps the provider's non-streaming chat completion response back to RosettaAI's GenerateResult.
   * @param response - The raw response object from the provider's SDK.
   * @param modelId - The model ID used for the request (needed as it's not always in the response).
   * @returns The unified RosettaAI GenerateResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromProviderResponse(response: any, modelId: string): GenerateResult // Provider-specific response type

  /**
   * Maps the provider's streaming chat completion response chunks to RosettaAI's StreamChunk union type.
   * @param stream - The async iterable stream object from the provider's SDK.
   * @returns An async iterable yielding unified RosettaAI StreamChunk objects.
   */
  mapProviderStream(stream: AsyncIterable<any>): AsyncIterable<StreamChunk> // Provider-specific stream chunk type

  // --- Embedding Mapping ---

  /**
   * Maps RosettaAI EmbedParams to the provider's specific parameters for embeddings.
   * @param params - The unified RosettaAI embedding parameters.
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToEmbedParams(params: EmbedParams): any // Provider-specific embed params type

  /**
   * Maps the provider's embedding response back to RosettaAI's EmbedResult.
   * @param response - The raw embedding response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI EmbedResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromEmbedResponse(response: any, modelId: string): EmbedResult // Provider-specific embed response type

  // --- Audio Mapping (STT/Translate) ---

  /**
   * Maps RosettaAI TranscribeParams to the provider's specific parameters for transcription.
   * @param params - The unified RosettaAI transcription parameters.
   * @param file - The prepared audio file data (e.g., FileLike, Uploadable).
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToTranscribeParams(params: TranscribeParams, file: any): any // Provider-specific STT params type

  /**
   * Maps the provider's transcription response back to RosettaAI's TranscriptionResult.
   * @param response - The raw transcription response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI TranscriptionResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromTranscribeResponse(response: any, modelId: string): TranscriptionResult // Provider-specific STT response type

  /**
   * Maps RosettaAI TranslateParams to the provider's specific parameters for translation.
   * @param params - The unified RosettaAI translation parameters.
   * @param file - The prepared audio file data.
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToTranslateParams(params: TranslateParams, file: any): any // Provider-specific Translate params type

  /**
   * Maps the provider's translation response back to RosettaAI's TranscriptionResult.
   * @param response - The raw translation response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI TranscriptionResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromTranslateResponse(response: any, modelId: string): TranscriptionResult // Provider-specific Translate response type

  // --- Error Handling ---

  /**
   * Wraps a provider-specific error into a standardized RosettaAIError (usually ProviderAPIError).
   * @param error - The error caught from the provider's SDK.
   * @param provider - The provider associated with the error.
   * @returns A RosettaAIError instance.
   */
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError
}
