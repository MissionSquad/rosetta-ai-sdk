import {
  GenerateParams,
  GenerateResult,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  StreamChunk,
  RosettaTool,
  ProviderKey,
  SpeechParams,
  AudioStreamChunk,
  RosettaVoiceList
} from '../../types'
import { RosettaAIError } from '../../errors'
import { CustomProviderConfig } from '../../types/custom.types'

/**
 * Defines the common interface for provider-specific mapping logic.
 * Each provider implementation will handle the transformation between
 * RosettaAI's unified types and the provider's specific request/response formats.
 * Custom providers will primarily implement the `execute*` methods.
 */
// eslint-disable-next-line @typescript-eslint/interface-name-prefix
export interface IProviderMapper {
  /** The provider key (enum value or custom string) this mapper handles. */
  readonly provider: ProviderKey

  // --- Chat/Completion Mapping ---

  /**
   * Maps RosettaAI GenerateParams to the provider's specific parameters for chat completion.
   * Includes mapping messages, tools, and other parameters.
   * Responsible for mapping `RosettaToolResult` messages from history to the provider's format.
   * **Required for built-in providers.** Optional for custom providers if `executeGenerate` is implemented.
   *
   * @param params - The unified RosettaAI generation parameters, potentially including tools.
   * @returns The provider-specific parameters object (type `any` for flexibility).
   * @throws {MappingError} If required parameters are missing or invalid for the provider.
   * @throws {InvalidToolDefinitionError} If a provided tool definition is invalid (e.g., bad JSON schema).
   */
  mapToProviderParams?(params: GenerateParams): any // Provider-specific params type

  /**
   * Maps the provider's non-streaming chat completion response back to RosettaAI's GenerateResult.
   * Responsible for parsing tool call arguments, validating them against the `zodSchema`
   * from the original `RosettaTool` definitions, and throwing `ToolArgumentValidationError` on failure.
   * Returns the raw `RosettaToolCallRequest` (with string arguments) in the result for Phase 1.
   * **Required for built-in providers.** Optional for custom providers if `executeGenerate` is implemented.
   *
   * @param response - The raw response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @param originalTools - The original `RosettaTool` definitions passed to the generate call, used for validation.
   * @returns The unified RosettaAI GenerateResult.
   * @throws {MappingError} If the response structure is unexpected or argument parsing fails.
   * @throws {ToolArgumentValidationError} If tool arguments fail Zod validation.
   */
  mapFromProviderResponse?(response: any, modelId: string, originalTools?: RosettaTool<any>[]): GenerateResult // Provider-specific response type

  /**
   * Maps the provider's streaming chat completion response chunks to RosettaAI's StreamChunk union type.
   * Responsible for parsing streamed tool call arguments, validating them against the `zodSchema`
   * from the original `RosettaTool` definitions, and throwing `ToolArgumentValidationError` on failure
   * (which should be yielded as an error chunk).
   * Yields the raw tool call chunks (`tool_call_start`, `tool_call_delta`, `tool_call_done`) for Phase 1.
   * **Required for built-in providers.** Optional for custom providers if `executeStream` is implemented.
   *
   * @param stream - The async iterable stream object from the provider's SDK.
   * @param originalParams - The original `GenerateParams` passed to the stream call, used for model ID and tool validation.
   * @returns An async iterable yielding unified RosettaAI StreamChunk objects.
   * @throws {MappingError} If argument parsing fails during streaming.
   * @throws {ToolArgumentValidationError} If tool arguments fail Zod validation during streaming (yielded as error).
   */
  mapProviderStream?(stream: AsyncIterable<any>, originalParams: GenerateParams): AsyncIterable<StreamChunk> // Provider-specific stream chunk type

  // --- Embedding Mapping ---

  /**
   * Maps RosettaAI EmbedParams to the provider's specific parameters for embeddings.
   * **Required for built-in providers supporting embeddings.** Optional for custom providers if `executeEmbed` is implemented.
   * @param params - The unified RosettaAI embedding parameters.
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToEmbedParams?(params: EmbedParams): any // Provider-specific embed params type

  /**
   * Maps the provider's embedding response back to RosettaAI's EmbedResult.
   * **Required for built-in providers supporting embeddings.** Optional for custom providers if `executeEmbed` is implemented.
   * @param response - The raw embedding response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI EmbedResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromEmbedResponse?(response: any, modelId: string): EmbedResult // Provider-specific embed response type

  // --- Audio Mapping (STT/Translate) ---

  /**
   * Maps RosettaAI TranscribeParams to the provider's specific parameters for transcription.
   * **Required for built-in providers supporting transcription.** Optional for custom providers if `executeTranscribe` is implemented.
   * @param params - The unified RosettaAI transcription parameters.
   * @param file - The prepared audio file data (e.g., FileLike, Uploadable).
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToTranscribeParams?(params: TranscribeParams, file: any): any // Provider-specific STT params type

  /**
   * Maps the provider's transcription response back to RosettaAI's TranscriptionResult.
   * **Required for built-in providers supporting transcription.** Optional for custom providers if `executeTranscribe` is implemented.
   * @param response - The raw transcription response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI TranscriptionResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromTranscribeResponse?(response: any, modelId: string): TranscriptionResult // Provider-specific STT response type

  /**
   * Maps RosettaAI TranslateParams to the provider's specific parameters for translation.
   * **Required for built-in providers supporting translation.** Optional for custom providers if `executeTranslate` is implemented.
   * @param params - The unified RosettaAI translation parameters.
   * @param file - The prepared audio file data.
   * @returns The provider-specific parameters object.
   * @throws {MappingError | UnsupportedFeatureError} If parameters are invalid or unsupported.
   */
  mapToTranslateParams?(params: TranslateParams, file: any): any // Provider-specific Translate params type

  /**
   * Maps the provider's translation response back to RosettaAI's TranscriptionResult.
   * **Required for built-in providers supporting translation.** Optional for custom providers if `executeTranslate` is implemented.
   * @param response - The raw translation response object from the provider's SDK.
   * @param modelId - The model ID used for the request.
   * @returns The unified RosettaAI TranscriptionResult.
   * @throws {MappingError} If the response structure is unexpected.
   */
  mapFromTranslateResponse?(response: any, modelId: string): TranscriptionResult // Provider-specific Translate response type

  // --- Error Handling ---

  /**
   * Wraps a provider-specific error into a standardized RosettaAIError (usually ProviderAPIError).
   * **Required for all providers (built-in and custom).**
   * @param error - The error caught from the provider's SDK or the custom execution logic.
   * @param provider - The provider key associated with the error.
   * @returns A RosettaAIError instance.
   */
  wrapProviderError(error: unknown, provider: ProviderKey): RosettaAIError

  // --- Custom Provider Execution Methods (Optional) ---
  // These methods allow custom providers to bypass the standard client/mapping flow
  // and directly handle the API interaction.

  /**
   * **Optional:** Executes a non-streaming generation request for a custom provider.
   * If implemented, this method is called instead of the standard `getClientForProvider` and SDK calls.
   * It should handle the entire API interaction, including authentication and error handling.
   * The result should be mapped back to `GenerateResult` *before* returning.
   * Implementations should handle tool definition sending and tool call receiving/validation based on `providerConfig.toolConfig`.
   *
   * @param mappedParams - The parameters already mapped by `mapToProviderParams` (if implemented, otherwise raw `GenerateParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `GenerateParams` passed to the SDK method (contains `tools` for validation).
   * @returns A promise resolving to the unified `GenerateResult`.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   * @throws {ToolArgumentValidationError} If received tool arguments fail validation.
   */
  executeGenerate?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams // Pass original params for access to tools
  ): Promise<GenerateResult>

  /**
   * **Optional:** Executes a streaming generation request for a custom provider.
   * If implemented, this method is called instead of the standard `getClientForProvider` and SDK calls.
   * It should handle the API interaction and yield `StreamChunk` objects directly.
   * Implementations should handle tool definition sending and tool call receiving/validation based on `providerConfig.toolConfig`.
   *
   * @param mappedParams - The parameters already mapped by `mapToProviderParams` (if implemented, otherwise raw `GenerateParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `GenerateParams` passed to the SDK method (contains `tools` for validation).
   * @returns An async iterable yielding unified `StreamChunk` objects.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails during setup (yield error chunk for stream errors).
   * @throws {ToolArgumentValidationError} If received tool arguments fail validation (yield error chunk).
   */
  executeStream?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams, // Pass original params for access to tools
    abortSignal?: AbortSignal
  ): AsyncIterable<StreamChunk>

  /**
   * **Optional:** Executes an embedding request for a custom provider.
   *
   * @param mappedParams - The parameters already mapped by `mapToEmbedParams` (if implemented, otherwise raw `EmbedParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `EmbedParams` passed to the SDK method.
   * @returns A promise resolving to the unified `EmbedResult`.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   */
  executeEmbed?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: EmbedParams
  ): Promise<EmbedResult>

  /**
   * **Optional:** Executes a Text-to-Speech request for a custom provider (non-streaming).
   *
   * @param mappedParams - The parameters (likely raw `SpeechParams` as TTS is less common for built-ins).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `SpeechParams` passed to the SDK method.
   * @returns A promise resolving to the audio data as a Buffer.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   */
  executeGenerateSpeech?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: SpeechParams
  ): Promise<Buffer>

  /**
   * **Optional:** Executes a streaming Text-to-Speech request for a custom provider.
   *
   * @param mappedParams - The parameters (likely raw `SpeechParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `SpeechParams` passed to the SDK method.
   * @returns An async iterable yielding `AudioStreamChunk` objects.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails during setup.
   */
  executeStreamSpeech?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: SpeechParams,
    abortSignal?: AbortSignal
  ): AsyncIterable<AudioStreamChunk>

  /**
   * **Optional:** Executes an audio transcription request for a custom provider.
   *
   * @param mappedParams - The parameters already mapped by `mapToTranscribeParams` (if implemented, otherwise raw `TranscribeParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `TranscribeParams` passed to the SDK method.
   * @returns A promise resolving to the unified `TranscriptionResult`.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   */
  executeTranscribe?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: TranscribeParams
  ): Promise<TranscriptionResult>

  /**
   * **Optional:** Executes an audio translation request for a custom provider.
   *
   * @param mappedParams - The parameters already mapped by `mapToTranslateParams` (if implemented, otherwise raw `TranslateParams`).
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @param originalParams - The original `TranslateParams` passed to the SDK method.
   * @returns A promise resolving to the unified `TranscriptionResult`.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   */
  executeTranslate?(
    mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: TranslateParams
  ): Promise<TranscriptionResult>

  /**
   * **Optional:** Lists voices for a custom provider (e.g., TTS voice catalog).
   *
   * @param apiKey - The API key for the custom provider.
   * @param providerConfig - The full configuration object for this custom provider.
   * @returns A promise resolving to a provider-agnostic RosettaVoiceList.
   * @throws {ProviderAPIError | RosettaAIError} If the custom API call fails.
   */
  executeListVoices?(
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig
  ): Promise<RosettaVoiceList>
}
