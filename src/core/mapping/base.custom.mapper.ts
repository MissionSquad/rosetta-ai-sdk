import {
  GenerateParams,
  GenerateResult,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  StreamChunk,
  ProviderKey,
  SpeechParams,
  AudioStreamChunk,
  RosettaTool,
  RosettaToolCallRequest
} from '../../types'
import {
  RosettaAIError,
  ProviderAPIError,
  UnsupportedFeatureError,
  ToolArgumentValidationError, // Import ToolArgumentValidationError
  MappingError // Import MappingError
} from '../../errors'
import { IProviderMapper } from './base.mapper'
import { CustomProviderConfig } from '../../types/custom.types'

/**
 * Abstract base class for creating custom provider mappers.
 * Provides default implementations that throw `UnsupportedFeatureError` for all
 * standard mapping and execution methods. Subclasses should override the methods
 * corresponding to the features their custom provider supports, typically the `execute*` methods.
 *
 * It also provides a default implementation for `wrapProviderError` and a helper
 * method `validateToolArguments` for validating tool call arguments using Zod.
 */
export abstract class BaseCustomMapper implements IProviderMapper {
  /** The unique key for this custom provider. */
  readonly provider: ProviderKey
  /** The configuration object passed during initialization. */
  protected readonly config: CustomProviderConfig

  constructor(config: CustomProviderConfig) {
    this.config = config
    this.provider = config.providerKey
  }

  // --- Default Implementations (Throw UnsupportedFeatureError) ---

  mapToProviderParams?(params: GenerateParams): any {
    // If executeGenerate is implemented, this might not be needed.
    // If needed, the subclass must override it.
    // For simplicity, we can return the original params if no mapping is done.
    console.warn(`[${this.provider}] mapToProviderParams not implemented. Passing raw params to executeGenerate.`)
    return params
  }

  mapFromProviderResponse?(_response: any, _modelId: string, _originalTools?: RosettaTool<any>[]): GenerateResult {
    throw new UnsupportedFeatureError(this.provider, 'mapFromProviderResponse (required if not using executeGenerate)')
  }

  mapProviderStream?(_stream: AsyncIterable<any>, _originalParams: GenerateParams): AsyncIterable<StreamChunk> {
    throw new UnsupportedFeatureError(this.provider, 'mapProviderStream (required if not using executeStream)')
  }

  mapToEmbedParams?(params: EmbedParams): any {
    console.warn(`[${this.provider}] mapToEmbedParams not implemented. Passing raw params to executeEmbed.`)
    return params
  }

  mapFromEmbedResponse?(_response: any, _modelId: string): EmbedResult {
    throw new UnsupportedFeatureError(this.provider, 'mapFromEmbedResponse (required if not using executeEmbed)')
  }

  mapToTranscribeParams?(params: TranscribeParams, _file: any): any {
    console.warn(`[${this.provider}] mapToTranscribeParams not implemented. Passing raw params to executeTranscribe.`)
    return params
  }

  mapFromTranscribeResponse?(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(
      this.provider,
      'mapFromTranscribeResponse (required if not using executeTranscribe)'
    )
  }

  mapToTranslateParams?(params: TranslateParams, _file: any): any {
    console.warn(`[${this.provider}] mapToTranslateParams not implemented. Passing raw params to executeTranslate.`)
    return params
  }

  mapFromTranslateResponse?(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(
      this.provider,
      'mapFromTranslateResponse (required if not using executeTranslate)'
    )
  }

  // --- Default Error Wrapping ---

  wrapProviderError(error: unknown, provider: ProviderKey): RosettaAIError {
    // Basic wrapping, subclasses can override for more specific error parsing
    if (error instanceof RosettaAIError) {
      return error // Don't re-wrap SDK errors
    }
    const message = error instanceof Error ? error.message : String(error ?? 'Unknown custom provider error')
    // Attempt to get status code if it's a fetch-like error response
    const statusCode =
      typeof error === 'object' && error !== null && 'status' in error ? Number(error.status) : undefined

    return new ProviderAPIError(message, provider as string, statusCode, undefined, undefined, error)
  }

  // --- Default Execution Methods (Throw UnsupportedFeatureError) ---
  // Subclasses MUST override these for the features they support.

  executeGenerate?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: GenerateParams // Updated signature
  ): Promise<GenerateResult> {
    throw new UnsupportedFeatureError(this.provider, 'generate (executeGenerate not implemented)')
  }

  executeStream?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: GenerateParams // Updated signature
  ): AsyncIterable<StreamChunk> {
    throw new UnsupportedFeatureError(this.provider, 'stream (executeStream not implemented)')
  }

  executeEmbed?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: EmbedParams
  ): Promise<EmbedResult> {
    throw new UnsupportedFeatureError(this.provider, 'embed (executeEmbed not implemented)')
  }

  executeGenerateSpeech?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: SpeechParams
  ): Promise<Buffer> {
    throw new UnsupportedFeatureError(this.provider, 'generateSpeech (executeGenerateSpeech not implemented)')
  }

  executeStreamSpeech?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: SpeechParams
  ): AsyncIterable<AudioStreamChunk> {
    throw new UnsupportedFeatureError(this.provider, 'streamSpeech (executeStreamSpeech not implemented)')
  }

  executeTranscribe?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: TranscribeParams
  ): Promise<TranscriptionResult> {
    throw new UnsupportedFeatureError(this.provider, 'transcribe (executeTranscribe not implemented)')
  }

  executeTranslate?(
    _mappedParams: any,
    _apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    _originalParams: TranslateParams
  ): Promise<TranscriptionResult> {
    throw new UnsupportedFeatureError(this.provider, 'translate (executeTranslate not implemented)')
  }

  // --- Helper Methods ---

  /**
   * Validates the arguments received for a tool call against the Zod schema defined in the original tool list.
   * This should be called within `executeGenerate` or `executeStream` when processing tool calls from the custom provider's API.
   *
   * @param call - An object containing the tool call details (name, arguments, optional ID). Arguments should be pre-parsed if received as a string.
   * @param originalTools - The array of `RosettaTool` definitions originally passed to the generate/stream request.
   * @throws {ToolArgumentValidationError} If the arguments fail validation against the corresponding Zod schema.
   * @throws {MappingError} If the tool definition cannot be found.
   */
  protected validateToolArguments(
    call: { name: string; arguments: any; id?: string },
    originalTools?: RosettaTool<any>[]
  ): void {
    if (!originalTools || originalTools.length === 0) {
      console.warn(
        `[${this.provider}] Skipping validation for tool '${call.name}': No original tool definitions provided.`
      )
      return
    }

    const toolDefinition = originalTools.find(t => t.function.name === call.name)

    if (!toolDefinition) {
      // Option 1: Throw an error if the model calls an undefined tool
      throw new MappingError(
        `Received tool call for unknown tool '${call.name}'. Ensure it was defined in the request.`,
        this.provider as string, // Cast ProviderKey to string
        'validateToolArguments'
      )
      // Option 2: Warn and skip validation
      // console.warn(`[${this.provider}] Skipping validation for unknown tool '${call.name}'.`);
      // return;
    }

    // Validate arguments using Zod schema
    const validationResult = toolDefinition.function.zodSchema.safeParse(call.arguments)

    if (!validationResult.success) {
      throw new ToolArgumentValidationError(
        `Arguments failed validation for tool '${call.name}'.`,
        validationResult.error.issues,
        call.name,
        call.id
      )
    }
    // Validation passed
  }

  /**
   * Example helper for making a basic fetch request.
   * Subclasses would likely need more sophisticated versions handling authentication, retries etc.
   */
  protected async basicFetch(url: string, options: RequestInit, apiKey?: string): Promise<Response> {
    const headers = new Headers(options.headers)
    if (apiKey) {
      // Example: Add Authorization header - adjust based on provider needs
      headers.set('Authorization', `Bearer ${apiKey}`)
    }

    try {
      const response = await fetch(url, { ...options, headers })
      if (!response.ok) {
        // Throw a generic error that wrapProviderError can handle
        const errorBody = await response.text().catch(() => 'Failed to read error body')
        const error = new Error(`API request failed with status ${response.status}: ${errorBody}`)
        ;(error as any).status = response.status // Attach status for wrapProviderError
        throw error
      }
      return response
    } catch (error) {
      // Wrap network errors etc.
      throw this.wrapProviderError(error, this.provider)
    }
  }

  /**
   * Parses tool call arguments based on the configured format.
   *
   * @param rawArguments - The arguments received from the provider API.
   * @param toolName - The name of the tool being called (for error context).
   * @param toolCallId - The ID of the tool call (for error context).
   * @returns The parsed arguments object.
   * @throws {MappingError} If parsing fails for 'jsonString' format.
   */
  protected parseToolArguments(rawArguments: any, toolName: string, toolCallId?: string): any {
    const format = this.config.toolConfig?.toolCallInputFormat ?? 'jsonString'

    if (format === 'jsonObject') {
      if (typeof rawArguments !== 'object' || rawArguments === null) {
        console.warn(
          `[${
            this.provider
          }] Expected tool arguments for '${toolName}' to be an object (format: jsonObject), but received ${typeof rawArguments}. Attempting to use as is.`
        )
      }
      return rawArguments ?? {} // Return as is, or empty object if null/undefined
    } else {
      // Default to 'jsonString'
      if (typeof rawArguments !== 'string') {
        console.warn(
          `[${
            this.provider
          }] Expected tool arguments for '${toolName}' to be a JSON string (format: jsonString), but received ${typeof rawArguments}. Attempting JSON.stringify.`
        )
        try {
          // Attempt to stringify non-string input before parsing, might indicate an issue upstream
          rawArguments = JSON.stringify(rawArguments)
        } catch (stringifyError) {
          throw new MappingError(
            `Failed to stringify unexpected argument type for tool '${toolName}' (ID: ${toolCallId}) before JSON parsing. Received type: ${typeof rawArguments}`,
            this.provider as string,
            'parseToolArguments',
            stringifyError
          )
        }
      }
      try {
        // Handle empty string case - return empty object
        return rawArguments.trim() === '' ? {} : JSON.parse(rawArguments)
      } catch (parseError) {
        throw new MappingError(
          `Failed to parse JSON string arguments for tool '${toolName}' (ID: ${toolCallId})`,
          this.provider as string,
          'parseToolArguments validation',
          parseError
        )
      }
    }
  }

  /**
   * Maps received tool calls from the provider's response format to RosettaToolCallRequest[].
   * This is a basic helper; subclasses might need more complex logic depending on the provider API structure.
   * It assumes the provider response contains an array of tool call objects.
   * It uses `parseToolArguments` and `validateToolArguments`.
   *
   * @param providerResponse - The raw response object from the custom provider API.
   * @param toolCallsPath - Path to the array of tool calls within the response (e.g., ['choices', 0, 'message', 'tool_calls']).
   * @param idPath - Path to the tool call ID within each tool call object.
   * @param namePath - Path to the tool name within each tool call object.
   * @param argsPath - Path to the tool arguments within each tool call object.
   * @param originalTools - The original RosettaTool definitions for validation.
   * @returns An array of RosettaToolCallRequest objects or undefined.
   * @throws {ToolArgumentValidationError} If validation fails.
   * @throws {MappingError} If parsing fails or paths are invalid.
   */
  protected mapAndValidateToolCallsHelper(
    providerResponse: any,
    toolCallsPath: (string | number)[],
    idPath: (string | number)[],
    namePath: (string | number)[],
    argsPath: (string | number)[],
    originalTools?: RosettaTool<any>[]
  ): RosettaToolCallRequest[] | undefined {
    const rawToolCalls = this.safeGetResponsePath(providerResponse, toolCallsPath)

    if (!Array.isArray(rawToolCalls) || rawToolCalls.length === 0) {
      return undefined
    }

    const mappedCalls: RosettaToolCallRequest[] = []
    for (const rawCall of rawToolCalls) {
      const id = this.safeGetResponsePath(rawCall, idPath) as string | undefined
      const name = this.safeGetResponsePath(rawCall, namePath) as string | undefined
      const rawArgs = this.safeGetResponsePath(rawCall, argsPath)

      if (!id || !name) {
        console.warn(`[${this.provider}] Skipping tool call due to missing id or name:`, rawCall)
        continue
      }

      // Parse arguments based on config
      const parsedArgs = this.parseToolArguments(rawArgs, name, id)

      // Validate arguments
      this.validateToolArguments({ name, arguments: parsedArgs, id }, originalTools)

      // Store the raw arguments string as required by RosettaToolCallRequest
      const argumentsString =
        this.config.toolConfig?.toolCallInputFormat === 'jsonObject' ? JSON.stringify(parsedArgs) : String(rawArgs)

      mappedCalls.push({
        id,
        type: 'function', // Assuming only function tools for now
        function: { name, arguments: argumentsString }
      })
    }

    return mappedCalls.length > 0 ? mappedCalls : undefined
  }

  /**
   * Safely accesses a nested property within a provider response object.
   * @param response The provider response object.
   * @param path An array of keys/indices representing the path.
   * @returns The value at the path, or undefined if not found.
   */
  protected safeGetResponsePath(response: any, path: (string | number)[]): any {
    let current = response
    for (const key of path) {
      if (current === null || typeof current === 'undefined') {
        return undefined
      }
      try {
        current = current[key]
      } catch (e) {
        return undefined // Handle potential errors during access
      }
    }
    return current
  }
}
