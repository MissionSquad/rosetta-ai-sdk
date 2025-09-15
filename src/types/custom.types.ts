import { IProviderMapper } from '../core/mapping/base.mapper'

/**
 * Defines the configuration structure for a custom AI provider.
 */
export interface CustomProviderConfig {
  /**
   * A unique string identifier for this custom provider.
   * Must not conflict with built-in Provider enum values or other custom provider keys.
   */
  providerKey: string

  /**
   * The API key for the custom provider.
   * If not provided directly, the SDK might attempt to load it from an environment variable
   * named `${providerKey.toUpperCase().replace(/-/g, '_')}_API_KEY` (e.g., `MY_CUSTOM_PROVIDER_API_KEY`).
   */
  apiKey?: string

  /**
   * The constructor function for the custom provider's mapper class.
   * This class must implement the `IProviderMapper` interface.
   * The constructor will receive this `CustomProviderConfig` object as its argument.
   *
   * @example
   * import { MyCustomMapper } from './my-custom-mapper';
   * // ...
   * const config: CustomProviderConfig = {
   *   providerKey: 'my-custom-provider',
   *   mapper: MyCustomMapper,
   *   // ... other config
   * };
   */
  mapper: new (config: CustomProviderConfig) => IProviderMapper

  /**
   * An array indicating the core features supported by this custom provider.
   * This helps the SDK perform upfront checks and potentially throw `UnsupportedFeatureError`.
   */
  supportedFeatures: Array<
    | 'generate'
    | 'stream'
    | 'embed'
    | 'tts' // Text-to-Speech
    | 'stt' // Speech-to-Text (Transcription)
    | 'translate' // Speech Translation
    | 'tool_use'
    | 'image_input'
    | 'json_mode'
    | 'list_models' // Model listing
    | 'list_voices' // Voice listing
    // Add other potential features as needed
  >

  /**
   * Optional configuration related to tool usage for this provider.
   * This guides the custom mapper on how to format tool definitions for the API,
   * how to interpret tool call requests from the API, and how to format tool results
   * when sending them back.
   */
  toolConfig?: {
    /**
     * Specifies how the custom provider's API expects tool definitions to be sent.
     * - `jsonSchema`: Send the standard JSON Schema object from `RosettaTool.function.parameters`.
     * - `simplified`: Send a custom, potentially simplified format (mapper must implement).
     * - `none`: The provider does not support receiving tool definitions explicitly.
     * @default 'jsonSchema'
     */
    toolDefinitionFormat?: 'jsonSchema' | 'simplified' | 'none'

    /**
     * Specifies the format in which the custom provider's API returns tool call arguments.
     * - `jsonString`: Arguments are received as a single JSON string (needs parsing).
     * - `jsonObject`: Arguments are received as a pre-parsed JSON object.
     * @default 'jsonString'
     */
    toolCallInputFormat?: 'jsonString' | 'jsonObject'

    /**
     * Specifies how the custom provider's API expects tool results (from the user) to be sent back.
     * - `jsonString`: Send results as a single JSON string within the designated content field.
     * - `jsonObject`: Send results as a JSON object within the designated content field/structure.
     * - `separateFields`: Send results using distinct fields for tool ID, content, error status (mapper must implement).
     * @default 'jsonString'
     */
    toolResultFormat?: 'jsonString' | 'jsonObject' | 'separateFields'

    /**
     * Indicates if the provider requires a specific tool_choice parameter format different
     * from the standard ones ('auto', 'none', 'required', specific function). If true,
     * the mapper needs to handle the mapping from Rosetta's `toolChoice` parameter.
     * @default false
     */
    requiresCustomToolChoiceFormat?: boolean

    /** Any other provider-specific flags or configurations related to tool handling. */
    [key: string]: any
  }

  /**
   * Optional default model ID to use for this custom provider if not specified in the request.
   */
  defaultModel?: string

  /**
   * Optional default embedding model ID for this custom provider.
   */
  defaultEmbeddingModel?: string

  /**
   * Optional default TTS model ID for this custom provider.
   */
  defaultTtsModel?: string

  /**
   * Optional default STT model ID for this custom provider.
   */
  defaultSttModel?: string

  /**
   * Optional base URL for the custom provider's API endpoint.
   * This might be used by the custom mapper's `execute*` methods and for default model listing.
   */
  baseURL?: string

  /**
   * Optional: The relative path from `baseURL` to the model listing endpoint.
   * If not provided, defaults to `/models`. Ignored if `modelListUrl` is set.
   * Example: `/openai/models` or `/v1/models`
   */
  modelListPath?: string

  /**
   * Optional: The absolute URL for the model listing endpoint.
   * If provided, this overrides `baseURL` and `modelListPath` for model listing requests.
   * Example: `https://api.customprovider.com/api/inventory/models`
   */
  modelListUrl?: string

  /**
   * Optional default maximum retries for API calls made by the custom mapper.
   */
  defaultMaxRetries?: number

  /**
   * Optional default request timeout in milliseconds for API calls made by the custom mapper.
   */
  defaultTimeoutMs?: number

  /**
   * Any other provider-specific configuration needed by the custom mapper.
   * The mapper implementation can access these via the `config` object passed to its constructor.
   */
  [key: string]: any
}
