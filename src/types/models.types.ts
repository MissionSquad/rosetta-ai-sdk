import { Provider } from './common.types' // Assuming Provider enum exists in common.types.ts

/**
 * Optional properties specific to certain models, primarily seen in Anthropic/Groq responses.
 */
export interface RosettaModelProperties {
  /** A brief description of the model's capabilities or intended use. */
  description?: string
  /** A summary of the model's strengths or key features. */
  strengths?: string
  /** Indicates if the model has strong multilingual capabilities. */
  multilingual?: boolean
  /** Indicates if the model supports vision (image) input. */
  vision?: boolean
  /** Anthropic specific: Indicates support for extended thinking steps. */
  extended_thinking?: boolean
  /** Anthropic specific: A qualitative measure of the model's latency. */
  comparative_latency?: string
  /** Anthropic specific: Estimated cost per million input tokens. */
  cost_input_mtok?: number
  /** Anthropic specific: Estimated cost per million output tokens. */
  cost_output_mtok?: number
  /** Anthropic specific: The cutoff date for the model's training data. */
  training_data_cutoff?: string
  /** Anthropic specific: Maximum completion tokens potentially available under specific conditions. */
  extended_max_completion_tokens?: number
  // Add any other observed optional properties here
}

/**
 * Represents a single AI model available through a provider,
 * combining common fields and provider-specific optional details.
 */
export interface RosettaModel {
  /** Unique identifier for the model (provider-specific format). */
  id: string
  /** Object type, always "model". */
  object: 'model'
  /** The organization that owns the model (e.g., "openai", "google", "anthropic", "meta"). */
  owned_by: string
  /** Unix timestamp (seconds) when the model was created, or null if not available. */
  created?: number | null
  /** Indicates if the model is currently active or available for use. */
  active?: boolean
  /** The maximum context window size in tokens. */
  context_window?: number
  /** Indicates availability in public applications (observed as null, kept for potential future use). */
  public_apps?: string | null // Keeping as string | null for flexibility, though examples show null
  /** The maximum number of tokens that can be generated in a single completion request. */
  max_completion_tokens?: number
  /** Additional, often provider-specific, properties and capabilities of the model. */
  properties?: RosettaModelProperties
  /** The provider this model belongs to. Added by RosettaAI for context. */
  readonly provider: Provider // Added by our SDK logic
  /** Store the raw data from the provider for debugging or advanced use */
  readonly rawData?: Record<string, any> // Store the original object
}

/**
 * Represents the standard list response structure for models.
 */
export interface RosettaModelList {
  /** Object type, always "list". */
  object: 'list'
  /** An array of available models. */
  data: RosettaModel[]
}

/**
 * Defines the source and configuration for fetching model lists for a provider.
 * - `sdkMethod`: Use a dedicated method from the provider's SDK (e.g., `groq.models.list()`).
 * - `apiEndpoint`: Fetch the list from a specified REST API endpoint.
 * - `staticList`: Use a predefined, hardcoded list within the SDK (e.g., for Anthropic).
 */
export type ModelListingSourceType = 'sdkMethod' | 'apiEndpoint' | 'staticList'

/** Configuration for using an SDK method to list models. */
export interface SdkMethodSource {
  type: 'sdkMethod'
  // No specific config needed here, logic will use the existing provider client.
  // Method to call is implicitly known (e.g., groq.models.list).
}

/** Configuration for fetching the model list from an API endpoint. */
export interface ApiEndpointSource {
  type: 'apiEndpoint'
  /** The URL of the API endpoint to fetch the model list from. */
  url: string
  /** Optional Zod schema or validation function for the raw API response */
  // validationSchema?: ZodSchema<any> | ((data: any) => boolean); // Example using Zod
}

/** Configuration for using a static, predefined list of models. */
export interface StaticListSource {
  type: 'staticList'
  // Static data will be stored internally, no external config needed here.
}

/**
 * Union type representing the possible configurations for listing models for a provider.
 * This can be set globally in `RosettaAIConfig` or passed per-call to `listModels`.
 */
export type ModelListingSourceConfig = SdkMethodSource | ApiEndpointSource | StaticListSource
