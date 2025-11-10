import { ProviderKey } from './common.types' // Import ProviderKey
import { ModelListingSourceConfig } from './models.types'
import { CustomProviderConfig } from './custom.types'

/**
 * Optional provider-specific configuration settings that can override global defaults
 * or provide parameters unique to a provider (like Azure deployment IDs).
 */
export interface ProviderOptions {
  /** Base URL override for the provider's API endpoint. */
  baseURL?: string
  /** Azure OpenAI specific deployment ID for chat/completion models. Overrides `azureOpenAIDefaultChatDeploymentName`. */
  azureChatDeploymentId?: string
  /** Azure OpenAI specific deployment ID for embedding models. Overrides `azureOpenAIDefaultEmbeddingDeploymentName`. */
  azureEmbeddingDeploymentId?: string
  /** Google API version (e.g., 'v1beta', 'v1alpha', 'v1'). Affects available features. */
  googleApiVersion?: 'v1beta' | 'v1alpha' | 'v1'
  /** Whether to use Vertex AI instead of Gemini API for Google provider. */
  googleVertexAI?: boolean
  /** Google Cloud project ID for Vertex AI. Required if googleVertexAI is true. */
  googleVertexAIProject?: string
  /** Google Cloud location/region for Vertex AI (e.g., 'us-central1'). Required if googleVertexAI is true. */
  googleVertexAILocation?: string
  /**
   * Safety settings for Google provider. Allows customization of content filtering thresholds.
   * If not specified, defaults to BLOCK_MEDIUM_AND_ABOVE for all categories.
   *
   * @example
   * ```typescript
   * {
   *   googleSafetySettings: [
   *     { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
   *     { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' }
   *   ]
   * }
   * ```
   */
  googleSafetySettings?: Array<{
    category: 'HARM_CATEGORY_HARASSMENT' | 'HARM_CATEGORY_HATE_SPEECH' | 'HARM_CATEGORY_SEXUALLY_EXPLICIT' | 'HARM_CATEGORY_DANGEROUS_CONTENT'
    threshold: 'BLOCK_NONE' | 'BLOCK_ONLY_HIGH' | 'BLOCK_MEDIUM_AND_ABOVE' | 'BLOCK_LOW_AND_ABOVE'
  }>
  // Add other provider-specific config options here as needed
}

/**
 * Configuration object for the RosettaAI client.
 * API keys can be provided here or loaded from standard environment variables.
 */
export interface RosettaAIConfig {
  /** API key for Anthropic. Falls back to `process.env.ANTHROPIC_API_KEY`. */
  anthropicApiKey?: string
  /** API key for Google Generative AI (Gemini API). Falls back to `process.env.GOOGLE_API_KEY`.
   * Not required if using Vertex AI with Application Default Credentials. */
  googleApiKey?: string
  /** Google Cloud project ID for Vertex AI. Falls back to `process.env.GOOGLE_CLOUD_PROJECT`. */
  googleVertexAIProject?: string
  /** Google Cloud location for Vertex AI. Falls back to `process.env.GOOGLE_CLOUD_LOCATION`. */
  googleVertexAILocation?: string
  /** API key for Groq. Falls back to `process.env.GROQ_API_KEY`. */
  groqApiKey?: string
  /** API key for Standard OpenAI. Falls back to `process.env.OPENAI_API_KEY`. Ignored if Azure config is provided and valid. */
  openaiApiKey?: string

  /** Azure OpenAI API key. Falls back to `process.env.AZURE_OPENAI_API_KEY`. */
  azureOpenAIApiKey?: string
  /** Azure OpenAI endpoint URL. Falls back to `process.env.AZURE_OPENAI_ENDPOINT`. Required if using Azure. */
  azureOpenAIEndpoint?: string
  /** Default Azure OpenAI deployment name/ID for chat models. Falls back to `process.env.AZURE_OPENAI_DEPLOYMENT_NAME`. */
  azureOpenAIDefaultChatDeploymentName?: string
  /** Default Azure OpenAI deployment name/ID for embedding models. Falls back to `process.env.ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`. */
  azureOpenAIDefaultEmbeddingDeploymentName?: string
  /** Azure OpenAI API Version string (e.g., '2024-05-01-preview'). Falls back to `process.env.AZURE_OPENAI_API_VERSION`. */
  azureOpenAIApiVersion?: string

  /** Optional provider-specific configurations applied to all requests for that provider unless overridden per-request. */
  providerOptions?: Partial<Record<ProviderKey, ProviderOptions>> // Use ProviderKey

  /** Default chat/completion model ID to use if not specified in request, keyed by provider. E.g., `{ openai: 'gpt-4o-mini', 'my-custom': 'model-x' }`. */
  defaultModels?: Partial<Record<ProviderKey, string>> // Use ProviderKey
  /** Default embedding model ID to use if not specified, keyed by provider. E.g., `{ openai: 'text-embedding-3-small' }`. */
  defaultEmbeddingModels?: Partial<Record<ProviderKey, string>> // Use ProviderKey
  /** Default TTS model ID to use if not specified, keyed by provider. E.g., `{ openai: 'tts-1' }`. */
  defaultTtsModels?: Partial<Record<ProviderKey, string>> // Use ProviderKey
  /** Default STT model ID to use if not specified, keyed by provider. E.g., `{ openai: 'whisper-1' }`. */
  defaultSttModels?: Partial<Record<ProviderKey, string>> // Use ProviderKey

  /** Default maximum retries for API calls (where supported by underlying SDK). Defaults to 2. */
  defaultMaxRetries?: number
  /** Default request timeout in milliseconds (where supported by underlying SDK). Defaults to 60000 (1 minute). */
  defaultTimeoutMs?: number

  /** Optional configuration for how model lists are retrieved per provider. */
  modelListingConfig?: Partial<Record<ProviderKey, ModelListingSourceConfig>> // Use ProviderKey

  /** Optional array of custom provider configurations. */
  customProviders?: CustomProviderConfig[]
  /**
   * If true, the `generate` method will transform the response from any provider
   * into the standard OpenAI Chat Completion format and attach it to the `GenerateResult`.
   * @default false
   */
  openAICompletions?: boolean
}
