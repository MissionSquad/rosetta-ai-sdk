import { Provider } from './common.types'

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
  /** Google API version (e.g., 'v1beta', 'v1alpha'). Affects available features. */
  googleApiVersion?: 'v1beta' | 'v1alpha'
  // Add other provider-specific config options here as needed
}

/**
 * Configuration object for the RosettaAI client.
 * API keys can be provided here or loaded from standard environment variables.
 */
export interface RosettaAIConfig {
  /** API key for Anthropic. Falls back to `process.env.ANTHROPIC_API_KEY`. */
  anthropicApiKey?: string
  /** API key for Google Generative AI. Falls back to `process.env.GOOGLE_API_KEY`. */
  googleApiKey?: string
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
  providerOptions?: Partial<Record<Provider, ProviderOptions>>

  /** Default chat/completion model ID to use if not specified in request, keyed by provider. E.g., `{ openai: 'gpt-4o-mini' }`. */
  defaultModels?: Partial<Record<Provider, string>>
  /** Default embedding model ID to use if not specified, keyed by provider. E.g., `{ openai: 'text-embedding-3-small' }`. */
  defaultEmbeddingModels?: Partial<Record<Provider, string>>
  /** Default TTS model ID to use if not specified, keyed by provider. E.g., `{ openai: 'tts-1' }`. */
  defaultTtsModels?: Partial<Record<Provider, string>>
  /** Default STT model ID to use if not specified, keyed by provider. E.g., `{ openai: 'whisper-1' }`. */
  defaultSttModels?: Partial<Record<Provider, string>>

  /** Default maximum retries for API calls (where supported by underlying SDK). Defaults to 2. */
  defaultMaxRetries?: number
  /** Default request timeout in milliseconds (where supported by underlying SDK). Defaults to 60000 (1 minute). */
  defaultTimeoutMs?: number
}
