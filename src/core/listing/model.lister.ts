// src/core/listing/model.lister.ts
import Groq from 'groq-sdk'
import {
  RosettaModelList,
  RosettaModel,
  Provider,
  ModelListingSourceConfig,
  ModelListingSourceType,
  ProviderKey,
  CustomProviderConfig
} from '../../types'
import {
  ProviderAPIError,
  MappingError,
  ConfigurationError,
  RosettaAIError,
  UnsupportedFeatureError
} from '../../errors'
import { anthropicStaticModels } from './static-data/anthropic.models'
import { fetchAndValidateModelsFromApi } from './fetch.utils'

// Internal function to handle listing for a specific provider
export async function listModelsForProvider(
  providerKey: ProviderKey,
  config: {
    // Pass necessary parts of RosettaAIConfig and clients
    sourceConfig?: ModelListingSourceConfig
    apiKey?: string
    groqClient?: Groq // Pass Groq client if available for built-in Groq
    customConfig?: CustomProviderConfig // Pass custom config if providerKey is custom
  }
): Promise<RosettaModelList> {
  const { sourceConfig, apiKey, groqClient, customConfig } = config
  let sourceType: ModelListingSourceType
  let finalUrl: string | undefined
  const isCustom = !!customConfig

  // --- Determine Source Type and URL ---
  if (sourceConfig) {
    sourceType = sourceConfig.type
    if (sourceConfig.type === 'apiEndpoint') {
      finalUrl = sourceConfig.url
    }
  } else if (isCustom) {
    // Default logic for custom providers if no sourceConfig override
    sourceType = 'apiEndpoint' // Assume API endpoint for custom unless overridden
    if (customConfig.modelListUrl) {
      finalUrl = customConfig.modelListUrl
    } else if (customConfig.baseURL) {
      const path = customConfig.modelListPath ?? '/models' // Default path
      // Basic URL joining (consider using URL constructor for robustness)
      const baseUrlTrimmed = customConfig.baseURL.endsWith('/')
        ? customConfig.baseURL.slice(0, -1)
        : customConfig.baseURL
      const pathTrimmed = path.startsWith('/') ? path : `/${path}`
      finalUrl = baseUrlTrimmed + pathTrimmed
    } else {
      throw new ConfigurationError(
        `Cannot determine model list URL for custom provider '${providerKey}'. Configure 'modelListUrl' or 'baseURL' in CustomProviderConfig.`
      )
    }
  } else {
    // Default logic for built-in providers if no sourceConfig override
    const provider = providerKey as Provider // Safe cast as !isCustom
    if (provider === Provider.Groq && groqClient) sourceType = 'sdkMethod'
    else if (provider === Provider.Anthropic) sourceType = 'staticList'
    else if (provider === Provider.Google) {
      sourceType = 'apiEndpoint'
      finalUrl = 'https://generativelanguage.googleapis.com/v1beta/openai/models' // Updated Google URL
    } else if (provider === Provider.OpenAI) {
      sourceType = 'apiEndpoint'
      finalUrl = 'https://api.openai.com/v1/models' // Standard OpenAI URL
    } else {
      throw new ConfigurationError(`Model listing source type for provider ${providerKey} could not be determined.`)
    }
  }

  // --- Execute based on source type ---
  try {
    switch (sourceType) {
      case 'staticList':
        if (providerKey !== Provider.Anthropic) {
          throw new ConfigurationError(`Static list is only configured for Anthropic, not ${providerKey}.`)
        }
        // Return a deep copy to prevent modification of the original static data
        return JSON.parse(JSON.stringify(anthropicStaticModels))

      case 'sdkMethod':
        if (providerKey !== Provider.Groq || !groqClient) {
          throw new ConfigurationError(`SDK method listing is only configured for Groq with an active client.`)
        }
        const groqResponse = await groqClient.models.list()
        // Map Groq's response - handle potential extra fields AT RUNTIME
        const groqModels: RosettaModel[] = groqResponse.data.map(
          (groqModel: any): RosettaModel => ({
            id: groqModel.id,
            object: 'model',
            owned_by: groqModel.owned_by,
            created: typeof groqModel.created === 'number' ? groqModel.created : undefined,
            active: typeof groqModel.active === 'boolean' ? groqModel.active : undefined,
            context_window: typeof groqModel.context_window === 'number' ? groqModel.context_window : undefined,
            public_apps: groqModel.public_apps ?? undefined,
            max_completion_tokens:
              typeof groqModel.max_completion_tokens === 'number' ? groqModel.max_completion_tokens : undefined,
            // No 'properties' observed in Groq example, set undefined
            properties: undefined,
            provider: providerKey,
            rawData: groqModel // Store original
          })
        )
        return { object: 'list', data: groqModels }

      case 'apiEndpoint':
        if (!finalUrl) {
          throw new ConfigurationError(`API endpoint URL for ${providerKey} could not be determined.`)
        }
        // API key check for built-in providers. For custom providers, API key might be optional (e.g., local server)
        if (!isCustom && !apiKey) {
          throw new ConfigurationError(`API key for ${providerKey} is required but missing for model listing.`)
        }
        // Call fetch utility, passing the providerKey (built-in or custom)
        return await fetchAndValidateModelsFromApi(finalUrl, providerKey, apiKey)

      default:
        const _exhaustiveCheck: never = sourceType
        throw new ConfigurationError(`Unsupported model listing source type: ${_exhaustiveCheck}`)
    }
  } catch (error) {
    // Don't re-wrap RosettaAIError types
    if (
      error instanceof ProviderAPIError ||
      error instanceof MappingError ||
      error instanceof ConfigurationError ||
      error instanceof UnsupportedFeatureError || // Include UnsupportedFeatureError
      error instanceof RosettaAIError // Catch base SDK error too
    ) {
      throw error
    }
    // Wrap other errors
    const message = error instanceof Error ? error.message : String(error)
    throw new ProviderAPIError(
      `Failed to list models for ${providerKey} using ${sourceType}: ${message}`,
      providerKey, // Pass the correct providerKey
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
