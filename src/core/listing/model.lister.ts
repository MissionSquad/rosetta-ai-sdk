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
    isAzureOpenAI?: boolean // Flag indicating if we're using Azure OpenAI
  }
): Promise<RosettaModelList> {
  const { sourceConfig, apiKey, groqClient, customConfig, isAzureOpenAI } = config
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
      // For OpenAI, handle Azure OpenAI differently
      sourceType = 'apiEndpoint'

      // If no URL is provided in sourceConfig and we're using Azure OpenAI,
      // we should have received a URL from the caller that includes the Azure endpoint
      if (isAzureOpenAI) {
        // If we don't have a URL at this point, it means the caller didn't provide one
        // This is unexpected since the caller should have created a sourceConfig with the Azure URL
        if (!finalUrl) {
          console.warn(
            'RosettaAI Warning: Azure OpenAI is active but no deployments endpoint URL was provided. Falling back to standard OpenAI URL, which will likely fail with Azure key.'
          )
          finalUrl = 'https://api.openai.com/v1/models' // Fallback, but will likely fail
        }
      } else {
        // Standard OpenAI
        finalUrl = 'https://api.openai.com/v1/models'
      }
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

        // Special handling for Azure OpenAI response format
        if (isAzureOpenAI) {
          try {
            console.log(`RosettaAI: Fetching Azure OpenAI deployments from: ${finalUrl}`)
            const response = await fetch(finalUrl, {
              method: 'GET',
              headers: {
                Accept: 'application/json',
                'api-key': apiKey || '' // Azure uses 'api-key' header instead of 'Authorization: Bearer'
              }
            })

            if (!response.ok) {
              let errorBody = `Status: ${response.status}`
              try {
                errorBody = await response.text()
              } catch {
                /* Ignore body parsing errors */
              }
              throw new ProviderAPIError(
                `Failed to fetch models from Azure OpenAI API: ${errorBody}`,
                providerKey,
                response.status
              )
            }

            const azureData = await response.json()

            // Transform Azure OpenAI deployments format to match OpenAI models format
            if (Array.isArray(azureData.data)) {
              const models: RosettaModel[] = azureData.data.map(
                (deployment: any): RosettaModel => {
                  return {
                    id: deployment.id || deployment.model,
                    object: 'model',
                    owned_by: 'azure',
                    created: deployment.created ? new Date(deployment.created).getTime() / 1000 : undefined,
                    active: true, // Assume all deployments are active
                    provider: providerKey,
                    rawData: deployment
                  }
                }
              )

              return {
                object: 'list',
                data: models
              }
            } else {
              throw new MappingError(
                `Invalid Azure OpenAI API response structure: expected array in 'data' field.`,
                providerKey,
                'Azure deployments response format'
              )
            }
          } catch (error) {
            if (error instanceof RosettaAIError) {
              throw error
            }
            // Wrap other errors
            const message = error instanceof Error ? error.message : String(error)
            throw new ProviderAPIError(
              `Error fetching Azure OpenAI deployments: ${message}`,
              providerKey,
              undefined,
              undefined,
              undefined,
              error
            )
          }
        } else {
          // Standard API endpoint handling (OpenAI, Google, custom providers)
          return await fetchAndValidateModelsFromApi(finalUrl, providerKey, apiKey)
        }

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
