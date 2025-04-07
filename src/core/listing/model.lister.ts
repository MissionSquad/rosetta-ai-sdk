// src/core/listing/model.lister.ts
import Groq from 'groq-sdk'
import { RosettaModelList, RosettaModel, Provider, ModelListingSourceConfig, ModelListingSourceType } from '../../types'
import { ProviderAPIError, MappingError, ConfigurationError, RosettaAIError } from '../../errors'
import { anthropicStaticModels } from './static-data/anthropic.models'
import { fetchAndValidateModelsFromApi } from './fetch.utils'

// Internal function to handle listing for a specific provider
export async function listModelsForProvider(
  provider: Provider,
  config: {
    // Pass necessary parts of RosettaAIConfig and clients
    sourceConfig?: ModelListingSourceConfig
    apiKey?: string
    groqClient?: Groq // Pass Groq client if available
  }
): Promise<RosettaModelList> {
  const source = config.sourceConfig // Determine source type

  // Determine source type (could default based on provider if config not present)
  let sourceType: ModelListingSourceType
  if (source) {
    sourceType = source.type
  } else {
    // Default logic if no explicit config
    if (provider === Provider.Groq && config.groqClient) sourceType = 'sdkMethod'
    else if (provider === Provider.Anthropic) sourceType = 'staticList'
    else if (provider === Provider.Google || provider === Provider.OpenAI) sourceType = 'apiEndpoint'
    else throw new ConfigurationError(`Model listing source type for provider ${provider} could not be determined.`)
  }

  // --- Execute based on source type ---
  try {
    switch (sourceType) {
      case 'staticList':
        if (provider !== Provider.Anthropic) {
          throw new ConfigurationError(`Static list is only configured for Anthropic, not ${provider}.`)
        }
        // Return a deep copy to prevent modification of the original static data
        return JSON.parse(JSON.stringify(anthropicStaticModels))

      case 'sdkMethod':
        if (provider !== Provider.Groq || !config.groqClient) {
          throw new ConfigurationError(`SDK method listing is only configured for Groq with an active client.`)
        }
        const groqResponse = await config.groqClient.models.list()
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
            provider: Provider.Groq,
            rawData: groqModel // Store original
          })
        )
        return { object: 'list', data: groqModels }

      case 'apiEndpoint':
        let url: string
        if (source && source.type === 'apiEndpoint' && source.url) {
          url = source.url
        } else {
          // Default URLs if no config provided
          if (provider === Provider.Google) url = 'https://generativelanguage.googleapis.com/v1beta/openai/models'
          // Updated Google URL
          else if (provider === Provider.OpenAI) url = 'https://api.openai.com/v1/models'
          else throw new ConfigurationError(`API endpoint URL for ${provider} not configured.`)
        }
        return await fetchAndValidateModelsFromApi(url, provider, config.apiKey)

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
      error instanceof RosettaAIError // Catch base SDK error too
    ) {
      throw error
    }
    // Wrap other errors
    const message = error instanceof Error ? error.message : String(error)
    throw new ProviderAPIError(
      `Failed to list models for ${provider} using ${sourceType}: ${message}`,
      provider,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
