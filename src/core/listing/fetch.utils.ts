// src/core/listing/fetch.utils.ts
import { z } from 'zod'
import { RosettaModel, RosettaModelList, ProviderKey, Provider } from '../../types'
import { ProviderAPIError, MappingError, RosettaAIError } from '../../errors'

/**
 * Determines if a URL belongs to the Cohere API
 */
function isCohereApiUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url)
    return (
      hostname === 'api.cohere.com' ||
      hostname.endsWith('.cohere.com') ||
      hostname === 'api.cohere.ai' ||
      hostname.endsWith('.cohere.ai')
    )
  } catch {
    return false
  }
}

/**
 * Transforms Cohere's response format to match OpenAI's expected format
 */
function transformCohereResponse(rawJson: any): any {
  // Check if this looks like a Cohere response
  if (!rawJson.models || !Array.isArray(rawJson.models)) {
    return rawJson // Not a Cohere response, return as-is
  }

  // Transform the response
  const transformed = {
    object: 'list' as const,
    data: rawJson.models.map((model: any) => ({
      // Map 'name' to 'id' as that's what Cohere uses as the identifier
      id: model.name || model.id,
      object: 'model' as const,
      // Use a default for owned_by since Cohere doesn't provide this
      owned_by: 'cohere',
      // Map other fields while preserving them in rawData
      created: model.created,
      active: model.active ?? true, // Assume active if not specified
      context_window: model.context_length || model.context_window,
      // Preserve all original data
      ...model
    }))
  }

  // Remove the original 'models' and 'next_page_token' from each data item
  transformed.data = transformed.data.map((item: any) => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { models, next_page_token, ...cleanItem } = item
    return cleanItem
  })

  return transformed
}

// Zod schema for the MINIMUM expected API response structure
const BaseApiResponseSchema = z
  .object({
    object: z.literal('list'),
    data: z.array(
      z
        .object({
          id: z.string(),
          object: z.literal('model'),
          owned_by: z.string()
          // Allow other fields to pass through
        })
        .passthrough()
    )
  })
  .strict() // Use strict to prevent unexpected top-level fields

/**
 * Fetches and validates model list from an API endpoint.
 */
export async function fetchAndValidateModelsFromApi(
  url: string,
  providerKey: ProviderKey, // Accept ProviderKey (string or enum)
  apiKey: string | undefined
): Promise<RosettaModelList> {
  // --- API Key Check for Built-in Providers ---
  // Check if the providerKey is one of the built-in Provider enum values
  const isBuiltIn = Object.values(Provider).includes(providerKey as Provider)
  if (isBuiltIn && !apiKey) {
    // Throw the specific error the test expects
    throw new ProviderAPIError(
      `API key for ${providerKey} is required but missing for model listing.`,
      providerKey
      // No status code here as the error happens before the request
    )
  }

  // For Cohere, the model list URL is fixed and different from the chat completions compatibility endpoint.
  const isCohere = isCohereApiUrl(url)
  let finalUrl = url

  if (isCohere) {
    // Cohere's model list is at a fixed endpoint, different from their compatibility endpoint path
    const cohereModelsUrl = 'https://api.cohere.com/v1/models'
    const urlObj = new URL(cohereModelsUrl)
    urlObj.searchParams.set('page_size', '1000')
    finalUrl = urlObj.toString()
    console.log(`RosettaAI: Detected Cohere provider. Overriding model list URL to: ${finalUrl}`)
  }

  // API key might be optional for custom providers (e.g., local servers)
  const headers: HeadersInit = {
    Accept: 'application/json'
  }
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`
  } else {
    // Check if API key is actually required (e.g., not a local server)
    // This is a basic check; more robust logic might be needed based on URL patterns
    if (!url.startsWith('http://localhost') && !url.startsWith('http://127.0.0.1')) {
      console.warn(`API key is missing for non-local API endpoint: ${url}. Request might fail.`)
      // Warning for non-local custom providers without keys is still reasonable
      console.warn(`API key is missing for non-local custom API endpoint: ${url}. Request might fail.`)
    }
  }

  try {
    const response = await fetch(finalUrl, {
      method: 'GET',
      headers: headers
    })

    if (!response.ok) {
      let errorBody = `Status: ${response.status}`
      try {
        errorBody = await response.text()
      } catch {
        /* Ignore body parsing errors */
      }
      throw new ProviderAPIError(
        `Failed to fetch models from ${providerKey} API: ${errorBody}`,
        providerKey,
        response.status
      )
    }

    let rawJson = await response.json()

    // Transform Cohere response if detected
    if (isCohere) {
      console.log(`RosettaAI: Transforming Cohere response format`)
      rawJson = transformCohereResponse(rawJson)
    }

    // --- CRITICAL VALIDATION STEP ---
    const validationResult = BaseApiResponseSchema.safeParse(rawJson)
    if (!validationResult.success) {
      console.error(`Validation Error for ${providerKey} API Response (${url}):`, validationResult.error.errors)
      throw new MappingError(
        `Invalid API response structure received from ${providerKey}.`,
        providerKey,
        'fetchAndValidateModelsFromApi validation',
        validationResult.error // Include ZodError for details
      )
    }

    const validatedData = validationResult.data // Now typed according to schema

    // --- Mapping ---
    const models: RosettaModel[] = validatedData.data.map(
      (rawModel: any): RosettaModel => {
        // Map known fields + safely access optional ones seen in examples
        return {
          id: rawModel.id,
          object: 'model',
          owned_by: rawModel.owned_by,
          created: typeof rawModel.created === 'number' ? rawModel.created : undefined, // Only if number
          active: typeof rawModel.active === 'boolean' ? rawModel.active : undefined, // Optional field handling
          context_window: typeof rawModel.context_window === 'number' ? rawModel.context_window : undefined,
          public_apps: rawModel.public_apps ?? undefined, // Handle null or undefined
          max_completion_tokens:
            typeof rawModel.max_completion_tokens === 'number' ? rawModel.max_completion_tokens : undefined,
          properties: rawModel.properties
            ? {
                // Map known properties if they exist
                description: rawModel.properties.description,
                strengths: rawModel.properties.strengths,
                multilingual: rawModel.properties.multilingual,
                vision: rawModel.properties.vision
              }
            : undefined,
          provider: providerKey,
          rawData: rawModel // Store original
        }
      }
    )

    return {
      object: 'list',
      data: models
    }
  } catch (error) {
    if (error instanceof RosettaAIError) {
      // Don't re-wrap our errors
      throw error
    }
    // Wrap fetch/parsing errors
    const message = error instanceof Error ? error.message : String(error)
    throw new ProviderAPIError(
      `Network or parsing error fetching models for ${providerKey}: ${message}`,
      providerKey,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
