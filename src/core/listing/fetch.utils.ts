// src/core/listing/fetch.utils.ts (Simplified Example)
import { z } from 'zod'
import { RosettaModel, RosettaModelList, Provider } from '../../types'
import { ProviderAPIError, MappingError, RosettaAIError } from '../../errors'

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
  provider: Provider,
  apiKey: string | undefined
): Promise<RosettaModelList> {
  if (!apiKey) {
    throw new ProviderAPIError(`API key for ${provider} is required but missing for model listing.`, provider, 401)
  }

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: 'application/json'
      }
    })

    if (!response.ok) {
      let errorBody = `Status: ${response.status}`
      try {
        errorBody = await response.text()
      } catch {
        /* Ignore body parsing errors */
      }
      throw new ProviderAPIError(`Failed to fetch models from ${provider} API: ${errorBody}`, provider, response.status)
    }

    const rawJson = await response.json()

    // --- CRITICAL VALIDATION STEP ---
    const validationResult = BaseApiResponseSchema.safeParse(rawJson)
    if (!validationResult.success) {
      console.error(`Validation Error for ${provider} API Response (${url}):`, validationResult.error.errors)
      throw new MappingError(
        `Invalid API response structure received from ${provider}.`,
        provider,
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
          provider: provider,
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
      `Network or parsing error fetching models for ${provider}: ${message}`,
      provider,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
