import { TokenUsage, GenerateParams } from '../../types'
import { safeGet } from '../utils' // Import safeGet from the existing utils file

/**
 * Maps various provider-specific usage objects to the unified RosettaAI TokenUsage type.
 * Handles differences in field names (e.g., input_tokens vs prompt_tokens).
 *
 * @param providerUsage - The usage object from the provider's response (type `any` for flexibility).
 * @returns A unified TokenUsage object or undefined if no usage data is available.
 */
export function mapTokenUsage(providerUsage: any): TokenUsage | undefined {
  if (!providerUsage || typeof providerUsage !== 'object') {
    return undefined
  }

  // Explicitly check for Google's promptTokenCount first if it exists
  let promptTokens = safeGet<number>(providerUsage, 'promptTokenCount') // Google style

  // If Google's field wasn't found, check other common names
  if (promptTokens === undefined) {
    promptTokens =
      safeGet<number>(providerUsage, 'prompt_tokens') ?? // OpenAI/Groq style
      safeGet<number>(providerUsage, 'input_tokens') ?? // Anthropic style
      undefined
  }

  // Explicitly check for Google's candidatesTokenCount first if it exists
  let completionTokens = safeGet<number>(providerUsage, 'candidatesTokenCount') // Google Generate style

  // If Google's field wasn't found, check other common names
  if (completionTokens === undefined) {
    completionTokens =
      safeGet<number>(providerUsage, 'completion_tokens') ?? // OpenAI/Groq style
      safeGet<number>(providerUsage, 'output_tokens') ?? // Anthropic style
      undefined
  }

  // Explicitly check for Google's totalTokenCount first if it exists
  let totalTokens = safeGet<number>(providerUsage, 'totalTokenCount') // Google style

  // If Google's field wasn't found, check other common names
  if (totalTokens === undefined) {
    totalTokens = safeGet<number>(providerUsage, 'total_tokens') ?? undefined // OpenAI/Groq style
  }

  // Calculate totalTokens if prompt and completion are available but total is not
  if (totalTokens === undefined && promptTokens !== undefined && completionTokens !== undefined) {
    totalTokens = promptTokens + completionTokens
  }

  // Google specific fields
  const cachedContentTokenCount = safeGet<number>(providerUsage, 'cachedContentTokenCount') ?? undefined

  // Only return usage if at least one field was found
  if (
    promptTokens !== undefined ||
    completionTokens !== undefined ||
    totalTokens !== undefined ||
    cachedContentTokenCount !== undefined
  ) {
    return {
      promptTokens,
      completionTokens,
      totalTokens,
      cachedContentTokenCount
    }
  }

  return undefined
}

/**
 * Maps common generation parameters (temperature, topP, maxTokens, stop)
 * from RosettaAI's GenerateParams to a generic structure.
 * Provider-specific mappers will handle mapping these to the correct provider field names.
 *
 * @param params - The unified RosettaAI generation parameters.
 * @returns An object containing the mapped base parameters.
 */
export function mapBaseParams(
  params: GenerateParams
): {
  temperature?: number
  topP?: number
  maxTokens?: number
  stopSequences?: string[]
} {
  const stopSequences = Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined

  return {
    temperature: params.temperature,
    topP: params.topP,
    maxTokens: params.maxTokens,
    stopSequences: stopSequences
  }
}

/**
 * Maps the base RosettaAI tool choice options to a standardized intermediate representation
 * or directly to common provider values ('auto', 'none').
 * Provider-specific mappers handle translation to provider-specific values like 'any', 'required', or tool objects.
 *
 * @param toolChoice - The tool choice parameter from GenerateParams.
 * @returns A standardized representation (e.g., 'auto', 'none', 'required', { type: 'function', function: { name: string } }) or undefined.
 */
export function mapBaseToolChoice(
  toolChoice: GenerateParams['toolChoice']
): 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } } | undefined {
  if (typeof toolChoice === 'string' && ['none', 'auto', 'required'].includes(toolChoice)) {
    return toolChoice as 'none' | 'auto' | 'required'
  } else if (typeof toolChoice === 'object' && toolChoice.type === 'function' && toolChoice.function?.name) {
    // Return the structure directly if it matches the function type
    return toolChoice
  } else if (toolChoice) {
    // If it's defined but not a recognized format, default to 'auto' and warn?
    // Or return undefined and let the provider mapper handle the default? Let's return undefined.
    console.warn(`Unsupported tool_choice format encountered in common mapping: ${JSON.stringify(toolChoice)}`)
    return undefined
  }
  return undefined // Return undefined if toolChoice is not set
}

/**
 * Placeholder for mapping base content parts.
 * Due to role constraints (e.g., images only for user), this is complex
 * and likely better handled within each provider's `mapToProviderParams` logic.
 *
 * @param part - The RosettaContentPart to map.
 * @param role - The role of the message containing the part.
 * @param provider - The target provider.
 * @returns Provider-specific content part object (type `any`).
 */
// export function mapBaseContentPart(part: RosettaContentPart, role: string, provider: Provider): any {
//   // Implementation would involve checks based on role and provider capabilities
//   // For now, this logic resides within individual provider mappers.
//   throw new Error('mapBaseContentPart is not implemented. Handle content mapping within provider mappers.');
// }
