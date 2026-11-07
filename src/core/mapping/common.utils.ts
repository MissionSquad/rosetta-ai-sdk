import { BillingTokenDetail, BillingUsage, TokenUsage, GenerateParams, GenerateResult } from '../../types'
import { StructuredOutputValidationError } from '../../errors'
import { safeGet } from '../utils' // Import safeGet from the existing utils file

/** Reads a provider model identifier without changing its value. */
export function mapReportedModel(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined
}

/**
 * Maps various provider-specific usage objects to the unified RosettaAI TokenUsage type.
 * Handles differences in field names (e.g., input_tokens vs prompt_tokens).
 *
 * @param providerUsage - The usage object from the provider's response (type `any` for flexibility).
 * @returns A unified TokenUsage object or undefined if no usage data is available.
 */
export function mapTokenUsage(providerUsage: any, format?: UsageFormat, serviceTier?: unknown): TokenUsage | undefined {
  if (format) return mapExplicitTokenUsage(providerUsage, format, serviceTier)

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

type UsageFormat = 'openai' | 'openai-responses' | 'anthropic' | 'google' | 'groq'

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>
  return undefined
}

/** Normalize inclusive provider counts without fabricating missing components. */
function mapExplicitTokenUsage(value: unknown, format: UsageFormat, serviceTier: unknown): TokenUsage | undefined {
  const usage = asRecord(value)
  if (!usage) return undefined
  let invalid = false
  const count = (key: string, optional = false): number | undefined => {
    const value = usage[key]
    if (value === undefined) return optional ? 0 : undefined
    if (value === null) { invalid = true; return undefined }
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
      invalid = true
      return undefined
    }
    return value
  }
  const sum = (...values: Array<number | undefined>): number | undefined => {
    let result = 0
    for (const value of values) {
      if (value === undefined) return undefined
      result += value
    }
    if (!Number.isSafeInteger(result)) { invalid = true; return undefined }
    return result
  }
  let promptTokens: number | undefined
  let completionTokens: number | undefined
  let totalTokens: number | undefined
  if (format === 'anthropic') {
    promptTokens = sum(count('input_tokens'), count('cache_creation_input_tokens'), count('cache_read_input_tokens'))
    completionTokens = count('output_tokens')
  } else if (format === 'google') {
    promptTokens = sum(count('promptTokenCount'), count('toolUsePromptTokenCount', true))
    completionTokens = sum(count('candidatesTokenCount'), count('thoughtsTokenCount', true))
    totalTokens = count('totalTokenCount')
  } else {
    const responses = format === 'openai-responses'
    promptTokens = count(responses ? 'input_tokens' : 'prompt_tokens')
    completionTokens = count(responses ? 'output_tokens' : 'completion_tokens')
    totalTokens = count('total_tokens')
  }
  const calculated = sum(promptTokens, completionTokens)
  if (totalTokens !== undefined && calculated !== undefined && totalTokens !== calculated) invalid = true
  totalTokens ??= calculated
  const cachedContentTokenCount = format === 'google' ? count('cachedContentTokenCount') : undefined
  const billing = format === 'groq' ? undefined : mapBillingUsage(usage, format, serviceTier, promptTokens, completionTokens)
  return {
    promptTokens, completionTokens, totalTokens,
    ...(billing ? { billing } : {}),
    ...(cachedContentTokenCount !== undefined ? { cachedContentTokenCount } : {}),
    ...(invalid || totalTokens === undefined ? { incomplete: true } : {})
  }
}

function mapBillingUsage(
  usage: Record<string, unknown>,
  format: Exclude<UsageFormat, 'groq'>,
  serviceTier: unknown,
  promptTokens: number | undefined,
  completionTokens: number | undefined
): BillingUsage {
  const billing: BillingUsage = {}
  const count = (value: unknown): number | undefined => {
    if (value === undefined) return undefined
    if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value
    billing.incomplete = true
    return undefined
  }
  const detailObject = (value: unknown): Record<string, unknown> | undefined => {
    if (value == null) return undefined
    const result = asRecord(value)
    if (!result) billing.incomplete = true
    return result
  }
  const label = (value: unknown): string | undefined => {
    if (value == null) return undefined
    if (typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(value)) return value
    billing.incomplete = true
    return undefined
  }
  const details = (value: unknown): BillingTokenDetail[] | undefined => {
    if (value == null) return undefined
    if (Array.isArray(value)) {
      const items: BillingTokenDetail[] = []
      for (const entry of value) {
        const item = asRecord(entry)
        const modality = label(item?.modality)
        const tokenCount = count(item?.tokenCount)
        if (modality === undefined || tokenCount === undefined) {
          billing.incomplete = true
          return undefined
        }
        items.push({ modality, tokenCount })
      }
      return items
    }
    billing.incomplete = true
    return undefined
  }

  if (format === 'anthropic') {
    const cacheCreation = detailObject(usage.cache_creation)
    billing.cachedInputTokens = count(usage.cache_read_input_tokens)
    billing.cacheWriteInputTokens = count(usage.cache_creation_input_tokens)
    billing.cacheWrite5mInputTokens = count(cacheCreation?.ephemeral_5m_input_tokens)
    billing.cacheWrite1hInputTokens = count(cacheCreation?.ephemeral_1h_input_tokens)
    billing.serviceTier = label(usage.service_tier)
    billing.inferenceGeo = label(usage.inference_geo)
    billing.speed = label(usage.speed)
  } else if (format === 'google') {
    billing.serviceTier = label(serviceTier ?? usage.serviceTier)
    billing.cachedInputTokens = count(usage.cachedContentTokenCount)
    billing.inputTokenDetails = details(usage.promptTokensDetails)
    billing.cachedInputTokenDetails = details(usage.cacheTokensDetails)
    billing.toolUseInputTokenDetails = details(usage.toolUsePromptTokensDetails)
    billing.outputTokenDetails = details(usage.candidatesTokensDetails)
  } else {
    const responses = format === 'openai-responses'
    const input = detailObject(usage[responses ? 'input_tokens_details' : 'prompt_tokens_details'])
    billing.cachedInputTokens = count(input?.cached_tokens)
    const cacheWriteTokens = input?.cache_write_tokens
    // Chat Completions defaults omitted writes to zero. Responses preserves absence.
    billing.cacheWriteInputTokens = count(!responses && cacheWriteTokens === undefined ? 0 : cacheWriteTokens)
    billing.serviceTier = label(serviceTier)
    if (responses) {
      const cacheWrite = detailObject(input?.cache_write_tokens_details)
      billing.cacheWrite30mInputTokens = count(cacheWrite?.ttl_30m_tokens)
      billing.cacheWrite12hInputTokens = count(cacheWrite?.ttl_12h_tokens)
      if (billing.cachedInputTokens === undefined) billing.incomplete = true
    } else {
      const output = detailObject(usage.completion_tokens_details)
      billing.cacheWrite30mInputTokens = billing.cacheWriteInputTokens
      billing.inputAudioTokens = count(input?.audio_tokens)
      billing.outputAudioTokens = count(output?.audio_tokens)
      const textTokens = count(input?.text_tokens)
      const imageTokens = count(input?.image_tokens)
      const inputDetails: BillingTokenDetail[] = []
      if (textTokens !== undefined) inputDetails.push({ modality: 'TEXT', tokenCount: textTokens })
      if (imageTokens !== undefined) inputDetails.push({ modality: 'IMAGE', tokenCount: imageTokens })
      if (inputDetails.length) billing.inputTokenDetails = inputDetails
    }
  }

  const cachedAndWritten = (billing.cachedInputTokens ?? 0) + (billing.cacheWriteInputTokens ?? 0)
  const split = format === 'anthropic'
    ? [billing.cacheWrite5mInputTokens, billing.cacheWrite1hInputTokens]
    : [billing.cacheWrite30mInputTokens, billing.cacheWrite12hInputTokens]
  const [shortWrite, longWrite] = split
  if ((promptTokens !== undefined && (cachedAndWritten > promptTokens || (billing.inputAudioTokens ?? 0) > promptTokens)) ||
      (completionTokens !== undefined && (billing.outputAudioTokens ?? 0) > completionTokens) ||
      (billing.cacheWriteInputTokens !== undefined && shortWrite !== undefined && longWrite !== undefined &&
        shortWrite + longWrite !== billing.cacheWriteInputTokens)) {
    billing.incomplete = true
  }
  // Omit optional keys that the provider did not report, preserving explicit zeros.
  for (const key of Object.keys(billing) as Array<keyof BillingUsage>) {
    if (billing[key] === undefined) delete billing[key]
  }
  return billing
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
  reasoningEffort?: GenerateParams['reasoningEffort']
  verbosity?: GenerateParams['verbosity']
  stopSequences?: string[]
} {
  const stopSequences = Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined

  // Normalize and apply precedence for token limits
  const hasMct = params.maxCompletionTokens != null
  const hasMt = params.maxTokens != null

  const normalizedMct = hasMct && params.maxCompletionTokens !== -1 ? params.maxCompletionTokens : undefined
  const normalizedMt = hasMt && params.maxTokens !== -1 ? params.maxTokens : undefined

  // Precedence: maxCompletionTokens > maxTokens
  const effectiveMaxTokens = normalizedMct ?? normalizedMt

  // Create an object with only defined parameters
  const result: {
    temperature?: number
    topP?: number
    maxTokens?: number
    reasoningEffort?: GenerateParams['reasoningEffort']
    verbosity?: GenerateParams['verbosity']
    stopSequences?: string[]
  } = {}

  if (params.temperature !== undefined) result.temperature = params.temperature
  if (params.topP !== undefined) result.topP = params.topP
  if (effectiveMaxTokens !== undefined) result.maxTokens = effectiveMaxTokens
  if (params.reasoningEffort !== undefined) result.reasoningEffort = params.reasoningEffort
  if (params.verbosity !== undefined) result.verbosity = params.verbosity
  if (stopSequences !== undefined) result.stopSequences = stopSequences

  return result
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

export type OpenAIResponseFormatParam =
  | { type: 'text' }
  | { type: 'json_object' }
  | {
      type: 'json_schema'
      json_schema: {
        name: string
        strict: boolean
        schema: Record<string, unknown>
      }
    }

/**
 * Normalize Rosetta's responseFormat into OpenAI Chat Completions `response_format`.
 *
 * This is used for OpenAI and OpenAI-compatible providers.
 */
export function mapToOpenAIResponseFormat(
  responseFormat: GenerateParams['responseFormat']
): OpenAIResponseFormatParam | undefined {
  if (!responseFormat) return undefined

  if (responseFormat.type === 'text') return { type: 'text' }

  if (responseFormat.type === 'json_object') {
    return { type: 'json_object' }
  }

  if (responseFormat.type === 'json_schema') {
    return {
      type: 'json_schema',
      json_schema: {
        name: responseFormat.json_schema.name ?? 'response',
        strict: responseFormat.json_schema.strict ?? true,
        schema: responseFormat.json_schema.schema
      }
    }
  }

  return undefined
}

/**
 * Parses structured output and, when supplied, enforces the caller's runtime Zod schema.
 * A missing validator preserves the SDK's existing best-effort JSON parsing behavior.
 *
 * @throws {StructuredOutputValidationError} If validated output is malformed JSON or fails its Zod schema.
 */
export function parseStructuredOutput(
  responseFormat: GenerateParams['responseFormat'],
  content: string | null
): GenerateResult['parsedContent'] | undefined {
  if (
    !responseFormat ||
    (responseFormat.type !== 'json_object' && responseFormat.type !== 'json_schema') ||
    typeof content !== 'string' ||
    !content.trim()
  ) {
    return undefined
  }

  const zodSchema = responseFormat.type === 'json_schema' ? responseFormat.json_schema.zodSchema : undefined
  const schemaName = responseFormat.type === 'json_schema' ? responseFormat.json_schema.name : undefined
  let parsed: unknown
  try {
    parsed = JSON.parse(content)
  } catch (error) {
    if (zodSchema) {
      throw new StructuredOutputValidationError('Provider output is not valid JSON', [], schemaName, content, error)
    }
    return undefined
  }

  if (!zodSchema) return parsed as GenerateResult['parsedContent']

  const validation = zodSchema.safeParse(parsed)
  if (!validation.success) {
    throw new StructuredOutputValidationError(
      'Provider output failed runtime schema validation',
      validation.error.issues,
      schemaName,
      parsed
    )
  }
  return validation.data as GenerateResult['parsedContent']
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
