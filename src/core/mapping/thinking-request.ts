/**
 * Provider-neutral thinking request resolution.
 *
 * MissionSquad agents persist model options once and may be pointed at any provider. Historically,
 * disclosed reasoning was enabled by placing provider-specific request fields in `extraParams`
 * (e.g. Google's `thinkingConfig`, Anthropic's `thinking`). Because every mapper spreads
 * `extraParams` into its provider payload, a config written for one provider produced hard 400s on
 * the strict APIs of the others ("thinkingConfig: Extra inputs are not permitted" on Anthropic,
 * "Unknown parameter: 'thinkingConfig'" on OpenAI).
 *
 * This module gives the strict-dialect mappers (Anthropic, Google, OpenAI, Azure OpenAI, Groq) one
 * shared policy:
 *
 * - Well-known thinking/reasoning keys that belong to a *different* provider are removed from
 *   `extraParams` before the payload is built, so they can never invalidate the request.
 * - An unambiguous "disclose reasoning" intent expressed through a foreign key is translated into
 *   the neutral `thinking: true` request, which each mapper then renders in its own dialect.
 * - Keys native to the target provider are left untouched so explicit per-provider overrides keep
 *   working (mapped fields still take precedence over `extraParams` on collision, per the
 *   `GenerateParams.extraParams` contract).
 *
 * The OpenAI-compatible mapper intentionally does NOT use this policy: custom endpoints speak
 * unknown dialects (some accept `thinking`, `reasoning`, `include_reasoning`, ...), so its
 * `extraParams` remain a raw passthrough.
 */

/**
 * Request-side thinking/reasoning keys with well-known provider-specific meanings.
 *
 * - `thinking`        — Anthropic Messages API (`{type: 'adaptive' | 'enabled' | 'disabled'}`)
 * - `thinkingConfig`  — Google GenerateContentConfig (`{includeThoughts, thinkingBudget, ...}`)
 * - `reasoning`       — OpenAI Responses API / OpenRouter chat dialect (object form)
 * - `reasoning_effort` — OpenAI Chat Completions / Groq
 * - `reasoning_format` — Groq
 * - `include_reasoning` — OpenRouter-style disclosure toggle
 */
const PROVIDER_THINKING_EXTRA_PARAM_KEYS = [
  'thinking',
  'thinkingConfig',
  'reasoning',
  'reasoning_effort',
  'reasoning_format',
  'include_reasoning'
] as const

export interface ResolvedThinkingRequest {
  /** True when the caller asked for disclosed reasoning, directly or via a translated foreign key. */
  thinkingRequested: boolean
  /** `extraParams` with foreign provider thinking keys removed. Undefined iff the input was undefined. */
  extraParams?: Record<string, unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Returns true when a stripped foreign key unambiguously requested disclosed reasoning.
 *
 * Only positive, explicit signals are translated:
 * - Google `thinkingConfig.includeThoughts === true`
 * - Anthropic `thinking.type` of `'adaptive'` or `'enabled'`
 * - OpenRouter-style `include_reasoning === true`
 *
 * Negative or ambiguous values (e.g. `thinking: {type: 'disabled'}`, a bare `reasoning_effort`)
 * never force thinking on; they are simply dropped for providers where the key is invalid.
 */
function impliesThinkingRequest(key: string, value: unknown): boolean {
  if (key === 'thinkingConfig') {
    return isRecord(value) && value.includeThoughts === true
  }
  if (key === 'thinking') {
    return isRecord(value) && (value.type === 'adaptive' || value.type === 'enabled')
  }
  if (key === 'include_reasoning') {
    return value === true
  }
  return false
}

/**
 * Resolves the effective thinking request for a strict-dialect provider mapper.
 *
 * @param params - The `thinking` and `extraParams` from GenerateParams.
 * @param nativeKeys - Thinking keys valid in this provider's request dialect; these are preserved
 *   in `extraParams` verbatim. Every other key from {@link PROVIDER_THINKING_EXTRA_PARAM_KEYS} is
 *   removed and, when it carries an unambiguous positive intent, folded into `thinkingRequested`.
 */
export function resolveThinkingRequest(
  params: { thinking?: boolean; extraParams?: Record<string, unknown> },
  nativeKeys: readonly string[]
): ResolvedThinkingRequest {
  let thinkingRequested = params.thinking === true

  if (params.extraParams === undefined) {
    return { thinkingRequested }
  }

  const sanitized: Record<string, unknown> = { ...params.extraParams }
  for (const key of PROVIDER_THINKING_EXTRA_PARAM_KEYS) {
    if (nativeKeys.includes(key) || !(key in sanitized)) continue
    if (impliesThinkingRequest(key, sanitized[key])) {
      thinkingRequested = true
    }
    delete sanitized[key]
  }

  return { thinkingRequested, extraParams: sanitized }
}
