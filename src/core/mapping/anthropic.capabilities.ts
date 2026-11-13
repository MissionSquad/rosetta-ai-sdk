/**
 * Version-based capability rules for Anthropic Claude models.
 *
 * Each rule is a floor on the family version rather than a list of ids, so a new release in an
 * existing family inherits the newest behaviour instead of silently falling back to the oldest.
 * Every rule was verified live on 2026-11-12 with `examples/probe-new-models.ts`; re-run the
 * probe before moving a floor.
 */

/** Beta that enables `thinking.block_binding` and `input_transformations` reporting. */
export const ANTHROPIC_THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'

type AnthropicFamily = 'sonnet' | 'opus' | 'haiku' | 'fable' | 'mythos'

interface AnthropicModelVersion {
  family: AnthropicFamily
  major: number
  minor: number
}

function isAnthropicFamily(value: string): value is AnthropicFamily {
  return value === 'sonnet' || value === 'opus' || value === 'haiku' || value === 'fable' || value === 'mythos'
}

/**
 * Parses `claude-<family>-<major>[-<minor>][-<yyyymmdd>][:1m]`.
 * An 8-digit second segment is a release date, not a minor version
 * (`claude-sonnet-4-20250514` is 4.0). Returns undefined for any other shape.
 */
function parseAnthropicModel(model: string): AnthropicModelVersion | undefined {
  const match = model
    .trim()
    .toLowerCase()
    .replace(/:1m$/, '')
    .match(/^claude-([a-z]+)-(\d+)(?:-(\d+))?(?:-\d+)*$/)
  if (!match) return undefined

  const family = match[1]
  const majorText = match[2]
  if (family === undefined || majorText === undefined || !isAnthropicFamily(family)) return undefined

  const minorText = match[3]
  const minor = minorText === undefined || /^\d{8}$/.test(minorText) ? 0 : Number(minorText)
  return { family, major: Number(majorText), minor }
}

function isAtLeast(version: AnthropicModelVersion, major: number, minor: number): boolean {
  return version.major > major || (version.major === major && version.minor >= minor)
}

/**
 * Whether the model rejects `temperature` / `top_p` with a 400.
 * True for every Fable and Mythos model, Opus 4.7+, Sonnet 5+ and Haiku 5+.
 */
export function anthropicRejectsSamplingParams(model: string): boolean {
  const version = parseAnthropicModel(model)
  if (!version) return false
  switch (version.family) {
    case 'fable':
    case 'mythos':
      return true
    case 'opus':
      return isAtLeast(version, 4, 7)
    case 'sonnet':
    case 'haiku':
      return isAtLeast(version, 5, 0)
  }
}

/**
 * Whether the model rejects `thinking: { type: 'enabled', budget_tokens }` and must be asked for
 * thinking with `{ type: 'adaptive' }`. The live split is identical to the sampling rule.
 */
export function anthropicRequiresAdaptiveThinking(model: string): boolean {
  return anthropicRejectsSamplingParams(model)
}

/**
 * Whether the model rejects `tool_choice` of type `any` / `tool` with a 400.
 * True for Fable and Mythos 5.1+, Opus 5.5+ and Sonnet 5.5+ (Haiku 5.5 still accepts it).
 */
export function anthropicRejectsForcedToolChoice(model: string): boolean {
  const version = parseAnthropicModel(model)
  if (!version) return false
  switch (version.family) {
    case 'fable':
    case 'mythos':
      return isAtLeast(version, 5, 1)
    case 'opus':
    case 'sonnet':
      return isAtLeast(version, 5, 5)
    case 'haiku':
      return false
  }
}

/**
 * Whether the model binds each thinking block to the system prompt, tool set and messages that
 * preceded it, and fails a replayed block when any of those changed (a 400 where enforced).
 * True for Fable and Mythos 5.1+, Opus 5.5+, Sonnet 5.5+ and Haiku 5.5+. Models below the floor
 * reject or ignore `thinking.block_binding`, so it must only be sent when this returns true.
 */
export function anthropicRunsThinkingHistoryCheck(model: string): boolean {
  const version = parseAnthropicModel(model)
  if (!version) return false
  switch (version.family) {
    case 'fable':
    case 'mythos':
      return isAtLeast(version, 5, 1)
    case 'opus':
    case 'sonnet':
    case 'haiku':
      return isAtLeast(version, 5, 5)
  }
}

/** Beta flags the mapped request needs for this model (empty when none). */
export function getRequiredAnthropicBetas(model: string): string[] {
  return anthropicRunsThinkingHistoryCheck(model) ? [ANTHROPIC_THINKING_BINDING_BETA] : []
}

/**
 * Combines caller-selected beta flags with the ones the model requires, without duplicates.
 *
 * @param requested Betas chosen by the caller or implied by the model id suffix.
 * @param model The model id, with or without a `:1m` suffix.
 * @returns The beta list to send; empty when the regular (non-beta) endpoint should be used.
 */
export function mergeAnthropicBetas(requested: readonly string[] | undefined, model: string): string[] {
  return Array.from(new Set([...(requested ?? []), ...getRequiredAnthropicBetas(model)]))
}
