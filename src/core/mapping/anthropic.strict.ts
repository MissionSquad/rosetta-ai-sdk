import type { JSONSchema7, JSONSchema7Definition } from 'json-schema'

/**
 * Anthropic strict tool use support.
 *
 * Anthropic's standard (buffered) tool-use streaming guarantees that a tool's
 * accumulated `input` is *valid JSON*, but NOT that it conforms to the tool's
 * declared JSON schema. A model can therefore emit a syntactically valid but
 * schema-incomplete tool call (e.g. an object missing a `required` field),
 * which rosetta then rejects at its local zod `safeParse` boundary — a terminal
 * failure that kills the stream.
 *
 * Anthropic's documented fix is **strict tool use**: set `strict: true` as a
 * top-level field on the tool definition (a sibling of `name` / `description` /
 * `input_schema`). It is generally available (no beta header) and guarantees
 * that `tool_use.input` validates exactly against the tool's schema
 * server-side. It is supported on Claude 4.5 and later models and requires the
 * schema to declare `additionalProperties: false` and a `required` array on
 * every object schema, and rejects a set of unsupported JSON Schema
 * constructs.
 *
 * Source of truth for the constraints below:
 * https://platform.claude.com/docs/en/build-with-claude/structured-outputs
 * https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
 */

/**
 * JSON Schema keywords that Anthropic strict tool use does not support. When
 * strict mode is applied these are stripped from the *copy* of the schema sent
 * to Anthropic (the caller's schema and rosetta's local zod validation are left
 * untouched).
 *
 * Notes:
 * - `pattern` is deliberately NOT included: it is not part of Anthropic's
 *   documented unsupported list, so it is preserved.
 * - `minItems` of 0 or 1 is technically accepted, but `maxItems` and higher
 *   `minItems` values are not; stripping `minItems` unconditionally is a safe
 *   relaxation (rosetta's zod schema still enforces the constraint locally).
 */
const STRICT_UNSUPPORTED_KEYWORDS: ReadonlySet<string> = new Set([
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'multipleOf',
  'minLength',
  'maxLength',
  'minItems',
  'maxItems'
])

/**
 * Model-ID prefixes known NOT to support strict tool use. Strict tool use is GA
 * on Claude 4.5 and later; everything before that (Claude 3.x/2.x, the
 * dedicated Claude 4.0/4.1 models, and their dated snapshot forms) is excluded.
 *
 * This is a defensive denylist rather than a registry lookup: the Anthropic
 * registry in this SDK is entirely 4.5+, but callers may configure arbitrary
 * model IDs (e.g. the still-live but deprecated `claude-opus-4-1`). Matching by
 * prefix keeps arbitrary/unknown 4.5+ IDs enabled while reliably excluding the
 * known-unsupported families, including their `-20250514` / `-20250805` dated
 * forms (matched via the `claude-*-4-2025` prefixes).
 */
const STRICT_UNSUPPORTED_MODEL_PREFIXES: readonly string[] = [
  'claude-1',
  'claude-2',
  'claude-instant',
  'claude-3',
  'claude-opus-4-0',
  'claude-opus-4-1',
  // The two `-4-2025` entries below target the dated 4.0 snapshot IDs
  // (`claude-opus-4-20250514`, `claude-sonnet-4-20250514`). Dated 4.5+ snapshots
  // are named `claude-{family}-4-5-YYYYMMDD`, so they never match these.
  'claude-opus-4-2025',
  'claude-sonnet-4-0',
  'claude-sonnet-4-2025'
]

/**
 * Returns whether the given Anthropic model ID supports strict tool use.
 *
 * Normalization, in order: trim, lower-case, then strip a *single trailing*
 * colon-delimited suffix. The strip is end-anchored (`/:[^:]+$/`) and suffix
 * agnostic, so `:1m`, `:thinking`, `:batch`, and any future variant are all
 * handled, while a `:`-bearing segment in the middle of an ID is left intact
 * (an unanchored replace could splice an ID into a different — potentially
 * denylisted — one).
 *
 * Matching is then gated on the `claude-` prefix: a non-Anthropic ID (e.g.
 * `gpt-4o`) is never strict-eligible, even though it matches no entry in
 * {@link STRICT_UNSUPPORTED_MODEL_PREFIXES}. Unknown `claude-*` IDs that miss
 * the denylist remain enabled by design — new Anthropic releases are 4.5+ and
 * therefore strict-capable, so defaulting them to `true` is forward compatible.
 *
 * @param model - The (possibly suffixed, e.g. `:1m`) model ID from the request.
 * @returns `true` for Claude 4.5+ models; `false` for known pre-4.5 families,
 *   non-`claude-` IDs, and an empty/absent model ID.
 */
export function supportsStrictToolUse(model: string | undefined | null): boolean {
  if (typeof model !== 'string') {
    return false
  }
  // End-anchored strip of a single trailing colon-delimited suffix.
  const normalized = model.trim().toLowerCase().replace(/:[^:]+$/, '')
  if (!normalized.startsWith('claude-')) {
    // Covers the empty/whitespace-only ID as well as any non-Anthropic ID.
    return false
  }
  return !STRICT_UNSUPPORTED_MODEL_PREFIXES.some(prefix => normalized.startsWith(prefix))
}

/**
 * The result of normalizing a tool's JSON schema for strict tool use.
 *
 * @property eligible - Whether the schema can be sent with `strict: true`. This
 *   is `false` when the schema contains a construct strict mode rejects that
 *   cannot be safely rewritten (a `$ref`/`$defs` reference, a recursive object
 *   graph, or an `additionalProperties` set to something other than `false`).
 * @property schema - A normalized copy of the input schema. When `eligible` is
 *   `true` it is a full deep copy sharing no nodes with the input; when
 *   `eligible` is `false` (notably for cyclic graphs, where normalization
 *   short-circuits) it may share nodes with the input and must be discarded.
 *   Only ever sent to Anthropic when `eligible` is `true`; the caller's schema
 *   object is never mutated in either case.
 */
export interface StrictSchemaNormalizationResult {
  eligible: boolean
  schema: JSONSchema7
}

/**
 * Produces a strict-eligible copy of a tool's JSON schema.
 *
 * Recursively, while building a fresh copy of the schema graph:
 * - sets `additionalProperties: false` on every object schema that does not
 *   already set it;
 * - ensures a `required` array exists on every object schema (an empty array
 *   when there are no required properties — optional properties remain valid
 *   under strict mode);
 * - strips the unsupported constraint keywords in {@link STRICT_UNSUPPORTED_KEYWORDS};
 * - flags the schema ineligible when it uses a construct strict mode rejects
 *   and that cannot be safely rewritten (`$ref`/`$defs`, a cyclic object graph,
 *   or `additionalProperties` set to `true` / a sub-schema).
 *
 * The caller's schema object is never mutated. The returned copy is a full
 * deep copy only when `eligible` is `true`: cycle detection short-circuits by
 * returning the original node (marking the schema ineligible), so an
 * ineligible result may share nodes with the input — see
 * {@link StrictSchemaNormalizationResult}.
 *
 * @param schema - The tool's `input_schema` (already validated to be an object
 *   schema by the caller).
 * @returns The normalization result — see {@link StrictSchemaNormalizationResult}.
 */
export function normalizeSchemaForStrict(schema: JSONSchema7): StrictSchemaNormalizationResult {
  let eligible = true
  const visiting = new WeakSet<object>()

  const normalizeDefinitionMap = (value: unknown): Record<string, JSONSchema7Definition> => {
    const out: Record<string, JSONSchema7Definition> = {}
    if (value !== null && typeof value === 'object') {
      const source = value as Record<string, JSONSchema7Definition>
      for (const propName of Object.keys(source)) {
        out[propName] = normalizeDefinition(source[propName]!)
      }
    }
    return out
  }

  const normalizeDefinition = (definition: JSONSchema7Definition): JSONSchema7Definition => {
    if (typeof definition === 'boolean') {
      return definition
    }
    return normalizeNode(definition)
  }

  const normalizeNode = (node: JSONSchema7): JSONSchema7 => {
    if (visiting.has(node)) {
      // A node reachable from itself through the object graph is a recursive
      // schema, which strict mode does not support.
      eligible = false
      return node
    }
    visiting.add(node)

    const result: Record<string, unknown> = {}
    const source = node as Record<string, unknown>

    for (const key of Object.keys(source)) {
      const value = source[key]

      if (STRICT_UNSUPPORTED_KEYWORDS.has(key)) {
        // Drop the unsupported constraint from the copy sent to Anthropic.
        continue
      }

      switch (key) {
        case '$ref':
          // `$ref`/`$defs` are permitted by strict mode only when non-recursive.
          // Recursion cannot be detected reliably from a reference graph, so
          // gate conservatively: any reference-bearing schema is ineligible.
          eligible = false
          result[key] = value
          break
        case '$defs':
        case 'definitions':
          eligible = false
          result[key] = normalizeDefinitionMap(value)
          break
        case 'properties':
        case 'patternProperties':
          result[key] = normalizeDefinitionMap(value)
          break
        case 'additionalProperties':
          if (value === false) {
            result[key] = false
          } else {
            // `additionalProperties: true` or a sub-schema cannot be coerced to
            // `false` without changing semantics, so the schema is ineligible.
            eligible = false
            result[key] = typeof value === 'boolean' ? value : normalizeDefinition(value as JSONSchema7Definition)
          }
          break
        case 'items':
        case 'additionalItems':
        case 'contains':
        case 'propertyNames':
        case 'not':
        case 'if':
        case 'then':
        case 'else':
          result[key] = Array.isArray(value)
            ? (value as JSONSchema7Definition[]).map(normalizeDefinition)
            : normalizeDefinition(value as JSONSchema7Definition)
          break
        case 'anyOf':
        case 'allOf':
        case 'oneOf':
          result[key] = (value as JSONSchema7Definition[]).map(normalizeDefinition)
          break
        default:
          result[key] = value
      }
    }

    // Strict mode requires `additionalProperties: false` and a `required` array
    // on every object schema.
    const isObjectSchema = result['type'] === 'object' || 'properties' in result
    if (isObjectSchema) {
      if (!('additionalProperties' in result)) {
        result['additionalProperties'] = false
      }
      if (!('required' in result)) {
        result['required'] = []
      }
    }

    visiting.delete(node)
    return result as JSONSchema7
  }

  const normalizedSchema = normalizeNode(schema)
  return { eligible, schema: normalizedSchema }
}
