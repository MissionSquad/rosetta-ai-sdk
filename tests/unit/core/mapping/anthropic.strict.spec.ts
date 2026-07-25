import type { JSONSchema7 } from 'json-schema'
import { normalizeSchemaForStrict, supportsStrictToolUse } from '../../../../src/core/mapping/anthropic.strict'

describe('Anthropic strict tool use support', () => {
  describe('supportsStrictToolUse (model gating)', () => {
    it('[Easy] returns true for Claude 4.5+ models (GA target range)', () => {
      const supported = [
        'claude-fable-5',
        'claude-mythos-5',
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-opus-4-8',
        'claude-opus-4-7',
        'claude-opus-4-6',
        'claude-opus-4-5',
        'claude-opus-4-5-20251101',
        'claude-sonnet-4-6',
        'claude-sonnet-4-5',
        'claude-sonnet-4-5-20250929',
        'claude-haiku-4-5',
        'claude-haiku-4-5-20251001'
      ]
      for (const model of supported) {
        expect(supportsStrictToolUse(model)).toBe(true)
      }
    })

    it('[Easy] tolerates the :1m context suffix', () => {
      expect(supportsStrictToolUse('claude-opus-4-6:1m')).toBe(true)
      expect(supportsStrictToolUse('claude-sonnet-4-5:1m')).toBe(true)
    })

    it('[Medium] strips any trailing colon-delimited suffix, not just :1m', () => {
      expect(supportsStrictToolUse('claude-sonnet-4-5:thinking')).toBe(true)
      expect(supportsStrictToolUse('claude-opus-4-6:batch')).toBe(true)
      expect(supportsStrictToolUse('claude-haiku-4-5:some-future-suffix')).toBe(true)
      // Pre-4.5 families stay denied whatever the suffix is.
      expect(supportsStrictToolUse('claude-opus-4-1:thinking')).toBe(false)
      expect(supportsStrictToolUse('claude-opus-4-1:1m')).toBe(false)
    })

    it('[Medium] strips the suffix only at the end of the id, never mid-string', () => {
      // The previous unanchored `replace(':1m', '')` spliced a mid-string ':1m'
      // out, turning these into the denylisted 'claude-sonnet-4-0' /
      // 'claude-opus-4-1'. The end-anchored strip instead removes the whole
      // trailing ':1m-0' / ':1m-1' segment (everything after the last colon),
      // yielding 'claude-sonnet-4' / 'claude-opus-4' — no denylist match, no
      // splice into a different ID.
      expect(supportsStrictToolUse('claude-sonnet-4:1m-0')).toBe(true)
      expect(supportsStrictToolUse('claude-opus-4:1m-1')).toBe(true)
      // The claude- prefix gate still applies to mid-string-suffix ids.
      expect(supportsStrictToolUse('gpt-4:1man-x')).toBe(false)
    })

    it('[Easy] returns false for non-Anthropic model ids', () => {
      const nonAnthropic = ['gpt-4o', 'gpt-5', 'gemini-2.5-pro', 'grok-4', 'claude', 'anthropic.claude-sonnet-4-5']
      for (const model of nonAnthropic) {
        expect(supportsStrictToolUse(model)).toBe(false)
      }
    })

    it('[Medium] returns false for known pre-4.5 families and their dated forms', () => {
      const unsupported = [
        'claude-3-haiku-20240307',
        'claude-3-5-sonnet-20241022',
        'claude-3-opus-20240229',
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2',
        'claude-opus-4-1',
        'claude-opus-4-1-20250805',
        'claude-opus-4-0',
        'claude-opus-4-20250514',
        'claude-sonnet-4-0',
        'claude-sonnet-4-20250514',
        'claude-sonnet-4-0:1m'
      ]
      for (const model of unsupported) {
        expect(supportsStrictToolUse(model)).toBe(false)
      }
    })

    it('[Easy] returns false for an absent or empty model id', () => {
      expect(supportsStrictToolUse(undefined)).toBe(false)
      expect(supportsStrictToolUse(null)).toBe(false)
      expect(supportsStrictToolUse('')).toBe(false)
      expect(supportsStrictToolUse('   ')).toBe(false)
    })
  })

  describe('normalizeSchemaForStrict', () => {
    it('[Easy] sets additionalProperties:false and preserves an existing required array', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location']
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      expect(result.schema).toEqual({
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
        additionalProperties: false
      })
    })

    it('[Medium] synthesizes an empty required array when properties exist but required is absent', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: { note: { type: 'string' } }
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      expect(result.schema).toEqual({
        type: 'object',
        properties: { note: { type: 'string' } },
        additionalProperties: false,
        required: []
      })
    })

    it('[Medium] treats a schema with properties but no explicit type:object as an object schema', () => {
      const schema: JSONSchema7 = {
        properties: { x: { type: 'string' } }
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      expect(result.schema).toEqual({
        properties: { x: { type: 'string' } },
        additionalProperties: false,
        required: []
      })
    })

    it('[Medium] normalizes nested object schemas recursively', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: { name: { type: 'string' } },
            required: ['name']
          }
        },
        required: ['user']
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      expect(result.schema).toEqual({
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: { name: { type: 'string' } },
            required: ['name'],
            additionalProperties: false
          }
        },
        required: ['user'],
        additionalProperties: false
      })
    })

    it('[Medium] normalizes object branches inside anyOf and array items', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: {
          items: {
            type: 'array',
            items: {
              type: 'object',
              properties: { id: { type: 'string' } },
              required: ['id']
            }
          },
          choice: {
            anyOf: [
              { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] },
              { type: 'null' }
            ]
          }
        },
        required: ['items']
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      const props = result.schema.properties as Record<string, JSONSchema7>
      expect((props.items!.items as JSONSchema7).additionalProperties).toBe(false)
      const anyOf = props.choice!.anyOf as JSONSchema7[]
      expect(anyOf[0]!.additionalProperties).toBe(false)
      expect(anyOf[0]!.required).toEqual(['a'])
      expect(anyOf[1]).toEqual({ type: 'null' })
    })

    it('[Medium] strips unsupported constraint keywords but keeps pattern, enum, and format', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: {
          age: { type: 'integer', minimum: 0, maximum: 120, multipleOf: 1, exclusiveMinimum: -1, exclusiveMaximum: 200 },
          name: { type: 'string', minLength: 1, maxLength: 50, pattern: '^[a-z]+$' },
          tags: { type: 'array', items: { type: 'string' }, minItems: 1, maxItems: 10 },
          unit: { type: 'string', enum: ['c', 'f'] },
          when: { type: 'string', format: 'date-time' }
        },
        required: ['age']
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(true)
      const props = result.schema.properties as Record<string, JSONSchema7>
      expect(props.age).toEqual({ type: 'integer' })
      expect(props.name).toEqual({ type: 'string', pattern: '^[a-z]+$' })
      expect(props.tags).toEqual({ type: 'array', items: { type: 'string' } })
      expect(props.unit).toEqual({ type: 'string', enum: ['c', 'f'] })
      expect(props.when).toEqual({ type: 'string', format: 'date-time' })
    })

    it('[Hard] does not mutate the caller schema', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: { location: { type: 'string', minLength: 2 } }
      }
      const snapshot = JSON.parse(JSON.stringify(schema))
      normalizeSchemaForStrict(schema)
      expect(schema).toEqual(snapshot)
      expect(schema.additionalProperties).toBeUndefined()
      expect((schema.properties!.location as JSONSchema7).minLength).toBe(2)
    })

    it('[Medium] marks the schema ineligible when additionalProperties is true (semantics cannot be overridden)', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
        additionalProperties: true
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(false)
      // The explicit `true` is preserved (not silently coerced to false).
      expect(result.schema.additionalProperties).toBe(true)
    })

    it('[Medium] marks the schema ineligible when additionalProperties is a sub-schema', () => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: {},
        additionalProperties: { type: 'string' }
      }
      const result = normalizeSchemaForStrict(schema)
      expect(result.eligible).toBe(false)
    })

    it('[Medium] marks the schema ineligible when it uses $ref or $defs', () => {
      const refSchema: JSONSchema7 = {
        type: 'object',
        properties: { child: { $ref: '#/$defs/node' } },
        required: ['child'],
        $defs: {
          node: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] }
        }
      }
      const result = normalizeSchemaForStrict(refSchema)
      expect(result.eligible).toBe(false)
    })
  })
})
