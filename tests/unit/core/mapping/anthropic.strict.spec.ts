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
