/**
 * OpenAI Responses mapper computer-use coverage.
 *
 * The fixtures in this file intentionally mirror the installed openai@6.27.0
 * Responses declarations and the Comms Agent computer-use contract.
 */

import { describe, expect, it } from '@jest/globals'
import {
  ComputerAction as OpenAIComputerAction,
  Response,
  ResponseComputerToolCall
} from 'openai/resources/responses/responses'
import {
  mapFromOpenAIResponsesResponse,
  mapOpenAIComputerCallToDecision,
  mapOpenAIResponsesStream,
  mapToOpenAIResponsesParams
} from '../../../../src/core/mapping/openai.responses.mapper'
import {
  mapOpenAIComputerAction,
  normalizeProviderDelta,
  normalizeProviderPoint
} from '../../../../src/core/mapping/openai.computer-use'
import {
  CreateResponseParams,
  ResponsesInputItem,
  ResponsesStreamChunk,
  ResponsesTool
} from '../../../../src/types/responses.types'
import { Provider } from '../../../../src/types/common.types'
import {
  ComputerUseMappingError,
  InvalidToolDefinitionError,
  MappingError,
  ToolArgumentValidationError
} from '../../../../src/errors'
import { z } from 'zod'

const dimensions = { width: 1280, height: 720 }

function asOpenAIAction(value: unknown): OpenAIComputerAction {
  return value as OpenAIComputerAction
}

function computerCall(action: unknown, overrides: Record<string, unknown> = {}): ResponseComputerToolCall {
  return ({
    id: 'item_computer_1',
    call_id: 'call_computer_1',
    pending_safety_checks: [],
    status: 'completed',
    type: 'computer_call',
    actions: [action],
    ...overrides
  } as unknown) as ResponseComputerToolCall
}

function singularComputerCall(action: unknown): ResponseComputerToolCall {
  return ({
    id: 'item_singular_1',
    call_id: 'call_singular_1',
    pending_safety_checks: [],
    status: 'completed',
    type: 'computer_call',
    action
  } as unknown) as ResponseComputerToolCall
}

function computerResponse(action: unknown, overrides: Record<string, unknown> = {}): Response {
  return ({
    id: 'resp_computer_1',
    model: 'gpt-5',
    output: [computerCall(action)],
    output_text: '',
    ...overrides
  } as unknown) as Response
}

function expectComputerError(fn: () => unknown, code: ComputerUseMappingError['code']): void {
  try {
    fn()
  } catch (error) {
    expect(error).toBeInstanceOf(ComputerUseMappingError)
    expect((error as ComputerUseMappingError).code).toBe(code)
    return
  }
  throw new Error(`Expected ${code} to be thrown`)
}

describe('OpenAI Responses mapper', () => {
  describe('request mapping', () => {
    it('maps the GA computer tool and a forced computer tool choice exactly', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        input: 'Inspect the browser state.',
        tools: [{ type: 'computer' }],
        tool_choice: { type: 'computer' }
      }

      expect(mapToOpenAIResponsesParams(params)).toEqual({
        model: 'gpt-5',
        input: 'Inspect the browser state.',
        tools: [{ type: 'computer' }],
        tool_choice: { type: 'computer' },
        stream: false
      })
    })

    it('wraps mixed text and image input as one user message using native content parts', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        input: [
          { type: 'input_text', text: 'What is visible?' },
          { type: 'input_image', image_url: 'https://example.test/source.png' },
          { type: 'input_image', image: { mimeType: 'image/jpeg', base64Data: 'encoded-image' } }
        ]
      }

      expect(mapToOpenAIResponsesParams(params)).toEqual({
        model: 'gpt-5',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              { type: 'input_text', text: 'What is visible?' },
              { type: 'input_image', image_url: 'https://example.test/source.png', detail: 'auto' },
              { type: 'input_image', image_url: 'data:image/jpeg;base64,encoded-image', detail: 'auto' }
            ]
          }
        ],
        stream: false
      })
    })

    it('preserves explicit Responses function strictness without imposing it on existing tools', () => {
      const parameters = {
        type: 'object' as const,
        properties: { value: { type: 'string' as const } },
        required: ['value'],
        additionalProperties: false
      }
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        tools: [
          { type: 'function', name: 'explicit_non_strict', parameters, strict: false },
          { type: 'function', name: 'provider_default', parameters }
        ]
      }

      expect(mapToOpenAIResponsesParams(params).tools).toEqual([
        { type: 'function', name: 'explicit_non_strict', parameters, strict: false },
        { type: 'function', name: 'provider_default', parameters, strict: null }
      ])
    })

    it('keeps GA web search available as a tool and maps code interpreter explicitly', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        tools: [{ type: 'web_search' }, { type: 'code_interpreter' }]
      }

      expect(mapToOpenAIResponsesParams(params).tools).toEqual([
        { type: 'web_search' },
        { type: 'code_interpreter', container: { type: 'auto' } }
      ])
    })

    it('maps response continuation, screenshot data, and caller acknowledgements without provider leakage', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        previous_response_id: 'resp_previous',
        input: [
          {
            type: 'computer_call_output',
            call_id: 'call_computer_1',
            output: {
              type: 'computer_screenshot',
              image: { mimeType: 'image/png', base64Data: 'fresh-screenshot' }
            },
            acknowledged_safety_checks: [
              { id: 'safety_1', code: 'policy_check', message: 'Caller authorized this continuation.' },
              { id: 'safety_2', code: null, message: null }
            ]
          }
        ]
      }

      expect(mapToOpenAIResponsesParams(params)).toEqual({
        model: 'gpt-5',
        previous_response_id: 'resp_previous',
        input: [
          {
            type: 'computer_call_output',
            call_id: 'call_computer_1',
            output: { type: 'computer_screenshot', image_url: 'data:image/png;base64,fresh-screenshot' },
            acknowledged_safety_checks: [
              { id: 'safety_1', code: 'policy_check', message: 'Caller authorized this continuation.' },
              { id: 'safety_2', code: null, message: null }
            ]
          }
        ],
        stream: false
      })
    })

    it('preserves a Responses screenshot file ID in a continuation', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        input: [
          {
            type: 'computer_call_output',
            call_id: 'call_computer_1',
            output: { type: 'computer_screenshot', file_id: 'file_screenshot_1' }
          }
        ]
      }

      expect(mapToOpenAIResponsesParams(params)).toEqual({
        model: 'gpt-5',
        input: [
          {
            type: 'computer_call_output',
            call_id: 'call_computer_1',
            output: { type: 'computer_screenshot', file_id: 'file_screenshot_1' }
          }
        ],
        stream: false
      })
    })

    it('rejects an unknown runtime input discriminator instead of treating it as computer output', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        input: [({ type: 'future_input' } as unknown) as ResponsesInputItem]
      }

      expect(() => mapToOpenAIResponsesParams(params)).toThrow(MappingError)
      expect(() => mapToOpenAIResponsesParams(params)).toThrow('Unsupported Responses input item type: future_input')
    })

    it('rejects an unknown runtime tool discriminator instead of mapping code interpreter', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        tools: [({ type: 'future_tool' } as unknown) as ResponsesTool]
      }

      expect(() => mapToOpenAIResponsesParams(params)).toThrow(InvalidToolDefinitionError)
      expect(() => mapToOpenAIResponsesParams(params)).toThrow('Unsupported Responses tool type: future_tool')
    })

    it('excludes forced GA web search from the public type and rejects legacy runtime input', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        // @ts-expect-error openai@6.27.0 declares web_search as a tool, but not as a forced tool choice.
        tool_choice: { type: 'web_search' }
      }

      expect(() => mapToOpenAIResponsesParams(params)).toThrow(MappingError)
    })

    it('rejects stop sequences that the installed Responses request type does not declare', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-5',
        stop: 'END'
      }

      expect(() => mapToOpenAIResponsesParams(params)).toThrow(MappingError)
    })
  })

  describe('function calls', () => {
    it('preserves the distinct provider item ID and function call correlation ID', () => {
      const raw = computerResponse(
        { type: 'wait' },
        {
          output: [
            {
              type: 'function_call',
              id: 'fc_item_1',
              call_id: 'call_function_1',
              name: 'inspect_state',
              arguments: '{"scope":"visible"}',
              status: 'completed'
            }
          ]
        }
      )

      const mapped = mapFromOpenAIResponsesResponse(raw)
      expect(mapped.output).toEqual([
        {
          type: 'function_call',
          id: 'fc_item_1',
          call_id: 'call_function_1',
          name: 'inspect_state',
          arguments: '{"scope":"visible"}'
        }
      ])
      expect(mapped.tool_calls).toEqual([
        {
          id: 'fc_item_1',
          call_id: 'call_function_1',
          type: 'function',
          function: { name: 'inspect_state', arguments: '{"scope":"visible"}' }
        }
      ])
    })
  })

  describe('native computer calls', () => {
    it('maps the contract scroll fixture with identity, safety, usage, and raw response preservation', () => {
      const raw = computerResponse(
        { type: 'scroll', x: 1279, y: 719, scroll_x: -1279, scroll_y: 719 },
        {
          id: 'resp_1',
          output: [
            computerCall(
              { type: 'scroll', x: 1279, y: 719, scroll_x: -1279, scroll_y: 719 },
              {
                id: 'cu_item_1',
                call_id: 'call_1',
                pending_safety_checks: [{ id: 'safe_1' }]
              }
            )
          ],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 }
        }
      )

      const mapped = mapFromOpenAIResponsesResponse(raw)
      expect(mapped).toEqual({
        id: 'resp_1',
        model: 'gpt-5',
        output: [
          {
            type: 'computer_call',
            status: 'completed',
            decision: {
              schemaVersion: '1',
              actionId: 'call_1',
              actions: [{ kind: 'scroll', point: { x: 1, y: 1 }, deltaX: -1, deltaY: 1 }],
              providerTraceId: 'cu_item_1',
              responseId: 'resp_1',
              pendingSafetyChecks: [{ id: 'safe_1', code: null, message: null }]
            }
          }
        ],
        output_text: '',
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        rawResponse: raw
      })
      expect(mapped.rawResponse).toBe(raw)
    })

    it('maps the native screenshot fixture to the canonical request action', () => {
      const mapped = mapFromOpenAIResponsesResponse(
        computerResponse(
          { type: 'screenshot' },
          {
            id: 'resp_2',
            output: [
              computerCall({ type: 'screenshot' }, { id: 'cu_item_2', call_id: 'call_2', pending_safety_checks: [] })
            ]
          }
        )
      )

      expect(mapped.output).toEqual([
        {
          type: 'computer_call',
          status: 'completed',
          decision: {
            schemaVersion: '1',
            actionId: 'call_2',
            actions: [{ kind: 'request_screenshot' }],
            providerTraceId: 'cu_item_2',
            responseId: 'resp_2',
            pendingSafetyChecks: []
          }
        }
      ])
    })

    it('preserves safety strings and nulls, normalizes only missing values, and retains order', () => {
      const call = computerCall(
        { type: 'wait' },
        {
          pending_safety_checks: [
            { id: 'missing' },
            { id: 'undefined', code: undefined, message: undefined },
            { id: 'null', code: null, message: null },
            { id: 'strings', code: 'code-exact', message: 'Message bytes stay exact.' }
          ]
        }
      )

      expect(mapOpenAIComputerCallToDecision(call, 'resp_safety').pendingSafetyChecks).toEqual([
        { id: 'missing', code: null, message: null },
        { id: 'undefined', code: null, message: null },
        { id: 'null', code: null, message: null },
        { id: 'strings', code: 'code-exact', message: 'Message bytes stay exact.' }
      ])
    })

    it.each([
      [
        'left click',
        { type: 'click', button: 'left', x: 0, y: 0 },
        { kind: 'click', point: { x: 0, y: 0 }, button: 'left' }
      ],
      [
        'right click',
        { type: 'click', button: 'right', x: 1279, y: 719 },
        { kind: 'click', point: { x: 1, y: 1 }, button: 'right' }
      ],
      [
        'double click',
        { type: 'double_click', x: 640, y: 360 },
        { kind: 'double_click', point: { x: 640 / 1279, y: 360 / 719 }, button: 'left' }
      ],
      ['move', { type: 'move', x: 640, y: 360 }, { kind: 'move', point: { x: 640 / 1279, y: 360 / 719 } }],
      [
        'drag',
        {
          type: 'drag',
          path: [
            { x: 0, y: 0 },
            { x: 1279, y: 719 }
          ]
        },
        {
          kind: 'drag',
          path: [
            { x: 0, y: 0 },
            { x: 1, y: 1 }
          ],
          button: 'left'
        }
      ],
      [
        'scroll with default center',
        { type: 'scroll', scroll_x: 1, scroll_y: -1 },
        { kind: 'scroll', point: { x: 0.5, y: 0.5 }, deltaX: 1 / 1279, deltaY: -1 / 719 }
      ],
      ['keypress', { type: 'keypress', keys: ['CTRL', 'aLt'] }, { kind: 'press_key', keys: ['Control', 'Alt'] }],
      ['type', { type: 'type', text: 'hello' }, { kind: 'type_text', text: 'hello' }],
      ['wait', { type: 'wait' }, { kind: 'wait', milliseconds: 1000 }],
      ['screenshot', { type: 'screenshot' }, { kind: 'request_screenshot' }]
    ])('maps every native action: %s', (_name, action, expected) => {
      expect(mapOpenAIComputerAction(asOpenAIAction(action), 'pixels', dimensions)).toEqual(expected)
    })

    it.each([
      ['ALT', 'Alt'],
      ['OPTION', 'Alt'],
      ['ARROWDOWN', 'ArrowDown'],
      ['DOWN', 'ArrowDown'],
      ['ARROWLEFT', 'ArrowLeft'],
      ['LEFT', 'ArrowLeft'],
      ['ARROWRIGHT', 'ArrowRight'],
      ['RIGHT', 'ArrowRight'],
      ['ARROWUP', 'ArrowUp'],
      ['UP', 'ArrowUp'],
      ['BACKSPACE', 'Backspace'],
      ['CONTROL', 'Control'],
      ['CTRL', 'Control'],
      ['DELETE', 'Delete'],
      ['DEL', 'Delete'],
      ['END', 'End'],
      ['ENTER', 'Enter'],
      ['RETURN', 'Enter'],
      ['ESC', 'Escape'],
      ['ESCAPE', 'Escape'],
      ['HOME', 'Home'],
      ['META', 'Meta'],
      ['CMD', 'Meta'],
      ['COMMAND', 'Meta'],
      ['PAGEDOWN', 'PageDown'],
      ['PAGEUP', 'PageUp'],
      ['SHIFT', 'Shift'],
      ['TAB', 'Tab']
    ])('normalizes the closed key alias %s case-insensitively', (providerKey, canonicalKey) => {
      expect(
        mapOpenAIComputerAction(
          asOpenAIAction({ type: 'keypress', keys: [providerKey.toLowerCase()] }),
          'pixels',
          dimensions
        )
      ).toEqual({ kind: 'press_key', keys: [canonicalKey] })
    })
  })

  describe('coordinate conversion', () => {
    it('uses inclusive pixel divisors, 0-1000 divisors, and normalized values without clamping', () => {
      expect(normalizeProviderPoint(1279, 719, 'pixels', dimensions)).toEqual({ x: 1, y: 1 })
      expect(normalizeProviderDelta(-1279, 'x', 'pixels', dimensions)).toBe(-1)
      expect(normalizeProviderDelta(719, 'y', 'pixels', dimensions)).toBe(1)

      expect(normalizeProviderPoint(1000, 500, '0-1000', dimensions)).toEqual({ x: 1, y: 0.5 })
      expect(normalizeProviderDelta(-1000, 'x', '0-1000', dimensions)).toBe(-1)
      expect(normalizeProviderDelta(500, 'y', '0-1000', dimensions)).toBe(0.5)

      expect(normalizeProviderPoint(0.25, 0.75, 'normalized', dimensions)).toEqual({ x: 0.25, y: 0.75 })
      expect(normalizeProviderDelta(-0.5, 'x', 'normalized', dimensions)).toBe(-0.5)
    })
  })

  describe('fail-closed validation', () => {
    it.each([
      ['singular-only action', singularComputerCall({ type: 'wait' }), 'PROVIDER_ACTION_SHAPE_UNSUPPORTED'],
      [
        'mixed singular and batch action',
        computerCall({ type: 'wait' }, { action: { type: 'wait' } }),
        'PROVIDER_ACTION_SHAPE_UNSUPPORTED'
      ],
      [
        'missing singular and batch action',
        ({
          id: 'item_missing',
          call_id: 'call_missing',
          pending_safety_checks: [],
          status: 'completed',
          type: 'computer_call'
        } as unknown) as ResponseComputerToolCall,
        'PROVIDER_ACTION_SHAPE_UNSUPPORTED'
      ],
      ['empty action batch', computerCall({ type: 'wait' }, { actions: [] }), 'PROVIDER_ACTION_BATCH_UNSUPPORTED'],
      [
        'multiple actions',
        computerCall({ type: 'wait' }, { actions: [{ type: 'wait' }, { type: 'screenshot' }] }),
        'PROVIDER_ACTION_BATCH_UNSUPPORTED'
      ],
      ['empty action id', computerCall({ type: 'wait' }, { call_id: '' }), 'PROVIDER_ACTION_INVALID'],
      ['overlong action id', computerCall({ type: 'wait' }, { call_id: 'a'.repeat(201) }), 'PROVIDER_ACTION_INVALID']
    ] as const)('rejects %s with the exact error code', (_name, call, code) => {
      expectComputerError(() => mapOpenAIComputerCallToDecision(call, 'resp_1'), code)
    })

    it.each([
      { type: 'click', button: 'left', x: 0, y: 0, keys: [] },
      { type: 'double_click', x: 0, y: 0, keys: [] },
      { type: 'move', x: 0, y: 0, keys: [] },
      {
        type: 'drag',
        path: [
          { x: 0, y: 0 },
          { x: 1, y: 1 }
        ],
        keys: []
      },
      { type: 'scroll', x: 0, y: 0, scroll_x: 1, scroll_y: 1, keys: [] }
    ])('rejects an own modifiers property on every mouse action: $type', action => {
      expectComputerError(
        () => mapOpenAIComputerAction(asOpenAIAction(action), 'pixels', dimensions),
        'PROVIDER_ACTION_MODIFIERS_UNSUPPORTED'
      )
    })

    it.each([
      [
        'mouse modifiers',
        { type: 'click', button: 'left', x: 0, y: 0, keys: ['SHIFT'] },
        'PROVIDER_ACTION_MODIFIERS_UNSUPPORTED'
      ],
      ['wheel button', { type: 'click', button: 'wheel', x: 0, y: 0 }, 'PROVIDER_ACTION_UNSUPPORTED'],
      ['back button', { type: 'click', button: 'back', x: 0, y: 0 }, 'PROVIDER_ACTION_UNSUPPORTED'],
      ['forward button', { type: 'click', button: 'forward', x: 0, y: 0 }, 'PROVIDER_ACTION_UNSUPPORTED'],
      ['short drag', { type: 'drag', path: [{ x: 0, y: 0 }] }, 'PROVIDER_ACTION_INVALID'],
      [
        'long drag',
        { type: 'drag', path: Array.from({ length: 33 }, () => ({ x: 0, y: 0 })) },
        'PROVIDER_ACTION_INVALID'
      ],
      ['unknown key', { type: 'keypress', keys: ['UNKNOWN'] }, 'PROVIDER_ACTION_INVALID'],
      ['duplicate normalized key', { type: 'keypress', keys: ['ALT', 'OPTION'] }, 'PROVIDER_ACTION_INVALID'],
      [
        'too many keys',
        { type: 'keypress', keys: ['ALT', 'CTRL', 'SHIFT', 'META', 'ENTER'] },
        'PROVIDER_ACTION_INVALID'
      ],
      ['pixel point outside width', { type: 'click', button: 'left', x: 1280, y: 0 }, 'PROVIDER_ACTION_INVALID'],
      ['pixel point outside height', { type: 'click', button: 'left', x: 0, y: 720 }, 'PROVIDER_ACTION_INVALID'],
      ['pixel delta outside width', { type: 'scroll', scroll_x: 1280, scroll_y: 1 }, 'PROVIDER_ACTION_INVALID'],
      ['pixel delta outside height', { type: 'scroll', scroll_x: 1, scroll_y: -720 }, 'PROVIDER_ACTION_INVALID'],
      ['non-finite coordinate', { type: 'move', x: Number.NaN, y: 0 }, 'PROVIDER_ACTION_INVALID'],
      [
        'non-finite delta',
        { type: 'scroll', scroll_x: 1, scroll_y: Number.POSITIVE_INFINITY },
        'PROVIDER_ACTION_INVALID'
      ],
      ['zero scroll', { type: 'scroll', scroll_x: 0, scroll_y: 0 }, 'PROVIDER_ACTION_INVALID'],
      ['unknown action', { type: 'unknown' }, 'PROVIDER_ACTION_UNSUPPORTED'],
      ['partial scroll point', { type: 'scroll', x: 0, scroll_x: 1, scroll_y: 1 }, 'PROVIDER_ACTION_INVALID']
    ] as const)('rejects %s with the exact error code', (_name, action, code) => {
      expectComputerError(() => mapOpenAIComputerAction(asOpenAIAction(action), 'pixels', dimensions), code)
    })

    it.each([
      ['0-1000 negative point', () => normalizeProviderPoint(-1, 0, '0-1000', dimensions)],
      ['0-1000 oversized point', () => normalizeProviderPoint(1001, 0, '0-1000', dimensions)],
      ['0-1000 negative delta', () => normalizeProviderDelta(-1001, 'x', '0-1000', dimensions)],
      ['0-1000 oversized delta', () => normalizeProviderDelta(1001, 'y', '0-1000', dimensions)],
      ['normalized negative point', () => normalizeProviderPoint(-0.01, 0, 'normalized', dimensions)],
      ['normalized oversized point', () => normalizeProviderPoint(1.01, 0, 'normalized', dimensions)],
      ['normalized negative delta', () => normalizeProviderDelta(-1.01, 'x', 'normalized', dimensions)],
      ['normalized oversized delta', () => normalizeProviderDelta(1.01, 'y', 'normalized', dimensions)]
    ])('rejects provider-coordinate boundary violations: %s', (_name, invoke) => {
      expectComputerError(invoke, 'PROVIDER_ACTION_INVALID')
    })

    it('does not emit a mapped computer output when any native action shape is rejected', () => {
      const raw = computerResponse({ type: 'wait' }, { output: [computerCall({ type: 'wait' }, { actions: [] })] })
      expectComputerError(() => mapFromOpenAIResponsesResponse(raw), 'PROVIDER_ACTION_BATCH_UNSUPPORTED')
      expect(raw.output).toEqual([computerCall({ type: 'wait' }, { actions: [] })])
    })
  })

  describe('streaming', () => {
    it('maps a completed stream event through the same computer decision path', async () => {
      const raw = computerResponse({ type: 'screenshot' }, { id: 'resp_stream' })
      const stream = (async function*(): AsyncIterable<unknown> {
        yield { type: 'response.completed', response: raw }
      })()
      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(stream)) chunks.push(chunk)

      expect(chunks).toEqual([
        {
          type: 'response.completed',
          data: {
            id: 'resp_stream',
            model: 'gpt-5',
            output: [
              {
                type: 'computer_call',
                status: 'completed',
                decision: {
                  schemaVersion: '1',
                  actionId: 'call_computer_1',
                  actions: [{ kind: 'request_screenshot' }],
                  providerTraceId: 'item_computer_1',
                  responseId: 'resp_stream',
                  pendingSafetyChecks: []
                }
              }
            ],
            output_text: '',
            rawResponse: raw
          }
        }
      ])
    })

    it('maps a failed response event without losing provider error context', async () => {
      const stream = (async function*(): AsyncIterable<unknown> {
        yield { type: 'response.created', response: { id: 'resp_failed', model: 'gpt-5' } }
        yield { type: 'response.failed', error: { message: 'Rate limit exceeded', code: 'rate_limit' } }
      })()
      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(stream)) chunks.push(chunk)

      expect(chunks).toEqual([
        { type: 'response.created', data: { id: 'resp_failed', model: 'gpt-5' } },
        {
          type: 'response.failed',
          data: { error: { message: 'Rate limit exceeded', code: 'rate_limit' } }
        }
      ])
    })

    it('emits a typed error for schema-invalid streamed function arguments', async () => {
      const tools: ResponsesTool[] = [
        {
          type: 'function',
          name: 'get_weather',
          parameters: {
            type: 'object',
            properties: { location: { type: 'string' } },
            required: ['location']
          },
          zodSchema: z.object({ location: z.string() })
        }
      ]
      const stream = (async function*(): AsyncIterable<unknown> {
        yield { type: 'response.tool_call.start', tool_call: { id: 'call_weather', name: 'get_weather' } }
        yield {
          type: 'response.tool_call.done',
          tool_call: { id: 'call_weather', name: 'get_weather', arguments: '{"location":123}' }
        }
      })()
      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(stream, tools)) chunks.push(chunk)

      expect(chunks).toHaveLength(2)
      expect(chunks[0]).toEqual({
        type: 'response.tool_call.start',
        data: { id: 'call_weather', name: 'get_weather' }
      })
      expect(chunks[1].type).toBe('error')
      if (chunks[1].type !== 'error') throw new Error('Expected a streamed validation error')
      expect(chunks[1].data.error).toBeInstanceOf(ToolArgumentValidationError)
    })
  })
})
