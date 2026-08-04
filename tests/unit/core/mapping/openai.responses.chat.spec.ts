import { z } from 'zod'
import type { ResponseStreamEvent, Response } from 'openai/resources/responses/responses'

import {
  generateViaOpenAIResponses,
  mapOpenAIResponsesChatResponse,
  mapOpenAIResponsesChatStream,
  mapToOpenAIResponsesChatParams,
  shouldUseOpenAIResponsesApi
} from '../../../../src/core/mapping/openai.responses.chat'
import { GenerateParams, Provider, StreamChunk } from '../../../../src/types'
import { ToolArgumentValidationError } from '../../../../src/errors'

const WEATHER_TOOL = {
  type: 'function' as const,
  function: {
    name: 'get_weather',
    description: 'Get current weather for a city',
    parameters: {
      type: 'object' as const,
      properties: { city: { type: 'string' } },
      required: ['city']
    },
    zodSchema: z.object({ city: z.string() })
  }
}

const baseParams: GenerateParams = {
  provider: Provider.OpenAI,
  model: 'gpt-5.6-terra',
  messages: [{ role: 'user', content: 'Hello' }]
}

function reasoningItem(id: string): Record<string, unknown> {
  return {
    id,
    type: 'reasoning',
    summary: [{ type: 'summary_text', text: 'Considering options.' }],
    encrypted_content: 'opaque-encrypted-blob'
  }
}

async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

describe('shouldUseOpenAIResponsesApi', () => {
  it('routes gpt-5.x models to the Responses API', () => {
    expect(shouldUseOpenAIResponsesApi({ ...baseParams, model: 'gpt-5.6-terra' })).toBe(true)
    expect(shouldUseOpenAIResponsesApi({ ...baseParams, model: 'gpt-5.4-mini' })).toBe(true)
    expect(shouldUseOpenAIResponsesApi({ ...baseParams, model: 'gpt-5' })).toBe(true)
  })

  it('keeps non-gpt-5 models on Chat Completions', () => {
    expect(shouldUseOpenAIResponsesApi({ ...baseParams, model: 'gpt-4o' })).toBe(false)
    expect(shouldUseOpenAIResponsesApi({ ...baseParams, model: 'o3-mini' })).toBe(false)
  })

  it('honors the openaiPreferChatCompletions escape hatch', () => {
    expect(
      shouldUseOpenAIResponsesApi({
        ...baseParams,
        model: 'gpt-5.6-terra',
        providerOptions: { openaiPreferChatCompletions: true }
      })
    ).toBe(false)
  })
})

describe('mapToOpenAIResponsesChatParams', () => {
  it('maps system messages to instructions and keeps the conversation in input', () => {
    const params: GenerateParams = {
      ...baseParams,
      messages: [
        { role: 'system', content: 'Be helpful.' },
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello there' },
        { role: 'user', content: 'Question?' }
      ]
    }
    const mapped = mapToOpenAIResponsesChatParams(params)
    expect(mapped.instructions).toBe('Be helpful.')
    expect(mapped.input).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello there' },
      { role: 'user', content: 'Question?' }
    ])
    expect(mapped.store).toBe(false)
    expect(mapped.include).toEqual(['reasoning.encrypted_content'])
  })

  it('maps image content parts to input_image entries', () => {
    const params: GenerateParams = {
      ...baseParams,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Describe' },
            { type: 'image', image: { mimeType: 'image/png', base64Data: 'aaa=' } }
          ]
        }
      ]
    }
    const mapped = mapToOpenAIResponsesChatParams(params)
    expect(mapped.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'Describe' },
          { type: 'input_image', image_url: 'data:image/png;base64,aaa=', detail: 'auto' }
        ]
      }
    ])
  })

  it('replays assistant tool calls as reasoning items, message, then function_call items', () => {
    const params: GenerateParams = {
      ...baseParams,
      messages: [
        { role: 'user', content: 'Weather in Paris? ' },
        {
          role: 'assistant',
          content: 'Checking.',
          toolCalls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
              providerMetadata: { openaiResponses: { reasoningItems: [reasoningItem('rs_1')] } }
            }
          ]
        },
        { role: 'tool', toolCallId: 'call_1', content: '18C' }
      ]
    }
    const mapped = mapToOpenAIResponsesChatParams(params)
    expect(mapped.input).toEqual([
      { role: 'user', content: 'Weather in Paris? ' },
      reasoningItem('rs_1'),
      { role: 'assistant', content: 'Checking.' },
      { type: 'function_call', call_id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' },
      { type: 'function_call_output', call_id: 'call_1', output: '18C' }
    ])
  })

  it('rejects null or empty system content and null user content like the chat path', () => {
    expect(() =>
      mapToOpenAIResponsesChatParams({ ...baseParams, messages: [{ role: 'system', content: null }] })
    ).toThrow(`Role 'system' requires non-null content.`)
    expect(() =>
      mapToOpenAIResponsesChatParams({ ...baseParams, messages: [{ role: 'system', content: '' }] })
    ).toThrow(`Role 'system' requires non-empty string content.`)
    expect(() =>
      mapToOpenAIResponsesChatParams({ ...baseParams, messages: [{ role: 'user', content: null }] })
    ).toThrow(`Role 'user' requires non-null content.`)
    // Empty user strings remain allowed, matching Chat Completions.
    expect(
      mapToOpenAIResponsesChatParams({ ...baseParams, messages: [{ role: 'user', content: '' }] }).input
    ).toEqual([{ role: 'user', content: '' }])
  })

  it('throws when a tool message is missing its toolCallId', () => {
    const params: GenerateParams = {
      ...baseParams,
      messages: [{ role: 'tool', content: 'orphan result' }]
    }
    expect(() => mapToOpenAIResponsesChatParams(params)).toThrow('missing toolCallId')
  })

  it('maps tools to flat Responses definitions with strict disabled', () => {
    const mapped = mapToOpenAIResponsesChatParams({ ...baseParams, tools: [WEATHER_TOOL] })
    expect(mapped.tools).toEqual([
      {
        type: 'function',
        name: 'get_weather',
        description: 'Get current weather for a city',
        parameters: WEATHER_TOOL.function.parameters,
        strict: false
      }
    ])
  })

  it('maps tool_choice variants', () => {
    expect(mapToOpenAIResponsesChatParams({ ...baseParams, toolChoice: 'auto' }).tool_choice).toBe('auto')
    expect(mapToOpenAIResponsesChatParams({ ...baseParams, toolChoice: 'required' }).tool_choice).toBe('required')
    expect(
      mapToOpenAIResponsesChatParams({
        ...baseParams,
        toolChoice: { type: 'function', function: { name: 'get_weather' } }
      }).tool_choice
    ).toEqual({ type: 'function', name: 'get_weather' })
  })

  it('requests reasoning summaries with a working effort when thinking is enabled', () => {
    // gpt-5.6 defaults to effort 'none', which performs no reasoning — a thinking request
    // upgrades to 'medium' so there is reasoning to disclose.
    const mapped = mapToOpenAIResponsesChatParams({ ...baseParams, thinking: true })
    expect(mapped.reasoning).toEqual({ effort: 'medium', summary: 'auto' })
  })

  it('keeps an explicit reasoningEffort alongside thinking summaries', () => {
    const mapped = mapToOpenAIResponsesChatParams({ ...baseParams, thinking: true, reasoningEffort: 'xhigh' })
    expect(mapped.reasoning).toEqual({ effort: 'xhigh', summary: 'auto' })
  })

  it('passes reasoningEffort through without summaries when thinking is off', () => {
    const mapped = mapToOpenAIResponsesChatParams({ ...baseParams, reasoningEffort: 'low' })
    expect(mapped.reasoning).toEqual({ effort: 'low' })
  })

  it('omits reasoning entirely when neither thinking nor effort are set', () => {
    const mapped = mapToOpenAIResponsesChatParams(baseParams)
    expect(mapped.reasoning).toBeUndefined()
  })

  it('translates a foreign persisted thinkingConfig into a summary request and strips it', () => {
    const mapped = mapToOpenAIResponsesChatParams({
      ...baseParams,
      extraParams: { thinkingConfig: { includeThoughts: true }, customFlag: 'kept' }
    })
    expect(mapped.reasoning).toEqual({ effort: 'medium', summary: 'auto' })
    expect(mapped).not.toHaveProperty('thinkingConfig')
    expect((mapped as unknown as Record<string, unknown>).customFlag).toBe('kept')
  })

  it('omits sampling parameters for gpt-5.6 and maps token limit and json schema', () => {
    const mapped = mapToOpenAIResponsesChatParams({
      ...baseParams,
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 2048,
      responseFormat: {
        type: 'json_schema',
        json_schema: { name: 'result', schema: { type: 'object', properties: {} } }
      }
    })
    expect(mapped).not.toHaveProperty('temperature')
    expect(mapped).not.toHaveProperty('top_p')
    expect(mapped.max_output_tokens).toBe(2048)
    expect(mapped.text).toEqual({
      format: { type: 'json_schema', name: 'result', strict: true, schema: { type: 'object', properties: {} } }
    })
  })

  it('maps verbosity into text config for models that support it', () => {
    const mapped = mapToOpenAIResponsesChatParams({ ...baseParams, verbosity: 'low' })
    expect(mapped.text).toEqual({ verbosity: 'low' })
  })
})

describe('mapOpenAIResponsesChatResponse', () => {
  const makeResponse = (output: unknown[], overrides: Partial<Response> = {}): Response =>
    ({
      id: 'resp_1',
      model: 'gpt-5.6-terra',
      status: 'completed',
      output,
      usage: {
        input_tokens: 10,
        output_tokens: 20,
        total_tokens: 30,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 5 }
      },
      ...overrides
    }) as unknown as Response

  it('maps text, reasoning summaries, and usage', () => {
    const response = makeResponse([
      reasoningItem('rs_1'),
      { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'Answer.' }] }
    ])
    const result = mapOpenAIResponsesChatResponse(response, 'gpt-5.6-terra', undefined)
    expect(result.content).toBe('Answer.')
    expect(result.thinkingSteps).toBe('Considering options.')
    expect(result.finishReason).toBe('stop')
    expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 20, totalTokens: 30 })
  })

  it('maps function calls, validates arguments, and attaches reasoning replay metadata', () => {
    const response = makeResponse([
      reasoningItem('rs_1'),
      { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' }
    ])
    const result = mapOpenAIResponsesChatResponse(response, 'gpt-5.6-terra', [WEATHER_TOOL])
    expect(result.finishReason).toBe('tool_calls')
    expect(result.toolCalls).toEqual([
      {
        id: 'call_1',
        type: 'function',
        function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
        providerMetadata: { openaiResponses: { reasoningItems: [reasoningItem('rs_1')] } }
      }
    ])
  })

  it('throws ToolArgumentValidationError for invalid tool arguments', () => {
    const response = makeResponse([
      { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'get_weather', arguments: '{"city":42}' }
    ])
    expect(() => mapOpenAIResponsesChatResponse(response, 'gpt-5.6-terra', [WEATHER_TOOL])).toThrow(
      ToolArgumentValidationError
    )
  })

  it('maps refusals to content_filter and incomplete max tokens to length', () => {
    const refusal = makeResponse([
      { type: 'message', id: 'msg_1', content: [{ type: 'refusal', refusal: 'Cannot help with that.' }] }
    ])
    const refusalResult = mapOpenAIResponsesChatResponse(refusal, 'gpt-5.6-terra', undefined)
    expect(refusalResult.content).toBe('Cannot help with that.')
    expect(refusalResult.finishReason).toBe('content_filter')

    const truncated = makeResponse([{ type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'par' }] }], {
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' }
    } as Partial<Response>)
    expect(mapOpenAIResponsesChatResponse(truncated, 'gpt-5.6-terra', undefined).finishReason).toBe('length')
  })

  it('auto-parses JSON answers', () => {
    const response = makeResponse([
      { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: '{"a":1}' }] }
    ])
    expect(mapOpenAIResponsesChatResponse(response, 'gpt-5.6-terra', undefined).parsedContent).toEqual({ a: 1 })
  })
})

describe('mapOpenAIResponsesChatStream', () => {
  async function* eventStream(events: unknown[]): AsyncIterable<ResponseStreamEvent> {
    for (const event of events) yield event as ResponseStreamEvent
  }

  const createdEvent = {
    type: 'response.created',
    response: { id: 'resp_1', model: 'gpt-5.6-terra', status: 'in_progress', output: [] }
  }
  const completedEvent = (overrides: Record<string, unknown> = {}) => ({
    type: 'response.completed',
    response: {
      id: 'resp_1',
      model: 'gpt-5.6-terra',
      status: 'completed',
      output: [],
      usage: {
        input_tokens: 7,
        output_tokens: 9,
        total_tokens: 16,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 3 }
      },
      ...overrides
    }
  })

  it('maps reasoning summaries and text into canonical thinking/content chunks', async () => {
    const chunks = await collect(
      mapOpenAIResponsesChatStream(
        eventStream([
          createdEvent,
          { type: 'response.reasoning_summary_part.added', item_id: 'rs_1', output_index: 0, summary_index: 0 },
          { type: 'response.reasoning_summary_text.delta', item_id: 'rs_1', output_index: 0, summary_index: 0, delta: 'Thinking ' },
          { type: 'response.reasoning_summary_text.delta', item_id: 'rs_1', output_index: 0, summary_index: 0, delta: 'hard.' },
          { type: 'response.reasoning_summary_part.done', item_id: 'rs_1', output_index: 0, summary_index: 0 },
          { type: 'response.output_item.done', output_index: 0, item: reasoningItem('rs_1') },
          { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 1, content_index: 0, delta: 'Ans' },
          { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 1, content_index: 0, delta: 'wer' },
          completedEvent()
        ]),
        'gpt-5.6-terra',
        undefined
      )
    )

    expect(chunks.map(c => c.type)).toEqual([
      'message_start',
      'thinking_start',
      'thinking_delta',
      'thinking_delta',
      'thinking_stop',
      'content_delta',
      'content_delta',
      'message_stop',
      'final_usage',
      'final_result'
    ])
    const final = chunks.at(-1) as Extract<StreamChunk, { type: 'final_result' }>
    expect(final.data.result.content).toBe('Answer')
    expect(final.data.result.thinkingSteps).toBe('Thinking hard.')
    expect(final.data.result.finishReason).toBe('stop')
    expect(final.data.result.usage).toEqual({ promptTokens: 7, completionTokens: 9, totalTokens: 16 })
  })

  it('maps streamed function calls with validation and reasoning replay metadata', async () => {
    const fnItem = { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'get_weather', arguments: '' }
    const chunks = await collect(
      mapOpenAIResponsesChatStream(
        eventStream([
          createdEvent,
          { type: 'response.output_item.done', output_index: 0, item: reasoningItem('rs_1') },
          { type: 'response.output_item.added', output_index: 1, item: fnItem },
          { type: 'response.function_call_arguments.delta', item_id: 'fc_1', output_index: 1, delta: '{"city":' },
          { type: 'response.function_call_arguments.delta', item_id: 'fc_1', output_index: 1, delta: '"Paris"}' },
          { type: 'response.function_call_arguments.done', item_id: 'fc_1', output_index: 1, arguments: '{"city":"Paris"}' },
          {
            type: 'response.output_item.done',
            output_index: 1,
            item: { ...fnItem, arguments: '{"city":"Paris"}' }
          },
          completedEvent()
        ]),
        'gpt-5.6-terra',
        [WEATHER_TOOL]
      )
    )

    expect(chunks.map(c => c.type)).toEqual([
      'message_start',
      'tool_call_start',
      'tool_call_delta',
      'tool_call_delta',
      'tool_call_done',
      'message_stop',
      'final_usage',
      'final_result'
    ])
    const final = chunks.at(-1) as Extract<StreamChunk, { type: 'final_result' }>
    expect(final.data.result.finishReason).toBe('tool_calls')
    expect(final.data.result.toolCalls).toEqual([
      {
        id: 'call_1',
        type: 'function',
        function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
        providerMetadata: { openaiResponses: { reasoningItems: [reasoningItem('rs_1')] } }
      }
    ])
  })

  it('yields an error chunk when streamed tool arguments fail validation', async () => {
    const fnItem = { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'get_weather', arguments: '' }
    const chunks = await collect(
      mapOpenAIResponsesChatStream(
        eventStream([
          createdEvent,
          { type: 'response.output_item.added', output_index: 0, item: fnItem },
          {
            type: 'response.output_item.done',
            output_index: 0,
            item: { ...fnItem, arguments: '{"city":42}' }
          },
          completedEvent()
        ]),
        'gpt-5.6-terra',
        [WEATHER_TOOL]
      )
    )
    const error = chunks.find(c => c.type === 'error') as Extract<StreamChunk, { type: 'error' }>
    expect(error).toBeDefined()
    expect(error.data.error).toBeInstanceOf(ToolArgumentValidationError)
  })

  it('streams JSON answers as json_delta/json_done', async () => {
    const chunks = await collect(
      mapOpenAIResponsesChatStream(
        eventStream([
          createdEvent,
          { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 0, content_index: 0, delta: '{"a":' },
          { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 0, content_index: 0, delta: '1}' },
          completedEvent()
        ]),
        'gpt-5.6-terra',
        undefined
      )
    )
    expect(chunks.map(c => c.type)).toEqual([
      'message_start',
      'json_delta',
      'json_delta',
      'json_done',
      'message_stop',
      'final_usage',
      'final_result'
    ])
    const final = chunks.at(-1) as Extract<StreamChunk, { type: 'final_result' }>
    expect(final.data.result.parsedContent).toEqual({ a: 1 })
  })

  it('maps response.failed into an error chunk', async () => {
    const chunks = await collect(
      mapOpenAIResponsesChatStream(
        eventStream([
          createdEvent,
          {
            type: 'response.failed',
            response: { id: 'resp_1', model: 'gpt-5.6-terra', status: 'failed', output: [], error: { code: 'server_error', message: 'boom' } }
          }
        ]),
        'gpt-5.6-terra',
        undefined
      )
    )
    const error = chunks.find(c => c.type === 'error') as Extract<StreamChunk, { type: 'error' }>
    expect(error).toBeDefined()
    expect(String(error.data.error.message)).toContain('boom')
  })
})

describe('generateViaOpenAIResponses', () => {
  it('creates a non-streaming response and maps it', async () => {
    const create = jest.fn().mockResolvedValue({
      id: 'resp_1',
      model: 'gpt-5.6-terra',
      status: 'completed',
      output: [{ type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'Hi.' }] }],
      usage: {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 0 }
      }
    })
    const client = { responses: { create } } as any
    const result = await generateViaOpenAIResponses(client, { ...baseParams, thinking: true }, undefined)
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-5.6-terra',
        store: false,
        reasoning: { effort: 'medium', summary: 'auto' }
      })
    )
    expect(result.content).toBe('Hi.')
  })
})
