import OpenAI from 'openai'
import { z } from 'zod'
import type { Response, ResponseStreamEvent } from 'openai/resources/responses/responses'
import { OpenAICompatibleMapper } from '../../../../src/core/mapping/openai-compatible.mapper'
import { OpenAIMapper } from '../../../../src/core/mapping/openai.mapper'
import { generateViaOpenAIResponses, mapOpenAIResponsesChatStream } from '../../../../src/core/mapping/openai.responses.chat'
import { StructuredOutputValidationError, ToolArgumentValidationError } from '../../../../src/errors'
import type { CustomProviderConfig, GenerateParams, StreamChunk } from '../../../../src/types'
import type { ProviderUsageObserver } from '../../../../src/types/usage.types'

const tool = { type: 'function' as const, function: {
  name: 'validate', description: 'Validate', parameters: { type: 'object' as const, properties: { value: { type: 'number' as const } } },
  zodSchema: z.object({ value: z.number() })
} }
const config: CustomProviderConfig = { providerKey: 'fixture', mapper: OpenAICompatibleMapper,
  apiKey: 'fixture-key', baseURL: 'https://fixture.invalid/v1', supportedFeatures: ['generate', 'stream'] }
const params: GenerateParams = { provider: 'fixture', model: 'fixture-model', messages: [{ role: 'user', content: 'Test' }], tools: [tool] }
function observer() {
  return { beforeDispatch: jest.fn<Promise<void>, []>(async () => undefined), markDispatched: jest.fn(),
    observeUsage: jest.fn(), transportComplete: jest.fn(), observeRequestId: jest.fn() } satisfies ProviderUsageObserver
}
function completion(invalidTool = true): OpenAI.Chat.Completions.ChatCompletion {
  return { id: 'chat-fixture', object: 'chat.completion', created: 1, model: 'fixture-model',
    choices: [{ index: 0, finish_reason: invalidTool ? 'tool_calls' : 'stop', logprobs: null, message: {
      role: 'assistant', content: invalidTool ? null : '{"value":"invalid"}', refusal: null,
      ...(invalidTool ? { tool_calls: [{ id: 'call-1', type: 'function' as const, function: { name: 'validate', arguments: '{"value":"invalid"}' } }] } : {})
    } }], usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 } }
}
function install(mapper: OpenAICompatibleMapper, create: jest.Mock): void {
  Object.defineProperty(mapper, 'openaiClient', { value: { chat: { completions: { create } } } })
}
async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}
async function* chatStream(withUsage = true): AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> {
  yield { id: 'stream-fixture', object: 'chat.completion.chunk', created: 1, model: 'fixture-model', choices: [{ index: 0,
    delta: { role: 'assistant', tool_calls: [{ index: 0, id: 'call-1', type: 'function', function: { name: 'validate', arguments: '{"value":"invalid"}' } }] }, finish_reason: 'tool_calls' }] }
  if (withUsage) yield { id: 'stream-fixture', object: 'chat.completion.chunk', created: 1, model: 'fixture-model', choices: [], usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 } }
}
const invalidFunction = { type: 'function_call', id: 'item-1', call_id: 'call-1', name: 'validate', arguments: '{"value":"invalid"}' }
function response(output: unknown[] = []): Response {
  // Fixture deliberately contains only fields consumed by the mapper, as in
  // existing Responses tests; the transport is mocked rather than a live SDK.
  return { id: 'response-fixture', object: 'response', model: 'gpt-5.6-terra', status: 'completed', output,
    usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12, input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 2 } } } as unknown as Response
}
async function* responseEvents(events: unknown[]): AsyncIterable<ResponseStreamEvent> {
  for (const event of events) yield event as ResponseStreamEvent
}

describe('OpenAI inference lifecycle mapper boundaries', () => {
  test('custom nonstream observes raw usage before typed tool validation failure', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn().mockResolvedValue(completion())
    install(mapper, create)
    const seen = observer()
    await expect(mapper.executeGenerate({}, config.apiKey, config, params, seen)).rejects.toBeInstanceOf(ToolArgumentValidationError)
    expect(seen.beforeDispatch).toHaveBeenCalledTimes(1)
    expect(seen.markDispatched).toHaveBeenCalledTimes(1)
    expect(seen.transportComplete).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage).toHaveBeenCalledWith({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    expect(seen.observeRequestId).toHaveBeenCalledWith('chat-fixture')
    expect(seen.beforeDispatch.mock.invocationCallOrder[0]).toBeLessThan(create.mock.invocationCallOrder[0]!)
    expect(seen.markDispatched.mock.invocationCallOrder[0]).toBeLessThan(create.mock.invocationCallOrder[0]!)
  })
  test('custom JSON schema validation failure retains raw observed usage', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    install(mapper, jest.fn().mockResolvedValue(completion(false)))
    const seen = observer()
    await expect(mapper.executeGenerate({}, config.apiKey, config, { ...params, tools: undefined,
      responseFormat: { type: 'json_schema', json_schema: { name: 'value', schema: { type: 'object' }, zodSchema: z.object({ value: z.number() }) } } }, seen)).rejects.toBeInstanceOf(StructuredOutputValidationError)
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
  })
  test('admission rejection retains exact application error identity and sends nothing', async () => {
    const denied = new Error('Fixture admission denied')
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn()
    install(mapper, create)
    const seen = observer()
    seen.beforeDispatch.mockRejectedValue(denied)
    await expect(mapper.executeGenerate({}, config.apiKey, config, params, seen)).rejects.toBe(denied)
    await expect(collect(mapper.executeStream({}, config.apiKey, config, params, undefined, seen))).rejects.toBe(denied)
    expect(create).not.toHaveBeenCalled()
    expect(seen.markDispatched).not.toHaveBeenCalled()
    expect(seen.transportComplete).not.toHaveBeenCalled()
  })
  test('cancellation during async stream admission is verified not dispatched', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn()
    install(mapper, create)
    const abort = new AbortController()
    const seen = observer()
    seen.beforeDispatch.mockImplementation(async () => { abort.abort() })
    expect(await collect(mapper.executeStream({}, config.apiKey, config, params, abort.signal, seen))).toEqual([])
    expect(create).not.toHaveBeenCalled()
    expect(seen.markDispatched).not.toHaveBeenCalled()
    expect(seen.observeUsage).not.toHaveBeenCalled()
  })
  test('custom stream drains trailing usage before exposing the same typed tool error', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    install(mapper, jest.fn(async () => chatStream()))
    const seen = observer()
    const chunks = await collect(mapper.executeStream({}, config.apiKey, config, params, undefined, seen))
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage).toHaveBeenCalledWith({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    expect(seen.transportComplete).toHaveBeenCalledTimes(1)
    const failure = chunks.find(chunk => chunk.type === 'error')
    expect(failure?.type).toBe('error')
    if (failure?.type === 'error') expect(failure.data.error).toBeInstanceOf(ToolArgumentValidationError)
    expect(chunks.some(chunk => chunk.type === 'tool_call_done' || chunk.type === 'final_result')).toBe(false)
  })
  test('native OpenAI mapping observes before validation and drains trailing usage too', async () => {
    const mapper = new OpenAIMapper()
    const seen = observer()
    expect(() => mapper.mapFromProviderResponse(completion(), 'fixture-model', [tool], seen)).toThrow(ToolArgumentValidationError)
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    seen.observeUsage.mockClear()
    const chunks = await collect(mapper.mapProviderStream(chatStream() as unknown as Parameters<OpenAIMapper['mapProviderStream']>[0], params, seen))
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(chunks.at(-1)?.type).toBe('error')
    expect(mapper.supportsInferenceLifecycle).toBe(true)
  })
  test('a first finished candidate is not a complete multi-candidate request', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    async function* candidates() {
      yield { id: 'multi', object: 'chat.completion.chunk', created: 1, model: 'fixture-model',
        choices: [{ index: 0, delta: { content: 'First' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 } }
      yield { id: 'multi', object: 'chat.completion.chunk', created: 1, model: 'fixture-model',
        choices: [{ index: 1, delta: {}, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 } }
    }
    install(mapper, jest.fn(async () => candidates()))
    const seen = observer()
    await collect(mapper.executeStream({}, config.apiKey, config,
      { ...params, tools: undefined, extraParams: { n: 2 } }, undefined, seen))
    expect(seen.observeUsage.mock.calls.map(call => call[1])).toEqual([false, true])
  })
  test('an early sample stays non-final when no provider terminal usage arrives', async () => {
    const mapper = new OpenAICompatibleMapper(config)
    async function* earlyUsage() {
      yield { id: 'early', model: 'fixture-model', object: 'chat.completion.chunk', created: 1,
        choices: [{ index: 0, delta: { content: 'Partial' }, finish_reason: null }], usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 } }
      yield { id: 'early', model: 'fixture-model', object: 'chat.completion.chunk', created: 1,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }
    }
    install(mapper, jest.fn(async () => earlyUsage()))
    const seen = observer()
    await collect(mapper.executeStream({}, config.apiKey, config, { ...params, tools: undefined }, undefined, seen))
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage.mock.calls[0]?.[1]).toBe(false)
    expect(seen.transportComplete).toHaveBeenCalledTimes(1)
  })
})

describe('automatic OpenAI Responses lifecycle', () => {
  test('nonstream usage is observed before tool validation; admission failures stay unchanged', async () => {
    const create = jest.fn().mockResolvedValue(response([invalidFunction]))
    const client = { responses: { create } } as unknown as OpenAI
    const seen = observer()
    await expect(generateViaOpenAIResponses(client, { ...params, provider: 'openai', model: 'gpt-5.6-terra' }, [tool], seen)).rejects.toBeInstanceOf(ToolArgumentValidationError)
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage).toHaveBeenCalledWith({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    expect(seen.transportComplete).toHaveBeenCalledTimes(1)
    const denied = new Error('Denied')
    seen.beforeDispatch.mockRejectedValueOnce(denied)
    create.mockClear()
    await expect(generateViaOpenAIResponses(client, { ...params, provider: 'openai', model: 'gpt-5.6-terra' }, [tool], seen)).rejects.toBe(denied)
    expect(create).not.toHaveBeenCalled()
  })
  test('terminal usage follows invalid output-item arguments without a duplicate report', async () => {
    const seen = observer()
    const chunks = await collect(mapOpenAIResponsesChatStream(responseEvents([
      { type: 'response.output_item.added', item: invalidFunction },
      { type: 'response.output_item.done', item: invalidFunction },
      { type: 'response.completed', response: response([invalidFunction]) }
    ]), 'gpt-5.6-terra', [tool], seen))
    expect(seen.observeUsage).toHaveBeenCalledTimes(1)
    expect(seen.observeUsage).toHaveBeenCalledWith({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    expect(seen.transportComplete).toHaveBeenCalledTimes(1)
    const failure = chunks.find(chunk => chunk.type === 'error')
    if (failure?.type !== 'error') throw Error('Expected typed validation error')
    expect(failure.data.error).toBeInstanceOf(ToolArgumentValidationError)
    expect(chunks.some(chunk => chunk.type === 'tool_call_done' || chunk.type === 'final_result')).toBe(false)
  })
  test('response.failed usage is captured before exposing the provider error', async () => {
    const seen = observer()
    const failed = { ...response(), status: 'failed', error: { code: 'server_error', message: 'Fixture failure' } }
    const chunks = await collect(mapOpenAIResponsesChatStream(responseEvents([{ type: 'response.failed', response: failed }]), 'gpt-5.6-terra', undefined, seen))
    expect(seen.observeUsage).toHaveBeenCalledWith({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    expect(chunks.at(-1)?.type).toBe('error')
  })
})
