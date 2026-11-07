import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { z } from 'zod'
import { InferenceCall } from '../../../src/core/inference-lifecycle'
import { RosettaAI } from '../../../src/core/rosetta-ai'
import { OpenAICompatibleMapper } from '../../../src/core/mapping/openai-compatible.mapper'
import { BaseCustomMapper } from '../../../src/core/mapping/base.custom.mapper'
import { ToolArgumentValidationError, UnsupportedFeatureError } from '../../../src/errors'
import type { GenerateParams, InferenceLifecycle, InferenceOutcome, StreamChunk } from '../../../src/types'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
async function bounded<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([promise, new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error('Lifecycle fixture did not reach expected event')), 5000)
    })])
  } finally { if (timer) clearTimeout(timer) }
}
const tool = { type: 'function' as const, function: {
  name: 'validate', description: 'Validate', parameters: { type: 'object' as const, properties: { value: { type: 'number' as const } } },
  zodSchema: z.object({ value: z.number() })
} }
const params = (model = 'plain'): GenerateParams => ({ provider: 'fixture', model, messages: [{ role: 'user', content: 'Test' }] })

let server: Server, baseURL: string
let requests: Array<Record<string, unknown>> = []
let pendingNative: { accepted: () => void; closed: () => void } | undefined
beforeAll(async () => {
  server = createServer((req, res) => {
    let text = ''
    req.on('data', chunk => { text += chunk })
    req.on('end', () => {
      const body = JSON.parse(text) as Record<string, unknown>
      requests.push(body)
      const model = String(body.model)
      if (body.stream && model === 'native-pending') {
        // No headers: the native SDK create() promise is still pending and no
        // provider stream resource has been registered when cancellation starts.
        res.on('close', () => pendingNative?.closed())
        pendingNative?.accepted()
        return
      }
      const usage = { prompt_tokens: model === 'second' ? 20 : 5, completion_tokens: 7, total_tokens: model === 'second' ? 27 : 12,
        prompt_tokens_details: { cached_tokens: 2 } }
      if (body.stream) {
        res.setHeader('content-type', 'text/event-stream')
        const chunk = (choices: unknown[], extra: Record<string, unknown> = {}) => `data: ${JSON.stringify({ id: `stream-${model}`, object: 'chat.completion.chunk', created: 1, model, choices, ...extra })}\n\n`
        if (model === 'invalid-tool') {
          res.write(chunk([{ index: 0, delta: { role: 'assistant', tool_calls: [{ index: 0, id: 'call-1', type: 'function', function: { name: 'validate', arguments: '{"value":"invalid"}' } }] }, finish_reason: 'tool_calls' }]))
        } else {
          res.write(chunk([{ index: 0, delta: { role: 'assistant', content: 'Started' }, finish_reason: null }]))
        }
        if (model === 'hold') return // Real request remains active until SDK cancellation/return.
        res.write(chunk([{ index: 0, delta: {}, finish_reason: model === 'invalid-tool' ? 'tool_calls' : 'stop' }]))
        if (model !== 'no-usage') res.write(chunk([], { usage }))
        res.end('data: [DONE]\n\n')
        return
      }
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ id: `chat-${model}`, object: 'chat.completion', created: 1, model,
        choices: [{ index: 0, finish_reason: model === 'invalid-tool' ? 'tool_calls' : 'stop', message: {
          role: 'assistant', content: model === 'invalid-tool' ? null : 'Reply',
          ...(model === 'invalid-tool' ? { tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'validate', arguments: '{"value":"invalid"}' } }] } : {})
        } }], ...(model === 'no-usage' ? {} : { usage }) }))
    })
  })
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve) })
  baseURL = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`
})
beforeEach(() => { requests = []; pendingNative = undefined })
afterAll(async () => {
  server.closeAllConnections()
  await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
})
function sdk(inferenceLifecycle?: InferenceLifecycle): RosettaAI {
  return new RosettaAI({ inferenceLifecycle, customProviders: [{ providerKey: 'fixture', mapper: OpenAICompatibleMapper,
    apiKey: 'fixture-key', baseURL, supportedFeatures: ['generate', 'stream', 'tool_use'], defaultMaxRetries: 0 }] })
}
async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}
function recorder() {
  const outcomes: InferenceOutcome[] = []
  const finished = deferred<InferenceOutcome>()
  const finish = jest.fn(async (outcome: InferenceOutcome) => { outcomes.push(outcome); finished.resolve(outcome) })
  const admit = jest.fn<ReturnType<InferenceLifecycle['admit']>, Parameters<InferenceLifecycle['admit']>>(async () => ({ finish }))
  return { outcomes, finished, finish, admit, client: sdk({ admit }) }
}

test('settlement snapshots billing details independently of consumer mutations', async () => {
  const finish = jest.fn(async (_outcome: InferenceOutcome) => undefined)
  const call = new InferenceCall({ admit: async () => ({ finish }) }, { operation: 'stream', provider: 'google', model: 'gemini' })
  await call.beforeDispatch()
  call.markDispatched()
  const billing = { cachedInputTokens: 1, inputTokenDetails: [{ modality: 'TEXT', tokenCount: 5 }] }
  call.observeUsage({ promptTokens: 5, completionTokens: 2, billing }, true)
  billing.cachedInputTokens = 4
  billing.inputTokenDetails[0]!.tokenCount = 20
  await call.finish('completed')
  expect(finish).toHaveBeenCalledWith(expect.objectContaining({
    usageComplete: true,
    usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7,
      billing: { cachedInputTokens: 1, inputTokenDetails: [{ modality: 'TEXT', tokenCount: 5 }] } }
  }))
})

describe('SDK inference lifecycle integration', () => {
  test.each(['generate', 'stream'] as const)('unsupported custom mapper rejects metered %s before admission or transport', async operation => {
    class UnsupportedMapper extends BaseCustomMapper {
      override async executeGenerate(): Promise<never> { throw new Error('Unsupported mapper must never execute') }
      override async *executeStream(): AsyncIterable<StreamChunk> { throw new Error('Unsupported mapper must never execute') }
    }
    const generate = jest.spyOn(UnsupportedMapper.prototype, 'executeGenerate')
    const stream = jest.spyOn(UnsupportedMapper.prototype, 'executeStream')
    const admit = jest.fn(async () => ({ finish: async () => undefined }))
    const client = new RosettaAI({ inferenceLifecycle: { admit }, customProviders: [{ providerKey: 'fixture', mapper: UnsupportedMapper,
      apiKey: 'fixture-key', baseURL, supportedFeatures: ['generate', 'stream'] }] })
    if (operation === 'generate') await expect(client.generate(params())).rejects.toBeInstanceOf(UnsupportedFeatureError)
    else {
      const chunks = await collect(client.stream(params()))
      expect(chunks).toHaveLength(1)
      if (chunks[0]?.type !== 'error') throw new Error('Expected unsupported lifecycle error')
      expect(chunks[0].data.error).toBeInstanceOf(UnsupportedFeatureError)
    }
    expect(admit).not.toHaveBeenCalled()
    expect(generate).not.toHaveBeenCalled()
    expect(stream).not.toHaveBeenCalled()
    expect(requests).toEqual([])
  })

  test('omitting lifecycle retains ordinary generate and stream behavior', async () => {
    const client = sdk()
    expect((await client.generate(params())).content).toBe('Reply')
    const chunks = await collect(client.stream(params()))
    expect(chunks.some(chunk => chunk.type === 'final_result')).toBe(true)
    expect(chunks.some(chunk => chunk.type === 'error')).toBe(false)
    expect(requests).toHaveLength(2)
  })

  test.each(['generate', 'stream'] as const)('denied %s admission preserves identity and makes no request', async operation => {
    const denied = new Error('Quota denied')
    const admit = jest.fn(async () => { throw denied })
    const client = sdk({ admit })
    if (operation === 'generate') await expect(client.generate(params())).rejects.toBe(denied)
    else {
      const chunks = await collect(client.stream(params()))
      expect(chunks).toHaveLength(1)
      expect(chunks[0]?.type).toBe('error')
      if (chunks[0]?.type === 'error') expect(chunks[0].data.error).toBe(denied)
    }
    expect(admit).toHaveBeenCalledTimes(1)
    expect(requests).toEqual([])
  })

  test('awaits exactly one settlement before returning nonstream success', async () => {
    const entered = deferred<InferenceOutcome>(), release = deferred<void>()
    const finish = jest.fn(async (outcome: InferenceOutcome) => { entered.resolve(outcome); await release.promise })
    const admit = jest.fn(async () => ({ finish }))
    let exposed = false
    const result = sdk({ admit }).generate(params()).then(value => { exposed = true; return value })
    const outcome = await bounded(entered.promise)
    expect(exposed).toBe(false)
    expect(outcome).toMatchObject({ status: 'completed', dispatched: true, usageComplete: true, usage: { promptTokens: 5, completionTokens: 7, totalTokens: 12 } })
    release.resolve()
    expect((await result).content).toBe('Reply')
    expect(finish).toHaveBeenCalledTimes(1)
    expect(admit).toHaveBeenCalledWith({ operation: 'generate', provider: 'fixture', model: 'plain' })
  })

  test.each(['plain', 'invalid-tool'] as const)('awaits settlement before exposing %s terminal stream events', async model => {
    const entered = deferred<InferenceOutcome>(), release = deferred<void>()
    const finish = jest.fn(async (outcome: InferenceOutcome) => { entered.resolve(outcome); await release.promise })
    const client = sdk({ admit: async () => ({ finish }) })
    const observed: StreamChunk[] = []
    const consume = (async () => {
      for await (const chunk of client.stream({ ...params(model), ...(model === 'invalid-tool' ? { tools: [tool] } : {}) })) observed.push(chunk)
    })()
    try {
      expect(await bounded(entered.promise)).toMatchObject({ status: model === 'plain' ? 'completed' : 'failed', dispatched: true, usageComplete: true })
      expect(observed.some(chunk => chunk.type === 'final_result' || chunk.type === 'error' || chunk.type === 'message_stop')).toBe(false)
    } finally { release.resolve() }
    await bounded(consume)
    expect(observed.some(chunk => chunk.type === (model === 'plain' ? 'final_result' : 'error'))).toBe(true)
    expect(finish).toHaveBeenCalledTimes(1)
  })

  test.each(['generate', 'stream'] as const)('settles raw usage before %s terminal tool validation failure', async operation => {
    const recorded = recorder()
    const input = { ...params('invalid-tool'), tools: [tool] }
    if (operation === 'generate') {
      await expect(recorded.client.generate(input)).rejects.toBeInstanceOf(ToolArgumentValidationError)
    } else {
      const chunks = await collect(recorded.client.stream(input))
      const terminal = chunks.find(chunk => chunk.type === 'error')
      if (terminal?.type !== 'error') throw new Error('Expected stream validation error')
      expect(terminal.data.error).toBeInstanceOf(ToolArgumentValidationError)
      expect(chunks.some(chunk => chunk.type === 'final_result')).toBe(false)
    }
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(recorded.outcomes[0]).toMatchObject({ status: 'failed', dispatched: true, usageComplete: true, usage: { totalTokens: 12, billing: { cachedInputTokens: 2 } } })
  })

  test.each(['generate', 'stream'] as const)('%s missing provider usage remains unknown', async operation => {
    const recorded = recorder()
    if (operation === 'generate') await recorded.client.generate(params('no-usage'))
    else await collect(recorded.client.stream(params('no-usage')))
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(recorded.outcomes[0]).toMatchObject({ status: 'completed', dispatched: true, usage: null, usageComplete: false })
  })

  test('concurrent calls on one cached client keep admissions and usage isolated', async () => {
    const outcomes = new Map<string, InferenceOutcome>()
    const client = sdk({ admit: async info => ({ finish: async outcome => { outcomes.set(info.model, outcome) } }) })
    await Promise.all([client.generate(params('first')), client.generate(params('second'))])
    expect(outcomes.size).toBe(2)
    expect(outcomes.get('first')?.usage?.totalTokens).toBe(12)
    expect(outcomes.get('second')?.usage?.totalTokens).toBe(27)
    expect(requests).toHaveLength(2)
  })

  test.each(['cancel', 'return', 'throw'] as const)('unused stream %s never admits or dispatches', async action => {
    const recorded = recorder()
    const stream = recorded.client.stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    if (action === 'cancel') { stream.cancel(); await iterator.next() }
    else if (action === 'return') await iterator.return?.()
    else await expect(iterator.throw?.(new Error('Consumer stopped'))).rejects.toThrow('Consumer stopped')
    expect(recorded.admit).not.toHaveBeenCalled()
    expect(recorded.finish).not.toHaveBeenCalled()
    expect(requests).toEqual([])
  })

  test('cancellation during admission settles verified no-dispatch after admission resolves', async () => {
    const entered = deferred<void>(), release = deferred<void>()
    const recorded = recorder()
    const client = sdk({ admit: async () => { entered.resolve(); await release.promise; return { finish: recorded.finish } } })
    const stream = client.stream(params('hold'))
    const next = stream[Symbol.asyncIterator]().next()
    await bounded(entered.promise)
    stream.cancel()
    release.resolve()
    await bounded(next)
    expect(await bounded(recorded.finished.promise)).toMatchObject({ status: 'cancelled', dispatched: false, usageComplete: true })
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(requests).toEqual([])
  })

  test.each(['cancel', 'return', 'throw'] as const)('active paused stream %s finalizes exactly once', async action => {
    const recorded = recorder()
    const stream = recorded.client.stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    expect((await bounded(iterator.next())).done).toBe(false)
    if (action === 'cancel') stream.cancel()
    else if (action === 'return') await bounded(iterator.return!())
    else await expect(bounded(iterator.throw!(new Error('Consumer stopped')))).rejects.toThrow('Consumer stopped')
    expect(await bounded(recorded.finished.promise)).toMatchObject({ status: 'cancelled', dispatched: true, usage: null, usageComplete: false })
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('native cancellation while transport create awaits headers settles cancelled once', async () => {
    const accepted = deferred<void>(), closed = deferred<void>()
    pendingNative = { accepted: () => accepted.resolve(), closed: () => closed.resolve() }
    const recorded = recorder()
    const client = new RosettaAI({ openaiApiKey: 'fixture-key', defaultMaxRetries: 0,
      providerOptions: { openai: { baseURL } }, inferenceLifecycle: { admit: recorded.admit } })
    const stream = client.stream({ ...params('native-pending'), provider: 'openai' })
    const next = stream[Symbol.asyncIterator]().next()
    await bounded(accepted.promise)
    expect(recorded.finish).not.toHaveBeenCalled()
    stream.cancel()
    expect(stream.signal.aborted).toBe(true)
    expect((await bounded(next)).done).toBe(true)
    expect(await bounded(recorded.finished.promise)).toMatchObject({ status: 'cancelled', dispatched: true, usage: null, usageComplete: false })
    await bounded(closed.promise)
    // Let queued iterator cleanup finish too; Jest treats an unhandled cleanup
    // rejection as a test failure rather than suppressing it in this fixture.
    await new Promise<void>(resolve => setImmediate(resolve))
    expect(recorded.admit).toHaveBeenCalledTimes(1)
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('concurrent next calls use one iterator, admission and settlement', async () => {
    const recorded = recorder()
    const stream = recorded.client.stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    const results = await bounded(Promise.all([iterator.next(), iterator.next()]))
    expect(results.every(result => !result.done)).toBe(true)
    expect(stream[Symbol.asyncIterator]()).toBe(iterator)
    stream.cancel()
    await bounded(recorded.finished.promise)
    expect(recorded.admit).toHaveBeenCalledTimes(1)
    expect(recorded.finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('active iterator return exposes the original settlement error exactly once', async () => {
    const unavailable = new Error('Accounting unavailable')
    const finish = jest.fn(async () => { throw unavailable })
    const stream = sdk({ admit: async () => ({ finish }) }).stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    await bounded(iterator.next())
    await expect(bounded(iterator.return!())).rejects.toBe(unavailable)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('return after cancellation cleanup has failed still reports the original settlement error', async () => {
    const unavailable = new Error('Cancelled accounting unavailable'), entered = deferred<void>()
    const finish = jest.fn(async () => { entered.resolve(); throw unavailable })
    const stream = sdk({ admit: async () => ({ finish }) }).stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    await bounded(iterator.next())
    stream.cancel()
    await bounded(entered.promise)
    // Background cancellation must handle its rejection, but an explicit later
    // close must still observe that failure rather than an already-closed success.
    await new Promise<void>(resolve => setImmediate(resolve))
    await expect(bounded(iterator.return!())).rejects.toBe(unavailable)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('explicit returns join pending cancellation settlement and cannot complete early', async () => {
    const entered = deferred<void>(), release = deferred<void>()
    const finish = jest.fn(async (_outcome: InferenceOutcome) => { entered.resolve(); await release.promise })
    const stream = sdk({ admit: async () => ({ finish }) }).stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    await bounded(iterator.next())
    stream.cancel()
    await bounded(entered.promise)
    let closed = 0
    const first = iterator.return!().then(result => { closed++; return result })
    const second = iterator.return!().then(result => { closed++; return result })
    try {
      await new Promise<void>(resolve => setImmediate(resolve))
      expect(closed).toBe(0)
      expect(finish).toHaveBeenCalledTimes(1)
    } finally { release.resolve() }
    expect((await bounded(Promise.all([first, second]))).every(result => result.done)).toBe(true)
    expect(closed).toBe(2)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ status: 'cancelled', dispatched: true, usageComplete: false })
    expect(requests).toHaveLength(1)
  })

  test.each(['return', 'throw', 'next'] as const)('cancel then %s joins the pending finisher and preserves its failure identity', async action => {
    const unavailable = new Error('Cancelled accounting unavailable')
    const entered = deferred<void>(), release = deferred<void>()
    const finish = jest.fn(async () => { entered.resolve(); await release.promise; throw unavailable })
    const stream = sdk({ admit: async () => ({ finish }) }).stream(params('hold'))
    const iterator = stream[Symbol.asyncIterator]()
    await bounded(iterator.next())
    stream.cancel()
    await bounded(entered.promise)
    const close = action === 'return' ? iterator.return!() : action === 'throw' ? iterator.throw!(new Error('Consumer stopped')) : iterator.next()
    // Attach the rejection observer before releasing the finisher. The test
    // must not manufacture an unhandled promise rejection itself.
    const rejected = expect(bounded(close)).rejects.toBe(unavailable)
    release.resolve()
    await rejected
    await new Promise<void>(resolve => setImmediate(resolve))
    expect(finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })

  test('native pending-create cancellation shares its settlement failure with pending next and explicit return', async () => {
    const accepted = deferred<void>(), closed = deferred<void>()
    pendingNative = { accepted: () => accepted.resolve(), closed: () => closed.resolve() }
    const unavailable = new Error('Native cancellation accounting unavailable')
    const finish = jest.fn(async (_outcome: InferenceOutcome) => { throw unavailable })
    const admit = jest.fn(async () => ({ finish }))
    const client = new RosettaAI({ openaiApiKey: 'fixture-key', defaultMaxRetries: 0,
      providerOptions: { openai: { baseURL } }, inferenceLifecycle: { admit } })
    const stream = client.stream({ ...params('native-pending'), provider: 'openai' })
    const iterator = stream[Symbol.asyncIterator]()
    // Observe the request's original pending next immediately; cancellation can
    // reject it before the explicit return operation below is awaited.
    const next = iterator.next().then(result => ({ result }), error => ({ error }))
    await bounded(accepted.promise)
    stream.cancel()
    await expect(bounded(iterator.return!())).rejects.toBe(unavailable)
    const terminal = await bounded(next)
    expect('error' in terminal && terminal.error).toBe(unavailable)
    await bounded(closed.promise)
    await new Promise<void>(resolve => setImmediate(resolve))
    expect(admit).toHaveBeenCalledTimes(1)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ status: 'cancelled', dispatched: true, usageComplete: false })
    expect(requests).toHaveLength(1)
  })

  test.each(['generate', 'stream'] as const)('%s settlement callback error preserves identity and is not retried', async operation => {
    const denied = new Error('Accounting unavailable')
    const finish = jest.fn(async () => { throw denied })
    const client = sdk({ admit: async () => ({ finish }) })
    const result = operation === 'generate' ? client.generate(params()) : collect(client.stream(params()))
    await expect(result).rejects.toBe(denied)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(requests).toHaveLength(1)
  })
})
