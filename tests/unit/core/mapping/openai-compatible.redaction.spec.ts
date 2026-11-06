import { createServer, Server } from 'node:http'
import { AddressInfo } from 'node:net'
import { inspect } from 'node:util'
import OpenAI from 'openai'
import { OpenAICompatibleMapper } from '../../../../src/core/mapping/openai-compatible.mapper'
import { RosettaAI } from '../../../../src/core/rosetta-ai'
import { ProviderAPIError, ToolArgumentValidationError } from '../../../../src/errors'
import { CustomProviderConfig, GenerateParams, StreamChunk } from '../../../../src/types'

const apiKey = 'SENTINEL_CUSTOM_API_KEY'
const params: GenerateParams = {
  provider: 'fixture',
  model: 'fixture-model',
  messages: [{ role: 'user', content: 'Fixture' }]
}
function config(baseURL: string): CustomProviderConfig {
  return {
    providerKey: 'fixture',
    mapper: OpenAICompatibleMapper,
    apiKey,
    baseURL,
    supportedFeatures: ['generate', 'stream'],
    defaultMaxRetries: 0
  }
}
async function close(server: Server): Promise<void> {
  server.closeAllConnections()
  await new Promise<void>((resolve, reject) => server.close(error => (error ? reject(error) : resolve())))
}
function assertSafe(error: unknown, baseURL: string, status: number): void {
  expect(error).toBeInstanceOf(ProviderAPIError)
  expect(error).toMatchObject({
    statusCode: status,
    errorCode: 'fixture_code',
    errorType: 'fixture_type',
    customProvider: 'fixture'
  })
  expect((error as ProviderAPIError).underlyingError).toBeUndefined()
  expect((error as Error).message).toContain('[REDACTED]')
  const printed = inspect(error, { depth: 10 })
  expect(printed).not.toContain(apiKey)
  expect(printed).not.toContain(baseURL)
  expect(JSON.stringify(error)).not.toContain(apiKey)
  expect(JSON.stringify(error)).not.toContain(baseURL)
}

describe('OpenAI-compatible inference error redaction', () => {
  it.each([400, 401])(
    'redacts HTTP%s provider echoes before throwing or yielding and retains metadata',
    async status => {
      let baseURL = ''
      const server = createServer((req, res) => {
        req.resume()
        req.on('end', () => {
          res.statusCode = status
          res.setHeader('content-type', 'application/json')
          res.end(
            JSON.stringify({
              error: {
                message: `Rejected ${apiKey} at ${baseURL}/chat/completions`,
                code: 'fixture_code',
                type: 'fixture_type'
              }
            })
          )
        })
      })
      await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
      baseURL = `http://127.0.0.1:${(server.address() as AddressInfo).port}/private-fixture/v1`
      try {
        const cfg = config(baseURL)
        const mapper = new OpenAICompatibleMapper(cfg)
        try {
          await mapper.executeGenerate({}, apiKey, cfg, params)
          throw new Error('Expected provider failure')
        } catch (error) {
          assertSafe(error, baseURL, status)
        }
        const chunks: StreamChunk[] = []
        for await (const chunk of mapper.executeStream({}, apiKey, cfg, params)) chunks.push(chunk)
        const failure = chunks.find(chunk => chunk.type === 'error')
        expect(failure?.type).toBe('error')
        if (failure?.type === 'error') assertSafe(failure.data.error, baseURL, status)
      } finally {
        await close(server)
      }
    }
  )

  it('sanitizes error chunks emitted by mapOpenAIStream after the stream has started', async () => {
    const baseURL = 'https://private-fixture.invalid/v1'
    const cfg = config(baseURL)
    const mapper = new OpenAICompatibleMapper(cfg)
    const upstream = OpenAI.APIError.generate(
      400,
      { error: { message: `Rejected ${apiKey} ${baseURL}`, code: 'fixture_code', type: 'fixture_type' } },
      'Fixture error',
      new Headers()
    )
    async function* chunks() {
      yield {
        id: 'stream-fixture',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'fixture-model',
        choices: [{ index: 0, delta: { role: 'assistant', content: 'Started' }, finish_reason: null }]
      }
      throw upstream
    }
    Object.defineProperty(mapper, 'openaiClient', {
      value: { chat: { completions: { create: jest.fn(async () => chunks()) } } }
    })
    const seen: StreamChunk[] = []
    for await (const chunk of mapper.executeStream({}, apiKey, cfg, params)) seen.push(chunk)
    expect(seen.some(chunk => chunk.type === 'content_delta')).toBe(true)
    const failure = seen.find(chunk => chunk.type === 'error')
    expect(failure?.type).toBe('error')
    if (failure?.type === 'error') assertSafe(failure.data.error, baseURL, 400)
  })

  it('drops malformed non-string provider metadata instead of retaining secret-bearing objects', async () => {
    const baseURL = 'https://private-fixture.invalid/v1'
    const cfg = config(baseURL)
    const mapper = new OpenAICompatibleMapper(cfg)
    const upstream = OpenAI.APIError.generate(400, {
      error: { message: 'Rejected request', code: { secret: apiKey }, type: { endpoint: baseURL } }
    }, 'Fixture error', new Headers())
    Object.defineProperty(mapper, 'openaiClient', { value: { chat: { completions: { create: jest.fn().mockRejectedValue(upstream) } } } })
    const check = (error: unknown): void => {
      expect(error).toBeInstanceOf(ProviderAPIError)
      expect(error).toMatchObject({ statusCode: 400, customProvider: 'fixture' })
      expect((error as ProviderAPIError).errorCode).toBeUndefined()
      expect((error as ProviderAPIError).errorType).toBeUndefined()
      expect((error as ProviderAPIError).underlyingError).toBeUndefined()
      expect(inspect(error, { depth: 10 })).not.toContain(apiKey)
      expect(inspect(error, { depth: 10 })).not.toContain(baseURL)
      expect(JSON.stringify(error)).not.toContain(apiKey)
      expect(JSON.stringify(error)).not.toContain(baseURL)
    }
    try { await mapper.executeGenerate({}, apiKey, cfg, params); throw new Error('Expected provider failure') }
    catch (error) { check(error) }
    const chunks: StreamChunk[] = []
    for await (const chunk of mapper.executeStream({}, apiKey, cfg, params)) chunks.push(chunk)
    const failure = chunks.find(chunk => chunk.type === 'error')
    expect(failure?.type).toBe('error')
    if (failure?.type === 'error') check(failure.data.error)
  })

  it('preserves typed tool validation failures and pre-aborted stream behavior', async () => {
    const cfg = config('https://private-fixture.invalid/v1')
    const mapper = new OpenAICompatibleMapper(cfg)
    const validation = new ToolArgumentValidationError('fixture', [], 'tool', 'call', {})
    const create = jest.fn().mockRejectedValue(validation)
    Object.defineProperty(mapper, 'openaiClient', { value: { chat: { completions: { create } } } })
    await expect(mapper.executeGenerate({}, apiKey, cfg, params)).rejects.toBe(validation)
    const aborted = new AbortController()
    aborted.abort()
    const seen: StreamChunk[] = []
    for await (const chunk of mapper.executeStream({}, apiKey, cfg, params, aborted.signal)) seen.push(chunk)
    expect(seen).toEqual([])
    expect(create).toHaveBeenCalledTimes(1)
  })

  it('does not log a configured private URL when initializing a custom provider', () => {
    const spy = jest.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      new RosettaAI({ customProviders: [config('https://private-fixture.invalid/v1')] })
      expect(inspect(spy.mock.calls)).not.toContain('private-fixture.invalid')
      expect(inspect(spy.mock.calls)).not.toContain(apiKey)
      expect(inspect(spy.mock.calls)).toContain('Initialized custom provider: fixture')
    } finally {
      spy.mockRestore()
    }
  })
})
