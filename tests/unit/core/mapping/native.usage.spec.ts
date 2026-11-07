import { GoogleMapper } from '../../../../src/core/mapping/google.mapper'
import { GenerateContentResponse, FinishReason, HttpResponse } from '@google/genai'
import { mapTokenUsage } from '../../../../src/core/mapping/common.utils'
import { AnthropicMapper } from '../../../../src/core/mapping/anthropic.mapper'
import type { ProviderUsageObserver } from '../../../../src/types/usage.types'
import type { GenerateParams } from '../../../../src/types'
import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages'

describe('native provider inclusive usage', () => {
  test('includes Anthropic cache input and Google thoughts/tool input exactly once', () => {
    expect(mapTokenUsage({ input_tokens: 10, cache_creation_input_tokens: 4, cache_read_input_tokens: 10, output_tokens: 6 }, 'anthropic')).toEqual({ promptTokens: 24, completionTokens: 6, totalTokens: 30, billing: { cachedInputTokens: 10, cacheWriteInputTokens: 4 } })
    expect(mapTokenUsage({ promptTokenCount: 10, toolUsePromptTokenCount: 3, candidatesTokenCount: 4, thoughtsTokenCount: 5, totalTokenCount: 22 }, 'google')).toEqual({ promptTokens: 13, completionTokens: 9, totalTokens: 22, billing: {} })
    expect(mapTokenUsage({ prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, prompt_tokens_details: { cached_tokens: 8 }, completion_tokens_details: { reasoning_tokens: 3 } }, 'openai')).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15, billing: { cachedInputTokens: 8, cacheWriteInputTokens: 0, cacheWrite30mInputTokens: 0 } })
  })
  test('distinguishes optional Google components from unknown required counts', () => {
    expect(mapTokenUsage({ promptTokenCount: 2, candidatesTokenCount: 3 }, 'google')).toEqual({ promptTokens: 2, completionTokens: 3, totalTokens: 5, billing: {} })
    expect(mapTokenUsage({ input_tokens: 2, output_tokens: 3, cache_creation_input_tokens: null, cache_read_input_tokens: 0 }, 'anthropic')).toMatchObject({ promptTokens: undefined, completionTokens: 3, totalTokens: undefined, incomplete: true })
    expect(mapTokenUsage({ totalTokenCount: 8 }, 'google')).toMatchObject({ promptTokens: undefined, completionTokens: undefined, totalTokens: 8 })
  })
  test('preserves allowlisted Gemini modality counts without retaining provider objects', () => {
    expect(mapTokenUsage({
      promptTokenCount: 10, cachedContentTokenCount: 4, toolUsePromptTokenCount: 2,
      candidatesTokenCount: 3, thoughtsTokenCount: 2, totalTokenCount: 17, serviceTier: 'priority',
      promptTokensDetails: [{ modality: 'TEXT', tokenCount: 6, extra: 'ignored' }, { modality: 'AUDIO', tokenCount: 4 }],
      cacheTokensDetails: [{ modality: 'TEXT', tokenCount: 4 }],
      toolUsePromptTokensDetails: [{ modality: 'TEXT', tokenCount: 2 }],
      candidatesTokensDetails: [{ modality: 'IMAGE', tokenCount: 3 }]
    }, 'google')).toEqual({
      promptTokens: 12, completionTokens: 5, totalTokens: 17, cachedContentTokenCount: 4,
      billing: {
        serviceTier: 'priority',
        cachedInputTokens: 4,
        inputTokenDetails: [{ modality: 'TEXT', tokenCount: 6 }, { modality: 'AUDIO', tokenCount: 4 }],
        cachedInputTokenDetails: [{ modality: 'TEXT', tokenCount: 4 }],
        toolUseInputTokenDetails: [{ modality: 'TEXT', tokenCount: 2 }],
        outputTokenDetails: [{ modality: 'IMAGE', tokenCount: 3 }]
      }
    })
  })
  test('defaults omitted Chat cache writes to zero and preserves missing Responses cache writes', () => {
    expect(mapTokenUsage({ prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }, 'openai'))
      .toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15,
        billing: { cacheWriteInputTokens: 0, cacheWrite30mInputTokens: 0 } })
    expect(mapTokenUsage({ input_tokens: 10, output_tokens: 5, total_tokens: 15, input_tokens_details: { cached_tokens: 0 } }, 'openai-responses'))
      .toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15, billing: { cachedInputTokens: 0 } })
  })
  test('marks missing required Responses cache reads as incomplete', () => {
    expect(mapTokenUsage({ input_tokens: 10, output_tokens: 5, total_tokens: 15 }, 'openai-responses'))
      .toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15, billing: { incomplete: true } })
  })
  test('marks explicitly null billing counts as malformed', () => {
    expect(mapTokenUsage({
      prompt_tokens: 10, completion_tokens: 5, total_tokens: 15,
      prompt_tokens_details: { cached_tokens: null, cache_write_tokens: null, audio_tokens: null }
    }, 'openai')).toEqual({
      promptTokens: 10, completionTokens: 5, totalTokens: 15,
      billing: { incomplete: true }
    })
  })
  test('marks invalid billing data separately from inclusive token usage', () => {
    expect(mapTokenUsage({
      input_tokens: 10, output_tokens: 5, total_tokens: 15,
      input_tokens_details: { cached_tokens: 5, cache_write_tokens: 2, cache_write_tokens_details: { ttl_30m_tokens: 1, ttl_12h_tokens: 2 } }
    }, 'openai-responses')).toEqual({
      promptTokens: 10, completionTokens: 5, totalTokens: 15,
      billing: { cachedInputTokens: 5, cacheWriteInputTokens: 2, cacheWrite30mInputTokens: 1, cacheWrite12hInputTokens: 2, incomplete: true }
    })
  })
  test.each([-1, NaN, Infinity, 1.5])('marks invalid reported counts %s incomplete', count => {
    expect(mapTokenUsage({ prompt_tokens: count, completion_tokens: 2 }, 'openai')).toMatchObject({ promptTokens: undefined, completionTokens: 2, incomplete: true })
  })
  test('Anthropic cumulative delta replaces fields and preserves known null components', async () => {
    const observed: unknown[] = []
    const observer: ProviderUsageObserver = { beforeDispatch: async () => {}, markDispatched() {}, observeRequestId() {}, transportComplete() {}, observeUsage(usage, final) { observed.push([usage, final]) } }
    const events = [
      { type: 'message_start', message: { id: 'm', model: 'claude', usage: { input_tokens: 10, cache_creation_input_tokens: 4, cache_read_input_tokens: 10, output_tokens: 1, cache_creation: { ephemeral_5m_input_tokens: 3, ephemeral_1h_input_tokens: 1 }, speed: 'fast', service_tier: 'standard', inference_geo: 'us' } } },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 12, cache_creation_input_tokens: null, cache_read_input_tokens: null, output_tokens: 6 } },
      { type: 'message_stop' }
    ] as unknown as RawMessageStreamEvent[]
    async function* stream() { yield* events }
    for await (const _chunk of new AnthropicMapper().mapProviderStream(stream(), {} as GenerateParams, observer)) { /* consume */ }
    const billing = { cachedInputTokens: 10, cacheWriteInputTokens: 4, cacheWrite5mInputTokens: 3, cacheWrite1hInputTokens: 1,
      speed: 'fast', serviceTier: 'standard', inferenceGeo: 'us' }
    expect(observed).toEqual([
      [{ promptTokens: 24, completionTokens: 1, totalTokens: 25, billing }, false],
      [{ promptTokens: 26, completionTokens: 6, totalTokens: 32, billing }, true]
    ])
  })
  test('drains trailing Anthropic usage after malformed tool JSON without emitting tool completion', async () => {
    const observeUsage = jest.fn()
    const observer: ProviderUsageObserver = { beforeDispatch: async () => {}, markDispatched() {}, observeRequestId() {}, transportComplete() {}, observeUsage }
    const events = [
      { type: 'message_start', message: { id: 'm', model: 'claude', usage: { input_tokens: 10, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 0 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tool', name: 'test', input: {} } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{bad' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 6 } },
      { type: 'message_stop' }
    ] as unknown as RawMessageStreamEvent[]
    async function* stream() { yield* events }
    const chunks = []
    for await (const chunk of new AnthropicMapper().mapProviderStream(stream(), {} as GenerateParams, observer)) chunks.push(chunk)
    expect(observeUsage).toHaveBeenLastCalledWith({ promptTokens: 10, completionTokens: 6, totalTokens: 16, billing: { cachedInputTokens: 0, cacheWriteInputTokens: 0 } }, true)
    expect(chunks.some(chunk => chunk.type === 'tool_call_done' || chunk.type === 'final_result')).toBe(false)
    expect(chunks[chunks.length - 1].type).toBe('error')
    let consumedWithoutObserver = 0
    async function* legacyStream() {
      for (const event of events) { consumedWithoutObserver++; yield event }
    }
    for await (const _chunk of new AnthropicMapper().mapProviderStream(legacyStream(), {} as GenerateParams)) { /* consume */ }
    expect(consumedWithoutObserver).toBe(4)
  })

  test('Google final usage waits for every configured candidate', async () => {
    const observeUsage = jest.fn()
    const observer: ProviderUsageObserver = { beforeDispatch: async () => {}, markDispatched() {}, observeRequestId() {}, transportComplete() {}, observeUsage }
    async function* stream(): AsyncIterable<GenerateContentResponse> {
      yield { candidates: [{ index: 0, finishReason: FinishReason.STOP }],
        sdkHttpResponse: new HttpResponse(new Response(null, { headers: { 'x-gemini-service-tier': 'flex' } })), usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12 } } as GenerateContentResponse
      yield { candidates: [{ index: 1, finishReason: FinishReason.STOP }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 6, totalTokenCount: 16,
        cachedContentTokenCount: 2, promptTokensDetails: [{ modality: 'AUDIO', tokenCount: 10 }] } } as GenerateContentResponse
    }
    const params: GenerateParams = { model: 'gemini', messages: [{ role: 'user', content: 'test' }], extraParams: { candidateCount: 2 } }
    expect(new GoogleMapper().mapToProviderParams(params).config.candidateCount).toBe(2)
    for await (const _chunk of new GoogleMapper().mapProviderStream(stream(), params, observer)) { /* consume */ }
    expect(observeUsage.mock.calls).toEqual([
      [{ promptTokens: 10, completionTokens: 2, totalTokens: 12, billing: { serviceTier: 'flex' } }, false],
      [{ promptTokens: 10, completionTokens: 6, totalTokens: 16, cachedContentTokenCount: 2,
        billing: { serviceTier: 'flex', cachedInputTokens: 2, inputTokenDetails: [{ modality: 'AUDIO', tokenCount: 10 }] } }, true]
    ])
  })
})
