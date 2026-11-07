import { GoogleMapper } from '../../../../src/core/mapping/google.mapper'
import { GenerateContentResponse, FinishReason } from '@google/genai'
import { mapTokenUsage } from '../../../../src/core/mapping/common.utils'
import { AnthropicMapper } from '../../../../src/core/mapping/anthropic.mapper'
import type { ProviderUsageObserver } from '../../../../src/types/usage.types'
import type { GenerateParams } from '../../../../src/types'
import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages'

describe('native provider inclusive usage', () => {
  test('includes Anthropic cache input and Google thoughts/tool input exactly once', () => {
    expect(mapTokenUsage({ input_tokens: 10, cache_creation_input_tokens: 4, cache_read_input_tokens: 10, output_tokens: 6 }, 'anthropic')).toEqual({ promptTokens: 24, completionTokens: 6, totalTokens: 30 })
    expect(mapTokenUsage({ promptTokenCount: 10, toolUsePromptTokenCount: 3, candidatesTokenCount: 4, thoughtsTokenCount: 5, totalTokenCount: 22 }, 'google')).toEqual({ promptTokens: 13, completionTokens: 9, totalTokens: 22 })
    expect(mapTokenUsage({ prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, prompt_tokens_details: { cached_tokens: 8 }, completion_tokens_details: { reasoning_tokens: 3 } }, 'openai')).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 })
  })
  test('distinguishes optional Google components from unknown required counts', () => {
    expect(mapTokenUsage({ promptTokenCount: 2, candidatesTokenCount: 3 }, 'google')).toEqual({ promptTokens: 2, completionTokens: 3, totalTokens: 5 })
    expect(mapTokenUsage({ input_tokens: 2, output_tokens: 3, cache_creation_input_tokens: null, cache_read_input_tokens: 0 }, 'anthropic')).toMatchObject({ promptTokens: undefined, completionTokens: 3, totalTokens: undefined, incomplete: true })
    expect(mapTokenUsage({ totalTokenCount: 8 }, 'google')).toMatchObject({ promptTokens: undefined, completionTokens: undefined, totalTokens: 8 })
  })
  test.each([-1, NaN, Infinity, 1.5])('marks invalid reported counts %s incomplete', count => {
    expect(mapTokenUsage({ prompt_tokens: count, completion_tokens: 2 }, 'openai')).toMatchObject({ promptTokens: undefined, completionTokens: 2, incomplete: true })
  })
  test('Anthropic cumulative delta replaces fields and preserves known null components', async () => {
    const observed: unknown[] = []
    const observer: ProviderUsageObserver = { beforeDispatch: async () => {}, markDispatched() {}, observeRequestId() {}, transportComplete() {}, observeUsage(usage, final) { observed.push([usage, final]) } }
    const events = [
      { type: 'message_start', message: { id: 'm', model: 'claude', usage: { input_tokens: 10, cache_creation_input_tokens: 4, cache_read_input_tokens: 10, output_tokens: 1 } } },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 12, cache_creation_input_tokens: null, cache_read_input_tokens: null, output_tokens: 6 } },
      { type: 'message_stop' }
    ] as unknown as RawMessageStreamEvent[]
    async function* stream() { yield* events }
    for await (const _chunk of new AnthropicMapper().mapProviderStream(stream(), {} as GenerateParams, observer)) { /* consume */ }
    expect(observed).toEqual([[{ promptTokens: 24, completionTokens: 1, totalTokens: 25 }, false], [{ promptTokens: 26, completionTokens: 6, totalTokens: 32 }, true]])
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
    expect(observeUsage).toHaveBeenLastCalledWith({ promptTokens: 10, completionTokens: 6, totalTokens: 16 }, true)
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
      yield { candidates: [{ index: 0, finishReason: FinishReason.STOP }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12 } } as GenerateContentResponse
      yield { candidates: [{ index: 1, finishReason: FinishReason.STOP }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 6, totalTokenCount: 16 } } as GenerateContentResponse
    }
    const params: GenerateParams = { model: 'gemini', messages: [{ role: 'user', content: 'test' }], extraParams: { candidateCount: 2 } }
    expect(new GoogleMapper().mapToProviderParams(params).config.candidateCount).toBe(2)
    for await (const _chunk of new GoogleMapper().mapProviderStream(stream(), params, observer)) { /* consume */ }
    expect(observeUsage.mock.calls).toEqual([
      [{ promptTokens: 10, completionTokens: 2, totalTokens: 12 }, false],
      [{ promptTokens: 10, completionTokens: 6, totalTokens: 16 }, true]
    ])
  })
})
