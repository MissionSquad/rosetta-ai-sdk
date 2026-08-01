import {
  normalizeGenerateResultThinking,
  normalizeResponsesThinkingStream,
  normalizeThinkingStream
} from '../../../../src/core/streaming/thinking-normalizer'
import { GenerateResult, ResponsesStreamChunk, StreamChunk } from '../../../../src/types'

async function* streamChunks<T>(chunks: readonly T[]): AsyncIterable<T> {
  for (const chunk of chunks) yield chunk
}

async function collect<T>(source: AsyncIterable<T>): Promise<T[]> {
  const chunks: T[] = []
  for await (const chunk of source) chunks.push(chunk)
  return chunks
}

function result(overrides: Partial<GenerateResult> = {}): GenerateResult {
  return {
    content: null,
    finishReason: 'stop',
    model: 'test-model',
    ...overrides
  }
}

function finalResultChunk(value: GenerateResult): StreamChunk {
  return { type: 'final_result', data: { result: value } }
}

function responseResult() {
  return {
    id: 'resp-1',
    output: [{ type: 'output_text' as const, text: 'answer' }],
    output_text: 'answer',
    model: 'test-model',
    finish_reason: 'stop'
  }
}

describe('thinking stream normalization', () => {
  describe('canonical lifecycle repair', () => {
    it('preserves one well-formed start/delta/stop cycle', async () => {
      const input: StreamChunk[] = [
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'checking' } },
        { type: 'thinking_stop' }
      ]

      await expect(collect(normalizeThinkingStream(streamChunks(input)))).resolves.toEqual(input)
    })

    it('repairs deltas without starts, repeated starts, early/repeated stops, and empty deltas', async () => {
      const input: StreamChunk[] = [
        { type: 'thinking_stop' },
        { type: 'thinking_delta', data: { delta: '' } },
        { type: 'thinking_delta', data: { delta: 'first' } },
        { type: 'thinking_start' },
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'replacement' } },
        { type: 'thinking_stop' },
        { type: 'thinking_stop' }
      ]

      const output = await collect(normalizeThinkingStream(streamChunks(input)))

      expect(output).toEqual([
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'first' } },
        { type: 'thinking_stop' },
        { type: 'thinking_start' },
        { type: 'thinking_stop' },
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'replacement' } },
        { type: 'thinking_stop' }
      ])
      expect(
        output.some((chunk, index) => chunk.type === 'thinking_start' && output[index - 1]?.type === 'thinking_start')
      ).toBe(false)
      expect(
        output.some((chunk, index) => chunk.type === 'thinking_stop' && output[index - 1]?.type === 'thinking_stop')
      ).toBe(false)
    })

    const boundaries: Array<{ name: string; chunk: StreamChunk }> = [
      { name: 'answer content', chunk: { type: 'content_delta', data: { delta: 'answer' } } },
      { name: 'JSON content', chunk: { type: 'json_delta', data: { delta: '{"ok":true}', snapshot: '{"ok":true}' } } },
      {
        name: 'tool call start',
        chunk: {
          type: 'tool_call_start',
          data: { index: 0, toolCall: { id: 'call-1', type: 'function', function: { name: 'lookup' } } }
        }
      },
      { name: 'code execution start', chunk: { type: 'code_execution_start', data: { id: 'code-1', code: '1 + 1' } } },
      { name: 'message stop', chunk: { type: 'message_stop', data: { finishReason: 'stop' } } },
      { name: 'final result', chunk: finalResultChunk(result({ content: 'answer' })) },
      { name: 'error', chunk: { type: 'error', data: { error: new Error('provider failed') } } }
    ]

    it.each(boundaries)('stops active thinking before $name', async ({ chunk }) => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_start' },
            { type: 'thinking_delta', data: { delta: 'thought' } },
            chunk
          ])
        )
      )

      expect(output.slice(0, 3)).toEqual([
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'thought' } },
        { type: 'thinking_stop' }
      ])
      expect(output[3]?.type).toBe(chunk.type)
    })

    it('stops an active text cycle on natural iterator completion', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_start' },
            { type: 'thinking_delta', data: { delta: 'unfinished thought' } }
          ])
        )
      )

      expect(output).toEqual([
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'unfinished thought' } },
        { type: 'thinking_stop' }
      ])
    })

    it('stops an explicit signal with no disclosed text on natural completion', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([{ type: 'thinking_start' }])
        )
      )

      expect(output).toEqual([{ type: 'thinking_start' }, { type: 'thinking_stop' }])
    })

    it('preserves non-thinking chunks by identity and relative order', async () => {
      const created: StreamChunk = { type: 'message_start', data: { provider: 'openai', model: 'gpt-test' } }
      const citation: StreamChunk = {
        type: 'citation_delta',
        data: { index: 0, citation: { sourceId: 'source-1', title: 'Source' } }
      }
      const toolDelta: StreamChunk = {
        type: 'tool_call_delta',
        data: { index: 0, id: 'call-1', functionArgumentChunk: '{"id":1}' }
      }
      const usage: StreamChunk = {
        type: 'final_usage',
        data: { usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 } }
      }

      const output = await collect(normalizeThinkingStream(streamChunks([created, citation, toolDelta, usage])))

      expect(output).toHaveLength(4)
      expect(output[0]).toBe(created)
      expect(output[1]).toBe(citation)
      expect(output[2]).toBe(toolDelta)
      expect(output[3]).toBe(usage)
    })
  })

  describe('leading raw thinking tags', () => {
    for (const tag of ['think', 'thinking', 'analysis']) {
      it(`extracts <${tag}> when every opening/closing character boundary is split`, async () => {
        const tagged = `<${tag}>private reasoning</${tag}>\npublic answer`

        for (let split = 0; split <= tagged.length; split += 1) {
          const output = await collect(
            normalizeThinkingStream(
              streamChunks<StreamChunk>([
                { type: 'content_delta', data: { delta: tagged.slice(0, split) } },
                { type: 'content_delta', data: { delta: tagged.slice(split) } }
              ])
            )
          )

          expect(output.filter(chunk => chunk.type === 'thinking_start')).toHaveLength(1)
          expect(output.filter(chunk => chunk.type === 'thinking_stop')).toHaveLength(1)
          expect(
            output
              .filter(
                (chunk): chunk is Extract<StreamChunk, { type: 'thinking_delta' }> => chunk.type === 'thinking_delta'
              )
              .map(chunk => chunk.data.delta)
              .join('')
          ).toBe('private reasoning')
          expect(
            output
              .filter(
                (chunk): chunk is Extract<StreamChunk, { type: 'content_delta' }> => chunk.type === 'content_delta'
              )
              .map(chunk => chunk.data.delta)
              .join('')
          ).toBe('public answer')
          expect(output.findIndex(chunk => chunk.type === 'thinking_stop')).toBeLessThan(
            output.findIndex(chunk => chunk.type === 'content_delta')
          )
        }
      })
    }

    it('handles mixed-case tags, discards leading whitespace, removes one CRLF, and preserves remaining answer whitespace', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'content_delta', data: { delta: ' \t<ThInKiNg>reason</tHiNkInG>\r' } },
            { type: 'content_delta', data: { delta: '\n\n  answer' } }
          ])
        )
      )

      expect(output).toEqual([
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'reason' } },
        { type: 'thinking_stop' },
        { type: 'content_delta', data: { delta: '\n  answer' } }
      ])
    })

    it('treats an unclosed supported leading block as thinking and stops it at completion', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([{ type: 'content_delta', data: { delta: '<analysis>still private' } }])
        )
      )

      expect(output).toEqual([
        { type: 'thinking_start' },
        { type: 'thinking_delta', data: { delta: 'still private' } },
        { type: 'thinking_stop' }
      ])
    })

    it('keeps a stray closing tag and supported-looking tags after answer text literal', async () => {
      const stray = '</think> literal close'
      const later = 'Answer first\n<think>quoted code</think>'
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'content_delta', data: { delta: stray } },
            { type: 'content_delta', data: { delta: later } }
          ])
        )
      )

      expect(output).toEqual([
        { type: 'content_delta', data: { delta: stray } },
        { type: 'content_delta', data: { delta: later } }
      ])
    })

    it('flushes an incomplete unsupported tag prefix as answer before a terminal boundary', async () => {
      const messageStop: StreamChunk = { type: 'message_stop', data: { finishReason: 'stop' } }
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([{ type: 'content_delta', data: { delta: '<thi' } }, messageStop])
        )
      )

      expect(output).toEqual([{ type: 'content_delta', data: { delta: '<thi' } }, messageStop])
      expect(output[1]).toBe(messageStop)
    })
  })

  describe('final result reconciliation', () => {
    it('joins two disclosed cycles with exactly two newlines and excludes both from answer content', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_start' },
            { type: 'thinking_delta', data: { delta: 'cycle one' } },
            { type: 'thinking_stop' },
            { type: 'thinking_start' },
            { type: 'thinking_delta', data: { delta: 'cycle two' } },
            { type: 'thinking_stop' },
            { type: 'content_delta', data: { delta: 'answer only' } },
            finalResultChunk(result())
          ])
        )
      )
      const final = output.find(
        (chunk): chunk is Extract<StreamChunk, { type: 'final_result' }> => chunk.type === 'final_result'
      )

      expect(final?.data.result.content).toBe('answer only')
      expect(final?.data.result.thinkingSteps).toBe('cycle one\n\ncycle two')
      expect(final?.data.result.content).not.toContain('cycle')
    })

    it('gives streamed disclosed text precedence over mapper thinkingSteps', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_delta', data: { delta: 'stream thought' } },
            { type: 'thinking_stop' },
            finalResultChunk(result({ content: 'answer', thinkingSteps: 'mapper thought' }))
          ])
        )
      )
      const final = output.at(-1) as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.thinkingSteps).toBe('stream thought')
    })

    it('retains mapper thinkingSteps when the stream disclosed no text', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_start' },
            { type: 'thinking_stop' },
            finalResultChunk(result({ content: 'answer', thinkingSteps: 'mapper thought' }))
          ])
        )
      )
      const final = output.at(-1) as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.thinkingSteps).toBe('mapper thought')
    })

    it('uses a non-null mapper answer instead of accumulated answer deltas', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'content_delta', data: { delta: 'partial streamed answer' } },
            finalResultChunk(result({ content: 'authoritative mapper answer' }))
          ])
        )
      )
      const final = output.at(-1) as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.content).toBe('authoritative mapper answer')
    })

    it('preserves a final-result-only answer', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([finalResultChunk(result({ content: 'final-only answer' }))])
        )
      )
      const final = output[0] as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.content).toBe('final-only answer')
    })

    it('parses complete JSON when accumulated answer supplies a null mapper result', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'content_delta', data: { delta: '{"ok":' } },
            { type: 'content_delta', data: { delta: 'true}' } },
            finalResultChunk(result({ parsedContent: { stale: true } }))
          ])
        )
      )
      const final = output.at(-1) as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.content).toBe('{"ok":true}')
      expect(final.data.result.parsedContent).toEqual({ ok: true })
    })

    it('sanitizes final-result-only leading thinking, extracts it, and reparses the remaining JSON', async () => {
      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            finalResultChunk(
              result({
                content: '<think>private</think>\n{"answer":42}',
                parsedContent: { stale: true }
              })
            )
          ])
        )
      )
      const final = output[0] as Extract<StreamChunk, { type: 'final_result' }>

      expect(final.data.result.content).toBe('{"answer":42}')
      expect(final.data.result.thinkingSteps).toBe('private')
      expect(final.data.result.parsedContent).toEqual({ answer: 42 })
    })

    it('sets parsedContent to null when sanitation changes content to invalid JSON', () => {
      const normalized = normalizeGenerateResultThinking(
        result({
          content: '<analysis>private</analysis>\nplain answer',
          parsedContent: { stale: true }
        })
      )

      expect(normalized.content).toBe('plain answer')
      expect(normalized.thinkingSteps).toBe('private')
      expect(normalized.parsedContent).toBeNull()
    })

    it('normalizes a thought-only tagged result to null answer content', () => {
      const normalized = normalizeGenerateResultThinking(
        result({ content: '<think>private only</think>', parsedContent: { stale: true } })
      )

      expect(normalized.content).toBeNull()
      expect(normalized.thinkingSteps).toBe('private only')
      expect(normalized.parsedContent).toBeNull()
    })

    it('preserves parsedContent when no sanitation or content fallback changes the answer', () => {
      const parsedContent = { existing: true }
      const original = result({ content: 'plain answer', thinkingSteps: 'existing thought', parsedContent })
      const normalized = normalizeGenerateResultThinking(original)

      expect(normalized).toBe(original)
      expect(normalized.parsedContent).toBe(parsedContent)
    })

    it('normalizes an empty thinkingSteps value to null when content is null', () => {
      const original = result({ content: null, thinkingSteps: '' })
      const normalized = normalizeGenerateResultThinking(original)

      expect(normalized).not.toBe(original)
      expect(normalized.content).toBeNull()
      expect(normalized.thinkingSteps).toBeNull()
    })

    it('combines mapper thinkingSteps with distinct leading-tag thinking during result sanitation', () => {
      const normalized = normalizeGenerateResultThinking(
        result({ content: '<think>tag thought</think>\nanswer', thinkingSteps: 'mapper thought' })
      )

      expect(normalized.content).toBe('answer')
      expect(normalized.thinkingSteps).toBe('mapper thought\n\ntag thought')
    })

    it('preserves tool calls, usage, citations, code results, finish reason, model, raw response, provider state, and container', async () => {
      const toolCalls: NonNullable<GenerateResult['toolCalls']> = [
        { id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{"id":1}' } }
      ]
      const usage = { promptTokens: 10, completionTokens: 5, totalTokens: 15 }
      const citations: NonNullable<GenerateResult['citations']> = []
      const codeExecutionResults: NonNullable<GenerateResult['codeExecutionResults']> = []
      const rawResponse = { opaque: true }
      const providerState = { openAICompatible: { reasoningDetails: [{ type: 'reasoning.text', text: 'private' }] } }
      const container = { id: 'container-1', expiresAt: 'tomorrow' }
      const mapperResult = result({
        content: null,
        toolCalls,
        usage,
        citations,
        codeExecutionResults,
        finishReason: 'tool_calls',
        model: 'provider-model',
        rawResponse,
        providerState,
        container
      })

      const output = await collect(
        normalizeThinkingStream(
          streamChunks<StreamChunk>([
            { type: 'thinking_delta', data: { delta: 'thought' } },
            finalResultChunk(mapperResult)
          ])
        )
      )
      const normalized = (output.at(-1) as Extract<StreamChunk, { type: 'final_result' }>).data.result

      expect(normalized.toolCalls).toBe(toolCalls)
      expect(normalized.usage).toBe(usage)
      expect(normalized.citations).toBe(citations)
      expect(normalized.codeExecutionResults).toBe(codeExecutionResults)
      expect(normalized.finishReason).toBe('tool_calls')
      expect(normalized.model).toBe('provider-model')
      expect(normalized.rawResponse).toBe(rawResponse)
      expect(normalized.providerState).toBe(providerState)
      expect(normalized.container).toBe(container)
    })
  })
})

describe('Responses thinking stream normalization', () => {
  it('repairs malformed lifecycle ordering and ignores empty deltas', async () => {
    const output = await collect(
      normalizeResponsesThinkingStream(
        streamChunks<ResponsesStreamChunk>([
          { type: 'thinking_stop' },
          { type: 'thinking_delta', data: { delta: '' } },
          { type: 'thinking_delta', data: { delta: 'first' } },
          { type: 'thinking_start' },
          { type: 'thinking_delta', data: { delta: 'second' } },
          { type: 'thinking_stop' },
          { type: 'thinking_stop' }
        ])
      )
    )

    expect(output).toEqual([
      { type: 'thinking_start' },
      { type: 'thinking_delta', data: { delta: 'first' } },
      { type: 'thinking_stop' },
      { type: 'thinking_start' },
      { type: 'thinking_delta', data: { delta: 'second' } },
      { type: 'thinking_stop' }
    ])
  })

  const boundaries: Array<{ name: string; chunk: ResponsesStreamChunk }> = [
    { name: 'output text delta', chunk: { type: 'response.output_text.delta', data: { delta: 'answer' } } },
    { name: 'output text done', chunk: { type: 'response.output_text.done', data: { text: 'answer' } } },
    { name: 'tool call start', chunk: { type: 'response.tool_call.start', data: { id: 'call-1', name: 'lookup' } } },
    { name: 'completion', chunk: { type: 'response.completed', data: responseResult() } },
    { name: 'failure', chunk: { type: 'response.failed', data: { error: { message: 'failed' } } } },
    { name: 'cancellation', chunk: { type: 'response.cancelled', data: { reason: 'cancelled' } } },
    { name: 'error', chunk: { type: 'error', data: { error: new Error('failed') } } }
  ]

  it.each(boundaries)('stops active thinking before Responses $name', async ({ chunk }) => {
    const output = await collect(
      normalizeResponsesThinkingStream(
        streamChunks<ResponsesStreamChunk>([{ type: 'thinking_delta', data: { delta: 'thought' } }, chunk])
      )
    )

    expect(output.slice(0, 3)).toEqual([
      { type: 'thinking_start' },
      { type: 'thinking_delta', data: { delta: 'thought' } },
      { type: 'thinking_stop' }
    ])
    expect(output[3]).toBe(chunk)
  })

  it('keeps non-boundary Responses events ordered while thinking remains active, then stops naturally', async () => {
    const created: ResponsesStreamChunk = { type: 'response.created', data: { id: 'resp-1', model: 'test-model' } }
    const toolDelta: ResponsesStreamChunk = { type: 'response.tool_call.delta', data: { id: 'call-1', delta: '{}' } }
    const toolDone: ResponsesStreamChunk = {
      type: 'response.tool_call.done',
      data: { id: 'call-1', name: 'lookup', arguments: '{}' }
    }
    const output = await collect(
      normalizeResponsesThinkingStream(
        streamChunks<ResponsesStreamChunk>([
          { type: 'thinking_delta', data: { delta: 'thought' } },
          created,
          toolDelta,
          toolDone
        ])
      )
    )

    expect(output).toEqual([
      { type: 'thinking_start' },
      { type: 'thinking_delta', data: { delta: 'thought' } },
      created,
      toolDelta,
      toolDone,
      { type: 'thinking_stop' }
    ])
    expect(output[2]).toBe(created)
    expect(output[3]).toBe(toolDelta)
    expect(output[4]).toBe(toolDone)
  })
})
