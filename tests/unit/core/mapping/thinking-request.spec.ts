import { resolveThinkingRequest } from '../../../../src/core/mapping/thinking-request'

describe('resolveThinkingRequest', () => {
  it('passes through when no thinking signal exists', () => {
    expect(resolveThinkingRequest({}, [])).toEqual({ thinkingRequested: false })
    expect(resolveThinkingRequest({ thinking: true }, [])).toEqual({ thinkingRequested: true })
  })

  it('leaves extraParams undefined when the input had none', () => {
    const resolved = resolveThinkingRequest({ thinking: true }, ['thinking'])
    expect(resolved.extraParams).toBeUndefined()
  })

  it('does not mutate the caller extraParams object', () => {
    const extraParams = { thinkingConfig: { includeThoughts: true }, keep: 1 }
    const resolved = resolveThinkingRequest({ extraParams }, [])
    expect(extraParams).toEqual({ thinkingConfig: { includeThoughts: true }, keep: 1 })
    expect(resolved.extraParams).toEqual({ keep: 1 })
  })

  it('preserves native keys and strips foreign ones', () => {
    const resolved = resolveThinkingRequest(
      {
        extraParams: {
          thinking: { type: 'adaptive' },
          thinkingConfig: { includeThoughts: true },
          reasoning: { effort: 'high' },
          reasoning_effort: 'high',
          reasoning_format: 'parsed',
          include_reasoning: true,
          unrelated: 'kept'
        }
      },
      ['thinking']
    )
    expect(resolved.extraParams).toEqual({ thinking: { type: 'adaptive' }, unrelated: 'kept' })
    expect(resolved.thinkingRequested).toBe(true)
  })

  it('translates a stripped Google thinkingConfig with includeThoughts into a thinking request', () => {
    const resolved = resolveThinkingRequest({ extraParams: { thinkingConfig: { includeThoughts: true } } }, [])
    expect(resolved).toEqual({ thinkingRequested: true, extraParams: {} })
  })

  it('translates stripped Anthropic adaptive/enabled thinking objects into a thinking request', () => {
    for (const type of ['adaptive', 'enabled']) {
      const resolved = resolveThinkingRequest({ extraParams: { thinking: { type } } }, [])
      expect(resolved.thinkingRequested).toBe(true)
      expect(resolved.extraParams).toEqual({})
    }
  })

  it('translates a stripped include_reasoning: true into a thinking request', () => {
    const resolved = resolveThinkingRequest({ extraParams: { include_reasoning: true } }, [])
    expect(resolved).toEqual({ thinkingRequested: true, extraParams: {} })
  })

  it('never forces thinking on from negative or ambiguous signals', () => {
    const resolved = resolveThinkingRequest(
      {
        extraParams: {
          thinking: { type: 'disabled' },
          thinkingConfig: { includeThoughts: false, thinkingBudget: 512 },
          include_reasoning: false,
          reasoning_effort: 'high',
          reasoning: { effort: 'high' }
        }
      },
      []
    )
    expect(resolved).toEqual({ thinkingRequested: false, extraParams: {} })
  })

  it('ignores malformed values while still stripping the keys', () => {
    const resolved = resolveThinkingRequest(
      { extraParams: { thinking: 'yes', thinkingConfig: [1, 2], include_reasoning: 'true' } },
      []
    )
    expect(resolved).toEqual({ thinkingRequested: false, extraParams: {} })
  })
})
