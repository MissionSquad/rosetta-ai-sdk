import { getGpt5Support, Gpt5Support } from '../../../../src'

describe('getGpt5Support public export', () => {
  it('looks up known aliases and snapshots and returns undefined for unknown identifiers', () => {
    const support: Gpt5Support | undefined = getGpt5Support('gpt-5.6-terra')
    expect(support?.allowedReasoningEfforts).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max'])
    expect(getGpt5Support('gpt-5.6-terra-2026-09-01')).toEqual(support)
    expect(getGpt5Support('gpt-5-chat-latest')?.allowedReasoningEfforts).toEqual([])
    expect(getGpt5Support('gpt-4o')).toBeUndefined()
    expect(getGpt5Support('constructor')).toBeUndefined()
  })

  it('isolates the shared lookup table from mutation by callers', () => {
    const support = getGpt5Support('gpt-5.6-terra')!
    support.supportsVerbosity = false
    support.allowedReasoningEfforts.length = 0

    expect(getGpt5Support('gpt-5.6-terra')).toMatchObject({
      supportsVerbosity: true,
      allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max']
    })
  })
})
