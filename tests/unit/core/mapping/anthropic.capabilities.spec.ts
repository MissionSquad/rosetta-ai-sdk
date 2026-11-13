import {
  ANTHROPIC_THINKING_BINDING_BETA,
  anthropicRejectsForcedToolChoice,
  anthropicRejectsSamplingParams,
  anthropicRequiresAdaptiveThinking,
  anthropicRunsThinkingHistoryCheck,
  getRequiredAnthropicBetas,
  mergeAnthropicBetas
} from '../../../../src/core/mapping/anthropic.capabilities'

interface LiveRow {
  model: string
  /** Rejects temperature / top_p, and needs adaptive (not budgeted) thinking. */
  rejectsSampling: boolean
  rejectsForcedToolChoice: boolean
  runsHistoryCheck: boolean
}

// One row per model probed live on 2026-11-12 (examples/probe-new-models.ts).
const LIVE_MATRIX: LiveRow[] = [
  { model: 'claude-haiku-4-5', rejectsSampling: false, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-sonnet-4-5', rejectsSampling: false, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-4-5', rejectsSampling: false, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-sonnet-4-6', rejectsSampling: false, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-4-6', rejectsSampling: false, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-4-7', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-4-8', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-sonnet-5', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-5', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-fable-5', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: false },
  { model: 'claude-opus-5-5', rejectsSampling: true, rejectsForcedToolChoice: true, runsHistoryCheck: true },
  { model: 'claude-sonnet-5-5', rejectsSampling: true, rejectsForcedToolChoice: true, runsHistoryCheck: true },
  { model: 'claude-haiku-5-5', rejectsSampling: true, rejectsForcedToolChoice: false, runsHistoryCheck: true },
  { model: 'claude-fable-5-1', rejectsSampling: true, rejectsForcedToolChoice: true, runsHistoryCheck: true },
  { model: 'claude-fable-5-2', rejectsSampling: true, rejectsForcedToolChoice: true, runsHistoryCheck: true }
]

describe('anthropic.capabilities', () => {
  for (const { model, rejectsSampling, rejectsForcedToolChoice, runsHistoryCheck } of LIVE_MATRIX) {
    describe(model, () => {
      it('matches the live-probed behaviour', () => {
        expect(anthropicRejectsSamplingParams(model)).toBe(rejectsSampling)
        expect(anthropicRequiresAdaptiveThinking(model)).toBe(rejectsSampling)
        expect(anthropicRejectsForcedToolChoice(model)).toBe(rejectsForcedToolChoice)
        expect(anthropicRunsThinkingHistoryCheck(model)).toBe(runsHistoryCheck)
      })

      it('gives the same answers with a :1m suffix', () => {
        expect(anthropicRejectsSamplingParams(`${model}:1m`)).toBe(rejectsSampling)
        expect(anthropicRunsThinkingHistoryCheck(`${model}:1m`)).toBe(runsHistoryCheck)
      })
    })
  }

  it('reads a trailing 8-digit segment as a release date, not a minor version', () => {
    // claude-sonnet-4-20250514 is Sonnet 4.0, claude-opus-4-5-20251101 is Opus 4.5.
    expect(anthropicRejectsSamplingParams('claude-sonnet-4-20250514')).toBe(false)
    expect(anthropicRejectsSamplingParams('claude-opus-4-5-20251101')).toBe(false)
    expect(anthropicRejectsSamplingParams('claude-haiku-4-5-20251001')).toBe(false)
    expect(anthropicRejectsSamplingParams('claude-opus-4-7-20260101')).toBe(true)
    expect(anthropicRunsThinkingHistoryCheck('claude-opus-5-5-20260901')).toBe(true)
  })

  it('treats Mythos like Fable of the same version', () => {
    expect(anthropicRejectsSamplingParams('claude-mythos-5')).toBe(true)
    expect(anthropicRejectsForcedToolChoice('claude-mythos-5')).toBe(false)
    expect(anthropicRejectsForcedToolChoice('claude-mythos-5-2')).toBe(true)
    expect(anthropicRunsThinkingHistoryCheck('claude-mythos-5-1')).toBe(true)
  })

  it('applies the newest behaviour to future versions of a known family', () => {
    expect(anthropicRejectsSamplingParams('claude-opus-6')).toBe(true)
    expect(anthropicRejectsForcedToolChoice('claude-sonnet-6-1')).toBe(true)
    expect(anthropicRunsThinkingHistoryCheck('claude-haiku-6')).toBe(true)
  })

  it('returns false for ids it cannot parse', () => {
    for (const model of ['', 'gpt-6-sol', 'claude-3-5-sonnet-20241022', 'claude-instant-1', 'claude-opus-latest']) {
      expect(anthropicRejectsSamplingParams(model)).toBe(false)
      expect(anthropicRequiresAdaptiveThinking(model)).toBe(false)
      expect(anthropicRejectsForcedToolChoice(model)).toBe(false)
      expect(anthropicRunsThinkingHistoryCheck(model)).toBe(false)
    }
  })

  describe('beta flags', () => {
    it('requires the binding-controls beta only for models that run the history check', () => {
      expect(getRequiredAnthropicBetas('claude-fable-5-2')).toEqual([ANTHROPIC_THINKING_BINDING_BETA])
      expect(getRequiredAnthropicBetas('claude-opus-5')).toEqual([])
      expect(getRequiredAnthropicBetas('claude-haiku-4-5')).toEqual([])
    })

    it('merges requested and required betas without duplicates', () => {
      expect(mergeAnthropicBetas(undefined, 'claude-opus-5')).toEqual([])
      expect(mergeAnthropicBetas(['context-1m-2025-08-07'], 'claude-sonnet-4-6:1m')).toEqual(['context-1m-2025-08-07'])
      expect(mergeAnthropicBetas(['custom-beta'], 'claude-opus-5-5')).toEqual([
        'custom-beta',
        ANTHROPIC_THINKING_BINDING_BETA
      ])
      expect(mergeAnthropicBetas([ANTHROPIC_THINKING_BINDING_BETA], 'claude-opus-5-5')).toEqual([
        ANTHROPIC_THINKING_BINDING_BETA
      ])
    })
  })
})
