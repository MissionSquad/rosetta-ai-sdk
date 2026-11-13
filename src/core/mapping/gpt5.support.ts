import type { ReasoningEffort } from '../../types'

export interface Gpt5Support {
  chatCompletionsSupported: boolean
  allowedReasoningEfforts: ReasoningEffort[]
  defaultReasoningEffort?: ReasoningEffort
  fixedReasoningEffort?: ReasoningEffort
  supportsVerbosity: boolean
  supportsSampling: 'never' | 'always' | 'only_with_reasoning_none'
  supportsFunctionToolsWithReasoning?: boolean
  /**
   * gpt-5.6 rejects Chat Completions requests that attach function tools unless
   * reasoning_effort is explicitly 'none' — merely omitting the parameter also 400s
   * ("Function tools with reasoning_effort are not supported ... set reasoning_effort
   * to 'none'"). When set, the mapper sends the explicit 'none' instead of omitting.
   */
  functionToolsRequireExplicitNone?: boolean
}

// Chat Completions-supported GPT-5 and GPT-6 models verified for this implementation.
// GPT-6 and gpt-5.6 rows were verified live on 2026-11-12 with examples/probe-new-models.ts.
const GPT5_CHAT_COMPLETIONS_SUPPORT: Record<string, Gpt5Support> = {
  'gpt-5': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5-mini': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5-nano': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5.1': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5.2': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5.4': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: false
  },
  // Covers gpt-5.6-terra / gpt-5.6-sol (and future gpt-5.6-* variants) via prefix matching.
  'gpt-5.6': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none',
    supportsFunctionToolsWithReasoning: false,
    functionToolsRequireExplicitNone: true
  },
  // GPT-6: every model defaults to medium effort and rejects 'minimal'. Sol, Luna and 6.1 Terra
  // accept 'none' (and sampling parameters only alongside it).
  'gpt-6-sol': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none',
    supportsFunctionToolsWithReasoning: false,
    functionToolsRequireExplicitNone: true
  },
  'gpt-6-luna': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none',
    supportsFunctionToolsWithReasoning: false,
    functionToolsRequireExplicitNone: true
  },
  'gpt-6.1-terra': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none',
    supportsFunctionToolsWithReasoning: false,
    functionToolsRequireExplicitNone: true
  },
  // Astra and 6.1 Sol reject effort 'none', so they never accept sampling parameters and cannot
  // attach function tools on Chat Completions at all (the Responses API is the only route).
  'gpt-6-astra': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: false
  },
  'gpt-6.1-sol': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: false
  },
  'gpt-5-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: [],
    supportsVerbosity: true,
    supportsSampling: 'always',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5.1-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['medium'],
    fixedReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  },
  'gpt-5.2-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['medium'],
    fixedReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never',
    supportsFunctionToolsWithReasoning: true
  }
}

// Verified GPT-5 inventory that is treated as Responses-only / unsupported on Chat Completions here.
const GPT5_CHAT_COMPLETIONS_UNSUPPORTED: Record<string, Gpt5Support> = {
  'gpt-5-codex': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5-pro': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: ['high'],
    fixedReasoningEffort: 'high',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5-search-api': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.1-codex': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.1-codex-max': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.1-codex-mini': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.2-codex': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.2-pro': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: ['high'],
    fixedReasoningEffort: 'high',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5.3-chat-latest': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.3-codex': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: [],
    supportsVerbosity: false,
    supportsSampling: 'never'
  },
  'gpt-5.4-pro': {
    chatCompletionsSupported: false,
    allowedReasoningEfforts: ['high'],
    fixedReasoningEffort: 'high',
    supportsVerbosity: true,
    supportsSampling: 'never'
  }
}

export const GPT5_SUPPORT_TABLE: Record<string, Gpt5Support> = {
  ...GPT5_CHAT_COMPLETIONS_SUPPORT,
  ...GPT5_CHAT_COMPLETIONS_UNSUPPORTED
}

const GPT5_SUPPORT_KEYS = Object.keys(GPT5_SUPPORT_TABLE).sort((a, b) => b.length - a.length)

/** Returns a copy of Rosetta's known capabilities for an OpenAI GPT-5 or GPT-6 model. */
export function getGpt5Support(model: string): Gpt5Support | undefined {
  const key = Object.prototype.hasOwnProperty.call(GPT5_SUPPORT_TABLE, model)
    ? model
    : GPT5_SUPPORT_KEYS.find(key => model.startsWith(`${key}-`))
  const support = key === undefined ? undefined : GPT5_SUPPORT_TABLE[key]
  if (support === undefined) return undefined

  return { ...support, allowedReasoningEfforts: [...support.allowedReasoningEfforts] }
}
