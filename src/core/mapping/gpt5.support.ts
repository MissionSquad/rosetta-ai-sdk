import type { ReasoningEffort } from '../../types'

export interface Gpt5Support {
  chatCompletionsSupported: boolean
  allowedReasoningEfforts: ReasoningEffort[]
  defaultReasoningEffort?: ReasoningEffort
  fixedReasoningEffort?: ReasoningEffort
  supportsVerbosity: boolean
  supportsSampling: 'never' | 'always' | 'only_with_reasoning_none'
}

// Chat Completions-supported GPT-5 models verified for this implementation.
const GPT5_CHAT_COMPLETIONS_SUPPORT: Record<string, Gpt5Support> = {
  'gpt-5': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5-mini': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5-nano': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5.1': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5.2': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'only_with_reasoning_none'
  },
  'gpt-5.4': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh'],
    defaultReasoningEffort: 'none',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: [],
    supportsVerbosity: true,
    supportsSampling: 'always'
  },
  'gpt-5.1-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['medium'],
    fixedReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never'
  },
  'gpt-5.2-chat-latest': {
    chatCompletionsSupported: true,
    allowedReasoningEfforts: ['medium'],
    fixedReasoningEffort: 'medium',
    supportsVerbosity: true,
    supportsSampling: 'never'
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

export function getGpt5Support(model: string): Gpt5Support | undefined {
  const exact = GPT5_SUPPORT_TABLE[model]
  if (exact) return exact

  const prefix = GPT5_SUPPORT_KEYS.find(key => model.startsWith(`${key}-`))
  return prefix ? GPT5_SUPPORT_TABLE[prefix] : undefined
}
