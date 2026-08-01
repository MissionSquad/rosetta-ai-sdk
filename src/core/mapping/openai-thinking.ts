import { OpenAICompatibleAssistantProviderState } from '../../types'

const THINKING_TAG_NAMES = ['think', 'thinking', 'analysis'] as const

export interface OpenAIThinkingExtraction {
  thinkingDeltas: string[]
  answerDeltas: string[]
  hasThinkingSignal: boolean
  reasoningDetails?: unknown[]
  structuredContent?: unknown[]
}

export interface LeadingThinkingParseResult {
  thinking: string
  content: string
  hasThinkingSignal: boolean
}

interface StreamingParseResult {
  thinkingDelta: string
  contentDelta: string
  hasThinkingSignal: boolean
}

type ParserState = 'leading' | 'thinking' | 'after_close' | 'answer'

export function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function cloneReplayValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cloneReplayValue)
  if (isUnknownRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, cloneReplayValue(entry)]))
  }
  return value
}

export function cloneReplayRecordArray(value: unknown): unknown[] | undefined {
  if (!Array.isArray(value) || !value.every(isUnknownRecord)) return undefined
  return value.map(cloneReplayValue)
}

export function cloneOpenAICompatibleAssistantProviderState(
  value: unknown
): OpenAICompatibleAssistantProviderState | undefined {
  if (!isUnknownRecord(value)) return undefined
  const reasoningDetails = cloneReplayRecordArray(value.reasoningDetails)
  const structuredContent = cloneReplayRecordArray(value.structuredContent)
  if (!reasoningDetails && !structuredContent) return undefined
  return {
    ...(reasoningDetails ? { reasoningDetails } : {}),
    ...(structuredContent ? { structuredContent } : {})
  }
}

function extractReasoningDetails(
  value: unknown
): {
  text: string
  hasSignal: boolean
  replay?: unknown[]
} {
  const replay = cloneReplayRecordArray(value)
  if (!replay) return { text: '', hasSignal: false }

  let text = ''
  let hasSignal = false
  for (const entry of replay) {
    if (!isUnknownRecord(entry)) continue
    if (entry.type === 'reasoning.text' && typeof entry.text === 'string') {
      text += entry.text
      hasSignal = true
    } else if (entry.type === 'reasoning.summary' && typeof entry.summary === 'string') {
      text += entry.summary
      hasSignal = true
    } else if (entry.type === 'reasoning.encrypted') {
      hasSignal = true
    }
  }
  return { text, hasSignal, replay }
}

function firstNonEmptyString(record: Record<string, unknown>, keys: readonly string[]): string | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.length > 0) return value
  }
  return undefined
}

function extractStructuredContent(
  value: unknown
): {
  thinking: string
  answers: string[]
  replay?: unknown[]
} {
  const replay = cloneReplayRecordArray(value)
  if (!replay) return { thinking: '', answers: [] }

  let thinking = ''
  const answers: string[] = []
  for (const item of replay) {
    if (!isUnknownRecord(item)) continue
    if (item.type === 'thinking' && Array.isArray(item.thinking)) {
      for (const part of item.thinking) {
        if (isUnknownRecord(part) && typeof part.text === 'string') thinking += part.text
      }
    } else if (item.type === 'text' && typeof item.text === 'string') {
      answers.push(item.text)
    }
  }
  return { thinking, answers, replay }
}

/** Extracts disclosed reasoning and answer text from an OpenAI-compatible message or delta. */
export function extractOpenAIThinking(value: unknown): OpenAIThinkingExtraction {
  if (!isUnknownRecord(value)) {
    return { thinkingDeltas: [], answerDeltas: [], hasThinkingSignal: false }
  }

  const details = extractReasoningDetails(value.reasoning_details)
  const alias = firstNonEmptyString(value, ['reasoning_content', 'reasoning', 'thinking', 'analysis'])
  const structured = extractStructuredContent(value.content)
  const hasTextualDetails = details.text.length > 0
  const selectedAlias = hasTextualDetails ? undefined : alias
  const selectedStructuredThinking = hasTextualDetails || selectedAlias ? '' : structured.thinking

  const thinking = details.text || selectedAlias || selectedStructuredThinking
  const answerDeltas = typeof value.content === 'string' ? [value.content] : structured.answers

  return {
    thinkingDeltas: thinking.length > 0 ? [thinking] : [],
    answerDeltas,
    hasThinkingSignal: details.hasSignal || thinking.length > 0,
    ...(details.replay ? { reasoningDetails: details.replay } : {}),
    ...(structured.replay ? { structuredContent: structured.replay } : {})
  }
}

function longestSuffixPrefix(value: string, prefix: string): number {
  const max = Math.min(value.length, prefix.length - 1)
  for (let length = max; length > 0; length -= 1) {
    if (value.slice(-length).toLowerCase() === prefix.slice(0, length).toLowerCase()) return length
  }
  return 0
}

/** Stateful parser for a supported leading reasoning tag whose delimiters may span stream chunks. */
export class LeadingThinkingTagParser {
  private state: ParserState = 'leading'
  private pending = ''
  private closingTag = ''
  private signalled = false

  push(chunk: string): StreamingParseResult {
    if (chunk.length === 0) return { thinkingDelta: '', contentDelta: '', hasThinkingSignal: false }

    let remaining = chunk
    let thinkingDelta = ''
    let contentDelta = ''
    let hasThinkingSignal = false

    while (remaining.length > 0) {
      if (this.state === 'answer') {
        contentDelta += remaining
        remaining = ''
        continue
      }

      if (this.state === 'leading') {
        this.pending += remaining
        remaining = ''
        const candidate = this.pending.trimStart()
        const leadingWhitespaceLength = this.pending.length - candidate.length
        const matchingTag = THINKING_TAG_NAMES.find(name => candidate.toLowerCase().startsWith(`<${name}>`))
        if (matchingTag) {
          const openingTagLength = matchingTag.length + 2
          remaining = candidate.slice(openingTagLength)
          this.pending = ''
          this.closingTag = `</${matchingTag}>`
          this.state = 'thinking'
          this.signalled = true
          hasThinkingSignal = true
          continue
        }

        const mayBecomeTag = THINKING_TAG_NAMES.some(name => `<${name}>`.startsWith(candidate.toLowerCase()))
        if (candidate.length === 0 || mayBecomeTag) continue

        contentDelta += this.pending.slice(0, leadingWhitespaceLength) + candidate
        this.pending = ''
        this.state = 'answer'
        continue
      }

      if (this.state === 'thinking') {
        this.pending += remaining
        remaining = ''
        const closeIndex = this.pending.toLowerCase().indexOf(this.closingTag.toLowerCase())
        if (closeIndex >= 0) {
          thinkingDelta += this.pending.slice(0, closeIndex)
          remaining = this.pending.slice(closeIndex + this.closingTag.length)
          this.pending = ''
          this.state = 'after_close'
          continue
        }

        const suffixLength = longestSuffixPrefix(this.pending, this.closingTag)
        const safeLength = this.pending.length - suffixLength
        thinkingDelta += this.pending.slice(0, safeLength)
        this.pending = this.pending.slice(safeLength)
        continue
      }

      this.pending += remaining
      remaining = ''
      if (this.pending === '\r') continue
      if (this.pending.startsWith('\r\n')) {
        remaining = this.pending.slice(2)
      } else if (this.pending.startsWith('\n')) {
        remaining = this.pending.slice(1)
      } else {
        remaining = this.pending
      }
      this.pending = ''
      this.state = 'answer'
    }

    return { thinkingDelta, contentDelta, hasThinkingSignal }
  }

  finish(): StreamingParseResult {
    let thinkingDelta = ''
    let contentDelta = ''
    if (this.state === 'leading' || this.state === 'answer') contentDelta = this.pending
    else if (this.state === 'thinking') thinkingDelta = this.pending
    else if (this.state === 'after_close' && this.pending === '\r') contentDelta = this.pending
    this.pending = ''
    this.state = 'answer'
    return { thinkingDelta, contentDelta, hasThinkingSignal: false }
  }

  get hasThinkingSignal(): boolean {
    return this.signalled
  }
}

/** Parses one complete provider response using the same leading-tag rules as streaming. */
export function parseLeadingThinkingTags(content: string): LeadingThinkingParseResult {
  const parser = new LeadingThinkingTagParser()
  const pushed = parser.push(content)
  const finished = parser.finish()
  return {
    thinking: pushed.thinkingDelta + finished.thinkingDelta,
    content: pushed.contentDelta + finished.contentDelta,
    hasThinkingSignal: parser.hasThinkingSignal
  }
}
