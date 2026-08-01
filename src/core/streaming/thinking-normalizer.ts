import { GenerateResult, ResponsesStreamChunk, StreamChunk, ThinkingStreamChunk } from '../../types'
import { LeadingThinkingTagParser, parseLeadingThinkingTags } from '../mapping/openai-thinking'

class ThinkingLifecycle {
  private active = false
  private currentText = ''
  private readonly completedTexts: string[] = []

  start(): ThinkingStreamChunk[] {
    const chunks: ThinkingStreamChunk[] = []
    if (this.active) chunks.push(...this.stop())
    this.active = true
    this.currentText = ''
    chunks.push({ type: 'thinking_start' })
    return chunks
  }

  delta(delta: string): ThinkingStreamChunk[] {
    if (delta.length === 0) return []
    const chunks = this.active ? [] : this.start()
    this.currentText += delta
    chunks.push({ type: 'thinking_delta', data: { delta } })
    return chunks
  }

  stop(): ThinkingStreamChunk[] {
    if (!this.active) return []
    this.active = false
    if (this.currentText.length > 0) this.completedTexts.push(this.currentText)
    this.currentText = ''
    return [{ type: 'thinking_stop' }]
  }

  get disclosedText(): string {
    const texts = [...this.completedTexts]
    if (this.active && this.currentText.length > 0) texts.push(this.currentText)
    return texts.join('\n\n')
  }
}

function parseCompleteJson(content: string): GenerateResult['parsedContent'] | undefined {
  try {
    const parsed: unknown = JSON.parse(content)
    if (Array.isArray(parsed)) return parsed
    if (typeof parsed === 'object' && parsed !== null) return parsed as Record<string, unknown>
  } catch {
    // The caller distinguishes failed parsing from a valid object/array result.
  }
  return undefined
}

function withContentAndParsedResult(
  result: GenerateResult,
  content: string | null,
  contentChanged: boolean
): GenerateResult {
  if (typeof content === 'string') {
    const parsedContent = parseCompleteJson(content)
    if (parsedContent !== undefined) {
      return { ...result, content, parsedContent }
    }
  }
  if (contentChanged) return { ...result, content, parsedContent: null }
  return content === result.content ? result : { ...result, content }
}

/** Removes a supported leading reasoning block from a non-streaming result. */
export function normalizeGenerateResultThinking(result: GenerateResult): GenerateResult {
  const existingThinking = result.thinkingSteps === '' ? null : result.thinkingSteps
  if (typeof result.content !== 'string') {
    if (result.thinkingSteps === existingThinking) return result
    return { ...result, thinkingSteps: existingThinking }
  }

  const parsed = parseLeadingThinkingTags(result.content)
  const normalizedContent = parsed.hasThinkingSignal && parsed.content.length === 0 ? null : parsed.content
  const parsedThinking = parsed.thinking.length > 0 ? parsed.thinking : null
  let thinkingSteps = existingThinking
  if (parsedThinking && !existingThinking) thinkingSteps = parsedThinking
  else if (parsedThinking && existingThinking && parsedThinking !== existingThinking) {
    thinkingSteps = `${existingThinking}\n\n${parsedThinking}`
  }

  const contentChanged = normalizedContent !== result.content
  const thinkingChanged = thinkingSteps !== result.thinkingSteps
  if (!contentChanged && !thinkingChanged) return result

  return withContentAndParsedResult({ ...result, thinkingSteps }, normalizedContent, contentChanged)
}

function isStreamThinkingBoundary(chunk: StreamChunk): boolean {
  return (
    chunk.type === 'json_delta' ||
    chunk.type === 'json_done' ||
    chunk.type === 'tool_call_start' ||
    chunk.type === 'code_execution_start' ||
    chunk.type === 'message_stop' ||
    chunk.type === 'final_result' ||
    chunk.type === 'error'
  )
}

/** Applies the provider-neutral thought lifecycle and leading-tag sanitation to a stream. */
export async function* normalizeThinkingStream(source: AsyncIterable<StreamChunk>): AsyncIterable<StreamChunk> {
  const lifecycle = new ThinkingLifecycle()
  const tagParser = new LeadingThinkingTagParser()
  let tagParserFinished = false
  let accumulatedAnswer = ''

  const finishTagParser = (): StreamChunk[] => {
    if (tagParserFinished) return []
    tagParserFinished = true
    const parsed = tagParser.finish()
    const chunks: StreamChunk[] = []
    chunks.push(...lifecycle.delta(parsed.thinkingDelta))
    if (parsed.contentDelta.length > 0) {
      chunks.push(...lifecycle.stop())
      accumulatedAnswer += parsed.contentDelta
      chunks.push({ type: 'content_delta', data: { delta: parsed.contentDelta } })
    }
    return chunks
  }

  for await (const chunk of source) {
    if (chunk.type === 'thinking_start') {
      for (const normalized of lifecycle.start()) yield normalized
      continue
    }
    if (chunk.type === 'thinking_delta') {
      for (const normalized of lifecycle.delta(chunk.data.delta)) yield normalized
      continue
    }
    if (chunk.type === 'thinking_stop') {
      for (const normalized of lifecycle.stop()) yield normalized
      continue
    }

    if (chunk.type === 'content_delta' && !tagParserFinished) {
      const parsed = tagParser.push(chunk.data.delta)
      if (parsed.hasThinkingSignal) {
        for (const normalized of lifecycle.start()) yield normalized
      }
      for (const normalized of lifecycle.delta(parsed.thinkingDelta)) yield normalized
      if (parsed.contentDelta.length > 0) {
        for (const normalized of lifecycle.stop()) yield normalized
        accumulatedAnswer += parsed.contentDelta
        yield { type: 'content_delta', data: { delta: parsed.contentDelta } }
      }
      continue
    }

    if (isStreamThinkingBoundary(chunk)) {
      for (const normalized of finishTagParser()) yield normalized
      for (const normalized of lifecycle.stop()) yield normalized
    }

    if (chunk.type === 'content_delta') {
      for (const normalized of lifecycle.stop()) yield normalized
      accumulatedAnswer += chunk.data.delta
      yield chunk
      continue
    }

    if (chunk.type === 'json_delta') accumulatedAnswer += chunk.data.delta

    if (chunk.type === 'final_result') {
      const mapperResult = normalizeGenerateResultThinking(chunk.data.result)
      const streamedThinking = lifecycle.disclosedText
      const content = mapperResult.content !== null ? mapperResult.content : accumulatedAnswer || null
      const thinkingSteps = streamedThinking.length > 0 ? streamedThinking : mapperResult.thinkingSteps
      const contentChanged = content !== mapperResult.content
      const normalizedResult = withContentAndParsedResult(
        thinkingSteps === mapperResult.thinkingSteps ? mapperResult : { ...mapperResult, thinkingSteps },
        content,
        contentChanged
      )
      yield { type: 'final_result', data: { result: normalizedResult } }
      continue
    }

    yield chunk
  }

  for (const normalized of finishTagParser()) yield normalized
  for (const normalized of lifecycle.stop()) yield normalized
}

function isResponsesThinkingBoundary(chunk: ResponsesStreamChunk): boolean {
  return (
    chunk.type === 'response.output_text.delta' ||
    chunk.type === 'response.output_text.done' ||
    chunk.type === 'response.tool_call.start' ||
    chunk.type === 'response.completed' ||
    chunk.type === 'response.failed' ||
    chunk.type === 'response.cancelled' ||
    chunk.type === 'error'
  )
}

/** Applies the same thought lifecycle contract to OpenAI Responses API events. */
export async function* normalizeResponsesThinkingStream(
  source: AsyncIterable<ResponsesStreamChunk>
): AsyncIterable<ResponsesStreamChunk> {
  const lifecycle = new ThinkingLifecycle()
  for await (const chunk of source) {
    if (chunk.type === 'thinking_start') {
      for (const normalized of lifecycle.start()) yield normalized
      continue
    }
    if (chunk.type === 'thinking_delta') {
      for (const normalized of lifecycle.delta(chunk.data.delta)) yield normalized
      continue
    }
    if (chunk.type === 'thinking_stop') {
      for (const normalized of lifecycle.stop()) yield normalized
      continue
    }
    if (isResponsesThinkingBoundary(chunk)) {
      for (const normalized of lifecycle.stop()) yield normalized
    }
    yield chunk
  }
  for (const normalized of lifecycle.stop()) yield normalized
}
