import { RosettaToolCallRequest, Citation } from './common.types'

/**
 * Represents token usage statistics for an API call.
 * Fields are optional as not all providers/responses include all details.
 */
export interface TokenUsage {
  promptTokens?: number
  completionTokens?: number
  totalTokens?: number
  /** Tokens related to cached content (Google specific). */
  cachedContentTokenCount?: number
}

/**
 * The result structure for a non-streaming generation request (`RosettaAI.generate`).
 */
export interface GenerateResult {
  /** The primary text content of the response, or null if none was generated (e.g., only tool calls). */
  content: string | null
  /** Tool calls requested by the model, if any. */
  toolCalls?: RosettaToolCallRequest[]
  /** Reason the generation finished (e.g., 'stop', 'length', 'tool_calls', 'content_filter', 'error', 'recitation_filter'). */
  finishReason: string | null
  /** Token usage statistics, if provided by the API. */
  usage?: TokenUsage
  /** Citations or grounding information, if provided by the API (e.g., Google grounding). */
  citations?: Citation[]
  /** Intermediate thinking steps, if requested and provided (Anthropic specific). */
  thinkingSteps?: string | null
  /** The parsed JSON object if `responseFormat: { type: 'json_object' }` was requested and parsing succeeded. Null otherwise. */
  parsedContent?: Record<string, unknown> | Array<unknown> | null
  /** The exact model ID string used for the completion, as reported by the provider. */
  model: string
  /** The raw response object from the underlying SDK (use with caution, structure varies). */
  rawResponse?: unknown
}

/**
 * The result structure for an embedding request (`RosettaAI.embed`).
 */
export interface EmbedResult {
  /** An array of embedding vectors. Each inner array corresponds to an input string. */
  embeddings: number[][]
  /** Token usage statistics for the embedding operation, if provided. */
  usage?: TokenUsage
  /** The exact model ID string used for the embedding (or deployment ID for Azure). */
  model: string
  /** The raw response object from the underlying SDK. */
  rawResponse?: unknown
}

/**
 * The result structure for audio transcription or translation requests (`RosettaAI.transcribe`, `RosettaAI.translate`).
 */
export interface TranscriptionResult {
  /** The transcribed or translated text. */
  text: string
  /** Optional: Language detected or used (ISO-639-1). Provided by some models/modes. */
  language?: string
  /** Optional: Duration of the audio processed in seconds. Provided by some models/modes. */
  duration?: number
  /** Optional: Segment-level details (text, timestamps, etc.). Structure varies by provider and response format. */
  segments?: unknown[]
  /** Optional: Word-level timestamps. Structure varies by provider and response format. */
  words?: unknown[]
  /** The exact model ID string used for the transcription/translation (or deployment ID for Azure). */
  model: string
  /** The raw response object from the underlying SDK. */
  rawResponse?: unknown
}
