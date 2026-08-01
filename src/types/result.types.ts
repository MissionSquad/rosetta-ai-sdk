import { VoiceSettings } from '@elevenlabs/elevenlabs-js/api'
import { RosettaToolCallRequest, Citation, ProviderKey, GenerateResultProviderState, CodeExecutionResultInfo } from './common.types'
import { OpenAICompletion } from './openai.types' // Import the new type

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
  /** Provider-neutral disclosed reasoning, when returned by the selected provider. */
  thinkingSteps?: string | null
  /** Managed code execution results emitted by providers that support programmatic execution. */
  codeExecutionResults?: CodeExecutionResultInfo[]
  /** The parsed JSON object if `responseFormat: { type: 'json_object' }` was requested and parsing succeeded. Null otherwise. */
  parsedContent?: Record<string, unknown> | Array<unknown> | null
  /** The exact model ID string used for the completion, as reported by the provider. */
  model: string
  /** The raw response object from the underlying SDK (use with caution, structure varies). */
  rawResponse?: unknown
  /** Provider-specific response state for downstream persistence and replay. */
  providerState?: GenerateResultProviderState
  /** Provider-native container/session metadata, when returned. */
  container?: {
    id: string
    expiresAt?: string | null
  }
  /**
   * If `openAICompletions` was true in the config, this field will contain the
   * response transformed into the standard OpenAI Chat Completion format.
   */
  openAIResponse?: OpenAICompletion
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

/**
 * Voice metadata (provider-agnostic) for TTS voice catalogs.
 */
export interface RosettaVoice {
  /** Provider-specific voice identifier */
  id: string
  /** Human-readable name if provided by the provider */
  name?: string
  /** Optional labels/tags attached to the voice */
  labels?: Record<string, string>
  /** Optional voice category/type provided by the provider */
  category?: string
  /** The description of the voice. */
  description?: string
  /** Optional preview URL to sample the voice */
  previewUrl?: string
  /** The settings of the voice. */
  settings?: VoiceSettings
  /** Whether the authenticated user owns the voice (provider-dependent) */
  owned?: boolean
  /** Provider key for reference */
  provider: ProviderKey
  /** Raw provider response for advanced use */
  rawData?: unknown
}

/**
 * Standard list wrapper for voice catalogs.
 */
export interface RosettaVoiceList {
  object: 'list'
  data: RosettaVoice[]
}
