import { TokenUsage, GenerateResult } from './result.types'
import { Citation } from './common.types'
import { Provider } from './common.types'

/**
 * Discriminated union representing the different types of events
 * yielded by the `RosettaAI.stream` async generator.
 */
export type StreamChunk =
  // --- Lifecycle & Metadata ---
  /** Signals the start of the stream, providing initial metadata like provider and model ID. */
  | { type: 'message_start'; data: { provider: Provider; model: string } }
  /** Signals the end of the message generation, indicating the reason for stopping. */
  | { type: 'message_stop'; data: { finishReason: string | null } }
  /** Contains the final token usage statistics for the entire operation, usually sent at the very end. */
  | { type: 'final_usage'; data: { usage: TokenUsage } }
  /** Contains the fully aggregated `GenerateResult` object, often sent as the last event. */
  | { type: 'final_result'; data: { result: GenerateResult } }
  /** Indicates an error occurred either during stream setup or processing. */
  | { type: 'error'; data: { error: Error } }

  // --- Content & Thinking ---
  /** A chunk of the primary text content being generated. */
  | { type: 'content_delta'; data: { delta: string } }
  /** Anthropic: Signals that the model has started its internal "thinking" process (if requested). */
  | { type: 'thinking_start' }
  /** Anthropic: A chunk of the model's internal "thinking" text (if requested). */
  | { type: 'thinking_delta'; data: { delta: string } }
  /** Anthropic: Signals that the model has finished its internal "thinking" process. */
  | { type: 'thinking_stop' }

  // --- Tool Calls ---
  /** Signals the start of a tool call request from the model. Provides the tool name and ID. */
  | {
      type: 'tool_call_start'
      data: { index: number; toolCall: { id: string; type: 'function'; function: { name: string } } }
    }
  /** A chunk of the JSON arguments string being streamed for a specific tool call. */
  | { type: 'tool_call_delta'; data: { index: number; id: string; functionArgumentChunk: string } }
  /** Signals that the argument stream for a specific tool call has finished. */
  | { type: 'tool_call_done'; data: { index: number; id: string } }

  // --- Structured Output (JSON) ---
  /** A chunk of the raw JSON string being streamed when JSON mode is active. Includes the accumulated snapshot and a partial parse attempt. */
  | { type: 'json_delta'; data: { delta: string; parsed?: any; snapshot: string } }
  /** Signals that the JSON output stream is finished. Includes the final raw snapshot and the fully parsed object (if successful). */
  | { type: 'json_done'; data: { parsed: any | null; snapshot: string } }

  // --- Citations / Grounding ---
  /** Contains partial information about a citation as it's being streamed (support varies). */
  | { type: 'citation_delta'; data: { index: number; citation: Partial<Citation> & { sourceId: string } } }
  /** Contains the complete information for a specific citation once fully received. */
  | { type: 'citation_done'; data: { index: number; citation: Citation } }

/**
 * Type for streaming audio data chunks (e.g., from TTS `streamSpeech`).
 */
export type AudioStreamChunk =
  /** Contains a chunk of the raw audio data as a Buffer. */
  | { type: 'audio_chunk'; data: Buffer }
  /** Signals that the audio stream has finished. */
  | { type: 'audio_stop' }
  /** Indicates an error occurred during audio stream generation or processing. */
  | { type: 'error'; data: { error: Error } }
