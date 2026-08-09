import { ProviderKey, RosettaMessage, RosettaTool, RosettaAudioData, GenerateParamsProviderState } from './common.types'
import { ProviderOptions } from './config.types'
import { z } from 'zod'

export type ReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'
export type Verbosity = 'low' | 'medium' | 'high'

export type RosettaResponseFormat =
  | { type: 'text' }
  | {
      type: 'json_object'
      /**
       * Optional JSON schema to guide the model's JSON output.
       *
       * Notes:
       * - OpenAI ignores schema for `json_object` (JSON mode).
       * - For `json_object`, this is treated as informational by the SDK and may be ignored by providers.
       */
      schema?: Record<string, unknown>
    }
  | {
      type: 'json_schema'
      json_schema: {
        /** Optional for providers that don’t require it; required for OpenAI (SDK will default when mapping). */
        name?: string
        /** Defaults to true when supported (OpenAI). */
        strict?: boolean
        /** JSON Schema (draft-07-ish). */
        schema: Record<string, unknown>
        /** Optional runtime validator for provider output parsed through this schema. */
        zodSchema?: z.ZodTypeAny
      }
    }

/**
 * Parameters for generating chat completions (streaming or non-streaming).
 */
export interface GenerateParams {
  /** The provider to use for this request (e.g., 'openai', 'anthropic', or a custom string key). */
  provider: ProviderKey
  /** The specific model ID for the chosen provider. Optional if a default is configured. */
  model?: string
  /** An array of messages forming the conversation history and the current prompt. */
  messages: RosettaMessage[]
  /** The maximum number of tokens to generate in the response. */
  maxTokens?: number
  /** The maximum number of completion tokens to generate in the response (OpenAI-style). */
  maxCompletionTokens?: number
  /** Controls randomness: lower values (e.g., 0.2) make output more focused, higher values (e.g., 0.8) make it more random. */
  temperature?: number
  /** Nucleus sampling parameter: considers only tokens comprising the top `topP` probability mass. */
  topP?: number
  /** GPT-5 reasoning effort control for supported OpenAI Chat Completions models. */
  reasoningEffort?: ReasoningEffort
  /** GPT-5 verbosity control for supported OpenAI Chat Completions models. */
  verbosity?: Verbosity
  /** Sequence(s) where the API will stop generating further tokens. */
  stop?: string | string[] | null
  /**
   * An array of tools the model may call.
   * Each tool must include a `zodSchema` for argument validation.
   */
  tools?: RosettaTool<any>[] // Use RosettaTool<any> for flexibility
  /**
   * Enables provider-native programmatic tool calling when supported.
   * The Anthropic mapper uses this to inject code execution and tool caller metadata.
   */
  programmaticToolCalling?: boolean
  /**
   * Reuses an existing provider container/session for stateful code execution.
   */
  container?: string
  /**
   * Provider-specific request state. Prefer this over legacy provider-specific top-level fields.
   */
  providerState?: GenerateParamsProviderState
  /** Controls whether the model is forced to call a tool ('required' or specific function), allowed to choose ('auto'), or prevented ('none'). */
  toolChoice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } }

  /** Request the model to respond in a specific format (e.g., JSON). Support varies by provider/model. */
  responseFormat?: RosettaResponseFormat

  /** Request the model to provide citations or grounding for its response. Support varies by provider/model. */
  grounding?: {
    enabled: boolean
    /** Source for grounding (e.g., 'web' for Google Search). Provider-specific interpretation. */
    source?: 'web' | string[]
  }

  /**
   * Request disclosed reasoning, provider-neutrally. Mappers render this in their own dialect:
   * Anthropic sends `thinking` (`{type: 'adaptive', display: 'summarized'}` on models with
   * adaptive thinking, `{type: 'enabled', budget_tokens: 1024}` otherwise); Google merges
   * `thinkingConfig: {includeThoughts: true}` into the request config; OpenAI gpt-5.x models
   * route through the Responses API with `reasoning: {summary: 'auto'}` (summary emission is
   * best-effort on OpenAI's side — reasoning always runs, but a summary is not guaranteed per
   * response). Surfaces without a disclosure control (other OpenAI models and Azure on Chat
   * Completions, Groq, OpenAI-compatible endpoints) accept the flag as a no-op — models that
   * disclose reasoning there do so without a request toggle.
   */
  thinking?: boolean

  // Internal flag, not set by user directly on top-level call
  /** @internal */
  stream?: boolean

  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions

  /**
   * Additional provider-specific parameters to pass through to the provider API.
   * These are spread into the final provider payload under explicitly mapped fields,
   * meaning mapped fields (temperature, topP, etc.) take precedence if there is a collision.
   * Use this for provider-specific parameters not covered by the unified interface
   * (e.g., repetition_penalty, presence_penalty, frequency_penalty, top_k, seed, logprobs).
   */
  extraParams?: Record<string, unknown>
}

/**
 * Parameters for generating embeddings.
 */
export interface EmbedParams {
  /** The provider to use for this request. */
  provider: ProviderKey
  /** The specific embedding model ID. Optional if a default is configured. */
  model?: string
  /** The input text(s) to embed. Can be a single string or an array for batching (if supported). */
  input: string | string[]
  /** Optional: Specify the desired output format (e.g., 'float', 'base64'). Support varies. */
  encodingFormat?: 'float' | 'base64' // Check provider specifics
  /** Optional: Desired dimension size for the output embeddings (OpenAI specific). */
  dimensions?: number
  /**
   * Additional provider-specific parameters to pass through to the provider API.
   * Mapped fields take precedence over extraParams in case of collision.
   */
  extraParams?: Record<string, unknown>
  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions
}

/**
 * Parameters for generating speech (Text-to-Speech).
 */
export interface SpeechParams {
  /** The provider to use for this request (built-in or custom). */
  provider: ProviderKey
  /** The specific TTS model ID. Optional if a default is configured (e.g., 'tts-1' for OpenAI, 'playai-tts' for Groq). */
  model?: string
  /** The text to synthesize into speech. */
  input: string
  /** The voice to use (provider-specific options, e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer' for OpenAI,
   * or 'Fritz-PlayAI', 'Arista-PlayAI', etc. for Groq). */
  voice: string
  /** The desired audio output format (e.g., 'mp3', 'opus', 'wav'). Defaults to 'mp3' for OpenAI, 'wav' for Groq. */
  responseFormat?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm'
  /** Optional: Speed of the generated speech (0.25 to 4.0). Defaults to 1.0. */
  speed?: number
  /** Optional: Provider-specific TTS options bag (e.g., ElevenLabs voice settings). */
  ttsOptions?: Record<string, unknown>
  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions
  ttsNormalize?: boolean
}

/**
 * Base parameters for audio processing (Transcription/Translation).
 */
interface BaseAudioParams {
  /** The provider to use for this request (built-in or custom). */
  provider: ProviderKey
  /** The specific STT/translation model ID. Optional if a default is configured (e.g., 'whisper-1'). */
  model?: string
  /** The audio data to process. */
  audio: RosettaAudioData
  /** Optional: Language of the input audio (ISO-639-1). Hint for transcription accuracy. */
  language?: string
  /** Optional: Prompt to guide the model's style or provide context. */
  prompt?: string
  /** Optional: Desired format for the response text (e.g., 'json', 'text', 'srt'). Defaults to 'json'. */
  responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt' // Check provider specifics
  /** Optional: Granularity of timestamps (word or segment level). Support varies. */
  timestampGranularities?: ('word' | 'segment')[]
  /**
   * Additional provider-specific parameters to pass through to the provider API.
   * Mapped fields take precedence over extraParams in case of collision.
   */
  extraParams?: Record<string, unknown>
  /** Optional: Enable speaker diarization (identify different speakers). Provider-specific, e.g., ElevenLabs. */
  diarize?: boolean
  /** Optional: Tag non-speech audio events (e.g., laughter, applause). Provider-specific, e.g., ElevenLabs. */
  tagAudioEvents?: boolean
  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions
}

/**
 * Parameters for transcribing audio (Speech-to-Text).
 */
export type TranscribeParams = BaseAudioParams

/**
 * Parameters for translating audio into English text.
 */
export type TranslateParams = Omit<BaseAudioParams, 'language'> // Language is not applicable for translation to English
