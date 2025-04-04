import { Provider, RosettaMessage, RosettaTool, RosettaAudioData } from './common.types'
import { ProviderOptions } from './config.types'

/**
 * Parameters for generating chat completions (streaming or non-streaming).
 */
export interface GenerateParams {
  /** The provider to use for this request (e.g., 'openai', 'anthropic'). */
  provider: Provider
  /** The specific model ID for the chosen provider. Optional if a default is configured. */
  model?: string
  /** An array of messages forming the conversation history and the current prompt. */
  messages: RosettaMessage[]
  /** The maximum number of tokens to generate in the response. */
  maxTokens?: number
  /** Controls randomness: lower values (e.g., 0.2) make output more focused, higher values (e.g., 0.8) make it more random. */
  temperature?: number
  /** Nucleus sampling parameter: considers only tokens comprising the top `topP` probability mass. */
  topP?: number
  /** Sequence(s) where the API will stop generating further tokens. */
  stop?: string | string[] | null
  /** An array of tools the model may call. Currently only 'function' type is broadly supported. */
  tools?: RosettaTool[]
  /** Controls whether the model is forced to call a tool ('required' or specific function), allowed to choose ('auto'), or prevented ('none'). */
  toolChoice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } }

  /** Request the model to respond in a specific format (e.g., JSON). Support varies by provider/model. */
  responseFormat?: {
    type: 'text' | 'json_object'
    /** Optional JSON schema to guide the model's JSON output (currently informational, used in system prompt construction where applicable). */
    schema?: Record<string, unknown>
  }

  /** Request the model to provide citations or grounding for its response. Support varies by provider/model. */
  grounding?: {
    enabled: boolean
    /** Source for grounding (e.g., 'web' for Google Search). Provider-specific interpretation. */
    source?: 'web' | string[]
  }

  /** Request the model to output intermediate thinking steps (Anthropic specific). */
  thinking?: boolean

  // Internal flag, not set by user directly on top-level call
  /** @internal */
  stream?: boolean

  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions

  // Add other common parameters like 'presence_penalty', 'frequency_penalty' if needed
}

/**
 * Parameters for generating embeddings.
 */
export interface EmbedParams {
  /** The provider to use for this request. */
  provider: Provider
  /** The specific embedding model ID. Optional if a default is configured. */
  model?: string
  /** The input text(s) to embed. Can be a single string or an array for batching (if supported). */
  input: string | string[]
  /** Optional: Specify the desired output format (e.g., 'float', 'base64'). Support varies. */
  encodingFormat?: 'float' | 'base64' // Check provider specifics
  /** Optional: Desired dimension size for the output embeddings (OpenAI specific). */
  dimensions?: number
  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions
}

/**
 * Parameters for generating speech (Text-to-Speech).
 */
export interface SpeechParams {
  /** The provider to use for this request (currently primarily OpenAI). */
  provider: Provider.OpenAI // Restrict for now as it's the main provider with this
  /** The specific TTS model ID. Optional if a default is configured (e.g., 'tts-1'). */
  model?: string
  /** The text to synthesize into speech. */
  input: string
  /** The voice to use (provider-specific options, e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer' for OpenAI). */
  voice: string
  /** The desired audio output format (e.g., 'mp3', 'opus'). Defaults to 'mp3'. */
  responseFormat?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm' // Check OpenAI specifics
  /** Optional: Speed of the generated speech (0.25 to 4.0). Defaults to 1.0. */
  speed?: number
  /** Provider-specific options overriding global config for this call. */
  providerOptions?: ProviderOptions
}

/**
 * Base parameters for audio processing (Transcription/Translation).
 */
interface BaseAudioParams {
  /** The provider to use for this request (e.g., OpenAI, Groq). */
  provider: Provider
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
