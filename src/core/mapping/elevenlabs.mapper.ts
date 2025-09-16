import { Readable } from 'stream'
import { BaseCustomMapper } from './base.custom.mapper'
import { CustomProviderConfig } from '../../types/custom.types'
import {
  AudioStreamChunk,
  ProviderKey,
  SpeechParams,
  TranscribeParams,
  TranscriptionResult,
  RosettaVoiceList
} from '../../types'
import { ProviderAPIError, MappingError } from '../../errors'

// IMPORTANT: This relies on the official ElevenLabs engineering guide present in this repo.
// We will verify signatures against the installed SDK '@elevenlabs/elevenlabs-js' .d.ts after installation.
// eslint-disable-next-line @typescript-eslint/consistent-type-imports
import type { ElevenLabsClient as ElevenLabsClientType } from '@elevenlabs/elevenlabs-js'
import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js'
import type { TextToSpeechRequest } from '@elevenlabs/elevenlabs-js/api/resources/textToSpeech/client/requests/TextToSpeechRequest'
import type { StreamTextToSpeechRequest } from '@elevenlabs/elevenlabs-js/api/resources/textToSpeech/client/requests/StreamTextToSpeechRequest'
import type { BodySpeechToTextV1SpeechToTextPost } from '@elevenlabs/elevenlabs-js/api/resources/speechToText/client/requests'
import type { FileLike as ElevenFileLike } from '@elevenlabs/elevenlabs-js/core/file'
import { TextToSpeechConvertRequestOutputFormat } from '@elevenlabs/elevenlabs-js/api/resources/textToSpeech/types'
import type { SpeechToTextConvertResponse } from '@elevenlabs/elevenlabs-js/api/resources/speechToText/types'
import type { SpeechToTextChunkResponseModel } from '@elevenlabs/elevenlabs-js/api/types/SpeechToTextChunkResponseModel'
import type { MultichannelSpeechToTextResponseModel } from '@elevenlabs/elevenlabs-js/api/types/MultichannelSpeechToTextResponseModel'
import type { SpeechToTextWebhookResponseModel } from '@elevenlabs/elevenlabs-js/api/types/SpeechToTextWebhookResponseModel'
import type { GetVoicesResponse } from '@elevenlabs/elevenlabs-js/api/types/GetVoicesResponse'
import type { Voice as ElevenVoice } from '@elevenlabs/elevenlabs-js/api/types/Voice'

/**
 * Optional text normalization for TTS to improve pronunciation of numbers, currencies, etc.
 * Enabled by default. Can be disabled by setting custom provider config 'ttsNormalize' to false.
 * This implementation avoids external dependencies for SDK footprint minimization.
 */
function normalizeTextForTTS(text: string): string {
  let normalized = text

  // Currencies: $42.50 -> "42.50 dollars", £ -> "pounds", € -> "euros"
  normalized = normalized.replace(/([$£€])(\d[\d,.]*)/g, (_m, curr, amount) => {
    const currencyName = curr === '$' ? 'dollars' : curr === '£' ? 'pounds' : 'euros'
    return `${amount} ${currencyName}`
  })

  // Phone numbers: 555-555-5555 -> "5 5 5, 5 5 5, 5 5 5 5"
  normalized = normalized.replace(/(\d{3})-(\d{3})-(\d{4})/g, (_m, a, b, c) => {
    const spell = (s: string) => s.split('').join(' ')
    return `${spell(a)}, ${spell(b)}, ${spell(c)}`
  })

  // URLs: example.com/path -> "example dot com slash path"
  normalized = normalized.replace(/([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(\/[^\s]*)?/g, (_m, domain, pathPart) => {
    const spokenDomain = String(domain).replace(/\./g, ' dot ')
    const spokenPath = pathPart ? ` slash ${String(pathPart).slice(1).replace(/\//g, ' slash ')}` : ''
    return `${spokenDomain}${spokenPath}`
  })

  return normalized
}

/**
 * Map Rosetta's generic audio responseFormat to ElevenLabs expected output_format strings.
 * Conservative mapping based on documented examples in the engineering guide.
 * Unrecognized formats fall back to SDK defaults (no explicit output_format).
 */
function mapTtsOutputFormat(
  fmt: SpeechParams['responseFormat'] | undefined
): TextToSpeechConvertRequestOutputFormat | undefined {
  switch (fmt) {
    case 'mp3':
      return 'mp3_44100_128'
    case 'opus':
      return 'opus_48000_128'
    case 'wav':
      // Map WAV request to uncompressed PCM at 44.1kHz
      return 'pcm_44100'
    case 'pcm':
      // Default to telephony-friendly PCM if unspecified
      return 'pcm_16000'
    case 'aac':
    case 'flac':
      console.warn(`[elevenlabs] responseFormat '${fmt}' not supported by ElevenLabs outputFormat tokens; using SDK default.`)
      return undefined
    default:
      return undefined
  }
}

/** Type guards for stream variants returned by the ElevenLabs SDK */
function isNodeReadable(stream: any): stream is Readable {
  return !!stream && (typeof stream.read === 'function' || typeof stream.pipe === 'function' || typeof stream.on === 'function')
}
function isWebReadableStream(stream: any): stream is ReadableStream<Uint8Array> {
  return !!stream && typeof stream.getReader === 'function'
}

/** Iterate over both Node.js Readable and Web ReadableStream<Uint8Array> uniformly. */
async function* iterateStreamChunks(stream: unknown): AsyncGenerator<Uint8Array | string> {
  if (isNodeReadable(stream)) {
    for await (const chunk of stream as any) {
      yield chunk as any
    }
    return
  }
  if (isWebReadableStream(stream)) {
    const reader = (stream as ReadableStream<Uint8Array>).getReader()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (value) yield value
      }
    } finally {
      try {
        reader.releaseLock()
      } catch {}
    }
    return
  }
  throw new MappingError('Unsupported stream type returned by ElevenLabs SDK.', 'elevenlabs')
}

/** Collect any supported stream into a Buffer without blocking the event loop. */
async function collectStreamToBuffer(stream: unknown): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of iterateStreamChunks(stream)) {
    if (chunk instanceof Uint8Array) chunks.push(Buffer.from(chunk))
    else if (typeof chunk === 'string') chunks.push(Buffer.from(chunk))
    else throw new MappingError(`Unexpected audio stream chunk type: ${typeof chunk}`, 'elevenlabs')
  }
  return Buffer.concat(chunks)
}

/** Type guards for STT union response */
function isSttChunk(resp: SpeechToTextConvertResponse): resp is SpeechToTextChunkResponseModel {
  return typeof (resp as any)?.text === 'string' && Array.isArray((resp as any)?.words)
}
function isSttMultichannel(resp: SpeechToTextConvertResponse): resp is MultichannelSpeechToTextResponseModel {
  return Array.isArray((resp as any)?.transcripts)
}
function isSttWebhook(resp: SpeechToTextConvertResponse): resp is SpeechToTextWebhookResponseModel {
  return typeof (resp as any)?.message === 'string'
}

/**
 * ElevenLabs Custom Provider Mapper
 *
 * Implements TTS (buffer + streaming) and STT using the ElevenLabs engineering guide.
 * This class integrates with Rosetta's custom provider execution hooks.
 */
export class ElevenLabsMapper extends BaseCustomMapper {
  private _client?: ElevenLabsClientType

  constructor(config: CustomProviderConfig) {
    super(config)
  }

  /** Create or reuse the ElevenLabs client. Options verified against engineering guide. */
  private getClient(explicitApiKey?: string): ElevenLabsClientType {
    if (this._client) return this._client

    const apiKey =
      explicitApiKey ??
      this.config.apiKey ??
      process.env.ELEVENLABS_API_KEY

    if (!apiKey) {
      throw new ProviderAPIError('Missing ELEVENLABS_API_KEY for ElevenLabs provider.', this.provider)
    }

    // baseUrl is optional; engineering guide shows it for global preview
    const baseUrl = this.config.baseURL

    // The ElevenLabsClient will also read the env var automatically if no apiKey is provided.
    this._client = baseUrl ? new ElevenLabsClient({ apiKey, baseUrl }) : new ElevenLabsClient({ apiKey })

    return this._client
  }

  /** Non-streaming TTS: return a Buffer of synthesized audio. */
  override async executeGenerateSpeech(
    _mappedParams: TextToSpeechRequest,
    apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    originalParams: SpeechParams
  ): Promise<Buffer> {
    const client = this.getClient(apiKey)

    const voiceId = originalParams.voice
    if (!voiceId) {
      throw new MappingError('ElevenLabs TTS requires a voiceId (SpeechParams.voice).', this.provider)
    }

    const modelId = originalParams.model ?? this.config.defaultTtsModel
    if (!modelId) {
      throw new MappingError('TTS model_id is required (set SpeechParams.model or defaultTtsModel).', this.provider)
    }

    const output_format = mapTtsOutputFormat(originalParams.responseFormat)

    try {
      const shouldNormalize = originalParams?.ttsNormalize !== false
      const inputText = shouldNormalize ? normalizeTextForTTS(originalParams.input) : originalParams.input
      const ttsReq: TextToSpeechRequest = {
        text: inputText,
        modelId,
        ...(output_format ? { outputFormat: output_format } : {})
      }
      const stream = await client.textToSpeech.convert(voiceId, ttsReq)
      return await collectStreamToBuffer(stream)
    } catch (err) {
      throw this.wrapProviderError(err, this.provider)
    }
  }

  /** Streaming TTS: yield AudioStreamChunk events as bytes are produced. */
  override async *executeStreamSpeech(
    _mappedParams: StreamTextToSpeechRequest,
    apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    originalParams: SpeechParams
  ): AsyncIterable<AudioStreamChunk> {
    const client = this.getClient(apiKey)

    const voiceId = originalParams.voice
    if (!voiceId) {
      yield { type: 'error', data: { error: new MappingError('voiceId (SpeechParams.voice) is required.', this.provider) } }
      return
    }

    const modelId = originalParams.model ?? this.config.defaultTtsModel
    if (!modelId) {
      yield { type: 'error', data: { error: new MappingError('TTS model_id is required.', this.provider) } }
      return
    }

    let stream: unknown
    try {
      const shouldNormalize = originalParams?.ttsNormalize !== false
      const inputText = shouldNormalize ? normalizeTextForTTS(originalParams.input) : originalParams.input
      const streamReq: StreamTextToSpeechRequest = {
        text: inputText,
        modelId
      }
      stream = await client.textToSpeech.stream(voiceId, streamReq)
    } catch (e) {
      const wrapped = this.wrapProviderError(e, this.provider)
      yield { type: 'error', data: { error: wrapped } }
      return
    }

    try {
      for await (const chunk of iterateStreamChunks(stream)) {
        if (chunk instanceof Uint8Array) {
          yield { type: 'audio_chunk', data: Buffer.from(chunk) }
        } else if (typeof chunk === 'string') {
          yield { type: 'audio_chunk', data: Buffer.from(chunk) }
        } else {
          console.warn(`[${this.provider}] Unexpected stream chunk type: ${typeof chunk}`)
        }
      }
      yield { type: 'audio_stop' }
    } catch (e) {
      const wrapped = this.wrapProviderError(e, this.provider)
      yield { type: 'error', data: { error: wrapped } }
    }
  }

  /** Synchronous STT Transcription via speechToText.convert */
  override async executeTranscribe(
    _mappedParams: BodySpeechToTextV1SpeechToTextPost,
    apiKey: string | undefined,
    _providerConfig: CustomProviderConfig,
    originalParams: TranscribeParams
  ): Promise<TranscriptionResult> {
    const client = this.getClient(apiKey)

    const modelId = originalParams.model ?? this.config.defaultSttModel
    if (!modelId) {
      throw new MappingError('STT model_id is required (set TranscribeParams.model or defaultSttModel).', this.provider)
    }

    // Build ElevenLabs FileLike from RosettaAudioData (Buffer or Node Readable)
    const data = originalParams.audio.data
    let file: ElevenFileLike | undefined
    if (Buffer.isBuffer(data)) {
      file = data
    } else if (data instanceof Readable) {
      file = data
    }

    const sttReq: BodySpeechToTextV1SpeechToTextPost = {
      modelId,
      ...(file ? { file } : {}),
      ...(originalParams.language ? { languageCode: originalParams.language } : {})
    }

    // Only include optional flags if explicitly desired by upstream (avoid assumptions)
    // if (originalParams.timestampGranularities?.includes('word')) {
    //   request['diarize'] = true // Enable if word-level timestamps require diarization in your product context
    // }

    try {
      const response = await client.speechToText.convert(sttReq)

      let text: string
      let language: string | undefined
      let duration: number | undefined
      let segments: unknown[] | undefined
      let words: unknown[] | undefined

      if (isSttChunk(response)) {
        text = response.text
        language = response.languageCode
        words = response.words
      } else if (isSttMultichannel(response)) {
        text = response.transcripts.map(t => t.text).join(' ')
        language = response.transcripts[0]?.languageCode
        words = response.transcripts.flatMap(t => (Array.isArray(t.words) ? t.words : []))
      } else if (isSttWebhook(response)) {
        throw new MappingError(
          'ElevenLabs STT returned a webhook acknowledgement. Use synchronous conversion (webhook: false) for inline results.',
          this.provider
        )
      } else {
        throw new MappingError('Unexpected ElevenLabs STT response shape.', this.provider)
      }

      const result: TranscriptionResult = {
        text,
        language,
        duration,
        segments,
        words,
        model: modelId,
        rawResponse: response
      }
      return result
    } catch (e) {
      throw this.wrapProviderError(e, this.provider)
    }
  }

  /** Lists available voices for the authenticated ElevenLabs account. */
  override async executeListVoices(
    apiKey: string | undefined,
    _providerConfig: CustomProviderConfig
  ): Promise<RosettaVoiceList> {
    const client = this.getClient(apiKey)
    const resp: GetVoicesResponse = await client.voices.getAll({})
    const voices = (resp.voices ?? []) as ElevenVoice[]

    const data = voices.map(v => ({
      id: v.voiceId,
      name: v.name,
      labels: v.labels,
      category: v.category ? String(v.category) : undefined,
      description: v.description,
      previewUrl: v.previewUrl,
      settings: v.settings,
      owned: v.isOwner,
      provider: this.provider,
      rawData: v
    }))

    return { object: 'list', data }
  }

  /** Optionally override error parsing for ElevenLabs error envelopes */
  override wrapProviderError(error: unknown, provider: ProviderKey) {
    // Inspect common error statuses documented in the engineering guide (401/429 etc.) if available on the error object.
    // Default to base implementation for consistent normalization.
    return super.wrapProviderError(error, provider)
  }
}
