// Mappers for OpenAI Audio features (Transcription, Translation)

import OpenAI from 'openai'
import { Uploadable as OpenAIUploadable } from 'openai/uploads'
import { TranscriptionResult, Provider, TranscribeParams, TranslateParams } from '../../types'
import { MappingError } from '../../errors'
import { safeGet } from '../utils'

// --- Parameter Mapping ---

export function mapToOpenAITranscribeParams(
  params: TranscribeParams,
  file: OpenAIUploadable
): OpenAI.Audio.TranscriptionCreateParams {
  // OpenAI uses standard model names for Whisper.
  if (params.timestampGranularities && params.timestampGranularities.length > 0) {
    // Timestamp granularities are supported by OpenAI
  }
  // OpenAI supports: json, text, srt, verbose_json, vtt
  const supportedFormats: OpenAI.Audio.TranscriptionCreateParams['response_format'][] = [
    'json',
    'text',
    'srt',
    'verbose_json',
    'vtt'
  ]
  let responseFormat = params.responseFormat ?? 'json'
  if (responseFormat && !supportedFormats.includes(responseFormat as any)) {
    console.warn(
      `OpenAI STT format '${responseFormat}' not directly supported or recognized. Supported: ${supportedFormats.join(
        ', '
      )}. Defaulting to 'json'.`
    )
    responseFormat = 'json'
  }

  return {
    model: params.model!, // e.g., 'whisper-1'
    file: file,
    language: params.language,
    prompt: params.prompt,
    response_format: responseFormat as OpenAI.Audio.TranscriptionCreateParams['response_format'],
    temperature: undefined, // Temperature not typically supported in OpenAI STT
    timestamp_granularities: params.timestampGranularities as ('word' | 'segment')[] | undefined
  }
}

export function mapToOpenAITranslateParams(
  params: TranslateParams,
  file: OpenAIUploadable
): OpenAI.Audio.TranslationCreateParams {
  // OpenAI uses standard model names for Whisper translation.
  // OpenAI supports: json, text, srt, verbose_json, vtt
  const supportedFormats: OpenAI.Audio.TranslationCreateParams['response_format'][] = [
    'json',
    'text',
    'srt',
    'verbose_json',
    'vtt'
  ]
  let responseFormat = params.responseFormat ?? 'json'
  if (responseFormat && !supportedFormats.includes(responseFormat as any)) {
    console.warn(
      `OpenAI Translate format '${responseFormat}' not directly supported or recognized. Supported: ${supportedFormats.join(
        ', '
      )}. Defaulting to 'json'.`
    )
    responseFormat = 'json'
  }
  return {
    model: params.model!, // e.g., 'whisper-1'
    file: file,
    prompt: params.prompt,
    response_format: responseFormat as OpenAI.Audio.TranslationCreateParams['response_format'],
    temperature: undefined // Temperature not typically supported
  }
}

// --- Result Mapping ---

export function mapFromOpenAITranscriptionResponse(
  response: OpenAI.Audio.Transcriptions.Transcription,
  modelUsed: string
): TranscriptionResult {
  // OpenAI Transcription response structure is generally { text: string, ...optional_verbose_fields }
  if (typeof response.text !== 'string') {
    throw new MappingError('Unexpected transcription response format from OpenAI. Missing text.', Provider.OpenAI)
  }
  // Use safeGet for optional fields which might be present in verbose_json response format
  const language = safeGet<string>(response, 'language')
  const duration = safeGet<number>(response, 'duration')
  const segments = safeGet<unknown[]>(response, 'segments') // Type is complex, keep as unknown[]
  const words = safeGet<unknown[]>(response, 'words') // Type is complex, keep as unknown[]

  return {
    text: response.text,
    language,
    duration,
    segments,
    words,
    model: modelUsed, // Model isn't part of the response object itself
    rawResponse: response
  }
}

export function mapFromOpenAITranslationResponse(
  response: OpenAI.Audio.Translations.Translation,
  modelUsed: string
): TranscriptionResult {
  // OpenAI Translation response structure is similar: { text: string, ...optional_verbose_fields }
  if (typeof response.text !== 'string') {
    throw new MappingError('Unexpected translation response format from OpenAI. Missing text.', Provider.OpenAI)
  }
  // Verbose fields might also be present for translations
  const language = safeGet<string>(response, 'language') // Usually null/undefined for translations
  const duration = safeGet<number>(response, 'duration')
  const segments = safeGet<unknown[]>(response, 'segments')
  const words = safeGet<unknown[]>(response, 'words')

  return {
    text: response.text,
    language, // Often not relevant/present for translation to English
    duration,
    segments,
    words,
    model: modelUsed, // Model isn't part of the response object
    rawResponse: response
  }
}
