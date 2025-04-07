// Mappers for Groq Audio features (STT, Translate)

import Groq from 'groq-sdk'
import { Uploadable as GroqUploadable } from 'groq-sdk/core'
import { TranscribeParams, TranslateParams, TranscriptionResult } from '../../types'
import { safeGet } from '../utils'

// --- Parameter Mapping ---

export function mapToGroqSttParams(
  params: TranscribeParams,
  file: GroqUploadable
): Groq.Audio.TranscriptionCreateParams {
  if (params.timestampGranularities && params.timestampGranularities.length > 0) {
    console.warn("Groq provider does not support 'timestampGranularities'. Parameter ignored.")
  }
  // Groq supports: json, text, srt, verbose_json, vtt
  // Groq may support srt, vtt. Check this later. For now, we'll use json, text, verbose_json
  const supportedFormats: Groq.Audio.TranscriptionCreateParams['response_format'][] = ['json', 'text', 'verbose_json']
  let responseFormat = params.responseFormat ?? 'json' // Apply default if undefined
  if (responseFormat && !supportedFormats.includes(responseFormat as any)) {
    console.warn(
      `Groq STT format '${responseFormat}' not directly supported or recognized. Supported: ${supportedFormats.join(
        ', '
      )}. Defaulting to 'json'.`
    )
    responseFormat = 'json'
  }
  return {
    model: params.model!,
    file: file,
    language: params.language,
    prompt: params.prompt,
    response_format: responseFormat as Groq.Audio.TranscriptionCreateParams['response_format'],
    temperature: undefined // Groq STT doesn't support temperature param in current types
  }
}

export function mapToGroqTranslateParams(
  params: TranslateParams,
  file: GroqUploadable
): Groq.Audio.TranslationCreateParams {
  const supportedFormats: Groq.Audio.TranslationCreateParams['response_format'][] = ['json', 'text', 'verbose_json']
  let responseFormat = params.responseFormat ?? 'json' // Apply default if undefined
  if (responseFormat && !supportedFormats.includes(responseFormat as any)) {
    console.warn(
      `Groq Translate format '${responseFormat}' not directly supported or recognized. Supported: ${supportedFormats.join(
        ', '
      )}. Defaulting to 'json'.`
    )
    responseFormat = 'json'
  }
  return {
    model: params.model!,
    file: file,
    prompt: params.prompt,
    response_format: responseFormat as Groq.Audio.TranslationCreateParams['response_format'],
    temperature: undefined // Groq Translate also doesn't support temperature
  }
}

// --- Result Mapping ---

// Helper to extract text, handling potential string or object responses
function extractTextFromGroqAudioResponse(response: Groq.Audio.Transcription | Groq.Audio.Translation): string {
  if (response === null) {
    console.warn('Received null audio response from Groq.')
    return '[Unparsable Response]' // Return specific string for null
  }
  if (typeof response === 'string') {
    return response
  } else if (
    typeof response === 'object' &&
    response !== null &&
    'text' in response &&
    typeof response.text === 'string'
  ) {
    return response.text
  } else {
    // If it's not string or {text: string}, it might be SRT/VTT string, or unexpected JSON.
    // Try String() conversion as a fallback.
    console.warn('Received non-standard audio response format from Groq, attempting String() conversion:', response)
    try {
      return String(response)
    } catch (e) {
      console.error('Error converting Groq audio response to string:', e)
      return '[Unparsable Response]' // Return specific string on conversion error
    }
  }
}

export function mapFromGroqTranscriptionResponse(
  response: Groq.Audio.Transcription,
  model: string
): TranscriptionResult {
  const textResult = extractTextFromGroqAudioResponse(response)
  // If the response was verbose_json, try to extract details safely
  const verboseResponse = typeof response === 'object' && response !== null ? (response as any) : {}

  return {
    text: textResult,
    // Safely access potential properties from verbose_json format
    language: safeGet<string>(verboseResponse, 'language'),
    duration: safeGet<number>(verboseResponse, 'duration'),
    segments: safeGet<unknown[]>(verboseResponse, 'segments'),
    words: safeGet<unknown[]>(verboseResponse, 'words'),
    model: model,
    rawResponse: response
  }
}

export function mapFromGroqTranslationResponse(response: Groq.Audio.Translation, model: string): TranscriptionResult {
  const textResult = extractTextFromGroqAudioResponse(response)
  // Translation response might also be verbose, allow extracting details if present
  const verboseResponse = typeof response === 'object' && response !== null ? (response as any) : {}

  // Translation typically doesn't return language (it's implicitly English)
  // or detailed segments/words unless verbose_json was requested AND supported.
  return {
    text: textResult,
    language: safeGet<string>(verboseResponse, 'language'), // Might be null/undefined
    duration: safeGet<number>(verboseResponse, 'duration'),
    segments: safeGet<unknown[]>(verboseResponse, 'segments'),
    words: safeGet<unknown[]>(verboseResponse, 'words'),
    model: model,
    rawResponse: response
  }
}
