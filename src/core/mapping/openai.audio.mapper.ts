// Mappers for OpenAI Audio features (Transcription, Translation)

import OpenAI from 'openai'
import { TranscriptionResult, Provider } from '../../types'
import { MappingError } from '../../errors'
import { safeGet } from '../utils'

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
