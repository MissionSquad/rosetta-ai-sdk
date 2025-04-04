import {
  mapFromOpenAITranscriptionResponse,
  mapFromOpenAITranslationResponse
} from '../../../../src/core/mapping/openai.audio.mapper'
import { TranscriptionResult } from '../../../../src/types'
import { MappingError } from '../../../../src/errors'
import OpenAI from 'openai'

describe('OpenAI Audio Mapper', () => {
  const modelUsed = 'whisper-1-test'

  describe('mapFromOpenAITranscriptionResponse', () => {
    it('should map basic transcription response (text only)', () => {
      const response: OpenAI.Audio.Transcriptions.Transcription = {
        text: 'Hello world.'
      }
      const result: TranscriptionResult = mapFromOpenAITranscriptionResponse(response, modelUsed)

      expect(result.text).toBe('Hello world.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(modelUsed)
      expect(result.rawResponse).toBe(response)
    })

    it('should map verbose transcription response', () => {
      const response: OpenAI.Audio.Transcriptions.Transcription = {
        text: 'Verbose response.',
        language: 'en',
        duration: 5.12,
        segments: [
          {
            id: 0,
            seek: 0,
            start: 0.0,
            end: 5.12,
            text: 'Verbose response.',
            tokens: [123],
            temperature: 0,
            avg_logprob: -0.5,
            compression_ratio: 1,
            no_speech_prob: 0.1
          }
        ],
        words: [
          { word: 'Verbose', start: 0.1, end: 0.5 },
          { word: 'response.', start: 0.6, end: 1.2 }
        ]
      } as any // Cast needed as SDK type might not fully represent verbose_json
      const result: TranscriptionResult = mapFromOpenAITranscriptionResponse(response, modelUsed)

      expect(result.text).toBe('Verbose response.')
      expect(result.language).toBe('en')
      expect(result.duration).toBe(5.12)
      expect(result.segments).toBeDefined()
      expect(result.segments).toHaveLength(1)
      expect(result.words).toBeDefined()
      expect(result.words).toHaveLength(2)
      expect(result.model).toBe(modelUsed)
    })

    it('should throw MappingError if text is missing', () => {
      const invalidResponse = {} as any // Missing text
      expect(() => mapFromOpenAITranscriptionResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAITranscriptionResponse(invalidResponse, modelUsed)).toThrow(
        'Unexpected transcription response format from OpenAI. Missing text.'
      )
    })
  })

  describe('mapFromOpenAITranslationResponse', () => {
    it('should map basic translation response (text only)', () => {
      const response: OpenAI.Audio.Translations.Translation = {
        text: 'Translated text.'
      }
      const result: TranscriptionResult = mapFromOpenAITranslationResponse(response, modelUsed)

      expect(result.text).toBe('Translated text.')
      expect(result.language).toBeUndefined() // Language usually not present/relevant
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(modelUsed)
      expect(result.rawResponse).toBe(response)
    })

    it('should map verbose translation response', () => {
      // Structure is similar to transcription verbose response
      const response: OpenAI.Audio.Translations.Translation = {
        text: 'Verbose translated text.',
        // language: 'en', // Language might be present but often not meaningful for translation to English
        duration: 6.0,
        segments: [
          {
            id: 0,
            seek: 0,
            start: 0.0,
            end: 6.0,
            text: 'Verbose translated text.',
            tokens: [456],
            temperature: 0,
            avg_logprob: -0.4,
            compression_ratio: 1.1,
            no_speech_prob: 0.05
          }
        ],
        words: [
          { word: 'Verbose', start: 0.2, end: 0.6 },
          { word: 'translated', start: 0.7, end: 1.5 },
          { word: 'text.', start: 1.6, end: 2.0 }
        ]
      } as any // Cast needed as SDK type might not fully represent verbose_json
      const result: TranscriptionResult = mapFromOpenAITranslationResponse(response, modelUsed)

      expect(result.text).toBe('Verbose translated text.')
      expect(result.language).toBeUndefined() // Explicitly check if language is undefined or handle if present
      expect(result.duration).toBe(6.0)
      expect(result.segments).toBeDefined()
      expect(result.segments).toHaveLength(1)
      expect(result.words).toBeDefined()
      expect(result.words).toHaveLength(3)
      expect(result.model).toBe(modelUsed)
    })

    it('should throw MappingError if text is missing', () => {
      const invalidResponse = {} as any // Missing text
      expect(() => mapFromOpenAITranslationResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAITranslationResponse(invalidResponse, modelUsed)).toThrow(
        'Unexpected translation response format from OpenAI. Missing text.'
      )
    })
  })
})
