import OpenAI from 'openai'
import { Uploadable } from 'openai/uploads'
import {
  mapToOpenAITranscribeParams,
  mapToOpenAITranslateParams,
  mapFromOpenAITranscriptionResponse,
  mapFromOpenAITranslationResponse
} from '../../../../src/core/mapping/openai.audio.mapper'
import { TranscribeParams, TranslateParams, TranscriptionResult, Provider } from '../../../../src/types'
import { MappingError } from '../../../../src/errors'

// Mock Uploadable for audio tests
const mockAudioFile: Uploadable = {
  [Symbol.toStringTag]: 'File',
  name: 'mock.mp3'
} as any // Cast to bypass full FileLike implementation details

describe('OpenAI Audio Mapper', () => {
  const model = 'whisper-1'
  const baseAudioData = {
    data: Buffer.from(''),
    filename: 'a.mp3',
    mimeType: 'audio/mpeg' as const
  }

  describe('mapToOpenAITranscribeParams', () => {
    const baseParams: TranscribeParams = {
      provider: Provider.OpenAI,
      model: model,
      audio: baseAudioData
    }

    it('[Easy] should map basic parameters', () => {
      const result = mapToOpenAITranscribeParams(baseParams, mockAudioFile)
      expect(result.model).toBe(model)
      expect(result.file).toBe(mockAudioFile)
      expect(result.language).toBeUndefined()
      expect(result.prompt).toBeUndefined()
      expect(result.response_format).toBe('json') // Default
      expect(result.temperature).toBeUndefined()
      expect(result.timestamp_granularities).toBeUndefined()
    })

    it('[Easy] should map language and prompt', () => {
      const params: TranscribeParams = {
        ...baseParams,
        language: 'fr',
        prompt: 'Context'
      }
      const result = mapToOpenAITranscribeParams(params, mockAudioFile)
      expect(result.language).toBe('fr')
      expect(result.prompt).toBe('Context')
    })

    it('[Easy] should map responseFormat: text', () => {
      const params: TranscribeParams = {
        ...baseParams,
        responseFormat: 'text'
      }
      const result = mapToOpenAITranscribeParams(params, mockAudioFile)
      expect(result.response_format).toBe('text')
    })

    it('[Easy] should map responseFormat: verbose_json', () => {
      const params: TranscribeParams = {
        ...baseParams,
        responseFormat: 'verbose_json'
      }
      const result = mapToOpenAITranscribeParams(params, mockAudioFile)
      expect(result.response_format).toBe('verbose_json')
    })

    it('[Easy] should map timestampGranularities', () => {
      const params: TranscribeParams = {
        ...baseParams,
        timestampGranularities: ['word', 'segment']
      }
      const result = mapToOpenAITranscribeParams(params, mockAudioFile)
      expect(result.timestamp_granularities).toEqual(['word', 'segment'])
    })

    it('[Medium] should warn and default for unsupported responseFormat', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: TranscribeParams = { ...baseParams, responseFormat: 'unsupported' as any }
      const result = mapToOpenAITranscribeParams(params, mockAudioFile)
      expect(result.response_format).toBe('json') // Defaults to json
      expect(warnSpy).toHaveBeenCalledWith(
        "OpenAI STT format 'unsupported' not directly supported or recognized. Supported: json, text, srt, verbose_json, vtt. Defaulting to 'json'."
      )
      warnSpy.mockRestore()
    })
  })

  describe('mapToOpenAITranslateParams', () => {
    const baseParams: TranslateParams = {
      provider: Provider.OpenAI,
      model: model,
      audio: baseAudioData
    }

    it('[Easy] should map basic parameters', () => {
      const result = mapToOpenAITranslateParams(baseParams, mockAudioFile)
      expect(result.model).toBe(model)
      expect(result.file).toBe(mockAudioFile)
      expect(result.prompt).toBeUndefined()
      expect(result.response_format).toBe('json') // Default
      expect(result.temperature).toBeUndefined()
    })

    it('[Easy] should map prompt', () => {
      const params: TranslateParams = {
        ...baseParams,
        prompt: 'Translate context'
      }
      const result = mapToOpenAITranslateParams(params, mockAudioFile)
      expect(result.prompt).toBe('Translate context')
    })

    it('[Easy] should map responseFormat: text', () => {
      const params: TranslateParams = { ...baseParams, responseFormat: 'text' }
      const result = mapToOpenAITranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('text')
    })

    it('[Easy] should map responseFormat: verbose_json', () => {
      const params: TranslateParams = {
        ...baseParams,
        responseFormat: 'verbose_json'
      }
      const result = mapToOpenAITranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('verbose_json')
    })

    it('[Medium] should warn and default for unsupported responseFormat', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: TranslateParams = { ...baseParams, responseFormat: 'invalid' as any }
      const result = mapToOpenAITranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('json') // Defaults to json
      expect(warnSpy).toHaveBeenCalledWith(
        "OpenAI Translate format 'invalid' not directly supported or recognized. Supported: json, text, srt, verbose_json, vtt. Defaulting to 'json'."
      )
      warnSpy.mockRestore()
    })
  })

  describe('mapFromOpenAITranscriptionResponse', () => {
    const modelUsed = 'whisper-1-test'

    it('[Easy] should map basic transcription response (text only)', () => {
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

    it('[Easy] should map verbose transcription response', () => {
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
      } as any
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

    it('[Medium] should throw MappingError if text is missing', () => {
      const invalidResponse = {} as any
      expect(() => mapFromOpenAITranscriptionResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAITranscriptionResponse(invalidResponse, modelUsed)).toThrow(
        'Unexpected transcription response format from OpenAI. Missing text.'
      )
    })

    it('[Medium] should handle verbose response with missing optional fields', () => {
      const response = {
        text: 'Missing fields text.',
        language: 'fr'
        // Missing duration, segments, words
      } as any
      const result = mapFromOpenAITranscriptionResponse(response, modelUsed)
      expect(result.text).toBe('Missing fields text.')
      expect(result.language).toBe('fr')
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
    })
  })

  describe('mapFromOpenAITranslationResponse', () => {
    const modelUsed = 'whisper-1-test-trans'

    it('[Easy] should map basic translation response (text only)', () => {
      const response: OpenAI.Audio.Translations.Translation = {
        text: 'Translated text.'
      }
      const result: TranscriptionResult = mapFromOpenAITranslationResponse(response, modelUsed)

      expect(result.text).toBe('Translated text.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(modelUsed)
      expect(result.rawResponse).toBe(response)
    })

    it('[Easy] should map verbose translation response', () => {
      const response: OpenAI.Audio.Translations.Translation = {
        text: 'Verbose translated text.',
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
      } as any
      const result: TranscriptionResult = mapFromOpenAITranslationResponse(response, modelUsed)

      expect(result.text).toBe('Verbose translated text.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBe(6.0)
      expect(result.segments).toBeDefined()
      expect(result.segments).toHaveLength(1)
      expect(result.words).toBeDefined()
      expect(result.words).toHaveLength(3)
      expect(result.model).toBe(modelUsed)
    })

    it('[Medium] should throw MappingError if text is missing', () => {
      const invalidResponse = {} as any
      expect(() => mapFromOpenAITranslationResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromOpenAITranslationResponse(invalidResponse, modelUsed)).toThrow(
        'Unexpected translation response format from OpenAI. Missing text.'
      )
    })

    it('[Medium] should handle verbose response with missing optional fields', () => {
      const response = {
        text: 'Missing fields translation.',
        duration: 7.89
        // Missing language, segments, words
      } as any
      const result = mapFromOpenAITranslationResponse(response, modelUsed)
      expect(result.text).toBe('Missing fields translation.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBe(7.89)
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
    })
  })
})
