import {
  mapToGroqSttParams,
  mapToGroqTranslateParams,
  mapFromGroqTranscriptionResponse,
  mapFromGroqTranslationResponse
} from '../../../../src/core/mapping/groq.audio.mapper'
import { TranscribeParams, TranslateParams, TranscriptionResult, Provider } from '../../../../src/types'
import Groq from 'groq-sdk'
import { Uploadable } from 'groq-sdk/core'

// Mock Uploadable for audio tests
const mockAudioFile: Uploadable = {
  [Symbol.toStringTag]: 'File',
  name: 'mock.mp3'
} as any // Cast to bypass full FileLike implementation details

describe('Groq Audio Mapper', () => {
  const model = 'whisper-large-v3'
  const baseAudioData = {
    data: Buffer.from(''),
    filename: 'a.mp3',
    mimeType: 'audio/mpeg' as const
  }

  describe('mapToGroqSttParams', () => {
    const baseParams: TranscribeParams = {
      provider: Provider.Groq,
      model: model,
      audio: baseAudioData
    }

    it('[Easy] should map basic parameters', () => {
      const result = mapToGroqSttParams(baseParams, mockAudioFile)
      expect(result.model).toBe(model)
      expect(result.file).toBe(mockAudioFile)
      expect(result.language).toBeUndefined()
      expect(result.prompt).toBeUndefined()
      expect(result.response_format).toBe('json') // Default
      expect(result.temperature).toBeUndefined()
    })

    it('[Easy] should map language and prompt', () => {
      const params: TranscribeParams = {
        ...baseParams,
        language: 'fr',
        prompt: 'Context'
      }
      const result = mapToGroqSttParams(params, mockAudioFile)
      expect(result.language).toBe('fr')
      expect(result.prompt).toBe('Context')
    })

    it('[Easy] should map responseFormat: text', () => {
      const params: TranscribeParams = {
        ...baseParams,
        responseFormat: 'text'
      }
      const result = mapToGroqSttParams(params, mockAudioFile)
      expect(result.response_format).toBe('text')
    })

    it('[Easy] should map responseFormat: verbose_json', () => {
      const params: TranscribeParams = {
        ...baseParams,
        responseFormat: 'verbose_json'
      }
      const result = mapToGroqSttParams(params, mockAudioFile)
      expect(result.response_format).toBe('verbose_json')
    })

    it('[Medium] should warn and default for unsupported responseFormat', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: TranscribeParams = { ...baseParams, responseFormat: 'srt' }
      const result = mapToGroqSttParams(params, mockAudioFile)
      expect(result.response_format).toBe('json')
      expect(warnSpy).toHaveBeenCalledWith(
        "Groq STT format 'srt' not directly supported or recognized. Supported: json, text, verbose_json. Defaulting to 'json'."
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should warn for timestampGranularities', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: TranscribeParams = {
        ...baseParams,
        timestampGranularities: ['word']
      }
      mapToGroqSttParams(params, mockAudioFile)
      expect(warnSpy).toHaveBeenCalledWith(
        "Groq provider does not support 'timestampGranularities'. Parameter ignored."
      )
      warnSpy.mockRestore()
    })
  })

  describe('mapToGroqTranslateParams', () => {
    const baseParams: TranslateParams = {
      provider: Provider.Groq,
      model: model,
      audio: baseAudioData
    }

    it('[Easy] should map basic parameters', () => {
      const result = mapToGroqTranslateParams(baseParams, mockAudioFile)
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
      const result = mapToGroqTranslateParams(params, mockAudioFile)
      expect(result.prompt).toBe('Translate context')
    })

    it('[Easy] should map responseFormat: text', () => {
      const params: TranslateParams = { ...baseParams, responseFormat: 'text' }
      const result = mapToGroqTranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('text')
    })

    it('[Easy] should map responseFormat: verbose_json', () => {
      const params: TranslateParams = {
        ...baseParams,
        responseFormat: 'verbose_json'
      }
      const result = mapToGroqTranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('verbose_json')
    })

    it('[Medium] should warn and default for unsupported responseFormat', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: TranslateParams = { ...baseParams, responseFormat: 'vtt' }
      const result = mapToGroqTranslateParams(params, mockAudioFile)
      expect(result.response_format).toBe('json')
      expect(warnSpy).toHaveBeenCalledWith(
        "Groq Translate format 'vtt' not directly supported or recognized. Supported: json, text, verbose_json. Defaulting to 'json'."
      )
      warnSpy.mockRestore()
    })
  })

  describe('mapFromGroqTranscriptionResponse', () => {
    it('[Easy] should map basic text string response', () => {
      const response = 'This is the transcription text.' as any
      const result = mapFromGroqTranscriptionResponse(response, model)
      expect(result.text).toBe('This is the transcription text.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('should map basic JSON response (only text)', () => {
      const response: Groq.Audio.Transcription = {
        text: 'Transcription from JSON.'
      }
      const result = mapFromGroqTranscriptionResponse(response, model)
      expect(result.text).toBe('Transcription from JSON.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('should map verbose JSON response', () => {
      const response: Groq.Audio.Transcription = {
        text: 'Verbose transcription.',
        language: 'en',
        duration: 12.34,
        segments: [{ id: 1, text: 'Segment 1' }],
        words: [{ word: 'Verbose', start: 0.1, end: 0.5 }]
      } as any // Cast if type doesn't match perfectly
      const result = mapFromGroqTranscriptionResponse(response, model)
      expect(result.text).toBe('Verbose transcription.')
      expect(result.language).toBe('en')
      expect(result.duration).toBe(12.34)
      expect(result.segments).toEqual([{ id: 1, text: 'Segment 1' }])
      expect(result.words).toEqual([{ word: 'Verbose', start: 0.1, end: 0.5 }])
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('[Medium] should map verbose JSON response with missing optional fields', () => {
      const response: Groq.Audio.Transcription = {
        text: 'Missing fields.',
        language: 'fr'
        // duration, segments, words are missing
      } as any
      const result = mapFromGroqTranscriptionResponse(response, model)
      expect(result.text).toBe('Missing fields.')
      expect(result.language).toBe('fr')
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
    })

    it('[Hard] should handle unexpected response format (null)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response = null as any
      const result = mapFromGroqTranscriptionResponse(response, model)
      // FIX: Expect the specific fallback string for null
      expect(result.text).toBe('[Unparsable Response]')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(warnSpy).toHaveBeenCalledWith('Received null audio response from Groq.')
      warnSpy.mockRestore()
    })

    it('[Hard] should handle unexpected response format (number)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response = 42 as any
      const result = mapFromGroqTranscriptionResponse(response, model)
      // FIX: Expect the string representation of the number
      expect(result.text).toBe('42')
      expect(warnSpy).toHaveBeenCalledWith(
        'Received unexpected audio response format from Groq, attempting String() conversion:',
        42
      )
      warnSpy.mockRestore()
    })
  })

  describe('mapFromGroqTranslationResponse', () => {
    it('[Easy] should map basic text string response', () => {
      const response = 'This is the translation text.' as any
      const result = mapFromGroqTranslationResponse(response, model)
      expect(result.text).toBe('This is the translation text.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('should map basic JSON response (only text)', () => {
      const response: Groq.Audio.Translation = {
        text: 'Translation from JSON.'
      }
      const result = mapFromGroqTranslationResponse(response, model)
      expect(result.text).toBe('Translation from JSON.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('should map verbose JSON response', () => {
      const response: Groq.Audio.Translation = {
        text: 'Verbose translation.',
        // language: 'en', // Typically not present in translation
        duration: 5.67,
        segments: [{ id: 0, text: 'Segment A' }],
        words: [{ word: 'Verbose', start: 0.2, end: 0.6 }]
      } as any // Cast if type doesn't match perfectly
      const result = mapFromGroqTranslationResponse(response, model)
      expect(result.text).toBe('Verbose translation.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBe(5.67)
      expect(result.segments).toEqual([{ id: 0, text: 'Segment A' }])
      expect(result.words).toEqual([{ word: 'Verbose', start: 0.2, end: 0.6 }])
      expect(result.model).toBe(model)
      expect(result.rawResponse).toBe(response)
    })

    it('[Medium] should map verbose JSON response with missing optional fields', () => {
      const response: Groq.Audio.Translation = {
        text: 'Missing fields translation.',
        duration: 7.89
        // language, segments, words are missing
      } as any
      const result = mapFromGroqTranslationResponse(response, model)
      expect(result.text).toBe('Missing fields translation.')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBe(7.89)
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
    })

    it('[Hard] should handle unexpected response format (array)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response = ['unexpected'] as any
      const result = mapFromGroqTranslationResponse(response, model)
      // FIX: Expect the standard string representation of the array
      expect(result.text).toBe('unexpected')
      expect(result.language).toBeUndefined()
      expect(result.duration).toBeUndefined()
      expect(result.segments).toBeUndefined()
      expect(result.words).toBeUndefined()
      expect(result.model).toBe(model)
      expect(
        warnSpy
      ).toHaveBeenCalledWith('Received unexpected audio response format from Groq, attempting String() conversion:', [
        'unexpected'
      ])
      warnSpy.mockRestore()
    })

    it('[Hard] should handle unexpected response format (object without text)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response = { some_other_field: 'value' } as any
      const result = mapFromGroqTranslationResponse(response, model)
      // FIX: Expect the standard string representation of the object
      expect(result.text).toBe('[object Object]')
      expect(warnSpy).toHaveBeenCalledWith(
        'Received unexpected audio response format from Groq, attempting String() conversion:',
        {
          some_other_field: 'value'
        }
      )
      warnSpy.mockRestore()
    })
  })
})
