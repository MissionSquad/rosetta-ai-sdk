import { mapTokenUsage, mapBaseParams, mapBaseToolChoice } from '../../../../src/core/mapping/common.utils'
import { GenerateParams, TokenUsage } from '../../../../src/types'

describe('Common Mapping Utilities', () => {
  describe('mapTokenUsage', () => {
    it('[Easy] should map usage with prompt_tokens and completion_tokens', () => {
      const providerUsage = { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 }
      const expected: TokenUsage = { promptTokens: 10, completionTokens: 20, totalTokens: 30 }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })

    it('[Easy] should map usage with input_tokens and output_tokens', () => {
      const providerUsage = { input_tokens: 15, output_tokens: 25, total_tokens: 40 }
      const expected: TokenUsage = { promptTokens: 15, completionTokens: 25, totalTokens: 40 }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })

    it('[Easy] should map usage with candidatesTokenCount (Google Generate)', () => {
      const providerUsage = { promptTokenCount: 5, candidatesTokenCount: 10, totalTokenCount: 15 }
      const expected: TokenUsage = { promptTokens: 5, completionTokens: 10, totalTokens: 15 }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })

    it('[Easy] should map usage with totalTokenCount (Google Embed)', () => {
      const providerUsage = { totalTokenCount: 12 }
      const expected: TokenUsage = {
        promptTokens: undefined,
        completionTokens: undefined,
        totalTokens: 12,
        cachedContentTokenCount: undefined
      }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })

    it('[Easy] should map usage with cachedContentTokenCount (Google)', () => {
      const providerUsage = {
        promptTokenCount: 5,
        candidatesTokenCount: 10,
        totalTokenCount: 15,
        cachedContentTokenCount: 3
      }
      const expected: TokenUsage = {
        promptTokens: 5,
        completionTokens: 10,
        totalTokens: 15,
        cachedContentTokenCount: 3
      }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })

    it('[Easy] should return undefined for null or undefined input', () => {
      expect(mapTokenUsage(null)).toBeUndefined()
      expect(mapTokenUsage(undefined)).toBeUndefined()
    })

    it('[Easy] should return undefined for empty object input', () => {
      expect(mapTokenUsage({})).toBeUndefined()
    })

    it('[Easy] should handle missing optional fields gracefully', () => {
      const providerUsage = { prompt_tokens: 10, total_tokens: 10 } // Missing completion_tokens
      const expected: TokenUsage = { promptTokens: 10, completionTokens: undefined, totalTokens: 10 }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)

      const providerUsage2 = { completion_tokens: 5, total_tokens: 5 } // Missing prompt_tokens
      const expected2: TokenUsage = { promptTokens: undefined, completionTokens: 5, totalTokens: 5 }
      expect(mapTokenUsage(providerUsage2)).toEqual(expected2)

      const providerUsage3 = { prompt_tokens: 7, completion_tokens: 8 } // Missing total_tokens
      const expected3: TokenUsage = { promptTokens: 7, completionTokens: 8, totalTokens: 15 }
      expect(mapTokenUsage(providerUsage3)).toEqual(expected3)
    })

    it('[Medium] should prioritize specific field names (e.g., prompt_tokens over input_tokens if both exist)', () => {
      const providerUsage = { prompt_tokens: 11, input_tokens: 99, completion_tokens: 22, total_tokens: 33 }
      const expected: TokenUsage = { promptTokens: 11, completionTokens: 22, totalTokens: 33 }
      expect(mapTokenUsage(providerUsage)).toEqual(expected)
    })
  })

  describe('mapBaseParams', () => {
    const baseGenerateParams: GenerateParams = {
      provider: 'openai' as any, // Provider doesn't matter for this test
      messages: []
    }

    it('[Easy] should map temperature, topP, maxTokens', () => {
      const params: GenerateParams = { ...baseGenerateParams, temperature: 0.7, topP: 0.8, maxTokens: 100 }
      const expected = { temperature: 0.7, topP: 0.8, maxTokens: 100, stopSequences: undefined }
      expect(mapBaseParams(params)).toEqual(expected)
    })

    it('[Easy] should map stop sequence (string)', () => {
      const params: GenerateParams = { ...baseGenerateParams, stop: '\n' }
      const expected = {
        temperature: undefined,
        topP: undefined,
        maxTokens: undefined,
        stopSequences: ['\n']
      }
      expect(mapBaseParams(params)).toEqual(expected)
    })

    it('[Easy] should map stop sequences (array)', () => {
      const params: GenerateParams = { ...baseGenerateParams, stop: ['\n', 'Human:'] }
      const expected = {
        temperature: undefined,
        topP: undefined,
        maxTokens: undefined,
        stopSequences: ['\n', 'Human:']
      }
      expect(mapBaseParams(params)).toEqual(expected)
    })

    it('[Easy] should handle null stop sequence', () => {
      const params: GenerateParams = { ...baseGenerateParams, stop: null }
      const expected = {
        temperature: undefined,
        topP: undefined,
        maxTokens: undefined,
        stopSequences: undefined
      }
      expect(mapBaseParams(params)).toEqual(expected)
    })

    it('[Easy] should handle undefined parameters', () => {
      const params: GenerateParams = { ...baseGenerateParams } // No optional params
      const expected = {
        temperature: undefined,
        topP: undefined,
        maxTokens: undefined,
        stopSequences: undefined
      }
      expect(mapBaseParams(params)).toEqual(expected)
    })

    it('[Medium] should map a combination of parameters', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        temperature: 0.5,
        maxTokens: 50,
        stop: ['stop']
      }
      const expected = { temperature: 0.5, topP: undefined, maxTokens: 50, stopSequences: ['stop'] }
      expect(mapBaseParams(params)).toEqual(expected)
    })
  })

  describe('mapBaseToolChoice', () => {
    it("[Easy] should map 'auto'", () => {
      expect(mapBaseToolChoice('auto')).toBe('auto')
    })

    it("[Easy] should map 'none'", () => {
      expect(mapBaseToolChoice('none')).toBe('none')
    })

    it("[Easy] should map 'required'", () => {
      expect(mapBaseToolChoice('required')).toBe('required')
    })

    it('[Easy] should map specific function object', () => {
      const toolChoice: GenerateParams['toolChoice'] = { type: 'function', function: { name: 'my_func' } }
      expect(mapBaseToolChoice(toolChoice)).toEqual(toolChoice)
    })

    it('[Easy] should return undefined for undefined input', () => {
      expect(mapBaseToolChoice(undefined)).toBeUndefined()
    })

    it('[Medium] should return undefined and warn for invalid string input', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      expect(mapBaseToolChoice('invalid_string' as any)).toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(
        'Unsupported tool_choice format encountered in common mapping: "invalid_string"'
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should return undefined and warn for invalid object input (missing type)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const toolChoice = { function: { name: 'my_func' } } as any
      expect(mapBaseToolChoice(toolChoice)).toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(
        `Unsupported tool_choice format encountered in common mapping: ${JSON.stringify(toolChoice)}`
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should return undefined and warn for invalid object input (invalid type)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const toolChoice = { type: 'invalid', function: { name: 'my_func' } } as any
      expect(mapBaseToolChoice(toolChoice)).toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(
        `Unsupported tool_choice format encountered in common mapping: ${JSON.stringify(toolChoice)}`
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should return undefined and warn for invalid function object (missing name)', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const toolChoice = { type: 'function', function: {} } as any
      expect(mapBaseToolChoice(toolChoice)).toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(
        `Unsupported tool_choice format encountered in common mapping: ${JSON.stringify(toolChoice)}`
      )
      warnSpy.mockRestore()
    })
  })
})
