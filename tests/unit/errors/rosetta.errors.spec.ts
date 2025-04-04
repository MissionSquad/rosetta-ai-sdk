import {
  RosettaAIError,
  ConfigurationError,
  UnsupportedFeatureError,
  ProviderAPIError,
  MappingError
} from '../../../src/errors'
import { Provider } from '../../../src/types/common.types'

describe('RosettaAI Errors', () => {
  it('RosettaAIError should be base class', () => {
    const error = new RosettaAIError('Base error message')
    expect(error).toBeInstanceOf(RosettaAIError)
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe('RosettaAIError')
    expect(error.message).toBe('Base error message')
    expect(error.timestamp).toBeInstanceOf(Date)
  })

  it('ConfigurationError should construct correctly', () => {
    const error = new ConfigurationError('Missing API key')
    expect(error).toBeInstanceOf(ConfigurationError)
    expect(error).toBeInstanceOf(RosettaAIError)
    expect(error.name).toBe('ConfigurationError')
    expect(error.message).toBe('Configuration Error: Missing API key')
  })

  it('UnsupportedFeatureError should construct correctly', () => {
    const error = new UnsupportedFeatureError(Provider.Groq, 'Streaming TTS')
    expect(error).toBeInstanceOf(UnsupportedFeatureError)
    expect(error).toBeInstanceOf(RosettaAIError)
    expect(error.name).toBe('UnsupportedFeatureError')
    expect(error.message).toBe("Provider 'groq' does not support the requested feature: Streaming TTS")
    expect(error.provider).toBe(Provider.Groq)
    expect(error.feature).toBe('Streaming TTS')
  })

  describe('ProviderAPIError', () => {
    it('should construct with minimal info', () => {
      const error = new ProviderAPIError('Network issue', Provider.Anthropic)
      expect(error).toBeInstanceOf(ProviderAPIError)
      expect(error).toBeInstanceOf(RosettaAIError)
      expect(error.name).toBe('ProviderAPIError')
      expect(error.message).toBe('[anthropic] API Error : Network issue')
      expect(error.provider).toBe(Provider.Anthropic)
      expect(error.statusCode).toBeUndefined()
      expect(error.errorCode).toBeUndefined()
      expect(error.errorType).toBeUndefined()
      expect(error.underlyingError).toBeUndefined()
    })

    it('should construct with status code and codes/types', () => {
      const error = new ProviderAPIError('Rate limit exceeded', Provider.OpenAI, 429, 'rate_limit_exceeded', 'requests')
      expect(error.message).toBe('[openai] API Error (Status 429) [Code: rate_limit_exceeded] : Rate limit exceeded')
      expect(error.provider).toBe(Provider.OpenAI)
      expect(error.statusCode).toBe(429)
      expect(error.errorCode).toBe('rate_limit_exceeded')
      expect(error.errorType).toBe('requests')
      expect(error.underlyingError).toBeUndefined()
    })

    it('should construct with underlying error', () => {
      const underlying = new Error('Original SDK error')
      underlying.stack = 'Original stack trace'
      const error = new ProviderAPIError('Failed request', Provider.Google, 500, null, 'server_error', underlying)
      expect(error.message).toBe('[google] API Error (Status 500) : Failed request')
      expect(error.provider).toBe(Provider.Google)
      expect(error.statusCode).toBe(500)
      expect(error.errorCode).toBeNull()
      expect(error.errorType).toBe('server_error')
      expect(error.underlyingError).toBe(underlying)
      // Check if stack includes underlying error stack
      expect(error.stack).toContain('ProviderAPIError: [google] API Error (Status 500) : Failed request')
      expect(error.stack).toContain('Caused by: Original stack trace')
    })
  })

  describe('MappingError', () => {
    it('should construct with minimal info', () => {
      const error = new MappingError('Could not map role')
      expect(error).toBeInstanceOf(MappingError)
      expect(error).toBeInstanceOf(RosettaAIError)
      expect(error.name).toBe('MappingError')
      expect(error.message).toBe('Mapping Error : Could not map role')
      expect(error.provider).toBeUndefined()
      expect(error.context).toBeUndefined()
    })

    it('should construct with provider and context', () => {
      const error = new MappingError('Invalid content part', Provider.Anthropic, 'mapContentToAnthropic')
      expect(error.message).toBe('Mapping Error [anthropic] [Context: mapContentToAnthropic]: Invalid content part')
      expect(error.provider).toBe(Provider.Anthropic)
      expect(error.context).toBe('mapContentToAnthropic')
    })

    it('should construct with cause', () => {
      const cause = new TypeError('Cannot read property')
      cause.stack = 'TypeError stack trace'
      const error = new MappingError('Unexpected type', Provider.Google, 'mapToGoogleParams', cause)
      expect(error.message).toBe('Mapping Error [google] [Context: mapToGoogleParams]: Unexpected type')
      expect(error.provider).toBe(Provider.Google)
      expect(error.context).toBe('mapToGoogleParams')
      expect(error.stack).toContain(
        'MappingError: Mapping Error [google] [Context: mapToGoogleParams]: Unexpected type'
      )
      expect(error.stack).toContain('Caused by: TypeError stack trace')
    })
  })
})
