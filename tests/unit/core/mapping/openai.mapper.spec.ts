import OpenAI from 'openai'
import { Stream } from 'openai/streaming'
import { ChatCompletionChunk } from 'openai/resources/chat/completions'
import { Uploadable } from 'openai/uploads'
import { OpenAIMapper } from '../../../../src/core/mapping/openai.mapper'
import * as OpenAIEmbedMapper from '../../../../src/core/mapping/openai.embed.mapper'
import * as OpenAIAudioMapper from '../../../../src/core/mapping/openai.audio.mapper'
// Import the actual module to spy on its exports
import * as OpenAICommon from '../../../../src/core/mapping/openai.common'
import {
  GenerateParams,
  Provider,
  RosettaImageData,
  StreamChunk,
  EmbedParams,
  TranscribeParams,
  TranslateParams
} from '../../../../src/types'
import {
  MappingError,
  UnsupportedFeatureError,
  ProviderAPIError,
  InvalidToolDefinitionError
} from '../../../../src/errors'
import { z } from 'zod' // Import Zod

// Mock the sub-mappers used by OpenAIMapper
jest.mock('../../../../src/core/mapping/openai.embed.mapper')
jest.mock('../../../../src/core/mapping/openai.audio.mapper')

const mockMapToOpenAIEmbedParams = OpenAIEmbedMapper.mapToOpenAIEmbedParams as jest.Mock
const mockMapFromOpenAIEmbedResponse = OpenAIEmbedMapper.mapFromOpenAIEmbedResponse as jest.Mock
const mockMapToOpenAITranscribeParams = OpenAIAudioMapper.mapToOpenAITranscribeParams as jest.Mock
const mockMapFromOpenAITranscriptionResponse = OpenAIAudioMapper.mapFromOpenAITranscriptionResponse as jest.Mock
const mockMapToOpenAITranslateParams = OpenAIAudioMapper.mapToOpenAITranslateParams as jest.Mock
const mockMapFromOpenAITranslationResponse = OpenAIAudioMapper.mapFromOpenAITranslationResponse as jest.Mock

// Helper type for ChatCompletionMessage mock to satisfy the interface
type MockChatCompletionMessage = OpenAI.Chat.Completions.ChatCompletionMessage & {
  refusal: string | null
}

// Helper async generator for stream tests
async function* mockOpenAIStreamGenerator(
  chunks: ChatCompletionChunk[]
): AsyncGenerator<ChatCompletionChunk, void, undefined> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield chunk
  }
}

// Helper async generator that throws an error
async function* mockOpenAIErrorStreamGenerator(
  chunks: ChatCompletionChunk[],
  errorToThrow: Error
): AsyncGenerator<ChatCompletionChunk, void, undefined> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield chunk
  }
  throw errorToThrow
}

// Helper to collect stream chunks
async function collectStreamChunks(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) {
    chunks.push(chunk)
  }
  return chunks
}

describe('OpenAI Mapper', () => {
  let mapper: OpenAIMapper
  // Spies for common functions
  let spyMapFromOpenAIResponse: jest.SpyInstance
  let spyMapOpenAIStream: jest.SpyInstance
  let spyWrapOpenAIError: jest.SpyInstance

  beforeEach(() => {
    mapper = new OpenAIMapper()
    jest.clearAllMocks() // Clear mocks before each test

    // Create spies on the actual imported module functions
    spyMapFromOpenAIResponse = jest.spyOn(OpenAICommon, 'mapFromOpenAIResponse')
    spyMapOpenAIStream = jest.spyOn(OpenAICommon, 'mapOpenAIStream')
    spyWrapOpenAIError = jest.spyOn(OpenAICommon, 'wrapOpenAIError')

    // Provide default mock implementations for spies if needed (optional, depends on test)
    spyMapFromOpenAIResponse.mockImplementation((res, _model) => res as any) // Simple pass-through
    spyMapOpenAIStream.mockImplementation(async function*(stream, _provider) {
      yield* stream as any
    }) // Simple pass-through
    spyWrapOpenAIError.mockImplementation(err => err as any) // Simple pass-through
  })

  afterEach(() => {
    // Restore spies after each test
    spyMapFromOpenAIResponse.mockRestore()
    spyMapOpenAIStream.mockRestore()
    spyWrapOpenAIError.mockRestore()
  })

  it('[Easy] should have the correct provider property', () => {
    expect(mapper.provider).toBe(Provider.OpenAI)
  })

  // mapContentForOpenAIRole is tested indirectly via mapToProviderParams

  describe('mapToProviderParams (Generate)', () => {
    const baseParams: GenerateParams = {
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      messages: []
    }

    it('[Easy] should map basic text messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello there.' },
          { role: 'assistant', content: 'Hi! How can I help?' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello there.' },
        { role: 'assistant', content: 'Hi! How can I help?' }
      ])
      expect(result.model).toBe('gpt-4o-mini')
      expect(result.stream).toBe(false)
    })

    it('[Easy] should map user message with text and image', () => {
      const imageData: RosettaImageData = { mimeType: 'image/jpeg', base64Data: 'imgdata' }
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'What is this?' },
              { type: 'image', image: imageData }
            ]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,imgdata' } }
          ]
        }
      ])
    })

    it('[Easy] should map assistant message with tool calls and null content', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Find weather in SF.' },
          {
            role: 'assistant',
            content: null, // Explicitly null
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                function: { name: 'getWeather', arguments: '{"location": "SF"}' }
              }
            ]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        { role: 'user', content: 'Find weather in SF.' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            { id: 'call_123', type: 'function', function: { name: 'getWeather', arguments: '{"location": "SF"}' } }
          ]
        }
      ])
    })

    it('[Easy] should map tool result message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Find weather in SF.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                function: { name: 'getWeather', arguments: '{"location": "SF"}' }
              }
            ]
          },
          {
            role: 'tool',
            toolCallId: 'call_123',
            content: '{"temperature": 75, "unit": "F"}'
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 'call_123',
        content: '{"temperature": 75, "unit": "F"}'
      })
    })

    it('[Easy] should map tools and tool_choice', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'myFunc',
              parameters: { type: 'object', properties: {} },
              zodSchema: z.object({})
            }
          }
        ],
        toolChoice: { type: 'function', function: { name: 'myFunc' } }
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.tools).toEqual([
        { type: 'function', function: { name: 'myFunc', parameters: { type: 'object', properties: {} } } }
      ])
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'myFunc' } })
    })

    it('[Easy] should set stream flag and options correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true // Explicitly set for testing mapping logic
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.stream).toBe(true)
      expect((result as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming).stream_options).toEqual({
        include_usage: true
      })
    })

    it('[Easy] should map response_format', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Return JSON' }],
        responseFormat: { type: 'json_object' }
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.response_format).toEqual({ type: 'json_object' })
    })

    it('[Easy] should map response_format json_schema with defaults', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Return JSON' }],
        responseFormat: { type: 'json_schema', json_schema: { schema: { type: 'object', properties: { a: { type: 'number' } } } } }
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.response_format).toEqual({
        type: 'json_schema',
        json_schema: {
          name: 'response',
          strict: true,
          schema: { type: 'object', properties: { a: { type: 'number' } } }
        }
      })
    })

    it('[Easy] should throw error for unsupported features', () => {
      const paramsThinking: GenerateParams = { ...baseParams, messages: [], thinking: true }
      const paramsGrounding: GenerateParams = { ...baseParams, messages: [], grounding: { enabled: true } }
      expect(() => mapper.mapToProviderParams(paramsThinking)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToProviderParams(paramsGrounding)).toThrow(UnsupportedFeatureError)
    })

    it('[Medium] should map assistant message with both text content and tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Get weather and say something.' },
          {
            role: 'assistant',
            content: 'Okay, getting weather...', // Text content present
            toolCalls: [
              {
                id: 'call_weather',
                type: 'function',
                function: { name: 'getWeather', arguments: '{"location": "London"}' }
              }
            ]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: 'Okay, getting weather...', // Text content should be preserved
        tool_calls: [
          {
            id: 'call_weather',
            type: 'function',
            function: { name: 'getWeather', arguments: '{"location": "London"}' }
          }
        ]
      })
    })

    it('[Medium] should map system message with array content to string', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          {
            role: 'system',
            content: [
              { type: 'text', text: 'Be ' },
              { type: 'text', text: 'concise.' }
            ]
          },
          { role: 'user', content: 'Hello' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[0]).toEqual({ role: 'system', content: 'Be concise.' })
    })

    it('[Medium] should map complex message history', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool A' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'tA', type: 'function', function: { name: 'toolA', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'tA', content: '{"resA": 1}' },
          { role: 'user', content: 'Now call tool B' },
          {
            role: 'assistant',
            content: 'Calling B',
            toolCalls: [{ id: 'tB', type: 'function', function: { name: 'toolB', arguments: '{"p": 2}' } }]
          },
          { role: 'tool', toolCallId: 'tB', content: '{"resB": "ok"}' },
          { role: 'user', content: 'Thanks' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toHaveLength(7)
      expect(result.messages[0]).toEqual({ role: 'user', content: 'Call tool A' })
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 'tA', type: 'function', function: { name: 'toolA', arguments: '{}' } }]
      })
      expect(result.messages[2]).toEqual({ role: 'tool', tool_call_id: 'tA', content: '{"resA": 1}' })
      expect(result.messages[3]).toEqual({ role: 'user', content: 'Now call tool B' })
      expect(result.messages[4]).toEqual({
        role: 'assistant',
        content: 'Calling B',
        tool_calls: [{ id: 'tB', type: 'function', function: { name: 'toolB', arguments: '{"p": 2}' } }]
      })
      expect(result.messages[5]).toEqual({ role: 'tool', tool_call_id: 'tB', content: '{"resB": "ok"}' })
      expect(result.messages[6]).toEqual({ role: 'user', content: 'Thanks' })
    })

    it('[Medium] should throw MappingError for tool message missing toolCallId', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'tool', arguments: '{}' } }]
          },
          { role: 'tool', content: '{"res": 1}' } // Missing toolCallId
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Tool message requires toolCallId.')
    })

    it('[Medium] should map temperature and topP together', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.temperature).toBe(0.7)
      expect(result.top_p).toBe(0.9)
    })

    it('[Medium] should drop sampling and default reasoning for gpt-5', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5',
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.temperature).toBeUndefined()
      expect(result.top_p).toBeUndefined()
      expect(result.reasoning_effort).toBe('medium')
      expect(result.max_completion_tokens).toBeUndefined()
    })

    it('[Medium] should preserve sampling for gpt-5.2 when reasoningEffort is none', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5.2',
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9,
        reasoningEffort: 'none',
        verbosity: 'low',
        maxTokens: 800
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.temperature).toBe(0.7)
      expect(result.top_p).toBe(0.9)
      expect(result.reasoning_effort).toBe('none')
      expect(result.verbosity).toBe('low')
      expect(result.max_completion_tokens).toBe(800)
    })

    it('[Medium] should drop sampling for gpt-5.2 when reasoningEffort is high', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5.2',
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9,
        reasoningEffort: 'high'
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.temperature).toBeUndefined()
      expect(result.top_p).toBeUndefined()
      expect(result.reasoning_effort).toBe('high')
    })

    it('[Medium] should use family-aware lookup for dated gpt-5.2 models', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5.2-2025-12-11',
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        reasoningEffort: 'none',
        verbosity: 'high'
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.temperature).toBe(0.7)
      expect(result.reasoning_effort).toBe('none')
      expect(result.verbosity).toBe('high')
    })

    it('[Medium] should map fixed reasoning for gpt-5.2-chat-latest', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5.2-chat-latest',
        messages: [{ role: 'user', content: 'Generate.' }],
        reasoningEffort: 'none',
        temperature: 0.7,
        verbosity: 'medium'
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.reasoning_effort).toBe('medium')
      expect(result.temperature).toBeUndefined()
      expect(result.top_p).toBeUndefined()
      expect(result.verbosity).toBe('medium')
    })

    it('[Medium] should reject responses-only GPT-5 variants on chat completions', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5-pro',
        messages: [{ role: 'user', content: 'Generate.' }]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        "Provider 'openai' does not support the requested feature: Chat Completions for model 'gpt-5-pro'"
      )
    })

    it('[Medium] should conservatively normalize unknown future GPT-5 family models', () => {
      const params: GenerateParams = {
        ...baseParams,
        model: 'gpt-5.9-experimental',
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9,
        reasoningEffort: 'high',
        verbosity: 'high',
        maxTokens: 321
      }
      const result = mapper.mapToProviderParams(params) as any
      expect(result.temperature).toBeUndefined()
      expect(result.top_p).toBeUndefined()
      expect(result.reasoning_effort).toBeUndefined()
      expect(result.verbosity).toBeUndefined()
      expect(result.max_completion_tokens).toBe(321)
    })

    it('[Medium] should map toolChoice required and none', () => {
      const paramsRequired: GenerateParams = { ...baseParams, messages: [], toolChoice: 'required' }
      const paramsNone: GenerateParams = { ...baseParams, messages: [], toolChoice: 'none' }
      const resultRequired = mapper.mapToProviderParams(paramsRequired)
      const resultNone = mapper.mapToProviderParams(paramsNone)
      expect(resultRequired.tool_choice).toBe('required')
      expect(resultNone.tool_choice).toBe('none')
    })

    it('[Medium] should throw MappingError for invalid tool parameters schema', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [],
        tools: [
          {
            type: 'function',
            function: {
              name: 'bad_tool',
              parameters: 'not an object' as any,
              zodSchema: z.object({})
            }
          }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(InvalidToolDefinitionError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        "Invalid Tool Definition for 'bad_tool': Invalid parameters schema. Expected JSON Schema object."
      )
    })

    it('[Hard] should throw MappingError for empty string system message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: '' },
          { role: 'user', content: 'Hi' }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow("Role 'system' requires non-empty string content.")
    })

    it('[Hard] should map empty string user message successfully', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: '' }]
      }
      expect(() => mapper.mapToProviderParams(params)).not.toThrow()
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([{ role: 'user', content: '' }])
    })

    it('[Hard] should map empty string tool message successfully', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'tool', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 't1', content: '' } // Empty content
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).not.toThrow()
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[2]).toEqual({ role: 'tool', tool_call_id: 't1', content: '' })
    })

    it('[Hard] should handle assistant message with empty content array (maps to empty string)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: [] } // Empty array
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[1]).toEqual({ role: 'assistant', content: '' }) // Maps to empty string
    })

    describe('extraParams passthrough', () => {
      it('should spread extraParams into the provider payload', () => {
        const params: GenerateParams = {
          ...baseParams,
          messages: [{ role: 'user', content: 'Hello' }],
          extraParams: {
            presence_penalty: 0.5,
            frequency_penalty: 0.3,
            seed: 42
          }
        }
        const result = mapper.mapToProviderParams(params)
        expect(result.presence_penalty).toBe(0.5)
        expect(result.frequency_penalty).toBe(0.3)
        expect(result.seed).toBe(42)
      })

      it('should not let extraParams override explicitly mapped fields', () => {
        const params: GenerateParams = {
          ...baseParams,
          messages: [{ role: 'user', content: 'Hello' }],
          temperature: 0.7,
          extraParams: {
            temperature: 0.1,
            custom_param: 'value'
          }
        }
        const result = mapper.mapToProviderParams(params)
        expect(result.temperature).toBe(0.7)
        expect(result.custom_param).toBe('value')
      })

      it('should handle undefined extraParams gracefully', () => {
        const params: GenerateParams = {
          ...baseParams,
          messages: [{ role: 'user', content: 'Hello' }]
        }
        const result = mapper.mapToProviderParams(params)
        expect(result.model).toBe('gpt-4o-mini')
      })

      it('should handle empty extraParams gracefully', () => {
        const params: GenerateParams = {
          ...baseParams,
          messages: [{ role: 'user', content: 'Hello' }],
          extraParams: {}
        }
        const result = mapper.mapToProviderParams(params)
        expect(result.model).toBe('gpt-4o-mini')
      })
    })
  })

  describe('mapFromProviderResponse (Generate)', () => {
    it('[Easy] should delegate to common function', () => {
      const mockResponse = {} as OpenAI.Chat.Completions.ChatCompletion
      mapper.mapFromProviderResponse(mockResponse, 'model')
      // Use the spy to check the call
      // Undefined as the third argument because it expects 3 args
      expect(spyMapFromOpenAIResponse).toHaveBeenCalledWith(mockResponse, 'model', undefined)
    })
  })

  describe('mapProviderStream (Generate)', () => {
    it('[Easy] should delegate to common function', async () => {
      const mockStream = (mockOpenAIStreamGenerator([]) as any) as Stream<ChatCompletionChunk>
      const mockOriginalParams: GenerateParams = {
        provider: Provider.OpenAI,
        model: 'test-model-id',
        messages: [],
        tools: [] // Explicitly pass empty tools array
      }
      // Consume the stream to trigger the delegation, passing originalParams
      for await (const _ of mapper.mapProviderStream(mockStream, mockOriginalParams)) {
        // No-op, just consume
      }
      // Use the spy to check the call, expecting modelId and tools from originalParams
      expect(spyMapOpenAIStream).toHaveBeenCalledWith(
        mockStream,
        Provider.OpenAI,
        'test-model-id', // Expect modelId from originalParams
        [] // Expect tools array from originalParams
      )
    })
  })

  describe('Embedding Methods', () => {
    const embedParams: EmbedParams = { provider: Provider.OpenAI, input: 'test', model: 'embed-model' }
    const embedResponse = { data: [], model: 'embed-model', object: 'list', usage: null } as any
    const mappedEmbedResponse = { embeddings: [], model: 'mapped-embed-model' } as any

    beforeEach(() => {
      mockMapToOpenAIEmbedParams.mockReturnValue({ model: 'mapped-params' })
      mockMapFromOpenAIEmbedResponse.mockReturnValue(mappedEmbedResponse)
    })

    it('[Easy] mapToEmbedParams should delegate', () => {
      const result = mapper.mapToEmbedParams(embedParams)
      expect(mockMapToOpenAIEmbedParams).toHaveBeenCalledWith(embedParams)
      expect(result).toEqual({ model: 'mapped-params' })
    })

    it('[Easy] mapFromEmbedResponse should delegate', () => {
      const result = mapper.mapFromEmbedResponse(embedResponse, 'embed-model')
      expect(mockMapFromOpenAIEmbedResponse).toHaveBeenCalledWith(embedResponse, 'embed-model')
      expect(result).toEqual(mappedEmbedResponse)
    })
  })

  describe('Audio Methods', () => {
    const transcribeParams: TranscribeParams = {
      provider: Provider.OpenAI,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' },
      model: 'whisper-1'
    }
    const translateParams: TranslateParams = {
      provider: Provider.OpenAI,
      audio: { data: Buffer.from(''), filename: 'b.wav', mimeType: 'audio/wav' },
      model: 'whisper-1'
    }
    const audioFile = {} as Uploadable
    const transcribeResponse = { text: 'transcribed' } as any
    const translateResponse = { text: 'translated' } as any
    const mappedTranscribeResponse = { text: 'mapped-transcribed' } as any
    const mappedTranslateResponse = { text: 'mapped-translated' } as any

    beforeEach(() => {
      mockMapToOpenAITranscribeParams.mockReturnValue({ model: 'mapped-stt-params' })
      mockMapFromOpenAITranscriptionResponse.mockReturnValue(mappedTranscribeResponse)
      mockMapToOpenAITranslateParams.mockReturnValue({ model: 'mapped-translate-params' })
      mockMapFromOpenAITranslationResponse.mockReturnValue(mappedTranslateResponse)
    })

    it('[Easy] mapToTranscribeParams should delegate', () => {
      const result = mapper.mapToTranscribeParams(transcribeParams, audioFile)
      expect(mockMapToOpenAITranscribeParams).toHaveBeenCalledWith(transcribeParams, audioFile)
      expect(result).toEqual({ model: 'mapped-stt-params' })
    })

    it('[Easy] mapFromTranscribeResponse should delegate', () => {
      const result = mapper.mapFromTranscribeResponse(transcribeResponse, 'whisper-1')
      expect(mockMapFromOpenAITranscriptionResponse).toHaveBeenCalledWith(transcribeResponse, 'whisper-1')
      expect(result).toEqual(mappedTranscribeResponse)
    })

    it('[Easy] mapToTranslateParams should delegate', () => {
      const result = mapper.mapToTranslateParams(translateParams, audioFile)
      expect(mockMapToOpenAITranslateParams).toHaveBeenCalledWith(translateParams, audioFile)
      expect(result).toEqual({ model: 'mapped-translate-params' })
    })

    it('[Easy] mapFromTranslateResponse should delegate', () => {
      const result = mapper.mapFromTranslateResponse(translateResponse, 'whisper-1')
      expect(mockMapFromOpenAITranslationResponse).toHaveBeenCalledWith(translateResponse, 'whisper-1')
      expect(result).toEqual(mappedTranslateResponse)
    })
  })

  describe('wrapProviderError', () => {
    it('[Easy] should delegate error wrapping to common function', () => {
      const underlying = new OpenAI.APIError(400, {}, '', new Headers())
      mapper.wrapProviderError(underlying, Provider.OpenAI)
      // Use the spy to check the call
      expect(spyWrapOpenAIError).toHaveBeenCalledWith(underlying, Provider.OpenAI)
    })

    it('[Easy] should return the result from the common function', () => {
      const underlying = new Error('Generic')
      const wrappedError = new ProviderAPIError('Wrapped', Provider.OpenAI)
      // Mock the *spy's* return value for this specific test
      spyWrapOpenAIError.mockReturnValue(wrappedError)

      const result = mapper.wrapProviderError(underlying, Provider.OpenAI)
      // FIX: Change assertion to toEqual
      expect(result).toEqual(wrappedError)
    })
  })
})
