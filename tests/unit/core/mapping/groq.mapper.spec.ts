import Groq from 'groq-sdk'
import { Uploadable } from 'groq-sdk/core'
import { GroqMapper } from '../../../../src/core/mapping/groq.mapper'
import * as GroqEmbedMapper from '../../../../src/core/mapping/groq.embed.mapper'
import * as GroqAudioMapper from '../../../../src/core/mapping/groq.audio.mapper'
import {
  GenerateParams,
  Provider,
  RosettaImageData,
  StreamChunk,
  EmbedParams,
  TranscribeParams,
  TranslateParams,
  GenerateResult // Import result type
} from '../../../../src/types'
import { MappingError, UnsupportedFeatureError, ProviderAPIError } from '../../../../src/errors'
import { ChatCompletionChunk } from 'groq-sdk/resources/chat/completions'

// Mock the sub-mappers
jest.mock('../../../../src/core/mapping/groq.embed.mapper')
jest.mock('../../../../src/core/mapping/groq.audio.mapper')

const mockMapToGroqEmbedParams = GroqEmbedMapper.mapToGroqEmbedParams as jest.Mock
const mockMapFromGroqEmbedResponse = GroqEmbedMapper.mapFromGroqEmbedResponse as jest.Mock
const mockMapToGroqSttParams = GroqAudioMapper.mapToGroqSttParams as jest.Mock
const mockMapFromGroqTranscriptionResponse = GroqAudioMapper.mapFromGroqTranscriptionResponse as jest.Mock
const mockMapToGroqTranslateParams = GroqAudioMapper.mapToGroqTranslateParams as jest.Mock
const mockMapFromGroqTranslationResponse = GroqAudioMapper.mapFromGroqTranslationResponse as jest.Mock

// Helper async generator to simulate Groq stream
async function* mockGroqStreamGenerator(chunks: ChatCompletionChunk[]): AsyncIterable<ChatCompletionChunk> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1)) // Simulate slight delay
    yield chunk
  }
}

// Helper async generator that throws an error
async function* mockGroqErrorStreamGenerator(
  chunks: ChatCompletionChunk[],
  errorToThrow: Error
): AsyncIterable<ChatCompletionChunk> {
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

// Mock Uploadable for audio tests
const mockAudioFile: Uploadable = {
  [Symbol.toStringTag]: 'File',
  name: 'mock.mp3'
} as any // Cast to bypass full FileLike implementation details

describe('Groq Mapper', () => {
  let mapper: GroqMapper
  let warnSpy: jest.SpyInstance // Declare spy instance

  beforeEach(() => {
    mapper = new GroqMapper()
    jest.clearAllMocks() // Clear mocks before each test
    warnSpy = jest.spyOn(console, 'warn').mockImplementation() // Initialize and mock spy
  })

  afterEach(() => {
    warnSpy.mockRestore() // Restore console.warn
  })

  it('[Easy] should have the correct provider property', () => {
    expect(mapper.provider).toBe(Provider.Groq)
  })

  // mapContentForGroqRole is tested indirectly via mapToProviderParams

  describe('mapToProviderParams (Generate)', () => {
    const baseParams: GenerateParams = {
      provider: Provider.Groq,
      model: 'llama3-8b-8192',
      messages: [{ role: 'user', content: 'Placeholder' }] // Add placeholder
    }

    it('[Easy] should map basic text messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Be fast.' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi.' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        { role: 'system', content: 'Be fast.' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi.' }
      ])
      expect(result.model).toBe('llama3-8b-8192')
      expect(result.stream).toBeUndefined()
    })

    it('[Easy] should map user message with image (Groq might ignore it)', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'imgdata' }
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'What is this?' },
              { type: 'image', image: imageData } // Image part
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
            { type: 'image_url', image_url: { url: 'data:image/png;base64,imgdata' } }
          ]
        }
      ])
    })

    it('[Easy] should map assistant message with tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null, // Null content allowed with tool calls
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                function: { name: 'my_tool', arguments: '{"arg": 1}' }
              }
            ]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        { role: 'user', content: 'Call tool.' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{"arg": 1}' } }]
        }
      ])
    })

    it('[Easy] should map tool result message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'call_123', content: '{"status": "done"}' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toEqual([
        { role: 'user', content: 'Call tool.' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
        },
        { role: 'tool', tool_call_id: 'call_123', content: '{"status": "done"}' }
      ])
    })

    it('[Easy] should map tools correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Use the tool.' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Gets weather',
              parameters: { type: 'object', properties: { location: { type: 'string' } } }
            }
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.tools).toEqual([
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Gets weather',
            parameters: { type: 'object', properties: { location: { type: 'string' } } }
          }
        }
      ])
    })

    it('[Easy] should map tool_choice (auto/none)', () => {
      const paramsAuto: GenerateParams = { ...baseParams, toolChoice: 'auto' }
      const paramsNone: GenerateParams = { ...baseParams, toolChoice: 'none' }
      const resultAuto = mapper.mapToProviderParams(paramsAuto)
      const resultNone = mapper.mapToProviderParams(paramsNone)
      expect(resultAuto.tool_choice).toBe('auto')
      expect(resultNone.tool_choice).toBe('none')
    })

    it('[Easy] should map tool_choice (required to auto with warning)', () => {
      // warnSpy is active due to beforeEach
      const params: GenerateParams = { ...baseParams, toolChoice: 'required' }
      const result = mapper.mapToProviderParams(params)
      expect(result.tool_choice).toBe('auto')
      expect(warnSpy).toHaveBeenCalledWith("'required' tool_choice mapped to 'auto' for Groq.")
      // warnSpy is restored in afterEach
    })

    it('[Easy] should set stream flag correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Stream this.' }],
        stream: true
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.stream).toBe(true)
    })

    it('[Easy] should throw error for unsupported features (thinking)', () => {
      const paramsThinking: GenerateParams = { ...baseParams, thinking: true }
      expect(() => mapper.mapToProviderParams(paramsThinking)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToProviderParams(paramsThinking)).toThrow(
        "Provider 'groq' does not support the requested feature: Thinking steps"
      )
    })

    it('[Easy] should throw error for unsupported features (grounding)', () => {
      const paramsGrounding: GenerateParams = { ...baseParams, grounding: { enabled: true } }
      expect(() => mapper.mapToProviderParams(paramsGrounding)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToProviderParams(paramsGrounding)).toThrow(
        "Provider 'groq' does not support the requested feature: Grounding/Citations"
      )
    })

    it('[Easy] should throw error for unsupported tool type', () => {
      const params: GenerateParams = {
        ...baseParams,
        tools: [{ type: 'retrieval' } as any] // Unsupported type
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Unsupported tool type for Groq: retrieval')
    })

    it('[Easy] should warn about JSON response format request', () => {
      // warnSpy is active due to beforeEach
      const params: GenerateParams = { ...baseParams, responseFormat: { type: 'json_object' } }
      mapper.mapToProviderParams(params)
      expect(warnSpy).toHaveBeenCalledWith(
        'JSON response format requested, but Groq support is unconfirmed via standard parameters.'
      )
      // warnSpy is restored in afterEach
    })

    it('[Easy] should map temperature, topP, and stop parameters', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate text.' }],
        temperature: 0.6,
        topP: 0.9,
        stop: ['\n', ' Human:']
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.temperature).toBe(0.6)
      expect(result.top_p).toBe(0.9)
      expect(result.stop).toEqual(['\n', ' Human:'])
    })

    it('[Medium] should map tool_choice (function)', () => {
      const params: GenerateParams = {
        ...baseParams,
        toolChoice: { type: 'function', function: { name: 'my_func' } }
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'my_func' } })
    })

    // FIX: Test expectation updated to undefined based on code fix
    it('[Medium] should map tool_choice (invalid format to undefined with warning)', () => {
      // warnSpy is active due to beforeEach
      const params: GenerateParams = { ...baseParams, toolChoice: { type: 'invalid' } as any }
      const result = mapper.mapToProviderParams(params)
      expect(result.tool_choice).toBeUndefined() // Expect undefined now
      // FIX: Update expected warning message to match common.utils.ts
      expect(warnSpy).toHaveBeenCalledWith(
        `Unsupported tool_choice format encountered in common mapping: ${JSON.stringify({
          type: 'invalid'
        })}`
      )
      // warnSpy is restored in afterEach
    })

    it('[Medium] should map assistant message with both text content and tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Get weather and say hi.' },
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

    it('[Medium] should throw error for invalid tool parameters schema (missing type: object)', () => {
      const params: GenerateParams = {
        ...baseParams,
        tools: [
          {
            type: 'function',
            function: {
              name: 'bad_tool',
              parameters: { properties: { location: { type: 'string' } } } // Invalid
            }
          }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        'Invalid parameters schema for tool bad_tool. Groq requires top-level \'type: "object"\'.'
      )
    })

    it('[Hard] should throw error if assistant message has null content but no tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: null } // Null content, no tool calls
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        'Assistant message content cannot be null if no tool calls are present.'
      )
    })

    // FIX: Test expectation updated based on code fix
    it('[Hard] should throw error if system/user/tool message content is empty string', () => {
      const paramsSys: GenerateParams = { ...baseParams, messages: [{ role: 'system', content: '' }] }
      const paramsUser: GenerateParams = { ...baseParams, messages: [{ role: 'user', content: '' }] }
      const paramsTool: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'call_123', content: '' } // Empty content
        ]
      }
      expect(() => mapper.mapToProviderParams(paramsSys)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(paramsUser)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(paramsTool)).toThrow(MappingError) // This should now throw
      expect(() => mapper.mapToProviderParams(paramsSys)).toThrow(
        'Conversation cannot consist only of a system message'
      )
      expect(() => mapper.mapToProviderParams(paramsUser)).toThrow(
        "Role 'user' requires non-empty string content for Groq."
      )
      expect(() => mapper.mapToProviderParams(paramsTool)).toThrow(
        // Check the error message
        "Role 'tool' requires non-empty string content for Groq."
      )
    })

    // --- New Tests ---
    // FIX: Update test to expect MappingError based on code fix
    it('[Easy] mapToProviderParams - should throw error for system message only', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'system', content: 'System only' }]
      }
      // Groq requires at least one non-system message usually
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Conversation cannot consist only of a system message.')
    })

    // FIX: Update test to expect MappingError based on code fix
    it('[Easy] mapToProviderParams - should throw error for assistant message only', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'assistant', content: 'Assistant only' }]
      }
      // Conversation must start with user
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Conversation cannot start with an assistant message.')
    })
  })

  describe('mapFromProviderResponse (Generate)', () => {
    const modelUsed = 'llama3-8b-8192-test'

    it('[Easy] should map basic text response', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_123',
        object: 'chat.completion',
        created: 1700000000,
        model: 'llama3-8b-8192-test-id',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Groq response.' },
            finish_reason: 'stop',
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Groq response.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
      expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 })
      expect(result.model).toBe('llama3-8b-8192-test-id')
    })

    it('[Easy] should map response with tool calls', () => {
      const toolCalls: Groq.Chat.Completions.ChatCompletionMessageToolCall[] = [
        { id: 'call_abc', type: 'function', function: { name: 'get_info', arguments: '{"id": 1}' } }
      ]
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_456',
        object: 'chat.completion',
        created: 1700000001,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: null, tool_calls: toolCalls },
            finish_reason: 'tool_calls',
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 15, completion_tokens: 8, total_tokens: 23 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toEqual([
        { id: 'call_abc', type: 'function', function: { name: 'get_info', arguments: '{"id": 1}' } }
      ])
      expect(result.finishReason).toBe('tool_calls')
      expect(result.usage?.totalTokens).toBe(23)
    })

    it('[Easy] should map length finish reason', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_789',
        object: 'chat.completion',
        created: 1700000002,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Too long...' },
            finish_reason: 'length',
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 5, completion_tokens: 50, total_tokens: 55 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Too long...')
      expect(result.finishReason).toBe('length')
      expect(result.usage?.completionTokens).toBe(50)
    })

    it('[Easy] should handle missing choices gracefully', () => {
      // warnSpy is active due to beforeEach
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_err',
        object: 'chat.completion',
        created: 1700000003,
        model: modelUsed,
        choices: [],
        usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('error')
      expect(result.usage?.totalTokens).toBe(5)
      expect(warnSpy).toHaveBeenCalledWith('Groq response missing choices.')
      // warnSpy is restored in afterEach
    })

    it('[Easy] should handle null message in choice', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_null_msg',
        object: 'chat.completion',
        created: 1700000006,
        model: modelUsed,
        choices: [{ index: 0, message: null as any, finish_reason: 'stop', logprobs: null }]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
    })

    it('[Easy] should attempt to parse JSON content', () => {
      const jsonString = '{"result": "ok"}'
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_json',
        object: 'chat.completion',
        created: 1700000004,
        model: modelUsed,
        choices: [
          { index: 0, message: { role: 'assistant', content: jsonString }, finish_reason: 'stop', logprobs: null }
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toEqual({ result: 'ok' })
    })

    it('[Easy] should handle unparsable JSON content gracefully', () => {
      const jsonString = '{"result": "ok"'
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_bad_json',
        object: 'chat.completion',
        created: 1700000004,
        model: modelUsed,
        choices: [
          { index: 0, message: { role: 'assistant', content: jsonString }, finish_reason: 'stop', logprobs: null }
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toBeNull()
    })

    it('[Easy] should map other finish reasons directly', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_other',
        object: 'chat.completion',
        created: 1700000005,
        model: modelUsed,
        choices: [
          { index: 0, message: { role: 'assistant', content: 'Response' }, finish_reason: 'stop', logprobs: null }
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.finishReason).toBe('stop')
    })

    it('[Medium] should handle response with both text and tool calls', () => {
      const toolCalls: Groq.Chat.Completions.ChatCompletionMessageToolCall[] = [
        { id: 'call_mix', type: 'function', function: { name: 'func_mix', arguments: '{}' } }
      ]
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_mix',
        object: 'chat.completion',
        created: 1700000010,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Mixed response.', tool_calls: toolCalls },
            finish_reason: 'tool_calls',
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 20, completion_tokens: 15, total_tokens: 35 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Mixed response.')
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls![0].id).toBe('call_mix')
      expect(result.finishReason).toBe('tool_calls')
    })

    it('[Medium] should handle response with null/undefined usage', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_no_usage',
        object: 'chat.completion',
        created: 1700000011,
        model: modelUsed,
        choices: [
          { index: 0, message: { role: 'assistant', content: 'Response' }, finish_reason: 'stop', logprobs: null }
        ],
        usage: undefined
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.usage).toBeUndefined()
    })

    it('[Medium] should handle response with empty tool_calls array', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_empty_tools',
        object: 'chat.completion',
        created: 1700000012,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'No tools called.', tool_calls: [] },
            finish_reason: 'stop',
            logprobs: null
          }
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('No tools called.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
    })

    it('[Hard] should handle response with logprobs gracefully', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_logprobs',
        object: 'chat.completion',
        created: 1700000007,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Response' },
            finish_reason: 'stop',
            logprobs: { content: null }
          }
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Response')
      expect(result.finishReason).toBe('stop')
      expect(result.rawResponse).toEqual(response)
    })

    // --- New Tests ---
    it('[Easy] mapFromProviderResponse - should handle null usage object', () => {
      const response: Groq.Chat.Completions.ChatCompletion = {
        id: 'chat_null_usage',
        object: 'chat.completion',
        created: 1700000000,
        model: modelUsed,
        choices: [
          { index: 0, message: { role: 'assistant', content: 'Response' }, finish_reason: 'stop', logprobs: null }
        ],
        usage: null as any // Explicitly null
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.usage).toBeUndefined()
    })
  })

  describe('mapProviderStream', () => {
    const modelId = 'llama3-stream-test'
    const baseChunkProps = { id: 'chatcmpl-123', object: 'chat.completion.chunk' as const, created: 1700000000 }

    it('[Hard] should handle basic text stream', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Hello' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' world' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Groq, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Hello' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: ' world' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 } }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Hello world',
          finishReason: 'stop',
          model: modelId,
          usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 }
        })
      )
    })

    it('[Hard] should handle stream with a single tool call', async () => {
      const toolCallId = 'call_tool_1'
      const toolName = 'get_weather'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: toolCallId, type: 'function', function: { name: toolName } }] },
              logprobs: null,
              finish_reason: null
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"loc' } }] }, finish_reason: null }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, function: { arguments: 'ation": "SF"}' } }] },
              finish_reason: null
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 10, completion_tokens: 8, total_tokens: 18 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, tool_start, delta, delta, tool_done, stop, usage, final
      expect(results[1]).toEqual({
        type: 'tool_call_start',
        data: { index: 0, toolCall: { id: toolCallId, type: 'function', function: { name: toolName } } }
      })
      expect(results[2]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: '{"loc' }
      })
      expect(results[3]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: 'ation": "SF"}' }
      })
      expect(results[4]).toEqual({ type: 'tool_call_done', data: { index: 0, id: toolCallId } })
      expect(results[5]).toEqual({ type: 'message_stop', data: { finishReason: 'tool_calls' } })
      expect(results[6]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 10, completionTokens: 8, totalTokens: 18 } }
      })
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result as GenerateResult
      expect(finalResult.content).toBeNull()
      expect(finalResult.finishReason).toBe('tool_calls')
      expect(finalResult.toolCalls).toEqual([
        { id: toolCallId, type: 'function', function: { name: toolName, arguments: '{"location": "SF"}' } }
      ])
      expect(finalResult.usage).toEqual({ promptTokens: 10, completionTokens: 8, totalTokens: 18 })
    })

    it('[Hard] should handle stream with text and tool call', async () => {
      const toolCallId = 'call_tool_2'
      const toolName = 'send_message'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Okay, ' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: toolCallId, type: 'function', function: { name: toolName } }] },
              logprobs: null,
              finish_reason: null
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'sending...' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, function: { arguments: '{"to": "Bob"}' } }] },
              finish_reason: null
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 12, completion_tokens: 10, total_tokens: 22 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(9) // start, delta, tool_start, delta, tool_delta, tool_done, stop, usage, final
      expect(results[0].type).toBe('message_start')
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Okay, ' } })
      expect(results[2].type).toBe('tool_call_start')
      expect(results[3]).toEqual({ type: 'content_delta', data: { delta: 'sending...' } })
      expect(results[4].type).toBe('tool_call_delta')
      expect(results[5].type).toBe('tool_call_done')
      expect(results[6].type).toBe('message_stop')
      expect(results[7].type).toBe('final_usage')
      expect(results[8].type).toBe('final_result')

      const finalResult = (results[8] as any).data.result as GenerateResult
      expect(finalResult.content).toBe('Okay, sending...')
      expect(finalResult.finishReason).toBe('tool_calls')
      expect(finalResult.toolCalls).toHaveLength(1)
      expect(finalResult.toolCalls![0]).toEqual(
        expect.objectContaining({ function: { name: toolName, arguments: '{"to": "Bob"}' } })
      )
      expect(finalResult.usage).toEqual({ promptTokens: 12, completionTokens: 10, totalTokens: 22 })
    })

    it('[Hard] should handle stream ending with length limit', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'This is too' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'length', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 3, completion_tokens: 3, total_tokens: 6 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'length' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('length')
      expect((results[4] as any).data.result.content).toBe('This is too')
    })

    it('[Hard] should handle empty stream', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 2, completion_tokens: 0, total_tokens: 2 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(4) // start, stop, usage, final
      expect(results[1]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[2].type).toBe('final_usage')
      expect(results[3].type).toBe('final_result')
      expect((results[3] as any).data.result.content).toBeNull()
      expect((results[3] as any).data.result.toolCalls).toBeUndefined()
    })

    it('[Hard] should handle stream error', async () => {
      const apiError = new Groq.APIError(400, { error: { message: 'Bad request' } }, 'Error', {})
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Hello' }, logprobs: null, finish_reason: null }]
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqErrorStreamGenerator(mockChunks, apiError))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      const errorChunk = results[2] as { type: 'error'; data: { error: Error } }
      expect(errorChunk.data.error).toBeInstanceOf(ProviderAPIError)
      expect(errorChunk.data.error.message).toContain('Bad request')
      expect((errorChunk.data.error as ProviderAPIError).provider).toBe(Provider.Groq)
      expect((errorChunk.data.error as ProviderAPIError).statusCode).toBe(400)
    })

    it('[Hard] should handle stream with JSON content and parse final result', async () => {
      const jsonString = '{"key": "value", "items": [1, 2]}'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: '{"key":' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' "value",' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' "items":' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' [1, 2]}' }, logprobs: null, finish_reason: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }],
          x_groq: { usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 } }
        }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, delta x 4, stop, usage, final
      expect(results[5].type).toBe('message_stop')
      expect(results[6].type).toBe('final_usage')
      expect(results[7].type).toBe('final_result')

      const finalResult = (results[7] as any).data.result as GenerateResult
      expect(finalResult.content).toBe(jsonString)
      expect(finalResult.parsedContent).toEqual({ key: 'value', items: [1, 2] }) // Check parsed JSON
    })

    // --- New Tests ---
    // FIX: Update test expectation based on code fix
    it('[Medium] mapProviderStream - should handle usage arriving before finish_reason', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: 'Text' } }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          x_groq: { usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 } }
        }, // Usage chunk
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] } // Finish reason chunk
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 5 chunks now (start, delta, stop, usage, final)
      expect(results).toHaveLength(5)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('message_stop')
      expect(results[3].type).toBe('final_usage') // Usage yielded after stop
      expect((results[3] as any).data.usage.totalTokens).toBe(6)
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.usage.totalTokens).toBe(6) // Usage included in final result
    })

    // FIX: Update test expectation based on code fix
    it('[Hard] mapProviderStream - should handle stream ending abruptly during tool call delta', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: 't1', type: 'function', function: { name: 'toolA' } }] }
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"a":' } }] } }]
        }
        // Stream ends here, no finish_reason or usage
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 6 chunks now (start, tool_start, tool_delta, stop, usage, final)
      expect(results).toHaveLength(6)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('tool_call_start')
      expect(results[2].type).toBe('tool_call_delta')
      expect(results[3].type).toBe('message_stop') // Default stop
      expect((results[3] as any).data.finishReason).toBe('stop')
      expect(results[4].type).toBe('final_usage') // Usage will be undefined
      expect((results[4] as any).data.usage).toBeUndefined()
      expect(results[5].type).toBe('final_result')
      // Tool call might not be fully aggregated in final result if stream ended abruptly
      expect((results[5] as any).data.result.toolCalls).toBeUndefined()
    })

    // FIX: Update test expectation based on code fix
    it('[Hard] mapProviderStream - should ignore unexpected chunk types gracefully', async () => {
      const mockChunks: any[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        { type: 'unexpected_event', data: 'ignore me' }, // Unexpected chunk
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: 'Hello' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }
      ]
      const stream = mapper.mapProviderStream(mockGroqStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 5 chunks now (start, delta, stop, usage, final) - usage is missing but yielded
      expect(results).toHaveLength(5)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('message_stop')
      expect(results[3].type).toBe('final_usage') // Usage yielded even if undefined
      expect((results[3] as any).data.usage).toBeUndefined()
      expect(results[4].type).toBe('final_result')
    })
  })

  describe('Embedding Methods', () => {
    const embedParams: EmbedParams = { provider: Provider.Groq, input: 'test', model: 'nomic-embed-text-v1.5' }
    const embedResponse = { data: [], model: 'nomic-embed-text-v1.5', object: 'list', usage: null } as any
    const mappedEmbedResponse = { embeddings: [], model: 'mapped-embed-model' } as any

    beforeEach(() => {
      mockMapToGroqEmbedParams.mockReturnValue({ mapped: true })
      mockMapFromGroqEmbedResponse.mockReturnValue(mappedEmbedResponse)
    })

    it('[Easy] mapToEmbedParams should delegate', () => {
      const result = mapper.mapToEmbedParams(embedParams)
      expect(mockMapToGroqEmbedParams).toHaveBeenCalledWith(embedParams)
      expect(result).toEqual({ mapped: true })
    })

    it('[Easy] mapFromEmbedResponse should delegate', () => {
      const result = mapper.mapFromEmbedResponse(embedResponse, 'nomic-embed-text-v1.5')
      expect(mockMapFromGroqEmbedResponse).toHaveBeenCalledWith(embedResponse, 'nomic-embed-text-v1.5')
      expect(result).toEqual(mappedEmbedResponse)
    })
  })

  describe('Audio Methods', () => {
    const transcribeParams: TranscribeParams = {
      provider: Provider.Groq,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' },
      model: 'whisper-large-v3'
    }
    const translateParams: TranslateParams = {
      provider: Provider.Groq,
      audio: { data: Buffer.from(''), filename: 'b.wav', mimeType: 'audio/wav' },
      model: 'whisper-large-v3'
    }
    const audioFile = {} as Uploadable
    const transcribeResponse = { text: 'transcribed' } as any
    const translateResponse = { text: 'translated' } as any
    const mappedTranscribeResponse = { text: 'mapped-transcribed' } as any
    const mappedTranslateResponse = { text: 'mapped-translated' } as any

    beforeEach(() => {
      mockMapToGroqSttParams.mockReturnValue({ mapped: 'stt' })
      mockMapFromGroqTranscriptionResponse.mockReturnValue(mappedTranscribeResponse)
      mockMapToGroqTranslateParams.mockReturnValue({ mapped: 'translate' })
      mockMapFromGroqTranslationResponse.mockReturnValue(mappedTranslateResponse)
    })

    it('[Easy] mapToTranscribeParams should delegate', () => {
      const result = mapper.mapToTranscribeParams(transcribeParams, audioFile)
      expect(mockMapToGroqSttParams).toHaveBeenCalledWith(transcribeParams, audioFile)
      expect(result).toEqual({ mapped: 'stt' })
    })

    it('[Easy] mapFromTranscribeResponse should delegate', () => {
      const result = mapper.mapFromTranscribeResponse(transcribeResponse, 'whisper-large-v3')
      expect(mockMapFromGroqTranscriptionResponse).toHaveBeenCalledWith(transcribeResponse, 'whisper-large-v3')
      expect(result).toEqual(mappedTranscribeResponse)
    })

    it('[Easy] mapToTranslateParams should delegate', () => {
      const result = mapper.mapToTranslateParams(translateParams, audioFile)
      expect(mockMapToGroqTranslateParams).toHaveBeenCalledWith(translateParams, audioFile)
      expect(result).toEqual({ mapped: 'translate' })
    })

    it('[Easy] mapFromTranslateResponse should delegate', () => {
      const result = mapper.mapFromTranslateResponse(translateResponse, 'whisper-large-v3')
      expect(mockMapFromGroqTranslationResponse).toHaveBeenCalledWith(translateResponse, 'whisper-large-v3')
      expect(result).toEqual(mappedTranslateResponse)
    })
  })

  describe('wrapProviderError', () => {
    // FIX: Update mock error structure and test expectations
    it('[Easy] should wrap Groq.APIError', () => {
      // FIX: Use a plain object that structurally matches the expected error
      // and bypass the instanceof check by setting the prototype or adjusting the check.
      const underlying = {
        status: 401,
        message: 'Auth Error', // Outer message
        error: {
          // The nested structure wrapProviderError expects
          message: 'Bad key',
          code: 'auth_failed',
          type: 'invalid_request'
        },
        name: 'APIError' // To potentially help with identification if needed
      }
      // Make it look like an instance for the `instanceof` check
      Object.setPrototypeOf(underlying, Groq.APIError.prototype)

      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBe(401)
      // FIX: Expect correct values based on code fix and updated mock
      expect(wrapped.errorCode).toBe('auth_failed') // Should now pass
      expect(wrapped.errorType).toBe('invalid_request') // Should now pass
      expect(wrapped.message).toContain('Bad key') // Should use nested message
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap generic Error', () => {
      const underlying = new Error('Generic network failure')
      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Generic network failure')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap unknown/string error', () => {
      const underlying = 'Something went wrong'
      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Something went wrong')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should not re-wrap RosettaAIError', () => {
      const underlying = new MappingError('Already mapped', Provider.Groq)
      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBe(underlying)
      expect(wrapped).toBeInstanceOf(MappingError)
    })

    // --- New Tests ---
    // FIX: Update test expectation based on code fix
    it('[Easy] wrapProviderError - should handle Groq.APIError without nested error object', () => {
      // FIX: Use plain object mock
      const underlying = {
        status: 500,
        message: 'Server Error', // Outer message
        error: undefined, // No nested error
        name: 'APIError'
      }
      Object.setPrototypeOf(underlying, Groq.APIError.prototype)

      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBe(500)
      expect(wrapped.errorCode).toBeUndefined() // No nested code
      expect(wrapped.errorType).toBeUndefined() // No nested type
      // FIX: Expect the correct fallback message
      expect(wrapped.message).toContain('Server Error') // Uses outer message
      expect(wrapped.underlyingError).toBe(underlying)
    })

    // FIX: Update test expectation based on code fix
    it('[Easy] wrapProviderError - should handle Groq.APIError with null nested error object', () => {
      // FIX: Use plain object mock
      const underlying = {
        status: 404,
        message: 'Not Found', // Outer message
        error: null, // Null nested error
        name: 'APIError'
      }
      Object.setPrototypeOf(underlying, Groq.APIError.prototype)

      const wrapped = mapper.wrapProviderError(underlying, Provider.Groq)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Groq)
      expect(wrapped.statusCode).toBe(404)
      expect(wrapped.errorCode).toBeUndefined()
      expect(wrapped.errorType).toBeUndefined()
      // FIX: Expect the correct fallback message
      expect(wrapped.message).toContain('Not Found') // Uses outer message
      expect(wrapped.underlyingError).toBe(underlying)
    })
  })
})
