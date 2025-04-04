import {
  mapToOpenAIParams,
  mapFromOpenAIResponse,
  mapContentForOpenAIRole,
  mapOpenAIStream, // Import the stream mapper
  mapUsageFromOpenAI, // Import usage mapper
  mapToolCallsFromOpenAI // Import tool call mapper
} from '../../../../src/core/mapping/openai.mapper'
import {
  GenerateParams,
  Provider,
  RosettaMessage,
  RosettaImageData,
  StreamChunk, // Import StreamChunk
  TokenUsage // Import TokenUsage
} from '../../../../src/types'
import { MappingError, UnsupportedFeatureError, ProviderAPIError } from '../../../../src/errors' // Import ProviderAPIError
import OpenAI from 'openai'
import { Stream } from 'openai/streaming' // Import Stream type
import {
  ChatCompletionChunk, // Import Chunk type
  ChatCompletionMessageToolCall, // Import ToolCall type
  ChatCompletionTokenLogprob // Import Logprob type
} from 'openai/resources/chat/completions'
import { CompletionUsage } from 'openai/resources/completions' // Import CompletionUsage

// Helper type for ChatCompletionMessage mock to satisfy the interface
type MockChatCompletionMessage = OpenAI.Chat.Completions.ChatCompletionMessage & {
  // Add potentially missing optional fields if needed, but `refusal` is the key one
  refusal: string | null // Make refusal explicitly nullable as per the type definition
}

// Helper async generator for stream tests
async function* mockOpenAIStreamGenerator(
  chunks: ChatCompletionChunk[]
): AsyncGenerator<ChatCompletionChunk, void, undefined> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1)) // Simulate delay
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
  // --- NEW: Tests for mapUsageFromOpenAI ---
  describe('mapUsageFromOpenAI', () => {
    it('[Easy] should map a valid usage object', () => {
      const usage: CompletionUsage = { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 }
      const expected: TokenUsage = { promptTokens: 10, completionTokens: 20, totalTokens: 30 }
      expect(mapUsageFromOpenAI(usage)).toEqual(expected)
    })

    it('[Easy] should return undefined for null or undefined input', () => {
      expect(mapUsageFromOpenAI(null)).toBeUndefined()
      expect(mapUsageFromOpenAI(undefined)).toBeUndefined()
    })

    it('[Easy] should handle missing optional fields (completion_tokens)', () => {
      const usage = { prompt_tokens: 10, total_tokens: 10 } as CompletionUsage // Missing completion_tokens
      const expected: TokenUsage = { promptTokens: 10, completionTokens: undefined, totalTokens: 10 }
      expect(mapUsageFromOpenAI(usage)).toEqual(expected)
    })
  })
  // --- END: Tests for mapUsageFromOpenAI ---

  // --- NEW: Tests for mapToolCallsFromOpenAI ---
  describe('mapToolCallsFromOpenAI', () => {
    it('[Easy] should map valid tool calls', () => {
      const toolCalls: ChatCompletionMessageToolCall[] = [
        { id: 'call_1', type: 'function', function: { name: 'func1', arguments: '{"a":1}' } },
        { id: 'call_2', type: 'function', function: { name: 'func2', arguments: '{}' } }
      ]
      const expected = [
        { id: 'call_1', type: 'function', function: { name: 'func1', arguments: '{"a":1}' } },
        { id: 'call_2', type: 'function', function: { name: 'func2', arguments: '{}' } }
      ]
      expect(mapToolCallsFromOpenAI(toolCalls)).toEqual(expected)
    })

    it('[Easy] should return undefined for null, undefined, or empty array input', () => {
      expect(mapToolCallsFromOpenAI(null as any)).toBeUndefined()
      expect(mapToolCallsFromOpenAI(undefined)).toBeUndefined()
      expect(mapToolCallsFromOpenAI([])).toBeUndefined()
    })

    it('[Easy] should filter out tool calls with missing id or function name', () => {
      const toolCalls = [
        { id: 'call_1', type: 'function', function: { name: 'func1', arguments: '{}' } },
        { id: null, type: 'function', function: { name: 'func2', arguments: '{}' } }, // Missing id
        { id: 'call_3', type: 'function', function: { name: null, arguments: '{}' } } // Missing name
      ] as any
      const expected = [{ id: 'call_1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      expect(mapToolCallsFromOpenAI(toolCalls)).toEqual(expected)
    })

    it('[Easy] should default arguments to "{}" if missing', () => {
      const toolCalls = [{ id: 'call_1', type: 'function', function: { name: 'func1', arguments: undefined } }] as any
      const expected = [{ id: 'call_1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      expect(mapToolCallsFromOpenAI(toolCalls)).toEqual(expected)
    })
  })
  // --- END: Tests for mapToolCallsFromOpenAI ---

  describe('mapContentForOpenAIRole', () => {
    it('should map string content', () => {
      expect(mapContentForOpenAIRole('Hello', 'user')).toBe('Hello')
      expect(mapContentForOpenAIRole('World', 'assistant')).toBe('World')
    })

    it('should map text part array for user', () => {
      const content: RosettaMessage['content'] = [{ type: 'text', text: 'Part 1' }]
      const expected = [{ type: 'text', text: 'Part 1' }]
      expect(mapContentForOpenAIRole(content, 'user')).toEqual(expected)
    })

    it('should map mixed text/image parts array for user', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'base64string' }
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Describe this:' },
        { type: 'image', image: imageData }
      ]
      const expected = [
        { type: 'text', text: 'Describe this:' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,base64string' } }
      ]
      expect(mapContentForOpenAIRole(content, 'user')).toEqual(expected)
    })

    // Test Fixed: Expect error instead of filtering
    it('should throw error when filtering non-text parts for assistant', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'base64string' }
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Result:' },
        { type: 'image', image: imageData } // Invalid for assistant
      ]
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(
        "Image content parts only allowed for 'user' role, not 'assistant'."
      )
    })

    // Test Fixed: Expect error instead of returning null
    it('should throw error for assistant if only non-text parts exist', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'base64string' }
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }] // Invalid for assistant
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(
        "Image content parts only allowed for 'user' role, not 'assistant'."
      )
    })

    it('should throw error if image part provided for non-user role', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'base64string' }
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }]
      expect(() => mapContentForOpenAIRole(content, 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(MappingError) // Error thrown during mapping attempt
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(MappingError)
    })

    it('should map text parts array to string for tool role', () => {
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Tool ' },
        { type: 'text', text: 'Result' }
      ]
      expect(mapContentForOpenAIRole(content, 'tool')).toBe('Tool Result')
    })

    it('should throw error for non-text parts for tool role', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'base64string' }
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }]
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(MappingError)
    })

    it('should return null for null content if role is assistant or tool', () => {
      expect(mapContentForOpenAIRole(null, 'assistant')).toBeNull()
      expect(mapContentForOpenAIRole(null, 'tool')).toBeNull()
    })

    it('[Medium] should throw error for null content if role is user or system', () => {
      expect(() => mapContentForOpenAIRole(null, 'user')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(null, 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(null, 'user')).toThrow("Role 'user' requires non-null content.")
      expect(() => mapContentForOpenAIRole(null, 'system')).toThrow("Role 'system' requires non-null content.")
    })

    it('[Easy] should map empty content array for user role', () => {
      const content: RosettaMessage['content'] = []
      expect(mapContentForOpenAIRole(content, 'user')).toEqual([])
    })

    // Test Fixed: Adjust expectation for assistant, expect throw for tool/system
    it('[Medium] should handle empty content array for assistant/tool/system roles', () => {
      const content: RosettaMessage['content'] = []
      // Assistant: Should return null (handled by caller if no tool calls)
      expect(mapContentForOpenAIRole(content, 'assistant')).toBeNull()
      // Tool & System: Should throw error
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(
        "Role 'tool' requires non-empty content array. Received empty array."
      )
      expect(() => mapContentForOpenAIRole(content, 'system')).toThrow(
        "Role 'system' requires non-empty content array. Received empty array."
      )
    })
  })

  describe('mapToOpenAIParams', () => {
    const baseParams: GenerateParams = {
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      messages: []
    }

    it('should map basic text messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello there.' },
          { role: 'assistant', content: 'Hi! How can I help?' }
        ]
      }
      const result = mapToOpenAIParams(params)
      expect(result.messages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello there.' },
        { role: 'assistant', content: 'Hi! How can I help?' }
      ])
      expect(result.model).toBe('gpt-4o-mini')
      expect(result.stream).toBe(false)
    })

    it('should map user message with text and image', () => {
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
      const result = mapToOpenAIParams(params)
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

    it('should map assistant message with tool calls and null content', () => {
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
      const result = mapToOpenAIParams(params)
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

    it('should map assistant message with tool calls and empty string content', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Find weather in SF.' },
          {
            role: 'assistant',
            content: '', // Empty string
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
      const result = mapToOpenAIParams(params)
      expect(result.messages[1]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: null, // Should become null when tool calls are present
          tool_calls: expect.any(Array)
        })
      )
    })

    it('should map tool result message', () => {
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
      const result = mapToOpenAIParams(params)
      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 'call_123',
        content: '{"temperature": 75, "unit": "F"}'
      })
    })

    it('should map tools and tool_choice', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }],
        tools: [
          {
            type: 'function',
            function: { name: 'myFunc', parameters: { type: 'object', properties: {} } }
          }
        ],
        toolChoice: { type: 'function', function: { name: 'myFunc' } }
      }
      const result = mapToOpenAIParams(params)
      expect(result.tools).toEqual([
        { type: 'function', function: { name: 'myFunc', parameters: { type: 'object', properties: {} } } }
      ])
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'myFunc' } })
    })

    it('should set stream flag and options correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true // Explicitly set for testing mapping logic
      }
      const result = mapToOpenAIParams(params)
      expect(result.stream).toBe(true)
      expect((result as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming).stream_options).toEqual({
        include_usage: true
      })
    })

    it('should map response_format', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Return JSON' }],
        responseFormat: { type: 'json_object' }
      }
      const result = mapToOpenAIParams(params)
      expect(result.response_format).toEqual({ type: 'json_object' })
    })

    it('should throw error for unsupported features', () => {
      const paramsThinking: GenerateParams = { ...baseParams, messages: [], thinking: true }
      const paramsGrounding: GenerateParams = { ...baseParams, messages: [], grounding: { enabled: true } }
      expect(() => mapToOpenAIParams(paramsThinking)).toThrow(UnsupportedFeatureError)
      expect(() => mapToOpenAIParams(paramsGrounding)).toThrow(UnsupportedFeatureError)
    })

    it('should map assistant message with both text content and tool calls', () => {
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
      const result = mapToOpenAIParams(params)
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

    // --- NEW TESTS (Easy/Medium) ---
    it('[Easy] should map system message with array content to string', () => {
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
      const result = mapToOpenAIParams(params)
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
      const result = mapToOpenAIParams(params)
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
      expect(() => mapToOpenAIParams(params)).toThrow(MappingError)
      expect(() => mapToOpenAIParams(params)).toThrow('Tool message requires toolCallId.')
    })

    it('[Medium] should map temperature and topP together', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.9
      }
      const result = mapToOpenAIParams(params)
      expect(result.temperature).toBe(0.7)
      expect(result.top_p).toBe(0.9)
    })

    it('[Medium] should map toolChoice required and none', () => {
      const paramsRequired: GenerateParams = { ...baseParams, messages: [], toolChoice: 'required' }
      const paramsNone: GenerateParams = { ...baseParams, messages: [], toolChoice: 'none' }
      const resultRequired = mapToOpenAIParams(paramsRequired)
      const resultNone = mapToOpenAIParams(paramsNone)
      expect(resultRequired.tool_choice).toBe('required')
      expect(resultNone.tool_choice).toBe('none')
    })

    // FIX: Correct the mock data for tool parameters to be an object
    it('[Medium] should throw MappingError for invalid tool parameters schema', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [],
        tools: [
          {
            type: 'function',
            function: {
              name: 'bad_tool',
              parameters: 'not an object' as any // Invalid schema - Cast to any to bypass initial check
            }
          }
        ]
      }
      // The error should be caught during the mapping process
      expect(() => mapToOpenAIParams(params)).toThrow(MappingError)
      expect(() => mapToOpenAIParams(params)).toThrow('Invalid parameters schema for tool bad_tool.')
    })
    // --- END NEW TESTS ---
  })

  describe('mapFromOpenAIResponse', () => {
    const modelUsed = 'gpt-4o-mini-test'

    // Helper to create a valid mock message satisfying the type
    const createMockMessage = (
      content: string | null,
      toolCalls?: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[]
    ): MockChatCompletionMessage => ({
      role: 'assistant',
      content: content,
      tool_calls: toolCalls,
      refusal: null // Explicitly add refusal as null to satisfy the type
    })

    it('should map basic text response', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: 1677652288,
        model: 'gpt-4o-mini-test-id',
        choices: [
          {
            index: 0,
            message: createMockMessage('This is the response.'), // Use helper
            finish_reason: 'stop',
            logprobs: null // Add missing logprobs property
          }
        ],
        usage: { prompt_tokens: 9, completion_tokens: 12, total_tokens: 21 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe('This is the response.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
      expect(result.usage).toEqual({ promptTokens: 9, completionTokens: 12, totalTokens: 21 })
      expect(result.model).toBe('gpt-4o-mini-test-id') // Use model from response
      expect(result.parsedContent).toBeNull()
    })

    it('should map response with tool calls', () => {
      const toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [
        {
          id: 'call_abc',
          type: 'function',
          function: { name: 'getWeather', arguments: '{"location":"Boston"}' }
        }
      ]
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-456',
        object: 'chat.completion',
        created: 1677652290,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: createMockMessage(null, toolCalls), // Use helper
            finish_reason: 'tool_calls',
            logprobs: null // Add missing logprobs property
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toEqual([
        { id: 'call_abc', type: 'function', function: { name: 'getWeather', arguments: '{"location":"Boston"}' } }
      ])
      expect(result.finishReason).toBe('tool_calls')
      expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 })
      expect(result.model).toBe(modelUsed)
    })

    it('[Easy] should map response with length finish reason', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-789',
        object: 'chat.completion',
        created: 1677652292,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: createMockMessage('This response was cut off...'), // Use helper
            finish_reason: 'length',
            logprobs: null // Add missing logprobs property
          }
        ],
        usage: { prompt_tokens: 5, completion_tokens: 100, total_tokens: 105 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe('This response was cut off...')
      expect(result.finishReason).toBe('length')
      expect(result.usage?.completionTokens).toBe(100)
    })

    it('should attempt to parse JSON content', () => {
      const jsonString = '{"key": "value", "num": 123}'
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-json',
        object: 'chat.completion',
        created: 1677652294,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: createMockMessage(jsonString), // Use helper
            finish_reason: 'stop',
            logprobs: null // Add missing logprobs property
          }
        ],
        usage: { prompt_tokens: 8, completion_tokens: 10, total_tokens: 18 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.finishReason).toBe('stop')
      expect(result.parsedContent).toEqual({ key: 'value', num: 123 })
    })

    it('should handle unparsable JSON content gracefully', () => {
      const jsonString = '{"key": "value", "num": 123' // Missing closing brace
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-badjson',
        object: 'chat.completion',
        created: 1677652296,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: createMockMessage(jsonString), // Use helper
            finish_reason: 'stop',
            logprobs: null // Add missing logprobs property
          }
        ]
      }
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.finishReason).toBe('stop')
      expect(result.parsedContent).toBeNull() // Should be null if parsing fails
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to auto-parse potential JSON from OpenAI:'),
        expect.any(SyntaxError) // Check that the second argument is a SyntaxError
      )
      warnSpy.mockRestore()
    })

    // --- NEW TESTS (Easy/Medium) ---
    it('[Easy] should handle missing usage field', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-nousage',
        object: 'chat.completion',
        created: 1677652300,
        model: modelUsed,
        choices: [{ index: 0, message: createMockMessage('Response'), finish_reason: 'stop', logprobs: null }],
        usage: undefined // Missing usage
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.usage).toBeUndefined()
    })

    it('[Easy] should handle empty choices array', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-nochoice',
        object: 'chat.completion',
        created: 1677652302,
        model: modelUsed,
        choices: [], // Empty choices
        usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('error') // Default finish reason for missing choice
      expect(result.usage?.totalTokens).toBe(5)
      expect(warnSpy).toHaveBeenCalledWith('OpenAI response missing choices.')
      warnSpy.mockRestore()
    })

    it('[Medium] should handle content_filter finish reason', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-filter',
        object: 'chat.completion',
        created: 1677652304,
        model: modelUsed,
        choices: [{ index: 0, message: createMockMessage(null), finish_reason: 'content_filter', logprobs: null }],
        usage: { prompt_tokens: 10, completion_tokens: 0, total_tokens: 10 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('content_filter')
    })

    it('[Medium] should handle response with refusal content (future proofing)', () => {
      // This tests if the mapper ignores the 'refusal' field if present,
      // as RosettaAI doesn't currently have a dedicated field for it.
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-refusal',
        object: 'chat.completion',
        created: 1677652306,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: null, refusal: 'Refused due to safety.' } as any, // Mock refusal
            finish_reason: 'stop', // Finish reason might still be stop
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 0, total_tokens: 10 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull() // Content is null
      expect(result.finishReason).toBe('stop')
      // No specific assertion for refusal, just ensure it doesn't crash
      expect(result.rawResponse).toBeDefined()
    })

    it('[Easy] should handle non-JSON string response (parsedContent should be null)', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-nonjson',
        object: 'chat.completion',
        created: 1677652308,
        model: modelUsed,
        choices: [{ index: 0, message: createMockMessage('Just plain text.'), finish_reason: 'stop', logprobs: null }],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe('Just plain text.')
      expect(result.parsedContent).toBeNull()
    })

    it('[Medium] should handle response where choices[0].message is null', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-nullmsg',
        object: 'chat.completion',
        created: 1677652310,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: null as any, // Null message
            finish_reason: 'stop',
            logprobs: null
          }
        ],
        usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop') // Finish reason from choice is still used
    })

    // FIX: Add top_logprobs: null to mock logprobs data
    it('[Hard] should handle response with logprobs gracefully', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-logprobs',
        object: 'chat.completion',
        created: 1677652312,
        model: modelUsed,
        choices: [
          {
            index: 0,
            message: createMockMessage('Response with logprobs'),
            finish_reason: 'stop',
            logprobs: {
              content: [{ token: 'Resp', logprob: -0.1, bytes: null, top_logprobs: null }] // Example logprobs with top_logprobs
            }
          }
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe('Response with logprobs')
      expect(result.finishReason).toBe('stop')
      // We don't map logprobs, just ensure it doesn't crash and raw response is present
      expect(result.rawResponse).toBeDefined()
      expect((result.rawResponse as any).choices[0].logprobs).toBeDefined()
    })
    // --- END NEW TESTS ---
  })

  // --- NEW: mapOpenAIStream Tests ---
  describe('mapOpenAIStream', () => {
    const modelId = 'gpt-stream-test'
    const baseChunkProps = { id: 'chatcmpl-stream-123', object: 'chat.completion.chunk' as const, created: 1700000000 }

    it('[Easy] should handle basic text streaming', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Stream ' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'response.' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
        },
        // Usage chunk (from stream_options)
        {
          ...baseChunkProps,
          model: modelId,
          choices: [], // Usage chunk has empty choices
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 }
        }
      ]
      // FIX: Cast the mock generator to satisfy the type checker
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.OpenAI, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Stream ' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: 'response.' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 } }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Stream response.',
          finishReason: 'stop',
          model: modelId,
          usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 }
        })
      )
    })

    it('[Easy] should handle streaming ending with length', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Too long' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'length', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'length' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('length')
      expect((results[4] as any).data.result.content).toBe('Too long')
    })

    it('[Medium] should handle streaming ending with content_filter', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Unsafe' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'content_filter', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'content_filter' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('content_filter')
      expect((results[4] as any).data.result.content).toBe('Unsafe')
    })

    it('[Medium] should handle stream error (APIError)', async () => {
      const apiError = new OpenAI.APIError(500, { message: 'Server issue' }, 'Internal Server Error', {})
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Partial' }, finish_reason: null, logprobs: null }]
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream(
        (mockOpenAIErrorStreamGenerator(mockChunks, apiError) as any) as Stream<ChatCompletionChunk>
      )
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      const errorChunk = results[2] as { type: 'error'; data: { error: Error } }
      expect(errorChunk.data.error).toBeInstanceOf(ProviderAPIError)
      expect(errorChunk.data.error.message).toContain('Server issue')
      expect((errorChunk.data.error as ProviderAPIError).provider).toBe(Provider.OpenAI)
      expect((errorChunk.data.error as ProviderAPIError).statusCode).toBe(500)
    })

    it('[Hard] should handle streaming with a single tool call', async () => {
      const toolCallId = 'call_t1'
      const toolName = 'func_stream'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: null }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: toolCallId, type: 'function', function: { name: toolName } }] },
              finish_reason: null,
              logprobs: null
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"a":' } }] }, finish_reason: null }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: ' 1}' } }] }, finish_reason: null }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, tool_start, delta, delta, tool_done, stop, usage, final
      expect(results[1]).toEqual({
        type: 'tool_call_start',
        data: { index: 0, toolCall: { id: toolCallId, type: 'function', function: { name: toolName } } }
      })
      expect(results[2]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: '{"a":' }
      })
      expect(results[3]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: ' 1}' }
      })
      expect(results[4]).toEqual({ type: 'tool_call_done', data: { index: 0, id: toolCallId } })
      expect(results[5]).toEqual({ type: 'message_stop', data: { finishReason: 'tool_calls' } })
      expect(results[6].type).toBe('final_usage')
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result
      expect(finalResult.content).toBeNull()
      expect(finalResult.finishReason).toBe('tool_calls')
      expect(finalResult.toolCalls).toEqual([
        { id: toolCallId, type: 'function', function: { name: toolName, arguments: '{"a": 1}' } }
      ])
    })

    it('[Hard] should handle streaming with JSON mode', async () => {
      const jsonString = '{"status": "complete", "value": 100}'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: '{"status":' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' "complete",' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: ' "value": 100}' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 6, completion_tokens: 10, total_tokens: 16 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, json_delta x 3, json_done, stop, usage, final
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('json_delta')
      expect((results[1] as any).data.delta).toBe('{"status":')
      expect((results[1] as any).data.snapshot).toBe('{"status":')
      expect((results[1] as any).data.parsed).toBeUndefined() // Cannot parse yet

      expect(results[2].type).toBe('json_delta')
      expect((results[2] as any).data.delta).toBe(' "complete",')
      expect((results[2] as any).data.snapshot).toBe('{"status": "complete",')
      expect((results[2] as any).data.parsed).toBeUndefined()

      expect(results[3].type).toBe('json_delta')
      expect((results[3] as any).data.delta).toBe(' "value": 100}')
      expect((results[3] as any).data.snapshot).toBe(jsonString)
      expect((results[3] as any).data.parsed).toEqual({ status: 'complete', value: 100 }) // Parsed successfully

      expect(results[4].type).toBe('json_done') // Yielded when finish_reason arrives
      expect((results[4] as any).data.snapshot).toBe(jsonString)
      expect((results[4] as any).data.parsed).toEqual({ status: 'complete', value: 100 })

      expect(results[5].type).toBe('message_stop')
      expect(results[6].type).toBe('final_usage')
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result
      expect(finalResult.content).toBe(jsonString)
      expect(finalResult.parsedContent).toEqual({ status: 'complete', value: 100 })
    })

    it('[Hard] should handle streaming with concurrent tool calls', async () => {
      const toolCallId1 = 'call_t1_conc'
      const toolName1 = 'func_conc1'
      const toolCallId2 = 'call_t2_conc'
      const toolName2 = 'func_conc2'
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: null }, finish_reason: null, logprobs: null }]
        },
        // Start tool 1
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: toolCallId1, type: 'function', function: { name: toolName1 } }] },
              finish_reason: null,
              logprobs: null
            }
          ]
        },
        // Start tool 2
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 1, id: toolCallId2, type: 'function', function: { name: toolName2 } }] },
              finish_reason: null,
              logprobs: null
            }
          ]
        },
        // Args for tool 1
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"p1":' } }] }, finish_reason: null }
          ]
        },
        // Args for tool 2
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 1, delta: { tool_calls: [{ index: 1, function: { arguments: '{"p2":' } }] }, finish_reason: null }
          ]
        },
        // Args for tool 1
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: ' 1}' } }] }, finish_reason: null }
          ]
        },
        // Args for tool 2
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            { index: 1, delta: { tool_calls: [{ index: 1, function: { arguments: ' 2}' } }] }, finish_reason: null }
          ]
        },
        // Finish
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 15, completion_tokens: 12, total_tokens: 27 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      // Expected: start, tool1_start, tool2_start, tool1_delta, tool2_delta, tool1_delta, tool2_delta, tool1_done, tool2_done, stop, usage, final
      expect(results).toHaveLength(12)
      expect(results[1].type).toBe('tool_call_start')
      expect((results[1] as any).data.index).toBe(0)
      expect((results[1] as any).data.toolCall.id).toBe(toolCallId1)

      expect(results[2].type).toBe('tool_call_start')
      expect((results[2] as any).data.index).toBe(1)
      expect((results[2] as any).data.toolCall.id).toBe(toolCallId2)

      expect(results[3].type).toBe('tool_call_delta')
      expect((results[3] as any).data.index).toBe(0)
      expect((results[3] as any).data.id).toBe(toolCallId1)
      expect((results[3] as any).data.functionArgumentChunk).toBe('{"p1":')

      expect(results[4].type).toBe('tool_call_delta')
      expect((results[4] as any).data.index).toBe(1) // Index should be 1 for tool 2
      expect((results[4] as any).data.id).toBe(toolCallId2)
      expect((results[4] as any).data.functionArgumentChunk).toBe('{"p2":')

      expect(results[5].type).toBe('tool_call_delta')
      expect((results[5] as any).data.index).toBe(0)
      expect((results[5] as any).data.functionArgumentChunk).toBe(' 1}')

      expect(results[6].type).toBe('tool_call_delta')
      expect((results[6] as any).data.index).toBe(1) // Index should be 1 for tool 2
      expect((results[6] as any).data.functionArgumentChunk).toBe(' 2}')

      // Tool done events might arrive out of order depending on when finish_reason arrives,
      // but the mapper should yield them when finish_reason is received.
      const doneEvents = results.filter(c => c.type === 'tool_call_done')
      expect(doneEvents).toHaveLength(2)
      expect(doneEvents).toContainEqual({ type: 'tool_call_done', data: { index: 0, id: toolCallId1 } })
      expect(doneEvents).toContainEqual({ type: 'tool_call_done', data: { index: 1, id: toolCallId2 } })

      expect(results.find(c => c.type === 'message_stop')).toEqual({
        type: 'message_stop',
        data: { finishReason: 'tool_calls' }
      })
      expect(results.find(c => c.type === 'final_usage')).toBeDefined()
      expect(results.find(c => c.type === 'final_result')).toBeDefined()

      const finalResult = (results.find(c => c.type === 'final_result') as any).data.result
      expect(finalResult.toolCalls).toHaveLength(2)
      expect(finalResult.toolCalls).toEqual(
        expect.arrayContaining([
          { id: toolCallId1, type: 'function', function: { name: toolName1, arguments: '{"p1": 1}' } },
          { id: toolCallId2, type: 'function', function: { name: toolName2, arguments: '{"p2": 2}' } }
        ])
      )
    })

    it('[Medium] should handle stream where final parsedContent should be null (non-JSON)', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { content: 'Just text' }, finish_reason: null, logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('message_stop')
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      const finalResult = (results[4] as any).data.result
      expect(finalResult.content).toBe('Just text')
      expect(finalResult.parsedContent).toBeNull() // Ensure parsedContent is null
    })

    // FIX: Add top_logprobs: null to mock logprobs data
    it('[Hard] should handle stream with logprobs gracefully', async () => {
      const mockChunks: ChatCompletionChunk[] = [
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: '' },
              finish_reason: null,
              logprobs: { content: [{ token: '<|im_start|>', logprob: -0.1, bytes: null, top_logprobs: null }] } // Logprobs in first chunk
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { content: 'Text' },
              finish_reason: null,
              logprobs: { content: [{ token: 'Text', logprob: -0.2, bytes: null, top_logprobs: null }] } // Logprobs in delta chunk
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 }
        }
      ]
      // FIX: Cast the mock generator
      const stream = mapOpenAIStream((mockOpenAIStreamGenerator(mockChunks) as any) as Stream<ChatCompletionChunk>)
      const results = await collectStreamChunks(stream)

      // Ensure stream processes without error and yields expected core chunks
      expect(results).toHaveLength(5) // start, delta, stop, usage, final
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('message_stop')
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      // We don't map logprobs, just ensure it doesn't crash
    })
  })
  // --- END mapOpenAIStream Tests ---
})
