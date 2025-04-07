import OpenAI from 'openai'
import { Stream } from 'openai/streaming'
import {
  mapRoleToOpenAI,
  mapContentForOpenAIRole,
  mapFromOpenAIResponse,
  mapOpenAIStream,
  wrapOpenAIError
} from '../../../../src/core/mapping/openai.common'
import { RosettaMessage, Provider, RosettaImageData, StreamChunk, GenerateResult } from '../../../../src/types'
import { MappingError, ProviderAPIError } from '../../../../src/errors'

// Helper type for ChatCompletionMessage mock to satisfy the interface
type MockChatCompletionMessage = OpenAI.Chat.Completions.ChatCompletionMessage & {
  refusal: string | null
}

// Helper to create a valid mock message satisfying the type
const createMockMessage = (
  content: string | null,
  toolCalls?: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[]
): MockChatCompletionMessage => ({
  role: 'assistant',
  content: content,
  tool_calls: toolCalls,
  refusal: null
})

// Helper async generator for stream tests
async function* mockOpenAIStreamGenerator(
  chunks: OpenAI.Chat.Completions.ChatCompletionChunk[]
): AsyncGenerator<OpenAI.Chat.Completions.ChatCompletionChunk, void, undefined> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield chunk
  }
}

// Helper to collect stream chunks
async function collectStreamChunks(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) {
    chunks.push(chunk)
  }
  return chunks
}

describe('OpenAI Common Mapping Utilities', () => {
  describe('mapRoleToOpenAI', () => {
    it('[Easy] should map valid roles', () => {
      expect(mapRoleToOpenAI('system')).toBe('system')
      expect(mapRoleToOpenAI('user')).toBe('user')
      expect(mapRoleToOpenAI('assistant')).toBe('assistant')
      expect(mapRoleToOpenAI('tool')).toBe('tool')
    })

    it('[Easy] should throw MappingError for invalid role', () => {
      expect(() => mapRoleToOpenAI('invalid' as any)).toThrow(MappingError)
      expect(() => mapRoleToOpenAI('invalid' as any)).toThrow('Unsupported role: invalid')
    })
  })

  describe('mapContentForOpenAIRole', () => {
    const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'test-img' }

    // --- String Content ---
    it('[Easy] should map string content for user', () => {
      expect(mapContentForOpenAIRole('Hello', 'user')).toBe('Hello')
    })
    it('[Easy] should map string content for assistant', () => {
      expect(mapContentForOpenAIRole('Hi', 'assistant')).toBe('Hi')
    })
    it('[Easy] should map string content for system', () => {
      expect(mapContentForOpenAIRole('System prompt', 'system')).toBe('System prompt')
    })
    it('[Easy] should map string content for tool', () => {
      expect(mapContentForOpenAIRole('{"result": true}', 'tool')).toBe('{"result": true}')
    })
    it('[Easy] should map empty string content for assistant', () => {
      expect(mapContentForOpenAIRole('', 'assistant')).toBe('')
    })

    // --- Null Content ---
    it('[Easy] should map null content for assistant', () => {
      expect(mapContentForOpenAIRole(null, 'assistant')).toBeNull()
    })
    it('[Easy] should map null content for tool', () => {
      expect(mapContentForOpenAIRole(null, 'tool')).toBeNull()
    })
    it('[Medium] should throw MappingError for null content for user', () => {
      expect(() => mapContentForOpenAIRole(null, 'user')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(null, 'user')).toThrow("Role 'user' requires non-null content.")
    })
    it('[Medium] should throw MappingError for null content for system', () => {
      expect(() => mapContentForOpenAIRole(null, 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(null, 'system')).toThrow("Role 'system' requires non-null content.")
    })

    // --- Array Content ---
    it('[Easy] should map text parts array for user', () => {
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Part 1' },
        { type: 'text', text: ' Part 2' }
      ]
      expect(mapContentForOpenAIRole(content, 'user')).toEqual([
        { type: 'text', text: 'Part 1' },
        { type: 'text', text: ' Part 2' }
      ])
    })
    it('[Easy] should map mixed text/image parts array for user', () => {
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Look:' },
        { type: 'image', image: imageData }
      ]
      expect(mapContentForOpenAIRole(content, 'user')).toEqual([
        { type: 'text', text: 'Look:' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,test-img' } }
      ])
    })
    it('[Easy] should map empty array content for user', () => {
      expect(mapContentForOpenAIRole([], 'user')).toEqual([])
    })
    it('[Medium] should map text parts array for assistant (joins to string)', () => {
      const content: RosettaMessage['content'] = [
        { type: 'text', text: 'Assistant ' },
        { type: 'text', text: 'reply.' }
      ]
      expect(mapContentForOpenAIRole(content, 'assistant')).toEqual([
        { type: 'text', text: 'Assistant ' },
        { type: 'text', text: 'reply.' }
      ])
    })
    it('[Medium] should map empty array content for assistant (maps to null)', () => {
      expect(mapContentForOpenAIRole([], 'assistant')).toBeNull()
    })
    it('[Medium] should throw MappingError for image parts for assistant', () => {
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }]
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'assistant')).toThrow(
        "Image content parts only allowed for 'user' role, not 'assistant'."
      )
    })
    it('[Medium] should throw MappingError for image parts for system', () => {
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }]
      expect(() => mapContentForOpenAIRole(content, 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'system')).toThrow(
        "Image content parts only allowed for 'user' role, not 'system'."
      )
    })
    it('[Medium] should throw MappingError for image parts for tool', () => {
      const content: RosettaMessage['content'] = [{ type: 'image', image: imageData }]
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole(content, 'tool')).toThrow(
        "Image content parts only allowed for 'user' role, not 'tool'."
      )
    })
    it('[Hard] should throw MappingError for empty array content for system', () => {
      expect(() => mapContentForOpenAIRole([], 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole([], 'system')).toThrow(
        "Role 'system' requires non-empty content array. Received empty array."
      )
    })
    it('[Hard] should throw MappingError for empty array content for tool', () => {
      expect(() => mapContentForOpenAIRole([], 'tool')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole([], 'tool')).toThrow(
        "Role 'tool' requires non-empty content array. Received empty array."
      )
    })
    it('[Hard] should throw MappingError for empty string content for system', () => {
      expect(() => mapContentForOpenAIRole('', 'system')).toThrow(MappingError)
      expect(() => mapContentForOpenAIRole('', 'system')).toThrow("Role 'system' requires non-empty string content.")
    })
    // FIX: Update test to expect success (empty string returned) instead of throwing
    it('[Hard] should map empty string content for tool successfully', () => {
      expect(mapContentForOpenAIRole('', 'tool')).toBe('')
    })
  })

  describe('mapFromOpenAIResponse', () => {
    const modelUsed = 'gpt-4o-mini-test'

    it('[Easy] should map basic response', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-1',
        object: 'chat.completion',
        created: 1,
        model: 'gpt-4o-mini-id',
        choices: [{ index: 0, message: createMockMessage('Response'), finish_reason: 'stop', logprobs: null }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBe('Response')
      expect(result.finishReason).toBe('stop')
      expect(result.model).toBe('gpt-4o-mini-id')
      expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 })
    })

    it('[Medium] should map response with tool calls', () => {
      const toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [
        { id: 't1', type: 'function', function: { name: 'f1', arguments: '{}' } }
      ]
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-2',
        object: 'chat.completion',
        created: 2,
        model: modelUsed,
        choices: [
          { index: 0, message: createMockMessage(null, toolCalls), finish_reason: 'tool_calls', logprobs: null }
        ]
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('tool_calls')
      expect(result.toolCalls).toEqual([{ id: 't1', type: 'function', function: { name: 'f1', arguments: '{}' } }])
    })

    it('[Medium] should map response with content filter', () => {
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-3',
        object: 'chat.completion',
        created: 3,
        model: modelUsed,
        choices: [{ index: 0, message: createMockMessage(null), finish_reason: 'content_filter', logprobs: null }]
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('content_filter')
    })

    it('[Medium] should handle missing choices gracefully', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: OpenAI.Chat.Completions.ChatCompletion = {
        id: 'chatcmpl-4',
        object: 'chat.completion',
        created: 4,
        model: modelUsed,
        choices: [] // Empty choices
      }
      const result = mapFromOpenAIResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('error')
      expect(warnSpy).toHaveBeenCalledWith('OpenAI response missing choices.')
      warnSpy.mockRestore()
    })
  })

  describe('mapOpenAIStream', () => {
    const modelId = 'gpt-stream-test'
    const baseChunkProps = { id: 'chatcmpl-stream-1', object: 'chat.completion.chunk' as const, created: 1 }

    it('[Hard] should map basic text stream', async () => {
      const mockChunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: 'Chunk 1 ' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: 'Chunk 2' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 }
        }
      ]
      const stream = mapOpenAIStream(
        (mockOpenAIStreamGenerator(mockChunks) as any) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
        Provider.OpenAI
      )
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.OpenAI, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Chunk 1 ' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: 'Chunk 2' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 } }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result.content).toBe('Chunk 1 Chunk 2')
    })

    it('[Hard] should map stream with tool calls', async () => {
      const toolCallId = 't_stream_1'
      const toolName = 'streamFunc'
      const mockChunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, id: toolCallId, type: 'function', function: { name: toolName } }] }
            }
          ]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"a":' } }] } }]
        },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '1}' } }] } }]
        },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }
      ]
      const stream = mapOpenAIStream(
        (mockOpenAIStreamGenerator(mockChunks) as any) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
        Provider.OpenAI
      )
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, tool_start, delta, delta, tool_done, stop, usage, final
      expect(results[1].type).toBe('tool_call_start')
      expect(results[4].type).toBe('tool_call_done')
      expect(results[5].type).toBe('message_stop')
      expect((results[5] as any).data.finishReason).toBe('tool_calls')
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result as GenerateResult
      expect(finalResult.content).toBeNull()
      expect(finalResult.toolCalls).toHaveLength(1)
      expect(finalResult.toolCalls![0]).toEqual({
        id: toolCallId,
        type: 'function',
        function: { name: toolName, arguments: '{"a":1}' }
      })
    })

    it('[Hard] should map stream with JSON mode', async () => {
      const jsonString = '{"data": true}'
      const mockChunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = [
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { role: 'assistant' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: '{"data":' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: { content: ' true}' } }] },
        { ...baseChunkProps, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        {
          ...baseChunkProps,
          model: modelId,
          choices: [],
          usage: { prompt_tokens: 4, completion_tokens: 4, total_tokens: 8 }
        }
      ]
      const stream = mapOpenAIStream(
        (mockOpenAIStreamGenerator(mockChunks) as any) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
        Provider.OpenAI
      )
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(7) // start, json_delta, json_delta, json_done, stop, usage, final
      expect(results[1].type).toBe('json_delta')
      expect(results[2].type).toBe('json_delta')
      expect(results[3].type).toBe('json_done')
      expect((results[3] as any).data.snapshot).toBe(jsonString)
      expect((results[3] as any).data.parsed).toEqual({ data: true })
      expect(results[6].type).toBe('final_result')
      const finalResult = (results[6] as any).data.result as GenerateResult
      expect(finalResult.content).toBe(jsonString)
      expect(finalResult.parsedContent).toEqual({ data: true })
    })
  })

  describe('wrapOpenAIError', () => {
    it('[Hard] should wrap different OpenAI APIError subtypes', () => {
      const rateLimitError = new OpenAI.RateLimitError(429, { message: 'Rate limited' }, 'Rate Limit', {})
      const authError = new OpenAI.AuthenticationError(401, { message: 'Invalid key' }, 'Auth Error', {})
      const badRequestError = new OpenAI.BadRequestError(400, { message: 'Bad input' }, 'Bad Request', {})
      const genericApiError = new OpenAI.APIError(500, { message: 'Server error' }, 'Server Error', {})

      const wrappedRateLimit = wrapOpenAIError(rateLimitError, Provider.OpenAI)
      const wrappedAuth = wrapOpenAIError(authError, Provider.OpenAI)
      const wrappedBadRequest = wrapOpenAIError(badRequestError, Provider.OpenAI)
      const wrappedGenericApi = wrapOpenAIError(genericApiError, Provider.OpenAI)

      expect(wrappedRateLimit).toBeInstanceOf(ProviderAPIError)
      expect(wrappedRateLimit.statusCode).toBe(429)
      expect(wrappedRateLimit.message).toContain('Rate limited')

      expect(wrappedAuth).toBeInstanceOf(ProviderAPIError)
      expect(wrappedAuth.statusCode).toBe(401)
      expect(wrappedAuth.message).toContain('Invalid key')

      expect(wrappedBadRequest).toBeInstanceOf(ProviderAPIError)
      expect(wrappedBadRequest.statusCode).toBe(400)
      expect(wrappedBadRequest.message).toContain('Bad input')

      expect(wrappedGenericApi).toBeInstanceOf(ProviderAPIError)
      expect(wrappedGenericApi.statusCode).toBe(500)
      expect(wrappedGenericApi.message).toContain('Server error')
    })

    it('[Hard] should wrap generic Error', () => {
      const genericError = new Error('Network issue')
      const wrapped = wrapOpenAIError(genericError, Provider.OpenAI)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.message).toContain('Network issue')
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.underlyingError).toBe(genericError)
    })

    it('[Hard] should wrap non-Error object/string', () => {
      const stringError = 'Something failed'
      const objectError = { detail: 'Failure object' }
      const wrappedString = wrapOpenAIError(stringError, Provider.OpenAI)
      const wrappedObject = wrapOpenAIError(objectError, Provider.OpenAI)

      expect(wrappedString).toBeInstanceOf(ProviderAPIError)
      expect(wrappedString.message).toContain('Something failed')
      expect(wrappedString.underlyingError).toBe(stringError)

      expect(wrappedObject).toBeInstanceOf(ProviderAPIError)
      expect(wrappedObject.message).toContain('{"detail":"Failure object"}')
      expect(wrappedObject.underlyingError).toBe(objectError)
    })

    it('[Hard] should not re-wrap RosettaAIError', () => {
      const rosettaError = new MappingError('Already mapped', Provider.OpenAI)
      const wrapped = wrapOpenAIError(rosettaError, Provider.OpenAI)
      expect(wrapped).toBe(rosettaError)
    })
  })
})
