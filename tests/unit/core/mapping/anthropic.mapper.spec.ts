import {
  mapToAnthropicParams,
  mapFromAnthropicResponse,
  mapAnthropicStream // Import stream mapper
} from '../../../../src/core/mapping/anthropic.mapper'
import { GenerateParams, Provider, RosettaImageData, StreamChunk } from '../../../../src/types'
import { MappingError, ProviderAPIError } from '../../../../src/errors' // Import ProviderAPIError
import Anthropic from '@anthropic-ai/sdk'
import { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages'

// Helper to create mock Anthropic Message with required fields
const createMockAnthropicMessage = (
  content: Anthropic.ContentBlock[],
  stopReason: Anthropic.Messages.Message['stop_reason'],
  usage: { input_tokens: number; output_tokens: number },
  model: string = 'claude-3-haiku-20240307'
  // Removed thinking parameter as it's part of content blocks now
): Anthropic.Messages.Message => ({
  id: `msg_${Date.now()}`,
  type: 'message',
  role: 'assistant',
  content: content,
  model: model,
  stop_reason: stopReason,
  stop_sequence: null, // Add required property
  usage: {
    ...usage,
    cache_creation_input_tokens: null,
    cache_read_input_tokens: null
  }
})

// Helper to create mock TextBlock with required fields
const createMockTextBlock = (text: string): Anthropic.TextBlock => ({
  type: 'text',
  text: text,
  citations: null // Add required property
})

// Helper to create mock ToolUseBlock
const createMockToolUseBlock = (id: string, name: string, input: any): Anthropic.ToolUseBlock => ({
  id: id,
  type: 'tool_use',
  name: name,
  input: input
})

// Helper to create mock ThinkingBlock
const createMockThinkingBlock = (thinking: string): Anthropic.ThinkingBlock => ({
  type: 'thinking',
  thinking: thinking,
  signature: null // Add required property
})

// Helper async generator for stream tests
async function* mockAnthropicStreamGenerator(events: RawMessageStreamEvent[]): AsyncIterable<RawMessageStreamEvent> {
  for (const event of events) {
    await new Promise(resolve => setTimeout(resolve, 1)) // Simulate delay
    yield event
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

describe('Anthropic Mapper', () => {
  describe('mapToAnthropicParams', () => {
    const baseParams: GenerateParams = {
      provider: Provider.Anthropic,
      model: 'claude-3-haiku-20240307',
      messages: []
    }

    it('should map basic text messages and system prompt', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'You are Claude.' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there.' }
        ]
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.system).toBe('You are Claude.')
      expect(result.messages).toEqual([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there.' }
      ])
      expect(result.model).toBe('claude-3-haiku-20240307')
      expect(result.stream).toBeUndefined() // Should not be present for non-streaming
    })

    it('should map user message with text and image', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'imgdata' }
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Describe this image:' },
              { type: 'image', image: imageData }
            ]
          }
        ]
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this image:' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'imgdata' } }
          ]
        }
      ])
    })

    // Test Fixed: Expect assistant message with tool_use block to be included
    it('should map tool result message correctly (including preceding assistant message)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Use the tool.' },
          {
            role: 'assistant',
            content: 'Okay, using the tool.', // Text content
            toolCalls: [{ id: 'tool_123', type: 'function', function: { name: 'my_tool', arguments: '{"p":1}' } }]
          },
          { role: 'tool', toolCallId: 'tool_123', content: '{"result": "ok"}' }
        ]
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([
        { role: 'user', content: 'Use the tool.' },
        // Assistant message with text and tool_use block IS included now
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Okay, using the tool.' },
            { type: 'tool_use', id: 'tool_123', name: 'my_tool', input: { p: 1 } }
          ]
        },
        // Tool result message mapped to user role with tool_result block
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'tool_123',
              content: '{"result": "ok"}'
            }
          ]
        }
      ])
    })

    it('should map tools correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Use the tool.' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Gets weather',
              parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] }
            }
          }
        ]
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tools).toEqual([
        {
          name: 'get_weather',
          description: 'Gets weather',
          input_schema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] }
        }
      ])
    })

    it('should throw MappingError for invalid tool parameters schema (missing type: object)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Use the tool.' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Gets weather',
              parameters: { properties: { location: { type: 'string' } }, required: ['location'] }
            }
          }
        ]
      }
      expect(() => mapToAnthropicParams(params)).toThrow(MappingError)
      expect(() => mapToAnthropicParams(params)).toThrow(
        "Invalid parameters schema for tool 'get_weather'. Anthropic requires a JSON Schema object with top-level 'type: \"object\"'."
      )
    })

    it('should set stream flag correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Stream this.' }],
        stream: true
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsStreaming
      expect(result.stream).toBe(true)
    })

    it('should throw MappingError for multiple system messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Sys 1' },
          { role: 'system', content: 'Sys 2' }
        ]
      }
      expect(() => mapToAnthropicParams(params)).toThrow(MappingError)
      expect(() => mapToAnthropicParams(params)).toThrow('Multiple system messages not supported by Anthropic.')
    })

    it('should throw MappingError if system message content is not string', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'system', content: [{ type: 'text', text: 'Sys 1' }] }]
      }
      expect(() => mapToAnthropicParams(params)).toThrow(MappingError)
      expect(() => mapToAnthropicParams(params)).toThrow('Anthropic system prompt must be string.')
    })

    // Test Fixed: Assistant message with tool calls is now included
    it('should map assistant message with only tool calls (null content)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call the tool.' },
          {
            role: 'assistant',
            content: null, // Null content
            toolCalls: [{ id: 'tool_abc', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
          }
        ]
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([
        { role: 'user', content: 'Call the tool.' },
        // Assistant message IS included, content array only has tool_use block
        {
          role: 'assistant',
          content: [{ type: 'tool_use', id: 'tool_abc', name: 'my_tool', input: {} }]
        }
      ])
    })

    // --- New Tests (Easy) ---
    it('should map temperature, topP, and stop_sequences', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.8,
        stop: ['\n', 'Human:']
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.temperature).toBe(0.7)
      expect(result.top_p).toBe(0.8)
      expect(result.stop_sequences).toEqual(['\n', 'Human:'])
    })

    it('should map toolChoice auto and none', () => {
      const paramsAuto: GenerateParams = { ...baseParams, messages: [], toolChoice: 'auto' }
      const paramsNone: GenerateParams = { ...baseParams, messages: [], toolChoice: 'none' }
      const resultAuto = mapToAnthropicParams(paramsAuto) as Anthropic.Messages.MessageCreateParamsNonStreaming
      const resultNone = mapToAnthropicParams(paramsNone) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(resultAuto.tool_choice).toEqual({ type: 'auto' })
      expect(resultNone.tool_choice).toEqual({ type: 'none' })
    })

    it('should map empty message array', () => {
      const params: GenerateParams = { ...baseParams, messages: [] }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([])
    })

    // --- New Tests (Medium) ---
    it('should map toolChoice required', () => {
      const params: GenerateParams = { ...baseParams, messages: [], toolChoice: 'required' }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tool_choice).toEqual({ type: 'any' })
    })

    it('should map toolChoice for a specific tool', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [],
        toolChoice: { type: 'function', function: { name: 'my_specific_tool' } }
      }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tool_choice).toEqual({ type: 'tool', name: 'my_specific_tool' })
    })

    // Test Fixed: Expect budget_tokens to be 1024
    it('should map thinking parameter with correct budget_tokens', () => {
      const params: GenerateParams = { ...baseParams, messages: [], thinking: true }
      const result = mapToAnthropicParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.thinking).toEqual({ type: 'enabled', budget_tokens: 1024 }) // Check for 1024
    })

    it('should throw MappingError for invalid tool result message (missing toolCallId)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Use tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 't', arguments: '{}' } }]
          },
          { role: 'tool', content: '{"res": 1}' } // Missing toolCallId
        ]
      }
      expect(() => mapToAnthropicParams(params)).toThrow(MappingError)
      expect(() => mapToAnthropicParams(params)).toThrow(
        'Invalid tool result message format for Anthropic. Requires toolCallId and string content.'
      )
    })

    it('should throw MappingError for invalid tool result message (non-string content)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Use tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 't', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 't1', content: [{ type: 'text', text: 'Invalid' }] } // Non-string content
        ]
      }
      expect(() => mapToAnthropicParams(params)).toThrow(MappingError)
      expect(() => mapToAnthropicParams(params)).toThrow(
        'Invalid tool result message format for Anthropic. Requires toolCallId and string content.'
      )
    })
  })

  describe('mapFromAnthropicResponse', () => {
    const modelUsed = 'claude-3-opus-20240229'

    it('should map basic text response', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Response text.')],
        'end_turn',
        { input_tokens: 10, output_tokens: 5 },
        'claude-3-opus-20240229-test'
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.content).toBe('Response text.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop') // 'end_turn' maps to 'stop'
      expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 })
      expect(result.model).toBe('claude-3-opus-20240229-test') // Use model from response
    })

    it('should map response with tool calls', () => {
      const response = createMockAnthropicMessage(
        [
          createMockTextBlock('Okay, using the tool.'),
          createMockToolUseBlock('toolu_abc', 'get_weather', { location: 'London' })
        ],
        'tool_use',
        { input_tokens: 20, output_tokens: 15 },
        modelUsed
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.content).toBe('Okay, using the tool.') // Text content is preserved
      expect(result.toolCalls).toEqual([
        { id: 'toolu_abc', type: 'function', function: { name: 'get_weather', arguments: '{"location":"London"}' } }
      ])
      expect(result.finishReason).toBe('tool_calls')
      expect(result.usage).toEqual({ promptTokens: 20, completionTokens: 15, totalTokens: 35 })
      expect(result.model).toBe(modelUsed)
    })

    it('should map max_tokens finish reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('This is cut off')],
        'max_tokens',
        { input_tokens: 5, output_tokens: 100 },
        modelUsed
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.content).toBe('This is cut off')
      expect(result.finishReason).toBe('length') // 'max_tokens' maps to 'length'
      expect(result.usage?.completionTokens).toBe(100)
    })

    it('should handle null content in response', () => {
      const response = createMockAnthropicMessage([], 'end_turn', { input_tokens: 5, output_tokens: 0 }, modelUsed)
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.content).toBeNull() // Should map empty content to null
      expect(result.finishReason).toBe('stop')
    })

    // --- New Tests (Easy) ---
    it('should map stop_sequence finish reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Stopped by sequence.')],
        'stop_sequence',
        { input_tokens: 10, output_tokens: 4 },
        modelUsed
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.finishReason).toBe('stop')
    })

    it('should handle null stop_reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Response')],
        null, // Null stop_reason
        { input_tokens: 10, output_tokens: 4 },
        modelUsed
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.finishReason).toBe('unknown')
    })

    // --- New Tests (Medium) ---
    // FIX: Update test to place thinking block inside content
    it('should map response containing thinking block', () => {
      const response = createMockAnthropicMessage(
        [
          createMockThinkingBlock('Thinking process steps...'), // Thinking block in content
          createMockTextBlock('Final Answer.')
        ],
        'end_turn',
        { input_tokens: 10, output_tokens: 5 },
        modelUsed
      )
      const result = mapFromAnthropicResponse(response, modelUsed)
      expect(result.content).toBe('Final Answer.')
      expect(result.thinkingSteps).toBe('Thinking process steps...') // Mapper should extract this
      expect(result.finishReason).toBe('stop')
    })

    // Test for citations deferred until feature implementation/confirmation
    // it('should map response containing citations', () => { ... });
  })

  // --- New Tests (mapAnthropicStream) ---
  describe('mapAnthropicStream', () => {
    const modelId = 'claude-3-stream-test'
    const baseMessageStart: Anthropic.Messages.MessageStartEvent = {
      type: 'message_start',
      message: {
        id: 'msg_stream_123',
        type: 'message',
        role: 'assistant',
        content: [],
        model: modelId,
        stop_reason: null,
        stop_sequence: null, // Add required property
        usage: { input_tokens: 10, output_tokens: 0, cache_creation_input_tokens: null, cache_read_input_tokens: null }
      }
    }

    // FIX: Update mock data to include required properties
    it('[Medium] should handle basic text streaming', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } }, // Add citations
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ' world' } },
        { type: 'content_block_stop', index: 0 },
        {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null }, // Add stop_sequence
          usage: { output_tokens: 2 }
        },
        { type: 'message_stop' }
      ]

      const stream = mapAnthropicStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 6 chunks now
      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final_result
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Anthropic, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Hello' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: ' world' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 10, completionTokens: 2, totalTokens: 12 } }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Hello world',
          finishReason: 'stop',
          model: modelId,
          usage: { promptTokens: 10, completionTokens: 2, totalTokens: 12 }
        })
      )
    })

    // FIX: Update mock data to include required properties
    it('[Medium] should handle stream ending with max_tokens', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } }, // Add citations
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Too long' } },
        { type: 'content_block_stop', index: 0 },
        {
          type: 'message_delta',
          delta: { stop_reason: 'max_tokens', stop_sequence: null }, // Add stop_sequence
          usage: { output_tokens: 2 }
        },
        { type: 'message_stop' }
      ]

      const stream = mapAnthropicStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 5 chunks now
      expect(results).toHaveLength(5) // start, delta, stop, usage, final_result
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Anthropic, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Too long' } })
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'length' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('length')
      expect((results[4] as any).data.result.content).toBe('Too long')
    })

    // --- Hard Tests ---
    // FIX: Update mock data to include required properties
    it('[Hard] should handle tool call streaming', async () => {
      const toolCallId = 'toolu_stream_abc'
      const toolName = 'stream_tool'
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'tool_use', id: toolCallId, name: toolName, input: {} }
        },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"arg":' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: ' 123}' } },
        { type: 'content_block_stop', index: 0 },
        {
          type: 'message_delta',
          delta: { stop_reason: 'tool_use', stop_sequence: null }, // Add stop_sequence
          usage: { output_tokens: 5 }
        },
        { type: 'message_stop' }
      ]

      const stream = mapAnthropicStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, tool_start, delta, delta, tool_done, stop, usage, final_result
      expect(results[1]).toEqual({
        type: 'tool_call_start',
        data: { index: 0, toolCall: { id: toolCallId, type: 'function', function: { name: toolName } } }
      })
      expect(results[2]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: '{"arg":' }
      })
      expect(results[3]).toEqual({
        type: 'tool_call_delta',
        data: { index: 0, id: toolCallId, functionArgumentChunk: ' 123}' }
      })
      expect(results[4]).toEqual({ type: 'tool_call_done', data: { index: 0, id: toolCallId } })
      expect(results[5]).toEqual({ type: 'message_stop', data: { finishReason: 'tool_calls' } })
      expect(results[6].type).toBe('final_usage')
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result
      expect(finalResult.content).toBeNull()
      expect(finalResult.toolCalls).toEqual([
        { id: toolCallId, type: 'function', function: { name: toolName, arguments: '{"arg": 123}' } }
      ])
    })

    // FIX: Update mock data to include required properties
    it('[Hard] should handle thinking steps streaming', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: null } }, // Add signature
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Step 1...' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Step 2.' } },
        { type: 'content_block_stop', index: 0 },
        // Assume text content follows thinking
        { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '', citations: null } }, // Add citations
        { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Answer.' } },
        { type: 'content_block_stop', index: 1 },
        {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null }, // Add stop_sequence
          usage: { output_tokens: 3 }
        },
        { type: 'message_stop' }
      ]

      const stream = mapAnthropicStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      // Expect: start, thinking_start, thinking_delta, thinking_delta, thinking_stop, delta, stop, usage, final
      expect(results).toHaveLength(9)
      expect(results[1]).toEqual({ type: 'thinking_start' })
      expect(results[2]).toEqual({ type: 'thinking_delta', data: { delta: 'Step 1...' } })
      expect(results[3]).toEqual({ type: 'thinking_delta', data: { delta: 'Step 2.' } })
      expect(results[4]).toEqual({ type: 'thinking_stop' })
      expect(results[5]).toEqual({ type: 'content_delta', data: { delta: 'Answer.' } })
      expect(results[6]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[7].type).toBe('final_usage')
      expect(results[8].type).toBe('final_result')
      const finalResult = (results[8] as any).data.result
      expect(finalResult.content).toBe('Answer.')
      expect(finalResult.thinkingSteps).toBe('Step 1...Step 2.')
    })

    it('[Hard] should handle stream error', async () => {
      const apiError = new Anthropic.APIError(
        500,
        { error: { type: 'server_error', message: 'Internal failure' } },
        'Server Error',
        {}
      )
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } }, // Add citations
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } }
        // Error occurs here
      ]

      // Create a generator that throws after yielding initial events
      async function* errorGenerator(): AsyncIterable<RawMessageStreamEvent> {
        for (const event of events) {
          yield event
        }
        throw apiError
      }

      const stream = mapAnthropicStream(errorGenerator())
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      const errorChunk = results[2] as { type: 'error'; data: { error: Error } }
      // FIX: Expect ProviderAPIError now
      expect(errorChunk.data.error).toBeInstanceOf(ProviderAPIError)
      expect(errorChunk.data.error.message).toContain('Internal failure')
    })
  })
})
