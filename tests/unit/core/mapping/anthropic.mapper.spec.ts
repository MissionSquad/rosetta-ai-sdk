import { AnthropicMapper } from '../../../../src/core/mapping/anthropic.mapper'
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
import { MappingError, ProviderAPIError, RosettaAIError, UnsupportedFeatureError } from '../../../../src/errors'
import Anthropic from '@anthropic-ai/sdk'
import { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages'

// Helper to create mock Anthropic Message with required fields
const createMockAnthropicMessage = (
  content: Anthropic.ContentBlock[],
  stopReason: Anthropic.Messages.Message['stop_reason'],
  usage: { input_tokens: number; output_tokens: number },
  model: string = 'claude-3-haiku-20240307'
): Anthropic.Messages.Message => ({
  id: `msg_${Date.now()}`,
  type: 'message',
  role: 'assistant',
  content: content,
  model: model,
  stop_reason: stopReason,
  stop_sequence: null,
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
  citations: null
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
  signature: null
})

// Helper async generator for stream tests
async function* mockAnthropicStreamGenerator(events: RawMessageStreamEvent[]): AsyncIterable<RawMessageStreamEvent> {
  for (const event of events) {
    await new Promise(resolve => setTimeout(resolve, 1)) // Simulate delay
    yield event
  }
}

// Helper async generator that throws an error
async function* mockAnthropicErrorStreamGenerator(
  events: RawMessageStreamEvent[],
  errorToThrow: Error
): AsyncIterable<RawMessageStreamEvent> {
  for (const event of events) {
    await new Promise(resolve => setTimeout(resolve, 1))
    yield event
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

describe('Anthropic Mapper', () => {
  let mapper: AnthropicMapper

  beforeEach(() => {
    mapper = new AnthropicMapper()
  })

  it('[Easy] should have the correct provider property', () => {
    expect(mapper.provider).toBe(Provider.Anthropic)
  })

  describe('mapToProviderParams', () => {
    const baseParams: GenerateParams = {
      provider: Provider.Anthropic,
      model: 'claude-3-haiku-20240307',
      messages: [{ role: 'user', content: 'Placeholder' }] // Add placeholder message
    }

    it('[Easy] should map basic text messages and system prompt', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'You are Claude.' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there.' }
        ]
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.system).toBe('You are Claude.')
      expect(result.messages).toEqual([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there.' }
      ])
      expect(result.model).toBe('claude-3-haiku-20240307')
      expect(result.stream).toBeUndefined() // Should not be present for non-streaming
    })

    it('[Easy] should map user message with text and image', () => {
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
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
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

    it('[Easy] should map tool result message correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Use the tool.' },
          {
            role: 'assistant',
            content: 'Okay, using the tool.',
            toolCalls: [{ id: 'tool_123', type: 'function', function: { name: 'my_tool', arguments: '{"p":1}' } }]
          },
          { role: 'tool', toolCallId: 'tool_123', content: '{"result": "ok"}' }
        ]
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([
        { role: 'user', content: 'Use the tool.' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Okay, using the tool.' },
            { type: 'tool_use', id: 'tool_123', name: 'my_tool', input: { p: 1 } }
          ]
        },
        {
          role: 'user', // Tool result mapped to user role
          content: [{ type: 'tool_result', tool_use_id: 'tool_123', content: '{"result": "ok"}' }]
        }
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
              parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] }
            }
          }
        ]
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tools).toEqual([
        {
          name: 'get_weather',
          description: 'Gets weather',
          input_schema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] }
        }
      ])
    })

    it('[Easy] should set stream flag correctly', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Stream this.' }],
        stream: true
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsStreaming
      expect(result.stream).toBe(true)
    })

    it('[Easy] should map temperature, topP, and stop_sequences', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate.' }],
        temperature: 0.7,
        topP: 0.8,
        stop: ['\n', 'Human:']
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.temperature).toBe(0.7)
      expect(result.top_p).toBe(0.8)
      expect(result.stop_sequences).toEqual(['\n', 'Human:'])
    })

    it('[Easy] should map toolChoice auto and none', () => {
      // FIX: Add a user message to avoid the "No messages" error
      const paramsAuto: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Test' }],
        toolChoice: 'auto'
      }
      const paramsNone: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Test' }],
        toolChoice: 'none'
      }
      const resultAuto = mapper.mapToProviderParams(paramsAuto) as Anthropic.Messages.MessageCreateParamsNonStreaming
      const resultNone = mapper.mapToProviderParams(paramsNone) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(resultAuto.tool_choice).toEqual({ type: 'auto' })
      expect(resultNone.tool_choice).toEqual({ type: 'none' })
    })

    it('[Medium] should map toolChoice required to any', () => {
      // FIX: Add a user message
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Test' }],
        toolChoice: 'required'
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tool_choice).toEqual({ type: 'any' })
    })

    it('[Medium] should map toolChoice for a specific tool', () => {
      // FIX: Add a user message
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Test' }],
        toolChoice: { type: 'function', function: { name: 'my_specific_tool' } }
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.tool_choice).toEqual({ type: 'tool', name: 'my_specific_tool' })
    })

    it('[Medium] should map thinking parameter', () => {
      // FIX: Add a user message
      const params: GenerateParams = { ...baseParams, messages: [{ role: 'user', content: 'Test' }], thinking: true }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.thinking).toEqual({ type: 'enabled', budget_tokens: 1024 })
    })

    it('[Medium] should throw MappingError for multiple system messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Sys 1' },
          { role: 'system', content: 'Sys 2' },
          { role: 'user', content: 'Hello' } // Keep user message
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Multiple system messages not supported by Anthropic.')
    })

    it('[Medium] should throw MappingError if system message content is not string', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: [{ type: 'text', text: 'Sys 1' }] },
          { role: 'user', content: 'Hello' } // Keep user message
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Anthropic system prompt must be string.')
    })

    it('[Medium] should throw MappingError for invalid tool parameters schema', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Use the tool.' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Gets weather',
              parameters: { properties: { location: { type: 'string' } } } // Missing type: object
            }
          }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        "Invalid parameters schema for tool 'get_weather'. Anthropic requires a JSON Schema object with top-level 'type: \"object\"'."
      )
    })

    it('[Medium] should throw MappingError for invalid tool result message (missing toolCallId)', () => {
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
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        'Invalid tool result message format for Anthropic. Requires toolCallId and string content.'
      )
    })

    it('[Medium] should throw MappingError for invalid tool result message (non-string content)', () => {
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
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        'Invalid tool result message format for Anthropic. Requires toolCallId and string content.'
      )
    })

    it('[Hard] should map assistant message with only tool calls (null content)', () => {
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
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      expect(result.messages).toEqual([
        { role: 'user', content: 'Call the tool.' },
        {
          role: 'assistant',
          content: [{ type: 'tool_use', id: 'tool_abc', name: 'my_tool', input: {} }]
        }
      ])
    })

    it('[Hard] should map assistant message with empty string content and tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call the tool.' },
          {
            role: 'assistant',
            content: '', // Empty string content
            toolCalls: [{ id: 'tool_abc', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsNonStreaming
      // FIX: Expectation updated based on code fix - empty text block should NOT be included
      expect(result.messages).toEqual([
        { role: 'user', content: 'Call the tool.' },
        {
          role: 'assistant',
          content: [
            // No text block expected here
            { type: 'tool_use', id: 'tool_abc', name: 'my_tool', input: {} }
          ]
        }
      ])
    })
  })

  describe('mapFromProviderResponse', () => {
    const modelUsed = 'claude-3-opus-20240229'

    it('[Easy] should map basic text response', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Response text.')],
        'end_turn',
        { input_tokens: 10, output_tokens: 5 },
        'claude-3-opus-20240229-test'
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Response text.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
      // FIX: Update usage expectation
      expect(result.usage).toEqual({
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 15,
        cachedContentTokenCount: undefined
      })
      expect(result.model).toBe('claude-3-opus-20240229-test')
    })

    it('[Easy] should map response with tool calls', () => {
      const response = createMockAnthropicMessage(
        [
          createMockTextBlock('Okay, using the tool.'),
          createMockToolUseBlock('toolu_abc', 'get_weather', { location: 'London' })
        ],
        'tool_use',
        { input_tokens: 20, output_tokens: 15 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Okay, using the tool.')
      expect(result.toolCalls).toEqual([
        { id: 'toolu_abc', type: 'function', function: { name: 'get_weather', arguments: '{"location":"London"}' } }
      ])
      expect(result.finishReason).toBe('tool_calls')
      // FIX: Update usage expectation
      expect(result.usage).toEqual({
        promptTokens: 20,
        completionTokens: 15,
        totalTokens: 35,
        cachedContentTokenCount: undefined
      })
      expect(result.model).toBe(modelUsed)
    })

    it('[Easy] should map max_tokens finish reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('This is cut off')],
        'max_tokens',
        { input_tokens: 5, output_tokens: 100 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('This is cut off')
      expect(result.finishReason).toBe('length')
      expect(result.usage?.completionTokens).toBe(100)
    })

    it('[Easy] should handle null content in response', () => {
      const response = createMockAnthropicMessage([], 'end_turn', { input_tokens: 5, output_tokens: 0 }, modelUsed)
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('stop')
    })

    it('[Easy] should map stop_sequence finish reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Stopped by sequence.')],
        'stop_sequence',
        { input_tokens: 10, output_tokens: 4 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.finishReason).toBe('stop')
    })

    it('[Easy] should handle null stop_reason', () => {
      const response = createMockAnthropicMessage(
        [createMockTextBlock('Response')],
        null,
        { input_tokens: 10, output_tokens: 4 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.finishReason).toBe('unknown')
    })

    it('[Medium] should map response containing thinking block', () => {
      const response = createMockAnthropicMessage(
        [createMockThinkingBlock('Thinking process steps...'), createMockTextBlock('Final Answer.')],
        'end_turn',
        { input_tokens: 10, output_tokens: 5 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Final Answer.')
      expect(result.thinkingSteps).toBe('Thinking process steps...')
      expect(result.finishReason).toBe('stop')
    })

    it('[Medium] should handle response with only tool calls (no text)', () => {
      const response = createMockAnthropicMessage(
        [createMockToolUseBlock('toolu_xyz', 'another_tool', {})],
        'tool_use',
        { input_tokens: 12, output_tokens: 8 },
        modelUsed
      )
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toEqual([
        { id: 'toolu_xyz', type: 'function', function: { name: 'another_tool', arguments: '{}' } }
      ])
      expect(result.finishReason).toBe('tool_calls')
    })
  })

  describe('mapProviderStream', () => {
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
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 0, cache_creation_input_tokens: null, cache_read_input_tokens: null }
      }
    }

    it('[Medium] should handle basic text streaming', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ' world' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 2 } },
        { type: 'message_stop' }
      ]

      const stream = mapper.mapProviderStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final_result
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Anthropic, model: modelId } })
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Hello' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: ' world' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      // FIX: Update usage expectation to include calculated totalTokens
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: {
          usage: { promptTokens: 10, completionTokens: 2, totalTokens: 12, cachedContentTokenCount: undefined }
        }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Hello world',
          finishReason: 'stop',
          model: modelId,
          // FIX: Update usage expectation to include calculated totalTokens
          usage: { promptTokens: 10, completionTokens: 2, totalTokens: 12, cachedContentTokenCount: undefined }
        })
      )
    })

    it('[Medium] should handle stream ending with max_tokens', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Too long' } },
        { type: 'content_block_stop', index: 0 },
        {
          type: 'message_delta',
          delta: { stop_reason: 'max_tokens', stop_sequence: null },
          usage: { output_tokens: 2 }
        },
        { type: 'message_stop' }
      ]

      const stream = mapper.mapProviderStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final_result
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'length' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('length')
    })

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
        { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 5 } },
        { type: 'message_stop' }
      ]

      const stream = mapper.mapProviderStream(mockAnthropicStreamGenerator(events))
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

    it('[Hard] should handle thinking steps streaming', async () => {
      const events: RawMessageStreamEvent[] = [
        baseMessageStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: null } },
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Step 1...' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Step 2.' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '', citations: null } },
        { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Answer.' } },
        { type: 'content_block_stop', index: 1 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 3 } },
        { type: 'message_stop' }
      ]

      const stream = mapper.mapProviderStream(mockAnthropicStreamGenerator(events))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(9) // start, thinking_start, delta, delta, thinking_stop, delta, stop, usage, final
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
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } }
      ]

      const stream = mapper.mapProviderStream(mockAnthropicErrorStreamGenerator(events, apiError))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      const errorChunk = results[2] as { type: 'error'; data: { error: Error } }
      expect(errorChunk.data.error).toBeInstanceOf(ProviderAPIError)
      expect(errorChunk.data.error.message).toContain('Internal failure')
      expect((errorChunk.data.error as ProviderAPIError).provider).toBe(Provider.Anthropic)
      expect((errorChunk.data.error as ProviderAPIError).statusCode).toBe(500)
    })
  })

  describe('Unsupported Features', () => {
    const dummyEmbedParams: EmbedParams = { provider: Provider.Anthropic, input: 'test' }
    const dummyTranscribeParams: TranscribeParams = {
      provider: Provider.Anthropic,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' }
    }
    const dummyTranslateParams: TranslateParams = {
      provider: Provider.Anthropic,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' }
    }

    it('[Easy] should throw UnsupportedFeatureError for mapToEmbedParams', () => {
      expect(() => mapper.mapToEmbedParams(dummyEmbedParams)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToEmbedParams(dummyEmbedParams)).toThrow(
        "Provider 'anthropic' does not support the requested feature: Embeddings"
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for mapFromEmbedResponse', () => {
      expect(() => mapper.mapFromEmbedResponse({}, 'model')).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapFromEmbedResponse({}, 'model')).toThrow(
        "Provider 'anthropic' does not support the requested feature: Embeddings"
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for mapToTranscribeParams', () => {
      expect(() => mapper.mapToTranscribeParams(dummyTranscribeParams, {})).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToTranscribeParams(dummyTranscribeParams, {})).toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Transcription"
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for mapFromTranscribeResponse', () => {
      expect(() => mapper.mapFromTranscribeResponse({}, 'model')).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapFromTranscribeResponse({}, 'model')).toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Transcription"
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for mapToTranslateParams', () => {
      expect(() => mapper.mapToTranslateParams(dummyTranslateParams, {})).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToTranslateParams(dummyTranslateParams, {})).toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Translation"
      )
    })

    it('[Easy] should throw UnsupportedFeatureError for mapFromTranslateResponse', () => {
      expect(() => mapper.mapFromTranslateResponse({}, 'model')).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapFromTranslateResponse({}, 'model')).toThrow(
        "Provider 'anthropic' does not support the requested feature: Audio Translation"
      )
    })
  })

  describe('wrapProviderError', () => {
    it('[Easy] should wrap Anthropic APIError', () => {
      const underlying = new Anthropic.APIError(
        429,
        { error: { type: 'rate_limit_error', message: 'Limit exceeded' } },
        'Rate Limit',
        {}
      )
      const wrapped = mapper.wrapProviderError(underlying, Provider.Anthropic)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Anthropic)
      expect(wrapped.statusCode).toBe(429)
      expect(wrapped.errorCode).toBe('rate_limit_error')
      expect(wrapped.errorType).toBe('rate_limit_error')
      expect(wrapped.message).toContain('Limit exceeded')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap generic Error', () => {
      const underlying = new Error('Generic network failure')
      const wrapped = mapper.wrapProviderError(underlying, Provider.Anthropic)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Anthropic)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Generic network failure')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap unknown/string error', () => {
      const underlying = 'Something went wrong'
      const wrapped = mapper.wrapProviderError(underlying, Provider.Anthropic)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Anthropic)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Something went wrong')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should not re-wrap RosettaAIError', () => {
      const underlying = new MappingError('Already mapped', Provider.Anthropic)
      const wrapped = mapper.wrapProviderError(underlying, Provider.Anthropic)
      expect(wrapped).toBe(underlying) // Should return the original error instance
      expect(wrapped).toBeInstanceOf(MappingError)
    })
  })
})
