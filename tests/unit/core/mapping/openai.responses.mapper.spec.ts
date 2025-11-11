/**
 * Unit tests for OpenAI Responses API mapper
 */

import { describe, it, expect } from '@jest/globals'
import {
  mapToOpenAIResponsesParams,
  mapFromOpenAIResponsesResponse,
  mapOpenAIResponsesStream
} from '../../../../src/core/mapping/openai.responses.mapper'
import {
  CreateResponseParams,
  ResponsesTool,
  ResponseResult,
  ResponsesStreamChunk
} from '../../../../src/types/responses.types'
import { Provider } from '../../../../src/types/common.types'
import { MappingError, InvalidToolDefinitionError, ToolArgumentValidationError } from '../../../../src/errors'
import { z } from 'zod'

describe('OpenAI Responses Mapper', () => {
  describe('mapToOpenAIResponsesParams', () => {
    it('should map basic parameters correctly', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        instructions: 'You are a helpful assistant',
        input: 'Hello, how are you?',
        temperature: 0.7,
        max_tokens: 100
      }

      const mapped = mapToOpenAIResponsesParams(params)

      expect(mapped).toMatchObject({
        model: 'gpt-4o',
        instructions: 'You are a helpful assistant',
        input: 'Hello, how are you?',
        temperature: 0.7,
        max_tokens: 100,
        stream: false
      })
    })

    it('should map multimodal input items', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: [
          { type: 'input_text', text: 'What is in this image?' },
          {
            type: 'input_image',
            image: {
              mimeType: 'image/jpeg',
              base64Data: 'base64encodeddata'
            }
          }
        ]
      }

      const mapped = mapToOpenAIResponsesParams(params)

      expect(mapped.input).toHaveLength(2)
      expect(mapped.input[0]).toEqual({ type: 'input_text', text: 'What is in this image?' })
      expect(mapped.input[1]).toEqual({
        type: 'input_image',
        image_url: 'data:image/jpeg;base64,base64encodeddata'
      })
    })

    it('should map built-in tools', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'Search for TypeScript tutorials',
        tools: [
          { type: 'web_search' },
          { type: 'code_interpreter' },
          {
            type: 'image_generation',
            options: { size: '1024x1024', quality: 'hd' }
          }
        ]
      }

      const mapped = mapToOpenAIResponsesParams(params)

      expect(mapped.tools).toHaveLength(3)
      expect(mapped.tools[0]).toEqual({ type: 'web_search' })
      expect(mapped.tools[1]).toEqual({ type: 'code_interpreter' })
      expect(mapped.tools[2]).toEqual({
        type: 'image_generation',
        options: { size: '1024x1024', quality: 'hd' }
      })
    })

    it('should map custom function tools', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'Get weather for San Francisco',
        tools: [
          {
            type: 'function',
            name: 'getWeather',
            description: 'Get current weather',
            parameters: {
              type: 'object',
              properties: {
                location: { type: 'string' }
              },
              required: ['location']
            },
            zodSchema: z.object({ location: z.string() })
          }
        ]
      }

      const mapped = mapToOpenAIResponsesParams(params)

      expect(mapped.tools).toHaveLength(1)
      expect(mapped.tools[0]).toEqual({
        type: 'function',
        name: 'getWeather',
        description: 'Get current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' }
          },
          required: ['location']
        }
      })
    })

    it('should map tool_choice correctly', () => {
      const params1: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'test',
        tool_choice: 'auto'
      }

      const mapped1 = mapToOpenAIResponsesParams(params1)
      expect(mapped1.tool_choice).toBe('auto')

      const params2: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'test',
        tool_choice: { type: 'web_search' }
      }

      const mapped2 = mapToOpenAIResponsesParams(params2)
      expect(mapped2.tool_choice).toEqual({ type: 'web_search' })
    })

    it('should map response_format for structured outputs', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'Extract package info',
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'PackageInfo',
            strict: true,
            schema: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                version: { type: 'string' }
              },
              required: ['name', 'version']
            }
          }
        }
      }

      const mapped = mapToOpenAIResponsesParams(params)

      expect(mapped.response_format).toEqual({
        type: 'json_schema',
        json_schema: {
          name: 'PackageInfo',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              version: { type: 'string' }
            },
            required: ['name', 'version']
          }
        }
      })
    })

    it('should include previous_response_id for stateful conversations', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'Continue the conversation',
        previous_response_id: 'resp_abc123'
      }

      const mapped = mapToOpenAIResponsesParams(params)
      expect(mapped.previous_response_id).toBe('resp_abc123')
    })

    it('should throw error for invalid tool definition', () => {
      const params: CreateResponseParams = {
        provider: Provider.OpenAI,
        model: 'gpt-4o',
        input: 'test',
        tools: [
          {
            type: 'function',
            name: '',
            parameters: {} as any,
            zodSchema: z.object({})
          }
        ]
      }

      expect(() => mapToOpenAIResponsesParams(params)).toThrow(InvalidToolDefinitionError)
    })
  })

  describe('mapFromOpenAIResponsesResponse', () => {
    it('should map basic response correctly', () => {
      const rawResponse = {
        id: 'resp_123',
        model: 'gpt-4o',
        output: [
          { type: 'text', text: 'Hello! I am doing well, thank you for asking.' }
        ],
        output_text: 'Hello! I am doing well, thank you for asking.',
        usage: {
          input_tokens: 10,
          output_tokens: 15,
          total_tokens: 25
        },
        finish_reason: 'stop'
      }

      const result = mapFromOpenAIResponsesResponse(rawResponse)

      expect(result).toMatchObject({
        id: 'resp_123',
        model: 'gpt-4o',
        output_text: 'Hello! I am doing well, thank you for asking.',
        usage: {
          input_tokens: 10,
          output_tokens: 15,
          total_tokens: 25
        },
        finish_reason: 'stop'
      })
      expect(result.output).toHaveLength(1)
      expect(result.output[0]).toEqual({
        type: 'output_text',
        text: 'Hello! I am doing well, thank you for asking.'
      })
    })

    it('should map response with tool calls', () => {
      const rawResponse = {
        id: 'resp_123',
        model: 'gpt-4o',
        output: [],
        output_text: '',
        tool_calls: [
          {
            id: 'call_abc',
            function: {
              name: 'getWeather',
              arguments: '{"location":"San Francisco"}'
            }
          }
        ],
        usage: {
          input_tokens: 20,
          output_tokens: 5,
          total_tokens: 25
        }
      }

      const result = mapFromOpenAIResponsesResponse(rawResponse)

      expect(result.tool_calls).toHaveLength(1)
      expect(result.tool_calls![0]).toEqual({
        id: 'call_abc',
        type: 'function',
        function: {
          name: 'getWeather',
          arguments: '{"location":"San Francisco"}'
        }
      })
    })

    it('should validate tool call arguments with Zod schema', () => {
      const rawResponse = {
        id: 'resp_123',
        model: 'gpt-4o',
        output: [],
        output_text: '',
        tool_calls: [
          {
            id: 'call_abc',
            function: {
              name: 'getWeather',
              arguments: '{"location":"San Francisco"}'
            }
          }
        ]
      }

      const tools: ResponsesTool[] = [
        {
          type: 'function',
          name: 'getWeather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' }
            },
            required: ['location']
          },
          zodSchema: z.object({ location: z.string() })
        }
      ]

      // Should not throw - valid arguments
      const result = mapFromOpenAIResponsesResponse(rawResponse, tools)
      expect(result.tool_calls).toHaveLength(1)
    })

    it('should throw ToolArgumentValidationError for invalid tool arguments', () => {
      const rawResponse = {
        id: 'resp_123',
        model: 'gpt-4o',
        output: [],
        output_text: '',
        tool_calls: [
          {
            id: 'call_abc',
            function: {
              name: 'getWeather',
              arguments: '{"location":123}'
            }
          }
        ]
      }

      const tools: ResponsesTool[] = [
        {
          type: 'function',
          name: 'getWeather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' }
            },
            required: ['location']
          },
          zodSchema: z.object({ location: z.string() })
        }
      ]

      expect(() => mapFromOpenAIResponsesResponse(rawResponse, tools)).toThrow(ToolArgumentValidationError)
    })

    it('should handle response with multiple output types', () => {
      const rawResponse = {
        id: 'resp_123',
        model: 'gpt-4o',
        output: [
          { type: 'text', text: 'Here is an image:' },
          { type: 'image', image_url: 'https://example.com/image.png' }
        ],
        output_text: 'Here is an image:',
        usage: {
          input_tokens: 10,
          output_tokens: 20,
          total_tokens: 30
        }
      }

      const result = mapFromOpenAIResponsesResponse(rawResponse)

      expect(result.output).toHaveLength(2)
      expect(result.output[0]).toEqual({ type: 'output_text', text: 'Here is an image:' })
      expect(result.output[1]).toEqual({ type: 'image', image_url: 'https://example.com/image.png' })
    })
  })

  describe('mapOpenAIResponsesStream', () => {
    it('should map semantic streaming events', async () => {
      // Mock stream with semantic events
      const mockStream = (async function* () {
        yield { type: 'response.created', response: { id: 'resp_123', model: 'gpt-4o' } }
        yield { type: 'response.output_text.delta', delta: 'Hello' }
        yield { type: 'response.output_text.delta', delta: ' world' }
        yield { type: 'response.output_text.done', text: 'Hello world' }
        yield {
          type: 'response.completed',
          response: {
            id: 'resp_123',
            model: 'gpt-4o',
            output: [{ type: 'text', text: 'Hello world' }],
            output_text: 'Hello world',
            usage: { input_tokens: 5, output_tokens: 2, total_tokens: 7 }
          }
        }
      })()

      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(mockStream)) {
        chunks.push(chunk)
      }

      expect(chunks).toHaveLength(5)
      expect(chunks[0].type).toBe('response.created')
      expect(chunks[1].type).toBe('response.output_text.delta')
      expect(chunks[2].type).toBe('response.output_text.delta')
      expect(chunks[3].type).toBe('response.output_text.done')
      expect(chunks[4].type).toBe('response.completed')
    })

    it('should map tool call streaming events', async () => {
      const mockStream = (async function* () {
        yield { type: 'response.created', response: { id: 'resp_123', model: 'gpt-4o' } }
        yield { type: 'response.tool_call.start', tool_call: { id: 'call_abc', name: 'getWeather' } }
        yield { type: 'response.tool_call.delta', tool_call: { id: 'call_abc' }, delta: '{"location"' }
        yield { type: 'response.tool_call.delta', tool_call: { id: 'call_abc' }, delta: ':"SF"}' }
        yield {
          type: 'response.tool_call.done',
          tool_call: { id: 'call_abc', name: 'getWeather', arguments: '{"location":"SF"}' }
        }
      })()

      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(mockStream)) {
        chunks.push(chunk)
      }

      expect(chunks).toHaveLength(5)
      expect(chunks[0].type).toBe('response.created')
      expect(chunks[1].type).toBe('response.tool_call.start')
      expect(chunks[2].type).toBe('response.tool_call.delta')
      expect(chunks[3].type).toBe('response.tool_call.delta')
      expect(chunks[4].type).toBe('response.tool_call.done')
    })

    it('should yield error event for failed response', async () => {
      const mockStream = (async function* () {
        yield { type: 'response.created', response: { id: 'resp_123', model: 'gpt-4o' } }
        yield { type: 'response.failed', error: { message: 'Rate limit exceeded', code: 'rate_limit' } }
      })()

      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(mockStream)) {
        chunks.push(chunk)
      }

      expect(chunks).toHaveLength(2)
      expect(chunks[1].type).toBe('response.failed')
      expect((chunks[1] as any).data.error.message).toBe('Rate limit exceeded')
    })

    it('should validate tool arguments during streaming', async () => {
      const tools: ResponsesTool[] = [
        {
          type: 'function',
          name: 'getWeather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' }
            },
            required: ['location']
          },
          zodSchema: z.object({ location: z.string() })
        }
      ]

      const mockStream = (async function* () {
        yield { type: 'response.created', response: { id: 'resp_123', model: 'gpt-4o' } }
        yield { type: 'response.tool_call.start', tool_call: { id: 'call_abc', name: 'getWeather' } }
        yield {
          type: 'response.tool_call.done',
          tool_call: { id: 'call_abc', name: 'getWeather', arguments: '{"location":123}' }
        }
      })()

      const chunks: ResponsesStreamChunk[] = []
      for await (const chunk of mapOpenAIResponsesStream(mockStream, tools)) {
        chunks.push(chunk)
      }

      // Should yield error chunk for invalid arguments
      const errorChunk = chunks.find(c => c.type === 'error')
      expect(errorChunk).toBeDefined()
    })
  })
})
