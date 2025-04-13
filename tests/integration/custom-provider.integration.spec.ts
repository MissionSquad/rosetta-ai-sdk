import http from 'http'
import { z } from 'zod'
import {
  RosettaAI,
  RosettaMessage,
  RosettaTool,
  GenerateParams,
  StreamChunk,
  CustomProviderConfig,
  GenerateResult,
  ProviderAPIError,
  ToolArgumentValidationError,
  MappingError
} from '../../src'
import { BaseCustomMapper } from '../../src/core/mapping/base.custom.mapper'
import {
  startMockApiServer,
  stopMockApiServer,
  getLastGenerateRequest,
  getLastStreamRequest
} from '../mocks/mock-custom-api'

const MOCK_API_PORT = 3031 // Use a different port for tests
const MOCK_API_URL = `http://localhost:${MOCK_API_PORT}`

// --- Mock Custom Mapper Implementation ---
class MockApiClientMapper extends BaseCustomMapper {
  // Override executeGenerate to call the mock API
  async executeGenerate(
    _mappedParams: any, // We'll use originalParams directly for simplicity here
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): Promise<GenerateResult> {
    if (!apiKey) throw new Error('API key required for mock custom provider')

    const url = `${providerConfig.baseURL}/generate`
    console.log(`[MockApiClientMapper] Calling mock API: POST ${url}`)

    const requestBody = {
      model: originalParams.model ?? providerConfig.defaultModel,
      messages: originalParams.messages,
      max_tokens: originalParams.maxTokens,
      temperature: originalParams.temperature,
      tools: originalParams.tools?.map(t => ({
        // Simulate sending tool definitions if needed by API
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters // Send JSON schema part
      })),
      stream: false
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({ error: 'Failed to parse error response' }))
      throw new ProviderAPIError(
        `Mock API Error: ${errorBody.error || response.statusText}`,
        this.provider,
        response.status
      )
    }

    const apiResponse = await response.json()

    // --- Map API Response to GenerateResult ---
    const choice = apiResponse.choices?.[0]
    if (!choice) {
      throw new Error('Invalid response structure from mock API')
    }

    // Basic mapping (adapt based on actual mock API response structure)
    const content = choice.message?.content ?? null
    const finishReason = choice.finish_reason ?? 'unknown'
    const usage = apiResponse.usage
      ? {
          promptTokens: apiResponse.usage.prompt_tokens,
          completionTokens: apiResponse.usage.completion_tokens,
          totalTokens: apiResponse.usage.total_tokens
        }
      : undefined

    // Map and Validate Tool Calls
    const toolCalls = choice.message?.tool_calls
      ?.map((tc: any) => {
        if (tc.type !== 'function') return null // Ignore non-function calls

        // Parse arguments based on config (default: jsonString)
        const parsedArgs = this.parseToolArguments(tc.function.arguments, tc.function.name, tc.id)

        // Validate arguments using helper
        this.validateToolArguments({ name: tc.function.name, arguments: parsedArgs, id: tc.id }, originalParams.tools)

        // Return RosettaToolCallRequest format (with raw string args)
        return {
          id: tc.id,
          type: 'function',
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments // Keep raw string
          }
        }
      })
      ?.filter((tc: any) => tc !== null) // Filter out nulls

    return {
      content: content,
      toolCalls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
      finishReason: finishReason,
      usage: usage,
      model: apiResponse.model, // Use model from response
      rawResponse: apiResponse
    }
  }

  // Override executeStream to call the mock API's SSE endpoint
  async *executeStream(
    _mappedParams: any,
    apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): AsyncIterable<StreamChunk> {
    if (!apiKey) throw new Error('API key required for mock custom provider')

    const url = `${providerConfig.baseURL}/stream`
    console.log(`[MockApiClientMapper] Calling mock API: POST ${url} (SSE)`)

    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null // Read raw bytes
    const decoder = new TextDecoder() // Decoder for UTF-8
    let buffer = ''

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
          Accept: 'text/event-stream'
        },
        body: JSON.stringify({
          model: originalParams.model ?? providerConfig.defaultModel,
          messages: originalParams.messages,
          max_tokens: originalParams.maxTokens,
          temperature: originalParams.temperature,
          tools: originalParams.tools?.map(t => ({
            name: t.function.name,
            description: t.function.description,
            input_schema: t.function.parameters
          })),
          stream: true
        })
        // Note: Fetch doesn't automatically handle AbortController for SSE in Node < 16
        // For real implementations, consider using libraries like 'eventsource' or node-fetch with AbortSignal
      })

      if (!response.ok || !response.body) {
        const errorBody = await response.json().catch(() => ({ error: 'Failed to parse error response' }))
        throw new ProviderAPIError(
          `Mock API Stream Error: ${errorBody.error || response.statusText}`,
          this.provider,
          response.status
        )
      }

      reader = response.body.getReader() // Get the raw byte reader
      const toolCallArgAccumulators: Record<string, string> = {} // Accumulate args per tool call ID
      let lastReceivedToolCallName: string | undefined = undefined // Store name for validation

      while (true) {
        console.log('[DEBUG executeStream] Waiting for reader.read()...') // Log loop start
        const { value, done } = await reader.read()
        console.log(`[DEBUG executeStream] reader.read() returned: done=${done}, value length=${value?.length ?? 0}`) // Log read result

        if (done) {
          console.log('[MockApiClientMapper] SSE stream finished (reader done).')
          break
        }
        if (!value) {
          console.log('[DEBUG executeStream] Received empty value from reader, continuing.')
          continue
        }

        // Decode the Uint8Array chunk to string and append to buffer
        buffer += decoder.decode(value, { stream: true }) // Use stream: true for multi-byte chars
        console.log(`[DEBUG executeStream] Buffer after decode: "${buffer.replace(/\n/g, '\\n')}"`) // Log buffer

        let boundaryIndex: number
        // Process all complete messages in the buffer
        while ((boundaryIndex = buffer.indexOf('\n\n')) !== -1) {
          const message = buffer.substring(0, boundaryIndex)
          buffer = buffer.substring(boundaryIndex + 2) // Remove message and boundary from buffer
          console.log(`[DEBUG executeStream] Processing message: "${message.replace(/\n/g, '\\n')}"`) // Log message

          let eventType = 'message'
          let eventData = ''
          let retry: number | undefined = undefined

          // Parse the SSE message lines
          message.split('\n').forEach(line => {
            if (line.startsWith('event:')) {
              eventType = line.substring(6).trim()
            } else if (line.startsWith('data:')) {
              // Append data, handling potential multi-line data fields by adding back newline
              eventData += (eventData ? '\n' : '') + line.substring(5).trim()
            } else if (line.startsWith('retry:')) {
              const retryValue = parseInt(line.substring(6).trim(), 10)
              if (!isNaN(retryValue)) {
                retry = retryValue
              }
            } else if (line.startsWith('id:')) {
              // ID field - ignore for this mock
            } else if (line.trim() === '') {
              // Empty line - ignore
            } else {
              // console.warn(`[MockApiClientMapper] Ignoring unknown SSE line: ${line}`)
            }
          })

          console.log(`[DEBUG executeStream] Parsed event - Type: ${eventType}, Data: "${eventData}", Retry: ${retry}`) // DEBUG

          if (eventData) {
            if (eventType === 'chunk') {
              try {
                const parsedData = JSON.parse(eventData)
                console.log(`[DEBUG executeStream] Parsed chunk data:`, parsedData) // Log parsed data

                // Basic validation that it looks like a StreamChunk
                if (typeof parsedData !== 'object' || !parsedData || !('type' in parsedData)) {
                  console.error('[MockApiClientMapper] Invalid data format for event type "chunk":', parsedData)
                  throw new Error('Invalid chunk data format received from mock API')
                }

                const chunkData = parsedData as StreamChunk // Cast after basic check

                // Store tool name when start chunk arrives
                if (chunkData.type === 'tool_call_start') {
                  lastReceivedToolCallName = chunkData.data.toolCall.function.name
                }

                // --- Tool Argument Validation during Stream ---
                if (chunkData.type === 'tool_call_delta') {
                  toolCallArgAccumulators[chunkData.data.id] =
                    (toolCallArgAccumulators[chunkData.data.id] ?? '') + chunkData.data.functionArgumentChunk
                } else if (chunkData.type === 'tool_call_done') {
                  const toolCallId = chunkData.data.id
                  const accumulatedArgs = toolCallArgAccumulators[toolCallId] ?? '{}'
                  const toolDefinition = originalParams.tools?.find(t => t.function.name === lastReceivedToolCallName)

                  if (toolDefinition) {
                    let parsedArgsForValidation: any
                    try {
                      parsedArgsForValidation = JSON.parse(accumulatedArgs)
                    } catch (parseError) {
                      throw new MappingError(
                        `Failed to parse accumulated arguments for tool call ${toolCallId}`,
                        this.provider as string,
                        'executeStream tool validation',
                        parseError
                      )
                    }
                    // Use originalParams.tools for validation
                    this.validateToolArguments(
                      { name: toolDefinition.function.name, arguments: parsedArgsForValidation, id: toolCallId },
                      originalParams.tools // Pass tools from originalParams
                    )
                    console.log(`[DEBUG executeStream] Tool args validated for ${toolCallId}`) // DEBUG
                  } else {
                    console.warn(
                      `[${this.provider}] Could not find definition for tool call ID ${toolCallId} during stream validation.`
                    )
                  }
                  delete toolCallArgAccumulators[toolCallId]
                  lastReceivedToolCallName = undefined
                }
                // --- End Tool Argument Validation ---

                console.log(`[DEBUG executeStream] YIELDING chunkData:`, chunkData) // Log before yield
                yield chunkData // <--- THE YIELD
                console.log(`[DEBUG executeStream] AFTER YIELDING chunkData`) // Log after yield
              } catch (e) {
                console.error(
                  '[MockApiClientMapper] Error parsing or processing "chunk" event data:',
                  e,
                  'Data:',
                  eventData
                )
                // Yield an error chunk instead of throwing directly from the generator
                const mappingError = new MappingError(
                  `Failed to parse or process stream chunk data: ${(e as Error).message}`,
                  this.provider as string,
                  'executeStream chunk processing',
                  e
                )
                console.log(`[DEBUG executeStream] YIELDING ERROR chunk:`, mappingError) // Log error yield
                yield {
                  type: 'error',
                  data: { error: mappingError }
                }
                console.log(`[DEBUG executeStream] AFTER YIELDING ERROR chunk`) // Log after error yield
                // Optionally break or return here if you want to stop processing on the first chunk error
                // return;
              }
            } else if (eventType === 'error') {
              try {
                const parsedData = JSON.parse(eventData)
                console.error('[MockApiClientMapper] Received error event:', parsedData)
                const providerError = new ProviderAPIError(
                  parsedData?.error?.message || 'Unknown stream error event',
                  this.provider as string
                )
                console.log(`[DEBUG executeStream] YIELDING ERROR event:`, providerError) // Log error yield
                yield {
                  type: 'error',
                  data: { error: providerError }
                }
                console.log(`[DEBUG executeStream] AFTER YIELDING ERROR event`) // Log after error yield
              } catch (e) {
                console.error('[MockApiClientMapper] Error parsing "error" event data:', e, 'Data:', eventData)
                const mappingError = new MappingError(
                  'Failed to parse stream error event',
                  this.provider as string,
                  'executeStream error event parsing',
                  e
                )
                console.log(`[DEBUG executeStream] YIELDING ERROR chunk (parse fail):`, mappingError) // Log error yield
                yield {
                  type: 'error',
                  data: { error: mappingError }
                }
                console.log(`[DEBUG executeStream] AFTER YIELDING ERROR chunk (parse fail)`) // Log after error yield
              }
              // Optionally break or return here if you want to stop processing on the first error event
              // return;
            } else if (eventType === 'end') {
              console.log('[MockApiClientMapper] Received end event, returning.')
              if (buffer.trim()) {
                console.warn('[MockApiClientMapper] End event received with unprocessed buffer:', buffer)
              }
              return // End the generator
            } else {
              // console.warn(`[MockApiClientMapper] Received unhandled SSE event type: ${eventType}`)
            }
          }
        } // end while boundary
      } // end while true
    } catch (error) {
      console.error('[MockApiClientMapper] Stream execution error (outer catch):', error)
      // Wrap and yield the error
      const wrappedError = this.wrapProviderError(error, this.provider)
      console.log(`[DEBUG executeStream] YIELDING WRAPPED ERROR (outer catch):`, wrappedError) // Log outer error yield
      yield { type: 'error', data: { error: wrappedError } }
      console.log(`[DEBUG executeStream] AFTER YIELDING WRAPPED ERROR (outer catch)`) // Log after outer error yield
    } finally {
      // Ensure the reader is cancelled if the loop exits unexpectedly
      if (reader) {
        // Decode any final bytes in the buffer before cancelling
        if (buffer) {
          decoder.decode(undefined, { stream: false }) // Finalize decoding
          console.warn('[MockApiClientMapper] Final unprocessed buffer content:', buffer)
        }
        reader.cancel().catch(e => console.error('[MockApiClientMapper] Error cancelling stream reader:', e))
      }
      console.log('[MockApiClientMapper] executeStream finished (finally block).') // DEBUG
    }
  }
}

// --- Test Setup ---
const mockProviderKey = 'mock-custom-provider'
const mockApiKey = 'mock-api-key-123'

const mockProviderConfig: CustomProviderConfig = {
  providerKey: mockProviderKey,
  mapper: MockApiClientMapper, // Use the mapper that calls the API
  supportedFeatures: ['generate', 'stream', 'tool_use'],
  baseURL: MOCK_API_URL,
  apiKey: mockApiKey,
  defaultModel: 'mock-default-model',
  toolConfig: {
    toolDefinitionFormat: 'jsonSchema',
    toolCallInputFormat: 'jsonString',
    toolResultFormat: 'jsonString'
  }
}

// --- Test Suite ---
describe('Custom Provider Integration Tests', () => {
  let server: http.Server
  let rosetta: RosettaAI

  beforeAll(async () => {
    server = await startMockApiServer(MOCK_API_PORT)
    rosetta = new RosettaAI({ customProviders: [mockProviderConfig] })
  })

  afterAll(async () => {
    await stopMockApiServer(server)
  })

  beforeEach(() => {
    // Reset last requests before each test if needed (or rely on mock server reset)
  })

  it('should register the custom provider', () => {
    expect(rosetta.getConfiguredProviders()).toContain(mockProviderKey)
  })

  it('should successfully call generate via the custom provider', async () => {
    const params: GenerateParams = {
      provider: mockProviderKey,
      messages: [{ role: 'user', content: 'Hello from test' }],
      model: 'test-model-override' // Override default
    }

    const result = await rosetta.generate(params)

    expect(result).toBeDefined()
    expect(result.content).toContain('Mock response to: Hello from test')
    expect(result.model).toBe('test-model-override') // Mock API should echo model
    expect(result.finishReason).toBe('stop')
    expect(result.usage?.totalTokens).toBe(15)

    // Verify the request received by the mock API
    const lastRequest = getLastGenerateRequest()
    expect(lastRequest).toBeDefined()
    expect(lastRequest.model).toBe('test-model-override')
    expect(lastRequest.messages).toEqual([{ role: 'user', content: 'Hello from test' }])
  })

  it('should successfully call stream via the custom provider', async () => {
    const params: GenerateParams = {
      provider: mockProviderKey,
      messages: [{ role: 'user', content: 'Stream test' }]
    }

    const stream = rosetta.stream(params)
    let receivedChunks = 0
    let firstChunkType: string | null = null
    let deltaCount = 0
    let stopReason: string | null = null
    let usage: any = null
    let model: string | null = null

    try {
      for await (const chunk of stream) {
        console.log('[Direct Test Loop - Simple Stream] Received chunk:', chunk.type) // DEBUG
        receivedChunks++
        if (receivedChunks === 1) {
          firstChunkType = chunk.type
          if (chunk.type === 'message_start') model = chunk.data.model
        }
        if (chunk.type === 'content_delta') deltaCount++
        if (chunk.type === 'message_stop') stopReason = chunk.data.finishReason
        if (chunk.type === 'final_usage') usage = chunk.data.usage
        if (chunk.type === 'error') throw chunk.data.error // Re-throw error to fail test
      }
    } catch (error) {
      console.error('Error processing stream directly in test:', error)
      throw error // Fail test on error
    }

    console.log(`[Direct Test Loop - Simple Stream] Finished. Total chunks: ${receivedChunks}`) // DEBUG
    // Check counts (adjust based on mock API streamChunks array)
    // message_start, delta x 3, stop, usage = 6 chunks
    expect(receivedChunks).toBeGreaterThanOrEqual(6)
    expect(firstChunkType).toBe('message_start')
    expect(model).toBe('mock-custom-stream-model')
    expect(deltaCount).toBe(3)
    expect(stopReason).toBe('stop')
    expect(usage?.totalTokens).toBe(9)

    // Verify the request received by the mock API
    const lastRequest = getLastStreamRequest()
    expect(lastRequest).toBeDefined()
    expect(lastRequest.model).toBe('mock-default-model') // Used default
    expect(lastRequest.messages).toEqual([{ role: 'user', content: 'Stream test' }])
    expect(lastRequest.stream).toBe(true)
  })

  describe('Tool Use Integration', () => {
    const weatherTool: RosettaTool = {
      type: 'function',
      function: {
        name: 'get_current_weather',
        description: 'Gets the current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'City and state/country' },
            unit: { type: 'string', enum: ['celsius', 'fahrenheit'] }
          },
          required: ['location']
        },
        zodSchema: z.object({
          location: z.string(),
          unit: z.enum(['celsius', 'fahrenheit']).optional()
        })
      }
    }

    it('should handle tool call request in generate', async () => {
      const params: GenerateParams = {
        provider: mockProviderKey,
        messages: [{ role: 'user', content: "What's the weather in Test City?" }],
        tools: [weatherTool]
      }

      const result = await rosetta.generate(params)

      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('tool_calls')
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls![0].id).toBe('tool_call_abc')
      expect(result.toolCalls![0].type).toBe('function')
      expect(result.toolCalls![0].function.name).toBe('get_current_weather')
      expect(result.toolCalls![0].function.arguments).toBe('{"location":"Test City","unit":"celsius"}') // Raw string

      // Verify mock API received tool definition
      const lastRequest = getLastGenerateRequest()
      expect(lastRequest.tools).toBeDefined()
      expect(lastRequest.tools).toHaveLength(1)
      expect(lastRequest.tools[0].name).toBe('get_current_weather')
      expect(lastRequest.tools[0].input_schema).toEqual(weatherTool.function.parameters)
    })

    it('should handle sending tool result back in generate', async () => {
      const messages: RosettaMessage[] = [
        { role: 'user', content: "What's the weather in Test City?" },
        {
          role: 'assistant',
          content: null,
          toolCalls: [
            {
              id: 'tool_call_abc',
              type: 'function',
              function: { name: 'get_current_weather', arguments: '{"location":"Test City","unit":"celsius"}' }
            }
          ]
        },
        {
          role: 'tool',
          toolCallId: 'tool_call_abc',
          content: '{"temperature": 20, "unit": "celsius", "condition": "Clear"}' // Tool result
        }
      ]
      const params: GenerateParams = {
        provider: mockProviderKey,
        messages: messages,
        tools: [weatherTool] // Need to pass tools again for context if provider requires
      }

      const result = await rosetta.generate(params)

      // Mock API should respond with the final answer
      expect(result.content).toContain('Okay, the weather in Test City is 20 degrees Celsius.')
      expect(result.finishReason).toBe('stop')
      expect(result.toolCalls).toBeUndefined()

      // Verify the last request sent to the mock API included the tool result message
      const lastRequest = getLastGenerateRequest()
      expect(lastRequest).toBeDefined() // Ensure request was captured
      expect(lastRequest.messages).toHaveLength(3)
      expect(lastRequest.messages[2].role).toBe('tool')
      // Access tool_call_id from the captured request body
      expect(lastRequest.messages[2].toolCallId).toBe('tool_call_abc') // Check the captured request
      expect(lastRequest.messages[2].content).toBe('{"temperature": 20, "unit": "celsius", "condition": "Clear"}')
    })

    it('should handle tool call stream events', async () => {
      const params: GenerateParams = {
        provider: mockProviderKey,
        messages: [{ role: 'user', content: "What's the weather in Stream City?" }],
        tools: [weatherTool]
      }

      const stream = rosetta.stream(params)
      let startChunkData: any = null
      let deltaChunksContent = ''
      let doneChunkData: any = null
      let stopChunkData: any = null
      let finalResultData: any = null
      let receivedChunks = 0

      try {
        for await (const chunk of stream) {
          console.log('[Direct Tool Test Loop] Received chunk:', chunk.type) // DEBUG
          receivedChunks++
          switch (chunk.type) {
            case 'tool_call_start':
              startChunkData = chunk.data
              break
            case 'tool_call_delta':
              deltaChunksContent += chunk.data.functionArgumentChunk
              break
            case 'tool_call_done':
              doneChunkData = chunk.data
              break
            case 'message_stop':
              stopChunkData = chunk.data
              break
            case 'final_result':
              finalResultData = chunk.data.result
              break
            case 'error':
              throw chunk.data.error
          }
        }
      } catch (error) {
        console.error('Error processing tool stream directly in test:', error)
        throw error
      }

      console.log(`[Direct Tool Test Loop] Finished. Total chunks: ${receivedChunks}`) // DEBUG
      // Check counts (adjust based on mock API streamChunks array)
      // message_start, tool_start, delta x 3, tool_done, stop, usage = 8 chunks (final_result might not be yielded)
      expect(receivedChunks).toBeGreaterThanOrEqual(8)

      expect(startChunkData).toBeDefined()
      expect(startChunkData.toolCall.id).toBe('stream_tool_xyz')
      expect(startChunkData.toolCall.function.name).toBe('get_current_weather')

      expect(deltaChunksContent).toBe('{"location": "Stream City", "unit": "fahrenheit"}')

      expect(doneChunkData).toBeDefined()
      expect(doneChunkData.id).toBe('stream_tool_xyz')

      expect(stopChunkData).toBeDefined()
      expect(stopChunkData.finishReason).toBe('tool_calls')

      // Check final result if it was yielded
      if (finalResultData) {
        expect(finalResultData.content).toBeNull() // Mock stream doesn't yield text with tool call
        expect(finalResultData.toolCalls).toHaveLength(1)
        expect(finalResultData.toolCalls![0].function.arguments).toBe(
          '{"location": "Stream City", "unit": "fahrenheit"}'
        )
      }

      // Verify mock API received tool definition
      const lastRequest = getLastStreamRequest()
      expect(lastRequest.tools).toBeDefined()
      expect(lastRequest.tools).toHaveLength(1)
      expect(lastRequest.tools[0].name).toBe('get_current_weather')
    })

    it('should throw ToolArgumentValidationError if mock API returns invalid args (generate)', async () => {
      const params: GenerateParams = {
        provider: mockProviderKey,
        messages: [{ role: 'user', content: "What's the weather?" }], // No location provided
        tools: [weatherTool]
      }

      // Configure mock server to return a tool call with invalid args (missing location)
      // This requires modifying the mock server logic or adding a specific trigger.
      // For simplicity, we'll assume the mock server *can* return invalid args.
      // We need to adjust the mock server's /generate endpoint to sometimes return bad args.
      // Let's simulate this by having the mapper receive bad args and fail validation.

      // Mock the fetch call within the mapper to return bad args
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          id: 'mock-gen-bad-args',
          model: 'mock-custom-model',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: 'tool_call_bad',
                    type: 'function',
                    function: {
                      name: 'get_current_weather',
                      arguments: JSON.stringify({ unit: 'celsius' }) // Missing 'location'
                    }
                  }
                ]
              },
              finish_reason: 'tool_calls'
            }
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        })
      })
      const originalFetch = global.fetch
      global.fetch = mockFetch // Override global fetch for this test

      await expect(rosetta.generate(params)).rejects.toThrow(ToolArgumentValidationError)
      await expect(rosetta.generate(params)).rejects.toThrow(
        "Tool Argument Validation Error for 'get_current_weather': Arguments failed validation"
      )

      global.fetch = originalFetch // Restore original fetch
    })
  })

  // Add more tests: error handling, different tool configs, edge cases, etc.
})
