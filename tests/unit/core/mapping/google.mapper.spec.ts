import {
  mapToGoogleParams,
  mapFromGoogleResponse,
  mapFromGoogleEmbedResponse, // Import embed mapper
  mapFromGoogleEmbedBatchResponse, // Import embed mapper
  mapGoogleStream // Import stream mapper
} from '../../../../src/core/mapping/google.mapper'
import {
  GenerateParams,
  Provider,
  RosettaImageData,
  StreamChunk // Import StreamChunk
  // Removed unused RosettaMessage import
} from '../../../../src/types'
import { MappingError, ProviderAPIError } from '../../../../src/errors' // Import necessary errors
import {
  GenerateContentRequest,
  StartChatParams,
  GenerateContentResponse,
  Part,
  FunctionCall,
  EmbedContentResponse,
  BatchEmbedContentsResponse,
  // Import necessary enums and types from the SDK
  FinishReason,
  HarmCategory,
  HarmProbability,
  BlockReason,
  // Removed unused GoogleTool import
  FunctionDeclarationsTool, // Import specific tool type
  GoogleSearchRetrievalTool, // Import specific tool type
  GenerateContentCandidate // Import Candidate type
} from '@google/generative-ai'

// Helper async generator for stream tests
async function* mockGoogleStreamGenerator(chunks: GenerateContentResponse[]): AsyncIterable<GenerateContentResponse> {
  for (const chunk of chunks) {
    await new Promise(resolve => setTimeout(resolve, 1)) // Simulate delay
    yield chunk
  }
}

// Helper async generator that throws an error
async function* mockGoogleErrorStreamGenerator(
  chunks: GenerateContentResponse[],
  errorToThrow: Error
): AsyncIterable<GenerateContentResponse> {
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

// --- NEW: Implementation for createMockCandidate ---
// Helper to create mock candidate
const createMockCandidate = (
  parts: Part[],
  finishReason: FinishReason | null, // Allow null finish reason
  citations?: any, // Use 'any' for flexibility in mocking potentially evolving types
  safetyRatings?: any[]
): GenerateContentCandidate => ({
  index: 0,
  content: { role: 'model', parts },
  finishReason: finishReason ?? undefined, // Map null to undefined if necessary, or handle based on type
  // Ensure safetyRatings is always an array
  safetyRatings: safetyRatings ?? [],
  // Ensure citationMetadata structure is correct
  citationMetadata: citations ? { citationSources: citations } : undefined,
  // Add potentially missing optional fields if the Candidate type requires them
  finishMessage: undefined // Example optional field
})
// --- END: Implementation for createMockCandidate ---

describe('Google Mapper', () => {
  describe('mapToGoogleParams', () => {
    const baseParams: GenerateParams = {
      provider: Provider.Google,
      model: 'gemini-1.5-flash-latest',
      messages: []
    }

    it('should map basic text messages (single turn)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(isChat).toBe(false)
      expect(result.contents).toEqual([{ role: 'user', parts: [{ text: 'Hello' }] }])
      expect(result.systemInstruction).toBeUndefined()
      expect(result.tools).toBeUndefined()
      expect(result.generationConfig?.maxOutputTokens).toBeUndefined()
    })

    it('should map basic text messages (chat)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello there' },
          { role: 'user', content: 'How are you?' }
        ]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Hi' }] },
        { role: 'model', parts: [{ text: 'Hello there' }] }
      ])
      expect(result.contents).toEqual([{ text: 'How are you?' }]) // Last user message becomes contents for sendMessage
      expect(result.systemInstruction).toBeUndefined()
    })

    it('should map system instruction', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Be concise.' },
          { role: 'user', content: 'Hello' }
        ]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] } // Chat due to system prompt
      expect(isChat).toBe(true)
      expect(result.systemInstruction).toEqual({ role: 'system', parts: [{ text: 'Be concise.' }] })
      expect(result.history).toEqual([]) // System prompt doesn't go in history
      expect(result.contents).toEqual([{ text: 'Hello' }])
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
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(isChat).toBe(false)
      expect(result.contents[0].parts).toEqual([
        { text: 'What is this?' },
        { inlineData: { mimeType: 'image/jpeg', data: 'imgdata' } }
      ])
    })

    it('should map assistant message with tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call the tool.' },
          {
            role: 'assistant',
            content: 'Okay, calling it.', // Text content alongside tool call
            toolCalls: [
              {
                id: 'call_123', // Google doesn't use ID this way, but we map it
                type: 'function',
                function: { name: 'my_tool', arguments: '{"arg": 1}' }
              }
            ]
          }
        ]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] } // Chat because history > 0
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Call the tool.' }] },
        {
          role: 'model',
          parts: [
            { text: 'Okay, calling it.' }, // Text part first
            { functionCall: { name: 'my_tool', args: { arg: 1 } } } // Then function call part
          ]
        }
      ])
      expect(result.contents).toEqual([]) // No final user message
    })

    it('should map tool result message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call the tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'call_123', content: '{"result": "success"}' }
        ]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Call the tool.' }] },
        { role: 'model', parts: [{ functionCall: { name: 'my_tool', args: {} } }] },
        // Tool result mapped to function role with functionResponse part
        { role: 'function', parts: [{ functionResponse: { name: 'my_tool', response: { result: 'success' } } }] }
      ])
      expect(result.contents).toEqual([])
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
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.tools).toBeDefined()
      expect(result.tools).toHaveLength(1)
      expect((result.tools![0] as FunctionDeclarationsTool).functionDeclarations).toEqual([
        {
          name: 'get_weather',
          description: 'Gets weather',
          parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] }
        }
      ])
    })

    it('should map grounding tool', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Explain this.' }],
        grounding: { enabled: true, source: 'web' }
      }
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.tools).toBeDefined()
      expect(result.tools).toHaveLength(1)
      expect((result.tools![0] as GoogleSearchRetrievalTool).googleSearchRetrieval).toEqual({}) // Empty object for default web search
    })

    it('should map generationConfig (maxTokens, temp, topP, stop)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate text.' }],
        maxTokens: 100,
        temperature: 0.5,
        topP: 0.8,
        stop: ['\n']
      }
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig).toBeDefined()
      expect(result.generationConfig?.maxOutputTokens).toBe(100)
      expect(result.generationConfig?.temperature).toBe(0.5)
      expect(result.generationConfig?.topP).toBe(0.8)
      expect(result.generationConfig?.stopSequences).toEqual(['\n'])
    })

    it('should map responseFormat (JSON)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Return JSON.' }],
        responseFormat: { type: 'json_object' }
      }
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig?.responseMimeType).toBe('application/json')
      warnSpy.mockRestore() // Restore console.warn
    })

    it('should throw MappingError for multiple system messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Sys 1' },
          { role: 'system', content: 'Sys 2' },
          { role: 'user', content: 'Hello' }
        ]
      }
      expect(() => mapToGoogleParams(params)).toThrow(MappingError)
      expect(() => mapToGoogleParams(params)).toThrow('Multiple system messages not supported by Google.')
    })

    it('should throw MappingError if system message content is not string', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: [{ type: 'text', text: 'Sys 1' }] }, // Invalid content type
          { role: 'user', content: 'Hello' }
        ]
      }
      expect(() => mapToGoogleParams(params)).toThrow(MappingError)
      expect(() => mapToGoogleParams(params)).toThrow('Google system instruction must be string.')
    })

    // --- NEW TEST ---
    it('should throw MappingError when mapping tool result without preceding function call', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Some message.' },
          // Missing assistant message with the functionCall part
          { role: 'tool', toolCallId: 'call_456', content: '{"result": "failure"}' }
        ]
      }
      expect(() => mapToGoogleParams(params)).toThrow(MappingError)
      expect(() => mapToGoogleParams(params)).toThrow(
        'Cannot find function name for tool result (ID: call_456). Ensure model message with FunctionCall precedes this tool message.'
      )
    })

    // --- NEW: Easy Tests from Plan ---
    it('[Easy] should handle empty messages array (single turn)', () => {
      const params: GenerateParams = { ...baseParams, messages: [] }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(isChat).toBe(false)
      expect(result.contents).toEqual([{ role: 'user', parts: [] }]) // Maps to user role with empty parts
    })

    it('[Easy] should handle empty messages array with system prompt (chat)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'system', content: 'System instruction' }]
      }
      const { googleMappedParams, isChat } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.systemInstruction).toEqual({ role: 'system', parts: [{ text: 'System instruction' }] })
      expect(result.history).toEqual([])
      expect(result.contents).toEqual([]) // No user message parts
    })

    it('[Easy] should warn for grounding source other than web', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Explain.' }],
        // FIX: Cast invalid source to any to bypass TS error while testing warning
        grounding: { enabled: true, source: 'invalid-source' as any }
      }
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect((result.tools![0] as GoogleSearchRetrievalTool).googleSearchRetrieval).toBeDefined()
      expect(warnSpy).toHaveBeenCalledWith(
        "Only 'web' grounding source currently mapped for Google Search Retrieval. Ignoring source: invalid-source"
      )
      warnSpy.mockRestore()
    })

    it('[Easy] should warn for responseFormat schema presence', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'JSON please.' }],
        responseFormat: { type: 'json_object', schema: { type: 'object' } }
      }
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig?.responseMimeType).toBe('application/json')
      expect(warnSpy).toHaveBeenCalledWith(
        'Google JSON mode requested via responseFormat. Ensure schema is described in the prompt. `schema` parameter is ignored for Google GenerationConfig.'
      )
      warnSpy.mockRestore()
    })
    // --- End Easy Tests ---

    // --- NEW: Medium Tests from Plan ---
    it('[Medium] should map tool result with non-JSON string content', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call tool.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'call_str', type: 'function', function: { name: 'string_tool', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'call_str', content: 'Tool result was just this string.' } // Non-JSON string
        ]
      }
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const { googleMappedParams } = mapToGoogleParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      // FIX: Add check for history existence
      expect(result.history).toBeDefined()
      expect(result.history![2]).toEqual({
        role: 'function',
        parts: [
          {
            functionResponse: {
              name: 'string_tool',
              response: { content: 'Tool result was just this string.' } // Wrapped in object
            }
          }
        ]
      })
      expect(warnSpy).toHaveBeenCalledWith(
        'Tool result content for string_tool was not valid JSON. Wrapping as { content: "..." }'
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should determine isChat=true with only system prompt', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'system', content: 'System instruction' }]
      }
      const { isChat } = mapToGoogleParams(params)
      expect(isChat).toBe(true)
    })

    it('[Medium] should determine isChat=true with history', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello' }
        ]
      }
      const { isChat } = mapToGoogleParams(params)
      expect(isChat).toBe(true)
    })

    it('[Medium] should determine isChat=false with only single user message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hi' }]
      }
      const { isChat } = mapToGoogleParams(params)
      expect(isChat).toBe(false)
    })
    // --- End Medium Tests ---
  })

  describe('mapFromGoogleResponse', () => {
    const modelUsed = 'gemini-1.5-flash-latest-test'

    // Helper to create mock candidate (already defined above)

    it('should map basic text response', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Response text.' }], FinishReason.STOP)],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 }
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe('Response text.')
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('stop')
      expect(result.usage).toEqual({
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 15,
        cachedContentTokenCount: undefined
      })
      expect(result.model).toBe(modelUsed)
      expect(result.citations).toBeUndefined()
    })

    it('should map response with tool calls', () => {
      const functionCall: FunctionCall = { name: 'get_weather', args: { location: 'Paris' } }
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ functionCall }], FinishReason.STOP)], // Finish reason might be STOP even with tool call
        usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 10, totalTokenCount: 30 }
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBeNull() // No text part
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls![0].type).toBe('function')
      expect(result.toolCalls![0].function.name).toBe('get_weather')
      expect(result.toolCalls![0].function.arguments).toBe('{"location":"Paris"}')
      expect(result.finishReason).toBe('tool_calls') // Overridden because tool call exists
      expect(result.usage?.totalTokens).toBe(30)
    })

    it('should map response with citations', () => {
      const response: GenerateContentResponse = {
        candidates: [
          createMockCandidate([{ text: 'Grounded response.' }], FinishReason.STOP, [
            { startIndex: 0, endIndex: 8, uri: 'http://example.com' }
          ])
        ],
        usageMetadata: { promptTokenCount: 15, candidatesTokenCount: 5, totalTokenCount: 20 }
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe('Grounded response.')
      expect(result.citations).toBeDefined()
      expect(result.citations).toHaveLength(1)
      expect(result.citations![0]).toEqual({
        sourceId: 'http://example.com',
        startIndex: 0,
        endIndex: 8,
        text: undefined
      })
      expect(result.finishReason).toBe('stop')
    })

    it('should map MAX_TOKENS finish reason', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Cut off...' }], FinishReason.MAX_TOKENS)]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe('Cut off...')
      expect(result.finishReason).toBe('length')
    })

    it('should map SAFETY finish reason', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        candidates: [
          createMockCandidate([], FinishReason.SAFETY, undefined, [
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, probability: HarmProbability.HIGH }
          ])
        ]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('content_filter')
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google candidate blocked due to safety.'))
      warnSpy.mockRestore()
    })

    it('should handle prompt blocked response', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        promptFeedback: {
          blockReason: BlockReason.SAFETY,
          safetyRatings: [{ category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, probability: HarmProbability.MEDIUM }]
        }
        // No candidates field when prompt is blocked
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toBeUndefined()
      expect(result.finishReason).toBe('content_filter') // Mapped from blockReason
      expect(result.usage).toBeUndefined() // No usage when prompt blocked
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google prompt blocked.'))
      warnSpy.mockRestore()
    })

    it('should handle missing candidates gracefully', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        // No candidates array
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 0, totalTokenCount: 5 }
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('error') // Indicate unexpected state
      expect(result.usage?.totalTokens).toBe(5)
      expect(warnSpy).toHaveBeenCalledWith('Google response or candidate is missing despite no prompt block.')
      warnSpy.mockRestore()
    })

    it('should attempt to parse JSON content', () => {
      const jsonString = '{"data": 1}'
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: jsonString }], FinishReason.STOP)]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toEqual({ data: 1 })
    })

    it('should handle unparsable JSON content gracefully', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const jsonString = '{"data": 1' // Invalid JSON
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: jsonString }], FinishReason.STOP)]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toBeNull() // Parsing failed
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to auto-parse potential JSON from Google:'),
        expect.any(Error) // Check that the second argument is an error object
      )
      warnSpy.mockRestore()
    })

    // --- NEW: Medium Tests from Plan ---
    it('[Medium] should map RECITATION finish reason', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Recited text.' }], FinishReason.RECITATION)]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe('Recited text.')
      expect(result.finishReason).toBe('recitation_filter')
    })

    it('[Medium] should map OTHER block reason', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        // FIX: Add missing safetyRatings property
        promptFeedback: { blockReason: BlockReason.OTHER, safetyRatings: [] }
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.finishReason).toBe('error') // Mapped from OTHER blockReason
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google prompt blocked.'))
      warnSpy.mockRestore()
    })

    it('[Medium] should handle invalid function call arguments gracefully', () => {
      const functionCall: FunctionCall = { name: 'bad_args_tool', args: 'not json' as any } // Invalid args
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ functionCall }], FinishReason.STOP)]
      }
      // Expect mapToolCallsFromGoogle to handle stringification
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls![0].function.arguments).toBe('"not json"') // Should be stringified
      expect(result.finishReason).toBe('tool_calls') // Overridden
    })

    it('[Medium] should handle null parts in content gracefully', () => {
      const response: GenerateContentResponse = {
        candidates: [
          createMockCandidate(
            [
              { text: 'Part 1' },
              null as any, // Simulate null part
              { text: 'Part 3' }
            ],
            FinishReason.STOP
          )
        ]
      }
      const result = mapFromGoogleResponse(response, modelUsed)
      expect(result.content).toBe('Part 1Part 3') // Null part should be ignored by the fixed filter
      expect(result.finishReason).toBe('stop')
    })
    // --- End Medium Tests ---
  })

  // --- Embeddings ---
  describe('mapFromGoogleEmbedResponse', () => {
    const modelUsed = 'embedding-001-test'

    it('should map a single embedding response', () => {
      const response: EmbedContentResponse = {
        embedding: { values: [0.1, 0.2, 0.3] }
        // usageMetadata: { totalTokenCount: 10 } // Usage metadata is not part of the official type
      }
      const result = mapFromGoogleEmbedResponse(response, modelUsed)
      expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toBeUndefined()
      expect(result.rawResponse).toBe(response)
    })

    it('should throw MappingError if embedding structure is invalid', () => {
      const invalidResponse = { embedding: null } as any // Invalid structure
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid single embedding response structure from Google.'
      )
    })
  })

  describe('mapFromGoogleEmbedBatchResponse', () => {
    const modelUsed = 'embedding-001-test-batch'

    it('should map a batch embedding response', () => {
      const response: BatchEmbedContentsResponse = {
        embeddings: [{ values: [0.1, 0.2] }, { values: [0.3, 0.4] }]
        // usageMetadata: { totalTokenCount: 25 } // Usage metadata is not part of the official type
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.3, 0.4]
      ])
      expect(result.model).toBe(modelUsed)
      expect(result.usage).toBeUndefined()
      expect(result.rawResponse).toBe(response)
    })

    it('should handle missing embeddings in batch response', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: BatchEmbedContentsResponse = {
        embeddings: [
          { values: [0.1, 0.2] },
          null as any, // Simulate a missing embedding object
          { values: [0.5, 0.6] }
        ]
      }
      const result = mapFromGoogleEmbedBatchResponse(response, modelUsed)
      expect(result.embeddings).toEqual([
        [0.1, 0.2],
        [0.5, 0.6]
      ]) // Null entry is filtered out
      expect(warnSpy).toHaveBeenCalledWith('Some embeddings were missing values in Google batch response.')
      warnSpy.mockRestore()
    })

    it('should throw MappingError if batch structure is invalid', () => {
      const invalidResponse = { embeddings: null } as any // Invalid structure
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedBatchResponse(invalidResponse, modelUsed)).toThrow(
        'Invalid batch embedding response structure from Google.'
      )
    })

    it('should throw MappingError if all embeddings are missing values', () => {
      const response = { embeddings: [null, undefined] } as any // All missing values
      expect(() => mapFromGoogleEmbedBatchResponse(response, modelUsed)).toThrow(MappingError)
      expect(() => mapFromGoogleEmbedBatchResponse(response, modelUsed)).toThrow(
        'All embeddings were missing values in Google batch response.'
      )
    })
  })

  // --- NEW: mapGoogleStream Tests ---
  describe('mapGoogleStream', () => {
    // Removed unused modelId variable

    it('[Hard] should handle basic text stream', async () => {
      const mockChunks: GenerateContentResponse[] = [
        { candidates: [createMockCandidate([{ text: 'Hello ' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        { candidates: [createMockCandidate([{ text: 'world' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        {
          candidates: [createMockCandidate([], FinishReason.STOP)],
          usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 }
        }
      ]
      const stream = mapGoogleStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Google, model: '' } }) // Model unknown initially
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Hello ' } })
      expect(results[2]).toEqual({ type: 'content_delta', data: { delta: 'world' } })
      expect(results[3]).toEqual({ type: 'message_stop', data: { finishReason: 'stop' } })
      expect(results[4]).toEqual({
        type: 'final_usage',
        data: { usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7, cachedContentTokenCount: undefined } }
      })
      expect(results[5].type).toBe('final_result')
      expect((results[5] as any).data.result).toEqual(
        expect.objectContaining({
          content: 'Hello world',
          finishReason: 'stop',
          model: '', // Model still unknown
          usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7, cachedContentTokenCount: undefined }
        })
      )
    })

    it('[Hard] should handle stream with tool call', async () => {
      const toolName = 'stream_tool'
      const mockChunks: GenerateContentResponse[] = [
        {
          candidates: [createMockCandidate([{ functionCall: { name: toolName, args: { a: 1 } } }], FinishReason.STOP)]
        }, // Tool call in one chunk
        {
          candidates: [], // Empty candidate list in usage chunk
          usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 }
        }
      ]
      const stream = mapGoogleStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // Expected: start, tool_start, tool_delta, tool_done, stop, usage, final
      expect(results).toHaveLength(7)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('tool_call_start')
      expect((results[1] as any).data.toolCall.function.name).toBe(toolName)
      const toolCallId = (results[1] as any).data.toolCall.id // Get the generated ID

      expect(results[2].type).toBe('tool_call_delta')
      expect((results[2] as any).data.id).toBe(toolCallId)
      expect((results[2] as any).data.functionArgumentChunk).toBe('{"a":1}') // Arguments streamed at once

      expect(results[3].type).toBe('tool_call_done')
      expect((results[3] as any).data.id).toBe(toolCallId)

      expect(results[4].type).toBe('message_stop')
      expect((results[4] as any).data.finishReason).toBe('tool_calls') // Overridden

      expect(results[5].type).toBe('final_usage')
      expect(results[6].type).toBe('final_result')
      const finalResult = (results[6] as any).data.result
      expect(finalResult.content).toBeNull()
      expect(finalResult.finishReason).toBe('tool_calls')
      expect(finalResult.toolCalls).toHaveLength(1)
      expect(finalResult.toolCalls[0]).toEqual(
        expect.objectContaining({ function: { name: toolName, arguments: '{"a":1}' } })
      )
    })

    it('[Hard] should handle stream error', async () => {
      const apiError = new Error('Google stream failed')
      const mockChunks: GenerateContentResponse[] = [
        { candidates: [createMockCandidate([{ text: 'Partial ' }], FinishReason.FINISH_REASON_UNSPECIFIED)] }
      ]
      const stream = mapGoogleStream(mockGoogleErrorStreamGenerator(mockChunks, apiError))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta')
      expect(results[2].type).toBe('error')
      const errorChunk = results[2] as { type: 'error'; data: { error: Error } }
      expect(errorChunk.data.error).toBeInstanceOf(ProviderAPIError)
      expect(errorChunk.data.error.message).toContain('Google stream failed')
      expect((errorChunk.data.error as ProviderAPIError).provider).toBe(Provider.Google)
    })

    it('[Hard] should handle stream ending with SAFETY', async () => {
      const mockChunks: GenerateContentResponse[] = [
        { candidates: [createMockCandidate([{ text: 'Unsafe ' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        {
          candidates: [
            createMockCandidate([], FinishReason.SAFETY, undefined, [
              { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, probability: HarmProbability.MEDIUM }
            ])
          ],
          usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 1, totalTokenCount: 6 }
        }
      ]
      const stream = mapGoogleStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // FIX: Expect 5 chunks now (start, delta, stop, usage, final)
      expect(results).toHaveLength(5)
      expect(results[0].type).toBe('message_start')
      expect(results[1]).toEqual({ type: 'content_delta', data: { delta: 'Unsafe ' } })
      expect(results[2]).toEqual({ type: 'message_stop', data: { finishReason: 'content_filter' } })
      expect(results[3].type).toBe('final_usage')
      expect(results[4].type).toBe('final_result')
      expect((results[4] as any).data.result.finishReason).toBe('content_filter')
      expect((results[4] as any).data.result.content).toBe('Unsafe ')
    })

    it('[Hard] should handle stream with citations', async () => {
      const mockChunks: GenerateContentResponse[] = [
        {
          candidates: [
            createMockCandidate([{ text: 'Grounded ' }], FinishReason.FINISH_REASON_UNSPECIFIED, [
              { startIndex: 0, endIndex: 8, uri: 'cite1.com' }
            ])
          ]
        },
        { candidates: [createMockCandidate([{ text: 'text.' }], FinishReason.STOP)] },
        { usageMetadata: { promptTokenCount: 6, candidatesTokenCount: 3, totalTokenCount: 9 } }
      ]
      const stream = mapGoogleStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // Expected: start, delta(cite_delta, cite_done), delta, stop, usage, final
      expect(results).toHaveLength(8)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('content_delta') // Text delta
      expect(results[2].type).toBe('citation_delta') // Citation delta
      expect((results[2] as any).data.citation.sourceId).toBe('cite1.com')
      expect(results[3].type).toBe('citation_done') // Citation done
      expect(results[4].type).toBe('content_delta') // Text delta
      expect(results[5].type).toBe('message_stop')
      expect(results[6].type).toBe('final_usage')
      expect(results[7].type).toBe('final_result')
      const finalResult = (results[7] as any).data.result
      expect(finalResult.content).toBe('Grounded text.')
      expect(finalResult.citations).toHaveLength(1)
      expect(finalResult.citations[0].sourceId).toBe('cite1.com')
    })

    it('[Hard] should handle stream with JSON content', async () => {
      const jsonString = '{"key": "value"}'
      const mockChunks: GenerateContentResponse[] = [
        { candidates: [createMockCandidate([{ text: '{"key":' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        { candidates: [createMockCandidate([{ text: ' "value"}' }], FinishReason.STOP)] },
        { usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 4, totalTokenCount: 8 } }
      ]
      const stream = mapGoogleStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      // Expected: start, json_delta, json_delta, json_done, stop, usage, final
      expect(results).toHaveLength(7)
      expect(results[0].type).toBe('message_start')
      expect(results[1].type).toBe('json_delta')
      expect((results[1] as any).data.delta).toBe('{"key":')
      expect((results[1] as any).data.snapshot).toBe('{"key":')
      expect((results[1] as any).data.parsed).toBeUndefined()

      expect(results[2].type).toBe('json_delta')
      expect((results[2] as any).data.delta).toBe(' "value"}')
      expect((results[2] as any).data.snapshot).toBe(jsonString)
      expect((results[2] as any).data.parsed).toEqual({ key: 'value' }) // Parsed successfully

      expect(results[3].type).toBe('json_done')
      expect((results[3] as any).data.snapshot).toBe(jsonString)
      expect((results[3] as any).data.parsed).toEqual({ key: 'value' })

      expect(results[4].type).toBe('message_stop')
      expect(results[5].type).toBe('final_usage')
      expect(results[6].type).toBe('final_result')
      const finalResult = (results[6] as any).data.result
      expect(finalResult.content).toBe(jsonString)
      expect(finalResult.parsedContent).toEqual({ key: 'value' })
    })
  })
  // --- End mapGoogleStream Tests ---
})
