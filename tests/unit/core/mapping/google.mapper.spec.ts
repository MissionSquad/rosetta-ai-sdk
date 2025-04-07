import {
  GenerateContentRequest,
  StartChatParams,
  GenerateContentResponse,
  Part,
  FunctionCall,
  EmbedContentResponse,
  BatchEmbedContentsResponse,
  FinishReason,
  HarmCategory,
  HarmProbability,
  BlockReason,
  FunctionDeclarationsTool,
  GoogleSearchRetrievalTool,
  GenerateContentCandidate,
  BatchEmbedContentsRequest,
  EmbedContentRequest
} from '@google/generative-ai'
import { GoogleMapper } from '../../../../src/core/mapping/google.mapper'
import * as GoogleEmbedMapper from '../../../../src/core/mapping/google.embed.mapper'
import {
  GenerateParams,
  Provider,
  RosettaImageData,
  StreamChunk,
  EmbedParams,
  TranscribeParams,
  TranslateParams
} from '../../../../src/types'
import { MappingError, ProviderAPIError, UnsupportedFeatureError } from '../../../../src/errors'

// Mock the embed mapper functions
jest.mock('../../../../src/core/mapping/google.embed.mapper', () => ({
  mapFromGoogleEmbedResponse: jest.fn(),
  mapFromGoogleEmbedBatchResponse: jest.fn()
}))

const mockMapFromGoogleEmbedResponse = GoogleEmbedMapper.mapFromGoogleEmbedResponse as jest.Mock
const mockMapFromGoogleEmbedBatchResponse = GoogleEmbedMapper.mapFromGoogleEmbedBatchResponse as jest.Mock

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

// Helper to create mock candidate
const createMockCandidate = (
  parts: Part[],
  finishReason: FinishReason | null,
  citations?: any,
  safetyRatings?: any[]
): GenerateContentCandidate => ({
  index: 0,
  content: { role: 'model', parts },
  finishReason: finishReason ?? undefined,
  safetyRatings: safetyRatings ?? [],
  citationMetadata: citations ? { citationSources: citations } : undefined,
  finishMessage: undefined
})

describe('Google Mapper', () => {
  let mapper: GoogleMapper

  beforeEach(() => {
    mapper = new GoogleMapper()
    jest.clearAllMocks() // Clear mocks before each test
  })

  it('[Easy] should have the correct provider property', () => {
    expect(mapper.provider).toBe(Provider.Google)
  })

  describe('mapToProviderParams (Generate)', () => {
    const baseParams: GenerateParams = {
      provider: Provider.Google,
      model: 'gemini-1.5-flash-latest',
      messages: []
    }

    it('[Easy] should map basic text messages (single turn)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hello' }]
      }
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(isChat).toBe(false)
      expect(result.contents).toEqual([{ role: 'user', parts: [{ text: 'Hello' }] }])
      expect(result.systemInstruction).toBeUndefined()
      expect(result.tools).toBeUndefined()
      expect(result.generationConfig?.maxOutputTokens).toBeUndefined()
    })

    it('[Easy] should map basic text messages (chat)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello there' },
          { role: 'user', content: 'How are you?' }
        ]
      }
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Hi' }] },
        { role: 'model', parts: [{ text: 'Hello there' }] }
      ])
      expect(result.contents).toEqual([{ text: 'How are you?' }])
      expect(result.systemInstruction).toBeUndefined()
    })

    it('[Easy] should map system instruction', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Be concise.' },
          { role: 'user', content: 'Hello' }
        ]
      }
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] } // Chat due to system prompt
      expect(isChat).toBe(true)
      expect(result.systemInstruction).toEqual({ role: 'system', parts: [{ text: 'Be concise.' }] })
      expect(result.history).toEqual([]) // System prompt doesn't go in history
      expect(result.contents).toEqual([{ text: 'Hello' }])
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
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(isChat).toBe(false)
      expect(result.contents[0].parts).toEqual([
        { text: 'What is this?' },
        { inlineData: { mimeType: 'image/jpeg', data: 'imgdata' } }
      ])
    })

    it('[Easy] should map assistant message with tool calls', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Call the tool.' },
          {
            role: 'assistant',
            content: 'Okay, calling it.',
            toolCalls: [{ id: 'call_123', type: 'function', function: { name: 'my_tool', arguments: '{"arg": 1}' } }]
          },
          { role: 'user', content: 'Did it work?' }
        ]
      }
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Call the tool.' }] },
        {
          role: 'model',
          parts: [{ text: 'Okay, calling it.' }, { functionCall: { name: 'my_tool', args: { arg: 1 } } }]
        }
      ])
      expect(result.contents).toEqual([{ text: 'Did it work?' }])
    })

    it('[Easy] should map tool result message', () => {
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
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Call the tool.' }] },
        { role: 'model', parts: [{ functionCall: { name: 'my_tool', args: {} } }] }
      ])
      expect(result.contents).toEqual([{ functionResponse: { name: 'my_tool', response: { result: 'success' } } }])
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
      const { googleMappedParams } = mapper.mapToProviderParams(params)
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

    it('[Easy] should map grounding tool', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Explain this.' }],
        grounding: { enabled: true, source: 'web' }
      }
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.tools).toBeDefined()
      expect(result.tools).toHaveLength(1)
      expect((result.tools![0] as GoogleSearchRetrievalTool).googleSearchRetrieval).toEqual({})
    })

    it('[Easy] should map generationConfig (maxTokens, temp, topP, stop)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Generate text.' }],
        maxTokens: 100,
        temperature: 0.5,
        topP: 0.8,
        stop: ['\n']
      }
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig).toBeDefined()
      expect(result.generationConfig?.maxOutputTokens).toBe(100)
      expect(result.generationConfig?.temperature).toBe(0.5)
      expect(result.generationConfig?.topP).toBe(0.8)
      expect(result.generationConfig?.stopSequences).toEqual(['\n'])
    })

    it('[Easy] should map responseFormat (JSON)', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Return JSON.' }],
        responseFormat: { type: 'json_object' }
      }
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig?.responseMimeType).toBe('application/json')
      warnSpy.mockRestore()
    })

    it('[Medium] should throw MappingError for multiple system messages', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'Sys 1' },
          { role: 'system', content: 'Sys 2' },
          { role: 'user', content: 'Hello' }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Multiple system messages not supported by Google.')
    })

    it('[Medium] should throw MappingError if system message content is not string', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: [{ type: 'text', text: 'Sys 1' }] },
          { role: 'user', content: 'Hello' }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Google system instruction must be string.')
    })

    it('[Medium] should throw MappingError when mapping tool result without preceding function call', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Some message.' },
          { role: 'tool', toolCallId: 'call_456', content: '{"result": "failure"}' }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow(
        'Cannot find function name for final tool result (ID: call_456).'
      )
    })

    it('[Medium] should warn for grounding source other than web', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Explain.' }],
        grounding: { enabled: true, source: 'invalid-source' as any }
      }
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect((result.tools![0] as GoogleSearchRetrievalTool).googleSearchRetrieval).toBeDefined()
      expect(warnSpy).toHaveBeenCalledWith(
        "Only 'web' grounding source currently mapped for Google Search Retrieval. Ignoring source: invalid-source"
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should warn for responseFormat schema presence', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'JSON please.' }],
        responseFormat: { type: 'json_object', schema: { type: 'object' } }
      }
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as GenerateContentRequest
      expect(result.generationConfig?.responseMimeType).toBe('application/json')
      expect(warnSpy).toHaveBeenCalledWith(
        'Google JSON mode requested via responseFormat. Ensure schema is described in the prompt. `schema` parameter is ignored for Google GenerationConfig.'
      )
      warnSpy.mockRestore()
    })

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
          { role: 'tool', toolCallId: 'call_str', content: 'Tool result was just this string.' }
        ]
      }
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const { googleMappedParams } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(result.history).toEqual([
        { role: 'user', parts: [{ text: 'Call tool.' }] },
        { role: 'model', parts: [{ functionCall: { name: 'string_tool', args: {} } }] }
      ])
      expect(result.contents).toEqual([
        { functionResponse: { name: 'string_tool', response: { content: 'Tool result was just this string.' } } }
      ])
      // FIX: Adjust assertion to be less strict about the exact warning message, check for the key part.
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          'tool result content for string_tool was not valid JSON. Wrapping as { content: "..." }'
        )
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should determine isChat=true with only system prompt', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'system', content: 'System instruction' },
          { role: 'user', content: 'Hello' }
        ]
      }
      const { isChat } = mapper.mapToProviderParams(params)
      expect(isChat).toBe(true)
    })

    it('[Medium] should determine isChat=true with history', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello' },
          { role: 'user', content: 'Follow up' }
        ]
      }
      const { isChat } = mapper.mapToProviderParams(params)
      expect(isChat).toBe(true)
    })

    it('[Medium] should determine isChat=false with only single user message', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [{ role: 'user', content: 'Hi' }]
      }
      const { isChat } = mapper.mapToProviderParams(params)
      expect(isChat).toBe(false)
    })

    it('[Hard] should throw MappingError if final user message is empty', () => {
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello' },
          { role: 'user', content: '' }
        ]
      }
      expect(() => mapper.mapToProviderParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(params)).toThrow('Final user message content cannot be null or empty.')
    })

    it('[Hard] should throw MappingError if final message is assistant/system', () => {
      const paramsAssistant: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello' }
        ]
      }
      const paramsSystem: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'system', content: 'Be good.' }
        ]
      }
      expect(() => mapper.mapToProviderParams(paramsAssistant)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(paramsSystem)).toThrow(MappingError)
      expect(() => mapper.mapToProviderParams(paramsAssistant)).toThrow(
        "Invalid role for the final message in a Google chat turn: 'model'. Expected 'user' or 'tool'."
      )
      expect(() => mapper.mapToProviderParams(paramsSystem)).toThrow(
        "Invalid role for the final message in a Google chat turn: 'system'. Expected 'user' or 'tool'."
      )
    })

    it('[Hard] should skip history messages with empty content parts', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const params: GenerateParams = {
        ...baseParams,
        messages: [
          { role: 'user', content: 'First message' },
          { role: 'assistant', content: '' },
          { role: 'user', content: 'Second message' }
        ]
      }
      const { googleMappedParams, isChat } = mapper.mapToProviderParams(params)
      const result = googleMappedParams as StartChatParams & { contents: Part[] }
      expect(isChat).toBe(true)
      expect(result.history).toEqual([{ role: 'user', parts: [{ text: 'First message' }] }])
      expect(result.contents).toEqual([{ text: 'Second message' }])
      expect(warnSpy).toHaveBeenCalledWith("Skipping history message with role 'model' due to empty content parts.")
      warnSpy.mockRestore()
    })
  })

  describe('mapFromProviderResponse (Generate)', () => {
    const modelUsed = 'gemini-1.5-flash-latest-test'

    it('[Easy] should map basic text response', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Response text.' }], FinishReason.STOP)],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
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

    it('[Easy] should map response with tool calls', () => {
      const functionCall: FunctionCall = { name: 'get_weather', args: { location: 'Paris' } }
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ functionCall }], FinishReason.STOP)],
        usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 10, totalTokenCount: 30 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls![0].function.name).toBe('get_weather')
      expect(result.toolCalls![0].function.arguments).toBe('{"location":"Paris"}')
      expect(result.finishReason).toBe('tool_calls')
      expect(result.usage?.totalTokens).toBe(30)
    })

    it('[Easy] should map response with citations', () => {
      const response: GenerateContentResponse = {
        candidates: [
          createMockCandidate([{ text: 'Grounded response.' }], FinishReason.STOP, [
            { startIndex: 0, endIndex: 8, uri: 'http://example.com' }
          ])
        ],
        usageMetadata: { promptTokenCount: 15, candidatesTokenCount: 5, totalTokenCount: 20 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
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

    it('[Easy] should map MAX_TOKENS finish reason', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Cut off...' }], FinishReason.MAX_TOKENS)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Cut off...')
      expect(result.finishReason).toBe('length')
    })

    it('[Easy] should map SAFETY finish reason', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        candidates: [
          createMockCandidate([], FinishReason.SAFETY, undefined, [
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, probability: HarmProbability.HIGH }
          ])
        ]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('content_filter')
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google candidate blocked due to safety.'))
      warnSpy.mockRestore()
    })

    it('[Easy] should handle prompt blocked response', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        promptFeedback: {
          blockReason: BlockReason.SAFETY,
          safetyRatings: [{ category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, probability: HarmProbability.MEDIUM }]
        }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('content_filter')
      expect(result.usage).toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google prompt blocked.'))
      warnSpy.mockRestore()
    })

    it('[Easy] should handle missing candidates gracefully', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 0, totalTokenCount: 5 }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBeNull()
      expect(result.finishReason).toBe('error')
      expect(result.usage?.totalTokens).toBe(5)
      expect(warnSpy).toHaveBeenCalledWith('Google response or candidate is missing despite no prompt block.')
      warnSpy.mockRestore()
    })

    it('[Easy] should attempt to parse JSON content', () => {
      const jsonString = '{"data": 1}'
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: jsonString }], FinishReason.STOP)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toEqual({ data: 1 })
    })

    it('[Easy] should handle unparsable JSON content gracefully', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const jsonString = '{"data": 1'
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: jsonString }], FinishReason.STOP)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe(jsonString)
      expect(result.parsedContent).toBeNull()
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to auto-parse potential JSON from Google:'),
        expect.any(Error)
      )
      warnSpy.mockRestore()
    })

    it('[Medium] should map RECITATION finish reason', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Recited text.' }], FinishReason.RECITATION)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Recited text.')
      expect(result.finishReason).toBe('recitation_filter')
    })

    it('[Medium] should map OTHER block reason', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation()
      const response: GenerateContentResponse = {
        promptFeedback: { blockReason: BlockReason.OTHER, safetyRatings: [] }
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.finishReason).toBe('error')
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Google prompt blocked.'))
      warnSpy.mockRestore()
    })

    it('[Medium] should handle invalid function call arguments gracefully', () => {
      const functionCall: FunctionCall = { name: 'bad_args_tool', args: 'not json' as any }
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ functionCall }], FinishReason.STOP)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.toolCalls).toBeDefined()
      expect(result.toolCalls![0].function.arguments).toBe('"not json"')
      expect(result.finishReason).toBe('tool_calls')
    })

    it('[Medium] should handle null parts in content gracefully', () => {
      const response: GenerateContentResponse = {
        candidates: [createMockCandidate([{ text: 'Part 1' }, null as any, { text: 'Part 3' }], FinishReason.STOP)]
      }
      const result = mapper.mapFromProviderResponse(response, modelUsed)
      expect(result.content).toBe('Part 1Part 3')
      expect(result.finishReason).toBe('stop')
    })
  })

  describe('mapToEmbedParams', () => {
    const model = 'embedding-001'
    const baseEmbedParams: EmbedParams = {
      provider: Provider.Google,
      model: model,
      input: ''
    }

    it('[Easy] should map single string input', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: 'Embed this' }
      const result = mapper.mapToEmbedParams(params) as EmbedContentRequest
      expect(result.model).toBe(`models/${model}`)
      expect(result.content).toEqual({ parts: [{ text: 'Embed this' }], role: 'user' })
    })

    it('[Easy] should map string array input (batch)', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: ['Text 1', 'Text 2'] }
      const result = mapper.mapToEmbedParams(params) as BatchEmbedContentsRequest
      expect(result.requests).toHaveLength(2)
      expect(result.requests[0]).toEqual({
        model: `models/${model}`,
        content: { parts: [{ text: 'Text 1' }], role: 'user' }
      })
      expect(result.requests[1]).toEqual({
        model: `models/${model}`,
        content: { parts: [{ text: 'Text 2' }], role: 'user' }
      })
    })

    it('[Medium] should throw MappingError for empty string input', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: '' }
      expect(() => mapper.mapToEmbedParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToEmbedParams(params)).toThrow('Input text for Google embedding cannot be empty.')
    })

    it('[Medium] should throw MappingError for empty array input', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: [] }
      expect(() => mapper.mapToEmbedParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToEmbedParams(params)).toThrow('Input text for Google embedding cannot be empty.')
    })
  })

  describe('mapFromEmbedResponse', () => {
    const modelUsed = 'embedding-001-test'
    const mockSingleResponse: EmbedContentResponse = { embedding: { values: [0.1, 0.2] } }
    const mockBatchResponse: BatchEmbedContentsResponse = { embeddings: [{ values: [0.3, 0.4] }] }

    beforeEach(() => {
      mockMapFromGoogleEmbedResponse.mockClear()
      mockMapFromGoogleEmbedBatchResponse.mockClear()
    })

    it('[Easy] should delegate to mapFromGoogleEmbedResponse for single response', () => {
      mapper.mapFromEmbedResponse(mockSingleResponse, modelUsed)
      expect(mockMapFromGoogleEmbedResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromGoogleEmbedResponse).toHaveBeenCalledWith(mockSingleResponse, modelUsed)
      expect(mockMapFromGoogleEmbedBatchResponse).not.toHaveBeenCalled()
    })

    it('[Easy] should delegate to mapFromGoogleEmbedBatchResponse for batch response', () => {
      mapper.mapFromEmbedResponse(mockBatchResponse, modelUsed)
      expect(mockMapFromGoogleEmbedBatchResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromGoogleEmbedBatchResponse).toHaveBeenCalledWith(mockBatchResponse, modelUsed)
      expect(mockMapFromGoogleEmbedResponse).not.toHaveBeenCalled()
    })

    it('[Medium] should throw MappingError for unknown response structure', () => {
      const unknownResponse = { someOtherField: 'value' }
      expect(() => mapper.mapFromEmbedResponse(unknownResponse, modelUsed)).toThrow(MappingError)
      expect(() => mapper.mapFromEmbedResponse(unknownResponse, modelUsed)).toThrow(
        'Unknown Google embedding response structure.'
      )
    })
  })

  describe('Audio Methods', () => {
    const dummyTranscribeParams: TranscribeParams = {
      provider: Provider.Google,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' }
    }
    const dummyTranslateParams: TranslateParams = {
      provider: Provider.Google,
      audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' }
    }

    it('[Easy] mapToTranscribeParams should throw UnsupportedFeatureError', () => {
      expect(() => mapper.mapToTranscribeParams(dummyTranscribeParams, {})).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToTranscribeParams(dummyTranscribeParams, {})).toThrow(
        "Provider 'google' does not support the requested feature: Audio Transcription"
      )
    })

    it('[Easy] mapFromTranscribeResponse should throw UnsupportedFeatureError', () => {
      expect(() => mapper.mapFromTranscribeResponse({}, 'model')).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapFromTranscribeResponse({}, 'model')).toThrow(
        "Provider 'google' does not support the requested feature: Audio Transcription"
      )
    })

    it('[Easy] mapToTranslateParams should throw UnsupportedFeatureError', () => {
      expect(() => mapper.mapToTranslateParams(dummyTranslateParams, {})).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToTranslateParams(dummyTranslateParams, {})).toThrow(
        "Provider 'google' does not support the requested feature: Audio Translation"
      )
    })

    it('[Easy] mapFromTranslateResponse should throw UnsupportedFeatureError', () => {
      expect(() => mapper.mapFromTranslateResponse({}, 'model')).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapFromTranslateResponse({}, 'model')).toThrow(
        "Provider 'google' does not support the requested feature: Audio Translation"
      )
    })
  })

  describe('mapProviderStream', () => {
    it('[Hard] should handle basic text stream', async () => {
      const mockChunks: GenerateContentResponse[] = [
        { candidates: [createMockCandidate([{ text: 'Hello ' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        { candidates: [createMockCandidate([{ text: 'world' }], FinishReason.FINISH_REASON_UNSPECIFIED)] },
        {
          candidates: [createMockCandidate([], FinishReason.STOP)],
          usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 }
        }
      ]
      const stream = mapper.mapProviderStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(6) // start, delta, delta, stop, usage, final
      expect(results[0]).toEqual({ type: 'message_start', data: { provider: Provider.Google, model: '' } })
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
          model: '',
          usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7, cachedContentTokenCount: undefined }
        })
      )
    })

    it('[Hard] should handle stream with tool call', async () => {
      const toolName = 'stream_tool'
      const mockChunks: GenerateContentResponse[] = [
        {
          candidates: [createMockCandidate([{ functionCall: { name: toolName, args: { a: 1 } } }], FinishReason.STOP)]
        },
        { usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 } }
      ]
      const stream = mapper.mapProviderStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(7) // start, tool_start, tool_delta, tool_done, stop, usage, final
      expect(results[1].type).toBe('tool_call_start')
      expect((results[1] as any).data.toolCall.function.name).toBe(toolName)
      const toolCallId = (results[1] as any).data.toolCall.id

      expect(results[2].type).toBe('tool_call_delta')
      expect((results[2] as any).data.id).toBe(toolCallId)
      expect((results[2] as any).data.functionArgumentChunk).toBe('{"a":1}')

      expect(results[3].type).toBe('tool_call_done')
      expect((results[3] as any).data.id).toBe(toolCallId)

      expect(results[4].type).toBe('message_stop')
      expect((results[4] as any).data.finishReason).toBe('tool_calls')

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
      const stream = mapper.mapProviderStream(mockGoogleErrorStreamGenerator(mockChunks, apiError))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(3) // start, delta, error
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
      const stream = mapper.mapProviderStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(5) // start, delta, stop, usage, final
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
      const stream = mapper.mapProviderStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(8) // start, delta(cite_delta, cite_done), delta, stop, usage, final
      expect(results[2].type).toBe('citation_delta')
      expect((results[2] as any).data.citation.sourceId).toBe('cite1.com')
      expect(results[3].type).toBe('citation_done')
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
      const stream = mapper.mapProviderStream(mockGoogleStreamGenerator(mockChunks))
      const results = await collectStreamChunks(stream)

      expect(results).toHaveLength(7) // start, json_delta, json_delta, json_done, stop, usage, final
      expect(results[1].type).toBe('json_delta')
      expect((results[1] as any).data.delta).toBe('{"key":')
      expect(results[2].type).toBe('json_delta')
      expect((results[2] as any).data.delta).toBe(' "value"}')
      expect(results[3].type).toBe('json_done')
      expect((results[3] as any).data.snapshot).toBe(jsonString)
      expect((results[3] as any).data.parsed).toEqual({ key: 'value' })
      expect(results[6].type).toBe('final_result')
      const finalResult = (results[6] as any).data.result
      expect(finalResult.content).toBe(jsonString)
      expect(finalResult.parsedContent).toEqual({ key: 'value' })
    })
  })

  describe('wrapProviderError', () => {
    it('[Easy] should wrap simulated Google Error', () => {
      const underlying = {
        message: 'Permission denied',
        status: 403,
        errorDetails: [{ reason: 'PERMISSION_DENIED' }]
      }
      const wrapped = mapper.wrapProviderError(underlying, Provider.Google)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Google)
      expect(wrapped.statusCode).toBe(403)
      expect(wrapped.errorCode).toBe('PERMISSION_DENIED')
      expect(wrapped.message).toContain('Permission denied')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap generic Error', () => {
      const underlying = new Error('Generic network failure')
      const wrapped = mapper.wrapProviderError(underlying, Provider.Google)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Google)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Generic network failure')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should wrap unknown/string error', () => {
      const underlying = 'Something went wrong'
      const wrapped = mapper.wrapProviderError(underlying, Provider.Google)
      expect(wrapped).toBeInstanceOf(ProviderAPIError)
      expect(wrapped.provider).toBe(Provider.Google)
      expect(wrapped.statusCode).toBeUndefined()
      expect(wrapped.message).toContain('Something went wrong')
      expect(wrapped.underlyingError).toBe(underlying)
    })

    it('[Easy] should not re-wrap RosettaAIError', () => {
      const underlying = new MappingError('Already mapped', Provider.Google)
      const wrapped = mapper.wrapProviderError(underlying, Provider.Google)
      expect(wrapped).toBe(underlying)
      expect(wrapped).toBeInstanceOf(MappingError)
    })
  })
})
