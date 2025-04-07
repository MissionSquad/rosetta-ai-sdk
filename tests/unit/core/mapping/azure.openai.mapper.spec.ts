import { AzureOpenAIMapper } from '../../../../src/core/mapping/azure.openai.mapper'
// import * as OpenAIMapper from '../../../../src/core/mapping/openai.mapper' // No longer needed
import * as OpenAIAudioMapper from '../../../../src/core/mapping/openai.audio.mapper'
import * as OpenAIEmbedMapper from '../../../../src/core/mapping/openai.embed.mapper'
import * as OpenAICommon from '../../../../src/core/mapping/openai.common' // Import common module
import {
  GenerateParams,
  EmbedParams,
  Provider,
  RosettaAIConfig,
  RosettaImageData,
  StreamChunk,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  EmbedResult,
  GenerateResult
} from '../../../../src/types'
import { ConfigurationError, MappingError, ProviderAPIError, UnsupportedFeatureError } from '../../../../src/errors'
import OpenAI from 'openai'
import { Stream } from 'openai/streaming'
import { Uploadable } from 'openai/uploads'

// Mock the base OpenAI common functions and sub-mappers
// We mock the *exported functions* from the common/sub-mapper modules that Azure mapper delegates to.
jest.mock('../../../../src/core/mapping/openai.common', () => ({
  ...jest.requireActual('../../../../src/core/mapping/openai.common'), // Keep other functions if needed
  mapFromOpenAIResponse: jest.fn(),
  mapOpenAIStream: jest.fn(),
  wrapOpenAIError: jest.fn(err => err) // Simple pass-through for testing
}))

jest.mock('../../../../src/core/mapping/openai.embed.mapper', () => ({
  // mapToOpenAIEmbedParams: jest.fn(), // Azure mapper reimplements this logic for deployment ID
  mapFromOpenAIEmbedResponse: jest.fn()
}))

jest.mock('../../../../src/core/mapping/openai.audio.mapper', () => ({
  mapToOpenAITranscribeParams: jest.fn(),
  mapFromOpenAITranscriptionResponse: jest.fn(),
  mapToOpenAITranslateParams: jest.fn(),
  mapFromOpenAITranslationResponse: jest.fn()
}))

// Mock implementations for delegation checks
const mockMapFromOpenAIBaseResponse = OpenAICommon.mapFromOpenAIResponse as jest.Mock
const mockMapOpenAIStream = OpenAICommon.mapOpenAIStream as jest.Mock
const mockWrapOpenAIError = OpenAICommon.wrapOpenAIError as jest.Mock // Mock the error wrapper
const mockMapFromOpenAIBaseEmbedResponse = OpenAIEmbedMapper.mapFromOpenAIEmbedResponse as jest.Mock
const mockMapToOpenAITranscribeParams = OpenAIAudioMapper.mapToOpenAITranscribeParams as jest.Mock
const mockMapFromOpenAITranscriptionResponse = OpenAIAudioMapper.mapFromOpenAITranscriptionResponse as jest.Mock
const mockMapToOpenAITranslateParams = OpenAIAudioMapper.mapToOpenAITranslateParams as jest.Mock
const mockMapFromOpenAITranslationResponse = OpenAIAudioMapper.mapFromOpenAITranslationResponse as jest.Mock

// Helper async generator for stream delegation test
async function* mockBaseStreamGenerator(): AsyncIterable<StreamChunk> {
  yield { type: 'message_start', data: { provider: Provider.OpenAI, model: 'delegated-model' } }
  yield { type: 'content_delta', data: { delta: 'Delegated ' } }
  yield { type: 'content_delta', data: { delta: 'Stream' } }
  yield { type: 'message_stop', data: { finishReason: 'stop' } }
}

// Define mock objects for delegation tests
const mockValidStream = ({
  controller: new AbortController(),
  iterator: jest.fn(),
  tee: jest.fn(),
  toReadableStream: jest.fn(),
  [Symbol.asyncIterator]: mockBaseStreamGenerator
} as any) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>

// FIX: Add choices array to mockResponse
const mockValidResponse = {
  id: 'res-123',
  object: 'chat.completion',
  created: 1677652288,
  model: 'delegated-model-from-resp',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Delegated Response' },
      finish_reason: 'stop',
      logprobs: null
    }
  ],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
} as OpenAI.Chat.Completions.ChatCompletion

const mockValidEmbedResponse = {
  object: 'list' as const,
  data: [{ object: 'embedding' as const, index: 0, embedding: [0.1, 0.2] }],
  model: 'delegated-embed-model-from-resp',
  usage: { prompt_tokens: 10, total_tokens: 10 }
} as OpenAI.Embeddings.CreateEmbeddingResponse

const mockAudioFile: Uploadable = {
  [Symbol.toStringTag]: 'File',
  name: 'mock.mp3'
} as any

describe('Azure OpenAI Mapper (V2)', () => {
  let mapper: AzureOpenAIMapper
  const baseConfig: RosettaAIConfig = {
    azureOpenAIApiKey: 'dummy-key',
    azureOpenAIEndpoint: 'dummy-endpoint',
    azureOpenAIApiVersion: 'dummy-version',
    azureOpenAIDefaultChatDeploymentName: 'default-chat-deploy',
    azureOpenAIDefaultEmbeddingDeploymentName: 'default-embed-deploy'
  }

  const baseGenerateParams: GenerateParams = {
    provider: Provider.OpenAI, // Provider is OpenAI when using Azure client
    messages: [{ role: 'user', content: 'Hello' }]
  }

  const baseEmbedParams: EmbedParams = {
    provider: Provider.OpenAI,
    input: 'Embed this'
  }

  const baseTranscribeParams: TranscribeParams = {
    provider: Provider.OpenAI,
    model: 'azure-whisper-stt', // Model name used for delegation check
    audio: { data: Buffer.from(''), filename: 'a.mp3', mimeType: 'audio/mpeg' }
  }

  const baseTranslateParams: TranslateParams = {
    provider: Provider.OpenAI,
    model: 'azure-whisper-trans', // Model name used for delegation check
    audio: { data: Buffer.from(''), filename: 'b.wav', mimeType: 'audio/wav' }
  }

  beforeEach(() => {
    mapper = new AzureOpenAIMapper(baseConfig)
    jest.clearAllMocks()
    // Setup default mock returns for delegation checks
    mockMapFromOpenAIBaseResponse.mockReturnValue({
      content: 'Delegated Response',
      finishReason: 'stop',
      model: 'delegated-model'
    } as GenerateResult)
    mockMapOpenAIStream.mockImplementation(mockBaseStreamGenerator)
    mockMapFromOpenAIBaseEmbedResponse.mockReturnValue({
      embeddings: [[0.1, 0.2]],
      model: 'delegated-embed-model'
    } as EmbedResult)
    mockMapToOpenAITranscribeParams.mockReturnValue({ model: 'mapped-stt-params' })
    mockMapFromOpenAITranscriptionResponse.mockReturnValue({ text: 'mapped-transcribed' } as TranscriptionResult)
    mockMapToOpenAITranslateParams.mockReturnValue({ model: 'mapped-translate-params' })
    mockMapFromOpenAITranslationResponse.mockReturnValue({ text: 'mapped-translated' } as TranscriptionResult)
    mockWrapOpenAIError.mockImplementation(err => err) // Reset error wrapper mock
  })

  it('[Easy] should have the correct provider property', () => {
    expect(mapper.provider).toBe(Provider.OpenAI)
  })

  describe('mapToProviderParams (Generate)', () => {
    it('[Easy] should throw ConfigurationError if chat deployment ID is missing', () => {
      const configWithoutDeploy: RosettaAIConfig = { ...baseConfig, azureOpenAIDefaultChatDeploymentName: undefined }
      const mapperNoDeploy = new AzureOpenAIMapper(configWithoutDeploy)
      expect(() => mapperNoDeploy.mapToProviderParams(baseGenerateParams)).toThrow(ConfigurationError)
      expect(() => mapperNoDeploy.mapToProviderParams(baseGenerateParams)).toThrow(
        'Azure chat deployment ID/name must be configured.'
      )
    })

    it('[Easy] should map basic text messages using default deployment ID', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'system', content: 'System prompt.' },
          { role: 'user', content: 'User query.' },
          { role: 'assistant', content: 'Assistant reply.' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.model).toBe('default-chat-deploy')
      expect(result.messages).toEqual([
        { role: 'system', content: 'System prompt.' },
        { role: 'user', content: 'User query.' },
        { role: 'assistant', content: 'Assistant reply.' }
      ])
      expect(result.stream).toBe(false)
    })

    it('[Easy] should use deployment ID from config default', () => {
      const result = mapper.mapToProviderParams(baseGenerateParams)
      expect(result.model).toBe('default-chat-deploy')
    })

    it('[Easy] should use deployment ID from providerOptions (overriding config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureChatDeploymentId: 'provider-chat-deploy' }
        }
      }
      const mapperWithOptions = new AzureOpenAIMapper(configWithProviderOpts)
      const result = mapperWithOptions.mapToProviderParams(baseGenerateParams)
      expect(result.model).toBe('provider-chat-deploy')
    })

    it('[Easy] should use deployment ID from params (overriding providerOptions and config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureChatDeploymentId: 'provider-chat-deploy' }
        }
      }
      const mapperWithOptions = new AzureOpenAIMapper(configWithProviderOpts)
      const paramsWithOverride: GenerateParams = {
        ...baseGenerateParams,
        providerOptions: { azureChatDeploymentId: 'param-chat-deploy' }
      }
      const result = mapperWithOptions.mapToProviderParams(paramsWithOverride)
      expect(result.model).toBe('param-chat-deploy')
    })

    it('[Easy] should map toolChoice: none, auto, required', () => {
      const paramsNone: GenerateParams = { ...baseGenerateParams, toolChoice: 'none' }
      const paramsAuto: GenerateParams = { ...baseGenerateParams, toolChoice: 'auto' }
      const paramsRequired: GenerateParams = { ...baseGenerateParams, toolChoice: 'required' }
      expect(mapper.mapToProviderParams(paramsNone).tool_choice).toBe('none')
      expect(mapper.mapToProviderParams(paramsAuto).tool_choice).toBe('auto')
      expect(mapper.mapToProviderParams(paramsRequired).tool_choice).toBe('required')
    })

    it('[Easy] should map toolChoice: specific function', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        toolChoice: { type: 'function', function: { name: 'my_func' } }
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'my_func' } })
    })

    it('[Easy] should map responseFormat: text, json_object', () => {
      const paramsText: GenerateParams = { ...baseGenerateParams, responseFormat: { type: 'text' } }
      const paramsJson: GenerateParams = { ...baseGenerateParams, responseFormat: { type: 'json_object' } }
      expect(mapper.mapToProviderParams(paramsText).response_format).toEqual({ type: 'text' })
      expect(mapper.mapToProviderParams(paramsJson).response_format).toEqual({ type: 'json_object' })
    })

    it('[Easy] should map temperature, topP, maxTokens, stop', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        temperature: 0.8,
        topP: 0.9,
        maxTokens: 150,
        stop: ['\n', 'stop']
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.temperature).toBe(0.8)
      expect(result.top_p).toBe(0.9)
      expect(result.max_tokens).toBe(150)
      expect(result.stop).toEqual(['\n', 'stop'])
    })

    it('[Medium] should map user message with image content', () => {
      const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'test-img-data' }
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Describe:' },
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
            { type: 'text', text: 'Describe:' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,test-img-data' } }
          ]
        }
      ])
    })

    it('[Medium] should map assistant message with tool calls and null content', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'user', content: 'Call tool' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
          }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      })
    })

    it('[Medium] should map tool result message', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'user', content: 'Call tool' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 't1', content: '{"result": "ok"}' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 't1',
        content: '{"result": "ok"}'
      })
    })

    it('[Medium] should map tools array', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        tools: [{ type: 'function', function: { name: 'myFunc', parameters: { type: 'object' } } }]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.tools).toEqual([{ type: 'function', function: { name: 'myFunc', parameters: { type: 'object' } } }])
    })

    it('[Medium] should set stream flag and options', () => {
      const params: GenerateParams = { ...baseGenerateParams, stream: true }
      const result = mapper.mapToProviderParams(params)
      expect(result.stream).toBe(true)
      expect((result as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming).stream_options).toEqual({
        include_usage: true
      })
    })

    it('[Medium] should throw UnsupportedFeatureError for thinking/grounding', () => {
      const paramsThinking: GenerateParams = { ...baseGenerateParams, thinking: true }
      const paramsGrounding: GenerateParams = { ...baseGenerateParams, grounding: { enabled: true } }
      expect(() => mapper.mapToProviderParams(paramsThinking)).toThrow(UnsupportedFeatureError)
      expect(() => mapper.mapToProviderParams(paramsGrounding)).toThrow(UnsupportedFeatureError)
    })

    it('[Hard] should map complex message history', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'system', content: 'System.' },
          { role: 'user', content: 'User 1' },
          { role: 'assistant', content: 'Assistant 1' },
          {
            role: 'user',
            content: [
              { type: 'text', text: 'User 2 with image' },
              { type: 'image', image: { mimeType: 'image/gif', base64Data: 'gif' } }
            ]
          },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 'tA', type: 'function', function: { name: 'toolA', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 'tA', content: '{"resA": true}' },
          { role: 'user', content: 'User 3' }
        ]
      }
      const result = mapper.mapToProviderParams(params)
      expect(result.messages).toHaveLength(7)
      expect(result.messages[0]).toEqual({ role: 'system', content: 'System.' })
      expect(result.messages[3]).toEqual({
        role: 'user',
        content: [
          { type: 'text', text: 'User 2 with image' },
          { type: 'image_url', image_url: { url: 'data:image/gif;base64,gif' } }
        ]
      })
      expect(result.messages[4]).toEqual({
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 'tA', type: 'function', function: { name: 'toolA', arguments: '{}' } }]
      })
      expect(result.messages[5]).toEqual({ role: 'tool', tool_call_id: 'tA', content: '{"resA": true}' })
    })
  })

  describe('mapToEmbedParams', () => {
    it('[Easy] should throw ConfigurationError if embedding deployment ID is missing', () => {
      const configWithoutDeploy: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultEmbeddingDeploymentName: undefined
      }
      const mapperNoDeploy = new AzureOpenAIMapper(configWithoutDeploy)
      expect(() => mapperNoDeploy.mapToEmbedParams(baseEmbedParams)).toThrow(ConfigurationError)
      expect(() => mapperNoDeploy.mapToEmbedParams(baseEmbedParams)).toThrow(
        'Azure embedding deployment ID/name must be configured.'
      )
    })

    it('[Easy] should map basic string input using default deployment ID', () => {
      const result = mapper.mapToEmbedParams(baseEmbedParams)
      expect(result.model).toBe('default-embed-deploy')
      expect(result.input).toBe('Embed this')
    })

    it('[Easy] should use deployment ID from config default', () => {
      const result = mapper.mapToEmbedParams(baseEmbedParams)
      expect(result.model).toBe('default-embed-deploy')
    })

    it('[Easy] should use deployment ID from providerOptions (overriding config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureEmbeddingDeploymentId: 'provider-embed-deploy' }
        }
      }
      const mapperWithOptions = new AzureOpenAIMapper(configWithProviderOpts)
      const result = mapperWithOptions.mapToEmbedParams(baseEmbedParams)
      expect(result.model).toBe('provider-embed-deploy')
    })

    it('[Easy] should use deployment ID from params (overriding providerOptions and config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureEmbeddingDeploymentId: 'provider-embed-deploy' }
        }
      }
      const mapperWithOptions = new AzureOpenAIMapper(configWithProviderOpts)
      const paramsWithOverride: EmbedParams = {
        ...baseEmbedParams,
        providerOptions: { azureEmbeddingDeploymentId: 'param-embed-deploy' }
      }
      const result = mapperWithOptions.mapToEmbedParams(paramsWithOverride)
      expect(result.model).toBe('param-embed-deploy')
    })

    it('[Easy] should map encodingFormat: base64', () => {
      const params: EmbedParams = { ...baseEmbedParams, encodingFormat: 'base64' }
      const result = mapper.mapToEmbedParams(params)
      expect(result.encoding_format).toBe('base64')
    })

    it('[Easy] should map dimensions', () => {
      const params: EmbedParams = { ...baseEmbedParams, dimensions: 512 }
      const result = mapper.mapToEmbedParams(params)
      expect(result.dimensions).toBe(512)
    })

    it('[Medium] should map array input', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: ['text1', 'text2'] }
      const result = mapper.mapToEmbedParams(params)
      expect(result.input).toEqual(['text1', 'text2'])
    })

    it('[Medium] should map a combination of encodingFormat and dimensions', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: 'Combo', encodingFormat: 'base64', dimensions: 1024 }
      const result = mapper.mapToEmbedParams(params)
      expect(result.encoding_format).toBe('base64')
      expect(result.dimensions).toBe(1024)
    })

    it('[Medium] should throw MappingError for invalid input type', () => {
      const params = { ...baseEmbedParams, input: 123 } as any // Invalid input type
      expect(() => mapper.mapToEmbedParams(params)).toThrow(MappingError)
      expect(() => mapper.mapToEmbedParams(params)).toThrow('Invalid input type for Azure OpenAI embeddings.')
    })
  })

  describe('Delegation Checks', () => {
    // FIX: Use the updated mockValidResponse with choices
    const mockResponse = mockValidResponse
    const mockStream = mockValidStream
    const mockEmbedResponse = mockValidEmbedResponse
    const mockTranscribeResponse = { text: 'transcribed' } as any
    const mockTranslateResponse = { text: 'translated' } as any

    it('[Hard] mapFromProviderResponse should delegate to base mapper', () => {
      const result = mapper.mapFromProviderResponse(mockResponse, 'my-deploy-id')
      expect(mockMapFromOpenAIBaseResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromOpenAIBaseResponse).toHaveBeenCalledWith(mockResponse, 'my-deploy-id')
      expect(result).toEqual({ content: 'Delegated Response', finishReason: 'stop', model: 'delegated-model' })
    })

    it('[Hard] mapProviderStream should delegate to base stream mapper', async () => {
      const stream = mapper.mapProviderStream(mockStream)
      const results: StreamChunk[] = []
      for await (const chunk of stream) {
        results.push(chunk)
      }
      // FIX: Check the correct mock function
      expect(mockMapOpenAIStream).toHaveBeenCalledTimes(1)
      expect(mockMapOpenAIStream).toHaveBeenCalledWith(mockStream, Provider.OpenAI) // Pass provider
      expect(results).toHaveLength(4)
      expect(results[0].type).toBe('message_start')
      expect(results[3].type).toBe('message_stop')
    })

    it('[Hard] mapFromEmbedResponse should delegate to base embed mapper', () => {
      const result = mapper.mapFromEmbedResponse(mockEmbedResponse, 'my-embed-deploy-id')
      expect(mockMapFromOpenAIBaseEmbedResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromOpenAIBaseEmbedResponse).toHaveBeenCalledWith(mockEmbedResponse, 'my-embed-deploy-id')
      expect(result).toEqual({ embeddings: [[0.1, 0.2]], model: 'delegated-embed-model' })
    })

    it('[Hard] mapToTranscribeParams should delegate', () => {
      mapper.mapToTranscribeParams(baseTranscribeParams, mockAudioFile)
      expect(mockMapToOpenAITranscribeParams).toHaveBeenCalledWith(baseTranscribeParams, mockAudioFile)
    })

    it('[Hard] mapFromTranscribeResponse should delegate', () => {
      mapper.mapFromTranscribeResponse(mockTranscribeResponse, 'azure-whisper-stt')
      expect(mockMapFromOpenAITranscriptionResponse).toHaveBeenCalledWith(mockTranscribeResponse, 'azure-whisper-stt')
    })

    it('[Hard] mapToTranslateParams should delegate', () => {
      mapper.mapToTranslateParams(baseTranslateParams, mockAudioFile)
      expect(mockMapToOpenAITranslateParams).toHaveBeenCalledWith(baseTranslateParams, mockAudioFile)
    })

    it('[Hard] mapFromTranslateResponse should delegate', () => {
      mapper.mapFromTranslateResponse(mockTranslateResponse, 'azure-whisper-trans')
      expect(mockMapFromOpenAITranslationResponse).toHaveBeenCalledWith(mockTranslateResponse, 'azure-whisper-trans')
    })
  })

  describe('wrapProviderError', () => {
    it('[Easy] should delegate error wrapping to common function', () => {
      const underlying = new OpenAI.APIError(400, {}, '', {})
      mapper.wrapProviderError(underlying, Provider.OpenAI)
      expect(mockWrapOpenAIError).toHaveBeenCalledTimes(1)
      expect(mockWrapOpenAIError).toHaveBeenCalledWith(underlying, Provider.OpenAI)
    })

    it('[Easy] should return the result from the common function', () => {
      const underlying = new Error('Generic')
      const wrappedError = new ProviderAPIError('Wrapped', Provider.OpenAI)
      mockWrapOpenAIError.mockReturnValue(wrappedError) // Mock the return value

      const result = mapper.wrapProviderError(underlying, Provider.OpenAI)
      expect(result).toBe(wrappedError)
    })
  })
})
