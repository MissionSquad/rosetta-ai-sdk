import {
  mapToAzureOpenAIParams,
  mapFromAzureOpenAIResponse,
  mapAzureOpenAIStream,
  mapToAzureOpenAIEmbedParams,
  mapFromAzureOpenAIEmbedResponse
} from '../../../../src/core/mapping/azure.openai.mapper'
import * as OpenAIMapper from '../../../../src/core/mapping/openai.mapper' // Import base mapper to mock
import {
  GenerateParams,
  EmbedParams,
  Provider,
  RosettaAIConfig,
  // RosettaMessage, // Removed unused import
  RosettaImageData,
  StreamChunk
} from '../../../../src/types'
import { ConfigurationError, MappingError } from '../../../../src/errors' // Import MappingError
import OpenAI from 'openai'
import { Stream } from 'openai/streaming'

// Mock the base OpenAI mapper functions
jest.mock('../../../../src/core/mapping/openai.mapper', () => ({
  mapFromOpenAIResponse: jest.fn(),
  mapOpenAIStream: jest.fn(),
  mapFromOpenAIEmbedResponse: jest.fn(),
  // Keep other exports if needed, but these are the ones we mock for delegation checks
  // We don't need the param mappers from the base file here.
  mapRoleToOpenAI: jest.requireActual('../../../../src/core/mapping/openai.mapper').mapRoleToOpenAI, // Use actual role mapper if needed internally by azure mapper
  mapContentForOpenAIRole: jest.requireActual('../../../../src/core/mapping/openai.mapper').mapContentForOpenAIRole // Use actual content mapper
}))

// Mock implementations for delegation checks
const mockMapFromOpenAIBaseResponse = OpenAIMapper.mapFromOpenAIResponse as jest.Mock
const mockMapOpenAIStream = OpenAIMapper.mapOpenAIStream as jest.Mock
const mockMapFromOpenAIBaseEmbedResponse = OpenAIMapper.mapFromOpenAIEmbedResponse as jest.Mock

// Helper async generator for stream delegation test
async function* mockBaseStreamGenerator(): AsyncIterable<StreamChunk> {
  yield { type: 'message_start', data: { provider: Provider.OpenAI, model: 'delegated-model' } }
  yield { type: 'content_delta', data: { delta: 'Delegated ' } }
  yield { type: 'content_delta', data: { delta: 'Stream' } }
  yield { type: 'message_stop', data: { finishReason: 'stop' } }
}

// --- FIX: Define more complete mock objects for delegation tests ---
// Cast to any to bypass strict type checking for the mock object structure
const mockValidStream = ({
  // Simulate the structure of OpenAI's Stream object
  controller: new AbortController(),
  iterator: jest.fn(),
  tee: jest.fn(),
  toReadableStream: jest.fn(),
  [Symbol.asyncIterator]: mockBaseStreamGenerator // Use the helper generator
} as any) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>

const mockValidEmbedResponse = {
  // Simulate the structure of OpenAI's CreateEmbeddingResponse
  object: 'list' as const,
  data: [{ object: 'embedding' as const, index: 0, embedding: [0.1, 0.2] }],
  model: 'delegated-embed-model-from-resp',
  usage: { prompt_tokens: 10, total_tokens: 10 }
} as OpenAI.Embeddings.CreateEmbeddingResponse
// --- END FIX ---

describe('Azure OpenAI Mapper', () => {
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

  beforeEach(() => {
    jest.clearAllMocks()
    // Setup default mock returns for delegation checks
    mockMapFromOpenAIBaseResponse.mockReturnValue({
      content: 'Delegated Response',
      finishReason: 'stop',
      model: 'delegated-model'
    })
    mockMapOpenAIStream.mockImplementation(mockBaseStreamGenerator)
    mockMapFromOpenAIBaseEmbedResponse.mockReturnValue({
      embeddings: [[0.1, 0.2]],
      model: 'delegated-embed-model'
    })
  })

  // --- Easy Tests ---
  describe('[Easy] mapToAzureOpenAIParams', () => {
    it('should throw ConfigurationError if chat deployment ID is missing', () => {
      const configWithoutDeploy: RosettaAIConfig = { ...baseConfig, azureOpenAIDefaultChatDeploymentName: undefined }
      expect(() => mapToAzureOpenAIParams(baseGenerateParams, configWithoutDeploy)).toThrow(ConfigurationError)
      expect(() => mapToAzureOpenAIParams(baseGenerateParams, configWithoutDeploy)).toThrow(
        'Azure chat deployment ID/name must be configured.'
      )
    })

    it('should map basic text messages using default deployment ID', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'system', content: 'System prompt.' },
          { role: 'user', content: 'User query.' },
          { role: 'assistant', content: 'Assistant reply.' }
        ]
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.model).toBe('default-chat-deploy')
      expect(result.messages).toEqual([
        { role: 'system', content: 'System prompt.' },
        { role: 'user', content: 'User query.' },
        { role: 'assistant', content: 'Assistant reply.' }
      ])
      expect(result.stream).toBe(false)
    })

    it('should use deployment ID from config default', () => {
      const result = mapToAzureOpenAIParams(baseGenerateParams, baseConfig)
      expect(result.model).toBe('default-chat-deploy')
    })

    it('should use deployment ID from providerOptions (overriding config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureChatDeploymentId: 'provider-chat-deploy' }
        }
      }
      const result = mapToAzureOpenAIParams(baseGenerateParams, configWithProviderOpts)
      expect(result.model).toBe('provider-chat-deploy')
    })

    it('should use deployment ID from params (overriding providerOptions and config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureChatDeploymentId: 'provider-chat-deploy' }
        }
      }
      const paramsWithOverride: GenerateParams = {
        ...baseGenerateParams,
        providerOptions: { azureChatDeploymentId: 'param-chat-deploy' }
      }
      const result = mapToAzureOpenAIParams(paramsWithOverride, configWithProviderOpts)
      expect(result.model).toBe('param-chat-deploy')
    })

    // --- NEW Easy Tests ---
    it('should map toolChoice: none', () => {
      const params: GenerateParams = { ...baseGenerateParams, toolChoice: 'none' }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.tool_choice).toBe('none')
    })

    it('should map toolChoice: auto', () => {
      const params: GenerateParams = { ...baseGenerateParams, toolChoice: 'auto' }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.tool_choice).toBe('auto')
    })

    it('should map toolChoice: required', () => {
      const params: GenerateParams = { ...baseGenerateParams, toolChoice: 'required' }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.tool_choice).toBe('required')
    })

    it('should map toolChoice: specific function', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        toolChoice: { type: 'function', function: { name: 'my_func' } }
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'my_func' } })
    })

    it('should map responseFormat: text', () => {
      const params: GenerateParams = { ...baseGenerateParams, responseFormat: { type: 'text' } }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.response_format).toEqual({ type: 'text' })
    })

    it('should map responseFormat: json_object', () => {
      const params: GenerateParams = { ...baseGenerateParams, responseFormat: { type: 'json_object' } }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.response_format).toEqual({ type: 'json_object' })
    })

    it('should map temperature, topP, maxTokens, stop', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        temperature: 0.8,
        topP: 0.9,
        maxTokens: 150,
        stop: ['\n', 'stop']
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.temperature).toBe(0.8)
      expect(result.top_p).toBe(0.9)
      expect(result.max_tokens).toBe(150)
      expect(result.stop).toEqual(['\n', 'stop'])
    })
    // --- END NEW Easy Tests ---
  })

  describe('[Easy] mapToAzureOpenAIEmbedParams', () => {
    it('should throw ConfigurationError if embedding deployment ID is missing', () => {
      const configWithoutDeploy: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultEmbeddingDeploymentName: undefined
      }
      expect(() => mapToAzureOpenAIEmbedParams(baseEmbedParams, configWithoutDeploy)).toThrow(ConfigurationError)
      expect(() => mapToAzureOpenAIEmbedParams(baseEmbedParams, configWithoutDeploy)).toThrow(
        'Azure embedding deployment ID/name must be configured.'
      )
    })

    it('should map basic string input using default deployment ID', () => {
      const result = mapToAzureOpenAIEmbedParams(baseEmbedParams, baseConfig)
      expect(result.model).toBe('default-embed-deploy')
      expect(result.input).toBe('Embed this')
    })

    it('should use deployment ID from config default', () => {
      const result = mapToAzureOpenAIEmbedParams(baseEmbedParams, baseConfig)
      expect(result.model).toBe('default-embed-deploy')
    })

    it('should use deployment ID from providerOptions (overriding config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureEmbeddingDeploymentId: 'provider-embed-deploy' }
        }
      }
      const result = mapToAzureOpenAIEmbedParams(baseEmbedParams, configWithProviderOpts)
      expect(result.model).toBe('provider-embed-deploy')
    })

    it('should use deployment ID from params (overriding providerOptions and config)', () => {
      const configWithProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        providerOptions: {
          [Provider.OpenAI]: { azureEmbeddingDeploymentId: 'provider-embed-deploy' }
        }
      }
      const paramsWithOverride: EmbedParams = {
        ...baseEmbedParams,
        providerOptions: { azureEmbeddingDeploymentId: 'param-embed-deploy' }
      }
      const result = mapToAzureOpenAIEmbedParams(paramsWithOverride, configWithProviderOpts)
      expect(result.model).toBe('param-embed-deploy')
    })

    // --- NEW Easy Tests ---
    it('should map encodingFormat: base64', () => {
      const params: EmbedParams = { ...baseEmbedParams, encodingFormat: 'base64' }
      const result = mapToAzureOpenAIEmbedParams(params, baseConfig)
      expect(result.encoding_format).toBe('base64')
    })

    it('should map dimensions', () => {
      const params: EmbedParams = { ...baseEmbedParams, dimensions: 512 }
      const result = mapToAzureOpenAIEmbedParams(params, baseConfig)
      expect(result.dimensions).toBe(512)
    })
    // --- END NEW Easy Tests ---
  })

  // --- Medium Tests ---
  describe('[Medium] mapToAzureOpenAIParams', () => {
    const imageData: RosettaImageData = { mimeType: 'image/png', base64Data: 'test-img-data' }

    it('should map user message with image content', () => {
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
      const result = mapToAzureOpenAIParams(params, baseConfig)
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

    it('should map assistant message with tool calls and null content', () => {
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
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      })
    })

    it('should map tool result message', () => {
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
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 't1',
        content: '{"result": "ok"}'
      })
    })

    it('should map tools array', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        tools: [{ type: 'function', function: { name: 'myFunc', parameters: { type: 'object' } } }]
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.tools).toEqual([{ type: 'function', function: { name: 'myFunc', parameters: { type: 'object' } } }])
    })

    it('should map toolChoice options', () => {
      const paramsAuto: GenerateParams = { ...baseGenerateParams, toolChoice: 'auto' }
      const paramsNone: GenerateParams = { ...baseGenerateParams, toolChoice: 'none' }
      const paramsRequired: GenerateParams = { ...baseGenerateParams, toolChoice: 'required' }
      const paramsFunc: GenerateParams = {
        ...baseGenerateParams,
        toolChoice: { type: 'function', function: { name: 'myFunc' } }
      }
      expect(mapToAzureOpenAIParams(paramsAuto, baseConfig).tool_choice).toBe('auto')
      expect(mapToAzureOpenAIParams(paramsNone, baseConfig).tool_choice).toBe('none')
      expect(mapToAzureOpenAIParams(paramsRequired, baseConfig).tool_choice).toBe('required')
      expect(mapToAzureOpenAIParams(paramsFunc, baseConfig).tool_choice).toEqual({
        type: 'function',
        function: { name: 'myFunc' }
      })
    })

    it('should map responseFormat json_object', () => {
      const params: GenerateParams = { ...baseGenerateParams, responseFormat: { type: 'json_object' } }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.response_format).toEqual({ type: 'json_object' })
    })

    it('should set stream flag and options', () => {
      const params: GenerateParams = { ...baseGenerateParams, stream: true }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.stream).toBe(true)
      expect((result as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming).stream_options).toEqual({
        include_usage: true
      })
    })

    it('should map assistant message with text content and tool calls', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'user', content: 'Call tool and say hi' },
          {
            role: 'assistant',
            content: 'Okay, calling tool.', // Has text content
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
          }
        ]
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: 'Okay, calling tool.', // Content is preserved
        tool_calls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      })
    })

    // --- NEW Medium Tests ---
    it('should map a combination of temp, topP, maxTokens, stop', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        temperature: 0.6,
        topP: 0.85,
        maxTokens: 200,
        stop: ['\n']
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.temperature).toBe(0.6)
      expect(result.top_p).toBe(0.85)
      expect(result.max_tokens).toBe(200)
      expect(result.stop).toEqual(['\n'])
    })

    it('should map user message with multiple image parts', () => {
      const imageData1: RosettaImageData = { mimeType: 'image/jpeg', base64Data: 'jpegdata' }
      const imageData2: RosettaImageData = { mimeType: 'image/png', base64Data: 'pngdata' }
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Compare:' },
              { type: 'image', image: imageData1 },
              { type: 'image', image: imageData2 }
            ]
          }
        ]
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages[0].content).toEqual([
        { type: 'text', text: 'Compare:' },
        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,jpegdata' } },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,pngdata' } }
      ])
    })

    it('should map complex message history with various roles', () => {
      const params: GenerateParams = {
        ...baseGenerateParams,
        messages: [
          { role: 'system', content: 'System prompt.' },
          { role: 'user', content: 'User message 1.' },
          { role: 'assistant', content: 'Assistant reply 1.' },
          {
            role: 'assistant',
            content: null,
            toolCalls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
          },
          { role: 'tool', toolCallId: 't1', content: '{"res": 1}' },
          { role: 'user', content: 'User message 2.' }
        ]
      }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages).toHaveLength(6)
      expect(result.messages[0]).toEqual({ role: 'system', content: 'System prompt.' })
      expect(result.messages[1]).toEqual({ role: 'user', content: 'User message 1.' })
      expect(result.messages[2]).toEqual({ role: 'assistant', content: 'Assistant reply 1.' })
      expect(result.messages[3]).toEqual({
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 't1', type: 'function', function: { name: 'func1', arguments: '{}' } }]
      })
      expect(result.messages[4]).toEqual({ role: 'tool', tool_call_id: 't1', content: '{"res": 1}' })
      expect(result.messages[5]).toEqual({ role: 'user', content: 'User message 2.' })
    })
    // --- END NEW Medium Tests ---
  })

  describe('[Medium] mapToAzureOpenAIEmbedParams', () => {
    it('should map array input', () => {
      const params: EmbedParams = { ...baseEmbedParams, input: ['text1', 'text2'] }
      const result = mapToAzureOpenAIEmbedParams(params, baseConfig)
      expect(result.input).toEqual(['text1', 'text2'])
    })

    it('should map dimensions and encodingFormat', () => {
      const params: EmbedParams = { ...baseEmbedParams, dimensions: 1024, encodingFormat: 'base64' }
      const result = mapToAzureOpenAIEmbedParams(params, baseConfig)
      expect(result.dimensions).toBe(1024)
      expect(result.encoding_format).toBe('base64')
    })

    // --- NEW Medium Tests ---
    it('should map a combination of encodingFormat and dimensions', () => {
      const params: EmbedParams = { ...baseEmbedParams, encodingFormat: 'base64', dimensions: 768 }
      const result = mapToAzureOpenAIEmbedParams(params, baseConfig)
      expect(result.encoding_format).toBe('base64')
      expect(result.dimensions).toBe(768)
    })
    // --- END NEW Medium Tests ---
  })

  // --- Hard Tests ---
  describe('[Hard] Delegation Checks', () => {
    // FIX: Use the corrected mock objects
    const mockResponse = { id: 'res-123' } as OpenAI.Chat.Completions.ChatCompletion
    const mockStream = mockValidStream
    const mockEmbedResponse = mockValidEmbedResponse

    it('mapFromAzureOpenAIResponse should delegate to base mapper', () => {
      const result = mapFromAzureOpenAIResponse(mockResponse, 'my-deploy-id')
      expect(mockMapFromOpenAIBaseResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromOpenAIBaseResponse).toHaveBeenCalledWith(mockResponse, 'my-deploy-id')
      expect(result).toEqual({ content: 'Delegated Response', finishReason: 'stop', model: 'delegated-model' }) // From mock return
    })

    it('mapAzureOpenAIStream should delegate to base stream mapper', async () => {
      const stream = mapAzureOpenAIStream(mockStream)
      // FIX: Explicitly type results array
      const results: StreamChunk[] = []
      for await (const chunk of stream) {
        results.push(chunk)
      }
      expect(mockMapOpenAIStream).toHaveBeenCalledTimes(1)
      expect(mockMapOpenAIStream).toHaveBeenCalledWith(mockStream)
      expect(results).toHaveLength(4) // From mockBaseStreamGenerator
      // FIX: Access type property correctly
      expect(results[0].type).toBe('message_start')
      expect(results[3].type).toBe('message_stop')
    })

    it('mapFromAzureOpenAIEmbedResponse should delegate to base embed mapper', () => {
      const result = mapFromAzureOpenAIEmbedResponse(mockEmbedResponse, 'my-embed-deploy-id')
      expect(mockMapFromOpenAIBaseEmbedResponse).toHaveBeenCalledTimes(1)
      expect(mockMapFromOpenAIBaseEmbedResponse).toHaveBeenCalledWith(mockEmbedResponse, 'my-embed-deploy-id')
      expect(result).toEqual({ embeddings: [[0.1, 0.2]], model: 'delegated-embed-model' }) // From mock return
    })
  })

  describe('[Hard] mapToAzureOpenAIParams', () => {
    it('should map complex message history', () => {
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
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages).toHaveLength(7)
      expect(result.messages[0]).toEqual({ role: 'system', content: 'System.' })
      expect(result.messages[1]).toEqual({ role: 'user', content: 'User 1' })
      expect(result.messages[2]).toEqual({ role: 'assistant', content: 'Assistant 1' })
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
      expect(result.messages[6]).toEqual({ role: 'user', content: 'User 3' })
    })

    // --- NEW Hard Tests ---
    it('should map with empty message array', () => {
      const params: GenerateParams = { ...baseGenerateParams, messages: [] }
      const result = mapToAzureOpenAIParams(params, baseConfig)
      expect(result.messages).toEqual([])
    })

    it('should prioritize deployment IDs correctly (param > providerOptions > config)', () => {
      const config: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultChatDeploymentName: 'config-default',
        providerOptions: { [Provider.OpenAI]: { azureChatDeploymentId: 'provider-option' } }
      }
      const params: GenerateParams = {
        ...baseGenerateParams,
        providerOptions: { azureChatDeploymentId: 'param-override' }
      }
      const result = mapToAzureOpenAIParams(params, config)
      expect(result.model).toBe('param-override')

      const paramsWithoutOverride: GenerateParams = { ...baseGenerateParams }
      const result2 = mapToAzureOpenAIParams(paramsWithoutOverride, config)
      expect(result2.model).toBe('provider-option')

      const configWithoutProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultChatDeploymentName: 'config-default'
      }
      const result3 = mapToAzureOpenAIParams(paramsWithoutOverride, configWithoutProviderOpts)
      expect(result3.model).toBe('config-default')
    })
    // --- END NEW Hard Tests ---
  })

  // --- NEW Hard Tests ---
  describe('[Hard] mapToAzureOpenAIEmbedParams', () => {
    it('should prioritize deployment IDs correctly (param > providerOptions > config)', () => {
      const config: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultEmbeddingDeploymentName: 'config-default-embed',
        providerOptions: { [Provider.OpenAI]: { azureEmbeddingDeploymentId: 'provider-option-embed' } }
      }
      const params: EmbedParams = {
        ...baseEmbedParams,
        providerOptions: { azureEmbeddingDeploymentId: 'param-override-embed' }
      }
      const result = mapToAzureOpenAIEmbedParams(params, config)
      expect(result.model).toBe('param-override-embed')

      const paramsWithoutOverride: EmbedParams = { ...baseEmbedParams }
      const result2 = mapToAzureOpenAIEmbedParams(paramsWithoutOverride, config)
      expect(result2.model).toBe('provider-option-embed')

      const configWithoutProviderOpts: RosettaAIConfig = {
        ...baseConfig,
        azureOpenAIDefaultEmbeddingDeploymentName: 'config-default-embed'
      }
      const result3 = mapToAzureOpenAIEmbedParams(paramsWithoutOverride, configWithoutProviderOpts)
      expect(result3.model).toBe('config-default-embed')
    })

    // This test relies on the base mapper's error handling, just verifying integration
    it('should throw MappingError for invalid input type (delegated)', () => {
      const params = { ...baseEmbedParams, input: 123 } as any // Invalid input type
      // Expect the error to bubble up from the base mapper via the Azure mapper
      expect(() => mapToAzureOpenAIEmbedParams(params, baseConfig)).toThrow(MappingError)
      expect(() => mapToAzureOpenAIEmbedParams(params, baseConfig)).toThrow(
        'Invalid input type for OpenAI embeddings. Expected string or string[].'
      )
    })
  })
  // --- END NEW Hard Tests ---
})
