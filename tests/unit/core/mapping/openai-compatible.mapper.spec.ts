import OpenAI from 'openai'

import { OpenAICompatibleMapper } from '../../../../src/core/mapping/openai-compatible.mapper'
import { CustomProviderConfig, GenerateParams, RosettaMessage, StreamChunk } from '../../../../src/types'

const CUSTOM_PROVIDER = 'custom-mistral'
const REQUEST_MODEL = 'custom-model'

function createConfig(): CustomProviderConfig {
  return {
    providerKey: CUSTOM_PROVIDER,
    mapper: OpenAICompatibleMapper,
    supportedFeatures: ['generate', 'stream'],
    defaultModel: REQUEST_MODEL,
    baseURL: 'https://custom-provider.invalid/v1'
  }
}

function createResponse(content = 'mapped response'): OpenAI.Chat.Completions.ChatCompletion {
  return {
    id: 'chatcmpl-custom',
    object: 'chat.completion',
    created: 1,
    model: REQUEST_MODEL,
    choices: [
      {
        index: 0,
        finish_reason: 'stop',
        logprobs: null,
        message: {
          role: 'assistant',
          content,
          refusal: null
        }
      }
    ]
  }
}

function installChatCreate(mapper: OpenAICompatibleMapper, create: jest.Mock): void {
  Object.defineProperty(mapper, 'openaiClient', {
    configurable: true,
    value: {
      chat: {
        completions: { create }
      }
    }
  })
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be a record`)
  }
  return value as Record<string, unknown>
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value
}

function capturedRequest(create: jest.Mock): Record<string, unknown> {
  if (create.mock.calls.length !== 1) throw new Error('Expected exactly one chat completion request')
  return requireRecord(create.mock.calls[0][0], 'chat completion request')
}

async function collect(source: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of source) chunks.push(chunk)
  return chunks
}

describe('OpenAICompatibleMapper replay metadata', () => {
  it('reattaches validated cloned replay metadata only to a custom assistant message', async () => {
    const config = createConfig()
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn().mockResolvedValue(createResponse())
    installChatCreate(mapper, create)

    const reasoningDetails = [
      {
        type: 'reasoning.encrypted',
        data: 'opaque-continuation',
        metadata: { sequence: 1 }
      }
    ]
    const structuredContent = [
      {
        type: 'thinking',
        thinking: [{ type: 'text', text: 'disclosed thought', metadata: { sequence: 2 } }]
      },
      { type: 'text', text: 'provider-native answer' }
    ]
    const messages: RosettaMessage[] = [
      {
        role: 'user',
        content: 'continue',
        providerState: {
          openAICompatible: {
            reasoningDetails: [{ type: 'reasoning.encrypted', data: 'must-not-attach-to-user' }]
          }
        }
      },
      {
        role: 'assistant',
        content: 'answer-only history',
        providerState: {
          openAICompatible: { reasoningDetails, structuredContent }
        }
      }
    ]
    const params: GenerateParams = { provider: CUSTOM_PROVIDER, model: REQUEST_MODEL, messages }

    await mapper.executeGenerate(undefined, undefined, config, params)

    const request = capturedRequest(create)
    const sentMessages = requireArray(request.messages, 'messages')
    const sentUser = requireRecord(sentMessages[0], 'user message')
    const sentAssistant = requireRecord(sentMessages[1], 'assistant message')
    const sentReasoning = requireArray(sentAssistant.reasoning_details, 'assistant reasoning_details')
    const sentStructured = requireArray(sentAssistant.content, 'assistant structured content')

    expect(sentUser).toEqual({ role: 'user', content: 'continue' })
    expect(sentAssistant).toEqual({
      role: 'assistant',
      content: structuredContent,
      reasoning_details: reasoningDetails
    })
    expect(sentReasoning).not.toBe(reasoningDetails)
    expect(sentStructured).not.toBe(structuredContent)
    expect(requireRecord(sentReasoning[0], 'reasoning detail')).not.toBe(reasoningDetails[0])
    expect(requireRecord(sentReasoning[0], 'reasoning detail').metadata).not.toBe(reasoningDetails[0].metadata)
    expect(requireRecord(sentStructured[0], 'structured content')).not.toBe(structuredContent[0])
    expect(requireArray(requireRecord(sentStructured[0], 'structured content').thinking, 'thinking parts')).not.toBe(
      structuredContent[0].thinking
    )
  })

  it('ignores malformed replay state instead of attaching a partial provider extension', async () => {
    const config = createConfig()
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn().mockResolvedValue(createResponse())
    installChatCreate(mapper, create)

    const params: GenerateParams = {
      provider: CUSTOM_PROVIDER,
      model: REQUEST_MODEL,
      messages: [
        {
          role: 'assistant',
          content: 'safe answer',
          providerState: {
            openAICompatible: {
              reasoningDetails: [{ type: 'reasoning.text', text: 'valid-looking' }, 'invalid-entry'],
              structuredContent: [null]
            }
          }
        }
      ]
    }

    await mapper.executeGenerate(undefined, undefined, config, params)

    const sentMessage = requireRecord(
      requireArray(capturedRequest(create).messages, 'messages')[0],
      'assistant message'
    )
    expect(sentMessage).toEqual({ role: 'assistant', content: 'safe answer' })
    expect(sentMessage).not.toHaveProperty('reasoning_details')
  })

  it('never replays reasoning aliases or raw-tag text from provider state', async () => {
    const config = createConfig()
    const mapper = new OpenAICompatibleMapper(config)
    const create = jest.fn().mockResolvedValue(createResponse())
    installChatCreate(mapper, create)

    const unsupportedReplayState = {
      reasoningDetails: undefined,
      reasoning_content: 'reasoning-content secret',
      reasoning: 'reasoning secret',
      thinking: 'thinking secret',
      analysis: 'analysis secret',
      rawTagText: '<think>raw tag secret</think>'
    }
    const params: GenerateParams = {
      provider: CUSTOM_PROVIDER,
      model: REQUEST_MODEL,
      messages: [
        {
          role: 'assistant',
          content: 'clean answer',
          providerState: { openAICompatible: unsupportedReplayState }
        }
      ]
    }

    await mapper.executeGenerate(undefined, undefined, config, params)

    const sentMessage = requireRecord(
      requireArray(capturedRequest(create).messages, 'messages')[0],
      'assistant message'
    )
    expect(sentMessage).toEqual({ role: 'assistant', content: 'clean answer' })
    expect(JSON.stringify(sentMessage)).not.toContain('secret')
    expect(JSON.stringify(sentMessage)).not.toContain('<think>')
  })

  it('keeps replay metadata and message-start provider identity on the actual custom provider stream', async () => {
    const config = createConfig()
    const mapper = new OpenAICompatibleMapper(config)
    const reasoningDetails = [{ type: 'reasoning.encrypted', data: 'opaque-stream-state' }]
    const structuredContent = [
      { type: 'thinking', thinking: [{ type: 'text', text: 'stream thought' }] },
      { type: 'text', text: 'stream answer' }
    ]
    const providerChunks: unknown[] = [
      {
        id: 'chatcmpl-custom-stream',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'custom-provider-model-id',
        choices: [
          {
            index: 0,
            delta: { reasoning_details: reasoningDetails, content: structuredContent },
            finish_reason: null
          }
        ]
      },
      {
        id: 'chatcmpl-custom-stream',
        object: 'chat.completion.chunk',
        created: 2,
        model: 'custom-provider-model-id',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
      }
    ]
    async function* providerStream(): AsyncGenerator<unknown, void, undefined> {
      for (const chunk of providerChunks) yield chunk
    }
    const create = jest.fn().mockResolvedValue(providerStream())
    installChatCreate(mapper, create)

    const params: GenerateParams = {
      provider: CUSTOM_PROVIDER,
      model: REQUEST_MODEL,
      messages: [{ role: 'user', content: 'stream it' }]
    }
    const chunks = await collect(mapper.executeStream(undefined, undefined, config, params))
    const messageStart = chunks.find(
      (chunk): chunk is Extract<StreamChunk, { type: 'message_start' }> => chunk.type === 'message_start'
    )
    const finalResult = chunks.find(
      (chunk): chunk is Extract<StreamChunk, { type: 'final_result' }> => chunk.type === 'final_result'
    )

    expect(messageStart?.data).toEqual({ provider: CUSTOM_PROVIDER, model: 'custom-provider-model-id' })
    expect(finalResult?.data.result.model).toBe('custom-provider-model-id')
    expect(finalResult?.data.result.content).toBe('stream answer')
    expect(finalResult?.data.result.providerState).toEqual({
      openAICompatible: { reasoningDetails, structuredContent }
    })
    expect(finalResult?.data.result.providerState?.openAICompatible?.reasoningDetails).not.toBe(reasoningDetails)
    expect(finalResult?.data.result.providerState?.openAICompatible?.structuredContent).not.toBe(structuredContent)
  })
})
