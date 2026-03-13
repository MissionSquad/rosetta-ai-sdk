/* eslint-disable no-console */
import Anthropic from '@anthropic-ai/sdk'
import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages'
import dotenv from 'dotenv'
import fs from 'fs'
import path from 'path'
import { z } from 'zod'

import {
  GenerateParams,
  GenerateResult,
  Provider,
  RosettaMessage,
  RosettaTool,
  RosettaToolCallRequest
} from '../src'
import { AnthropicMapper } from '../src/core/mapping/anthropic.mapper'
import { StreamChunk } from '../src/types'

dotenv.config()
dotenv.config({ path: path.resolve(__dirname, '../.env') })
dotenv.config({ path: path.resolve(__dirname, '../../.env') })

type IterationArtifact = {
  iteration: number
  request: unknown
  rawEvents: RawMessageStreamEvent[]
  mappedChunks: StreamChunk[]
  finalResult: GenerateResult | null
  streamError: string | null
  messagesAfterIteration: RosettaMessage[]
}

type ToolExecutor = (toolCall: RosettaToolCallRequest) => Promise<string>

const model = process.env.ROSETTA_PTC_DEBUG_MODEL ?? 'claude-sonnet-4-6'
const maxIterations = parsePositiveInt(process.env.ROSETTA_PTC_DEBUG_MAX_ITERATIONS, 6)
const outputRoot = path.resolve(
  process.cwd(),
  process.env.ROSETTA_PTC_DEBUG_OUTPUT_DIR ?? 'tmp/anthropic-ptc-debug'
)
const prompt =
  process.env.ROSETTA_PTC_DEBUG_PROMPT ??
  [
    'First call getCoreConfig.',
    'Then call listProviders.',
    'Only after both tool calls, summarize the available providers and models.',
    'Do not stop after the first tool call.'
  ].join(' ')

const mockCoreConfig = {
  success: true,
  models: {
    'claude-sonnet-4-6': {
      providerKey: 'anthropic',
      model: 'claude-sonnet-4-6'
    },
    'llama-3.3-70b-versatile': {
      providerKey: 'groq',
      model: 'llama-3.3-70b-versatile'
    }
  },
  providers: ['anthropic', 'groq', 'openai']
}

const mockProviders = {
  success: true,
  providers: {
    anthropic: { providerKey: 'anthropic', capabilities: ['llm'] },
    groq: { providerKey: 'groq', capabilities: ['llm'] },
    openai: { providerKey: 'openai', capabilities: ['llm'] }
  }
}

const tools: RosettaTool[] = [
  {
    type: 'function',
    function: {
      name: 'getCoreConfig',
      description: 'Returns the current core configuration including configured providers and models as JSON.',
      parameters: {
        type: 'object',
        properties: {},
        additionalProperties: false
      },
      zodSchema: z.object({})
    }
  },
  {
    type: 'function',
    function: {
      name: 'listProviders',
      description: 'Returns the configured providers and their capabilities as JSON.',
      parameters: {
        type: 'object',
        properties: {},
        additionalProperties: false
      },
      zodSchema: z.object({})
    }
  }
]

const toolExecutors: Record<string, ToolExecutor> = {
  async getCoreConfig(): Promise<string> {
    return JSON.stringify(mockCoreConfig)
  },
  async listProviders(): Promise<string> {
    return JSON.stringify(mockProviders)
  }
}

async function main(): Promise<void> {
  const apiKey = process.env.ANTHROPIC_API_KEY
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is required.')
  }

  const client = new Anthropic({ apiKey })
  const mapper = new AnthropicMapper()
  const runDirectory = path.join(outputRoot, timestampLabel())
  fs.mkdirSync(runDirectory, { recursive: true })

  const conversationMessages: RosettaMessage[] = [{ role: 'user', content: prompt }]
  let containerId: string | undefined
  const artifacts: IterationArtifact[] = []

  console.log(`Anthropic PTC debug run directory: ${runDirectory}`)
  console.log(`Model: ${model}`)
  console.log(`Prompt: ${prompt}`)

  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const params: GenerateParams = {
      provider: Provider.Anthropic,
      model,
      messages: cloneJson(conversationMessages),
      tools,
      programmaticToolCalling: true,
      toolChoice: 'auto',
      maxTokens: 1024,
      stream: true,
      ...(containerId
        ? {
            providerState: {
              anthropic: {
                containerId
              }
            }
          }
        : {})
    }

    const mappedRequest = mapper.mapToProviderParams(params) as Anthropic.Messages.MessageCreateParamsStreaming
    const rawEvents: RawMessageStreamEvent[] = []
    const mappedChunks: StreamChunk[] = []
    let finalResult: GenerateResult | null = null
    let streamError: string | null = null

    console.log(`\n=== Iteration ${iteration} Request ===`)
    console.log(JSON.stringify(mappedRequest, null, 2))

    const providerStream = await createAnthropicStream(client, mappedRequest)

    for await (const chunk of mapper.mapProviderStream(
      tapAsyncIterable(providerStream, event => {
        rawEvents.push(cloneJson(event))
      }),
      params
    )) {
      mappedChunks.push(cloneJson(chunk))
      if (chunk.type === 'container_info') {
        containerId = chunk.data.containerId
      }
      if (chunk.type === 'error') {
        streamError = chunk.data.error instanceof Error ? chunk.data.error.message : String(chunk.data.error)
        console.error(`=== Iteration ${iteration} Stream Error ===`)
        console.error(streamError)
      }
      if (chunk.type === 'final_result') {
        finalResult = chunk.data.result
        const nextContainerId = finalResult.providerState?.anthropic?.containerId
        if (typeof nextContainerId === 'string' && nextContainerId.trim() !== '') {
          containerId = nextContainerId
        }
      }
    }

    artifacts.push({
      iteration,
      request: cloneJson(mappedRequest),
      rawEvents,
      mappedChunks,
      finalResult: finalResult ? cloneJson(finalResult) : null,
      streamError,
      messagesAfterIteration: cloneJson(conversationMessages)
    })

    if (finalResult == null) {
      writeArtifacts(runDirectory, artifacts)
      throw new Error(
        streamError
          ? `Iteration ${iteration} completed without a final_result chunk. Stream error: ${streamError}`
          : `Iteration ${iteration} completed without a final_result chunk.`
      )
    }

    console.log(`=== Iteration ${iteration} Final Result ===`)
    console.log(
      JSON.stringify(
        {
          finishReason: finalResult.finishReason,
          content: finalResult.content,
          toolCalls: finalResult.toolCalls,
          providerState: finalResult.providerState,
          container: finalResult.container
        },
        null,
        2
      )
    )

    conversationMessages.push(buildAssistantHistoryMessage(finalResult))

    if (!finalResult.toolCalls || finalResult.toolCalls.length === 0) {
      break
    }

    const toolMessages = await executeToolCalls(finalResult.toolCalls)
    conversationMessages.push(...toolMessages)
  }

  writeArtifacts(runDirectory, artifacts)
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback
  }

  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function timestampLabel(): string {
  return new Date().toISOString().replace(/[:.]/g, '-')
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

async function createAnthropicStream(
  client: Anthropic,
  mappedRequest: Anthropic.Messages.MessageCreateParamsStreaming
): Promise<AsyncIterable<RawMessageStreamEvent>> {
  return client.messages.create(mappedRequest, {})
}

async function* tapAsyncIterable<T>(
  iterable: AsyncIterable<T>,
  onValue: (value: T) => void
): AsyncIterable<T> {
  for await (const value of iterable) {
    onValue(value)
    yield value
  }
}

function buildAssistantHistoryMessage(result: GenerateResult): RosettaMessage {
  const assistantMessage: RosettaMessage = {
    role: 'assistant',
    content: result.content,
    ...(result.toolCalls ? { toolCalls: result.toolCalls } : {})
  }

  const rawContentBlocks = result.providerState?.anthropic?.rawContentBlocks
  if (Array.isArray(rawContentBlocks) && rawContentBlocks.length > 0) {
    assistantMessage.providerState = {
      anthropic: {
        rawContentBlocks: cloneJson(rawContentBlocks)
      }
    }
    assistantMessage.rawContentBlocks = cloneJson(rawContentBlocks)
  }

  return assistantMessage
}

async function executeToolCalls(toolCalls: RosettaToolCallRequest[]): Promise<RosettaMessage[]> {
  const messages: RosettaMessage[] = []

  for (const toolCall of toolCalls) {
    const executor = toolExecutors[toolCall.function.name]
    if (!executor) {
      messages.push({
        role: 'tool',
        toolCallId: toolCall.id,
        content: JSON.stringify({ error: `No local executor registered for ${toolCall.function.name}` }),
        isError: true
      })
      continue
    }

    try {
      const toolResult = await executor(toolCall)
      messages.push({
        role: 'tool',
        toolCallId: toolCall.id,
        content: toolResult
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      messages.push({
        role: 'tool',
        toolCallId: toolCall.id,
        content: JSON.stringify({ error: message }),
        isError: true
      })
    }
  }

  return messages
}

function writeArtifacts(runDirectory: string, artifacts: IterationArtifact[]): void {
  fs.mkdirSync(runDirectory, { recursive: true })

  for (const artifact of artifacts) {
    const prefix = `iteration-${artifact.iteration}`
    fs.writeFileSync(path.join(runDirectory, `${prefix}-request.json`), `${JSON.stringify(artifact.request, null, 2)}\n`)
    fs.writeFileSync(path.join(runDirectory, `${prefix}-raw-events.json`), `${JSON.stringify(artifact.rawEvents, null, 2)}\n`)
    fs.writeFileSync(path.join(runDirectory, `${prefix}-mapped-chunks.json`), `${JSON.stringify(artifact.mappedChunks, null, 2)}\n`)
    fs.writeFileSync(path.join(runDirectory, `${prefix}-final-result.json`), `${JSON.stringify(artifact.finalResult, null, 2)}\n`)
    fs.writeFileSync(path.join(runDirectory, `${prefix}-stream-error.txt`), `${artifact.streamError ?? ''}\n`)
    fs.writeFileSync(
      path.join(runDirectory, `${prefix}-messages-after-iteration.json`),
      `${JSON.stringify(artifact.messagesAfterIteration, null, 2)}\n`
    )
  }
}

main().catch(error => {
  console.error('Anthropic PTC debug harness failed.')
  console.error(error)
  process.exitCode = 1
})
