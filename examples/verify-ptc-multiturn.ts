/* eslint-disable no-console */
/**
 * Anthropic programmatic tool calling (PTC) across two user turns, through RosettaAI.
 *
 * Reproduces the history MissionSquad replays on the second turn of a chat: the first turn's
 * tool scaffold (assistant turns with raw content blocks: thinking, server_tool_use,
 * code_execution_tool_result, tool_use) is kept, the final answer is kept as plain text, and a
 * new user message is appended. verify-model-matrix.ts does not cover this (no PTC, one turn).
 *
 * Usage (from the repo root; requests are billed, about 6 calls per model):
 *   examples/node_modules/.bin/ts-node examples/verify-ptc-multiturn.ts
 *   examples/node_modules/.bin/ts-node examples/verify-ptc-multiturn.ts --models=claude-opus-5,claude-fable-5-2
 */
import dotenv from 'dotenv'
import { z } from 'zod'
import { RosettaAI, Provider, RosettaTool, RosettaMessage, GenerateParams, GenerateResult } from '../src'

dotenv.config()

const DEFAULT_MODELS = [
  'claude-sonnet-4-6',
  'claude-opus-4-8',
  'claude-sonnet-5',
  'claude-opus-5',
  'claude-fable-5',
  'claude-opus-5-5',
  'claude-sonnet-5-5',
  'claude-fable-5-1',
  'claude-fable-5-2'
]

const MAX_ITERATIONS = 6

/**
 * How a completed turn is stored before the next user turn:
 *   scaffold  tool turns keep raw blocks, final answer is text only (MissionSquad API today)
 *   complete  tool turns and the final answer all keep their raw blocks
 *   plain     no raw blocks anywhere: plain tool calls, tool results and text
 *   collapsed only the user message and the final answer text are kept
 */
type HistoryMode = 'scaffold' | 'complete' | 'plain' | 'collapsed'

function stringTool(name: string, description: string): RosettaTool {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        properties: { input: { type: 'string', description: 'Input data' } },
        required: ['input'],
        additionalProperties: false
      },
      zodSchema: z.object({ input: z.unknown() })
    }
  }
}

const tools: RosettaTool[] = [
  stringTool('encodeBase64', 'Encode input data to Base64'),
  stringTool('decodeBase64', 'Decode Base64 input data'),
  stringTool('encodeUrl', 'Encode input data to URL-encoded format')
]

function runTool(name: string, input: string): string {
  if (name === 'encodeBase64') return Buffer.from(input, 'utf8').toString('base64')
  if (name === 'decodeBase64') return Buffer.from(input, 'base64').toString('utf8')
  if (name === 'encodeUrl') return encodeURIComponent(input)
  throw new Error(`Unknown tool ${name}`)
}

function shorten(text: string, max = 400): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

function describeBlocks(blocks: unknown[] | undefined): string {
  if (!Array.isArray(blocks)) return 'none'
  return blocks
    .map(block => (typeof block === 'object' && block !== null && 'type' in block ? String(block.type) : 'unknown'))
    .join(',')
}

/**
 * Runs one user turn as a PTC tool loop and appends the history the way MissionSquad stores it:
 * tool-call turns keep their raw content blocks, the final answer is stored as text only.
 */
async function runTurn(
  rosetta: RosettaAI,
  model: string,
  messages: RosettaMessage[],
  mode: HistoryMode
): Promise<string> {
  let containerId: string | undefined
  const turnStart = messages.length

  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const params: GenerateParams = {
      provider: Provider.Anthropic,
      model,
      messages,
      tools,
      toolChoice: 'auto',
      temperature: 0.2,
      maxTokens: 4000,
      programmaticToolCalling: true,
      ...(containerId ? { providerState: { anthropic: { containerId } } } : {})
    }
    const result: GenerateResult = await rosetta.generate(params)
    containerId = result.providerState?.anthropic?.containerId ?? containerId
    const rawContentBlocks = result.providerState?.anthropic?.rawContentBlocks
    console.log(
      `    iteration ${iteration + 1}: finish=${result.finishReason} toolCalls=${result.toolCalls?.length ??
        0} blocks=${describeBlocks(rawContentBlocks)}`
    )

    if (!result.toolCalls || result.toolCalls.length === 0) {
      messages.push({
        role: 'assistant',
        content: result.content,
        ...(mode === 'complete' && Array.isArray(rawContentBlocks) && rawContentBlocks.length > 0
          ? { providerState: { anthropic: { rawContentBlocks } } }
          : {})
      })
      if (mode === 'plain') {
        for (let index = turnStart; index < messages.length; index++) {
          const message = messages[index]
          if (message?.role === 'assistant') {
            messages[index] = {
              role: 'assistant',
              content: message.content,
              toolCalls: message.toolCalls?.map(({ id, type, function: fn }) => ({ id, type, function: fn }))
            }
          }
        }
      }
      if (mode === 'collapsed') {
        messages.splice(turnStart, messages.length - turnStart, { role: 'assistant', content: result.content })
      }
      return result.content ?? ''
    }

    messages.push({
      role: 'assistant',
      content: result.content,
      toolCalls: result.toolCalls,
      ...(Array.isArray(rawContentBlocks) && rawContentBlocks.length > 0
        ? { providerState: { anthropic: { rawContentBlocks } } }
        : {})
    })
    for (const toolCall of result.toolCalls) {
      const parsed = z.object({ input: z.unknown() }).parse(JSON.parse(toolCall.function.arguments))
      messages.push({
        role: 'tool',
        toolCallId: toolCall.id,
        content: JSON.stringify({
          ok: true,
          contentType: 'text/plain',
          text: runTool(toolCall.function.name, String(parsed.input))
        })
      })
    }
  }
  throw new Error(`No final answer after ${MAX_ITERATIONS} iterations.`)
}

async function verifyModel(rosetta: RosettaAI, model: string, mode: HistoryMode): Promise<boolean> {
  console.log(`\n${model} (history: ${mode})`)
  const messages: RosettaMessage[] = [
    { role: 'system', content: 'You are a concise assistant. Use your tools when they help answer the question.' },
    {
      role: 'user',
      content:
        "Think about which is larger, 17^3 or 70^2. Then use the encodeBase64 tool to encode the larger number followed by '-msq-7f3a9c', and use decodeBase64 on the result to verify it. Report the encoded string and the decoded check."
    }
  ]

  try {
    console.log('  turn 1')
    const first = await runTurn(rosetta, model, messages, mode)
    console.log(`  [pass] turn 1: ${shorten(first, 100)}`)
  } catch (error) {
    console.log(`  [FAIL] turn 1: ${shorten(error instanceof Error ? error.message : String(error))}`)
    return false
  }

  messages.push({
    role: 'user',
    content: "Now URL-encode the phrase 'tools & reasoning = ok?' with the encodeUrl tool and tell me the result."
  })
  try {
    console.log('  turn 2')
    const second = await runTurn(rosetta, model, messages, mode)
    console.log(`  [pass] turn 2: ${shorten(second, 100)}`)
    return true
  } catch (error) {
    console.log(`  [FAIL] turn 2: ${shorten(error instanceof Error ? error.message : String(error))}`)
    return false
  }
}

async function main(): Promise<void> {
  const modelsArg = process.argv.slice(2).find(arg => arg.startsWith('--models='))
  const models = modelsArg
    ? modelsArg
        .slice('--models='.length)
        .split(',')
        .map(model => model.trim())
        .filter(Boolean)
    : DEFAULT_MODELS
  const historyArg = process.argv.slice(2).find(arg => arg.startsWith('--history='))
  const requested = historyArg ? historyArg.slice('--history='.length) : 'scaffold'
  if (requested !== 'scaffold' && requested !== 'complete' && requested !== 'plain' && requested !== 'collapsed') {
    throw new Error(`Unknown history mode '${requested}'.`)
  }
  const rosetta = new RosettaAI()

  let failures = 0
  for (const model of models) {
    if (!(await verifyModel(rosetta, model, requested))) failures += 1
  }
  console.log(`\n${failures === 0 ? 'All models passed.' : `${failures} model(s) failed.`}`)
  if (failures > 0) process.exitCode = 1
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 1
})
