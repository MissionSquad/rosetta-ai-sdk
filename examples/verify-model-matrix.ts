/* eslint-disable no-console */
/**
 * End-to-end model matrix: runs every listed model through RosettaAI (mappers included) with the
 * request shapes MissionSquad Core produces, and prints pass/fail per model per step.
 *
 * Unlike probe-new-models.ts (which calls the provider APIs directly to learn what they accept),
 * this verifies that Rosetta's gating makes each model work. Run it after changing a mapper,
 * a capability table, or the static model list.
 *
 * Steps per model:
 *   basic     generate with temperature 0.1 (Core's default agent temperature) and 2000 tokens
 *   tools     generate tool loop with thinking requested, replaying provider state each turn
 *   fallback  mid-loop request with the tools removed (Core's tool-failure finalization shape)
 *   stream    the same tool loop through stream()
 *
 * Usage (from the repo root; requests are billed, about 10 small calls per model):
 *   examples/node_modules/.bin/ts-node examples/verify-model-matrix.ts
 *   examples/node_modules/.bin/ts-node examples/verify-model-matrix.ts --provider=openai
 *   examples/node_modules/.bin/ts-node examples/verify-model-matrix.ts --provider=anthropic --models=claude-fable-5-2,claude-opus-5
 */
import dotenv from 'dotenv'
import { z } from 'zod'
import { RosettaAI, Provider, RosettaTool, RosettaMessage, GenerateParams, GenerateResult } from '../src'

dotenv.config()

const ANTHROPIC_MODELS = [
  'claude-haiku-4-5',
  'claude-sonnet-4-5',
  'claude-opus-4-5',
  'claude-sonnet-4-6',
  'claude-opus-4-6',
  'claude-opus-4-7',
  'claude-opus-4-8',
  'claude-sonnet-5',
  'claude-opus-5',
  'claude-fable-5',
  'claude-opus-5-5',
  'claude-sonnet-5-5',
  'claude-haiku-5-5',
  'claude-fable-5-1',
  'claude-fable-5-2'
]

const OPENAI_MODELS = [
  'gpt-4o',
  'gpt-4.1',
  'gpt-5',
  'gpt-5-mini',
  'gpt-5.1',
  'gpt-5.2',
  'gpt-5.4',
  'gpt-5.6-sol',
  'gpt-5.6-terra',
  'gpt-5.6-luna',
  'gpt-6-sol',
  'gpt-6-luna',
  'gpt-6-astra',
  'gpt-6.1-sol',
  'gpt-6.1-terra'
]

type MatrixProvider = Provider.Anthropic | Provider.OpenAI
type StepName = 'basic' | 'tools' | 'fallback' | 'stream'

interface StepResult {
  step: StepName
  ok: boolean
  detail: string
}

const weatherTool: RosettaTool = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string', description: 'City name' } },
      required: ['city'],
      additionalProperties: false
    },
    zodSchema: z.object({ city: z.string() })
  }
}

const TOOL_PROMPT =
  'Work out which of Paris, Lyon and Marseille is furthest north, then call get_weather for that city and tell me the result in one sentence.'
const MAX_ITERATIONS = 4

function shorten(text: string, max = 160): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

function errorText(error: unknown): string {
  return shorten(error instanceof Error ? error.message : String(error))
}

/** Assistant history message that replays provider state the way MissionSquad Core does. */
function assistantMessage(result: GenerateResult): RosettaMessage {
  const rawContentBlocks = result.providerState?.anthropic?.rawContentBlocks
  return {
    role: 'assistant',
    content: result.content,
    toolCalls: result.toolCalls,
    ...(Array.isArray(rawContentBlocks) && rawContentBlocks.length > 0
      ? { providerState: { anthropic: { rawContentBlocks } } }
      : {})
  }
}

function toolResultMessages(result: GenerateResult): RosettaMessage[] {
  return (result.toolCalls ?? []).map(toolCall => ({
    role: 'tool' as const,
    toolCallId: toolCall.id,
    content: JSON.stringify({ temp_c: 18, sky: 'clear' })
  }))
}

function baseParams(provider: MatrixProvider, model: string, messages: RosettaMessage[]): GenerateParams {
  return { provider, model, messages, temperature: 0.1, maxTokens: 4000 }
}

async function streamToResult(rosetta: RosettaAI, params: GenerateParams): Promise<GenerateResult> {
  let finalResult: GenerateResult | undefined
  for await (const chunk of rosetta.stream(params)) {
    if (chunk.type === 'error') throw chunk.data.error
    if (chunk.type === 'final_result') finalResult = chunk.data.result
  }
  if (!finalResult) throw new Error('Stream ended without a final_result chunk.')
  return finalResult
}

async function runToolLoop(
  provider: MatrixProvider,
  model: string,
  call: (params: GenerateParams) => Promise<GenerateResult>,
  onFirstToolTurn?: (messages: RosettaMessage[]) => Promise<void>
): Promise<StepResult['detail']> {
  const messages: RosettaMessage[] = [{ role: 'user', content: TOOL_PROMPT }]
  let toolCallCount = 0
  let sawThinking = false

  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const result = await call({
      ...baseParams(provider, model, messages),
      tools: [weatherTool],
      toolChoice: 'auto',
      thinking: true
    })
    if (result.thinkingSteps) sawThinking = true
    messages.push(assistantMessage(result))

    if (!result.toolCalls || result.toolCalls.length === 0) {
      if (toolCallCount === 0) throw new Error('Model answered without calling the tool.')
      if (!result.content || result.content.trim() === '') throw new Error('Final answer was empty.')
      return `${toolCallCount} tool call(s), thinking ${sawThinking ? 'shown' : 'not shown'}, answer ${JSON.stringify(
        shorten(result.content, 60)
      )}`
    }

    toolCallCount += result.toolCalls.length
    messages.push(...toolResultMessages(result))
    if (iteration === 0 && onFirstToolTurn) await onFirstToolTurn([...messages])
  }
  throw new Error(`No final answer after ${MAX_ITERATIONS} iterations.`)
}

async function verifyModel(rosetta: RosettaAI, provider: MatrixProvider, model: string): Promise<StepResult[]> {
  const results: StepResult[] = []
  const step = async (name: StepName, run: () => Promise<string>): Promise<void> => {
    try {
      results.push({ step: name, ok: true, detail: await run() })
    } catch (error) {
      results.push({ step: name, ok: false, detail: errorText(error) })
    }
  }

  await step('basic', async () => {
    const result = await rosetta.generate({
      ...baseParams(provider, model, [{ role: 'user', content: 'Respond with "OK".' }]),
      maxTokens: 2000
    })
    if (!result.content || result.content.trim() === '')
      throw new Error(`Empty content (finishReason=${result.finishReason}).`)
    return `content ${JSON.stringify(shorten(result.content, 40))}`
  })

  let fallback: StepResult = { step: 'fallback', ok: false, detail: 'not reached: the tool loop made no tool call' }
  await step('tools', () =>
    runToolLoop(
      provider,
      model,
      params => rosetta.generate(params),
      async messages => {
        try {
          // Same history, tools removed: Core sends this to force a final answer after tool failures.
          const result = await rosetta.generate(baseParams(provider, model, messages))
          fallback = { step: 'fallback', ok: true, detail: `finishReason=${result.finishReason}` }
        } catch (error) {
          fallback = { step: 'fallback', ok: false, detail: errorText(error) }
        }
      }
    )
  )
  results.push(fallback)

  await step('stream', () => runToolLoop(provider, model, params => streamToResult(rosetta, params)))
  return results
}

function parseArgs(): { provider: 'anthropic' | 'openai' | 'all'; models?: string[] } {
  let provider: 'anthropic' | 'openai' | 'all' = 'all'
  let models: string[] | undefined
  for (const arg of process.argv.slice(2)) {
    if (arg.startsWith('--provider=')) {
      const value = arg.slice('--provider='.length)
      if (value !== 'anthropic' && value !== 'openai' && value !== 'all') {
        throw new Error(`Unknown provider '${value}'. Use anthropic, openai or all.`)
      }
      provider = value
    } else if (arg.startsWith('--models=')) {
      models = arg
        .slice('--models='.length)
        .split(',')
        .map(model => model.trim())
        .filter(Boolean)
    } else {
      throw new Error(`Unknown argument '${arg}'.`)
    }
  }
  return { provider, models }
}

async function runProvider(rosetta: RosettaAI, provider: MatrixProvider, models: string[]): Promise<number> {
  const settled = await Promise.all(
    models.map(async model => ({ model, results: await verifyModel(rosetta, provider, model) }))
  )
  let failures = 0
  console.log(`\n=== ${provider} ===`)
  for (const { model, results } of settled) {
    console.log(`\n${model}`)
    for (const result of results) {
      if (!result.ok) failures += 1
      console.log(`  [${result.ok ? 'pass' : 'FAIL'}] ${result.step.padEnd(8)} ${result.detail}`)
    }
  }
  return failures
}

async function main(): Promise<void> {
  const { provider, models } = parseArgs()
  const rosetta = new RosettaAI()
  let failures = 0

  if (provider === 'anthropic' || provider === 'all') {
    failures += await runProvider(
      rosetta,
      Provider.Anthropic,
      provider === 'anthropic' && models ? models : ANTHROPIC_MODELS
    )
  }
  if (provider === 'openai' || provider === 'all') {
    failures += await runProvider(rosetta, Provider.OpenAI, provider === 'openai' && models ? models : OPENAI_MODELS)
  }

  console.log(`\n${failures === 0 ? 'All steps passed.' : `${failures} step(s) failed.`}`)
  if (failures > 0) process.exitCode = 1
}

main().catch((error: unknown) => {
  console.error(errorText(error))
  process.exitCode = 1
})
