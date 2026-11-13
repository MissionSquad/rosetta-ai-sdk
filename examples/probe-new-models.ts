/**
 * Live capability probe for new Anthropic and OpenAI models.
 *
 * Sends the request shapes Rosetta / MissionSquad Core produce (and the variants they would
 * need) straight to the provider APIs and prints what each model accepts or rejects. Use it
 * before adding a model family to the mappers, so gating is based on observed behaviour.
 *
 * It deliberately bypasses the Rosetta mappers: the point is to learn what the provider does
 * with a given payload, not what Rosetta currently strips from it.
 *
 * Usage (from the repo root, Node >= 22.18 runs TypeScript directly):
 *   node examples/probe-new-models.ts
 *   node examples/probe-new-models.ts --provider=anthropic
 *   node examples/probe-new-models.ts --provider=openai --models=gpt-6-sol,gpt-6.1-sol
 *
 * Reads ANTHROPIC_API_KEY / OPENAI_API_KEY from .env. Every request is small, but they are
 * billed: a full run is roughly 80 Anthropic and 70 OpenAI requests.
 */

import Anthropic from '@anthropic-ai/sdk'
import OpenAI from 'openai'
import dotenv from 'dotenv'

dotenv.config()

const DEFAULT_ANTHROPIC_MODELS = [
  'claude-opus-5', // baseline: known to work on the platform today
  'claude-opus-5-5',
  'claude-sonnet-5-5',
  'claude-haiku-5-5',
  'claude-fable-5-1',
  'claude-fable-5-2'
]

const DEFAULT_OPENAI_MODELS = [
  'gpt-5.6-sol', // baseline: known to work on the platform today
  'gpt-6-sol',
  'gpt-6-luna',
  'gpt-6-astra',
  'gpt-6.1-sol',
  'gpt-6.1-terra'
]

const THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'

type JsonRecord = Record<string, unknown>

interface ProbeResult {
  model: string
  probe: string
  status: number | 'ok' | 'error'
  detail: string
}

interface CallOutcome {
  ok: boolean
  status: number | 'ok' | 'error'
  body?: JsonRecord
  message: string
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function recordArray(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function shorten(text: string, max = 220): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
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

// ---------------------------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------------------------

const WEATHER_TOOL: JsonRecord = {
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  input_schema: {
    type: 'object',
    properties: { city: { type: 'string', description: 'City name' } },
    required: ['city'],
    additionalProperties: false
  }
}

const EXTRA_TOOL: JsonRecord = {
  name: 'get_time',
  description: 'Get the current local time for a city.',
  input_schema: {
    type: 'object',
    properties: { city: { type: 'string', description: 'City name' } },
    required: ['city'],
    additionalProperties: false
  }
}

const TOOL_USER_PROMPT =
  'Work out which of Paris, Lyon and Marseille is furthest north, then call get_weather for that city and tell me the result.'

async function callAnthropic(client: Anthropic, body: JsonRecord, beta?: string): Promise<CallOutcome> {
  try {
    const response = await client.post<unknown>('/v1/messages', {
      body,
      ...(beta ? { headers: { 'anthropic-beta': beta } } : {})
    })
    return { ok: true, status: 'ok', body: isRecord(response) ? response : undefined, message: '' }
  } catch (error) {
    if (error instanceof Anthropic.APIError) {
      return { ok: false, status: typeof error.status === 'number' ? error.status : 'error', message: error.message }
    }
    return { ok: false, status: 'error', message: error instanceof Error ? error.message : String(error) }
  }
}

function describeAnthropicResponse(body: JsonRecord | undefined): string {
  if (!body) return 'no body'
  const blocks = recordArray(body.content)
  const types = blocks.map(block => String(block.type)).join(',') || 'none'
  const parts = [`stop_reason=${String(body.stop_reason)}`, `blocks=${types}`]
  if (Array.isArray(body.input_transformations)) {
    const transformations = recordArray(body.input_transformations)
      .map(entry => `${String(entry.type)}:${String(entry.reason)}@${String(entry.path)}`)
      .join(' ')
    parts.push(`input_transformations=[${transformations}]`)
  }
  return parts.join(' ')
}

async function probeAnthropicModel(client: Anthropic, model: string): Promise<ProbeResult[]> {
  const results: ProbeResult[] = []
  const record = (probe: string, outcome: CallOutcome): void => {
    results.push({
      model,
      probe,
      status: outcome.status,
      detail: outcome.ok ? describeAnthropicResponse(outcome.body) : shorten(outcome.message)
    })
  }
  const ping = [{ role: 'user', content: 'Respond with "OK".' }]

  record(
    'baseline (no sampling, no thinking param)',
    await callAnthropic(client, { model, max_tokens: 256, messages: ping })
  )
  record(
    'temperature 0.1 (model-add test today)',
    await callAnthropic(client, { model, max_tokens: 256, temperature: 0.1, messages: ping })
  )
  record('top_p 0.9', await callAnthropic(client, { model, max_tokens: 256, top_p: 0.9, messages: ping }))
  record('max_tokens 10 (model-add test today)', await callAnthropic(client, { model, max_tokens: 10, messages: ping }))
  record(
    'thinking enabled + budget_tokens 1024',
    await callAnthropic(client, {
      model,
      max_tokens: 2048,
      thinking: { type: 'enabled', budget_tokens: 1024 },
      messages: ping
    })
  )
  record(
    'thinking adaptive + display summarized',
    await callAnthropic(client, {
      model,
      max_tokens: 2048,
      thinking: { type: 'adaptive', display: 'summarized' },
      messages: ping
    })
  )
  record(
    'tool_choice any (forced tool use)',
    await callAnthropic(client, {
      model,
      max_tokens: 1024,
      tools: [WEATHER_TOOL],
      tool_choice: { type: 'any' },
      messages: [{ role: 'user', content: 'What is the weather in Paris?' }]
    })
  )

  record(
    'structured output (output_config.format json_schema)',
    await callAnthropic(client, {
      model,
      max_tokens: 1024,
      output_config: {
        format: {
          type: 'json_schema',
          schema: {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false
          }
        }
      },
      messages: [{ role: 'user', content: 'Reply with the word OK in the answer field.' }]
    })
  )

  // --- Multi-turn tool use: produce a tool_use turn, then replay it in different shapes. ---
  const firstUser = { role: 'user', content: TOOL_USER_PROMPT }
  const system = 'You are a concise assistant.'
  // High effort on turn 1 only: adaptive thinking skips trivial turns, and the replay probes
  // need at least one thinking block to check. Effort is not part of the bound prefix.
  const firstTurnBody: JsonRecord = {
    model,
    max_tokens: 8192,
    system,
    tools: [WEATHER_TOOL],
    tool_choice: { type: 'auto' },
    messages: [firstUser]
  }
  let firstTurn = await callAnthropic(client, { ...firstTurnBody, output_config: { effort: 'xhigh' } })
  if (!firstTurn.ok && firstTurn.status === 400 && firstTurn.message.includes('effort')) {
    // Pre-4.7 models reject the effort parameter or the xhigh level; fall back to the default.
    firstTurn = await callAnthropic(client, firstTurnBody)
  }
  record('tool loop turn 1 (tools, auto)', firstTurn)

  const assistantBlocks = recordArray(firstTurn.body?.content)
  const toolUses = assistantBlocks.filter(block => block.type === 'tool_use')
  const thinkingCount = assistantBlocks.filter(block => block.type === 'thinking' || block.type === 'redacted_thinking')
    .length
  if (!firstTurn.ok || toolUses.length === 0) {
    results.push({
      model,
      probe: 'tool loop replay',
      status: 'error',
      detail: 'skipped: turn 1 produced no tool_use block'
    })
    return results
  }
  results.push({
    model,
    probe: 'tool loop turn 1 thinking blocks',
    status: 'ok',
    detail: `${thinkingCount} replayable thinking block(s)${
      thinkingCount === 0 ? ' — history-check probes below prove nothing for this model' : ''
    }`
  })

  const toolResults = {
    role: 'user',
    content: toolUses.map(block => ({
      type: 'tool_result',
      tool_use_id: block.id,
      content: '{"temp_c":18,"sky":"clear"}'
    }))
  }
  const verbatimAssistant = { role: 'assistant', content: assistantBlocks }
  const strippedAssistant = {
    role: 'assistant',
    content: assistantBlocks.filter(block => block.type !== 'thinking' && block.type !== 'redacted_thinking')
  }
  const editedUser = {
    role: 'user',
    content: `${TOOL_USER_PROMPT}\n\nContext from documents that you must use:\n\nParis is the capital of France.`
  }
  const observed = { type: 'adaptive', block_binding: { prefix_mismatch_behavior: 'drop_block' } }

  const replay = (overrides: JsonRecord): JsonRecord => ({
    model,
    max_tokens: 4096,
    system,
    tools: [WEATHER_TOOL],
    tool_choice: { type: 'auto' },
    messages: [firstUser, verbatimAssistant, toolResults],
    ...overrides
  })

  // What the platform sends today (no beta header): does the request succeed?
  record('replay unchanged (PTC shape today)', await callAnthropic(client, replay({})))
  record(
    'replay with thinking stripped (non-PTC shape today)',
    await callAnthropic(client, replay({ messages: [firstUser, strippedAssistant, toolResults] }))
  )
  record(
    'replay after earlier user message edited (RAG/scrape shape)',
    await callAnthropic(client, replay({ messages: [editedUser, verbatimAssistant, toolResults] }))
  )
  record(
    'replay after a tool was added (MCP rediscovery shape)',
    await callAnthropic(client, replay({ tools: [WEATHER_TOOL, EXTRA_TOOL] }))
  )
  record(
    'replay after system prompt changed',
    await callAnthropic(client, replay({ system: `${system} Today is Thursday.` }))
  )
  const withoutTools = replay({})
  delete withoutTools.tools
  delete withoutTools.tool_choice
  record('replay with tools removed (tool-failure finalization shape)', await callAnthropic(client, withoutTools))

  // Same shapes with the binding-controls beta: reports which blocks fail the history check
  // even on accounts where the check is not enforced.
  record(
    '[observed] replay unchanged',
    await callAnthropic(client, replay({ thinking: observed }), THINKING_BINDING_BETA)
  )
  record(
    '[observed] earlier user message edited',
    await callAnthropic(
      client,
      replay({ thinking: observed, messages: [editedUser, verbatimAssistant, toolResults] }),
      THINKING_BINDING_BETA
    )
  )
  record(
    '[observed] tool added',
    await callAnthropic(
      client,
      replay({ thinking: observed, tools: [WEATHER_TOOL, EXTRA_TOOL] }),
      THINKING_BINDING_BETA
    )
  )
  record(
    '[observed] system prompt changed',
    await callAnthropic(
      client,
      replay({ thinking: observed, system: `${system} Today is Thursday.` }),
      THINKING_BINDING_BETA
    )
  )
  const observedWithoutTools = replay({ thinking: observed })
  delete observedWithoutTools.tools
  delete observedWithoutTools.tool_choice
  record('[observed] tools removed', await callAnthropic(client, observedWithoutTools, THINKING_BINDING_BETA))

  return results
}

// ---------------------------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------------------------

const OPENAI_CHAT_TOOL: JsonRecord = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
      additionalProperties: false
    }
  }
}

const OPENAI_RESPONSES_TOOL: JsonRecord = {
  type: 'function',
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  parameters: {
    type: 'object',
    properties: { city: { type: 'string' } },
    required: ['city'],
    additionalProperties: false
  },
  strict: true
}

async function callOpenAI(
  client: OpenAI,
  path: '/chat/completions' | '/responses',
  body: JsonRecord
): Promise<CallOutcome> {
  try {
    const response = await client.post<unknown>(path, { body })
    return { ok: true, status: 'ok', body: isRecord(response) ? response : undefined, message: '' }
  } catch (error) {
    if (error instanceof OpenAI.APIError) {
      return { ok: false, status: typeof error.status === 'number' ? error.status : 'error', message: error.message }
    }
    return { ok: false, status: 'error', message: error instanceof Error ? error.message : String(error) }
  }
}

function describeChatCompletion(body: JsonRecord | undefined): string {
  const choice = recordArray(body?.choices)[0]
  if (!choice) return 'no choices'
  const message = isRecord(choice.message) ? choice.message : {}
  const content = typeof message.content === 'string' ? message.content : ''
  const toolCalls = recordArray(message.tool_calls).length
  return `finish_reason=${String(choice.finish_reason)} content=${JSON.stringify(
    shorten(content, 40)
  )} tool_calls=${toolCalls}`
}

function describeResponse(body: JsonRecord | undefined): string {
  if (!body) return 'no body'
  const output = recordArray(body.output)
  const types = output.map(item => String(item.type)).join(',') || 'none'
  const text = output
    .filter(item => item.type === 'message')
    .flatMap(item => recordArray(item.content))
    .map(part => (typeof part.text === 'string' ? part.text : ''))
    .join('')
  const incomplete = isRecord(body.incomplete_details) ? ` incomplete=${String(body.incomplete_details.reason)}` : ''
  const reasoning = isRecord(body.reasoning) ? ` effort=${String(body.reasoning.effort)}` : ''
  return `status=${String(body.status)}${incomplete}${reasoning} output=${types} text=${JSON.stringify(
    shorten(text, 40)
  )}`
}

async function probeOpenAIModel(client: OpenAI, model: string): Promise<ProbeResult[]> {
  const results: ProbeResult[] = []
  const chat = async (probe: string, body: JsonRecord): Promise<void> => {
    const outcome = await callOpenAI(client, '/chat/completions', { model, ...body })
    results.push({
      model,
      probe: `chat: ${probe}`,
      status: outcome.status,
      detail: outcome.ok ? describeChatCompletion(outcome.body) : shorten(outcome.message)
    })
  }
  const responses = async (probe: string, body: JsonRecord): Promise<void> => {
    const outcome = await callOpenAI(client, '/responses', { model, store: false, ...body })
    results.push({
      model,
      probe: `responses: ${probe}`,
      status: outcome.status,
      detail: outcome.ok ? describeResponse(outcome.body) : shorten(outcome.message)
    })
  }
  const messages = [{ role: 'user', content: 'Respond with "OK".' }]
  const input = 'Respond with "OK".'
  const weather = 'What is the weather in Paris? Use the tool.'

  await chat('temperature 0.1 + 10 tokens (model-add test today)', {
    messages,
    temperature: 0.1,
    max_completion_tokens: 10
  })
  await chat('no temperature + 10 tokens', { messages, max_completion_tokens: 10 })
  await chat('no temperature + 2000 tokens', { messages, max_completion_tokens: 2000 })
  await chat('reasoning_effort none + temperature 0.1', {
    messages,
    reasoning_effort: 'none',
    temperature: 0.1,
    max_completion_tokens: 200
  })
  await chat('reasoning_effort max', { messages, reasoning_effort: 'max', max_completion_tokens: 2000 })
  await chat('function tool, reasoning_effort omitted', {
    messages: [{ role: 'user', content: weather }],
    tools: [OPENAI_CHAT_TOOL],
    max_completion_tokens: 2000
  })
  await chat('function tool, reasoning_effort none', {
    messages: [{ role: 'user', content: weather }],
    tools: [OPENAI_CHAT_TOOL],
    reasoning_effort: 'none',
    max_completion_tokens: 2000
  })

  await responses('16 output tokens, default effort (model-add test via Responses)', {
    input,
    max_output_tokens: 16,
    include: ['reasoning.encrypted_content']
  })
  await responses('2000 output tokens, default effort', { input, max_output_tokens: 2000 })
  await responses('temperature 0.1, default effort', { input, temperature: 0.1, max_output_tokens: 2000 })
  for (const effort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']) {
    await responses(`reasoning effort ${effort}`, { input, reasoning: { effort }, max_output_tokens: 4000 })
  }
  await responses('effort none + temperature 0.1', {
    input,
    reasoning: { effort: 'none' },
    temperature: 0.1,
    max_output_tokens: 2000
  })
  await responses('effort none + 16 output tokens', { input, reasoning: { effort: 'none' }, max_output_tokens: 16 })
  await responses('text.verbosity low', { input, text: { verbosity: 'low' }, max_output_tokens: 2000 })
  await responses('reasoning summary auto', { input, reasoning: { summary: 'auto' }, max_output_tokens: 2000 })
  await responses('function tool, default effort', {
    input: weather,
    tools: [OPENAI_RESPONSES_TOOL],
    max_output_tokens: 2000
  })

  return results
}

// ---------------------------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------------------------

function printResults(title: string, results: ProbeResult[]): void {
  console.log(`\n=== ${title} ===`)
  let currentModel = ''
  for (const result of results) {
    if (result.model !== currentModel) {
      currentModel = result.model
      console.log(`\n${currentModel}`)
    }
    console.log(`  [${String(result.status).padEnd(5)}] ${result.probe}\n          ${result.detail}`)
  }
}

async function main(): Promise<void> {
  const { provider, models } = parseArgs()

  if (provider === 'anthropic' || provider === 'all') {
    if (!process.env.ANTHROPIC_API_KEY) {
      console.log('\nSkipping Anthropic: ANTHROPIC_API_KEY is not set.')
    } else {
      const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY, maxRetries: 1 })
      const targets = provider === 'anthropic' && models ? models : DEFAULT_ANTHROPIC_MODELS
      const settled = await Promise.all(targets.map(model => probeAnthropicModel(client, model)))
      printResults('Anthropic', settled.flat())
    }
  }

  if (provider === 'openai' || provider === 'all') {
    if (!process.env.OPENAI_API_KEY) {
      console.log('\nSkipping OpenAI: OPENAI_API_KEY is not set.')
    } else {
      const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY, maxRetries: 1 })
      const targets = provider === 'openai' && models ? models : DEFAULT_OPENAI_MODELS
      // Fail fast on a bad key instead of printing one 401 per probe.
      try {
        await client.models.list()
      } catch (error) {
        if (error instanceof OpenAI.AuthenticationError) {
          console.log('\nSkipping OpenAI: OPENAI_API_KEY was rejected (401).')
          return
        }
        throw error
      }
      const settled = await Promise.all(targets.map(model => probeOpenAIModel(client, model)))
      printResults('OpenAI', settled.flat())
    }
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 1
})
