import OpenAI from 'openai'
import type {
  Response,
  ResponseCreateParamsNonStreaming,
  ResponseInput,
  ResponseInputItem,
  ResponseFunctionToolCall,
  ResponseOutputItem,
  ResponseOutputText,
  ResponseReasoningItem,
  ResponseStreamEvent,
  Tool as OpenAIResponsesTool,
  ToolChoiceFunction,
  ToolChoiceOptions
} from 'openai/resources/responses/responses'
import type { Reasoning } from 'openai/resources/shared'

import {
  GenerateParams,
  GenerateResult,
  Provider,
  RosettaMessage,
  RosettaTool,
  RosettaToolCallRequest,
  StreamChunk,
  TokenUsage
} from '../../types'
import { MappingError, ProviderAPIError, ToolArgumentValidationError } from '../../errors'
import { mapBaseParams, mapBaseToolChoice } from './common.utils'
import { getGpt5Support } from './gpt5.support'
import { isGPT5Model, wrapOpenAIError } from './openai.common'
import { resolveThinkingRequest } from './thinking-request'

/**
 * OpenAI Responses API path for the standard chat surface (generate/stream).
 *
 * gpt-5.4+ models reject function tools combined with reasoning on /v1/chat/completions
 * (gpt-5.6 rejects tools unless reasoning_effort is explicitly 'none'), and Chat Completions
 * never discloses reasoning content. The Responses API supports both: tools work at any
 * reasoning effort, and `reasoning: {summary}` streams disclosed reasoning summaries.
 *
 * This module maps the provider-neutral GenerateParams surface onto /v1/responses and maps
 * responses/streams back into the canonical GenerateResult/StreamChunk contracts, so callers
 * see no difference from the Chat Completions path apart from working reasoning.
 *
 * Requests are stateless (`store: false`) with `include: ['reasoning.encrypted_content']`.
 * Reasoning items that precede tool calls are carried opaquely on the first tool call's
 * `providerMetadata.openaiResponses` and replayed ahead of the corresponding function_call
 * items on the next request (same pattern as Google's thoughtSignature). Replay is
 * best-effort: the API accepts histories without reasoning items (verified live on
 * gpt-5.4 / gpt-5.6-terra / gpt-5.6-sol), so dropped metadata degrades continuity, not
 * correctness.
 */

interface OpenAIResponsesToolCallMetadata {
  /** Reasoning items (with encrypted content) that preceded this tool call's function_call items. */
  reasoningItems?: unknown[]
}

/** Decides whether an OpenAI request should use the Responses API instead of Chat Completions. */
export function shouldUseOpenAIResponsesApi(params: GenerateParams): boolean {
  if (params.providerOptions?.openaiPreferChatCompletions === true) return false
  return isGPT5Model(params.model ?? '')
}

function contentToText(content: RosettaMessage['content']): string {
  if (content === null) return ''
  if (typeof content === 'string') return content
  return content
    .map(part => (part.type === 'text' ? part.text : ''))
    .filter(text => text.length > 0)
    .join('\n')
}

function extractReasoningItemsFromToolCalls(toolCalls: RosettaToolCallRequest[]): unknown[] {
  const items: unknown[] = []
  for (const toolCall of toolCalls) {
    const metadata = toolCall.providerMetadata?.openaiResponses as OpenAIResponsesToolCallMetadata | undefined
    if (metadata?.reasoningItems && Array.isArray(metadata.reasoningItems)) {
      items.push(...metadata.reasoningItems)
    }
  }
  return items
}

function mapMessagesToResponsesInput(messages: RosettaMessage[]): { instructions?: string; input: ResponseInput } {
  const systemTexts: string[] = []
  const input: ResponseInput = []

  for (const msg of messages) {
    switch (msg.role) {
      case 'system': {
        // Match the Chat Completions mapping: reject null/empty system content instead of
        // silently sending an unintended empty prompt.
        if (msg.content === null) {
          throw new MappingError(`Role 'system' requires non-null content.`, Provider.OpenAI)
        }
        const text = contentToText(msg.content)
        if (text.length === 0) {
          throw new MappingError(`Role 'system' requires non-empty string content.`, Provider.OpenAI)
        }
        systemTexts.push(text)
        break
      }
      case 'user': {
        // Match the Chat Completions mapping: null user content is an upstream bug, not a
        // blank message. Empty strings remain allowed, as on the chat path.
        if (msg.content === null) {
          throw new MappingError(`Role 'user' requires non-null content.`, Provider.OpenAI)
        }
        if (typeof msg.content === 'string') {
          input.push({ role: 'user', content: msg.content })
        } else {
          input.push({
            role: 'user',
            content: msg.content.map(part => {
              if (part.type === 'text') return { type: 'input_text' as const, text: part.text }
              return {
                type: 'input_image' as const,
                image_url: `data:${part.image.mimeType};base64,${part.image.base64Data}`,
                detail: 'auto' as const
              }
            })
          })
        }
        break
      }
      case 'assistant': {
        const text = contentToText(msg.content)
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          // Observed output ordering on /v1/responses is [reasoning, message?, function_call...];
          // replay mirrors it so encrypted reasoning stays adjacent to the calls it produced.
          for (const reasoningItem of extractReasoningItemsFromToolCalls(msg.toolCalls)) {
            input.push(reasoningItem as ResponseInputItem)
          }
          if (text.length > 0) input.push({ role: 'assistant', content: text })
          for (const toolCall of msg.toolCalls) {
            input.push({
              type: 'function_call',
              call_id: toolCall.id,
              name: toolCall.function.name,
              arguments: toolCall.function.arguments
            })
          }
        } else if (text.length > 0) {
          input.push({ role: 'assistant', content: text })
        }
        break
      }
      case 'tool': {
        if (!msg.toolCallId) {
          throw new MappingError(`Tool message missing toolCallId for Responses input.`, Provider.OpenAI)
        }
        input.push({ type: 'function_call_output', call_id: msg.toolCallId, output: contentToText(msg.content) })
        break
      }
      default: {
        const _e: never = msg.role
        throw new MappingError(`Unhandled role type during Responses input construction: ${_e}`, Provider.OpenAI)
      }
    }
  }

  return { instructions: systemTexts.length > 0 ? systemTexts.join('\n\n') : undefined, input }
}

function mapToolsToResponsesTools(tools: GenerateParams['tools']): OpenAIResponsesTool[] | undefined {
  if (!tools || tools.length === 0) return undefined
  return tools.map(tool => {
    if (tool.type !== 'function') {
      throw new MappingError(`Unsupported tool type for OpenAI Responses: ${tool.type}`, Provider.OpenAI)
    }
    return {
      type: 'function' as const,
      name: tool.function.name,
      description: tool.function.description ?? null,
      parameters: (tool.function.parameters as Record<string, unknown>) ?? null,
      // The Responses API defaults strict to true, which rejects schemas that are not
      // strict-mode clean (e.g. missing additionalProperties: false). Chat Completions
      // defaulted to non-strict, so keep that behavior for parity with existing tools.
      strict: false
    }
  })
}

function mapToolChoiceToResponses(
  toolChoice: GenerateParams['toolChoice']
): ToolChoiceOptions | ToolChoiceFunction | undefined {
  const base = mapBaseToolChoice(toolChoice)
  if (!base) return undefined
  if (base === 'auto' || base === 'none' || base === 'required') return base
  if (typeof base === 'object' && base.type === 'function') {
    return { type: 'function', name: base.function.name }
  }
  return undefined
}

/**
 * The Responses API rejects `max_output_tokens` below 16 with a 400
 * (`integer_below_min_value`); Chat Completions accepts caps down to 1, so neutral
 * `maxTokens` values below the floor are clamped up to keep tiny-budget requests
 * (validation pings, cheap probes) working across the dialect switch.
 */
const RESPONSES_MIN_OUTPUT_TOKENS = 16

/**
 * Resolves the effective `reasoning` request parameter.
 *
 * A neutral thinking request needs reasoning to actually run (effort above 'none') and asks
 * for disclosed summaries. Without thinking, an explicit reasoningEffort passes through and
 * the model default applies otherwise.
 */
function resolveResponsesReasoning(
  model: string,
  requestedEffort: GenerateParams['reasoningEffort'],
  thinkingRequested: boolean
): Reasoning | undefined {
  const support = getGpt5Support(model)

  let effort = requestedEffort
  if (support && effort !== undefined && support.allowedReasoningEfforts.length > 0) {
    if (!support.allowedReasoningEfforts.includes(effort)) {
      effort = support.defaultReasoningEffort
    }
  }
  if (support?.fixedReasoningEffort) {
    effort = support.fixedReasoningEffort
  }

  if (thinkingRequested && (effort === undefined || effort === 'none')) {
    const supportedDefault =
      support && support.defaultReasoningEffort && support.defaultReasoningEffort !== 'none'
        ? support.defaultReasoningEffort
        : undefined
    effort = supportedDefault ?? 'medium'
  }

  if (effort === undefined && !thinkingRequested) return undefined

  return {
    ...(effort !== undefined ? { effort } : {}),
    ...(thinkingRequested ? { summary: 'auto' as const } : {})
  }
}

export function mapToOpenAIResponsesChatParams(params: GenerateParams): ResponseCreateParamsNonStreaming {
  const model = params.model!
  const baseMappedParams = mapBaseParams(params)
  // Native key for this dialect is the Responses `reasoning` object; Chat Completions'
  // reasoning_effort is expressed through the first-class reasoningEffort param instead.
  const { thinkingRequested, extraParams: sanitizedExtraParams } = resolveThinkingRequest(params, ['reasoning'])

  const { instructions, input } = mapMessagesToResponsesInput(params.messages)
  const tools = mapToolsToResponsesTools(params.tools)
  const toolChoice = mapToolChoiceToResponses(params.toolChoice)
  const reasoning = resolveResponsesReasoning(model, baseMappedParams.reasoningEffort, thinkingRequested)
  const support = getGpt5Support(model)

  let textConfig: ResponseCreateParamsNonStreaming['text']
  if (params.responseFormat?.type === 'json_schema') {
    textConfig = {
      format: {
        type: 'json_schema',
        name: params.responseFormat.json_schema.name ?? 'response',
        strict: params.responseFormat.json_schema.strict ?? true,
        schema: params.responseFormat.json_schema.schema as Record<string, unknown>
      }
    }
  } else if (params.responseFormat?.type === 'json_object') {
    textConfig = { format: { type: 'json_object' } }
  }
  if (
    support?.supportsVerbosity &&
    baseMappedParams.verbosity !== undefined &&
    ['low', 'medium', 'high'].includes(baseMappedParams.verbosity)
  ) {
    textConfig = { ...(textConfig ?? {}), verbosity: baseMappedParams.verbosity }
  }

  const allowsSampling =
    support?.supportsSampling === 'always' ||
    (support?.supportsSampling === 'only_with_reasoning_none' && reasoning?.effort === 'none')

  const payload: ResponseCreateParamsNonStreaming = {
    ...((sanitizedExtraParams ?? {}) as Partial<ResponseCreateParamsNonStreaming>),
    model,
    input,
    ...(instructions ? { instructions } : {}),
    ...(tools ? { tools } : {}),
    ...(toolChoice ? { tool_choice: toolChoice } : {}),
    ...(reasoning ? { reasoning } : {}),
    ...(textConfig ? { text: textConfig } : {}),
    ...(baseMappedParams.maxTokens !== undefined
      ? { max_output_tokens: Math.max(RESPONSES_MIN_OUTPUT_TOKENS, baseMappedParams.maxTokens) }
      : {}),
    ...(allowsSampling && baseMappedParams.temperature !== undefined
      ? { temperature: baseMappedParams.temperature }
      : {}),
    ...(allowsSampling && baseMappedParams.topP !== undefined ? { top_p: baseMappedParams.topP } : {}),
    // Stateless operation with opaque reasoning replay, so no provider-side conversation
    // storage is required and ZDR organizations are supported.
    store: false,
    include: ['reasoning.encrypted_content']
  }

  return payload
}

function summaryText(item: ResponseReasoningItem): string {
  if (!Array.isArray(item.summary)) return ''
  return item.summary
    .map(part => part.text ?? '')
    .filter(text => text.length > 0)
    .join('\n\n')
}

function validateToolCall(
  toolCall: { id: string; name: string; arguments: string },
  originalTools: RosettaTool<any>[] | undefined,
  finishContext: string
): void {
  const toolDefinition = originalTools?.find(t => t.function.name === toolCall.name)
  if (!toolDefinition) {
    console.warn(`Skipping validation for unknown Responses tool '${toolCall.name}'.`)
    return
  }
  let parsedArgs: unknown
  try {
    parsedArgs = JSON.parse(toolCall.arguments || '{}')
  } catch (parseError) {
    throw new MappingError(
      `Failed to parse arguments for tool '${toolCall.name}' (ID: ${toolCall.id}).`,
      Provider.OpenAI,
      finishContext,
      parseError
    )
  }
  const validationResult = toolDefinition.function.zodSchema.safeParse(parsedArgs)
  if (!validationResult.success) {
    throw new ToolArgumentValidationError(
      `Arguments failed validation for tool '${toolCall.name}'.`,
      validationResult.error.issues,
      toolCall.name,
      toolCall.id
    )
  }
}

function attachReasoningMetadata(toolCalls: RosettaToolCallRequest[], reasoningItems: ResponseReasoningItem[]): void {
  if (reasoningItems.length === 0) return
  const first = toolCalls[0]
  if (!first) return
  first.providerMetadata = {
    ...(first.providerMetadata ?? {}),
    openaiResponses: { reasoningItems } satisfies OpenAIResponsesToolCallMetadata
  }
}

function mapFinishReason(response: Response, hasToolCalls: boolean, hasRefusal: boolean): string {
  if (hasToolCalls) return 'tool_calls'
  if (hasRefusal) return 'content_filter'
  if (response.status === 'incomplete') {
    return response.incomplete_details?.reason === 'max_output_tokens' ? 'length' : 'stop'
  }
  return 'stop'
}

function mapResponsesUsage(usage: Response['usage']): TokenUsage | undefined {
  if (!usage) return undefined
  return {
    promptTokens: usage.input_tokens,
    completionTokens: usage.output_tokens,
    totalTokens: usage.total_tokens
  }
}

export function mapOpenAIResponsesChatResponse(
  response: Response,
  modelUsed: string,
  originalTools: RosettaTool<any>[] | undefined
): GenerateResult {
  const outputItems: ResponseOutputItem[] = response.output ?? []
  const reasoningItems = outputItems.filter((item): item is ResponseReasoningItem => item.type === 'reasoning')

  const textParts: string[] = []
  let refusalText: string | null = null
  for (const item of outputItems) {
    if (item.type !== 'message') continue
    for (const part of item.content) {
      if (part.type === 'output_text') textParts.push((part as ResponseOutputText).text)
      else if (part.type === 'refusal') refusalText = part.refusal
    }
  }

  const toolCalls: RosettaToolCallRequest[] = []
  for (const item of outputItems) {
    if (item.type !== 'function_call') continue
    const fnCall = item as ResponseFunctionToolCall
    validateToolCall(
      { id: fnCall.call_id, name: fnCall.name, arguments: fnCall.arguments },
      originalTools,
      'mapOpenAIResponsesChatResponse validation'
    )
    toolCalls.push({
      id: fnCall.call_id,
      type: 'function',
      function: { name: fnCall.name, arguments: fnCall.arguments }
    })
  }
  attachReasoningMetadata(toolCalls, reasoningItems)

  const content = textParts.length > 0 ? textParts.join('') : refusalText
  let parsedJson: Record<string, unknown> | Array<unknown> | null = null
  if (content && toolCalls.length === 0) {
    const trimmed = content.trim()
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        parsedJson = JSON.parse(trimmed)
      } catch {
        // Non-JSON content despite a JSON-looking prefix — leave parsedContent null.
      }
    }
  }

  const thinkingSteps = reasoningItems.map(summaryText).filter(text => text.length > 0).join('\n\n') || null

  return {
    content,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    finishReason: mapFinishReason(response, toolCalls.length > 0, refusalText !== null),
    usage: mapResponsesUsage(response.usage),
    citations: undefined,
    parsedContent: parsedJson,
    thinkingSteps,
    model: response.model ?? modelUsed,
    rawResponse: response
  }
}

type StreamingToolCallState = {
  itemId: string
  toolIndex: number
  callId: string
  name: string
  argumentsAccumulator: string
  done: boolean
}

export async function* mapOpenAIResponsesChatStream(
  stream: AsyncIterable<ResponseStreamEvent>,
  modelId: string,
  originalTools: RosettaTool<any>[] | undefined
): AsyncIterable<StreamChunk> {
  let accumulatedContent = ''
  let accumulatedThinking = ''
  let isJsonMode = false
  let messageStartYielded = false
  let refusalText: string | null = null
  let finalUsage: TokenUsage | undefined
  let responseSnapshot: Response | undefined
  const toolCallsByItemId = new Map<string, StreamingToolCallState>()
  const reasoningItems: ResponseReasoningItem[] = []

  const aggregatedResult: GenerateResult = {
    content: '',
    toolCalls: [],
    finishReason: null,
    usage: undefined,
    model: modelId,
    thinkingSteps: null,
    citations: undefined,
    parsedContent: null,
    rawResponse: undefined
  }

  try {
    for await (const event of stream) {
      switch (event.type) {
        case 'response.created': {
          responseSnapshot = event.response
          if (event.response.model) aggregatedResult.model = event.response.model
          if (!messageStartYielded) {
            yield { type: 'message_start', data: { provider: Provider.OpenAI, model: aggregatedResult.model } }
            messageStartYielded = true
          }
          break
        }
        case 'response.reasoning_summary_part.added': {
          // Each summary part is a distinct disclosed thought cycle.
          yield { type: 'thinking_start' }
          break
        }
        case 'response.reasoning_summary_text.delta': {
          if (event.delta.length === 0) break
          accumulatedThinking += event.delta
          aggregatedResult.thinkingSteps = accumulatedThinking
          yield { type: 'thinking_delta', data: { delta: event.delta } }
          break
        }
        case 'response.reasoning_summary_part.done': {
          yield { type: 'thinking_stop' }
          break
        }
        case 'response.output_text.delta': {
          const delta = event.delta
          if (delta.length === 0) break
          accumulatedContent += delta
          aggregatedResult.content = accumulatedContent
          if (!isJsonMode && accumulatedContent.trim().match(/^[{[]/)) isJsonMode = true
          if (isJsonMode) {
            let partialParsed: unknown
            try {
              partialParsed = JSON.parse(accumulatedContent)
            } catch {}
            yield { type: 'json_delta', data: { delta, parsed: partialParsed, snapshot: accumulatedContent } }
          } else {
            yield { type: 'content_delta', data: { delta } }
          }
          break
        }
        case 'response.refusal.done': {
          refusalText = event.refusal
          break
        }
        case 'response.output_item.added': {
          if (event.item.type === 'function_call') {
            const fnCall = event.item as ResponseFunctionToolCall
            const toolIndex = toolCallsByItemId.size
            toolCallsByItemId.set(fnCall.id ?? fnCall.call_id, {
              itemId: fnCall.id ?? fnCall.call_id,
              toolIndex,
              callId: fnCall.call_id,
              name: fnCall.name,
              argumentsAccumulator: fnCall.arguments ?? '',
              done: false
            })
            yield {
              type: 'tool_call_start',
              data: {
                index: toolIndex,
                toolCall: { id: fnCall.call_id, type: 'function', function: { name: fnCall.name } }
              }
            }
          }
          break
        }
        case 'response.function_call_arguments.delta': {
          const state = toolCallsByItemId.get(event.item_id)
          if (!state || event.delta.length === 0) break
          state.argumentsAccumulator += event.delta
          yield {
            type: 'tool_call_delta',
            data: { index: state.toolIndex, id: state.callId, functionArgumentChunk: event.delta }
          }
          break
        }
        case 'response.function_call_arguments.done': {
          const state = toolCallsByItemId.get(event.item_id)
          if (state) state.argumentsAccumulator = event.arguments
          break
        }
        case 'response.output_item.done': {
          if (event.item.type === 'reasoning') {
            reasoningItems.push(event.item as ResponseReasoningItem)
          } else if (event.item.type === 'function_call') {
            const fnCall = event.item as ResponseFunctionToolCall
            const state = toolCallsByItemId.get(fnCall.id ?? fnCall.call_id)
            if (state && !state.done) {
              state.argumentsAccumulator = fnCall.arguments ?? state.argumentsAccumulator
              validateToolCall(
                { id: state.callId, name: state.name, arguments: state.argumentsAccumulator },
                originalTools,
                'mapOpenAIResponsesChatStream validation'
              )
              state.done = true
              yield { type: 'tool_call_done', data: { index: state.toolIndex, id: state.callId } }
              aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
              aggregatedResult.toolCalls.push({
                id: state.callId,
                type: 'function',
                function: { name: state.name, arguments: state.argumentsAccumulator }
              })
            }
          }
          break
        }
        case 'response.completed':
        case 'response.incomplete': {
          responseSnapshot = event.response
          finalUsage = mapResponsesUsage(event.response.usage)
          aggregatedResult.usage = finalUsage
          break
        }
        case 'response.failed': {
          const providerMessage = event.response.error?.message ?? 'OpenAI Responses stream reported failure.'
          throw new ProviderAPIError(providerMessage, Provider.OpenAI, undefined, event.response.error?.code)
        }
        case 'error': {
          throw new ProviderAPIError(event.message ?? 'OpenAI Responses stream error.', Provider.OpenAI, undefined, event.code)
        }
        default:
          break
      }
    }

    if (isJsonMode) {
      let finalParsedJson: Record<string, unknown> | Array<unknown> | null = null
      try {
        finalParsedJson = JSON.parse(accumulatedContent)
      } catch {}
      yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: accumulatedContent } }
      aggregatedResult.parsedContent = finalParsedJson
    }

    const hasToolCalls = (aggregatedResult.toolCalls?.length ?? 0) > 0
    if (hasToolCalls) {
      attachReasoningMetadata(aggregatedResult.toolCalls!, reasoningItems)
    }
    if (aggregatedResult.content === '' && refusalText) aggregatedResult.content = refusalText
    if (!isJsonMode && aggregatedResult.content === '') aggregatedResult.content = null
    if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
    if (accumulatedThinking.length === 0) aggregatedResult.thinkingSteps = null

    const finishReason = responseSnapshot
      ? mapFinishReason(responseSnapshot, hasToolCalls, refusalText !== null)
      : hasToolCalls
        ? 'tool_calls'
        : 'stop'
    aggregatedResult.finishReason = finishReason

    yield { type: 'message_stop', data: { finishReason } }
    if (finalUsage) {
      yield { type: 'final_usage', data: { usage: finalUsage } }
    }
    yield { type: 'final_result', data: { result: aggregatedResult } }
  } catch (error) {
    const mappedError = wrapOpenAIError(error, Provider.OpenAI)
    yield { type: 'error', data: { error: mappedError } }
  }
}

/** Executes a non-streaming Responses API generation for the standard chat surface. */
export async function generateViaOpenAIResponses(
  client: OpenAI,
  params: GenerateParams,
  originalTools: RosettaTool<any>[] | undefined
): Promise<GenerateResult> {
  const mappedParams = mapToOpenAIResponsesChatParams(params)
  const response = await client.responses.create(mappedParams)
  return mapOpenAIResponsesChatResponse(response, params.model!, originalTools)
}
