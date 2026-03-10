import Anthropic, { APIError } from '@anthropic-ai/sdk'
import {
  RawMessageStreamEvent,
  MessageParam as AnthropicMessageParam,
  ToolUnion as AnthropicToolParam,
  ThinkingConfigParam as AnthropicThinkingConfig,
  ContentBlockParam as AnthropicContentBlockParam,
  TextBlockParam as AnthropicTextBlockParam,
  ImageBlockParam as AnthropicImageBlockParam,
  RawContentBlockStopEvent,
  ToolUseBlockParam, // INPUT type for tool use
  ToolResultBlockParam, // INPUT type for tool result
  CodeExecutionToolResultBlock
} from '@anthropic-ai/sdk/resources/messages'
import { Tool as AnthropicToolType } from '@anthropic-ai/sdk/resources'
import {
  Message as AnthropicMessage,
  ContentBlock as AnthropicResponseContentBlock
} from '@anthropic-ai/sdk/resources/messages'
import { JSONSchema7 } from 'json-schema' // Import JSONSchema7

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult,
  RosettaTool // Import RosettaTool
} from '../../types'
import {
  MappingError,
  ProviderAPIError,
  RosettaAIError,
  UnsupportedFeatureError,
  InvalidToolDefinitionError, // Import new errors
  ToolArgumentValidationError
} from '../../errors'
import { safeGet } from '../utils'
import { IProviderMapper } from './base.mapper'
import { mapTokenUsage, mapBaseParams, mapBaseToolChoice } from './common.utils'

// Type alias for the stream type from Anthropic SDK
type AnthropicMessageStream = AsyncIterable<RawMessageStreamEvent>

type RosettaToolCaller = NonNullable<RosettaToolCallRequest['caller']>

type ServerToolUseAccumulator = {
  id: string
  jsonAccumulator: string
  lastCode: string
  inputSnapshot: unknown
}

type ToolUseAccumulator = {
  id: string
  name: string
  jsonAccumulator: string
  inputSnapshot: unknown
  index: number
  caller?: RosettaToolCaller
}

export class AnthropicMapper implements IProviderMapper {
  readonly provider = Provider.Anthropic

  private extractToolCaller(caller?: {
    type: RosettaToolCaller['type']
    tool_id?: string
  }): RosettaToolCaller | undefined {
    if (!caller) {
      return undefined
    }

    return {
      type: caller.type,
      ...(('tool_id' in caller && typeof caller.tool_id === 'string') ? { toolId: caller.tool_id } : {})
    }
  }

  private mapToolCallerToAnthropicParam(caller: RosettaToolCaller): ToolUseBlockParam['caller'] {
    if (caller.type === 'direct') {
      return { type: 'direct' }
    }
    if (caller.type === 'code_execution_20250825') {
      return { type: 'code_execution_20250825', tool_id: caller.toolId ?? '' }
    }
    return { type: 'code_execution_20260120', tool_id: caller.toolId ?? '' }
  }

  private extractCodeExecutionCode(input: unknown): string {
    if (typeof input !== 'object' || input === null || !('code' in input)) {
      return ''
    }

    const candidate = (input as { code?: unknown }).code
    return typeof candidate === 'string' ? candidate : ''
  }

  private tryExtractCodeFromAccumulatedJSON(accumulator: string): string | null {
    try {
      const parsed = JSON.parse(accumulator) as unknown
      return this.extractCodeExecutionCode(parsed)
    } catch {
      return null
    }
  }

  private tryPartialParseJSON(accumulator: string): unknown | undefined {
    try {
      return JSON.parse(accumulator) as unknown
    } catch {
      return undefined
    }
  }

  private resolveStreamedToolInput(toolData: ToolUseAccumulator): unknown {
    if (toolData.jsonAccumulator !== '') {
      try {
        return JSON.parse(toolData.jsonAccumulator) as unknown
      } catch (parseError) {
        throw new MappingError(
          `Failed to parse arguments for tool '${toolData.name}' (ID: ${toolData.id})`,
          this.provider,
          'mapProviderStream validation',
          parseError
        )
      }
    }

    return toolData.inputSnapshot ?? {}
  }

  private resolveAccumulatedJSONInput(
    accumulator: { id: string; jsonAccumulator: string; inputSnapshot: unknown },
    contextLabel: string
  ): unknown {
    if (accumulator.jsonAccumulator !== '') {
      try {
        return JSON.parse(accumulator.jsonAccumulator) as unknown
      } catch (parseError) {
        throw new MappingError(
          `Failed to parse streamed ${contextLabel} input (ID: ${accumulator.id})`,
          this.provider,
          'mapProviderStream validation',
          parseError
        )
      }
    }

    return accumulator.inputSnapshot ?? {}
  }

  private cloneRawContentBlock<T>(block: T): T {
    return JSON.parse(JSON.stringify(block)) as T
  }

  private mapCodeExecutionResultChunk(block: CodeExecutionToolResultBlock): StreamChunk {
    const content = block.content

    if (content.type === 'code_execution_result') {
      return {
        type: 'code_execution_result',
        data: {
          toolUseId: block.tool_use_id,
          stdout: content.stdout,
          stderr: content.stderr,
          returnCode: content.return_code,
          contentFileIds: content.content.map(output => output.file_id)
        }
      }
    }

    if (content.type === 'encrypted_code_execution_result') {
      return {
        type: 'code_execution_result',
        data: {
          toolUseId: block.tool_use_id,
          stdout: '',
          stderr: content.stderr,
          returnCode: content.return_code,
          encryptedStdout: content.encrypted_stdout
        }
      }
    }

    return {
      type: 'code_execution_result',
      data: {
        toolUseId: block.tool_use_id,
        stdout: '',
        stderr: '',
        returnCode: 1,
        errorCode: content.error_code
      }
    }
  }

  // --- Parameter Mapping ---

  private mapRoleToAnthropic(role: RosettaMessage['role']): AnthropicMessageParam['role'] {
    switch (role) {
      case 'user':
        return 'user'
      case 'assistant':
        return 'assistant'
      // System and Tool roles are handled structurally, not directly mapped here.
      case 'system':
      case 'tool':
        throw new MappingError(
          `Role '${role}' should be handled structurally for Anthropic.`,
          this.provider,
          'mapRoleToAnthropic'
        )
      default:
        // Ensure exhaustive check works with `never`
        const _exhaustiveCheck: never = role
        throw new MappingError(`Unsupported role: ${_exhaustiveCheck}`, this.provider)
    }
  }

  private mapContentToAnthropic(content: RosettaMessage['content']): string | Array<AnthropicContentBlockParam> {
    if (content === null) {
      // Anthropic requires content for user/assistant messages unless it's purely tool calls/results.
      // Returning empty string might be problematic depending on context. Handled in mapToProviderParams.
      console.warn('Mapping null content to empty string for Anthropic input.')
      return ''
    }
    if (typeof content === 'string') {
      return content
    }
    // Handle empty array case - return empty string as Anthropic content cannot be empty array
    if (Array.isArray(content) && content.length === 0) {
      console.warn('Mapping empty content array to empty string for Anthropic input.')
      return ''
    }

    const parts: AnthropicContentBlockParam[] = content.map(part => {
      if (part.type === 'text') {
        const textParam: AnthropicTextBlockParam = { type: 'text', text: part.text }
        return textParam
      } else if (part.type === 'image') {
        const imageParam: AnthropicImageBlockParam = {
          type: 'image',
          source: { type: 'base64', media_type: part.image.mimeType, data: part.image.base64Data }
        }
        return imageParam
      } else {
        // Ensure exhaustive check works with `never`
        const _exhaustiveCheck: never = part
        throw new MappingError(`Unsupported content part type: ${(_exhaustiveCheck as any).type}`, this.provider)
      }
    })
    return parts
  }

  mapToProviderParams(
    params: GenerateParams
  ): Anthropic.Messages.MessageCreateParamsNonStreaming | Anthropic.Messages.MessageCreateParamsStreaming {
    let systemPrompt: string | undefined = undefined
    const messages: AnthropicMessageParam[] = []

    for (const msg of params.messages) {
      if (msg.role === 'system') {
        if (systemPrompt) throw new MappingError('Multiple system messages not supported by Anthropic.', this.provider)
        if (typeof msg.content !== 'string')
          throw new MappingError('Anthropic system prompt must be string.', this.provider)
        systemPrompt = msg.content
        continue
      }
      if (msg.role === 'tool') {
        if (!msg.toolCallId || typeof msg.content !== 'string') {
          throw new MappingError(
            'Invalid tool result message format for Anthropic. Requires toolCallId and string content.',
            this.provider
          )
        }
        // Map RosettaToolResult (role='tool') to Anthropic's user message with tool_result block
        const toolResultBlock: ToolResultBlockParam = {
          type: 'tool_result',
          tool_use_id: msg.toolCallId,
          content: msg.content, // Assuming content is already stringified JSON or simple string
          is_error: msg.isError // Pass the error flag if present
        }
        messages.push({
          role: 'user',
          content: [toolResultBlock]
        })
      } else if (msg.role === 'assistant' && Array.isArray(msg.rawContentBlocks) && msg.rawContentBlocks.length > 0) {
        messages.push({
          role: 'assistant',
          content: msg.rawContentBlocks as AnthropicContentBlockParam[]
        })
      } else if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
        const assistantContent = this.mapContentToAnthropic(msg.content)
        const contentBlocks: AnthropicContentBlockParam[] = []

        // Add text block if assistantContent is a non-empty string or a non-empty array containing text
        if (typeof assistantContent === 'string' && assistantContent.length > 0) {
          contentBlocks.push({ type: 'text', text: assistantContent })
        } else if (Array.isArray(assistantContent)) {
          assistantContent.forEach(block => {
            if (block.type === 'text' || block.type === 'image') {
              // @ts-ignore - Assuming block.source.data exists for image block if type is image
              if (block.type === 'text' || (block.type === 'image' && block.source.data)) {
                contentBlocks.push(block)
              }
            } else {
              console.warn(
                `Ignoring unexpected content block type '${block.type}' in assistant message with tool calls.`
              )
            }
          })
        }

        // Add tool_use blocks
        msg.toolCalls.forEach(toolCall => {
          try {
            const toolUseBlock: ToolUseBlockParam = {
              type: 'tool_use',
              id: toolCall.id,
              name: toolCall.function.name,
              input: JSON.parse(toolCall.function.arguments || '{}'), // Parse arguments string
              ...(toolCall.caller
                ? {
                    caller: this.mapToolCallerToAnthropicParam(toolCall.caller)
                  }
                : {})
            }
            contentBlocks.push(toolUseBlock)
          } catch (e) {
            throw new MappingError(
              `Failed to parse arguments for tool_use block ${toolCall.id}`,
              this.provider,
              'mapToProviderParams toolCall mapping',
              e
            )
          }
        })

        if (contentBlocks.length === 0) {
          // This case should only happen if toolCalls were present but msg.content was null/empty
          // and resulted in an empty contentBlocks array. Anthropic requires at least one block.
          if (msg.toolCalls.length === 0) {
            // This sub-case should theoretically not be reachable if msg.toolCalls.length > 0 check passed
            throw new MappingError(
              'Assistant message resulted in empty content blocks without tool calls.',
              this.provider
            )
          }
          // If only tool_use blocks exist, that's valid.
        }

        messages.push({ role: 'assistant', content: contentBlocks })
      } else {
        // Handle regular user/assistant messages
        const mappedContent = this.mapContentToAnthropic(msg.content)
        // Anthropic requires non-empty content for user/assistant roles unless it's purely tool calls/results
        if (
          (msg.role === 'user' || msg.role === 'assistant') &&
          ((typeof mappedContent === 'string' && mappedContent === '') ||
            (Array.isArray(mappedContent) && mappedContent.length === 0))
        ) {
          // Allow empty assistant message only if it contains tool calls (handled above)
          if (msg.role !== 'assistant' || !msg.toolCalls || msg.toolCalls.length === 0) {
            throw new MappingError(
              `Role '${msg.role}' requires non-empty content for Anthropic. Received empty content after mapping.`,
              this.provider
            )
          }
        }
        messages.push({
          role: this.mapRoleToAnthropic(msg.role as 'user' | 'assistant'),
          content: mappedContent
        })
      }
    }

    if (messages.length === 0 && !systemPrompt) {
      // Anthropic requires at least one message.
      throw new MappingError('No user or assistant messages provided for Anthropic.', this.provider)
    }

    // Map RosettaTool definitions to AnthropicToolParam
    const isProgrammaticToolCalling = params.programmaticToolCalling === true
    if (isProgrammaticToolCalling && params.extraParams?.disable_parallel_tool_use === true) {
      throw new UnsupportedFeatureError(this.provider, 'disable_parallel_tool_use with programmaticToolCalling')
    }

    const tools: AnthropicToolParam[] | undefined =
      params.tools && params.tools.length > 0
        ? [
            ...(isProgrammaticToolCalling
              ? ([{ type: 'code_execution_20260120', name: 'code_execution' }] as const)
              : []),
            ...params.tools.map(tool => {
              if (tool.type !== 'function') {
                throw new InvalidToolDefinitionError(`Unsupported tool type: ${tool.type}`, tool.function.name)
              }
              const inputSchemaSource = tool.function.parameters as JSONSchema7

              // Validate that the schema type is 'object' as required by Anthropic
              if (inputSchemaSource.type !== 'object') {
                throw new InvalidToolDefinitionError(
                  `Invalid parameters schema for tool '${tool.function.name}'. Anthropic requires the top-level 'type' property to be exactly 'object'. Received: type='${inputSchemaSource.type}'`,
                  tool.function.name
                )
              }

              // Ensure zodSchema exists
              if (!tool.function.zodSchema) {
                throw new InvalidToolDefinitionError(`Missing zodSchema for validation.`, tool.function.name)
              }

              // Cast is now safe because we've checked inputSchemaSource.type === 'object'
              const inputSchema: AnthropicToolType.InputSchema = inputSchemaSource as AnthropicToolType.InputSchema

              return {
                name: tool.function.name,
                description: tool.function.description,
                input_schema: inputSchema,
                ...(isProgrammaticToolCalling
                  ? {
                      allowed_callers: tool.allowedCallers ?? ['code_execution_20260120']
                    }
                  : {})
              }
            })
          ]
        : undefined

    const baseToolChoice = mapBaseToolChoice(params.toolChoice)
    let anthropicToolChoice: Anthropic.Messages.ToolChoice | undefined = undefined
    if (baseToolChoice) {
      if (baseToolChoice === 'auto' || baseToolChoice === 'none') {
        anthropicToolChoice = { type: baseToolChoice }
      } else if (baseToolChoice === 'required') {
        anthropicToolChoice = { type: 'any' }
      } else if (typeof baseToolChoice === 'object' && baseToolChoice.type === 'function') {
        if (isProgrammaticToolCalling) {
          throw new UnsupportedFeatureError(this.provider, 'forced toolChoice with programmaticToolCalling')
        }
        anthropicToolChoice = { type: 'tool', name: baseToolChoice.function.name }
      } else {
        console.warn(`Unhandled baseToolChoice format: ${JSON.stringify(baseToolChoice)}`)
      }
    }

    let thinkingParam: AnthropicThinkingConfig | undefined = undefined
    if (params.thinking) {
      thinkingParam = { type: 'enabled', budget_tokens: 1024 }
    }

    let systemParam: string | AnthropicTextBlockParam[] | undefined
    if (typeof systemPrompt === 'string') {
      systemParam = systemPrompt
    } else {
      systemParam = undefined
    }

    if (params.responseFormat?.type === 'json_object') {
      throw new UnsupportedFeatureError(this.provider, 'responseFormat: json_object (use json_schema)')
    }

    const output_config: Anthropic.Messages.OutputConfig | undefined =
      params.responseFormat?.type === 'json_schema'
        ? {
            format: {
              type: 'json_schema' as const,
              schema: params.responseFormat.json_schema.schema
            }
          }
        : undefined

    const baseMappedParams = mapBaseParams(params)

    const basePayload = {
      ...(params.extraParams ?? {}),
      model: params.model!.replace(':1m', ''),
      messages: messages,
      system: systemParam,
      max_tokens: baseMappedParams.maxTokens ?? 4096,
      temperature: baseMappedParams.temperature,
      top_p: baseMappedParams.topP,
      stop_sequences: baseMappedParams.stopSequences,
      tools: tools,
      tool_choice: anthropicToolChoice,
      ...(thinkingParam && { thinking: thinkingParam }),
      ...(output_config && { output_config }),
      ...(params.container ? { container: params.container } : {})
    }

    if (params.stream) {
      const streamPayload: Anthropic.Messages.MessageCreateParamsStreaming = { ...basePayload, stream: true }
      return streamPayload
    } else {
      const nonStreamPayload: Anthropic.Messages.MessageCreateParamsNonStreaming = basePayload
      return nonStreamPayload
    }
  }

  // --- Result Mapping ---

  private mapAndValidateToolCallsFromAnthropic(
    contentBlocks: AnthropicResponseContentBlock[] | undefined,
    originalTools?: RosettaTool<any>[]
  ): RosettaToolCallRequest[] | undefined {
    if (!Array.isArray(contentBlocks)) return undefined

    const toolCalls: RosettaToolCallRequest[] = []
    for (const block of contentBlocks) {
      if (block.type === 'tool_use') {
        const toolDefinition = originalTools?.find(t => t.function.name === block.name)
        if (!toolDefinition) {
          console.warn(`Received tool call for unknown tool '${block.name}'. Skipping validation.`)
          toolCalls.push({
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
            ...(this.extractToolCaller(block.caller) ? { caller: this.extractToolCaller(block.caller) } : {})
          })
          continue
        }

        // Validate arguments using Zod schema
        const validationResult = toolDefinition.function.zodSchema.safeParse(block.input)
        if (!validationResult.success) {
          throw new ToolArgumentValidationError(
            `Arguments failed validation for tool '${block.name}'.`,
            validationResult.error.issues,
            block.name,
            block.id
          )
        }

        // Arguments are valid, add the raw tool call request
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) }, // Return raw string args
          ...(this.extractToolCaller(block.caller) ? { caller: this.extractToolCaller(block.caller) } : {})
        })
      }
    }

    return toolCalls.length > 0 ? toolCalls : undefined
  }

  mapFromProviderResponse(
    response: AnthropicMessage,
    model: string,
    originalTools?: RosettaTool<any>[]
  ): GenerateResult {
    let combinedTextContent: string | null = null
    let thinkingText: string | null = null
    const responseContent = response.content as AnthropicResponseContentBlock[]

    if (Array.isArray(responseContent)) {
      const textParts: string[] = []
      responseContent.forEach(block => {
        if (block.type === 'text') {
          textParts.push(block.text)
        } else if (block.type === 'thinking' && typeof block.thinking === 'string') {
          thinkingText = block.thinking
        }
      })
      if (textParts.length > 0) {
        combinedTextContent = textParts.join('')
      }
    }

    // Map and validate tool calls
    const toolCalls = this.mapAndValidateToolCallsFromAnthropic(responseContent, originalTools)

    const finishReason =
      response.stop_reason === 'tool_use'
        ? 'tool_calls'
        : response.stop_reason === 'max_tokens'
        ? 'length'
        : response.stop_reason === 'stop_sequence'
        ? 'stop'
        : response.stop_reason === 'end_turn'
        ? 'stop'
        : response.stop_reason ?? 'unknown'

    const usage = mapTokenUsage(response.usage)

    return {
      content: combinedTextContent,
      toolCalls: toolCalls, // Raw tool calls
      finishReason: finishReason,
      usage: usage,
      thinkingSteps: thinkingText,
      citations: undefined,
      parsedContent: null,
      model: response.model ?? model,
      rawResponse: response,
      ...(response.container
        ? {
            container: {
              id: response.container.id,
              expiresAt: response.container.expires_at
            }
          }
        : {})
    }
  }

  // --- Stream Mapping ---

  async *mapProviderStream(
    stream: AnthropicMessageStream,
    originalParams: GenerateParams // Changed from originalTools
  ): AsyncIterable<StreamChunk> {
    const originalTools = originalParams.tools // Extract tools for validation
    let currentUsage: TokenUsage | undefined
    let finalFinishReason: string | null = null
    let thinkingStarted = false
    let model = ''
    const toolCallArgAccumulators: Record<string, ToolUseAccumulator> = {}
    const toolCallIdByIndex: Record<number, string> = {}
    const serverToolUseBlocks: Record<number, ServerToolUseAccumulator> = {}
    const rawContentBlocksByIndex: Array<unknown | undefined> = []
    let aggregatedResult: GenerateResult | null = null

    try {
      for await (const event of stream) {
        if (typeof event !== 'object' || !event || !('type' in event)) {
          console.warn('Received unexpected event format from Anthropic stream:', event)
          continue
        }

        switch (event.type) {
          case 'message_start':
            model = safeGet<string>(event.message, 'model') ?? ''
            yield { type: 'message_start', data: { provider: this.provider, model: model } }
            currentUsage = mapTokenUsage(safeGet<Anthropic.Usage>(event.message, 'usage'))
            aggregatedResult = {
              content: '',
              toolCalls: [],
              finishReason: null,
              usage: undefined,
              model: model,
              thinkingSteps: null,
              citations: undefined,
              parsedContent: null,
              rawResponse: undefined
            }
            if (event.message.container) {
              const container = {
                id: event.message.container.id,
                expiresAt: event.message.container.expires_at
              }
              yield {
                type: 'container_info',
                data: {
                  containerId: container.id,
                  expiresAt: container.expiresAt
                }
              }
              aggregatedResult.container = container
            }
            break
          case 'content_block_start':
            if (event.content_block.type === 'thinking') {
              rawContentBlocksByIndex[event.index] = this.cloneRawContentBlock(event.content_block)
              yield { type: 'thinking_start' }
              thinkingStarted = true
            } else if (event.content_block.type === 'tool_use') {
              const toolUse = event.content_block
              const index = event.index
              rawContentBlocksByIndex[index] = this.cloneRawContentBlock(toolUse)
              toolCallArgAccumulators[toolUse.id] = {
                id: toolUse.id,
                name: toolUse.name,
                jsonAccumulator: '',
                inputSnapshot: toolUse.input,
                index,
                caller: this.extractToolCaller(toolUse.caller)
              }
              toolCallIdByIndex[index] = toolUse.id
              yield {
                type: 'tool_call_start',
                data: { index, toolCall: { id: toolUse.id, type: 'function', function: { name: toolUse.name } } }
              }
            } else if (event.content_block.type === 'server_tool_use' && event.content_block.name === 'code_execution') {
              const serverToolUse = event.content_block
              const initialCode = this.extractCodeExecutionCode(serverToolUse.input)
              rawContentBlocksByIndex[event.index] = this.cloneRawContentBlock(serverToolUse)
              serverToolUseBlocks[event.index] = {
                id: serverToolUse.id,
                jsonAccumulator: '',
                lastCode: initialCode,
                inputSnapshot: serverToolUse.input
              }
              yield {
                type: 'code_execution_start',
                data: {
                  id: serverToolUse.id,
                  code: initialCode
                }
              }
            } else if (event.content_block.type === 'code_execution_tool_result') {
              rawContentBlocksByIndex[event.index] = this.cloneRawContentBlock(event.content_block)
              yield this.mapCodeExecutionResultChunk(event.content_block)
            } else {
              rawContentBlocksByIndex[event.index] = this.cloneRawContentBlock(event.content_block)
            }
            break
          case 'content_block_delta':
            if (event.delta.type === 'text_delta') {
              yield { type: 'content_delta', data: { delta: event.delta.text } }
              if (aggregatedResult) aggregatedResult.content = (aggregatedResult.content ?? '') + event.delta.text
              const rawTextBlock = rawContentBlocksByIndex[event.index] as { type?: string; text?: unknown } | undefined
              if (rawTextBlock?.type === 'text') {
                rawTextBlock.text = `${typeof rawTextBlock.text === 'string' ? rawTextBlock.text : ''}${event.delta.text}`
              }
            } else if (event.delta.type === 'thinking_delta') {
              if (!thinkingStarted) {
                yield { type: 'thinking_start' }
                thinkingStarted = true
              }
              yield { type: 'thinking_delta', data: { delta: event.delta.thinking } }
              if (aggregatedResult)
                aggregatedResult.thinkingSteps = (aggregatedResult.thinkingSteps ?? '') + event.delta.thinking
              const rawThinkingBlock = rawContentBlocksByIndex[event.index] as
                | { type?: string; thinking?: unknown }
                | undefined
              if (rawThinkingBlock?.type === 'thinking') {
                rawThinkingBlock.thinking = `${
                  typeof rawThinkingBlock.thinking === 'string' ? rawThinkingBlock.thinking : ''
                }${event.delta.thinking}`
              }
            } else if (event.delta.type === 'input_json_delta') {
              const index = event.index
              const serverToolUse = serverToolUseBlocks[index]
              if (serverToolUse) {
                serverToolUse.jsonAccumulator += event.delta.partial_json
                const partialInput = this.tryPartialParseJSON(serverToolUse.jsonAccumulator)
                if (partialInput !== undefined) {
                  serverToolUse.inputSnapshot = partialInput
                  const rawServerToolUseBlock = rawContentBlocksByIndex[index] as
                    | { type?: string; input?: unknown }
                    | undefined
                  if (rawServerToolUseBlock?.type === 'server_tool_use') {
                    rawServerToolUseBlock.input = partialInput
                  }
                }
                const parsedCode = this.tryExtractCodeFromAccumulatedJSON(serverToolUse.jsonAccumulator)
                if (parsedCode !== null && parsedCode.startsWith(serverToolUse.lastCode)) {
                  const codeDelta = parsedCode.slice(serverToolUse.lastCode.length)
                  if (codeDelta.length > 0) {
                    serverToolUse.lastCode = parsedCode
                    yield {
                      type: 'code_execution_delta',
                      data: {
                        id: serverToolUse.id,
                        codeDelta,
                        snapshot: parsedCode
                      }
                    }
                  }
                }
                break
              }
              const currentToolCallId = toolCallIdByIndex[index]
              if (currentToolCallId && toolCallArgAccumulators[currentToolCallId]) {
                const toolData = toolCallArgAccumulators[currentToolCallId]
                toolData.jsonAccumulator += event.delta.partial_json
                const partialInput = this.tryPartialParseJSON(toolData.jsonAccumulator)
                if (partialInput !== undefined) {
                  toolData.inputSnapshot = partialInput
                  const rawToolUseBlock = rawContentBlocksByIndex[index] as { type?: string; input?: unknown } | undefined
                  if (rawToolUseBlock?.type === 'tool_use') {
                    rawToolUseBlock.input = partialInput
                  }
                }
                yield {
                  type: 'tool_call_delta',
                  data: { index, id: currentToolCallId, functionArgumentChunk: event.delta.partial_json }
                }
              } else {
                console.warn(`Received input_json_delta for unknown tool index: ${index}`)
              }
            }
            break
          case 'content_block_stop':
            const stoppedEvent = event as RawContentBlockStopEvent
            const stoppedBlockIndex = stoppedEvent.index
            if (serverToolUseBlocks[stoppedBlockIndex]) {
              const serverToolUse = serverToolUseBlocks[stoppedBlockIndex]
              const rawServerToolUseBlock = rawContentBlocksByIndex[stoppedBlockIndex] as
                | { type?: string; input?: unknown }
                | undefined
              if (rawServerToolUseBlock?.type === 'server_tool_use') {
                rawServerToolUseBlock.input = this.resolveAccumulatedJSONInput(serverToolUse, 'server tool')
              }
              delete serverToolUseBlocks[stoppedBlockIndex]
              break
            }
            const finishedToolCallId = toolCallIdByIndex[stoppedBlockIndex]

            if (finishedToolCallId && toolCallArgAccumulators[finishedToolCallId]) {
              const toolData = toolCallArgAccumulators[finishedToolCallId]
              const toolDefinition = originalTools?.find(t => t.function.name === toolData.name)
              const resolvedToolInput = this.resolveStreamedToolInput(toolData)

              if (toolDefinition) {
                const validationResult = toolDefinition.function.zodSchema.safeParse(resolvedToolInput)
                if (!validationResult.success) {
                  throw new ToolArgumentValidationError(
                    `Streamed arguments failed validation for tool '${toolData.name}'.`,
                    validationResult.error.issues,
                    toolData.name,
                    toolData.id
                  )
                }
              } else {
                console.warn(`Skipping validation for unknown streamed tool '${toolData.name}'.`)
              }

              // Yield done chunk after validation (or skipping)
              yield { type: 'tool_call_done', data: { index: stoppedBlockIndex, id: finishedToolCallId } }

              const rawToolUseBlock = rawContentBlocksByIndex[stoppedBlockIndex] as
                | { type?: string; input?: unknown }
                | undefined
              if (rawToolUseBlock?.type === 'tool_use') {
                rawToolUseBlock.input = resolvedToolInput
              }

              // Add raw tool call to aggregated result
              if (aggregatedResult) {
                aggregatedResult.toolCalls = aggregatedResult.toolCalls ?? []
                aggregatedResult.toolCalls.push({
                  id: toolData.id,
                  type: 'function',
                  function: {
                    name: toolData.name,
                    arguments: JSON.stringify(resolvedToolInput)
                  },
                  ...(toolData.caller ? { caller: toolData.caller } : {})
                })
              }
              delete toolCallArgAccumulators[finishedToolCallId]
              delete toolCallIdByIndex[stoppedBlockIndex]
            }
            if (thinkingStarted) {
              yield { type: 'thinking_stop' }
              thinkingStarted = false
            }
            break
          case 'message_delta':
            if (event.delta.container) {
              const container = {
                id: event.delta.container.id,
                expiresAt: event.delta.container.expires_at
              }
              const shouldEmitContainerInfo =
                !aggregatedResult?.container ||
                aggregatedResult.container.id !== container.id ||
                aggregatedResult.container.expiresAt !== container.expiresAt

              if (shouldEmitContainerInfo) {
                yield {
                  type: 'container_info',
                  data: {
                    containerId: container.id,
                    expiresAt: container.expiresAt
                  }
                }
              }

              if (aggregatedResult) {
                aggregatedResult.container = container
              }
            }
            const deltaUsage = mapTokenUsage(event.usage)
            if (deltaUsage?.completionTokens !== undefined) {
              currentUsage = {
                promptTokens: currentUsage?.promptTokens,
                completionTokens: deltaUsage.completionTokens,
                totalTokens:
                  currentUsage?.promptTokens !== undefined
                    ? currentUsage.promptTokens + deltaUsage.completionTokens
                    : deltaUsage.completionTokens
              }
            }
            if (event.delta.stop_reason) {
              finalFinishReason =
                event.delta.stop_reason === 'tool_use'
                  ? 'tool_calls'
                  : event.delta.stop_reason === 'max_tokens'
                  ? 'length'
                  : event.delta.stop_reason === 'stop_sequence'
                  ? 'stop'
                  : event.delta.stop_reason === 'end_turn'
                  ? 'stop'
                  : event.delta.stop_reason ?? 'unknown'
            }
            break
          case 'message_stop':
            finalFinishReason = finalFinishReason ?? 'stop'
            yield { type: 'message_stop', data: { finishReason: finalFinishReason } }
            if (currentUsage) {
              yield { type: 'final_usage', data: { usage: currentUsage } }
              if (aggregatedResult) aggregatedResult.usage = currentUsage
            }
            if (aggregatedResult) {
              aggregatedResult.finishReason = finalFinishReason
              if (aggregatedResult.content === '') aggregatedResult.content = null
              if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
              const rawContentBlocks = rawContentBlocksByIndex.filter(block => block !== undefined)
              if (rawContentBlocks.length > 0) {
                aggregatedResult.rawResponse = {
                  content: rawContentBlocks
                }
              }
              yield { type: 'final_result', data: { result: aggregatedResult } }
            } else {
              console.warn('Message stop received but no aggregated result was built.')
            }
            break
        }
      }
    } catch (error) {
      // Wrap and yield errors, including ToolArgumentValidationError
      const mappedError = this.wrapProviderError(error, this.provider)
      yield { type: 'error', data: { error: mappedError } }
    }
  }

  // --- Embedding Mapping ---
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToEmbedParams(_params: EmbedParams): any {
    throw new UnsupportedFeatureError(this.provider, 'Embeddings')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromEmbedResponse(_response: any, _modelId: string): EmbedResult {
    throw new UnsupportedFeatureError(this.provider, 'Embeddings')
  }

  // --- Audio Mapping ---
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranscribeParams(_params: TranscribeParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranscribeResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranslateParams(_params: TranslateParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranslateResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    // Handle specific validation errors first
    if (error instanceof ToolArgumentValidationError || error instanceof InvalidToolDefinitionError) {
      return error
    }
    if (error instanceof RosettaAIError) {
      return error
    }

    const isAnthropicAPIErrorLike = (e: any): e is APIError =>
      typeof e === 'object' &&
      e !== null &&
      typeof e.status === 'number' &&
      typeof e.message === 'string' &&
      'error' in e

    if (error instanceof Anthropic.APIError || isAnthropicAPIErrorLike(error)) {
      const anthropicError = error as APIError
      const nestedErrorType = safeGet<string>(anthropicError, 'error', 'type') // Corrected path
      const finalCode = nestedErrorType
      const finalType = nestedErrorType
      return new ProviderAPIError(anthropicError.message, provider, anthropicError.status, finalCode, finalType, error)
    }

    if (error instanceof Error) {
      return new ProviderAPIError(error.message, provider, undefined, undefined, undefined, error)
    }
    return new ProviderAPIError(
      String(error ?? 'Unknown error occurred'),
      provider,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
