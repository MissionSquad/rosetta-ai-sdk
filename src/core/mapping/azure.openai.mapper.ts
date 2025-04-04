import OpenAI from 'openai'
// Import necessary types, including specific content part types
import {
  ChatCompletionToolChoiceOption as OpenAIToolChoiceOption,
  ChatCompletionTool as OpenAITool,
  ChatCompletionMessageParam as OpenAIMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionContentPart, // General content part for User messages
  ChatCompletionContentPartText, // Specific text part for Assistant messages
  ChatCompletionContentPartRefusal, // Specific refusal part for Assistant messages
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionDeveloperMessageParam // Import if supporting 'developer' role
} from 'openai/resources/chat/completions'
import { FunctionDefinition as OpenAIFunctionDef } from 'openai/resources/shared'

import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  Provider,
  RosettaAIConfig,
  EmbedParams,
  EmbedResult,
  RosettaMessage // Import RosettaMessage
} from '../../types'
import { Stream } from 'openai/streaming'
import { MappingError, UnsupportedFeatureError, ConfigurationError } from '../../errors'

// Re-export needed functions from base mapper
import {
  mapFromOpenAIResponse as mapFromOpenAIBaseResponse,
  mapFromOpenAIEmbedResponse as mapFromOpenAIBaseEmbedResponse,
  mapOpenAIStream
} from './openai.mapper'

// --- Parameter Mapping (Chat/Completion) ---

// Reuse role mapping
import { mapRoleToOpenAI } from './openai.mapper'

// Adjusted content mapping for flexibility
function mapContentForOpenAIRole(
  content: RosettaMessage['content'],
  role: OpenAI.Chat.Completions.ChatCompletionRole
):
  | string
  | ChatCompletionContentPart[]
  | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal>
  | null {
  if (content === null) {
    // Null content is only valid for assistant (with tool calls) or tool (though string preferred)
    if (role === 'assistant' || role === 'tool') return null
    // System/User/Developer require content
    throw new MappingError(`Role '${role}' requires non-null content.`, Provider.OpenAI)
  }
  if (typeof content === 'string') {
    // String content is valid for all roles except potentially Tool if JSON expected?
    // Tool content should be string according to types.
    return content
  }

  // Handle array of parts
  const mappedParts: ChatCompletionContentPart[] = content.map(part => {
    if (part.type === 'text') {
      return { type: 'text', text: part.text } as ChatCompletionContentPartText
    } else if (part.type === 'image') {
      // Image parts are only valid for the 'user' role content array
      if (role !== 'user') {
        throw new MappingError(
          `Image content parts are only allowed for the 'user' role, not '${role}'.`,
          Provider.OpenAI
        )
      }
      return {
        type: 'image_url',
        image_url: { url: `data:${part.image.mimeType};base64,${part.image.base64Data}` }
      } // Structurally matches ChatCompletionContentPartImage
    } else {
      const _e: never = part
      throw new MappingError(`Unsupported content part type: ${(_e as any).type}`, Provider.OpenAI)
    }
  })

  // Adjust return type based on role constraints
  if (role === 'user') {
    return mappedParts // User can have text/image parts
  } else if (role === 'assistant') {
    // Assistant content must be string or array of Text/Refusal parts
    const assistantContentParts = mappedParts.filter((p): p is ChatCompletionContentPartText => p.type === 'text')
    if (assistantContentParts.length === mappedParts.length) {
      // If only text parts remain, return them
      return assistantContentParts
    } else {
      // If non-text parts were filtered out (e.g., image attempt), return only text or error?
      console.warn(`Non-text/refusal content parts were filtered out for assistant message.`)
      // Decide: return only text, empty array, or throw? Returning only text seems safest.
      if (assistantContentParts.length > 0) return assistantContentParts
      // If NO text parts remain after filtering, return null (if tool calls exist) or empty string.
      return null // Allow null only if tool calls handle the empty content case later
    }
  } else if (role === 'system' || role === 'developer') {
    // System/Developer content can be string or TextPart array
    const textParts = mappedParts.filter((p): p is ChatCompletionContentPartText => p.type === 'text')
    if (textParts.length !== mappedParts.length) {
      throw new MappingError(`Role '${role}' content array can only contain text parts.`, Provider.OpenAI)
    }
    return textParts // Return array of text parts
  } else if (role === 'tool') {
    // Tool content must be string (or technically null). Stringify complex content.
    try {
      return JSON.stringify(mappedParts)
    } catch {
      throw new MappingError(`Could not stringify complex content parts for tool message role.`, Provider.OpenAI)
    }
  } else {
    // Fallback/error for unhandled roles (like 'function')
    throw new MappingError(`Cannot map content parts for role '${role}'.`, Provider.OpenAI)
  }
}

export function mapToAzureOpenAIParams(
  params: GenerateParams,
  config: RosettaAIConfig
): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
  const deploymentId =
    params.providerOptions?.azureChatDeploymentId ??
    config.providerOptions?.[Provider.OpenAI]?.azureChatDeploymentId ??
    config.azureOpenAIDefaultChatDeploymentName

  if (!deploymentId) {
    throw new ConfigurationError('Azure chat deployment ID/name must be configured.')
  }

  const messages: OpenAIMessageParam[] = params.messages.map(msg => {
    const role = mapRoleToOpenAI(msg.role)
    // Map content specifically for the target role
    const content = mapContentForOpenAIRole(msg.content, role)

    switch (role) {
      case 'system':
        // Content mapping already validated type
        return { role: 'system', content: content as string } as ChatCompletionSystemMessageParam // Cast needed if map returns union
      case 'user':
        // Content mapping already validated type
        return {
          role: 'user',
          content: content as string | ChatCompletionContentPart[]
        } as ChatCompletionUserMessageParam // Cast needed if map returns union
      case 'assistant':
        // FIX: Construct assistant message ensuring content type matches
        const assistantMsg: ChatCompletionAssistantMessageParam = {
          role: 'assistant',
          // Assign mapped content, ensuring it fits Assistant's allowed types
          content: content as string | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal> | null
        }
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
            id: tc.id,
            type: tc.type as 'function',
            function: { name: tc.function.name, arguments: tc.function.arguments }
          }))
          // Ensure content is null if tool calls exist and content was originally null/empty
          if (assistantMsg.content === '' || content === null) {
            assistantMsg.content = null
          }
        } else if (assistantMsg.content === null) {
          // If no tool calls, content cannot be null, default to empty string
          assistantMsg.content = ''
        }
        return assistantMsg
      case 'tool':
        if (!msg.toolCallId) throw new MappingError('Tool message requires toolCallId.', Provider.OpenAI)
        // Tool content mapping results in string or null. Type requires string | TextPart[]? Check SDK again.
        // From completions.d.ts: ChatCompletionToolMessageParam.content is string | Array<ChatCompletionContentPartText>
        // Let's ensure it's string for now as that's most common.
        let toolContentString: string
        if (typeof content === 'string') {
          toolContentString = content
        } else if (content === null) {
          toolContentString = '' // Prefer empty string over null if possible
        } else {
          // If content ended up as parts (e.g. text parts), join them or stringify
          toolContentString = Array.isArray(content)
            ? content.map(c => (c as ChatCompletionContentPartText).text).join('')
            : String(content)
          console.warn(
            'Tool content was complex, converting to string. Ensure the receiving model expects this format.'
          )
        }
        return {
          role: 'tool',
          tool_call_id: msg.toolCallId,
          content: toolContentString // Ensure content is string here
        } as ChatCompletionToolMessageParam
      case 'developer':
        if (typeof content !== 'string' && !(Array.isArray(content) && content.every(p => p.type === 'text'))) {
          throw new MappingError(`Developer message content must be string or text parts array.`, Provider.OpenAI)
        }
        return {
          role: 'developer',
          content: content as string | ChatCompletionContentPartText[]
        } as ChatCompletionDeveloperMessageParam
      // Skip 'function' role case

      default:
        throw new MappingError(`Unhandled role type during message construction: ${role}`, Provider.OpenAI)
    }
  })

  // --- Tool Mapping --- (No changes from previous version)
  const tools: OpenAITool[] | undefined = params.tools?.map(tool => {
    if (tool.type !== 'function')
      throw new MappingError(`Unsupported tool type for Azure OpenAI: ${tool.type}`, Provider.OpenAI)
    const parameters = tool.function.parameters as OpenAIFunctionDef['parameters']
    if (typeof parameters !== 'object' || parameters === null)
      throw new MappingError(`Invalid parameters schema for tool ${tool.function.name}.`, Provider.OpenAI)
    return {
      type: tool.type,
      function: { name: tool.function.name, description: tool.function.description, parameters: parameters }
    }
  })

  // --- Tool Choice Mapping --- (No changes from previous version)
  let toolChoice: OpenAIToolChoiceOption | undefined
  if (typeof params.toolChoice === 'string' && ['none', 'auto', 'required'].includes(params.toolChoice))
    toolChoice = params.toolChoice as 'none' | 'auto' | 'required'
  else if (typeof params.toolChoice === 'object' && params.toolChoice.type === 'function')
    toolChoice = { type: 'function', function: { name: params.toolChoice.function.name } }
  else if (params.toolChoice) toolChoice = 'auto'

  // --- Response Format Mapping --- (No changes from previous version)
  let responseFormat: { type: 'text' | 'json_object' } | undefined
  if (params.responseFormat?.type === 'json_object') {
    responseFormat = { type: 'json_object' }
  } else if (params.responseFormat?.type === 'text') {
    responseFormat = { type: 'text' }
  }

  // --- Unsupported Feature Checks --- (No changes from previous version)
  if (params.thinking) throw new UnsupportedFeatureError(Provider.OpenAI, 'Thinking steps')
  if (params.grounding) throw new UnsupportedFeatureError(Provider.OpenAI, 'Grounding/Citations')

  // --- Construct Final Payload ---
  const basePayload = {
    model: deploymentId,
    messages,
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    stop: params.stop,
    tools,
    tool_choice: toolChoice,
    response_format: responseFormat
  }

  if (params.stream) {
    return {
      ...basePayload,
      stream: true,
      stream_options: { include_usage: true }
    } as ChatCompletionCreateParamsStreaming
  } else {
    return { ...basePayload, stream: false } as ChatCompletionCreateParamsNonStreaming
  }
}

// --- Result Mapping (Chat/Completion) --- (No changes needed)
export function mapFromAzureOpenAIResponse(response: ChatCompletion, modelUsed: string): GenerateResult {
  return mapFromOpenAIBaseResponse(response, modelUsed)
}

// --- Stream Mapping (Chat/Completion) --- (No changes needed)
export async function* mapAzureOpenAIStream(stream: Stream<ChatCompletionChunk>): AsyncIterable<StreamChunk> {
  yield* mapOpenAIStream(stream)
}

// --- Parameter Mapping (Embeddings) --- (No changes needed)
export function mapToAzureOpenAIEmbedParams(
  params: EmbedParams,
  config: RosettaAIConfig
): OpenAI.Embeddings.EmbeddingCreateParams {
  const deploymentId =
    params.providerOptions?.azureEmbeddingDeploymentId ??
    config.providerOptions?.[Provider.OpenAI]?.azureEmbeddingDeploymentId ??
    config.azureOpenAIDefaultEmbeddingDeploymentName
  if (!deploymentId) {
    throw new ConfigurationError('Azure embedding deployment ID/name must be configured.')
  }
  let inputData: OpenAI.Embeddings.EmbeddingCreateParams['input']
  if (typeof params.input === 'string' || Array.isArray(params.input)) {
    inputData = params.input
  } else {
    throw new MappingError('Invalid input type for Azure OpenAI embeddings.', Provider.OpenAI)
  }
  return {
    model: deploymentId,
    input: inputData,
    encoding_format: params.encodingFormat,
    dimensions: params.dimensions
  }
}

// --- Result Mapping (Embeddings) --- (No changes needed)
export function mapFromAzureOpenAIEmbedResponse(
  response: OpenAI.Embeddings.CreateEmbeddingResponse,
  modelUsed: string
): EmbedResult {
  return mapFromOpenAIBaseEmbedResponse(response, modelUsed)
}

// Audio mapping reuses standard OpenAI mapping logic.
