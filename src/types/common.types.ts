import { z } from 'zod'
import { JSONSchema7 } from 'json-schema'

/**
 * Enumeration of supported AI providers.
 */
export enum Provider {
  Anthropic = 'anthropic',
  Google = 'google',
  Groq = 'groq',
  OpenAI = 'openai' // Represents both OpenAI standard and Azure OpenAI
}

/**
 * Represents either a built-in provider enum value or a custom provider key string.
 */
export type ProviderKey = Provider | string

export type ImageMimeType = 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp'
/**
 * Represents raw image data, typically Base64 encoded.
 * @property mimeType - The MIME type of the image (e.g., 'image/jpeg', 'image/png').
 * @property base64Data - The Base64 encoded string of the image data.
 */
export interface RosettaImageData {
  mimeType: ImageMimeType
  base64Data: string
}

/**
 * Represents raw audio data for input.
 * @property data - The audio data as a Buffer or NodeJS.ReadableStream.
 * @property filename - A filename for the audio (required by some APIs).
 * @property mimeType - The MIME type of the audio (e.g., 'audio/mpeg', 'audio/wav').
 */
export interface RosettaAudioData {
  data: Buffer | NodeJS.ReadableStream
  filename: string
  mimeType: 'audio/mpeg' | 'audio/wav' | 'audio/ogg' | 'audio/webm'
}

/**
 * A discriminated union representing different parts of a message's content.
 * Supports text and image inputs.
 */
export type RosettaContentPart = { type: 'text'; text: string } | { type: 'image'; image: RosettaImageData }
// | { type: 'audio'; audio: RosettaAudioData }; // Future: If models support inline audio content parts

/**
 * Anthropic-specific replay state that must be preserved on assistant history messages.
 */
export interface AnthropicMessageProviderState {
  /** Exact provider-native content blocks to replay on follow-up turns. */
  rawContentBlocks?: unknown[]
  /** Preserves an empty-but-real assistant turn boundary for programmatic tool calling resumes. */
  assistantTurnBoundary?: boolean
}

/**
 * Provider-specific message state that must survive history persistence and replay.
 */
export interface RosettaMessageProviderState {
  anthropic?: AnthropicMessageProviderState
}

/**
 * Anthropic-specific request state required to resume a stateful tool-calling session.
 */
export interface AnthropicGenerateParamsProviderState {
  /** Provider-issued container/session identifier to reuse on the next request. */
  containerId?: string
}

/**
 * Provider-specific request state.
 */
export interface GenerateParamsProviderState {
  anthropic?: AnthropicGenerateParamsProviderState
}

/**
 * Anthropic-specific response state surfaced for downstream persistence and replay.
 */
export interface AnthropicGenerateResultProviderState {
  /** Provider-issued container/session identifier returned on the response. */
  containerId?: string
  /** Provider-issued container/session expiration, when returned. */
  expiresAt?: string | null
  /** Replayable provider-native content blocks captured from the assistant turn. */
  rawContentBlocks?: unknown[]
  /** Preserves an empty-but-real assistant turn boundary for downstream history replay. */
  assistantTurnBoundary?: boolean
}

/**
 * Provider-specific response state.
 */
export interface GenerateResultProviderState {
  anthropic?: AnthropicGenerateResultProviderState
}

/**
 * Represents a single message in a conversation.
 * @property role - The role of the message sender ('system', 'user', 'assistant', 'tool').
 * @property content - The content of the message, can be simple text or an array of content parts (for multimodal).
 * @property toolCalls - For 'assistant' role: An array of tool calls requested by the model.
 * @property toolCallId - For 'tool' role: The ID of the tool call this message is a response to.
 */
export interface RosettaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | RosettaContentPart[] | null
  toolCalls?: RosettaToolCallRequest[]
  toolCallId?: string
  /** Optional flag for tool messages indicating an error during execution. */
  isError?: boolean
  /**
   * Provider-specific message state that must be preserved verbatim across turns.
   */
  providerState?: RosettaMessageProviderState
  /**
   * @deprecated Prefer `providerState.anthropic.rawContentBlocks`.
   * Raw provider-native content blocks for assistant messages.
   * Used when an upstream provider requires exact content-block echo-back.
   */
  rawContentBlocks?: unknown[]
}

/**
 * Defines a tool (currently only functions) that the model can be instructed to use.
 * Includes both the JSON schema for provider communication and a Zod schema for validation.
 *
 * @template T - The Zod type representing the structure of the function's arguments. Defaults to `z.ZodTypeAny`.
 * @property type - The type of the tool (currently 'function').
 * @property function - Details of the function.
 * @property function.name - The name of the function to be called.
 * @property function.description - A description of what the function does, used by the model.
 * @property function.parameters - A JSON Schema object describing the expected arguments for the function (used for provider API).
 * @property function.zodSchema - A Zod schema defining the structure and types of the function's arguments (used for validation).
 */
export interface RosettaTool<T extends z.ZodTypeAny = z.ZodTypeAny> {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: JSONSchema7 // Keep JSON Schema for provider mapping
    zodSchema: T // Add Zod schema for validation
  }
  /**
   * Provider-specific caller restrictions for tools that support programmatic invocation.
   */
  allowedCallers?: Array<'direct' | 'code_execution_20250825' | 'code_execution_20260120'>
}

/**
 * Represents a tool call requested by the model in its response.
 * Contains the raw arguments string as received from the provider.
 *
 * @property id - A unique identifier for this specific tool call instance.
 * @property type - The type of tool called (currently 'function').
 * @property function - Details of the function call.
 * @property function.name - The name of the function the model wants to call.
 * @property function.arguments - A JSON string containing the arguments the model generated for the function call.
 */
export interface RosettaToolCallRequest {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string // Raw JSON string
  }
  /**
   * Provider-specific caller metadata for tools that report invocation source.
   */
  caller?: {
    type: 'direct' | 'code_execution_20250825' | 'code_execution_20260120'
    toolId?: string
  }
  /**
   * Provider-specific metadata that must be echoed back in subsequent requests.
   * Used by Google Gemini to carry thought signatures through the tool-calling loop.
   */
  providerMetadata?: Record<string, unknown>
}

/**
 * Represents a tool call received from the provider *after* successful argument validation
 * against the corresponding `RosettaTool`'s `zodSchema`.
 * This type is defined for clarity but is not directly returned by the SDK in Phase 1.
 *
 * @template T - The Zod type used for validation, inferred from the corresponding `RosettaTool`.
 * @property id - The unique identifier for the tool call instance.
 * @property type - The type of tool called (currently 'function').
 * @property function - Details of the validated function call.
 * @property function.name - The name of the function called.
 * @property function.arguments - The parsed and validated arguments, matching the structure defined by `T`.
 * @property function.rawArguments - The original, unparsed arguments string received from the provider.
 */
export interface ValidatedRosettaToolCall<T extends z.ZodTypeAny = z.ZodTypeAny> {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: z.infer<T> // Parsed and validated arguments
    rawArguments: string // Original arguments string from provider
  }
}

/**
 * Represents the result of executing a tool, to be sent back to the model.
 * This data will typically be formatted into a `RosettaMessage` with `role: 'tool'`.
 * @property toolCallId - The ID of the tool call this result corresponds to.
 * @property content - The output/result from the tool execution (usually stringified).
 * @property isError - Optional flag indicating if the tool execution resulted in an error.
 */
export interface RosettaToolResult {
  toolCallId: string
  content: string // Stringified result from tool execution
  isError?: boolean // Optional flag
}

/**
 * Represents citation metadata, often related to grounding.
 * @property text - The text content of the citation (may not always be provided).
 * @property sourceId - An identifier linking to the source material (e.g., URI, document ID).
 * @property startIndex - The starting index in the main response content where the citation applies (optional).
 * @property endIndex - The ending index in the main response content where the citation applies (optional).
 */
export interface Citation {
  text?: string
  sourceId: string
  startIndex?: number
  endIndex?: number
}

/**
 * Provider-agnostic code execution result details surfaced from providers that support managed execution.
 */
export interface CodeExecutionResultInfo {
  toolUseId: string
  stdout: string
  stderr: string
  returnCode: number
  encryptedStdout?: string
  errorCode?: string
  contentFileIds?: string[]
}
