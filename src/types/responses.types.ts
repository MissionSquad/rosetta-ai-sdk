import { Provider, RosettaImageData } from './common.types'
import { ThinkingStreamChunk } from './stream.types'
import { JSONSchema7 } from 'json-schema'
import { z } from 'zod'
import { ComputerUseDecision } from './computer-use.types'

/**
 * OpenAI Responses API - New stateful conversation interface.
 * This is a separate interface from Chat Completions, designed for agent-ready interactions.
 *
 * Key differences from Chat Completions:
 * - Stateful conversations via previous_response_id (no history replay needed)
 * - Separates instructions (developer/system intent) from input (user content)
 * - Built-in tools (web_search, file_search, image_generation, code_interpreter)
 * - Semantic streaming events (not just content deltas)
 *
 * Currently OpenAI-only. Other providers may adopt similar patterns in the future.
 */

// ===== Input Types =====

/**
 * Multimodal input item for Responses API.
 */
export type ResponsesInputItem =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string }
  | { type: 'input_image'; image: RosettaImageData }
  | ResponsesComputerCallOutput

export type ResponsesComputerScreenshot =
  | { type: 'computer_screenshot'; image_url: string }
  | { type: 'computer_screenshot'; file_id: string }
  | { type: 'computer_screenshot'; image: RosettaImageData }

export interface ResponsesAcknowledgedSafetyCheck {
  id: string
  code?: string | null
  message?: string | null
}

/** Caller-authorized continuation for one native computer call. */
export interface ResponsesComputerCallOutput {
  type: 'computer_call_output'
  call_id: string
  output: ResponsesComputerScreenshot
  acknowledged_safety_checks?: ResponsesAcknowledgedSafetyCheck[]
}

/**
 * Built-in tool types available in Responses API.
 */
export type ResponsesBuiltInToolType =
  | 'web_search'
  | 'file_search'
  | 'image_generation'
  | 'code_interpreter'
  | 'computer'

/**
 * Configuration for built-in tools.
 */
export interface ResponsesWebSearchTool {
  type: 'web_search'
}

export interface ResponsesFileSearchTool {
  type: 'file_search'
  /** Optional: IDs of vector stores to search */
  vector_store_ids?: string[]
}

export interface ResponsesImageGenerationTool {
  type: 'image_generation'
  /** Optional: Image generation options */
  options?: {
    size?: '1024x1024' | '1792x1024' | '1024x1792'
    quality?: 'standard' | 'hd'
    style?: 'vivid' | 'natural'
  }
}

export interface ResponsesCodeInterpreterTool {
  type: 'code_interpreter'
}

/** OpenAI GA native computer tool. */
export interface ResponsesComputerTool {
  type: 'computer'
}

/**
 * Custom function tool for Responses API.
 * Similar to Chat Completions but in Responses context.
 */
export interface ResponsesFunctionTool<T extends z.ZodTypeAny = z.ZodTypeAny> {
  type: 'function'
  name: string
  description?: string
  parameters: JSONSchema7
  /** Preserve the caller's OpenAI strict-mode selection; null leaves the provider default in effect. */
  strict?: boolean | null
  /** Zod schema for runtime validation */
  zodSchema?: T
}

/**
 * Union of all tool types for Responses API.
 */
export type ResponsesTool<T extends z.ZodTypeAny = z.ZodTypeAny> =
  | ResponsesWebSearchTool
  | ResponsesFileSearchTool
  | ResponsesImageGenerationTool
  | ResponsesCodeInterpreterTool
  | ResponsesComputerTool
  | ResponsesFunctionTool<T>

/**
 * Tool choice options for Responses API.
 */
export type ResponsesToolChoice =
  | 'auto'
  | 'required'
  | 'none'
  | { type: Exclude<ResponsesBuiltInToolType, 'web_search'> }
  | { type: 'function'; name: string }

/**
 * Response format for structured outputs.
 */
export interface ResponsesFormat {
  type: 'json_schema'
  json_schema: {
    name: string
    strict?: boolean
    schema: JSONSchema7
  }
}

/**
 * Parameters for creating a Response via OpenAI Responses API.
 */
export interface CreateResponseParams {
  /** Provider must be OpenAI (responses API is OpenAI-specific) */
  provider: Provider.OpenAI | 'openai'
  /** Model to use (e.g., 'gpt-4o', 'gpt-4o-mini') */
  model?: string
  /** Developer/system instructions (replaces system messages) */
  instructions?: string
  /** User input - can be string or multimodal array */
  input?: string | ResponsesInputItem[]
  /** Tools available to the model (built-in + custom functions) */
  tools?: ResponsesTool[]
  /** Control which tool(s) the model can/must use */
  tool_choice?: ResponsesToolChoice
  /** Request structured JSON output */
  response_format?: ResponsesFormat
  /** Chain from previous response (stateful conversations) */
  previous_response_id?: string
  /** Maximum tokens to generate */
  max_tokens?: number
  /** Temperature (0-2) */
  temperature?: number
  /** Top-p sampling */
  top_p?: number
  /** Stop sequences */
  stop?: string | string[]
  /** Enable streaming */
  stream?: boolean
  /** Optional metadata */
  metadata?: Record<string, string>
}

// ===== Output Types =====

/**
 * Output item types from Responses API.
 */
export type ResponsesOutputItem =
  | { type: 'output_text'; text: string }
  | { type: 'function_call'; id?: string; call_id: string; name: string; arguments: string }
  | { type: 'image'; image_url: string }
  | ResponsesComputerCall

/** Provider-neutral representation of a native Responses computer call. */
export interface ResponsesComputerCall {
  type: 'computer_call'
  status: 'in_progress' | 'completed' | 'incomplete'
  decision: ComputerUseDecision
}

/**
 * Tool call in Responses API.
 */
export interface ResponsesToolCall {
  /** Provider output-item identity used for tracing. */
  id?: string
  /** Function-call correlation identity used when returning tool output. */
  call_id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

/**
 * Token usage information.
 */
export interface ResponsesUsage {
  input_tokens: number
  output_tokens: number
  total_tokens: number
}

/**
 * Result from a non-streaming Responses API call.
 */
export interface ResponseResult {
  /** Unique response ID (use as previous_response_id in next turn) */
  id: string
  /** Array of output items */
  output: ResponsesOutputItem[]
  /** Convenience accessor - concatenated text output */
  output_text: string
  /** Tool calls requested by the model */
  tool_calls?: ResponsesToolCall[]
  /** Token usage statistics */
  usage?: ResponsesUsage
  /** Model used */
  model: string
  /** Finish reason */
  finish_reason?: string
  /** Raw response from OpenAI SDK */
  rawResponse?: unknown
}

// ===== Streaming Types =====

/**
 * Semantic event types for Responses API streaming.
 * Based on OpenAI's event-driven streaming model.
 */
export type ResponsesStreamChunk =
  | { type: 'response.created'; data: { id: string; model: string } }
  | { type: 'response.output_text.delta'; data: { delta: string } }
  | { type: 'response.output_text.done'; data: { text: string } }
  | { type: 'response.tool_call.start'; data: { id: string; name: string } }
  | { type: 'response.tool_call.delta'; data: { id: string; delta: string } }
  | { type: 'response.tool_call.done'; data: { id: string; name: string; arguments: string } }
  | { type: 'response.completed'; data: ResponseResult }
  | { type: 'response.failed'; data: { error: { message: string; code?: string } } }
  | { type: 'response.cancelled'; data: { reason?: string } }
  | { type: 'error'; data: { error: Error } }
  | ThinkingStreamChunk
