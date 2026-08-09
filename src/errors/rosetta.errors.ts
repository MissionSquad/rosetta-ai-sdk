import { Provider, ProviderKey } from '../types'
import { z } from 'zod'

/**
 * Base error class for all errors originating from the RosettaAI SDK.
 * Includes a timestamp for when the error occurred.
 */
export class RosettaAIError extends Error {
  public readonly timestamp: Date

  constructor(message: string) {
    super(message)
    this.name = 'RosettaAIError'
    this.timestamp = new Date()
    // Maintain proper stack trace in V8 environments (Node.js)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor)
    }
  }
}

/**
 * Error indicating an issue with the SDK's configuration,
 * such as missing API keys, invalid endpoints, or missing model deployment IDs.
 */
export class ConfigurationError extends RosettaAIError {
  constructor(message: string) {
    super(`Configuration Error: ${message}`)
    this.name = 'ConfigurationError'
  }
}

/**
 * Error indicating that a requested feature (e.g., image input, JSON mode, specific tool use)
 * is not supported by the selected AI provider or the specific model being used.
 */
export class UnsupportedFeatureError extends RosettaAIError {
  public readonly provider?: Provider
  public readonly customProvider?: string
  public readonly feature: string

  constructor(provider: ProviderKey, feature: string) {
    super(`Provider '${provider}' does not support the requested feature: ${feature}`)
    this.name = 'UnsupportedFeatureError'
    // Check if provider is a value within the Provider enum
    const isBuiltIn = Object.values(Provider).includes(provider as Provider)
    this.provider = isBuiltIn ? (provider as Provider) : undefined
    this.customProvider = !isBuiltIn ? provider : undefined
    this.feature = feature
  }
}

/**
 * Error originating directly from a provider's API, such as rate limits,
 * authentication failures, invalid requests, or server errors.
 * Contains details about the provider, HTTP status code, provider-specific error codes/types,
 * and optionally the original error object from the underlying SDK.
 */
export class ProviderAPIError extends RosettaAIError {
  /** The provider that generated the error. */
  public readonly provider?: Provider
  /** The custom provider key that generated the error */
  public readonly customProvider?: string
  /** The HTTP status code returned by the API (e.g., 429, 401, 500), if available. */
  public readonly statusCode?: number
  /** A provider-specific error code string (e.g., 'invalid_api_key', 'rate_limit_exceeded'), if available. */
  public readonly errorCode?: string | null
  /** A provider-specific error type string, if available. */
  public readonly errorType?: string | null
  /** The original error object thrown by the underlying provider SDK, if available. */
  public readonly underlyingError?: unknown

  constructor(
    message: string,
    provider: ProviderKey,
    statusCode?: number,
    errorCode?: string | null,
    errorType?: string | null,
    underlyingError?: unknown
  ) {
    const statusString = statusCode ? `(Status ${statusCode}) ` : ''
    const codeString = errorCode ? `[Code: ${errorCode}] ` : ''
    super(`[${provider}] API Error ${statusString}${codeString}: ${message}`)

    this.name = 'ProviderAPIError'
    // Correctly assign provider/customProvider based on whether the key is in the Provider enum
    const isBuiltIn = Object.values(Provider).includes(provider as Provider)
    this.provider = isBuiltIn ? (provider as Provider) : undefined
    this.customProvider = !isBuiltIn ? provider : undefined
    this.statusCode = statusCode
    this.errorCode = errorCode
    this.errorType = errorType
    this.underlyingError = underlyingError

    // Attempt to capture stack from underlying error if it's an Error instance
    if (underlyingError instanceof Error && underlyingError.stack) {
      this.stack = `${this.name}: ${this.message}\nCaused by: ${underlyingError.stack}`
    }
  }
}

/**
 * Error indicating a failure during data mapping or processing within the SDK,
 * such as converting between RosettaAI types and provider-specific formats.
 */
export class MappingError extends RosettaAIError {
  /** The provider involved in the mapping, if applicable. */
  public readonly provider?: Provider
  /** The custom provider key that generated the error */
  public readonly customProvider?: string
  /** Contextual information about where the mapping error occurred (e.g., function name). */
  public readonly context?: string

  constructor(message: string, provider?: ProviderKey, context?: string, cause?: unknown) {
    const providerString = provider ? `[${provider}]` : ''
    const ctxString = context ? ` [Context: ${context}]` : ''
    super(`Mapping Error ${providerString}${ctxString}: ${message}`)
    this.name = 'MappingError'
    const isBuiltIn = Object.values(Provider).includes(provider as Provider)
    this.provider = isBuiltIn ? (provider as Provider) : undefined
    this.customProvider = !isBuiltIn ? provider : undefined
    this.context = context
    if (cause instanceof Error && cause.stack) {
      this.stack = `${this.name}: ${this.message}\nCaused by: ${cause.stack}`
    }
  }
}

export type ComputerUseMappingErrorCode =
  | 'PROVIDER_ACTION_SHAPE_UNSUPPORTED'
  | 'PROVIDER_ACTION_BATCH_UNSUPPORTED'
  | 'PROVIDER_ACTION_MODIFIERS_UNSUPPORTED'
  | 'PROVIDER_ACTION_UNSUPPORTED'
  | 'PROVIDER_ACTION_INVALID'

/** A typed, fail-closed provider-to-canonical computer-use mapping failure. */
export class ComputerUseMappingError extends MappingError {
  public readonly code: ComputerUseMappingErrorCode

  constructor(code: ComputerUseMappingErrorCode, message: string, provider?: ProviderKey, cause?: unknown) {
    super(`${code}: ${message}`, provider, 'computer_use', cause)
    this.name = 'ComputerUseMappingError'
    this.code = code
  }
}

/** A typed failure for structured provider output that does not satisfy its runtime schema. */
export class StructuredOutputValidationError extends RosettaAIError {
  public readonly issues: z.ZodIssue[]
  public readonly schemaName?: string
  public readonly receivedInput?: unknown
  public readonly underlyingError?: unknown

  constructor(
    message: string,
    issues: z.ZodIssue[],
    schemaName?: string,
    receivedInput?: unknown,
    underlyingError?: unknown
  ) {
    super(`Structured Output Validation Error${schemaName ? ` for '${schemaName}'` : ''}: ${message}`)
    this.name = 'StructuredOutputValidationError'
    this.issues = issues
    this.schemaName = schemaName
    this.receivedInput = receivedInput
    this.underlyingError = underlyingError
    if (underlyingError instanceof Error && underlyingError.stack) {
      this.stack = `${this.name}: ${this.message}\nCaused by: ${underlyingError.stack}`
    }
  }
}

/**
 * Error indicating an issue with a tool definition provided to the SDK.
 * This could be an invalid JSON schema for `parameters` or an invalid `zodSchema`.
 */
export class InvalidToolDefinitionError extends RosettaAIError {
  constructor(message: string, toolName?: string) {
    super(`Invalid Tool Definition${toolName ? ` for '${toolName}'` : ''}: ${message}`)
    this.name = 'InvalidToolDefinitionError'
  }
}

/**
 * Error indicating that the arguments provided by the LLM for a tool call
 * failed validation against the tool's `zodSchema`.
 */
export class ToolArgumentValidationError extends RosettaAIError {
  /** The Zod validation issues. */
  public readonly issues: z.ZodIssue[]
  /** The name of the tool whose arguments failed validation. */
  public readonly toolName?: string
  /** The ID of the specific tool call that failed validation. */
  public readonly toolCallId?: string
  /**
   * The raw arguments the model produced for the tool call that failed schema
   * validation (the parsed value passed to `zodSchema.safeParse`). Exposed so
   * consumers driving the tool-execution loop can build an `is_error`
   * tool_result and let the model retry, rather than treating the failure as
   * terminal. Present when the SDK has the model's arguments available.
   */
  public readonly receivedInput?: unknown

  constructor(message: string, issues: z.ZodIssue[], toolName?: string, toolCallId?: string, receivedInput?: unknown) {
    super(`Tool Argument Validation Error${toolName ? ` for '${toolName}'` : ''}: ${message}`)
    this.name = 'ToolArgumentValidationError'
    this.issues = issues
    this.toolName = toolName
    this.toolCallId = toolCallId
    this.receivedInput = receivedInput
  }
}

/**
 * Error indicating a failure during the *user's* execution of a tool function.
 * This error is defined for completeness but is typically thrown by the user's
 * tool implementation, not directly by the SDK in Phase 1.
 */
export class ToolExecutionError extends RosettaAIError {
  /** The name of the tool that failed during execution. */
  public readonly toolName?: string
  /** The ID of the specific tool call that failed execution. */
  public readonly toolCallId?: string
  /** The original error thrown by the tool's execution logic. */
  public readonly underlyingError?: unknown

  constructor(message: string, toolName?: string, toolCallId?: string, underlyingError?: unknown) {
    super(`Tool Execution Error${toolName ? ` for '${toolName}'` : ''}: ${message}`)
    this.name = 'ToolExecutionError'
    this.toolName = toolName
    this.toolCallId = toolCallId
    this.underlyingError = underlyingError
    if (underlyingError instanceof Error && underlyingError.stack) {
      this.stack = `${this.name}: ${this.message}\nCaused by: ${underlyingError.stack}`
    }
  }
}
