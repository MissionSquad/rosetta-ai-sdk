import { Provider } from '../types'

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
  public readonly provider: Provider
  public readonly feature: string

  constructor(provider: Provider, feature: string) {
    super(`Provider '${provider}' does not support the requested feature: ${feature}`)
    this.name = 'UnsupportedFeatureError'
    this.provider = provider
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
  public readonly provider: Provider
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
    provider: Provider,
    statusCode?: number,
    errorCode?: string | null,
    errorType?: string | null,
    underlyingError?: unknown
  ) {
    const statusString = statusCode ? `(Status ${statusCode}) ` : ''
    const codeString = errorCode ? `[Code: ${errorCode}] ` : ''
    super(`[${provider}] API Error ${statusString}${codeString}: ${message}`)

    this.name = 'ProviderAPIError'
    this.provider = provider
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
  /** Contextual information about where the mapping error occurred (e.g., function name). */
  public readonly context?: string

  constructor(message: string, provider?: Provider, context?: string, cause?: unknown) {
    const providerString = provider ? `[${provider}]` : ''
    const ctxString = context ? ` [Context: ${context}]` : ''
    super(`Mapping Error ${providerString}${ctxString}: ${message}`)
    this.name = 'MappingError'
    this.provider = provider
    this.context = context
    if (cause instanceof Error && cause.stack) {
      this.stack = `${this.name}: ${this.message}\nCaused by: ${cause.stack}`
    }
  }
}
