import type { ProviderKey } from './common.types'
import type { TokenUsage } from './result.types'

/** A single logical generation; SDK transport retries remain within this call. */
export interface InferenceCallInfo {
  operation: 'generate' | 'stream'
  provider: ProviderKey
  model: string
}

/** Only reported token counts are carried; absent usage is never inferred as zero. */
export interface InferenceOutcome {
  status: 'completed' | 'failed' | 'cancelled'
  dispatched: boolean
  usage: TokenUsage | null
  usageComplete: boolean
  providerRequestId?: string
}

/** Optional server-owned admission and durable accounting callbacks. */
export interface InferenceLifecycle {
  admit(call: InferenceCallInfo): Promise<{
    finish(result: InferenceOutcome): Promise<void>
  }>
}

/** @internal Implemented by the SDK, called before transport and before parsing. */
export interface ProviderUsageObserver {
  beforeDispatch(): Promise<void>
  markDispatched(): void
  observeUsage(snapshot: TokenUsage | undefined, final: boolean): void
  transportComplete(): void
  observeRequestId(id: unknown): void
}
