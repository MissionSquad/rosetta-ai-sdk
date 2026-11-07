import type { InferenceCallInfo, InferenceLifecycle, InferenceOutcome, ProviderUsageObserver, TokenUsage } from '../types'

type Admission = Awaited<ReturnType<InferenceLifecycle['admit']>>

/** Per-invocation state. Never shared by concurrent calls on a cached SDK client. */
export class InferenceCall implements ProviderUsageObserver {
  private admissionPromise?: Promise<Admission>
  private admission?: Admission
  private finishPromise?: Promise<void>
  private callbackErrors = new Set<unknown>()
  private dispatched = false
  private usage: TokenUsage | null = null
  private finalUsage = false
  private requestId?: string
  private transportEnded = false

  constructor(private readonly lifecycle: InferenceLifecycle | undefined, private readonly info: InferenceCallInfo) {}

  async beforeDispatch(): Promise<void> {
    if (!this.lifecycle) return
    if (!this.admissionPromise) {
      this.admissionPromise = Promise.resolve().then(() => this.lifecycle!.admit(Object.freeze({ ...this.info }))).then(admission => {
        if (!admission || typeof admission.finish !== 'function') throw new Error('Invalid inference admission')
        this.admission = admission
        return admission
      }).catch(error => { throw this.callbackFailure(error) })
    }
    await this.admissionPromise
  }

  markDispatched(): void { this.dispatched = true }

  observeRequestId(id: unknown): void {
    if (typeof id === 'string' && id.length > 0 && id.length <= 512) this.requestId = id
  }

  observeUsage(snapshot: TokenUsage | undefined, final: boolean): void {
    if (!snapshot) { this.finalUsage = false; return }
    const counts = [snapshot.promptTokens, snapshot.completionTokens, snapshot.totalTokens, snapshot.cachedContentTokenCount]
    if (counts.some(value => value !== undefined && (!Number.isSafeInteger(value) || value < 0))) {
      this.finalUsage = false
      return
    }
    const usage = { ...snapshot }
    const split = usage.promptTokens !== undefined && usage.completionTokens !== undefined
      ? usage.promptTokens + usage.completionTokens : undefined
    if (split !== undefined && !Number.isSafeInteger(split)) { this.finalUsage = false; return }
    if (usage.totalTokens !== undefined && split !== undefined && usage.totalTokens !== split) {
      this.finalUsage = false
      return
    }
    if (usage.totalTokens === undefined && split !== undefined) usage.totalTokens = split
    this.usage = usage
    this.finalUsage = final && usage.totalTokens !== undefined && usage.incomplete !== true
  }

  transportComplete(): void { this.transportEnded = true }
  get transportCompleted(): boolean { return this.transportEnded }
  isCallbackError(error: unknown): boolean { return this.callbackErrors.has(error) }

  async finish(status: InferenceOutcome['status']): Promise<void> {
    // A rejected admission never acquired a receipt and must not be settled.
    if (!this.admission) return
    if (!this.finishPromise) {
      const outcome: InferenceOutcome = {
        status, dispatched: this.dispatched,
        usage: this.usage ? { ...this.usage } : null,
        usageComplete: !this.dispatched || this.finalUsage,
        ...(this.requestId ? { providerRequestId: this.requestId } : {})
      }
      this.finishPromise = Promise.resolve().then(() => this.admission!.finish(Object.freeze(outcome)))
        .catch(error => { throw this.callbackFailure(error) })
    }
    await this.finishPromise
  }

  private callbackFailure(error: unknown): Error {
    const failure = error instanceof Error ? error : new Error('Inference lifecycle callback failed')
    this.callbackErrors.add(failure)
    return failure
  }
}
