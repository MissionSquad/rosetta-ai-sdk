import { InferenceCall } from '../../../src/core/inference-lifecycle'
import type { InferenceOutcome, TokenUsage } from '../../../src/types'

async function fixture() {
  const finish = jest.fn<Promise<void>, [InferenceOutcome]>(async () => undefined)
  const admit = jest.fn(async () => ({ finish }))
  const call = new InferenceCall({ admit }, { operation: 'generate', provider: 'fixture', model: 'fixture-model' })
  await call.beforeDispatch()
  call.markDispatched()
  return { call, finish, admit }
}

describe('per-call inference usage collector', () => {
  test('a provider-final total-only report is complete without inventing input/output', async () => {
    const { call, finish } = await fixture()
    call.observeUsage({ totalTokens: 30 }, true)
    await call.finish('completed')
    expect(finish).toHaveBeenCalledWith({ status: 'completed', dispatched: true, usageComplete: true, usage: { totalTokens: 30 } })
    const usage = finish.mock.calls[0]?.[0].usage
    expect(usage).not.toHaveProperty('promptTokens')
    expect(usage).not.toHaveProperty('completionTokens')
  })

  test('repeated snapshots replace earlier counts rather than being added', async () => {
    const { call, finish, admit } = await fixture()
    call.observeUsage({ promptTokens: 5, completionTokens: 1, totalTokens: 6 }, false)
    call.observeUsage({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    call.observeUsage({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, true)
    await call.beforeDispatch()
    await Promise.all([call.finish('completed'), call.finish('completed')])
    expect(admit).toHaveBeenCalledTimes(1)
    expect(finish).toHaveBeenCalledTimes(1)
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ usageComplete: true, usage: { promptTokens: 5, completionTokens: 7, totalTokens: 12 } })
  })

  const invalid: Array<[string, TokenUsage]> = [
    ['negative input', { promptTokens: -1, completionTokens: 7, totalTokens: 6 }],
    ['negative output', { promptTokens: 5, completionTokens: -1, totalTokens: 4 }],
    ['negative total', { totalTokens: -1 }],
    ['unsafe total', { totalTokens: Number.MAX_SAFE_INTEGER + 1 }],
    ['unsafe split sum', { promptTokens: Number.MAX_SAFE_INTEGER, completionTokens: 1 }],
    ['noninteger count', { totalTokens: 1.5 }],
    ['nonfinite count', { totalTokens: Infinity }],
    ['NaN count', { totalTokens: NaN }],
    ['negative cached count', { totalTokens: 12, cachedContentTokenCount: -1 }]
  ]
  test.each(invalid)('%s is incomplete and does not invent a charge', async (_name, snapshot) => {
    const { call, finish } = await fixture()
    call.observeUsage(snapshot, true)
    call.transportComplete()
    await call.finish('failed')
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ dispatched: true, usage: null, usageComplete: false })
  })

  test.each(invalid)('retains prior valid partial usage after %s', async (_name, snapshot) => {
    const { call, finish } = await fixture()
    call.observeUsage({ promptTokens: 5 }, false)
    call.observeUsage(snapshot, true)
    await call.finish('failed')
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ usageComplete: false, usage: { promptTokens: 5 } })
    expect(finish.mock.calls[0]?.[0].usage).not.toHaveProperty('completionTokens')
    expect(finish.mock.calls[0]?.[0].usage).not.toHaveProperty('totalTokens')
  })

  test('a mismatched split and total is reported as incomplete, never as final usage', async () => {
    const { call, finish } = await fixture()
    call.observeUsage({ promptTokens: 5 }, false)
    call.observeUsage({ promptTokens: 5, completionTokens: 7, totalTokens: 13 }, true)
    call.transportComplete()
    await call.finish('completed')
    expect(finish.mock.calls[0]?.[0]).toMatchObject({
      dispatched: true, usageComplete: false,
      usage: { promptTokens: 5, completionTokens: 7, totalTokens: 13, incomplete: true }
    })
  })

  test('transport completion does not promote an early sample to provider-final usage', async () => {
    const { call, finish } = await fixture()
    call.observeUsage({ promptTokens: 5, completionTokens: 7, totalTokens: 12 }, false)
    call.transportComplete()
    await call.finish('completed')
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ usageComplete: false, usage: { totalTokens: 12 } })
  })

  test('a complete split supplies the total; explicit incompleteness remains authoritative', async () => {
    const { call, finish } = await fixture()
    call.observeUsage({ promptTokens: 5, completionTokens: 7, incomplete: true }, true)
    await call.finish('completed')
    expect(finish.mock.calls[0]?.[0]).toMatchObject({ usageComplete: false, usage: { promptTokens: 5, completionTokens: 7, totalTokens: 12, incomplete: true } })
  })
})
