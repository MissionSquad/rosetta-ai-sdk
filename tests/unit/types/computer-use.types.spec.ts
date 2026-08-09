import {
  authorityContextSchema,
  computerActionKinds,
  computerActionResultSchema,
  computerActionSchema,
  computerActionToolArgsSchema,
  computerKeys,
  computerObservationSchema,
  computerUseCapabilitiesSchema,
  computerUseDecisionSchema,
  computerUseTurnSchema,
  normalizedPointSchema,
  pendingSafetyCheckSchema
} from '../../../src/types/computer-use.types'
import { ZodTypeAny } from 'zod'

const VALID_HASH = 'a'.repeat(64)
const VALID_TIMESTAMP = '2026-08-08T18:30:00Z'
const VALID_CLICK = {
  kind: 'click',
  point: { x: 0.812, y: 0.744 },
  button: 'left'
} as const

const makeObservation = () => ({
  schemaVersion: '1',
  observationId: 'observation-1',
  screenshot: {
    mediaType: 'image/png',
    data: 'base64-data',
    width: 1280,
    height: 720,
    sha256: VALID_HASH
  },
  url: 'https://example.com/schedule',
  capturedAt: VALID_TIMESTAMP,
  browserSessionId: 'browser-session-1',
  controlEpoch: 3,
  allowedActionKinds: [...computerActionKinds],
  goal: 'Open Schedule'
})

const makeAuthority = () => ({
  commsAgentId: 'agent-1',
  authority: 'execute_bounded',
  allowedDomains: ['example.com'],
  approvalId: 'approval-1',
  approvalHash: VALID_HASH,
  emergencyStopVersion: 2
})

const makeResult = () => ({
  schemaVersion: '1',
  actionId: 'action-1',
  action: VALID_CLICK,
  observationSha256: VALID_HASH,
  startedAt: VALID_TIMESTAMP,
  completedAt: '2026-08-08T18:30:01+00:00',
  status: 'executed',
  verification: 'matched',
  resultingUrl: 'https://example.com/schedule',
  screenshotSha256: 'b'.repeat(64)
})

const makeTurn = () => ({
  schemaVersion: '1',
  taskId: 'task-1',
  goal: 'Open Schedule',
  step: 1,
  observation: makeObservation(),
  activeApplication: 'chromium',
  domSnapshot: '<main>Schedule</main>',
  accessibilityTree: 'main: Schedule',
  recentActions: [makeResult()],
  allowedActions: [...computerActionKinds],
  authority: makeAuthority(),
  maxActionsThisTurn: 1
})

const expectAccepted = (schema: ZodTypeAny, value: unknown): void => {
  expect(schema.safeParse(value).success).toBe(true)
}

const expectRejected = (schema: ZodTypeAny, value: unknown): void => {
  expect(schema.safeParse(value).success).toBe(false)
}

describe('canonical computer-use schemas', () => {
  describe('NormalizedPoint', () => {
    it.each([
      ['origin', { x: 0, y: 0 }],
      ['inclusive endpoint', { x: 1, y: 1 }],
      ['interior point', { x: 0.5, y: 0.25 }]
    ])('accepts the %s', (_name, point) => {
      expectAccepted(normalizedPointSchema, point)
    })

    it.each([
      ['negative x', { x: -Number.EPSILON, y: 0 }],
      ['x above one', { x: 1 + Number.EPSILON, y: 0 }],
      ['negative y', { x: 0, y: -Number.EPSILON }],
      ['y above one', { x: 0, y: 1 + Number.EPSILON }],
      ['NaN', { x: Number.NaN, y: 0 }],
      ['positive infinity', { x: Number.POSITIVE_INFINITY, y: 0 }],
      ['negative infinity', { x: 0, y: Number.NEGATIVE_INFINITY }]
    ])('rejects %s', (_name, point) => {
      expectRejected(normalizedPointSchema, point)
    })
  })

  describe('ComputerUseCapabilities', () => {
    const baseCapabilities = {
      vision: true,
      nativeComputerTool: true,
      structuredToolCalls: true,
      batchedActions: false,
      acceptsDomSnapshot: false,
      acceptsAccessibilityTree: true
    } as const

    it.each(['normalized', 'pixels', '0-1000'] as const)('accepts %s coordinates', coordinateSystem => {
      expectAccepted(computerUseCapabilitiesSchema, {
        ...baseCapabilities,
        coordinateSystem,
        stateContinuation: 'response-id'
      })
    })

    it.each(['client-history', 'response-id', 'provider-session'] as const)(
      'accepts %s continuation',
      stateContinuation => {
        expectAccepted(computerUseCapabilitiesSchema, {
          ...baseCapabilities,
          coordinateSystem: 'pixels',
          stateContinuation
        })
      }
    )

    it('requires literal true vision support', () => {
      expectRejected(computerUseCapabilitiesSchema, {
        ...baseCapabilities,
        vision: false,
        coordinateSystem: 'pixels',
        stateContinuation: 'response-id'
      })
    })

    it('rejects unknown capability keys', () => {
      expectRejected(computerUseCapabilitiesSchema, {
        ...baseCapabilities,
        coordinateSystem: 'pixels',
        stateContinuation: 'response-id',
        unexpected: true
      })
    })
  })

  describe('ComputerAction', () => {
    const dragPath = (length: number) =>
      Array.from({ length }, (_, index) => ({
        x: length === 1 ? 0 : index / (length - 1),
        y: length === 1 ? 0 : index / (length - 1)
      }))

    const validActions = [
      ['click', VALID_CLICK],
      ['double_click', { kind: 'double_click', point: { x: 0, y: 1 }, button: 'left' }],
      ['move', { kind: 'move', point: { x: 1, y: 0 } }],
      ['drag', { kind: 'drag', path: dragPath(2), button: 'left' }],
      ['scroll', { kind: 'scroll', point: { x: 0.5, y: 0.5 }, deltaX: -1, deltaY: 1 }],
      ['type_text', { kind: 'type_text', text: 'Schedule' }],
      ['press_key', { kind: 'press_key', keys: ['Control', 'Shift', 'Tab'] }],
      ['wait', { kind: 'wait', milliseconds: 1000 }],
      ['navigate_back', { kind: 'navigate_back' }],
      ['request_screenshot', { kind: 'request_screenshot' }],
      ['done', { kind: 'done', summary: 'Schedule is open' }],
      ['handoff', { kind: 'handoff', reason: 'provider_safety', severity: 'action_required' }]
    ]

    it.each(validActions)('accepts the %s action', (_kind, action) => {
      expectAccepted(computerActionSchema, action)
    })

    it.each(['left', 'middle', 'right'] as const)('accepts canonical click button %s', button => {
      expectAccepted(computerActionSchema, { ...VALID_CLICK, button })
    })

    it('accepts drag path boundaries', () => {
      expectAccepted(computerActionSchema, { kind: 'drag', path: dragPath(2), button: 'left' })
      expectAccepted(computerActionSchema, { kind: 'drag', path: dragPath(32), button: 'left' })
    })

    it.each([0, 1, 33])('rejects a drag path containing %i points', length => {
      expectRejected(computerActionSchema, { kind: 'drag', path: dragPath(length), button: 'left' })
    })

    it('requires left button for double-click and drag', () => {
      expectRejected(computerActionSchema, { kind: 'double_click', point: { x: 0.5, y: 0.5 }, button: 'right' })
      expectRejected(computerActionSchema, { kind: 'drag', path: dragPath(2), button: 'middle' })
    })

    it.each([
      ['minimum delta', { deltaX: -1, deltaY: 0 }],
      ['maximum delta', { deltaX: 1, deltaY: 0 }],
      ['vertical delta', { deltaX: 0, deltaY: 1 }]
    ])('accepts scroll at the %s boundary', (_name, deltas) => {
      expectAccepted(computerActionSchema, {
        kind: 'scroll',
        point: { x: 0.5, y: 0.5 },
        ...deltas
      })
    })

    it.each([
      ['both deltas zero', 0, 0],
      ['deltaX below range', -1 - Number.EPSILON, 0],
      ['deltaX above range', 1 + Number.EPSILON, 0],
      ['deltaY below range', 0, -1 - Number.EPSILON],
      ['deltaY above range', 0, 1 + Number.EPSILON],
      ['NaN delta', Number.NaN, 1],
      ['infinite delta', 0, Number.POSITIVE_INFINITY]
    ])('rejects scroll with %s', (_name, deltaX, deltaY) => {
      expectRejected(computerActionSchema, {
        kind: 'scroll',
        point: { x: 0.5, y: 0.5 },
        deltaX,
        deltaY
      })
    })

    it('counts Unicode scalar values rather than UTF-16 code units', () => {
      expectAccepted(computerActionSchema, { kind: 'type_text', text: '😀'.repeat(20_000) })
      expectRejected(computerActionSchema, { kind: 'type_text', text: '😀'.repeat(20_001) })
    })

    it('rejects empty text and unpaired UTF-16 surrogates', () => {
      expectRejected(computerActionSchema, { kind: 'type_text', text: '' })
      expectRejected(computerActionSchema, { kind: 'type_text', text: '\ud800' })
      expectRejected(computerActionSchema, { kind: 'type_text', text: '\udc00' })
    })

    it('accepts exactly the 17 case-sensitive canonical keys', () => {
      expect(computerKeys).toHaveLength(17)
      for (const key of computerKeys) {
        expectAccepted(computerActionSchema, { kind: 'press_key', keys: [key] })
      }
    })

    it('accepts one through four unique canonical keys', () => {
      expectAccepted(computerActionSchema, { kind: 'press_key', keys: ['Alt'] })
      expectAccepted(computerActionSchema, {
        kind: 'press_key',
        keys: ['Control', 'Shift', 'ArrowDown', 'Enter']
      })
    })

    it.each([
      ['no keys', []],
      ['five keys', ['Control', 'Shift', 'ArrowDown', 'ArrowUp', 'Enter']],
      ['duplicate keys', ['Control', 'Control']],
      ['unknown key', ['Space']],
      ['wrong-case key', ['enter']]
    ])('rejects press_key with %s', (_name, keys) => {
      expectRejected(computerActionSchema, { kind: 'press_key', keys })
    })

    it.each([100, 5000])('accepts wait boundary %i ms', milliseconds => {
      expectAccepted(computerActionSchema, { kind: 'wait', milliseconds })
    })

    it.each([99, 5001, Number.NaN, Number.POSITIVE_INFINITY])('rejects wait duration %p', milliseconds => {
      expectRejected(computerActionSchema, { kind: 'wait', milliseconds })
    })

    it('accepts empty and 1,000-character summaries but rejects 1,001 characters', () => {
      expectAccepted(computerActionSchema, { kind: 'done', summary: '' })
      expectAccepted(computerActionSchema, { kind: 'done', summary: 'x'.repeat(1000) })
      expectRejected(computerActionSchema, { kind: 'done', summary: 'x'.repeat(1001) })
    })

    it.each(['captcha', 'authentication', 'protected_control', 'uncertain', 'policy', 'provider_safety'])(
      'accepts handoff reason %s',
      reason => {
        expectAccepted(computerActionSchema, { kind: 'handoff', reason, severity: 'routine' })
      }
    )

    it.each(['routine', 'action_required', 'urgent'])('accepts handoff severity %s', severity => {
      expectAccepted(computerActionSchema, { kind: 'handoff', reason: 'uncertain', severity })
    })

    it('rejects unknown action kinds and missing variant fields', () => {
      expectRejected(computerActionSchema, { kind: 'screenshot' })
      expectRejected(computerActionSchema, { kind: 'click', button: 'left' })
    })
  })

  describe('ComputerActionToolArgs', () => {
    const exactContractExample = {
      schemaVersion: '1',
      actionId: 'a-7',
      action: VALID_CLICK,
      rationale: 'Schedule control'
    }

    it('accepts the exact canonical computer_action example', () => {
      expect(computerActionToolArgsSchema.parse(exactContractExample)).toEqual(exactContractExample)
    })

    it('accepts actionId and rationale boundaries', () => {
      expectAccepted(computerActionToolArgsSchema, {
        ...exactContractExample,
        actionId: 'a'.repeat(200),
        rationale: 'r'.repeat(1000)
      })
      expectAccepted(computerActionToolArgsSchema, { ...exactContractExample, rationale: '' })
    })

    it.each([
      ['missing actionId', undefined],
      ['empty actionId', ''],
      ['overlong actionId', 'a'.repeat(201)]
    ])('rejects %s', (_name, actionId) => {
      const value: Record<string, unknown> = { ...exactContractExample, actionId }
      if (actionId === undefined) delete value.actionId
      expectRejected(computerActionToolArgsSchema, value)
    })

    it('rejects an overlong rationale and wrong schema version', () => {
      expectRejected(computerActionToolArgsSchema, { ...exactContractExample, rationale: 'r'.repeat(1001) })
      expectRejected(computerActionToolArgsSchema, { ...exactContractExample, schemaVersion: '2' })
    })
  })

  describe('ComputerActionResult', () => {
    it('accepts the complete result and every optional field', () => {
      expectAccepted(computerActionResultSchema, {
        ...makeResult(),
        errorCode: 'NONE',
        message: 'Completed'
      })
    })

    it('accepts a minimal result and all status/verification literals', () => {
      for (const status of ['executed', 'rejected', 'failed', 'skipped']) {
        for (const verification of ['matched', 'mismatched', 'not_applicable']) {
          const { resultingUrl: _url, screenshotSha256: _screenshot, ...minimalResult } = makeResult()
          void _url
          void _screenshot
          expectAccepted(computerActionResultSchema, { ...minimalResult, status, verification })
        }
      }
    })

    it.each([
      ['short hash', 'a'.repeat(63)],
      ['long hash', 'a'.repeat(65)],
      ['uppercase hash', 'A'.repeat(64)],
      ['non-hex hash', 'g'.repeat(64)]
    ])('rejects %s', (_name, observationSha256) => {
      expectRejected(computerActionResultSchema, { ...makeResult(), observationSha256 })
    })

    it('rejects an invalid optional screenshot hash', () => {
      expectRejected(computerActionResultSchema, { ...makeResult(), screenshotSha256: 'not-a-hash' })
    })

    it('rejects empty and overlong result action IDs', () => {
      expectRejected(computerActionResultSchema, { ...makeResult(), actionId: '' })
      expectRejected(computerActionResultSchema, { ...makeResult(), actionId: 'a'.repeat(201) })
    })

    it.each([
      ['missing timezone', '2026-08-08T18:30:00'],
      ['non-UTC offset', '2026-08-08T18:30:00-06:00'],
      ['invalid date', '2026-13-40T25:61:61Z'],
      ['arbitrary text', 'today']
    ])('rejects %s timestamps', (_name, startedAt) => {
      expectRejected(computerActionResultSchema, { ...makeResult(), startedAt })
    })

    it('accepts RFC 3339 UTC Z and +00:00 timestamps', () => {
      expectAccepted(computerActionResultSchema, makeResult())
      expectAccepted(computerActionResultSchema, {
        ...makeResult(),
        startedAt: '2026-08-08T18:30:00.123+00:00'
      })
    })
  })

  describe('ComputerObservation, AuthorityContext, and ComputerUseTurn', () => {
    it.each(['image/png', 'image/jpeg'])('accepts fixed-size %s observations', mediaType => {
      expectAccepted(computerObservationSchema, {
        ...makeObservation(),
        screenshot: { ...makeObservation().screenshot, mediaType }
      })
    })

    it.each([
      ['wrong width', { width: 1279 }],
      ['wrong height', { height: 721 }],
      ['wrong media type', { mediaType: 'image/webp' }]
    ])('rejects a screenshot with %s', (_name, override) => {
      expectRejected(computerObservationSchema, {
        ...makeObservation(),
        screenshot: { ...makeObservation().screenshot, ...override }
      })
    })

    it('rejects invalid screenshot hashes and capturedAt timestamps', () => {
      expectRejected(computerObservationSchema, {
        ...makeObservation(),
        screenshot: { ...makeObservation().screenshot, sha256: 'A'.repeat(64) }
      })
      expectRejected(computerObservationSchema, { ...makeObservation(), capturedAt: '2026-08-08' })
    })

    it.each(['observe', 'draft', 'request_approval', 'execute_approved', 'execute_bounded'])(
      'accepts authority %s',
      authority => {
        expectAccepted(authorityContextSchema, { ...makeAuthority(), authority })
      }
    )

    it('accepts complete and minimal turns', () => {
      expectAccepted(computerUseTurnSchema, makeTurn())
      const { domSnapshot: _dom, accessibilityTree: _tree, ...minimalTurn } = makeTurn()
      void _dom
      void _tree
      expectAccepted(computerUseTurnSchema, minimalTurn)
    })

    it('requires the exact schema, application, and max-actions literals', () => {
      expectRejected(computerUseTurnSchema, { ...makeTurn(), schemaVersion: '2' })
      expectRejected(computerUseTurnSchema, { ...makeTurn(), activeApplication: 'firefox' })
      expectRejected(computerUseTurnSchema, { ...makeTurn(), maxActionsThisTurn: 2 })
    })
  })

  describe('ComputerUseDecision and pending safety checks', () => {
    const exactScrollDecision = {
      schemaVersion: '1',
      actionId: 'call_1',
      actions: [
        {
          kind: 'scroll',
          point: { x: 1, y: 1 },
          deltaX: -1,
          deltaY: 1
        }
      ],
      providerTraceId: 'cu_item_1',
      responseId: 'resp_1',
      pendingSafetyChecks: [{ id: 'safe_1', code: null, message: null }]
    }

    const exactScreenshotDecision = {
      schemaVersion: '1',
      actionId: 'call_2',
      actions: [{ kind: 'request_screenshot' }],
      providerTraceId: 'cu_item_2',
      responseId: 'resp_2',
      pendingSafetyChecks: []
    }

    it('accepts the exact normative scroll decision fixture', () => {
      expect(computerUseDecisionSchema.parse(exactScrollDecision)).toEqual(exactScrollDecision)
    })

    it('accepts the exact normative screenshot decision fixture', () => {
      expect(computerUseDecisionSchema.parse(exactScreenshotDecision)).toEqual(exactScreenshotDecision)
    })

    it('accepts string and null pending safety fields', () => {
      expectAccepted(pendingSafetyCheckSchema, { id: 'safe-1', code: 'policy', message: 'Review required' })
      expectAccepted(pendingSafetyCheckSchema, { id: 'safe-1', code: null, message: null })
    })

    it('requires explicit code and message fields in canonical safety checks', () => {
      expectRejected(pendingSafetyCheckSchema, { id: 'safe-1' })
      expectRejected(pendingSafetyCheckSchema, { id: 'safe-1', code: undefined, message: undefined })
    })

    it('rejects zero or multiple decision actions', () => {
      expectRejected(computerUseDecisionSchema, { ...exactScrollDecision, actions: [] })
      expectRejected(computerUseDecisionSchema, {
        ...exactScrollDecision,
        actions: [VALID_CLICK, { kind: 'request_screenshot' }]
      })
    })

    it.each([
      ['missing', undefined],
      ['empty', ''],
      ['over 200 characters', 'a'.repeat(201)]
    ])('rejects an actionId that is %s', (_name, actionId) => {
      const decision: Record<string, unknown> = { ...exactScrollDecision, actionId }
      if (actionId === undefined) delete decision.actionId
      expectRejected(computerUseDecisionSchema, decision)
    })
  })

  describe('strict unknown-key rejection', () => {
    it.each([
      ['point', normalizedPointSchema, { x: 0.5, y: 0.5, z: 0 }],
      ['action', computerActionSchema, { ...VALID_CLICK, unexpected: true }],
      [
        'nested action point',
        computerActionSchema,
        { ...VALID_CLICK, point: { ...VALID_CLICK.point, unexpected: true } }
      ],
      [
        'tool args',
        computerActionToolArgsSchema,
        { schemaVersion: '1', actionId: 'a-7', action: VALID_CLICK, rationale: '', unexpected: true }
      ],
      ['action result', computerActionResultSchema, { ...makeResult(), unexpected: true }],
      ['observation', computerObservationSchema, { ...makeObservation(), unexpected: true }],
      [
        'nested screenshot',
        computerObservationSchema,
        {
          ...makeObservation(),
          screenshot: { ...makeObservation().screenshot, unexpected: true }
        }
      ],
      ['authority', authorityContextSchema, { ...makeAuthority(), unexpected: true }],
      ['turn', computerUseTurnSchema, { ...makeTurn(), unexpected: true }],
      [
        'nested turn observation',
        computerUseTurnSchema,
        { ...makeTurn(), observation: { ...makeObservation(), unexpected: true } }
      ],
      [
        'nested turn authority',
        computerUseTurnSchema,
        { ...makeTurn(), authority: { ...makeAuthority(), unexpected: true } }
      ],
      [
        'nested turn result',
        computerUseTurnSchema,
        { ...makeTurn(), recentActions: [{ ...makeResult(), unexpected: true }] }
      ],
      [
        'decision',
        computerUseDecisionSchema,
        {
          schemaVersion: '1',
          actionId: 'action-1',
          actions: [VALID_CLICK],
          pendingSafetyChecks: [],
          unexpected: true
        }
      ],
      ['pending safety check', pendingSafetyCheckSchema, { id: 'safe-1', code: null, message: null, extra: true }]
    ])('rejects unknown keys in %s', (_name, schema, value) => {
      expectRejected(schema, value)
    })
  })
})
