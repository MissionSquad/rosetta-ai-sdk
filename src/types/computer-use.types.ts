import { z } from 'zod'

export const COMPUTER_USE_SCHEMA_VERSION = '1' as const
export const COMPUTER_USE_VIEWPORT_WIDTH = 1280 as const
export const COMPUTER_USE_VIEWPORT_HEIGHT = 720 as const

export const computerActionKinds = [
  'click',
  'double_click',
  'move',
  'drag',
  'scroll',
  'type_text',
  'press_key',
  'wait',
  'navigate_back',
  'request_screenshot',
  'done',
  'handoff'
] as const

export const computerKeys = [
  'Alt',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'Backspace',
  'Control',
  'Delete',
  'End',
  'Enter',
  'Escape',
  'Home',
  'Meta',
  'PageDown',
  'PageUp',
  'Shift',
  'Tab'
] as const

const actionIdSchema = z
  .string()
  .min(1)
  .max(200)
const sha256Schema = z.string().regex(/^[0-9a-f]{64}$/)
const utcTimestampSchema = z
  .string()
  .datetime({ offset: true })
  .refine(value => /(?:Z|\+00:00)$/.test(value), 'Timestamp must be RFC 3339 UTC')

function countUnicodeScalars(value: string): number | null {
  let count = 0
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index)
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (!Number.isFinite(next) || next < 0xdc00 || next > 0xdfff) return null
      index += 1
    } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
      return null
    }
    count += 1
  }
  return count
}

const computerTextSchema = z.string().superRefine((value, context) => {
  const scalarCount = countUnicodeScalars(value)
  if (scalarCount === null) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: 'Text must contain only Unicode scalar values' })
  } else if (scalarCount < 1 || scalarCount > 20_000) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: 'Text must contain 1 to 20,000 Unicode scalar values' })
  }
})

/** A finite point in the canonical inclusive 0–1 coordinate space. */
export const normalizedPointSchema = z
  .object({
    x: z
      .number()
      .finite()
      .min(0)
      .max(1),
    y: z
      .number()
      .finite()
      .min(0)
      .max(1)
  })
  .strict()

export type NormalizedPoint = z.infer<typeof normalizedPointSchema>

export const mouseButtonSchema = z.enum(['left', 'middle', 'right'])
export type MouseButton = z.infer<typeof mouseButtonSchema>

export const computerActionKindSchema = z.enum(computerActionKinds)
export type ComputerActionKind = z.infer<typeof computerActionKindSchema>

export const computerKeySchema = z.enum(computerKeys)
export type ComputerKey = z.infer<typeof computerKeySchema>

/** Provider capabilities used to choose the native or forced-tool computer-use path. */
export const computerUseCapabilitiesSchema = z
  .object({
    vision: z.literal(true),
    nativeComputerTool: z.boolean(),
    structuredToolCalls: z.boolean(),
    batchedActions: z.boolean(),
    acceptsDomSnapshot: z.boolean(),
    acceptsAccessibilityTree: z.boolean(),
    coordinateSystem: z.enum(['normalized', 'pixels', '0-1000']),
    stateContinuation: z.enum(['client-history', 'response-id', 'provider-session'])
  })
  .strict()

export type ComputerUseCapabilities = z.infer<typeof computerUseCapabilitiesSchema>

const clickActionSchema = z
  .object({ kind: z.literal('click'), point: normalizedPointSchema, button: mouseButtonSchema })
  .strict()
const doubleClickActionSchema = z
  .object({ kind: z.literal('double_click'), point: normalizedPointSchema, button: z.literal('left') })
  .strict()
const moveActionSchema = z.object({ kind: z.literal('move'), point: normalizedPointSchema }).strict()
const dragPathSchema = z
  .tuple([normalizedPointSchema, normalizedPointSchema])
  .rest(normalizedPointSchema)
  .refine(path => path.length <= 32, 'Drag path cannot contain more than 32 points')
const dragActionSchema = z
  .object({
    kind: z.literal('drag'),
    path: dragPathSchema,
    button: z.literal('left')
  })
  .strict()
const scrollActionSchema = z
  .object({
    kind: z.literal('scroll'),
    point: normalizedPointSchema,
    deltaX: z
      .number()
      .finite()
      .min(-1)
      .max(1),
    deltaY: z
      .number()
      .finite()
      .min(-1)
      .max(1)
  })
  .strict()
const typeTextActionSchema = z.object({ kind: z.literal('type_text'), text: computerTextSchema }).strict()
const pressKeyActionSchema = z
  .object({
    kind: z.literal('press_key'),
    keys: z
      .array(computerKeySchema)
      .min(1)
      .max(4)
      .refine(keys => new Set(keys).size === keys.length, 'Keys must be unique')
  })
  .strict()
const waitActionSchema = z
  .object({
    kind: z.literal('wait'),
    milliseconds: z
      .number()
      .finite()
      .min(100)
      .max(5000)
  })
  .strict()
const navigateBackActionSchema = z.object({ kind: z.literal('navigate_back') }).strict()
const requestScreenshotActionSchema = z.object({ kind: z.literal('request_screenshot') }).strict()
const doneActionSchema = z.object({ kind: z.literal('done'), summary: z.string().max(1000) }).strict()
const handoffActionSchema = z
  .object({
    kind: z.literal('handoff'),
    reason: z.enum(['captcha', 'authentication', 'protected_control', 'uncertain', 'policy', 'provider_safety']),
    severity: z.enum(['routine', 'action_required', 'urgent'])
  })
  .strict()

const discriminatedComputerActionSchema = z.discriminatedUnion('kind', [
  clickActionSchema,
  doubleClickActionSchema,
  moveActionSchema,
  dragActionSchema,
  scrollActionSchema,
  typeTextActionSchema,
  pressKeyActionSchema,
  waitActionSchema,
  navigateBackActionSchema,
  requestScreenshotActionSchema,
  doneActionSchema,
  handoffActionSchema
])

/** The provider-neutral V1 action union. */
export const computerActionSchema = discriminatedComputerActionSchema.superRefine((action, context) => {
  if (action.kind === 'scroll' && action.deltaX === 0 && action.deltaY === 0) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: 'Scroll deltas cannot both be zero' })
  }
})

export type ComputerAction = z.infer<typeof computerActionSchema>

/** Arguments of the standard forced `computer_action` Rosetta function tool. */
export const computerActionToolArgsSchema = z
  .object({
    schemaVersion: z.literal(COMPUTER_USE_SCHEMA_VERSION),
    actionId: actionIdSchema,
    action: computerActionSchema,
    rationale: z.string().max(1000)
  })
  .strict()

export type ComputerActionToolArgs = z.infer<typeof computerActionToolArgsSchema>

/** Auditable result returned after a worker handles one canonical action. */
export const computerActionResultSchema = z
  .object({
    schemaVersion: z.literal(COMPUTER_USE_SCHEMA_VERSION),
    actionId: actionIdSchema,
    action: computerActionSchema,
    observationSha256: sha256Schema,
    startedAt: utcTimestampSchema,
    completedAt: utcTimestampSchema,
    status: z.enum(['executed', 'rejected', 'failed', 'skipped']),
    verification: z.enum(['matched', 'mismatched', 'not_applicable']),
    errorCode: z.string().optional(),
    message: z.string().optional(),
    resultingUrl: z.string().optional(),
    screenshotSha256: sha256Schema.optional()
  })
  .strict()

export type ComputerActionResult = z.infer<typeof computerActionResultSchema>

const computerScreenshotSchema = z
  .object({
    mediaType: z.enum(['image/png', 'image/jpeg']),
    data: z.string(),
    width: z.literal(COMPUTER_USE_VIEWPORT_WIDTH),
    height: z.literal(COMPUTER_USE_VIEWPORT_HEIGHT),
    sha256: sha256Schema
  })
  .strict()

/** A fresh, fixed-size screenshot observation and its browser-control identity. */
export const computerObservationSchema = z
  .object({
    schemaVersion: z.literal(COMPUTER_USE_SCHEMA_VERSION),
    observationId: z.string(),
    screenshot: computerScreenshotSchema,
    url: z.string(),
    capturedAt: utcTimestampSchema,
    browserSessionId: z.string(),
    controlEpoch: z.number().finite(),
    allowedActionKinds: z.array(computerActionKindSchema),
    goal: z.string()
  })
  .strict()

export type ComputerObservation = z.infer<typeof computerObservationSchema>

export const authorityContextSchema = z
  .object({
    commsAgentId: z.string(),
    authority: z.enum(['observe', 'draft', 'request_approval', 'execute_approved', 'execute_bounded']),
    allowedDomains: z.array(z.string()),
    approvalId: z.string().optional(),
    approvalHash: z.string().optional(),
    emergencyStopVersion: z.number().finite()
  })
  .strict()

export type AuthorityContext = z.infer<typeof authorityContextSchema>

/** One computer-use decision request, limited to one action. */
export const computerUseTurnSchema = z
  .object({
    schemaVersion: z.literal(COMPUTER_USE_SCHEMA_VERSION),
    taskId: z.string(),
    goal: z.string(),
    step: z.number().finite(),
    observation: computerObservationSchema,
    activeApplication: z.literal('chromium'),
    domSnapshot: z.string().optional(),
    accessibilityTree: z.string().optional(),
    recentActions: z.array(computerActionResultSchema),
    allowedActions: z.array(computerActionKindSchema),
    authority: authorityContextSchema,
    maxActionsThisTurn: z.literal(1)
  })
  .strict()

export type ComputerUseTurn = z.infer<typeof computerUseTurnSchema>

export const pendingSafetyCheckSchema = z
  .object({
    id: z.string(),
    code: z.string().nullable(),
    message: z.string().nullable()
  })
  .strict()

export type PendingSafetyCheck = z.infer<typeof pendingSafetyCheckSchema>

/** Canonical decision produced by a native or forced-tool provider adapter. */
export const computerUseDecisionSchema = z
  .object({
    schemaVersion: z.literal(COMPUTER_USE_SCHEMA_VERSION),
    actionId: actionIdSchema,
    actions: z.tuple([computerActionSchema]),
    providerTraceId: z.string().optional(),
    responseId: z.string().optional(),
    pendingSafetyChecks: z.array(pendingSafetyCheckSchema)
  })
  .strict()

export type ComputerUseDecision = z.infer<typeof computerUseDecisionSchema>

/** Provider-neutral computer-use model contract. */
export interface ComputerUseModel {
  readonly id: string
  readonly capabilities: ComputerUseCapabilities
  decide(turn: ComputerUseTurn): Promise<ComputerUseDecision>
}
