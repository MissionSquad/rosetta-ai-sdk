import { ComputerAction as OpenAIComputerAction } from 'openai/resources/responses/responses'
import {
  ComputerAction,
  ComputerKey,
  ComputerUseCapabilities,
  NormalizedPoint,
  computerActionSchema,
  computerKeySchema,
  normalizedPointSchema
} from '../../types/computer-use.types'
import { Provider } from '../../types/common.types'
import { ComputerUseMappingError } from '../../errors'

export interface ProviderCoordinateDimensions {
  width: number
  height: number
}

type ProviderCoordinateSystem = ComputerUseCapabilities['coordinateSystem']
type CoordinateAxis = 'x' | 'y'

const PROVIDER_KEY_ALIASES: Readonly<Record<string, ComputerKey>> = {
  ALT: 'Alt',
  OPTION: 'Alt',
  ARROWDOWN: 'ArrowDown',
  DOWN: 'ArrowDown',
  ARROWLEFT: 'ArrowLeft',
  LEFT: 'ArrowLeft',
  ARROWRIGHT: 'ArrowRight',
  RIGHT: 'ArrowRight',
  ARROWUP: 'ArrowUp',
  UP: 'ArrowUp',
  BACKSPACE: 'Backspace',
  CONTROL: 'Control',
  CTRL: 'Control',
  DELETE: 'Delete',
  DEL: 'Delete',
  END: 'End',
  ENTER: 'Enter',
  RETURN: 'Enter',
  ESC: 'Escape',
  ESCAPE: 'Escape',
  HOME: 'Home',
  META: 'Meta',
  CMD: 'Meta',
  COMMAND: 'Meta',
  PAGEDOWN: 'PageDown',
  PAGEUP: 'PageUp',
  SHIFT: 'Shift',
  TAB: 'Tab'
}

function invalid(message: string, cause?: unknown): ComputerUseMappingError {
  return new ComputerUseMappingError('PROVIDER_ACTION_INVALID', message, Provider.OpenAI, cause)
}

function axisSize(axis: CoordinateAxis, dimensions: ProviderCoordinateDimensions): number {
  const size = axis === 'x' ? dimensions.width : dimensions.height
  if (!Number.isFinite(size) || size <= 1) {
    throw invalid(`Provider ${axis}-axis size must be finite and greater than one`)
  }
  return size
}

function normalizeProviderCoordinate(
  value: number,
  axis: CoordinateAxis,
  coordinateSystem: ProviderCoordinateSystem,
  dimensions: ProviderCoordinateDimensions,
  delta: boolean
): number {
  if (!Number.isFinite(value)) throw invalid(`Provider ${axis} value must be finite`)

  let normalized: number
  if (coordinateSystem === 'pixels') {
    const maximum = axisSize(axis, dimensions) - 1
    const minimum = delta ? -maximum : 0
    if (value < minimum || value > maximum) {
      throw invalid(`Provider pixel ${axis} value ${value} is outside ${minimum}..${maximum}`)
    }
    normalized = value / maximum
  } else if (coordinateSystem === '0-1000') {
    const minimum = delta ? -1000 : 0
    if (value < minimum || value > 1000) {
      throw invalid(`Provider 0-1000 ${axis} value ${value} is outside ${minimum}..1000`)
    }
    normalized = value / 1000
  } else {
    const minimum = delta ? -1 : 0
    if (value < minimum || value > 1) {
      throw invalid(`Provider normalized ${axis} value ${value} is outside ${minimum}..1`)
    }
    normalized = value
  }

  return normalized
}

/**
 * Converts a provider point to the canonical inclusive 0–1 coordinate space.
 *
 * @throws {ComputerUseMappingError} If dimensions or coordinate values are outside the contract.
 */
export function normalizeProviderPoint(
  x: number,
  y: number,
  coordinateSystem: ProviderCoordinateSystem,
  dimensions: ProviderCoordinateDimensions
): NormalizedPoint {
  const parsed = normalizedPointSchema.safeParse({
    x: normalizeProviderCoordinate(x, 'x', coordinateSystem, dimensions, false),
    y: normalizeProviderCoordinate(y, 'y', coordinateSystem, dimensions, false)
  })
  if (!parsed.success) throw invalid('Normalized provider point failed canonical validation', parsed.error)
  return parsed.data
}

/**
 * Converts one provider signed delta using the corresponding axis divisor.
 *
 * @throws {ComputerUseMappingError} If dimensions or the delta are outside the contract.
 */
export function normalizeProviderDelta(
  value: number,
  axis: CoordinateAxis,
  coordinateSystem: ProviderCoordinateSystem,
  dimensions: ProviderCoordinateDimensions
): number {
  return normalizeProviderCoordinate(value, axis, coordinateSystem, dimensions, true)
}

/**
 * Applies the contract's closed provider-key alias table and post-normalization uniqueness rule.
 *
 * @throws {ComputerUseMappingError} If a key is unsupported or the normalized combination is invalid.
 */
export function normalizeProviderKeys(keys: readonly string[]): ComputerKey[] {
  if (keys.length < 1 || keys.length > 4) throw invalid('Provider keypress must contain one to four keys')

  const normalized = keys.map(key => PROVIDER_KEY_ALIASES[key.toUpperCase()])
  if (normalized.some(key => key === undefined)) throw invalid('Provider keypress contains an unsupported key')

  const parsed = computerKeySchema
    .array()
    .min(1)
    .max(4)
    .safeParse(normalized)
  if (!parsed.success) throw invalid('Provider keypress failed canonical validation', parsed.error)
  if (new Set(parsed.data).size !== parsed.data.length) {
    throw invalid('Provider keypress contains duplicate keys after alias normalization')
  }
  return parsed.data
}

function isMouseAction(action: OpenAIComputerAction): boolean {
  return ['click', 'double_click', 'move', 'drag', 'scroll'].includes(action.type)
}

function assertNoMouseModifiers(action: OpenAIComputerAction): void {
  if (isMouseAction(action) && Object.prototype.hasOwnProperty.call(action, 'keys')) {
    throw new ComputerUseMappingError(
      'PROVIDER_ACTION_MODIFIERS_UNSUPPORTED',
      `OpenAI ${action.type} action contains unsupported mouse modifiers`,
      Provider.OpenAI
    )
  }
}

/**
 * Maps one installed OpenAI GA computer action to the closed canonical action union.
 *
 * @throws {ComputerUseMappingError} If provider semantics cannot map exactly to the V1 contract.
 */
export function mapOpenAIComputerAction(
  action: OpenAIComputerAction,
  coordinateSystem: ProviderCoordinateSystem,
  dimensions: ProviderCoordinateDimensions
): ComputerAction {
  assertNoMouseModifiers(action)

  let mapped: unknown
  switch (action.type) {
    case 'click':
      if (action.button !== 'left' && action.button !== 'right') {
        throw new ComputerUseMappingError(
          'PROVIDER_ACTION_UNSUPPORTED',
          `OpenAI click button '${action.button}' has no canonical V1 mapping`,
          Provider.OpenAI
        )
      }
      mapped = {
        kind: 'click',
        point: normalizeProviderPoint(action.x, action.y, coordinateSystem, dimensions),
        button: action.button
      }
      break
    case 'double_click':
      mapped = {
        kind: 'double_click',
        point: normalizeProviderPoint(action.x, action.y, coordinateSystem, dimensions),
        button: 'left'
      }
      break
    case 'move':
      mapped = {
        kind: 'move',
        point: normalizeProviderPoint(action.x, action.y, coordinateSystem, dimensions)
      }
      break
    case 'drag':
      if (!Array.isArray(action.path)) throw invalid('OpenAI drag path must be an array')
      mapped = {
        kind: 'drag',
        path: action.path.map(point => normalizeProviderPoint(point.x, point.y, coordinateSystem, dimensions)),
        button: 'left'
      }
      break
    case 'scroll': {
      const hasX = typeof action.x !== 'undefined'
      const hasY = typeof action.y !== 'undefined'
      if (hasX !== hasY) throw invalid('OpenAI scroll must provide both point coordinates or neither')
      mapped = {
        kind: 'scroll',
        point:
          hasX && hasY ? normalizeProviderPoint(action.x, action.y, coordinateSystem, dimensions) : { x: 0.5, y: 0.5 },
        deltaX: normalizeProviderDelta(action.scroll_x, 'x', coordinateSystem, dimensions),
        deltaY: normalizeProviderDelta(action.scroll_y, 'y', coordinateSystem, dimensions)
      }
      break
    }
    case 'keypress':
      if (!Array.isArray(action.keys) || action.keys.some(key => typeof key !== 'string')) {
        throw invalid('OpenAI keypress keys must be an array of strings')
      }
      mapped = { kind: 'press_key', keys: normalizeProviderKeys(action.keys) }
      break
    case 'type':
      mapped = { kind: 'type_text', text: action.text }
      break
    case 'wait':
      mapped = { kind: 'wait', milliseconds: 1000 }
      break
    case 'screenshot':
      mapped = { kind: 'request_screenshot' }
      break
    default:
      throw new ComputerUseMappingError(
        'PROVIDER_ACTION_UNSUPPORTED',
        'OpenAI returned an unknown computer action',
        Provider.OpenAI
      )
  }

  const parsed = computerActionSchema.safeParse(mapped)
  if (!parsed.success) throw invalid(`OpenAI ${action.type} action failed canonical validation`, parsed.error)
  return parsed.data
}
