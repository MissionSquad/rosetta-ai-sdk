import { z } from 'zod'
import { BaseCustomMapper } from '../../../../src/core/mapping/base.custom.mapper'
import { ToolArgumentValidationError } from '../../../../src/errors'

class FixtureMapper extends BaseCustomMapper {
  mapFixture(argumentsText: string) {
    return this.mapAndValidateToolCallsHelper(
      { calls: [{ id: 'call-1', name: 'open_session', arguments: argumentsText }] },
      ['calls'], ['id'], ['name'], ['arguments'],
      [{
        type: 'function',
        function: { name: 'open_session', parameters: { type: 'object' }, zodSchema: z.object({ sessionId: z.string() }) }
      }]
    )
  }
}

describe('BaseCustomMapper tool arguments', () => {
  it.each([
    ['{"session_id":"wrong-case"}', { session_id: 'wrong-case' }],
    ['{"sessionId":', '{"sessionId":']
  ])(
    'preserves invalid tool arguments on the typed error',
    (rawInput, rejectedInput) => {
      const mapper = new FixtureMapper({ providerKey: 'fixture', mapper: FixtureMapper, supportedFeatures: ['tool_use'] })
      let error: unknown
      try {
        mapper.mapFixture(rawInput)
      } catch (caught) {
        error = caught
      }
      expect(error).toBeInstanceOf(ToolArgumentValidationError)
      expect(error).toMatchObject({ toolName: 'open_session', toolCallId: 'call-1', receivedInput: rejectedInput })
    }
  )
})
