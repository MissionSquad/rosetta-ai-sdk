import { anthropicStaticModels } from '../../../../../src/core/listing/static-data/anthropic.models'
import { RosettaModel, Provider } from '../../../../../src/types'

describe('Anthropic Static Models Data', () => {
  it('[Easy] should have the correct overall structure', () => {
    expect(anthropicStaticModels).toBeDefined()
    expect(anthropicStaticModels.object).toBe('list')
    expect(Array.isArray(anthropicStaticModels.data)).toBe(true)
    expect(anthropicStaticModels.data.length).toBeGreaterThan(0) // Ensure there's at least one model
  })

  it('[Easy] should contain valid RosettaModel objects', () => {
    const sampleModel = anthropicStaticModels.data[0] // Check the first model as a sample
    expect(sampleModel).toBeDefined()

    // Check required fields
    expect(typeof sampleModel.id).toBe('string')
    expect(sampleModel.id).not.toBe('')
    expect(sampleModel.object).toBe('model')
    expect(typeof sampleModel.owned_by).toBe('string')
    expect(sampleModel.owned_by).toBe('anthropic')
    expect(sampleModel.provider).toBe(Provider.Anthropic)

    // Check optional fields (types or undefined)
    expect(sampleModel.created === null || typeof sampleModel.created === 'number').toBe(true)
    expect(sampleModel.active === undefined || typeof sampleModel.active === 'boolean').toBe(true)
    expect(sampleModel.context_window === undefined || typeof sampleModel.context_window === 'number').toBe(true)
    expect(sampleModel.public_apps === undefined || sampleModel.public_apps === null).toBe(true) // Check null specifically
    expect(
      sampleModel.max_completion_tokens === undefined || typeof sampleModel.max_completion_tokens === 'number'
    ).toBe(true)

    // Check properties structure (if present)
    if (sampleModel.properties) {
      expect(typeof sampleModel.properties).toBe('object')
      expect(
        sampleModel.properties.description === undefined || typeof sampleModel.properties.description === 'string'
      ).toBe(true)
      expect(
        sampleModel.properties.multilingual === undefined || typeof sampleModel.properties.multilingual === 'boolean'
      ).toBe(true)
      expect(sampleModel.properties.vision === undefined || typeof sampleModel.properties.vision === 'boolean').toBe(
        true
      )
      // Add checks for other optional properties if needed
    }

    // Check rawData presence
    expect(sampleModel.rawData).toBeDefined()
    expect(typeof sampleModel.rawData).toBe('object')
    expect(sampleModel.rawData?.id).toBe(sampleModel.id) // Verify rawData contains original id
  })

  it('[Medium] should ensure all models have required fields and correct provider', () => {
    anthropicStaticModels.data.forEach((model: RosettaModel) => {
      expect(typeof model.id).toBe('string')
      expect(model.id).not.toBe('')
      expect(model.object).toBe('model')
      expect(typeof model.owned_by).toBe('string')
      expect(model.provider).toBe(Provider.Anthropic)
      expect(model.rawData).toBeDefined()
    })
  })

  it('[Medium] should ensure created is null or number', () => {
    anthropicStaticModels.data.forEach((model: RosettaModel) => {
      expect(model.created === null || typeof model.created === 'number').toBe(true)
    })
  })
})
