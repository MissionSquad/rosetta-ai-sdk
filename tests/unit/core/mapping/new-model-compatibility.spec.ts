import { z } from 'zod'

import { AnthropicMapper } from '../../../../src/core/mapping/anthropic.mapper'
import { OpenAIMapper } from '../../../../src/core/mapping/openai.mapper'
import { isGPT5Model, isOpenAIResponsesChatModel } from '../../../../src/core/mapping/openai.common'
import {
  mapToOpenAIResponsesChatParams,
  shouldUseOpenAIResponsesApi
} from '../../../../src/core/mapping/openai.responses.chat'
import { getGpt5Support } from '../../../../src/core/mapping/gpt5.support'
import { anthropicStaticModels } from '../../../../src/core/listing/static-data/anthropic.models'
import { UnsupportedFeatureError } from '../../../../src/errors'
import { GenerateParams, Provider } from '../../../../src/types'

/**
 * Regression matrix for the 2026-11 model families (Claude 5.1/5.2/5.5, GPT-6) alongside the
 * models that already worked. Expected values come from live probes (examples/probe-new-models.ts).
 */

const WEATHER_TOOL = {
  type: 'function' as const,
  function: {
    name: 'get_weather',
    description: 'Get current weather for a city',
    parameters: {
      type: 'object' as const,
      properties: { city: { type: 'string' } },
      required: ['city']
    },
    zodSchema: z.object({ city: z.string() })
  }
}

function anthropicParams(model: string, overrides: Partial<GenerateParams> = {}): GenerateParams {
  return { provider: Provider.Anthropic, model, messages: [{ role: 'user', content: 'Hello' }], ...overrides }
}

function openAIParams(model: string, overrides: Partial<GenerateParams> = {}): GenerateParams {
  return { provider: Provider.OpenAI, model, messages: [{ role: 'user', content: 'Hello' }], ...overrides }
}

describe('Anthropic mapper: new and existing model families', () => {
  const mapper = new AnthropicMapper()
  let warnSpy: jest.SpyInstance

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  afterEach(() => {
    warnSpy.mockRestore()
  })

  it.each(['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1', 'claude-fable-5-2'])(
    'strips temperature and top_p for %s',
    model => {
      const mapped = mapper.mapToProviderParams(anthropicParams(model, { temperature: 0.1, topP: 0.9 }))
      expect(mapped.temperature).toBeUndefined()
      expect(mapped.top_p).toBeUndefined()
    }
  )

  it.each(['claude-haiku-4-5', 'claude-sonnet-4-5', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-opus-4-6:1m'])(
    'keeps temperature for %s',
    model => {
      const mapped = mapper.mapToProviderParams(anthropicParams(model, { temperature: 0.1 }))
      expect(mapped.temperature).toBe(0.1)
    }
  )

  it('requests adaptive thinking on new models and budgeted thinking on pre-4.7 models', () => {
    expect(mapper.mapToProviderParams(anthropicParams('claude-haiku-5-5', { thinking: true })).thinking).toMatchObject({
      type: 'adaptive',
      display: 'summarized'
    })
    expect(mapper.mapToProviderParams(anthropicParams('claude-opus-5', { thinking: true })).thinking).toEqual({
      type: 'adaptive',
      display: 'summarized'
    })
    expect(mapper.mapToProviderParams(anthropicParams('claude-haiku-4-5', { thinking: true })).thinking).toEqual({
      type: 'enabled',
      budget_tokens: 1024
    })
  })

  it.each(['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1', 'claude-fable-5-2:1m'])(
    'asks %s to drop thinking blocks that fail the history check, with or without a thinking request',
    model => {
      const withoutThinking = mapper.mapToProviderParams(anthropicParams(model))
      expect(withoutThinking.thinking).toEqual({
        type: 'adaptive',
        block_binding: { prefix_mismatch_behavior: 'drop_block' }
      })

      const withThinking = mapper.mapToProviderParams(anthropicParams(model, { thinking: true }))
      expect(withThinking.thinking).toEqual({
        type: 'adaptive',
        display: 'summarized',
        block_binding: { prefix_mismatch_behavior: 'drop_block' }
      })
    }
  )

  it.each([
    'claude-haiku-4-5',
    'claude-sonnet-4-6',
    'claude-opus-4-8',
    'claude-sonnet-5',
    'claude-opus-5',
    'claude-fable-5'
  ])('sends no thinking config to %s unless thinking is requested', model => {
    // block_binding is rejected by models that do not run the check (400 on claude-haiku-4-5).
    expect(mapper.mapToProviderParams(anthropicParams(model)).thinking).toBeUndefined()
  })

  it.each(['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-fable-5-1', 'claude-fable-5-2'])(
    'downgrades forced tool choice to auto for %s',
    model => {
      const required = mapper.mapToProviderParams(
        anthropicParams(model, { tools: [WEATHER_TOOL], toolChoice: 'required' })
      )
      expect(required.tool_choice).toEqual({ type: 'auto' })

      const named = mapper.mapToProviderParams(
        anthropicParams(model, {
          tools: [WEATHER_TOOL],
          toolChoice: { type: 'function', function: { name: 'get_weather' } }
        })
      )
      expect(named.tool_choice).toEqual({ type: 'auto' })
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('does not support forced tool choice'))
    }
  )

  it.each(['claude-haiku-4-5', 'claude-sonnet-4-6', 'claude-opus-5', 'claude-fable-5', 'claude-haiku-5-5'])(
    'keeps forced tool choice for %s',
    model => {
      const required = mapper.mapToProviderParams(
        anthropicParams(model, { tools: [WEATHER_TOOL], toolChoice: 'required' })
      )
      expect(required.tool_choice).toEqual({ type: 'any' })

      const named = mapper.mapToProviderParams(
        anthropicParams(model, {
          tools: [WEATHER_TOOL],
          toolChoice: { type: 'function', function: { name: 'get_weather' } }
        })
      )
      expect(named.tool_choice).toEqual({ type: 'tool', name: 'get_weather' })
    }
  )

  it('leaves tool_choice auto untouched on every family', () => {
    for (const model of ['claude-haiku-4-5', 'claude-opus-5', 'claude-fable-5-2']) {
      const mapped = mapper.mapToProviderParams(anthropicParams(model, { tools: [WEATHER_TOOL], toolChoice: 'auto' }))
      expect(mapped.tool_choice).toEqual({ type: 'auto' })
    }
    expect(warnSpy).not.toHaveBeenCalledWith(expect.stringContaining('forced tool choice'))
  })

  it('lists the new models with 1M context and 128K output', () => {
    for (const id of [
      'claude-fable-5-2',
      'claude-fable-5-1',
      'claude-opus-5-5',
      'claude-sonnet-5-5',
      'claude-haiku-5-5'
    ]) {
      const model = anthropicStaticModels.data.find(entry => entry.id === id)
      expect(model).toBeDefined()
      expect(model?.context_window).toBe(1000000)
      expect(model?.max_completion_tokens).toBe(128000)
      expect(model?.properties?.extended_thinking).toBe(true)
    }
  })

  it('still lists the models that were already present', () => {
    for (const id of [
      'claude-fable-5',
      'claude-opus-5',
      'claude-sonnet-5',
      'claude-opus-4-6',
      'claude-sonnet-4-6:1m',
      'claude-haiku-4-5'
    ]) {
      expect(anthropicStaticModels.data.some(entry => entry.id === id)).toBe(true)
    }
  })
})

describe('OpenAI: Responses routing gate', () => {
  it.each([
    'gpt-5',
    'gpt-5-mini',
    'gpt-5.1',
    'gpt-5.4',
    'gpt-5.6-sol',
    'gpt-6-sol',
    'gpt-6-luna',
    'gpt-6-astra',
    'gpt-6.1-sol',
    'gpt-6.1-terra'
  ])('routes %s through the Responses API', model => {
    expect(isOpenAIResponsesChatModel(model)).toBe(true)
    expect(isGPT5Model(model)).toBe(true)
    expect(shouldUseOpenAIResponsesApi(openAIParams(model))).toBe(true)
  })

  it.each(['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-35-turbo', 'gpt-3.5-turbo', 'o3', 'o4-mini'])(
    'keeps %s on Chat Completions',
    model => {
      expect(isOpenAIResponsesChatModel(model)).toBe(false)
      expect(shouldUseOpenAIResponsesApi(openAIParams(model))).toBe(false)
    }
  )

  it('honours the Chat Completions escape hatch for GPT-6', () => {
    expect(
      shouldUseOpenAIResponsesApi(openAIParams('gpt-6-sol', { providerOptions: { openaiPreferChatCompletions: true } }))
    ).toBe(false)
  })
})

describe('OpenAI: GPT-6 capability table', () => {
  it.each(['gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra'])(
    '%s accepts effort none and sampling only alongside it',
    model => {
      expect(getGpt5Support(model)).toMatchObject({
        allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        defaultReasoningEffort: 'medium',
        supportsSampling: 'only_with_reasoning_none',
        supportsVerbosity: true,
        functionToolsRequireExplicitNone: true
      })
    }
  )

  it.each(['gpt-6-astra', 'gpt-6.1-sol'])('%s rejects effort none and never accepts sampling', model => {
    const support = getGpt5Support(model)
    expect(support).toMatchObject({
      allowedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
      defaultReasoningEffort: 'medium',
      supportsSampling: 'never'
    })
    expect(support?.functionToolsRequireExplicitNone).toBeUndefined()
  })

  it('does not let gpt-6.1-sol inherit the gpt-6-sol row (or the reverse)', () => {
    expect(getGpt5Support('gpt-6.1-sol')?.allowedReasoningEfforts).not.toContain('none')
    expect(getGpt5Support('gpt-6-sol')?.allowedReasoningEfforts).toContain('none')
    expect(getGpt5Support('gpt-6-sol-2026-10-01')?.allowedReasoningEfforts).toContain('none')
  })

  it('leaves the older GPT-5 rows unchanged', () => {
    expect(getGpt5Support('gpt-5')).toMatchObject({
      allowedReasoningEfforts: ['minimal', 'low', 'medium', 'high'],
      defaultReasoningEffort: 'medium',
      supportsSampling: 'never'
    })
    expect(getGpt5Support('gpt-5.4')).toMatchObject({
      allowedReasoningEfforts: ['none', 'low', 'medium', 'high', 'xhigh'],
      defaultReasoningEffort: 'none',
      supportsSampling: 'never'
    })
    expect(getGpt5Support('gpt-5.2')?.supportsSampling).toBe('only_with_reasoning_none')
  })
})

describe('OpenAI: Responses request mapping for GPT-6', () => {
  it.each(['gpt-6-sol', 'gpt-6-luna', 'gpt-6-astra', 'gpt-6.1-sol', 'gpt-6.1-terra'])(
    'drops temperature for %s at default effort (the add-model test shape)',
    model => {
      const mapped = mapToOpenAIResponsesChatParams(openAIParams(model, { temperature: 0.1, topP: 0.9, maxTokens: 10 }))
      expect(mapped).not.toHaveProperty('temperature')
      expect(mapped).not.toHaveProperty('top_p')
      expect(mapped).not.toHaveProperty('reasoning')
      expect(mapped.max_output_tokens).toBe(16)
    }
  )

  it('keeps temperature for gpt-6-sol only with an explicit effort of none', () => {
    const mapped = mapToOpenAIResponsesChatParams(
      openAIParams('gpt-6-sol', { temperature: 0.1, reasoningEffort: 'none' })
    )
    expect(mapped.temperature).toBe(0.1)
    expect(mapped.reasoning).toEqual({ effort: 'none' })
  })

  it('replaces an unsupported effort of none with the model default on gpt-6-astra and drops temperature', () => {
    const mapped = mapToOpenAIResponsesChatParams(
      openAIParams('gpt-6-astra', { temperature: 0.1, reasoningEffort: 'none' })
    )
    expect(mapped.reasoning).toEqual({ effort: 'medium' })
    expect(mapped).not.toHaveProperty('temperature')
  })

  it('replaces minimal (rejected by every GPT-6 model) with the model default', () => {
    expect(
      mapToOpenAIResponsesChatParams(openAIParams('gpt-6.1-sol', { reasoningEffort: 'minimal' })).reasoning
    ).toEqual({
      effort: 'medium'
    })
  })

  it('passes effort max through for GPT-6 and gpt-5.6', () => {
    for (const model of ['gpt-6-sol', 'gpt-6-astra', 'gpt-6.1-sol', 'gpt-5.6-sol']) {
      expect(mapToOpenAIResponsesChatParams(openAIParams(model, { reasoningEffort: 'max' })).reasoning).toEqual({
        effort: 'max'
      })
    }
  })

  it('combines function tools with reasoning and a summary when thinking is requested', () => {
    const mapped = mapToOpenAIResponsesChatParams(
      openAIParams('gpt-6.1-sol', {
        tools: [WEATHER_TOOL],
        toolChoice: 'auto',
        thinking: true,
        reasoningEffort: 'high'
      })
    )
    expect(mapped.tools).toHaveLength(1)
    expect(mapped.reasoning).toEqual({ effort: 'high', summary: 'auto' })
  })

  it('keeps existing behaviour for gpt-5 (minimal allowed, max replaced by its default)', () => {
    expect(mapToOpenAIResponsesChatParams(openAIParams('gpt-5', { reasoningEffort: 'minimal' })).reasoning).toEqual({
      effort: 'minimal'
    })
    expect(mapToOpenAIResponsesChatParams(openAIParams('gpt-5', { reasoningEffort: 'max' })).reasoning).toEqual({
      effort: 'medium'
    })
  })
})

describe('OpenAI: Chat Completions path for GPT-6 (escape hatch)', () => {
  const mapper = new OpenAIMapper()

  it('drops temperature at default effort', () => {
    const mapped = mapper.mapToProviderParams(openAIParams('gpt-6-sol', { temperature: 0.1 }))
    expect(mapped.temperature).toBeUndefined()
  })

  it('sends an explicit effort of none to attach function tools where none is accepted', () => {
    const mapped = mapper.mapToProviderParams(
      openAIParams('gpt-6-sol', { tools: [WEATHER_TOOL], reasoningEffort: 'high' })
    )
    expect(mapped.reasoning_effort).toBe('none')
  })

  it.each(['gpt-6-astra', 'gpt-6.1-sol'])(
    'rejects function tools for %s, which cannot take them on this API',
    model => {
      expect(() => mapper.mapToProviderParams(openAIParams(model, { tools: [WEATHER_TOOL] }))).toThrow(
        UnsupportedFeatureError
      )
    }
  )

  it('clamps effort max to xhigh, the highest Chat Completions accepts', () => {
    expect(mapper.mapToProviderParams(openAIParams('gpt-6-astra', { reasoningEffort: 'max' })).reasoning_effort).toBe(
      'xhigh'
    )
    expect(mapper.mapToProviderParams(openAIParams('gpt-5.6-sol', { reasoningEffort: 'max' })).reasoning_effort).toBe(
      'xhigh'
    )
  })

  it('leaves gpt-4o untouched (temperature and token limit forwarded)', () => {
    const mapped = mapper.mapToProviderParams(openAIParams('gpt-4o', { temperature: 0.1, maxTokens: 10 }))
    expect(mapped.temperature).toBe(0.1)
    expect(mapped.max_completion_tokens).toBe(10)
    expect(mapped).not.toHaveProperty('reasoning_effort')
  })
})
