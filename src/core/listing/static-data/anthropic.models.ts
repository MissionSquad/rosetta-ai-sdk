// src/core/listing/static-data/anthropic.models.ts
import { RosettaModelList, RosettaModel, Provider } from '../../../types'

// Raw data matching the provided JSON structure, maybe use an intermediate raw type if needed.
// Placeholder structure based on mapping logic in the plan. Replace with actual JSON if available.
const rawAnthropicData = {
  object: 'list',
  data: [
    {
      id: 'claude-3-opus-20240229',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096,
      properties: {
        description: 'Most powerful model for highly complex tasks.',
        strengths: 'Complex analysis, math, coding, research, long-context understanding.',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'slowest',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'August 2023',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-3-sonnet-20240229',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096,
      properties: {
        description: 'Ideal balance of intelligence and speed for enterprise workloads.',
        strengths: 'Data processing, RAG, coding, quality control, content generation.',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'medium',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'August 2023',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-3-haiku-20240307',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096,
      properties: {
        description: 'Fastest, most compact model for near-instant responsiveness.',
        strengths: 'Customer interactions, content moderation, cost-saving tasks, logistics.',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'fastest',
        cost_input_mtok: 0.25,
        cost_output_mtok: 1.25,
        training_data_cutoff: 'August 2023',
        extended_max_completion_tokens: null
      }
    }
    // Add other Anthropic models here if known
  ]
}

// Statically type and process the raw data into the RosettaModelList format
export const anthropicStaticModels: RosettaModelList = {
  object: 'list',
  data: rawAnthropicData.data.map(
    (rawModel: any): RosettaModel => ({
      id: rawModel.id,
      object: 'model',
      owned_by: rawModel.owned_by,
      created: rawModel.created, // Will be null based on data
      active: rawModel.active,
      context_window: rawModel.context_window,
      public_apps: rawModel.public_apps,
      max_completion_tokens: rawModel.max_completion_tokens,
      properties: rawModel.properties
        ? {
            // Map properties safely
            description: rawModel.properties.description,
            strengths: rawModel.properties.strengths,
            multilingual: rawModel.properties.multilingual,
            vision: rawModel.properties.vision,
            extended_thinking: rawModel.properties.extended_thinking,
            comparative_latency: rawModel.properties.comparative_latency,
            cost_input_mtok: rawModel.properties.cost_input_mtok,
            cost_output_mtok: rawModel.properties.cost_output_mtok,
            training_data_cutoff: rawModel.properties.training_data_cutoff,
            extended_max_completion_tokens: rawModel.properties.extended_max_completion_tokens
          }
        : undefined,
      provider: Provider.Anthropic,
      rawData: rawModel // Store original
    })
  )
}
