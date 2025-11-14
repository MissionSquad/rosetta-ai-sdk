// src/core/listing/static-data/anthropic.models.ts
import { RosettaModelList, RosettaModel, Provider } from '../../../types'

// Raw data matching the provided JSON structure, maybe use an intermediate raw type if needed.
// Placeholder structure based on mapping logic in the plan. Replace with actual JSON if available.
const rawAnthropicData = {
  object: 'list',
  data: [
    // --- Claude 4.5 Haiku ---
    {
      id: 'claude-haiku-4-5-20251001',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our fastest model with near-frontier intelligence',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 1.0,
        cost_output_mtok: 5.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-haiku-4-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our fastest model with near-frontier intelligence (alias)',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 1.0,
        cost_output_mtok: 5.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 4.5 Sonnet ---
    {
      id: 'claude-sonnet-4-5-20250929',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our smartest model for complex agents and coding',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-sonnet-4-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our smartest model for complex agents and coding (alias)',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 4.5 Sonnet (1M Context) ---
    {
      id: 'claude-sonnet-4-5-20250929:1m',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our smartest model for complex agents and coding (1M context)',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null,
        beta_features: ['context-1m-2025-08-07']
      }
    },
    {
      id: 'claude-sonnet-4-5:1m',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our smartest model for complex agents and coding (1M context, alias)',
        strengths: 'High level of intelligence and capability with good speed (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null,
        beta_features: ['context-1m-2025-08-07']
      }
    },
    // --- Claude 4.1 Opus ---
    {
      id: 'claude-opus-4-1-20250805',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 32000,
      properties: {
        description: "Anthropic's most powerful Claude 4 model",
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderately Fast',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-opus-4-1',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 32000,
      properties: {
        description: "Anthropic's most powerful Claude 4 model (alias)",
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderately Fast',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 4 Opus ---
    {
      id: 'claude-opus-4-20250514',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 32000,
      properties: {
        description: "Anthropic's previously most powerful Claude 4 model",
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderately Fast',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-opus-4-0',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 32000,
      properties: {
        description: "Anthropic's most powerful Claude 4 model (alias)",
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderately Fast',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 4 Sonnet ---
    {
      id: 'claude-sonnet-4-20250514',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: "Anthropic's balanced Claude 4 model for intelligence and speed",
        strengths: 'High level of intelligence and capability with good speed (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-sonnet-4-0',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: "Anthropic's balanced Claude 4 model for intelligence and speed (alias)",
        strengths: 'High level of intelligence and capability with good speed (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Mar 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.7 Sonnet ---
    {
      id: 'claude-3-7-sonnet-20250219',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our most intelligent model',
        strengths: 'Highest level of intelligence and capability with toggleable extended thinking',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Nov 2024',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-3-7-sonnet-latest',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Our most intelligent model',
        strengths: 'Highest level of intelligence and capability with toggleable extended thinking',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Nov 2024',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.5 Sonnet (Upgraded Version) ---
    {
      id: 'claude-3-5-sonnet-20241022',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192,
      properties: {
        description: 'Our previous most intelligent model',
        strengths: 'High level of intelligence and capability',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Apr 2024',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-3-5-sonnet-latest',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192,
      properties: {
        description: 'Our previous most intelligent model',
        strengths: 'High level of intelligence and capability',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Apr 2024',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.5 Sonnet (Previous Version) ---
    {
      id: 'claude-3-5-sonnet-20240620',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192,
      properties: {
        description: 'Our previous most intelligent model',
        strengths: 'High level of intelligence and capability',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Apr 2024',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.5 Haiku ---
    {
      id: 'claude-3-5-haiku-20241022',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192,
      properties: {
        description: 'Our fastest model',
        strengths: 'Intelligence at blazing speeds',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.8,
        cost_output_mtok: 4.0,
        training_data_cutoff: 'July 2024',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-3-5-haiku-latest',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192,
      properties: {
        description: 'Our fastest model',
        strengths: 'Intelligence at blazing speeds',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.8,
        cost_output_mtok: 4.0,
        training_data_cutoff: 'July 2024',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3 Opus (deprecated) ---
    // --- Claude 3 Haiku ---
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
        description: 'Fastest and most compact model for near-instant responsiveness',
        strengths: 'Quick and accurate targeted performance',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.25,
        cost_output_mtok: 1.25,
        training_data_cutoff: 'Aug 2023',
        extended_max_completion_tokens: null
      }
    }
    // Note: The text does not list a 'claude-3-haiku-latest' alias for the 20240307 version,
    // so it's not included here to match the provided data exactly.
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
            extended_max_completion_tokens: rawModel.properties.extended_max_completion_tokens,
            beta_features: rawModel.properties.beta_features
          }
        : undefined,
      provider: Provider.Anthropic,
      rawData: rawModel // Store original
    })
  )
}
