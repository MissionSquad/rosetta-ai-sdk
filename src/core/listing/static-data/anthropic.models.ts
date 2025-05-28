// src/core/listing/static-data/anthropic.models.ts
import { RosettaModelList, RosettaModel, Provider } from '../../../types'

// Raw data matching the provided JSON structure, maybe use an intermediate raw type if needed.
// Placeholder structure based on mapping logic in the plan. Replace with actual JSON if available.
const rawAnthropicData = {
  object: 'list',
  data: [
    // --- Claude 4 Opus ---
    {
      id: 'claude-opus-4-20250514',
      object: 'model',
      owned_by: 'anthropic',
      created: null, // Placeholder
      active: true,
      context_window: 200000, // Assuming similar to recent models
      public_apps: null,
      max_completion_tokens: 32000, // Updated based on feedback
      properties: {
        description: 'Anthropic\'s most powerful Claude 4 model',
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true, // Assuming for a new Opus model
        comparative_latency: 'Moderately Fast', // Matches feedback
        cost_input_mtok: 15.0, // Matches feedback
        cost_output_mtok: 75.0, // Matches feedback
        training_data_cutoff: 'Mar 2025', // Updated based on feedback
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-opus-4-latest',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 32000, // Updated based on feedback
      properties: {
        description: 'Anthropic\'s most powerful Claude 4 model (latest alias)',
        strengths: 'Highest level of intelligence and capability for complex tasks (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderately Fast', // Matches feedback
        cost_input_mtok: 15.0, // Matches feedback
        cost_output_mtok: 75.0, // Matches feedback
        training_data_cutoff: 'Mar 2025', // Updated based on feedback
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 4 Sonnet ---
    {
      id: 'claude-sonnet-4-20250514',
      object: 'model',
      owned_by: 'anthropic',
      created: null, // Placeholder
      active: true,
      context_window: 200000, // Assuming similar to recent models
      public_apps: null,
      max_completion_tokens: 64000, // Matches feedback
      properties: {
        description: 'Anthropic\'s balanced Claude 4 model for intelligence and speed',
        strengths: 'High level of intelligence and capability with good speed (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true, // Assuming for a new Sonnet model
        comparative_latency: 'Fast', // Matches feedback
        cost_input_mtok: 3.0, // Matches feedback
        cost_output_mtok: 15.0, // Matches feedback
        training_data_cutoff: 'Mar 2025', // Updated based on feedback
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-sonnet-4-latest',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000, // Matches feedback
      properties: {
        description: 'Anthropic\'s balanced Claude 4 model for intelligence and speed (latest alias)',
        strengths: 'High level of intelligence and capability with good speed (Claude 4 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast', // Matches feedback
        cost_input_mtok: 3.0, // Matches feedback
        cost_output_mtok: 15.0, // Matches feedback
        training_data_cutoff: 'Mar 2025', // Updated based on feedback
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
      max_completion_tokens: 64000, // From "Max output" 64000 tokens
      properties: {
        description: 'Our most intelligent model',
        strengths: 'Highest level of intelligence and capability with toggleable extended thinking',
        multilingual: true, // From "Yes"
        vision: true, // From "Yes"
        extended_thinking: true, // From "Yes"
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0, // From "$3.00"
        cost_output_mtok: 15.0, // From "$15.00"
        training_data_cutoff: 'Nov 2024',
        extended_max_completion_tokens: null // Not specified in new data
      }
    },
    {
      id: 'claude-3-7-sonnet-latest', // Alias from the text
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000, // Mirrors claude-3-7-sonnet-20250219
      properties: {
        description: 'Our most intelligent model',
        strengths: 'Highest level of intelligence and capability with toggleable extended thinking',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Nov 2024', // Assumed to be same as the underlying model
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.5 Sonnet (Upgraded Version) ---
    {
      id: 'claude-3-5-sonnet-20241022', // Upgraded version from API list
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192, // From "Max output" 8192 tokens
      properties: {
        description: 'Our previous most intelligent model', // From "Claude 3.5 Sonnet" column
        strengths: 'High level of intelligence and capability', // From "Claude 3.5 Sonnet" column
        multilingual: true, // From "Yes"
        vision: true, // From "Yes"
        extended_thinking: false, // From "No"
        comparative_latency: 'Fast', // From "Claude 3.5 Sonnet" column
        cost_input_mtok: 3.0, // From "$3.00"
        cost_output_mtok: 15.0, // From "$15.00"
        training_data_cutoff: 'Apr 2024', // From "Claude 3.5 Sonnet" column
        extended_max_completion_tokens: null // Not specified in new data
      }
    },
    {
      id: 'claude-3-5-sonnet-latest', // Alias from the text
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192, // Mirrors claude-3-5-sonnet-20241022
      properties: {
        description: 'Our previous most intelligent model',
        strengths: 'High level of intelligence and capability',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Apr 2024', // Assumed to be same as the underlying model
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3.5 Sonnet (Previous Version) ---
    {
      id: 'claude-3-5-sonnet-20240620', // Previous version from API list
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000, // Assumed same as other 3.5 Sonnet
      public_apps: null,
      max_completion_tokens: 8192, // Assumed same as upgraded 3.5 Sonnet based on table column
      properties: {
        description: 'Our previous most intelligent model', // Using data from "Claude 3.5 Sonnet" column as specific data isn't provided
        strengths: 'High level of intelligence and capability', // Using data from "Claude 3.5 Sonnet" column
        multilingual: true, // Using data from "Claude 3.5 Sonnet" column
        vision: true, // Using data from "Claude 3.5 Sonnet" column
        extended_thinking: false, // Using data from "Claude 3.5 Sonnet" column
        comparative_latency: 'Fast', // Using data from "Claude 3.5 Sonnet" column
        cost_input_mtok: 3.0, // Using data from "Claude 3.5 Sonnet" column
        cost_output_mtok: 15.0, // Using data from "Claude 3.5 Sonnet" column
        training_data_cutoff: 'Apr 2024', // Using data from "Claude 3.5 Sonnet" column, actual cutoff might differ slightly but not specified
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
      max_completion_tokens: 8192, // From "Max output" 8192 tokens
      properties: {
        description: 'Our fastest model',
        strengths: 'Intelligence at blazing speeds',
        multilingual: true, // From "Yes"
        vision: true, // From "Yes"
        extended_thinking: false, // From "No"
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.8, // From "$0.80"
        cost_output_mtok: 4.0, // From "$4.00"
        training_data_cutoff: 'July 2024',
        extended_max_completion_tokens: null // Not specified in new data
      }
    },
    {
      id: 'claude-3-5-haiku-latest', // Alias from the text
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 8192, // Mirrors claude-3-5-haiku-20241022
      properties: {
        description: 'Our fastest model',
        strengths: 'Intelligence at blazing speeds',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.8,
        cost_output_mtok: 4.0,
        training_data_cutoff: 'July 2024', // Assumed to be same as the underlying model
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3 Opus ---
    {
      id: 'claude-3-opus-20240229',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096, // From "Max output" 4096 tokens
      properties: {
        description: 'Powerful model for complex tasks',
        strengths: 'Top-level intelligence, fluency, and understanding',
        multilingual: true, // From "Yes"
        vision: true, // From "Yes"
        extended_thinking: false, // From "No" - Updated based on new table
        comparative_latency: 'Moderately fast',
        cost_input_mtok: 15.0, // From "$15.00"
        cost_output_mtok: 75.0, // From "$75.00"
        training_data_cutoff: 'Aug 2023',
        extended_max_completion_tokens: null // Not specified in new data
      }
    },
    {
      id: 'claude-3-opus-latest', // Alias from the text
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096, // Mirrors claude-3-opus-20240229
      properties: {
        description: 'Powerful model for complex tasks',
        strengths: 'Top-level intelligence, fluency, and understanding',
        multilingual: true,
        vision: true,
        extended_thinking: false,
        comparative_latency: 'Moderately fast',
        cost_input_mtok: 15.0,
        cost_output_mtok: 75.0,
        training_data_cutoff: 'Aug 2023', // Assumed to be same as the underlying model
        extended_max_completion_tokens: null
      }
    },
    // --- Claude 3 Haiku ---
    {
      id: 'claude-3-haiku-20240307',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 4096, // From "Max output" 4096 tokens
      properties: {
        description: 'Fastest and most compact model for near-instant responsiveness',
        strengths: 'Quick and accurate targeted performance',
        multilingual: true, // From "Yes"
        vision: true, // From "Yes"
        extended_thinking: false, // From "No" - Updated based on new table
        comparative_latency: 'Fastest',
        cost_input_mtok: 0.25, // From "$0.25"
        cost_output_mtok: 1.25, // From "$1.25"
        training_data_cutoff: 'Aug 2023',
        extended_max_completion_tokens: null // Not specified in new data
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
            extended_max_completion_tokens: rawModel.properties.extended_max_completion_tokens
          }
        : undefined,
      provider: Provider.Anthropic,
      rawData: rawModel // Store original
    })
  )
}
