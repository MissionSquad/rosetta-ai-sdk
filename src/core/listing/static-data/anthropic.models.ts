// src/core/listing/static-data/anthropic.models.ts
import { RosettaModelList, RosettaModel, Provider } from '../../../types'

// Raw data matching Anthropic's current model lineup.
// See: https://docs.anthropic.com/en/docs/about-claude/models/overview
const rawAnthropicData = {
  object: 'list',
  data: [
    // =========================================================================
    // LATEST MODELS
    // =========================================================================

    // --- Claude Fable 5 ---
    {
      id: 'claude-fable-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'Next-generation intelligence for long-running agents',
        strengths: "Anthropic's most capable widely released model for demanding reasoning and long-horizon agentic work (Claude 5 series)",
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Slower',
        cost_input_mtok: 10.0,
        cost_output_mtok: 50.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null
      }
    },

    // --- Claude Opus 5 ---
    {
      id: 'claude-opus-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'For complex agentic coding and enterprise work',
        strengths: 'Step-change over Opus 4.8 in deep reasoning, agentic coding, and long-horizon work (Claude 5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'May 2026',
        extended_max_completion_tokens: null
      }
    },

    // --- Claude Sonnet 5 ---
    // Introductory pricing of $2/$10 per MTok applies through 2026-08-31;
    // the durable sticker price is recorded here.
    {
      id: 'claude-sonnet-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'The best combination of speed and intelligence',
        strengths: 'Near-Opus quality on coding and agentic work with fast response times (Claude 5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null
      }
    },

    // --- Claude Opus 4.8 ---
    {
      id: 'claude-opus-4-8',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: "Anthropic's most capable model for complex reasoning and agentic coding",
        strengths: 'Top-tier complex reasoning, long-horizon agentic coding, and high-autonomy work (Claude 4.8 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null
      }
    },

    // --- Claude Opus 4.7 ---
    {
      id: 'claude-opus-4-7',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'Our most capable generally available model for complex reasoning and agentic coding',
        strengths: 'Step-change improvement in agentic coding over Opus 4.6 (Claude 4.7 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null
      }
    },

    // --- Claude Opus 4.6 ---
    {
      id: 'claude-opus-4-6',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'The most intelligent model for building agents and coding',
        strengths: 'Top-tier reasoning, coding, and complex task performance (Claude 4.6 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Aug 2025',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude Opus 4.6 (1M Context) ---
    {
      id: 'claude-opus-4-6:1m',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 128000,
      properties: {
        description: 'The most intelligent model for building agents and coding (1M context)',
        strengths: 'Top-tier reasoning, coding, and complex task performance (Claude 4.6 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Aug 2025',
        extended_max_completion_tokens: null,
        beta_features: ['context-1m-2025-08-07']
      }
    },

    // --- Claude Sonnet 4.6 ---
    {
      id: 'claude-sonnet-4-6',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'The best combination of speed and intelligence',
        strengths: 'High intelligence with fast response times (Claude 4.6 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null
      }
    },
    // --- Claude Sonnet 4.6 (1M Context) ---
    {
      id: 'claude-sonnet-4-6:1m',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 1000000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'The best combination of speed and intelligence (1M context)',
        strengths: 'High intelligence with fast response times (Claude 4.6 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fast',
        cost_input_mtok: 3.0,
        cost_output_mtok: 15.0,
        training_data_cutoff: 'Jan 2026',
        extended_max_completion_tokens: null,
        beta_features: ['context-1m-2025-08-07']
      }
    },

    // --- Claude Haiku 4.5 ---
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
        description: 'The fastest model with near-frontier intelligence',
        strengths: 'Near-frontier intelligence at fastest speeds (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fastest',
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
        description: 'The fastest model with near-frontier intelligence (alias)',
        strengths: 'Near-frontier intelligence at fastest speeds (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Fastest',
        cost_input_mtok: 1.0,
        cost_output_mtok: 5.0,
        training_data_cutoff: 'Jul 2025',
        extended_max_completion_tokens: null
      }
    },

    // =========================================================================
    // LEGACY MODELS
    // =========================================================================

    // --- Claude Sonnet 4.5 ---
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
        description: 'Previous generation smart model for complex agents and coding',
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
        description: 'Previous generation smart model for complex agents and coding (alias)',
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
    // --- Claude Sonnet 4.5 (1M Context) ---
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
        description: 'Previous generation smart model for complex agents and coding (1M context)',
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
        description: 'Previous generation smart model for complex agents and coding (1M context, alias)',
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

    // --- Claude Opus 4.5 ---
    {
      id: 'claude-opus-4-5-20251101',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Previous generation powerful model for complex tasks',
        strengths: 'Highest level of intelligence and capability (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Aug 2025',
        extended_max_completion_tokens: null
      }
    },
    {
      id: 'claude-opus-4-5',
      object: 'model',
      owned_by: 'anthropic',
      created: null,
      active: true,
      context_window: 200000,
      public_apps: null,
      max_completion_tokens: 64000,
      properties: {
        description: 'Previous generation powerful model for complex tasks (alias)',
        strengths: 'Highest level of intelligence and capability (Claude 4.5 series)',
        multilingual: true,
        vision: true,
        extended_thinking: true,
        comparative_latency: 'Moderate',
        cost_input_mtok: 5.0,
        cost_output_mtok: 25.0,
        training_data_cutoff: 'Aug 2025',
        extended_max_completion_tokens: null
      }
    }
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
