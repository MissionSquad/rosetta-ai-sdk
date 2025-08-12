/* eslint-disable no-console */
// Embeddings Example
import { RosettaAI, Provider, EmbedParams, RosettaAIError, CustomProviderConfig } from '../src'
import { OpenAICompatibleMapper } from '../src/core/mapping/openai-compatible.mapper'
import dotenv from 'dotenv'

dotenv.config()

// GPUStack custom provider configuration
const gpustackProviderKey = 'gpustack'
const gpustackProviderConfig: CustomProviderConfig = {
  providerKey: gpustackProviderKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'embed', 'image_input'],
  apiKey: process.env.GPUSTACK_API_KEY,
  defaultModel: process.env.ROSETTA_DEFAULT_GPUSTACK_MODEL ?? 'qwen2.5-coder-14b-instruct',
  defaultEmbeddingModel: 'qwen3-embedding-0.6b', // Specifically use this model
  toolConfig: {
    toolDefinitionFormat: 'jsonSchema',
    toolCallInputFormat: 'jsonString',
    toolResultFormat: 'jsonString'
  }
}

async function runEmbeddings() {
  if (!process.env.GPUSTACK_API_KEY) {
    console.warn('Warning: GPUSTACK_API_KEY not found in environment. GPUStack provider will be skipped.')
  }

  const rosetta = new RosettaAI({
    customProviders: [gpustackProviderConfig]
  })

  // Filter providers that support embeddings
  const providers = rosetta.getConfiguredProviders().filter(p => {
    // Check if it's a standard provider that supports embeddings
    if (['openai', 'google'].includes(p as string)) return true
    // Check if it's our custom gpustack provider
    if (p === gpustackProviderKey) return true
    return false
  })

  if (providers.length === 0) {
    console.error('No configured providers support embeddings (OpenAI, Google, or GPUStack needed).')
    return
  }

  console.log(`--- Testing Embeddings ---`)
  const textsToEmbed = [
    'The quick brown fox jumps over the lazy dog.',
    'Software development requires careful planning.',
    'Embeddings represent text in a vector space.'
  ]

  for (const provider of providers) {
    console.log(`\n--- Provider: ${provider} ---`)
    try {
      // Select embedding model (default or fallback)
      let model: string | undefined

      if (provider === Provider.OpenAI) {
        model = rosetta.config.defaultEmbeddingModels?.[provider] ?? 'text-embedding-3-small'
      } else if (provider === Provider.Google) {
        model = rosetta.config.defaultEmbeddingModels?.[provider] ?? 'text-embedding-004'
      } else if (provider === gpustackProviderKey) {
        model = gpustackProviderConfig.defaultEmbeddingModel
      }

      if (!model) {
        console.log(`Skipping ${provider}: No default embedding model configured or fallback available.`)
        continue
      }
      console.log(`Using model: ${model}`)

      // OpenAI, Google, and GPUStack support batching via array input
      const inputData = textsToEmbed

      const params: EmbedParams = {
        provider: provider as Provider,
        model,
        input: inputData
        // Optionally add dimensions for OpenAI: dimensions: 256
      }

      // Generate embeddings
      const result = await rosetta.embed(params)

      console.log(`Generated ${result.embeddings.length} embedding vector(s).`)
      result.embeddings.forEach((embeddingVector, index) => {
        console.log(
          `  Input ${index + 1} (Vector Length: ${embeddingVector.length}): [${embeddingVector
            .slice(0, 4)
            .map(n => n.toFixed(4))
            .join(', ')}...]`
        )
      })
      console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
      console.log('Model Used:', result.model) // Display the actual model string used
    } catch (error) {
      if (error instanceof RosettaAIError) {
        console.error(`Error with ${provider} embeddings: ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected error with ${provider} embeddings:`, error)
      }
    }
    // Delay between provider calls
    await new Promise(resolve => setTimeout(resolve, 1000))
  } // End provider loop

  console.log('--------------------\nEmbeddings Test Complete.')
}

// Run the example
runEmbeddings().catch(err => console.error('Unhandled error in embeddings example script:', err))
