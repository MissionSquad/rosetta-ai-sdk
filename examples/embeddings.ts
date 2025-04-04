/* eslint-disable no-console */
// Embeddings Example
import { RosettaAI, Provider, EmbedParams, RosettaAIError } from '../src'
import dotenv from 'dotenv'

dotenv.config()

async function runEmbeddings() {
  const rosetta = new RosettaAI()

  // Filter providers that support embeddings
  const providers = rosetta
    .getConfiguredProviders()
    .filter(p => [Provider.OpenAI, Provider.Google, Provider.Groq].includes(p))

  if (providers.length === 0) {
    console.error('No configured providers support embeddings (OpenAI, Google, Groq needed).')
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
      const model =
        rosetta.config.defaultEmbeddingModels?.[provider] ??
        (provider === Provider.OpenAI
          ? 'text-embedding-3-small'
          : provider === Provider.Google
          ? 'text-embedding-004' // Google's latest embedding model
          : provider === Provider.Groq
          ? 'nomic-embed-text-v1.5' // Groq's embedding model
          : undefined)

      if (!model) {
        console.log(`Skipping ${provider}: No default embedding model configured or fallback available.`)
        continue
      }
      console.log(`Using model: ${model}`)

      // Prepare parameters - Groq might only support single string input
      let inputData: string | string[]
      if (provider === Provider.Groq) {
        console.log(`(Note: Groq currently processes only the first text for embeddings in this example)`)
        inputData = textsToEmbed[0]!
      } else {
        // OpenAI and Google support batching via array input
        inputData = textsToEmbed
      }

      const params: EmbedParams = {
        provider,
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
