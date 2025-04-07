/* eslint-disable no-console */
// Model Listing Example
import { RosettaAI, Provider, RosettaAIError, RosettaModelList } from '../src'
import dotenv from 'dotenv'

dotenv.config()

async function runModelListing() {
  console.log('Initializing RosettaAI for Model Listing...')
  const rosetta = new RosettaAI()
  const configuredProviders = rosetta.getConfiguredProviders()

  if (configuredProviders.length === 0) {
    console.error('No providers configured. Please set API keys in your .env file.')
    return
  }

  console.log(`Configured providers: ${configuredProviders.join(', ')}\n`)

  // --- Example 1: List models for a specific provider ---
  const providerToList = Provider.OpenAI // Change this to test others like Provider.Groq, Provider.Anthropic, Provider.Google
  if (configuredProviders.includes(providerToList)) {
    console.log(`--- Listing Models for: ${providerToList} ---`)
    try {
      const modelList: RosettaModelList = await rosetta.listModels(providerToList)
      console.log(`Found ${modelList.data.length} models:`)
      modelList.data.forEach(model => {
        // Print some key details
        console.log(
          `  - ID: ${model.id.padEnd(35)} Owner: ${model.owned_by.padEnd(15)} Context: ${model.context_window ?? 'N/A'}`
        )
        // Optionally print more properties
        // if (model.properties) {
        //   console.log(`    Properties: ${JSON.stringify(model.properties)}`);
        // }
      })
    } catch (error) {
      if (error instanceof RosettaAIError) {
        console.error(`Error listing models for ${providerToList}: ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected error listing models for ${providerToList}:`, error)
      }
    }
  } else {
    console.log(`Skipping single provider test for ${providerToList} as it's not configured.`)
  }

  // --- Example 2: List models for ALL configured providers ---
  console.log('\n--- Listing Models for ALL Configured Providers ---')
  try {
    const allModelsResult = await rosetta.listAllModels()

    for (const provider of configuredProviders) {
      console.log(`\n--- Results for: ${provider} ---`)
      const result = allModelsResult[provider]

      if (result instanceof RosettaAIError) {
        // Handle errors for specific providers
        console.error(`  Error: ${result.name} - ${result.message}`)
      } else if (result) {
        // Process successful list
        console.log(`  Found ${result.data.length} models:`)
        // Print only the first few models for brevity
        result.data.slice(0, 5).forEach(model => {
          console.log(`    - ID: ${model.id.padEnd(35)} Owner: ${model.owned_by}`)
        })
        if (result.data.length > 5) {
          console.log(`    ... and ${result.data.length - 5} more.`)
        }
      } else {
        // Should not happen if provider is in configuredProviders, but handle defensively
        console.log('  No result found (unexpected).')
      }
    }
  } catch (error) {
    // This catch block is unlikely to be hit for listAllModels itself,
    // as errors are returned within the result object per provider.
    console.error('Unexpected error during listAllModels execution:', error)
  }

  console.log('\n--- Model Listing Example Complete ---')
}

// Run the example
runModelListing().catch(err => console.error('Unhandled error in model listing example script:', err))
