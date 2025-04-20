/* eslint-disable no-console */
// Test script for model listing with multiple providers
import { RosettaAI, RosettaAIError, RosettaModelList } from '../src'
import dotenv from 'dotenv'

dotenv.config()

async function testModelListing() {
  console.log('=== Testing Model Listing with Multiple Providers ===')

  // Create a RosettaAI instance with multiple providers configured
  const rosetta = new RosettaAI()
  const configuredProviders = rosetta.getConfiguredProviders()

  if (configuredProviders.length === 0) {
    console.error('No providers configured. Please set API keys in your .env file.')
    return
  }

  if (configuredProviders.length < 2) {
    console.warn('This test is designed to verify model listing with multiple providers.')
    console.warn(
      `Currently only ${configuredProviders.length} provider is configured: ${configuredProviders.join(', ')}`
    )
    console.warn('For a more thorough test, configure at least 2 providers in your .env file.')
  }

  console.log(`Configured providers: ${configuredProviders.join(', ')}\n`)

  // Test 1: List models for each provider individually
  console.log('=== Test 1: Listing Models for Each Provider Individually ===')
  for (const provider of configuredProviders) {
    console.log(`\n--- Listing Models for: ${provider} ---`)
    try {
      const modelList: RosettaModelList = await rosetta.listModels(provider)
      console.log(`Success! Found ${modelList.data.length} models for ${provider}`)
      // Print first few models
      modelList.data.slice(0, 3).forEach(model => {
        console.log(`  - ID: ${model.id.padEnd(35)} Owner: ${model.owned_by.padEnd(15)}`)
      })
      if (modelList.data.length > 3) {
        console.log(`  ... and ${modelList.data.length - 3} more.`)
      }
    } catch (error) {
      console.error(`Error listing models for ${provider}:`)
      if (error instanceof RosettaAIError) {
        console.error(`  ${error.name}: ${error.message}`)
      } else {
        console.error(`  Unexpected error: ${error}`)
      }
    }
  }

  // Test 2: List models for all providers at once
  console.log('\n=== Test 2: Listing Models for All Providers at Once ===')
  try {
    const allModelsResult = await rosetta.listAllModels()

    for (const provider of configuredProviders) {
      console.log(`\n--- Results for: ${provider} ---`)
      const result = allModelsResult[provider]

      if (result instanceof RosettaAIError) {
        console.error(`  Error: ${result.name} - ${result.message}`)
      } else if (result) {
        console.log(`  Success! Found ${result.data.length} models`)
        // Print first few models
        result.data.slice(0, 3).forEach(model => {
          console.log(`    - ID: ${model.id.padEnd(35)} Owner: ${model.owned_by}`)
        })
        if (result.data.length > 3) {
          console.log(`    ... and ${result.data.length - 3} more.`)
        }
      } else {
        console.log('  No result found (unexpected).')
      }
    }
  } catch (error) {
    console.error('Unexpected error during listAllModels execution:')
    console.error(error)
  }

  console.log('\n=== Model Listing Test Complete ===')
}

// Run the test
testModelListing().catch(err => console.error('Unhandled error in test script:', err))
