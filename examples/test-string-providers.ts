/* eslint-disable no-console */
// Test script for using string provider names instead of enum values
import { RosettaAI, RosettaAIError } from '../src'
import dotenv from 'dotenv'

dotenv.config()

async function testStringProviders() {
  console.log('=== Testing String Provider Names ===')

  // Create a RosettaAI instance
  const rosetta = new RosettaAI()
  const configuredProviders = rosetta.getConfiguredProviders()

  if (configuredProviders.length === 0) {
    console.error('No providers configured. Please set API keys in your .env file.')
    return
  }

  console.log(`Configured providers: ${configuredProviders.join(', ')}\n`)

  // Test 1: Use string provider names instead of enum values
  console.log('=== Test 1: Using String Provider Names ===')

  // Test with different string formats for provider names
  const testProviders = [
    'openai', // lowercase
    'OPENAI', // uppercase
    'OpenAI', // mixed case
    'google', // another provider
    'invalid' // invalid provider (should fail)
  ]

  for (const providerStr of testProviders) {
    console.log(`\n--- Testing with provider string: "${providerStr}" ---`)

    // Define normalizedKey outside the try block so it's accessible in the catch block
    const normalizedKey = providerStr.toLowerCase()

    try {
      // Check if this provider is actually configured
      const isConfigured = configuredProviders.some(p => typeof p === 'string' && p.toLowerCase() === normalizedKey)

      if (!isConfigured && normalizedKey !== 'invalid') {
        console.log(`Provider "${providerStr}" is not configured, skipping test.`)
        continue
      }

      // For the invalid provider, we expect an error
      if (normalizedKey === 'invalid') {
        console.log(`Testing with invalid provider "${providerStr}" (expecting error)...`)
        try {
          await rosetta.listModels(providerStr)
          console.error('ERROR: Expected an error for invalid provider, but none was thrown!')
        } catch (error) {
          console.log(
            `Success! Got expected error for invalid provider: ${
              error instanceof Error ? error.message : String(error)
            }`
          )
        }
        continue
      }

      // For valid providers, test listModels
      console.log(`Listing models for "${providerStr}"...`)
      const modelList = await rosetta.listModels(providerStr)
      console.log(`Success! Found ${modelList.data.length} models for "${providerStr}"`)

      // Print first few models
      modelList.data.slice(0, 3).forEach(model => {
        console.log(`  - ID: ${model.id.padEnd(35)} Owner: ${model.owned_by.padEnd(15)}`)
      })
      if (modelList.data.length > 3) {
        console.log(`  ... and ${modelList.data.length - 3} more.`)
      }
    } catch (error) {
      if (normalizedKey !== 'invalid') {
        console.error(`Error listing models for "${providerStr}":`)
        if (error instanceof RosettaAIError) {
          console.error(`  ${error.name}: ${error.message}`)
        } else {
          console.error(`  Unexpected error: ${error}`)
        }
      }
    }
  }

  // Test 2: Test generate with string provider
  if (configuredProviders.some(p => typeof p === 'string' && p.toLowerCase() === 'openai')) {
    console.log('\n=== Test 2: Generate with String Provider ===')
    try {
      const result = await rosetta.generate({
        provider: 'openai', // Use string instead of Provider.OpenAI
        messages: [{ role: 'user', content: 'Say hello!' }]
        // Model will use default if configured
      })
      console.log('Success! Generated content:')
      console.log(`"${result.content}"`)
    } catch (error) {
      console.error('Error generating with string provider:')
      if (error instanceof RosettaAIError) {
        console.error(`  ${error.name}: ${error.message}`)
      } else {
        console.error(`  Unexpected error: ${error}`)
      }
    }
  } else {
    console.log('\nSkipping Test 2: OpenAI provider not configured.')
  }

  console.log('\n=== String Provider Test Complete ===')
}

// Run the test
testStringProviders().catch(err => console.error('Unhandled error in test script:', err))
