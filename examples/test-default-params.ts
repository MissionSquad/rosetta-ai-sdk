import { RosettaAI } from '../src'
import dotenv from 'dotenv'

// Load environment variables from .env file
dotenv.config()

async function testDefaultParams() {
  // Initialize RosettaAI with OpenAI API key
  const rosetta = new RosettaAI()

  try {
    // Make a request without specifying temperature
    // This should not include temperature in the request to OpenAI
    const result = await rosetta.generate({
      provider: 'openai',
      model: 'o3-mini',
      messages: [
        {
          role: 'user',
          content: 'Hello, how are you?'
        }
      ]
      // Note: temperature is intentionally not specified
    })

    console.log('Response received successfully:')
    console.log(`Content: ${result.content}`)
    console.log(`Model: ${result.model}`)
    console.log(`Finish reason: ${result.finishReason}`)

    if (result.usage) {
      console.log('Usage:')
      console.log(`  Prompt tokens: ${result.usage.promptTokens}`)
      console.log(`  Completion tokens: ${result.usage.completionTokens}`)
      console.log(`  Total tokens: ${result.usage.totalTokens}`)
    }
  } catch (error) {
    console.error('Error:', error)
  }
}

testDefaultParams()
