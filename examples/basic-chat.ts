/* eslint-disable no-console */
// Basic Chat Completion Example
import { RosettaAI, Provider, ConfigurationError, ProviderAPIError, RosettaAIError } from '../src'
import dotenv from 'dotenv'

// Load environment variables from .env file
dotenv.config()

async function runBasicChat() {
  console.log('Initializing RosettaAI...')
  try {
    const rosetta = new RosettaAI() // Configuration loaded automatically from env/constructor
    const providers = rosetta.getConfiguredProviders()

    if (providers.length === 0) {
      console.error('No providers configured. Please set API keys in your .env file or pass them to the constructor.')
      return
    }

    console.log(`Available providers: ${providers.join(', ')}\n`)

    const commonPrompt = "Explain 'separation of concerns' in software engineering simply."

    // Iterate through each configured provider
    for (const provider of providers) {
      console.log(`--- Testing Provider: ${provider} ---`)
      try {
        // Use configured default model or fallback logic
        const model =
          rosetta.config.defaultModels?.[provider] ??
          (provider === Provider.Anthropic
            ? 'claude-3-haiku-20240307'
            : provider === Provider.Google
            ? 'gemini-1.5-flash-latest'
            : provider === Provider.Groq
            ? 'llama3-8b-8192'
            : provider === Provider.OpenAI
            ? 'gpt-4o-mini'
            : undefined) // Fallback model IDs

        if (!model) {
          console.log(`Skipping ${provider}: No default model configured or fallback available.`)
          continue
        }
        console.log(`Using model: ${model}`)

        // Make the generation request
        const result = await rosetta.generate({
          provider: provider,
          model: model,
          messages: [
            { role: 'system', content: 'You are a helpful, concise assistant.' },
            { role: 'user', content: commonPrompt }
          ],
          maxTokens: 150,
          temperature: 0.7
        })

        // Display results
        console.log(`[${provider} Response - Model: ${result.model}]`)
        console.log('Content:', result.content ?? '[No Content]')
        console.log('Finish Reason:', result.finishReason)
        console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
        console.log('---------------\n')
      } catch (error) {
        // Handle errors specific to this provider - Enhanced Example
        if (error instanceof ProviderAPIError) {
          // Specific handling for API errors (rate limits, auth issues, etc.)
          console.error(
            `API Error from ${provider} (Status: ${error.statusCode ?? 'N/A'}, Code: ${error.errorCode ?? 'N/A'}): ${
              error.message
            }`
          )
          // Optionally log the underlying error for more details during development
          // console.error("Underlying Provider Error:", error.underlyingError);
        } else if (error instanceof ConfigurationError) {
          // Handle configuration issues specific to this provider call (e.g., invalid model for provider)
          console.error(`Configuration Error for ${provider}: ${error.message}`)
        } else if (error instanceof RosettaAIError) {
          // Catch other SDK-specific errors
          console.error(`RosettaAI Error with ${provider}: ${error.name} - ${error.message}`)
        } else {
          // Catch unexpected errors
          console.error(`Unexpected error with ${provider}:`, error)
        }
        console.log('---------------\n')
      }
      // Add a small delay between provider calls to avoid rate limiting issues
      await new Promise(resolve => setTimeout(resolve, 1500))
    }
  } catch (error) {
    // Catch initialization errors
    if (error instanceof ConfigurationError) {
      console.error(`Initialization failed: ${error.message}`)
    } else {
      console.error('Unexpected initialization error:', error)
    }
  }
}

// Run the example
runBasicChat().catch(err => console.error('Unhandled error in example script:', err))
