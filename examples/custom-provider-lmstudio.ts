/* eslint-disable no-console */
/**
 * Custom Provider Example: LM Studio (OpenAI Compatible)
 *
 * This example demonstrates how to configure and use a custom provider
 * that is compatible with the OpenAI API standard, like LM Studio,
 * using the refactored `OpenAICompatibleMapper`.
 *
 * Prerequisites:
 * 1. Install dependencies: `yarn install` (in the examples directory)
 * 2. Ensure LM Studio is running and serving a model on the configured URL (default: http://localhost:1234/v1).
 * 3. (Optional) Set `LMSTUDIO_BASE_URL` and `ROSETTA_DEFAULT_LMSTUDIO_MODEL` in `examples/.env`.
 * 4. Run this example: `npm run example:custom-lms`
 */

import dotenv from 'dotenv'
import { z } from 'zod'

import {
  RosettaAI,
  RosettaMessage,
  RosettaTool,
  GenerateResult,
  CustomProviderConfig,
  ProviderAPIError,
  ToolArgumentValidationError,
  Provider
} from '../src' // Adjust path as needed
import { OpenAICompatibleMapper } from '../src/core/mapping/openai-compatible.mapper'

// Load environment variables from examples/.env
dotenv.config({ path: '.env' })

// --- LM Studio Provider Configuration ---
const lmstudioProviderKey = 'lmstudio' // Unique key for this provider
const lmstudioProviderConfig: CustomProviderConfig = {
  providerKey: lmstudioProviderKey,
  mapper: OpenAICompatibleMapper, // Use the reusable OpenAI-compatible mapper
  supportedFeatures: ['generate', 'stream', 'tool_use'], // Assuming standard OpenAI compatibility
  baseURL: process.env.LMSTUDIO_BASE_URL || 'http://localhost:1234/v1', // LM Studio's OpenAI-compatible endpoint
  // Define a default model for LM Studio (user needs to ensure this model is loaded in LM Studio)
  defaultModel: process.env.ROSETTA_DEFAULT_LMSTUDIO_MODEL ?? 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF',
  // Tool configuration matching OpenAI's expected format (default for OpenAICompatibleMapper)
  toolConfig: {
    toolDefinitionFormat: 'jsonSchema',
    toolCallInputFormat: 'jsonString',
    toolResultFormat: 'jsonString'
  }
}
// --- Example Usage ---
async function runLMStudioExamples() {
  console.log('--- Custom Provider Example: LM Studio ---')
  console.log(`Connecting to LM Studio at: ${lmstudioProviderConfig.baseURL}`)
  console.log(`Using default model: ${lmstudioProviderConfig.defaultModel}`)
  if (lmstudioProviderConfig.apiKey) {
    console.log('Using API Key provided via LMSTUDIO_API_KEY.')
  } else {
    console.log('No API Key provided (typical for local LM Studio).')
  }

  // Initialize RosettaAI with the custom provider configuration
  const rosetta = new RosettaAI({
    customProviders: [lmstudioProviderConfig]
  })
  console.log(`Configured providers: ${rosetta.getConfiguredProviders().join(', ')}`)
  if (!rosetta.getConfiguredProviders().includes(lmstudioProviderKey)) {
    console.error(`Error: Failed to register custom provider '${lmstudioProviderKey}'. Check configuration.`)
    return
  }
  // --- 1. Basic Generation ---
  console.log('--- 1. Basic Generation (LM Studio) ---')
  try {
    const result = await rosetta.generate({
      provider: lmstudioProviderKey as Provider, // Cast for GenerateParams type
      model: lmstudioProviderConfig.defaultModel, // Use the default model
      messages: [{ role: 'user', content: 'Write a short tagline for a new AI SDK.' }],
      maxTokens: 50,
      temperature: 0.7 // LM Studio default temp
      // topP: 1
    })
    console.log(`[${result.model}] Response: ${result.content}`)
    console.log(`Usage: ${JSON.stringify(result.usage)}`)
  } catch (error) {
    console.error('[Basic Generation Error]', error)
    if (error instanceof ProviderAPIError && error.message.includes('fetch failed')) {
      console.error('Hint: Ensure LM Studio is running and accessible at the configured URL.')
    }
  }

  // --- 2. Streaming Generation ---
  console.log('--- 2. Streaming Generation (LM Studio) ---')
  let fullStreamedContent = ''
  let finalStreamResult: GenerateResult | null = null
  try {
    const stream = rosetta.stream({
      provider: lmstudioProviderKey as Provider, // Cast for GenerateParams type
      model: lmstudioProviderConfig.defaultModel,
      messages: [{ role: 'user', content: 'Explain the concept of "API compatibility" briefly.' }],
      maxTokens: 100,
      temperature: 0.7
      // topP: 1
    })

    process.stdout.write(`[${lmstudioProviderConfig.defaultModel} Stream] `)
    for await (const chunk of stream) {
      if (chunk.type === 'content_delta') {
        process.stdout.write(chunk.data.delta)
        fullStreamedContent += chunk.data.delta
      } else if (chunk.type === 'message_stop') {
        console.log(`--- Stream Stop (Reason: ${chunk.data.finishReason}) ---`)
      } else if (chunk.type === 'final_usage') {
        console.log(`--- Stream Usage: ${JSON.stringify(chunk.data.usage)} ---`)
      } else if (chunk.type === 'final_result') {
        finalStreamResult = chunk.data.result
        console.log('--- Final Aggregated Result Received ---')
      } else if (chunk.type === 'error') {
        console.error('--- Stream Error ---', chunk.data.error)
        break // Stop processing on error
      }
    }
    console.log('--- End of Stream ---')
    // Log the accumulated content and the final result object
    console.log('Accumulated Stream Content:', fullStreamedContent)
    if (finalStreamResult) {
      console.log('Final Aggregated Result Object:', JSON.stringify(finalStreamResult, null, 2))
    } else {
      console.log('Final Aggregated Result Object: Not received.')
    }
  } catch (error) {
    // Catch errors during stream setup (e.g., initial API call failure)
    console.error('[Streaming Setup Error]', error)
    if (error instanceof ProviderAPIError && error.message.includes('fetch failed')) {
      console.error('Hint: Ensure LM Studio is running and accessible at the configured URL.')
    }
  }

  // --- 3. Tool Use ---
  console.log('--- 3. Tool Use (LM Studio) ---')
  // Define a simple tool
  const GetCapitalToolSchema = z.object({
    country: z.string().describe('The country for which to find the capital city.')
  })
  const getCapitalTool: RosettaTool<typeof GetCapitalToolSchema> = {
    type: 'function',
    function: {
      name: 'get_capital_city',
      description: 'Retrieves the capital city of a given country.',
      parameters: {
        type: 'object',
        properties: {
          country: { type: 'string', description: 'The country name.' }
        },
        required: ['country']
      },
      zodSchema: GetCapitalToolSchema // Include Zod schema
    }
  }

  // Mock tool implementation
  async function getCapitalCity(country: string): Promise<string> {
    console.log(`[TOOL EXECUTION] Finding capital for ${country}...`)
    await new Promise(r => setTimeout(r, 100)) // Simulate delay
    const capitals: { [key: string]: string } = {
      france: 'Paris',
      japan: 'Tokyo',
      brazil: 'Brasília'
    }
    const result = capitals[country.toLowerCase()] || `Unknown country: ${country}`
    return JSON.stringify({ capital: result })
  }

  // Conversation loop for tool use
  const toolMessages: RosettaMessage[] = [{ role: 'user', content: "What's the capital of France?" }]
  const toolModel = lmstudioProviderConfig.defaultModel
  console.log(`[Tool Use] Attempting with model: ${toolModel}`)
  console.log(`(Note: Tool use support depends heavily on the specific model loaded in LM Studio)`)

  try {
    for (let i = 0; i < 3; i++) {
      // Limit turns
      console.log(`[Tool Use Turn ${i + 1}] Sending request...`)
      const response = await rosetta.generate({
        provider: lmstudioProviderKey as Provider, // Cast for GenerateParams type
        model: toolModel, // Use the LM Studio model
        messages: toolMessages,
        tools: [getCapitalTool],
        toolChoice: 'auto',
        temperature: 0.7
      })

      console.log('[Assistant Raw Response]', {
        content: response.content,
        toolCalls: response.toolCalls
      })

      toolMessages.push({ role: 'assistant', content: response.content, toolCalls: response.toolCalls })

      if (response.toolCalls && response.toolCalls.length > 0) {
        const toolResults: RosettaMessage[] = []
        for (const call of response.toolCalls) {
          if (call.function.name === 'get_capital_city') {
            let result = ''
            let isError = false
            try {
              // Arguments are validated by the mapper/SDK before returning
              const args = JSON.parse(call.function.arguments)
              result = await getCapitalCity(args.country)
              console.log(` -> Tool Result OK for call ${call.id}`)
            } catch (e) {
              isError = true
              result = JSON.stringify({ error: (e as Error).message })
              console.error(`[Tool Execution Error] ${e}`)
            }
            toolResults.push({ role: 'tool', toolCallId: call.id, content: result, isError })
          } else {
            // Handle unknown tool call
            const toolName = call.function?.name ?? 'unknown tool'
            console.warn(`[WARNING] Model called unknown/unsupported tool: ${toolName}`)
            toolResults.push({
              role: 'tool',
              toolCallId: call.id,
              content: JSON.stringify({ error: `Tool '${toolName}' is not implemented.` }),
              isError: true
            })
          }
        }
        toolMessages.push(...toolResults)
      } else {
        console.log('[Final Answer]:', response.content ?? '[No text content]')
        break // Exit loop if no tool calls
      }
    }
  } catch (error) {
    if (error instanceof ToolArgumentValidationError) {
      console.error('[Tool Argument Validation Error]', error.message, error.issues)
    } else if (error instanceof ProviderAPIError && error.statusCode === 400) {
      console.error(
        `[Tool Use Error] Received 400 Bad Request. This might indicate the model '${toolModel}' does not support tool use or the request format is incorrect for LM Studio. Check LM Studio's documentation and the loaded model's capabilities.`,
        error
      )
    } else if (error instanceof ProviderAPIError && error.message.includes('fetch failed')) {
      console.error('[Tool Use Error] Connection failed. Ensure LM Studio is running and accessible.')
    } else {
      console.error('[Tool Use Error]', error)
    }
  }

  console.log('--- LM Studio Example Complete ---')
}

runLMStudioExamples().catch(console.error)
