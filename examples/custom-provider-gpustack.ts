/* eslint-disable no-console */
/**
 * Custom Provider Example: Gpustack (OpenAI Compatible)
 *
 * This example demonstrates how to configure and use a custom provider
 * that is compatible with the OpenAI API standard, like Gpustack,
 * using the refactored `OpenAICompatibleMapper`.
 *
 * Prerequisites:
 * 1. Install dependencies: `yarn install` (in the examples directory)
 * 2. Add your Gpustack API key to `examples/.env`:
 *    GPUSTACK_API_KEY=your_gpustack_api_key
 * 3. Run this example: `yarn example:custom-gpustack`
 */

import dotenv from 'dotenv'
import fs from 'fs'
import path from 'path'
import { z } from 'zod'

import {
  RosettaAI,
  RosettaMessage,
  RosettaTool,
  GenerateResult,
  CustomProviderConfig,
  ProviderAPIError,
  ToolArgumentValidationError,
  Provider,
  EmbedParams,
  EmbedResult
} from '../src'
import { OpenAICompatibleMapper } from '../src/core/mapping/openai-compatible.mapper'

// Load environment variables from examples/.env
dotenv.config({ path: '.env' })

// --- Gpustack OpenAI Compatible Mapper Implementation ---
const gpustackProviderKey = 'gpustack' // Unique key for this provider
const gpustackProviderConfig: CustomProviderConfig = {
  providerKey: gpustackProviderKey,
  mapper: OpenAICompatibleMapper, // Use the reusable OpenAI-compatible mapper
  supportedFeatures: ['generate', 'stream', 'tool_use', 'embed', 'image_input'], // Features supported by Gpustack's OpenAI endpoint
  // baseURL: process.env.GPUSTACK_BASE_URL, // Gpustack's OpenAI-compatible endpoint, loaded from environment
  apiKey: process.env.GPUSTACK_API_KEY, // Loaded from environment
  // Define a default model for Gpustack
  defaultModel: process.env.ROSETTA_DEFAULT_GPUSTACK_MODEL ?? 'qwen2.5-coder-14b-instruct',
  // Define a default embedding model for Gpustack
  defaultEmbeddingModel: process.env.ROSETTA_DEFAULT_EMBEDDING_GPUSTACK_MODEL ?? 'nomic-embed-text-v1.5',
  // Tool configuration matching OpenAI's expected format (default for OpenAICompatibleMapper)
  toolConfig: {
    toolDefinitionFormat: 'jsonSchema',
    toolCallInputFormat: 'jsonString',
    toolResultFormat: 'jsonString'
  }
}
// --- Example Usage ---
async function runGpustackExamples() {
  console.log('--- Custom Provider Example: Gpustack ---')
  if (!gpustackProviderConfig.apiKey) {
    console.error(`Error: GPUSTACK_API_KEY environment variable not set. Please add it to examples/.env`)
    return
  }

  // Initialize RosettaAI with the custom provider configuration
  const rosetta = new RosettaAI({
    customProviders: [gpustackProviderConfig]
  })

  console.log(`Configured providers: ${rosetta.getConfiguredProviders().join(', ')}`)
  if (!rosetta.getConfiguredProviders().includes(gpustackProviderKey)) {
    console.error(`Error: Failed to register custom provider '${gpustackProviderKey}'. Check configuration.`)
    return
  }
  // --- 1. Basic Generation ---
  console.log('--- 1. Basic Generation (Gpustack) ---')
  try {
    const result = await rosetta.generate({
      provider: gpustackProviderKey as Provider, // Cast for GenerateParams type
      model: gpustackProviderConfig.defaultModel, // Use the default model
      messages: [{ role: 'user', content: 'Write a short tagline for a new AI SDK.' }],
      maxTokens: 50,
      temperature: 1, // from curl example
      topP: 1 // from curl example
    })
    console.log(`[${result.model}] Response: ${result.content}`)
    console.log(`Usage: ${JSON.stringify(result.usage)}`)
  } catch (error) {
    console.error('[Basic Generation Error]', error)
  }

  // --- 2. Streaming Generation ---
  console.log('--- 2. Streaming Generation (Gpustack) ---')
  let fullStreamedContent = ''
  let finalStreamResult: GenerateResult | null = null
  try {
    const stream = rosetta.stream({
      provider: gpustackProviderKey as Provider, // Cast for GenerateParams type
      model: gpustackProviderConfig.defaultModel,
      messages: [{ role: 'user', content: 'Explain the concept of "API compatibility" briefly.' }],
      maxTokens: 50,
      temperature: 1,
      topP: 1
    })

    process.stdout.write(`[${gpustackProviderConfig.defaultModel} Stream] `)
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
  }

  // --- 3. Tool Use ---
  console.log('--- 3. Tool Use (Gpustack) ---')
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
  const toolModel = gpustackProviderConfig.defaultModel
  console.log(`[Tool Use] Attempting with model: ${toolModel}`)

  try {
    for (let i = 0; i < 3; i++) {
      // Limit turns
      console.log(`[Tool Use Turn ${i + 1}] Sending request...`)
      const response = await rosetta.generate({
        provider: gpustackProviderKey as Provider, // Cast for GenerateParams type
        model: toolModel, // Use the Gpustack model
        messages: toolMessages,
        tools: [getCapitalTool],
        toolChoice: 'auto',
        temperature: 1,
        topP: 1
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
        `[Tool Use Error] Received 400 Bad Request. This might indicate the model '${toolModel}' does not support tool use or the request format is incorrect for Gpustack. Check Gpustack's documentation for supported models and API details.`,
        error
      )
    } else {
      console.error('[Tool Use Error]', error)
    }
  }

  // --- 4. Embeddings ---
  console.log('--- 4. Embeddings (Gpustack) ---')
  try {
    const embedParams: EmbedParams = {
      provider: gpustackProviderKey as Provider, // Cast for EmbedParams type
      input: ['This is the first text.', 'This is the second text to embed.']
    }
    const result: EmbedResult = await rosetta.embed(embedParams)

    console.log(`[${result.model}] Generated ${result.embeddings.length} embeddings.`)
    result.embeddings.forEach((vec, i) => {
      console.log(`  Embedding ${i + 1} (Dim: ${vec.length}): [${vec.slice(0, 3).join(', ')}...]`)
    })
    console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
  } catch (error) {
    console.error('[Embeddings Error]', error)
    if (error instanceof ProviderAPIError && error.statusCode === 404) {
      console.error(
        `Hint: Ensure the embedding model ('nomic-embed-text-v1.5' or configured default) is available on your Gpustack endpoint.`
      )
    }
  }

  // --- 5. Image Input ---
  console.log('--- 5. Image Input (Gpustack) ---')
  try {
    // Define the path to the image. Assumes 'document.png' is in the 'examples' directory.
    const imagePath = path.join(__dirname, 'document.png')

    // Check if the image file exists before proceeding
    if (!fs.existsSync(imagePath)) {
      console.warn(`[Image Input] Warning: Image file not found at ${imagePath}. Skipping example.`)
    } else {
      // Read the image file and convert it to a base64 string
      const imageBuffer = fs.readFileSync(imagePath)
      const base64Data = imageBuffer.toString('base64')
      const mimeType = 'image/png' // The MIME type for the image

      // Define the system prompt for the vision task
      const systemPrompt =
        'Extract the text from the above document as if you were reading it naturally. Diagrams can be represented as text descriptions with simple text line drawings.'

      // NOTE: Ensure the model used supports vision/image inputs.
      // 'llava-v1.6-34b' is a common type of vision model, but you may need to
      // replace it with the specific vision-capable model available on your Gpustack endpoint.
      const visionModel = 'nanonets-ocr-s'
      console.log(`[Image Input] Attempting with model: ${visionModel}`)

      const response = await rosetta.generate({
        provider: gpustackProviderKey as Provider,
        model: visionModel,
        messages: [
          {
            role: 'system',
            content: systemPrompt
          },
          {
            role: 'user',
            content: [
              {
                type: 'image',
                image: {
                  mimeType,
                  base64Data
                }
              }
            ]
          }
        ],
        maxTokens: 4096 // Allow for a longer response to extract text
      })

      console.log(`[Image Input Response]:\n${response.content}`)
      console.log(`[Image Input Usage]: ${JSON.stringify(response.usage)}`)
    }
  } catch (error) {
    console.error('[Image Input Error]', error)
    if (error instanceof ProviderAPIError) {
      console.error(
        `Hint: Ensure the vision model is correct and available on your Gpustack endpoint and that the API key has the correct permissions.`
      )
    }
  }

  console.log('--- Gpustack Example Complete ---')
}

runGpustackExamples().catch(console.error)
