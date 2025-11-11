/**
 * OpenAI Responses API Example
 *
 * Demonstrates the use of OpenAI's Responses API through RosettaAI.
 * This is a separate, stateful interface designed for agent-ready interactions.
 *
 * Key differences from Chat Completions:
 * - Stateful conversations via previous_response_id (no history replay)
 * - Separates instructions (developer intent) from input (user content)
 * - Built-in tools (web_search, file_search, image_generation, code_interpreter)
 * - Semantic streaming events (not just content deltas)
 */

import { RosettaAI, Provider, CreateResponseParams, ResponsesTool } from '../src'
import { z } from 'zod'
import dotenv from 'dotenv'

dotenv.config()

async function main() {
  // Initialize RosettaAI with OpenAI credentials
  const rosetta = new RosettaAI({
    openaiApiKey: process.env.OPENAI_API_KEY
  })

  console.log('\n=== OpenAI Responses API Examples ===\n')

  // Example 1: Basic Response
  await basicResponse(rosetta)

  // Example 2: Stateful Conversation
  await statefulConversation(rosetta)

  // Example 3: Built-in Tools (Web Search)
  await builtInTools(rosetta)

  // Example 4: Custom Function Tools
  await customFunctionTools(rosetta)

  // Example 5: Structured Output
  await structuredOutput(rosetta)

  // Example 6: Streaming Response
  await streamingResponse(rosetta)

  // Example 7: Multimodal Input
  await multimodalInput(rosetta)
}

/**
 * Example 1: Basic Response
 * Simple request with instructions and input
 */
async function basicResponse(rosetta: RosettaAI) {
  console.log('--- Example 1: Basic Response ---')

  try {
    const result = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      instructions: 'You are a helpful assistant that provides concise answers.',
      input: 'What is the capital of France?',
      max_tokens: 50
    })

    console.log(`Response ID: ${result.id}`)
    console.log(`Output: ${result.output_text}`)
    console.log(`Usage: ${JSON.stringify(result.usage)}`)
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 2: Stateful Conversation
 * Chain multiple turns without replaying history
 */
async function statefulConversation(rosetta: RosettaAI) {
  console.log('--- Example 2: Stateful Conversation ---')

  try {
    // First turn
    const turn1 = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      instructions: 'You are a helpful math tutor.',
      input: 'What is 15 + 27?',
      max_tokens: 100
    })

    console.log(`Turn 1 Response ID: ${turn1.id}`)
    console.log(`Turn 1 Output: ${turn1.output_text}`)

    // Second turn - reference previous response
    const turn2 = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      instructions: 'You are a helpful math tutor.',
      input: 'Now multiply that result by 3',
      previous_response_id: turn1.id, // Stateful - no need to replay history!
      max_tokens: 100
    })

    console.log(`\nTurn 2 Response ID: ${turn2.id}`)
    console.log(`Turn 2 Output: ${turn2.output_text}`)
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 3: Built-in Tools (Web Search)
 * Use OpenAI's built-in web search tool
 */
async function builtInTools(rosetta: RosettaAI) {
  console.log('--- Example 3: Built-in Tools (Web Search) ---')

  try {
    const result = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o',
      instructions: 'You are a helpful assistant with access to web search.',
      input: 'What are the latest TypeScript 5.5 features?',
      tools: [
        { type: 'web_search' }
      ],
      tool_choice: 'auto',
      max_tokens: 500
    })

    console.log(`Response ID: ${result.id}`)
    console.log(`Output: ${result.output_text}`)

    if (result.tool_calls) {
      console.log(`\nTool calls made: ${result.tool_calls.length}`)
      for (const call of result.tool_calls) {
        console.log(`  - ${call.function.name}`)
      }
    }
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 4: Custom Function Tools
 * Define and use custom functions
 */
async function customFunctionTools(rosetta: RosettaAI) {
  console.log('--- Example 4: Custom Function Tools ---')

  // Define a custom function tool with Zod schema for validation
  const getWeatherTool: ResponsesTool = {
    type: 'function',
    name: 'getCurrentWeather',
    description: 'Get the current weather for a location',
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: 'City and state/country (e.g., "San Francisco, CA")'
        },
        unit: {
          type: 'string',
          enum: ['celsius', 'fahrenheit'],
          description: 'Temperature unit'
        }
      },
      required: ['location']
    },
    zodSchema: z.object({
      location: z.string(),
      unit: z.enum(['celsius', 'fahrenheit']).optional()
    })
  }

  try {
    const result = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      instructions: 'You are a helpful assistant that can check the weather.',
      input: "What's the weather like in San Francisco?",
      tools: [getWeatherTool],
      tool_choice: 'auto',
      max_tokens: 200
    })

    console.log(`Response ID: ${result.id}`)
    console.log(`Output: ${result.output_text}`)

    if (result.tool_calls && result.tool_calls.length > 0) {
      console.log('\nTool calls requested:')
      for (const call of result.tool_calls) {
        console.log(`  Function: ${call.function.name}`)
        console.log(`  Arguments: ${call.function.arguments}`)

        // Parse and validate arguments
        try {
          const args = JSON.parse(call.function.arguments)
          console.log(`  Parsed: ${JSON.stringify(args, null, 2)}`)
        } catch (e) {
          console.error('  Failed to parse arguments')
        }
      }

      // In a real application, you would:
      // 1. Execute the function
      // 2. Send the result back in a new request with previous_response_id
      console.log('\n(In production: execute function and continue conversation)')
    }
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 5: Structured Output
 * Request JSON output with a specific schema
 */
async function structuredOutput(rosetta: RosettaAI) {
  console.log('--- Example 5: Structured Output ---')

  try {
    const result = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o',
      instructions: 'Extract package information from the user input.',
      input: 'I need to install typescript version 5.5.4',
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'PackageInfo',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              package_name: { type: 'string' },
              version: { type: 'string' },
              install_command: { type: 'string' }
            },
            required: ['package_name', 'version', 'install_command'],
            additionalProperties: false
          }
        }
      },
      max_tokens: 200
    })

    console.log(`Response ID: ${result.id}`)
    console.log(`Output: ${result.output_text}`)

    // Parse the structured output
    try {
      const parsed = JSON.parse(result.output_text)
      console.log('\nParsed structured output:')
      console.log(JSON.stringify(parsed, null, 2))
    } catch (e) {
      console.error('Failed to parse JSON output')
    }
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 6: Streaming Response
 * Process semantic events as they arrive
 */
async function streamingResponse(rosetta: RosettaAI) {
  console.log('--- Example 6: Streaming Response ---')

  try {
    const stream = rosetta.streamResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o-mini',
      instructions: 'You are a helpful assistant.',
      input: 'Write a haiku about TypeScript',
      max_tokens: 100
    })

    console.log('Streaming output:')
    let fullText = ''

    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'response.created':
          console.log(`\n[Stream started - ID: ${chunk.data.id}]`)
          break

        case 'response.output_text.delta':
          process.stdout.write(chunk.data.delta)
          fullText += chunk.data.delta
          break

        case 'response.output_text.done':
          console.log('\n[Text output complete]')
          break

        case 'response.completed':
          console.log(`\n[Stream completed]`)
          console.log(`Final usage: ${JSON.stringify(chunk.data.usage)}`)
          break

        case 'response.failed':
          console.error(`\n[Stream failed]: ${chunk.data.error.message}`)
          break

        case 'error':
          console.error(`\n[Error]: ${chunk.data.error.message}`)
          break

        default:
          console.log(`\n[Event: ${(chunk as any).type}]`)
      }
    }

    console.log(`\nFull accumulated text: ${fullText}`)
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

/**
 * Example 7: Multimodal Input
 * Send images along with text
 */
async function multimodalInput(rosetta: RosettaAI) {
  console.log('--- Example 7: Multimodal Input ---')

  try {
    // Example with base64 encoded image
    const result = await rosetta.createResponse({
      provider: Provider.OpenAI,
      model: 'gpt-4o',
      instructions: 'You are a helpful assistant that can analyze images.',
      input: [
        { type: 'input_text', text: 'What is in this image?' },
        {
          type: 'input_image',
          image: {
            mimeType: 'image/png',
            base64Data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
          }
        }
      ],
      max_tokens: 200
    })

    console.log(`Response ID: ${result.id}`)
    console.log(`Output: ${result.output_text}`)
  } catch (error) {
    console.error('Error:', error)
  }

  console.log()
}

// Run the examples
main().catch(console.error)
