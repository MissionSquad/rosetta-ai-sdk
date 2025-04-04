/* eslint-disable no-console */
// Tool Use (Function Calling) Example - Non-Streaming
import { RosettaAI, Provider, RosettaTool, RosettaMessage, RosettaAIError } from '../src'
import dotenv from 'dotenv'

dotenv.config()

// 1. Define Tool & Implementation
const getWeatherTool: RosettaTool = {
  type: 'function',
  function: {
    name: 'getCurrentWeather',
    description: 'Get the current weather conditions for a specific location.',
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: 'The city and state/country, e.g., San Francisco, CA or London, UK'
        },
        unit: {
          type: 'string',
          enum: ['celsius', 'fahrenheit'],
          description: 'The temperature unit to use.'
        }
      },
      required: ['location'] // Location is required
    }
  }
}

// Simple async function simulating an API call for weather
async function getCurrentWeather(location: string, unit: string = 'fahrenheit'): Promise<string> {
  console.log(`[TOOL EXECUTION] Fetching weather for ${location} in ${unit}...`)
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300))

  // Simulate different responses based on location
  if (location.toLowerCase().includes('tokyo')) {
    const temperature = unit === 'celsius' ? 15 : 59
    return JSON.stringify({ location, temperature, unit, condition: 'Cloudy with a chance of rain.' })
  }
  if (location.toLowerCase().includes('bogota')) {
    // Simulate an error case
    console.error(`[TOOL EXECUTION] Error: API unavailable for Bogota.`)
    throw new Error(`Weather API service is currently unavailable for Bogota.`)
  }
  // Default response
  const temperature = unit === 'celsius' ? 22 : 72
  return JSON.stringify({ location, temperature, unit, condition: 'Sunny and pleasant.' })
}

// REMOVED: Unused findLastToolCallName function.
// This function is used internally by the SDK's Google mapper, but is not needed in the example itself.

// 2. Conversation Loop Logic
async function runToolUseChat(initialPrompt: string) {
  console.log(`\n--- Tool Use Example Starting With: "${initialPrompt}" ---`)
  const rosetta = new RosettaAI()

  // Filter providers that support tool use (adjust if needed)
  const providers = rosetta
    .getConfiguredProviders()
    .filter(p => [Provider.OpenAI, Provider.Anthropic, Provider.Google, Provider.Groq].includes(p))

  if (providers.length === 0) {
    console.error('No configured providers support tool use (OpenAI, Anthropic, Google, Groq needed).')
    return
  }

  // Use the first available tool-supporting provider
  const provider: Provider = providers[0]!
  console.log(`Using provider: ${provider}`)

  // Initial conversation message
  const messages: RosettaMessage[] = [{ role: 'user', content: initialPrompt }]

  try {
    // Get default model or fallback
    const model =
      rosetta.config.defaultModels?.[provider] ??
      (provider === Provider.Anthropic
        ? 'claude-3-sonnet-20240229' // Sonnet or Opus recommended for tools
        : provider === Provider.Google
        ? 'gemini-1.5-flash-latest' // Or gemini-1.5-pro-latest
        : provider === Provider.Groq
        ? 'llama3-70b-8192' // Larger Groq models might be better
        : provider === Provider.OpenAI
        ? 'gpt-4o-mini' // Or gpt-4o
        : undefined)

    if (!model) {
      console.log(`Cannot run tool use test for ${provider}: No default model configured or fallback available.`)
      return
    }
    console.log(`Using model: ${model}`)

    // Limit iterations to prevent infinite loops
    const maxIterations = 5
    for (let i = 0; i < maxIterations; i++) {
      console.log(`\n[Iteration ${i + 1}] Sending messages to ${provider}...`)
      // console.log("Current Messages:", JSON.stringify(messages, null, 2)); // Debug: show messages being sent

      const response = await rosetta.generate({
        provider,
        model,
        messages,
        tools: [getWeatherTool], // Provide the tool definition
        toolChoice: 'auto' // Let the model decide ('auto' is often default)
      })

      console.log('[Assistant Raw Response]', {
        finishReason: response.finishReason,
        content: response.content,
        toolCalls: response.toolCalls?.map(tc => ({ id: tc.id, name: tc.function.name, args: tc.function.arguments })) // Log tool call info
      })

      // Add the assistant's response (content and potential tool calls) to the history
      messages.push({
        role: 'assistant',
        content: response.content, // Content might be null if only tool calls are made
        toolCalls: response.toolCalls
      })

      // Check if the model requested tool calls
      if (response.toolCalls && response.toolCalls.length > 0) {
        console.log(`[Tool calls requested: ${response.toolCalls.length}]`)
        const toolResultMessages: RosettaMessage[] = []

        // Execute each requested tool call
        await Promise.all(
          response.toolCalls.map(async call => {
            if (call.type === 'function' && call.function.name === 'getCurrentWeather') {
              let toolResultContent: string
              try {
                // Parse arguments provided by the model
                const args = JSON.parse(call.function.arguments)
                // Execute the actual tool function
                toolResultContent = await getCurrentWeather(args.location, args.unit)
                console.log(` -> Tool Result OK for call ${call.id}`)
              } catch (e) {
                // Handle errors during argument parsing or tool execution
                const errorMessage = e instanceof Error ? e.message : String(e)
                console.error(` -> Tool Error for call ${call.id}:`, errorMessage)
                // Return error information back to the model
                toolResultContent = JSON.stringify({ error: errorMessage })
              }
              // Create a 'tool' role message with the result
              toolResultMessages.push({
                role: 'tool',
                toolCallId: call.id, // Link result to the specific call ID
                content: toolResultContent
              })
            } else {
              // Handle cases where the model calls an unknown or unsupported tool
              const toolName = call.function?.name ?? 'unknown tool'
              console.warn(`[WARNING] Model called unknown/unsupported tool: ${toolName}`)
              toolResultMessages.push({
                role: 'tool',
                toolCallId: call.id,
                content: JSON.stringify({ error: `Tool '${toolName}' is not implemented.` })
              })
            }
          })
        )

        // Add all tool results to the message history for the next turn
        messages.push(...toolResultMessages)
      } else {
        // If no tool calls, the model provided a final answer
        console.log('\n[Final Response from Assistant]:')
        console.log(response.content ?? '[No text content provided]')
        console.log('\nUsage:', response.usage ? JSON.stringify(response.usage) : 'N/A')
        break // Exit the loop
      }

      // Safety break if max iterations are reached
      if (i === maxIterations - 1) {
        console.warn('[WARNING] Maximum conversation iterations reached.')
      }
    } // End of loop
  } catch (error) {
    if (error instanceof RosettaAIError) {
      console.error(`Error during tool use chat with ${provider}: ${error.name} - ${error.message}`)
    } else {
      console.error(`Unexpected error during tool use chat with ${provider}:`, error)
    }
  } finally {
    console.log('--- Tool Use Example Complete ---')
  }
}

// 3. Run Different Scenarios
async function runAllToolTests() {
  await runToolUseChat("What's the weather like in Tokyo? Please use Celsius.")
  await new Promise(r => setTimeout(r, 1500)) // Delay between tests
  await runToolUseChat('Can you tell me the weather in Bogota?')
  await new Promise(r => setTimeout(r, 1500)) // Delay between tests
  await runToolUseChat('Hi, how are you today?') // Scenario where the tool shouldn't be called
}

// Execute the tests
runAllToolTests().catch(err => console.error('Unhandled error in tool use example script:', err))
