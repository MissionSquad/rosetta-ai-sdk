/* eslint-disable no-console */
/**
 * Custom Provider Example: Novita AI (OpenAI Compatible)
 *
 * This example demonstrates how to configure and use a custom provider
 * that is compatible with the OpenAI API standard, like Novita AI.
 *
 * It defines a simple custom mapper (`NovitaMapper`) that leverages the
 * official `openai` Node.js SDK internally, configured with Novita's
 * specific base URL and API key handling.
 *
 * Prerequisites:
 * 1. Install dependencies: `npm install`
 * 2. Add your Novita API key to `examples/.env`:
 *    NOVITA_API_KEY=your_novita_api_key
 * 3. Run this example: `npm run example:custom-novita`
 */

import dotenv from 'dotenv'
import { z } from 'zod'
import OpenAI from 'openai' // Use the official OpenAI SDK
import { Stream } from 'openai/streaming'

import {
  RosettaAI,
  RosettaMessage,
  RosettaTool,
  GenerateParams,
  GenerateResult,
  StreamChunk,
  CustomProviderConfig,
  ProviderAPIError,
  RosettaAIError,
  ToolArgumentValidationError,
  MappingError,
  Provider, // Import Provider enum
  ProviderKey // Import ProviderKey type
} from '../src' // Adjust path as needed
import { BaseCustomMapper } from '../src/core/mapping/base.custom.mapper' // Adjust path
import {
  mapFromOpenAIResponse,
  mapOpenAIStream,
  mapContentForOpenAIRole, // Import content mapping helper
  mapRoleToOpenAI, // Import role mapping helper
  wrapOpenAIError
} from '../src/core/mapping/openai.common' // Reuse common OpenAI mapping logic
import { mapBaseToolChoice } from '../src/core/mapping/common.utils' // Import tool choice mapper

// Load environment variables from examples/.env
dotenv.config({ path: '.env' })

// --- Novita AI Custom Mapper Implementation ---

class NovitaMapper extends BaseCustomMapper {
  private openaiClient: OpenAI

  constructor(config: CustomProviderConfig) {
    super(config) // Pass config to base class

    // Initialize the OpenAI client with Novita's specifics
    this.openaiClient = new OpenAI({
      apiKey: config.apiKey, // API key loaded via config/env
      baseURL: config.baseURL, // Novita's OpenAI-compatible base URL
      maxRetries: config.defaultMaxRetries ?? 2,
      timeout: config.defaultTimeoutMs ?? 60 * 1000
    })
  }

  // Helper to map RosettaMessages to OpenAI format
  private mapMessagesToOpenAI(messages: RosettaMessage[]): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    return messages.map(msg => {
      const role = mapRoleToOpenAI(msg.role)
      const content = mapContentForOpenAIRole(msg.content, role)

      // Construct the message param based on role, ensuring required fields
      switch (role) {
        case 'system':
          if (typeof content !== 'string' || content === '') {
            throw new MappingError('System message content must be a non-empty string.', this.provider)
          }
          return { role, content }
        case 'user':
          if (content === null || (Array.isArray(content) && content.length === 0)) {
            throw new MappingError('User message content cannot be empty.', this.provider)
          }
          return { role, content: content as string | OpenAI.Chat.Completions.ChatCompletionContentPart[] }
        case 'assistant':
          const assistantMsg: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam = {
            role,
            content: content as string | null
          }
          if (msg.toolCalls && msg.toolCalls.length > 0) {
            assistantMsg.tool_calls = msg.toolCalls.map(tc => ({
              id: tc.id,
              type: tc.type,
              function: { name: tc.function.name, arguments: tc.function.arguments }
            }))
            // Ensure content is null if tool calls exist and original content was null/empty
            if (content === null || content === '') {
              assistantMsg.content = null
            }
          } else if (assistantMsg.content === null) {
            // If no tool calls, content cannot be null for OpenAI assistant message
            assistantMsg.content = '' // Default to empty string if no tool calls and content was null
          }
          return assistantMsg
        case 'tool':
          if (!msg.toolCallId) {
            throw new MappingError('Tool message requires toolCallId.', this.provider)
          }
          if (typeof content !== 'string') {
            // Should ideally not happen if mapContentForOpenAIRole works correctly
            throw new MappingError('Tool message content must map to a string.', this.provider)
          }
          // Allow empty string for tool content
          return { role, tool_call_id: msg.toolCallId, content: content }
        default:
          // Should be unreachable if mapRoleToOpenAI is correct
          throw new MappingError(`Unhandled role: ${role}`, this.provider)
      }
    })
  }

  // Helper to map RosettaTools to OpenAI format
  private mapToolsToOpenAI(tools?: RosettaTool<any>[]): OpenAI.Chat.Completions.ChatCompletionTool[] | undefined {
    return tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Unsupported tool type: ${tool.type}`, this.provider)
      }
      return {
        type: tool.type,
        function: {
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters as OpenAI.FunctionDefinition['parameters'] // Basic cast
        }
      }
    })
  }

  // Helper to map Rosetta toolChoice to OpenAI format
  private mapToolChoiceToOpenAI(
    toolChoice?: GenerateParams['toolChoice']
  ): OpenAI.Chat.Completions.ChatCompletionToolChoiceOption | undefined {
    const baseChoice = mapBaseToolChoice(toolChoice)
    if (baseChoice === 'auto' || baseChoice === 'none' || baseChoice === 'required') {
      return baseChoice
    } else if (typeof baseChoice === 'object' && baseChoice.type === 'function') {
      return { type: 'function', function: { name: baseChoice.function.name } }
    }
    return undefined
  }

  // Override executeGenerate to use the configured OpenAI client
  // eslint-disable-next-line prettier/prettier 
  override async executeGenerate(
    _mappedParams: any, // We'll map within this method for simplicity
    _apiKey: string | undefined, // Already available in this.openaiClient
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): Promise<GenerateResult> {
    const model = originalParams.model ?? providerConfig.defaultModel
    if (!model) {
      throw new Error('Model must be specified for Novita AI (or set a default).')
    }

    // Map Rosetta messages/tools to OpenAI format using the helpers
    const messages = this.mapMessagesToOpenAI(originalParams.messages)
    const tools = this.mapToolsToOpenAI(originalParams.tools)
    const tool_choice = this.mapToolChoiceToOpenAI(originalParams.toolChoice)

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
      model: model,
      messages: messages,
      max_tokens: originalParams.maxTokens,
      temperature: originalParams.temperature,
      stream: false,
      tools,
      tool_choice
    }

    try {
      console.log('[NovitaMapper] Calling Novita API (via OpenAI SDK)... Params:', JSON.stringify(openAIParams))
      const response = await this.openaiClient.chat.completions.create(openAIParams)
      console.log('[NovitaMapper] Received response.')

      // Map the OpenAI SDK response back to GenerateResult using common helper
      // Pass original tools for validation within mapFromOpenAIResponse
      return mapFromOpenAIResponse(response, model, originalParams.tools)
    } catch (error) {
      // Wrap potential OpenAI SDK errors using the custom wrapper
      throw this.wrapProviderError(error, this.provider)
    }
  }

  // Override executeStream to use the configured OpenAI client
  override async *executeStream(
    _mappedParams: any,
    _apiKey: string | undefined,
    providerConfig: CustomProviderConfig,
    originalParams: GenerateParams
  ): AsyncIterable<StreamChunk> {
    const model = originalParams.model ?? providerConfig.defaultModel
    if (!model) {
      // Yield error chunk instead of throwing directly
      yield {
        type: 'error',
        data: { error: new Error('Model must be specified for Novita AI (or set a default).') }
      }
      return
    }

    // Map Rosetta messages/tools to OpenAI format using the helpers
    let messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]
    let tools: OpenAI.Chat.Completions.ChatCompletionTool[] | undefined
    let tool_choice: OpenAI.Chat.Completions.ChatCompletionToolChoiceOption | undefined
    try {
      messages = this.mapMessagesToOpenAI(originalParams.messages)
      tools = this.mapToolsToOpenAI(originalParams.tools)
      tool_choice = this.mapToolChoiceToOpenAI(originalParams.toolChoice)
    } catch (mappingError) {
      yield { type: 'error', data: { error: mappingError as Error } }
      return
    }

    const openAIParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
      model: model,
      messages: messages,
      max_tokens: originalParams.maxTokens,
      temperature: originalParams.temperature,
      stream: true,
      stream_options: { include_usage: true }, // Request usage data
      tools,
      tool_choice
    }

    try {
      console.log('[NovitaMapper] Calling Novita API for stream (via OpenAI SDK)... Params:', JSON.stringify(openAIParams))
      const stream = await this.openaiClient.chat.completions.create(openAIParams)
      console.log('[NovitaMapper] Received stream.')

      // NOTE on Streaming with OpenAI-Compatible Providers:
      // We are reusing the `mapOpenAIStream` helper here. Initial testing showed
      // Novita streams were not producing content_delta chunks. Investigation
      // revealed Novita sends `usage` data within the *same* chunk as `delta.content`.
      // The original `mapOpenAIStream` had a `continue` statement after processing
      // `chunk.usage`, which skipped the `delta.content` check for these chunks.
      //
      // FIX APPLIED (in `src/core/mapping/openai.common.ts`):
      // Removed the `continue` statement within the `if (aggregatedResult && chunk.usage)`
      // block in `mapOpenAIStream` to allow processing of other fields in the same chunk.
      //
      // FUTURE: Implement `usage_delta` chunks:
      // If intermediate usage reporting is desired, the `mapOpenAIStream` helper (or a
      // custom stream mapper) could be modified to yield a `usage_delta` chunk
      // when `chunk.usage` is encountered, e.g., by adding:
      // `yield { type: 'usage_delta', data: { usage: finalUsage } };`
      // within the `if (aggregatedResult && chunk.usage)` block.
      // // Raw Chunk from Novita
      // {
      //   "id": "chatcmpl-b806f7e408f34c589e310e2b2ce68c73",
      //   "object": "chat.completion.chunk",
      //   "created": 1744497025,
      //   "model": "meta-llama/llama-3.1-8b-instruct",
      //   "choices": [
      //     {
      //       "index": 0,
      //       "delta": {
      //         "content": "API co" // <<< Content Delta is here
      //       },
      //       "finish_reason": null,
      //       // ...
      //     }
      //   ],
      //   "system_fingerprint": "",
      //   "usage": { // <<< Usage is ALSO present in the SAME chunk
      //     "prompt_tokens": 46,
      //     "completion_tokens": 3,
      //     "total_tokens": 49,
      //     // ...
      //   }
      // }

      // Map the OpenAI SDK stream chunks using the common helper
      // Pass the provider, model ID, and original tools from originalParams
      yield* mapOpenAIStream(
        stream as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>, // Use the original stream
        Provider.OpenAI, // Use the correct Provider enum value for the common helper
        model, // Pass the determined model ID
        originalParams.tools // Pass the tools from the original params
      )
    } catch (error) {
      // Wrap potential OpenAI SDK errors using the custom wrapper
      // Errors during stream setup are caught here. Errors during iteration are handled within mapOpenAIStream.
      // Yield error chunk instead of throwing
      yield { type: 'error', data: { error: this.wrapProviderError(error, this.provider) } }
    }
  }

  // Override wrapProviderError to potentially handle Novita-specific error formats
  // if they differ significantly from standard OpenAI errors.
  override wrapProviderError(error: unknown, provider: ProviderKey): RosettaAIError {
    // If Novita returns errors exactly like OpenAI, this is sufficient.
    // If not, parse the error structure here and return a ProviderAPIError.
    // Example: Check if error has { code, reason, message } structure
    if (
      typeof error === 'object' &&
      error !== null &&
      'code' in error &&
      'reason' in error &&
      'message' in error
    ) {
      const novitaError = error as { code: number; reason: string; message: string }
      return new ProviderAPIError(
        novitaError.message,
        provider, // Use the ProviderKey directly
        novitaError.code, // Use Novita code as status if appropriate, or map it
        novitaError.reason, // Use Novita reason as errorCode
        undefined, // No specific errorType from Novita format
        error
      )
    }
    // Fallback to the common OpenAI wrapper
    console.warn('[NovitaMapper] Wrapping error using default OpenAI wrapper.')
    // **FIX:** Cast the custom provider key to Provider.OpenAI for the common wrapper
    return wrapOpenAIError(error, Provider.OpenAI)
  }

  // Other methods (embed, audio) are not implemented as Novita docs focus on completions.
  // The base class will throw UnsupportedFeatureError for them.
}

// --- Novita AI Provider Configuration ---
const novitaProviderKey = 'novita' // Unique key for this provider
const novitaProviderConfig: CustomProviderConfig = {
  providerKey: novitaProviderKey,
  mapper: NovitaMapper, // Use our custom mapper
  supportedFeatures: ['generate', 'stream', 'tool_use'], // Based on Novita docs
  baseURL: 'https://api.novita.ai/v3/openai', // Novita's OpenAI-compatible endpoint
  apiKey: process.env.NOVITA_API_KEY, // Load from environment
  // Optional: Define a default model for Novita
  defaultModel: process.env.ROSETTA_DEFAULT_NOVITA_MODEL ?? 'meta-llama/llama-3.1-8b-instruct',
  // Tool configuration matching OpenAI's expected format
  toolConfig: {
    toolDefinitionFormat: 'jsonSchema',
    toolCallInputFormat: 'jsonString',
    toolResultFormat: 'jsonString'
  }
}
// --- Example Usage ---
async function runNovitaExamples() {
  console.log('--- Custom Provider Example: Novita AI ---')
  if (!novitaProviderConfig.apiKey) {
    console.error(
      `Error: NOVITA_API_KEY environment variable not set. Please add it to examples/.env`
    )
    return
  }

  // Initialize RosettaAI with the custom provider configuration
  const rosetta = new RosettaAI({
    customProviders: [novitaProviderConfig]
    // You can also configure built-in providers here if needed
    // openaiApiKey: process.env.OPENAI_API_KEY,
  })
  console.log(`Configured providers: ${rosetta.getConfiguredProviders().join(', ')}`)
  if (!rosetta.getConfiguredProviders().includes(novitaProviderKey)) {
    console.error(`Error: Failed to register custom provider '${novitaProviderKey}'. Check configuration.`)
    return
  }
  // --- 1. Basic Generation ---
  console.log('--- 1. Basic Generation (Novita) ---')
  try {
    const result = await rosetta.generate({
      provider: novitaProviderKey as Provider, // Cast for GenerateParams type
      // model: 'meta-llama/llama-3.1-8b-instruct', // Optional: Override default
      messages: [{ role: 'user', content: 'Write a short tagline for a new AI SDK.' }],
      maxTokens: 50
    })
    console.log(`[${result.model}] Response: ${result.content}`)
    console.log(`Usage: ${JSON.stringify(result.usage)}`)
  } catch (error) {
    console.error('[Basic Generation Error]', error)
  }

  // --- 2. Streaming Generation ---
  console.log('--- 2. Streaming Generation (Novita) ---')
  let fullStreamedContent = ''
  let finalStreamResult: GenerateResult | null = null
  try {
    const stream = rosetta.stream({
      provider: novitaProviderKey as Provider, // Cast for GenerateParams type
      messages: [{ role: 'user', content: 'Explain the concept of "API compatibility" briefly.' }],
      maxTokens: 50
    })

    process.stdout.write(`[${novitaProviderConfig.defaultModel} Stream] `)
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
  console.log('--- 3. Tool Use (Novita) ---')
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
  // **FIX:** Use a model known to support function calling according to Novita docs
  const toolModel = 'deepseek/deepseek-v3-0324' // Example model from Novita docs
  console.log(`[Tool Use] Attempting with model: ${toolModel}`)

  try {
    for (let i = 0; i < 3; i++) { // Limit turns
      console.log(`[Tool Use Turn ${i + 1}] Sending request...`)
      const response = await rosetta.generate({
        provider: novitaProviderKey as Provider, // Cast for GenerateParams type
        model: toolModel, // **FIX:** Explicitly set the tool-compatible model
        messages: toolMessages,
        tools: [getCapitalTool],
        toolChoice: 'auto'
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
        `[Tool Use Error] Received 400 Bad Request. This might indicate the model '${toolModel}' does not support tool use or the request format is incorrect for Novita. Check Novita's documentation for supported models and API details.`,
        error
      )
    } else {
      console.error('[Tool Use Error]', error)
    }
  }

  console.log('--- Novita AI Example Complete ---')
}

runNovitaExamples().catch(console.error)
