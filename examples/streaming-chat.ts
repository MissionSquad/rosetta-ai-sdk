/* eslint-disable no-console */
// Streaming Chat Completion Example
import { RosettaAI, Provider, RosettaAIError, ProviderAPIError, GenerateResult } from '../src'
import dotenv from 'dotenv'

dotenv.config()

async function runStreamingChat() {
  const rosetta = new RosettaAI()
  const providers = rosetta.getConfiguredProviders()

  if (providers.length === 0) {
    console.error('No providers configured.')
    return
  }

  // Select a provider known to work well with streaming (e.g., Anthropic or OpenAI)
  // Or just pick the first available one for demonstration
  let providerToTest: Provider | undefined
  if (providers.includes(Provider.Anthropic)) providerToTest = Provider.Anthropic
  else if (providers.includes(Provider.OpenAI)) providerToTest = Provider.OpenAI
  else if (providers.includes(Provider.Groq)) providerToTest = Provider.Groq
  else if (providers.includes(Provider.Google)) providerToTest = Provider.Google
  else {
    console.error('No suitable provider found for streaming test (Anthropic, OpenAI, Groq, Google preferred).')
    return
  }

  console.log(`--- Testing Streaming with: ${providerToTest} ---`)

  try {
    // Use default model or fallback
    const model =
      rosetta.config.defaultModels?.[providerToTest] ??
      (providerToTest === Provider.Anthropic
        ? 'claude-3-haiku-20240307'
        : providerToTest === Provider.Google
        ? 'gemini-1.5-flash-latest'
        : providerToTest === Provider.Groq
        ? 'llama3-8b-8192'
        : providerToTest === Provider.OpenAI
        ? 'gpt-4o-mini'
        : undefined)

    if (!model) {
      console.log(`Cannot run streaming test for ${providerToTest}: No default model configured or fallback available.`)
      return
    }
    console.log(`Using model: ${model}`)

    // Define the prompt
    const prompt =
      providerToTest === Provider.Anthropic
        ? 'Explain quantum entanglement step-by-step using an analogy.'
        : 'Write a short, funny story about a cat learning Python and encountering a bug.'

    // Enable thinking steps only for Anthropic if desired
    const enableThinking = false //providerToTest === Provider.Anthropic

    // FIX: Increase maxTokens when thinking is enabled for Anthropic
    // Anthropic requires max_tokens > thinking.budget_tokens (which defaults to 1024 in the mapper)
    const maxTokensForRequest = enableThinking ? 1200 : 200

    // Initiate the stream
    const stream = rosetta.stream({
      provider: providerToTest,
      model: model,
      messages: [{ role: 'user', content: prompt }],
      maxTokens: maxTokensForRequest, // Use adjusted maxTokens
      thinking: enableThinking
    })

    console.log(`[${providerToTest} Streaming Response]`)
    let fullContent = ''
    let finalResult: GenerateResult | undefined
    let currentToolArgs = '' // Accumulator for tool arguments delta

    // Process the stream chunks
    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'message_start':
          console.log(`\nStream started (Model: ${chunk.data.model})...`)
          break
        case 'content_delta':
          process.stdout.write(chunk.data.delta) // Write text delta directly
          fullContent += chunk.data.delta
          break
        case 'thinking_start':
          process.stdout.write('\n[Thinking...]')
          break
        case 'thinking_delta':
          process.stdout.write('.') // Indicate thinking progress
          break
        case 'thinking_stop':
          process.stdout.write('[Done Thinking]\n')
          break
        case 'tool_call_start':
          console.log(
            `\n[Tool Call Start #${chunk.data.index}: ${chunk.data.toolCall.function.name} (ID: ${chunk.data.toolCall.id})]`
          )
          console.log(`  Args (streaming): `)
          currentToolArgs = '' // Reset for new tool call
          break
        case 'tool_call_delta':
          // Accumulate and print argument chunks
          process.stdout.write(chunk.data.functionArgumentChunk)
          currentToolArgs += chunk.data.functionArgumentChunk
          break
        case 'tool_call_done':
          // Attempt to parse the accumulated arguments when the call is done
          try {
            const parsedArgs = JSON.parse(currentToolArgs)
            console.log(
              `\n[Tool Call Args Complete #${chunk.data.index} (ID: ${chunk.data.id})]\n  Parsed:`,
              parsedArgs
            )
          } catch {
            console.log(
              `\n[Tool Call Args Complete #${chunk.data.index} (ID: ${chunk.data.id})]\n  Raw: ${currentToolArgs}`
            )
          }
          break
        case 'json_delta':
          // Handle streaming JSON (clear console for better view)
          console.clear()
          console.log('Streaming JSON Snapshot:', chunk.data.snapshot)
          console.log('Partial Parse Attempt:', chunk.data.parsed)
          break
        case 'json_done':
          console.clear()
          console.log('\nFinal Parsed JSON:', chunk.data.parsed)
          console.log('Final JSON Snapshot:', chunk.data.snapshot)
          fullContent = chunk.data.snapshot // Update full content with final JSON
          break
        case 'citation_delta':
        case 'citation_done':
          console.log(
            `\n[Citation ${chunk.data.index} ${chunk.type === 'citation_done' ? 'Complete' : 'Update'}: Source ID ${
              chunk.data.citation.sourceId
            }, Start: ${chunk.data.citation.startIndex ?? 'N/A'}, End: ${chunk.data.citation.endIndex ?? 'N/A'}]`
          )
          break
        case 'message_stop':
          console.log(`\n--- Stream Stopped (Reason: ${chunk.data.finishReason}) ---`)
          break
        case 'final_usage':
          console.log('\nFinal Usage:', chunk.data.usage)
          break
        case 'final_result':
          // The final aggregated result object
          finalResult = chunk.data.result
          console.log('\n--- Final Result Object Received ---')
          // You can access finalResult.content, finalResult.toolCalls etc. here
          break
        case 'error':
          console.error('\n--- Stream Error ---')
          console.error(`${chunk.data.error.name}: ${chunk.data.error.message}`)
          if (chunk.data.error instanceof ProviderAPIError) {
            // console.error("Provider Error Details:", chunk.data.error.underlyingError);
          }
          console.error('--------------------')
          // Depending on the error, you might want to break the loop
          return // Exit function on stream error
        default:
          // Ensure all chunk types are handled (compile-time check)
          const _: never = chunk
          console.log(`\nUnknown chunk type encountered: ${(_ as any).type}`)
      }
    }
    console.log('\n--- End of Stream ---')
    console.log('Final Accumulated Content:', fullContent)
    // Clarification: finalResult contains the complete response object after the stream finishes.
    // It aggregates all content, tool calls, usage, etc., mirroring the non-streaming GenerateResult.
    // Useful if you need the structured response object after processing the stream deltas.
    console.log(
      'Final Aggregated Result Object (if received):',
      finalResult ? JSON.stringify(finalResult, null, 2) : 'N/A'
    )
  } catch (error) {
    // Catch errors during stream setup (e.g., invalid model, auth error)
    if (error instanceof RosettaAIError) {
      console.error(`\nError setting up stream for ${providerToTest}: ${error.name} - ${error.message}`)
    } else {
      console.error(`\nUnexpected error setting up stream for ${providerToTest}:`, error)
    }
  }
}

runStreamingChat().catch(err => console.error('Unhandled error in example script:', err))
