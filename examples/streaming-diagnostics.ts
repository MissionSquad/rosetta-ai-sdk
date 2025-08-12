/* eslint-disable no-console */
// Streaming Diagnostics Script
import { RosettaAI, Provider, RosettaAIError, ProviderAPIError, StreamChunk } from '../src'
import dotenv from 'dotenv'

dotenv.config()

// Function to add a delay
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function runStreamingDiagnostics() {
  const rosetta = new RosettaAI()
  const configuredProviders = rosetta.getConfiguredProviders();
  const providersToTest: (Provider | string)[] = [
    Provider.OpenAI,
    Provider.Anthropic,
    Provider.Google,
    Provider.Groq
  ].filter(p => configuredProviders.includes(p as Provider));

  if (providersToTest.length === 0) {
    console.error('No providers configured in the .env file. Please add API keys for OpenAI, Anthropic, Google, or Groq.');
    return;
  }

  for (const provider of providersToTest) {
    console.log(`\n\n=================================================`);
    console.log(`--- Running Diagnostics for: ${provider.toUpperCase()} ---`);
    console.log(`=================================================\n`);

    try {
      const model = rosetta.config.defaultModels?.[provider as Provider] ?? getFallbackModel(provider as Provider);

      if (!model) {
        console.error(`No default model configured or fallback available for ${provider}. Skipping.`);
        continue;
      }

      console.log(`Using model: ${model}`);

      const stream = rosetta.stream({
        provider: provider as Provider,
        model: model,
        messages: [{ role: 'user', content: 'Tell me a short story about a robot who discovers music.' }],
        maxTokens: 50,
      });

      let chunkCount = 0;
      const startTime = Date.now();

      for await (const chunk of stream) {
        chunkCount++;
        const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
        logChunk(chunk, chunkCount, elapsedTime, provider as Provider);
      }

      console.log(`\n--- Diagnostics Complete for ${provider} ---`);
      console.log(`Total chunks received: ${chunkCount}`);
      console.log(`Total time: ${((Date.now() - startTime) / 1000).toFixed(2)}s`);

    } catch (error) {
      console.error(`\n--- ERROR during ${provider} test ---`);
      if (error instanceof RosettaAIError) {
        console.error(`RosettaAI Error: ${error.name} - ${error.message}`);
        if (error instanceof ProviderAPIError) {
          console.error("Underlying Provider Error:", error.underlyingError);
        }
      } else {
        console.error('Unexpected Error:', error);
      }
    }
    // Add a delay between provider tests to avoid rate limiting
    await delay(2000);
  }
}

function getFallbackModel(provider: Provider): string | undefined {
    switch (provider) {
        case Provider.Anthropic: return 'claude-3-haiku-20240307';
        case Provider.Google: return 'gemini-1.5-flash-latest';
        case Provider.Groq: return 'llama3-8b-8192';
        case Provider.OpenAI: return 'gpt-4o-mini';
        default: return undefined;
    }
}

function logChunk(chunk: StreamChunk, count: number, time: string, provider: Provider) {
    console.log(`\n[${provider}] Chunk #${count} | Time: ${time}s`);
    console.log(`  - Type: ${chunk.type}`);
    
    // Log the raw data of the chunk for detailed analysis, if it exists
    if ('data' in chunk) {
      console.log(`  - Data: ${JSON.stringify(chunk.data, null, 2)}`);
    } else {
      console.log(`  - Data: (No data property for this chunk type)`);
    }

    // Specific handling for content delta to see the text stream
    if (chunk.type === 'content_delta') {
        console.log(`  - Text Delta: "${chunk.data.delta}"`);
    }
    if (chunk.type === 'tool_call_delta') {
        console.log(`  - Tool Args Delta: "${chunk.data.functionArgumentChunk}"`);
    }
}


runStreamingDiagnostics().catch(err => {
  console.error('\nFATAL SCRIPT ERROR:', err)
});
