# RosettaAI SDK

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![npm version](https://badge.fury.io/js/@missionsquad/RosettaAI.svg)](https://badge.fury.io/js/@missionsquad/RosettaAI)
[![Node.js CI](https://github.com/MissionSquad/rosetta-ai-sdk/actions/workflows/node.js.yml/badge.svg)](https://github.com/MissionSquad/rosetta-ai-sdk/actions/workflows/node.js.yml)

**RosettaAI** is a unified TypeScript SDK designed for seamless interaction with multiple Large Language Model (LLM) providers like OpenAI (including Azure), Anthropic, Google Generative AI, and Groq. It provides a consistent interface (`generate`, `stream`, `embed`, `generateSpeech`, `transcribe`, etc.) that abstracts away the provider-specific implementations, allowing you to switch between backends with minimal code changes.

Built with Node.js v20+ and TypeScript v5.5+ in mind, it emphasizes type safety, robustness, and adherence to modern backend development best practices.

## Key Features

- **Unified API:** Interact with different LLMs using a single, consistent interface for common tasks.
- **Provider Support:** Seamlessly switch between major AI providers:
  - OpenAI (Standard & Azure)
  - Anthropic
  - Google Generative AI
  - Groq
- **Core Functionality:**
  - Chat Completions (Streaming & Non-Streaming)
  - Text Embeddings
  - Tool Use / Function Calling
  - Multimodal Input (Images)
  - Text-to-Speech (TTS) via OpenAI/Azure
  - Speech-to-Text (STT) via OpenAI/Azure & Groq
  - Audio Translation via OpenAI/Azure & Groq
- **Advanced Features (Provider-dependent):**
  - JSON Mode / Structured Output (OpenAI/Azure direct support, others via prompting)
  - Grounding / Citations (Google)
  - Thinking Steps (Anthropic)
- **Type Safe:** Leverages TypeScript's strong typing for improved developer experience, autocompletion, and compile-time error checking.
- **Robust Error Handling:** Provides classified errors (`ConfigurationError`, `ProviderAPIError`, `UnsupportedFeatureError`, `MappingError`) for easier debugging and programmatic handling.
- **Flexible Configuration:** Easily configure API keys and defaults via `.env` files or direct constructor arguments.

## Supported Features Matrix

This matrix provides a general guide to feature support across providers. Provider capabilities and model support change frequently, so always consult the official provider documentation for the latest details.

| Feature             | OpenAI (Azure) | Anthropic | Google | Groq | Notes                                  |
| :------------------ | :------------: | :-------: | :----: | :--: | :------------------------------------- |
| Chat (Generate)     |       ✅       |    ✅     |   ✅   |  ✅  |                                        |
| Chat (Stream)       |       ✅       |    ✅     |   ✅   |  ✅  |                                        |
| Image Input         |       ✅       |    ✅     |   ✅   |  ⚠️  | Groq support varies by model           |
| Tool Use            |       ✅       |    ✅     |   ✅   |  ✅  | Implementation details differ slightly |
| Embeddings          |       ✅       |    ❌     |   ✅   |  ✅  | Anthropic has no public embedding API  |
| JSON Mode           |       ✅       |    ❌     |   ⚠️   |  ⚠️  | OpenAI/Azure best; others via prompt   |
| Grounding/Citations |       ❌       |    ❌     |   ✅   |  ❌  | Via Google Search tool integration     |
| Thinking Steps      |       ❌       |    ✅     |   ❌   |  ❌  | Anthropic specific feature             |
| TTS                 |       ✅       |    ❌     |   ❌   |  ❌  | Via OpenAI/Azure Audio API             |
| STT                 |       ✅       |    ❌     |   ⚠️   |  ✅  | Google requires separate Speech client |
| STT (Translate)     |       ✅       |    ❌     |   ❌   |  ✅  | To English                             |

✅ = Supported | ⚠️ = Partial/Limited/Via Prompting | ❌ = Not Supported

**Note:** This SDK aims to provide a common interface but cannot enable features unsupported by the underlying provider API.

## Installation

```bash
npm install rosetta-ai-sdk
# or
yarn add rosetta-ai-sdk
# or
pnpm add rosetta-ai-sdk
```

Ensure you have **Node.js v20 or later** installed.

## Configuration

RosettaAI can be configured using environment variables (loaded via `dotenv` or set directly) or by passing a configuration object to the constructor.

**1. Environment Variables (.env file):**

Create a `.env` file in your project root. See `.env.example` for all available options.

- **API Keys (Required for desired providers):**

  ```dotenv
  # .env
  ANTHROPIC_API_KEY=sk-ant-...
  GOOGLE_API_KEY=AIza...
  GROQ_API_KEY=gsk_...
  OPENAI_API_KEY=sk-... # For standard OpenAI OR Azure if AZURE_OPENAI_API_KEY is not set
  ```

- **Azure OpenAI (Alternative to Standard OpenAI):** If using Azure, provide these instead of/in addition to `OPENAI_API_KEY`. RosettaAI will prioritize Azure if its key and endpoint are set.

  ```dotenv
  AZURE_OPENAI_API_KEY=your_azure_openai_api_key
  AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
  AZURE_OPENAI_API_VERSION=2024-05-01-preview # Check Azure docs for appropriate version
  AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-deployment-name # Default CHAT deployment ID
  ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name # Default EMBEDDING deployment ID
  ```

- **Optional Defaults:** Set default models for each provider to avoid specifying them in every call.

  ```dotenv
  # Chat Models
  ROSETTA_DEFAULT_ANTHROPIC_MODEL=claude-3-haiku-20240307
  ROSETTA_DEFAULT_GOOGLE_MODEL=gemini-1.5-flash-latest
  ROSETTA_DEFAULT_GROQ_MODEL=llama3-8b-8192
  ROSETTA_DEFAULT_OPENAI_MODEL=gpt-4o-mini

  # Embedding Models
  ROSETTA_DEFAULT_EMBEDDING_GOOGLE_MODEL=text-embedding-004
  ROSETTA_DEFAULT_EMBEDDING_OPENAI_MODEL=text-embedding-3-small
  ROSETTA_DEFAULT_EMBEDDING_GROQ_MODEL=nomic-embed-text-v1.5

  # Audio Models
  ROSETTA_DEFAULT_TTS_OPENAI_MODEL=tts-1
  ROSETTA_DEFAULT_STT_OPENAI_MODEL=whisper-1
  ROSETTA_DEFAULT_STT_GROQ_MODEL=whisper-large-v3
  ```

**2. Constructor Configuration:**

Pass configuration options directly when creating the `RosettaAI` instance. Constructor arguments override environment variables.

```typescript
import { RosettaAI, Provider, RosettaAIConfig } from 'rosetta-ai-sdk'

const config: RosettaAIConfig = {
  openaiApiKey: 'sk-...',
  googleApiKey: 'AIza...',
  // Anthropic and Groq will be loaded from environment if keys exist there

  defaultModels: {
    [Provider.OpenAI]: 'gpt-4o-mini',
    [Provider.Google]: 'gemini-1.5-flash-latest'
  },

  // Example: Explicit Azure configuration
  azureOpenAIApiKey: 'azure-key',
  azureOpenAIEndpoint: 'https://your-azure.openai.azure.com/',
  azureOpenAIApiVersion: '2024-05-01-preview',
  azureOpenAIDefaultChatDeploymentName: 'my-gpt4-deployment',
  azureOpenAIDefaultEmbeddingDeploymentName: 'my-embedding-deployment',

  // Optional: Override default retries/timeout
  defaultMaxRetries: 3,
  defaultTimeoutMs: 90000 // 90 seconds
}

const rosetta = new RosettaAI(config)
```

## Quick Start

```typescript
import { RosettaAI, Provider, RosettaMessage } from 'rosetta-ai-sdk'
import dotenv from 'dotenv'

// Load .env file (if using)
dotenv.config()

async function main() {
  // Initialize (reads config from constructor or process.env)
  const rosetta = new RosettaAI()

  // --- Basic Chat Completion (Non-Streaming) ---
  console.log('\n--- Basic Generation ---')
  try {
    const messages: RosettaMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' }
    ]

    const result = await rosetta.generate({
      provider: Provider.OpenAI, // Or Provider.Anthropic, Provider.Google, Provider.Groq
      // model: 'gpt-4o-mini', // Optional: uses configured default if not set
      messages: messages,
      maxTokens: 50
    })

    console.log(`[${result.model}] Response:`)
    console.log(result.content)
    console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
  } catch (error) {
    console.error(`Generation failed: ${error.name} - ${error.message}`)
  }

  // --- Streaming Chat Completion ---
  console.log('\n--- Streaming Generation ---')
  try {
    const stream = rosetta.stream({
      provider: Provider.Groq, // Or any other configured provider
      // model: 'llama3-8b-8192', // Optional
      messages: [{ role: 'user', content: 'Write a short haiku about TypeScript.' }],
      maxTokens: 60
    })

    console.log(`[Streaming Response]`)
    let fullContent = ''
    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'content_delta':
          process.stdout.write(chunk.data.delta)
          fullContent += chunk.data.delta
          break
        case 'message_stop':
          console.log(`\n--- Stream Stopped (Reason: ${chunk.data.finishReason}) ---`)
          break
        case 'final_usage':
          console.log('\nFinal Usage:', chunk.data.usage ? JSON.stringify(chunk.data.usage) : 'N/A')
          break
        case 'error':
          console.error('\nStream Error:', chunk.data.error.message)
          return // Stop processing on error
        // Add cases for other chunk types (tool_call_*, json_*, thinking_*, etc.) if needed
        default:
          // console.log(`Chunk: ${chunk.type}`); // Log other chunk types if interested
          break
      }
    }
    console.log('\n--- Stream Complete ---')
    // console.log("Final Accumulated Content:", fullContent);
  } catch (error) {
    // Errors during stream *setup* (e.g., invalid config) are caught here
    console.error(`Streaming setup failed: ${error.name} - ${error.message}`)
  }
}

main().catch(console.error)
```

## Core Concepts & Usage Examples

### Initialization

```typescript
import { RosettaAI, Provider, RosettaAIConfig } from 'rosetta-ai-sdk'
import dotenv from 'dotenv'

dotenv.config() // Load .env

// Option 1: Automatic configuration from environment variables
const rosettaAuto = new RosettaAI()

// Option 2: Explicit configuration via constructor
const config: RosettaAIConfig = {
  openaiApiKey: 'sk-...',
  anthropicApiKey: 'sk-ant-...',
  defaultModels: {
    [Provider.OpenAI]: 'gpt-4o-mini',
    [Provider.Anthropic]: 'claude-3-haiku-20240307'
  }
}
const rosettaManual = new RosettaAI(config)

// Check configured providers
console.log('Available Providers:', rosettaManual.getConfiguredProviders())
```

### Chat Completions

#### `generate()` (Non-Streaming)

Use `generate` for simple request-response interactions.

```typescript
import { RosettaAI, Provider, RosettaMessage } from 'rosetta-ai-sdk'
// ... initialization ...

const messages: RosettaMessage[] = [
  { role: 'system', content: 'You are a concise poet.' },
  { role: 'user', content: 'Write a 2-line poem about the moon.' }
]

try {
  const result = await rosetta.generate({
    provider: Provider.Anthropic,
    model: 'claude-3-haiku-20240307', // Specify model
    messages: messages,
    maxTokens: 30,
    temperature: 0.8
  })

  console.log(result.content)
  console.log('Finish Reason:', result.finishReason)
  console.log('Usage:', result.usage)
} catch (error) {
  // Handle errors (see Error Handling section)
  console.error(error)
}
```

#### `stream()` (Streaming)

Use `stream` to process responses chunk-by-chunk, ideal for real-time applications.

```typescript
import { RosettaAI, Provider, RosettaMessage, StreamChunk } from 'rosetta-ai-sdk'
// ... initialization ...

const messages: RosettaMessage[] = [{ role: 'user', content: 'Explain the concept of "event loop" in Node.js simply.' }]

try {
  const stream = rosetta.stream({
    provider: Provider.OpenAI,
    model: 'gpt-4o-mini',
    messages: messages,
    maxTokens: 150
  })

  let fullResponse = ''
  let toolArgs = '' // Example accumulator for tool arguments

  for await (const chunk of stream) {
    switch (chunk.type) {
      case 'message_start':
        console.log(`Stream started (Model: ${chunk.data.model})`)
        break
      case 'content_delta':
        process.stdout.write(chunk.data.delta)
        fullResponse += chunk.data.delta
        break
      case 'tool_call_start':
        console.log(`\nTool Call Start: ${chunk.data.toolCall.function.name} (ID: ${chunk.data.toolCall.id})`)
        toolArgs = '' // Reset for new tool call
        break
      case 'tool_call_delta':
        process.stdout.write(chunk.data.functionArgumentChunk)
        toolArgs += chunk.data.functionArgumentChunk
        break
      case 'tool_call_done':
        console.log(`\nTool Call Done (ID: ${chunk.data.id}). Args: ${toolArgs}`)
        break
      case 'message_stop':
        console.log(`\nStream Stop Reason: ${chunk.data.finishReason}`)
        break
      case 'final_usage':
        console.log(`\nFinal Usage: ${JSON.stringify(chunk.data.usage)}`)
        break
      case 'final_result':
        console.log('\nFinal Aggregated Result Received.')
        // console.log(chunk.data.result); // Access the complete GenerateResult object
        break
      case 'error':
        console.error(`\nStream Error: ${chunk.data.error.message}`)
        break
      // Add cases for json_*, thinking_*, citation_* as needed
      default:
        const _: never = chunk // Exhaustiveness check
        console.log(`\nUnknown chunk type: ${(_ as any).type}`)
    }
  }
  console.log('\n--- Stream Complete ---')
} catch (error) {
  // Handle stream setup errors
  console.error(error)
}
```

### Embeddings

Generate vector representations of text using `embed`. Supported by OpenAI/Azure, Google, and Groq.

```typescript
import { RosettaAI, Provider, EmbedParams } from 'rosetta-ai-sdk'
// ... initialization ...

const textsToEmbed = ['RosettaAI simplifies LLM interactions.', 'TypeScript adds static typing to JavaScript.']

try {
  const params: EmbedParams = {
    provider: Provider.OpenAI, // Or Provider.Google, Provider.Groq
    model: 'text-embedding-3-small', // Use appropriate model
    input: textsToEmbed
    // dimensions: 256 // Optional: For OpenAI models supporting reduced dimensions
  }

  const result = await rosetta.embed(params)

  console.log(`Generated ${result.embeddings.length} embeddings.`)
  result.embeddings.forEach((vec, i) => {
    console.log(`Embedding ${i + 1} (Dim: ${vec.length}): [${vec.slice(0, 3).join(', ')}...]`)
  })
  console.log('Usage:', result.usage)
  console.log('Model Used:', result.model)
} catch (error) {
  console.error(error)
}
```

### Tool Use / Function Calling

Instruct models to use predefined tools (functions) to interact with external systems or data.

```typescript
import { RosettaAI, Provider, RosettaTool, RosettaMessage } from 'rosetta-ai-sdk'
// ... initialization ...

// 1. Define your tool
const getWeatherTool: RosettaTool = {
  type: 'function',
  function: {
    name: 'getCurrentWeather',
    description: 'Get the current weather for a location.',
    parameters: {
      type: 'object',
      properties: { location: { type: 'string', description: 'City and state/country' } },
      required: ['location']
    }
  }
}

// 2. Implement the tool function
async function getCurrentWeather(location: string): Promise<string> {
  console.log(`[TOOL] Getting weather for ${location}...`)
  // Simulate API call
  await new Promise(r => setTimeout(r, 100))
  return JSON.stringify({ temperature: 72, condition: 'Sunny' })
}

// 3. Conversation Loop
async function runToolConversation() {
  const messages: RosettaMessage[] = [{ role: 'user', content: "What's the weather in San Francisco?" }]
  const maxTurns = 5 // Prevent infinite loops

  for (let i = 0; i < maxTurns; i++) {
    console.log(`\nTurn ${i + 1}: Sending request...`)
    try {
      const response = await rosetta.generate({
        provider: Provider.OpenAI, // Or Anthropic, Google, Groq
        messages: messages,
        tools: [getWeatherTool],
        toolChoice: 'auto' // Let the model decide
      })

      console.log('Assistant:', response.content ?? '[No text content]')
      messages.push({ role: 'assistant', content: response.content, toolCalls: response.toolCalls })

      if (response.toolCalls && response.toolCalls.length > 0) {
        console.log('Tool Calls Requested:', response.toolCalls.length)
        const toolResults: RosettaMessage[] = []
        for (const call of response.toolCalls) {
          if (call.function.name === 'getCurrentWeather') {
            try {
              const args = JSON.parse(call.function.arguments)
              const result = await getCurrentWeather(args.location)
              toolResults.push({ role: 'tool', toolCallId: call.id, content: result })
            } catch (e) {
              console.error(`Tool execution error: ${e.message}`)
              toolResults.push({ role: 'tool', toolCallId: call.id, content: JSON.stringify({ error: e.message }) })
            }
          }
        }
        messages.push(...toolResults) // Add results for the next turn
      } else {
        console.log('\n--- Conversation End ---')
        break // Exit loop if no tool calls
      }
    } catch (error) {
      console.error(error)
      break
    }
  }
}

runToolConversation()
```

### Multimodal (Image Input)

Send images along with text prompts to multimodal models (OpenAI, Anthropic, Google).

```typescript
import { RosettaAI, Provider, RosettaMessage, RosettaImageData, ImageMimeType } from 'rosetta-ai-sdk'
import fs from 'fs/promises'
import path from 'path'
// ... initialization ...

async function describeImage(imagePath: string) {
  try {
    // 1. Load and encode the image
    const buffer = await fs.readFile(imagePath)
    const base64Data = buffer.toString('base64')
    const ext = path.extname(imagePath).toLowerCase()
    const mimeType: ImageMimeType = ext === '.png' ? 'image/png' : 'image/jpeg' // Add more types as needed
    const imageData: RosettaImageData = { mimeType, base64Data }

    // 2. Construct the message
    const messages: RosettaMessage[] = [
      {
        role: 'user',
        content: [
          // Content is an array for multimodal
          { type: 'text', text: 'Describe this image in detail.' },
          { type: 'image', image: imageData }
        ]
      }
    ]

    // 3. Generate response
    const result = await rosetta.generate({
      provider: Provider.OpenAI, // Or Anthropic, Google
      model: 'gpt-4o-mini', // Use a vision-capable model
      messages: messages,
      maxTokens: 150
    })

    console.log('Image Description:', result.content)
  } catch (error) {
    console.error(error)
  }
}

// Ensure you have an image file (e.g., logo.png) in the same directory or provide the correct path
describeImage(path.join(__dirname, 'logo.png'))
```

### Audio (TTS, STT, Translation)

#### Text-to-Speech (TTS)

Generate speech from text using `generateSpeech` (non-streaming) or `streamSpeech`. Currently uses OpenAI/Azure.

```typescript
import { RosettaAI, Provider, SpeechParams } from 'rosetta-ai-sdk'
import fs from 'fs/promises'
import path from 'path'
// ... initialization ...

async function generateAudio() {
  const outputDir = path.join(__dirname, 'audio_output')
  await fs.mkdir(outputDir, { recursive: true })
  const filePath = path.join(outputDir, 'hello.mp3')

  try {
    const params: SpeechParams = {
      provider: Provider.OpenAI,
      input: 'Hello from RosettaAI!',
      voice: 'alloy', // Choose a voice
      // model: 'tts-1-hd', // Optional model override
      responseFormat: 'mp3'
    }

    // Non-streaming
    const audioBuffer = await rosetta.generateSpeech(params)
    await fs.writeFile(filePath, audioBuffer)
    console.log(`Audio saved to ${filePath}`)

    // Example: Streaming (optional)
    // const stream = rosetta.streamSpeech(params);
    // const writeStream = (await fs.open(path.join(outputDir, 'hello_stream.mp3'), 'w')).createWriteStream();
    // for await (const chunk of stream) {
    //   if (chunk.type === 'audio_chunk') writeStream.write(chunk.data);
    // }
    // writeStream.end();
    // console.log('Streamed audio saved.');
  } catch (error) {
    console.error(error)
  }
}

generateAudio()
```

#### Speech-to-Text (STT) & Translation

Transcribe audio to text using `transcribe` or translate audio to English using `translate`. Supported by OpenAI/Azure and Groq.

```typescript
import { RosettaAI, Provider, TranscribeParams, TranslateParams, RosettaAudioData } from 'rosetta-ai-sdk'
import fs from 'fs/promises'
import path from 'path'
// ... initialization ...

async function processAudio(audioPath: string) {
  try {
    // 1. Prepare audio data
    const buffer = await fs.readFile(audioPath)
    const audioData: RosettaAudioData = {
      data: buffer,
      filename: path.basename(audioPath),
      mimeType: 'audio/mpeg' // Adjust based on your file type (mp3, wav, etc.)
    }

    // 2. Transcribe
    console.log('\n--- Transcription ---')
    const transcribeParams: TranscribeParams = {
      provider: Provider.Groq, // Or Provider.OpenAI
      audio: audioData
      // model: 'whisper-large-v3', // Optional
      // language: 'en', // Optional language hint
    }
    const transcription = await rosetta.transcribe(transcribeParams)
    console.log(`[${transcribeParams.provider}] Transcription: ${transcription.text}`)

    // 3. Translate (to English)
    console.log('\n--- Translation ---')
    const translateParams: TranslateParams = {
      provider: Provider.Groq, // Or Provider.OpenAI
      audio: audioData
      // model: 'whisper-large-v3', // Optional
    }
    const translation = await rosetta.translate(translateParams)
    console.log(`[${translateParams.provider}] Translation: ${translation.text}`)
  } catch (error) {
    console.error(error)
  }
}

// Ensure you have an audio file (e.g., sample_audio.mp3) or use the TTS output
processAudio(path.join(__dirname, 'sample_audio.mp3')) // Provide path to your audio file
```

### Error Handling

RosettaAI throws specific error types to help you handle issues gracefully.

```typescript
import {
  RosettaAI,
  Provider,
  RosettaAIError,
  ConfigurationError,
  ProviderAPIError,
  UnsupportedFeatureError,
  MappingError
} from 'rosetta-ai-sdk'
// ... initialization ...

async function safeGenerate() {
  try {
    const result = await rosetta.generate({
      provider: Provider.OpenAI,
      model: 'invalid-model-id', // Intentionally invalid
      messages: [{ role: 'user', content: 'Test' }]
    })
    console.log(result.content)
  } catch (error) {
    if (error instanceof ConfigurationError) {
      console.error(`Configuration Error: ${error.message}`)
      // e.g., Missing API key, invalid deployment ID
    } else if (error instanceof ProviderAPIError) {
      console.error(
        `Provider API Error (${error.provider}): Status ${error.statusCode ?? 'N/A'}, Code: ${error.errorCode ??
          'N/A'} - ${error.message}`
      )
      // e.g., Rate limit, authentication error, invalid request to provider
      // console.error("Underlying error:", error.underlyingError); // Log original error if needed
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(`Unsupported Feature Error: ${error.provider} does not support ${error.feature}.`)
      // e.g., Trying TTS with Groq, Embeddings with Anthropic
    } else if (error instanceof MappingError) {
      console.error(`Internal SDK Mapping Error: ${error.message}`)
      // e.g., Failed to convert data between RosettaAI and provider format
    } else if (error instanceof RosettaAIError) {
      // Catch any other base SDK errors
      console.error(`RosettaAI Error: ${error.name} - ${error.message}`)
    } else {
      // Catch unexpected errors
      console.error('Unexpected Error:', error)
    }
  }
}

safeGenerate()
```

## API Reference

Detailed documentation for all exported classes, methods, types, and interfaces is available via JSDoc comments within the source code. Use your IDE's IntelliSense or generate HTML documentation using [TypeDoc](https://typedoc.org/).

Key exports include:

- **Client:** `RosettaAI`
- **Enums:** `Provider`
- **Configuration:** `RosettaAIConfig`, `ProviderOptions`
- **Core Parameters:** `GenerateParams`, `EmbedParams`, `SpeechParams`, `TranscribeParams`, `TranslateParams`
- **Core Results:** `GenerateResult`, `EmbedResult`, `TranscriptionResult`
- **Streaming:** `StreamChunk`, `AudioStreamChunk`
- **Common Types:** `RosettaMessage`, `RosettaContentPart`, `RosettaImageData`, `RosettaAudioData`, `RosettaTool`, `RosettaToolCallRequest`, `TokenUsage`, `Citation`
- **Errors:** `RosettaAIError`, `ConfigurationError`, `ProviderAPIError`, `UnsupportedFeatureError`, `MappingError`

## Examples

Runnable examples demonstrating various features can be found in the `/examples` directory:

- `basic-chat.ts`: Simple non-streaming generation.
- `streaming-chat.ts`: Handling streaming responses and various chunk types.
- `tool-use.ts`: Function calling/tool execution loop.
- `image-input.ts`: Sending images to multimodal models.
- `embeddings.ts`: Generating text embeddings.
- `audio.ts`: Text-to-Speech and Speech-to-Text/Translation.
- `structured-output.ts`: Requesting and validating JSON output.

**To run an example:**

1.  Ensure you have configured your API keys in a `.env` file (see Configuration).
2.  Make sure any required sample files (e.g., `logo.png`, `sample_audio.mp3`) exist in the `examples` directory if needed by the specific example.
3.  Run the build command: `npm run build`
4.  Execute the example using: `npm run example:<name>` (e.g., `npm run example:basic`, `npm run example:stream`).

## Development

1.  **Clone:** `git clone https://github.com/MissionSquad/rosetta-ai-sdk.git`
2.  **Install:** `cd rosetta-ai-sdk && npm install`
3.  **Build:** `npm run build` (Compiles TypeScript to JavaScript in `/dist`)
4.  **Test:** `npm test` (Runs Jest tests)
5.  **Lint:** `npm run lint` (Checks code style and potential type issues)
6.  **Format:** `npm run format` (Formats code using Prettier)

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure tests pass (`npm test`) and linting is clean (`npm run lint`).
5.  Add tests for new features or bug fixes.
6.  Commit your changes (`git commit -m 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Open a Pull Request.

Please adhere to the established code style and architectural patterns.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
