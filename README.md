# RosettaAI SDK

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![npm version](https://badge.fury.io/js/@missionsquad/RosettaAI.svg)](https://badge.fury.io/js/@missionsquad/RosettaAI)
[![Node.js CI](https://github.com/MissionSquad/rosetta-ai-sdk/actions/workflows/node.js.yml/badge.svg)](https://github.com/MissionSquad/rosetta-ai-sdk/actions/workflows/node.js.yml)

**RosettaAI** is a unified TypeScript SDK designed for seamless interaction with multiple Large Language Model (LLM) providers like OpenAI (including Azure), Anthropic, Google Generative AI, and Groq, as well as custom OpenAI-compatible providers. It provides a consistent interface (`generate`, `stream`, `embed`, `generateSpeech`, `transcribe`, `listModels`, etc.) that abstracts away the provider-specific implementations, allowing you to switch between backends with minimal code changes.

Built with Node.js v20+ and TypeScript v5.5+ in mind, it emphasizes type safety, robustness, and adherence to modern backend development best practices.

## Key Features

- **Unified API:** Interact with different LLMs using a single, consistent interface for common tasks.
- **Provider Support:** Seamlessly switch between major AI providers:
  - OpenAI (Standard & Azure)
  - Anthropic
  - Google Generative AI
  - Groq
  - Custom OpenAI-Compatible Providers (e.g., LM Studio, Novita, GPUStack)
- **Core Functionality:**
  - Chat Completions (Streaming & Non-Streaming)
  - Text Embeddings
  - Tool Use / Function Calling
  - Multimodal Input (Images)
  - Text-to-Speech (TTS) via OpenAI/Azure
  - Speech-to-Text (STT) via OpenAI/Azure & Groq
  - Audio Translation via OpenAI/Azure & Groq
  - Model Listing (Built-in & Custom Providers)
- **Advanced Features (Provider-dependent):**
  - JSON Mode / Structured Output (OpenAI/Azure direct support, others via prompting)
  - Grounding / Citations (Google)
  - Thinking Steps (Anthropic)
- **Type Safe:** Leverages TypeScript's strong typing for improved developer experience, autocompletion, and compile-time error checking.
- **Robust Error Handling:** Provides classified errors (`ConfigurationError`, `ProviderAPIError`, `UnsupportedFeatureError`, `MappingError`) for easier debugging and programmatic handling.
- **Flexible Configuration:** Easily configure API keys and defaults via `.env` files or direct constructor arguments.

## Supported Features Matrix

This matrix provides a general guide to feature support across providers. Provider capabilities and model support change frequently, so always consult the official provider documentation for the latest details. Custom provider support depends on their specific API implementation.

| Feature             | OpenAI (Azure) | Anthropic | Google | Groq | Custom (OpenAI-Compat) | Notes                                  |
| :------------------ | :------------: | :-------: | :----: | :--: | :--------------------: | :------------------------------------- |
| Chat (Generate)     |       ✅       |    ✅     |   ✅   |  ✅  |           ✅           |                                        |
| Chat (Stream)       |       ✅       |    ✅     |   ✅   |  ✅  |           ✅           |                                        |
| Image Input         |       ✅       |    ✅     |   ✅   |  ⚠️  |           ⚠️           | Groq/Custom support varies by model    |
| Tool Use            |       ✅       |    ✅     |   ✅   |  ✅  |           ✅           | Implementation details differ slightly |
| Embeddings          |       ✅       |    ❌     |   ✅   |  ✅  |           ⚠️           | Custom support varies                  |
| JSON Mode           |       ✅       |    ❌     |   ⚠️   |  ⚠️  |           ✅           | OpenAI/Azure best; others via prompt   |
| Grounding/Citations |       ❌       |    ❌     |   ✅   |  ❌  |           ❌           | Via Google Search tool integration     |
| Thinking Steps      |       ❌       |    ✅     |   ❌   |  ❌  |           ❌           | Anthropic specific feature             |
| TTS                 |       ✅       |    ❌     |   ❌   |  ✅  |           ❌           | Via OpenAI/Azure and Groq Audio APIs   |
| STT                 |       ✅       |    ❌     |   ⚠️   |  ✅  |           ❌           | Google requires separate Speech client |
| STT (Translate)     |       ✅       |    ❌     |   ❌   |  ✅  |           ❌           | To English                             |
| Model Listing       |       ✅       |    ✅     |   ✅   |  ✅  |           ✅           | Custom via OpenAI-compat `/models`     |

✅ = Supported | ⚠️ = Partial/Limited/Via Prompting/Varies | ❌ = Not Supported

**Note:** This SDK aims to provide a common interface but cannot enable features unsupported by the underlying provider API.

## Installation

```bash
npm install @missionsquad/rosetta-ai
# or
yarn add @missionsquad/rosetta-ai
# or
pnpm add @missionsquad/rosetta-ai
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

  # Custom Provider Example Keys
  NOVITA_API_KEY=your_novita_key
  GPUSTACK_API_KEY=your_gpustack_key
  # LMSTUDIO_API_KEY= # Often not needed for local
  ```

- **Azure OpenAI (Alternative to Standard OpenAI):** If using Azure, provide these instead of/in addition to `OPENAI_API_KEY`. RosettaAI will prioritize Azure if its key and endpoint are set.

  ```dotenv
  AZURE_OPENAI_API_KEY=your_azure_openai_api_key
  AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
  AZURE_OPENAI_API_VERSION=2024-05-01-preview # Check Azure docs for appropriate version
  AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-deployment-name # Default CHAT deployment ID
  ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name # Default EMBEDDING deployment ID
  ```

- **Custom Provider Base URLs (Optional):** Set base URLs if they differ from defaults or if not set in constructor config.

  ```dotenv
  # NOVITA_BASE_URL=https://api.novita.ai/v3/openai
  # GPUSTACK_BASE_URL=https://gpu.crypto-tech.cloud/v1-openai
  # LMSTUDIO_BASE_URL=http://localhost:1234/v1
  ```

- **Optional Defaults:** Set default models for each provider to avoid specifying them in every call.

  ```dotenv
  # Chat Models
  ROSETTA_DEFAULT_ANTHROPIC_MODEL=claude-3-haiku-20240307
  ROSETTA_DEFAULT_GOOGLE_MODEL=gemini-1.5-flash-latest
  ROSETTA_DEFAULT_GROQ_MODEL=llama3-8b-8192
  ROSETTA_DEFAULT_OPENAI_MODEL=gpt-4o-mini
  ROSETTA_DEFAULT_NOVITA_MODEL=meta-llama/llama-3.1-8b-instruct # Example custom default
  ROSETTA_DEFAULT_GPUSTACK_MODEL=ibm-granite-3.2-8b-instruct # Example custom default
  ROSETTA_DEFAULT_LMSTUDIO_MODEL=lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF # Example custom default

  # Embedding Models
  ROSETTA_DEFAULT_EMBEDDING_GOOGLE_MODEL=text-embedding-004
  ROSETTA_DEFAULT_EMBEDDING_OPENAI_MODEL=text-embedding-3-small
  ROSETTA_DEFAULT_EMBEDDING_GROQ_MODEL=nomic-embed-text-v1.5
  ROSETTA_DEFAULT_EMBEDDING_GPUSTACK_MODEL=nomic-embed-text-v1.5 # Example custom default

  # Audio Models
  ROSETTA_DEFAULT_TTS_OPENAI_MODEL=tts-1
  ROSETTA_DEFAULT_STT_OPENAI_MODEL=whisper-1
  ROSETTA_DEFAULT_STT_GROQ_MODEL=whisper-large-v3
  ```

**2. Constructor Configuration:**

Pass configuration options directly when creating the `RosettaAI` instance. Constructor arguments override environment variables.

```typescript
import { RosettaAI, Provider, RosettaAIConfig, CustomProviderConfig } from 'rosetta-ai-sdk'
import { OpenAICompatibleMapper } from 'rosetta-ai-sdk/dist/core/mapping/openai-compatible.mapper' // Adjust path if needed

// Example Custom Provider Config
const lmstudioConfig: CustomProviderConfig = {
  providerKey: 'lmstudio',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'list_models'], // Supports listing
  baseURL: process.env.LMSTUDIO_BASE_URL || 'http://localhost:1234/v1',
  apiKey: process.env.LMSTUDIO_API_KEY || undefined,
  defaultModel: process.env.ROSETTA_DEFAULT_LMSTUDIO_MODEL || 'local-model'
  // modelListPath: '/models' // Default, can omit
}

const customWithPathConfig: CustomProviderConfig = {
  providerKey: 'custom-path',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://api.example.com/v2',
  apiKey: 'key-123',
  defaultModel: 'model-a',
  modelListPath: '/openai-compat/models' // Custom path for model listing
}

const customWithUrlConfig: CustomProviderConfig = {
  providerKey: 'custom-url',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://api.example.com/v2', // Base for generate
  apiKey: 'key-456',
  defaultModel: 'model-b',
  modelListUrl: 'https://models.example.org/all' // Absolute URL for listing
}

// Main Config
const config: RosettaAIConfig = {
  openaiApiKey: 'sk-...',
  googleApiKey: 'AIza...',
  // Anthropic and Groq will be loaded from environment if keys exist there

  defaultModels: {
    [Provider.OpenAI]: 'gpt-4o-mini',
    [Provider.Google]: 'gemini-1.5-flash-latest',
    lmstudio: 'local-model-override' // Override default for lmstudio
  },

  // Example: Explicit Azure configuration
  azureOpenAIApiKey: 'azure-key',
  azureOpenAIEndpoint: 'https://your-azure.openai.azure.com/',
  azureOpenAIApiVersion: '2024-05-01-preview',
  azureOpenAIDefaultChatDeploymentName: 'my-gpt4-deployment',
  azureOpenAIDefaultEmbeddingDeploymentName: 'my-embedding-deployment',

  // Register Custom Providers
  customProviders: [lmstudioConfig, customWithPathConfig, customWithUrlConfig],

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
      provider: Provider.OpenAI, // Or Provider.Anthropic, Provider.Google, Provider.Groq, 'lmstudio'
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

  // --- List Models ---
  console.log('\n--- Model Listing ---')
  try {
    const providerToList = Provider.OpenAI // Or 'lmstudio', Provider.Groq, etc.
    if (rosetta.getConfiguredProviders().includes(providerToList)) {
      const models = await rosetta.listModels(providerToList)
      console.log(`Models for ${providerToList}:`)
      models.data.slice(0, 5).forEach(m => console.log(` - ${m.id} (Owned by: ${m.owned_by})`))
      if (models.data.length > 5) console.log(' ... and more')
    } else {
      console.log(`Provider ${providerToList} not configured, skipping model list.`)
    }
  } catch (error) {
    console.error(`Model listing failed: ${error.name} - ${error.message}`)
  }
}

main().catch(console.error)
```

## Core Concepts & Usage Examples

### Initialization

```typescript
import { RosettaAI, Provider, RosettaAIConfig, CustomProviderConfig } from 'rosetta-ai-sdk'
import { OpenAICompatibleMapper } from 'rosetta-ai-sdk/dist/core/mapping/openai-compatible.mapper'
import dotenv from 'dotenv'

dotenv.config() // Load .env

// Option 1: Automatic configuration from environment variables
const rosettaAuto = new RosettaAI()

// Option 2: Explicit configuration via constructor
const customConfig: CustomProviderConfig = {
  providerKey: 'my-custom',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'http://localhost:11434/v1', // Example: Ollama endpoint
  apiKey: 'ollama', // Example API key if needed
  modelListPath: '/api/tags' // Example: Ollama model listing path (might need custom mapper logic)
}
const config: RosettaAIConfig = {
  openaiApiKey: 'sk-...',
  anthropicApiKey: 'sk-ant-...',
  defaultModels: {
    [Provider.OpenAI]: 'gpt-4o-mini',
    [Provider.Anthropic]: 'claude-3-haiku-20240307',
    'my-custom': 'llama3'
  },
  customProviders: [customConfig]
}
const rosettaManual = new RosettaAI(config)

// Check configured providers
console.log('Available Providers:', rosettaManual.getConfiguredProviders())
```

### Chat Completions

#### `generate()` (Non-Streaming)

Use `generate` for simple request-response interactions.

```typescript
import { RosettaAI, Provider, RosettaMessage, ProviderKey } from 'rosetta-ai-sdk'
// ... initialization ...

const messages: RosettaMessage[] = [
  { role: 'system', content: 'You are a concise poet.' },
  { role: 'user', content: 'Write a 2-line poem about the moon.' }
]

try {
  const providerKey: ProviderKey = Provider.Anthropic // Or 'my-custom', Provider.OpenAI, etc.
  const result = await rosetta.generate({
    provider: providerKey,
    model: 'claude-3-haiku-20240307', // Specify model or use default
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
import { RosettaAI, Provider, RosettaMessage, StreamChunk, ProviderKey } from 'rosetta-ai-sdk'
// ... initialization ...

const messages: RosettaMessage[] = [{ role: 'user', content: 'Explain the concept of "event loop" in Node.js simply.' }]

try {
  const providerKey: ProviderKey = Provider.OpenAI // Or 'my-custom', Provider.Groq, etc.
  const stream = rosetta.stream({
    provider: providerKey,
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

Generate vector representations of text using `embed`. Supported by OpenAI/Azure, Google, Groq, and potentially custom providers.

```typescript
import { RosettaAI, Provider, EmbedParams, ProviderKey } from 'rosetta-ai-sdk'
// ... initialization ...

const textsToEmbed = ['RosettaAI simplifies LLM interactions.', 'TypeScript adds static typing to JavaScript.']

try {
  const providerKey: ProviderKey = Provider.OpenAI // Or Provider.Google, Provider.Groq, 'my-custom-embed-provider'
  const params: EmbedParams = {
    provider: providerKey,
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
import { RosettaAI, Provider, RosettaTool, RosettaMessage, ProviderKey } from 'rosetta-ai-sdk'
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
  const providerKey: ProviderKey = Provider.OpenAI // Or 'my-custom', Provider.Anthropic, etc.

  for (let i = 0; i < maxTurns; i++) {
    console.log(`\nTurn ${i + 1}: Sending request...`)
    try {
      const response = await rosetta.generate({
        provider: providerKey,
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
import { RosettaAI, Provider, RosettaMessage, RosettaImageData, ImageMimeType, ProviderKey } from 'rosetta-ai-sdk'
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
    const providerKey: ProviderKey = Provider.OpenAI // Or Provider.Anthropic, Provider.Google
    const result = await rosetta.generate({
      provider: providerKey,
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
import { RosettaAI, Provider, TranscribeParams, TranslateParams, RosettaAudioData, ProviderKey } from 'rosetta-ai-sdk'
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
    const transcribeProvider: ProviderKey = Provider.Groq // Or Provider.OpenAI
    const transcribeParams: TranscribeParams = {
      provider: transcribeProvider,
      audio: audioData
      // model: 'whisper-large-v3', // Optional
      // language: 'en', // Optional language hint
    }
    const transcription = await rosetta.transcribe(transcribeParams)
    console.log(`[${transcribeParams.provider}] Transcription: ${transcription.text}`)

    // 3. Translate (to English)
    console.log('\n--- Translation ---')
    const translateProvider: ProviderKey = Provider.Groq // Or Provider.OpenAI
    const translateParams: TranslateParams = {
      provider: translateProvider,
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

### Model Listing

List available models for configured providers (built-in and custom).

```typescript
import { RosettaAI, Provider, ProviderKey, RosettaModelList } from 'rosetta-ai-sdk'
// ... initialization ...

async function listProviderModels(providerKey: ProviderKey) {
  console.log(`\n--- Listing Models for: ${providerKey} ---`)
  try {
    const models: RosettaModelList = await rosetta.listModels(providerKey)
    console.log(`Found ${models.data.length} models:`)
    models.data.slice(0, 5).forEach(m => console.log(`  - ${m.id} (Owned by: ${m.owned_by})`))
    if (models.data.length > 5) console.log('  ...')
  } catch (error) {
    console.error(`Error listing models for ${providerKey}: ${error.message}`)
  }
}

async function listAllProviderModels() {
  console.log('\n--- Listing Models for ALL Configured Providers ---')
  try {
    const allResults = await rosetta.listAllModels()
    for (const providerKey in allResults) {
      console.log(`\nProvider: ${providerKey}`)
      const result = allResults[providerKey]
      if (result instanceof Error) {
        // Check if it's an error object
        console.error(`  Error: ${result.message}`)
      } else if (result) {
        console.log(`  Found ${result.data.length} models:`)
        result.data.slice(0, 5).forEach(m => console.log(`    - ${m.id}`))
        if (result.data.length > 5) console.log('    ...')
      }
    }
  } catch (error) {
    console.error(`Unexpected error during listAllModels: ${error.message}`)
  }
}

// Example calls
// listProviderModels(Provider.OpenAI);
// listProviderModels('lmstudio'); // Example custom provider key
// listAllProviderModels();
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
        `Provider API Error (${error.provider || error.customProvider}): Status ${error.statusCode ??
          'N/A'}, Code: ${error.errorCode ?? 'N/A'} - ${error.message}`
      )
      // e.g., Rate limit, authentication error, invalid request to provider
      // console.error("Underlying error:", error.underlyingError); // Log original error if needed
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(
        `Unsupported Feature Error: ${error.provider || error.customProvider} does not support ${error.feature}.`
      )
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

### OpenAI Responses API (Stateful Interface)

RosettaAI provides access to OpenAI's **Responses API** as a separate, OpenAI-specific interface. This is a stateful, agent-ready API designed for modern AI interactions.

**Key Differences from Chat Completions:**

- **Stateful conversations:** Use `previous_response_id` to chain turns without replaying message history
- **Separated concerns:** `instructions` (developer/system intent) + `input` (user content)
- **Built-in tools:** `web_search`, `file_search`, `image_generation`, `code_interpreter`
- **Semantic streaming:** Rich event types beyond simple content deltas
- **OpenAI-only:** Not available for other providers (by design)

#### Basic Usage

```typescript
import { RosettaAI, Provider } from 'rosetta-ai-sdk'
// ... initialization ...

// Simple response
const result = await rosetta.createResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  instructions: 'You are a helpful coding assistant',
  input: 'Explain async/await in JavaScript',
  max_tokens: 200
})

console.log(result.output_text)
console.log(`Response ID: ${result.id}`) // Use for stateful conversations
```

#### Stateful Conversations

```typescript
// First turn
const turn1 = await rosetta.createResponse({
  provider: Provider.OpenAI,
  instructions: 'You are a math tutor',
  input: 'What is 15 + 27?'
})

console.log(turn1.output_text) // "15 + 27 = 42"

// Second turn - reference previous response (no history replay!)
const turn2 = await rosetta.createResponse({
  provider: Provider.OpenAI,
  instructions: 'You are a math tutor',
  input: 'Now multiply that by 3',
  previous_response_id: turn1.id // Stateful!
})

console.log(turn2.output_text) // "42 × 3 = 126"
```

#### Built-in Tools

```typescript
// Web search
const result = await rosetta.createResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  input: 'What are the latest TypeScript features?',
  tools: [{ type: 'web_search' }],
  tool_choice: 'auto'
})

// Image generation
const imageResult = await rosetta.createResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  input: 'Generate a logo with a mountain and sun',
  tools: [
    {
      type: 'image_generation',
      options: { size: '1024x1024', quality: 'hd' }
    }
  ],
  tool_choice: { type: 'image_generation' }
})

// Code interpreter
const codeResult = await rosetta.createResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  input: 'Analyze this CSV data: ...',
  tools: [{ type: 'code_interpreter' }]
})
```

#### Custom Function Tools

```typescript
import { ResponsesTool } from 'rosetta-ai-sdk'
import { z } from 'zod'

const getWeatherTool: ResponsesTool = {
  type: 'function',
  name: 'getCurrentWeather',
  description: 'Get current weather for a location',
  parameters: {
    type: 'object',
    properties: {
      location: { type: 'string' },
      unit: { type: 'string', enum: ['celsius', 'fahrenheit'] }
    },
    required: ['location']
  },
  zodSchema: z.object({
    location: z.string(),
    unit: z.enum(['celsius', 'fahrenheit']).optional()
  })
}

const result = await rosetta.createResponse({
  provider: Provider.OpenAI,
  input: "What's the weather in San Francisco?",
  tools: [getWeatherTool],
  tool_choice: 'auto'
})

if (result.tool_calls) {
  for (const call of result.tool_calls) {
    const args = JSON.parse(call.function.arguments)
    // Execute your function...
    const weather = await getCurrentWeather(args)

    // Continue conversation with result
    const followUp = await rosetta.createResponse({
      provider: Provider.OpenAI,
      input: JSON.stringify(weather),
      previous_response_id: result.id
    })
  }
}
```

#### Structured Output

```typescript
const result = await rosetta.createResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  instructions: 'Extract package information',
  input: 'Install typescript version 5.5.4',
  response_format: {
    type: 'json_schema',
    json_schema: {
      name: 'PackageInfo',
      strict: true,
      schema: {
        type: 'object',
        properties: {
          package_name: { type: 'string' },
          version: { type: 'string' }
        },
        required: ['package_name', 'version'],
        additionalProperties: false
      }
    }
  }
})

const parsed = JSON.parse(result.output_text)
console.log(parsed) // { package_name: 'typescript', version: '5.5.4' }
```

#### Streaming Responses

```typescript
const stream = rosetta.streamResponse({
  provider: Provider.OpenAI,
  model: 'gpt-4o',
  instructions: 'You are a creative writer',
  input: 'Write a haiku about TypeScript',
  stream: true
})

for await (const chunk of stream) {
  switch (chunk.type) {
    case 'response.created':
      console.log(`Stream started: ${chunk.data.id}`)
      break

    case 'response.output_text.delta':
      process.stdout.write(chunk.data.delta)
      break

    case 'response.output_text.done':
      console.log('\n[Text complete]')
      break

    case 'response.tool_call.start':
      console.log(`\nTool call: ${chunk.data.name}`)
      break

    case 'response.tool_call.delta':
      process.stdout.write(chunk.data.delta)
      break

    case 'response.tool_call.done':
      console.log(`\nTool call complete: ${chunk.data.arguments}`)
      break

    case 'response.completed':
      console.log(`\nUsage: ${JSON.stringify(chunk.data.usage)}`)
      break

    case 'response.failed':
      console.error(`Failed: ${chunk.data.error.message}`)
      break

    case 'error':
      console.error(`Error: ${chunk.data.error.message}`)
      break
  }
}
```

**When to use Responses API vs Chat Completions:**

- **Use Responses API** for: Agent applications, stateful conversations, built-in tools, complex workflows
- **Use Chat Completions** for: Provider-agnostic code, simple Q&A, maximum portability

See `examples/responses-api.ts` for a complete demonstration.

## API Reference

Detailed documentation for all exported classes, methods, types, and interfaces is available via JSDoc comments within the source code. Use your IDE's IntelliSense or generate HTML documentation using [TypeDoc](https://typedoc.org/).

Key exports include:

- **Client:** `RosettaAI`
- **Enums:** `Provider`
- **Configuration:** `RosettaAIConfig`, `ProviderOptions`, `CustomProviderConfig`, `ModelListingSourceConfig`
- **Core Parameters:** `GenerateParams`, `EmbedParams`, `SpeechParams`, `TranscribeParams`, `TranslateParams`
- **Core Results:** `GenerateResult`, `EmbedResult`, `TranscriptionResult`, `RosettaModel`, `RosettaModelList`
- **Streaming:** `StreamChunk`, `AudioStreamChunk`
- **Common Types:** `ProviderKey`, `RosettaMessage`, `RosettaContentPart`, `RosettaImageData`, `RosettaAudioData`, `RosettaTool`, `RosettaToolCallRequest`, `TokenUsage`, `Citation`
- **Responses API (OpenAI):** `CreateResponseParams`, `ResponseResult`, `ResponsesStreamChunk`, `ResponsesTool`, `ResponsesInputItem`, `ResponsesToolChoice`, `ResponsesFormat`
- **Errors:** `RosettaAIError`, `ConfigurationError`, `ProviderAPIError`, `UnsupportedFeatureError`, `MappingError`, `InvalidToolDefinitionError`, `ToolArgumentValidationError`

## Examples

Runnable examples demonstrating various features can be found in the `/examples` directory:

- `basic-chat.ts`: Simple non-streaming generation.
- `streaming-chat.ts`: Handling streaming responses and various chunk types.
- `tool-use.ts`: Function calling/tool execution loop.
- `image-input.ts`: Sending images to multimodal models.
- `embeddings.ts`: Generating text embeddings.
- `audio.ts`: Text-to-Speech and Speech-to-Text/Translation.
- `structured-output.ts`: Requesting and validating JSON output.
- `list-models.ts`: Listing available models for configured providers.
- `responses-api.ts`: OpenAI Responses API with stateful conversations, built-in tools, and semantic streaming.
- `custom-provider-*.ts`: Examples for integrating custom OpenAI-compatible providers (Novita, GPUStack, LM Studio).

**To run an example:**

1.  Ensure you have configured your API keys in a `.env` file (see Configuration).
2.  Make sure any required sample files (e.g., `logo.png`, `sample_audio.mp3`) exist in the `examples` directory if needed by the specific example.
3.  Run the build command: `npm run build`
4.  Execute the example using: `npm run example:<name>` (e.g., `npm run example:basic`, `npm run example:stream`, `npm run example:responses`, `npm run example:listmodels`).

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
