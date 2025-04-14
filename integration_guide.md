# RosettaAI SDK Integration Guide

## 1. Introduction

Welcome to the RosettaAI SDK! This guide provides comprehensive instructions for integrating the `@missionsquad/rosetta-ai` SDK into your Node.js TypeScript projects (v20+).

RosettaAI acts as a unified interface, simplifying interactions with various Large Language Model (LLM) providers. By abstracting provider-specific complexities, it allows you to:

- **Write less code:** Use a consistent API (`generate`, `stream`, `embed`, `listModels`, etc.) for different backends.
- **Switch providers easily:** Change the `provider` parameter with minimal code modification.
- **Leverage type safety:** Benefit from TypeScript's strong typing for better developer experience and fewer runtime errors.
- **Integrate quickly:** Easily add support for new providers, especially those compatible with the OpenAI API standard.

This guide covers the integration of all officially supported providers (OpenAI, Azure OpenAI, Anthropic, Google Generative AI, Groq) as well as examples for integrating OpenAI-compatible custom providers (Novita AI, GPUStack, LM Studio). Following this guide should equip you to integrate any of these providers without needing to refer back to the SDK's source code for basic implementation details.

## 2. Setup

### Prerequisites

- **Node.js:** Version 20.0.0 or later. We recommend using [nvm](https://github.com/nvm-sh/nvm) to manage Node.js versions.
- **TypeScript:** Version 5.5 or later.
- **Package Manager:** npm, yarn, or pnpm.

### Installation

Install the SDK in your project:

```bash
# Using npm
npm install @missionsquad/rosetta-ai

# Using yarn
yarn add @missionsquad/rosetta-ai

# Using pnpm
pnpm add @missionsquad/rosetta-ai
```

### TypeScript Configuration (`tsconfig.json`)

Ensure your `tsconfig.json` is configured for modern Node.js development. Key settings include:

```json
{
  "compilerOptions": {
    "target": "ES2022", // Target modern ECMAScript features
    "module": "NodeNext", // Use Node.js's modern module system
    "moduleResolution": "NodeNext", // Aligns with 'module'
    "esModuleInterop": true, // Recommended for interoperability
    "strict": true, // Enable strict type checking (highly recommended)
    "skipLibCheck": true, // Optional: Speeds up compilation
    "forceConsistentCasingInFileNames": true
    // ... other options like outDir, rootDir
  },
  "include": ["src/**/*"], // Adjust to your source directory
  "exclude": ["node_modules"]
}
```

## 3. Configuration

RosettaAI can be configured in two primary ways:

1.  **Environment Variables:** Using a `.env` file (loaded with `dotenv`) or system environment variables.
2.  **Constructor Options:** Passing a configuration object directly to the `RosettaAI` constructor.

Constructor options override environment variables.

### 3.1 Environment Variables (`.env` file)

Create a `.env` file in your project root.

**Required API Keys (add keys for providers you intend to use):**

```dotenv
# .env

# --- Built-in Providers ---
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-... # Used for standard OpenAI OR Azure if AZURE_OPENAI_API_KEY is not set

# --- Azure OpenAI (Alternative to Standard OpenAI) ---
# If using Azure, provide these. They take precedence over OPENAI_API_KEY.
# AZURE_OPENAI_API_KEY=your_azure_openai_api_key
# AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-05-01-preview # Check Azure docs for appropriate version
# AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-deployment-name # Default CHAT deployment ID
# ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name # Default EMBEDDING deployment ID

# --- Custom Provider Examples ---
# Novita AI (OpenAI Compatible)
NOVITA_API_KEY=your_novita_api_key
# NOVITA_BASE_URL=https://api.novita.ai/v3/openai # Optional: Base URL for API calls

# GPUStack (OpenAI Compatible)
GPUSTACK_API_KEY=your_gpustack_api_key # Or however GPUStack handles auth
# GPUSTACK_BASE_URL=https://gpu.crypto-tech.cloud/v1-openai # Optional: Base URL for API calls

# LM Studio (Local OpenAI Compatible - Often no key needed)
# LMSTUDIO_API_KEY= # Usually empty or not needed
# LMSTUDIO_BASE_URL=http://localhost:1234/v1 # Optional: Override default URL
```

**Optional Default Models:**

Set default models to avoid specifying them in every API call.

```dotenv
# .env (continued)

# --- Optional Default Models ---

# Chat Models
ROSETTA_DEFAULT_ANTHROPIC_MODEL=claude-3-haiku-20240307
ROSETTA_DEFAULT_GOOGLE_MODEL=gemini-1.5-flash-latest
ROSETTA_DEFAULT_GROQ_MODEL=llama3-8b-8192
ROSETTA_DEFAULT_OPENAI_MODEL=gpt-4o-mini # Applies to standard OpenAI or Azure if AZURE_OPENAI_DEPLOYMENT_NAME is not set

# Custom Provider Defaults
ROSETTA_DEFAULT_NOVITA_MODEL=meta-llama/llama-3.1-8b-instruct # Example
ROSETTA_DEFAULT_GPUSTACK_MODEL=ibm-granite-3.2-8b-instruct # Example
ROSETTA_DEFAULT_LMSTUDIO_MODEL=lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF # Example (ensure model is loaded in LM Studio)

# Embedding Models
ROSETTA_DEFAULT_EMBEDDING_GOOGLE_MODEL=text-embedding-004
ROSETTA_DEFAULT_EMBEDDING_OPENAI_MODEL=text-embedding-3-small # Applies to standard OpenAI or Azure if ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is not set
ROSETTA_DEFAULT_EMBEDDING_GROQ_MODEL=nomic-embed-text-v1.5
ROSETTA_DEFAULT_EMBEDDING_GPUSTACK_MODEL=nomic-embed-text-v1.5 # Example custom default

# Audio Models (TTS/STT)
ROSETTA_DEFAULT_TTS_OPENAI_MODEL=tts-1 # Applies to standard OpenAI or Azure
ROSETTA_DEFAULT_STT_OPENAI_MODEL=whisper-1 # Applies to standard OpenAI or Azure
ROSETTA_DEFAULT_STT_GROQ_MODEL=whisper-large-v3
```

**Loading `.env`:**

Remember to install and load `dotenv` early in your application entry point:

```typescript
import dotenv from 'dotenv'
dotenv.config() // Load .env variables into process.env

// ... rest of your application logic ...
import { RosettaAI } from '@missionsquad/rosetta-ai'
const rosetta = new RosettaAI() // Now reads from process.env
```

### 3.2 Constructor Options

Pass configuration directly when creating the `RosettaAI` instance.

```typescript
import { RosettaAI, Provider, RosettaAIConfig, CustomProviderConfig } from '@missionsquad/rosetta-ai'
import { OpenAICompatibleMapper } from '@missionsquad/rosetta-ai/dist/core/mapping/openai-compatible.mapper' // Adjust path if needed

// --- Configuration for Custom Providers ---
const lmstudioProviderKey = 'lmstudio'
const lmstudioConfig: CustomProviderConfig = {
  providerKey: lmstudioProviderKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'list_models'], // Added list_models
  baseURL: process.env.LMSTUDIO_BASE_URL || 'http://localhost:1234/v1',
  apiKey: process.env.LMSTUDIO_API_KEY || undefined, // Often not needed
  defaultModel: process.env.ROSETTA_DEFAULT_LMSTUDIO_MODEL || 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF',
  // modelListPath: '/models', // Default, can be omitted
  toolConfig: {
    /* ... defaults are usually fine for OpenAI compatible ... */
  }
}

const novitaProviderKey = 'novita'
const novitaConfig: CustomProviderConfig = {
  providerKey: novitaProviderKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'list_models'], // Added list_models
  baseURL: 'https://api.novita.ai/v3/openai',
  apiKey: process.env.NOVITA_API_KEY, // Loaded from env here, but could be passed directly
  defaultModel: process.env.ROSETTA_DEFAULT_NOVITA_MODEL || 'meta-llama/llama-3.1-8b-instruct',
  // modelListPath: '/models', // Default, can be omitted
  toolConfig: {
    /* ... defaults ... */
  }
}

const gpustackProviderKey = 'gpustack'
const gpustackConfig: CustomProviderConfig = {
  providerKey: gpustackProviderKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'embed', 'list_models'], // Added list_models, embed
  baseURL: 'https://gpu.crypto-tech.cloud/v1-openai',
  apiKey: process.env.GPUSTACK_API_KEY,
  defaultModel: process.env.ROSETTA_DEFAULT_GPUSTACK_MODEL ?? 'ibm-granite-3.2-8b-instruct',
  defaultEmbeddingModel: process.env.ROSETTA_DEFAULT_EMBEDDING_GPUSTACK_MODEL ?? 'nomic-embed-text-v1.5',
  // modelListPath: '/models', // Default, can be omitted
  toolConfig: {
    /* ... defaults ... */
  }
}

// Example: Custom provider with a non-standard model list path/URL
const customProviderWithListPathKey = 'custom-path'
const customProviderWithListPathConfig: CustomProviderConfig = {
  providerKey: customProviderWithListPathKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://my-custom-api.com/api', // Base for generate
  apiKey: 'my-custom-key',
  defaultModel: 'custom-gen-model',
  modelListPath: '/inventory/llms' // Custom relative path for model listing
}

const customProviderWithListUrlKey = 'custom-url'
const customProviderWithListUrlConfig: CustomProviderConfig = {
  providerKey: customProviderWithListUrlKey,
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://my-custom-api.com/api', // Base for generate
  apiKey: 'my-custom-key',
  defaultModel: 'custom-gen-model',
  modelListUrl: 'https://models.my-other-service.com/list' // Absolute URL for model listing
}

// --- Main RosettaAI Configuration ---
const config: RosettaAIConfig = {
  // Built-in Provider Keys
  anthropicApiKey: 'sk-ant-...', // Direct key example
  googleApiKey: process.env.GOOGLE_API_KEY, // Load from env example
  groqApiKey: 'gsk_...',
  // openaiApiKey: 'sk-...', // Standard OpenAI key (optional if using Azure)

  // Azure Configuration (takes precedence over openaiApiKey)
  azureOpenAIApiKey: 'your_azure_key',
  azureOpenAIEndpoint: 'https://your-azure-resource.openai.azure.com/',
  azureOpenAIApiVersion: '2024-05-01-preview',
  azureOpenAIDefaultChatDeploymentName: 'my-gpt4-deployment',
  azureOpenAIDefaultEmbeddingDeploymentName: 'my-embedding-deployment',

  // Default Models (Overrides env vars if set)
  defaultModels: {
    [Provider.Anthropic]: 'claude-3-sonnet-20240229',
    [Provider.Google]: 'gemini-1.5-pro-latest',
    // OpenAI default uses Azure deployment name if Azure is configured
    [Provider.OpenAI]: 'gpt-4o', // Example: Overrides Azure default if needed, otherwise uses azureOpenAIDefaultChatDeploymentName
    // Custom provider defaults
    [lmstudioProviderKey]: 'local-llama3-8b',
    [novitaProviderKey]: 'novita-default-model',
    [gpustackProviderKey]: 'gpustack-default-model',
    [customProviderWithListPathKey]: 'custom-gen-model',
    [customProviderWithListUrlKey]: 'custom-gen-model'
  },
  defaultEmbeddingModels: {
    [Provider.Google]: 'text-embedding-004',
    // OpenAI default uses Azure deployment name if Azure is configured
    [Provider.OpenAI]: 'text-embedding-3-large', // Example: Overrides Azure default if needed
    [gpustackProviderKey]: 'gpustack-embed-model'
  },
  defaultSttModels: {
    [Provider.Groq]: 'whisper-large-v3'
  },

  // Custom Providers
  customProviders: [
    lmstudioConfig,
    novitaConfig,
    gpustackConfig,
    customProviderWithListPathConfig,
    customProviderWithListUrlConfig
  ],

  // Optional: Global Retries/Timeout
  defaultMaxRetries: 3,
  defaultTimeoutMs: 90000, // 90 seconds

  // Optional: Provider-specific options
  providerOptions: {
    [Provider.Google]: {
      googleApiVersion: 'v1beta' // Use beta API for Google
    },
    [Provider.OpenAI]: {
      // Example: Override specific deployment for a specific call later
      // azureChatDeploymentId: 'special-chat-deployment'
    }
  },

  // Optional: Override model listing source for specific built-in providers
  modelListingConfig: {
    [Provider.OpenAI]: { type: 'apiEndpoint', url: 'https://my-proxy.com/openai/models' } // Example: Use a proxy for OpenAI models
  }
}

// Initialize RosettaAI with the configuration object
const rosetta = new RosettaAI(config)

console.log('Initialized RosettaAI with constructor config.')
console.log('Available providers:', rosetta.getConfiguredProviders())
```

## 4. Core Concepts

### `RosettaAI` Client

The main entry point for interacting with the SDK. You instantiate it once with your configuration.

```typescript
import { RosettaAI } from '@missionsquad/rosetta-ai'
import dotenv from 'dotenv'

dotenv.config() // Load .env if using environment variables

const rosetta = new RosettaAI(/* optional config object */)
```

### `Provider` Enum & `ProviderKey` Type

- **`Provider` Enum:** Used to specify built-in providers (`Provider.OpenAI`, `Provider.Anthropic`, `Provider.Google`, `Provider.Groq`).
- **`ProviderKey` Type:** A string type that accepts either a `Provider` enum value or a custom provider key string (like `'lmstudio'`, `'novita'`, `'gpustack'`). This is used in the `provider` field of parameter objects (`GenerateParams`, `EmbedParams`, `listModels`, etc.).

```typescript
import { Provider, ProviderKey, GenerateParams } from '@missionsquad/rosetta-ai';

const openAIParams: GenerateParams = {
  provider: Provider.OpenAI, // Using enum for built-in
  messages: [...],
  // ...
};

const lmstudioParams: GenerateParams = {
  provider: 'lmstudio', // Using string key for custom provider
  messages: [...],
  // ...
};

// Listing models for a custom provider
// await rosetta.listModels('lmstudio');
```

## 5. Usage Examples

These examples demonstrate common use cases. Replace the `provider` value with any configured provider key (built-in enum or custom string).

### 5.1 Basic Chat (`generate`)

For simple request-response interactions.

```typescript
import {
  RosettaAI,
  Provider,
  RosettaMessage,
  GenerateParams,
  GenerateResult,
  ProviderKey
} from '@missionsquad/rosetta-ai'

async function basicChat(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Basic Chat (${provider}) ---`)
  try {
    const messages: RosettaMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' }
    ]

    const params: GenerateParams = {
      provider: provider,
      // model: 'optional-model-override', // Uses configured default if omitted
      messages: messages,
      maxTokens: 50,
      temperature: 0.7
    }

    const result: GenerateResult = await rosetta.generate(params)

    console.log(`[${result.model}] Response:`)
    console.log(result.content)
    console.log('Finish Reason:', result.finishReason)
    console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
  } catch (error) {
    console.error(`[${provider}] Generation failed: ${error.name} - ${error.message}`)
  }
}

// Assuming 'rosetta' is an initialized RosettaAI instance
// await basicChat(rosetta, Provider.OpenAI);
// await basicChat(rosetta, 'lmstudio'); // Example with custom provider
```

### 5.2 Streaming Chat (`stream`)

For processing responses chunk by chunk, ideal for real-time UI updates.

```typescript
import { RosettaAI, Provider, RosettaMessage, GenerateParams, StreamChunk, ProviderKey } from '@missionsquad/rosetta-ai'

async function streamingChat(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Streaming Chat (${provider}) ---`)
  try {
    const params: GenerateParams = {
      provider: provider,
      messages: [{ role: 'user', content: 'Write a short haiku about TypeScript.' }],
      maxTokens: 60
    }

    const stream: AsyncIterable<StreamChunk> = rosetta.stream(params)

    console.log(`[Streaming Response]`)
    let fullContent = ''
    let finalResult: GenerateResult | null = null

    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'message_start':
          console.log(`\nStream started (Model: ${chunk.data.model})...`)
          break
        case 'content_delta':
          process.stdout.write(chunk.data.delta) // Write delta to console
          fullContent += chunk.data.delta
          break
        case 'message_stop':
          console.log(`\n--- Stream Stopped (Reason: ${chunk.data.finishReason}) ---`)
          break
        case 'final_usage':
          console.log('\nFinal Usage:', chunk.data.usage ? JSON.stringify(chunk.data.usage) : 'N/A')
          break
        case 'final_result':
          finalResult = chunk.data.result
          console.log('\n--- Final Aggregated Result Received ---')
          break
        case 'error':
          console.error('\n--- Stream Error ---')
          console.error(`${chunk.data.error.name}: ${chunk.data.error.message}`)
          return // Stop processing on error
        // Add cases for other chunk types (tool_call_*, json_*, thinking_*, citation_*) as needed
        default:
          // console.log(`\n[Chunk Type: ${chunk.type}]`); // Log unhandled chunk types
          break
      }
    }
    console.log('\n--- Stream Complete ---')
    // console.log("Final Accumulated Content:", fullContent);
    // console.log("Final Result Object:", finalResult ? JSON.stringify(finalResult, null, 2) : 'N/A');
  } catch (error) {
    // Errors during stream *setup* (e.g., invalid config before iteration starts)
    console.error(`[${provider}] Streaming setup failed: ${error.name} - ${error.message}`)
  }
}

// await streamingChat(rosetta, Provider.Groq);
// await streamingChat(rosetta, 'novita');
```

### 5.3 Embeddings (`embed`)

Generate vector representations of text. Supported by OpenAI/Azure, Google, Groq, and potentially custom providers.

```typescript
import { RosettaAI, Provider, EmbedParams, EmbedResult, ProviderKey } from '@missionsquad/rosetta-ai'

async function generateEmbeddings(
  rosetta: RosettaAI,
  provider: Provider.OpenAI | Provider.Google | Provider.Groq | 'gpustack'
) {
  console.log(`\n--- Embeddings (${provider}) ---`)
  try {
    const textsToEmbed = ['RosettaAI simplifies LLM interactions.', 'TypeScript adds static typing to JavaScript.']

    // Note: Groq's nomic-embed-text might only handle single strings effectively in some SDK versions/setups.
    // Adjust input based on provider if needed.
    const inputData = provider === Provider.Groq ? textsToEmbed[0] : textsToEmbed

    const params: EmbedParams = {
      provider: provider,
      // model: 'optional-embedding-model-override', // Uses configured default if omitted
      input: inputData
      // dimensions: 256 // Optional: For OpenAI models supporting reduced dimensions
    }

    const result: EmbedResult = await rosetta.embed(params)

    console.log(`Generated ${result.embeddings.length} embedding(s) using model: ${result.model}`)
    result.embeddings.forEach((vec, i) => {
      console.log(`  Embedding ${i + 1} (Dim: ${vec.length}): [${vec.slice(0, 3).join(', ')}...]`)
    })
    console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
  } catch (error) {
    console.error(`[${provider}] Embedding failed: ${error.name} - ${error.message}`)
  }
}

// await generateEmbeddings(rosetta, Provider.OpenAI);
// await generateEmbeddings(rosetta, Provider.Google);
// await generateEmbeddings(rosetta, 'gpustack'); // Example with custom provider
```

### 5.4 Tool Use / Function Calling

Instruct models to use predefined tools.

```typescript
import {
  RosettaAI,
  Provider,
  RosettaTool,
  RosettaMessage,
  GenerateParams,
  GenerateResult,
  ProviderKey
} from '@missionsquad/rosetta-ai'
import { z } from 'zod' // Import Zod

// 1. Define Zod schema for validation
const GetWeatherToolSchema = z.object({
  location: z.string().describe('The city and state/country, e.g., San Francisco, CA'),
  unit: z
    .enum(['celsius', 'fahrenheit'])
    .optional()
    .default('fahrenheit')
})

// 2. Define the tool using RosettaTool interface, including the Zod schema
const getWeatherTool: RosettaTool<typeof GetWeatherToolSchema> = {
  type: 'function',
  function: {
    name: 'getCurrentWeather',
    description: 'Get the current weather for a specific location.',
    parameters: {
      // JSON Schema for the provider
      type: 'object',
      properties: {
        location: { type: 'string', description: 'The city and state/country' },
        unit: { type: 'string', enum: ['celsius', 'fahrenheit'], description: 'Temperature unit' }
      },
      required: ['location']
    },
    zodSchema: GetWeatherToolSchema // Zod schema for validation
  }
}

// 3. Implement the actual tool function
async function getCurrentWeather(location: string, unit: string = 'fahrenheit'): Promise<string> {
  console.log(`[TOOL EXECUTION] Getting weather for ${location} in ${unit}...`)
  // Simulate API call
  await new Promise(r => setTimeout(r, 100))
  // Return result as a JSON string
  return JSON.stringify({ temperature: unit === 'celsius' ? 22 : 72, condition: 'Sunny' })
}

// 4. Conversation Loop
async function runToolConversation(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Tool Use Conversation (${provider}) ---`)
  const messages: RosettaMessage[] = [{ role: 'user', content: "What's the weather in San Francisco?" }]
  const maxTurns = 5 // Prevent infinite loops

  for (let i = 0; i < maxTurns; i++) {
    console.log(`\nTurn ${i + 1}: Sending request...`)
    try {
      const params: GenerateParams = {
        provider: provider,
        messages: messages,
        tools: [getWeatherTool],
        toolChoice: 'auto' // Let the model decide
      }
      const response: GenerateResult = await rosetta.generate(params)

      console.log('Assistant:', response.content ?? '[No text content]')
      // Add assistant message (including tool calls) to history
      messages.push({ role: 'assistant', content: response.content, toolCalls: response.toolCalls })

      if (response.toolCalls && response.toolCalls.length > 0) {
        console.log('Tool Calls Requested:', response.toolCalls.length)
        const toolResults: RosettaMessage[] = []

        for (const call of response.toolCalls) {
          if (call.function.name === 'getCurrentWeather') {
            let resultJsonString: string
            let isError = false
            try {
              // Arguments are validated by the SDK's mapper before returning `response`
              // if the mapper throws ToolArgumentValidationError.
              // We still need to parse the raw string here for the tool implementation.
              const args = JSON.parse(call.function.arguments)
              // Execute the tool
              resultJsonString = await getCurrentWeather(args.location, args.unit)
              console.log(` -> Tool Result OK for call ${call.id}`)
            } catch (e) {
              isError = true
              console.error(` -> Tool Execution Error for call ${call.id}: ${e.message}`)
              resultJsonString = JSON.stringify({ error: e.message })
            }
            // Add tool result message
            toolResults.push({ role: 'tool', toolCallId: call.id, content: resultJsonString, isError })
          } else {
            console.warn(`[WARNING] Model called unknown tool: ${call.function.name}`)
            toolResults.push({
              role: 'tool',
              toolCallId: call.id,
              content: `{"error": "Unknown tool: ${call.function.name}"}`,
              isError: true
            })
          }
        }
        messages.push(...toolResults) // Add results for the next turn
      } else {
        console.log('\n--- Conversation End (No Tool Calls) ---')
        break // Exit loop if no tool calls
      }
    } catch (error) {
      console.error(`[${provider}] Tool conversation error: ${error.name} - ${error.message}`)
      // Handle specific errors like ToolArgumentValidationError if needed
      break
    }
  }
}

// await runToolConversation(rosetta, Provider.OpenAI);
// await runToolConversation(rosetta, 'gpustack');
```

### 5.5 Image Input

Send images along with text prompts (multimodal). Supported by OpenAI/Azure, Anthropic, Google.

```typescript
import {
  RosettaAI,
  Provider,
  RosettaMessage,
  RosettaImageData,
  ImageMimeType,
  GenerateParams,
  GenerateResult,
  ProviderKey
} from '@missionsquad/rosetta-ai'
import fs from 'fs/promises'
import path from 'path'

// Helper to load and encode image
async function loadImageData(imagePath: string): Promise<RosettaImageData | null> {
  try {
    const buffer = await fs.readFile(imagePath)
    const base64Data = buffer.toString('base64')
    const ext = path.extname(imagePath).toLowerCase()
    const mimeTypeMap: Record<string, ImageMimeType> = {
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.webp': 'image/webp'
    }
    const mimeType = mimeTypeMap[ext]
    if (!mimeType) {
      console.warn(`Unsupported image type: ${ext}`)
      return null
    }
    return { mimeType, base64Data }
  } catch (error) {
    console.error(`Error loading image ${imagePath}: ${error.message}`)
    return null
  }
}

async function describeImage(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Image Input (${provider}) ---`)
  const imagePath = path.join(__dirname, 'logo.png') // Ensure logo.png exists
  const imageData = await loadImageData(imagePath)

  if (!imageData) {
    console.error('Could not load image data.')
    return
  }

  try {
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

    const params: GenerateParams = {
      provider: provider,
      // model: 'optional-vision-model-override', // e.g., gpt-4-vision-preview, claude-3-opus-20240229, gemini-pro-vision
      messages: messages,
      maxTokens: 150
    }

    const result: GenerateResult = await rosetta.generate(params)

    console.log(`[${result.model}] Image Description:`)
    console.log(result.content)
  } catch (error) {
    console.error(`[${provider}] Image input failed: ${error.name} - ${error.message}`)
  }
}

// await describeImage(rosetta, Provider.OpenAI);
// await describeImage(rosetta, Provider.Google);
```

### 5.6 Text-to-Speech (TTS)

Generate speech from text. Currently uses OpenAI/Azure provider.

```typescript
import { RosettaAI, Provider, SpeechParams } from '@missionsquad/rosetta-ai'
import fs from 'fs/promises'
import path from 'path'

async function generateSpeech(rosetta: RosettaAI) {
  console.log(`\n--- Text-to-Speech (TTS) ---`)
  const outputDir = path.join(__dirname, 'audio_output')
  await fs.mkdir(outputDir, { recursive: true })
  const filePath = path.join(outputDir, 'tts_output.mp3')

  try {
    const params: SpeechParams = {
      provider: Provider.OpenAI, // Required provider
      input: 'Hello from RosettaAI! Speech synthesis is working.',
      voice: 'alloy', // e.g., alloy, echo, fable, onyx, nova, shimmer
      // model: 'tts-1-hd', // Optional model override
      responseFormat: 'mp3'
    }

    // Non-streaming generation
    const audioBuffer: Buffer = await rosetta.generateSpeech(params)
    await fs.writeFile(filePath, audioBuffer)
    console.log(`TTS audio saved to ${filePath} (${(audioBuffer.length / 1024).toFixed(1)} KB)`)

    // Example: Streaming TTS (optional)
    // const stream = rosetta.streamSpeech(params);
    // const writeStream = fs.createWriteStream(path.join(outputDir, 'tts_stream.mp3'));
    // for await (const chunk of stream) {
    //   if (chunk.type === 'audio_chunk') writeStream.write(chunk.data);
    //   else if (chunk.type === 'error') throw chunk.data.error;
    // }
    // writeStream.end();
    // console.log('Streamed TTS audio saved.');
  } catch (error) {
    console.error(`TTS failed: ${error.name} - ${error.message}`)
  }
}

// await generateSpeech(rosetta);
```

### 5.7 Speech-to-Text (STT) / Transcription

Transcribe audio to text. Supported by OpenAI/Azure and Groq.

```typescript
import {
  RosettaAI,
  Provider,
  TranscribeParams,
  RosettaAudioData,
  TranscriptionResult,
  ProviderKey
} from '@missionsquad/rosetta-ai'
import fs from 'fs/promises'
import path from 'path'

async function transcribeAudio(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Speech-to-Text (${provider}) ---`)
  // Use the TTS output or a sample file
  const audioPath = path.join(__dirname, 'audio_output', 'tts_output.mp3') // Or path/to/your/sample.mp3
  try {
    const buffer = await fs.readFile(audioPath)
    const audioData: RosettaAudioData = {
      data: buffer,
      filename: path.basename(audioPath),
      mimeType: 'audio/mpeg' // Adjust if using a different format
    }

    const params: TranscribeParams = {
      provider: provider,
      audio: audioData,
      // model: 'whisper-1', // Optional model override
      // language: 'en', // Optional language hint
      responseFormat: 'text' // Get plain text
    }

    const result: TranscriptionResult = await rosetta.transcribe(params)
    console.log(`[${result.model}] Transcription: "${result.text}"`)
  } catch (error) {
    if ((error as any).code === 'ENOENT') {
      console.error(
        `[${provider}] STT failed: Audio file not found at ${audioPath}. Run TTS first or provide a sample.`
      )
    } else {
      console.error(`[${provider}] STT failed: ${error.name} - ${error.message}`)
    }
  }
}

// await transcribeAudio(rosetta, Provider.OpenAI);
// await transcribeAudio(rosetta, Provider.Groq);
```

### 5.8 Audio Translation

Translate audio into English text. Supported by OpenAI/Azure and Groq.

```typescript
import {
  RosettaAI,
  Provider,
  TranslateParams,
  RosettaAudioData,
  TranscriptionResult,
  ProviderKey
} from '@missionsquad/rosetta-ai'
import fs from 'fs/promises'
import path from 'path'

async function translateAudio(rosetta: RosettaAI, provider: ProviderKey) {
  console.log(`\n--- Audio Translation (${provider}) ---`)
  // Use the TTS output or a sample file (ideally in another language)
  const audioPath = path.join(__dirname, 'audio_output', 'tts_output.mp3') // Or path/to/your/sample_other_language.mp3
  try {
    const buffer = await fs.readFile(audioPath)
    const audioData: RosettaAudioData = {
      data: buffer,
      filename: path.basename(audioPath),
      mimeType: 'audio/mpeg' // Adjust if using a different format
    }

    const params: TranslateParams = {
      provider: provider,
      audio: audioData,
      // model: 'whisper-1', // Optional model override
      responseFormat: 'text'
    }

    const result: TranscriptionResult = await rosetta.translate(params)
    console.log(`[${result.model}] Translation: "${result.text}"`)
  } catch (error) {
    if ((error as any).code === 'ENOENT') {
      console.error(
        `[${provider}] Translation failed: Audio file not found at ${audioPath}. Run TTS first or provide a sample.`
      )
    } else {
      console.error(`[${provider}] Translation failed: ${error.name} - ${error.message}`)
    }
  }
}

// await translateAudio(rosetta, Provider.OpenAI);
// await translateAudio(rosetta, Provider.Groq);
```

### 5.9 Error Handling

Catch specific error types exported by the SDK.

```typescript
import {
  RosettaAI,
  Provider,
  GenerateParams,
  RosettaAIError,
  ConfigurationError,
  ProviderAPIError,
  UnsupportedFeatureError,
  MappingError
} from '@missionsquad/rosetta-ai'

async function safeGenerate(rosetta: RosettaAI) {
  console.log(`\n--- Error Handling Example ---`)
  try {
    const params: GenerateParams = {
      provider: Provider.OpenAI,
      model: 'invalid-model-id', // Intentionally invalid
      messages: [{ role: 'user', content: 'Test' }]
    }
    await rosetta.generate(params)
  } catch (error) {
    if (error instanceof ConfigurationError) {
      console.error(`Caught Configuration Error: ${error.message}`)
      // e.g., Missing API key, invalid deployment ID
    } else if (error instanceof ProviderAPIError) {
      console.error(
        `Caught Provider API Error (${error.provider || error.customProvider}): Status ${error.statusCode ??
          'N/A'}, Code: ${error.errorCode ?? 'N/A'} - ${error.message}`
      )
      // e.g., Rate limit, authentication error, invalid request to provider
      // console.error("Underlying error:", error.underlyingError); // Log original error if needed
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(
        `Caught Unsupported Feature Error: ${error.provider || error.customProvider} does not support ${error.feature}.`
      )
      // e.g., Trying TTS with Groq, Embeddings with Anthropic
    } else if (error instanceof MappingError) {
      console.error(`Caught Internal SDK Mapping Error: ${error.message}`)
      // e.g., Failed to convert data between RosettaAI and provider format
    } else if (error instanceof RosettaAIError) {
      // Catch any other base SDK errors
      console.error(`Caught RosettaAI Error: ${error.name} - ${error.message}`)
    } else {
      // Catch unexpected errors
      console.error('Caught Unexpected Error:', error)
    }
  }
}

// await safeGenerate(rosetta);
```

### 5.10 Model Listing

List available models for configured providers (built-in and custom).

```typescript
import { RosettaAI, Provider, ProviderKey, RosettaModelList, RosettaAIError } from '@missionsquad/rosetta-ai'

async function listModels(rosetta: RosettaAI) {
  console.log(`\n--- Model Listing Example ---`)
  const configuredProviders = rosetta.getConfiguredProviders()

  // --- List models for a specific provider (built-in or custom) ---
  const providerToInspect: ProviderKey = 'lmstudio' // Change as needed (e.g., Provider.Groq, 'custom-path')
  if (configuredProviders.includes(providerToInspect)) {
    console.log(`\nListing models for: ${providerToInspect}`)
    try {
      // Optional: Override listing source for this specific call
      // const sourceOverride: ModelListingSourceConfig = { type: 'apiEndpoint', url: '...' };
      // const models: RosettaModelList = await rosetta.listModels(providerToInspect, sourceOverride);

      const models: RosettaModelList = await rosetta.listModels(providerToInspect)
      console.log(`Found ${models.data.length} models:`)
      models.data.slice(0, 5).forEach(m => console.log(`  - ${m.id} (Owned by: ${m.owned_by})`))
      if (models.data.length > 5) console.log('  ...')
    } catch (error) {
      console.error(`Error listing models for ${providerToInspect}: ${error.message}`)
    }
  } else {
    console.log(`Provider ${providerToInspect} not configured, skipping single list.`)
  }

  // --- List models for ALL configured providers ---
  console.log('\nListing models for ALL configured providers:')
  try {
    const allResults = await rosetta.listAllModels()
    for (const providerKey in allResults) {
      console.log(`\nProvider: ${providerKey}`)
      const result = allResults[providerKey]
      if (result instanceof RosettaAIError) {
        console.error(`  Error: ${result.message}`)
      } else if (result) {
        console.log(`  Found ${result.data.length} models:`)
        result.data.slice(0, 5).forEach(m => console.log(`    - ${m.id}`))
        if (result.data.length > 5) console.log('    ...')
      }
    }
  } catch (error) {
    // This catch is less likely for listAllModels itself, errors are per-provider
    console.error(`Unexpected error during listAllModels: ${error.message}`)
  }
}

// await listModels(rosetta);
```

## 6. Provider-Specific Details

### 6.1 OpenAI (Standard)

- **Provider Enum:** `Provider.OpenAI`
- **Configuration:**
  - `.env`: `OPENAI_API_KEY=sk-...`
  - Constructor: `openaiApiKey: 'sk-...'`
- **Default Models:**
  - Chat: `ROSETTA_DEFAULT_OPENAI_MODEL` (e.g., `gpt-4o-mini`)
  - Embedding: `ROSETTA_DEFAULT_EMBEDDING_OPENAI_MODEL` (e.g., `text-embedding-3-small`)
  - TTS: `ROSETTA_DEFAULT_TTS_OPENAI_MODEL` (e.g., `tts-1`)
  - STT: `ROSETTA_DEFAULT_STT_OPENAI_MODEL` (e.g., `whisper-1`)
- **Features:** Supports Chat (Generate/Stream), Image Input, Tool Use, Embeddings, JSON Mode, TTS, STT, Translation, Model Listing (via API).

### 6.2 Azure OpenAI

- **Provider Enum:** `Provider.OpenAI` (Uses the same enum as standard OpenAI)
- **Configuration (Takes precedence over standard OpenAI keys):**
  - `.env`:
    - `AZURE_OPENAI_API_KEY=...`
    - `AZURE_OPENAI_ENDPOINT=https://....openai.azure.com/`
    - `AZURE_OPENAI_API_VERSION=YYYY-MM-DD`
    - `AZURE_OPENAI_DEPLOYMENT_NAME=your-chat-deployment` (Default chat model)
    - `ROSETTA_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment` (Default embedding model)
  - Constructor: `azureOpenAIApiKey`, `azureOpenAIEndpoint`, `azureOpenAIApiVersion`, `azureOpenAIDefaultChatDeploymentName`, `azureOpenAIDefaultEmbeddingDeploymentName`.
- **Default Models:** Uses the deployment names set in the configuration.
- **Features:** Same as standard OpenAI, but capabilities depend on the specific models deployed in your Azure resource. Model listing uses the standard OpenAI endpoint unless overridden via `modelListingConfig`.

### 6.3 Anthropic

- **Provider Enum:** `Provider.Anthropic`
- **Configuration:**
  - `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
  - Constructor: `anthropicApiKey: 'sk-ant-...'`
- **Default Models:**
  - Chat: `ROSETTA_DEFAULT_ANTHROPIC_MODEL` (e.g., `claude-3-haiku-20240307`)
- **Features:** Supports Chat (Generate/Stream), Image Input, Tool Use, Thinking Steps. Model Listing uses a static list internal to the SDK.
- **Limitations:** No native Embeddings, TTS, STT, or Translation APIs. JSON mode requires careful prompting.

### 6.4 Google Generative AI

- **Provider Enum:** `Provider.Google`
- **Configuration:**
  - `.env`: `GOOGLE_API_KEY=AIza...`
  - Constructor: `googleApiKey: 'AIza...'`
- **Default Models:**
  - Chat: `ROSETTA_DEFAULT_GOOGLE_MODEL` (e.g., `gemini-1.5-flash-latest`)
  - Embedding: `ROSETTA_DEFAULT_EMBEDDING_GOOGLE_MODEL` (e.g., `text-embedding-004`)
- **Features:** Supports Chat (Generate/Stream), Image Input, Tool Use, Embeddings, Grounding/Citations (via Search tool), Model Listing (via API endpoint).
- **Limitations:** JSON mode requires prompting. STT requires a separate Google Cloud Speech client (not integrated into this SDK directly). No native TTS or Translation via the Generative AI API.

### 6.5 Groq

- **Provider Enum:** `Provider.Groq`
- **Configuration:**
  - `.env`: `GROQ_API_KEY=gsk_...`
  - Constructor: `groqApiKey: 'gsk_...'`
- **Default Models:**
  - Chat: `ROSETTA_DEFAULT_GROQ_MODEL` (e.g., `llama3-8b-8192`)
  - Embedding: `ROSETTA_DEFAULT_EMBEDDING_GROQ_MODEL` (e.g., `nomic-embed-text-v1.5`)
  - STT: `ROSETTA_DEFAULT_STT_GROQ_MODEL` (e.g., `whisper-large-v3`)
- **Features:** Supports Chat (Generate/Stream), Tool Use, Embeddings, STT, Translation, Model Listing (via SDK method).
- **Limitations:** Image input support varies by model. No native TTS. JSON mode requires prompting.

### 6.6 Custom Providers (OpenAI-Compatible Examples)

Integrating providers that offer an OpenAI-compatible API is straightforward using the built-in `OpenAICompatibleMapper`.

**General Pattern:**

1.  **Define a `CustomProviderConfig`:**
    - `providerKey`: A unique string (e.g., `'novita'`, `'gpustack'`, `'lmstudio'`).
    - `mapper`: Set to `OpenAICompatibleMapper`.
    - `supportedFeatures`: List the features the provider's endpoint supports (e.g., `['generate', 'stream', 'list_models']`). **Include `'list_models'` if the provider has an OpenAI-compatible `/models` endpoint.**
    - `baseURL`: The provider's OpenAI-compatible API endpoint URL (used for generate, stream, etc., and as base for default model listing).
    - `apiKey`: The API key (loaded from env or passed directly).
    - `defaultModel`: (Optional) The default model ID for this provider.
    - `toolConfig`: Usually the defaults are fine (`{ toolDefinitionFormat: 'jsonSchema', toolCallInputFormat: 'jsonString', toolResultFormat: 'jsonString' }`).
    - **Model Listing (Optional Overrides):**
      - `modelListPath?: string`: If the models endpoint is not at `/models` relative to `baseURL`, specify the path here (e.g., `/openai/models`). Defaults to `/models`.
      - `modelListUrl?: string`: If the models endpoint is at a completely different URL, specify the full URL here (overrides `baseURL` + `modelListPath`).
2.  **Add to `RosettaAIConfig`:** Include the config object in the `customProviders` array when initializing `RosettaAI`.
3.  **Use the `providerKey`:** Use the unique string key when calling SDK methods (`generate`, `stream`, `listModels`, etc.).

**Example Configurations (as used in Constructor Options):**

```typescript
import { CustomProviderConfig } from '@missionsquad/rosetta-ai'
import { OpenAICompatibleMapper } from '@missionsquad/rosetta-ai/dist/core/mapping/openai-compatible.mapper' // Adjust path if needed

// Novita AI
const novitaConfig: CustomProviderConfig = {
  providerKey: 'novita',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'list_models'], // Added list_models
  baseURL: 'https://api.novita.ai/v3/openai',
  apiKey: process.env.NOVITA_API_KEY,
  defaultModel: process.env.ROSETTA_DEFAULT_NOVITA_MODEL || 'meta-llama/llama-3.1-8b-instruct',
  // modelListPath: '/models', // Default, can omit
  toolConfig: {
    /* defaults */
  }
}

// GPUStack
const gpustackConfig: CustomProviderConfig = {
  providerKey: 'gpustack',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'embed', 'list_models'], // Added list_models, embed
  baseURL: 'https://gpu.crypto-tech.cloud/v1-openai',
  apiKey: process.env.GPUSTACK_API_KEY,
  defaultModel: process.env.ROSETTA_DEFAULT_GPUSTACK_MODEL ?? 'ibm-granite-3.2-8b-instruct',
  defaultEmbeddingModel: process.env.ROSETTA_DEFAULT_EMBEDDING_GPUSTACK_MODEL ?? 'nomic-embed-text-v1.5',
  // modelListPath: '/models', // Default, can omit
  toolConfig: {
    /* defaults */
  }
}

// LM Studio (Local)
const lmstudioConfig: CustomProviderConfig = {
  providerKey: 'lmstudio',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'stream', 'tool_use', 'list_models'], // Added list_models
  baseURL: process.env.LMSTUDIO_BASE_URL || 'http://localhost:1234/v1',
  apiKey: process.env.LMSTUDIO_API_KEY || undefined, // Often not needed
  defaultModel: process.env.ROSETTA_DEFAULT_LMSTUDIO_MODEL || 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF', // Ensure this model is loaded
  // modelListPath: '/models', // Default, can omit
  toolConfig: {
    /* defaults */
  }
}

// Custom provider with non-standard model list path
const customPathConfig: CustomProviderConfig = {
  providerKey: 'custom-path-provider',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://api.example.com/v2',
  apiKey: 'key-123',
  defaultModel: 'model-a',
  modelListPath: '/openai-compat/models' // Custom path relative to baseURL
}

// Custom provider with absolute model list URL
const customUrlConfig: CustomProviderConfig = {
  providerKey: 'custom-url-provider',
  mapper: OpenAICompatibleMapper,
  supportedFeatures: ['generate', 'list_models'],
  baseURL: 'https://api.example.com/v2', // Used for generate
  apiKey: 'key-456',
  defaultModel: 'model-b',
  modelListUrl: 'https://models.example.org/all' // Absolute URL for listing
}

// Then, in RosettaAI constructor:
// const rosetta = new RosettaAI({
//   customProviders: [novitaConfig, gpustackConfig, lmstudioConfig, customPathConfig, customUrlConfig],
//   // ... other config ...
// });
```

## 7. API Reference Summary

Key exports from the `@missionsquad/rosetta-ai` package:

- **Client:** `RosettaAI`
- **Enums:** `Provider`
- **Configuration:** `RosettaAIConfig`, `ProviderOptions`, `CustomProviderConfig`, `ModelListingSourceConfig`
- **Core Parameters:** `GenerateParams`, `EmbedParams`, `SpeechParams`, `TranscribeParams`, `TranslateParams`
- **Core Results:** `GenerateResult`, `EmbedResult`, `TranscriptionResult`, `RosettaModel`, `RosettaModelList`
- **Streaming:** `StreamChunk`, `AudioStreamChunk`
- **Common Types:** `ProviderKey`, `RosettaMessage`, `RosettaContentPart`, `RosettaImageData`, `RosettaAudioData`, `RosettaTool`, `RosettaToolCallRequest`, `TokenUsage`, `Citation`
- **Errors:** `RosettaAIError`, `ConfigurationError`, `ProviderAPIError`, `UnsupportedFeatureError`, `MappingError`, `InvalidToolDefinitionError`, `ToolArgumentValidationError`

Detailed documentation is available via JSDoc comments in the source code and can be viewed using IDE IntelliSense or generated using TypeDoc.

## 8. Conclusion

The RosettaAI SDK provides a powerful yet simple way to integrate multiple LLM providers into your TypeScript applications. By leveraging its unified interface, strong typing, and clear configuration patterns, you can build robust AI-powered features faster and maintain them more easily. Remember to consult the specific provider's documentation for the most up-to-date information on supported models and features.
