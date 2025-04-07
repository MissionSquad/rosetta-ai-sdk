import express, { Request, Response, NextFunction } from 'express'
import cors from 'cors'
import multer from 'multer'
import dotenv from 'dotenv'
import path from 'path'
import {
  RosettaAI,
  Provider,
  RosettaMessage,
  RosettaTool,
  GenerateParams,
  EmbedParams,
  SpeechParams,
  TranscribeParams,
  TranslateParams,
  RosettaAudioData,
  RosettaAIError,
  ConfigurationError,
  ProviderAPIError,
  UnsupportedFeatureError,
  MappingError,
  RosettaImageData, // Import image type
  ImageMimeType, // Import image mime type
  RosettaContentPart // Import content part type
} from '../src' // Adjust path based on your compiled output structure if needed

// Load environment variables from examples/.env
dotenv.config({ path: path.resolve(__dirname, '.env') })

const app = express()
const port = process.env.PORT || 3001

// --- Middleware ---
// CORS
const corsOptions = {
  origin: process.env.CORS_ORIGIN || '*', // Allow specific origin or all
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true
}
app.use(cors(corsOptions))

// JSON Body Parser
app.use(express.json({ limit: '10mb' })) // Increase limit for potential base64 images

// Multer for file uploads (store in memory for simplicity)
const storage = multer.memoryStorage()
const upload = multer({
  storage: storage,
  limits: { fileSize: 25 * 1024 * 1024 } // Limit file size (e.g., 25MB)
})

// --- Initialize RosettaAI ---
let rosetta: RosettaAI
try {
  // Pass empty config, relies on process.env loaded by dotenv
  rosetta = new RosettaAI()
  console.log(`RosettaAI initialized. Configured providers: ${rosetta.getConfiguredProviders().join(', ')}`)
} catch (error) {
  console.error('FATAL: Failed to initialize RosettaAI SDK:', error)
  process.exit(1) // Exit if SDK fails to initialize
}

// --- Tool Definition & Implementation (for Tool Use Example) ---
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

// --- API Routes ---

// GET /api/config
app.get('/api/config', (req: Request, res: Response) => {
  try {
    res.json(rosetta.getConfiguredProviders())
  } catch (error) {
    console.error('Error fetching configured providers:', error)
    res.status(500).json({ error: 'Failed to get provider configuration' })
  }
})

// POST /api/generate
app.post('/api/generate', async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Basic validation (more robust validation recommended for production)
    const params: GenerateParams = req.body
    if (!params.provider || !params.messages) {
      return res.status(400).json({ error: 'Missing required parameters: provider, messages' })
    }
    const result = await rosetta.generate(params)
    res.json(result)
  } catch (error) {
    next(error) // Pass error to the error handling middleware
  }
})

// POST /api/tool-use
app.post('/api/tool-use', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { provider, model, initialPrompt } = req.body
    if (!provider || !initialPrompt) {
      return res.status(400).json({ error: 'Missing required parameters: provider, initialPrompt' })
    }

    const history: RosettaMessage[] = [{ role: 'user', content: initialPrompt }]
    const maxIterations = 5 // Prevent infinite loops

    for (let i = 0; i < maxIterations; i++) {
      console.log(`[Tool Use API - Iteration ${i + 1}] Sending request...`)
      const response = await rosetta.generate({
        provider,
        model: model || undefined, // Use provided model or let SDK handle default
        messages: history,
        tools: [getWeatherTool], // Use the hardcoded tool
        toolChoice: 'auto'
      })

      // Add assistant's response (content and potential tool calls) to the history
      history.push({
        role: 'assistant',
        content: response.content,
        toolCalls: response.toolCalls
      })

      // Check if the model requested tool calls
      if (response.toolCalls && response.toolCalls.length > 0) {
        console.log(`[Tool Use API] Tool calls requested: ${response.toolCalls.length}`)
        const toolResultMessages: RosettaMessage[] = []

        // Execute each requested tool call
        await Promise.all(
          response.toolCalls.map(async call => {
            if (call.type === 'function' && call.function.name === 'getCurrentWeather') {
              let toolResultContent: string
              try {
                const args = JSON.parse(call.function.arguments)
                toolResultContent = await getCurrentWeather(args.location, args.unit)
                console.log(`[Tool Use API] -> Tool Result OK for call ${call.id}`)
              } catch (e) {
                const errorMessage = e instanceof Error ? e.message : String(e)
                console.error(`[Tool Use API] -> Tool Error for call ${call.id}:`, errorMessage)
                toolResultContent = JSON.stringify({ error: errorMessage })
              }
              toolResultMessages.push({
                role: 'tool',
                toolCallId: call.id,
                content: toolResultContent
              })
            } else {
              const toolName = call.function?.name ?? 'unknown tool'
              console.warn(`[Tool Use API] Model called unknown/unsupported tool: ${toolName}`)
              toolResultMessages.push({
                role: 'tool',
                toolCallId: call.id,
                content: JSON.stringify({ error: `Tool '${toolName}' is not implemented.` })
              })
            }
          })
        )
        // Add all tool results to the message history for the next turn
        history.push(...toolResultMessages)
      } else {
        // If no tool calls, the conversation is likely finished
        console.log('[Tool Use API] Conversation finished.')
        break // Exit the loop
      }

      // Safety break if max iterations are reached
      if (i === maxIterations - 1) {
        console.warn('[Tool Use API] Maximum conversation iterations reached.')
        // Add a final message indicating the limit was reached? Optional.
        history.push({ role: 'system', content: '[System Note: Max iterations reached]' })
      }
    } // End of loop

    // Return the full conversation history
    res.json({ history })
  } catch (error) {
    next(error) // Pass error to the error handling middleware
  }
})

// POST /api/generate-with-image
// Use multer middleware for the 'image' field
app.post(
  '/api/generate-with-image',
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Extract text fields from req.body
      const { provider, model, textPrompt, maxTokens } = req.body

      // Validate required fields
      if (!provider || !textPrompt) {
        return res.status(400).json({ error: 'Missing required parameters: provider, textPrompt' })
      }
      if (!req.file) {
        return res.status(400).json({ error: "Missing 'image' file in request." })
      }

      // Prepare image data
      const imageBuffer = req.file.buffer
      const base64Data = imageBuffer.toString('base64')
      const mimeType = req.file.mimetype as ImageMimeType // Basic assertion

      // Validate MIME type if necessary (optional)
      const allowedMimeTypes: ImageMimeType[] = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
      if (!allowedMimeTypes.includes(mimeType)) {
        return res.status(400).json({ error: `Unsupported image MIME type: ${mimeType}` })
      }

      const imageData: RosettaImageData = { mimeType, base64Data }

      // Construct the multimodal message
      const messages: RosettaMessage[] = [
        {
          role: 'user',
          content: [
            { type: 'text', text: textPrompt },
            { type: 'image', image: imageData }
          ] as RosettaContentPart[] // Explicitly type as array
        }
      ]

      // Prepare GenerateParams
      const params: GenerateParams = {
        provider: provider as Provider, // Assert provider type
        model: model || undefined, // Use provided model or let SDK use default
        messages: messages,
        maxTokens: maxTokens ? parseInt(maxTokens, 10) : undefined // Parse maxTokens if provided
      }

      // Call RosettaAI generate
      const result = await rosetta.generate(params)
      res.json(result)
    } catch (error) {
      next(error) // Pass error to the error handling middleware
    }
  }
)

// POST /api/stream
app.post('/api/stream', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const params: GenerateParams = req.body
    if (!params.provider || !params.messages) {
      return res.status(400).json({ error: 'Missing required parameters: provider, messages' })
    }

    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders() // Send headers immediately

    const stream = rosetta.stream(params)

    // Handle stream events
    for await (const chunk of stream) {
      if (chunk.type === 'error') {
        // Send error event and potentially close connection
        res.write(`event: error\ndata: ${JSON.stringify(chunk.data)}\n\n`)
        // Optionally close connection on error, or let client decide
        // res.end(); // Uncomment to close connection on first stream error
        console.error('SSE Stream Error:', chunk.data.error)
        // Do not throw here, error is sent via SSE
      } else {
        // Send message event
        res.write(`event: message\ndata: ${JSON.stringify(chunk)}\n\n`)
      }
    }

    // Signal end of stream (optional, client can detect closure)
    res.write(`event: end\ndata: Stream finished\n\n`)
    res.end()
  } catch (error) {
    // Catch errors during stream *setup* (before the loop)
    console.error('Error setting up SSE stream:', error)
    // Ensure headers aren't sent again if already flushed
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to start stream' })
    } else {
      // If headers sent, try to send an error event before closing
      res.write(`event: error\ndata: ${JSON.stringify({ error: { message: 'Stream setup failed' } })}\n\n`)
      res.end()
    }
    // next(error); // Don't call next if response is handled
  }

  // Handle client disconnect
  req.on('close', () => {
    console.log('SSE client disconnected.')
    // Clean up resources if necessary (e.g., abort ongoing provider requests if possible)
    res.end() // Ensure response stream is closed
  })
})

// POST /api/embed
app.post('/api/embed', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const params: EmbedParams = req.body
    if (!params.provider || !params.input) {
      return res.status(400).json({ error: 'Missing required parameters: provider, input' })
    }
    const result = await rosetta.embed(params)
    res.json(result)
  } catch (error) {
    next(error)
  }
})

// POST /api/tts
app.post('/api/tts', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const params: SpeechParams = req.body
    if (params.provider !== Provider.OpenAI) {
      return res.status(400).json({ error: "TTS currently only supports the 'openai' provider." })
    }
    if (!params.input || !params.voice) {
      return res.status(400).json({ error: 'Missing required parameters: input, voice' })
    }

    const audioBuffer = await rosetta.generateSpeech(params)

    // Determine content type
    let contentType = 'audio/mpeg' // Default MP3
    switch (params.responseFormat) {
      case 'opus':
        contentType = 'audio/opus'
        break
      case 'aac':
        contentType = 'audio/aac'
        break
      case 'flac':
        contentType = 'audio/flac'
        break
      case 'wav':
        contentType = 'audio/wav'
        break
      case 'pcm':
        contentType = 'audio/pcm'
        break
    }

    res.setHeader('Content-Type', contentType)
    res.send(audioBuffer)
  } catch (error) {
    next(error)
  }
})

// POST /api/transcribe
app.post('/api/transcribe', upload.single('audio'), async (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing 'audio' file in request." })
    }
    const params: Omit<TranscribeParams, 'audio'> = req.body // Get other params from body
    if (!params.provider) {
      return res.status(400).json({ error: 'Missing required parameter: provider' })
    }

    const audioData: RosettaAudioData = {
      data: req.file.buffer,
      filename: req.file.originalname,
      mimeType: req.file.mimetype as RosettaAudioData['mimeType'] // Basic type assertion
    }

    const fullParams: TranscribeParams = { ...params, audio: audioData }
    const result = await rosetta.transcribe(fullParams)
    res.json(result)
  } catch (error) {
    next(error)
  }
})

// POST /api/translate
app.post('/api/translate', upload.single('audio'), async (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing 'audio' file in request." })
    }
    const params: Omit<TranslateParams, 'audio'> = req.body
    if (!params.provider) {
      return res.status(400).json({ error: 'Missing required parameter: provider' })
    }

    const audioData: RosettaAudioData = {
      data: req.file.buffer,
      filename: req.file.originalname,
      mimeType: req.file.mimetype as RosettaAudioData['mimeType'] // Basic type assertion
    }

    const fullParams: TranslateParams = { ...params, audio: audioData }
    const result = await rosetta.translate(fullParams)
    res.json(result)
  } catch (error) {
    next(error)
  }
})

// --- Error Handling Middleware ---
// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(`[${new Date().toISOString()}] Error on ${req.method} ${req.path}:`, err)

  if (res.headersSent) {
    // If headers already sent (like in SSE), we can't send a JSON error response
    // Log it and potentially close the connection if it's still open
    console.error('Headers already sent, cannot send JSON error response.')
    if (!res.writableEnded) {
      res.end()
    }
    return
  }

  let statusCode = 500
  let errorMessage = 'Internal Server Error'
  let errorDetails: any = null

  if (err instanceof ConfigurationError) {
    statusCode = 400 // Bad request due to config issue client might influence
    errorMessage = err.message
  } else if (err instanceof UnsupportedFeatureError) {
    statusCode = 400 // Bad request - tried to use unsupported feature
    errorMessage = err.message
  } else if (err instanceof MappingError) {
    statusCode = 500 // Internal server error during mapping
    errorMessage = 'Internal SDK mapping error.'
    errorDetails = { message: err.message, provider: err.provider, context: err.context }
  } else if (err instanceof ProviderAPIError) {
    errorMessage = `Provider API Error (${err.provider}): ${err.message}`
    errorDetails = {
      provider: err.provider,
      statusCode: err.statusCode,
      errorCode: err.errorCode,
      errorType: err.errorType
      // underlyingError: err.underlyingError // Avoid sending potentially sensitive underlying errors
    }
    // Map provider status codes to HTTP status codes
    if (err.statusCode) {
      if (err.statusCode === 400) statusCode = 400
      // Bad Request from provider
      else if (err.statusCode === 401) statusCode = 401
      // Unauthorized
      else if (err.statusCode === 403) statusCode = 403
      // Forbidden
      else if (err.statusCode === 404) statusCode = 404
      // Not Found (e.g., invalid model)
      else if (err.statusCode === 429) statusCode = 429
      // Rate Limit
      else if (err.statusCode >= 500) statusCode = 503
      // Service Unavailable (provider issue)
      else statusCode = 502 // Bad Gateway (unexpected provider status)
    } else {
      statusCode = 503 // Service Unavailable if status unknown
    }
  } else if (err instanceof RosettaAIError) {
    // Catch-all for other specific SDK errors
    statusCode = 500
    errorMessage = err.message
  } else if (err.name === 'MulterError') {
    statusCode = 400
    errorMessage = `File upload error: ${err.message}`
  }
  // Add more specific error checks if needed

  res.status(statusCode).json({
    error: errorMessage,
    ...(errorDetails && { details: errorDetails }) // Include details if available
  })
})

// --- Static File Serving ---
// FIX: Serve static files from the correct 'examples' directory relative to project root
const examplesPath = path.join(process.cwd(), '') // Correct path to examples dir
console.log(`Serving static files from: ${examplesPath}`)
app.use(express.static(examplesPath))

// --- Root Route ---
// FIX: Serve index.html directly from the root path '/'
app.get('/', (req, res) => {
  res.sendFile(path.join(examplesPath, 'index.html'))
})

// --- Start Server ---
app.listen(port, () => {
  console.log(`RosettaAI Example Server listening on port ${port}`)
  console.log(`Access the frontend example at: http://localhost:${port}/ (or your configured host)`)
})
