import express, { Request, Response } from 'express'
import http from 'http'
import { z } from 'zod'
import { StreamChunk, Provider } from '../../src/types'

// Define expected request body schemas (optional but good practice)
const GenerateRequestSchema = z.object({
  model: z.string(),
  messages: z.array(z.any()), // Keep flexible for testing
  max_tokens: z.number().optional(),
  temperature: z.number().optional(),
  tools: z.array(z.any()).optional(), // Check if tools are received
  stream: z.boolean().optional()
})

const EmbedRequestSchema = z.object({
  model: z.string(),
  input: z.union([z.string(), z.array(z.string())])
})

// Store last received requests for inspection in tests
let lastGenerateRequest: any = null
let lastStreamRequest: any = null
let lastEmbedRequest: any = null

// --- Mock Stream Data ---
const streamChunks: StreamChunk[] = [
  { type: 'message_start', data: { provider: 'mock-custom-provider', model: 'mock-custom-stream-model' } },
  { type: 'content_delta', data: { delta: 'Mock ' } },
  { type: 'content_delta', data: { delta: 'stream ' } },
  { type: 'content_delta', data: { delta: 'response.' } },
  { type: 'message_stop', data: { finishReason: 'stop' } },
  { type: 'final_usage', data: { usage: { promptTokens: 6, completionTokens: 3, totalTokens: 9 } } }
]

const toolStreamChunks: StreamChunk[] = [
  { type: 'message_start', data: { provider: 'mock-custom-provider', model: 'mock-custom-stream-model-tool' } },
  {
    type: 'tool_call_start',
    data: { index: 0, toolCall: { id: 'stream_tool_xyz', type: 'function', function: { name: 'get_current_weather' } } }
  },
  {
    type: 'tool_call_delta',
    data: { index: 0, id: 'stream_tool_xyz', functionArgumentChunk: '{"location":' }
  },
  {
    type: 'tool_call_delta',
    data: { index: 0, id: 'stream_tool_xyz', functionArgumentChunk: ' "Stream City"' }
  },
  {
    type: 'tool_call_delta',
    data: { index: 0, id: 'stream_tool_xyz', functionArgumentChunk: ', "unit": "fahrenheit"}' }
  },
  { type: 'tool_call_done', data: { index: 0, id: 'stream_tool_xyz' } },
  { type: 'message_stop', data: { finishReason: 'tool_calls' } },
  { type: 'final_usage', data: { usage: { promptTokens: 12, completionTokens: 6, totalTokens: 18 } } }
]
// --- End Mock Stream Data ---

export function startMockApiServer(port: number): Promise<http.Server> {
  const app = express()
  app.use(express.json())

  // Endpoint for non-streaming generation
  app.post('/generate', (req: Request, res: Response) => {
    console.log('[Mock API /generate] Received request:', JSON.stringify(req.body))
    lastGenerateRequest = req.body // Store for inspection

    // Basic validation
    const parseResult = GenerateRequestSchema.safeParse(req.body)
    if (!parseResult.success) {
      console.error('[Mock API /generate] Invalid request body:', parseResult.error)
      return res.status(400).json({ error: 'Invalid request body', issues: parseResult.error.issues })
    }

    const { messages, tools, model: requestModel } = parseResult.data
    const lastUserMessage = messages.findLast((m: any) => m.role === 'user')?.content
    // Check if the history contains a tool result message
    const lastToolMessage = messages.findLast((m: any) => m.role === 'tool')?.content

    // Simulate different responses based on input or tools
    // --- Prioritize checking for tool result ---
    if (lastToolMessage) {
      // Simulate response after receiving tool result
      console.log('[Mock API /generate] Simulating response after tool result...')
      return res.json({
        id: 'mock-gen-456-final',
        model: requestModel || 'mock-custom-model',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Okay, the weather in Test City is 20 degrees Celsius.' // Final answer
            },
            finish_reason: 'stop'
          }
        ],
        usage: { prompt_tokens: 25, completion_tokens: 15, total_tokens: 40 }
      })
    } else if (tools && tools.length > 0 && lastUserMessage?.includes('weather')) {
      // Simulate a tool call response (only if no tool result was received)
      console.log('[Mock API /generate] Simulating tool call response...')
      return res.json({
        id: 'mock-gen-123-tool',
        model: requestModel || 'mock-custom-model', // Use requested model
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null, // No text content when calling tool
              tool_calls: [
                {
                  id: 'tool_call_abc',
                  type: 'function',
                  function: {
                    name: 'get_current_weather',
                    // Simulate arguments as a JSON string (matching default toolCallInputFormat)
                    arguments: JSON.stringify({ location: 'Test City', unit: 'celsius' })
                  }
                }
              ]
            },
            finish_reason: 'tool_calls'
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      })
    } else {
      // Simulate a simple text response
      console.log('[Mock API /generate] Simulating simple text response...')
      return res.json({
        id: 'mock-gen-789-text',
        model: requestModel || 'mock-custom-model',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: `Mock response to: ${lastUserMessage?.substring(0, 50) ?? 'empty prompt'}`
            },
            finish_reason: 'stop'
          }
        ],
        usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 }
        // provider: 'mock-custom-provider'
      })
    }
  })

  // Helper function to send SSE chunks sequentially
  const sendChunksSequentially = (res: Response, chunksToSend: StreamChunk[]) => {
    console.log(`[Mock API /stream] Starting to send ${chunksToSend.length} chunks sequentially...`)
    for (let i = 0; i < chunksToSend.length; i++) {
      if (res.writableEnded) {
        // This check might still be useful for logging, but shouldn't control flow
        console.log('[Mock API /stream] Connection closed before sending all chunks (during loop).')
        return // Stop sending if connection closed
      }
      const chunk = chunksToSend[i]
      const dataString = JSON.stringify(chunk)
      console.log(`[Mock API /stream] Sending event: chunk, data: ${dataString.substring(0, 100)}...`) // DEBUG
      res.write(`event: chunk\ndata: ${dataString}\n\n`)
    }
    console.log('[Mock API /stream] Finished sending all chunks.')
  }

  // Endpoint for streaming generation (SSE)
  app.post('/stream', (req: Request, res: Response) => {
    console.log('[Mock API /stream] Received request:', JSON.stringify(req.body))
    lastStreamRequest = req.body // Store for inspection

    // Basic validation
    const parseResult = GenerateRequestSchema.safeParse(req.body)
    if (!parseResult.success) {
      console.error('[Mock API /stream] Invalid request body:', parseResult.error)
      // Cannot send 400 easily with SSE once headers are sent, log and close.
      res.end()
      return
    }

    const { tools } = parseResult.data
    const lastUserMessage = req.body.messages?.findLast((m: any) => m.role === 'user')?.content

    // --- SSE Setup ---
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders() // Send headers immediately

    let streamFinishedSending = false // Flag to track normal completion

    // Determine which chunks to send
    const chunksToSend =
      tools && tools.length > 0 && lastUserMessage?.includes('weather') ? toolStreamChunks : streamChunks

    // Send all chunks synchronously (or as fast as possible)
    try {
      sendChunksSequentially(res, chunksToSend)
      // Signal end ONLY after sending all chunks
      if (!res.writableEnded) {
        console.log('[Mock API /stream] Sending end event.')
        res.write(`event: end\ndata: Stream finished\n\n`)
        streamFinishedSending = true // Mark as finished normally
      }
    } catch (error) {
      console.error('[Mock API /stream] Error during chunk sending:', error)
      // Attempt to send an error event if possible
      if (!res.writableEnded) {
        try {
          res.write(`event: error\ndata: ${JSON.stringify({ error: { message: 'Failed to send stream chunks' } })}\n\n`)
        } catch (writeError) {
          console.error('[Mock API /stream] Failed to write error event:', writeError)
        }
      }
    } finally {
      // Ensure the response is ended
      if (!res.writableEnded) {
        console.log('[Mock API /stream] Ending response stream.')
        res.end()
      }
    }

    // Handle client disconnect
    req.on('close', () => {
      // Only log unexpected closures
      if (!streamFinishedSending) {
        console.log('[Mock API /stream] Client disconnected unexpectedly.')
      } else {
        console.log('[Mock API /stream] Client disconnected after stream finished normally.')
      }
      // Ensure response ends if not already ended (e.g., due to error during sending)
      if (!res.writableEnded) {
        res.end()
      }
    })
  })

  // Endpoint for embedding
  app.post('/embed', (req: Request, res: Response) => {
    console.log('[Mock API /embed] Received request:', JSON.stringify(req.body))
    lastEmbedRequest = req.body // Store for inspection

    const parseResult = EmbedRequestSchema.safeParse(req.body)
    if (!parseResult.success) {
      console.error('[Mock API /embed] Invalid request body:', parseResult.error)
      return res.status(400).json({ error: 'Invalid request body', issues: parseResult.error.issues })
    }

    const input = parseResult.data.input
    const embeddings = Array.isArray(input)
      ? input.map((_, i) => [0.1 * (i + 1), 0.2 * (i + 1)]) // Simple mock vectors
      : [[0.1, 0.2]]

    console.log('[Mock API /embed] Sending mock embedding response...')
    res.json({
      object: 'list',
      data: embeddings.map((vec, i) => ({
        object: 'embedding',
        index: i,
        embedding: vec
      })),
      model: req.body.model || 'mock-custom-embed-model',
      usage: { total_tokens: Array.isArray(input) ? input.length * 5 : 5 } // Mock usage
    })
  })

  // Endpoint to get last requests (for test verification)
  app.get('/last-request/generate', (req: Request, res: Response) => {
    res.json(lastGenerateRequest)
  })
  app.get('/last-request/stream', (req: Request, res: Response) => {
    res.json(lastStreamRequest)
  })
  app.get('/last-request/embed', (req: Request, res: Response) => {
    res.json(lastEmbedRequest)
  })

  return new Promise(resolve => {
    const server = app.listen(port, () => {
      console.log(`[Mock API Server] Listening on port ${port}`)
      resolve(server)
    })
  })
}

export function stopMockApiServer(server: http.Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close(err => {
      if (err) {
        console.error('[Mock API Server] Error stopping server:', err)
        return reject(err)
      }
      console.log('[Mock API Server] Stopped.')
      resolve()
    })
  })
}

// --- Getters for last requests ---
export function getLastGenerateRequest(): any {
  return lastGenerateRequest
}
export function getLastStreamRequest(): any {
  return lastStreamRequest
}
export function getLastEmbedRequest(): any {
  return lastEmbedRequest
}
