/**
 * Enumeration of supported AI providers.
 */
export enum Provider {
  Anthropic = 'anthropic',
  Google = 'google',
  Groq = 'groq',
  OpenAI = 'openai' // Represents both OpenAI standard and Azure OpenAI
}

export type ImageMimeType = 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp'
/**
 * Represents raw image data, typically Base64 encoded.
 * @property mimeType - The MIME type of the image (e.g., 'image/jpeg', 'image/png').
 * @property base64Data - The Base64 encoded string of the image data.
 */
export interface RosettaImageData {
  mimeType: ImageMimeType
  base64Data: string
}

/**
 * Represents raw audio data for input.
 * @property data - The audio data as a Buffer or NodeJS.ReadableStream.
 * @property filename - A filename for the audio (required by some APIs).
 * @property mimeType - The MIME type of the audio (e.g., 'audio/mpeg', 'audio/wav').
 */
export interface RosettaAudioData {
  data: Buffer | NodeJS.ReadableStream
  filename: string
  mimeType: 'audio/mpeg' | 'audio/wav' | 'audio/ogg' | 'audio/webm'
}

/**
 * A discriminated union representing different parts of a message's content.
 * Supports text and image inputs.
 */
export type RosettaContentPart = { type: 'text'; text: string } | { type: 'image'; image: RosettaImageData }
// | { type: 'audio'; audio: RosettaAudioData }; // Future: If models support inline audio content parts

/**
 * Represents a single message in a conversation.
 * @property role - The role of the message sender ('system', 'user', 'assistant', 'tool').
 * @property content - The content of the message, can be simple text or an array of content parts (for multimodal).
 * @property toolCalls - For 'assistant' role: An array of tool calls requested by the model.
 * @property toolCallId - For 'tool' role: The ID of the tool call this message is a response to.
 */
export interface RosettaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | RosettaContentPart[] | null // FIX: Allow content to be potentially null
  toolCalls?: RosettaToolCallRequest[]
  toolCallId?: string
}

/**
 * Defines a tool (currently only functions) that the model can be instructed to use.
 * Based on the OpenAI function tool definition.
 * @property type - The type of the tool (currently 'function').
 * @property function - Details of the function.
 * @property function.name - The name of the function to be called.
 * @property function.description - A description of what the function does, used by the model.
 * @property function.parameters - A JSON Schema object describing the expected arguments for the function.
 */
export interface RosettaTool {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown> // JSON Schema definition
  }
}

/**
 * Represents a tool call requested by the model in its response.
 * @property id - A unique identifier for this specific tool call instance.
 * @property type - The type of tool called (currently 'function').
 * @property function - Details of the function call.
 * @property function.name - The name of the function the model wants to call.
 * @property function.arguments - A JSON string containing the arguments the model generated for the function call.
 */
export interface RosettaToolCallRequest {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string // Raw JSON string
  }
}

/**
 * Represents the result of executing a tool, to be sent back to the model.
 * This data will typically be formatted into a `RosettaMessage` with `role: 'tool'`.
 * @property toolCallId - The ID of the tool call this result corresponds to.
 * @property content - The output/result from the tool execution (usually stringified).
 * @property isError - Optional flag indicating if the tool execution resulted in an error.
 */
export interface RosettaToolResult {
  toolCallId: string
  content: string
  isError?: boolean
}

/**
 * Represents citation metadata, often related to grounding.
 * @property text - The text content of the citation (may not always be provided).
 * @property sourceId - An identifier linking to the source material (e.g., URI, document ID).
 * @property startIndex - The starting index in the main response content where the citation applies (optional).
 * @property endIndex - The ending index in the main response content where the citation applies (optional).
 */
export interface Citation {
  text?: string
  sourceId: string
  startIndex?: number
  endIndex?: number
}
