import { Readable } from 'stream'
import { toFile } from 'openai'
import { FileLike } from 'openai/uploads'
import { RosettaAudioData } from '../types'

/**
 * Prepares audio data (Buffer or ReadableStream) into a format suitable for SDK uploads.
 * Uses the openai.toFile helper for Buffers to ensure compatibility.
 * Returns a type compatible with both OpenAI and Groq upload parameters.
 *
 * @param audio The audio data object containing data, filename, and mimeType.
 * @returns A promise resolving to an object compatible with SDK file parameters (FileLike for Buffers, ReadableStream for streams).
 * @throws {TypeError} if the audio data type is not Buffer or NodeJS.ReadableStream.
 */
export async function prepareAudioUpload(audio: RosettaAudioData): Promise<FileLike | NodeJS.ReadableStream> {
  // Return the union of possible compatible types
  const { data, filename, mimeType } = audio

  if (Buffer.isBuffer(data)) {
    // Use the official OpenAI toFile helper to convert the Buffer.
    // This handles creating the correct internal structure (conforming to FileLike).
    // Pass filename and type options.
    const uploadableFile = await toFile(data, filename, { type: mimeType })
    // The result of toFile is compatible with OpenAI methods expecting Uploadable.
    // It also conforms structurally to FileLike, which Groq's Uploadable likely accepts.
    return uploadableFile
  } else if (data instanceof Readable) {
    // Both OpenAI and Groq SDKs generally accept NodeJS.ReadableStream directly
    // as part of their Uploadable type definitions.
    return data
  } else {
    throw new TypeError('Unsupported audio data type. Expected Buffer or NodeJS.ReadableStream.')
  }
}

/**
 * Utility to safely get nested properties from an object or properties from primitives.
 * Returns `undefined` if any intermediate property in the path is null or undefined,
 * or if the key/index doesn't exist or is invalid for the current value type.
 *
 * @template T The expected type of the target property.
 * @param obj The object or primitive to traverse.
 * @param path Sequence of property names or array indices to access.
 * @returns The value at the specified path, or `undefined` if the path is invalid.
 */
export function safeGet<T>(obj: any, ...path: (string | number)[]): T | undefined {
  let current: any = obj
  for (const key of path) {
    // Check for null/undefined at any point in the path
    if (current === null || typeof current === 'undefined') {
      return undefined
    }

    // Use Object() wrapper for `in` check to handle primitives correctly
    const currentWrapper = Object(current)

    try {
      // Check if the key exists in the object wrapper or if it's a valid array index
      if (!(key in currentWrapper)) {
        if (Array.isArray(current) && typeof key === 'number') {
          if (key < 0 || key >= current.length) {
            return undefined // Index out of bounds
          }
          // If index is within bounds, 'in' might have been false for sparse arrays, but access is valid.
        } else {
          return undefined // Key not found in object or primitive wrapper
        }
      }
    } catch (e) {
      // Handle cases where 'in' might throw (e.g., on certain host objects or proxies)
      return undefined
    }

    // Access the property/index using the original `current` value
    current = current[key]
  }
  return current as T
}
