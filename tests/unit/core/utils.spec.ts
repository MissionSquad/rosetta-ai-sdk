import { Readable } from 'stream'
import { safeGet, prepareAudioUpload } from '../../../src/core/utils' // Adjust path
import { RosettaAudioData } from '../../../src/types'
import * as OpenAIModule from 'openai' // Import the actual module to mock

// Mock the 'openai' module, specifically the 'toFile' function
jest.mock('openai', () => ({
  ...jest.requireActual('openai'), // Keep original parts of the module
  toFile: jest.fn() // Mock the toFile function
}))

// Helper to create a simple Readable stream
function createReadableStream(content: string): Readable {
  const stream = new Readable()
  stream.push(content)
  stream.push(null) // Signal end of stream
  return stream
}

describe('Core Utilities', () => {
  describe('safeGet', () => {
    const testObj = {
      a: 1,
      b: {
        c: 'hello',
        d: [10, 20, { e: true }]
      },
      f: null,
      g: undefined,
      h: [{ i: 100 }, { j: 200 }]
    }

    it('should get top-level properties', () => {
      expect(safeGet<number>(testObj, 'a')).toBe(1)
      expect(safeGet<object>(testObj, 'b')).toEqual({ c: 'hello', d: [10, 20, { e: true }] })
      expect(safeGet<null>(testObj, 'f')).toBeNull()
      expect(safeGet<undefined>(testObj, 'g')).toBeUndefined()
    })

    it('should get nested properties', () => {
      expect(safeGet<string>(testObj, 'b', 'c')).toBe('hello')
      expect(safeGet<number>(testObj, 'b', 'd', 0)).toBe(10)
      expect(safeGet<boolean>(testObj, 'b', 'd', 2, 'e')).toBe(true)
      expect(safeGet<number>(testObj, 'h', 0, 'i')).toBe(100)
    })

    it('should return undefined for invalid paths (missing key)', () => {
      expect(safeGet(testObj, 'x')).toBeUndefined()
      expect(safeGet(testObj, 'b', 'x')).toBeUndefined()
      expect(safeGet(testObj, 'b', 'd', 2, 'x')).toBeUndefined()
      expect(safeGet(testObj, 'h', 0, 'x')).toBeUndefined()
    })

    it('should return undefined for invalid paths (incorrect index)', () => {
      expect(safeGet(testObj, 'b', 'd', 5)).toBeUndefined()
      expect(safeGet(testObj, 'h', 2)).toBeUndefined()
      expect(safeGet(testObj, 'h', -1)).toBeUndefined() // Negative index
    })

    it('should return undefined if intermediate path is null or undefined', () => {
      expect(safeGet(testObj, 'f', 'x')).toBeUndefined() // testObj.f is null
      expect(safeGet(testObj, 'g', 'x')).toBeUndefined() // testObj.g is undefined
    })

    it('should return value for valid properties on primitives', () => {
      // This test should now pass with the fixed safeGet implementation
      expect(safeGet('string', 'length')).toBe(6) // Accessing property on primitive works
    })

    it('should return undefined for non-existent properties on primitives or non-objects', () => {
      expect(safeGet('string', 'a')).toBeUndefined() // Accessing non-existent property
      expect(safeGet(123, 'a')).toBeUndefined()
      expect(safeGet(null, 'a')).toBeUndefined()
      expect(safeGet(undefined, 'a')).toBeUndefined()
    })

    it('should return the object itself if path is empty', () => {
      expect(safeGet(testObj)).toBe(testObj)
      expect(safeGet(null)).toBeNull()
      expect(safeGet(undefined)).toBeUndefined()
    })
  })

  describe('prepareAudioUpload', () => {
    const mockToFile = OpenAIModule.toFile as jest.Mock

    beforeEach(() => {
      mockToFile.mockClear()
    })

    it('should call openai.toFile for Buffer input', async () => {
      const buffer = Buffer.from('fake audio data')
      const audioData: RosettaAudioData = {
        data: buffer,
        filename: 'test.mp3',
        mimeType: 'audio/mpeg'
      }
      const mockFileLike = { name: 'test.mp3', type: 'audio/mpeg', [Symbol.toStringTag]: 'File' } // Simulate FileLike structure
      mockToFile.mockResolvedValue(mockFileLike)

      const result = await prepareAudioUpload(audioData)

      expect(mockToFile).toHaveBeenCalledTimes(1)
      expect(mockToFile).toHaveBeenCalledWith(buffer, 'test.mp3', { type: 'audio/mpeg' })
      expect(result).toBe(mockFileLike) // Should return the result from toFile
    })

    it('should return the ReadableStream directly for stream input', async () => {
      const stream = createReadableStream('fake stream data')
      const audioData: RosettaAudioData = {
        data: stream,
        filename: 'test.wav',
        mimeType: 'audio/wav'
      }

      const result = await prepareAudioUpload(audioData)

      expect(mockToFile).not.toHaveBeenCalled()
      expect(result).toBe(stream) // Should return the original stream instance
      expect(result instanceof Readable).toBe(true)
    })

    it('should throw TypeError for unsupported data types', async () => {
      const invalidAudioData: RosettaAudioData = {
        data: 'this is a string' as any, // Invalid type
        filename: 'test.ogg',
        mimeType: 'audio/ogg'
      }

      await expect(prepareAudioUpload(invalidAudioData)).rejects.toThrow(TypeError)
      await expect(prepareAudioUpload(invalidAudioData)).rejects.toThrow(
        'Unsupported audio data type. Expected Buffer or NodeJS.ReadableStream.'
      )
      expect(mockToFile).not.toHaveBeenCalled()
    })
  })
})
