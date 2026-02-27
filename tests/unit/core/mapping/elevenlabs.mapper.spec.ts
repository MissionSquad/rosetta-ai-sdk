import { Readable } from 'stream'
import { ElevenLabsMapper } from '../../../../src/core/mapping/elevenlabs.mapper'
import {
  SpeechParams,
  TranscribeParams,
  RosettaAudioData,
  AudioStreamChunk,
  RosettaVoiceList
} from '../../../../src/types'
import { CustomProviderConfig } from '../../../../src/types/custom.types'
import { MappingError, ProviderAPIError } from '../../../../src/errors'

// Shared mock functions — referenced in both the mock factory and individual tests.
// Variable names prefixed with "mock" so Jest allows referencing them inside jest.mock().
const mockTtsConvert = jest.fn()
const mockTtsStream = jest.fn()
const mockSttConvert = jest.fn()
const mockVoicesGetAll = jest.fn()

jest.mock('@elevenlabs/elevenlabs-js', () => ({
  ElevenLabsClient: jest.fn().mockImplementation(() => ({
    textToSpeech: { convert: mockTtsConvert, stream: mockTtsStream },
    speechToText: { convert: mockSttConvert },
    voices: { getAll: mockVoicesGetAll }
  }))
}))

// Helper to collect stream chunks
async function collectAudioStreamChunks(stream: AsyncIterable<AudioStreamChunk>): Promise<AudioStreamChunk[]> {
  const chunks: AudioStreamChunk[] = []
  for await (const chunk of stream) {
    chunks.push(chunk)
  }
  return chunks
}

// Helper to create a Web ReadableStream from buffers (used for streaming TTS tests)
function createMockReadableStream(chunks: Buffer[]): ReadableStream<Uint8Array> {
  let index = 0
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(chunks[index++])
      } else {
        controller.close()
      }
    }
  })
}

// Helper to create a Node Readable stream from a single buffer (non-streaming TTS tests)
function createMockNodeReadable(data: Buffer): Readable {
  return new Readable({
    read() {
      this.push(data)
      this.push(null)
    }
  })
}

describe('ElevenLabsMapper', () => {
  let mapper: ElevenLabsMapper
  let mockConfig: CustomProviderConfig

  beforeEach(() => {
    mockConfig = {
      providerKey: 'elevenlabs',
      mapper: ElevenLabsMapper,
      supportedFeatures: ['tts', 'stt', 'list_voices'],
      apiKey: 'test-api-key',
      defaultTtsModel: 'eleven_flash_v2_5',
      defaultSttModel: 'scribe_v1'
    }

    mapper = new ElevenLabsMapper(mockConfig)

    jest.clearAllMocks()
  })

  describe('Text-to-Speech (TTS)', () => {
    describe('Non-streaming TTS', () => {
      it('should generate speech with basic parameters', async () => {
        const mockAudioData = Buffer.from('fake-audio-data')
        mockTtsConvert.mockResolvedValue(createMockNodeReadable(mockAudioData))

        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'Hello world',
          voice: '21m00Tcm4TlvDq8ikWAM',
          responseFormat: 'mp3'
        }

        const result = await mapper.executeGenerateSpeech({}, 'test-api-key', mockConfig, params)

        expect(result).toBeInstanceOf(Buffer)
        expect(mockTtsConvert).toHaveBeenCalledWith(
          '21m00Tcm4TlvDq8ikWAM',
          expect.objectContaining({
            text: 'Hello world',
            modelId: 'eleven_flash_v2_5',
            outputFormat: 'mp3_44100_128'
          })
        )
      })

      it('should apply text normalization by default', async () => {
        mockTtsConvert.mockResolvedValue(createMockNodeReadable(Buffer.from('fake-audio-data')))

        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'The price is $42.50',
          voice: '21m00Tcm4TlvDq8ikWAM'
        }

        await mapper.executeGenerateSpeech({}, 'test-api-key', mockConfig, params)

        expect(mockTtsConvert).toHaveBeenCalledWith(
          '21m00Tcm4TlvDq8ikWAM',
          expect.objectContaining({
            text: 'The price is 42.50 dollars'
          })
        )
      })

      it('should skip text normalization when disabled', async () => {
        mockTtsConvert.mockResolvedValue(createMockNodeReadable(Buffer.from('fake-audio-data')))

        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'The price is $42.50',
          voice: '21m00Tcm4TlvDq8ikWAM',
          ttsNormalize: false
        }

        await mapper.executeGenerateSpeech({}, 'test-api-key', mockConfig, params)

        expect(mockTtsConvert).toHaveBeenCalledWith(
          '21m00Tcm4TlvDq8ikWAM',
          expect.objectContaining({
            text: 'The price is $42.50'
          })
        )
      })

      it('should include voice settings from ttsOptions', async () => {
        mockTtsConvert.mockResolvedValue(createMockNodeReadable(Buffer.from('fake-audio-data')))

        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'Hello',
          voice: '21m00Tcm4TlvDq8ikWAM',
          ttsOptions: {
            stability: 0.5,
            similarityBoost: 0.75,
            style: 0.3,
            useSpeakerBoost: true
          }
        }

        await mapper.executeGenerateSpeech({}, 'test-api-key', mockConfig, params)

        expect(mockTtsConvert).toHaveBeenCalledWith(
          '21m00Tcm4TlvDq8ikWAM',
          expect.objectContaining({
            voiceSettings: {
              stability: 0.5,
              similarityBoost: 0.75,
              style: 0.3,
              useSpeakerBoost: true
            }
          })
        )
      })

      it('should throw MappingError if voice ID is missing', async () => {
        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'Hello',
          voice: ''
        }

        await expect(
          mapper.executeGenerateSpeech({}, 'test-api-key', mockConfig, params)
        ).rejects.toThrow(MappingError)
      })

      it('should throw MappingError if model is missing', async () => {
        const configWithoutModel = { ...mockConfig, defaultTtsModel: undefined }
        const mapperWithoutModel = new ElevenLabsMapper(configWithoutModel)

        const params: SpeechParams = {
          provider: 'elevenlabs',
          input: 'Hello',
          voice: '21m00Tcm4TlvDq8ikWAM'
        }

        await expect(
          mapperWithoutModel.executeGenerateSpeech({} as any, 'test-api-key', configWithoutModel, params)
        ).rejects.toThrow(MappingError)
      })
    })

    describe('Streaming TTS', () => {
      it('should stream audio chunks', async () => {
        const mockChunks = [Buffer.from('chunk1'), Buffer.from('chunk2'), Buffer.from('chunk3')]
        const mockStream = createMockReadableStream(mockChunks)
        mockTtsStream.mockResolvedValue(mockStream)

        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'Hello streaming',
          voice: '21m00Tcm4TlvDq8ikWAM'
        }

        const chunks = await collectAudioStreamChunks(
          mapper.executeStreamSpeech({}, 'test-api-key', mockConfig, params)
        )

        expect(chunks).toHaveLength(4) // 3 audio chunks + 1 stop
        expect(chunks[0].type).toBe('audio_chunk')
        expect(chunks[1].type).toBe('audio_chunk')
        expect(chunks[2].type).toBe('audio_chunk')
        expect(chunks[3].type).toBe('audio_stop')
      })

      it('should yield error chunk if voice ID is missing', async () => {
        const params: SpeechParams = {
          provider: 'elevenlabs',
          model: 'eleven_flash_v2_5',
          input: 'Hello',
          voice: ''
        }

        const chunks = await collectAudioStreamChunks(
          mapper.executeStreamSpeech({}, 'test-api-key', mockConfig, params)
        )

        expect(chunks).toHaveLength(1)
        expect(chunks[0].type).toBe('error')
        expect((chunks[0] as any).data.error).toBeInstanceOf(MappingError)
      })
    })
  })

  describe('Speech-to-Text (STT)', () => {
    describe('Basic transcription', () => {
      it('should transcribe audio with basic parameters', async () => {
        const mockResponse = {
          text: 'This is a test transcription',
          languageCode: 'en',
          words: [
            { text: 'This', start: 0.0, end: 0.2, type: 'word', speaker_id: 'speaker_0' },
            { text: 'is', start: 0.2, end: 0.3, type: 'word', speaker_id: 'speaker_0' },
            { text: 'a', start: 0.3, end: 0.4, type: 'word', speaker_id: 'speaker_0' },
            { text: 'test', start: 0.4, end: 0.6, type: 'word', speaker_id: 'speaker_0' }
          ]
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        const result = await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(result.text).toBe('This is a test transcription')
        expect(result.language).toBe('en')
        expect(result.words).toHaveLength(4)
        expect(result.model).toBe('scribe_v1')
        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            modelId: 'scribe_v1',
            file: audioBuffer
          })
        )
      })

      it('should support Readable stream as input', async () => {
        const mockResponse = {
          text: 'Transcription from stream',
          languageCode: 'en',
          words: []
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioStream = new Readable({
          read() {
            this.push(Buffer.from('fake-audio'))
            this.push(null)
          }
        })

        const audioData: RosettaAudioData = {
          data: audioStream,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        const result = await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(result.text).toBe('Transcription from stream')
        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            file: audioStream
          })
        )
      })

      it('should include language code when provided', async () => {
        const mockResponse = {
          text: 'Bonjour le monde',
          languageCode: 'fr',
          words: []
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData,
          language: 'fr'
        }

        await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            languageCode: 'fr'
          })
        )
      })
    })

    describe('Diarization', () => {
      it('should enable diarization when diarize is true', async () => {
        const mockResponse = {
          text: 'Speaker one speaks. Speaker two responds.',
          languageCode: 'en',
          words: [
            { text: 'Speaker', start: 0.0, end: 0.3, type: 'word', speaker_id: 'speaker_0' },
            { text: 'one', start: 0.3, end: 0.5, type: 'word', speaker_id: 'speaker_0' },
            { text: 'speaks', start: 0.5, end: 0.8, type: 'word', speaker_id: 'speaker_0' },
            { text: 'Speaker', start: 1.0, end: 1.3, type: 'word', speaker_id: 'speaker_1' },
            { text: 'two', start: 1.3, end: 1.5, type: 'word', speaker_id: 'speaker_1' },
            { text: 'responds', start: 1.5, end: 2.0, type: 'word', speaker_id: 'speaker_1' }
          ]
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'conversation.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData,
          diarize: true
        }

        const result = await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            diarize: true
          })
        )
        expect(result.words).toHaveLength(6)
        // Verify speaker IDs are present
        expect((result.words as any)[0].speaker_id).toBe('speaker_0')
        expect((result.words as any)[3].speaker_id).toBe('speaker_1')
      })

      it('should not include diarize parameter when undefined', async () => {
        const mockResponse = {
          text: 'No diarization',
          languageCode: 'en',
          words: []
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.not.objectContaining({
            diarize: expect.anything()
          })
        )
      })
    })

    describe('Audio event tagging', () => {
      it('should enable audio event tagging when tagAudioEvents is true', async () => {
        const mockResponse = {
          text: 'Hello world',
          languageCode: 'en',
          words: [
            { text: 'Hello', start: 0.0, end: 0.3, type: 'word', speaker_id: 'speaker_0' },
            { text: '[laughter]', start: 0.5, end: 1.0, type: 'audio_event', speaker_id: null },
            { text: 'world', start: 1.2, end: 1.5, type: 'word', speaker_id: 'speaker_0' }
          ]
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData,
          tagAudioEvents: true
        }

        const result = await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            tagAudioEvents: true
          })
        )
        expect(result.words).toHaveLength(3)
        expect((result.words as any)[1].type).toBe('audio_event')
      })

      it('should not include tagAudioEvents parameter when undefined', async () => {
        const mockResponse = {
          text: 'No event tagging',
          languageCode: 'en',
          words: []
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.not.objectContaining({
            tagAudioEvents: expect.anything()
          })
        )
      })
    })

    describe('Combined features', () => {
      it('should support both diarization and audio event tagging', async () => {
        const mockResponse = {
          text: 'Hello world',
          languageCode: 'en',
          words: [
            { text: 'Hello', start: 0.0, end: 0.3, type: 'word', speaker_id: 'speaker_0' },
            { text: '[applause]', start: 0.5, end: 1.0, type: 'audio_event', speaker_id: null },
            { text: 'world', start: 1.2, end: 1.5, type: 'word', speaker_id: 'speaker_1' }
          ]
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData,
          diarize: true,
          tagAudioEvents: true
        }

        await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(mockSttConvert).toHaveBeenCalledWith(
          expect.objectContaining({
            diarize: true,
            tagAudioEvents: true
          })
        )
      })
    })

    describe('Multichannel transcription', () => {
      it('should handle multichannel response', async () => {
        const mockResponse = {
          transcripts: [
            {
              text: 'Channel one audio',
              languageCode: 'en',
              words: [{ text: 'Channel', start: 0.0, end: 0.3, type: 'word' }]
            },
            {
              text: 'Channel two audio',
              languageCode: 'en',
              words: [{ text: 'Channel', start: 0.0, end: 0.3, type: 'word' }]
            }
          ]
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'multichannel.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        const result = await mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)

        expect(result.text).toBe('Channel one audio Channel two audio')
        expect(result.language).toBe('en')
        expect(result.words).toHaveLength(2)
      })
    })

    describe('Error handling', () => {
      it('should throw MappingError if model is missing', async () => {
        const configWithoutModel = { ...mockConfig, defaultSttModel: undefined }
        const mapperWithoutModel = new ElevenLabsMapper(configWithoutModel)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          audio: audioData
        }

        await expect(
          mapperWithoutModel.executeTranscribe({} as any, 'test-api-key', configWithoutModel, params)
        ).rejects.toThrow(MappingError)
      })

      it('should throw MappingError for webhook response', async () => {
        const mockResponse = {
          message: 'Transcription started, will be sent to webhook'
        }
        mockSttConvert.mockResolvedValue(mockResponse)

        const audioBuffer = Buffer.from('fake-audio-data')
        const audioData: RosettaAudioData = {
          data: audioBuffer,
          filename: 'test.mp3',
          mimeType: 'audio/mpeg'
        }

        const params: TranscribeParams = {
          provider: 'elevenlabs',
          model: 'scribe_v1',
          audio: audioData
        }

        await expect(mapper.executeTranscribe({}, 'test-api-key', mockConfig, params)).rejects.toThrow(
          MappingError
        )
      })
    })
  })

  describe('Voice Listing', () => {
    it('should list available voices', async () => {
      const mockVoices = [
        {
          voiceId: 'voice1',
          name: 'Rachel',
          labels: { accent: 'american', gender: 'female' },
          category: 'premade',
          description: 'A calm, friendly voice',
          previewUrl: 'https://example.com/preview1.mp3',
          settings: { stability: 0.5, similarityBoost: 0.75 },
          isOwner: false
        },
        {
          voiceId: 'voice2',
          name: 'Adam',
          labels: { accent: 'american', gender: 'male' },
          category: 'premade',
          description: 'A deep, narrative voice',
          previewUrl: 'https://example.com/preview2.mp3',
          settings: { stability: 0.6, similarityBoost: 0.8 },
          isOwner: false
        }
      ]

      mockVoicesGetAll.mockResolvedValue({ voices: mockVoices })

      const result: RosettaVoiceList = await mapper.executeListVoices('test-api-key', mockConfig)

      expect(result.object).toBe('list')
      expect(result.data).toHaveLength(2)
      expect(result.data[0].id).toBe('voice1')
      expect(result.data[0].name).toBe('Rachel')
      expect(result.data[0].provider).toBe('elevenlabs')
      expect(result.data[1].id).toBe('voice2')
      expect(result.data[1].name).toBe('Adam')
    })

    it('should handle empty voice list', async () => {
      mockVoicesGetAll.mockResolvedValue({ voices: [] })

      const result = await mapper.executeListVoices('test-api-key', mockConfig)

      expect(result.object).toBe('list')
      expect(result.data).toHaveLength(0)
    })
  })

  describe('Error wrapping', () => {
    it('should wrap provider errors correctly', () => {
      const originalError = new Error('ElevenLabs API error')
      const wrappedError = mapper.wrapProviderError(originalError, 'elevenlabs')

      expect(wrappedError).toBeInstanceOf(ProviderAPIError)
      expect(wrappedError.message).toContain('ElevenLabs API error')
    })

    it('should preserve existing RosettaAIError instances', () => {
      const originalError = new MappingError('Already a Rosetta error', 'elevenlabs')
      const wrappedError = mapper.wrapProviderError(originalError, 'elevenlabs')

      expect(wrappedError).toBe(originalError)
    })
  })
})
