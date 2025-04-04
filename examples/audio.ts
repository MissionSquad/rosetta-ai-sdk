/* eslint-disable no-console */
// Audio API Example (TTS & Transcription/Translation)
import {
  RosettaAI,
  Provider,
  SpeechParams,
  TranscribeParams,
  TranslateParams,
  RosettaAudioData,
  RosettaAIError,
  ProviderAPIError, // Import specific error types
  ConfigurationError,
  UnsupportedFeatureError
} from '../src'
import dotenv from 'dotenv'
import fs from 'fs/promises' // Use fs/promises for async operations
import { pipeline } from 'stream/promises' // Use pipeline for robust stream handling
import path from 'path'
import { Readable } from 'stream'

dotenv.config()

// Define output directory and paths
const outputDir = path.join(__dirname, 'audio_output')
const generatedSpeechFilePath = path.join(outputDir, 'generated_speech.mp3')
// IMPORTANT: Create or place a real audio file here for STT/Translate tests
const sampleAudioFilePath = path.join(__dirname, 'sample_audio.mp3')

// Helper function to check if a file exists
async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.stat(filePath)
    return true
  } catch (error) {
    if ((error as any).code === 'ENOENT') {
      return false
    }
    throw error // Re-throw other errors
  }
}

async function runAudioTests() {
  // Ensure output directory exists asynchronously
  try {
    await fs.mkdir(outputDir, { recursive: true })
    console.log(`Ensured output directory exists: ${outputDir}`)
  } catch (error) {
    console.error(`Failed to create output directory: ${outputDir}`, error)
    return // Stop if we can't create the directory
  }

  const rosetta = new RosettaAI()
  const configuredProviders = rosetta.getConfiguredProviders()

  // --- Test Text-to-Speech (TTS) ---
  console.log('\n--- Testing Text-to-Speech (TTS) ---')
  if (configuredProviders.includes(Provider.OpenAI)) {
    try {
      const ttsParams: SpeechParams = {
        provider: Provider.OpenAI, // TTS currently mapped via OpenAI
        input:
          'Hello from the Rosetta AI software development kit! This is a test of the text to speech functionality.',
        voice: 'nova', // Choose a voice (alloy, echo, fable, onyx, nova, shimmer)
        responseFormat: 'mp3' // Specify output format
        // model: 'tts-1' // Optional: specify model if not default
      }
      console.log(`Generating speech for: "${ttsParams.input.substring(0, 50)}..."`)

      // Generate speech (non-streaming)
      const audioBuffer = await rosetta.generateSpeech(ttsParams)

      // Save the generated audio file asynchronously
      await fs.writeFile(generatedSpeechFilePath, audioBuffer)
      console.log(
        `Speech saved to: ${path.relative(process.cwd(), generatedSpeechFilePath)} (${(
          audioBuffer.length / 1024
        ).toFixed(1)} KB)`
      )

      // Optional: Test streaming TTS
      console.log('\nTesting Streaming TTS...')
      const stream = rosetta.streamSpeech(ttsParams)
      const streamFilePath = path.join(outputDir, 'streamed_speech.mp3')
      const fileWriteStream = (await fs.open(streamFilePath, 'w')).createWriteStream() // Get WriteStream from file handle

      let streamedBytes = 0
      const audioChunks: Buffer[] = [] // Collect chunks to write via pipeline

      // Collect chunks from the SDK stream
      for await (const chunk of stream) {
        if (chunk.type === 'audio_chunk') {
          process.stdout.write('.') // Indicate progress
          audioChunks.push(chunk.data)
          streamedBytes += chunk.data.length
        } else if (chunk.type === 'audio_stop') {
          console.log('\nStreaming TTS finished.')
        } else if (chunk.type === 'error') {
          console.error('\nStreaming TTS Error:', chunk.data.error.message)
          // Close the file stream on error if it's open
          if (!fileWriteStream.closed) {
            fileWriteStream.close()
          }
          throw chunk.data.error // Re-throw to be caught by outer catch
        }
      }

      // Use pipeline to write collected chunks to the file stream
      // Create a Readable stream from the collected chunks
      const readableStream = new Readable({
        read() {
          for (const chunk of audioChunks) {
            this.push(chunk)
          }
          this.push(null) // Signal end of data
        }
      })

      await pipeline(readableStream, fileWriteStream) // Pipeline handles errors and closing

      console.log(
        `Streamed speech saved to: ${path.relative(process.cwd(), streamFilePath)} (${(streamedBytes / 1024).toFixed(
          1
        )} KB)`
      )
    } catch (error) {
      // Demonstrate more specific error handling
      if (error instanceof ConfigurationError) {
        console.error(`TTS Configuration Error: ${error.message}`)
      } else if (error instanceof UnsupportedFeatureError) {
        console.error(`TTS Feature Error: ${error.message}`)
      } else if (error instanceof ProviderAPIError) {
        console.error(`TTS Provider API Error (${error.provider}): ${error.statusCode ?? 'N/A'} - ${error.message}`)
        // console.error("Underlying TTS Provider Error:", error.underlyingError); // Optional: log more details
      } else if (error instanceof RosettaAIError) {
        console.error(`TTS Error: ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected TTS error:`, error)
      }
    }
  } else {
    console.warn('Skipping TTS tests: OpenAI provider not configured.')
  }

  // --- Test Speech-to-Text (STT) & Translation ---
  console.log('\n--- Testing Speech-to-Text (STT) & Translation ---')

  // Determine which audio file to use for STT/Translation asynchronously
  let inputAudioPath: string
  if (await fileExists(generatedSpeechFilePath)) {
    inputAudioPath = generatedSpeechFilePath
  } else if (await fileExists(sampleAudioFilePath)) {
    inputAudioPath = sampleAudioFilePath
  } else {
    console.error(
      `STT/Translation Error: Input audio file not found at "${generatedSpeechFilePath}" or "${sampleAudioFilePath}". Please create sample_audio.mp3 or run TTS first.`
    )
    return
  }
  console.log(`Using audio file for STT/Translation: ${path.basename(inputAudioPath)}`)

  // Filter providers that support STT
  const sttProviders = configuredProviders.filter(p => [Provider.OpenAI, Provider.Groq].includes(p))

  if (sttProviders.length === 0) {
    console.warn('Skipping STT/Translation tests: Neither OpenAI nor Groq provider configured.')
    return
  }

  // Prepare audio data once asynchronously
  let audioBuffer: Buffer
  try {
    audioBuffer = await fs.readFile(inputAudioPath)
  } catch (readError) {
    console.error(`Failed to read audio file "${inputAudioPath}":`, readError)
    return
  }

  const audioData: RosettaAudioData = {
    data: audioBuffer, // Pass the buffer directly
    filename: path.basename(inputAudioPath),
    // Determine mime type based on file used
    mimeType: path.extname(inputAudioPath) === '.mp3' ? 'audio/mpeg' : 'audio/wav' // Adjust if using different formats
  }

  // Test each STT provider
  for (const provider of sttProviders) {
    console.log(`\n--- STT/Translation Provider: ${provider} ---`)
    try {
      // --- Transcription ---
      console.log(`Transcribing audio...`)
      const transcribeParams: TranscribeParams = {
        provider,
        audio: audioData,
        responseFormat: 'text' // Request plain text output
        // language: 'en' // Optional: Provide language hint
        // model: 'whisper-1' // Optional: specify model
      }
      const transcriptionResult = await rosetta.transcribe(transcribeParams)
      console.log(`[${provider} Transcription Result - Model: ${transcriptionResult.model}]:`)
      console.log(`"${transcriptionResult.text}"`)

      // --- Translation (if applicable) ---
      // Both OpenAI and Groq whisper models typically support translation
      console.log(`\nTranslating audio to English...`)
      const translateParams: TranslateParams = {
        provider,
        audio: audioData,
        responseFormat: 'text'
        // model: 'whisper-1' // Optional: specify model
      }
      const translationResult = await rosetta.translate(translateParams)
      console.log(`[${provider} Translation Result - Model: ${translationResult.model}]:`)
      console.log(`"${translationResult.text}"`)
    } catch (error) {
      // Demonstrate more specific error handling
      if (error instanceof ConfigurationError) {
        console.error(`STT/Translation Configuration Error (${provider}): ${error.message}`)
      } else if (error instanceof UnsupportedFeatureError) {
        console.error(`STT/Translation Feature Error (${provider}): ${error.message}`)
      } else if (error instanceof ProviderAPIError) {
        console.error(
          `STT/Translation Provider API Error (${error.provider}): ${error.statusCode ?? 'N/A'} - ${error.message}`
        )
        // console.error("Underlying STT/Translation Provider Error:", error.underlyingError); // Optional
      } else if (error instanceof RosettaAIError) {
        console.error(`STT/Translation Error (${provider}): ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected STT/Translation error (${provider}):`, error)
      }
    }
    await new Promise(resolve => setTimeout(resolve, 1000)) // Delay between providers
  } // End provider loop

  console.log('---------------------------\nAudio Tests Complete.')
}

// Run the example tests
runAudioTests().catch(err => console.error('Unhandled error in audio example script:', err))
