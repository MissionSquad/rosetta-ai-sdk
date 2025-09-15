/* eslint-disable no-console */
/**
 * ElevenLabs Custom Provider Example (TTS + Streaming + STT)
 *
 * Plan:
 * 1) Configure ElevenLabs as a custom provider using ElevenLabsMapper:
 *    - providerKey: 'elevenlabs'
 *    - supportedFeatures: ['tts', 'stt']
 *    - defaultTtsModel: 'eleven_flash_v2_5'
 *    - defaultSttModel: 'scribe_v1'
 *    - apiKey from examples/.env -> ELEVENLABS_API_KEY (or process env)
 *    - optional baseURL via ELEVENLABS_BASE_URL
 * 2) TTS (non-streaming): generate MP3 and save to examples/audio_output/elevenlabs_generated.mp3
 * 3) TTS (streaming): stream MP3 chunks and save to examples/audio_output/elevenlabs_streamed.mp3
 * 4) STT (sync): transcribe the generated MP3 or fallback to examples/sample_audio.mp3
 * 5) Error handling: print clear diagnostics using Rosetta errors
 *
 * Usage:
 * - Add ELEVENLABS_API_KEY to examples/.env
 * - (Optional) ELEVENLABS_BASE_URL for global preview endpoint
 * - Run via a script (e.g., yarn example:custom-elevenlabs) after adding it to package.json
 */

import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs/promises'
import { Readable } from 'stream'
import { pipeline } from 'stream/promises'

import {
  RosettaAI,
  RosettaAIError,
  ProviderAPIError,
  ConfigurationError,
  UnsupportedFeatureError,
  SpeechParams,
  TranscribeParams,
  RosettaAudioData,
  CustomProviderConfig
} from '../src'
import { ElevenLabsMapper } from '../src'

dotenv.config({ path: '.env' })

const outputDir = path.join(__dirname, 'audio_output')
const generatedFile = path.join(outputDir, 'elevenlabs_generated.mp3')
const streamedFile = path.join(outputDir, 'elevenlabs_streamed.mp3')
const sampleAudio = path.join(__dirname, 'sample_audio.mp3')

// Example voice (Rachel) from public ElevenLabs docs; replace with your preferred voiceId
const DEFAULT_VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

async function ensureDir(dir: string) {
  await fs.mkdir(dir, { recursive: true })
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.stat(p)
    return true
  } catch {
    return false
  }
}

async function run() {
  await ensureDir(outputDir)

  // Prepare custom provider config for ElevenLabs
  const providerKey = 'elevenlabs'
  const customConfig: CustomProviderConfig = {
    providerKey,
    mapper: ElevenLabsMapper,
    supportedFeatures: ['tts', 'stt', 'list_voices'],
    apiKey: process.env.ELEVENLABS_API_KEY,
    baseURL: process.env.ELEVENLABS_BASE_URL, // optional (e.g., global preview endpoint)
    defaultTtsModel: process.env.ROSETTA_DEFAULT_TTS_ELEVENLABS_MODEL ?? 'eleven_flash_v2_5',
    defaultSttModel: process.env.ROSETTA_DEFAULT_STT_ELEVENLABS_MODEL ?? 'scribe_v1'
  }

  if (!customConfig.apiKey) {
    console.error(
      'Missing ELEVENLABS_API_KEY. Please set it in examples/.env or your environment to run this example.'
    )
    return
  }

  // Initialize Rosetta with the ElevenLabs custom provider
  const rosetta = new RosettaAI({
    customProviders: [customConfig]
  })

  const configured = rosetta.getConfiguredProviders()
  console.log('Configured providers:', configured.join(', '))
  if (!configured.includes(providerKey)) {
    console.error(`Failed to initialize custom provider '${providerKey}'. Check configuration.`)
    return
  }

  // --- Voice Listing ---
  try {
    console.log('\n--- ElevenLabs Voice Listing ---')
    const voiceList = await rosetta.listVoices(providerKey)
    console.log(`Voices: ${voiceList.data.length} found`)
    // Print first 10 voices with id and name
    for (const v of voiceList.data.slice(0, 10)) {
      console.log(`- ${v.name ?? '(no name)'} (${v.id})`)
    }
  } catch (e) {
    console.warn('Voice listing not available or failed:', e instanceof Error ? e.message : String(e))
  }

  // --- TTS (non-streaming) ---
  try {
    console.log('\n--- ElevenLabs TTS (non-streaming) ---')
    const ttsParams: SpeechParams = {
      provider: providerKey,
      // model: 'eleven_multilingual_v2', // Optional override; else uses defaultTtsModel
      input:
        'Hello from RosettaAI! This audio was generated using ElevenLabs TTS via the custom provider integration.',
      voice: DEFAULT_VOICE_ID,
      responseFormat: 'mp3'
    }
    console.log(`Generating MP3 speech...`)
    const audioBuffer = await rosetta.generateSpeech(ttsParams)
    await fs.writeFile(generatedFile, audioBuffer)
    console.log(
      `Saved non-streaming TTS audio to: ${path.relative(process.cwd(), generatedFile)} (${(
        audioBuffer.length / 1024
      ).toFixed(1)} KB)`
    )
  } catch (error) {
    if (error instanceof ConfigurationError) {
      console.error(`TTS Configuration Error: ${error.message}`)
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(`TTS Unsupported Feature: ${error.message}`)
    } else if (error instanceof ProviderAPIError) {
      console.error(`TTS Provider API Error (${error.provider}): ${error.statusCode ?? 'N/A'} - ${error.message}`)
    } else if (error instanceof RosettaAIError) {
      console.error(`TTS Rosetta Error: ${error.name} - ${error.message}`)
    } else {
      console.error(`Unexpected TTS error:`, error)
    }
  }

  // --- TTS (streaming) ---
  try {
    console.log('\n--- ElevenLabs TTS (streaming) ---')
    const streamParams: SpeechParams = {
      provider: providerKey,
      // model: 'eleven_flash_v2_5', // Optional override; else uses defaultTtsModel
      input: 'Streaming audio with ElevenLabs for low-latency playback.',
      voice: DEFAULT_VOICE_ID,
      responseFormat: 'mp3'
    }

    const audioStream = rosetta.streamSpeech(streamParams)
    const chunks: Buffer[] = []
    let total = 0

    for await (const evt of audioStream) {
      if (evt.type === 'audio_chunk') {
        chunks.push(evt.data)
        total += evt.data.length
        process.stdout.write('.')
      } else if (evt.type === 'audio_stop') {
        console.log('\nStreaming finished.')
      } else if (evt.type === 'error') {
        console.error('\nStreaming error:', evt.data.error)
        throw evt.data.error
      }
    }

    const fileHandle = await fs.open(streamedFile, 'w')
    try {
      const readable = new Readable({
        read() {
          for (const b of chunks) this.push(b)
          this.push(null)
        }
      })
      await pipeline(readable, fileHandle.createWriteStream())
    } finally {
      await fileHandle.close()
    }

    console.log(
      `Saved streaming TTS audio to: ${path.relative(process.cwd(), streamedFile)} (${(total / 1024).toFixed(1)} KB)`
    )
  } catch (error) {
    if (error instanceof ConfigurationError) {
      console.error(`Streaming TTS Configuration Error: ${error.message}`)
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(`Streaming TTS Unsupported Feature: ${error.message}`)
    } else if (error instanceof ProviderAPIError) {
      console.error(
        `Streaming TTS Provider API Error (${error.provider}): ${error.statusCode ?? 'N/A'} - ${error.message}`
      )
    } else if (error instanceof RosettaAIError) {
      console.error(`Streaming TTS Rosetta Error: ${error.name} - ${error.message}`)
    } else {
      console.error(`Unexpected Streaming TTS error:`, error)
    }
  }

  // --- STT (transcription) ---
  try {
    console.log('\n--- ElevenLabs STT (transcription) ---')
    const inputPath = (await fileExists(generatedFile)) ? generatedFile : sampleAudio
    if (!(await fileExists(inputPath))) {
      console.warn(
        `No input audio found for STT. Generate TTS first or place a file at ${path.relative(
          process.cwd(),
          sampleAudio
        )}`
      )
      return
    }
    const inputBuf = await fs.readFile(inputPath)
    const audioData: RosettaAudioData = {
      data: inputBuf, // Buffer or Readable are supported
      filename: path.basename(inputPath),
      mimeType: path.extname(inputPath).toLowerCase() === '.mp3' ? 'audio/mpeg' : 'audio/wav'
    }

    const sttParams: TranscribeParams = {
      provider: providerKey,
      // model: 'scribe_v1', // Optional override; else uses defaultSttModel
      audio: audioData,
      responseFormat: 'text'
    }
    const result = await rosetta.transcribe(sttParams)
    console.log(`[ElevenLabs STT] Model: ${result.model}`)
    console.log(`[ElevenLabs STT] Text: "${result.text}"`)
  } catch (error) {
    if (error instanceof ConfigurationError) {
      console.error(`STT Configuration Error: ${error.message}`)
    } else if (error instanceof UnsupportedFeatureError) {
      console.error(`STT Unsupported Feature: ${error.message}`)
    } else if (error instanceof ProviderAPIError) {
      console.error(`STT Provider API Error (${error.provider}): ${error.statusCode ?? 'N/A'} - ${error.message}`)
    } else if (error instanceof RosettaAIError) {
      console.error(`STT Rosetta Error: ${error.name} - ${error.message}`)
    } else {
      console.error(`Unexpected STT error:`, error)
    }
  }

  console.log('\n--- ElevenLabs Example Complete ---')
}

run().catch(err => console.error('Unhandled error in ElevenLabs example script:', err))
