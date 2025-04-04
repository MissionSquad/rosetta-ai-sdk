/* eslint-disable no-console */
// Image Input Example
import { RosettaAI, Provider, RosettaMessage, RosettaAIError, RosettaImageData, ImageMimeType } from '../src'
import dotenv from 'dotenv'
import fs from 'fs/promises'
import path from 'path'

dotenv.config()

// Helper function to encode image file to Base64
async function encodeImageToBase64(filePath: string): Promise<RosettaImageData | null> {
  try {
    // Read the file into a buffer
    const buffer = await fs.readFile(filePath)
    // Convert buffer to Base64 string
    const base64Data = buffer.toString('base64')
    // Determine MIME type from file extension
    const ext = path.extname(filePath).toLowerCase()
    let mimeType: ImageMimeType

    switch (ext) {
      case '.png':
        mimeType = 'image/png'
        break
      case '.jpg':
      case '.jpeg':
        mimeType = 'image/jpeg'
        break
      case '.gif':
        mimeType = 'image/gif'
        break
      case '.webp':
        mimeType = 'image/webp'
        break
      default:
        console.warn(`Unsupported image extension: ${ext}. Skipping file: ${filePath}`)
        return null // Return null for unsupported types
    }

    return { mimeType, base64Data }
  } catch (error) {
    // Handle file not found or other reading errors
    if ((error as any).code === 'ENOENT') {
      console.error(`Image file not found: "${filePath}"`)
    } else {
      console.error(`Failed to read or encode image "${filePath}": ${(error as any).message}`)
    }
    return null // Return null on error
  }
}

async function runImageInputChat() {
  console.log('--- Testing Image Input ---')
  const rosetta = new RosettaAI()

  // Filter providers that support image input (multimodal)
  const providers = rosetta
    .getConfiguredProviders()
    .filter(p => [Provider.OpenAI, Provider.Anthropic, Provider.Google].includes(p))

  if (providers.length === 0) {
    console.error('No configured providers support image input (OpenAI, Anthropic, Google needed).')
    return
  }

  // Path to the image file (ensure logo.png exists in the examples directory)
  const imagePath = path.join(__dirname, 'logo.png')
  const imageData = await encodeImageToBase64(imagePath)

  if (!imageData) {
    console.error(`Failed to load or encode image data from ${imagePath}. Aborting test.`)
    return
  }

  console.log(`Using image: ${path.basename(imagePath)} (MIME Type: ${imageData.mimeType})`)

  // Iterate through supported providers
  for (const provider of providers) {
    console.log(`\n--- Provider: ${provider} ---`)
    try {
      // Construct the multimodal message
      const messages: RosettaMessage[] = [
        {
          role: 'user',
          content: [
            // Content is an array of parts
            { type: 'text', text: 'What is shown in this image? Describe the main elements.' },
            { type: 'image', image: imageData } // Image part
          ]
        }
      ]

      // Select an appropriate model (defaults or specific multimodal models)
      const model =
        rosetta.config.defaultModels?.[provider] ??
        (provider === Provider.Anthropic
          ? 'claude-3-haiku-20240307' // Haiku supports vision
          : provider === Provider.Google
          ? 'gemini-1.5-flash-latest' // Flash supports vision
          : provider === Provider.OpenAI
          ? 'gpt-4o-mini' // GPT-4o mini supports vision
          : undefined)

      if (!model) {
        console.log(`Skipping ${provider}: No default or suitable multimodal model configured.`)
        continue
      }
      console.log(`Using model: ${model}`)

      // Make the generation request
      const result = await rosetta.generate({
        provider,
        model,
        messages,
        maxTokens: 100 // Limit response length
      })

      // Display results
      console.log(`[${provider} Response - Model: ${result.model}]:`)
      console.log('Content:', result.content ?? '[No Content]')
      console.log('Finish Reason:', result.finishReason)
      console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
    } catch (error) {
      if (error instanceof RosettaAIError) {
        console.error(`Error with ${provider}: ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected error with ${provider}:`, error)
      }
    }
    // Delay between provider calls
    await new Promise(resolve => setTimeout(resolve, 1500))
  } // End provider loop

  console.log('-----------------------\nImage Input Test Complete.')
}

// Run the example
runImageInputChat().catch(err => console.error('Unhandled error in image input example script:', err))
