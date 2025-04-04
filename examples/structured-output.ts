/* eslint-disable no-console */
// Structured Output (JSON Mode) Example
import { RosettaAI, Provider, RosettaAIError, GenerateParams } from '../src'
import dotenv from 'dotenv'
import { z, ZodError } from 'zod' // Import Zod for schema validation

dotenv.config()

// --- Define Zod Schema for Validation ---
// Example schema for extracting contact information
const ContactInfoSchema = z
  .object({
    name: z
      .string()
      .optional()
      .describe('The full name of the contact.'),
    email: z
      .string()
      .email()
      .optional()
      .describe('The email address of the contact.'),
    phone: z
      .string()
      .optional()
      .describe('The phone number of the contact.'),
    company: z
      .string()
      .optional()
      .describe('The company the contact works for.')
  })
  .describe('Structure to hold extracted contact information.')

// Basic function to generate a system prompt describing the desired JSON structure
// This helps guide models that don't have a strict JSON mode parameter.
function createJsonPrompt(description: string, schema: Record<string, any>): string {
  // Basic JSON schema description - can be enhanced
  const schemaString = JSON.stringify(schema, null, 2)
  return `Please extract the information according to the following description and respond ONLY with a single valid JSON object that adheres to the provided schema. Do not include any other text, explanations, or markdown formatting.

Description: ${description}

JSON Schema:
${schemaString}`
}

// Generate a JSON schema representation from Zod (basic version)
// Note: Libraries like zod-to-json-schema do this more robustly
function zodToJsonSchema(schema: z.ZodObject<any>): Record<string, any> {
  const properties: Record<string, any> = {}
  const required: string[] = []
  for (const key in schema.shape) {
    const field = schema.shape[key]
    const type =
      field instanceof z.ZodString
        ? 'string'
        : field instanceof z.ZodNumber
        ? 'number'
        : field instanceof z.ZodBoolean
        ? 'boolean'
        : field instanceof z.ZodArray
        ? 'array'
        : 'object' // Simplification
    properties[key] = { type }
    if (field._def.description) {
      properties[key].description = field._def.description
    }
    if (!field.isOptional()) {
      required.push(key)
    }
  }
  const jsonSchema: Record<string, any> = { type: 'object', properties }
  if (required.length > 0) {
    jsonSchema.required = required
  }
  return jsonSchema
}

async function runStructuredOutput() {
  const rosetta = new RosettaAI()

  // Filter providers that potentially support structured output
  // OpenAI has direct support; Google can be prompted.
  const providers = rosetta.getConfiguredProviders().filter(p => [Provider.OpenAI, Provider.Google].includes(p))

  if (providers.length === 0) {
    console.error('No configured providers support JSON output (OpenAI, Google needed).')
    return
  }

  const inputText =
    'Extract contact details: The main contact is Jane Doe at Example Corp. Reach her via email at jane.d@example.org or call 555-1234. Her colleague is John Smith (john.smith@sample.com).'
  const jsonSchema = zodToJsonSchema(ContactInfoSchema)
  const systemPrompt = createJsonPrompt(ContactInfoSchema.description ?? 'Extract contact information.', jsonSchema)

  for (const provider of providers) {
    console.log(`\n--- Testing JSON Output with: ${provider} ---`)
    try {
      // Select a capable model
      const model =
        rosetta.config.defaultModels?.[provider] ??
        (provider === Provider.OpenAI
          ? 'gpt-4o-mini' // GPT-4 Turbo or GPT-4o recommended for JSON mode
          : provider === Provider.Google
          ? 'gemini-1.5-flash-latest' // Gemini models can follow JSON instructions
          : undefined)

      if (!model) {
        console.log(`Skipping ${provider}: No default model configured or fallback available.`)
        continue
      }
      console.log(`Using model: ${model}`)

      // Prepare generation parameters
      const generateParams: GenerateParams = {
        provider,
        model,
        messages: [
          // Provide the schema description in the system prompt, especially for Google
          { role: 'system', content: systemPrompt },
          { role: 'user', content: inputText }
        ],
        // Request JSON format specifically for OpenAI
        responseFormat: provider === Provider.OpenAI ? { type: 'json_object' } : undefined,
        temperature: 0.1, // Lower temperature for more predictable structured output
        maxTokens: 200
      }

      // Generate the response
      const result = await rosetta.generate(generateParams)

      console.log(`[${provider} Response - Model: ${result.model}]`)
      console.log('Finish Reason:', result.finishReason)
      console.log('Usage:', result.usage ? JSON.stringify(result.usage) : 'N/A')
      console.log('Raw Content:', result.content ?? '[No Content]')

      // Attempt to parse the content (either directly from parsedContent or manually)
      let parsedData: any = result.parsedContent // Use SDK's parsed result if available

      if (!parsedData && result.content) {
        console.log('Attempting manual JSON parsing...')
        try {
          // Clean potential markdown fences (```json ... ```)
          const cleanContent = result.content.replace(/^```json\s*|```$/g, '').trim()
          parsedData = JSON.parse(cleanContent)
          console.log('(Manual parse successful)')
        } catch (parseError) {
          console.warn('Manual JSON parsing failed:', (parseError as Error).message)
        }
      }

      // Validate the parsed data against the Zod schema
      if (parsedData) {
        console.log('\nParsed JSON Data:')
        try {
          const validatedData = ContactInfoSchema.parse(parsedData)
          console.log('Schema Validation: SUCCESS')
          console.log('Validated Data:', JSON.stringify(validatedData, null, 2))
        } catch (validationError) {
          console.error('Schema Validation: FAILED')
          if (validationError instanceof ZodError) {
            console.error('Validation Issues:', JSON.stringify(validationError.errors, null, 2))
          } else {
            console.error('Unknown validation error:', validationError)
          }
          // Log the raw parsed data that failed validation
          console.log('Raw Parsed Data (Failed Validation):', JSON.stringify(parsedData, null, 2))
        }
      } else {
        console.log('\nFailed to obtain or parse JSON data from the response.')
      }
    } catch (error) {
      if (error instanceof RosettaAIError) {
        console.error(`Error during JSON output test (${provider}): ${error.name} - ${error.message}`)
      } else {
        console.error(`Unexpected error during JSON output test (${provider}):`, error)
      }
    }
    await new Promise(resolve => setTimeout(resolve, 1500)) // Delay
  } // End provider loop

  console.log('----------------------------------------\nStructured Output Test Complete.')
}

// Run the example
runStructuredOutput().catch(err => console.error('Unhandled error in structured output example script:', err))
