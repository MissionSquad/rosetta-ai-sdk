import {
  Content,
  Tool,
  GenerateContentParameters,
  GenerateContentConfig,
  GenerateContentResponse,
  CitationMetadata,
  Type as GoogleSchemaType,
  Schema as FunctionDeclarationSchema,
  Part,
  FinishReason,
  HarmCategory,
  HarmBlockThreshold,
  EmbedContentParameters,
  EmbedContentResponse
} from '@google/genai'
import {
  GenerateParams,
  GenerateResult,
  StreamChunk,
  RosettaMessage,
  RosettaToolCallRequest,
  TokenUsage,
  Provider,
  Citation,
  EmbedParams,
  EmbedResult,
  TranscribeParams,
  TranslateParams,
  TranscriptionResult
} from '../../types'
import { MappingError, ProviderAPIError, RosettaAIError, UnsupportedFeatureError } from '../../errors'
import { safeGet } from '../utils'
import { IProviderMapper } from './base.mapper'
import { mapTokenUsage, mapBaseParams } from './common.utils'
import * as GoogleEmbedMapper from './google.embed.mapper'

export class GoogleMapper implements IProviderMapper {
  readonly provider = Provider.Google

  // --- Parameter Mapping ---
  private mapRoleToGoogle(role: RosettaMessage['role']): 'user' | 'model' | 'function' | 'system' {
    switch (role) {
      case 'user':
        return 'user'
      case 'assistant':
        return 'model'
      case 'tool':
        return 'function'
      case 'system':
        return 'system'
      default:
        // Ensure exhaustive check works with `never`
        const _e: never = role
        throw new MappingError(`Unsupported role: ${_e}`, this.provider)
    }
  }

  private mapContentToGoogleParts(content: RosettaMessage['content']): Part[] {
    if (content === null) {
      console.warn('Mapping null content to empty parts array for Google history.')
      return []
    }
    if (typeof content === 'string') {
      // Handle empty string case - return empty array as Google requires non-empty parts for user messages
      if (content === '') {
        console.warn('Mapping empty string content to empty parts array for Google history.')
        return []
      }
      return [{ text: content }]
    }
    // Handle empty array case
    if (Array.isArray(content) && content.length === 0) {
      console.warn('Mapping empty content array to empty parts array for Google history.')
      return []
    }
    return content.map(part => {
      if (part.type === 'text') return { text: part.text }
      if (part.type === 'image') return { inlineData: { mimeType: part.image.mimeType, data: part.image.base64Data } }
      // Ensure exhaustive check works with `never`
      const _e: never = part
      throw new MappingError(`Unsupported content part: ${(_e as any).type}`, this.provider)
    })
  }

  private isFunctionDeclarationSchema(schema: any): schema is FunctionDeclarationSchema {
    return (
      typeof schema === 'object' &&
      schema !== null &&
      'type' in schema &&
      Object.values(GoogleSchemaType).includes(schema.type)
    )
  }

  /**
   * Removes properties that cause issues with the Google API from a schema object.
   * Specifically removes 'additionalProperties' and '$schema' properties.
   *
   * @param schema - The original schema object
   * @returns A new schema object with problematic properties removed
   */
  private cleanSchemaForGoogle(schema: any, visited = new Set<any>()): any {
    if (!schema || typeof schema !== 'object') {
      return schema;
    }

    if (visited.has(schema)) {
      // console.warn('RosettaAI GoogleMapper: Circular reference detected in schema. Returning as-is to break cycle.');
      return schema; // Break cycle
    }

    visited.add(schema);

    // Create a mutable copy to avoid modifying the original schema object at this level
    const cleanedSchema = { ...schema };

    // Remove standard problematic properties for Google
    delete cleanedSchema.additionalProperties;
    delete cleanedSchema.$schema;

    // Normalize/strip unsupported JSON Schema constructs for Google Function Declarations

    // Map lowercase/JSON Schema "type" (or array of types) to Google SchemaType
    const toGoogleType = (t: any): GoogleSchemaType | undefined => {
      const map: Record<string, GoogleSchemaType> = {
        string: GoogleSchemaType.STRING,
        number: GoogleSchemaType.NUMBER,
        integer: GoogleSchemaType.INTEGER,
        boolean: GoogleSchemaType.BOOLEAN,
        array: GoogleSchemaType.ARRAY,
        object: GoogleSchemaType.OBJECT
      };
      if (typeof t === 'string') return map[t.toLowerCase()];
      return undefined;
    };

    if (cleanedSchema.type !== undefined) {
      if (Array.isArray(cleanedSchema.type)) {
        // Choose a compatible single type (prefer the first recognized), fallback to STRING
        const first = cleanedSchema.type.find((x: any) => typeof x === 'string');
        const g = toGoogleType(first);
        cleanedSchema.type = g ?? GoogleSchemaType.STRING;
      } else if (typeof cleanedSchema.type === 'string') {
        const g = toGoogleType(cleanedSchema.type);
        if (g) cleanedSchema.type = g;
      }
    }

    // Convert const -> enum (Gemini doesn't accept "const" in schemas)
    if (Object.prototype.hasOwnProperty.call(cleanedSchema, 'const')) {
      const c = (cleanedSchema as any).const;
      delete (cleanedSchema as any).const;
      if (c !== undefined) {
        if (Array.isArray(cleanedSchema.enum)) {
          cleanedSchema.enum = Array.from(new Set([...(cleanedSchema.enum as any[]), c]));
        } else {
          cleanedSchema.enum = [c];
        }
      }
    }

    // Helper: extract enum values from union members (anyOf/oneOf) when they are const/enum literals
    const extractEnumFromUnion = (arr: any[]): any[] | null => {
      const values: any[] = [];
      for (const s of arr) {
        if (s && typeof s === 'object') {
          if (Object.prototype.hasOwnProperty.call(s, 'const')) {
            values.push((s as any).const);
          } else if (Array.isArray((s as any).enum)) {
            values.push(...(s as any).enum);
          }
        }
      }
      return values.length > 0 ? Array.from(new Set(values)) : null;
    };

    // Normalize anyOf/oneOf -> enum when possible; otherwise drop and fallback to STRING
    const unionKeys = ['anyOf', 'oneOf'] as const;
    for (const key of unionKeys) {
      const unionVal = (cleanedSchema as any)[key];
      if (Array.isArray(unionVal)) {
        const enums = extractEnumFromUnion(unionVal);
        delete (cleanedSchema as any)[key];
        if (enums && enums.length > 0) {
          cleanedSchema.enum = Array.isArray(cleanedSchema.enum)
            ? Array.from(new Set([...(cleanedSchema.enum as any[]), ...enums]))
            : enums;
          if (!cleanedSchema.type) cleanedSchema.type = GoogleSchemaType.STRING;
        } else {
          // Heterogeneous unions (e.g., string|number) are not representable: default to STRING
          if (!cleanedSchema.type) cleanedSchema.type = GoogleSchemaType.STRING;
        }
      }
    }

    // Defensive: remove snake_case union keys if present (SDK may transform keys)
    delete (cleanedSchema as any).any_of;
    delete (cleanedSchema as any).one_of;

    // Remove other unsupported JSON Schema keywords for Google function declarations
    const UNSUPPORTED = [
      'allOf','all_of','not','if','then','else','dependentSchemas','dependent_schemas',
      'patternProperties','pattern_properties','contains','unevaluatedProperties','unevaluated_properties',
      'nullable','$id','$ref','examples'
    ];
    for (const k of UNSUPPORTED) {
      if (Object.prototype.hasOwnProperty.call(cleanedSchema, k)) delete (cleanedSchema as any)[k];
    }

    // If enum exists and type is missing, default to STRING (Gemini expects a concrete type)
    if (cleanedSchema.enum && !cleanedSchema.type) {
      cleanedSchema.type = GoogleSchemaType.STRING;
    }

    // Handle exclusiveMinimum:
    if (typeof cleanedSchema.exclusiveMinimum === 'number') {
      if (typeof cleanedSchema.minimum === 'number' && cleanedSchema.minimum !== cleanedSchema.exclusiveMinimum) {
        console.warn(`RosettaAI GoogleMapper: Schema for property had both 'minimum: ${cleanedSchema.minimum}' and 'exclusiveMinimum: ${cleanedSchema.exclusiveMinimum}'. Prioritizing 'exclusiveMinimum' value and converting to 'minimum: ${cleanedSchema.exclusiveMinimum}'.`);
      }
      cleanedSchema.minimum = cleanedSchema.exclusiveMinimum;
      delete cleanedSchema.exclusiveMinimum;
    } else if (cleanedSchema.exclusiveMinimum === true && typeof cleanedSchema.minimum === 'number') {
      console.warn(`RosettaAI GoogleMapper: Converting 'exclusiveMinimum: true' with 'minimum: ${cleanedSchema.minimum}'. The bound will become inclusive. If strict inequality was intended, the original schema's 'minimum' value might need adjustment (e.g., incrementing for integers).`);
      delete cleanedSchema.exclusiveMinimum;
    }

    // Handle exclusiveMaximum (similar logic to exclusiveMinimum):
    if (typeof cleanedSchema.exclusiveMaximum === 'number') {
      if (typeof cleanedSchema.maximum === 'number' && cleanedSchema.maximum !== cleanedSchema.exclusiveMaximum) {
        console.warn(`RosettaAI GoogleMapper: Schema for property had both 'maximum: ${cleanedSchema.maximum}' and 'exclusiveMaximum: ${cleanedSchema.exclusiveMaximum}'. Prioritizing 'exclusiveMaximum' value and converting to 'maximum: ${cleanedSchema.exclusiveMaximum}'.`);
      }
      cleanedSchema.maximum = cleanedSchema.exclusiveMaximum;
      delete cleanedSchema.exclusiveMaximum;
    } else if (cleanedSchema.exclusiveMaximum === true && typeof cleanedSchema.maximum === 'number') {
      console.warn(`RosettaAI GoogleMapper: Converting 'exclusiveMaximum: true' with 'maximum: ${cleanedSchema.maximum}'. The bound will become inclusive. If strict inequality was intended, the original schema's 'maximum' value might need adjustment (e.g., decrementing for integers).`);
      delete cleanedSchema.exclusiveMaximum;
    }

    // Recursively clean nested 'properties' (for object schemas)
    // Pass the original sub-schema (schema.properties[key]) for cycle detection
    if (schema.properties && typeof schema.properties === 'object') {
      const newProperties: Record<string, any> = {};
      for (const key in schema.properties) {
        if (Object.prototype.hasOwnProperty.call(schema.properties, key)) {
          newProperties[key] = this.cleanSchemaForGoogle(schema.properties[key], visited);
        }
      }
      cleanedSchema.properties = newProperties;
    } else if (cleanedSchema.properties) { // Ensure properties is not carried over if original schema didn't have it as object
        delete cleanedSchema.properties 
    }


    // Recursively clean 'items' (for array schemas)
    // Pass the original sub-schema (schema.items) for cycle detection
    if (schema.items && typeof schema.items === 'object') {
      cleanedSchema.items = this.cleanSchemaForGoogle(schema.items, visited);
    } else if (cleanedSchema.items) { // Ensure items is not carried over if original schema didn't have it as object
        delete cleanedSchema.items
    }
    
    visited.delete(schema); // Clean up visited set for this path

    return cleanedSchema;
  }

  private findLastToolCallName(history: Content[], _toolCallId: string): string | undefined {
    for (let i = history.length - 1; i >= 0; i--) {
      const prevMsg = history[i]
      if (prevMsg?.role === 'model' && Array.isArray(prevMsg.parts)) {
        for (const part of prevMsg.parts) {
          if ('functionCall' in part && part.functionCall?.name) {
            return part.functionCall.name
          }
        }
      }
    }
    console.warn(`Could not determine preceding function name for tool result (ID: ${_toolCallId}) from history.`)
    return undefined
  }

  mapToProviderParams(
    params: GenerateParams
  ): GenerateContentParameters {
    let systemInstruction: Content | undefined = undefined
    const contents: Content[] = []
    const messagesToProcess = [...params.messages]

    const lastMessage = messagesToProcess.pop()
    if (!lastMessage) {
      throw new MappingError('No messages provided to map for Google.', this.provider)
    }

    messagesToProcess.forEach(msg => {
      const googleRole = this.mapRoleToGoogle(msg.role)

      if (googleRole === 'system') {
        if (systemInstruction)
          throw new MappingError('Multiple system messages not supported by Google.', this.provider)
        if (typeof msg.content !== 'string')
          throw new MappingError('Google system instruction must be string.', this.provider)
        // Note: In the new @google/genai SDK, systemInstruction is a Content object where
        // the role must be 'user' or 'model' (per SDK Content interface).
        // The systemInstruction field itself designates this as a system-level instruction.
        // See: https://ai.google.dev/gemini-api/docs/migrate#configuration
        systemInstruction = { role: 'user', parts: [{ text: msg.content }] }
        return
      }

      const parts = this.mapContentToGoogleParts(msg.content)
      // Skip adding entries if parts array is empty (from null/empty string/array content)
      if (parts.length === 0 && googleRole !== 'model') {
        // Allow empty parts for model role if tool calls are present
        // @ts-ignore
        if (!(googleRole === 'model' && msg.toolCalls && msg.toolCalls.length > 0)) {
          console.warn(`Skipping message with role '${googleRole}' due to empty content parts.`)
          return
        }
      }

      if (googleRole === 'model' && msg.toolCalls && msg.toolCalls.length > 0) {
        const functionCallParts: Part[] = msg.toolCalls.map(tc => {
          try {
            const part: Part = { functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) } }
            // Re-attach thought signature if present (required by Gemini 3+ for function calling)
            if (tc.providerMetadata?.thoughtSignature) {
              part.thoughtSignature = tc.providerMetadata.thoughtSignature as string
            }
            return part
          } catch (e) {
            throw new MappingError(
              `Failed to parse arguments for tool ${tc.function.name}`,
              this.provider,
              'mapToProviderParams toolCall mapping',
              e
            )
          }
        })
        const existingTextParts = parts.filter((p): p is Part => 'text' in p)
        // Ensure parts array is not empty if only function calls exist
        const finalParts = [...existingTextParts, ...functionCallParts]
        if (finalParts.length === 0) {
          // This case should be rare, but handle defensively
          console.warn(`Model message with tool calls resulted in empty parts array.`)
          return // Skip adding empty message
        }
        contents.push({ role: googleRole, parts: finalParts })
      } else if (googleRole === 'function') {
        if (!msg.toolCallId || typeof msg.content !== 'string') {
          throw new MappingError(
            'Invalid tool result message for Google. Requires toolCallId and string content.',
            this.provider
          )
        }
        const funcName = this.findLastToolCallName(contents, msg.toolCallId)
        if (!funcName) {
          throw new MappingError(
            `Cannot find function name for tool result (ID: ${msg.toolCallId}). Ensure model message with FunctionCall precedes this tool message.`,
            this.provider
          )
        }
        let respContent: any
        try {
          respContent = JSON.parse(msg.content)
        } catch {
          respContent = { content: msg.content } // Wrap non-JSON string content
          console.warn(`Tool result content for ${funcName} was not valid JSON. Wrapping as { content: "..." }`)
        }
        contents.push({
          role: googleRole,
          parts: [
            {
              functionResponse: {
                name: funcName,
                response: respContent
              }
            }
          ]
        })
      } else {
        // Only add if parts is not empty
        if (parts.length > 0) {
          contents.push({ role: googleRole, parts })
        } else {
          console.warn(`Skipping message with role '${googleRole}' due to empty content parts.`)
        }
      }
    })

    let lastMsgParts: Part[]
    const lastMessageRole = this.mapRoleToGoogle(lastMessage.role)

    if (lastMessageRole === 'function') {
      if (!lastMessage.toolCallId || typeof lastMessage.content !== 'string') {
        throw new MappingError(
          'Invalid last message: Tool result requires toolCallId and string content.',
          this.provider
        )
      }
      const funcName = this.findLastToolCallName(contents, lastMessage.toolCallId)
      if (!funcName) {
        throw new MappingError(
          `Cannot find function name for final tool result (ID: ${lastMessage.toolCallId}).`,
          this.provider
        )
      }
      let respContent: any
      try {
        respContent = JSON.parse(lastMessage.content)
      } catch {
        respContent = { content: lastMessage.content } // Wrap non-JSON string content
        console.warn(`Final tool result content for ${funcName} was not valid JSON. Wrapping as { content: "..." }`)
      }
      lastMsgParts = [
        {
          functionResponse: {
            name: funcName,
            response: respContent
          }
        }
      ]
    } else if (lastMessageRole === 'user') {
      lastMsgParts = this.mapContentToGoogleParts(lastMessage.content)
      if (lastMsgParts.length === 0) {
        // Google requires the final user message to have content parts
        throw new MappingError('Final user message content cannot be null or empty.', this.provider)
      }
    } else {
      throw new MappingError(
        `Invalid role for the final message: '${lastMessageRole}'. Expected 'user' or 'tool'.`,
        this.provider
      )
    }

    // Add the last message to contents
    contents.push({ role: lastMessageRole, parts: lastMsgParts })

    const googleTools: Tool[] | undefined = params.tools?.map(tool => {
      if (tool.type !== 'function') {
        throw new MappingError(`Only 'function' tools are currently supported for Google.`, this.provider)
      }
      const schema = tool.function.parameters
      // Clean the schema by removing/normalizing properties that cause issues with Google API
      const cleanedSchema = this.cleanSchemaForGoogle(schema)
      // Validate after cleaning so JSON Schema can be converted into Google's FunctionDeclarationSchema
      if (!this.isFunctionDeclarationSchema(cleanedSchema)) {
        throw new MappingError(
          `Invalid parameters schema for tool ${tool.function.name}. Expected FunctionDeclarationSchema.`,
          this.provider
        )
      }

      return {
        functionDeclarations: [
          {
            name: tool.function.name,
            description: tool.function.description,
            parameters: cleanedSchema
          }
        ]
      }
    })

    let finalTools = googleTools
    if (params.grounding?.enabled) {
      const searchTool: Tool = { googleSearchRetrieval: {} }
      if (params.grounding.source && params.grounding.source !== 'web') {
        console.warn(
          `Only 'web' grounding source currently mapped for Google Search Retrieval. Ignoring source: ${params.grounding.source}`
        )
      }
      finalTools = finalTools ? [...finalTools, searchTool] : [searchTool]
    }

    let responseMimeType: string | undefined
    let responseJsonSchema: any | undefined
    if (params.responseFormat?.type === 'json_object') {
      responseMimeType = 'application/json'
      if (params.responseFormat.schema) {
        responseJsonSchema = this.cleanSchemaForGoogle(params.responseFormat.schema)
      }
    } else if (params.responseFormat?.type === 'json_schema') {
      responseMimeType = 'application/json'
      responseJsonSchema = this.cleanSchemaForGoogle(params.responseFormat.json_schema.schema)
    }

    // Use common utility for base parameters
    const baseMappedParams = mapBaseParams(params)

    // Build GenerateContentConfig
    const config: GenerateContentConfig = {
      ...(params.extraParams ?? {}),
      maxOutputTokens: baseMappedParams.maxTokens,
      temperature: baseMappedParams.temperature,
      topP: baseMappedParams.topP,
      stopSequences: baseMappedParams.stopSequences,
      responseMimeType: responseMimeType,
      responseJsonSchema: responseJsonSchema,
      tools: finalTools,
      systemInstruction: systemInstruction,
      // Safety settings: use custom settings if provided, otherwise default to BLOCK_MEDIUM_AND_ABOVE
      // Default thresholds provide balanced protection without being overly restrictive
      safetySettings: params.providerOptions?.googleSafetySettings
        ? params.providerOptions.googleSafetySettings.map(s => ({
            category: s.category as HarmCategory,
            threshold: s.threshold as HarmBlockThreshold
          }))
        : [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }
          ]
    }

    // Return GenerateContentParameters (new SDK structure)
    if (!params.model) {
      throw new MappingError('Model parameter is required for Google provider', this.provider)
    }

    return {
      model: params.model,
      contents: contents,
      config: config
    }
  }

  // --- Result Mapping ---

  private mapToolCallsFromGoogle(functionCallParts: Part[] | undefined): RosettaToolCallRequest[] | undefined {
    if (!functionCallParts || functionCallParts.length === 0) return undefined
    return functionCallParts
      .filter(part => part.functionCall?.name)
      .map((part, index) => {
        const call = part.functionCall!
        const toolCall: RosettaToolCallRequest = {
          id: call.id || `google_func_${call.name}_${index}_${Date.now()}`,
          type: 'function',
          function: { name: call.name!, arguments: JSON.stringify(call.args ?? {}) }
        }
        // Preserve thought signature for echo-back in subsequent requests (required by Gemini 3+)
        if (part.thoughtSignature) {
          toolCall.providerMetadata = { thoughtSignature: part.thoughtSignature }
        }
        return toolCall
      })
  }

  private mapCitationsFromGoogle(metadata: CitationMetadata | undefined): Citation[] | undefined {
    if (!metadata?.citations || metadata.citations.length === 0) return undefined
    return metadata.citations.map((s: any, index: number) => ({
      sourceId: s.uri ?? `google_cite_idx_${index}`,
      startIndex: s.startIndex,
      endIndex: s.endIndex,
      text: undefined
    }))
  }

  mapFromProviderResponse(response: GenerateContentResponse | undefined, model: string): GenerateResult {
    const promptFeedbackReason = safeGet<string>(response, 'promptFeedback', 'blockReason')
    const promptFeedbackSafetyRatings = safeGet<any[]>(response, 'promptFeedback', 'safetyRatings')

    if (promptFeedbackReason) {
      const fr =
        promptFeedbackReason === 'SAFETY'
          ? 'content_filter'
          : promptFeedbackReason === 'OTHER'
          ? 'error'
          : promptFeedbackReason.toLowerCase()
      console.warn(`Google prompt blocked. Reason: ${fr}. Ratings: ${JSON.stringify(promptFeedbackSafetyRatings)}`)
      return {
        content: null,
        toolCalls: undefined,
        finishReason: fr,
        usage: mapTokenUsage(response?.usageMetadata), // Use common utility
        citations: undefined,
        parsedContent: null,
        thinkingSteps: undefined,
        model: model,
        rawResponse: response
      }
    }

    const candidate = response?.candidates?.[0]
    const candidateFinishReason = candidate?.finishReason
    const candidateSafetyRatings = candidate?.safetyRatings

    if (!response || !candidate) {
      console.warn('Google response or candidate is missing despite no prompt block.')
      return {
        content: null,
        toolCalls: undefined,
        finishReason: 'error',
        usage: mapTokenUsage(response?.usageMetadata), // Use common utility
        citations: undefined,
        parsedContent: null,
        thinkingSteps: undefined,
        model: model,
        rawResponse: response
      }
    }

    let textContent: string | null = null
    let toolCalls: RosettaToolCallRequest[] | undefined
    let parsedJson: any = null
    let finishReason = candidateFinishReason ?? 'unknown'

    if (candidate.content?.parts) {
      const textParts = candidate.content.parts.filter((p): p is Part => p && 'text' in p)
      if (textParts.length > 0) {
        textContent = textParts.map(p => p.text).join('')
        const isJsonLike = textContent?.trim().startsWith('{') || textContent?.trim().startsWith('[')
        const isBlocked = candidateFinishReason === 'SAFETY' || !!promptFeedbackReason
        if (isJsonLike && !isBlocked) {
          try {
            parsedJson = JSON.parse(textContent)
          } catch (e) {
            console.warn('Failed to auto-parse potential JSON from Google:', e)
          }
        }
      }

      const functionCallParts = candidate.content.parts.filter((p): p is Part => p && 'functionCall' in p)
      if (functionCallParts.length > 0) {
        const mappedCalls = this.mapToolCallsFromGoogle(functionCallParts)
        if (mappedCalls && mappedCalls.length > 0) {
          toolCalls = mappedCalls
          if (!['SAFETY', 'RECITATION', 'MAX_TOKENS'].includes(candidateFinishReason ?? '')) {
            finishReason = 'tool_calls'
          }
        }
      }
    }

    if (candidateFinishReason === 'SAFETY') {
      finishReason = 'content_filter'
      console.warn(`Google candidate blocked due to safety. Ratings: ${JSON.stringify(candidateSafetyRatings)}`)
    } else if (candidateFinishReason === 'RECITATION') {
      finishReason = 'recitation_filter'
    } else if (candidateFinishReason === 'MAX_TOKENS') {
      finishReason = 'length'
    } else if (candidateFinishReason === 'STOP' && !toolCalls) {
      finishReason = 'stop'
    } else if (finishReason === 'unknown' && candidate.content) {
      finishReason = 'stop'
    }

    const citations: Citation[] | undefined = this.mapCitationsFromGoogle(candidate.citationMetadata)

    return {
      content: textContent,
      toolCalls: toolCalls,
      finishReason: finishReason,
      usage: mapTokenUsage(response.usageMetadata), // Use common utility
      citations: citations,
      parsedContent: parsedJson,
      thinkingSteps: undefined,
      model: model,
      rawResponse: response
    }
  }

  // --- Stream Mapping ---

  async *mapProviderStream(
    stream: AsyncIterable<GenerateContentResponse>,
    _originalParams: GenerateParams // Changed from originalTools
  ): AsyncIterable<StreamChunk> {
    // Note: originalParams is not directly used in this implementation,
    // but included for interface consistency. Tool validation happens in mapFromProviderResponse.
    let currentUsage: TokenUsage | undefined
    let finalFinishReason: string | null = null
    let aggregatedText = ''
    const aggregatedCitations: Citation[] = []
    const aggregatedToolCalls: RosettaToolCallRequest[] = []
    const model = '' // Model name isn't directly in the stream chunks
    let isPotentiallyJson = false
    let aggregatedResult: GenerateResult | null = null

    try {
      // Yield message_start immediately (model unknown initially)
      yield { type: 'message_start', data: { provider: this.provider, model: model } }

      for await (const chunk of stream) {
        try {
          // Aggregate usage metadata if present
          if (chunk.usageMetadata) {
            currentUsage = mapTokenUsage(chunk.usageMetadata) // Use common utility
            if (aggregatedResult) aggregatedResult.usage = currentUsage
          }

          // --- FIX: Add check for candidates ---
          if (!chunk.candidates || !Array.isArray(chunk.candidates) || chunk.candidates.length === 0) {
            // console.warn('Google stream chunk missing candidates, skipping.') // Optional warning
            continue // Skip this chunk
          }
          // --- End FIX ---

          const candidate = chunk.candidates[0] // Safe to access index 0 now
          if (!candidate) continue // Should be redundant due to above check, but keep for safety

          // Initialize aggregated result on first valid candidate
          if (!aggregatedResult) {
            aggregatedResult = {
              content: '',
              toolCalls: [],
              finishReason: null,
              usage: currentUsage,
              model: model, // Will be empty initially, maybe update later if possible?
              thinkingSteps: null,
              citations: [],
              parsedContent: null,
              rawResponse: undefined
            }
          }

          // --- FIX: Wrap part processing in try-catch ---
          try {
            // --- Text Delta ---
            const textDelta =
              safeGet<Part[]>(candidate, 'content', 'parts') // Use safeGet
                ?.filter((p): p is Part => p && 'text' in p)
                .map(p => p.text ?? '')
                .join('') ?? ''

            if (textDelta) {
              if (!isPotentiallyJson && aggregatedText === '' && textDelta.trim().match(/^[{[]/)) {
                isPotentiallyJson = true
              }
              aggregatedText += textDelta
              if (aggregatedResult) aggregatedResult.content = aggregatedText

              if (isPotentiallyJson) {
                let partialParsed = undefined
                try {
                  partialParsed = JSON.parse(aggregatedText)
                } catch {}
                yield {
                  type: 'json_delta',
                  data: { delta: textDelta, parsed: partialParsed, snapshot: aggregatedText }
                }
              } else {
                yield { type: 'content_delta', data: { delta: textDelta } }
              }
            }

            // --- Function Call Delta ---
            const functionCallParts = safeGet<Part[]>(candidate, 'content', 'parts')?.filter(
              (p): p is Part => p && 'functionCall' in p
            )

            if (functionCallParts && functionCallParts.length > 0) {
              const newCalls = this.mapToolCallsFromGoogle(functionCallParts)
              if (newCalls) {
                for (const tc of newCalls) {
                  // Check if this specific tool call ID has already been fully processed and added
                  if (!aggregatedToolCalls.some(existing => existing.id === tc.id)) {
                    const overallIndex = aggregatedToolCalls.length
                    aggregatedToolCalls.push(tc) // Add the fully formed call
                    if (aggregatedResult && aggregatedResult.toolCalls) aggregatedResult.toolCalls.push(tc)

                    // Yield start, delta (full args), and done for this new call
                    yield {
                      type: 'tool_call_start',
                      data: {
                        index: overallIndex,
                        toolCall: { id: tc.id, type: 'function', function: { name: tc.function.name } }
                      }
                    }
                    yield {
                      type: 'tool_call_delta',
                      data: { index: overallIndex, id: tc.id, functionArgumentChunk: tc.function.arguments }
                    }
                    yield { type: 'tool_call_done', data: { index: overallIndex, id: tc.id } }
                    finalFinishReason = 'tool_calls' // Set finish reason if a tool call occurred
                  }
                }
              }
            }

            // --- Citation Delta ---
            const citationsChunk = this.mapCitationsFromGoogle(candidate.citationMetadata)
            if (citationsChunk) {
              for (const citation of citationsChunk) {
                // Check if this citation has already been processed
                if (
                  !aggregatedCitations.some(
                    existing => existing.sourceId === citation.sourceId && existing.startIndex === citation.startIndex
                  )
                ) {
                  const overallIndex = aggregatedCitations.length
                  aggregatedCitations.push(citation)
                  if (aggregatedResult && aggregatedResult.citations) aggregatedResult.citations.push(citation)

                  // Yield delta and done for this new citation
                  yield { type: 'citation_delta', data: { index: overallIndex, citation } }
                  yield { type: 'citation_done', data: { index: overallIndex, citation } }
                }
              }
            }
          } catch (partProcessingError) {
            console.error(
              'Error processing Google stream chunk parts:',
              partProcessingError,
              'Chunk:',
              JSON.stringify(chunk)
            )
            // Decide whether to yield an error and stop, or just log and continue
            // For now, let's yield an error chunk and let the main catch handle termination
            throw new MappingError(
              'Failed to process Google stream chunk content',
              this.provider,
              'mapProviderStream part processing',
              partProcessingError
            )
          }
          // --- End FIX ---

          // --- Finish Reason ---
          const reason = candidate.finishReason

          if (reason === FinishReason.SAFETY) {
            const safetyMessage =
              "\n\nThis response was blocked by the AI provider's safety filters. Please modify your request and try again."
            yield { type: 'content_delta', data: { delta: safetyMessage } }
            aggregatedText += safetyMessage
            if (aggregatedResult) aggregatedResult.content = aggregatedText
          }

          // Only update finalFinishReason if it's not already 'tool_calls'
          if (reason && reason !== FinishReason.FINISH_REASON_UNSPECIFIED && finalFinishReason !== 'tool_calls') {
            if (reason === FinishReason.SAFETY) finalFinishReason = 'content_filter'
            else if (reason === FinishReason.RECITATION) finalFinishReason = 'recitation_filter'
            else if (reason === FinishReason.MAX_TOKENS) finalFinishReason = 'length'
            else if (reason === FinishReason.STOP) finalFinishReason = 'stop'
            else finalFinishReason = reason.toLowerCase() // Use lowercase for others
            if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason
          }
        } catch (streamProcessingError) {
          // Catch errors during the loop itself (including the part processing error re-thrown above)
          console.error('Error during Google stream processing loop:', streamProcessingError)
          const mappedError = this.wrapProviderError(streamProcessingError, this.provider)
          yield { type: 'error', data: { error: mappedError } }
          return // Stop the generator on error
        }
      } // End main stream loop (for await...)

      // --- End of Stream ---
      // --- FIX: Ensure finalFinishReason is set ---
      if (finalFinishReason === null) {
        if (aggregatedText || aggregatedToolCalls.length > 0 || aggregatedCitations.length > 0) {
          finalFinishReason = 'stop' // Assume normal stop if content/tools/citations were generated
        } else {
          // If nothing was generated and no specific reason given, it might be an error or unknown state
          console.warn('Google stream finished without generating content or providing a finish reason.')
          finalFinishReason = 'unknown'
        }
      }
      // --- End FIX ---

      if (aggregatedResult) aggregatedResult.finishReason = finalFinishReason

      if (isPotentiallyJson) {
        let finalParsedJson = null
        try {
          finalParsedJson = JSON.parse(aggregatedText)
        } catch {}
        yield { type: 'json_done', data: { parsed: finalParsedJson, snapshot: aggregatedText } }
        if (aggregatedResult) aggregatedResult.parsedContent = finalParsedJson
      }

      yield { type: 'message_stop', data: { finishReason: finalFinishReason } }

      if (currentUsage) {
        yield { type: 'final_usage', data: { usage: currentUsage } }
      }

      if (aggregatedResult) {
        if (!isPotentiallyJson && aggregatedResult.content === '') aggregatedResult.content = null
        if (aggregatedResult.toolCalls?.length === 0) aggregatedResult.toolCalls = undefined
        if (aggregatedResult.citations?.length === 0) aggregatedResult.citations = undefined
        yield { type: 'final_result', data: { result: aggregatedResult } }
      } else {
        console.warn('Google stream finished, but no aggregated result was built.')
      }
    } catch (error) {
      // Catch setup errors or errors re-thrown from the loop's catch block
      const mappedError = this.wrapProviderError(error, this.provider)
      yield { type: 'error', data: { error: mappedError } }
    }
  }

  // --- Embedding Mapping ---
  mapToEmbedParams(params: EmbedParams): EmbedContentParameters {
    // New SDK uses a unified embedContent method
    const contents = Array.isArray(params.input) ? params.input : [params.input]

    // Validate inputs
    for (const text of contents) {
      if (typeof text !== 'string' || text === '') {
        throw new MappingError('Input text for Google embedding cannot be empty.', this.provider)
      }
    }

    return {
      ...(params.extraParams ?? {}),
      model: `models/${params.model!}`,
      contents: contents
    }
  }

  mapFromEmbedResponse(response: EmbedContentResponse, modelId: string): EmbedResult {
    // The new SDK returns EmbedContentResponse with embeddings array
    return GoogleEmbedMapper.mapFromGoogleEmbedResponse(response, modelId)
  }

  // --- Audio Mapping ---
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranscribeParams(_params: TranscribeParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranscribeResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Transcription')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapToTranslateParams(_params: TranslateParams, _file: any): any {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapFromTranslateResponse(_response: any, _modelId: string): TranscriptionResult {
    throw new UnsupportedFeatureError(this.provider, 'Audio Translation')
  }

  // --- Error Handling ---
  wrapProviderError(error: unknown, provider: Provider): RosettaAIError {
    if (error instanceof RosettaAIError) {
      return error
    }
    // Check for specific Google error structures
    if (
      typeof error === 'object' &&
      error !== null &&
      'message' in error &&
      (error.constructor?.name?.includes('Google') ||
        (error as any).code ||
        (error as any).status ||
        (error as any).httpStatus ||
        (error as any).errorDetails)
    ) {
      const gError = error as any
      const statusCode =
        gError.httpStatus ?? gError.status ?? safeGet<number>(gError, 'response', 'status') ?? undefined
      const errorCode = safeGet<string>(gError, 'errorDetails', 0, 'reason') ?? gError.code ?? undefined
      const errorType = gError.name ?? safeGet<string>(gError, 'errorDetails', 0, 'type') ?? undefined
      const message = (error as Error).message || 'Unknown Google API Error'
      return new ProviderAPIError(message, provider, statusCode, errorCode, errorType, error)
    }
    if (error instanceof Error) {
      return new ProviderAPIError(error.message, provider, undefined, undefined, undefined, error)
    }
    return new ProviderAPIError(
      String(error ?? 'Unknown error occurred'),
      provider,
      undefined,
      undefined,
      undefined,
      error
    )
  }
}
