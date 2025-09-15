/**
 * RosettaAI SDK Entry Point
 *
 * Exports the main client class and core types/errors.
 */

export { RosettaAI } from './core/rosetta-ai'
export * from './types' // Export all types from the types module
export * from './errors' // Export all custom errors
export * from './core/mapping/openai-compatible.mapper' // Export OpenAICompatibleMapper
export { ElevenLabsMapper } from './core/mapping/elevenlabs.mapper'
