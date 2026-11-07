/**
 * RosettaAI SDK Entry Point
 *
 * Exports the main client class and core types/errors.
 */

export { RosettaAI } from './core/rosetta-ai'
export * from './types' // Export all types from the types module
export * from './errors' // Export all custom errors
export * from './core/mapping/openai-compatible.mapper' // Export OpenAICompatibleMapper
export * from './computer-use/create-computer-action-tool'
export { ElevenLabsMapper } from './core/mapping/elevenlabs.mapper'
import { Gpt5Support as OpenAIGpt5Support } from './core/mapping/gpt5.support'
export { getGpt5Support } from './core/mapping/gpt5.support'
export type Gpt5Support = OpenAIGpt5Support
