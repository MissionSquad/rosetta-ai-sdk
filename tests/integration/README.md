# Integration Tests

This directory contains integration tests for the RosettaAI SDK. These tests interact with external systems or components, such as:

1.  **Mock APIs:** Testing interactions with custom providers using a simulated backend (e.g., `custom-provider.integration.spec.ts` uses `tests/mocks/mock-custom-api.ts`).
2.  **Real Provider APIs:** Testing interactions with actual AI provider APIs (e.g., OpenAI, Anthropic) to verify mapping and core functionality like tool use (`builtin-tool-use.spec.ts`).

## Running Tests

- **Run all tests:** `yarn test` (or `npm test`)
- **Run only integration tests:** `yarn test:integration` (or `npm run test:integration`)

## Configuration

- **Mock API Tests:** These tests typically start and stop their own mock server (like the Express server in `mock-custom-api.ts`) and don't require external API keys.
- **Built-in Provider Tests:** Tests interacting with real APIs (`builtin-*.spec.ts`) require API keys.
  - Create a `.env.test` file in the **project root** (alongside your main `.env`).
  - Copy relevant API key variables from `.env.example` into `.env.test`.
  - The tests in `builtin-tool-use.spec.ts` will load keys from `.env.test` first, falling back to `.env`.
  - **Caution:** These tests make real API calls and may incur costs. They are skipped using `describe.skip` if the necessary keys are not found in the environment.

## Files

- `README.md`: This file.
- `custom-provider.integration.spec.ts`: Tests for registering and using custom providers with a mock API backend.
- `builtin-tool-use.spec.ts`: Tests tool use functionality specifically for the built-in providers (OpenAI, Anthropic, Google, Groq), verifying the Phase 1 tool refactor against actual (or mocked SDK) interactions.
