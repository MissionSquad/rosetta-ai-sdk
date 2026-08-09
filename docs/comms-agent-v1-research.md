# Comms Agent V1 Rosetta research record

Date: 2026-08-08

## Inputs and prerequisite checks

- The sibling Comms Agent specification checkout is `missionsquad-marketing-employee`, on `bootstrap/ci` at `29650e19018f7ef953bd8a01262da1c14990b581` (`docs: close Rosetta computer-use mapping gaps`).
- The Rosetta feature worktree started at the required baseline `411b67c695a4852d26df53ada35eed7db6d97aec`. `git diff 411b67c...HEAD` was empty before implementation, so there is no post-baseline public-surface drift to reconcile.
- The governing files were reread in the required order: ADR-0043, the canonical contract, the Rosetta cross-repo brief, browser/computer-use spec, implementation prompt, and pinned recon.
- The active dependency graph is Yarn-based. `yarn.lock` resolves `openai@6.27.0`; the installed declaration file is `node_modules/openai/resources/responses/responses.d.mts` with SHA-256 `78b036165d131e925272aa390c6a55c6498e614e506218998260d014fedf73e9`.
- Current official OpenAI computer-use documentation was checked for the GA `{type:"computer"}` tool, `computer_call.actions[]`, `computer_call_output`, `call_id`, `previous_response_id`, screenshots, pending safety checks, and mouse modifier behavior. The installed declarations remain the source of truth for TypeScript signatures.

Conclusion: ADR-0043 resolves the earlier normalization, identity, safety-nullability, modifier, screenshot, batch, continuation, and fallback gaps. No new conflict requires a halt.

## Normative contract decisions

The implementation follows these exact requirements:

- All canonical serializable objects use strict Zod schemas and reject unknown keys (`computer-use-contract.md:1-3`).
- Canonical coordinates are finite normalized values. Pixel coordinates/deltas divide by the corresponding viewport axis minus one; 1,000-point values divide by 1,000; normalized values pass through. Invalid or oversized values reject and are never clamped (`computer-use-contract.md:118-122`; ADR-0043:17-18).
- Provider key values use the contract's closed 17-key allowlist and aliases, with uniqueness checked after normalization (`computer-use-contract.md:106-108`; ADR-0043:17).
- Native identity is preserved exactly: `call_id -> actionId`, output-item `id -> providerTraceId`, and response `id -> responseId` (`computer-use-contract.md:116`; ADR-0043:19).
- Missing or `undefined` safety `code` and `message` normalize to `null`; strings and explicit `null` are preserved (`computer-use-contract.md:144`; ADR-0043:20).
- V1 accepts only the GA computer tool and exactly one member in `actions[]`. Singular-only, mixed singular/batch, empty, and multi-action calls fail without execution or continuation (`computer-use-contract.md:124-126,170`; ADR-0043:21).
- Native screenshot maps to canonical `request_screenshot`; mouse actions carrying a `keys` property reject (`computer-use-contract.md:128-144`; ADR-0043:22-23).
- A caller may retry a rejected native shape once from the same fresh observation through forced `computer_action` or the same strict JSON schema. A repeated invalid result hands off to a human (`computer-use-contract.md:126`; `07-browser-and-computer-use.md:15-19,52`). Rosetta provides the validated tool/schema and mapping primitives; Mission Squad owns retry orchestration.
- OpenAI strict Structured Outputs supports nested `anyOf` but rejects composition keywords including `not`; its supported array subset also does not include `uniqueItems`. The provider-facing schema therefore carries the supported structural/range constraints, while the same mandatory Zod schema enforces zero-distance scroll and post-alias key uniqueness after parsing.

## Installed OpenAI 6.27.0 declaration evidence

The installed declaration file confirms:

- `Responses.create` accepts the declared Responses creation parameter variants.
- `ComputerTool` is the GA `{ type: "computer" }` tool.
- `ResponseComputerToolCall` contains required `id`, `call_id`, `pending_safety_checks`, `status`, and `type`, and exposes both optional legacy `action` and optional GA `actions`.
- `ResponseComputerToolCall.PendingSafetyCheck` permits optional/nullable `code` and `message`.
- `ResponseComputerToolCallOutput` is a response output item; the corresponding request-side `ComputerCallOutput` carries `call_id`, screenshot output, and optional acknowledged safety checks.
- `previous_response_id` is the continuation parameter.
- Raw `input_text` and `input_image` values are message content parts, not top-level `ResponseInputItem` values. Multimodal input must be wrapped in an input message before any top-level `computer_call_output` item is appended.
- The installed native mouse action declarations do not expose the current runtime `keys` extension. ADR-0043 explicitly requires a boundary-level raw property check and rejection rather than silently dropping it.

The current OpenAI guide documents optional mouse `keys`, while the installed 6.27.0 declarations omit it. This is not a conflict: ADR-0043 defines the compatible fail-closed behavior.

## Existing Rosetta surface reused

- Canonical exports continue through `src/types/index.ts` and the package root.
- `RosettaImageData`, `RosettaContentPart`, `RosettaMessage`, `RosettaTool`, `GenerateParams.tools`, and named function `toolChoice` remain the forced-tool path; no parallel image/message/tool abstractions are introduced.
- `OpenAICompatibleMapper` already maps images, tools, named tool choice, and validates returned tool arguments with the supplied Zod schema. Its non-streaming request currently omits `responseFormat` even though streaming maps it; the V1 strict-JSON fallback therefore requires the scoped non-streaming parity fix.
- The public stateful Responses mapper remains separate from the internal stateless OpenAI Responses chat adapter.

## Test obligations

Tests must cover every canonical variant and boundary, strict unknown-key rejection, exact valid and invalid contract fixtures, all aliases and duplicate normalization, timestamp/hash/action-ID limits, pixel/1,000-point/normalized conversion, every native mapping and rejection, exact identity and safety normalization, truthful GA batch rejection, native request/continuation shapes, streaming completion mapping, mandatory Zod validation, forced named-tool mapping, and same-schema structured-JSON fallback.

## External source

- OpenAI, “Computer use”: https://developers.openai.com/api/docs/guides/tools-computer-use
- OpenAI, “Structured model outputs — Supported schemas”: https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas
