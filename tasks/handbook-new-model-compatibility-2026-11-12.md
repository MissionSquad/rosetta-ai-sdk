# Handbook: new Claude and GPT-6 model compatibility

Date: 2026-11-12. Owner: Jayson. Status: rosetta 1.24.0 and core 1.46.0 are published; API 1.92.0 is merged to `main`; the web PR (missionsquad-web#218) is open against `test`. Section 6 platform checks are not done.

## 1. Goal

Make the newest Claude and OpenAI models work end to end on Mission Squad (add model, chat, tool use
combined with reasoning), without changing behaviour for models that work today.

In scope:

| Provider | New models |
|---|---|
| Anthropic | `claude-fable-5-2`, `claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` |
| OpenAI | `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra` |

Request path: missionsquad-web -> missionsquad-api -> missionsquad-core -> rosetta-ai-sdk. Work lands in
that reverse order (rosetta first), because each layer pins the one below it.

## 2. Root causes (verified in code)

1. **Core hardcodes the temperature on the add-model test.** `missionsquad-core/src/index.ts:1485`
   sends `temperature: 0.1` regardless of model options.
2. **Rosetta's OpenAI gate is the literal string "gpt-5".** `isGPT5Model` in
   `src/core/mapping/openai.common.ts:66` is `modelName.includes('gpt-5')`. It decides both sampling
   stripping (`openai.mapper.ts:199`) and Responses API routing (`openai.responses.chat.ts:63`).
   `gpt-6*` matches neither, so it goes to Chat Completions with the temperature intact.
3. **Rosetta's Anthropic gates are exact-match sets.** `modelsWithoutSamplingParams` and
   `modelsWithAdaptiveThinking` in `src/core/mapping/anthropic.mapper.ts:79-97` stop at
   Opus 5 / Sonnet 5 / Fable 5.
4. **Anthropic model listing is a static file** (`src/core/listing/static-data/anthropic.models.ts`)
   that has none of the five new models.
5. **New Claude models check replayed thinking blocks against the history.** Fable 5.2 rejects a
   request with a 400 when a thinking block is replayed after the system prompt, tool set or an
   earlier message changed. Core has paths that do this (section 3.3).

## 3. Verified behaviour (live, 2026-11-12)

Source: `examples/probe-new-models.ts`, run against the provider APIs directly. Re-run it before
changing any gate. "400" means the provider rejected the request.

### 3.1 Anthropic

| Model | `temperature` / `top_p` | Thinking `budget_tokens` | Adaptive thinking | Forced tool choice (`any`) | Runs history check |
|---|---|---|---|---|---|
| `claude-haiku-4-5` | ok | ok | 400 | ok | no |
| `claude-sonnet-4-5` | ok | ok | 400 | ok | no |
| `claude-opus-4-5` | ok | ok | 400 | ok | no |
| `claude-sonnet-4-6` | ok | ok | ok | ok | no |
| `claude-opus-4-6` | ok | ok | ok | ok | no |
| `claude-opus-4-7` | 400 | 400 | ok | ok | no |
| `claude-opus-4-8` | 400 | 400 | ok | ok | no |
| `claude-sonnet-5` | 400 | 400 | ok | ok | no |
| `claude-opus-5` | 400 | 400 | ok | ok | no |
| `claude-fable-5` | 400 | 400 | ok | ok | no |
| `claude-opus-5-5` | 400 | 400 | ok | 400 | yes (flagged, not enforced on this account) |
| `claude-sonnet-5-5` | 400 | 400 | ok | 400 | yes (flagged, not enforced on this account) |
| `claude-haiku-5-5` | 400 | 400 | ok | ok | yes (flagged, not enforced on this account) |
| `claude-fable-5-1` | 400 | 400 | ok | 400 | yes (flagged, not enforced on this account) |
| `claude-fable-5-2` | 400 | 400 | ok | 400 | yes (enforced: 400) |

Other observations:

- Multi-turn tool use with thinking blocks replayed unchanged works on every model above.
- Replaying with all thinking blocks removed works on every model above.
- `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` with beta header
  `thinking-binding-controls-2026-08-01` turns the Fable 5.2 400 into a successful request that
  reports the dropped block in `input_transformations`. This held for all four edit shapes tested
  (earlier user message edited, tool added, system prompt changed, tools removed).
- The same `block_binding` request returns 400 on `claude-haiku-4-5` ("adaptive thinking is not
  supported on this model"). It must only be sent to models that run the check.
- `claude-fable-5-2` at `max_tokens: 10` returns only a thinking block (`stop_reason: max_tokens`).
- Per Anthropic's migration reference (not verifiable from this account): accounts created on or
  after 2026-08-31 are enforced on Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1 as well.
- Mythos models were not probed (no access). They follow the Fable rule of the same version.

### 3.2 OpenAI

| Model | Chat Completions + `temperature: 0.1` | Chat Completions + tools, effort omitted | Responses + tools, default effort | Accepted `reasoning.effort` (Responses) | Default effort | `temperature` on Responses |
|---|---|---|---|---|---|---|
| `gpt-4o`, `gpt-4.1` | ok | ok | ok | none (parameter rejected) | n/a | ok |
| `gpt-5`, `gpt-5-mini` | 400 | ok | ok | minimal, low, medium, high | medium | 400 |
| `gpt-5.1` | ok | ok | ok | none, low, medium, high | none | ok at default effort |
| `gpt-5.2`, `gpt-5.4` | ok | ok | ok | none, low, medium, high, xhigh | none | ok at default effort |
| `gpt-5.6-sol`, `-terra`, `-luna` | 400 | 400 | ok | none, low, medium, high, xhigh, max | medium | only with effort `none` |
| `gpt-6-sol`, `gpt-6-luna`, `gpt-6.1-terra` | 400 | 400 | ok | none, low, medium, high, xhigh, max | medium | only with effort `none` |
| `gpt-6-astra`, `gpt-6.1-sol` | 400 | 400 | ok | low, medium, high, xhigh, max | medium | never (`none` is rejected) |

Other observations:

- The Chat Completions tool error on 5.6 and 6.x is: "Function tools with reasoning_effort are not
  supported ... use /v1/responses or set reasoning_effort to 'none'". Astra and 6.1 Sol cannot use
  tools on Chat Completions at all, because they reject `none`.
- Responses with `max_output_tokens: 16` at default effort returns text on 5.1+ and all 6.x. On
  `gpt-5` and `gpt-5-mini` it returns `status: incomplete` with empty text; 2000 tokens returns text.
- `text.verbosity: low` and `reasoning.summary: auto` are accepted on every gpt-5* and gpt-6* model.
- `o4-mini` now returns 404 (deprecated by OpenAI). Not caused by this work.

### 3.3 Where Mission Squad edits history (affects Fable 5.2 today, the other four on newer accounts)

| Path | File | Effect |
|---|---|---|
| Tool-failure fallback drops all tools | `missionsquad-core/src/utils/promptHandler.ts` `toolFailureFinalizationParams` | 400, caught, user sees the canned "tool kept failing" message |
| Tools rediscovered every loop iteration | `promptHandler.ts:1387` (`prepareRosettaTools` inside the loop) | 400 if an MCP server's tool list changes mid-turn |
| RAG and URL scraping rewrite only the latest user message | `missionsquad-core/src/index.ts:2187-2212` | 400 on the next turn only when an earlier tool scaffold is still replayed (recovery turns, interrupted turns) |

Completed PTC turns are safe: `collapseCompletedProgrammaticConversationMessages`
(`missionsquad-api/src/controllers/core.ts:5377`) drops the tool scaffold before history goes back
to the client. Non-PTC turns never replay thinking blocks.

## 4. Decisions

These are my recommendations. Change them here before implementation starts if you disagree.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Forced tool choice on models that reject it | Downgrade `required` / named-function tool choice to `auto` and `console.warn` | Core never sends forced tool choice; an API caller that does should degrade, not 400 |
| D2 | History check | Send `drop_block` plus the beta header on every model that runs the check | One rosetta change covers all three core paths in 3.3 and behaves the same on old and new Anthropic accounts. Cost: the model loses that reasoning on the affected request |
| D3 | Core's `?? 0.1` agent temperature default (`index.ts:1843`) | Leave it | Rosetta is the single place that strips sampling. Changing the default would alter behaviour for every existing model that accepts temperature |
| D4 | SDK upgrades (`@anthropic-ai/sdk` 0.100.1 -> 1.8.0, `openai` 6.27.0 -> 7.56.0) | Do not upgrade in this work | Both are major bumps. Anthropic's reference says newer SDK releases send API version `2026-10-21`, on which sampling parameters 400 on every model; that would break temperature on Haiku 4.5 / Sonnet 4.6 / Opus 4.6. Track as separate work |
| D5 | Existing static-list entries (4.6 context sizes, `:1m` variants, Sonnet 4.5 retiring 2026-11-30) | Leave untouched | Out of scope; no evidence they are broken |
| D6 | `gpt-5.6` table entry | Add `max` and `only_with_reasoning_none`; keep `defaultReasoningEffort: 'none'` | Live default is `medium`, but the field is rosetta's fallback and changing it alters the Chat Completions path for existing callers |

## 5. Implementation

Checkboxes are the tracking surface. Do not tick a box until its verification passed.

### Phase R: rosetta-ai-sdk (release 1.24.0)

Branch from `main`. The working tree currently has unrelated uncommitted changes (`LICENSE`,
`examples/anthropic-ptc-debug.ts`, `examples/anthropic-ptc-single-tool-repro.ts`); leave them out.

- [x] **R1. `ReasoningEffort` gains `max`.** `src/types/params.types.ts:5`:
  `'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'`. Run `yarn build`; fix any
  exhaustive switch the compiler flags in other mappers by treating `max` the way that mapper
  treats `xhigh`.

- [x] **R2. Replace the "gpt-5" string gate.** In `src/core/mapping/openai.common.ts` add:

  ```ts
  /**
   * True for OpenAI reasoning chat families that Rosetta routes through the Responses API and
   * models in gpt5.support.ts: gpt-5.x through gpt-9.x. A single-digit major is required so
   * Azure-style ids such as `gpt-35-turbo` do not match.
   */
  export function isOpenAIResponsesChatModel(modelName: string): boolean {
    return /gpt-[5-9](?!\d)/.test(modelName)
  }
  ```

  Keep `isGPT5Model` as a one-line deprecated wrapper that returns
  `isOpenAIResponsesChatModel(modelName)`. Replace its two call sites
  (`openai.mapper.ts:199`, `openai.responses.chat.ts:63`) with the new function. Export
  `isOpenAIResponsesChatModel` from `src/index.ts` next to `getGpt5Support` (core needs it in C2).

- [x] **R3. Capability table.** In `src/core/mapping/gpt5.support.ts`, add to
  `GPT5_CHAT_COMPLETIONS_SUPPORT`. Lookup is exact key, then longest key that the id starts with
  followed by `-`, so each id below needs its own key:

  | Key | `allowedReasoningEfforts` | `defaultReasoningEffort` | `supportsSampling` | `supportsFunctionToolsWithReasoning` | `functionToolsRequireExplicitNone` |
  |---|---|---|---|---|---|
  | `gpt-6-sol` | none, low, medium, high, xhigh, max | medium | `only_with_reasoning_none` | false | true |
  | `gpt-6-luna` | same | medium | `only_with_reasoning_none` | false | true |
  | `gpt-6.1-terra` | same | medium | `only_with_reasoning_none` | false | true |
  | `gpt-6-astra` | low, medium, high, xhigh, max | medium | `never` | false | omit |
  | `gpt-6.1-sol` | low, medium, high, xhigh, max | medium | `never` | false | omit |

  All five: `chatCompletionsSupported: true`, `supportsVerbosity: true`.
  Existing `gpt-5.6` entry (D6): add `'max'` to `allowedReasoningEfforts` and change
  `supportsSampling` to `'only_with_reasoning_none'`. Change nothing else in the table.
  Update the file's header comment: it now covers GPT-5 and GPT-6.

- [x] **R4. Chat Completions guard for models that cannot take tools there.** In
  `openai.mapper.ts`, inside the `if (support)` block, before the reasoning-effort resolution: if
  `hasFunctionTools && support.supportsFunctionToolsWithReasoning === false &&
  !support.allowedReasoningEfforts.includes('none')`, throw
  `new UnsupportedFeatureError(this.provider, \`function tools on Chat Completions for model '${params.model!}' (use the Responses API)\`)`.
  This path is only reachable with `providerOptions.openaiPreferChatCompletions`.
  Also on this path, when the resolved effort is `'max'`, send `'xhigh'`: Chat Completions listed
  `low, medium, high, xhigh` as the supported values on Astra and 6.1 Sol. Probe `gpt-6-sol` Chat
  Completions with `reasoning_effort: 'max'` first and only clamp where the API rejects it.

- [x] **R5. Anthropic capability predicates.** New file
  `src/core/mapping/anthropic.capabilities.ts`. One parser, four predicates. Parser: strip a
  trailing `:1m`; match `^claude-(sonnet|opus|haiku|fable|mythos)-(.+)$`; split the rest on `-`;
  first part is the major; the second part is the minor unless it is an 8-digit date (then minor
  is 0). Unparseable ids return `false` from every predicate (today's behaviour for unknown ids).
  "At least X.Y" means major greater, or major equal and minor greater or equal.

  | Function | True when |
  |---|---|
  | `anthropicRejectsSamplingParams(model)` | fable or mythos (any version); opus at least 4.7; sonnet at least 5.0; haiku at least 5.0 |
  | `anthropicRequiresAdaptiveThinking(model)` | same rule as above |
  | `anthropicRejectsForcedToolChoice(model)` | fable or mythos at least 5.1; opus at least 5.5; sonnet at least 5.5 |
  | `anthropicRunsThinkingHistoryCheck(model)` | fable or mythos at least 5.1; opus at least 5.5; sonnet at least 5.5; haiku at least 5.5 |

  Also export `getRequiredAnthropicBetas(model): string[]`, returning
  `['thinking-binding-controls-2026-08-01']` when `anthropicRunsThinkingHistoryCheck(model)` and
  `[]` otherwise. Every row of table 3.1 must be a test case (R9).

- [x] **R6. Use the predicates in `anthropic.mapper.ts`.**
  1. Delete both static sets and `shouldStripSamplingParams` (lines 78-101). Line 685 uses
     `anthropicRejectsSamplingParams(normalizedModel)`; line 639 uses
     `anthropicRequiresAdaptiveThinking(normalizedModel)`.
  2. Tool choice (lines 613-628), D1: when `anthropicRejectsForcedToolChoice(normalizedModel)` and
     the resolved choice would be `{ type: 'any' }` or `{ type: 'tool', ... }`, set
     `{ type: 'auto' }` and `console.warn` naming the model. `normalizedModel` is currently
     declared after this block (line 630); move the declaration above it.
  3. History check, D2: when `anthropicRunsThinkingHistoryCheck(normalizedModel)`, the request's
     `thinking` must be
     `{ type: 'adaptive', block_binding: { prefix_mismatch_behavior: 'drop_block' } }`, plus
     `display: 'summarized'` when thinking was requested. Omitting `thinking` and sending
     `{ type: 'adaptive' }` are equivalent on these models, so this does not turn thinking on.
     SDK 0.100.1 does not type `block_binding` (verified: no match in the SDK's `resources`).
     Do not cast. Declare a local type that intersects the SDK's adaptive thinking config type with
     `{ block_binding: { prefix_mismatch_behavior: 'error' | 'drop_block' } }`, build the value in a
     variable, and assign the variable (a non-literal subtype is assignable without an excess
     property error). Confirm the adaptive config type's exact exported name in
     `node_modules/@anthropic-ai/sdk/resources/messages/messages.d.ts` before writing it.
  4. Log drops: in `mapFromProviderResponse` and at `message_start` in the stream mapper, if the
     response object has a non-empty `input_transformations` array (narrow with `in` and
     `Array.isArray`), `console.warn` the entry types and reasons. No behaviour change.

- [x] **R7. Send the beta header.** `src/core/rosetta-ai.ts` has two Anthropic dispatch sites
  (generate at lines 550-565, stream at 885-900). In both, after computing
  `betas = betasFromOptions ?? betasFromSuffix`, append `getRequiredAnthropicBetas(model)` without
  duplicates. The existing "use `client.beta.messages` when betas is non-empty" branch then
  applies. Pass the model with any `:1m` suffix; the helper strips it.

- [x] **R8. Static model list.** Add five entries at the top of the LATEST block in
  `src/core/listing/static-data/anthropic.models.ts`, same shape as `claude-fable-5`. Omit
  `training_data_cutoff` (optional in `RosettaModel`; not verified for these models).

  | `id` | `context_window` | `max_completion_tokens` | `cost_input_mtok` | `cost_output_mtok` | `comparative_latency` |
  |---|---|---|---|---|---|
  | `claude-fable-5-2` | 1000000 | 128000 | 8.0 | 40.0 | Slower |
  | `claude-fable-5-1` | 1000000 | 128000 | 10.0 | 50.0 | Slower |
  | `claude-opus-5-5` | 1000000 | 128000 | 4.0 | 20.0 | Moderate |
  | `claude-sonnet-5-5` | 1000000 | 128000 | 2.0 | 10.0 | Fast |
  | `claude-haiku-5-5` | 1000000 | 128000 | 0.1 | 0.5 | Fastest |

  All five: `vision: true`, `extended_thinking: true`, `multilingual: true`. Haiku 5.5 comment:
  prices are for prompts up to 100K tokens ($0.50 / $2.50 beyond). Source: Anthropic model table
  cached 2026-11-03. Do not add `:1m` variants; 1M is the default on these models.

- [x] **R9. Tests** (`tests/unit/core/...`, jest):
  - New `mapping/anthropic.capabilities.spec.ts`: every row of table 3.1 for all four predicates,
    plus `:1m` suffix, dated ids (`claude-opus-4-5-20251101`, `claude-sonnet-4-20250514`), an
    unknown id, and a future id (`claude-opus-6`).
  - `mapping/anthropic.mapper.spec.ts`: temperature stripped for `claude-opus-5-5` and kept for
    `claude-sonnet-4-6`; adaptive thinking for `claude-haiku-5-5`, budget thinking for
    `claude-haiku-4-5`; `block_binding` present for `claude-fable-5-2` and absent for
    `claude-opus-5` and `claude-haiku-4-5`; `toolChoice: 'required'` maps to `auto` on
    `claude-opus-5-5` and `any` on `claude-opus-5`.
  - `mapping/openai.common.spec.ts`: `isOpenAIResponsesChatModel` true for `gpt-5`, `gpt-5.6-sol`,
    `gpt-6-sol`, `gpt-6.1-terra`; false for `gpt-4o`, `gpt-4.1`, `gpt-35-turbo`, `o3`.
  - `mapping/gpt5.support.spec.ts`: the five new keys and the `gpt-5.6` change.
  - `mapping/openai.responses.chat.spec.ts`: `gpt-6-sol` with `temperature: 0.1` and no effort has
    no `temperature`; with `reasoningEffort: 'none'` it has it; `gpt-6-astra` with
    `reasoningEffort: 'none'` sends `medium`; `reasoningEffort: 'max'` passes through.
  - `mapping/openai.mapper.spec.ts`: the R4 guard.
  - `listing/static-data/anthropic.models.spec.ts`: the five ids exist with 1M / 128K.

- [x] **R10. End-to-end example.** New `examples/verify-model-matrix.ts` that goes through
  `RosettaAI` (unlike the probe). For each model in section 6: (a) `generate` with
  `temperature: 0.1`, `maxTokens: 2000`; (b) a two-iteration tool loop with one zod-typed tool,
  `thinking: true`, replaying `providerState` from the first result; (c) the same loop through
  `stream`. Print pass/fail per model per step. Follow `examples/tool-use.ts` for the tool
  definition shape.

- [x] **R11. Release.** Commit `examples/probe-new-models.ts` and this handbook. Bump
  `package.json` to `1.24.0`. `yarn build`, `yarn lint`, `yarn test` must all pass. Publishing
  goes through `.github/workflows/publish-npm.yaml`; Jayson triggers it.

### Phase C: missionsquad-core (after rosetta 1.24.0 is published)

- [x] **C1. Add-model test.** `src/index.ts:1480-1488`: remove the `temperature: 0.1` line.
  Replace `maxTokens: 10` with
  `providerKey === Provider.OpenAI || providerKey === Provider.Anthropic ? 2000 : 10`.
  Reason: Fable 5.2 returns only thinking at 10 tokens, and `gpt-5` / `gpt-5-mini` return empty
  text at 16; 2000 is the value verified to return text on both. Other providers keep 10 so local
  and custom endpoints see no change. Verify `Provider` is already imported in this file.
- [x] **C2. Capability gate.** `src/utils/singleResponseCapabilities.ts:56`: replace
  `model.includes('gpt-5')` with `isOpenAIResponsesChatModel(model)` imported from
  `@missionsquad/rosetta-ai`.
- [x] **C3. Fable/Mythos 5.2 structured output.** Same file, line 51: the `nativeSchema` regex
  accepts `(?:fable|mythos)-5(?:-1)?`. Before widening it to `(?:-[12])?`, add an
  `output_config.format` json_schema probe to `examples/probe-new-models.ts` and confirm
  `claude-fable-5-2` accepts it. Widen only if it does.
- [x] **C4. No change needed (confirm only):** `toolFailureFinalizationParams`, the per-iteration
  `prepareRosettaTools`, the RAG/scrape rewrite, `src/utils/ptc.ts`, and the `?? 0.1` default (D3).
  R6/R7 make the first three succeed on checking models.
- [x] **C5. Release.** `package.json`: `@missionsquad/rosetta-ai` to `1.24.0`, version to
  `1.46.0`. `yarn build` and `yarn lint` must pass (core has no unit test script). Publish via
  `.github/workflows/publish.yaml`; Jayson triggers it.

### Phase A: missionsquad-api (after core 1.46.0 is published)

- [x] **A1. Pins.** `package.json`: `@missionsquad/missionsquad-core` `1.45.4` -> `1.46.0`,
  `@missionsquad/rosetta-ai` `1.22.0` -> `1.24.0`. The API is two rosetta versions behind; read
  the 1.23.0 commit (`8fe46d1`) for anything the API consumes.
- [x] **A2. Effort enum.** Add `'max'` in `src/types/evals.ts:13` and
  `src/utils/evalDefinition.ts:18`.
- [x] **A3. Pricing.** `src/config/modelPricingCatalog.ts`, next to the `claude-fable-5-1` row
  (line 193): add `['claude-fable-5-2', 'Claude Fable 5.2', [8, 0.2, 40]]` and
  `['claude-mythos-5-2', 'Claude Mythos 5.2', [8, 0.2, 40]]`. GPT-6 and the other new Claude
  models are already present.
- [x] **A4.** `yarn build` and `yarn test` pass. The working tree has unrelated uncommitted
  changes (`package.json`, `tasks/*`); do not sweep them into the commit.

### Phase W: missionsquad-web (branch `claude/claude-chatgpt-model-support-23778b`)

- [x] **W1. Sampling mirror.** `src/utils/models.ts`: delete `MODELS_WITHOUT_SAMPLING_PARAMS` and
  make `supportsSamplingParams` apply the R5 `anthropicRejectsSamplingParams` rule (reuse the
  family/version parsing already in `defaultsToPTC`; extract it into one helper). Update the
  KEEP IN SYNC comment to point at `anthropic.capabilities.ts`. Extend `src/utils/models.spec.ts`
  with the table 3.1 rows.
- [x] **W2. Effort `max`.** Add `'max'` in `src/types/models.ts:3`, `src/types/evals.ts:12`,
  `src/utils/evalEditor.ts:16`, the three value checks in
  `src/components/evals/EvalParameterEditor.vue` (lines 50, 67, 87) and the items list in
  `src/components/sub/ModelOptions.vue:142`.
- [ ] **W3.** `yarn lint`, `yarn test`, `yarn build` pass. Bump the version per repo convention.

## 6. Acceptance

All of the following, after every phase is deployed to a local stack (API on 8081, web on 8092).

**Rosetta matrix** (`examples/verify-model-matrix.ts`, every step passes):

- Anthropic: `claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-sonnet-4-6`,
  `claude-opus-4-6`, `claude-opus-4-7`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-opus-5`,
  `claude-fable-5`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`,
  `claude-fable-5-1`, `claude-fable-5-2`
- OpenAI: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.2`, `gpt-5.4`,
  `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`,
  `gpt-6.1-sol`, `gpt-6.1-terra`

**Platform checks:**

- [ ] Each of the ten new models can be added in the Models UI with the connection test on.
- [ ] One existing model per family still adds and chats: `claude-haiku-4-5`,
  `claude-sonnet-4-6`, `claude-opus-5`, `gpt-4o`, `gpt-5.4`, `gpt-5.6-sol`.
- [ ] Public chat, agent with at least one MCP tool, PTC on: `claude-opus-5-5`,
  `claude-sonnet-5-5`, `claude-fable-5-2` each complete a turn that calls the tool, then answer a
  follow-up turn.
- [ ] Public chat, same agent, non-PTC: `claude-haiku-5-5`.
- [ ] Public chat, agent with a tool and thinking on: `gpt-6-sol`, `gpt-6-astra`, `gpt-6.1-sol`
  each call the tool and show reasoning.
- [ ] Agent with RAG (embeddings) on `claude-fable-5-2`: two consecutive turns succeed.
- [ ] Force the tool-failure fallback on `claude-fable-5-2` (a tool that always errors): the
  model's own summary is returned, not the canned message, and rosetta logs an
  `input_transformations` warning.

## 7. Not in scope

- Upgrading the Anthropic or OpenAI SDK majors (D4).
- Mapping `reasoningEffort` to Anthropic's `output_config.effort` (rosetta does not send it today).
- Refusal fallbacks for Fable / Opus 5.5 (`fallbacks` parameter).
- Removing retired models from the static list, or correcting 4.6 context sizes (D5).
- Hiding the temperature control in the web UI for OpenAI models that ignore it.
- `o4-mini` deprecation.
- Fable 5.x requires 30-day data retention on the Anthropic organization. That is an account
  setting, not a code change; a 400 naming data retention means the org needs it enabled.

## 8. Change log

| Date | Change |
|---|---|
| 2026-11-12 | Created from the investigation and live probe results |
| 2026-11-12 | Phase R1-R10 done on branch `feat/new-model-compatibility`. `yarn build` and `yarn test` pass (1282 tests). `examples/verify-model-matrix.ts` passes all four steps on all 15 Claude and 15 GPT models |
| 2026-11-12 | R4: probed `reasoning_effort: 'max'` on Chat Completions; rejected on gpt-5.6 and every gpt-6 model, so `max` is clamped to `xhigh` on that path for all models |
| 2026-11-12 | R1 deviation: openai@6.27.0 types `reasoning.effort` without `max`, so `toOpenAIReasoningEffort` in `openai.responses.chat.ts` holds one documented type assertion. Remove it when the OpenAI SDK is upgraded |
| 2026-11-12 | R9: mapper-level cases live in one new file, `tests/unit/core/mapping/new-model-compatibility.spec.ts`, instead of being spread across the existing specs |
| 2026-11-12 | `yarn lint` fails on `main` too (ESLint crawls `examples/node_modules`, and the old parser cannot parse several existing files). Changed files have the same error count as `main`; new files lint clean |
| 2026-11-12 | The matrix example runs with `examples/node_modules/.bin/ts-node`; the repo root has no `ts-node`, so the `example:*` package scripts only work with a global install |
| 2026-11-12 | Released: rosetta-ai-sdk#82 merged (1.24.0 published), missionsquad-core#205 merged (1.46.0 published), missionsquad-api#251 merged (1.92.0). missionsquad-web#218 open against `test` |
| 2026-11-12 | C3: `claude-fable-5-2` accepts `output_config.format` json_schema (probed live), so core's native-schema regex was widened |
| 2026-11-12 | Correction: core does have unit tests (`yarn test:all`, 750 tests). The add-model test was updated for the new request shape after it failed in CI |
| 2026-11-12 | API: the branch was first cut from a stale local `main` and conflicted on `package.json` / `yarn.lock`; resolved by merging `origin/main` (1.91.0). The API version was bumped to 1.92.0 |
| 2026-11-12 | W3 is still open: web `yarn test` passes; `yarn lint` has 3 errors in files this work does not touch; `yarn build` and a browser check have not been run |
