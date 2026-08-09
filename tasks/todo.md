# Comms Agent V1 Rosetta checklist

## Research and gates

- [x] Preserve the dirty primary checkout by creating/using the isolated feature worktree.
- [x] Reconfirm Rosetta baseline `411b67c695a4852d26df53ada35eed7db6d97aec` and public-surface drift.
- [x] Reconfirm Comms Agent spec commit `29650e1` and reread the six updated sources in order.
- [x] Verify installed `openai@6.27.0` Responses signatures directly from its declarations.
- [x] Verify current official OpenAI computer-use documentation.
- [x] Confirm ADR-0043 leaves no unresolved implementation blocker.
- [x] Record the research and exact implementation constraints.

## Implementation

- [x] Add and publicly export strict canonical computer-use Zod schemas and inferred TypeScript types.
- [x] Add `computer_use` to custom-provider capabilities.
- [x] Add `createComputerActionTool()` with exact name/description, strict shared JSON schema, and mandatory Zod validation.
- [x] Add provider coordinate/key normalization with closed aliases, exact formulas, and typed failures.
- [x] Add GA OpenAI Responses computer tool, input/output, identity, action, safety, screenshot, batch, and continuation mappings.
- [x] Preserve strict no-continuation behavior for invalid native computer calls.
- [x] Make forced named tool use work through `OpenAICompatibleMapper`.
- [x] Make non-streaming strict JSON response format work through `OpenAICompatibleMapper` using the same schema.

## Tests

- [x] Add exhaustive valid/invalid canonical schema tests from the contract.
- [x] Add exhaustive coordinate, key, native-action, modifier, batch, identity, safety, screenshot, and continuation mapping tests.
- [x] Add exact normative scroll and screenshot round-trip fixture tests.
- [x] Add forced-tool mandatory-validation and same-schema strict-JSON fallback tests.
- [x] Preserve and update existing Responses mapper coverage.

## Verification and delivery

- [x] Run formatter on scoped files.
- [x] Run lint and distinguish repository-baseline failures from scoped results.
- [x] Run explicit TypeScript typecheck/build.
- [x] Run the complete unit suite.
- [x] Run the complete repository test suite, including integration tests.
- [x] Review the final diff for scope, public exports, declaration output, and accidental changes.
- [x] Add this file's review/results section and complete the checklist audit.
- [x] Commit the scoped changes with specification citations.
- [x] Push `agent/comms-agent-integration`.
- [x] Open PR #70 to `main` with specification citations and verify merge readiness.
- [ ] After merge/publish, record the exact published package version without starting downstream work.

## Review/results

- PR #70 follow-up reviewed every conversation comment, review submission, and inline thread. The
  suppressed Copilot finding was excluded. Confirmed fixes align forced web-search choice typing with
  the installed SDK, reject unknown runtime input/tool discriminators, and restore non-suppressed
  streaming failure/tool-validation regression coverage; intentional contract behavior and nitpicks
  were left unchanged.
- All five cross-repo work items are implemented against ADR-0043 and the canonical contract. Package version is set to `1.18.0`, the repository's established minor-version convention for a new public capability.
- Independent review findings were resolved: unsupported strict-schema `not`/`uniqueItems` keywords were removed, strict JSON results now use the same runtime Zod validator, Responses function strictness is caller-controlled, function item/call IDs remain distinct, and the declaration hash was corrected.
- `yarn build` and `yarn tsc --noEmit` pass.
- `yarn test` passes: 28 suites, 1,030 tests passed, 5 skipped.
- Escalated `yarn test:all` passes: 29 suites passed, 1 skipped; 1,037 tests passed, 8 skipped.
- Scoped ESLint reports no errors in every changed file its legacy parser can parse (one pre-existing `no-console` warning in `common.utils.ts`). The final complete `yarn lint --quiet` run remains blocked by 226 baseline errors across unrelated examples/source/tests, including parser failures because the pinned 2019 ESLint/Prettier stack cannot parse existing TypeScript syntax such as `override` and `import type`. The PR does not broaden scope into a repository-wide lint-toolchain migration.
- `npm pack --dry-run --cache /private/tmp/rosetta-npm-cache` passes for `@missionsquad/rosetta-ai@1.18.0`; the 180-file package includes the root declarations plus the new computer-use JavaScript, source maps, and declarations.
- `git diff --check` passes. The original dirty checkout remains untouched; all work is isolated on `agent/comms-agent-integration`.
- Commit `7a9c9e0` is pushed and PR #70 is open, non-draft, cleanly mergeable, and passed its required `build` check.
