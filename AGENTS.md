# Repository Development Rules

## Scope

- Keep this repository source-only. Personal notes, interview material, audio,
  documents, model weights, generated benchmark output, and scratch data stay
  local and must never be committed.
- `origin` is `Casten-Wang/nano-vllm`; `upstream` is
  `GeeeekExplorer/nano-vllm`. Never push to `upstream`.
- The next model target is text-only inference for `Qwen3.5-35B-A3B`. Do not
  add multimodal support unless it is explicitly scoped and designed first.

## Development Workflow

1. Update local `main` from `origin/main`.
2. Create a focused branch named `feature/<topic>`, `fix/<topic>`,
   `perf/<topic>`, or `test/<topic>`.
3. Inspect relevant source, tests, configuration, and upstream changes before
   editing.
4. Keep each change focused. Separate unrelated refactors and generated data.
5. Run the repository policy check, syntax check, and relevant tests.
6. Commit with a Conventional Commit prefix: `feat:`, `fix:`, `perf:`,
   `test:`, `refactor:`, `build:`, or `chore:`.
7. Push the development branch to `origin` and merge it into this fork's
   `main` through a pull request. Never push feature work to `upstream`.
8. Create a semantic version tag only for a tested release.

## Verification

- Run `python3 scripts/check_repo_policy.py` before every commit.
- Run `python3 -m compileall -q nanovllm projects scripts tests`.
- Run focused tests while iterating and the full available test suite before a
  pull request.
- GPU changes must record hardware, software versions, model revision, input
  shapes, warmup, repetitions, raw samples, correctness, memory, and latency or
  throughput. Do not claim a speedup from a single run.
- Compare performance on the same commit and environment. Report regressions
  and unsupported cases explicitly.

## Git Safety

- Review `git status`, staged file names, and `git diff --cached --check`
  before committing.
- Do not rewrite shared history, force-push, delete remote branches, or create
  release tags without explicit approval.
- Never commit credentials, tokens, private data, local datasets, or model
  artifacts.

## Upstream Contribution Lane

- Product development for this fork and contributions to upstream are separate
  lanes. Never open an upstream PR directly from a product-development branch.
- Start an upstream contribution from `upstream/main` on an
  `upstream-fix/<topic>` branch. Reproduce the problem on unmodified upstream
  first.
- Prefer one behavioral problem, one to three files, and the smallest complete
  patch. Do not bundle fork-only CI, repository policy, benchmarks, broad
  refactors, quantization experiments, or generated artifacts.
- Search open issues and pull requests before implementation. Coordinate on an
  existing PR instead of submitting a duplicate.
- A strong upstream PR contains a minimal reproduction, root cause, focused
  fix, before/after evidence, and honest limitations. Tests are required in
  this fork even when upstream historically merged patches without them.
- Large model support and experimental optimizations remain fork features until
  an independently useful component can be proposed against current upstream.
