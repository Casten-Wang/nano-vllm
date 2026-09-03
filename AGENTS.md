# Repository Development Rules

## Scope

- Keep this repository source-only. Personal notes, interview material, audio,
  documents, model weights, generated benchmark output, and scratch data stay
  local and must never be committed.
- `origin` is `Casten-Wang/nano-vllm`; `upstream` is
  `GeeeekExplorer/nano-vllm`. Never push to `upstream`.
- The primary model target is text-only inference for `Qwen3.6-35B-A3B`. Do not
  add multimodal support unless it is explicitly scoped and designed first.
- Qwen3.6 deliberately retains the upstream architecture identifiers
  `Qwen3_5Moe*`, `qwen3_5_moe_text`, and this repository's internal `qwen35*`
  module/config names. They are checkpoint compatibility identifiers, not a
  statement that Qwen3.5 remains the product target; do not mechanically rename
  them without a backward-compatible migration design.
- Treat `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.6-35B-A3B-FP8` as the official
  BF16 and FP8 validation tracks. No official Qwen3.6 GPTQ-Int4 checkpoint is
  currently part of the validation matrix. Any Qwen3.5 GPTQ work is an optional,
  explicitly labeled compatibility experiment and must not be reported as
  Qwen3.6 validation.

## Development Workflow

1. Use `feature/qwen35-text-foundation` as the long-lived product-development
   branch. Push completed, verified development commits only to this branch.
2. Keep `main` at the latest stable, fully verified milestone. Do not update
   `main` for every small commit and do not wait for all long-term optimization
   work to finish; fast-forward it only when a coherent milestone is ready.
3. Inspect relevant source, tests, configuration, and upstream changes before
   editing.
4. Keep each change focused. Separate unrelated refactors and generated data.
5. Run the repository policy check, syntax check, and relevant tests.
6. Commit with a Conventional Commit prefix: `feat:`, `fix:`, `perf:`,
   `test:`, `refactor:`, `build:`, or `chore:`.
7. Push normal work only to `origin/feature/qwen35-text-foundation`. Promote a
   tested milestone to `origin/main` separately. Never push to `upstream`.
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

## External Design Research

- Before designing a runtime, kernel, quantization, parallelism, scheduling, or
  memory optimization, inspect current primary-source implementations and
  relevant issues or pull requests in established projects. Use this research
  matrix when the topic applies:
  - GPU serving engines: vLLM, SGLang, TensorRT-LLM, LMDeploy, Hugging Face
    Text Generation Inference, and DeepSpeed-MII.
  - Local and cross-platform runtimes: llama.cpp, MLC LLM, and MLX-LM.
  - Kernels and primitives: FlashAttention, FlashInfer, Triton, and PyTorch.
  - Distributed serving and PD/KV systems: NVIDIA Dynamo, Mooncake, DistServe,
    NVIDIA Triton Inference Server, Ray Serve, and KServe.
  Do not force every project into every investigation: select the primary
  implementations that actually own the relevant mechanism, but always check
  vLLM and SGLang for server-side GPU work and llama.cpp for quantization or
  constrained-memory work.
- Record the exact repository, commit or release, file or PR, hardware scope,
  algorithmic idea, benchmark methodology, and known limitations used as
  references. Do not rely on recollection, summaries, or marketing claims.
- Adapt ideas to nano-vllm's architecture and verify them independently. Do not
  copy incompatible code, remove attribution, or assume another project's
  benchmark transfers to this runtime or hardware.
- Prefer proven interfaces and invariants, but keep this repository small:
  adopt only the minimum mechanism needed for the measured problem rather than
  importing a large framework abstraction wholesale.
- Maintain an ignored local external-contribution registry under
  `docs/upstream_candidates/` covering every framework in the research matrix.
  Record each useful issue as `observed`, `reproduced`,
  `candidate`, `blocked`, `duplicate`, or `rejected`, together with repository,
  exact revision, issue/PR links, reproduction, root cause, proposed minimal
  diff, tests, hardware requirements, overlap audit, and current owner or
  assignee. A design reference is not automatically a PR candidate.
- Track useful contribution opportunities found in any external project as
  local review candidates, not only opportunities in nano-vllm. Recheck the
  target project's current main, contribution guide, open issues, open PRs,
  and maintainer feedback before implementation and again before requesting
  submission approval.
- Never publicly claim an issue, comment, push a contribution branch, or open,
  update, close, or reopen a pull request in any third-party repository without
  the user's explicit approval for that exact action. Local investigation,
  reproduction, implementation, testing, and review-packet preparation should
  continue autonomously and must not wait for routine design approval.
- The same external-PR approval rule applies to every third-party repository,
  not only GeeeekExplorer/nano-vllm. Keep at most one external PR open across
  all third-party projects unless the user explicitly approves an exception.

## Optimization Tracks

- Maintain three explicit optimization tracks: memory/VRAM efficiency,
  scheduling, and prefill/decode disaggregation. Every experiment must name
  its baseline, target workload, success metrics, correctness oracle, and
  fallback path before it can become a default.
- Treat these tracks as a coordinated system rather than isolated features.
  Memory pressure is an input to scheduling; scheduling determines prefill and
  decode placement; disaggregation adds transfer memory, backpressure, and
  failure states that must feed back into admission and routing decisions.
- Execute the tracks in evidence-driven stages: first establish memory-capacity
  accounting and allocation/lifetime instrumentation; next improve the
  colocated scheduler; then prototype prefill/decode disaggregation on top of
  stable scheduler and transfer interfaces. A measured bottleneck may justify
  changing this order, but the evidence must be recorded.
- Treat prefill/decode disaggregation as a first-class research track. Measure
  KV transfer, placement and load-balancing cost, communication overlap,
  time-to-first-token, inter-token latency, and throughput before adopting it;
  retain a simple colocated path for deployments where disaggregation loses.
- Improve scheduling across continuous batching, chunked prefill, mixed
  prefill/decode traffic, fairness, preemption, and latency SLOs. Use
  deterministic workload traces and report both aggregate throughput and tail
  latency so one class of requests is not optimized at another's expense.
- Evaluate scheduler decisions with prompt/decode length distributions,
  prefix-cache hit patterns, KV-block pressure, request priorities, and
  multi-GPU topology. Report TTFT, TPOT/ITL, p50/p95/p99 request latency,
  throughput, starvation, preemption count, and recomputed tokens.
- Optimize host and device memory from measured tensor lifetimes. Prefer safe
  reuse of stable workspaces, staging buffers, KV-cache storage, and recurrent
  state over repeated temporary allocation. Prove shape, dtype, stream/event,
  aliasing, and CUDA Graph lifetime invariants before reusing storage.
- Investigate tensor-space reuse in this order: inventory large and frequent
  allocations; record live ranges and stream ownership; group compatible
  shape/dtype/alignment classes; introduce bounded reusable arenas or pools;
  then measure fragmentation and end-to-end effects. Never reuse storage whose
  prior asynchronous consumer has not completed, and zero or overwrite data
  whenever stale contents could become observable.
- Treat tensor-space reuse as a correctness-sensitive allocator change. Add
  peak allocated/reserved memory, allocation count, fragmentation, and
  end-to-end latency measurements, plus tests that detect overlapping live
  ranges and stale data. Do not accept lower allocator traffic alone as proof
  of an end-to-end improvement.
- For prefill/decode disaggregation, separate the work into placement and
  routing, KV-transfer protocol, backpressure/failure handling, and scheduler
  policy. Require an end-to-end comparison against the colocated path under
  both steady and bursty traffic; do not infer a win from isolated transfer
  bandwidth.
- Advance PD disaggregation through independently testable milestones:
  protocol correctness and recovery; bounded transfer memory; overlap and
  batching; topology-aware placement; load-aware routing; and finally
  multi-node fault handling. Admission control must account for both KV cache
  capacity and in-flight transfer/staging bytes.
- Advance scheduler work through deterministic policies first: explicit token
  and memory budgets, chunked-prefill/decode arbitration, starvation bounds,
  preemption cost accounting, and topology-aware placement. Add learned or
  adaptive policy only when it beats a deterministic baseline across multiple
  traces without violating latency or fairness limits.
- For these tracks, inspect current primary-source designs and relevant PRs in
  vLLM and SGLang, plus specialized systems such as Mooncake, Dynamo, DistServe,
  TensorRT-LLM, or PyTorch where applicable. Record why a design does or does
  not fit nano-vllm rather than copying its architecture wholesale.
- Prioritize work by dependency and validation cost: (1) memory accounting and
  safe tensor/workspace reuse, (2) deterministic memory-aware scheduling, and
  (3) multi-GPU prefill/decode disaggregation. Independent low-risk work may
  continue in parallel when it does not obscure the baseline for a later stage.
- For `Qwen3.6-35B-A3B`, evaluate all three tracks under TP4 and TP8 deployment
  assumptions. A design is not complete until its communication, replicated
  state, per-rank peak VRAM, and behavior under uneven request load are known.

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
- Never create, update, reopen, or close a pull request against a repository
  owned by someone else without the user's explicit approval for that exact
  action. Preparing a local branch and review packet does not grant submission
  permission.
- Before requesting approval to open an upstream PR, present the final diff,
  changed-file list, commit history, focused and full test results, known
  limitations, and proposed PR title and body. Submit only the reviewed commit
  SHA and text; any material change requires renewed approval.
- Keep review-ready upstream candidates local and record them for later user
  review. Waiting for that review must not block normal fork development.
- Apply this lane to vLLM, SGLang, TensorRT-LLM, llama.cpp, LMDeploy, TGI,
  DeepSpeed-MII, MLC LLM, MLX-LM, kernel libraries, and distributed-serving
  projects as well as nano-vllm. Use a separate local branch/worktree based on
  the exact target repository; never mix another project's patch into this
  fork's product branch.
- Start an upstream contribution from `upstream/main` on an
  `upstream-fix/<topic>` branch. Reproduce the problem on unmodified upstream
  first.
- Keep at most one upstream pull request open. Keep later candidate branches
  local; push a candidate to `origin` only after the current upstream pull
  request closes and the candidate is ready for review.
- Delete the remote upstream-contribution branch after its pull request is
  merged or closed, while retaining any still-useful local branch until its
  work has been incorporated or intentionally discarded.
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
