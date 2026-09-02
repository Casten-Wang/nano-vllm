# Contributing

## Remotes and branches

This fork uses two remotes:

- `origin`: `https://github.com/Casten-Wang/nano-vllm.git`
- `upstream`: `https://github.com/GeeeekExplorer/nano-vllm.git`

`main` is the latest stable milestone. Ongoing product work lives on the
long-lived `feature/qwen35-text-foundation` branch:

```bash
git switch feature/qwen35-text-foundation
git pull --ff-only origin feature/qwen35-text-foundation
```

Push each completed and verified development commit to that branch. Promote a
coherent, fully verified milestone to `main`; do not update `main` for every
small commit, and do not wait for the open-ended optimization roadmap to end.

## Before committing

```bash
python3 scripts/check_repo_policy.py
python3 -m compileall -q nanovllm projects scripts tests
python3 -m pytest -q
git diff --cached --check
```

If a GPU test cannot run in the current environment, state that explicitly in
the pull request. Never describe an unexecuted check as passing.

Only source, tests, build configuration, CI configuration, and these two
governance files belong in Git. The policy checker rejects local documents,
media, weights, archives, generated results, and interview tooling.

## Commits and pull requests

Use focused Conventional Commits, for example:

```text
feat: add Qwen3.5 MoE model configuration
fix: preserve KV block ownership during preemption
perf: fuse expert routing and token permutation
test: cover mixed linear and full attention layers
```

Push normal development to the product-development branch:

```bash
git push origin feature/qwen35-text-foundation
```

A pull request must explain the problem, approach, tests, limitations, and any
performance methodology. Large features should be split into independently
reviewable steps.

## Researching established runtimes

Before implementing an inference optimization, inspect the current code and
relevant issues or pull requests in vLLM, SGLang, and llama.cpp when applicable.
For specialized work, also consult the primary implementation in projects such
as TensorRT-LLM, FlashAttention, FlashInfer, or PyTorch. Record the repository,
revision, exact source location, supported hardware, benchmark method, and
limitations in the local evidence notes.

Use those projects to understand algorithms and engineering trade-offs, then
implement and verify the idea independently for nano-vllm. Respect licenses and
attribution, and never present another project's measurements as evidence for
this repository. Keep borrowed architectural ideas minimal enough to preserve
nano-vllm's educational scope.

Potential fixes discovered in any third-party project may be prepared as local
review candidates. Never create or update an external pull request without the
user's explicit approval for that exact revision and PR text.

## Releases

After the pull request is merged and the resulting `main` is verified, create
an annotated semantic version tag:

```bash
git switch main
git pull --ff-only origin main
git tag -a v1.1.0 -m "nano-vllm optimized v1.1.0"
git push origin main v1.1.0
```

Patch versions fix bugs, minor versions add backward-compatible functionality,
and major versions contain incompatible changes.

## Contributing fixes upstream

Opening or changing a pull request against a repository owned by someone else
always requires the user's explicit approval for that exact external action.
Do not treat permission to investigate, implement, commit, push to this fork,
or prepare a PR as permission to submit it upstream.

Before asking for submission approval, provide a review packet containing the
final diff and changed files, exact commit SHA(s), focused and full test
results, known limitations, and the proposed PR title and body. After approval,
submit only that reviewed revision. If the code or PR text changes materially,
request approval again. The same rule applies to updating, reopening, or
closing an existing upstream PR. A review-ready candidate remains local until
then and does not pause ongoing development in this fork.

Upstream contribution branches must start from the upstream repository rather
than this fork's `main`:

```bash
git fetch upstream
git switch -c upstream-fix/short-topic upstream/main
```

Before writing code, search upstream issues and pull requests for duplicates.
Reproduce the behavior on `upstream/main`, then keep the patch narrowly scoped.
Do not include this fork's `.gitignore`, governance, CI, broad instrumentation,
or unrelated optimization work in an upstream pull request.

The preferred upstream submission is one independently useful correctness or
performance fix with a minimal reproduction and before/after evidence. Large
features such as Qwen3.5 support are developed and validated in this fork, then
split only when a small component has clear value against current upstream.
Keep only one upstream pull request open at a time. Later candidate branches
remain local and are pushed to `origin` only when the current pull request has
closed and the next patch is review-ready. Remove merged or closed PR branches
from the fork so its remote branch list stays limited to `main`, the active
product-development branch, and the current upstream PR branch.
