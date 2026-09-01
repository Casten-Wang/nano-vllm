# Contributing

## Remotes and branches

This fork uses two remotes:

- `origin`: `https://github.com/Casten-Wang/nano-vllm.git`
- `upstream`: `https://github.com/GeeeekExplorer/nano-vllm.git`

Keep `main` releasable. Start work from an updated `main` on a focused branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/short-topic
```

Use `fix/`, `perf/`, or `test/` when those names describe the change better.

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

Push the branch to this fork and open a pull request into its `main`:

```bash
git push -u origin feature/short-topic
```

A pull request must explain the problem, approach, tests, limitations, and any
performance methodology. Large features should be split into independently
reviewable steps.

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
