# 062 — Sweeps: a spec of axes, expanded to runs, executed in parallel, indexed

**Status:** Accepted 2026-09-20
**Area:** infrastructure

## Context

The study is thousands of runs that differ in a handful of config values and a seed. Running
them by hand is not reproducible; a runner that knows about the study's variables would put
experimental design into public infrastructure. The runner needs to be generic over config keys,
resumable, parallel, and able to attach things a config does not describe.

## Options

| Option | Trade-off |
|---|---|
| Shell loop over `vsc-run` | Trivial, but no index, no resume, no way to attach non-config components, and the run naming is ad hoc |
| A workflow tool (Snakemake, Hydra multirun) | Capable, but a dependency and a second configuration language for a problem that is one file of Python |
| **A spec file of axes + a small runner** | Config-agnostic (dotted keys), resumable by config hash, parallel by process pool, extensible by two named hooks |

## Decision

`sweep.py`: `SweepSpec` (base config, `axes` as dotted key → values, `full` or `one_at_a_time`
design, seeds, explicit `cells`, `hook`, `post_run`, `workers`). `expand` produces deterministic
cells; each run's config is the base with the overrides applied and validated, named after the
sweep and a hash of the assignments. `run_sweep` skips any cell whose config hash already has a
successful row in `index.jsonl`, runs the rest in a fork-based process pool, and appends a row per
run as it completes. A failed run is a row with `status: error` and the exception, and the
sweep continues; failures are retried on the next invocation and the exit code is non-zero.

Axis keys that start with `_` are not config: they are passed to `hook(cfg, params)`, which
returns keyword arguments for `simulation.build` (a scorer wrapper, say). `post_run(run_dir)`
returns a dict merged into the row (a metric summary, say). Both are `module:function` strings
imported in the worker.

## Consequences

- The runner never learns what a study varies; the spec does, and the spec is a file that can be
  kept wherever its contents belong.
- Resume is by config hash, so editing the base config re-runs everything that changed and
  nothing that did not.
- Fork start method: workers inherit the parent's imports and `sys.path`, which is what makes
  `module:function` hooks defined outside the package importable. Fork after CUDA initialisation
  is unsafe, so a sweep that needs a GPU scorer must load the model inside the hook, not before.
