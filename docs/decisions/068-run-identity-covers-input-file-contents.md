# 068 — A run's identity covers the contents of the files it reads, not just their paths

**Status:** Adopted 2026-09-22. Public (infrastructure).

## The problem

A run's configuration names its inputs by path: `scorer.cache_path` for a cached importance map,
`scorer.calibration` for an affine score calibration. The resume identity (`config_hash`) was the
serialised configuration plus the hook parameters, so two runs reading *different contents* through
the *same path* were indistinguishable. Regenerating a cache and re-running the sweep therefore
reported every cell as already done, and the index silently described results that the files on
disk no longer produce.

This was found while re-running the paper-two transfer validation after re-fitting the confusion
model: the regenerated caches had the same names as the old ones. Nothing in the first study was
affected — its caches were written once and never rewritten — but the hazard is general, and
"every run is regenerable from config + seed" is a stated hard requirement for this project. A path
is not a value.

The hook-parameter case was already handled (decision on PR #30 hashes a `{path: ...}` parameter by
content). This extends the same rule to the configuration itself.

## Options

| Option | Verdict |
|---|---|
| Fold the content digests into `config_hash` | Correct in principle, rejected in practice: it changes the identity of all 17,240 runs already indexed, so the frozen pre-registered study would re-run in full and its recorded hashes would no longer match the registered ones. |
| Record digests on the index row; compare them at resume when present | **Chosen.** A row written before this existed carries no digests and is trusted as before, so the frozen study is untouched. Every row written from now on records what it actually read, and a changed input re-runs its cells. |
| Make the runner refuse to resume whenever any input file's mtime is newer than the index | Cheap but wrong twice over: mtime changes when content does not (a re-copy), and content can change while mtime is preserved. |
| Copy every input into the run directory | Correct and self-describing, but the importance caches are megabytes each and every cell would carry its own copy. |

## What it does

`input_digests(cfg)` walks the serialised configuration for any string under a key ending in
`path` (plus `calibration`), and records a twelve-character SHA-256 prefix of each file that
exists. It goes on the index row as `input_digests`. At resume a finished row is skipped when its
`config_hash` matches **and** either it carries no digests (a pre-existing row) or its digests still
match the files on disk. The log line reports how many of the cells to run are stale rather than
new, so a re-run that was meant to be a no-op says so.

## The limit worth stating

This detects a changed file, not a missing one: a cell whose cache has been deleted since it ran is
still reported as done, because the digest map simply omits paths that do not exist and an absent
input is indistinguishable from a configuration that never named one. That is the conservative
direction — it never invents work — but it means a deleted cache is caught by the run failing, not
by resume.
