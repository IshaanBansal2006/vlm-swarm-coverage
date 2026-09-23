# 070 — The run index tolerates a line torn by a crash

**Status:** Adopted 2026-09-23. Public (infrastructure).

## What happened

A study was interrupted mid-sweep. The index is append-only, and the kill landed between the
filesystem allocating a block and the data reaching it, so the file ended with 966 NUL bytes
where a row should have been. `read_index` parsed every line eagerly, so the next run raised
`JSONDecodeError: Expecting value: line 1 column 1` and the whole study refused to resume —
discarding 36 completed runs that were sitting in the file, intact, above the damage.

That is the wrong failure. The index exists so that work survives an interruption; a reader that
dies on the interruption's own signature defeats it.

## The decision

`read_index` returns every parseable row. A line of NUL bytes or whitespace is treated as an empty
torn tail and ignored silently. Any *other* unparseable line is skipped and reported in a warning
naming the file, the line numbers and how many rows were kept.

## Options

| Option | Verdict |
|---|---|
| **Skip unparseable lines, warn, keep the rest** | **Chosen.** Resume works, and damage is stated rather than hidden. |
| Keep failing on any bad line | Correct-looking and useless in practice: the one situation it triggers in is the one where the file is most valuable. |
| Skip bad lines silently | Rejected. Real data loss would look like a smaller study that still reports as finished, which is precisely the class of silent-wrong-answer this project has been bitten by repeatedly. |
| Truncate the file at the first bad line | Destructive, and wrong if the damage is mid-file rather than at the tail. |
| Write the index atomically (temp file + rename per row) | Robust but costs a rename per run in a hot loop, and the append-only file is also what makes concurrent workers cheap. Tolerating a torn tail is the cheaper half of the same guarantee. |

## The distinction worth keeping

A torn *final* line is a process that was killed; it is expected and benign. An unparseable line
anywhere else means something worse — a partial disk, an interleaved write, a corrupted file — and
the warning says so, because those results should not be trusted without a look.
