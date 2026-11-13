# Project instructions

## `journal/` — the research log

This repo keeps a research journal in `journal/`. Since 2026-11-13, every A/B experiment (a change,
tested and measured) has its own file in `journal/experiments/<NAME>.md` with a fixed template; see
`journal/experiments/README.md` and the approved example `journal/experiments/COMMIT_AHEAD.md`.
Reference material (data, tokenisation, metric definitions, configs, runbooks, ReaLJam) lives in
`journal/reference/<TOPIC>.md`. Every substantive finding gets written up in one of the two: the technical log — bugs and their root cause, exact
configs/checkpoints/job IDs, results tables, the reasoning trail including false starts and
corrections, and what is still open.

There used to be a second, thesis-distilled tier at `journal/thesis/<topic>.md`. **It was
removed on 2026-10-12 and must not be recreated** — the thesis is written by hand, so the
distilled copies were duplicated effort and drifted behind the technical ones. Write each
finding once, and keep the numbers and reasoning complete enough that a thesis section can be
written straight from the technical doc.

Full formatting rules and the rationale: **read `journal/README.md` before writing to it.**

## Two chord vocabularies -- check before mixing

Checkpoints use one of two incompatible chord vocabularies (5,902 vs 6,198 tokens; every chord
token id differs, melody ids are shared). Before feeding any token tensor to a model, or
comparing outputs across checkpoints, confirm both use the same one, and remap by name if not.
Which checkpoints use which, how to tell, and the remap helper: `journal/reference/TOKENIZERS.md`,
section 0.
