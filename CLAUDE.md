# CLAUDE.md — read before doing anything

**Always read `ROADMAP.md` first.** It is the source of truth for what this
project is, what's decided, and what to build next. This file is just the short
guardrail list so you don't drift.

## What this project is
Target-Speaker Voice Activity Detection (a.k.a. Personal VAD): given a short
enrollment of a target person, output a **time-stamped timeline of when that
person is speaking alone** in a single-mic dialog. Overlap and other speakers =
"not target." We deliver the detector/pipeline only; downstream analysis is out
of scope.

## Hard constraints (do not violate without updating ROADMAP.md)
- **Compute = Apple Silicon / CPU only.** Freeze pretrained nets; train ONLY a
  small LSTM head. Precompute and cache features to disk.
- **16 kHz mono everywhere.** Resample at ingestion.
- **Speaker-disjoint splits.** Test speakers (target + interferers) never appear
  in training. No utterance-level splitting.
- **Output is a timeline**, not a single yes/no. Dialog needs a *per-window*
  embedding sequence, not one global embedding.
- **Overlap is its own class** (4-class: silence / target-only / other-only /
  overlap). "Record" = target-only.
- **Metrics:** per-class F1 + false-trigger/miss rate. Never bare accuracy.
- **Data synthesis is seeded/deterministic.**
- **Don't commit `.env` / the HF token.**

## Conceptual traps to avoid
- The VAD does NOT separate speakers — only speech vs silence. Speaker identity
  comes from comparing embeddings to the enrollment embedding.
- The mel pathway is a **deliberately tiny 1D `Conv1d` over frequency** (§3). Do not
  upgrade it to a 2D CNN or depthwise-separable/MobileNet tricks without a measured
  gain.

## The author writes the real code — you write `*_example` reference files
This is a **learning project**: the author types the actual source line by line
while studying it. So **never create or edit the real source file** (e.g.
`src/foo.py`). Instead, write a fully-working reference implementation alongside it
with an `_example` suffix (e.g. `src/foo_example.py`). Same rules: simple,
explainable, runnable. The author reads it and re-implements the real file
themselves. When you need to run code to verify a phase, run the `_example` file.
(Docs/config like `ROADMAP.md`, `METRICS.md`, `requirements.txt`, `.gitignore` are
not "code to study" — write those normally.)

## Coding style (learning / portfolio project)
Keep code **simple and explainable** — the author must be able to read and defend
every line in a portfolio review. Prefer plain, straightforward Python over clever
abstractions. Avoid advanced/exotic techniques, heavy design patterns,
metaprogramming, premature optimization, and large frameworks where a small script
does the job. When in doubt, pick the boring, readable option and comment the *why*.

## How to work
Follow the phased plan in ROADMAP.md §6. Finish a phase's milestone before
starting the next. If you change scope or a locked decision, edit ROADMAP.md §2/§7
in the same change.
