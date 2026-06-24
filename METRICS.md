# Metrics — exact definitions

This file pins the evaluation protocol in `ROADMAP.md` §5 to **exact formulas** so
every phase (the Phase 3 baseline, the Phase 4 head, ablations) scores identically.
**Never report bare accuracy** — silence/non-target dominate the frames, so accuracy
is misleading (ROADMAP §5).

## Unit of evaluation: the frame

Predictions and ground truth are both **per-frame 4-class** label arrays on one
common frame grid (ROADMAP §3): `silence`, `target-only`, `other-only`, `overlap`.
Evaluate only on the speaker-disjoint val/test sets (ROADMAP §2).

The primary question is binary: **"is this frame `target-only`?"** ("record" =
`target-only`). So we collapse the 4 classes to a positive/negative view:

- **positive** = frame whose true (or predicted) class is `target-only`.
- **negative** = frame that is `silence`, `other-only`, or `overlap`.

(Per-class F1 for the other three classes is reported as secondary diagnostics, but
`target-only` is the headline.)

## Confusion counts (target-only as positive)

Over all evaluated frames:

| | predicted target-only | predicted non-target |
|---|---|---|
| **true target-only** | TP | FN |
| **true non-target**  | FP | TN |

## Primary metrics

```
precision = TP / (TP + FP)        # of frames we said "target", how many were
recall    = TP / (TP + FN)        # of true target frames, how many we caught
F1        = 2 * precision * recall / (precision + recall)
```

`F1` of the `target-only` class is the **primary number** every later phase must
beat (ROADMAP §6 Phase 3/4). If a denominator is 0, the metric is undefined → report
it as such, do not silently treat as 0 or 1.

## Use-case framing: false-trigger vs miss

The downstream use is "record the target alone." Two error types have different
costs, so report both explicitly (ROADMAP §5):

```
false_trigger_rate = FP / (FP + TN)   # fraction of non-target frames we wrongly recorded
miss_rate          = FN / (TP + FN)   # fraction of true target frames we dropped
                   = 1 - recall
```

- **False trigger** = recording non-target audio (a non-target frame labeled
  target-only). Pollutes the extracted output.
- **Miss** = dropping target audio (a target frame labeled non-target). Loses wanted
  content.

Which is costlier is decided in Phase 5; the decision threshold is then tuned **on
val** to favor the cheaper error (ROADMAP §5/§7).

## Segment-level (after smoothing)

After hysteresis + min-segment-duration smoothing (ROADMAP §3), compare predicted vs
true `target-only` **segments** (contiguous runs), not just frames:

- A predicted segment matches a true segment if they overlap within a stated
  **boundary tolerance** (e.g. ±0.2 s); report the tolerance used.
- Report a detection error rate (missed + false + boundary errors). Frame-level
  metrics stay primary; segment metrics show whether smoothing helped.

## F1-vs-TIR curve (overlap characterization)

Overlap rejection is governed by the **target-to-interferer ratio (TIR)** set during
synthesis — a dial, not a single number (ROADMAP §3-C). So do **not** summarize
overlap with one F1: sweep TIR over a realistic range and **plot `target-only` F1 as
a function of TIR**. That curve is the honest characterization (ROADMAP §5/§6 Phase 5).
