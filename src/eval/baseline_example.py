"""Phase 3 — the non-learned baseline (ROADMAP §6).

The simplest thing that could possibly work: **no LSTM, no mel, no training.** Just
gate on speech (Silero VAD) and threshold the cosine similarity to the enrollment
embedding (ROADMAP §3-A). A frame is predicted `target-only` when it is speech AND
its cosine to `e_target` clears a threshold:

    pred_target_only[t] = (vad[t] >= vad_thresh) AND (cos[t] >= cos_thresh)

Then a little smoothing (hysteresis + min-segment-duration, ROADMAP §3). We sweep the
cosine threshold **on val** and pick the F1-best operating point. This frame-level
target-only F1 is **the number every later phase must beat** (ROADMAP §6, METRICS.md).

Expected failure mode (and the reason Phase 4 exists): cosine can't tell `target-only`
from `overlap` — during overlap the target's voice is present, so cosine stays high
and the baseline false-triggers. Only the mel pathway (§3-B) separates those. We
print the false-positive breakdown by true class to make that visible.

Run (after caching val features):
  python -m src.features.build_feature_cache_example --split val --n 20
  python -m src.eval.baseline_example
"""

from pathlib import Path

import numpy as np

from src.eval.metrics_example import (
    format_metrics, frame_metrics, positives_from_labels, runs_of_true,
    segment_scores,
)
from src.synth.synthesize_scene_example import (
    LABEL_HOP, SR, OTHER_ONLY, OVERLAP, SILENCE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = REPO_ROOT / "data" / "features" / "val"

VAD_THRESH = 0.5          # Silero speech gate (fixed; the cosine threshold is swept)
HYST_MARGIN = 0.05        # hysteresis low threshold = cos_thresh - this
MIN_SEG_S = 0.20          # drop predicted target runs shorter than this
MIN_GAP_S = 0.10          # fill predicted negative gaps shorter than this


# --------------------------------------------------------------------------- #
# The detector + smoothing
# --------------------------------------------------------------------------- #

def predict_raw(cos: np.ndarray, vad: np.ndarray,
                cos_thresh: float, vad_thresh: float = VAD_THRESH) -> np.ndarray:
    """Speech-gated cosine threshold -> boolean target-only prediction (no smoothing)."""
    return (vad >= vad_thresh) & (cos >= cos_thresh)


def hysteresis(score: np.ndarray, gate: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Two-threshold gating: enter positive at `hi`, stay until score drops below `lo`.

    Reduces flicker at boundaries vs a single threshold. `gate` (speech) must hold for
    a frame to be positive at all.
    """
    out = np.zeros(len(score), dtype=bool)
    state = False
    for i in range(len(score)):
        if not gate[i]:
            state = False
        elif state:
            state = score[i] >= lo
        else:
            state = score[i] >= hi
        out[i] = state
    return out


def enforce_min_durations(mask: np.ndarray, min_seg_frames: int,
                          min_gap_frames: int) -> np.ndarray:
    """Fill short negative gaps, then drop short positive runs (min-segment smoothing)."""
    mask = mask.copy()
    # Fill short gaps first so a brief dropout doesn't split one real segment.
    for start, end in runs_of_true(~mask):
        if end - start < min_gap_frames:
            mask[start:end] = True
    # Then remove blips that are too short to be a real target turn.
    for start, end in runs_of_true(mask):
        if end - start < min_seg_frames:
            mask[start:end] = False
    return mask


def predict_smoothed(cos: np.ndarray, vad: np.ndarray, cos_thresh: float,
                     vad_thresh: float = VAD_THRESH) -> np.ndarray:
    """Baseline prediction with hysteresis + min-segment-duration smoothing."""
    speech = vad >= vad_thresh
    mask = hysteresis(cos, speech, hi=cos_thresh, lo=cos_thresh - HYST_MARGIN)
    min_seg = int(round(MIN_SEG_S * SR / LABEL_HOP))
    min_gap = int(round(MIN_GAP_S * SR / LABEL_HOP))
    return enforce_min_durations(mask, min_seg, min_gap)


# --------------------------------------------------------------------------- #
# Load cached val features
# --------------------------------------------------------------------------- #

def load_split(split_dir: Path):
    """Load every cached scene in a split. Returns a list of dicts with the arrays."""
    paths = sorted(split_dir.glob("scene_seed*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no cached features in {split_dir}. Run:\n"
            f"  python -m src.features.build_feature_cache_example "
            f"--split {split_dir.name} --n 20")
    scenes = []
    for p in paths:
        d = np.load(p)
        scenes.append({"cos": d["cos"], "vad": d["vad"], "labels": d["labels"]})
    return scenes


def pooled_metrics(scenes, cos_thresh, predictor):
    """Concatenate per-frame truth/prediction over all scenes, then score once."""
    true_pos, pred_pos = [], []
    for s in scenes:
        true_pos.append(positives_from_labels(s["labels"]))
        pred_pos.append(predictor(s["cos"], s["vad"], cos_thresh))
    true_pos = np.concatenate(true_pos)
    pred_pos = np.concatenate(pred_pos)
    return frame_metrics(true_pos, pred_pos), true_pos, pred_pos


# --------------------------------------------------------------------------- #
# Report (the Phase 3 milestone)
# --------------------------------------------------------------------------- #

def fp_breakdown_by_class(scenes, cos_thresh, predictor):
    """Where do false positives come from? Count FP frames per true non-target class."""
    counts = {SILENCE: 0, OTHER_ONLY: 0, OVERLAP: 0}
    for s in scenes:
        pred = predictor(s["cos"], s["vad"], cos_thresh)
        fp = pred & ~positives_from_labels(s["labels"])
        for cls in counts:
            counts[cls] += int(np.count_nonzero(fp & (s["labels"] == cls)))
    return counts


def main() -> None:
    scenes = load_split(VAL_DIR)
    n_frames = sum(len(s["labels"]) for s in scenes)
    print(f"loaded {len(scenes)} val scenes, {n_frames} frames "
          f"(~{n_frames * 0.01:.0f}s)\n")

    # 1. Sweep the cosine threshold on val (plain threshold), pick F1-best.
    print("cosine-threshold sweep on val (raw, speech-gated):")
    print(f"  {'thresh':>7}  {'P':>6} {'R':>6} {'F1':>6}  {'falseTrig':>9} {'miss':>6}")
    grid = np.round(np.arange(0.10, 0.625, 0.025), 3)
    best = None
    for t in grid:
        m, _, _ = pooled_metrics(scenes, t, predict_raw)
        f1 = m.f1 if m.f1 is not None else -1
        if best is None or f1 > best[0]:
            best = (f1, t, m)
        ft = m.false_trigger_rate or 0
        print(f"  {t:>7.3f}  {m.precision or 0:>6.3f} {m.recall or 0:>6.3f} "
              f"{m.f1 or 0:>6.3f}  {ft:>9.3f} {m.miss_rate or 0:>6.3f}")

    _, best_t, best_m = best
    print(f"\nF1-best threshold on val: cos >= {best_t:.3f}")
    print(f"  RAW      : {format_metrics(best_m)}")

    # 2. Same threshold, with smoothing.
    m_smooth, true_pos, pred_smooth = pooled_metrics(scenes, best_t, predict_smoothed)
    print(f"  SMOOTHED : {format_metrics(m_smooth)}")

    # 3. Where the baseline fails: false positives by true class (motivates Phase 4).
    fp = fp_breakdown_by_class(scenes, best_t, predict_smoothed)
    total_fp = sum(fp.values()) or 1
    print("\nfalse-positive frames by true class (smoothed @ best threshold):")
    print(f"  overlap   : {fp[OVERLAP]:6d} ({100*fp[OVERLAP]/total_fp:4.1f}%)  "
          "<- cosine can't reject overlap; this is Phase 4's job (mel branch, §3-B)")
    print(f"  other-only: {fp[OTHER_ONLY]:6d} ({100*fp[OTHER_ONLY]/total_fp:4.1f}%)")
    print(f"  silence   : {fp[SILENCE]:6d} ({100*fp[SILENCE]/total_fp:4.1f}%)")

    # 4. Segment-level diagnostic (secondary, METRICS.md).
    seg = segment_scores(true_pos, pred_smooth, tol_s=0.2)
    print(f"\nsegment-level (±{seg['tol_s']}s tol): "
          f"true={seg['n_true_segs']} pred={seg['n_pred_segs']} "
          f"detected={seg['detected']} missed={seg['missed']} "
          f"false={seg['false_segs']} "
          f"detErr={seg['detection_error_rate']:.3f}")

    print(f"\n>>> Phase 3 baseline target-only F1 (smoothed) = "
          f"{m_smooth.f1:.3f}  — the floor Phase 4 must beat.")


if __name__ == "__main__":
    main()
