"""Evaluation metrics — exact implementation of METRICS.md.

The headline question is binary: **"is this frame `target-only`?"** ("record" =
target-only). We collapse the 4-class labels to positive (target-only) vs negative
(silence / other-only / overlap) and report precision / recall / F1 for the
target-only class, plus the use-case framing (false-trigger vs miss). **Never bare
accuracy** (METRICS.md) — silence + non-target dominate the frames.

A metric with a zero denominator is *undefined* and returned as `None`, never
silently 0 or 1 (METRICS.md).
"""

from dataclasses import dataclass

import numpy as np

from src.synth.synthesize_scene_example import LABEL_HOP, SR, TARGET_ONLY


def positives_from_labels(labels: np.ndarray) -> np.ndarray:
    """Boolean 'is target-only' view of a 4-class label array (METRICS.md positive)."""
    return labels == TARGET_ONLY


@dataclass
class FrameMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else None      # undefined if we predicted no positives

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else None      # undefined if there are no true positives

    @property
    def f1(self):
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def false_trigger_rate(self):
        """FP / (FP + TN): fraction of non-target frames wrongly recorded."""
        d = self.fp + self.tn
        return self.fp / d if d else None

    @property
    def miss_rate(self):
        """FN / (TP + FN) = 1 - recall: fraction of true target frames dropped."""
        d = self.tp + self.fn
        return self.fn / d if d else None


def frame_metrics(true_pos: np.ndarray, pred_pos: np.ndarray) -> FrameMetrics:
    """Confusion counts with target-only as the positive class (METRICS.md table)."""
    true_pos = true_pos.astype(bool)
    pred_pos = pred_pos.astype(bool)
    tp = int(np.count_nonzero(true_pos & pred_pos))
    fp = int(np.count_nonzero(~true_pos & pred_pos))
    fn = int(np.count_nonzero(true_pos & ~pred_pos))
    tn = int(np.count_nonzero(~true_pos & ~pred_pos))
    return FrameMetrics(tp, fp, fn, tn)


def format_metrics(m: FrameMetrics) -> str:
    """One-line human summary; undefined metrics shown as 'n/a'."""
    def f(x):
        return "n/a " if x is None else f"{x:.3f}"
    return (f"P={f(m.precision)} R={f(m.recall)} F1={f(m.f1)} | "
            f"false_trigger={f(m.false_trigger_rate)} miss={f(m.miss_rate)} "
            f"(TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn})")


# --------------------------------------------------------------------------- #
# Segment-level (secondary diagnostic, METRICS.md "Segment-level")
# --------------------------------------------------------------------------- #

def runs_of_true(mask: np.ndarray):
    """Contiguous runs where mask is True -> list of (start_frame, end_frame_exclusive)."""
    mask = mask.astype(bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def segment_scores(true_pos: np.ndarray, pred_pos: np.ndarray,
                   tol_s: float = 0.2):
    """Match predicted vs true target-only segments with a boundary tolerance.

    Simple, defensible matching (METRICS.md calls this a secondary diagnostic):
    a true segment is 'detected' if a predicted segment overlaps it when both are
    expanded by `tol_s`; a predicted segment is a 'false segment' if it overlaps no
    true segment. Returns counts + a detection error rate.
    """
    tol = int(round(tol_s * SR / LABEL_HOP))  # tolerance in frames
    true_segs = runs_of_true(true_pos)
    pred_segs = runs_of_true(pred_pos)

    def overlaps(a, b):
        return (a[0] - tol) < b[1] and (b[0] - tol) < a[1]

    detected = sum(any(overlaps(t, p) for p in pred_segs) for t in true_segs)
    false_segs = sum(not any(overlaps(p, t) for t in true_segs) for p in pred_segs)
    missed = len(true_segs) - detected

    n_true = len(true_segs)
    det_err = (missed + false_segs) / n_true if n_true else None
    return {
        "n_true_segs": n_true, "n_pred_segs": len(pred_segs),
        "detected": detected, "missed": missed, "false_segs": false_segs,
        "detection_error_rate": det_err, "tol_s": tol_s,
    }
