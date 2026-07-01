"""Phase 5 cross-eval: score any saved model on any cached feature dir (clean/noisy).

Fills the 2x2 in RESULTS.md — train {clean, noisy} x test {clean, noisy}. Reuses the
Phase 4 diagnostics (target-only metrics, per-class F1, overlap-rejection) but lets you
point --model and --data anywhere, so we can measure the robustness gap directly:

  train clean, test noisy  -> how badly does noise hurt a model that never saw it?
  train noisy, test noisy  -> best case with matched noise
  train noisy, test clean  -> does noise-training cost anything on clean audio?

Run (after caching the noisy features + training the noisy model):
  python -m src.eval.noise_experiment_example --model data/models/personal_vad.pt \
                                              --data  data/features_noisy/val
"""

import argparse
from pathlib import Path

import numpy as np

from src.eval.eval_model_example import load_model, per_class_f1, predict_all
from src.eval.metrics_example import format_metrics, frame_metrics
from src.models.dataset_example import SceneFeatureDataset
from src.synth.synthesize_scene_example import CLASS_NAMES, OVERLAP, TARGET_ONLY

REPO_ROOT = Path(__file__).resolve().parents[2]


def score(model_path: Path, data_dir: Path) -> None:
    """Print target-only metrics, per-class F1 and overlap-rejection for one cell."""
    model = load_model(model_path)
    ds = SceneFeatureDataset(data_dir)
    pred, truth = predict_all(model, ds)

    m = frame_metrics(truth == TARGET_ONLY, pred == TARGET_ONLY)
    overlap_frames = int(np.count_nonzero(truth == OVERLAP))
    overlap_rejected = int(np.count_nonzero((truth == OVERLAP) & (pred != TARGET_ONLY)))
    rej = overlap_rejected / overlap_frames if overlap_frames else float("nan")
    f1s = per_class_f1(pred, truth)

    print(f"model : {model_path}")
    print(f"data  : {data_dir}  ({len(ds)} scenes)")
    print(f"  target-only : {format_metrics(m)}")
    print(f"  overlap-rej : {rej:.3f}")
    print("  per-class F1:", {CLASS_NAMES[c]: round(f1s[c], 3) for c in f1s})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a saved .pt checkpoint")
    ap.add_argument("--data", required=True, help="a cached feature dir (clean or noisy)")
    args = ap.parse_args()
    score((REPO_ROOT / args.model).resolve(), (REPO_ROOT / args.data).resolve())


if __name__ == "__main__":
    main()
