"""Phase 4 diagnostics: where does the trained head help, and where does it still err?

The headline F1 (train_example.py) says the model beats the baseline, but F1 alone
hides *which* errors changed. The project's whole thesis (ROADMAP §3-B) is that the
mel branch lets the head reject OVERLAP — the class cosine alone can't reject (it was
92.7% of the baseline's false positives). So we measure that directly:

- per-class F1 (one-vs-rest) for all 4 classes,
- false-positive frames by true class (does overlap still dominate?),
- overlap rejection rate (of true-overlap frames, how many we correctly do NOT call
  target-only) — the number that proves the mel branch earned its place.

Run:  python -m src.eval.eval_model_example
"""

from pathlib import Path

import numpy as np
import torch

from src.eval.baseline_example import VAL_DIR  # reuse the same val feature dir
from src.eval.metrics_example import format_metrics, frame_metrics
from src.models.dataset_example import SceneFeatureDataset
from src.models.model_example import PersonalVAD
from src.synth.synthesize_scene_example import (
    CLASS_NAMES, OTHER_ONLY, OVERLAP, SILENCE, TARGET_ONLY,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "data" / "models" / "personal_vad.pt"


def load_model(path: Path = MODEL_PATH) -> PersonalVAD:
    """Rebuild the head from a saved checkpoint (self-contained: mel stats travel with it)."""
    ckpt = torch.load(path, map_location="cpu")
    model = PersonalVAD()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_all(model, dataset):
    """4-class argmax prediction per frame, concatenated over all scenes (+ truth)."""
    preds, truths = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            s = dataset[i]
            logits = model(s["mel"][None], s["emb"][None], s["cos"][None],
                           s["vad"][None], s["e_target"][None])
            preds.append(logits[0].argmax(-1).numpy())
            truths.append(s["labels"].numpy())
    return np.concatenate(preds), np.concatenate(truths)


def per_class_f1(pred, truth):
    """One-vs-rest F1 for each of the 4 classes (a diagnostic, not the headline)."""
    out = {}
    for cls in (SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP):
        m = frame_metrics(truth == cls, pred == cls)
        out[cls] = m.f1
    return out


def main() -> None:
    val = SceneFeatureDataset(VAL_DIR)
    model = load_model()
    pred, truth = predict_all(model, val)

    # Headline: target-only frame metrics.
    m = frame_metrics(truth == TARGET_ONLY, pred == TARGET_ONLY)
    print(f"target-only (headline): {format_metrics(m)}\n")

    # Per-class F1.
    print("per-class F1 (one-vs-rest):")
    for cls, f1 in per_class_f1(pred, truth).items():
        print(f"  {CLASS_NAMES[cls]:12s}: {f1:.3f}" if f1 is not None
              else f"  {CLASS_NAMES[cls]:12s}: n/a")

    # False positives by true class: is overlap still the dominant failure?
    fp_mask = (pred == TARGET_ONLY) & (truth != TARGET_ONLY)
    fp = {cls: int(np.count_nonzero(fp_mask & (truth == cls)))
          for cls in (SILENCE, OTHER_ONLY, OVERLAP)}
    total_fp = sum(fp.values()) or 1
    print("\nfalse-positive frames by true class (model):")
    for cls in (OVERLAP, OTHER_ONLY, SILENCE):
        print(f"  {CLASS_NAMES[cls]:12s}: {fp[cls]:6d} ({100*fp[cls]/total_fp:4.1f}%)")

    # The thesis check: overlap rejection. Of true-overlap frames, how many did we
    # correctly NOT label target-only? Baseline rejected almost none (§3-B).
    overlap_frames = int(np.count_nonzero(truth == OVERLAP))
    overlap_rejected = int(np.count_nonzero((truth == OVERLAP) & (pred != TARGET_ONLY)))
    rate = overlap_rejected / overlap_frames if overlap_frames else None
    print(f"\noverlap rejection rate: {overlap_rejected}/{overlap_frames} = "
          f"{rate:.3f}  (fraction of overlap frames NOT called target-only)")
    print("  -> baseline rejected ~0 of these; this is the mel branch's job (§3-B).")


if __name__ == "__main__":
    main()
