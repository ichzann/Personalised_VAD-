"""Phase 4 milestone: train the LSTM head and beat the Phase 3 baseline.

Trains the small PersonalVAD head on cached TRAIN scene features, selects the best
epoch on VAL by target-only F1, and reports val F1 + false-trigger rate against the
Phase 3 baseline floor (ROADMAP §6: "done when it beats the Phase 3 baseline").

Everything the head sees was produced by the FROZEN backbones (Phase 2); only this
head trains (ROADMAP §2). One scene per optimizer step, weighted cross-entropy to
counter the silence-heavy class balance (ROADMAP §5).

Run (after caching features for both splits):
  python -m src.features.build_feature_cache_example --split train --n 150
  python -m src.features.build_feature_cache_example --split val   --n 20
  python -m src.models.train_example
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.eval.baseline_example import (
    load_split, pooled_metrics, predict_raw, predict_smoothed,
)
from src.eval.metrics_example import format_metrics, frame_metrics
from src.models.dataset_example import (
    SceneFeatureDataset, compute_class_weights, compute_mel_stats,
)
from src.models.model_example import PersonalVAD
from src.synth.synthesize_scene_example import TARGET_ONLY

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "data" / "features" / "train"
VAL_DIR = REPO_ROOT / "data" / "features" / "val"
MODEL_OUT = REPO_ROOT / "data" / "models" / "personal_vad.pt"

BASELINE_FLOOR = 0.702   # Phase 3 smoothed target-only F1 (recomputed below for parity)


def evaluate(model, val_dataset) -> "FrameMetrics":
    """Per-frame target-only metrics from the model's 4-class argmax over all val scenes."""
    model.eval()
    true_pos, pred_pos = [], []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            s = val_dataset[i]
            logits = model(s["mel"][None], s["emb"][None],
                           s["cos"][None], s["vad"][None], s["e_target"][None])
            pred = logits[0].argmax(-1)                  # (T,) 4-class
            true_pos.append((s["labels"] == TARGET_ONLY).numpy())
            pred_pos.append((pred == TARGET_ONLY).numpy())
    return frame_metrics(np.concatenate(true_pos), np.concatenate(pred_pos))


def baseline_on_val(val_dir: Path) -> float:
    """Recompute the Phase 3 baseline's best target-only F1 on the same val set."""
    scenes = load_split(val_dir)
    best = 0.0
    for t in np.round(np.arange(0.10, 0.625, 0.025), 3):
        m, _, _ = pooled_metrics(scenes, t, predict_smoothed)
        if m.f1 is not None and m.f1 > best:
            best = m.f1
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5, help="early-stop patience")
    ap.add_argument("--seed", type=int, default=0)
    # Dir/out overrides so the noisy run (Phase 5) trains on data/features_noisy and
    # saves its own checkpoint without clobbering the clean model.
    ap.add_argument("--train-dir", default=str(TRAIN_DIR))
    ap.add_argument("--val-dir", default=str(VAL_DIR))
    ap.add_argument("--out", default=str(MODEL_OUT))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_dir = Path(args.val_dir)
    out_path = Path(args.out)
    train_ds = SceneFeatureDataset(Path(args.train_dir))
    val_ds = SceneFeatureDataset(val_dir)
    print(f"train scenes: {len(train_ds)}   val scenes: {len(val_ds)}")

    # Normalization + class balance from TRAIN only (no val/test leakage).
    mel_mean, mel_std = compute_mel_stats(train_ds)
    class_weights = compute_class_weights(train_ds)
    print(f"class weights (sil/tgt/other/ovl): "
          f"{[round(float(w), 2) for w in class_weights]}")

    model = PersonalVAD()
    model.set_mel_stats(mel_mean, mel_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"head parameters: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    base_f1 = baseline_on_val(val_dir)
    print(f"Phase 3 baseline target-only F1 on this val set: {base_f1:.3f}\n")

    best_f1 = -1.0
    best_state = None
    since_best = 0
    order = np.arange(len(train_ds))

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(order)                    # shuffle scene order each epoch
        epoch_loss = 0.0
        for idx in order:
            s = train_ds[int(idx)]
            optimizer.zero_grad()
            logits = model(s["mel"][None], s["emb"][None],
                           s["cos"][None], s["vad"][None], s["e_target"][None])  # (1,T,4)
            loss = loss_fn(logits.reshape(-1, 4), s["labels"].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)

        m = evaluate(model, val_ds)
        f1 = m.f1 if m.f1 is not None else -1.0
        flag = ""
        if f1 > best_f1:
            best_f1, best_state, since_best = f1, {k: v.clone() for k, v in
                                                   model.state_dict().items()}, 0
            flag = "  <- best"
        else:
            since_best += 1
        print(f"epoch {epoch:2d}  loss={epoch_loss/len(train_ds):.3f}  "
              f"val: {format_metrics(m)}{flag}")
        if since_best >= args.patience:
            print(f"early stop (no val improvement in {args.patience} epochs)")
            break

    # Restore + save the best model.
    model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mel_mean": mel_mean, "mel_std": mel_std},
               out_path)

    final = evaluate(model, val_ds)
    print("\n" + "=" * 70)
    print("PHASE 4 RESULT (best epoch on val)")
    print(f"  model    : {format_metrics(final)}")
    print(f"  baseline : target-only F1 = {base_f1:.3f}")
    verdict = "BEATS" if final.f1 > base_f1 else "does NOT beat"
    print(f"  -> model F1 {final.f1:.3f} {verdict} baseline F1 {base_f1:.3f}")
    print(f"  saved {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
