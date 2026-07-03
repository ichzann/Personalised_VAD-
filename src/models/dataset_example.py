"""Phase 4 data loading: cached .npz scenes -> per-scene tensors for the head.

The frozen backbones already did the expensive work (Phase 2) and wrote each scene to
disk. Here we just load those arrays and hand them to the model. We train one scene
per step (batch size 1) — the simplest thing that avoids padding/masking variable-
length dialogs (ROADMAP §8 simplicity-first); a small BiLSTM on CPU handles a full
~35 s scene fine.
"""

from pathlib import Path

import numpy as np
import torch


class SceneFeatureDataset:
    """Loads cached scene feature bundles. Each item is one scene's tensors."""

    def __init__(self, features_dir: Path):
        self.paths = sorted(Path(features_dir).glob("scene_seed*.npz"))
        if not self.paths:
            raise FileNotFoundError(
                f"no cached features in {features_dir}. Build them with:\n"
                f"  python -m src.features.build_feature_cache_example "
                f"--split {Path(features_dir).name} --n 150")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        return {
            "mel": torch.from_numpy(d["logmel"]).float(),     # (T,40)
            "emb": torch.from_numpy(d["emb"]).float(),        # (T,192)
            "cos": torch.from_numpy(d["cos"]).float(),        # (T,)
            "vad": torch.from_numpy(d["vad"]).float(),        # (T,)
            "labels": torch.from_numpy(d["labels"]).long(),   # (T,)
            "e_target": torch.from_numpy(d["e_target"]).float(),  # (192,) enrollment
        }


def compute_mel_stats(dataset: SceneFeatureDataset):
    """Per-mel-bin mean/std over all training frames (for standardization).

    Computed on TRAIN only and saved into the model, so val/test never leak into the
    normalization (ROADMAP §2 speaker-disjoint spirit).
    """
    total = None
    sq = None
    n = 0
    for i in range(len(dataset)):
        mel = dataset[i]["mel"]                      # (T,40)
        if total is None:
            total = mel.sum(0)
            sq = (mel ** 2).sum(0)
        else:
            total += mel.sum(0)
            sq += (mel ** 2).sum(0)
        n += mel.shape[0]
    mean = total / n
    var = sq / n - mean ** 2
    return mean, var.clamp_min(1e-12).sqrt()


def compute_class_weights(dataset: SceneFeatureDataset, n_classes: int = 4):
    """Inverse-frequency class weights for cross-entropy.

    Silence + non-target dominate the frames (ROADMAP §5), so unweighted CE would
    ignore the rare but important target-only/overlap classes. Weight each class by
    total/(n_classes * count) so each contributes roughly equally to the loss.
    """
    counts = torch.zeros(n_classes)
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        counts += torch.bincount(labels, minlength=n_classes)
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights
