"""Batch feature caching: many scenes from a split -> cached .npz feature bundles.

Phase 2's demo cached one scene; Phase 3 (the baseline) and Phase 4 (training) need
a *set* of scenes. This loops over seeds for one split, synthesizes each scene
(deterministic from its seed, ROADMAP §8), runs the FROZEN backbones once (loaded a
single time and reused), and caches each feature bundle to disk (ROADMAP §2).

Speaker-disjoint by construction: every scene for split X draws only from X's
speakers (ROADMAP §2). Seeds are namespaced per split so train/val/test never share
a seed -> never collide.

Run:  python -m src.features.build_feature_cache_example --split val --n 20
Outputs (gitignored data/):
  data/features/<split>/scene_seed<SEED>.npz
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.features.extract_features_example import (
    extract_scene_features, load_embedder, load_vad_model, save_features,
)
from src.synth.speaker_splits_example import load_splits
from src.synth.synthesize_scene_example import synthesize_scene

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"
FEATURES_NOISY_DIR = REPO_ROOT / "data" / "features_noisy"
NOISE_DIR = REPO_ROOT / "data" / "fsd16k"

# Seed offset per split so the three splits use disjoint seed ranges (clarity, not
# correctness — speakers are already disjoint, but disjoint seeds avoid confusion).
SPLIT_SEED_BASE = {"train": 0, "val": 100_000, "test": 200_000}

# Noise-file split (Phase 5). Like speakers, noise files are held out by split so we
# never test on a noise clip seen in training (ROADMAP §4 "Noise/RIR sets also held
# out for eval"). Pure function of (sorted files, seed) -> reproducible.
NOISE_SPLIT_SEED = 4321


def noise_splits(noise_dir: Path = NOISE_DIR, seed: int = NOISE_SPLIT_SEED) -> dict:
    """Partition the FSD50K noise files into disjoint train/val/test lists (~70/15/15)."""
    files = sorted(noise_dir.glob("*.wav"))
    rng = np.random.default_rng(seed)
    order = files.copy()
    rng.shuffle(order)                       # deterministic given the seed
    n_test = int(0.15 * len(order))
    n_val = int(0.15 * len(order))
    return {
        "test": order[:n_test],
        "val": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }


def build_split(split: str, n_scenes: int, overwrite: bool = False,
                noisy: bool = False) -> list[Path]:
    """Synthesize + cache `n_scenes` feature bundles for one split. Returns the paths.

    `noisy=True` mixes a held-out FSD50K noise bed into every scene (Phase 5) and
    writes to `data/features_noisy/<split>` so the clean cache is untouched. Scenes
    use the same seeds as the clean cache, so the two differ only by the noise bed.
    """
    splits = load_splits()
    speakers = splits[split]
    out_dir = (FEATURES_NOISY_DIR if noisy else FEATURES_DIR) / split
    base = SPLIT_SEED_BASE[split]
    noise_files = noise_splits()[split] if noisy else None
    if noisy:
        print(f"noisy mode: {len(noise_files)} held-out noise files for '{split}'")

    print(f"loading frozen backbones (CPU)...")
    embedder = load_embedder()
    vad_model = load_vad_model()

    paths = []
    for i in tqdm(range(n_scenes), desc=f"caching {split}{' (noisy)' if noisy else ''}"):
        seed = base + i
        out_path = out_dir / f"scene_seed{seed}.npz"
        if out_path.exists() and not overwrite:
            paths.append(out_path)
            continue
        wav, labels, meta = synthesize_scene(seed, speakers, noise_files=noise_files)
        bundle = extract_scene_features(wav, labels, meta, embedder, vad_model)
        save_features(bundle, out_path)
        paths.append(out_path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--n", type=int, default=20, help="number of scenes to cache")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--noisy", action="store_true",
                    help="mix a held-out FSD50K noise bed (Phase 5); writes to data/features_noisy/")
    args = ap.parse_args()

    paths = build_split(args.split, args.n, args.overwrite, noisy=args.noisy)
    out_root = FEATURES_NOISY_DIR if args.noisy else FEATURES_DIR
    print(f"\ncached {len(paths)} {args.split} scenes -> {out_root / args.split}")


if __name__ == "__main__":
    main()
