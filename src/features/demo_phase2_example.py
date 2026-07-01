"""Phase 2 milestone demo: one scene -> e_target + x[t] sequence cached to disk.

This is the "done when" check for Phase 2 (ROADMAP §6): a single example yields the
enrollment embedding and an aligned per-frame feature sequence on disk. We
regenerate the scene from its seed (Phase 1 is deterministic), run the FROZEN
ECAPA + Silero backbones over it, and cache the feature bundle.

Run:  python -m src.features.demo_phase2_example
Outputs (gitignored data/):
  data/features/scene_seed<SEED>.npz   (logmel, emb, cos, vad, labels, e_target, ...)
"""

from pathlib import Path

import numpy as np

from src.features.extract_features_example import (
    extract_scene_features, load_embedder, load_vad_model, save_features,
)
from src.synth.speaker_splits_example import load_splits
from src.synth.synthesize_scene_example import synthesize_scene

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "features"

SEED = 10  # same scene as the Phase 1 demo (exercises all 4 classes)


def main() -> None:
    splits = load_splits()
    wav, labels, meta = synthesize_scene(SEED, splits["train"])
    print(f"scene seed={SEED}  target={meta['target']}  "
          f"interferers={meta['interferers']}  dur={meta['duration_s']:.1f}s")

    print("loading frozen backbones (first run downloads weights)...")
    embedder = load_embedder()
    vad_model = load_vad_model()

    print("extracting features...")
    bundle = extract_scene_features(wav, labels, meta, embedder, vad_model)

    out_path = OUT_DIR / f"scene_seed{SEED}.npz"
    save_features(bundle, out_path)

    T = bundle["logmel"].shape[0]
    print("\n--- cached feature bundle (all aligned to the 10 ms grid) ---")
    print(f"  frames T      = {T}  (~{T * 0.01:.1f}s)")
    print(f"  logmel        : {bundle['logmel'].shape}")
    print(f"  emb (e[t])    : {bundle['emb'].shape}")
    print(f"  cos (s[t])    : {bundle['cos'].shape}  "
          f"range [{bundle['cos'].min():.2f}, {bundle['cos'].max():.2f}]")
    print(f"  vad (p_speech): {bundle['vad'].shape}")
    print(f"  labels        : {bundle['labels'].shape}  "
          f"classes present {sorted(np.unique(bundle['labels']).tolist())}")
    print(f"  e_target      : {bundle['e_target'].shape}  "
          f"(||e_target|| = {np.linalg.norm(bundle['e_target']):.3f})")

    # Sanity: cosine should be higher on target-only frames than other-only frames.
    cos, lab = bundle["cos"], bundle["labels"]
    for name, cls in [("target-only", 1), ("other-only", 2), ("overlap", 3)]:
        m = lab == cls
        if m.any():
            print(f"  mean cos on {name:12s}: {cos[m].mean():+.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
