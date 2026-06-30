"""Phase 1 milestone demo: generate a labeled scene, save the wav + label plot.

This is the "done when" check for Phase 1 (ROADMAP §6): produce a labeled dialog and
confirm the labels visibly match the audio. Listen to the wav while looking at the
plot's label strip.

Run:  python -m src.synth.demo_phase1_example
Outputs (gitignored data/):
  data/phase1_demo/scene_seed<SEED>.wav
  data/phase1_demo/scene_seed<SEED>.png
  data/phase1_demo/scene_seed<SEED>.json   (meta)
"""

import json
from pathlib import Path

import soundfile as sf

from src.synth.speaker_splits_example import load_splits
from src.synth.synthesize_scene_example import SR, synthesize_scene
from src.synth.visualize_scene_example import class_summary, plot_scene

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "phase1_demo"

SEED = 10  # any int; the scene is fully determined by it (10 shows all 4 classes)


def main() -> None:
    splits = load_splits()
    # Synthesize from the TRAIN speakers (test/val stay unseen, ROADMAP §2).
    wav, labels, meta = synthesize_scene(SEED, splits["train"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"scene_seed{SEED}"
    sf.write(str(stem.with_suffix(".wav")), wav, SR, subtype="PCM_16")
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    plot_scene(wav, labels, meta, stem.with_suffix(".png"))

    print(f"target={meta['target']}  interferers={meta['interferers']}  "
          f"TIR={meta['tir_db']:.1f} dB  duration={meta['duration_s']:.1f}s  "
          f"events={meta['n_events']}")
    print(f"class fractions: {class_summary(labels)}")
    print(f"wrote {stem}.wav / .png / .json")


if __name__ == "__main__":
    main()
