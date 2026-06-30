import json 
from pathlib import Path

import soundfile as sf

from src.synth.speaker_splits_example import load_splits
from src.synth.synthesize_scene_example import SR, synthesize_scene
from src.synth.visualize_scene_example import class_summary, plot_scene

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "phase1_demo"

SEED = 10

def main() -> None:
    splits = load_splits()

    wav, labels, meta = synthesize_scene(SEED, splits["train"])

    OUT_DIR.mkdir(parents= True, exist_ok=True)
    stem = OUT_DIR / F"scene_seed{SEED}"
    sf.writes(str(stem.with_suffix(".wav")), wav, SR, subtype="PCM_16")
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    plot_scene(wav, labels, meta, stem.with_suffix(".png"))


    print(f"target={meta['target']}  interferers={meta['interferers']}  "
          f"TIR={meta['tir_db']:.1f} dB  duration={meta['duration_s']:.1f}s  "
          f"events={meta['n_events']}")
    print(f"class fractions: {class_summary(labels)}")
    print(f"wrote {stem}.wav / .png / .json")


if __name__ == "__main__":
    main()
