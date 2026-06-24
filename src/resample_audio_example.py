"""Resample all source audio to 16 kHz mono once, at ingestion (ROADMAP §8).

Everything downstream assumes 16 kHz mono (the pretrained VAD/embedder expect it),
so we make one clean, deterministic copy here instead of resampling ad hoc later.

  wav48/<spk>/<utt>.wav        (48.0 kHz)  -> data/wav16k/<spk>/<utt>.wav
  fsd50k_data/<clip>.wav       (44.1 kHz)  -> data/fsd16k/<clip>.wav

This step ONLY resamples + downmixes to mono. It does NOT trim silence or
normalize — silence trimming happens in Phase 1 synthesis, where all timing and
labels are derived from the trimmed extents (ROADMAP §4 step 1). Keeping this
script pure means the 16 kHz copy is a faithful, reversible-in-meaning version of
the source.

Idempotent: existing outputs are skipped, so re-running only fills in gaps.
"""

from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# 16 kHz mono everywhere — hard constraint from ROADMAP §2 (pretrained nets break
# at 48 kHz). Set once here so no other module picks its own sample rate.
TARGET_SR = 16000

REPO_ROOT = Path(__file__).resolve().parents[1]

# (source dir, output dir, whether to recurse into per-speaker subdirs)
JOBS = [
    (REPO_ROOT / "wav48", REPO_ROOT / "data" / "wav16k", True),
    (REPO_ROOT / "fsd50k_data", REPO_ROOT / "data" / "fsd16k", False),
]


def resample_file(src_path: Path, dst_path: Path) -> None:
    """Load one wav, downmix to mono, resample to 16 kHz, save as Int16 PCM.

    We use soundfile for read/write (plain libsndfile, no extra runtime deps) and
    torchaudio only for the high-quality resample kernel.
    """
    # always_2d -> shape (samples, channels) so the mono path is uniform.
    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=True)

    # Downmix to mono by averaging channels (VCTK/FSD here are already mono,
    # but average is the safe, obvious choice if any clip is stereo).
    mono = audio.mean(axis=1)  # shape: (samples,)

    if sr != TARGET_SR:
        waveform = torch.from_numpy(mono)
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        mono = waveform.numpy()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Save 16-bit PCM (same depth as the sources) for compact, lossless storage.
    sf.write(str(dst_path), mono, TARGET_SR, subtype="PCM_16")


def main() -> None:
    torch.manual_seed(0)  # determinism, even though resampling has no randomness

    for src_root, dst_root, recurse in JOBS:
        if not src_root.exists():
            print(f"[skip] source not found: {src_root}")
            continue

        pattern = "*/*.wav" if recurse else "*.wav"
        src_files = sorted(src_root.glob(pattern))

        converted = 0
        skipped = 0
        for src_path in tqdm(src_files, desc=src_root.name, unit="file"):
            rel = src_path.relative_to(src_root)
            dst_path = dst_root / rel
            if dst_path.exists():
                skipped += 1
                continue
            resample_file(src_path, dst_path)
            converted += 1

        print(
            f"{src_root.name}: {len(src_files)} source files "
            f"-> {dst_root} (converted {converted}, skipped {skipped})"
        )


if __name__ == "__main__":
    main()
