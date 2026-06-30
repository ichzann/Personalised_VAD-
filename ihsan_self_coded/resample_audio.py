from pathlib import Path 

import soundfile as sf 
import torch 
import torchaudio
from tqdm import tqdm

TARGET_SR = 16000

REPO_ROOT = Path(__file__).resolve().parents[1]

JOBS = [
    (REPO_ROOT / "wav48", REPO_ROOT / "data" / "wav16k", True),
    (REPO_ROOT / "fsd50k_data", REPO_ROOT / "data" / "fsd16k", False)
]


def resample_file(src_path: Path, dst_path: Path) -> None: 
    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=True)

    mono = audio.mean(axis=1)

    if sr != TARGET_SR:
        waveform = torch.from_numpy(mono)
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        mono = waveform.numpy()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), mono, TARGET_SR, subtype="PCM_16")

def main() -> None:
    torch.manual_seed(0)

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