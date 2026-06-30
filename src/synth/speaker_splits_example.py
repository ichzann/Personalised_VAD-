"""Speaker-disjoint train/val/test splits (ROADMAP §2, §4).

The single most important guardrail for honest numbers: a speaker that appears in
test must NEVER appear in train (otherwise we measure memorization, not
generalization). We split by *speaker*, never by utterance.

The split is a pure function of (sorted speaker list, SPLIT_SEED), so it is fully
reproducible even though `data/splits.json` itself is gitignored — re-running this
file always reproduces the same split.

Run:  python -m src.synth.speaker_splits_example
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WAV16K_DIR = REPO_ROOT / "data" / "wav16k"
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"

# Fixed seed for the split itself, independent of any per-scene seed. Changing it
# reshuffles who is train/val/test, so we pin it once.
SPLIT_SEED = 1234

# ~80 / 14 / 15 of the 109 VCTK speakers (ROADMAP §4). val + test are held out;
# train is everything else.
N_VAL = 14
N_TEST = 15

# Each target speaker's utterances are partitioned into an enrollment pool and a
# dialog pool so a scene never reuses the same recording for both (no leakage,
# ROADMAP §4 step 2). The first N_ENROLL_POOL files (sorted) are enrollment-only.
N_ENROLL_POOL = 20


def list_speakers(wav16k_dir: Path = WAV16K_DIR) -> list[str]:
    """Return sorted speaker ids (the sub-directory names, e.g. 'p225')."""
    speakers = [p.name for p in wav16k_dir.iterdir() if p.is_dir()]
    return sorted(speakers)


def build_speaker_splits(
    wav16k_dir: Path = WAV16K_DIR, seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Partition speakers into disjoint train/val/test lists (deterministic)."""
    speakers = list_speakers(wav16k_dir)

    # Shuffle a copy with a dedicated RNG so the order is fixed by the seed alone.
    rng = np.random.default_rng(seed)
    order = speakers.copy()
    rng.shuffle(order)  # in-place, deterministic given the seed

    test = sorted(order[:N_TEST])
    val = sorted(order[N_TEST : N_TEST + N_VAL])
    train = sorted(order[N_TEST + N_VAL :])
    return {"train": train, "val": val, "test": test}


def enrollment_dialog_split(
    speaker: str,
    wav16k_dir: Path = WAV16K_DIR,
    n_enroll: int = N_ENROLL_POOL,
) -> tuple[list[Path], list[Path]]:
    """Split one speaker's utterances into (enrollment pool, dialog pool).

    Enrollment = first `n_enroll` files (sorted); dialog = the rest. Disjoint by
    construction, so a scene's dialog audio is never reused as enrollment audio.
    """
    files = sorted((wav16k_dir / speaker).glob("*.wav"))
    return files[:n_enroll], files[n_enroll:]


def save_splits(splits: dict[str, list[str]], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path = SPLITS_PATH) -> dict[str, list[str]]:
    """Load the split, building + saving it on first use."""
    if not path.exists():
        splits = build_speaker_splits()
        save_splits(splits, path)
    return json.loads(path.read_text())


def main() -> None:
    splits = build_speaker_splits()
    save_splits(splits)
    for name, spk in splits.items():
        print(f"{name:5s}: {len(spk):3d} speakers -> {spk}")

    # Sanity: the three sets must be pairwise disjoint.
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)), "speaker leaked across splits!"
    print(f"\nOK: {len(all_ids)} speakers, all disjoint. Saved to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
