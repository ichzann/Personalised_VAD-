import json 
from pathlib import Path
import numpy as np 

REPO_ROOT = Path(__file__).resolve().parents[2]
wav16k_DIR = REPO_ROOT / "data" / "wav16k"
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"

SPLIT_SEED = 1234

N_VAL = 14
N_TEST = 15

N_ENROLL_POOL = 20

def list_speakers(wav16k_dir: Path = wav16k_DIR) -> list[str]:
    speakers = [p.name for p in wav16k_dir.iterdir() if p.is_dir()]
    return sorted(speakers)

def build_speaker_splits(wav16k_dir: Path = wav16k_DIR, seed: int = SPLIT_SEED): 

    speakers = list_speakers(wav16k_dir)

    rng = np.random.default_rng(seed)
    order = speakers.copy()
    rng.shuffle(order)

    test = sorted(order[:N_TEST])
    val = sorted(order[N_TEST : N_TEST + N_VAL])
    train = sorted(order[N_TEST + N_VAL :])
    return {"train": train, "val": val, "test": test}

def enrollment_dialog_split(speaker, wav16k_dir: Path = wav16k_DIR, n_enroll = N_ENROLL_POOL)
    
    files = sorted((wav16k_dir / speaker).glob("*wav"))
    return files[:n_enroll], files[n_enroll:]

def save_splits(splits, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path):
    if not path.exists():
        splits = build_speaker_splits()
        save_splits(splits, path)
    return json.loads(path.read_text())

def main():
    splits = build_speaker_splits()
    save_splits(splits)
    for name, spk in splits.item():
        print(f"{name:5s}: {len(spk):3d} speakers -> {spk}")
    
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)), "speaker leaked across splits!"
    print(f"\nOK: {len(all_ids)} speakers, all disjoint. Saved to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
