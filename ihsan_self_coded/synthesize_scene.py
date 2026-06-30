from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import soundfile as sf
from src.synth.speaker_splits_example import enrollment_dialog_split


SR = 16000
LABEL_HOP = 160 

SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP = 0,1,2,3
CLASS_NAMES = {SILENCE: "silence", TARGET_ONLY: "target-only", OTHER_ONLY: "other-only", OVERLAP: "overlap"}

@dataclass
class SceneConfig:
    """Knobs for one scene. Defaults give a short, clearly-labeled demo dialog."""
    n_interferers_choices: tuple = (1, 2)   # how many other speakers (N≈1-3)
    n_turns_range: tuple = (8, 14)          # number of placed utterances
    gap_range_s: tuple = (0.1, 0.7)        # silence between non-overlapping turns
    overlap_prob: float = 0.35              # chance a turn overlaps the previous one
    overlap_frac_range: tuple = (0.1, 0.6)  # how much of the previous turn is covered
    tir_db_range: tuple = (0.0, 12.0)       # target-to-interferer ratio (§3-C dial)
    target_turn_prob: float = 0.5           # how often the target takes a turn
    ref_rms: float = 0.06                   # each utterance normalized to this RMS
    n_enroll: int = 20                      # target utterances reserved for enrollment
    # Energy trim: keep frames within TRIM_DB of the utterance's peak RMS.
    trim_db: float = 30.0
    trim_margin_s: float = 0.02             # small pad so we don't clip plosives

def _frame_rms(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Root-mean-square energy per short frame."""
    n = 1 + max(0, (len(signal) - win) // hop)
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        frame = signal[i * hop : i * hop + win]
        rms[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
    return rms


