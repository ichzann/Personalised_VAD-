"""Phase 1 synthesis engine: synthesize_scene(seed) -> (wav, labels, meta).

Given a seed, lay 1 target + N interferers on a single-mic timeline with gaps,
turn-taking and deliberate overlaps, then emit a per-frame 4-class label array
(ROADMAP §3, §4). "Record" = target-only.

Design choices, all in the name of simplicity (ROADMAP §8):
- **Trimming is energy-based.** VCTK is studio-clean, so a simple RMS threshold
  finds the true speech onset/offset. We do NOT pull in a VAD model here (that's a
  Phase 2 backbone decision). ALL timing/labels come from these trimmed extents,
  never raw file boundaries (ROADMAP §4 step 1).
- **Noise bed is optional (Phase 5).** Pass `noise_files` to mix a scene-length
  FSD50K bed at a sampled SNR; omit it for the dry Phase 1 mixture (labels visibly
  match the audio). Reverb/RIR is still Phase 5 future work. The noise is drawn
  *after* all dialog placement, so one seed gives the SAME dialog with or without
  noise — a clean and a noisy cache differ only by the bed.
- **Labels on a fixed 10 ms grid.** Phase 2 aligns its feature grid to this.

Everything random flows from one seeded numpy Generator, so a seed fully determines
the scene (ROADMAP §8).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from src.synth.speaker_splits_example import enrollment_dialog_split

SR = 16000  # 16 kHz mono everywhere (ROADMAP §2)
LABEL_HOP = 160  # 10 ms per label frame at 16 kHz

# 4-class labels (ROADMAP §2). "record" = TARGET_ONLY.
SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP = 0, 1, 2, 3
CLASS_NAMES = {SILENCE: "silence", TARGET_ONLY: "target-only",
               OTHER_ONLY: "other-only", OVERLAP: "overlap"}


@dataclass
class SceneConfig:
    """Knobs for one scene. Defaults give a short, clearly-labeled demo dialog."""
    n_interferers_choices: tuple = (1, 2)   # how many other speakers (N≈1-3)
    n_turns_range: tuple = (8, 14)          # number of placed utterances
    gap_range_s: tuple = (0.15, 0.7)        # silence between non-overlapping turns
    overlap_prob: float = 0.35              # chance a turn overlaps the previous one
    overlap_frac_range: tuple = (0.2, 0.6)  # how much of the previous turn is covered
    tir_db_range: tuple = (-12.0, 12.0)     # target-to-interferer ratio (§3-C dial)
    target_turn_prob: float = 0.5           # how often the target takes a turn
    ref_rms: float = 0.06                   # each utterance normalized to this RMS
    snr_db_range: tuple = (5.0, 40.0)       # speech-to-noise ratio when a bed is added (Phase 5).
    # Wide span on purpose (ROADMAP §7 multi-condition): 5 dB = loud noise (hardest),
    # ~40 dB = faint quiet-room tone (nearly clean). Keeps the loud end for robustness
    # while adding a near-silent tail so the head also sees the low/no-noise regime.
    noise_crossfade_s: float = 0.1          # crossfade when concatenating short noise clips
    n_enroll: int = 20                      # target utterances reserved for enrollment
    # Energy trim: keep frames within TRIM_DB of the utterance's peak RMS.
    trim_db: float = 30.0
    trim_margin_s: float = 0.02             # small pad so we don't clip plosives


# ----------------------------------------------------------------------------- #
# Loading + trimming
# ----------------------------------------------------------------------------- #

def _frame_rms(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Root-mean-square energy per short frame."""
    n = 1 + max(0, (len(signal) - win) // hop)
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        frame = signal[i * hop : i * hop + win]
        rms[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
    return rms


def energy_trim(signal: np.ndarray, cfg: SceneConfig) -> tuple[np.ndarray, int, int]:
    """Trim leading/trailing silence by energy. Returns (trimmed, onset, offset).

    A frame counts as speech if its RMS is within `trim_db` of the loudest frame.
    We then cut to the first/last speech frame (plus a small margin).
    """
    win, hop = 400, 160  # 25 ms window, 10 ms hop
    rms = _frame_rms(signal, win, hop)
    peak = rms.max()
    threshold = peak * (10.0 ** (-cfg.trim_db / 20.0))
    speech = np.where(rms >= threshold)[0]
    if len(speech) == 0:  # silent clip (shouldn't happen for VCTK) -> keep as is
        return signal, 0, len(signal)

    margin = int(cfg.trim_margin_s * SR)
    onset = max(0, speech[0] * hop - margin)
    offset = min(len(signal), (speech[-1] + 1) * hop + margin)
    return signal[onset:offset], onset, offset


def load_trimmed(path: Path, cfg: SceneConfig) -> np.ndarray:
    """Load a 16 kHz utterance, trim silence, normalize to the reference RMS."""
    signal, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR} (run resample first)"
    trimmed, _, _ = energy_trim(signal, cfg)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2) + 1e-12)
    return (trimmed * (cfg.ref_rms / rms)).astype(np.float32)


def _sentence_id(path: Path) -> str:
    """'p225_017.wav' -> '017'. VCTK is parallel (same id = same sentence text),
    so we keep ids distinct within a scene to avoid overlapping identical text
    (ROADMAP §4 known limitations)."""
    return path.stem.split("_", 1)[1]


# ----------------------------------------------------------------------------- #
# Background noise bed (Phase 5, ROADMAP §4 step 5)
# ----------------------------------------------------------------------------- #

def _rms(signal: np.ndarray) -> float:
    """Root-mean-square level of a signal (with a tiny floor to avoid /0)."""
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2) + 1e-12))


def _crossfade_concat(a: np.ndarray, b: np.ndarray, fade: int) -> np.ndarray:
    """Concatenate two mono signals with a short linear crossfade (hides the seam)."""
    if len(a) == 0:
        return b.copy()
    fade = min(fade, len(a), len(b))
    if fade == 0:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    blended = a[-fade:] * (1.0 - ramp) + b[:fade] * ramp
    return np.concatenate([a[:-fade], blended, b[fade:]])


def build_noise_bed(rng, length: int, noise_files: list, cfg: SceneConfig) -> np.ndarray:
    """Build a scene-length mono noise bed from FSD50K clips.

    Clips are shorter than a scene, so we concatenate random ones with small
    crossfades (no clicks at the seams, ROADMAP §4 step 5) until the bed is long
    enough, then trim to exactly `length`. All picks come from the seeded rng.
    """
    fade = int(cfg.noise_crossfade_s * SR)
    bed = np.zeros(0, dtype=np.float32)
    while len(bed) < length:
        path = noise_files[int(rng.integers(len(noise_files)))]
        clip, sr = sf.read(str(path), dtype="float32")
        assert sr == SR, f"{path} is {sr} Hz, expected {SR} (run resample first)"
        if clip.ndim > 1:                  # fold any stray stereo clip to mono
            clip = clip.mean(axis=1).astype(np.float32)
        bed = _crossfade_concat(bed, clip, fade)
    return bed[:length]


# ----------------------------------------------------------------------------- #
# Scene synthesis
# ----------------------------------------------------------------------------- #

def _pick_speakers(rng, split_speakers: list[str], n_interferers: int):
    """Pick 1 target + n distinct interferers from the same split (disjoint)."""
    chosen = rng.choice(split_speakers, size=n_interferers + 1, replace=False)
    chosen = [str(s) for s in chosen]  # plain str, not np.str_ (clean meta/JSON)
    return chosen[0], chosen[1:]


def synthesize_scene(seed: int, split_speakers: list[str], cfg: SceneConfig = None,
                     noise_files: list = None):
    """Build one labeled scene. Returns (wav float32, labels int array, meta dict).

    Pass `noise_files` (a list of 16 kHz FSD50K wav paths) to add a shared
    background-noise bed at a sampled SNR (Phase 5). Omit it for a dry scene.
    """
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(seed)

    # 1. Cast the scene: target + interferers, all from the given (disjoint) split.
    n_interferers = int(rng.choice(cfg.n_interferers_choices))
    target, interferers = _pick_speakers(rng, split_speakers, n_interferers)
    roster = [target] + interferers  # index 0 is always the target

    # Per-interferer gain from a per-scene TIR (the §3-C dial). The target itself
    # stays at gain 1.0 and we scale the interferer relative to it:
    #   positive dB -> interferer quieter than the target,
    #   negative dB -> interferer LOUDER than the target (gain > 1.0).
    # The two-sided range matters: a one-sided (target-always-louder) range lets the
    # head cheat with "loudest = target" instead of using the enrollment embedding.
    # Letting the target be the quieter speaker in some scenes makes that shortcut
    # wrong half the time, so it can't be learned. One TIR per scene keeps the
    # §3-C F1-vs-TIR curve well-defined.
    tir_db = float(rng.uniform(*cfg.tir_db_range))
    interferer_gain = 10.0 ** (-tir_db / 20.0)

    # 2. Reserve the target's enrollment utterances; draw dialog from the rest.
    enroll_files, target_dialog = enrollment_dialog_split(target, n_enroll=cfg.n_enroll)
    dialog_pools = {target: list(target_dialog)}
    for spk in interferers:
        _, pool = enrollment_dialog_split(spk, n_enroll=cfg.n_enroll)
        dialog_pools[spk] = list(pool)

    # 3. Choose a turn order: usually the target (target_turn_prob), otherwise a
    #    random interferer. A speaker MAY take consecutive gapped turns (real dialog
    #    has multi-sentence turns), but an *overlap* must be a different speaker —
    #    no one overlaps themselves (that would sum two clips of one voice and get
    #    mislabeled single-speaker while sounding like overlap to the mel branch).
    n_turns = int(rng.integers(cfg.n_turns_range[0], cfg.n_turns_range[1] + 1))
    used_sentence_ids: set[str] = set()
    events = []           # placed utterances: dicts with samples + role
    cursor = 0            # end of the timeline so far, in samples
    prev = None           # (start, end) of the previously placed event
    last_speaker = None

    for _ in range(n_turns):
        # Decide placement first: only an overlap forbids repeating the last
        # speaker; a gapped turn passes no exclusion so repeats are allowed.
        is_overlap = prev is not None and rng.random() < cfg.overlap_prob
        forbid = last_speaker if is_overlap else None
        speaker = _choose_speaker(rng, roster, target, interferers, forbid, cfg)
        path = _draw_utterance(rng, dialog_pools[speaker], used_sentence_ids)
        if path is None:
            continue  # ran out of distinct sentences for this speaker; skip turn
        used_sentence_ids.add(_sentence_id(path))
        sig = load_trimmed(path, cfg)
        dur = len(sig)

        # Place: overlap the tail of the previous turn, or leave a gap after it.
        if is_overlap:
            frac = rng.uniform(*cfg.overlap_frac_range)
            start = max(0, prev[1] - int((prev[1] - prev[0]) * frac))
        else:
            gap = int(rng.uniform(*cfg.gap_range_s) * SR)
            start = cursor + gap
        end = start + dur

        is_target = speaker == target
        events.append({
            "speaker": speaker, "role": "target" if is_target else "interferer",
            "path": str(path), "start": start, "end": end,
            "gain": 1.0 if is_target else interferer_gain,
        })
        prev = (start, end)
        cursor = max(cursor, end)
        last_speaker = speaker

    # 4. Render the mixture and the per-sample activity of target vs others.
    total = cursor + int(0.3 * SR)  # small tail of trailing silence
    mix = np.zeros(total, dtype=np.float32)
    target_active = np.zeros(total, dtype=bool)
    other_active = np.zeros(total, dtype=bool)

    for ev in events:
        sig = load_trimmed(Path(ev["path"]), cfg) * ev["gain"]
        s, e = ev["start"], ev["start"] + len(sig)
        mix[s:e] += sig
        if ev["role"] == "target":
            target_active[s:e] = True
        else:
            other_active[s:e] = True

    # Add a shared background-noise bed at a sampled SNR (Phase 5, ROADMAP §4 step 5).
    # Drawn here — AFTER all dialog placement — so a given seed yields the SAME dialog
    # with or without noise (clean vs noisy caches differ only by the bed). Noise does
    # not touch the labels: labels come from speech activity, not energy, so silence
    # frames stay `silence` even though the bed has energy there (VAD must reject it).
    snr_db = None
    if noise_files:
        snr_db = float(rng.uniform(*cfg.snr_db_range))
        speech_mask = target_active | other_active
        if speech_mask.any():
            speech_rms = _rms(mix[speech_mask])          # level over speech only
            bed = build_noise_bed(rng, total, noise_files, cfg)
            target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
            bed *= target_noise_rms / _rms(bed)          # scale bed to hit the SNR
            mix = mix + bed.astype(np.float32)

    # Avoid clipping with a single uniform rescale (does not move any boundary, and a
    # uniform scale leaves the SNR unchanged).
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak

    labels = _labels_from_activity(target_active, other_active)

    # Realized overlap ratio = overlapped-speech time / total speech time. This is the
    # quantity the diarization literature reports and matches to the target domain
    # (real dialog ~10-35%); per-turn `overlap_prob` only controls it indirectly, so
    # we measure it instead of assuming it (same stance as the F1-vs-TIR curve, §3-C).
    speech_frames = int(np.count_nonzero(labels != SILENCE))
    overlap_frames = int(np.count_nonzero(labels == OVERLAP))
    overlap_ratio = overlap_frames / speech_frames if speech_frames else 0.0

    meta = {
        "seed": seed, "sr": SR, "label_hop": LABEL_HOP,
        "target": target, "interferers": interferers, "tir_db": tir_db,
        "snr_db": snr_db,  # None for a dry scene; float when a noise bed was added
        "duration_s": total / SR, "n_events": len(events),
        "overlap_ratio": overlap_ratio,
        "enrollment_files": [str(p) for p in enroll_files],
        "events": events, "class_names": CLASS_NAMES,
    }
    return mix, labels, meta


def _choose_speaker(rng, roster, target, interferers, last_speaker, cfg):
    """Turn-taking: prefer target, else a random interferer.

    `last_speaker` is the speaker to exclude (pass None to allow a repeat). The
    caller passes the previous speaker only for overlaps, so gapped turns may
    repeat while overlaps always switch speakers.
    """
    if rng.random() < cfg.target_turn_prob and last_speaker != target:
        return target
    options = [s for s in interferers if s != last_speaker] or interferers
    return str(options[int(rng.integers(len(options)))])


def _draw_utterance(rng, pool, used_sentence_ids):
    """Pick a random utterance whose sentence id hasn't been used in this scene."""
    candidates = [p for p in pool if _sentence_id(p) not in used_sentence_ids]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _labels_from_activity(target_active, other_active):
    """Collapse per-sample activity to a per-frame 4-class label (frame centers)."""
    n_frames = len(target_active) // LABEL_HOP
    centers = np.arange(n_frames) * LABEL_HOP + LABEL_HOP // 2
    t = target_active[centers]
    o = other_active[centers]
    labels = np.full(n_frames, SILENCE, dtype=np.int8)
    labels[t & ~o] = TARGET_ONLY
    labels[~t & o] = OTHER_ONLY
    labels[t & o] = OVERLAP
    return labels
