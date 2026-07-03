"""Phase 2 feature extraction with FROZEN backbones (ROADMAP §3, §6).

Backbone decision (ROADMAP §7, locked in Phase 2): **SpeechBrain ECAPA-TDNN**
(speaker embeddings) **+ Silero VAD** (speech/silence). Lighter on CPU than
pyannote and no HF token needed; it has no overlap detector, but overlap is
learned by the head from the mel pathway (§3-B), which is the whole point of the
1D-CNN-over-frequency branch.

Both nets are FROZEN (ROADMAP §2): we only ever run them in inference mode and
cache their outputs to disk. The ONLY trained part of the project is the LSTM head
+ the 1D-CNN mel pathway (Phase 4).

What one scene becomes on disk (all aligned to the Phase 1 10 ms label grid so we
never have to resample the labels):

    logmel : (T, N_MELS)  raw log-mel frames    -> trainable 1D CNN runs on these
                                                    in Phase 4 (we cache mel, not m[t])
    emb    : (T, 192)     ECAPA embedding e[t]   (sliding window, held across grid)
    cos    : (T,)         s[t] = cos(e[t], e_target)   <- the explicit comparison (§3-A)
    vad    : (T,)         p_speech[t] from Silero
    labels : (T,)         the Phase 1 4-class labels, carried through
    e_target : (192,)     enrollment embedding (one vector per target)

Design choices (ROADMAP §8 simplicity-first):
- **One common 10 ms grid.** The mel/VAD are naturally fine; the embedding is
  coarse (one per 0.25 s). We hold/repeat the coarse embedding across the fine grid
  (ROADMAP §3 windowing). x[t] is assembled from these arrays in Phase 4.
- **Centered embedding windows.** Offline-first (§2): we center the 1.5 s window on
  each frame. Streaming (Phase 7) swaps to a trailing window; nothing else changes.
- **Manual log-mel via torch.stft + a mel filterbank.** No extra heavy dependency,
  and every line is explainable in a portfolio review (the mel is load-bearing, §3-B).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SR = 16000          # 16 kHz mono everywhere (ROADMAP §2)
LABEL_HOP = 160     # 10 ms grid at 16 kHz (must match Phase 1's LABEL_HOP)


@dataclass
class FeatureConfig:
    """Knobs for feature extraction. Defaults follow ROADMAP §3."""
    # Mel: 25 ms window, 10 ms hop (hop == LABEL_HOP so mel lands on the label grid).
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = LABEL_HOP
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 8000.0           # Nyquist at 16 kHz
    # Sliding speaker embedding (ROADMAP §3 windowing). 1.5 s default; the {1.0/1.5/
    # 3.0 s} window ablation is Phase 5. Small hop -> many still-reliable embeddings.
    emb_win_s: float = 1.5
    emb_hop_s: float = 0.25
    # Window placement: "centered" = offline default (0.75 s lookahead);
    # "trailing" = window ENDS at the frame -> no lookahead, required for the live
    # demo / streaming (ROADMAP §6 Phase 7). Train on the same mode you run.
    emb_mode: str = "centered"
    # Enrollment: keep only speech (Silero) before averaging windows into e_target.
    vad_speech_thresh: float = 0.5


# --------------------------------------------------------------------------- #
# Log-mel (manual, so the load-bearing mel branch is fully transparent)
# --------------------------------------------------------------------------- #

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> torch.Tensor:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1).

    Standard Slaney-style triangles spaced evenly on the mel scale. Built once and
    reused for every frame/scene.
    """
    n_freqs = cfg.n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, SR / 2.0, n_freqs)

    # n_mels + 2 mel-spaced edges -> n_mels overlapping triangles.
    mel_min, mel_max = _hz_to_mel(cfg.fmin), _hz_to_mel(cfg.fmax)
    mel_points = np.linspace(mel_min, mel_max, cfg.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((cfg.n_mels, n_freqs), dtype=np.float32)
    for m in range(cfg.n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / (center - left + 1e-9)
        falling = (right - fft_freqs) / (right - center + 1e-9)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return torch.from_numpy(fb)


def compute_logmel(wav: np.ndarray, cfg: FeatureConfig, fb: torch.Tensor) -> np.ndarray:
    """Log-mel spectrogram on the 10 ms grid. Returns (n_frames, n_mels) float32."""
    x = torch.from_numpy(wav.astype(np.float32))
    # center=True pads by n_fft//2 so frame f is centered at sample f*hop -> the mel
    # grid lines up with the label-frame centers from Phase 1.
    spec = torch.stft(
        x, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
        window=torch.hann_window(cfg.win_length), center=True, return_complex=True,
    )
    power = spec.abs() ** 2                      # (n_freqs, n_frames)
    mel = fb @ power                             # (n_mels, n_frames)
    logmel = torch.log(mel + 1e-6).T             # (n_frames, n_mels)
    return logmel.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Frozen backbones (loaded once, inference only)
# --------------------------------------------------------------------------- #

def load_vad_model():
    """Silero VAD (frozen). Returns a callable model run on 16 kHz audio."""
    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    model.eval()
    return model


def load_embedder():
    """SpeechBrain ECAPA-TDNN speaker embedder (frozen)."""
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="data/pretrained/ecapa",          # cached weights (gitignored data/)
        run_opts={"device": "cpu"},               # CPU only (ROADMAP §2)
    )
    model.eval()
    return model


@torch.no_grad()
def speech_prob_per_frame(wav: np.ndarray, vad_model, n_frames: int) -> np.ndarray:
    """Silero speech probability resampled onto the 10 ms grid -> (n_frames,).

    Silero scores audio in fixed 512-sample chunks (32 ms at 16 kHz). We get one
    prob per chunk, expand it back over its samples, then read the value at each
    label-frame center so VAD lines up with mel/labels.
    """
    chunk = 512
    x = torch.from_numpy(wav.astype(np.float32))
    per_sample = np.zeros(len(wav), dtype=np.float32)
    for start in range(0, len(wav) - chunk + 1, chunk):
        prob = float(vad_model(x[start:start + chunk], SR).item())
        per_sample[start:start + chunk] = prob

    centers = np.arange(n_frames) * LABEL_HOP + LABEL_HOP // 2
    centers = np.clip(centers, 0, len(per_sample) - 1)
    return per_sample[centers]


@torch.no_grad()
def embed_segment(segment: np.ndarray, embedder) -> np.ndarray:
    """One L2-normalized ECAPA embedding (192-d) for a short waveform segment."""
    x = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)  # (1, samples)
    emb = embedder.encode_batch(x).squeeze().numpy().astype(np.float32)  # (192,)
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


def sliding_embeddings(wav: np.ndarray, embedder, cfg: FeatureConfig,
                       n_frames: int) -> np.ndarray:
    """Sliding ECAPA embeddings on each emb-hop, held onto the 10 ms grid.

    Returns (n_frames, 192). We compute one embedding per `emb_hop_s` (coarse) and
    repeat it across the fine frames it covers (ROADMAP §3: hold the coarse
    embedding across the fine grid). Padding decides the window placement: either
    way `padded[c : c+win]` is one window per coarse center c — centered on c
    (offline) or ending at c (trailing / streaming, zero lookahead).
    """
    win = int(cfg.emb_win_s * SR)
    hop = int(cfg.emb_hop_s * SR)
    half = win // 2
    if cfg.emb_mode == "trailing":
        padded = np.pad(wav, (win, 0))       # window ENDS at sample c
    else:
        padded = np.pad(wav, (half, half))   # window centered on sample c

    grid_centers = (np.arange(n_frames) * LABEL_HOP + LABEL_HOP // 2)
    out = np.zeros((n_frames, 192), dtype=np.float32)

    # Coarse embedding centers in samples; assign each fine frame to its nearest one.
    coarse_centers = np.arange(0, len(wav), hop)
    cache = {}
    for c in coarse_centers:
        seg = padded[c:c + win]                 # centered window around sample c
        cache[c] = embed_segment(seg, embedder)
    # Nearest coarse center for every fine frame (held/repeated embedding).
    idx = np.clip(np.round(grid_centers / hop).astype(int), 0, len(coarse_centers) - 1)
    for f in range(n_frames):
        out[f] = cache[coarse_centers[idx[f]]]
    return out


# --------------------------------------------------------------------------- #
# Enrollment embedding e_target
# --------------------------------------------------------------------------- #

@torch.no_grad()
def enrollment_embedding(enroll_paths, embedder, vad_model, cfg: FeatureConfig) -> np.ndarray:
    """Average ECAPA over speech windows of the enrollment audio -> e_target (192,).

    Run once per target person (ROADMAP §3). We keep only Silero-speech regions so
    leading/trailing silence in VCTK clips doesn't dilute the embedding, slide
    `emb_win_s` windows, embed each, and average + L2-normalize.
    """
    import soundfile as sf

    win = int(cfg.emb_win_s * SR)
    hop = int(cfg.emb_hop_s * SR)
    embs = []
    for path in enroll_paths:
        sig, sr = sf.read(str(path), dtype="float32")
        assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
        n_frames = len(sig) // LABEL_HOP
        speech = speech_prob_per_frame(sig, vad_model, n_frames) >= cfg.vad_speech_thresh
        for start in range(0, max(1, len(sig) - win + 1), hop):
            center_frame = (start + win // 2) // LABEL_HOP
            if center_frame < len(speech) and speech[center_frame]:
                embs.append(embed_segment(sig[start:start + win], embedder))
        # Short clip with no full window: embed whatever speech there is.
        if len(sig) < win and speech.any():
            embs.append(embed_segment(sig, embedder))

    if not embs:
        raise RuntimeError("no speech windows found in enrollment audio")
    mean = np.mean(embs, axis=0)
    return (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Top-level: one scene -> cached feature bundle
# --------------------------------------------------------------------------- #

def extract_scene_features(wav: np.ndarray, labels: np.ndarray, meta: dict,
                           embedder, vad_model, cfg: FeatureConfig = None) -> dict:
    """Turn one Phase 1 scene into the aligned feature arrays (all on the 10 ms grid)."""
    cfg = cfg or FeatureConfig()
    fb = mel_filterbank(cfg)

    logmel = compute_logmel(wav, cfg, fb)            # (T, n_mels)
    n_frames = logmel.shape[0]

    # Align Phase 1 labels to the mel length (they share the 10 ms grid; clip/pad by 1).
    labels = labels[:n_frames]
    if len(labels) < n_frames:
        labels = np.pad(labels, (0, n_frames - len(labels)), constant_values=0)

    vad = speech_prob_per_frame(wav, vad_model, n_frames)        # (T,)
    emb = sliding_embeddings(wav, embedder, cfg, n_frames)       # (T, 192)

    e_target = enrollment_embedding(meta["enrollment_files"], embedder, vad_model, cfg)
    cos = emb @ e_target                                         # (T,) both L2-normalized

    return {
        "logmel": logmel, "emb": emb, "cos": cos.astype(np.float32),
        "vad": vad.astype(np.float32), "labels": labels.astype(np.int8),
        "e_target": e_target, "seed": meta["seed"], "target": meta["target"],
        "tir_db": meta["tir_db"], "sr": SR, "label_hop": LABEL_HOP,
    }


def save_features(bundle: dict, out_path: Path) -> None:
    """Cache one scene's features to a compressed .npz (ROADMAP §2: cache to disk)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **bundle)
