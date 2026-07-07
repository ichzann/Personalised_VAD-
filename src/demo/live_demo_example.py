"""Live Personal VAD demo (ROADMAP §6, Phase 6/7): enroll a target from the mic,
then listen to a live dialog and light up while the target is speaking ALONE.

Three modes:

1) Enroll (once per target person — record them speaking cleanly ~15-20 s):
     python -m src.demo.live_demo_example --enroll --seconds 20
   Saves data/enroll/e_target.npz (+ the raw wav, so you can listen back).

2) Live detection (the demo):
     python -m src.demo.live_demo_example
   Streams the default mic. Every 0.25 s hop it computes the SAME per-frame
   features the model was trained on — log-mel, a TRAILING 1.5 s ECAPA embedding,
   cosine to e_target, Silero speech prob — and steps the causal LSTM, carrying
   its hidden state across hops. The terminal shows a green dot while the target
   speaks alone. Ctrl-C to stop.

3) File test (no mic — sanity-check the streaming pipeline on a known scene):
     python -m src.demo.live_demo_example --wav data/phase1_demo/scene_seed10.wav \
         --target data/features/scene_seed10.npz
   (--target accepts any .npz containing an `e_target` array, so a cached scene
   bundle works: the demo then tracks that scene's target speaker.)

IMPORTANT — use a model trained on TRAILING-window features. The default cache
uses centered embedding windows (0.75 s lookahead), which do not exist live:

    python -m src.features.build_feature_cache_example --split train --n 150 --noisy --trailing
    python -m src.features.build_feature_cache_example --split val   --n 20  --noisy --trailing
    python -m src.models.train_example \
        --train-dir data/features_noisy_trailing/train \
        --val-dir   data/features_noisy_trailing/val \
        --out       data/models/personal_vad_noisy_trailing.pt

A centered-trained model will still run (same shapes) but degraded — it was
trained on embeddings that could peek 0.75 s into the future.

Design notes (simplicity-first, ROADMAP §8):
- All streaming state lives in one small class: the audio ring buffer, Silero's
  chunk stream, and the LSTM hidden state. No threads beyond the audio callback.
- Expect ~1.5 s warm-up (the trailing window filling) and ~0.5 s of switching
  delay from the debounce — the live stand-in for min-segment smoothing.
"""

import argparse
import queue
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.eval_model_example import load_model
from src.features.extract_features_example import (
    LABEL_HOP, SR, FeatureConfig, compute_logmel, embed_segment,
    enrollment_embedding, load_embedder, load_vad_model, mel_filterbank,
)
from src.synth.synthesize_scene_example import CLASS_NAMES, TARGET_ONLY

REPO_ROOT = Path(__file__).resolve().parents[2]
ENROLL_DIR = REPO_ROOT / "data" / "enroll"
DEFAULT_TARGET = ENROLL_DIR / "e_target.npz"
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "personal_vad_noisy_trailing.pt"

HOP_S = 0.25                       # decision rate: one model step per 0.25 s
HOP = int(HOP_S * SR)              # 4000 samples = 25 label frames
HOP_FRAMES = HOP // LABEL_HOP      # 25 frames of 10 ms per hop
WIN = int(1.5 * SR)                # trailing embedding window (must match training)
EDGE = 2                           # skip the last 2 STFT frames of the buffer: they
                                   # touch the zero-padded edge, so their mel differs
                                   # from training. Costs 20 ms latency, keeps the
                                   # emitted frames identical to offline extraction.
VAD_CHUNK = 512                    # Silero's fixed chunk (32 ms at 16 kHz)
DEBOUNCE_HOPS = 2                  # dot flips only after ~0.5 s of agreement

GREEN, DIM, RESET = "\033[92m", "\033[2m", "\033[0m"


class StreamingDetector:
    """Feeds a live 16 kHz stream through the frozen backbones + the causal head.

    push(samples) accepts arbitrary-length blocks and returns one decision dict per
    completed 0.25 s hop. Everything that must survive between hops lives here:
    the trailing audio buffer, Silero's chunk stream, the LSTM hidden state, and
    the debounced on/off state of the indicator.
    """

    def __init__(self, model, embedder, vad_model, e_target: np.ndarray):
        self.model = model
        self.embedder = embedder
        self.vad_model = vad_model
        self.e_target = e_target.astype(np.float32)
        self.fb = mel_filterbank(FeatureConfig())

        self.buf = np.zeros(0, dtype=np.float32)  # trailing audio (bounded below)
        self.abs_total = 0                        # samples received since start
        self.hop_end = WIN                        # stream position of the next hop's end
        self.vad_leftover = np.zeros(0, dtype=np.float32)
        self.chunk_probs = []                     # Silero prob per 512-sample chunk
        self.lstm_state = None                    # carried across hops (causal LSTM)
        self.on = False                           # debounced indicator state
        self._consec = 0

    @torch.no_grad()
    def _vad_consume(self, samples: np.ndarray) -> None:
        """Run Silero over new samples in fixed 512-sample chunks. Its internal
        state carries across calls — the same regime as offline extraction."""
        x = np.concatenate([self.vad_leftover, samples])
        n = len(x) // VAD_CHUNK
        for i in range(n):
            chunk = torch.from_numpy(x[i * VAD_CHUNK:(i + 1) * VAD_CHUNK])
            self.chunk_probs.append(float(self.vad_model(chunk, SR).item()))
        self.vad_leftover = x[n * VAD_CHUNK:]

    def _vad_at(self, abs_sample: int) -> float:
        """Speech prob of the chunk containing an absolute sample position."""
        if not self.chunk_probs:
            return 0.0
        return self.chunk_probs[min(abs_sample // VAD_CHUNK, len(self.chunk_probs) - 1)]

    @torch.no_grad()
    def _process_hop(self, win: np.ndarray, abs_end: int) -> dict:
        """One 0.25 s step: features for the newest 25 frames -> causal LSTM -> dot."""
        # Mel over the whole trailing window; keep the newest 25 fully-contexted
        # frames (EDGE shifts us off the zero-padded right edge, see constant).
        logmel = compute_logmel(win, FeatureConfig(), self.fb)
        frames = logmel[-(HOP_FRAMES + EDGE):-EDGE]              # (25, 40)

        # One trailing embedding for the hop, held across its 25 frames — the same
        # hold/repeat the model saw in (trailing) training. cos is the s[t] feature.
        emb = embed_segment(win, self.embedder)                  # ends at abs_end
        cos = float(emb @ self.e_target)

        # Per-frame Silero prob at each frame center (absolute stream position).
        first_center = abs_end - (HOP_FRAMES + EDGE - 1) * LABEL_HOP
        centers = first_center + np.arange(HOP_FRAMES) * LABEL_HOP
        vad = np.array([self._vad_at(int(c)) for c in centers], dtype=np.float32)

        mel_t = torch.from_numpy(frames)[None]                       # (1,25,40)
        emb_t = torch.from_numpy(np.tile(emb, (HOP_FRAMES, 1)))[None]  # (1,25,192)
        cos_t = torch.full((1, HOP_FRAMES), cos)
        vad_t = torch.from_numpy(vad)[None]
        et_t = torch.from_numpy(self.e_target)[None]                 # (1,192)

        logits, self.lstm_state = self.model.step(mel_t, emb_t, cos_t, vad_t, et_t,
                                                  self.lstm_state)
        pred = logits[0].argmax(-1)                              # (25,) 4-class
        frac = float((pred == TARGET_ONLY).float().mean())
        majority = int(pred.mode().values)

        # Debounce: flip the dot only after DEBOUNCE_HOPS hops agree (~0.5 s) —
        # the live stand-in for hysteresis + min-segment smoothing.
        want = frac >= 0.5
        if want != self.on:
            self._consec += 1
            if self._consec >= DEBOUNCE_HOPS:
                self.on, self._consec = want, 0
        else:
            self._consec = 0

        # emb (the trailing 192-d ECAPA embedding for this hop) is surfaced for the
        # optional live embedding-map viz (src/demo/live_viz_example.py). Detection
        # itself ignores it, so this is the only hook the visualizer needs.
        return {"t": abs_end / SR, "on": self.on, "frac": frac,
                "cos": cos, "cls": majority, "emb": emb}

    def push(self, samples: np.ndarray) -> list[dict]:
        """Add new audio; returns one decision per completed hop (possibly none)."""
        samples = samples.astype(np.float32)
        self._vad_consume(samples)
        self.abs_total += len(samples)
        self.buf = np.concatenate([self.buf, samples])

        out = []
        while self.abs_total >= self.hop_end:      # enough audio for this hop
            # Slice the trailing WIN samples ending at stream position hop_end.
            end = len(self.buf) - (self.abs_total - self.hop_end)
            out.append(self._process_hop(self.buf[end - WIN:end], self.hop_end))
            self.hop_end += HOP
        # Drop audio older than the next hop will ever need (bounded memory).
        keep = WIN + (self.abs_total - self.hop_end) + HOP
        if len(self.buf) > keep:
            self.buf = self.buf[-keep:]
        return out


def show(r: dict) -> None:
    """Repaint one status line: the dot + what the model currently thinks."""
    dot = f"{GREEN}● TARGET SPEAKING{RESET}" if r["on"] else \
          f"{DIM}○ ---------------{RESET}"
    print(f"\r {r['t']:7.1f}s  {dot}  class={CLASS_NAMES[r['cls']]:<11s}"
          f" cos={r['cos']:+.2f}  target-frames={r['frac']:>4.0%}  ",
          end="", flush=True)


def run_enroll(seconds: float, out_path: Path, embedder, vad_model) -> None:
    """Record enrollment from the mic -> e_target.npz (+ raw wav for listening back)."""
    import sounddevice as sd
    import soundfile as sf

    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".wav")
    input(f"Press Enter, then have the target speak normally for {seconds:.0f}s...")
    print("recording...")
    rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(wav_path), rec[:, 0], SR)

    # Same enrollment path as training: Silero keeps speech, ECAPA windows averaged.
    e_target = enrollment_embedding([wav_path], embedder, vad_model, FeatureConfig())
    np.savez(out_path, e_target=e_target)
    print(f"saved {out_path}  (audio kept at {wav_path})")


def run_mic(det: StreamingDetector) -> None:
    """Live mode: mic callback fills a queue; the main loop consumes it."""
    import sounddevice as sd

    q = queue.Queue()

    def callback(indata, n_frames, t, status):
        if status:
            print(f"\n[audio] {status}", flush=True)
        q.put(indata[:, 0].copy())

    print(f"listening... (dot needs ~{WIN / SR:.1f}s of audio to warm up; Ctrl-C to stop)")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        try:
            while True:
                for r in det.push(q.get()):
                    show(r)
        except KeyboardInterrupt:
            print("\nstopped.")


def run_wav(det: StreamingDetector, path: Path) -> None:
    """File mode: stream a wav through the identical hop pipeline (no mic needed).
    Prints throughput (must be >1x real-time) and the detected target segments."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR} (resample first)"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    results = []
    t0 = time.time()
    for start in range(0, len(wav), HOP):
        for r in det.push(wav[start:start + HOP]):
            results.append(r)
            show(r)
    elapsed = time.time() - t0
    dur = len(wav) / SR
    print(f"\nprocessed {dur:.1f}s of audio in {elapsed:.1f}s "
          f"({dur / elapsed:.1f}x real-time)")

    # Debounced ON/OFF transitions -> target-only segments.
    segs, start_t = [], None
    for r in results:
        if r["on"] and start_t is None:
            start_t = r["t"] - HOP_S
        elif not r["on"] and start_t is not None:
            segs.append((start_t, r["t"]))
            start_t = None
    if start_t is not None:
        segs.append((start_t, results[-1]["t"]))
    print("detected target-only segments:")
    for a, b in segs:
        print(f"  {a:6.1f}s - {b:6.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enroll", action="store_true",
                    help="record the target from the mic and save e_target")
    ap.add_argument("--seconds", type=float, default=20,
                    help="enrollment recording length (--enroll)")
    ap.add_argument("--target", default=str(DEFAULT_TARGET),
                    help=".npz containing e_target (enroll output or a cached scene bundle)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="trained head checkpoint (train on --trailing features!)")
    ap.add_argument("--wav", default=None,
                    help="stream this 16 kHz wav instead of the mic (sanity test)")
    args = ap.parse_args()

    print("loading frozen backbones (CPU)...")
    embedder = load_embedder()
    vad_model = load_vad_model()

    if args.enroll:
        run_enroll(args.seconds, DEFAULT_TARGET, embedder, vad_model)
        return

    if not Path(args.target).exists():
        raise SystemExit(f"no enrollment at {args.target} — run --enroll first "
                         f"(or pass --target a cached scene .npz)")
    if not Path(args.model).exists():
        raise SystemExit(f"no model at {args.model} — build a trailing cache and "
                         f"train it first (see this file's docstring)")

    e_target = np.load(args.target)["e_target"]
    model = load_model(Path(args.model))
    det = StreamingDetector(model, embedder, vad_model, e_target)

    if args.wav:
        run_wav(det, Path(args.wav))
    else:
        run_mic(det)


if __name__ == "__main__":
    main()
