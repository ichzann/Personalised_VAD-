# Target Person Voice — Project Roadmap

> North-star document for humans **and** LLM assistants. Read this first before
> writing code or proposing changes. If a decision here is wrong or outdated,
> update *this file* in the same change — do not let code and roadmap drift apart.

Last updated: 2026-06-24

---

## 1. What we are building (one paragraph)

Given a short **enrollment** sample of a target person speaking cleanly, build a
model + pipeline that takes a **dialog recorded on a single microphone** (several
people) and outputs a **time-stamped timeline of when the target is speaking
*alone***. Silence, other speakers, and the target *overlapping* others all count
as "not target." The extracted target-only segments are meant for downstream
analysis, which is **out of scope** here — we deliver the detector/pipeline only.

This task has a name in the literature: **Personal VAD / Target-Speaker Voice
Activity Detection (TS-VAD)**. We follow that framing deliberately so we can reuse
known baselines and metrics instead of reinventing them. See §10.

## 2. Locked decisions (from project kickoff)

These are settled. Changing one is a scope change → update this section + §7.

| Decision | Choice | Why |
|---|---|---|
| Output | **Time-stamped timeline** (per-frame labels → target-only segments) | Needed to actually extract/record target-only audio |
| Compute | **Apple Silicon / CPU only** | No GPU → freeze pretrained nets, train only a small head |
| Domain target | **Real single shared mic eventually** | Needs reverb/RIR augmentation + a real test set |
| Project purpose | **Learning / portfolio** | Favor clarity, demonstrable milestones, honest ablations |
| Live vs offline | **Offline first, streaming-compatible by design** | Best accuracy + debuggability now; streaming as a stretch |
| Label granularity | **4-class** {silence, target-only, other-only, overlap} | Overlap-as-negative needs explicit modeling; labels are free from synthesis |
| Mel pathway | **Small 1D CNN over frequency** (plain `Conv1d`, 1–2 layers) | Cheap, streaming-safe (frame-local), matched to overlap; no 2D/MobileNet tricks |
| Sample rate | **16 kHz mono everywhere** | Pretrained VAD/embedder expect it; 48 kHz breaks them |
| Splits | **Speaker-disjoint** train/val/test | Otherwise we measure memorization, not generalization |
| Code style | **Simplicity first** — plain, explainable code | Learning/portfolio project; must be able to defend every line (see §8) |

## 3. Core mechanism (how it actually works)

The pretrained VAD only tells speech from silence — it **cannot** tell the target
from other speakers. Speaker discrimination comes from comparing each dialog
window's **speaker embedding** to the **enrollment embedding** (cosine similarity).
Overlap ("target + other") is detected as *concurrent speech*, not as target
absence — which is why we model it as its own class.

```
ENROLLMENT (run once per target person)
  target wav --(resample 16k)--> VAD (frozen) keep speech
            --> speaker embedder (frozen) over ~1.5s windows
            --> average --> e_target  (one vector)

DIALOG (per window t, hop ~0.2-0.5s, with overlap)
  dialog wav --(resample 16k)-->
     p_speech[t]  = VAD speech prob            (frozen)
     e[t]         = sliding speaker embedding   (frozen)
     s[t]         = cos(e[t], e_target)          <- explicit comparison (see A)
     m[t]         = mel frame -> small 1D CNN     <- overlap/energy feature (see B + Mel pathway)
     (optional)   = overlap/OSD cue
  x[t] = [ e[t] (or low-dim proj), s[t], m[t], p_speech[t], ... ]

HEAD  (the ONLY trained part)
  x[t] sequence --> (Bi)LSTM --> per-frame softmax over 4 classes
                --> map: "record" = argmax==target-only
                --> post-process: hysteresis + min-segment-duration smoothing
                --> timeline of target-only segments
```

**A. Hand the model the comparison — don't make it rediscover it.**
The "is this frame the target?" decision is learned by the head, but we feed it the
comparison instead of hoping it infers one: each dialog window carries
`s[t] = cos(e[t], e_target)` *as an input feature*, alongside `e_target` itself.
Concatenating raw enrollment + dialog streams and hoping the net learns to compare
is far less sample-efficient — a bad trade on CPU with a small head and synthetic
data. Keep `s[t]`; ablate it away later only to prove it earned its place.

**B. Overlap is a labeled class, and the mel is what makes it learnable.**
Frames are labeled 4-way {silence, target-only, other-only, overlap} (free from
synthesis); "record" = target-only. Key principle: **labels say *what* to output,
features decide *what's learnable*** — a correct label cannot be learned if the
inputs don't separate the classes. During overlap the (mixed) embedding sits
*between* target and interferer, so the embedding alone is ambiguous. The
**mel-spectrogram is load-bearing here**: additive energy, dual pitch tracks and
denser harmonics are what separate "target alone" from "target + other." So the mel
earns its place mainly via overlap/energy (the embedding already covers speaker
identity) — do not drop it.

**C. The one hard limit is the target-to-interferer ratio (TIR) — so measure it.**
The single case labels cannot rescue: when the interferer is *much quieter* than the
target, the mixture ≈ clean target in both embedding and mel, the classes genuinely
overlap in feature space, and even the best model errs (Bayes error, not a bug).
This is governed entirely by the **TIR set during synthesis** (§4) — a dial, not a
wall. So sweep TIR over a realistic range and **report target-only F1 as a function
of TIR** (§5); that curve is the honest characterization of overlap rejection, not a
single number. Corollary for the real-mic goal (§2): the model only learns to reject
the overlaps it saw, so the synthesis distribution (TIR range, #interferers, reverb)
must span real rooms or the rejection won't transfer.

**The TIR range must be two-sided** (decided: `tir_db ∈ [-12, +12]`, one draw per
scene). Positive dB = interferer quieter than the target; negative dB = interferer
*louder* than the target. A one-sided range (target always ≥ interferer) leaks a
loudness shortcut: the head can learn "loudest speaker = target" and skip the
enrollment-embedding comparison entirely — which then fails on real mics where the
target may sit farther from the mic. Letting the target be the quieter speaker in
some scenes makes that rule wrong ~half the time, so it can't be learned, while a
single TIR per scene keeps the F1-vs-TIR curve well-defined. (Time-*varying* level
within a scene is deliberately **not** done here — it would give a scene multiple
TIRs and blur the curve; per-utterance level jitter is deferred to Phase 5, §7.)

**Mel pathway (decided): a small 1D CNN over frequency.**
For each frame, run a tiny `Conv1d` over the mel/frequency bins (1–2 layers, a few
filters) to extract the local spectral pattern that signals overlap, then flatten to
a per-frame vector `m[t]`. The conv operates *within a single frame* (over frequency,
not time), so it adds no lookahead and the exact same code runs offline and
streaming. The LSTM does all the temporal modelling. Keep it plain `Conv1d` — no 2D
convs, no depthwise-separable / MobileNet tricks (see §8: simplicity first).

**Windowing / sizing.**
Speaker embeddings need context: use a sliding window for `e[t]` with a small hop
(~0.1–0.25 s) — many embeddings, each still reliable (shorter than ~0.5 s and the
embedding becomes noise, not signal). The small hop is correct *for us* because we
emit a per-frame timeline, not one embedding per segment (clustering diarization).
The **window length** is the real knob: the diarization literature's offline default
is **~3.0 s** (e.g. ECAPA on AMI), while ~1.0 s is favored only for streaming /
end-to-end tracking. Since we are offline-first, the window ablation must span
**{1.0, 1.5, 3.0} s** (do not stop at 1.5 s — too short costs identity stability).
The mel / `m[t]` stream is finer (~10–30 ms frames). Align both onto one common frame
grid before building `x[t]` (hold/repeat the coarse embedding across the fine grid).
Window length vs hop is an ablation (§6 Phase 5).

**Frozen backbone — two viable options (pick one in Phase 2):**
- **`pyannote.audio`** — pretrained segmentation incl. *overlapped-speech detection*
  + speaker embeddings. Directly helps the overlap problem. Needs the HF token.
- **SpeechBrain ECAPA-TDNN** (speaker embeddings) **+ Silero VAD**. Lighter, no OSD
  out of the box (overlap learned by the head from features).

## 4. Data

**Sources (already present):**
- `wav48/` — VCTK corpus: 109 speakers, ~231 read utterances each, 48 kHz mono,
  studio-clean, ~2 s/utterance. Used to build both enrollment and dialog speakers.
- `fsd50k_data/` — 4,351 noise wavs. Used as background noise field.


**Known limitations to keep in mind:** VCTK is read (not spontaneous) speech, short
utterances, no natural overlaps, and it's a *parallel* corpus (speakers read the
same sentences — avoid pairing identical sentence text across target/interferer).

**Synthesis engine (Phase 1) — given a seed, produce one labeled scene:**
1. **Trim each source utterance first.** VCTK clips have leading/trailing silence,
   so VAD/energy-trim every utterance to its true speech onset/offset. *All* timing
   and labels are derived from these trimmed extents, never from raw file
   boundaries — otherwise silence padding is mislabeled as speech and every
   timestamp downstream misaligns.
2. Pick 1 target + N interferers (N≈1–3), all from the *same split*,
   speaker-disjoint. The target's **dialog** utterances must be *different
   recordings* from its **enrollment** utterances (no audio reuse → no leakage).
3. Lay utterances on a timeline with controlled gaps, turn-taking and deliberate
   overlaps — the **target appears in the dialog** alongside interferers. Sample the
   target-to-interferer gain (TIR, §3-C) for overlap regions. Record exact
   boundaries → frame-level **4-class** label array.
   - *Turn-taking:* a speaker **may take consecutive gapped turns** (real dialog has
     multi-sentence turns; rigid no-repeat alternation is an unnatural grammar a
     sequence head could exploit). The one exception is **overlaps must switch
     speaker** — a speaker overlapping themselves sums two clips of one voice, which
     gets mislabeled single-speaker yet looks like overlap to the mel branch.
     `target_turn_prob` keeps a mild target bias so target-only stays well-sampled.
   - *Overlap as a measured ratio:* overlap is controlled by a per-turn probability,
     but the quantity the literature reports and matches to the target domain is the
     **realized overlap ratio** (overlapped-speech time ÷ total speech time; real
     dialog ≈ 10–35%, CHiME-6 ~34%). So **log it per scene in `meta`** and check the
     dataset lands in a realistic band — same measure-don't-assume stance as the
     F1-vs-TIR curve (§3-C). Sweepable later as an F1-vs-overlap-ratio ablation.
4. (For real-mic realism) convolve each speaker with a **room impulse response**;
   same room per scene, different positions per speaker. RIRs via `pyroomacoustics`
   (synthetic) or a recorded RIR set.
5. Build the noise bed to the scene length: if a clip is too short, **concatenate
   several FSD50K clips with short crossfades** (smooth the seams, no clicks). Mix to
   one channel and add this **shared** noise field at a sampled SNR.
6. Augmentation asymmetry: dialog gets heavier noise/reverb; enrollment lighter
   (mostly clean, some noisy) so the embedding is robust either way.
7. Seed + determinism so datasets are reproducible.

**Splits:** by speaker, e.g. ~80 train / ~14 val / ~15 test. Test speakers (target
*and* interferers) must be unseen in training. Noise/RIR sets also held out for eval. (once the pipeline is fully build, we have more data to populate)

## 5. Evaluation protocol

- **Primary:** frame-level precision / recall / **F1 for the `target-only` class**.
- **Use-case framing:** report **false-trigger rate** (recording non-target audio)
  vs **miss rate** (dropping target audio). Decide which is costlier for the
  downstream analysis and tune the decision threshold on val accordingly.
- **Segment-level:** after smoothing, compare predicted vs true target segments
  (e.g. detection error rate, boundary tolerance).
- **Never** report bare accuracy — silence/non-target dominate the frames.
- Always evaluate on the **speaker-disjoint test set** + (Phase 6) a **real**
  recorded test set.

## 6. Phased roadmap (each phase ends in a demonstrable milestone)

> Work top-to-bottom. Don't start a phase until the previous milestone is met.

- **Phase 0 — Setup & spec.** Repo structure, env (`requirements.txt`), resample
  all audio to 16 kHz, write the task spec + metric definitions. Rotate the HF token.
  *Done when:* clean env + 16 kHz data + this roadmap agreed.

- **Phase 1 — Synthesis engine.** `synthesize_scene(seed) -> (wav, labels, meta)`
  with speaker-disjoint splits. Sanity-listen + plot labels over the spectrogram.
  *Done when:* you generate a labeled dialog and the labels visibly match the audio.

- **Phase 2 — Feature extraction (frozen models).** Choose backbone (§3), compute
  enrollment embedding + dialog feature sequence + cosine sim, **cache to disk**.
  *Done when:* one example yields `e_target` and an `x[t]` sequence on disk.

- **Phase 3 — Non-learned baseline.** Pure cosine-similarity threshold personal VAD
  (no LSTM) + simple smoothing. *Done when:* you have val P/R/F1 from the simplest
  possible method — the number every later phase must beat.

- **Phase 4 — Train the LSTM head.** Small BiLSTM + the 1D-CNN mel pathway, 4-class,
  on cached features. *Done when:* it beats the Phase 3 baseline; report F1 +
  false-trigger rate.

- **Phase 5 — Overlap & robustness.** Improve the `overlap` class; add RIR/noise
  augmentation; ablations (enrollment length 3/5/10/20 s, embedding window {1.0/1.5/
  3.0 s} × hop, ± 1D-CNN mel feature, ± OSD cue, F1-vs-TIR curve, F1-vs-overlap-ratio).
  *Done when:* you have an ablation table.

- **Phase 6 — Real-data validation.** Record/collect ~10–20 min of real single-mic
  dialog, hand-label, evaluate, analyze the synthetic→real gap. *Done when:* you
  have real-audio numbers + an honest error analysis.

- **Phase 7 — Streaming (stretch).** Same model, made causal and run on a rolling
  buffer. Only two things change from offline: BiLSTM → **causal LSTM**
  (`bidirectional=False`) and chunked inference. The mel 1D-CNN and the
  `cos(e[t], e_target)` conditioning are unchanged.

  ```
  Enrollment (once, offline):  target wav -> embedder(frozen) -> e_target  [stored]
  Streaming (per hop ~0.25s, backward-looking only):
    trailing buffer (~1.5s) -> embedder(frozen) -> e[t]
    s[t] = cos(e[t], e_target)
    mel frame -> small 1D CNN (over freq) -> m[t]
    VAD -> p_speech[t]
    x[t] = [ m[t], e[t], s[t], p_speech[t] ]
    causal LSTM -> P(target-only|t) -> smoothing -> append target segment
  ```

  Zero lookahead latency, but the target decision *warms up* as the trailing buffer
  fills; the embedder over that buffer is the main compute cost (use a light one such
  as Resemblyzer). Since this use case tolerates seconds of delay, a simpler
  **chunked pseudo-live** path (keep the offline BiLSTM, run it on rolling ~5–10 s
  windows) is a valid, more accurate fallback if causal accuracy disappoints.
  *Done when:* a live/near-live demo that still identifies the target.

## 7. Open questions / deferred decisions

- Exact backbone (pyannote vs SpeechBrain+Silero) — decide in Phase 2.
- Speaker count / split sizes — confirm whether all 109 VCTK speakers or a subset.
- Number of interferers N and overlap-rate distribution — tune in Phase 1/5.
- Whether `other-only` and `overlap` stay separate classes or merge — revisit if
  overlap proves too hard to learn.
- How to source RIRs for "real single mic" realism (synthetic `pyroomacoustics`
  vs recorded RIR corpus) — decide before Phase 5.
- Decision-threshold target (favor low false-trigger or low miss) — set in Phase 5.
- Per-utterance / time-varying **level jitter** within a scene (robustness against a
  fixed per-file loudness) — deferred to Phase 5 augmentation. Phase 1 uses a single
  two-sided TIR per scene (§3-C) to keep the F1-vs-TIR curve clean.
- **Scene length is ~20–40 s, vs the 2–4 min "simulated conversations" in the
  diarization literature** — a conscious, scope-justified divergence: we train on
  cropped ~5–10 s windows, target single-mic *dialog* (not meetings), and are
  CPU-only. The implication is that training-audio volume and turn variety must come
  from generating **many short scenes**, not a few long ones. Revisit (allow up to
  ~60 s) only if longer-range turn dynamics prove to matter.

## 8. Conventions & guardrails (for humans and LLMs)

> **Simplicity first — this is a learning / portfolio project.** Prefer plain,
> readable code over clever abstractions. Avoid advanced or exotic techniques, heavy
> design patterns, metaprogramming, premature optimization, and large framework
> machinery where a small script does the job. If there is a simple way and a
> sophisticated way, take the simple one and add a one-line comment on *why*. Being
> able to read and defend every line in a portfolio review beats a marginal
> accuracy/speed gain.

**Do:**
- **The author writes the real source files; the assistant only writes `*_example`
  reference versions** (e.g. `src/foo_example.py`, not `src/foo.py`). This is a
  learning project — the author re-implements each file line by line while studying
  the example. Run the `_example` file when code must be executed to verify a phase.
  Docs/config (this file, `METRICS.md`, `requirements.txt`, `.gitignore`) are written
  normally.
- Write **simple, explainable** code (see banner)
- Keep all audio at **16 kHz mono**; resample at ingestion, not ad hoc.
- Keep heavy pretrained nets **frozen**; only the head trains. Cache features.
- Make data generation **seeded/deterministic**.
- Maintain **speaker-disjoint** splits; never leak a test speaker into train.
- Report per-class F1 + false-trigger/miss, not accuracy.
- Update §2 and §7 in the same change that alters scope.

**Don't:**
- Don't add heavy/2D CNNs or extra trainable branches — the mel pathway is a
  deliberately tiny 1D `Conv1d` over frequency (§3); anything bigger needs a
  measured gain.
- Don't expect the VAD to separate speakers; it only gates speech vs silence.
- Don't compute a single global embedding for the dialog — it must be per-window.
- Don't evaluate on utterance-level splits or commit the HF token.

## 9. Repo layout (target — create as phases need them)

```
data/            # generated scenes, cached features (gitignored)
src/
  synth/         # Phase 1 synthesis engine
  features/      # Phase 2 frozen-backbone feature extraction
  models/        # Phase 4 LSTM head + 1D-CNN mel pathway
  eval/          # metrics, baseline, plots
notebooks/       # exploration (dataset.ipynb lives here conceptually)
ROADMAP.md       # this file
CLAUDE.md        # short pointer + hard constraints for LLMs
requirements.txt
```

## 10. References (orient, don't copy blindly)

- Ding et al., *Personal VAD: Speaker-Conditioned Voice Activity Detection* (2019/20).
- Medennikov et al., *Target-Speaker Voice Activity Detection* (Interspeech 2020).
- Speaker embeddings: d-vector (Variani 2014), x-vector (Snyder 2018),
  **ECAPA-TDNN** (Desplanques 2020).
- Tooling: `pyannote.audio` (VAD/OSD/embeddings/diarization), **Silero VAD**,
  **SpeechBrain** (`spkrec-ecapa-voxceleb`), `Resemblyzer` (GE2E d-vector),
  `pyroomacoustics` (RIR), `audiomentations` / `torch-audiomentations` (aug).
- Data: VCTK corpus; FSD50K (Fonseca et al.).