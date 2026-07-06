"""Live speaker-embedding map for the Personal VAD demo (companion to
`live_demo_example.py`). While the detector streams, this draws each hop's ECAPA
embedding as a moving point in 2-D, colored by the model's class, so you can WATCH
the embedding walk between speakers as the dialog unfolds.

Why this is a separate file (ROADMAP §8 simplicity-first): the detection pipeline
does not change. `StreamingDetector` already computes the trailing embedding every
hop; the only hook is that `_process_hop` now also *returns* it (`"emb"`). This
file just consumes the per-hop dicts and plots them. Nothing in features, model, or
training is touched.

Two ideas do the work:

1) **Projector = PCA, NOT t-SNE.** t-SNE is the wrong tool for a *live* view: it is
   fit-transform only (no `.transform()` for a new point), so every hop would force
   a full refit, and its layout reshuffles on every refit — points teleport instead
   of "following." PCA is parametric and stable: fit the 2 principal axes on a
   buffer of embeddings, then project each new embedding with a single matmul, so it
   lands in a CONSISTENT place. It is ~10 lines of numpy (SVD), no new dependency.
   (t-SNE / UMAP are great as a periodic *still* snapshot of the whole buffer, but
   that is a separate, opt-in job — not the live tracker.)

2) **The "other" class is really several people — so cluster it.** ROADMAP §3: the
   VAD cannot separate speakers; identity lives in the ECAPA embedding. So we run a
   tiny ONLINE clustering over the other-only embeddings (nearest centroid by
   cosine; join if close enough, else start a new speaker) and color them Other #1,
   Other #2, ... This literally shows the embedding carrying speaker identity, and
   gives a live "how many other voices have I heard" count. A nice free confirmation
   of the project's thesis: OVERLAP embeddings are mixtures, so they land *between*
   the green target cluster and an Other cluster (ROADMAP §3-B) — you can see it.

Run (needs an enrollment + a trailing-trained model, same as the live demo):

    # sanity-check on a known scene, paced to real time so you can watch it:
    python -m src.demo.live_viz_example --wav data/phase1_demo/scene_seed10.wav \
        --target data/features/scene_seed10.npz

    # live from the mic (enroll first via live_demo_example.py --enroll):
    python -m src.demo.live_viz_example

Design notes:
- Silence hops are dropped (a near-silence embedding is junk and would only add
  noise to the projection) — we gate on the model's own class, which we already have.
- PCA is refit occasionally and each refit is sign-aligned to the previous axes, so
  the view does not flip (eigenvector signs are arbitrary). Redraws are throttled and
  the point buffer is bounded, mirroring the ring-buffer discipline in the detector,
  so the visualizer keeps up with real time.
"""

import argparse
import queue
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.demo.live_demo_example import (
    DEFAULT_MODEL, DEFAULT_TARGET, HOP, REPO_ROOT, StreamingDetector, show,
)
from src.eval.eval_model_example import load_model
from src.features.extract_features_example import SR, load_embedder, load_vad_model
from src.synth.synthesize_scene_example import (
    CLASS_NAMES, OTHER_ONLY, OVERLAP, SILENCE, TARGET_ONLY,
)

# Fixed colors for the two singleton classes (match visualize_scene_example.py);
# other-only is colored per discovered speaker from OTHER_PALETTE instead.
TARGET_RGB = to_rgb("#2ca02c")     # green = record (target-only)
OVERLAP_RGB = to_rgb("#d62728")    # red = overlap (lands between target & an 'other')
OTHER_PALETTE = [to_rgb(c) for c in (
    "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",   # orange, purple, brown, pink
    "#17becf", "#bcbd22", "#1f77b4", "#7f7f7f",   # cyan, olive, blue, gray
)]


class PCAProjector:
    """2-D PCA: fit principal axes on a buffer, project new points by matmul.

    Parametric and stable (unlike t-SNE): a fixed fit maps every embedding to a
    consistent location, so points move smoothly instead of jumping. Refit as the
    buffer grows and sign-align to the previous axes so the view does not flip.
    """

    def __init__(self):
        self.mean = None
        self.components = None          # (2, D) top-2 principal axes

    def fit(self, X: np.ndarray) -> None:
        if len(X) < 3:                  # need a few points for a meaningful plane
            return
        mean = X.mean(axis=0)
        # Top-2 right singular vectors of the centered data = principal axes.
        _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
        comps = vt[:2].copy()
        # Eigenvector signs are arbitrary; flip each to agree with the previous
        # fit so the whole cloud does not mirror when we refit.
        if self.components is not None:
            for i in range(2):
                if np.dot(comps[i], self.components[i]) < 0:
                    comps[i] = -comps[i]
        self.mean, self.components = mean, comps

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components is None:
            return np.zeros((len(X), 2), dtype=np.float32)
        return (X - self.mean) @ self.components.T


class SpeakerClusters:
    """Online leader clustering of 'other' embeddings -> distinct interferers.

    Streaming and dependency-free: for each new other-only embedding, take cosine to
    every centroid; join the nearest if it clears `thresh` (updating that centroid as
    a running mean, renormalized to the unit sphere), else start a new speaker. ECAPA
    embeddings are L2-normalized, so cosine == dot. `thresh` ~0.5 is a reasonable
    ECAPA start point but is data-dependent — expose it as a knob (--thresh).
    """

    def __init__(self, thresh: float = 0.5, min_count: int = 3):
        self.thresh = thresh
        self.min_count = min_count      # support needed to count as a real speaker
        self.centroids = []             # list of (D,) unit vectors
        self.counts = []

    def assign(self, emb: np.ndarray) -> int:
        if self.centroids:
            sims = [float(emb @ c) for c in self.centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.thresh:
                n = self.counts[best]
                c = (self.centroids[best] * n + emb) / (n + 1)
                self.centroids[best] = c / (np.linalg.norm(c) + 1e-9)
                self.counts[best] += 1
                return best
        self.centroids.append(emb.copy())
        self.counts.append(1)
        return len(self.centroids) - 1

    def n_confident(self) -> int:
        """Clusters with enough support to trust as a real speaker (debounces
        one-off singletons from transition/overlap frames)."""
        return sum(1 for n in self.counts if n >= self.min_count)


class LiveEmbeddingViz:
    """Accumulates per-hop embeddings and repaints a live PCA scatter.

    Feed it each decision dict from `StreamingDetector.push()` via `update(r)`. It
    drops silence hops, clusters the other-only ones, and (throttled) reprojects the
    bounded buffer with PCA and redraws.
    """

    REDRAW_S = 0.3        # repaint at most ~3x/s (projection is cheap; drawing isn't)
    REFIT_EVERY = 20      # refit PCA axes every N new points (they stabilize fast)

    def __init__(self, e_target: np.ndarray, thresh: float = 0.5,
                 max_points: int = 1500):
        self.e_target = e_target.astype(np.float32)
        self.pca = PCAProjector()
        self.clusters = SpeakerClusters(thresh=thresh)
        self.max_points = max_points            # bound memory (~6 min of hops)

        self.embs, self.cls, self.cid = [], [], []   # per-point buffers
        self._since_fit = 0
        self._last_draw = 0.0
        self._last_r = None
        self._setup_fig()

    def _setup_fig(self) -> None:
        import matplotlib
        # In a notebook the mic loop BLOCKS the kernel, so a native GUI window won't
        # repaint mid-loop (especially the macOS backend) — you'd just see the empty
        # starting figure. With `%matplotlib inline` we instead redraw the figure IN
        # PLACE via an IPython display handle, which refreshes in every Jupyter
        # frontend. Auto-detect the mode from the active backend so the same class
        # works both in the notebook (inline) and as a script (native window / Agg).
        self.inline = "inline" in matplotlib.get_backend().lower()
        self._dh = None
        if not self.inline:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5))
        self.ax.set_title("live speaker-embedding map (PCA of ECAPA)")
        self.ax.set_xlabel("PC 1"); self.ax.set_ylabel("PC 2")
        # Empty collections we mutate in place each redraw (fast; no re-add).
        self.scat = self.ax.scatter([], [], s=26)                       # all points
        self.cur = self.ax.scatter([], [], s=200, facecolors="none",    # current dot
                                   edgecolors="black", linewidths=1.6, zorder=5)
        self.star = self.ax.scatter([], [], marker="*", s=420, c="gold",  # enrollment
                                    edgecolors="black", linewidths=1.0, zorder=6)
        self.status = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                   va="top", ha="left", fontsize=9, family="monospace")
        # Static legend hint (other-only is really colored per discovered speaker).
        self.ax.legend(handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="black", markersize=15, label="target enroll"),
            Patch(color=TARGET_RGB, label="target-only"),
            Patch(color=OVERLAP_RGB, label="overlap"),
            Patch(color=OTHER_PALETTE[0], label="other #1"),
            Patch(color=OTHER_PALETTE[1], label="other #2"),
        ], loc="upper right", fontsize=8, framealpha=0.9)
        self.fig.tight_layout()
        if self.inline:
            from IPython.display import display
            self._dh = display(self.fig, display_id=True)  # updated in place each redraw
        else:
            plt.show(block=False)

    def update(self, r: dict) -> None:
        """Consume one hop dict. Silence is skipped so the map freezes when nobody
        speaks; otherwise the point is buffered and (throttled) the view repaints."""
        if r["cls"] == SILENCE:                 # junk embedding — don't fit or plot
            return
        emb = np.asarray(r["emb"], dtype=np.float32)
        cid = self.clusters.assign(emb) if r["cls"] == OTHER_ONLY else -1
        self.embs.append(emb); self.cls.append(r["cls"]); self.cid.append(cid)
        if len(self.embs) > self.max_points:    # keep buffers bounded
            self.embs = self.embs[-self.max_points:]
            self.cls = self.cls[-self.max_points:]
            self.cid = self.cid[-self.max_points:]
        self._since_fit += 1
        self._last_r = r

        now = time.time()
        if now - self._last_draw >= self.REDRAW_S:
            self._redraw()
            self._last_draw = now

    def _point_colors(self) -> np.ndarray:
        """RGBA per point: class/cluster color, with a recency fade (old -> faint)."""
        n = len(self.cls)
        ages = np.linspace(0.2, 1.0, n) if n > 1 else np.array([1.0])
        out = np.zeros((n, 4), dtype=np.float32)
        for i, (c, k) in enumerate(zip(self.cls, self.cid)):
            if c == TARGET_ONLY:
                rgb = TARGET_RGB
            elif c == OVERLAP:
                rgb = OVERLAP_RGB
            else:                               # OTHER_ONLY -> per-speaker color
                rgb = OTHER_PALETTE[k % len(OTHER_PALETTE)]
            out[i, :3], out[i, 3] = rgb, ages[i]
        return out

    def _redraw(self) -> None:
        X = np.asarray(self.embs, dtype=np.float32)
        # Refit occasionally; include e_target so the enroll star stays in frame.
        if self.pca.components is None or self._since_fit >= self.REFIT_EVERY:
            self.pca.fit(np.vstack([X, self.e_target[None]]))
            self._since_fit = 0

        P = self.pca.transform(X)                       # (N, 2) all points
        et = self.pca.transform(self.e_target[None])    # (1, 2) enrollment star
        self.scat.set_offsets(P)
        self.scat.set_facecolors(self._point_colors())
        self.cur.set_offsets(P[-1:])                    # newest point (highlight ring)
        self.star.set_offsets(et)

        # Set limits from the projected coordinates ourselves: autoscale_view() does
        # NOT pick up a scatter's offsets, so relim() would leave everything off-screen.
        allp = np.vstack([P, et])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        pad = 0.08 * (hi - lo) + 1e-6                   # small margin (avoid 0 width)
        self.ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        self.ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

        r = self._last_r
        self.status.set_text(
            f"t={r['t']:6.1f}s  now={CLASS_NAMES[r['cls']]:<11s} cos={r['cos']:+.2f}\n"
            f"other speakers heard: {self.clusters.n_confident()}"
        )
        if self.inline:
            self._dh.update(self.fig)           # refresh the in-notebook figure in place
        else:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)                    # let the native GUI event loop breathe

    def finish(self, out_png: Path = None) -> None:
        """Final repaint, save a PNG for the portfolio, and keep the window up."""
        if not self.embs:
            print("no speech seen — nothing to show.")
            return
        self._redraw()
        out_png = out_png or (REPO_ROOT / "data" / "viz" / "live_embedding_map.png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\nother speakers heard this session: {self.clusters.n_confident()}")
        print(f"saved final embedding map -> {out_png}")
        if self.inline:
            plt.close(self.fig)                 # stop the inline backend re-showing it
        else:
            plt.ioff()
            plt.show()                          # block so you can inspect the window


def run_mic_viz(det: StreamingDetector, viz: LiveEmbeddingViz) -> None:
    """Live mic mode: same callback/queue as live_demo, plus viz.update per hop."""
    import sounddevice as sd

    q = queue.Queue()

    def callback(indata, n_frames, t, status):
        if status:
            print(f"\n[audio] {status}", flush=True)
        q.put(indata[:, 0].copy())

    print("listening... speak, then have others speak; Ctrl-C (or close plot) to stop.")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        try:
            while True:
                for r in det.push(q.get()):
                    if not viz.inline:          # in a notebook the map's own status
                        show(r)                 # text replaces the console dot
                    viz.update(r)
        except KeyboardInterrupt:
            print("\nstopped.")
    viz.finish()


def run_wav_viz(det: StreamingDetector, viz: LiveEmbeddingViz, path: Path,
                realtime: bool = True) -> None:
    """File mode: stream a wav through the identical hop pipeline. Paced to real time
    by default so the animation is watchable (--no-realtime to run flat out)."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR} (resample first)"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    t0 = time.time()
    for start in range(0, len(wav), HOP):
        for r in det.push(wav[start:start + HOP]):
            if not viz.inline:
                show(r)
            viz.update(r)
        if realtime:                            # sleep so hop time tracks wall time
            lag = (start + HOP) / SR - (time.time() - t0)
            if lag > 0:
                plt.pause(min(lag, 0.25))       # plt.pause also pumps the GUI
    viz.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=str(DEFAULT_TARGET),
                    help=".npz with e_target (enroll output or a cached scene bundle)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="trained head checkpoint (train on --trailing features!)")
    ap.add_argument("--wav", default=None,
                    help="stream this 16 kHz wav instead of the mic")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="cosine threshold for online clustering of 'other' voices")
    ap.add_argument("--no-realtime", action="store_true",
                    help="in --wav mode, run flat out instead of pacing to real time")
    args = ap.parse_args()

    if not Path(args.target).exists():
        raise SystemExit(f"no enrollment at {args.target} — run "
                         f"`python -m src.demo.live_demo_example --enroll` first")
    # Same graceful fallback as the notebook: if the (best) trailing model hasn't
    # been trained yet, run the centered one so the demo still works — just degraded.
    model_path = Path(args.model)
    if not model_path.exists():
        fallback = REPO_ROOT / "data" / "models" / "personal_vad_noisy.pt"
        if args.model == str(DEFAULT_MODEL) and fallback.exists():
            print(f"trailing model not found -> falling back to {fallback}\n"
                  f"  (centered-trained: expect ~1 s lag and a fuzzier map; train the\n"
                  f"   trailing model for a crisp one — see live_demo_example.py's docstring)")
            model_path = fallback
        else:
            raise SystemExit(f"no model at {args.model} — build a trailing cache and "
                             f"train it (see live_demo_example.py's docstring)")

    print("loading frozen backbones (CPU)...")
    embedder = load_embedder()
    vad_model = load_vad_model()
    e_target = np.load(args.target)["e_target"]
    model = load_model(model_path)

    det = StreamingDetector(model, embedder, vad_model, e_target)
    viz = LiveEmbeddingViz(e_target, thresh=args.thresh)

    if args.wav:
        run_wav_viz(det, viz, Path(args.wav), realtime=not args.no_realtime)
    else:
        run_mic_viz(det, viz)


if __name__ == "__main__":
    main()
