"""Visual sanity check for Phase 1: plot the 4-class labels over the spectrogram.

The Phase 1 milestone is "the labels visibly match the audio" (ROADMAP §6). This
draws the mixture's spectrogram with a colored label strip beneath it on the same
time axis, so you can eyeball that target-only / other-only / overlap / silence line
up with what you hear.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.synth.synthesize_scene_example import (
    CLASS_NAMES, LABEL_HOP, SR, SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP,
)

# One color per class; index matches the label integer.
CLASS_COLORS = {
    SILENCE: "#dddddd", TARGET_ONLY: "#2ca02c",      # green = record
    OTHER_ONLY: "#1f77b4", OVERLAP: "#d62728",        # red = overlap
}


def plot_scene(wav: np.ndarray, labels: np.ndarray, meta: dict, out_png: Path) -> None:
    duration = len(wav) / SR
    cmap = ListedColormap([CLASS_COLORS[c] for c in (SILENCE, TARGET_ONLY,
                                                     OTHER_ONLY, OVERLAP)])

    fig, (ax_spec, ax_lab) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    # Spectrogram (25 ms window, 10 ms hop to match the label grid).
    ax_spec.specgram(wav, NFFT=400, Fs=SR, noverlap=240, cmap="magma")
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title(
        f"target={meta['target']}  interferers={meta['interferers']}  "
        f"TIR={meta['tir_db']:.1f} dB  seed={meta['seed']}"
    )

    # Label strip: one colored column per frame on the same time axis.
    ax_lab.imshow(
        labels[np.newaxis, :], aspect="auto", cmap=cmap, vmin=0, vmax=3,
        extent=[0, duration, 0, 1], interpolation="nearest",
    )
    ax_lab.set_yticks([])
    ax_lab.set_xlabel("time (s)")
    ax_lab.legend(
        handles=[Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
                 for c in (SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP)],
        loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=False,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def class_summary(labels: np.ndarray) -> dict[str, float]:
    """Fraction of frames per class — a quick numeric sanity check."""
    total = len(labels)
    return {CLASS_NAMES[c]: round(float((labels == c).sum()) / total, 3)
            for c in (SILENCE, TARGET_ONLY, OTHER_ONLY, OVERLAP)}
