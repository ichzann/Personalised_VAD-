"""Phase 4 model: the small trainable head (ROADMAP §3, §6).

This is the ONLY trained part of the project (ROADMAP §2). The frozen backbones
already produced, per 10 ms frame: a mel spectrogram, a speaker embedding e[t], the
cosine s[t] to the enrollment, and the VAD prob. The head turns that per-frame
feature sequence into a per-frame 4-class decision.

Architecture (kept deliberately small + explainable, ROADMAP §8):

    mel (T,40) --1D Conv over FREQUENCY (per frame)--> m[t] (T,64)   <- overlap cue (§3-B)
    e[t] (T,192) --Linear proj--> (T,32)                             <- speaker identity
    s[t] (T,1), p_speech[t] (T,1)                                    <- comparison + speech
    x[t] = concat[ m[t], proj(e[t]), s[t], p_speech[t] ]  (T,98)
    x[t] --> BiLSTM (the temporal model) --> Linear --> 4-class logits per frame

Key constraints honored:
- The mel conv runs **over frequency within a single frame** (Conv1d on the 40-bin
  axis), so it adds NO temporal lookahead — the exact same code runs offline and
  streaming (ROADMAP §3 "Mel pathway"). The BiLSTM does all temporal modeling.
- Plain `Conv1d`, a few filters, 2 layers. No 2D convs, no MobileNet tricks (§8).
- Mel standardization stats are stored as buffers so a saved model is self-contained.
"""

import torch
import torch.nn as nn

N_CLASSES = 4          # silence / target-only / other-only / overlap (ROADMAP §2)
N_MELS = 40
EMB_DIM = 192          # ECAPA embedding size


class MelFreqConv(nn.Module):
    """Tiny 1D CNN over the frequency axis, applied independently to each frame.

    Input  (B, T, n_mels) -> output (B, T, out_dim). We fold (B, T) into the batch
    so the conv only ever sees one frame's 40-bin spectrum at a time (no time mixing).
    """

    def __init__(self, n_mels: int = N_MELS, n_filters: int = 8, pooled: int = 8):
        super().__init__()
        # 1 input "channel" (the spectrum), a few filters, kernel over frequency.
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(pooled)   # shrink freq axis to a fixed length
        self.relu = nn.ReLU()
        self.out_dim = n_filters * pooled

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        b, t, f = mel.shape
        x = mel.reshape(b * t, 1, f)               # (B*T, 1, n_mels)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)                           # (B*T, n_filters, pooled)
        x = x.flatten(1)                           # (B*T, out_dim) = m[t]
        return x.reshape(b, t, self.out_dim)


class PersonalVAD(nn.Module):
    """Mel pathway + embedding projection + BiLSTM + per-frame 4-class classifier."""

    def __init__(self, emb_proj_dim: int = 32, lstm_hidden: int = 64,
                 mel_filters: int = 8):
        super().__init__()
        self.mel_conv = MelFreqConv(n_filters=mel_filters)
        self.emb_proj = nn.Sequential(nn.Linear(EMB_DIM, emb_proj_dim), nn.ReLU())

        # x[t] = [ m[t] , proj(e[t]) , s[t] , p_speech[t] ]
        in_dim = self.mel_conv.out_dim + emb_proj_dim + 1 + 1
        self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * lstm_hidden, N_CLASSES)

        # Mel standardization (set from train stats before training, ROADMAP: cache
        # features; here we just normalize them). Buffers travel with the checkpoint.
        self.register_buffer("mel_mean", torch.zeros(N_MELS))
        self.register_buffer("mel_std", torch.ones(N_MELS))

    def set_mel_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mel_mean.copy_(mean)
        self.mel_std.copy_(std.clamp_min(1e-6))

    def forward(self, mel, emb, cos, vad):
        """mel (B,T,40), emb (B,T,192), cos (B,T), vad (B,T) -> logits (B,T,4)."""
        mel = (mel - self.mel_mean) / self.mel_std
        m = self.mel_conv(mel)                     # (B,T,64)
        e = self.emb_proj(emb)                     # (B,T,32)
        x = torch.cat([m, e, cos.unsqueeze(-1), vad.unsqueeze(-1)], dim=-1)
        h, _ = self.lstm(x)                        # (B,T,2*hidden)
        return self.classifier(h)                  # (B,T,4)
