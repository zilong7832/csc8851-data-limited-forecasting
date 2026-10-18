"""Forecasting models. Every model maps a (B, L) input window to a (B, H) forecast."""
import torch
import torch.nn as nn


class LinearForecaster(nn.Module):
    """Direct linear map from the L past values to the H future values."""

    def __init__(self, lookback, horizon):
        super().__init__()
        self.proj = nn.Linear(lookback, horizon)

    def forward(self, x):
        return self.proj(x)


class PatchTransformer(nn.Module):
    """Compact patch-based Transformer encoder.

    The input window is split into non-overlapping patches of length P, each patch is
    linearly embedded to d dimensions, a learned positional embedding is added, the
    sequence of N = L / P tokens passes through a Transformer encoder, and the flattened
    token representations are mapped to the H-step forecast by a linear head.
    """

    def __init__(self, lookback, horizon, patch_len=24, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        assert lookback % patch_len == 0
        self.P, self.N = patch_len, lookback // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.N, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.N * d_model, horizon)

    def forward(self, x):
        tokens = self.embed(x.unfold(1, self.P, self.P))      # (B, N, d)
        h = self.encoder(self.drop(tokens + self.pos))
        return self.head(self.norm(h).flatten(1))


class InstanceNorm(nn.Module):
    """Per-window normalisation (non-affine RevIN, Kim et al. 2022) around any forecaster.

    Each input window is standardised by its own mean and standard deviation, and the
    forecast is mapped back with the same statistics. This lets models trained on one
    temperature level forecast series that later drift to a different level.
    """

    def __init__(self, model, eps=1e-5):
        super().__init__()
        self.model, self.eps = model, eps

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        s = x.std(1, keepdim=True) + self.eps
        return self.model((x - m) / s) * s + m


def seasonal_naive(x, horizon, period=24):
    """Repeat the last observed day (non-trained reference)."""
    last = x[:, -period:]
    reps = -(-horizon // period)
    return last.repeat(1, reps)[:, :horizon]


def build_model(name, lookback, horizon, instance_norm=True, **kw):
    if name == "linear":
        model = LinearForecaster(lookback, horizon)
    elif name == "transformer":
        model = PatchTransformer(lookback, horizon, **kw)
    else:
        raise ValueError(name)
    return InstanceNorm(model) if instance_norm else model


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
