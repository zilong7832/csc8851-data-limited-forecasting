"""ETTh data loading, chronological splits, and training-history subsets.

Split convention (ETT, hourly): 12 months train, 4 months validation, 4 months test,
i.e. 8640 / 2880 / 2880 hours. Validation and test forecast origins may look back
into the preceding period, so every forecast uses only past observations.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_LEN, VAL_LEN, TEST_LEN = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
TARGET = "OT"  # oil temperature


def load_series(name):
    """Return the univariate oil-temperature series of ETTh1 / ETTh2 as float32."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df[TARGET].to_numpy(dtype=np.float32)


class Windows(Dataset):
    """Sliding (input, target) windows whose forecast origins lie in [start, end)."""

    def __init__(self, z, start, end, lookback, horizon):
        self.z, self.L, self.H = z, lookback, horizon
        self.origins = np.arange(max(start, lookback), end - horizon + 1)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t = self.origins[i]
        return (torch.from_numpy(self.z[t - self.L:t]),
                torch.from_numpy(self.z[t:t + self.H]))


def make_splits(name, frac, lookback, horizon):
    """Build train/val/test datasets for one training-history fraction.

    The training subset is the most recent `frac` of the 12-month training period
    (contiguous, same endpoint). Inputs and targets of training windows stay inside it.
    Normalisation statistics are fitted on that subset only.
    """
    x = load_series(name)
    train_end, val_end = TRAIN_LEN, TRAIN_LEN + VAL_LEN
    test_end = val_end + TEST_LEN
    train_start = TRAIN_LEN - int(round(frac * TRAIN_LEN))

    mu = float(x[train_start:train_end].mean())
    sd = float(x[train_start:train_end].std())
    z = (x - mu) / sd

    # Training windows: inputs and targets inside [train_start, train_end).
    train = Windows(z[train_start:train_end], 0, train_end - train_start, lookback, horizon)
    # Validation / test windows: forecast origin inside the period, history may precede it.
    val = Windows(z[:val_end], train_end, val_end, lookback, horizon)
    test = Windows(z[:test_end], val_end, test_end, lookback, horizon)
    return train, val, test, (mu, sd)
