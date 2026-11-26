"""Declared study design, provenance and lightweight data checks."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import platform

import numpy as np
import pandas as pd
import torch

from data import DATA_DIR, TRAIN_LEN, VAL_LEN, TEST_LEN, make_splits
from models import build_model

ROOT = Path(__file__).resolve().parents[1]
TEAM = 17
L = 168
HORIZONS = (24, 96)
PERCENTAGES = (25, 50, 100)
MODELS = ("linear", "mlp", "lstm", "transformer")
SEEDS = (0, 1, 2)
RATES = (0.001, 0.0003)  # ties keep the first candidate
BATCH, MAX_EPOCHS, PATIENCE = 64, 100, 10
HASHES = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}


@dataclass(frozen=True)
class Setting:
    dataset: str
    horizon: int
    percent: int
    model: str
    variant: str = "main"

    @property
    def key(self):
        return f"{self.dataset}_h{self.horizon}_p{self.percent}_{self.model}_{self.variant}"

    def network(self):
        kwargs = {}
        if self.model == "transformer":
            kwargs["n_layers"] = 1 if self.variant == "depth1" else 2
        return build_model(self.model, L, self.horizon, instance_norm=self.variant != "noIN", **kwargs)


def main_settings():
    return [Setting(*items) for items in product(HASHES, HORIZONS, PERCENTAGES, MODELS)]


def ablations():
    return ([Setting("ETTh1", 24, p, m, "noIN") for p, m in product((25, 100), ("linear", "transformer"))]
            + [Setting("ETTh1", 24, p, "transformer", "depth1") for p in (25, 100)])


def naive_settings():
    return [Setting(d, h, p, "naive") for d, h, p in product(HASHES, HORIZONS, PERCENTAGES)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def plan():
    main = len(main_settings())
    n_tune = main * len(RATES)
    n_main_refit = main * (len(SEEDS) - 1)
    n_ablate = len(ablations()) * len(SEEDS)
    return dict(team=TEAM, selection_fits=n_tune, other_main_fits=n_main_refit,
                ablation_fits=n_ablate, total_fits=n_tune + n_main_refit + n_ablate,
                heldout_scores=(main + len(ablations())) * len(SEEDS) + len(naive_settings()))


def preflight():
    details = {}
    for name, digest in HASHES.items():
        path = DATA_DIR / (name + ".csv")
        if sha(path) != digest:
            raise ValueError(f"Data checksum mismatch: {path}")
        df = pd.read_csv(path, usecols=["date", "OT"])
        dates = pd.to_datetime(df.date)
        if len(df) < TRAIN_LEN + VAL_LEN + TEST_LEN or not np.isfinite(df.OT).all():
            raise ValueError(f"Incomplete temperature series: {name}")
        if not dates.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"Hourly timestamps are missing or repeated: {name}")
        counts = {}
        for h, p in product(HORIZONS, PERCENTAGES):
            train, val, test, (mu, sd) = make_splits(name, p / 100, L, h)
            sizes = [len(s) for s in (train, val, test)]
            expected = [TRAIN_LEN * p // 100 - L - h + 1, VAL_LEN - h + 1, TEST_LEN - h + 1]
            if sizes != expected or not (sd > 0 and np.isfinite(mu)):
                raise ValueError(f"Invalid data split {name, h, p}")
            counts[f"H{h}_p{p}"] = sizes
        details[name] = {"sha256": digest, "windows_train_val_test": counts}
    return details


def provenance(device):
    files = [ROOT / n for n in ("models.py", "data.py", "run_final.py")]
    files += sorted((ROOT / "final").glob("*.py"))
    info = {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "pandas": pd.__version__, "system": platform.system(),
            "device": torch.cuda.get_device_name(device), "threads": torch.get_num_threads(),
            "device_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
    return {"protocol": plan(), "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "input_sha256": HASHES, "runtime": info}
