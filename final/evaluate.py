"""Frozen-checkpoint evaluation, forecast examples and a paired shift simulation."""
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from data import DATA_DIR, TRAIN_LEN, VAL_LEN
from .engine import load_model, predict, windows
from .protocol import L, save_json

SIM_N = 512
SIM_SEED = 17026


@lru_cache(maxsize=2)
def timestamps(dataset):
    return pd.read_csv(DATA_DIR / (dataset + ".csv"), usecols=["date"]).date.to_numpy()


def simulated_paths():
    """Independent paths with a random 24h cycle, gradual slope and AR(1) noise."""
    rng = np.random.default_rng(SIM_SEED)
    time = np.arange(L + 24)[None, :]
    level = rng.uniform(13, 20, (SIM_N, 1))
    amplitude = rng.uniform(1, 4, (SIM_N, 1))
    phase = rng.uniform(0, 2 * np.pi, (SIM_N, 1))
    slope = rng.uniform(-0.008, 0.008, (SIM_N, 1))
    noise = rng.normal(0, 0.45, (SIM_N, L + 24))
    noise[:, 0] /= np.sqrt(1 - 0.8 ** 2)
    for t in range(1, L + 24):
        noise[:, t] += 0.8 * noise[:, t-1]
    paths = level + amplitude * np.sin(2 * np.pi * time / 24 + phase) + slope * time + noise
    shift = rng.choice([-6.0, 6.0], size=(SIM_N, 1))
    return paths[:, :L], paths[:, L:], shift


@torch.inference_mode()
def simulation(model, mu, sd, device):
    x, y, delta = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in simulated_paths()]
    base = predict(model, (x-mu)/sd, 24) * sd + mu
    moved = predict(model, (x+delta-mu)/sd, 24) * sd + mu
    base_mae = (base - y).abs().mean(dim=1)
    experiments = [("baseline", base, y, torch.zeros_like(delta)),
                   ("shared_shift", moved, y + delta, delta),
                   ("future_only_shift", base, y + delta, torch.zeros_like(delta))]
    results = []
    for name, estimate, target, expected_response in experiments:
        error = estimate - target
        path_mae = error.abs().mean(dim=1)
        change = path_mae - base_mae
        results.append(dict(scenario=name, n_paths=SIM_N, mae=path_mae.mean().item(),
                            rmse=error.square().mean().sqrt().item(),
                            paired_mae_change=change.mean().item(),
                            mcse_paired_change=change.std().item() / np.sqrt(SIM_N),
                            response_residual=(estimate-base-expected_response).abs().mean().item()))
    return results


@torch.inference_mode()
def evaluate(root, setting, lr, seed, device):
    out = root / "evaluations" / f"{setting.key}_s{seed}.json"
    if out.exists():
        return out
    x, y, (mu, sd) = windows(setting.dataset, setting.horizon, setting.percent, "test", device)
    model, fit = (None, dict(params=0, epochs=0, best_epoch=0, train_seconds=0.0))
    if setting.model != "naive":
        model, fit = load_model(root, setting, lr, seed, device)
    forecast = predict(model, x, setting.horizon)
    error = (forecast - y) * sd
    times = timestamps(setting.dataset)
    start = TRAIN_LEN + VAL_LEN
    examples = []
    # Fixed forecast origins selected by the protocol; the index is relative to this test period.
    for i in (0, len(x) // 2):
        examples.append(dict(origin_index=i, first_target_time=str(times[start+i]),
                             history=(x[i] * sd + mu).cpu().tolist(),
                             actual=(y[i] * sd + mu).cpu().tolist(),
                             forecast=(forecast[i] * sd + mu).cpu().tolist()))
    extra = []
    if setting.dataset == "ETTh1" and setting.horizon == 24 and setting.percent == 25:
        extra = simulation(model, mu, sd, device)
    metrics = {k: fit[k] for k in ("params", "epochs", "best_epoch", "train_seconds")}
    metrics.update(n_test=len(x), train_windows=TRAIN_LEN*setting.percent//100 - L-setting.horizon+1,
                   mae=error.abs().mean().item(), rmse=error.square().mean().sqrt().item())
    persistence = {}
    if setting.model == "naive" and setting.percent == 100:
        # Separate, cost-free check for slowly changing signals, beyond the five primary methods.
        last_value_error = (x[:, -1:] - y) * sd
        persistence = {"mae": last_value_error.abs().mean().item(),
                       "rmse": last_value_error.square().mean().sqrt().item()}
    save_json(out, dict(setting=asdict(setting), seed=seed, learning_rate=lr, metrics=metrics,
                        lead_mae=error.abs().mean(0).cpu().tolist(),
                        lead_rmse=error.square().mean(0).sqrt().cpu().tolist(),
                        origin_mae=error.abs().mean(1).cpu().tolist(),
                        examples=examples, simulation=extra, persistence=persistence))
    return out
