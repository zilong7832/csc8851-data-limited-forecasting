"""Validation-only fitting and batch evaluation on a single CUDA device."""
from copy import deepcopy
from functools import lru_cache
import os
import random
import time

import numpy as np
import torch

from data import make_splits
from models import n_params, seasonal_naive
from .protocol import BATCH, L, MAX_EPOCHS, PATIENCE


@lru_cache(maxsize=4)
def windows(dataset, horizon, percent, split, device):
    # Reuse the original split and origins. The strided view creates no window copies.
    parts = make_splits(dataset, percent / 100, L, horizon)
    item = parts[{"train": 0, "val": 1, "test": 2}[split]]
    t0 = int(item.origins[0])
    t1 = int(item.origins[-1])
    values = torch.as_tensor(item.z[t0-L:t1+horizon], device=device)
    matrix = values.unfold(0, L + horizon, 1)
    if matrix.shape[0] != len(item):
        raise RuntimeError("Unexpected window count")
    return matrix[:, :L], matrix[:, L:], parts[3]


@torch.inference_mode()
def predict(model, inputs, horizon):
    if model is None:
        return seasonal_naive(inputs, horizon)
    model.eval()
    return torch.cat([model(part.contiguous()) for part in inputs.split(256)])


def checkpoint_path(root, setting, lr, seed):
    return root / "checkpoints" / f"{setting.key}_lr{lr:g}_s{seed}.pt"


def read_fit(path):
    # Only load checkpoints written by this runner in its own run directory.
    return torch.load(path, map_location="cpu", weights_only=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit(root, setting, lr, seed, device):
    path = checkpoint_path(root, setting, lr, seed)
    if path.is_file():
        return read_fit(path)["summary"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    shuffle = torch.Generator(device=device).manual_seed(seed)
    x, y, (_, sd) = windows(setting.dataset, setting.horizon, setting.percent, "train", device)
    xv, yv, _ = windows(setting.dataset, setting.horizon, setting.percent, "val", device)
    model = setting.network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    hist = []
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for idx in torch.randperm(len(x), device=device, generator=shuffle).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[idx]) - y[idx]).square().mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(idx)
        val_mse = (predict(model, xv, setting.horizon) - yv).square().mean().item()
        train_mse = (loss_sum / len(x)).item()
        if not np.isfinite([val_mse, train_mse]).all():
            raise ArithmeticError(f"Non-finite loss: {setting.key}, seed {seed}")
        hist.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    sync(device)
    summary = dict(lr=lr, seed=seed, params=n_params(model), epochs=epoch, best_epoch=best_epoch,
                   best_val_mse=best, val_rmse_c=np.sqrt(best).item() * sd,
                   train_seconds=time.perf_counter() - start, history=hist)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state": {k: v.cpu() for k, v in state.items()}, "summary": summary}, tmp)
    os.replace(tmp, path)
    print(f"  {setting.key} lr={lr:g} seed={seed} epochs={epoch} "
          f"best val RMSE={summary['val_rmse_c']:.3f} C", flush=True)
    return summary


def load_model(root, setting, lr, seed, device):
    saved = read_fit(checkpoint_path(root, setting, lr, seed))
    model = setting.network().to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    return model, saved["summary"]
