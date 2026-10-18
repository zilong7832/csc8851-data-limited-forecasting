"""Train and evaluate forecasters across training-history fractions.

Example (preliminary report setting):
    python train.py --dataset ETTh1 --horizon 24 --models linear transformer --seeds 0
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import make_splits
from models import build_model, n_params, seasonal_naive

RESULTS = Path(__file__).resolve().parent / "results"


def predict(model, loader):
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x) if model is not None else seasonal_naive(x, y.shape[1]))
            ys.append(y)
    return torch.cat(preds), torch.cat(ys)


def metrics(pred, y, mu, sd):
    """MAE and RMSE in original units (degrees Celsius)."""
    err = (pred - y) * sd
    return {"mae": err.abs().mean().item(), "rmse": err.pow(2).mean().sqrt().item()}


def fit(model, train_dl, val_dl, lr, max_epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, hist = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        tr = []
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr.append(loss.item())
        model.eval()
        p, y = predict(model, val_dl)
        val = loss_fn(p, y).item()
        hist.append({"epoch": epoch + 1, "train_mse": float(np.mean(tr)), "val_mse": val})
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2"])
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--models", nargs="+", default=["naive", "linear", "transformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no_instance_norm", action="store_true",
                    help="disable per-window normalisation of the learned models")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or f"{args.dataset}_H{args.horizon}"
    out = RESULTS / f"{tag}.jsonl"

    for frac in args.fracs:
        train_ds, val_ds, test_ds, (mu, sd) = make_splits(
            args.dataset, frac, args.lookback, args.horizon)
        val_dl = DataLoader(val_ds, batch_size=512)
        test_dl = DataLoader(test_ds, batch_size=512)
        print(f"{args.dataset} frac={frac}: train/val/test windows = "
              f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
        for name in args.models:
            for seed in (args.seeds if name != "naive" else [0]):
                rec = {"dataset": args.dataset, "horizon": args.horizon, "frac": frac,
                       "model": name, "seed": seed, "n_train_windows": len(train_ds)}
                if name == "naive":
                    model, hist, secs, params = None, [], 0.0, 0
                else:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    kw = {"n_layers": args.n_layers} if name == "transformer" else {}
                    model = build_model(name, args.lookback, args.horizon,
                                        instance_norm=not args.no_instance_norm, **kw)
                    params = n_params(model)
                    g = torch.Generator().manual_seed(seed)
                    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                          shuffle=True, generator=g)
                    t0 = time.perf_counter()
                    hist = fit(model, train_dl, val_dl, args.lr, args.max_epochs, args.patience)
                    secs = time.perf_counter() - t0
                    model.eval()
                rec.update(params=params, train_seconds=secs, epochs=len(hist), history=hist)
                rec["val"] = metrics(*predict(model, val_dl), mu, sd)
                rec["test"] = metrics(*predict(model, test_dl), mu, sd)
                print(f"  {name:11s} seed={seed} params={params:6d} epochs={len(hist):3d} "
                      f"time={secs:6.1f}s  test MAE={rec['test']['mae']:.3f} "
                      f"RMSE={rec['test']['rmse']:.3f}")
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if name != "naive":
                    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                                "frac": frac, "seed": seed, "mu": mu, "sd": sd},
                               RESULTS / f"{tag}_{name}_f{frac}_s{seed}.pt")


if __name__ == "__main__":
    main()
