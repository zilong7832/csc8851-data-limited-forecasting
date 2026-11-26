#!/usr/bin/env python3
"""Run the reproducible Team 17 experiment or validate the package without training."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path

# cuBLAS reads this before its first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from final.engine import fit
from final.evaluate import evaluate
from final.protocol import (L, RATES, ROOT, SEEDS, ablations, main_settings, naive_settings,
                            plan, preflight, provenance, save_json)
from final.summarize import finish


@contextmanager
def exclusive(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"Another process is using {root}.") from exc
        yield


def check_models():
    # No optimizer, data fitting or scientific evaluation occurs here.
    x = torch.linspace(-1, 1, L).unsqueeze(0).repeat(2, 1)
    settings = {(s.model, s.variant, s.horizon): s for s in main_settings()+ablations()}
    with torch.inference_mode():
        for s in settings.values():
            net = s.network().eval()
            y = net(x)
            if y.shape != (2, s.horizon) or not torch.isfinite(y).all():
                raise ValueError(f"Forward shape or finiteness failed for {s.key}")
            if s.variant != "noIN" and not torch.allclose(net(x+3), y+3, atol=2e-5, rtol=1e-5):
                raise ValueError(f"Temperature translation check failed for {s.key}")


def check_gpu(device):
    # A small artificial batch exercises CUDA training kernels before the long run.
    x = torch.linspace(-1, 1, L, device=device).reshape(1, L).repeat(2, 1)
    variants = {(s.model, s.horizon, s.variant): s for s in main_settings() + ablations()}
    for setting in variants.values():
        model = setting.network().to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=RATES[0])
        loss = model(x).square().mean()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError(f"GPU gradients failed in {setting.key}")
        opt.step()
        model.eval()
        with torch.inference_mode():
            if not torch.isfinite(model(x)).all().item():
                raise ValueError(f"GPU inference failed in {setting.key}")
    torch.cuda.synchronize(device)


def train_all(root, device):
    selected = {}
    candidates, histories = [], []
    for i, setting in enumerate(main_settings(), 1):
        print(f"Main comparison {i}/{len(main_settings())}: {setting.key}", flush=True)
        trials = [fit(root, setting, lr, 0, device) for lr in RATES]
        winner = min(trials, key=lambda r: r["best_val_mse"])
        lr = selected[setting.key] = winner["lr"]
        for r in trials:
            candidates.append({**asdict(setting), "lr": r["lr"], "selected": r["lr"] == lr,
                               "best_val_mse": r["best_val_mse"], "best_epoch": r["best_epoch"],
                               "train_seconds": r["train_seconds"]})
        fits = [winner] + [fit(root, setting, lr, seed, device) for seed in SEEDS[1:]]
        for r in fits:
            histories.extend({**asdict(setting), "seed": r["seed"], "lr": lr, **v} for v in r["history"])
    for setting in ablations():
        lr = selected[replace(setting, variant="main").key]
        selected[setting.key] = lr
        for seed in SEEDS:
            r = fit(root, setting, lr, seed, device)
            histories.extend({**asdict(setting), "seed": seed, "lr": lr, **v} for v in r["history"])
    # The selection is completely fixed and recorded before any held-out scores are requested.
    save_json(root / "selection.json", selected)
    return selected, candidates, histories


def setup_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("Training needs a CUDA-enabled PyTorch environment on the GPU server. "
                         "Use --check for an offline package check.")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    check_gpu(device)
    return device


def run(root, device):
    info = provenance(device)
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != info:
        raise SystemExit("This output directory belongs to another code/data/runtime configuration. "
                         "Use a separate --out directory or resume in the original environment.")
    save_json(manifest, info)
    selected, candidates, histories = train_all(root, device)
    print("All model choices are frozen. Scoring the held-out periods.", flush=True)
    files = []
    for setting in main_settings() + ablations():
        for seed in SEEDS:
            files.append(evaluate(root, setting, selected[setting.key], seed, device))
    for setting in naive_settings():
        files.append(evaluate(root, setting, None, 0, device))
    archive = finish(root, files, candidates, histories)
    print(f"COMPLETE. Return this file: {archive.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check files, splits and model shapes without training")
    parser.add_argument("--gpu-check", action="store_true", help="exercise CUDA with tiny artificial batches only")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "team17", help="persistent run directory")
    args = parser.parse_args()
    checked = preflight()
    print("Team 17 study:", json.dumps(plan()))
    if args.check:
        torch.set_num_threads(1)
        check_models()
        print(json.dumps(checked, indent=2))
        print(f"Package checks passed. CUDA available: {torch.cuda.is_available()}. No models were trained.")
        return
    device = setup_cuda()
    if args.gpu_check:
        print(f"GPU preflight passed on {torch.cuda.get_device_name(device)}. No scientific fits were run.")
        return
    with exclusive(args.out.resolve()):
        run(args.out.resolve(), device)


if __name__ == "__main__":
    main()
