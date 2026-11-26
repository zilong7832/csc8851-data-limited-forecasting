"""Validate the finished grid, produce tables and figures, then build one return file."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from data import VAL_LEN
from .engine import checkpoint_path
from .protocol import SEEDS, Setting, ablations, main_settings, naive_settings, plan, save_json, sha

GROUP = ["dataset", "horizon", "percent", "model", "variant"]


def check_coverage(records, candidates, histories):
    def key(s, seed):
        return (*[s[k] for k in GROUP], seed)
    expected = {key(asdict(s), seed) for s in main_settings()+ablations() for seed in SEEDS}
    expected.update(key(asdict(s), 0) for s in naive_settings())
    actual = [key(r["setting"], r["seed"]) for r in records]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Held-out results are missing, repeated or outside the protocol.")
    from .protocol import RATES
    expected_tune = {(*[asdict(s)[k] for k in GROUP], lr) for s in main_settings() for lr in RATES}
    trial_keys = [(*[c[k] for k in GROUP], c["lr"]) for c in candidates]
    selected = [tuple(c[k] for k in GROUP) for c in candidates if c["selected"]]
    expected_groups = {k[:-1] for k in expected_tune}
    if (set(trial_keys) != expected_tune or len(trial_keys) != len(expected_tune)
            or set(selected) != expected_groups or len(selected) != len(expected_groups)):
        raise ValueError("Learning-rate search is incomplete or has duplicate selections.")
    learned = expected - {key(asdict(s), 0) for s in naive_settings()}
    if {key(row, row["seed"]) for row in histories} != learned:
        raise ValueError("Selected-checkpoint histories are incomplete.")
    trace = pd.DataFrame(histories).groupby(GROUP + ["seed"])
    for c in candidates:
        if not np.isfinite([c["best_val_mse"], c["train_seconds"]]).all():
            raise ValueError("Invalid validation trial.")
    for r in records:
        s = r["setting"]
        h = s["horizon"]
        n = VAL_LEN-h+1
        m = r["metrics"]
        if s["model"] != "naive":
            hrows = trace.get_group(key(s, r["seed"]))
            epoch_ids = hrows.epoch.to_numpy()
            best_epoch = int(hrows.loc[hrows.val_mse.idxmin(), "epoch"])
            if (not np.array_equal(epoch_ids, np.arange(1, m["epochs"]+1))
                    or best_epoch != m["best_epoch"]
                    or not np.isfinite(hrows[["train_mse", "val_mse"]]).all().all()):
                raise ValueError(f"Inconsistent training history for {s}")
        if m["n_test"] != n or len(r["origin_mae"]) != n or len(r["lead_mae"]) != h or len(r["lead_rmse"]) != h:
            raise ValueError(f"Window/lead coverage error in {s}")
        v = np.asarray(r["origin_mae"])
        if (not np.isfinite([m["mae"], m["rmse"], m["train_seconds"]]).all()
                or not np.allclose([np.mean(v), np.mean(r["lead_mae"]), np.mean(np.square(r["lead_rmse"]))],
                                   [m["mae"], m["mae"], m["rmse"]**2], rtol=2e-5, atol=1e-6)):
            raise ValueError(f"Metric aggregation disagrees for {s}")
        if [e["origin_index"] for e in r["examples"]] != [0, n//2]:
            raise ValueError(f"Predefined forecast origins missing in {s}")
        if s["dataset"] == "ETTh1" and h == 24 and s["percent"] == 25:
            if {z["scenario"] for z in r["simulation"]} != {"baseline", "shared_shift", "future_only_shift"}:
                raise ValueError("Simulation coverage is incomplete.")
            for z in r["simulation"]:
                if (z["n_paths"] != 512 or not np.isfinite([z["mae"], z["rmse"], z["paired_mae_change"],
                                                           z["mcse_paired_change"], z["response_residual"]]).all()):
                    raise ValueError(f"Invalid simulated metric for {s}")
                if s["variant"] != "noIN" and z["scenario"] == "shared_shift" and z["response_residual"] > 1e-3:
                    raise ValueError(f"Window-shift identity failed for {s}")
        if s["model"] == "naive" and s["percent"] == 100:
            if not np.isfinite([r["persistence"]["mae"], r["persistence"]["rmse"]]).all():
                raise ValueError("Persistence baseline is incomplete.")


def circular_interval(values, block, rng, repetitions=2000):
    """Circular moving blocks. Keep the final partial block at its exact length."""
    n = len(values)
    whole, tail = divmod(n, block)
    cs = np.r_[0.0, np.cumsum(np.tile(values, 2), dtype=np.float64)]
    starts = rng.integers(0, n, size=(repetitions, whole))
    sums = (cs[starts+block] - cs[starts]).sum(axis=1)
    if tail:
        rest = rng.integers(0, n, size=repetitions)
        sums += cs[rest+tail] - cs[rest]
    return np.quantile(sums/n, [0.025, 0.975])


def paired_comparisons(records):
    main = [r for r in records if r["setting"]["variant"] == "main"]
    rows = []
    for site in ("ETTh1", "ETTh2"):
        for h in (24, 96):
            for p in (25, 50, 100):
                group = [r for r in main if (r["setting"]["dataset"], r["setting"]["horizon"],
                                             r["setting"]["percent"]) == (site, h, p)]
                def avg_errors(name):
                    return np.mean([r["origin_mae"] for r in group if r["setting"]["model"] == name], axis=0)
                transformer = avg_errors("transformer")
                for other in ("naive", "linear", "mlp", "lstm"):
                    diff = transformer - avg_errors(other)
                    for block in (96, 168, 336):
                        lo, hi = circular_interval(diff, block, np.random.default_rng(17027))
                        rows.append(dict(dataset=site, horizon=h, percent=p, reference=other,
                                         delta_mae=diff.mean(), block_hours=block, low95=lo, high95=hi))
    return pd.DataFrame(rows)


def collate(records, candidates, histories, dest):
    metrics, leads, origins, examples, sims = [], [], [], [], []
    for r in records:
        group = {**r["setting"], "seed": r["seed"]}
        metrics.append({**group, "lr": r["learning_rate"], **r["metrics"]})
        leads.extend({**group, "lead": i, "mae": a, "rmse": b}
                     for i, (a,b) in enumerate(zip(r["lead_mae"], r["lead_rmse"]), 1))
        origins.append(pd.DataFrame({**group, "origin_index": np.arange(len(r["origin_mae"])),
                                     "mae": r["origin_mae"]}))
        for e in r["examples"]:
            for i, (a, f) in enumerate(zip(e["actual"], e["forecast"]), 1):
                examples.append({**group, "origin_index": e["origin_index"],
                                 "first_target_time": e["first_target_time"], "lead": i, "actual": a, "forecast": f})
        sims.extend({**group, **item} for item in r["simulation"])
    seed = pd.DataFrame(metrics)
    agg = seed.groupby(GROUP, as_index=False).agg(
        seeds=("seed", "count"), train_windows=("train_windows", "first"), test_windows=("n_test", "first"),
        mae_mean=("mae", "mean"), mae_seed_sd=("mae", "std"), rmse_mean=("rmse", "mean"),
        rmse_seed_sd=("rmse", "std"), params=("params", "first"),
        fit_seconds_mean=("train_seconds", "mean"), fit_seconds_sd=("train_seconds", "std"),
        best_epoch_median=("best_epoch", "median"))
    history = pd.DataFrame(histories)
    forecast = pd.DataFrame(examples)
    sim = pd.DataFrame(sims)
    tables = {"test_by_seed": seed, "test_summary": agg, "learning_rate_selection": pd.DataFrame(candidates),
              "loss_histories": history, "by_forecast_lead": pd.DataFrame(leads),
              "forecast_examples": forecast, "simulation": sim,
              "paired_mae_differences": paired_comparisons(records)}
    tables["last_value_reference"] = pd.DataFrame([
        {"dataset": r["setting"]["dataset"], "horizon": r["setting"]["horizon"], **r["persistence"]}
        for r in records if r["persistence"]])
    join_keys = ["dataset", "horizon", "percent", "model", "seed"]
    paired = seed[seed.variant.ne("main")].merge(seed[seed.variant.eq("main")], on=join_keys,
                    suffixes=("_changed", "_primary"), validate="many_to_one")
    for metric in ("mae", "rmse", "train_seconds", "params"):
        paired[metric+"_change"] = paired[metric+"_changed"] - paired[metric+"_primary"]
    tables["paired_ablations"] = paired
    # Full original time series are already in the server package. Return the history for each displayed origin.
    contexts = {}
    for r in records:
        if r["setting"]["model"] == "naive" and r["setting"]["percent"] == 100:
            s = r["setting"]
            contexts[f"{s['dataset']}_H{s['horizon']}"] = [
                {k:e[k] for k in ("first_target_time","origin_index","history","actual")} for e in r["examples"]]
    save_json(dest / "forecast_context.json", contexts)
    pd.concat(origins, ignore_index=True).to_csv(dest / "origin_mae.csv.gz", index=False, compression="gzip")
    for name, df in tables.items():
        df.to_csv(dest / (name + ".csv"), index=False)
    from .visuals import make_figures
    make_figures(dest / "figures", agg, history, sim, forecast, contexts)


def finish(root, files, candidates, histories):
    records = [json.loads(p.read_text()) for p in files]
    check_coverage(records, candidates, histories)
    archive = root / "Team17_RESULTS.zip"
    tmpzip = root / "Team17_RESULTS.zip.tmp"
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        dest = Path(tmp)
        collate(records, candidates, histories, dest)
        for name in ("manifest.json", "selection.json"):
            (dest / name).write_bytes((root / name).read_bytes())
        # Returning the fitted models allows arbitrary new forecast plots without retraining on the server.
        inputs = {str(p.relative_to(dest)): p for p in sorted(dest.rglob("*")) if p.is_file()}
        for record in records:
            setting = Setting(**record["setting"])
            if setting.model != "naive":
                ckpt = checkpoint_path(root, setting, record["learning_rate"], record["seed"])
                inputs["checkpoints/" + ckpt.name] = ckpt
        save_json(dest / "COMPLETE.json", {"status": "complete", **plan(),
                  "sha256": {name: sha(p) for name,p in inputs.items()}})
        inputs["COMPLETE.json"] = dest / "COMPLETE.json"
        with ZipFile(tmpzip, "w", ZIP_DEFLATED) as bundle:
            for name, p in inputs.items():
                bundle.write(p, "Team17_RESULTS/" + name)
        with ZipFile(tmpzip) as bundle:
            damaged = bundle.testzip()
            if damaged:
                raise IOError(f"Invalid return archive member: {damaged}")
    os.replace(tmpzip, archive)
    return archive
