# Small Transformers for Data-Limited Time Series Forecasting

CSC 8851 Deep Learning, Georgia State University, Fall 2026. Zilong Zhang, Team 17.

## Final study on a GPU server

The final runner implements the proposal’s two-station, two-horizon, three-history comparison, with four learned forecasters, a seasonal-naive reference, three training seeds and matched ablations. The return archive is generated after all expected results pass completeness and numerical consistency checks.

See the [Chinese server quick start](SERVER_GUIDE_ZH.md) and [full experimental protocol](final/PROTOCOL.md).

On a Linux GPU node with CUDA-enabled PyTorch installed:

```bash
python3 -m pip install -r requirements.txt
bash run_server.sh --gpu-check
bash run_server.sh
```

The final line prints the path to `Team17_RESULTS.zip`. The default is `runs/team17/Team17_RESULTS.zip`. Repeating the same command resumes completed fits. For an offline check without training, use `bash run_server.sh --check`.

## Preliminary study

The original CPU entry point and its archived validation metrics remain available below. For the final study, use the commands above.

### Setup

```bash
pip install -r requirements.txt
```

`data/ETTh1.csv` and `data/ETTh2.csv` are included. They come from the public
ETDataset repository (https://github.com/zhouhaoyi/ETDataset, `ETT-small/`). To
re-download:

```bash
curl -L -o data/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
curl -L -o data/ETTh2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
```

### Preliminary files

| File | Purpose |
|---|---|
| `data.py` | Loads the `OT` (oil temperature) column, makes the chronological 12/4/4-month split, the 25/50/100% training-history subsets, normalisation (fitted on the training subset only) and sliding windows. |
| `models.py` | Linear forecaster, patch-based Transformer, per-window instance normalisation wrapper, seasonal-naive reference. |
| `train.py` | Trains and evaluates the models for every history fraction; writes one JSON record per run to `results/<tag>.jsonl` (the file is rewritten on each invocation) (metrics, parameter count, training time, loss history). |
| `make_figures.py` | Prints the results table and draws the learning-curve figure (`--split val` by default). |

### Preliminary protocol

* Input: previous 168 hours of `OT`; output: next 24 (or 96) hours, predicted directly.
* Split: first 8640 h train, next 2880 h validation, next 2880 h test.
* Training history: the most recent 25%, 50% or 100% of the training period
  (1969 / 4129 / 8449 training windows at horizon 24). Training inputs and targets
  stay inside that subset; validation/test windows may look back into the preceding period.
* Loss: MSE on normalised values. Metrics: MAE and RMSE in degrees Celsius.
* Model selection uses the validation period only. The test period is evaluated (`--eval_test`)
  only once, for the final selected configurations; the preliminary report uses validation metrics.
* Optimiser: Adam, lr 1e-3, batch 64, at most 100 epochs, early stopping with
  patience 10 on validation MSE; the best validation checkpoint is evaluated.

### Reproducing the preliminary-report results

```bash
# Table 2 / Figure 2 of the preliminary report (ETTh1, 24-hour horizon, seed 0)
python train.py --dataset ETTh1 --horizon 24 --models naive linear transformer --seeds 0
python make_figures.py --tag ETTh1_H24 --out results

# Same setting without per-window instance normalisation (discussed in Sec. 4.2)
python train.py --dataset ETTh1 --horizon 24 --models linear transformer --seeds 0 \
    --no_instance_norm --tag ETTh1_H24_noIN
```

Results of these commands are stored in `results/`. Everything runs on a laptop CPU in
under a minute (Apple M3, 4 threads).

