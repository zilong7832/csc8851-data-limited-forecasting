# Small Transformers for Data-Limited Time Series Forecasting

Code for the CSC 8851 (Deep Learning, Fall 2026) course project by Zilong Zhang.
The project studies how the amount of available training history affects the accuracy
and cost of a compact patch-based Transformer relative to simpler forecasters on the
ETTh1 / ETTh2 oil-temperature series.

## Setup

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

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads the `OT` (oil temperature) column, makes the chronological 12/4/4-month split, the 25/50/100% training-history subsets, normalisation (fitted on the training subset only) and sliding windows. |
| `models.py` | Linear forecaster, patch-based Transformer, per-window instance normalisation wrapper, seasonal-naive reference. |
| `train.py` | Trains and evaluates the models for every history fraction; appends one JSON record per run to `results/<tag>.jsonl` (metrics, parameter count, training time, loss history). |
| `make_figures.py` | Prints the results table and draws the learning-curve figure. |

## Protocol

* Input: previous 168 hours of `OT`; output: next 24 (or 96) hours, predicted directly.
* Split: first 8640 h train, next 2880 h validation, next 2880 h test.
* Training history: the most recent 25%, 50% or 100% of the training period
  (1969 / 4129 / 8449 training windows at horizon 24). Training inputs and targets
  stay inside that subset; validation/test windows may look back into the preceding period.
* Loss: MSE on normalised values. Metrics: MAE and RMSE in degrees Celsius on the test period.
* Optimiser: Adam, lr 1e-3, batch 64, at most 100 epochs, early stopping with
  patience 10 on validation MSE; the best validation checkpoint is evaluated.

## Reproducing the preliminary-report results

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

## Planned for the final report

* MLP and LSTM baselines, ETTh2, the 96-hour horizon, three seeds per configuration,
  equal-budget validation hyperparameter search.
* Ablation on ETTh1 / 24 h: one vs. two encoder layers (`--n_layers 1`) at 25% and 100% history.
