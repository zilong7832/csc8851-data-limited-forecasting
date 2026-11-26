# CSC 8851 final evaluation protocol

## Scientific question

With only a recent stretch of oil-temperature readings, when does a supervised patch Transformer justify its computation relative to simpler forecasters? The primary comparison is test mean absolute error (MAE) in degrees Celsius. Report RMSE alongside MAE. Use within-setting comparisons and acknowledge ties and reversals. Report the normalized patch model even if a baseline performs better.

## Data and models

Use the bundled public [Electricity Transformer Temperature data](https://github.com/zhouhaoyi/ETDataset). Use only hourly OT. Each station is trained separately. The first 8,640 hours form the training period, the next 2,880 the validation period, the next 2,880 the test period. A model sees 168 hours and predicts all H future temperatures directly. Train on the most recent 2,160, 4,320 or 8,640 contiguous training hours with the same endpoint. At H=24 the training-window counts are 1,969 / 4,129 / 8,449; at H=96 they are 1,897 / 4,057 / 8,377. Each validation and test period has 2,857 or 2,785 origins, respectively. Test and validation targets lie inside their own periods. Their context can include the preceding period and any history already observed when forecasting. The remaining rows after the test period are unused. History length also changes which seasons enter training.

Fit the global mean and population standard deviation **only** to each active training subset. The four learning models also use the same non-affine reversible window normalization by default. At each origin, subtract the 168-hour sample mean, divide by the sample standard deviation plus 1e-5 in training-standardized units, fit or run the model, then restore the input-window statistics. Targets are never used to normalize inputs. Training MSE is evaluated on the restored prediction in training-standardized units, and reported errors are restored to °C.

| Model | Forecast calculation |
|---|---|
| Seasonal naive | Tile the last 24 hours to length H. No fitted parameters. |
| Direct linear | Single affine map from 168 values to H outputs. |
| MLP | 168 to 64 hidden units, ReLU, then H outputs. |
| LSTM | Single 32-unit LSTM over 168 scalar time steps; last hidden state projects to H. |
| Patch Transformer | Seven disjoint 24-hour tokens, 64-dimensional embedding, learned position vectors, two pre-norm encoder blocks, four heads, 128-unit feed-forward layers, GELU, dropout 0.1. Flatten seven final token vectors and project to H. |

The direct map and patch encoder reuse the preliminary implementation. The patch design adapts [PatchTST](https://arxiv.org/abs/2211.14730); direct forecasting is motivated by [Zeng et al.](https://arxiv.org/abs/2205.13504). The window wrapper follows non-affine [reversible instance normalization](https://github.com/ts-kim/RevIN). No external pretraining or station pooling. A separate diagnostic table reports last-value persistence at both sites and horizons.

## Selection, independent runs and timing

For each of 48 primary learned settings, compare Adam learning rates 0.001 and 0.0003, on seed 0. Each candidate uses all eligible hourly windows in shuffled batches of 64 for at most 100 epochs, early stopping after 10 epochs without a strict improvement in validation MSE. Choose the rate with the lowest saved validation MSE; exact ties use 0.001. Restore that checkpoint for seed 0. Independently train seeds 1 and 2 at the selected rate with the same stopping rule. All scientific fits use float32 on one CUDA GPU. Timing includes training and validation/early stopping, excluding data preparation, test inference and writing the checkpoint. It is reported for each fit, along with parameter count, realized epochs and best epoch. Times in the primary table refer to selected fits; tuning candidate costs are recorded separately. A selected seed-0 fit occurs in both tables and must be counted only once when totaling compute.

On ETTh1 at H=24 and 25% and 100% history, compare one with two Transformer layers. At the same settings, remove window normalization from the direct linear model and from the Transformer. Keep the matching primary model's selected rate, training windows, architecture apart from the indicated change, and seeds. Each variant still selects its best epoch on validation. The 96 tuning fits, 96 additional main-seed fits and 18 ablation fits are completed before unlocking test scoring. The final comparisons use the same frozen rules regardless of results. The code supports continuation at completed-fit and evaluation boundaries.

## Metrics, uncertainty and diagnostics

For each model, average absolute errors across all hourly origins and horizon steps. For RMSE, average squared errors over both axes before taking the square root. Then report the mean of each seed's metric and sample SD across the three seeds. The deterministic baseline has one score per cell; its seed SD field is empty. All time points carry equal weight per lead. The same realized timestamp may occur at multiple leads because origins overlap. Store per-lead MAE and RMSE and the mean absolute error over leads at each origin. Example traces are the first and middle available test origins, fixed in advance. PDF and PNG line plots show observed trajectories with seed-0 forecasts. The CSV also contains the other seeds and origins. Saved normalized training and validation losses expose optimization behavior and early stopping.

For the paired Transformer-minus-reference MAE contrast, first average absolute errors over seeds at each origin, preserving paired origins across models. A negative contrast favors the Transformer. Use a circular moving-block bootstrap on the hourly differences with 2,000 replicates, primary block length 168 hours and sensitivity lengths 96 and 336. The percentile intervals describe time variability conditional on these two observed test periods and an approximately stable dependence structure within each period. They do not represent station-level uncertainty or a multiple-comparison-adjusted superiority test. Seed SD describes training variability separately. The windows are observational and overlapping.

## Mechanism experiment and analytical prediction

For a window x, mean m and scale s, the wrapper computes f(x) = m + s g((x − m)/s). A constant c added to every input shifts m by c and leaves s unchanged. Therefore f(x+c) = f(x)+c in evaluation mode, up to floating-point error. Adding c to both history and target preserves absolute and squared errors. This identifies a structural property of normalization shared by architectures. An unobserved regime change that starts after the forecast origin has no such protection. The property does not establish a real-world accuracy ranking.

Generate 512 independent synthetic paths of length 192 with RNG seed 17026. Each uses a level drawn uniformly from 13 to 20 °C, a daily sine wave with amplitude uniform from 1 to 4 °C and phase uniform on [0, 2π), a slope uniform on [−0.008, 0.008] °C/hour and AR(1) noise of coefficient 0.8 with normal innovations of SD 0.45 °C. Initialize noise from its stationary distribution. For each path independently choose a shift of +6 or −6 °C with equal probability. Evaluate three paired versions of that path: unchanged, shift the entire input and future, or shift the future alone. Use frozen ETTh1 / 24 h / 25% models with their original training-subset scalers. All eligible main models and ablations see identical simulated paths. Report MAE, RMSE, paired change in MAE from the unchanged path, its Monte Carlo SE over independent paths, and the residual from the expected additive forecast response. This controlled simulation diagnoses a mechanism. It does not model the station's complete thermal dynamics.

## Reproduction and outputs

Follow [the run guide](../SERVER_GUIDE_ZH.md). The final return archive includes the per-seed test tables, depth and normalization contrasts, complete training histories for scored learned models, all learning-rate trial scores, per-lead and per-origin errors, paired intervals, window examples, simulation summaries, and figures. Parameters and GPU information are included through model counts, source hashes, data hashes and runtime records. The return archive includes all 162 scored learned checkpoints for new visualizations and independent verification without refitting. Figures can be regenerated from a completed run by repeating the same command.
