# Forward Risk Manager

This repo starts with a data conversion pipeline from QuantConnect Research exports into tidy CSVs that are easy to feed into PyTorch/PyG.

## Data You Should Export
From QuantConnect Research, export daily history for:
- Prices for your symbol universe (include any benchmark ticker you plan to use for evaluation/risk targets).
- Constituents membership (date + holdings/weights), if using membership-based graph filtering.
- Optional: Coarse Universe data for market cap.

## Expected Tidy Outputs
The converter writes:
- `data/processed/prices.csv`
  - Columns: `date,ticker,open,high,low,close,adj_close,volume`
- `data/processed/constituents.csv`
  - Columns: `date,ticker,is_member,weight,sector,market_cap`

## Artifact Layout
- Per-run artifacts: `runs/experiments/<run_id>/` with `metrics/`, `plots/`, `diagnostics/`, `models/`, `logs/`, and `manifest.json`.
- Published snapshots: `reports/published/<topic>/`.
- Global report index: `reports/index.csv`.

Keep raw/intermediate outputs out of `reports/` root.

One-time legacy consolidation:
```bash
python scripts/consolidate_artifacts.py --run-id legacy-20260211-colab
```

Publish curated artifacts from a run:
```bash
python scripts/publish_run.py --run-id <run_id>
```

## Current Results Snapshot (as of 2026-02-11)
Primary run analyzed: `runs/experiments/long_constituents/`.

Published index status:
- `reports/index.csv` currently has mixed provenance.
- Most scenario/hallucination/sweep benchmark metrics are from `long_constituents`.
- `train/*` published snapshots are still from `legacy-20260211-colab`.
- `goodness_backtest.csv` currently exists in `legacy-20260211-colab` only.

Benchmark (`runs/experiments/long_constituents/metrics/benchmark.csv`):

| mode | objective | eval_acc | eval_sep / eval_sc_gap | avg_epoch_s | graphs_per_s |
|---|---|---:|---:|---:|---:|
| `ff_layerwise` | `ff` | 0.866 | 1.719 (`eval_sep`) | 1.185 | 2383.5 |
| `ff_e2e` | `self_contrastive` | 0.999 | 0.730 (`eval_sc_gap`) | 1.169 | 2415.0 |
| `backprop` | `bce` | 0.983 | 0.548 (`eval_sep`) | 0.832 | 3395.3 |

`eval_acc` is a secondary metric. Prefer objective-aware separation (`eval_sep`/`eval_sc_gap`) plus AUROC/AUPRC/calibration metrics for model selection.

Sweep best (`runs/experiments/long_constituents/metrics/ff_sweep.csv`):
- Best objective-aware row (rank metric `eval_sep`): `2.071` in `ff_layerwise`.
- Speed for that row: `2818.1` graphs/s, `avg_epoch_s=1.002`.
- Key params: `goodness_temp=0.0524`, `goodness_target=3.4908`, `hall_steps=14`, `hall_lr=0.033614`, `hall_node_fraction=0.25`.

Scenario + stress (`runs/experiments/long_constituents/diagnostics/scenario_constraint_diagnostics.csv`, `runs/experiments/long_constituents/metrics/stress_test_report.csv`):
- Scenario constraint hit rate: `58.7%` over 150 scenarios.
- Mean target absolute error: `2.43%` (target drop is `-10%`).
- Mean non-target drift: `0.094%` absolute.
- Stress deltas (hallucinated minus real), `all` scope mean absolute:
  - `|delta total_return| = 0.12%`
  - `|delta volatility| = 0.04%`
  - `|delta cvar_95| = 0.08%`
- Target scope is intentionally harder and shows larger shifts (mean absolute `delta total_return ~= 15.75%`).

Hallucination calibration (`runs/experiments/long_constituents/diagnostics/hallucination_calibration_by_ticker.csv`):
- Median corr(real, halluc): `0.9937`.
- Median JS divergence: `0.0047`.
- Median tail ratio p99 (hall/real): `0.976`.
- Largest JS outliers currently include: `AEIS`, `INDB`, `CMP`, `MOH`, `SARO`.

## Converter Usage
Place your raw exports under `data/raw/` (any filenames). Then run:

```bash
python scripts/qc_export_to_tidy.py \
  --prices data/raw/prices \
  --constituents data/raw/constituents \
  --coarse data/raw/coarse \
  --out-dir data/processed
```

If your export columns have different names, override them, for example:

```bash
python scripts/qc_export_to_tidy.py \
  --prices data/raw/prices \
  --constituents data/raw/constituents \
  --price-date-col time \
  --price-ticker-col symbol \
  --adj-close-col adjusted_close \
  --constituent-ticker-col constituent_symbol \
  --weight-col weight
```

Note: If you pass a directory to `--prices`, every CSV in that directory is treated as price data. Keep constituents/coarse in separate directories or pass explicit files.

## Rolling Correlation Graphs
Build rolling correlation graphs using a window size (in trading days). The correlation matrix for each graph is computed from the last `window` days ending at each date.

For leakage-safe forecasting experiments, lag what the graph can see:
- `corr_lag_days`: lag for correlation edge construction.
- `feature_lag_days`: lag for node features.
- `membership_lag_days`: lag for constituents membership lookup.

Example (20-day window, top-10 edges per node):

```bash
python scripts/build_graphs.py --config configs/default.toml
```

You can also use a correlation threshold instead of top-k:

```bash
python scripts/build_graphs.py \
  --prices data/processed/prices.csv \
  --constituents data/processed/constituents.csv \
  --window 20 \
  --corr-lag-days 1 \
  --feature-lag-days 1 \
  --corr-threshold 0.3 \
  --out data/processed/graphs.pt
```

Tip: Use `--include-tickers <TICKER>` only if that ticker also exists in your prices data and you want it forced into membership.
You can disable the progress bar with `--no-progress` or `progress = false` in config.

## Optional: joblib Parallel Backend
If you want `joblib` parallelism for graph building on macOS, set:
```
[build_graphs]
parallel_backend = "joblib"
joblib_prefer = "threads"
joblib_n_jobs = 7
```

If joblib isn't installed, the builder will fall back to the threadpool backend.

## FF-GNN Training
Train a simple Forward-Forward GNN that uses graph topology during message passing and a per-graph goodness score.

```bash
python scripts/train_ff_gnn.py --config configs/default.toml
```

## Device Selection (Colab + macOS)
This scaffold uses PyTorch Geometric (PyG). The configs now default to `device = "auto"`, which picks:
- `cuda` first (recommended on Colab T4/A100/L4)
- `mps` next (Apple Silicon)
- `cpu` as fallback

Force a device explicitly if needed:
```bash
python scripts/train_ff_gnn.py --config configs/default.toml --device cuda
```

## Hallucinated Negatives
Set `neg_mode = "hallucinate"` in `configs/default.toml` to enable gradient-ascent negatives with realism constraints:
- L2 distance to original window
- Mean/std alignment
- Edge-correlation alignment

Tune the `hallucinate_*` fields in the config to control steps, learning rate, and penalty weights.
Recommended realism settings (especially for `window_plus_summary_fund`):
```
hallucinate_penalty_scope = "returns"
hallucinate_corr_scope = "returns"
hallucinate_freeze_non_return_features = true
```
These keep optimization focused on return-window channels while holding summary/fundamental channels fixed.

Temporal negatives are also supported:
```
neg_mode = "time_flip"
```
This flips the time window while keeping summary features unchanged (for `window_plus_summary`), teaching the model the arrow of time.

Additional hard negatives are available:
- `block_bootstrap`
- `cross_asset_mix`
- `phase_randomize`

## Feature Mode
`feature_mode = "window_plus_summary"` appends summary indicators to the raw return window:
- Realized volatility
- Momentum (sum of log returns)
- Volume shock
- Beta vs configured market ticker (`mdy_ticker`) or an automatic equal-weight market proxy when unavailable
- RSI

Ticker policy:
- `mdy_ticker = "AUTO"`: no hard dependency on a benchmark ETF; beta uses equal-weight market proxy if no explicit ticker is present.
- `risk_ticker = "AUTO"` / `econ_ticker = "AUTO"` / `target_ticker = "AUTO"`: scripts auto-select a viable ticker from available data and print the effective ticker used.

## GCN + Cached Edge Norm
Graphs now store normalized edge weights for faster GCN passes:
- `edge_norm = true`
- `edge_weight_mode = "raw"` (preserves correlation sign for adjacency)

## Layer-wise FF (Efficiency)
Set `ff_layerwise = true` to train layers sequentially using local FF losses.

## Partial Hallucinations (Efficiency)
Limit hallucination optimization to a subset of nodes:
```
hallucinate_node_fraction = 0.5
hallucinate_node_min = 20
```

You can also reduce correlation-penalty cost:
```
hallucinate_corr_every_n_steps = 2
hallucinate_corr_edge_fraction = 0.5
hallucinate_corr_edge_min = 32
```

## Plot Hallucinations
Generate a visual sanity check:

```bash
python scripts/plot_hallucination.py --config configs/default.toml
```

Pick a specific date:
```bash
python scripts/plot_hallucination.py --config configs/default.toml --date 2023-02-01
```

List available dates:
```bash
python scripts/plot_hallucination.py --config configs/default.toml --list-dates
```

Export the plotted windows to CSV:
```bash
python scripts/plot_hallucination.py --config configs/default.toml --save-csv runs/experiments/manual/diagnostics/hallucination_window.csv
```

Export all nodes to CSV:
```bash
python scripts/plot_hallucination.py --config configs/default.toml --save-csv-all runs/experiments/manual/diagnostics/hallucination_window_all.csv
```

## Goodness Temperature Sweep
Quickly probe how `goodness_temp` changes the scale of goodness without training:
```bash
python scripts/train_ff_gnn.py --config configs/default.toml --temp-sweep 0.25,0.5,1.0,2.0
```

You can also use a warm-start schedule:
```
neg_mode = "schedule"
neg_warmup_epochs = 8
```

Or a mixed schedule (recommended for harder negatives):
```
neg_mode = "mix"
neg_warmup_epochs = 20
neg_mix_start = 0.0
neg_mix_end = 0.7
neg_mix_ramp_epochs = 20
```

Or switch to self-contrastive FF (no explicit synthetic negatives):
```
neg_mode = "self_contrastive"
self_contrastive_temp = 0.2
self_contrastive_view_mode = "shuffle+noise"
self_contrastive_view_noise_std = 0.05
```

Stability add-ons:
- `neg_gate_margin`: if hallucinated negatives are too strong (`g_neg > g_pos + margin`), fall back to shuffle for that batch.
- `grad_clip`: gradient norm clipping to reduce instability.

Distance-forward auxiliary loss (graph-level pairwise margin):
```
distance_forward_weight = 0.05
distance_forward_margin = 0.15
```
This is applied in end-to-end FF mode (`ff_e2e` / non-layerwise) and can be combined with either synthetic negatives or `self_contrastive`.

## Training Plots
Set `log_csv` and `plot_path` in `configs/default.toml` to write a CSV of per-epoch metrics and a PNG plot.
Recommended run location: `runs/experiments/<run_id>/metrics/ff_train.csv` and `runs/experiments/<run_id>/plots/ff_train.png`.
Published snapshots should live under `reports/published/`.
The CSV now includes `hall_hardness` (avg `g_neg - g_pos` for hallucinated batches).
It also includes `dist_forward_loss` when distance-forward auxiliary training is enabled.

## Baseline Config
The current tuned baseline is stored at `configs/baseline.toml`.

## GPU Batch Auto-Tune
Set `auto_tune_batch = true` in `configs/default.toml` to probe larger batch sizes on CUDA or MPS and pick the biggest that fits.

## Parallelism Knobs
Colab-safe defaults in the configs are:
```
loader_workers = 0
torch_num_threads = 2
torch_num_interop_threads = 1
dataloader_persistent_workers = false
dataloader_prefetch_factor = 2
dataloader_pin_memory = true
dataloader_mp_context = ""
```
If your runtime is stable with multi-worker loading, you can increase `loader_workers`.

## Benchmarking (FF vs Backprop)
Run a small benchmark to compare speed and outcomes between:
- `ff_layerwise` (layer-wise FF)
- `ff_e2e` (end-to-end FF)
- `backprop` (standard supervised classifier on pos/neg)

```bash
python scripts/benchmark_training.py --config configs/default.toml
```

Customize via `configs/default.toml`:
```
[benchmark]
epochs = 5
batch_size = 32
eval_frac = 0.2
neg_mode = "mix"
eval_neg_mode = "auto"
eval_neg_modes = ["time_flip", "block_bootstrap", "cross_asset_mix", "phase_randomize"]
self_contrastive_eval_view_mode = "shuffle+noise"
self_contrastive_eval_noise_std = 0.05
ece_bins = 10
timing_warmup_epochs = 1
econ_enabled = true
econ_ticker = "AUTO"
econ_signal_window = 126
econ_signal_quantile = 0.5
econ_turnover_cost_bps = 0.0
out_csv = "runs/experiments/default/metrics/benchmark.csv"
```

For expanding walk-forward validation instead of a single holdout:
```
[benchmark]
split_mode = "walk_forward"
walk_forward_train_frac = 0.6
walk_forward_eval_frac = 0.2
walk_forward_step_frac = 0.1
walk_forward_min_train_graphs = 128
walk_forward_min_eval_graphs = 32
walk_forward_max_folds = 0
walk_forward_out_csv = "runs/experiments/default/metrics/benchmark_walk_forward_folds.csv"
```
This writes aggregate metrics (mean/std across folds) to `out_csv` and fold-level metrics with date ranges to `walk_forward_out_csv`.

The CSV includes `avg_epoch_s`, `graphs_per_s`, and outcome metrics like `eval_sep`, `eval_auroc`, `eval_auprc`, `eval_brier`, `eval_ece`, plus thresholded `eval_acc`.
It also appends economic columns (`econ_strategy_*`, `econ_bh_*`, `econ_ann_return_uplift`, `econ_sharpe_uplift`) computed from goodness-driven risk-on/off signals on the eval window.
If `eval_neg_mode = "auto"` with `neg_mode = "self_contrastive"`, benchmarking reports contrastive metrics (`eval_sc_loss`, `eval_sc_pos`, `eval_sc_neg`, `eval_sc_gap`, `eval_sc_acc`) and maps `eval_sep/eval_acc` to that objective.
`self_contrastive_eval_view_mode` and `self_contrastive_eval_noise_std` let you make retrieval eval harder than plain tiny-noise views.
This means `ff_e2e` can report near-1.0 `eval_acc` without being directly comparable to FF-separation rows; compare `eval_sc_gap` for self-contrastive rows and `eval_sep` for FF rows.
If `self_contrastive_eval_view_mode` is much harsher than training views (e.g. `time_flip+noise` at higher noise), `eval_sc_acc` can collapse even when training loss improves.
For e2e stability, start with matching views (`self_contrastive_view_mode = "noise"` and `self_contrastive_eval_view_mode = "noise"`) before increasing augmentation strength.

The script also writes a speed-vs-separation plot:
```
runs/experiments/default/plots/benchmark_speed_sep.png
```

And a bar chart summary:
```
runs/experiments/default/plots/benchmark.png
```

## Auto-Sweep (FF Hyperparams)
Run a lightweight grid search over FF settings with finance-first or objective-first ranking:

```bash
python scripts/ff_sweep.py --config configs/default.toml
```

Configure the sweep in `configs/default.toml`:
```
[sweep]
epochs = 3
batch_size = 32
eval_frac = 0.2
out_csv = "runs/experiments/default/metrics/ff_sweep.csv"
rank_mode = "finance_first"
econ_enabled = true
econ_ticker = "AUTO"
econ_signal_window = 126
econ_signal_quantile = 0.5
econ_turnover_cost_bps = 0.0
modes = ["ff_layerwise", "ff_e2e"]
goodness_temp = [0.25, 0.5]
goodness_target = [2.0, 2.5]
neg_mix_end = [0.3, 0.5]
hall_steps = [1, 3]
hall_lr = [0.03, 0.05]
hall_node_fraction = [0.1, 0.2]
top_k = 10
parallel_workers = 1
parallel_backend = "process"
parallel_mp_context = "spawn"
parallel_force_cpu = true
worker_torch_threads = 1
worker_torch_interop_threads = 1
worker_loader_workers = 0
```

`rank_mode` accepts:
- `finance_first` (default when economic metrics are enabled): ranks by `econ_sharpe_uplift` then `econ_ann_return_uplift`.
- `objective`: ranks by objective-aware separation (`eval_sep` / `eval_sc_gap`).
- any metric column name (e.g. `rank_mode = "econ_strategy_ann_return"`).

Plot sweep tradeoffs:
```bash
python scripts/plot_ff_sweep.py --csv runs/experiments/default/metrics/ff_sweep.csv
```

Pareto frontier plot:
```bash
python scripts/plot_ff_sweep.py --csv runs/experiments/default/metrics/ff_sweep.csv --pareto-out runs/experiments/default/plots/ff_sweep_pareto.png
```

Hallucination diagnostics (distribution overlay + diff histogram):
```bash
python scripts/plot_hallucination_diagnostics.py --csv runs/experiments/manual/diagnostics/hallucination_window_all.csv
```

Calibrate hallucinations (KL/JS + tail ratios):
```bash
python scripts/hallucination_calibration.py --csv runs/experiments/manual/metrics/scenario_book.csv
```
Add `--target-ticker` for focused diagnostics and `--out-by-ticker` for per-ticker calibration tables.

## Scenario Book + Stress Test Report
Generate a scenario book from multiple windows:
```bash
python scripts/scenario_book.py --config configs/default.toml --num-scenarios 10 --out runs/experiments/default/metrics/scenario_book.csv
```

You can also set defaults in `configs/default.toml`:
```
[scenario_book]
num_scenarios = 50
target_ticker = "AUTO"
target_drop = -0.10
constraint_mode = "exact"
constraint_tolerance = 0.01
constraint_weight = 20.0
adaptive = true
target_hit_rate = 0.6
target_tolerance = 0.01
max_adapt_steps = 40
diag_out = "runs/experiments/default/diagnostics/scenario_constraint_diagnostics.csv"
out = "runs/experiments/default/metrics/scenario_book.csv"
```
Then run:
```bash
python scripts/scenario_book.py --config configs/default.toml
```

Constrained “dreaming” (pick a ticker that exists in your graphs):
```bash
python scripts/scenario_book.py --config configs/default.toml \
  --num-scenarios 10 \
  --target-ticker AUTO \
  --target-drop -0.10 \
  --constraint-weight 10.0
```

Generate a stress test report (portfolio-level metrics + plot):
```bash
python scripts/stress_test_report.py --csv runs/experiments/default/metrics/scenario_book.csv --out-csv runs/experiments/default/metrics/stress_test_report.csv --out-plot runs/experiments/default/plots/stress_test_report.png
```
Add `--target-ticker` to produce `all`, `target`, and `non_target` scope diagnostics.
When `--target-ticker` is set, the report also synthesizes a `baseline_cov` scenario and writes `delta_vs_baseline_*` columns for direct hallucinated-vs-baseline comparison.

## Goodness Backtest
Check whether low goodness predicts higher forward volatility/drawdown:
```bash
python scripts/goodness_backtest.py --config configs/default.toml --ticker AUTO --horizons 5,21
```

This now also writes:
- regime/OOD summary CSV (`--out-events`) with `ood_auroc_low_goodness_vs_high_vol`
- strategy metrics CSV (`--out-strategy`) with `ann_return`, `ann_vol`, `sharpe`, `max_drawdown`, and `cvar_95_daily`
- goodness timeline plot (`--out-timeline`) with 2008/2020/2022 highlighted

## Sanity Checks
Run anti-triviality checks on benchmark outputs:
```bash
python scripts/sanity_checks.py --benchmark-csv runs/experiments/default/metrics/benchmark.csv
```

Generate a sweep summary report (top-K + Pareto):
```bash
python scripts/ff_sweep_summary.py --csv runs/experiments/default/metrics/ff_sweep.csv --out runs/experiments/default/logs/ff_sweep_summary.txt
```

Run a dedicated E2E-only sweep profile:
```bash
python scripts/ff_sweep.py --config configs/train_long_constituents.toml --section sweep_e2e
python scripts/ff_sweep_summary.py --csv runs/experiments/long_constituents/metrics/ff_sweep_e2e.csv --out runs/experiments/long_constituents/logs/ff_sweep_e2e_summary.txt
```

Generate a dual-track recommendation report (best `ff_e2e` by accuracy-focused score, best `ff_layerwise` by speed-focused score):
```bash
python scripts/dual_score_report.py \
  --benchmark runs/experiments/default/metrics/benchmark.csv \
  --sweep runs/experiments/default/metrics/ff_sweep.csv \
  --sweep-e2e runs/experiments/default/metrics/ff_sweep_e2e.csv \
  --out runs/experiments/default/logs/dual_score_report.txt \
  --out-csv runs/experiments/default/metrics/dual_score_report.csv
```
Use `--e2e-min-acc-ratio-vs-backprop` to down-rank e2e picks that are far below backprop accuracy.

## Long-History Data (2000–2024)
If your raw exports are split into year buckets under `data/raw/`, merge + clean them:

```bash
python scripts/merge_raw_years.py --raw-root data/raw --out-dir data/raw_merged
```

The merge script now resolves duplicate `(date,ticker)` price rows by selecting the smoothest per-ticker path across overlapping yearly exports, then deduplicates all merged tables by key.

Build graphs in two modes:

```bash
# Constituents only (2011+)
python scripts/build_graphs.py --config configs/long_constituents.toml

# All tickers (full history)
python scripts/build_graphs.py --config configs/long_alltickers.toml
```

The constituents config enables forward-fill (`membership_fill = "ffill"`) with a max gap of 63 days to reduce missing windows. You can disable by removing those settings.

Train against each graph set:

```bash
python scripts/train_ff_gnn.py --config configs/train_long_constituents.toml
python scripts/train_ff_gnn.py --config configs/train_long_alltickers.toml
```

## Sweep Parallelism Auto-Tuner
Find the fastest sweep parallelism settings on your machine:

```bash
python scripts/tune_sweep_parallel.py --config configs/default.toml
```

Configure candidates in `configs/default.toml`:
```
[sweep_tune]
out_csv = "runs/experiments/default/metrics/sweep_parallel_tune.csv"
device = "cpu"
epochs = 1
batch_size = 32
sample_graphs = 64
max_batches = 2
neg_mode = "shuffle"
parallel_backend = "process"
parallel_mp_context = "spawn"
parallel_workers = [1, 2, 3, 4]
worker_torch_threads = [1, 2]
worker_torch_interop_threads = [1]
worker_loader_workers = [0, 1]
apply = true
apply_to = "configs/default.toml"
apply_section = "sweep"
apply_min_improvement = 0.1
apply_backup = true
apply_backup_suffix = ".bak"
isolate_thread_settings = true
```

## Hard-Negative Curriculum
Optionally ramp hallucination strength over time:
```
[train.hallucinate_curriculum]
enabled = true
start_epoch = 10
ramp_epochs = 20
steps_start = 1
steps_end = 4
lr_start = 0.02
lr_end = 0.05
l2_start = 0.08
l2_end = 0.05
corr_start = 0.1
corr_end = 0.3
node_fraction_start = 0.2
node_fraction_end = 0.5
node_min_start = 10
node_min_end = 20
```

## Layer-wise Negatives (Advanced)
If you enable `ff_layerwise`, you can strengthen negatives in deeper layers:
```
layerwise_neg_mode = "shuffle+noise"
layerwise_noise_std = 0.08
layerwise_hall_corr = 0.0
layerwise_hall_mean = 0.01
layerwise_hall_std = 0.01
```
These settings apply only to layer-wise training.

## Notes
- If you don’t have `adj_close`, the converter falls back to `close`.
- If constituents don’t include `is_member`, it defaults to 1.
- `market_cap` is added if coarse data is provided and the column can be inferred.
