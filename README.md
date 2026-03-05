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

## Current Results Snapshot (as of 2026-02-25)
Latest notebook run IDs:
- Latest paper benchmark (500 epochs): `runs/experiments/paper_final_500_20260225_002036/`
- Latest full end-to-end run (sweep/scenario/calibration/backtest): `runs/experiments/e2e_runbook_20260215_143702/`
- Recovery ablation outputs: `runs/experiments/recovery_ablation/`

Published index status (`reports/index.csv`):
- Published benchmark/scenario/sweep/hallucination/train snapshots currently point to `long_constituents` (timestamp `2026-02-24T22:19:12Z`).
- Paper benchmark artifacts are now tracked for GitHub under `reports/published/benchmark/`:
  - `paper_final_500_latest_summary.csv`
  - `paper_final_500_latest_summary.md`
  - `paper_final_500_latest_summary.json`
  - `paper_final_500_latest_benchmark.csv`
  - `paper_final_500_latest_walk_forward_folds.csv`
- `sweep/latest_tune.csv` still points to `legacy-20260211-colab`.
- Notebook runbook artifacts (`e2e_runbook_*` and `paper_final_500_*`) remain local run artifacts under `runs/`; use `reports/published/` for tracked snapshots.

Latest paper benchmark rerun (`runs/experiments/paper_final_500_20260225_002036/metrics/paper_benchmark_summary.csv`, walk-forward with 3 folds):

| mode | quality metric | quality value | eval_auroc | eval_auprc | avg_epoch_s | graphs_per_s | econ_ann_return_uplift | econ_sharpe_uplift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ff_layerwise` | `eval_sep` | `6.51199` | `0.62829` | `0.67070` | `1.349` | `1070.2` | `-26.70%` | `-0.223` |
| `ff_e2e` | `eval_sep` | `14.23771` | `1.00000` | `1.00000` | `0.706` | `2045.5` | `-27.76%` | `-0.338` |
| `backprop` | `eval_auroc` | `1.00000` | `1.00000` | `1.00000` | `0.340` | `4253.3` | `-29.28%` | `-0.448` |

Latest completed benchmark status:
- Best quality: `ff_e2e` (`eval_sep=14.23771`, `eval_auroc=1.00000`).
- Fastest mode: `backprop` (`4253.3` graphs/s).
- Best economics in this run (Sharpe-based): `ff_layerwise` (`econ_sharpe_uplift=-0.223`, `econ_ann_return_uplift=-26.70%`).

How to read this latest paper result:
- Quality and economics are still not aligned: `ff_e2e` has the strongest separation objective (`eval_sep`) but does not win economics.
- `backprop` is best on speed and classifier metrics, but all three modes are negative on both annual return uplift and Sharpe uplift versus buy-and-hold in this run.
- Fold-level economics remain unstable for FF modes (`runs/experiments/paper_final_500_20260225_002036/metrics/benchmark_walk_forward_folds.csv`): `ff_layerwise` and `ff_e2e` switch Sharpe-uplift sign across folds, while `backprop` stays negative across all folds.
- `objective_track` matters when reading rows: `ff_*` rows are critic/separation-tracked; `backprop` is classifier/`eval_auroc`-tracked.

Recovery ablation runbook (`notebooks/recovery_ablation_runbook.ipynb`, outputs from `runs/experiments/recovery_ablation/metrics/`, focused `risk_head` rerun generated 2026-02-14 21:32):

| ablation_id | mode | eval_sep | delta_eval_sep_vs_baseline | econ_sharpe_uplift | delta_econ_sharpe_uplift_vs_baseline | econ_ann_return_uplift | delta_econ_ann_return_uplift_vs_baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_updated` | `ff_e2e` | `6.760634` | `0.000000` | `0.958215` | `0.000000` | `0.018112` | `0.000000` |
| `risk_head_weight_low` (`0.02`) | `ff_e2e` | `6.760453` | `-0.000181` | `0.958215` | `0.000000` | `0.018112` | `0.000000` |
| `risk_head_weight_lower` (`0.01`) | `ff_e2e` | `6.780601` | `+0.019966` | `1.041995` | `+0.083781` | `0.031896` | `+0.013783` |
| `risk_head_weight_min` (`0.005`) | `ff_e2e` | `6.794633` | `+0.033999` | `1.061382` | `+0.103168` | `0.035976` | `+0.017864` |
| `risk_head_off` | `ff_e2e` | `7.141854` | `+0.381220` | `-0.182601` | `-1.140815` | `-0.185359` | `-0.203472` |

Recovery ablation interpretation (latest focused rerun):
- In this focused rerun, baseline used `train.risk_loss_weight = 0.02`; `risk_head_weight_low` therefore matched baseline economics.
- Best economics came from keeping risk-head enabled but lowering weight to `0.005` (`risk_head_weight_min`): strongest `ff_e2e` Sharpe/return uplift and best `backprop` Sharpe uplift. The repo default is now `train.risk_loss_weight = 0.005`.
- Disabling risk-head (`risk_head_off`) increased separation but hurt economics sharply for `ff_e2e` and `backprop`; this matches the 500-epoch benchmark trend that higher separation alone does not guarantee higher Sharpe.
- `ff_layerwise` remained economically negative across all tested risk-head settings (Sharpe uplift ~`-0.713` except worse when risk-head is off).

Sweep best from latest full run (`runs/experiments/e2e_runbook_20260215_143702/metrics/ff_sweep.csv`):
- Best finance-first rank metric: `econ_sharpe_uplift = 0.9436` (`ff_e2e`).
- Speed for that row: `2311.3` graphs/s (`avg_epoch_s = 0.624`).
- Key params: `goodness_temp=0.2`, `goodness_target=1.5`, `hall_steps=1`, `hall_lr=0.03`, `hall_node_fraction=0.2`, `neg_mix_end=0.5`.
- Sweep size: 256 rows; top objective-aware ranks are dominated by `ff_e2e`.

Scenario + stress from latest full run (`runs/experiments/e2e_runbook_20260215_143702/diagnostics/scenario_constraint_diagnostics.csv`, `runs/experiments/e2e_runbook_20260215_143702/metrics/stress_test_report.csv`):
- Scenario constraint hit rate: `72.0%` over 50 scenarios.
- Mean target absolute error: `1.25%` (target drop is `-10%`).
- Mean non-target drift: `0.35%` absolute.
- Stress deltas (hallucinated minus real), `all` scope mean absolute:
  - `|delta total_return| = 0.58%`
  - `|delta max_drawdown| = 0.67%`
  - `|delta volatility| = 0.14%`
  - `|delta cvar_95| = 0.27%`
- Relative to baseline covariance stress, hallucinated scenarios are closer to real in `98%` of scenarios for total return and `88%` for max drawdown.

Hallucination calibration from latest full run (`runs/experiments/e2e_runbook_20260215_143702/diagnostics/hallucination_calibration.json`, `runs/experiments/e2e_runbook_20260215_143702/diagnostics/hallucination_calibration_by_ticker.csv`):
- Global corr(real, halluc): `0.9727`.
- MAE: `0.00346`, RMSE: `0.00556`.
- Tail ratio p99 (hall/real): `1.0382` (slight high-tail overshoot).
- Worst MAE tickers include: `DNR`, `SATS`, `ROSE`, `ENH`, `SVU`.

Goodness backtest (`runs/experiments/e2e_runbook_20260215_143702/diagnostics/goodness_strategy_metrics.csv`):
- `goodness_risk_on_off` vs buy-and-hold: lower annual return (`11.33%` vs `15.07%`) but better Sharpe (`0.586` vs `0.525`) and shallower max drawdown (`-37.1%` vs `-44.2%`).

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

## Local `Data FF` Consolidation
If your raw dataset lives outside Drive (for example `/Users/<you>/Desktop/Data FF`), consolidate it into a few CSVs first:

```bash
python scripts/consolidate_data_ff.py \
  --data-ff-root "/Users/<you>/Desktop/Data FF" \
  --out-dir "data/consolidated_ff_local"
```

This writes:
- `data/consolidated_ff_local/prices.csv`
- `data/consolidated_ff_local/macro.csv`
- `data/consolidated_ff_local/sec_submissions_entities.csv`
- `data/consolidated_ff_local/sec_companyfacts_selected.csv`

For a fast smoke run:

```bash
python scripts/consolidate_data_ff.py \
  --data-ff-root "/Users/<you>/Desktop/Data FF" \
  --out-dir "data/consolidated_ff_local_smoke" \
  --max-sec-files 1000 \
  --skip-companyfacts
```

If your DATA FF folder uses a `submissions` subfolder (not `SEC_XBRL_submissions`), pass `--submissions-dir submissions`.

After consolidation, you can build the **master graph** once (locally or on Colab) and use it for all runs. Sync `data/consolidated_ff_local/` to Drive, then on Colab open `notebooks/colab_setup.ipynb` or run `build_graphs.py` with `configs/master_graph_ff.toml`; the notebooks auto-detect consolidated data and use it for graph build and economic-profit–focused train/benchmark.

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

Each graph now carries relation metadata in addition to edge weights:
- `edge_relation_mask`: bitmask over edge sources (`corr_pos`, `corr_neg`, `lead_lag`, `sector_static`, `static_overlay`)
- `edge_lag_days`: lag depth for temporal lead-lag edges
- `edge_type`: compact primary relation id for relation-aware encoders

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

To use relation-aware message passing, set:
```toml
[train]
encoder_conv_type = "rgcn"
encoder_rgcn_num_relations = 8
```

### Encoder/Critic Split (Recommended)
Use explicit two-component training when you want strict role separation:

1. Encoder stage (`self_contrastive`, no time-flip gating):
```bash
python scripts/train_ff_gnn.py --config configs/default.toml \
  --neg-mode self_contrastive \
  --strict-component-split \
  --save-encoder runs/experiments/default/models/encoder.pt
```

2. Critic stage (FF discrimination with time-flip negatives, frozen encoder):
```bash
python scripts/train_ff_gnn.py --config configs/default.toml \
  --neg-mode time_flip+noise \
  --strict-component-split \
  --encoder-checkpoint-in runs/experiments/default/models/encoder.pt \
  --freeze-encoder \
  --save-critic runs/experiments/default/models/critic.pt
```

Or run both stages in sequence:
```bash
python scripts/train_two_stage.py --config configs/default.toml
```
`train_two_stage.py` reads optional `[encoder]` and `[critic]` sections for stage-specific overrides.

`scenario_book.py` now accepts `--critic-model` and, when `strict_component_split = true`, requires a critic checkpoint.

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
- `ff_rank_aux_weight`: rank-spread auxiliary on goodness.
- `ff_rank_use_portfolio_targets = true`: when graph dates are available, rank aux is aligned to forward-return targets (`portfolio_ticker`/`portfolio_horizon`) instead of pure unsupervised spread.

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

### Colab Notebook Resume (`notebooks/paper_final_benchmark_colab.ipynb`)
The paper notebook is set up to resume safely from benchmarking:
- Cell 3: set `RUN_ID_OVERRIDE` (for example `paper_final_500_20260218_024603`) to attach to an existing run instead of creating a new one.
- Cell 4: graph build auto-skips when `runs/experiments/<run_id>/data/graphs.pt` already exists (`FORCE_REBUILD_GRAPHS` can override).
- Cell 5: benchmark runs mode-by-mode (`ff_layerwise`, `ff_e2e`, `backprop`) and writes per-mode files:
  - `metrics/benchmark_ff_layerwise.csv`, `metrics/benchmark_ff_e2e.csv`, `metrics/benchmark_backprop.csv`
  - matching fold files `metrics/benchmark_walk_forward_folds_<mode>.csv`
- Re-running cell 5 skips completed modes and only executes missing ones, then merges back to:
  - `metrics/benchmark.csv`
  - `metrics/benchmark_walk_forward_folds.csv`
- If interruption happens mid-mode, that mode is rerun end-to-end (current benchmark script does not checkpoint inside a mode).

Customize via `configs/default.toml`:
```
[benchmark]
epochs = 5
batch_size = 32
eval_frac = 0.2
seeds = [7, 17, 29]
seed_bootstrap_samples = 2000
seed_bootstrap_alpha = 0.05
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
econ_short_borrow_bps = 0.0
econ_max_abs_exposure = 1.0
out_csv = "runs/experiments/default/metrics/benchmark.csv"
```
When `train.risk_head_enabled = true`, benchmark FF/backprop runs now include the same auxiliary risk loss path used in training and report `risk_loss_train`/`risk_head_enabled_effective`.

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
If you set multiple `benchmark.seeds`, the CSV includes per-seed rows plus an aggregate row with bootstrap confidence intervals (`*_ci_lo`, `*_ci_hi`).

The CSV includes `avg_epoch_s`, `graphs_per_s`, and outcome metrics like `eval_sep`, `eval_auroc`, `eval_auprc`, `eval_brier`, `eval_ece`, plus thresholded `eval_acc`.
It also appends economic columns (`econ_strategy_*`, `econ_bh_*`, `econ_ann_return_uplift`, `econ_sharpe_uplift`) computed from goodness-driven risk-on/off signals on the eval window.
If `eval_neg_mode = "auto"` with `neg_mode = "self_contrastive"`, benchmarking reports contrastive metrics (`eval_sc_loss`, `eval_sc_pos`, `eval_sc_neg`, `eval_sc_gap`, `eval_sc_acc`) and maps `eval_sep/eval_acc` to that objective.
`self_contrastive_eval_view_mode` and `self_contrastive_eval_noise_std` let you make retrieval eval harder than plain tiny-noise views.
Benchmark/sweep outputs include `objective_track` (`encoder`, `critic`, or `classifier`).
For `encoder` rows, extra `time_flip*` eval modes are skipped and listed in `eval_neg_modes_skipped` so arrow-of-time checks stay critic-only.
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

## Recovery Ablation Runner
Run a controlled one-factor-at-a-time ablation matrix covering:
- negative sampling/mix schedule
- risk-head weighting/disablement
- graph construction density/method
- goodness temperature + FF margin
- hallucination schedule strength
- split strategy (chronological vs walk-forward)

```bash
python scripts/recovery_ablation.py --config configs/default.toml
```

Outputs are written under `runs/experiments/recovery_ablation/`:
- `metrics/recovery_ablation_plan.csv` (full planned matrix)
- `metrics/recovery_ablation.csv` (consolidated benchmark rows + deltas vs baseline)
- `metrics/recovery_ablation_summary.csv` (best-by-family summary)

Useful options:
```bash
python scripts/recovery_ablation.py \
  --config configs/default.toml \
  --families negative_sampling,risk_head \
  --benchmark-epochs 3 \
  --modes ff_e2e,backprop
```

Dry-run (no graph builds or training):
```bash
python scripts/recovery_ablation.py --config configs/default.toml --dry-run
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
You can pin the critic checkpoint explicitly:
```bash
python scripts/scenario_book.py --config configs/default.toml --critic-model runs/experiments/default/models/critic.pt
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
critic_model = "runs/experiments/default/models/critic.pt"
adaptive = true
target_hit_rate = 0.6
target_tolerance = 0.01
max_adapt_steps = 40
diag_out = "runs/experiments/default/diagnostics/scenario_constraint_diagnostics.csv"
out = "runs/experiments/default/metrics/scenario_book.csv"
```
Scenario rows include metadata columns: `objective_track`, `energy_component`, `component_split_mode`, `encoder_checkpoint`, `critic_checkpoint`, `train_neg_mode`.
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
Notes:
- The easy-negative accuracy gate (`--easy-neg-acc-max`) is evaluated on critic/FF rows only. `backprop`/`bce` rows can legitimately saturate `eval_acc` and are excluded from this anti-triviality check.
- New benchmark aggregate rows preserve `objective_track` and `primary_eval_metric_name`. If you load an older `benchmark.csv` that predates this metadata, rerun `scripts/benchmark_training.py` (or use notebook fallback columns) before strict column assertions.

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
