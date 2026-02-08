# Forward Risk Manager

This repo starts with a data conversion pipeline from QuantConnect Research exports into tidy CSVs that are easy to feed into PyTorch/PyG.

## Data You Should Export
From QuantConnect Research, export daily history for:
- Prices for your symbol universe (S&P 400 constituents via MDY proxy).
- MDY ETF constituents (date + holdings/weights).
- Optional: Coarse Universe data for market cap.

## Expected Tidy Outputs
The converter writes:
- `data/processed/prices.csv`
  - Columns: `date,ticker,open,high,low,close,adj_close,volume`
- `data/processed/constituents.csv`
  - Columns: `date,ticker,is_member,weight,sector,market_cap`

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
  --corr-threshold 0.3 \
  --out data/processed/graphs.pt
```

Tip: Use `--include-tickers MDY` if you want the ETF as a global context node even though it’s not in the constituents list.
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

## PyG + MPS Note (macOS)
This scaffold uses PyTorch Geometric (PyG). For Apple Silicon, set `device = "mps"` in `configs/default.toml`. If you run into unsupported ops, switch to `cpu`.

## Hallucinated Negatives
Set `neg_mode = "hallucinate"` in `configs/default.toml` to enable gradient-ascent negatives with realism constraints:
- L2 distance to original window
- Mean/std alignment
- Edge-correlation alignment

Tune the `hallucinate_*` fields in the config to control steps, learning rate, and penalty weights.

Temporal negatives are also supported:
```
neg_mode = "time_flip"
```
This flips the time window while keeping summary features unchanged (for `window_plus_summary`), teaching the model the arrow of time.

## Feature Mode
`feature_mode = "window_plus_summary"` appends summary indicators to the raw return window:
- Realized volatility
- Momentum (sum of log returns)
- Volume shock
- Beta vs MDY
- RSI

## GCN + Cached Edge Norm
Graphs now store normalized edge weights for faster GCN passes:
- `edge_norm = true`
- `edge_weight_mode = "abs"` (uses absolute correlation for adjacency)

## Layer-wise FF (Efficiency)
Set `ff_layerwise = true` to train layers sequentially using local FF losses.

## Partial Hallucinations (Efficiency)
Limit hallucination optimization to a subset of nodes:
```
hallucinate_node_fraction = 0.5
hallucinate_node_min = 20
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
python scripts/plot_hallucination.py --config configs/default.toml --save-csv reports/hallucination_window.csv
```

Export all nodes to CSV:
```bash
python scripts/plot_hallucination.py --config configs/default.toml --save-csv-all reports/hallucination_window_all.csv
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

Stability add-ons:
- `neg_gate_margin`: if hallucinated negatives are too strong (`g_neg > g_pos + margin`), fall back to shuffle for that batch.
- `grad_clip`: gradient norm clipping to reduce instability.

## Training Plots
Set `log_csv` and `plot_path` in `configs/default.toml` to write a CSV of per-epoch metrics and a PNG plot.
Recommended publishable location: `reports/ff_train.csv` and `reports/ff_train.png`.
The CSV now includes `hall_hardness` (avg `g_neg - g_pos` for hallucinated batches).

## Baseline Config
The current tuned baseline is stored at `configs/baseline.toml`.

## MPS Batch Auto-Tune
Set `auto_tune_batch = true` in `configs/default.toml` to probe larger batch sizes on MPS and pick the biggest that fits.

## Parallelism Knobs (macOS)
For faster data loading and CPU ops:
```
loader_workers = 3
torch_num_threads = 8
torch_num_interop_threads = 2
dataloader_persistent_workers = true
dataloader_prefetch_factor = 2
dataloader_mp_context = "spawn"
```

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
eval_neg_mode = "shuffle"
timing_warmup_epochs = 1
out_csv = "reports/benchmark.csv"
```

The CSV includes `avg_epoch_s`, `graphs_per_s`, and outcome metrics like `eval_acc`, `eval_g_pos`, `eval_g_neg`, and `eval_sep`.

The script also writes a speed-vs-separation plot:
```
reports/benchmark_speed_sep.png
```

And a bar chart summary:
```
reports/benchmark.png
```

## Auto-Sweep (FF Hyperparams)
Run a lightweight grid search over FF settings and rank by `eval_sep`:

```bash
python scripts/ff_sweep.py --config configs/default.toml
```

Configure the sweep in `configs/default.toml`:
```
[sweep]
epochs = 3
batch_size = 32
eval_frac = 0.2
out_csv = "reports/ff_sweep.csv"
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

Plot sweep tradeoffs:
```bash
python scripts/plot_ff_sweep.py --csv reports/ff_sweep.csv
```

Pareto frontier plot:
```bash
python scripts/plot_ff_sweep.py --csv reports/ff_sweep.csv --pareto-out reports/ff_sweep_pareto.png
```

Hallucination diagnostics (distribution overlay + diff histogram):
```bash
python scripts/plot_hallucination_diagnostics.py --csv reports/hallucination_window_all.csv
```

Calibrate hallucinations (KL/JS + tail ratios):
```bash
python scripts/hallucination_calibration.py --csv reports/hallucination_window_all.csv
```

## Scenario Book + Stress Test Report
Generate a scenario book from multiple windows:
```bash
python scripts/scenario_book.py --config configs/default.toml --num-scenarios 10 --out reports/scenario_book.csv
```

You can also set defaults in `configs/default.toml`:
```
[scenario_book]
num_scenarios = 50
target_ticker = "MDY"
target_drop = -0.10
constraint_weight = 20.0
adaptive = true
target_hit_rate = 0.6
target_tolerance = 0.01
max_adapt_steps = 40
diag_out = "reports/scenario_constraint_diagnostics.csv"
out = "reports/scenario_book.csv"
```
Then run:
```bash
python scripts/scenario_book.py --config configs/default.toml
```

Constrained “dreaming” (pick a ticker that exists in your graphs; e.g., MDY):
```bash
python scripts/scenario_book.py --config configs/default.toml \
  --num-scenarios 10 \
  --target-ticker MDY \
  --target-drop -0.10 \
  --constraint-weight 10.0
```

Generate a stress test report (portfolio-level metrics + plot):
```bash
python scripts/stress_test_report.py --csv reports/scenario_book.csv --out-csv reports/stress_test_report.csv --out-plot reports/stress_test_report.png
```

## Goodness Backtest
Check whether low goodness predicts higher forward volatility/drawdown:
```bash
python scripts/goodness_backtest.py --config configs/default.toml --ticker MDY --horizons 5,21
```

Generate a sweep summary report (top-K + Pareto):
```bash
python scripts/ff_sweep_summary.py --csv reports/ff_sweep.csv --out reports/ff_sweep_summary.txt
```

## Long-History Data (2000–2024)
If your raw exports are split into year buckets under `data/raw/`, merge + clean them:

```bash
python scripts/merge_raw_years.py --raw-root data/raw --out-dir data/raw_merged
```

These exports are already tidy (date/ticker columns), so you can use `data/raw_merged` directly.

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
Find the fastest sweep parallelism settings on your Mac:

```bash
python scripts/tune_sweep_parallel.py --config configs/default.toml
```

Configure candidates in `configs/default.toml`:
```
[sweep_tune]
out_csv = "reports/sweep_parallel_tune.csv"
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
