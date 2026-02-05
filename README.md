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

## MPS Batch Auto-Tune
Set `auto_tune_batch = true` in `configs/default.toml` to probe larger batch sizes on MPS and pick the biggest that fits.

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

## Notes
- If you don’t have `adj_close`, the converter falls back to `close`.
- If constituents don’t include `is_member`, it defaults to 1.
- `market_cap` is added if coarse data is provided and the column can be inferred.
