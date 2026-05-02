# Turing Econ Large Benchmark, Slurm 59230

This folder is the published, lightweight snapshot for the `turing_econ_large`
benchmark run associated with Slurm job `59230`.

## Provenance

- Git commit after publishing artifacts: `c777aeeed9075637e85c013610ab4a43d2cc20a8`
- Slurm log: `slurm-59230.out`
- Runtime config used by the cluster wrapper: `runtime_turing_econ_large.toml`
- Source config in the repo: `configs/turing_econ_large.toml`
- Slurm submit script in the repo: `slurm/benchmark_1gpu_econ_large.sbatch`
- Benchmark driver: `scripts/benchmark_training.py`
- Main evaluation code: `src/frisk/econ_eval.py`
- Benchmark semantics/reporting helpers: `src/frisk/benchmarking/semantics.py`

## Run Summary

- Graph artifact used by the run: `data/processed/graphs_master_ff_rich.pt.sharded`
- Benchmark device: CUDA
- Benchmark epochs: `20`
- Batch size: `128`
- Split mode: walk-forward
- Walk-forward fractions: train `0.6`, eval `0.2`, step `0.1`
- Seeds: `7`, `27`, `47`
- Economic ticker: `SPY`
- Economic signal window: `252`
- Economic signal quantile: `0.6`
- OOS economic folds: `4`
- Transaction cost assumptions: turnover cost `1.0` bps, slippage `0.5` bps
- Risk and portfolio heads were enabled in the runtime config
- The repaired benchmark pass used `repair_eval_only = true`

The Slurm output reports `status: ok`, `seed_num_failed_runs: 0`, and
`walk_forward_num_failed_folds: 0`.

## Published Artifacts

Metrics are under `metrics/`.

- `benchmark.csv`, `benchmark_baseline.csv`, `benchmark_history.csv`,
  `benchmark_walk_forward_folds.csv`: original benchmark outputs
- `benchmark_repaired.csv`, `benchmark_repaired_baseline.csv`,
  `benchmark_repaired_history.csv`, `benchmark_repaired_walk_forward_folds.csv`:
  repaired benchmark outputs
- `benchmark_repaired_summary_named.csv`: named summary table
- `benchmark_repaired_ff_vs_backprop_paired_slices.csv` and
  `benchmark_repaired_best_ff_vs_backprop_slices.csv`: comparison slices for
  forward-forward vs backprop analyses

Plots are under `plots/`.

## Deliberately Excluded

The raw data, processed graph shard, model checkpoints, and resume checkpoint
files are not stored here. They are large generated artifacts and remain outside
Git under `data/` and `runs/experiments/.../models/`.

For a paper, cite this folder for the exact published result tables and figures,
the runtime TOML for experimental settings, and the Slurm log for the run trace.
Full computational reproduction also requires the external graph shard and the
same data snapshot used to build it.
