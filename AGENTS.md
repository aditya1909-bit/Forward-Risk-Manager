# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Python ML/research project (Forward-Forward GNN risk modeling). No Docker, web servers, or databases are needed.

### Running services

There are no long-running services. All functionality runs as CLI scripts under `scripts/` against local files. See `README.md` for full CLI reference.

### Quick reference

- **Install deps**: `pip install -r requirements.txt && pip install -e .`
- **Run tests**: `pytest tests/` (97 tests, all CPU-only, ~4s)
- **Lint**: No linter is configured in the project. `ruff check src/ tests/ scripts/` can be used ad-hoc (pre-existing warnings exist).
- **Train**: `python scripts/train_ff_gnn.py --config configs/default.toml` (requires graph data under `data/processed/graphs.pt`)
- **Benchmark**: `python scripts/benchmark_training.py --config configs/default.toml`

### Non-obvious caveats

- The repo has no raw data (`data/` is gitignored). Tests use in-memory synthetic data and work without any data files. Running scripts like `build_graphs.py` or `train_ff_gnn.py` requires generating synthetic data or providing real QuantConnect exports.
- `pytest` is not listed in `requirements.txt`; install it separately.
- PyTorch Geometric pulls in many transitive dependencies on first install (~90s). Subsequent installs are fast.
- The `~/.local/bin` directory must be on PATH for `pytest`, `ruff`, and similar tools installed via `pip install --user`. Use `export PATH="$HOME/.local/bin:$PATH"` or ensure it's in your shell profile.
- `torch_compile = true` in configs requires a CUDA-capable GPU. Set `--no-torch-compile` or use `torch_compile = false` when running on CPU-only environments.
- The `auto_tune_batch` feature probes GPU memory; disable with `--no-auto-tune-batch` on CPU.
