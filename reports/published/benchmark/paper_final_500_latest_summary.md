# Paper Benchmark Summary

Source: `runs/experiments/paper_final_500_20260225_002036/metrics/benchmark.csv`

| mode | quality_metric | quality_value | eval_auroc | eval_auprc | avg_epoch_s | graphs_per_s | econ_sharpe_uplift | econ_ann_return_uplift |
|---|---|---|---|---|---|---|---|---|
| ff_layerwise | eval_sep | 6.5120 | 0.6283 | 0.6707 | 1.349 | 1070.2 | -0.223 | -26.70% |
| ff_e2e | eval_sep | 14.2377 | 1.0000 | 1.0000 | 0.706 | 2045.5 | -0.338 | -27.76% |
| backprop | eval_auroc | 1.0000 | 1.0000 | 1.0000 | 0.340 | 4253.3 | -0.448 | -29.28% |

## Key Points
- Fastest: `backprop` (4253.3 graphs/s).
- Best quality: `ff_e2e` (eval_sep=14.2377).
- Best economics: `ff_layerwise` (Sharpe uplift=-0.223, ann return uplift=-26.70%).
