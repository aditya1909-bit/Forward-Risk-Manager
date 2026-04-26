# Turing Econ Large Sweep Summary

- Rows: 32
- Unpruned/final rows: 18
- Econ-evaluated rows: 18
- Best mode: ff_e2e
- Best rank metric: econ_sharpe_uplift_stability_adj
- rank_value: 0.0178726
- econ_sharpe_uplift: 0.0696142
- econ_sharpe_uplift_std: 0.103483
- econ_oos_sharpe_uplift_min: -0.339957
- econ_oos_sharpe_uplift_mean: 0.0348695
- econ_ann_return_uplift: -0.0938981
- econ_exposure_adjusted_sharpe_uplift: 0.0493739
- eval_sep_agg_min: 0.0383652
- eval_auroc_agg_min: 0.501639
- goodness_target: 2.2
- goodness_temp: 0.2
- neg_mix_end: 0.75
- hall_lr: 0.025
- hall_steps: 3
- hall_node_fraction: 0.35

## Top 5 By Rank
1. mode=ff_e2e rank_value=0.0178726 rank_metric=econ_sharpe_uplift_stability_adj gate=
2. mode=ff_e2e rank_value=-0.124975 rank_metric=econ_sharpe_uplift_stability_adj gate=
3. mode=ff_layerwise rank_value=-0.131973 rank_metric=econ_sharpe_uplift_stability_adj gate=
4. mode=ff_layerwise rank_value=-1e+06 rank_metric=eval_sep gate=successive_halving_pruned,econ_not_evaluated
5. mode=ff_layerwise rank_value=-1e+06 rank_metric=eval_sep gate=successive_halving_pruned,econ_not_evaluated
