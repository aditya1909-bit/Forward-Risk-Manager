from __future__ import annotations

import math


MODE_ALIASES = {
    "backprop": "backprop_contrastive",
}


def canonical_mode_name(mode: str) -> str:
    raw = str(mode).strip().lower()
    return MODE_ALIASES.get(raw, raw)


def mode_semantics(mode: str) -> dict[str, str]:
    canonical = canonical_mode_name(mode)
    if canonical in {"ff_e2e_core", "backprop_contrastive_core", "ff_fast"}:
        return {"task_family": "algorithmic_parity", "signal_family": "contrastive"}
    if canonical in {"ff_financial", "ff_accurate", "ff_bootstrap_rank"}:
        return {"task_family": "financial_value", "signal_family": "goodness_rank"}
    if canonical == "backprop_supervised_return":
        return {"task_family": "financial_value", "signal_family": "return_forecast"}
    if canonical in {"backprop_contrastive", "ff_e2e", "ff_layerwise"}:
        return {"task_family": "legacy_benchmark", "signal_family": "contrastive"}
    return {"task_family": "custom", "signal_family": "custom"}


def _financial_econ_strategy(out: dict) -> str:
    requested = str(out.get("econ_strategy_kind", "")).strip()
    if bool(out.get("econ_strategy_kind_explicit", False)) and requested:
        return requested
    return "both"


def apply_mode_profile(mode: str, cfg: dict) -> tuple[str, dict]:
    canonical = canonical_mode_name(mode)
    out = dict(cfg)
    out.update(mode_semantics(canonical))
    out["mode_canonical"] = canonical

    if canonical in {"ff_e2e_core", "backprop_contrastive_core", "ff_fast"}:
        out.update(
            {
                "risk_head_enabled": False,
                "portfolio_head_enabled": False,
                "ff_rank_aux_weight": 0.0,
                "ff_rank_corr_weight": 0.0,
                "distance_forward_weight": 0.0,
                "self_contrastive_ff_weight": 0.0,
                "sequence_critic_enabled": False,
                "residual_edge_weight_enabled": False,
                "neg_mode": "shuffle+noise",
                "eval_neg_mode": "shuffle+noise",
                "eval_neg_modes": [],
                "noise_std": float(out.get("noise_std", 0.05)),
                "neg_warmup_epochs": 0,
                "neg_mix_start": 0.0,
                "neg_mix_end": 0.0,
                "neg_mix_ramp_epochs": 1,
                "ff_mode": "classic",
                "ff_loss_type": "softplus_margin",
                "goodness_norm": "none",
                "goodness_reducer": "logsumexp",
            }
        )
    if canonical == "ff_fast":
        out.update({"amp": True, "torch_compile": True, "ff_concat_posneg": True})
    elif canonical == "ff_accurate":
        out.update(
            {
                "task_family": "financial_value",
                "signal_family": "goodness_rank",
                "ff_mode": "classic",
                "ff_loss_type": "symba",
                "goodness_norm": "layernorm",
                "goodness_reducer": "mean",
                "neg_mode": "shuffle+noise",
                "eval_neg_mode": "factor_hard",
                "eval_neg_modes": [],
                "ff_neg_mix": ["shuffle", "shuffle+noise", "factor_hard"],
                "ff_curriculum_epochs": [0.45, 0.35, 0.20],
                "ff_rank_aux_weight": max(float(out.get("ff_rank_aux_weight", 0.0)), 0.03),
                "ff_rank_corr_weight": max(float(out.get("ff_rank_corr_weight", 0.0)), 0.02),
                "energy_penalty_weight": max(float(out.get("energy_penalty_weight", 0.0)), 1e-4),
                "embedding_var_weight": max(float(out.get("embedding_var_weight", 0.0)), 0.02),
                "embedding_cov_weight": max(float(out.get("embedding_cov_weight", 0.0)), 0.01),
                "econ_strategy_kind": _financial_econ_strategy(out),
                "econ_ls_top_frac": float(out.get("econ_ls_top_frac", 0.2)),
                "econ_ls_bottom_frac": float(out.get("econ_ls_bottom_frac", 0.2)),
            }
        )
    elif canonical == "ff_financial":
        out.update(
            {
                "task_family": "financial_value",
                "signal_family": "goodness_rank",
                "ff_mode": "classic",
                "ff_loss_type": "symba",
                "goodness_norm": "layernorm",
                "goodness_reducer": "mean",
                "neg_mode": "shuffle+noise",
                "eval_neg_mode": "factor_hard",
                "eval_neg_modes": [],
                "ff_neg_mix": ["shuffle", "shuffle+noise", "factor_hard", "sector_swap"],
                "ff_curriculum_epochs": [0.40, 0.35, 0.25],
                "ff_rank_aux_weight": max(float(out.get("ff_rank_aux_weight", 0.0)), 0.025),
                "ff_rank_corr_weight": max(float(out.get("ff_rank_corr_weight", 0.0)), 0.02),
                "energy_penalty_weight": max(float(out.get("energy_penalty_weight", 0.0)), 1e-4),
                "embedding_var_weight": max(float(out.get("embedding_var_weight", 0.0)), 0.02),
                "embedding_cov_weight": max(float(out.get("embedding_cov_weight", 0.0)), 0.008),
                "portfolio_loss_type": str(out.get("portfolio_loss_type", "delta_cara")),
                "portfolio_loss_weight": max(float(out.get("portfolio_loss_weight", 0.0)), 0.003),
                "portfolio_baseline_exposure": float(out.get("portfolio_baseline_exposure", 1.0)),
                "portfolio_delta_scale": float(out.get("portfolio_delta_scale", 0.5)),
                "portfolio_cara_risk_aversion": float(out.get("portfolio_cara_risk_aversion", 4.0)),
                "econ_strategy_kind": _financial_econ_strategy(out),
                "econ_ls_top_frac": float(out.get("econ_ls_top_frac", 0.2)),
                "econ_ls_bottom_frac": float(out.get("econ_ls_bottom_frac", 0.2)),
            }
        )
    elif canonical == "ff_bootstrap_rank":
        out.update(
            {
                "task_family": "financial_value",
                "signal_family": "goodness_rank",
                "ff_mode": "classic",
                "ff_loss_type": "symba",
                "goodness_norm": "layernorm",
                "goodness_reducer": "mean",
                "neg_mode": "shuffle+noise",
                "eval_neg_mode": "factor_hard",
                "eval_neg_modes": [],
                "ff_neg_mix": ["shuffle", "shuffle+noise", "factor_hard", "cross_asset_mix"],
                "ff_curriculum_epochs": [0.40, 0.35, 0.25],
                "ff_rank_aux_weight": max(float(out.get("ff_rank_aux_weight", 0.0)), 0.03),
                "ff_rank_corr_weight": max(float(out.get("ff_rank_corr_weight", 0.0)), 0.03),
                "energy_penalty_weight": max(float(out.get("energy_penalty_weight", 0.0)), 1e-4),
                "embedding_var_weight": max(float(out.get("embedding_var_weight", 0.0)), 0.03),
                "embedding_cov_weight": max(float(out.get("embedding_cov_weight", 0.0)), 0.01),
                "bootstrap_graph_enabled": True,
                "bootstrap_graph_weight": max(float(out.get("bootstrap_graph_weight", 0.0)), 0.15),
                "bootstrap_graph_momentum": float(out.get("bootstrap_graph_momentum", 0.99)),
                "bootstrap_graph_view_mode": str(out.get("bootstrap_graph_view_mode", "cross_asset_mix")),
                "bootstrap_graph_view_noise_std": float(out.get("bootstrap_graph_view_noise_std", 0.03)),
                "bootstrap_graph_predictor_hidden_dim": int(
                    out.get("bootstrap_graph_predictor_hidden_dim", out.get("hidden_dim", 64))
                ),
                "portfolio_loss_type": str(out.get("portfolio_loss_type", "delta_cara")),
                "portfolio_loss_weight": max(float(out.get("portfolio_loss_weight", 0.0)), 0.004),
                "portfolio_baseline_exposure": float(out.get("portfolio_baseline_exposure", 1.0)),
                "portfolio_delta_scale": float(out.get("portfolio_delta_scale", 0.5)),
                "portfolio_cara_risk_aversion": float(out.get("portfolio_cara_risk_aversion", 4.0)),
                "econ_strategy_kind": _financial_econ_strategy(out),
                "econ_ls_top_frac": float(out.get("econ_ls_top_frac", 0.2)),
                "econ_ls_bottom_frac": float(out.get("econ_ls_bottom_frac", 0.2)),
                "econ_ls_uncertainty_scale": float(out.get("econ_ls_uncertainty_scale", 0.5)),
            }
        )
    elif canonical == "backprop_supervised_return":
        out.update({"task_family": "financial_value", "signal_family": "return_forecast"})
    elif canonical == "backprop_contrastive":
        out.update({"task_family": "legacy_benchmark", "signal_family": "contrastive"})
    return canonical, out


def goodness_kwargs(config: dict) -> dict[str, str]:
    return {
        "norm": str(config.get("goodness_norm", "none")).strip().lower(),
        "reducer": str(config.get("goodness_reducer", "logsumexp")).strip().lower(),
    }


def ff_loss_kwargs(config: dict) -> dict[str, str]:
    return {"loss_type": str(config.get("ff_loss_type", "softplus_margin")).strip().lower()}


def objective_track(objective: str) -> str:
    obj = str(objective).strip().lower()
    if obj == "self_contrastive":
        return "encoder"
    if obj in {"ff", "forward_forward", "forward-forward"} or obj.startswith("ff_"):
        return "critic"
    if obj in {"bce", "backprop", "supervised_return"}:
        return "classifier" if obj in {"bce", "backprop"} else "regressor"
    return "unknown"


def primary_metric_family(metric_name: str, row: dict) -> str:
    name = str(metric_name).strip().lower()
    if name.startswith("econ_"):
        return "economics"
    if name.startswith("eval_return_"):
        return "regression"
    if name in {"eval_auroc", "eval_auprc", "eval_acc", "eval_brier", "eval_ece"}:
        return "classifier"
    if name in {"eval_sep", "eval_sc_gap", "eval_sc_acc"}:
        return "contrastive"
    task_family = str(row.get("task_family", "")).strip().lower()
    if task_family == "financial_value":
        return "economics"
    return "unknown"


def _finite_or_none(value):
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def objective_primary_metric(metrics: dict) -> tuple[str, float]:
    task_family = str(metrics.get("task_family", "")).strip().lower()
    objective = str(metrics.get("eval_objective", "")).strip().lower()
    signal_family = str(metrics.get("signal_family", "")).strip().lower()
    if task_family == "financial_value":
        for key in (
            "econ_ls_oos_sharpe_uplift_min",
            "econ_ls_sharpe_uplift",
            "econ_ls_oos_ann_return_uplift_min",
            "econ_exposure_adjusted_sharpe_uplift",
            "econ_exposure_adjusted_ann_return_uplift",
            "econ_oos_sharpe_uplift_min",
            "econ_sharpe_uplift",
            "eval_return_corr",
            "eval_return_rank_corr",
            "eval_sep",
            "eval_auroc",
        ):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    if signal_family == "goodness_rank":
        for key in (
            "econ_ls_oos_sharpe_uplift_min",
            "econ_ls_sharpe_uplift",
            "econ_ls_oos_ann_return_uplift_min",
            "eval_return_rank_corr",
            "eval_sep",
        ):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    if task_family == "algorithmic_parity":
        for key in ("eval_auroc", "eval_auprc", "eval_sep", "eval_acc"):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    if objective == "self_contrastive":
        for key in ("eval_sc_gap", "eval_sep", "eval_sc_acc", "eval_acc"):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    if objective in {"bce", "backprop"}:
        for key in ("eval_auroc", "eval_auprc", "eval_sep", "eval_acc"):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    if objective == "supervised_return":
        for key in (
            "eval_return_corr",
            "eval_return_rank_corr",
            "econ_oos_sharpe_uplift_min",
            "econ_sharpe_uplift",
            "eval_return_mae",
        ):
            value = _finite_or_none(metrics.get(key))
            if value is not None:
                return key, value
    for key in ("eval_sep", "eval_auroc", "eval_auprc", "eval_acc"):
        value = _finite_or_none(metrics.get(key))
        if value is not None:
            return key, value
    return "none", float("nan")


def objective_primary_metric_robust(metrics: dict) -> tuple[str, float]:
    return objective_primary_metric(metrics)


def attach_primary_metrics(row: dict) -> None:
    metric_name, metric_value = objective_primary_metric(row)
    robust_name, robust_value = objective_primary_metric_robust(row)
    row["objective_track"] = objective_track(row.get("eval_objective", ""))
    row["primary_eval_metric_name"] = metric_name
    row["primary_eval_metric"] = metric_value
    row["primary_metric_family"] = primary_metric_family(metric_name, row)
    row["primary_eval_metric_robust_name"] = robust_name
    row["primary_eval_metric_robust"] = robust_value
