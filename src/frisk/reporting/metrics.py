from __future__ import annotations

import math


def to_finite_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def primary_metric_from_row(row):
    robust = to_finite_float(row.get("primary_eval_metric_robust"))
    if robust is not None:
        robust_name = str(row.get("primary_eval_metric_robust_name", "")).strip()
        return robust_name or "primary_eval_metric_robust", robust

    task_family = str(row.get("task_family", "")).strip().lower()
    primary_family = str(row.get("primary_metric_family", "")).strip().lower()
    if task_family == "financial_value" or primary_family == "economics":
        for key in (
            "econ_oos_sharpe_uplift_min",
            "econ_sharpe_uplift",
            "eval_return_corr",
            "eval_return_rank_corr",
        ):
            val = to_finite_float(row.get(key))
            if val is not None:
                return key, val

    objective = str(row.get("eval_objective", "")).strip().lower()
    if objective == "self_contrastive":
        sc_gap = to_finite_float(row.get("eval_sc_gap"))
        if sc_gap is not None:
            return "eval_sc_gap", sc_gap

    if objective in {"bce", "backprop"}:
        auroc = to_finite_float(row.get("eval_auroc"))
        if auroc is not None:
            return "eval_auroc", auroc
        auprc = to_finite_float(row.get("eval_auprc"))
        if auprc is not None:
            return "eval_auprc", auprc

    for key in ("eval_sep", "eval_auroc", "eval_auprc", "eval_sc_gap", "eval_acc"):
        val = to_finite_float(row.get(key))
        if val is not None:
            return key, val
    return "none", float("-inf")
