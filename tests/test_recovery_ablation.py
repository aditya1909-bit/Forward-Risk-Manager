from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_experiments_cover_requested_families():
    mod = _load_script("recovery_ablation.py")
    exps = mod._default_experiments()
    families = {e["family"] for e in exps}
    assert "baseline" in families
    assert "negative_sampling" in families
    assert "risk_head" in families
    assert "graph_params" in families
    assert "goodness_margin" in families
    assert "hallucination" in families
    assert "split_strategy" in families


def test_deep_update_merges_nested_dicts():
    mod = _load_script("recovery_ablation.py")
    base = {"train": {"neg_mode": "mix", "nested": {"a": 1, "b": 2}}}
    out = mod._deep_update(base, {"train": {"nested": {"b": 99, "c": 3}}})
    assert out["train"]["neg_mode"] == "mix"
    assert out["train"]["nested"]["a"] == 1
    assert out["train"]["nested"]["b"] == 99
    assert out["train"]["nested"]["c"] == 3


def test_build_signature_ignores_runtime_build_fields():
    mod = _load_script("recovery_ablation.py")
    a = {
        "prices": "data/processed/prices.csv",
        "top_k": 10,
        "out": "graphs_a.pt",
        "workers": 8,
        "progress": True,
    }
    b = {
        "prices": "data/processed/prices.csv",
        "top_k": 10,
        "out": "graphs_b.pt",
        "workers": 1,
        "progress": False,
    }
    assert mod._build_signature(a) == mod._build_signature(b)


def test_add_baseline_deltas_populates_expected_columns():
    mod = _load_script("recovery_ablation.py")
    rows = [
        {"ablation_id": "baseline_updated", "mode": "ff_e2e", "eval_sep": "0.1"},
        {"ablation_id": "trial_a", "mode": "ff_e2e", "eval_sep": "0.25"},
    ]
    mod._add_baseline_deltas(rows, baseline_id="baseline_updated")
    assert rows[0]["delta_eval_sep_vs_baseline"] == "0"
    assert rows[1]["delta_eval_sep_vs_baseline"] == "0.15"
