#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.cluster_runtime import resolve_cluster_layout
from frisk.notebook_runtime import write_toml_overrides


_RUN_KEYS = {
    "log_csv",
    "plot_path",
    "save_model",
    "save_encoder",
    "save_critic",
    "resume_state_path",
    "resume_dir",
    "out_csv",
    "baseline_out_csv",
    "walk_forward_out_csv",
    "history_out_csv",
    "bar_plot_path",
    "diag_out",
}
_CACHE_KEYS = {
    "risk_cache_dir",
    "portfolio_cache_dir",
    "auto_tune_cache_path",
}
_DATA_KEYS = {
    "graphs",
    "prices",
    "constituents",
    "fundamentals",
    "sec_companyfacts",
    "sec_submissions",
    "macro",
    "static_edges",
    "out",
}


def _load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _strip_prefix(path: str, prefix: str) -> str:
    text = str(path).strip()
    if text.startswith(prefix):
        return text[len(prefix) :].lstrip("/")
    return text


def _rewrite_path(raw: str, *, section: str, key: str, layout) -> str:
    text = str(raw).strip().replace("<netid>", layout.netid).replace("{netid}", layout.netid)
    if not text:
        return text
    value = Path(text).expanduser()
    if value.is_absolute():
        layout.require_safe_path(value, label=f"{section}.{key}")
        return str(value)
    if key in _CACHE_KEYS:
        suffix = _strip_prefix(text, "runs/cache")
        suffix = suffix or Path(text).name
        return str(layout.cache_root / suffix)
    if key in _RUN_KEYS:
        suffix = _strip_prefix(text, "runs")
        suffix = suffix or Path(text).name
        return str(layout.runs_root / suffix)
    if key in _DATA_KEYS:
        if key == "out":
            suffix = _strip_prefix(text, "data")
        else:
            suffix = _strip_prefix(text, "data")
        suffix = suffix or Path(text).name
        return str(layout.data_root / suffix)
    if text.startswith("reports/"):
        return str(layout.runs_root / text)
    if text.startswith("runs/"):
        return str(layout.runs_root / _strip_prefix(text, "runs"))
    if text.startswith("data/"):
        return str(layout.data_root / _strip_prefix(text, "data"))
    return text


def _merge_overlay_config(base_cfg: dict, overlay_cfg: dict) -> dict:
    out = dict(base_cfg)
    for section_name, overlay_section in overlay_cfg.items():
        if isinstance(overlay_section, dict) and isinstance(out.get(section_name), dict):
            merged_section = dict(out[section_name])
            merged_section.update(overlay_section)
            out[section_name] = merged_section
        else:
            out[section_name] = overlay_section
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite repo config paths for Emory Turing scratch.")
    parser.add_argument("--base-config", required=True, help="Base TOML config to rewrite.")
    parser.add_argument("--runtime-config", required=True, help="Output TOML path.")
    parser.add_argument(
        "--cluster-config",
        default="",
        help="Optional TOML file providing [cluster] defaults to merge before rewriting.",
    )
    parser.add_argument("--netid", default="", help="Override the resolved <netid> token.")
    parser.add_argument(
        "--scratch-root",
        default="",
        help="Optional override for cluster.scratch_root (useful for local dry runs/tests).",
    )
    parser.add_argument(
        "--graph-source",
        default="",
        help="Optional explicit graph artifact path on scratch to use for train.graphs.",
    )
    parser.add_argument(
        "--prices-source",
        default="",
        help="Optional explicit prices CSV path on scratch to use for build_graphs.prices.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    runtime_path = Path(args.runtime_config).expanduser().resolve()
    base_cfg = _load_config(base_path)
    cfg = base_cfg
    if args.cluster_config:
        cfg = _merge_overlay_config(cfg, _load_config(Path(args.cluster_config).expanduser().resolve()))

    cluster_cfg = dict(cfg.get("cluster", {}))
    if args.netid.strip():
        cluster_cfg["netid"] = args.netid.strip()
    if args.scratch_root.strip():
        cluster_cfg["scratch_root"] = args.scratch_root.strip()
        cluster_cfg["repo_root"] = str(Path(args.scratch_root).expanduser() / "repo")
        cluster_cfg["data_root"] = str(Path(args.scratch_root).expanduser() / "data")
        cluster_cfg["runs_root"] = str(Path(args.scratch_root).expanduser() / "runs")
        cluster_cfg["cache_root"] = str(Path(args.scratch_root).expanduser() / ".cache")
        cluster_cfg["logs_root"] = str(Path(args.scratch_root).expanduser() / "logs")
    cluster_cfg["enabled"] = True
    layout = resolve_cluster_layout(cluster_cfg)
    cluster_cfg.update(
        {
            "netid": layout.netid,
            "scratch_root": str(layout.scratch_root),
            "repo_root": str(layout.repo_root),
            "data_root": str(layout.data_root),
            "runs_root": str(layout.runs_root),
            "cache_root": str(layout.cache_root),
            "logs_root": str(layout.logs_root),
        }
    )

    overrides: dict[str, dict[str, object]] = {"cluster": cluster_cfg}
    for root in (
        layout.scratch_root,
        layout.repo_root,
        layout.data_root,
        layout.runs_root,
        layout.cache_root,
        layout.logs_root,
    ):
        root.mkdir(parents=True, exist_ok=True)

    for section_name, section in cfg.items():
        if not isinstance(section, dict) or section_name == "cluster":
            continue
        base_section = base_cfg.get(section_name, {})
        if not isinstance(base_section, dict):
            base_section = {}
        section_overrides: dict[str, object] = {}
        for key, value in section.items():
            if not isinstance(value, str):
                if base_section.get(key) != value:
                    section_overrides[key] = value
                continue
            rewritten = _rewrite_path(value, section=section_name, key=key, layout=layout)
            if rewritten != base_section.get(key):
                section_overrides[key] = rewritten
        if section_name == "train" and args.graph_source.strip():
            section_overrides["graphs"] = str(Path(args.graph_source).expanduser().resolve())
        if section_name == "build_graphs" and args.prices_source.strip():
            section_overrides["prices"] = str(Path(args.prices_source).expanduser().resolve())
        if section_overrides:
            overrides[section_name] = section_overrides

    write_toml_overrides(base_path, runtime_path, overrides)
    print(f"wrote runtime config: {runtime_path}")
    print(f"scratch_root: {layout.scratch_root}")
    print(f"repo_root: {layout.repo_root}")
    print(f"data_root: {layout.data_root}")
    print(f"runs_root: {layout.runs_root}")
    print(f"cache_root: {layout.cache_root}")
    print(f"logs_root: {layout.logs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
