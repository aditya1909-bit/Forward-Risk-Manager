from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_SUFFIX = "forward-risk-manager"
_HOME_PREFIX = Path("/home")
_SCRATCH_PREFIX = Path("/local/scratch")


@dataclass(frozen=True)
class SlurmMetadata:
    job_id: str
    job_name: str
    node_name: str
    partition: str
    gpus: int


@dataclass(frozen=True)
class ClusterLayout:
    enabled: bool
    netid: str
    scratch_root: Path
    repo_root: Path
    data_root: Path
    runs_root: Path
    cache_root: Path
    logs_root: Path
    stage_graphs_to_local_nvme: bool
    sync_back_enabled: bool

    @property
    def home_root(self) -> Path:
        return _HOME_PREFIX / self.netid

    def contains_home_path(self, path: str | Path) -> bool:
        raw = Path(path).expanduser()
        text = str(raw)
        home_text = str(self.home_root)
        if text == home_text or text.startswith(home_text + os.sep):
            return True
        resolved = raw.resolve(strict=False)
        return self.home_root == resolved or self.home_root in resolved.parents

    def require_safe_path(self, path: str | Path, *, label: str = "path") -> Path:
        raw = Path(path).expanduser()
        resolved = raw.resolve(strict=False)
        if self.enabled and self.contains_home_path(raw):
            raise ValueError(
                f"Cluster mode forbids writing {label} under {self.home_root}. "
                f"Use scratch under {self.scratch_root} instead."
            )
        return resolved

    def slurm_metadata(self, env: Mapping[str, str] | None = None) -> SlurmMetadata:
        data = dict(os.environ if env is None else env)
        gpu_fields = [
            data.get("SLURM_GPUS_ON_NODE", ""),
            data.get("SLURM_GPUS", ""),
            data.get("CUDA_VISIBLE_DEVICES", ""),
        ]
        gpu_count = 0
        for raw in gpu_fields:
            text = str(raw).strip()
            if not text:
                continue
            if text.isdigit():
                gpu_count = int(text)
                break
            gpu_count = len([item for item in text.split(",") if item.strip()])
            if gpu_count:
                break
        return SlurmMetadata(
            job_id=str(data.get("SLURM_JOB_ID", "")),
            job_name=str(data.get("SLURM_JOB_NAME", "")),
            node_name=str(data.get("SLURMD_NODENAME", data.get("HOSTNAME", ""))),
            partition=str(data.get("SLURM_JOB_PARTITION", "")),
            gpus=int(gpu_count),
        )


def _resolve_netid(
    config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    cfg = dict(config or {})
    data = dict(os.environ if env is None else env)
    for key in ("netid",):
        value = str(cfg.get(key, "")).strip()
        if value:
            return value
    for key in ("EMORY_NETID", "NETID", "USER", "USERNAME"):
        value = str(data.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Unable to resolve cluster netid from config or environment.")


def _replace_netid_tokens(value: str, netid: str) -> str:
    return value.replace("<netid>", netid).replace("{netid}", netid)


def _resolve_path(
    value: str | None,
    *,
    netid: str,
    fallback: Path,
) -> Path:
    text = _replace_netid_tokens(str(value or "").strip(), netid)
    if not text:
        return fallback
    return Path(text).expanduser()


def resolve_cluster_layout(
    cluster_cfg: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ClusterLayout:
    cfg = dict(cluster_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    netid = _resolve_netid(cfg, env)
    scratch_root = _resolve_path(
        str(cfg.get("scratch_root", "")),
        netid=netid,
        fallback=_SCRATCH_PREFIX / netid / _DEFAULT_SUFFIX,
    )
    repo_root = _resolve_path(
        str(cfg.get("repo_root", "")),
        netid=netid,
        fallback=scratch_root / "repo",
    )
    data_root = _resolve_path(
        str(cfg.get("data_root", "")),
        netid=netid,
        fallback=scratch_root / "data",
    )
    runs_root = _resolve_path(
        str(cfg.get("runs_root", "")),
        netid=netid,
        fallback=scratch_root / "runs",
    )
    cache_root = _resolve_path(
        str(cfg.get("cache_root", "")),
        netid=netid,
        fallback=scratch_root / ".cache",
    )
    logs_root = _resolve_path(
        str(cfg.get("logs_root", "")),
        netid=netid,
        fallback=scratch_root / "logs",
    )
    return ClusterLayout(
        enabled=enabled,
        netid=netid,
        scratch_root=scratch_root,
        repo_root=repo_root,
        data_root=data_root,
        runs_root=runs_root,
        cache_root=cache_root,
        logs_root=logs_root,
        stage_graphs_to_local_nvme=bool(cfg.get("stage_graphs_to_local_nvme", False)),
        sync_back_enabled=bool(cfg.get("sync_back_enabled", False)),
    )


def ensure_safe_cluster_path(
    path: str | Path | None,
    *,
    cluster_cfg: Mapping[str, Any] | None = None,
    label: str = "path",
    allow_empty: bool = True,
) -> str:
    if path is None:
        return ""
    text = str(path).strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError(f"{label} is empty")
    layout = resolve_cluster_layout(cluster_cfg)
    resolved = layout.require_safe_path(text, label=label)
    return str(resolved)


def cluster_path_roots(cluster_cfg: Mapping[str, Any] | None = None) -> dict[str, str]:
    layout = resolve_cluster_layout(cluster_cfg)
    return {
        "scratch_root": str(layout.scratch_root),
        "repo_root": str(layout.repo_root),
        "data_root": str(layout.data_root),
        "runs_root": str(layout.runs_root),
        "cache_root": str(layout.cache_root),
        "logs_root": str(layout.logs_root),
    }
