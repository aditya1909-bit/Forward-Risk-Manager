from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


NOTEBOOK_ENV_VAR = "FRM_REPO_DIR"


@dataclass(frozen=True)
class CommandRunResult:
    command: str
    returncode: int
    elapsed_s: float
    log_path: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class NotebookStage:
    step: str
    label: str
    required_outputs: tuple[str, ...] = ()
    eta_s: float = 0.0
    optional: bool = False


def in_colab() -> bool:
    return "google.colab" in sys.modules


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "unknown"
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def resolve_repo_root(
    *,
    env_var: str = NOTEBOOK_ENV_VAR,
    marker_relpaths: Sequence[str] = ("configs/default.toml",),
    extra_candidates: Iterable[str | Path] = (),
    start: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []

    env_repo = os.environ.get(env_var, "").strip()
    if env_repo:
        candidates.append(Path(env_repo))

    if start is not None:
        start_path = Path(start).resolve()
        candidates.extend([start_path, *start_path.parents])
    else:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])

    if in_colab():
        drive_root = Path("/content/drive/MyDrive")
        candidates.extend(
            [
                Path("/content/Forward-Risk-Manager"),
                Path("/content/drive/MyDrive/Forward-Risk-Manager"),
                Path("/content/drive/MyDrive/forward-risk-manager"),
            ]
        )
        if drive_root.exists():
            candidates.extend(
                sorted(
                    path
                    for path in drive_root.glob("*Forward*Risk*Manager*")
                    if path.is_dir()
                )
            )

    user_name = os.environ.get("EMORY_NETID", "").strip() or os.environ.get("USER", "").strip()
    if user_name:
        candidates.extend(
            [
                Path(f"/local/scratch/{user_name}/forward-risk-manager/repo"),
                Path(f"/local/scratch/{user_name}/forward-risk-manager"),
            ]
        )

    candidates.extend(Path(candidate) for candidate in extra_candidates)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / marker).exists() for marker in marker_relpaths):
            return candidate

    marker_desc = ", ".join(marker_relpaths)
    raise FileNotFoundError(
        f"Could not find repo root containing: {marker_desc}. "
        f"Set {env_var} or update candidate paths."
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_subdirs(root: str | Path, names: Iterable[str]) -> dict[str, Path]:
    base = ensure_dir(root)
    return {name: ensure_dir(base / name) for name in names}


def command_log_name(command: str, *, max_len: int = 120) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", command).strip("_")
    return (safe_name or "command")[:max_len]


def run_command(
    command: str,
    *,
    allow_fail: bool = False,
    tail_lines: int = 200,
    env_overrides: dict[str, str] | None = None,
    log_dir: str | Path | None = None,
    cwd: str | Path | None = None,
    print_command: bool = True,
) -> CommandRunResult:
    if print_command:
        print("\n" + "=" * 110)
        print(command)
        print("=" * 110)

    started = time.time()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    resolved_log_dir = ensure_dir(log_dir) if log_dir is not None else None
    log_path = None
    if resolved_log_dir is not None:
        log_path = resolved_log_dir / f"{int(started)}_{command_log_name(command)}.log"

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )
    assert proc.stdout is not None

    tail = deque(maxlen=max(40, int(tail_lines)))
    with (log_path.open("w", encoding="utf-8") if log_path is not None else open(os.devnull, "w")) as handle:
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
            tail.append(line.rstrip("\n"))
    returncode = proc.wait()
    elapsed_s = time.time() - started

    print(
        f"\ncompleted in {elapsed_s:.2f}s"
        + (f" | log: {log_path}" if log_path is not None else "")
    )

    if returncode != 0:
        if tail:
            print("---- command output tail ----")
            for line in tail:
                print(line)
            print("---- end tail ----")
        message = f"command failed ({returncode}): {command}"
        if not allow_fail:
            raise RuntimeError(message)
        print("WARNING:", message)

    return CommandRunResult(
        command=command,
        returncode=returncode,
        elapsed_s=elapsed_s,
        log_path=log_path,
    )


def _toml_value_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value_literal(item) for item in value) + "]"
    return json.dumps(str(value))


def write_toml_overrides(
    base_config_path: str | Path,
    runtime_config_path: str | Path,
    section_overrides: dict[str, dict[str, Any]],
) -> Path:
    src = Path(base_config_path)
    dst = Path(runtime_config_path)
    lines = src.read_text(encoding="utf-8").splitlines()

    section_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    key_res = {
        section: {key: re.compile(rf"^\s*{re.escape(key)}\s*=") for key in values}
        for section, values in section_overrides.items()
    }
    replaced = {
        (section, key): False
        for section, values in section_overrides.items()
        for key in values
    }
    seen_sections: set[str] = set()
    current_section: str | None = None
    output_lines: list[str] = []

    def flush_missing(section_name: str | None) -> None:
        if section_name not in section_overrides:
            return
        for key, value in section_overrides[section_name].items():
            if not replaced[(section_name, key)]:
                output_lines.append(f"{key} = {_toml_value_literal(value)}")
                replaced[(section_name, key)] = True

    for line in lines:
        match = section_re.match(line)
        if match:
            flush_missing(current_section)
            current_section = match.group(1).strip()
            seen_sections.add(current_section)
            output_lines.append(line)
            continue

        updated = False
        if current_section in section_overrides:
            for key, pattern in key_res[current_section].items():
                if pattern.match(line):
                    output_lines.append(
                        f"{key} = {_toml_value_literal(section_overrides[current_section][key])}"
                    )
                    replaced[(current_section, key)] = True
                    updated = True
                    break

        if not updated:
            output_lines.append(line)

    flush_missing(current_section)

    for section_name, overrides in section_overrides.items():
        if section_name in seen_sections:
            continue
        if output_lines and output_lines[-1].strip():
            output_lines.append("")
        output_lines.append(f"[{section_name}]")
        for key, value in overrides.items():
            output_lines.append(f"{key} = {_toml_value_literal(value)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return dst


def merge_csv_files(paths: Iterable[str | Path], out_path: str | Path) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)
            rows.extend(reader)

    if not rows:
        raise RuntimeError(f"No rows found to merge into {out_path}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime": int(stat.st_mtime),
                    "sha256": file_sha256(path),
                }
            )
        else:
            rows.append({"path": str(path), "missing": True})
    return rows


def build_source_fingerprint(
    build_config: str | Path,
    *,
    tracked_keys: Sequence[str],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore

    build_config_path = Path(build_config)
    with build_config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    section = cfg.get("build_graphs", {})
    tracked_paths: list[Path] = [build_config_path]
    for key in tracked_keys:
        value = str(section.get(key, "")).strip()
        if value:
            tracked_paths.append(Path(value))

    fingerprint = {
        "build_config": str(build_config_path),
        "files": fingerprint_files(tracked_paths),
    }
    if extra_fields:
        fingerprint.update(extra_fields)
    return fingerprint


def load_json(path: str | Path, *, default: Any = None) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        return default
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolved


def record_run_step(
    manifest_path: str | Path,
    *,
    step: str,
    command: str | None = None,
    status: str,
    log_path: str | Path | None = None,
    required_outputs: Iterable[str | Path] = (),
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest = load_json(manifest_path, default={}) or {}
    manifest.setdefault("updated_at_utc", utc_now_timestamp())
    manifest.setdefault("steps", {})
    manifest["updated_at_utc"] = utc_now_timestamp()
    manifest["steps"][step] = {
        "status": status,
        "command": command,
        "log_path": str(log_path) if log_path else "",
        "required_outputs": [str(path) for path in required_outputs],
        "metadata": metadata or {},
        "updated_at_utc": utc_now_timestamp(),
    }
    return write_json(manifest_path, manifest)


def step_is_complete(
    manifest_path: str | Path,
    step: str,
    *,
    required_outputs: Iterable[str | Path] = (),
) -> bool:
    manifest = load_json(manifest_path, default={}) or {}
    step_row = (manifest.get("steps") or {}).get(step) or {}
    if step_row.get("status") != "completed":
        return False

    output_paths = [Path(path) for path in step_row.get("required_outputs", [])]
    output_paths.extend(Path(path) for path in required_outputs)
    if not output_paths:
        return True
    return all(path.exists() for path in output_paths)


def stage_status_rows(
    manifest_path: str | Path,
    stages: Sequence[NotebookStage],
    *,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path, default={}) or {}
    step_rows = (manifest.get("steps") or {}) if isinstance(manifest, dict) else {}
    root_path = Path(root).resolve() if root is not None else None
    rows: list[dict[str, Any]] = []
    for stage in stages:
        step_row = step_rows.get(stage.step) or {}
        status = str(step_row.get("status", "")).strip() or "pending"
        metadata = step_row.get("metadata") or {}
        elapsed_s = metadata.get("elapsed_s")
        effective_eta = metadata.get("eta_s", stage.eta_s)
        outputs = [Path(path) for path in stage.required_outputs]
        if root_path is not None:
            outputs = [path if path.is_absolute() else root_path / path for path in outputs]
        outputs_ready = bool(outputs) and all(path.exists() for path in outputs)
        if status == "completed" and outputs and not outputs_ready:
            status = "stale"
        rows.append(
            {
                "step": stage.step,
                "label": stage.label,
                "status": status,
                "elapsed_s": elapsed_s,
                "eta_s": effective_eta,
                "optional": stage.optional,
                "outputs_ready": outputs_ready if outputs else None,
            }
        )
    return rows


def remaining_eta_seconds(
    manifest_path: str | Path,
    stages: Sequence[NotebookStage],
    *,
    root: str | Path | None = None,
) -> float:
    rows = stage_status_rows(manifest_path, stages, root=root)
    total = 0.0
    for row in rows:
        if row["status"] == "completed":
            continue
        eta_s = row.get("eta_s")
        if eta_s is None:
            continue
        try:
            eta_val = float(eta_s)
        except Exception:
            continue
        if eta_val > 0:
            total += eta_val
    return total


def run_tracked_command(
    *,
    step: str,
    label: str,
    command: str,
    manifest_path: str | Path,
    required_outputs: Iterable[str | Path] = (),
    eta_s: float = 0.0,
    optional: bool = False,
    allow_fail: bool = False,
    skip_if_complete: bool = True,
    tail_lines: int = 200,
    env_overrides: dict[str, str] | None = None,
    log_dir: str | Path | None = None,
    cwd: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> CommandRunResult | None:
    outputs = [Path(path) for path in required_outputs]
    if skip_if_complete and step_is_complete(manifest_path, step, required_outputs=outputs):
        print(f"Skipping {label}: outputs already exist.")
        record_run_step(
            manifest_path,
            step=step,
            command=command,
            status="completed",
            required_outputs=outputs,
            metadata={
                "skipped": True,
                "eta_s": float(eta_s),
                "optional": bool(optional),
                **(metadata or {}),
            },
        )
        return None

    record_run_step(
        manifest_path,
        step=step,
        command=command,
        status="started",
        required_outputs=outputs,
        metadata={
            "eta_s": float(eta_s),
            "optional": bool(optional),
            **(metadata or {}),
        },
    )
    started = time.time()
    try:
        result = run_command(
            command,
            allow_fail=allow_fail,
            tail_lines=tail_lines,
            env_overrides=env_overrides,
            log_dir=log_dir,
            cwd=cwd,
        )
    except Exception as exc:
        elapsed_s = time.time() - started
        record_run_step(
            manifest_path,
            step=step,
            command=command,
            status="failed",
            required_outputs=outputs,
            metadata={
                "elapsed_s": elapsed_s,
                "eta_s": float(eta_s),
                "optional": bool(optional),
                "error": str(exc),
                **(metadata or {}),
            },
        )
        raise

    status = "completed" if result.ok else "failed"
    record_run_step(
        manifest_path,
        step=step,
        command=command,
        status=status,
        log_path=result.log_path,
        required_outputs=outputs,
        metadata={
            "elapsed_s": result.elapsed_s,
            "eta_s": float(eta_s),
            "optional": bool(optional),
            "returncode": int(result.returncode),
            **(metadata or {}),
        },
    )
    return result
