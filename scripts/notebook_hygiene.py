#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_notebook(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def notebook_stats(path: Path) -> dict[str, Any]:
    notebook = _load_notebook(path)
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    total_outputs = sum(len(cell.get("outputs", [])) for cell in code_cells)
    total_output_chars = 0
    for cell in code_cells:
        for output in cell.get("outputs", []):
            total_output_chars += len("".join(output.get("text", [])))
            data = output.get("data", {})
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        total_output_chars += len("".join(value))
                    elif isinstance(value, str):
                        total_output_chars += len(value)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "cells": len(cells),
        "code_cells": len(code_cells),
        "total_outputs": total_outputs,
        "total_output_chars": total_output_chars,
    }


def strip_outputs(path: Path) -> bool:
    notebook = _load_notebook(path)
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        _write_notebook(path, notebook)
    return changed


def syntax_check(path: Path) -> list[str]:
    notebook = _load_notebook(path)
    errors: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = source.lstrip()
        if not stripped or stripped.startswith("!") or stripped.startswith("%"):
            continue
        try:
            compile(source, f"{path}#cell{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"{path}:cell{index}: {exc.msg} (line {exc.lineno})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or strip notebook outputs.")
    parser.add_argument("paths", nargs="+", help="Notebook paths.")
    parser.add_argument("--strip", action="store_true", help="Remove outputs and execution counts.")
    parser.add_argument(
        "--fail-on-outputs",
        action="store_true",
        help="Return non-zero if any notebook still has outputs.",
    )
    parser.add_argument(
        "--syntax-check",
        action="store_true",
        help="Compile Python code cells and return non-zero on syntax errors.",
    )
    args = parser.parse_args()

    had_error = False
    for raw_path in args.paths:
        path = Path(raw_path)
        if args.strip:
            changed = strip_outputs(path)
            print(f"strip {path}: changed={changed}")
        stats = notebook_stats(path)
        print(json.dumps(stats, sort_keys=True))
        if args.fail_on_outputs and stats["total_outputs"]:
            had_error = True
        if args.syntax_check:
            errors = syntax_check(path)
            for error in errors:
                print(error)
            had_error = had_error or bool(errors)

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
