#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _parse_gpu_cores(text: str) -> list[tuple[str, str]]:
    # Look for sections that include "Chipset Model" and "Total Number of Cores"
    results: list[tuple[str, str]] = []
    blocks = text.split("\n\n")
    for block in blocks:
        if "Chipset Model" not in block:
            continue
        m_model = re.search(r"Chipset Model:\s*(.+)", block)
        m_cores = re.search(r"Total Number of Cores:\s*(.+)", block)
        if m_model and m_cores:
            results.append((m_model.group(1).strip(), m_cores.group(1).strip()))
    return results


def _parse_cpu_cores(text: str) -> str | None:
    m = re.search(r"Total Number of Cores:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return None


def main() -> int:
    print("Querying system_profiler (this may take a few seconds)...")
    gpu_text = _run(["system_profiler", "SPDisplaysDataType"])
    gpu_entries = _parse_gpu_cores(gpu_text)
    if gpu_entries:
        print("GPU cores:")
        for model, cores in gpu_entries:
            print(f"  {model}: {cores}")
    else:
        print("GPU core count not found in SPDisplaysDataType.")

    cpu_text = _run(["system_profiler", "SPHardwareDataType"])
    cpu_cores = _parse_cpu_cores(cpu_text)
    if cpu_cores:
        print(f"CPU cores (for reference): {cpu_cores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
