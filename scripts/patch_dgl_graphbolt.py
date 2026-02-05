#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import site
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import site: {exc}")
        return 1

    candidates = site.getsitepackages() + [site.getusersitepackages()]
    target = None
    for base in candidates:
        if not base:
            continue
        p = Path(base) / "dgl" / "graphbolt" / "__init__.py"
        if p.exists():
            target = p
            break

    if not target:
        print("Could not find dgl/graphbolt/__init__.py in site-packages.")
        return 1

    text = target.read_text()
    if "try:\n    load_graphbolt()" in text and "Graphbolt library not available" in text:
        print(f"Graphbolt patch already applied: {target}")
        return 0

    marker = "\nload_graphbolt()\n"
    if marker not in text:
        print(f"Did not find Graphbolt load call in {target}")
        return 1

    patched = text.replace(
        marker,
        "\ntry:\n    load_graphbolt()\nexcept FileNotFoundError:\n    # Graphbolt library not available on this platform.\n    pass\n",
        1,
    )
    target.write_text(patched)
    print(f"Patched Graphbolt loader in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
