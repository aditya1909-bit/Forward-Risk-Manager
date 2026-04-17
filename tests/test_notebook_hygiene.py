from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "colab_setup.ipynb",
    ROOT / "notebooks" / "end_to_end_repo_runbook.ipynb",
    ROOT / "notebooks" / "graph_factory_colab.ipynb",
    ROOT / "notebooks" / "preliminary_paper_colab.ipynb",
    ROOT / "notebooks" / "paper_final_benchmark_colab.ipynb",
    ROOT / "notebooks" / "recovery_ablation_runbook.ipynb",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_are_output_stripped():
    for path in NOTEBOOKS:
        notebook = _load(path)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("outputs", []) == [], f"{path} still has stored outputs"
            assert cell.get("execution_count") is None, f"{path} still has execution counts"


def test_core_notebooks_import_shared_runtime():
    required = {
        ROOT / "notebooks" / "colab_setup.ipynb",
        ROOT / "notebooks" / "end_to_end_repo_runbook.ipynb",
        ROOT / "notebooks" / "graph_factory_colab.ipynb",
        ROOT / "notebooks" / "preliminary_paper_colab.ipynb",
        ROOT / "notebooks" / "paper_final_benchmark_colab.ipynb",
    }
    for path in required:
        notebook = _load(path)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        assert "from frisk.notebook_runtime import" in source, f"{path} does not use shared runtime"


def test_notebooks_do_not_reference_unimported_shlex_quote():
    for path in NOTEBOOKS:
        notebook = _load(path)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        if "shlex.quote(" in source:
            assert "import shlex" in source, f"{path} uses shlex.quote without importing shlex"
