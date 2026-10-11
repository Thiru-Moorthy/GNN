from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_markdown_report(path: str | Path, results: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P&ID GNN Connectivity Benchmark",
        "",
        f"Dataset type: {'synthetic demo' if results.get('is_synthetic') else 'real or mixed'}",
        "",
        "These metrics are computed only from explicit connected/not_connected labels. Ambiguous and unlabeled edges are excluded.",
        "",
        "## Summary",
        "",
    ]
    for method, metrics in results.get("methods", {}).items():
        lines.append(f"### {method}")
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(["## Counts", ""])
    for key, value in results.get("counts", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for item in results.get("limitations", []):
        lines.append(f"- {item}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
