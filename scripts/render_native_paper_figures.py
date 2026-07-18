#!/usr/bin/env python3
"""Render paper/figures/native_fig_*.json → PNG bar charts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def render_series_bar(path: Path, out: Path, value_key: str, ylabel: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    series = data.get("series") or []
    if not series:
        # alpha figure uses metrics dict
        metrics = data.get("metrics") or {}
        labels = list(metrics.keys())
        values = [float(metrics[k]) if isinstance(metrics[k], (int, float)) else 0.0 for k in labels]
        title = data.get("title", path.stem)
    else:
        labels = [str(s.get("name") or s.get("exp") or i) for i, s in enumerate(series)]
        values = [float(s[value_key]) for s in series]
        title = data.get("title", path.stem)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = plt.cm.Blues([(0.45 + 0.5 * i / max(1, len(values) - 1)) for i in range(len(values))])
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    root = Path("paper/figures")
    render_series_bar(root / "native_fig_d17_bpc.json", root / "native_fig_d17_bpc.png", "bpc", "BPC (lower better)")
    render_series_bar(
        root / "native_fig_d16_forgetting.json",
        root / "native_fig_d16_forgetting.png",
        "forgetting_score",
        "Forgetting score (lower better)",
    )
    # Alpha: single metrics object → fake series
    alpha = json.loads((root / "native_fig_d17b_alpha.json").read_text(encoding="utf-8"))
    fake = {
        "title": alpha.get("title"),
        "series": [
            {"name": "mean_alpha", "value": float(alpha["metrics"]["mean_alpha"])},
            {"name": "n_experts", "value": float(alpha["metrics"]["n_experts"])},
        ],
    }
    tmp = root / "_tmp_alpha_render.json"
    tmp.write_text(json.dumps(fake), encoding="utf-8")
    render_series_bar(tmp, root / "native_fig_d17b_alpha.png", "value", "Value")
    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
