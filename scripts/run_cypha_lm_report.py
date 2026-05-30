#!/usr/bin/env python3
"""Run all CyphaLM experiments in fast CI mode and write cypha_lm/REPORT.md."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "cypha_lm" / "REPORT.md"

EXPERIMENTS = [
    ("01_embedding_benchmark", "experiments/01_embedding_benchmark.py"),
    ("02_ssm_sequence_capacity", "experiments/02_ssm_sequence_capacity.py"),
    ("03_expert_self_organisation", "experiments/03_expert_self_organisation.py"),
    ("04_alpha_spectrum_emergence", "experiments/04_alpha_spectrum_emergence.py"),
    ("05_lm_training_toy_vocab", "experiments/05_lm_training_toy_vocab.py"),
    ("06_lm_training_code_corpus", "experiments/06_lm_training_code_corpus.py"),
    ("07_uncertainty_calibration", "experiments/07_uncertainty_calibration.py"),
    ("08_online_adaptation", "experiments/08_online_adaptation.py"),
    ("09_catastrophic_forgetting", "experiments/09_catastrophic_forgetting.py"),
    ("10_parameter_efficiency", "experiments/10_parameter_efficiency_vs_transformer.py"),
]

BENCHMARKS = [
    ("perplexity", "benchmarks/perplexity_eval.py"),
    ("memory", "benchmarks/memory_profile.py"),
    ("latency", "benchmarks/latency_profile.py"),
]


def _load_run(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        raise AttributeError(f"{script_path} has no main()")
    return mod.main()


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return ", ".join(_fmt(v) for v in val[:5]) + ("..." if len(val) > 5 else "")
    return str(val)


def _build_report(results: dict, bench: dict, elapsed: float, errors: list[str]) -> str:
    lines = [
        "# CyphaLM Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Mode: {'fast (CI)' if os.environ.get('CYPHA_LM_FAST') == '1' else 'full'}",
        f"Total runtime: {elapsed:.1f}s",
        "",
    ]

    if errors:
        lines.extend(["## Errors", ""])
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    lines.extend(["## Experiment Metrics", ""])
    for name, data in results.items():
        lines.append(f"### {name}")
        lines.append("")
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "generated_sample":
                    lines.append(f"- **{k}**: `{v}`")
                else:
                    lines.append(f"- **{k}**: {_fmt(v)}")
        else:
            lines.append(f"- result: {_fmt(data)}")
        lines.append("")

    lines.extend(["## Benchmarks", ""])
    for name, data in bench.items():
        lines.append(f"### {name}")
        lines.append("")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    lines.append(f"- **{k}**:")
                    for sk, sv in v.items():
                        lines.append(f"  - {sk}: {_fmt(sv)}")
                else:
                    lines.append(f"- **{k}**: {_fmt(v)}")
        lines.append("")

    exp01 = results.get("01_embedding_benchmark", {})
    exp03 = results.get("03_expert_self_organisation", {})
    exp05 = results.get("05_lm_training_toy_vocab", {})
    exp09 = results.get("09_catastrophic_forgetting", {})

    lines.extend(
        [
            "## Summary",
            "",
            f"- Izaac kNN accuracy: {_fmt((exp01.get('knn_accuracy') or {}).get('Izaac (zero-param)', 'n/a'))}",
            f"- Expert purity: {_fmt(exp03.get('expert_purity', 'n/a'))}",
            f"- Toy LM perplexity: {_fmt(exp05.get('perplexity', 'n/a'))}",
            f"- Forgetting retention: {_fmt(exp09.get('retention_ratio', 'n/a'))}",
            "",
            "Figures written to `paper/figures/`.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    os.environ["CYPHA_LM_FAST"] = "1"
    os.chdir(ROOT)

    results: dict = {}
    bench: dict = {}
    errors: list[str] = []
    t0 = time.perf_counter()

    for name, rel in EXPERIMENTS:
        path = ROOT / rel
        print(f"[run] {rel} ...", flush=True)
        try:
            results[name] = _load_run(path)
            print(f"  ok", flush=True)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"  FAIL: {exc}", flush=True)
            traceback.print_exc()

    for name, rel in BENCHMARKS:
        path = ROOT / rel
        print(f"[bench] {rel} ...", flush=True)
        try:
            bench[name] = _load_run(path)
            print(f"  ok", flush=True)
        except Exception as exc:
            errors.append(f"benchmark {name}: {exc}")
            print(f"  FAIL: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    report = _build_report(results, bench, elapsed, errors)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH} ({elapsed:.1f}s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
