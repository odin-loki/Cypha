"""
Master runner for cypha_bench domains d01–d17 and cross-domain analyses.

Usage (from repo root):
  python cypha_bench/run_all.py
  python cypha_bench/run_all.py --domain 3
  python cypha_bench/run_all.py --from-domain 10
  python cypha_bench/run_all.py --report-only
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

# Bootstrap before any cypha_bench imports.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.bootstrap import ensure_paths

ensure_paths()

DOMAINS = [
    ("d01", "cypha_bench.domains.d01_statistical_baselines", "run"),
    ("d02", "cypha_bench.domains.d02_regression", "run"),
    ("d03", "cypha_bench.domains.d03_classification", "run"),
    ("d04", "cypha_bench.domains.d04_generation_language", "run"),
    ("d05", "cypha_bench.domains.d05_chess", "run"),
    ("d06", "cypha_bench.domains.d06_go", "run"),
    ("d07", "cypha_bench.domains.d07_poker", "run"),
    ("d08", "cypha_bench.domains.d08_computer_vision", "run"),
    ("d09", "cypha_bench.domains.d09_documents", "run"),
    ("d10", "cypha_bench.domains.d10_time_series", "run"),
    ("d11", "cypha_bench.domains.d11_reinforcement_learning", "run"),
    ("d12", "cypha_bench.domains.d12_anomaly_detection", "run"),
    ("d13", "cypha_bench.domains.d13_compression", "run"),
    ("d14", "cypha_bench.domains.d14_symbolic_regression", "run"),
    ("d15", "cypha_bench.domains.d15_adversarial_robustness", "run"),
    ("d16", "cypha_bench.domains.d16_multitask", "run"),
    ("d17", "cypha_bench.domains.d17_cyphalm_integration", "run"),
]

CROSS_DOMAIN = [
    ("uncertainty_calibration", "cypha_bench.cross_domain.uncertainty_calibration", "run"),
    ("online_adaptation", "cypha_bench.cross_domain.online_adaptation", "run"),
    ("forgetting_resistance", "cypha_bench.cross_domain.forgetting_resistance", "run"),
    ("alpha_spectrum_global", "cypha_bench.cross_domain.alpha_spectrum_global", "run"),
]


def _run_module(module_path: str, fn_name: str) -> dict | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
        return fn()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cypha Bench domains d01–d17")
    parser.add_argument("--domain", type=int, default=None, help="Run single domain (1–17)")
    parser.add_argument("--from-domain", type=int, default=1, help="Start from domain number")
    parser.add_argument("--report-only", action="store_true", help="Skip experiments, build report")
    args = parser.parse_args()

    if args.report_only:
        from cypha_bench.report.generate_report import build_report

        out = build_report()
        print(f"Report: {out}")
        return

    domains = list(DOMAINS)
    if args.domain is not None:
        domains = [d for d in domains if int(d[0][1:]) == args.domain]
    elif args.from_domain > 1:
        domains = [d for d in domains if int(d[0][1:]) >= args.from_domain]

    failures: list[str] = []
    for tag, module_path, fn_name in domains:
        print(f"\n{'=' * 60}")
        print(f"Running {tag}: {module_path}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            _run_module(module_path, fn_name)
            print(f"  Completed in {time.time() - t0:.1f}s")
        except Exception as exc:
            msg = f"{tag}: {exc}"
            failures.append(msg)
            print(f"  FAILED: {exc}")
            import traceback

            traceback.print_exc()

    print("\nRunning cross-domain analyses...")
    for name, module_path, fn_name in CROSS_DOMAIN:
        print(f"  -> {name}")
        try:
            _run_module(module_path, fn_name)
        except Exception as exc:
            failures.append(f"cross_{name}: {exc}")
            print(f"     FAILED: {exc}")

    from cypha_bench.report.generate_report import build_report

    out = build_report()
    print(f"\nDone. Report: {out}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
