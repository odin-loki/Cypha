#!/usr/bin/env python3
"""
Production GPU benchmark bundle for Cypha (requires CuPy + CUDA in *this* interpreter).

Runs, in order:
  1. scripts/gpu_microbench.py   — raw fp64 GEMM CPU vs GPU
  2. scripts/gpu_fullbench.py    — encode + score_matrix + softmax + gate (CPU-patched vs GPU)
  3. scripts/tune_quality_performance.py — coarse grid + generation + heavy CuPy burn + per-cell GPU stress

Use Python 3.10–3.12 with a matching cupy-cuda* wheel (CuPy does not support Python 3.14 yet).

  python scripts/bench_gpu_production.py
  python scripts/bench_gpu_production.py --skip-tune
  python scripts/bench_gpu_production.py --medium-extra --medium-max-combos 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _require_cuda() -> None:
    try:
        import cupy  # noqa: F401, WPS433
    except ImportError as e:
        print(
            "CuPy is not importable in this Python.\n"
            "  • Use Python 3.10–3.12 (not 3.14) for official CuPy wheels.\n"
            "  • pip install cupy-cuda12x   # or cupy-cuda11x — match your CUDA\n"
            "  • https://docs.cupy.dev/en/stable/install.html\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    sys.path.insert(0, str(_ROOT))
    from cypha_accel.cuda_util import cuda_gemm_usable

    if not cuda_gemm_usable():
        print(
            "CuPy imported but cuda_gemm_usable() is False (no usable CUDA device / driver?).",
            file=sys.stderr,
        )
        raise SystemExit(3)


def _run(py: str, argv: list[str]) -> float:
    print(f"\n{'=' * 72}\n$ {py} {' '.join(argv)}\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([py, str(_ROOT / argv[0])] + argv[1:], cwd=str(_ROOT), check=True)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tune", action="store_true", help="Only micro + fullbench")
    ap.add_argument(
        "--medium-extra",
        action="store_true",
        help="After main tune, run medium preset subsample (more hyper combos, still jobs=1)",
    )
    ap.add_argument("--medium-max-combos", type=int, default=300)
    ap.add_argument("--python", default=sys.executable, help="Interpreter to use for child scripts")
    args = ap.parse_args()

    _require_cuda()

    from cypha_accel.cuda_util import warmup_cuda

    warmup_cuda()

    py = args.python
    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": py,
        "steps": [],
    }

    wall = _run(py, ["scripts/gpu_microbench.py", "--n", "8192", "--d", "256", "--k", "128", "--repeat", "7"])
    report["steps"].append({"name": "gpu_microbench", "wall_s": round(wall, 3)})

    wall = _run(
        py,
        ["scripts/gpu_fullbench.py", "--n", "4096", "--d", "128", "--k", "32", "--repeat", "7", "--train-steps", "500"],
    )
    report["steps"].append({"name": "gpu_fullbench", "wall_s": round(wall, 3)})

    if not args.skip_tune:
        tune_argv = [
            "scripts/tune_quality_performance.py",
            "--preset",
            "coarse",
            "--include-generation",
            "--jobs",
            "1",
            "--gpu-burn-passes",
            "64",
            "--gpu-batch-n",
            "12288",
            "--gpu-stress-repeats",
            "24",
            "--profile-top",
            "0",
        ]
        wall = _run(py, tune_argv)
        report["steps"].append({"name": "tune_gpu_heavy_coarse", "wall_s": round(wall, 3)})

        if args.medium_extra:
            med = [
                "scripts/tune_quality_performance.py",
                "--preset",
                "medium",
                "--max-combos",
                str(max(1, args.medium_max_combos)),
                "--include-generation",
                "--jobs",
                "1",
                "--gpu-burn-passes",
                "32",
                "--gpu-batch-n",
                "8192",
                "--gpu-stress-repeats",
                "12",
                "--profile-top",
                "0",
            ]
            wall = _run(py, med)
            report["steps"].append({"name": "tune_gpu_heavy_medium_subsample", "wall_s": round(wall, 3)})

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["total_wall_s"] = round(sum(s["wall_s"] for s in report["steps"]), 3)

    out_dir = _ROOT / "artifacts" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_gpu_production_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
