#!/usr/bin/env bash
# enwik8.8mb compress-faithful BPC screen: mem20 vs gate24 mem22 baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_LOSSY_BUILD:-$ROOT/native/build-lossy-screen}"
BENCH="$BUILD_DIR/cyphalm_lossy_bench"
OUT_DIR="${CYPHA_LOSSY_OUT:-$ROOT/bench/results/lossy_enwik_screen}"
CORPUS="$ROOT/bench/data/enwik8/enwik8.8mb"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

if [[ ! -f "$CORPUS" ]]; then
  echo "Missing $CORPUS — see bench/data/enwik8/README.md" >&2
  exit 1
fi

if [[ ! -x "$BENCH" ]]; then
  echo "Building cyphalm_lossy_bench..."
  cmake -S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc >/dev/null
  cmake --build "$BUILD_DIR" --target cyphalm_lossy_bench -j"$(nproc)" >/dev/null
fi

RAW_OUT="$OUT_DIR/enwik_screen_${STAMP}.json"
echo "=== cyphalm_lossy_enwik_screen (full 8MB, ~20+ min) ==="
"$BENCH" --enwik-screen | tee "$RAW_OUT"

python3 - "$RAW_OUT" "$OUT_DIR/enwik_screen_${STAMP}_summary.json" <<'PY'
import json, sys, datetime
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text())
bar = raw.get("gate24_quality_bar_bpc", 1.612)
ref = raw.get("cypha_gate24_ref_observe_bpc", 1.611729)
variants = []
baseline_bpc = None
for v in raw.get("variants", []):
    bpc = v.get("observe_bpc")
    if v["variant"] == "gate24_baseline_mem22":
        baseline_bpc = bpc
    row = {
        "variant": v["variant"],
        "hp_effective_mem": v.get("hp_effective_mem"),
        "observe_bpc": bpc,
        "observe_bpc_ms": v.get("observe_bpc_ms"),
        "observe_bytes_per_sec": v.get("observe_bytes_per_sec"),
        "vm_rss_kb_init": v.get("vm_rss_kb_init"),
        "delta_bpc_vs_baseline": (bpc - baseline_bpc) if baseline_bpc is not None and bpc is not None else None,
        "delta_bpc_vs_bar": (bpc - bar) if bpc is not None else None,
        "passes_gate24_bar": bpc is not None and bpc <= bar,
    }
    if baseline_bpc and v.get("vm_rss_kb_init") and v["variant"] != "gate24_baseline_mem22":
        row["rss_reduction_pct"] = round(100.0 * (1.0 - v["vm_rss_kb_init"] / raw["variants"][0]["vm_rss_kb_init"]), 1)
    variants.append(row)

mem20 = next((v for v in variants if v["variant"] == "lossy_mem20"), None)
verdict = "mem20_fails_gate24_bar_keep_mem22_default"
if mem20 and mem20.get("passes_gate24_bar"):
    verdict = "mem20_passes_gate24_bar"

out = {
    "date": datetime.date.today().isoformat(),
    "harness": "cyphalm_lossy_enwik_screen",
    "corpus": raw.get("corpus_path"),
    "corpus_sha256": raw.get("corpus_sha256"),
    "corpus_bytes": raw.get("corpus_bytes"),
    "metric": "observe_bit_serial_bpc",
    "gate24_quality_bar_bpc": bar,
    "cypha_gate24_ref_observe_bpc": ref,
    "verdict": verdict,
    "variants": variants,
    "recommendation": {
        "default_serve": "mem22 (gate24 baseline)" if verdict.endswith("keep_mem22_default") else "mem20 (passes bar)",
        "safe_levers": ["hp_serve_compact on mem22 (lossless RAM)", "hp_prune_cold_min_n on mem22 (re-screen per threshold)"],
        "opt_in_lossy": None,
    },
}
if mem20:
    d = mem20.get("delta_bpc_vs_baseline")
    rss = mem20.get("rss_reduction_pct")
    out["recommendation"]["opt_in_lossy"] = (
        f"mem20 tier — ~{rss}% RSS reduction, +{d:.4f} BPC on enwik8MB; "
        + ("not gate24-qualified" if not mem20.get("passes_gate24_bar") else "gate24-qualified")
    )

Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"verdict": verdict, "mem20_bpc": mem20.get("observe_bpc") if mem20 else None}, indent=2))
PY

SUMMARY="$OUT_DIR/enwik_screen_${STAMP}_summary.json"
DOCS_JSON="$ROOT/docs/reports/CYPHALM_LOSSY_ENWIK_SCREEN.json"
cp "$SUMMARY" "$DOCS_JSON"
echo "wrote $RAW_OUT"
echo "wrote $SUMMARY"
echo "copied to $DOCS_JSON"
