#!/usr/bin/env bash
# Generation quality harness: greedy + temperature samples via cyphalm_generation_harness.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_GEN_HARNESS_BUILD:-$ROOT/native/build-gen-harness}"
HARNESS="$BUILD_DIR/cyphalm_generation_harness"
OUT_DIR="${CYPHA_GEN_HARNESS_OUT:-$ROOT/bench/results/generation_harness}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

if [[ ! -x "$HARNESS" ]]; then
  echo "Building cyphalm_generation_harness..."
  cmake -S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc >/dev/null
  cmake --build "$BUILD_DIR" --target cyphalm_generation_harness -j"$(nproc)" >/dev/null
fi

MAX_TOKENS=32
TABLE_BITS=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --table-bits) TABLE_BITS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

JSON_OUT="$OUT_DIR/generation_harness_${STAMP}.json"
MD_OUT="$OUT_DIR/generation_harness_${STAMP}.md"
DOCS_JSON="$ROOT/docs/reports/CYPHALM_GENERATION_HARNESS.json"
DOCS_MD="$ROOT/docs/reports/CYPHALM_GENERATION_HARNESS.md"

echo "=== cyphalm_generation_harness ==="
"$HARNESS" --out "$JSON_OUT" --max-tokens "$MAX_TOKENS" --table-bits "$TABLE_BITS" >/dev/null

python3 - "$JSON_OUT" "$MD_OUT" <<'PY'
import json, sys, textwrap
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(src.read_text())

lines = [
    "# CyphaLM generation harness (qualitative)",
    "",
    f"- harness: `{data.get('harness')}`",
    f"- hp_table_bits: {data.get('hp_table_bits')}",
    f"- max_tokens: {data.get('max_tokens')}",
    f"- runs: {data.get('run_count')}",
    "",
    data.get("note", ""),
    "",
    "## Samples",
    "",
]

for run in data.get("runs", []):
    lines.append(f"### {run['prompt_id']} — {run['strategy']} (seed={run.get('seed', '-')})")
    lines.append(f"- source: {run.get('prompt_source', '')}")
    lines.append(f"- prompt ({run.get('prompt_bytes', 0)} B): `{run.get('prompt_text', '')[:120]}`")
    lines.append(f"- decode_ms: {run.get('decode_ms', 0):.1f}")
    comp = run.get("completion_text", "")
    lines.append(f"- completion ({run.get('generated_bytes', 0)} B):")
    lines.append("```")
    lines.append(comp if comp else "(empty)")
    lines.append("```")
    halted = run.get("halted_on_uncertainty") or run.get("halted_on_epistemic")
    if halted:
        lines.append("- halted early on uncertainty/epistemic gate")
    lines.append("")

lines.extend([
    "## Qualitative notes (human-readable, not scores)",
    "",
    "- **Cold start:** gate24 hp with no corpus warmup; expect repetitive or markup-like continuations.",
    "- **Greedy vs temperature:** greedy should be deterministic per prompt; temperature runs differ by seed.",
    "- **Structure:** XML/C/code prompts test whether continuations stay in-token class (tags, braces, semicolons).",
    "- **No automated quality metric** — inspect completions above for coherence, repetition, and charset drift.",
    "",
])

dst.write_text("\n".join(lines))
print(f"wrote {dst}")
PY

cp "$JSON_OUT" "$DOCS_JSON"
cp "$MD_OUT" "$DOCS_MD"
echo "wrote $JSON_OUT"
echo "wrote $MD_OUT"
echo "copied to docs/reports/CYPHALM_GENERATION_HARNESS.{json,md}"
