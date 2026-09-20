#!/usr/bin/env bash
# Generation quality harness: cold vs primed+penalty samples via cyphalm_generation_harness.
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
WARMUP_FILE="${CYPHA_GEN_WARMUP_FILE:-$ROOT/bench/data/canterbury/alice29.txt}"
WARMUP_BYTES=4096
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --table-bits) TABLE_BITS="$2"; shift 2 ;;
    --warmup-file) WARMUP_FILE="$2"; shift 2 ;;
    --warmup-bytes) WARMUP_BYTES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

JSON_OUT="$OUT_DIR/generation_harness_${STAMP}.json"
MD_OUT="$OUT_DIR/generation_harness_${STAMP}.md"
DOCS_JSON="$ROOT/docs/reports/CYPHALM_GENERATION_HARNESS.json"
DOCS_MD="$ROOT/docs/reports/CYPHALM_GENERATION_HARNESS.md"

echo "=== cyphalm_generation_harness (cold vs primed) ==="
"$HARNESS" --out "$JSON_OUT" --max-tokens "$MAX_TOKENS" --table-bits "$TABLE_BITS" \
  --warmup-file "$WARMUP_FILE" --warmup-bytes "$WARMUP_BYTES" >/dev/null

python3 - "$JSON_OUT" "$MD_OUT" <<'PY'
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(src.read_text())

lines = [
    "# CyphaLM generation harness (cold vs primed, qualitative)",
    "",
    f"- harness: `{data.get('harness')}`",
    f"- hp_table_bits: {data.get('hp_table_bits')}",
    f"- max_tokens: {data.get('max_tokens')}",
    f"- warmup_file: `{data.get('warmup_file')}`",
    f"- warmup_bytes: {data.get('warmup_bytes')}",
    f"- primed defaults: {data.get('primed_decode_defaults')}",
    f"- runs: {data.get('run_count')}",
    "",
    data.get("note", ""),
    "",
    "## Samples (before / after per prompt)",
    "",
]

by_prompt = {}
for run in data.get("runs", []):
    by_prompt.setdefault(run["prompt_id"], []).append(run)

for prompt_id, group in by_prompt.items():
    lines.append(f"### {prompt_id}")
    lines.append(f"- source: {group[0].get('prompt_source', '')}")
    lines.append(f"- prompt ({group[0].get('prompt_bytes', 0)} B): `{group[0].get('prompt_text', '')[:120]}`")
    lines.append("")
    for run in group:
        lines.append(f"#### {run['profile']} — {run['decode_params'].get('strategy')} "
                     f"(decode_ms={run.get('decode_ms', 0):.1f})")
        comp = run.get("completion_text", "")
        lines.append("```")
        lines.append(comp if comp else "(empty)")
        lines.append("```")
        lines.append("")
    lines.append("")

lines.extend([
    "## Latency notes",
    "",
    "- Compare `decode_ms` for `cold_*` vs `primed_*` per prompt/strategy.",
    "- Warmup cost is included in primed `decode_ms` (one-shot serve_advance over warmup corpus).",
    "- No automated quality metric — inspect completions for repetition, charset, and structure.",
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
