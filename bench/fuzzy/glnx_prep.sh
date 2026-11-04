#!/bin/bash
# GLNX data preprocessing (docs/GENERATION_PLAN.md §13.10): run CyphaLM's whole model once over the 1 MiB wiki holdout
# and save every expert's per-byte probability of the true byte, its bucket, and the content mixer's chain stages, so
# layer-2 experiments (bench/fuzzy/glnx.py) read a file in seconds instead of re-running the model. Odin Loch, 2026-11-01.
# Skips the run when OUT/experts.f32 exists and its manifest matches this binary and these flags (FORCE=1 redoes it).
# Usage: bench/fuzzy/glnx_prep.sh [OUT]   Settings: CYPHALM_TRIE, ENWIK8, DOLLY, EXTRA (more context8k flags).
set -euo pipefail
OUT=${1:-/root/work/glnx}
B=${CYPHALM_TRIE:-native/build/cyphalm_trie}
E=${ENWIK8:-/root/cdata/enwik8}
D=${DOLLY:-/root/cdata/dolly.jsonl}
EXTRA=${EXTRA:---frozen-mix --no-online-final --no-gln --no-lstm --one-rate --no-ngram --no-full-mix}  # the frozen chain's columns, as glnx.py reads them (FULL_DISTRIBUTION_MIXING.md: no full mix)
mkdir -p "$OUT"
key="$(sha256sum "$B" | cut -c1-16) $(basename "$E") $(basename "$D") $EXTRA"
if [ "${FORCE:-0}" != 1 ] && [ -f "$OUT/experts.f32" ] && [ -f "$OUT/manifest.txt" ] && grep -qxF "key $key" "$OUT/manifest.txt"; then
    echo "glnx_prep: $OUT/experts.f32 is current; nothing to do"
    exit 0
fi
"$B" context8k "$E" "$D" $EXTRA --dump-experts "$OUT/experts.f32" > "$OUT/c8.out" 2> "$OUT/c8.err"
{
    echo "key $key"
    echo "made $(date -u +%FT%TZ) by $B on $(uname -n)"
    echo "command $B context8k $E $D $EXTRA --dump-experts $OUT/experts.f32"
    echo "rows $(( $(stat -c %s "$OUT/experts.f32") / 4 / 36 )) columns 36 float32 (bench/fuzzy/glnx.py COLS)"
    echo "sha256 $(sha256sum "$OUT/experts.f32" | cut -c1-64)"
    grep '_bpc' "$OUT/c8.out" | sed 's/^/chain /'
} > "$OUT/manifest.txt"
cat "$OUT/manifest.txt"
