#!/bin/bash
# K5 on more Wikipedia (docs/reports/K5_SCALE.md): clean enwik9 past enwik8's bytes with SCRIBE's cleaner, drop every
# document that holds a 32-byte span of GEN-A's prompts or continuations or of SCRIBE's reference/test text, and write
# nested training texts (one document per line): train_7M, train_30M, train_100M, train_300M and train_all.
#
# usage: scripts/k5_clean_enwik9.sh ENWIK9 ENWIK8 SCRIBE_BIN OUT_DIR [PYTHON]
#   e.g. scripts/k5_clean_enwik9.sh /mnt/d/cypha_llm/enwik9 /mnt/d/cypha_llm/enwik8 \
#        ~/cypha_build/k5scale/scribe /mnt/d/cypha_llm/k5scale/clean ~/nn_venv/bin/python
# PYTHON needs numpy. enwik9's first 100,000,000 bytes are enwik8; 96 MiB = 100,663,296 bytes is the first whole
# MiB past them, and 954 MiB is past enwik9's end (1,000,000,000 bytes; the cleaner clamps it).
# Runtime on Odin's PC: about 2 minutes to clean, about 10 minutes for the check and the texts.
#
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
E9=$1 E8=$2 SCRIBE=$3 OUT=$4 PY=${5:-python3}
mkdir -p "$OUT"
[ -s "$OUT/clean_e9.jsonl" ] || "$SCRIBE" docs "enwik:$E9:96:954" --out "$OUT/clean_e9.jsonl"
[ -s "$OUT/scribe_reftest.jsonl" ] || "$SCRIBE" docs "enwik:$E8:48:64" --out "$OUT/scribe_reftest.jsonl"
[ -s "$OUT/scribe_pack.jsonl" ] || "$SCRIBE" docs "enwik:$E8:16:48" --out "$OUT/scribe_pack.jsonl"
[ -s "$OUT/gena_source.jsonl" ] || "$SCRIBE" docs "enwik:$E8:72:96" --out "$OUT/gena_source.jsonl"
"$PY" -I "$ROOT/scripts/k5_scale_prep.py" "$OUT/clean_e9.jsonl" "$OUT/scribe_reftest.jsonl" "$OUT/scribe_pack.jsonl" \
    "$OUT/gena_source.jsonl" "$ROOT/bench/eval" "$OUT"
sha256sum "$OUT"/train_*.txt > "$OUT/train_sha256.txt"
wc -c "$OUT"/train_*.txt
