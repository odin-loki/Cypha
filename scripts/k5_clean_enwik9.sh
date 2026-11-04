#!/bin/bash
# K5 on more Wikipedia (docs/reports/K5_SCALE.md): clean enwik9 past enwik8's bytes with SCRIBE's cleaner, remove
# GEN-A's and SCRIBE's reference/test text by 32-byte spans (scripts/k5_overlap.cpp), and write nested training texts,
# one document per line: train_7M, train_30M, train_100M, train_300M and train_all.
#
# usage: scripts/k5_clean_enwik9.sh ENWIK9 ENWIK8 SCRIBE_BIN OUT_DIR [PYTHON]
#   e.g. scripts/k5_clean_enwik9.sh /mnt/d/cypha_llm/enwik9 /mnt/d/cypha_llm/enwik8 \
#        ~/cypha_build/k5scale/scribe /mnt/d/cypha_llm/k5scale/clean python3
# enwik9's first 100,000,000 bytes are enwik8. 96 MiB (100,663,296 bytes) is the first whole MiB past them; 954 MiB is
# past enwik9's end (1,000,000,000 bytes; the cleaner clamps it).
#
# Forbidden (removed): 32-byte spans of GEN-A's prompts and continuations (dev and test) and of SCRIBE's reference and
# test texts (enwik8 [48, 64) MiB, cleaned). A document whose hits cover more than 10% of its bytes is dropped; in any
# other document each sentence that overlaps a hit is cut. Counted only: SCRIBE's pack (enwik8 [16, 48) MiB) and
# GEN-A's whole source region (enwik8 [72 MiB, end)).
#
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
E9=$1 E8=$2 SCRIBE=$3 OUT=$4 PY=${5:-python3}
mkdir -p "$OUT"
g++ -std=c++20 -O2 -o "$OUT/k5_overlap" "$ROOT/scripts/k5_overlap.cpp"
[ -s "$OUT/clean_e9.jsonl" ] || "$SCRIBE" docs "enwik:$E9:96:954" --out "$OUT/clean_e9.jsonl"
[ -s "$OUT/scribe_reftest.jsonl" ] || "$SCRIBE" docs "enwik:$E8:48:64" --out "$OUT/scribe_reftest.jsonl"
[ -s "$OUT/scribe_pack.jsonl" ] || "$SCRIBE" docs "enwik:$E8:16:48" --out "$OUT/scribe_pack.jsonl"
[ -s "$OUT/gena_source.jsonl" ] || "$SCRIBE" docs "enwik:$E8:72:96" --out "$OUT/gena_source.jsonl"
P="$PY -I $ROOT/scripts/k5_scale_prep.py"
for f in clean_e9 scribe_reftest scribe_pack gena_source; do $P lines "$OUT/$f.jsonl" "$OUT/$f.txt"; done
$P gena "$ROOT/bench/eval" "$OUT/gena_texts.txt"
"$OUT/k5_overlap" clean "$OUT/clean_e9.txt" "$OUT/kept_e9.txt" "$OUT/overlap_report.txt" \
    "$OUT/gena_texts.txt" "$OUT/scribe_reftest.txt" --info "$OUT/scribe_pack.txt" "$OUT/gena_source.txt"
$P nest "$OUT/kept_e9.txt" "$OUT"
# The second, independent pass: the largest text, as written, holds no forbidden span.
"$OUT/k5_overlap" check "$OUT/train_all.txt" "$OUT/gena_texts.txt" "$OUT/scribe_reftest.txt" | tee "$OUT/recheck.txt"
sha256sum "$OUT"/train_*.txt > "$OUT/train_sha256.txt"
wc -c "$OUT"/train_*.txt
