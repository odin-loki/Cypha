#!/usr/bin/env bash
# Assembles the Zenodo uploads, next to SCRIBE_paper.pdf.
#
# Version 2.0 (the default; the SCRIBE v3 paper):
#   SCRIBE_code_v2.0.zip     all the code (metric and tool with profiles v1, v2, v3 and v31, vectors, evaluation scripts,
#                            run scripts), the study plan, every SCRIBE report (v0 to v3.1 and the v3 paper's
#                            registration), the paper source, a standalone CMake build and the licence
#   SCRIBE_results_v2.0.zip  every result table of version 2.0 (results_v2.0/) and, for reference, version 1.0's (results/)
# Version 1.0 (kept for reference; the SCRIBE v2 paper as published): `bundle.sh v1.0` rebuilds
#   v1.0/SCRIBE_code_v1.0.zip and v1.0/SCRIBE_results_v1.0.zip from the version 1.0 sources.
#
# Usage: paper/scribe/bundle.sh [v2.0|v1.0]   (after build.sh and scripts/scribe_eval/v3_report.py)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ver="${1:-v2.0}"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

# The trie and CyphaLM tables belong to another project, not to the metric study, so they stay out.
strip_results() {  # $1: a copy of a results folder
  rm -f "$1"/cyphalm_* "$1"/*trie*
  if [ -f "$1/index.md" ]; then
    grep -v -E "wiki_fresh_trie|wiki_fresh_W_trie|cyphalm_cls" "$1/index.md" > "$1/index.md.tmp" && mv "$1/index.md.tmp" "$1/index.md"
  fi
  if [ -f "$1/index.json" ]; then
    python3 - "$1/index.json" <<'PY'
import json, sys
p = sys.argv[1]
skip = {"wiki_fresh_trie", "wiki_fresh_W_trie", "cyphalm_cls"}
d = json.load(open(p))
d = [x for x in d if x.get("name") not in skip]
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
PY
  fi
}

if [ "$ver" = v1.0 ]; then
  out="$here/v1.0"
  rm -f "$out"/SCRIBE_code_v1.0.zip "$out"/SCRIBE_results_v1.0.zip
  c="$stage/SCRIBE_code_v1.0"
  mkdir -p "$c/native/include/cypha/eval" "$c/native/tools" "$c/scripts" "$c/docs/reports" "$c/paper"
  cp "$repo/native/include/cypha/eval/scribe.hpp" "$c/native/include/cypha/eval/"
  cp "$repo/native/tools/eval/scribe.cpp" "$repo/native/tools/semantics/ppmi_svd.cpp" "$c/native/tools/"
  cp -r "$repo/scripts/scribe_eval" "$c/scripts/"
  find "$c/scripts" -name "__pycache__" -type d -prune -exec rm -rf {} +
  cp "$repo/docs/SCRIBE_PAPER_PLAN.md" "$c/docs/"
  cp "$repo/docs/reports/SCRIBE_V2.md" "$c/docs/reports/"
  cp "$out/SCRIBE_paper.md" "$here/paper.css" "$here/build.sh" "$c/paper/"
  cp "$here/code/README.md" "$here/code/CMakeLists.txt" "$here/code/requirements.txt" "$c/"
  cp "$repo/LICENSE" "$c/"
  (cd "$stage" && zip -q -r "$out/SCRIBE_code_v1.0.zip" SCRIBE_code_v1.0)
  r="$stage/SCRIBE_results_v1.0"
  mkdir -p "$r"
  cp -r "$here/results/." "$r/"
  strip_results "$r"
  (cd "$stage" && zip -q -r "$out/SCRIBE_results_v1.0.zip" SCRIBE_results_v1.0)
  echo "wrote v1.0/SCRIBE_code_v1.0.zip and v1.0/SCRIBE_results_v1.0.zip"
  exit 0
fi

[ "$ver" = v2.0 ] || { echo "unknown version $ver (v2.0 or v1.0)"; exit 2; }
rm -f "$here"/SCRIBE_code_v2.0.zip "$here"/SCRIBE_results_v2.0.zip
c="$stage/SCRIBE_code_v2.0"
mkdir -p "$c/native/include/cypha/eval" "$c/native/tools" "$c/scripts" "$c/docs/reports" "$c/paper"
cp "$repo/native/include/cypha/eval/scribe.hpp" "$c/native/include/cypha/eval/"
cp "$repo/native/tools/eval/scribe.cpp" "$repo/native/tools/semantics/ppmi_svd.cpp" "$c/native/tools/"
cp -r "$repo/scripts/scribe_eval" "$c/scripts/"
find "$c/scripts" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$repo/docs/SCRIBE_PAPER_PLAN.md" "$c/docs/"
for f in SCRIBE_V0 SCRIBE_V1 SCRIBE_V2 SCRIBE_V3 SCRIBE_V31 SCRIBE_V3_PAPER K5_UPGRADES K5_SCALE; do
  cp "$repo/docs/reports/$f.md" "$c/docs/reports/"
done
cp "$here/SCRIBE_paper.md" "$here/paper.css" "$here/build.sh" "$c/paper/"
cp "$here/code/README.md" "$here/code/CMakeLists.txt" "$here/code/requirements.txt" "$c/"
cp "$repo/LICENSE" "$c/"
(cd "$stage" && zip -q -r "$here/SCRIBE_code_v2.0.zip" SCRIBE_code_v2.0)

r="$stage/SCRIBE_results_v2.0"
mkdir -p "$r/v2.0" "$r/v1.0"
cp -r "$here/results_v2.0/." "$r/v2.0/"
cp -r "$here/results/." "$r/v1.0/"
strip_results "$r/v1.0"
(cd "$stage" && zip -q -r "$here/SCRIBE_results_v2.0.zip" SCRIBE_results_v2.0)

echo "wrote SCRIBE_code_v2.0.zip ($(du -h "$here/SCRIBE_code_v2.0.zip" | cut -f1)) and SCRIBE_results_v2.0.zip ($(du -h "$here/SCRIBE_results_v2.0.zip" | cut -f1))"
