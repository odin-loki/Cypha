#!/usr/bin/env bash
# Assembles the Zenodo uploads, next to SCRIBE_paper.pdf:
#   SCRIBE_code_v1.0.zip     all the code (metric, tool, vectors, evaluation scripts, run scripts), the study plan,
#                            the freeze record, the paper source, a standalone CMake build and the licence
#   SCRIBE_results_v1.0.zip  every result table of the paper (Markdown, CSV, JSON)
# Usage: paper/scribe/bundle.sh   (after build.sh and scripts/scribe_eval/report.py)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
rm -f "$here"/SCRIBE_code_v1.0.zip "$here"/SCRIBE_results_v1.0.zip "$here"/SCRIBE_zenodo_v1.0.zip

c="$stage/SCRIBE_code_v1.0"
mkdir -p "$c/native/include/cypha/eval" "$c/native/tools" "$c/scripts" "$c/docs/reports" "$c/paper"
cp "$repo/native/include/cypha/eval/scribe.hpp" "$c/native/include/cypha/eval/"
cp "$repo/native/tools/scribe.cpp" "$repo/native/tools/ppmi_svd.cpp" "$c/native/tools/"
cp -r "$repo/scripts/scribe_eval" "$c/scripts/"
find "$c/scripts" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$repo/docs/SCRIBE_PAPER_PLAN.md" "$c/docs/"
cp "$repo/docs/reports/SCRIBE_V2.md" "$c/docs/reports/"
cp "$here/SCRIBE_paper.md" "$here/paper.css" "$here/build.sh" "$c/paper/"
cp "$here/code/README.md" "$here/code/CMakeLists.txt" "$here/code/requirements.txt" "$c/"
cp "$repo/LICENSE" "$c/"
(cd "$stage" && zip -q -r "$here/SCRIBE_code_v1.0.zip" SCRIBE_code_v1.0)

r="$stage/SCRIBE_results_v1.0"
mkdir -p "$r"
# The trie and CyphaLM tables belong to another project, not to the metric study, so they stay out.
cp -r "$here/results/." "$r/"
rm -f "$r"/cyphalm_* "$r"/*trie*
grep -v -E "wiki_fresh_trie|wiki_fresh_W_trie|cyphalm_cls" "$r/index.md" > "$r/index.md.tmp" && mv "$r/index.md.tmp" "$r/index.md"
python3 - "$r/index.json" <<'PY'
import json, sys
p = sys.argv[1]
skip = {"wiki_fresh_trie", "wiki_fresh_W_trie", "cyphalm_cls"}
d = json.load(open(p))
d = [x for x in d if x.get("name") not in skip]
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
PY
(cd "$stage" && zip -q -r "$here/SCRIBE_results_v1.0.zip" SCRIBE_results_v1.0)

echo "wrote SCRIBE_code_v1.0.zip ($(du -h "$here/SCRIBE_code_v1.0.zip" | cut -f1)) and SCRIBE_results_v1.0.zip ($(du -h "$here/SCRIBE_results_v1.0.zip" | cut -f1))"
