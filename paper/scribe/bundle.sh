#!/usr/bin/env bash
# Assembles the Zenodo bundle: the paper (PDF, HTML, Markdown), metadata, and the result tables.
# Usage: paper/scribe/bundle.sh   (after build.sh and scripts/scribe_eval/report.py)
set -euo pipefail
cd "$(dirname "$0")"
out=SCRIBE_zenodo_v1.0.zip
rm -f "$out"
# The trie and CyphaLM tables belong to the Cypha project, not to the metric study, so they stay out of the bundle.
zip -q -r "$out" SCRIBE_paper.pdf SCRIBE_paper.html SCRIBE_paper.md paper.css zenodo.json UPLOAD.md results \
  -x "results/cyphalm_*" "results/*trie*"
echo "wrote $out ($(du -h "$out" | cut -f1))"
