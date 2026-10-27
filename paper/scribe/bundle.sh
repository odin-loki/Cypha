#!/usr/bin/env bash
# Assembles the Zenodo bundle: the paper (PDF, HTML, Markdown), metadata, and the result tables.
# Usage: paper/scribe/bundle.sh   (after build.sh and scripts/scribe_eval/report.py)
set -euo pipefail
cd "$(dirname "$0")"
out=SCRIBE_zenodo_v1.0.zip
rm -f "$out"
zip -q -r "$out" SCRIBE_paper.pdf SCRIBE_paper.html SCRIBE_paper.md paper.css zenodo.json UPLOAD.md results
echo "wrote $out ($(du -h "$out" | cut -f1))"
