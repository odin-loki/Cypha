#!/usr/bin/env bash
# Builds SCRIBE_paper.html and SCRIBE_paper.pdf from SCRIBE_paper.md (pandoc 3 + Playwright Chromium).
# Usage: paper/scribe/build.sh   (run from anywhere; needs pandoc and node with the playwright package)
set -euo pipefail
cd "$(dirname "$0")"
pandoc SCRIBE_paper.md -s --mathml --css paper.css --embed-resources --metadata pagetitle="SCRIBE" -o SCRIBE_paper.html
NODE_PATH="${NODE_PATH:-$(npm root -g)}" node - "$PWD/SCRIBE_paper.html" "$PWD/SCRIBE_paper.pdf" <<'JS'
const { chromium } = require('playwright');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + inp, { waitUntil: 'networkidle' });
  await p.pdf({ path: out, format: 'A4', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:7pt;width:100%;text-align:center;color:#666">SCRIBE · Odin Loch · <span class="pageNumber"></span>/<span class="totalPages"></span></div>',
    margin: { top: '16mm', bottom: '18mm', left: '15mm', right: '15mm' } });
  await b.close();
})();
JS
echo "built SCRIBE_paper.html and SCRIBE_paper.pdf"
