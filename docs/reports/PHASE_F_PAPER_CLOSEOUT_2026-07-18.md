# Phase F paper closeout — 2026-07-18

| Item | Status |
|------|--------|
| Native figure JSON | `paper/figures/native_fig_*.json` |
| Native figure PNG | rendered via `scripts/render_native_paper_figures.py` |
| Bibliography | expanded in `paper/CyphaLM_paper.md` §References (14 entries) |
| Footer / figure note | updated to cite native PNGs |
| `cypha_som` BoW checkbox | closed → archive README |
| Venue submit / arXiv upload | **human** (2027 Q1) |
| GitHub Release publish | **human** — needs `gh auth login`; offline: `pwsh -File scripts/publish_release.ps1 -DryRun` |

## Release dry-run

`publish_release.ps1 -DryRun` generates notes without calling `gh`. Live publish remains blocked until maintainer auth.
