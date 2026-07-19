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

`powershell -File scripts/publish_release.ps1 -DryRun` succeeded for tag `v2.3.24` (notes under `%TEMP%\cypha_release_notes_v2.3.24.md`). Live publish remains blocked until `gh auth login`.

## Stretch closed same day

Real WikiText BPE hybrid @300k → **BPC 4.154** (vocab 427) — does **not** beat char hybrid pin **2.873**. Documented STOP in wave-2 status.
