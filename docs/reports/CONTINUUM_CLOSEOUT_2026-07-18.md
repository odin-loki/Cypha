# Continuum closeout — 2026-07-18

Agent-actionable BoW / upgrade work closed. Human-only remainders: venue/arXiv upload, optional full `gh` scopes, optional 300k H15 re-row.

## Done this continuum

| Item | Result |
|------|--------|
| H15 post-NaN | Finite BPC **3.982** @5k; cell-sweep CSV **25/25** locally |
| D14 residual RFF | Full-tier mean R² **0.678** (opt-in; do not default kernel routing) |
| Math §0-bis mid-tier | All measured; eigenvalue `D_eff` / κ / scale-sign / blend-lr stay OFF or STOP |
| Paper package | `paper/arxiv_bundle/` + `paper/arxiv_bundle.zip` (markdown + native figs; PDF needs pandoc) |
| Release | `v2.3.24` live; Windows zip attached; packaging scripts fixed for CI |
| BPE @300k | STOP vs char hybrid pin |
| Sticky CL / eigenvalue D_eff | No promote |

## Human next

1. Upload arXiv / choose venue (bundle ready under `paper/arxiv_bundle/`)
2. Optional: install pandoc → PDF
3. Optional: wait for re-triggered Release CI for Linux + AppImage assets
4. Optional: `gh auth login` with `read:org` if full CLI desired
