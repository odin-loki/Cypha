# Continuum closeout — 2026-07-18

Agent-actionable BoW / upgrade work closed. Human-only remainders: venue/arXiv upload, optional pandoc PDF, optional 300k H15 re-row, ECG5000 for D10.

## Done this continuum

| Item | Result |
|------|--------|
| H15 post-NaN | Finite BPC **3.982** @5k; cell-sweep CSV **25/25** locally |
| D14 residual RFF | Full-tier mean R² **0.678** (opt-in) |
| Math §0-bis mid-tier | Measured; keep OFF / STOP |
| Paper package | `paper/arxiv_bundle/` + zip |
| Windows gate | **MSVC** — `windows_msvc` CI + `package_release_windows.ps1` (MinGW retired from gate) |
| MSVC compile | `NOMINMAX` + `<algorithm>` for `std::max` under `cl` |

## Human next

1. Upload arXiv / choose venue (`paper/arxiv_bundle/`)
2. Optional: pandoc → PDF; re-tag `v2.3.24` after green `windows_msvc` for fresh MSVC zip + Linux assets
3. Optional: 300k H15; ECG5000 dataset for D10>90%
