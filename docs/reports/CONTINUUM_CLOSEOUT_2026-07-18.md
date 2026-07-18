# Continuum closeout — 2026-07-18

Agent-actionable BoW / upgrade work closed (MSVC gate + release assets + ECG5000 + D10>90% + paper PDF).

## Done

| Item | Result |
|------|--------|
| Windows gate | `windows_msvc` (MinGW retired) |
| v2.3.24 assets | MSVC zip + Linux tar.gz + AppImage |
| ECG5000 | Download script; real `data_source=ecg5000` |
| D10A | **90.11%** default (enriched features; legacy 85.96% via `CYPHA_D10_ECG_ENRICH=0`) |
| Paper | `paper/arxiv_bundle/` MD + HTML + PDF (`xelatex`) |
| REST schema CI | `ChildProcess` move-only fix (Linux spawn was self-killing on assign) |

## Human next

1. Venue / arXiv upload from `paper/arxiv_bundle/`
2. Optional: 300k H15 re-row
