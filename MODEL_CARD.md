# Cypha model card

Competition / submission card for the living native runtime. Numbers below are
from `bench/BASELINE_LOCK.json` unless noted. Paper draft figures that still
cite 2.873 BPC are **historical** (L1 pin).

## Identity

| Field | Value |
|-------|-------|
| Name | Cypha (`cypha::Cypha`) |
| Version (latest GitHub release) | v2.3.25 (2026-07-18) |
| Intended next release | v2.4.0 (unreleased on `main`) |
| License | [CC BY 4.0](LICENSE) |
| Runtime | Native C++ only (`cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`) |
| Paper | `paper/arxiv_bundle/CyphaLM_paper.pdf` |

One public type owns classify, regress, latent sample, and next-token generate.

## Intended use

- Research / competition: online classification and regression on vector features, sequence modelling on WikiText-style corpora, event-forecasting benches (GDELT / VIEWS / MID).
- Not a drop-in transformer replacement. Proof surface is CTest parity + locked BPC, not a public LLM leaderboard.

## Production sequence pin (living)

| Item | Value |
|------|-------|
| Recipe | Hybrid GRIA+LSTM, L2 + Wave2 BPTT (Adam, bptt=8, lr=0.001) |
| Corpus | WikiText-2 official train/valid |
| Budget | 300k train tokens, 2k eval, seed 42 |
| **BPC** | **2.664** (`overnight_results` 2.664300395908913, run 2026-08-08) |
| Prior pins | L1 2.873 · SGD L2 2.816 (historical only) |

## Other locked / attested numbers

| Domain | Result | Notes |
|--------|--------|-------|
| D01 linear-sep | 0.9875 vs logistic 0.8875 | `BASELINE_LOCK` / CTest pins |
| D03 XOR (default) | ~76% latent RFF vs sklearn ~79% | Living default; linear LLR ~48% is historical |
| D10A ECG5000 | 90.11% | `CYPHA_D10_ECG_ENRICH=0` keeps 85.96% |
| Cell sweep | 36 variants; lock `status=historical` | July H19 2.921 is **not** the living spine |

## Training data

- Sequence lock: WikiText-2 (scripts: `scripts/download_wikitext2.ps1`). Gutenberg fallback when WikiText is absent.
- Forecast benches: sample CSVs under `bench/data/forecast/`; optional bulk via `scripts/fetch_forecast_data.ps1`.
- Classification benches: synthetic goldens + sklearn-style sets documented in `docs/RESEARCH_STATUS.md`.

## Limits (honest)

- Shared-model continual learning (D16B) remains open; zero-forgetting is per-file isolation (D16F).
- Cell-hypothesis sweep is a historical research tool; Hybrid 2.664 is the product default.
- Windows CI compiles but does not run the full CTest matrix (Linux `build_and_test` does).
- Paper PDF / HTML in `paper/arxiv_bundle/` may still mention 2.873 in body text; lock + this card are authoritative.

## Evaluation how-to

```powershell
powershell -File scripts\validate_baseline_lock.ps1 -Production
powershell -File scripts\cypha_native_validate_all.ps1
```

CTest tally is whatever `ctest -N -R native_` / `scripts/cypha_native_validate_all.ps1` reports — do not hardcode a count in submissions.
