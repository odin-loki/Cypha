# Cypha model card

Competition / submission card for the living native runtime. Numbers below are
from `bench/BASELINE_LOCK.json` unless noted. Paper draft figures that still
cite 2.873 BPC are **historical** (L1 pin).

## Identity

| Field | Value |
|-------|-------|
| Name | Cypha (`cypha::Cypha`) |
| Version (latest GitHub release) | v2.4.0 (2026-08-16) |
| Prior release | v2.3.25 (2026-07-18, One Cypha cutover) |
| License | [CC BY 4.0](LICENSE) |
| Runtime | Native C++ only (`cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`) |
| Paper | `paper/arxiv_bundle/CyphaLM_paper.pdf` |

One public type owns classify, regress, latent sample, and next-token generate.

## Intended use

- Research / competition: online classification and regression on vector features, sequence modelling on WikiText-style corpora, event-forecasting benches (GDELT / VIEWS / MID).
- Not a drop-in transformer replacement. Proof surface is CTest parity + locked BPC, not a public LLM leaderboard.

## Production sequence algorithm (living)

| Item | Value |
|------|-------|
| Algorithm | **hp** integer-exact context mixer ([CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm)) |
| Integration | `HpSequenceBackend` → `hp::Predictor`; `apply_hp_production_recipe()` (gate24: v78 + `HP_SLOT_MAX=24`) |
| Production knobs | `hp_table_bits=22`, `hp_slot_max=24`, `hp_mixer_lr=2`, `hp_gria=true`, byte vocab ≤ 256 |
| Compile profile | **gate24 only** — v78_flags.ps1 + `HP_SLOT_MAX=24` + XSIMD (no light/champ SKU matrix) |
| RAM hotspot | `HpSequenceBackend` holds `pred_` + `scratch_` + nine depth checkpoints; serve uses bit-tree DFS (`copy_state_from` backtrack) |
| Lab RSS (hp harness, mem 22) | **~1.5–2 GB** @ `SLOT_MAX=24` (`hp/tools/hp_harness.sh`) |
| Cypha BPC (default) | **`eval_bpc` / `compress_equivalent_bpc`**: bit-serial observe NLL |
| Cypha BPC (API / top-k) | **`serve_predict_next` + legacy fork log probs** (opt-in bit-tree: `CYPHA_HP_BIT_TREE_LOGPROBS=1`) |
| **Generation** | `generate_decode` / `cyphalm_generate` — serve path; greedy uses O(8) `serve_greedy_next` |
| **hp enwik8.8mb (measured 2026-09-19)** | **observe 1.611729**, **archive 1.611759** (vendored hp gate24). Upstream RECORD s24 ref **1.607**. See [`CYPHALM_LLM_EVAL.md`](docs/reports/CYPHALM_LLM_EVAL.md). Removed SKUs: [`REMOVED_HP_SKUS.md`](docs/history/REMOVED_HP_SKUS.md) |
| Historical pin | Hybrid GRIA+LSTM **2.664 BPC** @ 300k WikiText-2 (Aug 2026) — **superseded**; do not compare to hp BPC without relabeling |

> **Note:** Pre-hp BPC numbers in `bench/BASELINE_LOCK.json` are historical. New hp-backed BPC baselines are not yet locked in that file.

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

## History / superseded (not current product)

| Stack | Status | Doc |
|-------|--------|-----|
| Hybrid GRIA+LSTM (2.664 / 2.873 BPC pins) | Gated (`-DCYPHA_BUILD_LEGACY_CYPHALM=ON`); not default | [`docs/history/LEGACY_LLM.md`](docs/history/LEGACY_LLM.md) |
| RPSM (d21) | Removed 2026-09; superseded by hp | [`docs/history/REMOVED_RPSM.md`](docs/history/REMOVED_RPSM.md) |

Pre-removal source remains in git history on `main` and PR #1 commits.

## Limits (honest)

- Shared-model continual learning (D16B) remains open; zero-forgetting is per-file isolation (D16F).
- Cell-hypothesis sweep and Hybrid GRIA+LSTM pins are historical; **hp** is the product LLM algorithm.
- Windows CI compiles but does not run the full CTest matrix (Linux `build_and_test` does).
- Training is CPU. Optional CUDA is infer-only; GPU training is slower on this workload and is not a gap. Future speed path: portable SIMD via xsimd (`docs/FUTURE.md` §1b).
- Paper PDF / HTML in `paper/arxiv_bundle/` may still mention 2.873 in body text; lock + this card are authoritative.

## Evaluation how-to

```powershell
powershell -File scripts\validate_baseline_lock.ps1 -Production
powershell -File scripts\cypha_native_validate_all.ps1
```

CTest tally is whatever `ctest -N -R native_` / `scripts/cypha_native_validate_all.ps1` reports — do not hardcode a count in submissions.

Historical recipe / XOR dumps: [`data/archive/`](data/README.md). Do not treat `artifacts/` or `bench/results/` as committed evidence.
