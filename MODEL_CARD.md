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
| Integration | `HpSequenceBackend` → `hp::Predictor`; `apply_hp_production_recipe()` (light) / `apply_hp_gate24_recipe()` (quality screen) / `apply_hp_champ_recipe()` (champ) |
| Production knobs | `hp_table_bits=22`, `hp_slot_max=24`, `hp_mixer_lr=2`, `hp_gria=true`, byte vocab ≤ 256 |
| Compile SKUs | **light** (CI default, fast/dev, 0/78 v78, ~1.72 BPC enwik8MB) · **gate24** (`-DCYPHA_HP_PROFILE=gate24`, 78/78 v78 + SLOT_MAX=24, ~1.61 BPC class) · **champ** (`-DCYPHA_HP_PROFILE=champ`, 78/78 + SLOT_MAX=35, ~1.610 bar, ~15 GB RSS) |
| RAM hotspot | `HpSequenceBackend` holds `pred_` + `scratch_` + DFS checkpoint pool; full-vocab `next_byte_log_probs()` uses **bit-tree prefix DFS** (legacy 256-clone: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`); **BPC default uses bit-serial observe** |
| Lab RSS (hp harness, mem 22) | **~1.6 GB** @ `SLOT_MAX=24`; **~15 GB** @ `SLOT_MAX=35` (`hp/tools/hp_harness.sh` RECORD H34) |
| Cypha BPC (default) | **`eval_bpc` / `compress_equivalent_bpc`**: bit-serial observe NLL — **matches hp archive BPC** on same corpus/flags (see gap report) |
| Cypha BPC (API / top-k) | **`predict_next` + 256-clone path**: different metric; **not** hp archive BPC — use only when reporting REST/inference behavior |
| **hp profile (measured 2026-09-19)** | **Win metric:** enwik8.8mb vs **1.610906** (champ) / **1.611759** (gate24). **light:** archive/observe **1.721** (0/78 v78). **gate24:** archive **1.611759**, observe **1.611729** (78/78, RT PASS). **champ:** OOM @ 15 GiB VM. WikiText ~2.1 (light) is **not** comparable. See [`docs/reports/CYPHALM_LLM_EVAL.md`](docs/reports/CYPHALM_LLM_EVAL.md), [`docs/reports/CYPHALM_HP_ALGORITHM_PROFILE.md`](docs/reports/CYPHALM_HP_ALGORITHM_PROFILE.md) |
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
