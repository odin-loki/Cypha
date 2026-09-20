# Cypha model card

Competition / submission card for the living native runtime. Numbers below are
from `bench/BASELINE_LOCK.json` unless noted. Paper draft figures that still
cite 2.873 BPC are **historical** (L1 pin).

## Identity

| Field | Value |
|-------|-------|
| Name | Cypha (`cypha::Cypha`) |
| Version (latest GitHub release) | v2.5.0 (2026-09-20) |
| Prior release | v2.4.0 (2026-08-16, competition lock; retired) |
| License | [CC BY 4.0](LICENSE) |
| Runtime | Native C++ only (`cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`) |
| Paper | `paper/arxiv_bundle/CyphaLM_paper.pdf` |

One public type owns classify, regress, latent sample, and **byte-level LLM generation** (CyphaLM / hp gate24).

## CyphaLM — living byte LLM (hp gate24)

CyphaLM is a **trainable, servable byte LLM**, not only a compressor. The same vendored **hp** context mixer powers (1) **compress-faithful BPC training/eval**, (2) **fast inference / generation**, and (3) REST + CLI surfaces. **Inference latency is the headline metric**; training may be slower.

| Surface | APIs | Notes |
|---------|------|-------|
| **Train / BPC** | `train_step`, `observe_stream_bits`, `eval_bpc_compress_equivalent` | Bit-serial observe path (8 bits/byte). Quality bar: **~1.61 BPC** on enwik8 gate24 (see below). |
| **Serve / generate** | `serve_advance`, `serve_predict_next`, `serve_greedy_next`, `generate_decode` | No `train_step_count` / `adapt_after_predict` bookkeeping. Greedy: O(8) bit argmax. |
| **CLI** | `cyphalm_generate` | `--strategy greedy\|temperature\|top_k`, `--prompt`, `--max-tokens` |
| **REST** | `POST /generate`, `/generate/stream` | Same decode path as `generate_decode` |

Full API notes: [`docs/native/CYPHALM_SERVE.md`](docs/native/CYPHALM_SERVE.md).

### Quality bar (gate24, compress-faithful)

Corpus: `bench/data/enwik8/enwik8.8mb` — vendored hp gate24, mem 22, `HP_SLOT_MAX=24`.

| Metric | Value | Meaning |
|--------|-------|---------|
| **Cypha observe BPC** | **1.611729** | `eval_bpc_compress_equivalent` — bit-serial NLL |
| **hp archive BPC** | **1.611759** | hp `c`/`d` archive bytes |
| Upstream RECORD (s24) | 1.607 | Reference only; vendored-tree drift ~+0.005 BPC |

Generation log-probs use the **serve** path (bit-tree joint scoring by default; legacy 256-clone: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`). That is **inference math**, not the compress-faithful BPC claim above.

### Train vs serve (hp backend)

| Path | When | Cost profile |
|------|------|----------------|
| **Train** | `train_step`, BPC eval, `cyphalm_train` | Bit-serial observe; online table updates; no 256-vocab fan-out |
| **Serve** | `generate_decode`, `cyphalm_generate`, REST `/generate` | Prompt priming via `serve_advance`; greedy skips full-vocab scoring; top-k/temperature uses bit-tree log probs |

Removed light/champ hp SKUs: [`docs/history/REMOVED_HP_SKUS.md`](docs/history/REMOVED_HP_SKUS.md). Superseded Hybrid GRIA+LSTM stack: [`docs/history/LEGACY_LLM.md`](docs/history/LEGACY_LLM.md).

## Intended use

- Research / competition: online classification and regression on vector features, **byte-level language modelling and text generation** (CyphaLM), event-forecasting benches (GDELT / VIEWS / MID).
- Not a drop-in transformer replacement. Proof surface is CTest parity + **gate24 BPC (~1.61 enwik)** + generation smokes — not a public LLM leaderboard.

## Production sequence algorithm (living)

| Item | Value |
|------|-------|
| Algorithm | **hp** integer-exact context mixer ([CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm)) |
| Integration | `HpSequenceBackend` → `hp::Predictor`; `apply_hp_production_recipe()` (gate24: v78 + `HP_SLOT_MAX=24`) |
| Production knobs | `hp_table_bits=22`, `hp_slot_max=24`, `hp_mixer_lr=2`, `hp_gria=true`, byte vocab ≤ 256 |
| Compile profile | **gate24 only** — v78_flags.ps1 + `HP_SLOT_MAX=24` + XSIMD (no light/champ SKU matrix) |
| Priority | **Inference / generation latency** first; train and BPC eval may be slower |
| RAM hotspot | `HpSequenceBackend`: live `pred_` + scratch fork; bit-tree uses delta undo (one fork, patch backtrack) |
| Lab RSS (hp harness, mem 22) | **~1.5–2 GB** @ `SLOT_MAX=24` (`hp/tools/hp_harness.sh`) |
| Cypha BPC (train/eval) | **`eval_bpc` / `compress_equivalent_bpc`**: bit-serial observe NLL |
| Serve log probs | **`serve_predict_next`** — bit-tree default; legacy fork: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` |
| Generation | **`generate_decode` / `cyphalm_generate`** — serve path; greedy uses O(8) `serve_greedy_next` |
| Historical pin | Hybrid GRIA+LSTM **2.664 BPC** @ 300k WikiText-2 (Aug 2026) — **superseded**; see [`docs/history/LEGACY_LLM.md`](docs/history/LEGACY_LLM.md) |

> **Note:** Pre-hp BPC numbers in `bench/BASELINE_LOCK.json` are historical. Gate24 hp enwik numbers above are the living LLM quality bar (see [`CYPHALM_LLM_EVAL.md`](docs/reports/CYPHALM_LLM_EVAL.md)).

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
| hp light/champ SKUs | Removed 2026-09; **gate24 only** | [`docs/history/REMOVED_HP_SKUS.md`](docs/history/REMOVED_HP_SKUS.md) |
| RPSM (d21) | Removed 2026-09; superseded by hp | [`docs/history/REMOVED_RPSM.md`](docs/history/REMOVED_RPSM.md) |

Pre-removal source remains in git history on `main` and PR #1 commits.

## Limits (honest)

- Shared-model continual learning (D16B) remains open; zero-forgetting is per-file isolation (D16F).
- Cell-hypothesis sweep and Hybrid GRIA+LSTM pins are historical; **hp** is the product LLM algorithm.
- Windows CI compiles but does not run the full CTest matrix (Linux `build_and_test` does).
- Training is CPU. Optional CUDA is infer-only (DIF/classify path); CyphaLM hp train stays on CPU. GPU training is slower on this workload and is not a gap. **CyphaLM optimizes inference latency** over train throughput; future CPU speed path: portable SIMD via xsimd (`docs/FUTURE.md` §1b).
- Paper PDF / HTML in `paper/arxiv_bundle/` may still mention 2.873 in body text; lock + this card are authoritative.

## Evaluation how-to

**CyphaLM (LLM):**

```bash
cmake --build native/build --target cyphalm_generate cyphalm_serve_smoke -j$(nproc)
./native/build/cyphalm_generate --prompt "Hello " --max-tokens 32 --strategy greedy
bash scripts/measure_enwik_gate24.sh bench/data/enwik8/enwik8.8mb   # BPC quality bar (~1.61)
```

**Full native gate:**
```powershell
powershell -File scripts\validate_baseline_lock.ps1 -Production
powershell -File scripts\cypha_native_validate_all.ps1
```

CTest tally is whatever `ctest -N -R native_` / `scripts/cypha_native_validate_all.ps1` reports — do not hardcode a count in submissions.

Historical recipe / XOR dumps: [`data/archive/`](data/README.md). Do not treat `artifacts/` or `bench/results/` as committed evidence.
