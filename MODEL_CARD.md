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

One public type owns classify, regress, latent sample, and **byte-level generation**. The language model being taken forward is the **packed byte trie**. **hp gate24** remains in the tree as the comparison baseline.

## CyphaLM — packed trie first

The model to train on the corpus is the packed byte trie. hp gate24 is the comparison, not the training target. Gate24's observe score on enwik8 is **1.611729 BPC** at about **1.5 GB**. The trie mixture on the same size of text is **2.298 BPC**, peaking near **116 MB** on that slice and near **400 MB** across 10⁹ bytes. Gate24's tables are sized in advance. They do not grow with the data in a way that can be trained through a terabyte, and they are the wrong footprint for day-to-day use.

The serve CLI and REST `/generate` path still call hp. That wiring has not moved. The decision is which model the next train builds.

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

- Research / competition: online classification and regression on vector features, **byte-level language modelling and text generation** (CyphaLM). Event forecasting (GDELT / VIEWS / MID) is a scaffold under repair and has no results yet ([FC0](docs/reports/FC0_FORECAST_AUDIT.md), [plan](docs/FORECAST_PLAN.md)).
- Not a drop-in transformer replacement. Proof surface is CTest parity + **gate24 BPC (~1.61 enwik)** + generation smokes — not a public LLM leaderboard.

## Production sequence algorithm (living)

| Item | Value |
|------|-------|
| Algorithm | **hp** integer-exact context mixer ([CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm)) |
| Integration | `HpSequenceBackend` → `hp::Predictor`; `apply_hp_production_recipe()` (gate24: v78 + `HP_SLOT_MAX=24`) |
| Production knobs | `hp_table_bits=22`, `hp_slot_max=24`, `hp_mixer_lr=2`, `hp_gria=true`, byte vocab ≤ 256 |
| Compile profile | **gate24 only** — gate24-only vendored hp (v78 set baked in, no `-DHP_*` list; [strip](docs/reports/CYPHALM_HP_GATE24_STRIP.md)) + `HP_SLOT_MAX=24` + XSIMD |
| Priority | **Inference / generation latency** first; train and BPC eval may be slower |
| RAM hotspot | `HpSequenceBackend`: live `pred_` + scratch fork; bit-tree uses delta undo (one fork, patch backtrack) |
| Lab RSS (hp harness, mem 22) | **~1.5–2 GB** @ `SLOT_MAX=24` (`hp/tools/hp_harness.sh`) |
| Cypha BPC (train/eval) | **`eval_bpc` / `compress_equivalent_bpc`**: bit-serial observe NLL |
| Serve log probs | **`serve_predict_next`** — bit-tree default; legacy fork: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` |
| Generation | **`generate_decode` / `cyphalm_generate`** — serve path; greedy uses O(8) `serve_greedy_next` |
| Historical pin | Hybrid GRIA+LSTM **2.664 BPC** @ 300k WikiText-2 (Aug 2026) — **superseded**; see [`docs/history/LEGACY_LLM.md`](docs/history/LEGACY_LLM.md) |

> **Note:** Pre-hp BPC numbers in `bench/BASELINE_LOCK.json` are historical. Gate24 hp enwik numbers above are the living LLM quality bar (see [`CYPHALM_LLM_EVAL.md`](docs/reports/CYPHALM_LLM_EVAL.md)). The D17 hybrid **2.664** pin is a bench lock, not this quality bar.

## Small trie (not the shipping LLM)

A packed byte trie plus a six-expert mixture, measured October 2026 on enwik9's first 8 MiB. It is not gate24 and it is not the best writer. The tiny hp mixer (~1.92 BPC, ~40 MB, surface about 0.997 on its best sample) still writes closer to the file.

| Item | Measured |
|------|----------|
| Fixed backoff, orders 1–4 + match | **2.386 BPC**, surface about **0.74–0.83** |
| Learned mixture, frozen | **2.298 BPC**. Weights about **0.93** order 4, **0.06** match |
| Packed-layout accounting | **5.4 MB** stored counts |
| Live training RAM | Keeps singletons during the pass. About **116 MB** peak on the 8 MiB slice, about **400 MB** peak across 10⁹ bytes of enwik9. Packed upper bound at that size is about **45 MB**. |
| Dolly, 15,011 turns | Seen `User:` / `Assistant:` prompts copy the stored reply. Unseen prompts do not answer. Wiki BPC after that pass **2.341**. Stored about **11 MB**. |
| Tried and worse than backoff | PPMD ~**2.42** BPC. GRIA used as a probability gate **2.642**. Long chains and inheritance did not help. |

Hashes of whole contexts are uniform. The structure worth storing is the bytes, parent first. `r_eu` and `32/(32+n)` are not mix weights. Uncertainty for the match is `u = H(order-1) / ln(256)`.

BPC is the held-out negative log probability. It is not a generation score. Surface score `S = H·D·L` is implemented in `native/tools/surface_score.cpp`.

Details and the corpus plan: [`docs/CURRENT.md`](docs/CURRENT.md). Question answering and count-context results (2026-10-26): [`docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md`](docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md). Default plan: [`docs/PLAN.md`](docs/PLAN.md).

## Other locked / attested numbers

| Domain | Result | Notes |
|--------|--------|-------|
| D01 linear-sep | 0.9875 vs logistic 0.8875 | `BASELINE_LOCK` / CTest pins |
| D03 XOR (default) | ~76% latent RFF vs sklearn ~79% | Living default; linear LLR ~48% is historical |
| D10A ECG5000 | 90.11% | `CYPHA_D10_ECG_ENRICH=0` keeps 85.96% |
| Cell sweep | 36 variants; lock `status=historical` | July H19 2.921 is **not** the living spine |

## Training data

- Sequence lock: WikiText-2 (scripts: `scripts/download_wikitext2.ps1`). Gutenberg fallback when WikiText is absent.
- Forecast benches: placeholder CSVs under `bench/data/forecast/`, not drawn from the named datasets, and six of seven are UTF-16, which the loaders cannot read. Real extracts with a provenance manifest come with FORECAST_PLAN FC3. Do not use `scripts/fetch_forecast_data.ps1 -Bulk` output for results until then ([FC0](docs/reports/FC0_FORECAST_AUDIT.md) §1 G).
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
- Cell-hypothesis sweep and Hybrid GRIA+LSTM pins are historical. **hp gate24** is the comparison baseline (~1.61 BPC, ~1.5 GB). The language model being trained next is the **packed trie**.
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
