# One Cypha cutover

Inventory and cutover notes for collapsing CyphaDIF + CyphaLM into a single public type `cypha::Cypha`.

## Product brains (before)

| Brain | Prior packaging | Prior brand |
|-------|-----------------|-------------|
| Classify | `CyphaInferModel` + `CyphaDifMemoryState` + `.cypha` v3 | REST `"CyphaDIF"` |
| Regress | MoE/MKE + `regression_head.json` sidecar | bolted onto `/predict` |
| Latent sample | `generation.hpp` free funcs | `POST /dif/generate` (`space: latent`) |
| Tokens | `CyphaLMModel` / `cypha_lm_native` | `/lm/*`, Qt “CyphaLM” |

## Target

- One public type: `cypha::Cypha`
- One brand word: **Cypha**
- Capabilities: classify + regress + sample latents + next-token + text generate
- Latent sample ≠ full feature-row synthesis (no inverse encoder)

## Sequence spine (living)

- D17 hybrid GRIA+LSTM L2 + Wave2 BPTT BPC **2.664** (`bench/BASELINE_LOCK.json` production pin @ 300k) is the **living production default** for modeling and generating text (`apply_hybrid_production_recipe`: Adam, bptt=8, lr=0.001).
- Prior stacked-L2 pin **2.816** and pre-BPTT **2.873** are archived in lock notes / `data/archive/profiles/`.
- Predictive arithmetic coding (LLMZip-style) is integrated: next-token probs → range coder (`compress_tokens` / `decompress_tokens`, REST `/sequence/compress` + `/sequence/decompress`).
- Unified-context smoke winner **U06** (PGM→Wy) remains an **opt-in** cell variant (`--mode pgm_logits` / `apply_cell_variant("U06")`), not the product default.

## Route map

| Before | After |
|--------|--------|
| `/predict`, `/update` | unchanged paths, owned by `Cypha` |
| `/dif/generate` | `/sample` |
| `/dif/retrieve` | `/retrieve` |
| `/generate` (tokens) | unchanged, owned by `Cypha` |
| `/lm/*`, `/dif/*` | deleted |

## Phases

1. Own everything in `cypha::Cypha` (wrap then replace).
2. Burn dual APIs and brand strings.
3. Merge `cypha_lm_native` into `cypha_core`; drop hybrid as default.
4. Docs / lock purge.

## Status (2026-07-18) — Phase 3 ownership

**Done**

- `cypha::Cypha` in [`native/include/cypha/cypha.hpp`](../../native/include/cypha/cypha.hpp); `cypha_lm_native` is an INTERFACE alias compiled into `cypha_core`
- REST: `/sample`, `/retrieve`, `/sequence/load`, `/predict_next`; health `model_type=Cypha`; metrics `sequence_loaded` (+ `lm_loaded` alias)
- CLI: `--sequence-checkpoint` / `CYPHA_SEQUENCE_CHECKPOINT` (aliases: `--cyphalm-checkpoint`, `CYPHA_LM_CHECKPOINT`, `CYPHALM_CHECKPOINT`)
- Living sequence default: **Hybrid GRIA+LSTM** via `Cypha::init_default_sequence` → `apply_hybrid_production_recipe`; CLI `cyphalm_bench_native --mode hybrid` applies the same production recipe (Wave2 BPTT, Adam); U06/PGM→Wy opt-in; checkpoint JSON still persists unified/PGM flags when used
- Regression: `cypha/regression.hpp` (was `regression_stub`); `DifRegressorHead` deleted
- **`Cypha::save`**: merges `mem_` into retained `.cypha` root; no phantom REST `/save`
- REST sequence single-owner: `cyphalm_rest_configure(&g_mu, g_cypha)`; latent `/sample` + `/retrieve` via `g_cypha`
- Registry `LoadedModelBundle` owns `unique_ptr<Cypha>`; `predict`/`update` via `Cypha` for any slot (batch, ewc, replay_u01, kernel flags included)
- **`Cypha::predict`** honors `self_correct`; REST primary path admits `self_correct` + `return_explanation`
- **`Cypha::save`** writes GH session keys (`ood_sigma`, `gh_chi_session`, `gh_psi_session`, `gh_R_base`, `gh_inv_v_clean`)
- Qt **`studio_cypha_`** owns classify load/save/predict/sample + sequence (`model_` / `lm_cypha_` removed); `TrainingWorker` gets `infer()` / `mem()` pointers
- Docs/locks: `BASELINE_LOCK.json` cell sweep → `historical`; living FastAPI refs demoted; `FUTURE.md` hybrid wording; cell-variant CLI help demoted
- Smoke: `cypha_one_smoke`, `pgm_cell_smoke`, `pgm_checkpoint_roundtrip_smoke`, `native_cypha_rest_one_smoke` green

## Status (2026-07-18) — Phase 4 docs

**Done**

- Phase 4 living FastAPI claims cleared; smoke regress coverage; registry card `model_type=Cypha`

**Residual**

- PGM→Wy (U06) kept as opt-in cell / bench mode — not deleted

## Status (2026-07-19) — Hybrid default + predictive AC

**Done**

- Product + CLI default: Hybrid production recipe (`ngram_fuse_split`, no count prior, schedule_b when unset)
- Arithmetic coder + predictive codec on `predict_next`; `Cypha::{compress,decompress}_tokens`, `generate_via_bits`
- REST: `POST /sequence/compress`, `POST /sequence/decompress` (`bytes_hex`)
- CTests: `native_predictive_codec_smoke`, `native_predictive_codec_bench_smoke`; `cypha_one_smoke` asserts Hybrid + roundtrip

## Status (2026-08-08) — Wave2 BPTT re-pin + forecasting v1

**Done**

- Production recipe: **L2 stacked LSTM + Wave2 BPTT** (Adam, bptt=8, lr=0.001) via `apply_hybrid_production_recipe`
- `bench/BASELINE_LOCK.json` re-pinned **2.664 BPC** @ 300k; lock validators aligned (`validate_baseline_lock.ps1`, `baseline_lock_validate`, d27)
- Event forecasting Phases 1–9 shipped (`native/include/cypha/forecast/`, REST `/forecast/*`, bench `--domain-tag forecast`)
- SORF promotion gate passes (`orf_encoder_bench` → `promote=sorf`)

## Release

**v2.3.25** (2026-07-18) -- One Cypha cutover shipped at commit `5efc585`; CI release assets live. See [`../archive/reports/one_cypha/RELEASE_V2_3_25_2026-07-18.md`](../archive/reports/one_cypha/RELEASE_V2_3_25_2026-07-18.md). Next intended tag is **v2.4.0** after Linux CI is green.

Historical recipe traces: [`data/archive/profiles/`](../../data/archive/profiles/).
