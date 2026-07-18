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

## Historical pins (archived)

- D17 hybrid GRIA+LSTM BPC **2.873** (`bench/BASELINE_LOCK.json`) — historical only; not the living production spine.
- Unified-context smoke winner **U06** (PGM→Wy) is the internal token spine for `Cypha::predict_next` / `generate`.

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
- Living sequence default: **PGM→Wy** via product entry (`Cypha::init_default_sequence` → U06 / `apply_pgm_logits_recipe`); bare `CyphaLMConfig` struct defaults Hybrid for benches; checkpoint JSON persists unified/PGM flags + `pgm_wy`/`pgm_by` + **`pgm_cell` state**
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

- Hybrid sources kept for historical benches (`--mode hybrid`, profiles `_meta.status=historical`) — not deleted

## Release

**v2.3.25** (2026-07-18) — One Cypha cutover shipped at commit `5efc585`; CI release assets live. See [`RELEASE_V2_3_25_2026-07-18.md`](RELEASE_V2_3_25_2026-07-18.md).
