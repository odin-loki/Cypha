# CyphaLM Native Upgrade — Master Tracker

Native C++ port of CyphaLM (Tiers 0, 1, 2, 4). Python reference for D17 WikiText-2 @ 300k: **hybrid_gria_lstm → 2.873 BPC** (`cyphalm_d17_wikitext.json` `_meta.held_out_bpc`).

**Build:** `C:\Temp\cypha_native_build6` (Release, Ninja, outside OneDrive)  
**Bench:** `cyphalm_bench_native.exe --profile d17 --n-train 300000 --n-eval 2000 --threads 1`  
**Results log:** [`CYPHALM_NATIVE_BENCH_RESULTS.jsonl`](CYPHALM_NATIVE_BENCH_RESULTS.jsonl)

## D17 WikiText-2 @ 300k — build6 (2026-06-10)

| Mode | Native BPC | Python / prior | Δ vs Python hybrid |
|------|------------|----------------|--------------------|
| **char_lstm** | **2.900** | ~2.98 char baseline | −0.08 vs char |
| **hybrid** | **2.892** (v2 post LSTM fix) | **2.873** | **+0.019** |
| **ssm_gria** | **3.515** | gria_ngram stack | — |
| **ssm** | **4.458** | ssm_only | — |
| **spectral** | **4.458** | spectral PDE on | same as ssm @ 300k |
| **context_bank** | **3.604** (v2 post attn fix) | build3: 4.637 | — |

All runs: `train_epochs=2`, `schedule_b`, isolated exe `C:\Temp\cyphalm_bench_build6_run.exe`, `CYPHALM_TRAIN_LOG_EVERY=100000`.

## Tier status

| Tier | Scope | Status |
|------|-------|--------|
| **0** | Char LSTM, embed, corpus, bench CLI | ✅ char_lstm @ 2.900 BPC |
| **1** | CellAI SSM, CyphaDIF, GRIA low-rank, ngram fusion, views | ✅ ssm_gria 3.515; hybrid path wired |
| **2** | Unified `CyphaLMModel`, hybrid blend, BPTT-64, schedule_b | ✅ hybrid **2.892 BPC @ 300k**; checkpoint + REST |
| **4** | Hebbian / multiscale / spectral flags | ✅ flags wired; spectral = ssm numerically @ 300k |

## Critical fixes (build3 → build6)

| Bug | Fix |
|-----|-----|
| `eval_bpc` reset trained weights | Reset context only |
| Hybrid LSTM never trained | `hybrid_lstm_cache_` + `backward_step` / `apply_grads` |
| BPTT buffer avg discarded | `apply_bptt_delta_avg()` on `W_fast[0]` |
| DIF reset every block | Removed `dif_->reset()` from `reset_context()` (`carry_dif=True`) |
| Hybrid blend double LSTM forward | Cache `last_hybrid_log_g_` / `last_hybrid_log_l_` |
| RNG seed mismatch vs Python | GRIA `seed+2`, LSTM `seed+5`; sequential proj RNG |
| Context bank attn on wrong dims | Project attn via `proj_embed_` → field_dim (build7) |
| **Hybrid LSTM in-place `forward_step`** | Separate `h_new`/`c_new` buffers (was zeroing `c` before read) |

## Open parity gaps

1. **Hybrid parity** — ✅ **2.892 BPC @ 300k** (Python 2.873, Δ +0.019); root cause was in-place LSTM `forward_step`
2. **context_bank @ 300k** — ✅ **3.604 BPC** after `proj_embed_` attn fix (was 7.14)
3. **GRIA full-rank** — ✅ Python `W` import via rank-32 ALS factorization (`load_from_full_w`)

## Native REST LM (cypha_rest)

| Method | Path | Role |
|--------|------|------|
| POST | `/lm/load` | Load `{checkpoint_path}` → `.json` + `.npz` |
| GET | `/lm/metrics` | vocab, field_dim, context_mode, generation counts |
| POST | `/lm/predict_next` | `{ token_id }` → log_probs + uncertainty |
| POST | `/generate` | `{ prompt_ids, max_tokens, strategy, temperature, top_k, top_p, uncertainty_threshold, seed, stream }` |
| POST | `/generate/stream` | Same body; always SSE (`text/event-stream`) |

Startup: `--cyphalm-checkpoint <base>` or env `CYPHALM_CHECKPOINT` auto-loads LM at boot.

`/health` includes `lm_loaded`. Classifier routes unchanged.

## Operational notes (Windows)

- One bench process at a time; copy exe to `C:\Temp\cyphalm_bench_*_run.exe` before long runs.
- Kill stale `cyphalm*` from other sessions before starting.
- Use `--threads 1` for hybrid (OpenMP `0` unstable).
- Prefer `cmd /c "set VAR=...&& exe ... > out.json 2> err.log"` over PowerShell `&` (stderr progress aborts PS).

## Commands

```powershell
cmake -S native -B C:\Temp\cypha_native_build6 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build C:\Temp\cypha_native_build6 --target cyphalm_bench_native cyphalm_parity cyphalm_checkpoint_parity --parallel
ctest --test-dir C:\Temp\cypha_native_build6 -R native_cyphalm

# Regenerate Python→native checkpoint fixture
python scripts/generate_cyphalm_checkpoint_fixture.py

# Sequential 300k sweep (remaining modes)
powershell -File scripts/cyphalm_native_run_modes.ps1
```

## Results log

| Date | Tier | Result |
|------|------|--------|
| 2026-05-31 | fix | **`eval_bpc` bug:** was evaluating fresh untrained model (BPC≈8.0); fixed to reset context on trained weights only. |
| 2026-05-31 | fix | **`apply_bench_profile`:** loads `cyphalm_d17_wikitext.json` hyperparams (2 epochs, `gria_lr=0.08`, `schedule_b`). |
| 2026-05-31 | bench | **build2** D17 sweep (`C:\Temp\cypha_native_build2\cyphalm_bench_native.exe`, `--threads 0`, wikitext2): hybrid @ 40k **3.701 BPC** (sanity). |
| 2026-05-31 | bench | **build2** hybrid @ 300k **3.353 BPC** (Python ref **2.873**; Δ +0.480). |
| 2026-05-31 | bench | **build2** char_lstm @ 300k **2.920 BPC**; ssm **4.417**; ssm_gria **3.517**; spectral **4.417**; context_bank **6.510** (all `--profile d17 --n-eval 2000`). |
| 2026-05-31 | bench | Full build2 sweep JSONL: [`CYPHALM_NATIVE_BENCH_RESULTS.jsonl`](CYPHALM_NATIVE_BENCH_RESULTS.jsonl). Runs flaky under concurrent bench; use one process at a time. |

## Next iteration

- [x] Rebuild with context_bank + blend diagnostics; rerun `context_bank` @ 300k → 3.604
- [x] Hybrid 300k with LSTM fix → **2.892 BPC**
- [x] Checkpoint save/load + Python char_lstm numeric lock (`cyphalm_checkpoint_parity`)
- [x] Full-rank GRIA import for Python hybrid checkpoints (`load_from_full_w`)
- [x] Native REST `/lm/*` + `/generate` in `cypha_rest`
- [x] Decode strategies: `top_p`, `uncertainty_gated`; SSE `/generate/stream` + `stream=true`
- [x] Startup `--cyphalm-checkpoint` / `CYPHALM_CHECKPOINT` env
- [x] REST LM smoke: `native/scripts/smoke_cyphalm_rest_lm.py`; pytest `tests/test_cyphalm_rest_lm_smoke.py`
- [x] Checkpoint DIF + SSM state save/load (`NIGExpert::state_dict`, `CyphaDIF::get_state`, `CellAISSM::get_state`)
- [x] Parity runner repo-root sidecar resolution; pytest fixture key fix
- [x] `scripts/cyphalm_native_validate.ps1` (build + CTest + pytest + REST smoke)

## Optional next

- [x] Wire `proj_dif` into GRIA input for non-hybrid modes (`gria_input_core` in `cyphalm_model.cpp` mirrors Python `_gria_input`)
- [x] Tighten hybrid Python checkpoint load atol after full state import audit (sidecar `atol_bpc=0.02`; measured native delta ~0.002 BPC)
- [x] Full 300k jsonl refresh: `docs/native/CYPHALM_NATIVE_BENCH_RESULTS.jsonl` (build6 sweep — hybrid 2.892, context_bank 3.604, char_lstm 2.900 @ 300k)
