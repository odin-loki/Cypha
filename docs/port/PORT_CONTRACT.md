# Cypha -> C++ / CUDA / Qt -- frozen reference contracts

This document is the **normative checklist** for native ports. Behavior must match the frozen contracts in this document unless you explicitly version and document a breaking change.

## Product lead

| Layer | Living truth |
|-------|----------------|
| **Public type** | **`cypha::Cypha`** -- classify + regress + latent sample + sequence tokens |
| **REST** | **`cypha_rest`** -- `/predict`, `/update`, `/sample`, `/retrieve`, `/sequence/*`, `/generate` |
| **Sequence default** | **Hybrid GRIA+LSTM L2 + Wave2 BPTT** via `Cypha::init_default_sequence` → `apply_hybrid_production_recipe` (**2.664 BPC** @ 300k lock) |
| **Hybrid GRIA+LSTM** | **Living production pin** (D17 **2.664 BPC** in `bench/BASELINE_LOCK.json`; prior L1 **2.873** archived) |
| **Python names** | **Wire-format history only** (`.cypha` v3 keys, parity fixtures). Runtime packages removed (P7). |

> Dual brands "CyphaDIF" / "CyphaLM" in sections below are historical labels for the same product surfaces. Cutover: [`../reports/ONE_CYPHA_CUTOVER.md`](../reports/ONE_CYPHA_CUTOVER.md).

## 1. Binary state (`.cypha`)

- **Writers / readers**: `cypha_save_binary`, `cypha_load_binary`, **`cypha_save_binary_to_bytes`**, **`cypha_load_binary_from_bytes`** in native **`cypha_core`** (v3 bytes ↔ dict; same layout as native **`save_cypha_to_buffer`** / **`load_cypha_from_buffer`**).
- **Native (C++)**: **`save_cypha_file`** / **`save_cypha_to_buffer`** / **`load_cypha_file`** / **`load_cypha_from_buffer`** / **`clone_cnode`** (`native/include/cypha/load_cypha.hpp`) — same v3 on-disk layout as Python. **M1 / M2 / fixed kernels:** CTests **`native_parity`**, **`native_preprocessor`**, **`native_preprocessor_fit`** (scale + PCA + RFF via **`NumpyDefaultRng`**; **`fixtures/preprocessor_fit_rff/`**), **`native_nig_adapt`**, **`native_regression_mixture`** are gated by matching CTests **`native_*`** (override **`CYPHA_*_PARITY_BIN`** — see **`native/README.md`**). One-step latent **`DIFMemory.train`**: CTest **`native_memory_train`** (env **`CYPHA_MEMORY_TRAIN_PARITY_BIN`**). DIF-memory training state can be merged back into a loaded root via **`CyphaDifMemoryState::merge_state_into_root_for_save`**; CTest **`native_memory_train_roundtrip`** (subprocess + **`cypha_load_binary`** cross-check). One-step **`dif_train_step_vector`** loss: CTest **`native_train_step_vector`** (env **`CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN`**).
- **Magic**: `CYPHA\x00` (6 bytes).
- **Version**: single byte; current **3**. After magic: `version (u8)`, `endian_sentinel (u32) = 0x01020304`, `n_fields (u32)`, then keyed entries.
- **Endianness**: **little-endian** for all multi-byte scalars; if sentinel ≠ `0x01020304`, the file requires byte-swapping (big-endian writer).
- **Tensor rule**: arrays serialized as **float64**, **C-contiguous row-major**; no stride metadata in the file.
- **NIG `field_W_T` / `field_a_eff`**: Historical Python **`CyphaDIF.save_state()`** wrote **`field_W_T`**, **`field_h`**, **`field_step`**, and **`field_a_eff`** (float64 tensor, same values as **`Field._A_eff`** fp32 matvec matrix). Native **`load_cypha_*` → `CyphaInferModel`** uses optional **`field_a_eff`** when shapes match **`field_W_T`**; otherwise recomputes via **`recompute_field_a_eff`** (`native/src/nig_field.cpp`). Qt **`patch_infer_training_snapshot`** and **`memory_train_roundtrip`** (after merge) emit **`field_a_eff`** when a causal field is present. Older v3 files without **`field_a_eff`** still load.
- **Recursive dicts**: dtype `DICT` nests key/value pairs; the tree is the same shape as `save_state()` / `load_state()`.
- **`world.F_field`** (optional in older files; **written by native `Cypha::save` / Qt `patch_infer_training_snapshot`**; historical Python writer did the same): float64 tensor shape `(feat_dim, field_dim)` — field-conditioned shift for μ₀ (`WorldPrior.F_field`). When present, native loaders may omit external `f_field.json`. Older checkpoints without `F_field` still load via sidecar JSON.

Native loaders should accept **version 3** and reject unknown higher versions.

### Qt shell native save parity (`patch_infer_training_snapshot`)

After native training, **`patch_infer_training_snapshot`** in `native/qt/src/shell_main.cpp` writes all keys that the historical Python **`CyphaDIF._save_state()`** writer produced:

| Python `_save_state` key | Native write | Notes |
|---|---|---|
| `classes`, `world` (incl. `F_field`) | `merge_state_into_root_for_save` | exact parity |
| `enc_W`, `field_h`, `temperature`, `base_temp` | in-place update loop | update if present in root |
| `mahal_ema`, `mahal_std_ema`, `llr_scale_ema`, `llr_scale_n`, `llr_scale_baseline`, `llr_ema`, `mid_n`, `mid_freq`, `total_steps` | in-place update loop | update if present; new roots may lack key |
| `ctx_hist_packed`, `ctx_cooccur`, `ctx_cooccur_tot`, `ctx_last_label`, `mid_trans` | `root_map_assign` | insert or update |
| `field_W_T`, `w_inject`, `field_step`, `field_a_eff` | `root_map_assign` | field_a_eff as float64 tensor |
| `ll_world_ema` | `root_map_assign` | hardcoded **`-1.5`** — matches Python `_save_state` (also always writes -1.5) |
| `total_correct`, `feat_dim` | `root_map_assign` | |
|| `field_sr_vec` | `root_map_assign` (optional) | spectral-radius vector, shape `(field_dim,)` float64 — written when causal field is present; native reads if present, otherwise recomputes from `field_W_T`; older v3 without it still load cleanly |
| `ood_sigma`, `gh_chi_session`, `gh_psi_session`, `gh_R_base`, `gh_inv_v_clean` | `root_map_assign` (via `NativeSessionSnapshotPatch`) | `gh_R_base` always float 1.0 if GH unused; Python writes `None` → no functional diff (load checks `gh_inv_v_clean` first) |

**Known remaining gaps (non-functional):**
- Keys updated only in-place (the first loop) are **not inserted** if absent from the loaded `.cypha`. For any Python-generated root, all keys are present so this is not an issue in practice.
- **Key ordering** in the serialized map may differ from Python's insertion order → byte-identical `.cypha` files are not guaranteed, but inference parity is maintained.

### Python reference: `R @ D.T` backend (not in `.cypha`)

`cypha_core` uses `cypha_accel.score_batch.fused_score_llr` for the batched LLR core in `score_matrix` / `generate` (fuses `(H-μ₀)⊙inv_v`, `R @ D.T`, and MDL/context bias). Encoder batch projection uses `project_features`. **CuPy on GPU** when installed and a CUDA device is visible; otherwise **NumPy on CPU**. Numerics must match the reference formula in float64 (CTest **native_***, CTest **native_***). Native code may implement the same ops on any backend.

## 2. Core inference math (Cypha)

Treat this as the **single spec** shared by `infer`, `batch_infer`, `batch_infer_full`, **`cypha_rest`**, and the Qt shell.

1. **Latent**: `h = encoder.project(encoder_fn(x))` with `h` shape `(d,)`, `batch_encode` stacks to `(N, d)`.
2. **LLR**: `score_matrix(H, use_field)` — same μ₀ shift when `use_field=True` (field-conditioned prior). Column order = `memory._label_order`. Per-class MDL term uses **`world.v_mean / (n_obs_k + 1)`** (scalar mean of diagonal **variances** `v`), not `mean(inv_v)` — batch `score_matrix` must match `DIFMemory.classify`.
3. **Class probabilities**: `probs = softmax(LLR / (temperature + ε))` with ε = `1e-8` (`_EPS` in code). Batch path uses `_softmax_batch` (must match row-wise softmax of `infer`).
4. **World gate**: GH–NIG gate as in `DIFMemory.classify` / `world_gate_vector(..., gh_chi=1, gh_psi=1)` — **requires** `gh_chi > 0` and `gh_psi > 0` (`kInferWorldGateApiVersion=2`). The legacy sigmoid-only fallback when GH hyperparameters were non-positive is **removed**; callers must pass positive session hyperparameters or expect `std::runtime_error`.
5. **Confidence**: `conf_i = probs[i, argmax_i] * gate_i` (same as returned `(label, confidence)` from `infer` / `batch_infer`).

**Parity rule**: For the same loaded state, `temperature`, `use_field`, and inputs, `batch_infer` and `infer` must agree on **label and confidence** within floating tolerance (CTest **native_parity**). The batch GH gate calls `_nig_R_eff` row-wise, matching `DIFMemory.classify`. Fixtures assume `use_kernel_llr=False` and deliberation disabled (`deliberation_lo >= deliberation_hi`).

### Optional inference modifiers (kernel LLR, deliberation)

These features exist in native **`cypha_core`**. Parity fixtures for the core linear path are generated with both **kernel LLR** and **deliberation** disabled unless noted.

- **Kernel LLR** (`CyphaDIF(use_kernel_llr=True)`): blends Nyström RBF scores from `KernelMemory` with linear LLRs inside `infer` / `train_step`. `KernelMemory` maintains a reservoir of landmark points (`M=256` default, median-γ bandwidth, whitened Nyström `Φ(h)=K(h,B)K(B,B)^{-1/2}`) and per-class weight vectors; kernel score is `w_k·φ(h) − ½‖w_k‖²`. **Native:** `native/include/cypha/kernel_memory.hpp`, train/infer wired via `TrainStepExtras` / `CyphaInferOptions`; `gh_infer_at_h` accepts optional kernel blend when `.cypha` carries `kernel_mem`. **Persistence:** v3 `.cypha` keys and v3 binary round-trip include `kernel_mem` snapshot (CTest **native_parity**); C++ `export_snapshot` / `import_snapshot` (CTest `native_kernel_snapshot_roundtrip`); v3 `.cypha` root keys via `patch_kernel_into_root` / `try_load_kernel_from_root` (CTest `native_kernel_cypha_roundtrip`). Qt shell: train tab checkbox + `patch_kernel_into_root` on save; load via `try_load_kernel_from_root`. **Native `cypha_rest`:** loads kernel from `.cypha` on startup / registry **`POST /load`**; optional body keys **`use_kernel_llr`** / **`kernel_blend`** on **`POST /predict`** and **`POST /update`** create an in-session `KernelMemory` when enabled and none is loaded. **Parity:** `kernel_llr_parity` + CTest **native_*** (ctest `native_kernel_llr`). Bench opt-in profile: `bench/config/kernel_llr_profile.json`. See [`docs/FUTURE.md §0a`](../FUTURE.md).

- **Deliberation** (`deliberation_lo`, `deliberation_hi`): when `lo < hi` and the inferred confidence falls in `[lo, hi]`, `infer` / `infer_full` return label `__unknown__` and halve the confidence. Default is **`deliberation_lo=1.0`, `deliberation_hi=0.0`** (disabled; `lo >= hi` → no abstention). **Native (Phase 1):** `apply_deliberation` + `infer_at_h` (`native/src/infer_cpu.cpp`); keys loaded from `.cypha`; optional `deliberation_lo` / `deliberation_hi` on `POST /predict` when `use_gh=false`. Not applied on `gh_infer` (matches Python). CTest **`native_gh_infer_deliberation`** + `fixtures/gh_infer_deliberation/` (committed sidecar).

- **`gh_infer` / REST `use_gh`:** `gh_infer_at_h` + `cypha_rest` `POST /predict` when `use_gh=true` (session `g_gh_chi` / `g_gh_psi`; `anomaly_score` from `R_eff` vs `_mahal_ema`; `is_ood` when anomaly > 3.0). CTest **`native_gh_infer_deliberation`**. `/update` with `use_gh=true` remains `dif_gh_train_step_vector`.

## 3. REST API

**Authoritative server:** native **`cypha_rest`** (`--listen 127.0.0.1:7749` by default; `CYPHA_API_HOST` / `CYPHA_API_PORT`). Route shapes below are the stable wire contract.

> **Historical footnote (removed):** An in-tree Python FastAPI app (`create_app` / uvicorn) once mirrored these routes for parity smoke. It is gone; do not treat FastAPI error codes or `InferenceEngine` behaviour as normative.

Typical routes:

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | `{ status, model, uptime, n_predictions }` — `n_predictions` matches the engine counter (same as `/metrics` → `n_predictions`) |
| GET | `/ready` | **`200`** `{ "ready": true, "model_type": str }` when an engine is loaded; **`503`** `{ "ready": false, "reason": "no_model_loaded" }` when not |
| GET | `/metrics` | `uptime_seconds`, `model_loaded`, `model_type`, `n_predictions`, `n_corrections`, `registry_model_count`, **`loaded_model_count`**, **`active_model`**, optional `gh_chi_session` / `gh_psi_session` when Cypha classifier loaded, `session` or `null`, **`regression_head_loaded`** (bool — MoE sidecar active for `/predict`), **`sequence_loaded`** (bool — sequence checkpoint loaded; alias **`lm_loaded`**) |
| POST | `/predict` | Body: `{ "input": [float, ...], "use_gh": bool, "return_explanation": bool, "use_kernel_llr"?: bool, "kernel_blend"?: float, "model"?: "<name>/<version>" }` — omit **`model`** (or set to active key) for byte-compatible single-model default; named slot → **`404`** `{"detail":"model not loaded"}`; optional Nyström kernel LLR when `use_kernel_llr=true` (native creates in-session `KernelMemory` if `.cypha` has none); **`503`** `{ "detail": "No model loaded" }` when no engine |
| POST | `/update` | Body: `{ "input": [...], "correct_label": str, "use_gh": bool, "use_kernel_llr"?: bool, "kernel_blend"?: float, "model"?: "<name>/<version>" }`; success **`200`** → **`{ "loss": float, "n_corrections": int }`** only (no extra keys); omit **`model`** for active globals (single-model compat); named slot → **`404`** `{"detail":"model not loaded"}`; **`503`** `{ "detail": "No model loaded" }` when no model. **Optional:** when `regression_head.json` includes an **`mke`** block (see below), you may add **`regression_y`** (number) to run one scalar **`MKERegressor.train_step`**-style update; **`loss`** is then the router **`dif_train_step_vector`** loss. Optional **`router_train_label`** (string) overrides the router training label; optional **`replay_u01`** (array of numbers) fixes priority-replay uniforms (parity-style). Sending **`regression_y`** without an **`mke`** block → **`400`** `{"detail":"regression_y requires mke block in regression_head.json"}`. |
| POST | `/register` | Body `{ "name", "version", "model_cypha", "card_json", "preprocessor_json"?: str \| null, "overwrite"?: bool }` — absolute or relative **host** paths to existing files; copies into **`<registry_root>/<name>/<version>/`**. Success **`200`** → `{ "registered": true, "model_dir": "<path>" }`; **`503`** `{"detail":"No registry configured"}` when **`cypha_rest`** started without **`--registry`**; failure **`400`** `{"detail":"…"}` (missing sources, destination exists without **`overwrite`**, etc.). **`cypha_rest`** refreshes its in-process registry scan cache after success. CLI **`registry_register`**. |
| POST | `/adapt_temperature` | Body: `{ "calibration": [ { "input": [...], "correct_label": str }, ... ], "n_grid"?, "T_min"?, "T_max"?, "n_bins"? }` → `{ "temperature", "n_used" }` (ECE grid); **`503`** `{ "detail": "No model loaded" }` when no model |
| GET | `/models` | `{ "models": [ ModelCard dicts ], "active_model": "<name>/<version>" \| null }` — each row includes **`loaded`** (bool, in RAM) and **`active`** (bool, hot-swap target / default for omitting **`model`** on predict/update). Empty registry → **`{ "models": [], "active_model": null }`**; query **`summary=true`** (or `1`) → `{ "models": [ { "name", "version", "loaded", "active" }, ... ], "active_model": … }` |
| POST | `/load` | Body: `{ "name": str, "version"?: str }` (`version` defaults to **`latest`**), or `{ "model": "<name>/<version>" }`, or **empty body** → preload all registry bundles into RAM (returns `{ "loaded": [ "<name>/<version>", ... ] }`). On single-model load success: **`200`** `{ "loaded": <ModelCard dict>, "model": "<name>/<version>" }` — hot-swaps **active** globals and fills the in-memory map; **`503`** `{ "detail": "No registry configured" }` if **`cypha_rest`** started without **`--registry`**; **`404`** `{ "detail": "model not found" }` |
| GET | `/session` | `n_predictions`, `n_corrections`, `correction_accuracy`, `mean_confidence`, `mean_anomaly`, `n_ood_flagged`, `label_distribution`, `session_duration_s` (same keys in `/metrics` → `session` when a session exists). **`cypha_rest`** keeps an in-process session buffer when a model is loaded. |
| DELETE | `/session` | → `{ "cleared": true }` (**200**); clears prediction history and session GH χ/ψ (`InferenceSession.clear`). |
| GET | `/session/rng` | `{ "state": [uint32 × 624], "pos": int }` — snapshot of session RNG for deterministic replay cross-runtime. |
| POST | `/session/rng` | Body `{ "seed": int }` or `{ "state": [...], "pos": int }` — re-seed or restore; returns GET shape. Tested in `native/scripts/smoke_cypha_rest_mingw.ps1`. |
| GET | `/classes` | `{ "classes": { label: { "n_obs": float } } }`; **`503`** `{ "detail": "No model loaded" }` when no model |
| GET / POST | `/uncertainty-rank` | Body: `{ "rows": [[float, ...], ...], "top_n"?: int, "temperature"?: float, "curriculum"?: bool }` → `{ "indices", "entropies", "confidences", "top_n" }` (plus `"curriculum": true` when requested), ranked by descending entropy (or ascending confidence when `curriculum=true`). **GET note:** since GET request bodies are not read by the HTTP stack, **`cypha_rest`** requires the JSON payload as a URL-encoded query string: **`GET /uncertainty-rank?payload=<urlencoded-json>`** (e.g. `?payload=%7B%22rows%22%3A...%7D`); a GET with neither `?payload=` nor a body returns **`400`** `{"detail":"JSON body or ?payload=<urlencoded-json> required (GET bodies are not read by the HTTP stack)"}`. POST accepts a normal JSON body. **`503`** `{ "detail": "No model loaded" }` when no model; **`400`** `{ "detail": "…" }` for malformed `rows` / dimension mismatch / MKE-mode requests. |

**Malformed request body:** if the client sends **invalid JSON** on `POST /predict`, `/update`, or `/adapt_temperature`, **`cypha_rest`** responds with **`400`** and `{"detail":"bad json"}`. The same applies to **`POST /load`** when a registry is configured (parse fails before lookup). **Note:** with **no** registry, **`POST /load`** returns **`503`** before parsing the body, so a garbage body still yields **`{"detail":"No registry configured"}`** rather than **`bad json`**.

**Input dimension:** after optional preprocessor transform, vector length must match model latent dim; otherwise **`POST /predict`**, **`/update`**, and **`/adapt_temperature`** (per calibration row) return **`400`** and `{"detail":"input dim mismatch after preprocessor"}`.

**Replay on `/update`:** **`cypha_rest`** drives priority replay from an in-process **`std::mt19937`** session RNG by default. Optional **`replay_u01`** on **`POST /update`** is forwarded through **`TrainStepExtras`** for **classification** (`dif_train_step_vector` / GH) and for **MKE** (`mke_scalar_train_step` when **`regression_y`** + **`mke`**), mirroring parity harness fixed replay uniforms. When **`replay_u01`** is omitted, replay sampling uses the session RNG.

**Optional `/update` keys:** **`regression_y`** + **`mke`** block in **`regression_head.json`** (see below) runs **`mke_scalar_train_step`**; optional **`router_train_label`** overrides the router training label on the MKE path only (ignored on plain classification). **`regression_y`** without an **`mke`** block → **`400`** `{"detail":"regression_y requires mke block in regression_head.json"}`.

**On-disk registry (native tooling):** pre-built **`model.cypha`** + **`card.json`** can be installed under **`<root>/<name>/<version>/`** with **`native/registry_register`** (see **`native/README.md`**) or **`cypha::registry_register_bundle`** — same tree Python **`ModelRegistry`** scans. CTest **`native_registry_register`**; subprocess ctest **CTest **native_***** (env **`CYPHA_REGISTRY_REGISTER_BIN`**).

**Multi-model serving (native `cypha_rest` — FUTURE.md §5, bounded slice):** start with **`--registry <root>`** and optional **`--preload-registry`** to load every registry bundle into an in-process map keyed by **`name/version`**. Each slot has its own **`std::mutex`** (train vs infer serialisation per model). **`POST /predict`** and **`POST /update`** accept optional JSON **`"model": "<name>/<version>"`**; omitting **`model`** (or passing the active key) routes to the legacy single-model globals — byte-compatible when only one model is loaded. **`GET /models`** annotates each card with **`loaded`** / **`active`** and returns top-level **`active_model`**. **`POST /load`** hot-swaps the active globals and upserts the map; empty body preloads all registry entries. **`GET /metrics`** exposes **`loaded_model_count`** and **`active_model`**. LRU eviction of map slots is **not** implemented (follow-up).

**Predict response** (`PredictResponse`): `label`, `confidence`, `all_scores`, `anomaly_score`, `is_ood`, `regression_val`, `uncertainty`, optional `explanation`, `latency_ms`. When `return_explanation` is true, **`cypha_rest`** uses the top-level **`explanation`** key set exercised in **`native/scripts/smoke_cypha_rest_mingw.ps1`** (REST JSON shape smoke).

**Native `cypha_rest` — optional scalar regression head:** with `--regression-json regression_head.json` (or `regression_head.json` beside `model.cypha` on registry **`POST /load`**), JSON shape is (JSON Schema: [`schemas/regression_head.schema.json`](schemas/regression_head.schema.json)):

```json
{ "experts": { "<class_label>": { "mu": <float_or_[d]>, "var_ema": <float> }, ... } }
```

For each loaded class label (same strings as routing / `all_scores` keys), `mu` is the expert target EMA (scalar number or first element of an array for future vector targets). **`cypha_rest`** fills `regression_val` = Σ_k p_k·μ_k and `uncertainty` = √(Σ_k p_k·var_k) using the same softmax `p` as classification (`LLR / temperature` then `softmax_batch_like_python`). If the sidecar is absent, `regression_val` is JSON `null` and `uncertainty` is `0`. Load via **`--regression-json`** or env **`CYPHA_REGRESSION_HEAD`** (registry **`POST /load`** picks up `regression_head.json` beside the model when present).

**Optional `mke` block (native `cypha_rest` — online scalar MKERegressor step):** same file may include **`mke`** with **`d_in`**, **`D_rff`** (must equal classifier latent **`d`**), **`rff_W_rowmajor`**, **`rff_b`**, per-label **`w`** (length **`D_rff`**) and **`P`** (length **`D_rff`²** row-major), **`temperature`**, **`forgetting_factor`**, optional **`pi_floor`** (default **0.02**), optional **`gh_scales`** (length **K**). With **`mke`**, **`/predict`** uses RFF(**`input`**) as routing features (after preprocessor; **`input`** length must be **`d_in`**) and sets **`regression_val`** = Σ_k p_k·(w_k·φ); **`uncertainty`** still uses expert **`var_ema`** mixture when **`experts`** lists **`var_ema`** per label. **`POST /update`** with **`regression_y`** runs **`mke_scalar_train_step`** (see **`native/include/cypha/mke_scalar_train_step.hpp`**).

Qt or C++ clients should treat these JSON shapes as **stable** for v1; add fields additively rather than renaming.

## 4. Sequence REST (Cypha)

**Native `cypha_rest`:** sequence surface (`/sequence/load`, `/sequence/metrics`, `/predict_next`, `/generate`) when built with `cypha_lm_native` / `cypha_core`. Classifier routes unchanged.

| Method | Path | Role |
|--------|------|------|
| POST | `/sequence/load` | Body `{ "checkpoint_path": str }` → load Cypha sequence checkpoint (``.json`` + ``.npz`` pair). **200** `{ "loaded": true, "summary": {...} }`; **503** if sequence stack missing. |
| GET | `/sequence/metrics` | Sequence summary: vocab, experts, generation counts. **503** if no sequence model loaded. |
| POST | `/predict_next` | Body `{ "token_id": int }` → `{ log_probs, epistemic_var, aleatoric_var, routing_probs, dominant_expert, active_experts, top_k_tokens, top_k_probs }`. |
| POST | `/generate` | Body `{ "prompt_ids": [int,...], "max_tokens", "temperature", "strategy", "top_k", "top_p", "uncertainty_threshold", "stream" }`. Strategies: `greedy`, `temperature`, `top_k`, `top_p`, `uncertainty_gated`. **200** batch JSON with `generated_ids`, `per_step`, `per_step_metrics`; or **SSE** when `stream=true`. |
| POST | `/generate/stream` | Same body as `/generate`; always returns **SSE** (`text/event-stream`, one JSON object per token). |

**Environment:** `CYPHA_SEQUENCE_CHECKPOINT` — optional path to load Cypha sequence at app startup. Aliases: `CYPHALM_CHECKPOINT`, `CYPHA_LM_CHECKPOINT`.

**Streaming chunk shape (SSE `data:` lines):** `{ "index", "token_id", "loss", "epistemic_var", "aleatoric_var", "active_experts", "dominant_expert", "routing_probs", "done" }`. Final line: `{ "done": true }`.

**Expert routing:** each `predict_next` / generation step runs the Cypha sequence pipeline (Izaac → CellAI SSM → expert routing → GRIA). `routing_probs` and `dominant_expert` expose per-token expert field behaviour.

See [`native/qt/README.md`](../../native/qt/README.md) and [`examples/README.md`](../../examples/README.md). (Historical Python `cypha_lm/` tree removed.)

## 4a. Branch A text routing REST

**Native `cypha_rest`:** Branch A surface when built with `cypha_core` Branch A router (`native/include/cypha/branch_a_router.hpp`). See [`docs/CYPHA_BRANCH_A_EMBEDDINGS.md`](../CYPHA_BRANCH_A_EMBEDDINGS.md). Classifier + sequence routes unchanged.

| Method | Path | Role |
|--------|------|------|
| GET | `/route/health` | `{ router_trained, router_summary, ollama_url, ollama_model, ollama_reachable, sequence_loaded }` (`lm_loaded` alias) |
| POST | `/route/text` | Body `{ "text": str, "epistemic_threshold"?: float }` → `{ label, confidence, epistemic_var, abstain, embedding_backend, action, latency_ms }`. **`action`**: `cypha_route` (in-domain) or `fallback_llm` (OOD abstain). **400** if `text` empty; **500** if router not loaded. |
| POST | `/route/generate` | Route then **Cypha sequence** (in-domain, when `--sequence-checkpoint` / `POST /sequence/load`; alias `--cyphalm-checkpoint`) or **Ollama** (`POST /api/generate` on `CYPHA_OLLAMA_URL`) on abstain. Body adds `max_tokens`, `ollama_model`, `ollama_system`, `cypha_strategy` / `cypha_temperature` (aliases `cypha_lm_strategy` / `cypha_lm_temperature`). Response `{ route, generation, latency_ms }` with `generation.provider` = `"cypha"` | `"ollama"` | `"none"`. |
| POST | `/route/save` | Persist router to checkpoint dir (**200** `{ saved, checkpoint, summary }`; **400** if not trained). |

**Checkpoint (native JSON):** `--branch-a-json path/to/branch_a_router.json` or env **`CYPHA_BRANCH_A_CHECKPOINT`** (base or `.json` path). JSON references sibling **`model_cypha`** (v3 `.cypha` Cypha state), **`mean`** / **`std`** arrays (standardisation), optional **`projection`** (hashing SVD row-major), optional **`f_field_json`**. Python pickle **`.npz`** checkpoints are **not** loaded natively — use committed native JSON + `.cypha` bundles under **`fixtures/`**.

**Embedding:** native uses deterministic **MurmurHash3** feature hashing (`hash_n_features` default **512**, `hash_n_components` default **128**, unigram+bigram tokens, L2 norm) + optional fixed projection matrix from the checkpoint JSON. No `sentence-transformers` in C++.

**Epistemic gate:** Shannon entropy of routing softmax (same contract as Python `BenchClassifier.predict` → `infer_full["entropy"]`). Abstain when `epistemic_var > epistemic_threshold` (default **0.5**, overridable per request or `CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD` at export).

**`/metrics`:** adds **`branch_a_router`** summary object and **`sequence_loaded`** (sequence checkpoint loaded; alias **`lm_loaded`**).

## 4c. Latent sample / retrieve REST (Cypha)

Owned by **`cypha::Cypha`**. Latent sampling when a `.cypha` model is loaded — **not** token generation. Samples are latent `h` (`space: "latent"`), not full feature rows.

**Native `cypha_rest`:** `generation.hpp` via `Cypha::sample` / `retrieve`.

| Method | Path | Role |
|--------|------|------|
| POST | `/sample` | Former `/dif/generate`. Body `{ "input": [float,...], "mode": "langevin" \| "from_observation" \| "retrieval_augmented", ... }`. **`200`** → `{ "mode", "label", "n_samples", "space": "latent", "samples": [...] }`. |
| POST | `/retrieve` | Former `/dif/retrieve`. Body `{ "input", "database", "top_k"? }`. |
| POST | `/sequence/load` | Former `/lm/load`. |
| GET | `/sequence/metrics` | Former `/lm/metrics`. |
| POST | `/predict_next` | Former `/lm/predict_next`. |

**Input dimension:** same as `POST /predict`. Health `model_type` is **`Cypha`**.

**Note:** `POST /generate` = token autoregression. `POST /sample` = latent sampling.

## 4b. Native sequence stack (inside `cypha_core`)

> **Product note:** Living entry is `cypha::Cypha` sequence APIs. Hybrid `--mode hybrid` / `hybrid_gria_lstm` profiles remain for **historical BPC benches** and lock validation; they are not the shipping sequence default (PGM->Wy).

Sequence sources live under `native/include/cypha/cyphalm/`, `native/src/cyphalm/`, compiled into **`cypha_core`** (compat alias target `cypha_lm_native` remains). Public facade: **`cypha::Cypha`**. Build notes: [`docs/native/CYPHALM_NATIVE_BUILD.md`](../native/CYPHALM_NATIVE_BUILD.md).

### Reference parity rule

**Historical:** Python `CyphaLM` (`cypha_lm/`) was the golden reference for numerics, BPC targets, and checkpoint layout; that tree is **removed**. Native fixtures under `fixtures/cyphalm_*/` and CTests remain the authoritative parity gate. Do not treat native BPC from `cyphalm_bench_native` as a product claim until checkpoint parity is recorded in the master tracker.

### `cyphalm_bench_native` — BPC bench CLI

Trains and evaluates `CyphaLMModel` on a corpus profile, prints **JSON** (BPC, mode, thread count, corpus source) to stdout. Falls back to a deterministic synthetic corpus when bench data files are unavailable.

```
usage: cyphalm_bench_native --mode MODE --profile PROFILE
       [--n-train N] [--n-eval M] [--threads T]
```

| Flag | Values | Default | Role |
|------|--------|---------|------|
| `--mode` | see table below | `hybrid` | Selects `ContextMode` + tier flags via `apply_bench_mode` |
| `--profile` | `d17`, `d04` | `d17` | Corpus profile (`d17` → vocab 256; `d04` → vocab 128) |
| `--n-train` | int | `40000` | Training tokens |
| `--n-eval` | int | `2000` | Eval tokens for BPC |
| `--threads` | int | `0` | OpenMP / worker count (`0` = hardware default) |

**Bench modes** (`--mode` → native `ContextMode` / flags):

| `--mode` | `ContextMode` | Tier flags enabled |
|----------|---------------|-------------------|
| `char_lstm` | `CharLstm` | Char LSTM head only |
| `ssm` | `SsmGria` | CellAI SSM → GRIA |
| `hybrid` | `Hybrid` | GRIA + CharLSTM log-prob blend |
| `ssm_gria` | `GriaNgram` | SSM field + n-gram embed history → GRIA |
| `context_bank` | `GriaNgram` | + `use_context_bank` (linear attention ring, K=512) |
| `spectral` | `SsmGria` | + `use_spectral_pde` (FFT circulant SSM step) |

Example (smoke, synthetic corpus):

```powershell
cmake --build C:\Temp\cypha_native_build --target cyphalm_bench_native
C:\Temp\cypha_native_build\cyphalm_bench_native.exe --mode hybrid --profile d17 --n-train 500 --n-eval 100 --threads 4
```

Example output keys: `mode`, `profile`, `context_mode`, `bpc`, `threads`, `corpus`, `synthetic`, `vocab_size`.

### Parity and bench binaries

| Binary | Role |
|--------|------|
| **`cyphalm_bench_native`** | BPC sweep CLI (above). |
| **`cyphalm_parity`** | Meta-runner: `cyphalm_ssm_parity`, `cyphalm_model_parity`, `cyphalm_hebbian_parity`, and `cyphalm_char_lstm_parity` when `fixtures/cyphalm_*/sidecar.json` exist. |
| **`cyphalm_char_lstm_parity`** | Numeric check vs `fixtures/cyphalm_char_lstm/sidecar.json`. |
| **`cyphalm_ssm_parity`** | One-step SSM vs Python-exported golden (`cyphalm_ssm_golden.inc`). |
| **`cyphalm_model_parity`** | 10-token forward/train scaffold for `CyphaLMModel` modes. |
| **`cyphalm_hebbian_parity`** | Encoder + sparse SSM + graph diffuse vs Python-derived goldens. |
| **`cyphalm_checkpoint_parity`** | Save/load roundtrip + Python checkpoint BPC lock (char_lstm + hybrid GRIA `W`). |

**CTest:** `native_cyphalm_char_lstm`, `native_cyphalm_ssm`, `native_cyphalm_model_parity`, `native_cyphalm_hebbian`, `native_cyphalm_checkpoint_parity`, `native_cyphalm_parity_suite`.

**ctest:** CTest **native_*** — skip if binaries missing; env **`CYPHALM_PARITY_BIN`**, **`CYPHALM_BENCH_NATIVE_BIN`**, **`CYPHALM_CHAR_LSTM_PARITY_BIN`**.

**Checkpoint format:** Cypha sequence checkpoint (`CyphaLMModel::save` / historical `CyphaLM.save()`) JSON + NPZ; native loads Python full-rank GRIA via **`load_from_full_w`**. Native roundtrip also persists **`dif`** (NIG expert states) and **`ssm`** (h/s layer states). Includes U06 PGM flags + `pgm_wy`/`pgm_by` (+ PGM cell state when present). Fixtures: committed under **`fixtures/cyphalm_checkpoint/`**; gate with **`ctest -R native_cyphalm`**.

**Native REST sequence:** `cypha_rest` exposes **`POST /sequence/load`**, **`GET /sequence/metrics`**, **`POST /predict_next`**, **`POST /generate`**, **`POST /generate/stream`** (see master tracker). Startup: **`--sequence-checkpoint`** or env **`CYPHA_SEQUENCE_CHECKPOINT`** (aliases: **`--cyphalm-checkpoint`**, **`CYPHALM_CHECKPOINT`**, **`CYPHA_LM_CHECKPOINT`**). `/health` reports **`sequence_loaded`** (alias **`lm_loaded`**).

## 5. Parity fixtures (machine-checked)

- Directory: `fixtures/`
- **`manifest.json`**: model geometry, seeds, label order.
- **`reference.cypha`**: binary state after a fixed training schedule. *(Includes **Tier-1** (`ctx_hist_packed`, co-occurrence, last label); native **`CyphaInferModel::from_root`** restores them for **`cypha_parity`** / **`cypha_rest`**.)*
- **`expected.npz`**: `x_input`, `llr`, `probs`, `gates`, `conf_batch`, `pred_idx`, `serial_conf`, plus metadata arrays.
- **`native_parity.bin`** (optional sidecar): `F_field` + the same `x_input` / LLR / probs / gates buffers for the C++ `cypha_parity` tool (`native/`). **Version 2** appends **`batch_infer_full`** per-row **entropy** and **confidence** (argmax prob × gate) so native checks the explanation subset without Python. **Version 1** remains accepted. Regenerated only when the frozen contract changes — see **MAINTENANCE.md**.
- **`train_step_vector/sidecar.json`**: one online `train_step` (same row as `expected.npz` `x_input[0]`) — expected loss for native `train_step_vector_parity` (CTest `native_train_step_vector`).
- **`mke_train_step/`** (`before.cypha`, `f_field.json`, `sidecar.json`): one scalar **`MKERegressor.train_step`** vs native **`mke_train_step_parity`** (CTest **`native_mke_train_step`**). Regenerate: update committed sidecar + **`ctest -R native_mke_train_step`** (see **MAINTENANCE.md**).
- **`mke_train_extended/`**: same layout; sidecar **`fixture_schema` ≥ 2** with **`steps`**, optional **`replay_warmup`** + **`replay_u01`** when **`replay_ratio > 0`** — sequential **`MKERegressor.train_step`** checks vs **`mke_train_step_parity`** (CTest **`native_mke_train_extended`**). Regenerate: update committed sidecar + **`ctest -R native_mke_train_step`** (see **MAINTENANCE.md**).
- **`regression_head.json`** (optional): expert `mu` / `var_ema` per class label — native `cypha_rest` `/predict` regression fields (`native/scripts/smoke_cypha_rest_mingw.ps1`).

**Staleness note (as of 2026-05-31):** fixtures in `fixtures/` were last committed 2026-03-31. `cypha_core` received updates on 2026-05-30 (commit `1dbfa13`) that added `KernelMemory` and deliberation. **These additions do not affect fixture numerics** when generated with `use_kernel_llr=False` (the default) and `deliberation_lo >= deliberation_hi` (the default) — both are off in all current fixture generators. However, you should regenerate fixtures after any further changes to `score_matrix`, `train_step_vector`, or world-prior update paths.

Regenerate after intentional numerical changes:

```bash
```

Then run:

```bash
ctest --test-dir native/build -R native_parity --output-on-failure
```

A future **native** runtime should load `reference.cypha` (or an exported copy), run the same pipeline on `x_input`, and compare to `expected.npz` within agreed tolerances.

## 5. Suggested port order

1. **Read `.cypha` v3** + materialise weights / buffers in native memory.  
2. **`score_matrix` + softmax + GH gate** (batched).  
3. **`batch_encode`** for `VectorEncoder` (GEMM), then RFF path.  
4. **Online training** (`train_step` / world update) only after inference parity passes.  
5. **Qt shell** against the same REST JSON or a thin IPC mirroring it.

---

## 6. Native bench contract (`cypha_bench_run`)

Native benchmark runner: **`native/tools/cypha_bench_run.cpp`** → binary **`cypha_bench_run`**. Domain IDs **d01–d17** plus cross-domain analyses.

### CLI

```
usage: cypha_bench_run [--domain N] [--from-domain N] [--report-only] [--list-domains]
```

| Flag | Role |
|------|------|
| `--list-domains` | Print `dXX` tag and Python module path; exit 0. |
| `--domain N` | Run only domain **N** (e.g. `--domain 1` → **d01**). |
| `--from-domain N` | Run domains **N…17** (inclusive). |
| `--report-only` | Skip domain runs; run cross-domain analyses, rebuild **`BASELINE_REPORT.md`**, **`report/summary.json`**, and native figure data JSON under **`bench/report/figures/`**. |

**Environment:** **`CYPHA_REPO_ROOT`** — repo root when cwd is not the checkout. **`CYPHA_BENCH_FAST=1`** — smaller subsamples (via **`bench_scale`**).

### JSON table schema

Each domain writes **`bench/report/tables/<domain_id>.json`**:

```json
{
  "domain": "d03",
  "timestamp": "2026-06-11T12:00:00.000+00:00",
  "experiments": { ... }
}
```

- **`domain`**: short id (**`d01`** … **`d17`**, or **`cross_*`** for cross-domain tables).
- **`timestamp`**: UTC ISO-8601 with ms.
- **`experiments`**: domain payload (tasks, datasets, scores, assertions, etc.).

Cross-domain tables use the same envelope; **`experiments`** holds analysis summaries.

### `backend` field

Every domain **`experiments`** object includes **`backend`** describing the engine that produced Cypha scores:

| Value | Meaning |
|-------|---------|
| **`cypha_core`** | Native **Cypha** training/inference (`create_fresh_model_root`, `dif_train_step_vector`, encoders, etc.). |
| **`cypha_lm_native`** | Native **Cypha** sequence BPC bench (**d04**, **d17** LM sections). |
| **`native_stub`** | Reserved for unimplemented domains (should not appear once the domain is ported). |

Per-task entries may also carry **`backend": "cypha_core"`** when nested under **`tasks`** / **`datasets`**.

### sklearn-equivalent baselines (native)

**`native/src/bench/bench_baselines.cpp`** provides offline baselines wired into domains that compare against sklearn in Python:

- **`logistic_regression`** — L2 one-vs-rest logistic regression ( **`LogisticRegression(max_iter=500, C=1.0)`** ).
- **`ridge`** — **`Ridge(alpha=1.0)`** via **`ridge_fit_bias`**.

Results appear under **`experiments.baselines`** (or per-task **`baselines`**) with the same score keys as Python (**`accuracy`**, **`f1_macro`**, **`rmse`**, **`mae`**, **`r2`**).

### Figure data (native, no matplotlib)

**`generate_figure_data()`** reads saved tables and writes **`figure_data_v1`** JSON (+ optional CSV) to **`bench/report/figures/`**, indexed by **`figures_manifest.json`**. Mirrors Python **`save_figure`** bar charts for **d01**, **d02**, **d03**, **d05** where table data exists. PNG rendering remains optional (Python matplotlib or external tools from CSV/JSON).

### Related binaries

| Binary | Role |
|--------|------|
| **`cypha_bench_run`** | Master domain runner + report/figure emit. |
| **`cypha_bench_report`** | Report-only rebuild (same report helpers). |

---

## 7. Related (full product port)

- **[`PORT_FULL_STACK.md`](PORT_FULL_STACK.md)** — replace Python core + Cypha GUI + REST + Qt (milestones, risks).  
- **[`PREPROCESSOR_CONTRACT.md`](PREPROCESSOR_CONTRACT.md)** — registry `preprocessor.json` beside `.cypha`.
- **Experiments SQLite (M6):** CTests **`native_experiment_db_smoke`** / **`native_experiment_db_file`**; CTest **`native_experiment_db_crud`** (CMake-generated DDL) — env **`CYPHA_EXPERIMENT_DB_SMOKE_BIN`**, **`CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN`** — **`native/README.md`**.

---

*Update this file when bumping `_CYPHA_VERSION` or changing public API shapes.*
