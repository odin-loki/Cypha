# Cypha → C++ / CUDA / Qt — frozen reference contracts

This document is the **normative checklist** for native ports. Behavior must match this Python reference unless you explicitly version and document a breaking change.

## 1. Binary state (`.cypha`)

- **Writers / readers**: `cypha_save_binary`, `cypha_load_binary`, **`cypha_save_binary_to_bytes`**, **`cypha_load_binary_from_bytes`** in `Cypha.py` (v3 bytes ↔ dict; same layout as native **`save_cypha_to_buffer`** / **`load_cypha_from_buffer`**).
- **Native (C++)**: **`save_cypha_file`** / **`save_cypha_to_buffer`** / **`load_cypha_file`** / **`load_cypha_from_buffer`** / **`clone_cnode`** (`native/include/cypha/load_cypha.hpp`) — same v3 on-disk layout as Python. **M1 / M2 / fixed kernels:** CTests **`native_parity`**, **`native_preprocessor`**, **`native_nig_adapt`**, **`native_regression_mixture`** have matching subprocess pytest under **`tests/test_*_native*.py`** (override env vars **`CYPHA_*_PARITY_BIN`** — see **`native/README.md`**). One-step latent **`DIFMemory.train`**: CTest **`native_memory_train`** + pytest **`tests/test_memory_train_native_parity.py`** (env **`CYPHA_MEMORY_TRAIN_PARITY_BIN`**). DIF-memory training state can be merged back into a loaded root via **`CyphaDifMemoryState::merge_state_into_root_for_save`**; CTest **`native_memory_train_roundtrip`** + pytest **`tests/test_memory_train_roundtrip_native.py`** (subprocess + **`cypha_load_binary`** cross-check). One-step **`dif_train_step_vector`** loss: CTest **`native_train_step_vector`** + pytest **`tests/test_train_step_vector_native_parity.py`** (env **`CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN`**).
- **Magic**: `CYPHA\x00` (6 bytes).
- **Version**: single byte; current **3**. After magic: `version (u8)`, `endian_sentinel (u32) = 0x01020304`, `n_fields (u32)`, then keyed entries.
- **Endianness**: **little-endian** for all multi-byte scalars; if sentinel ≠ `0x01020304`, the file requires byte-swapping (big-endian writer).
- **Tensor rule**: arrays serialized as **float64**, **C-contiguous row-major**; no stride metadata in the file.
- **NIG `field_W_T` / `field_a_eff`**: Python **`CyphaDIF.save_state()`** persists **`field_W_T`**, **`field_h`**, **`field_step`**, and **`field_a_eff`** (float64 tensor, same values as **`Field._A_eff`** fp32 matvec matrix). Native **`load_cypha_*` → `CyphaInferModel`** uses optional **`field_a_eff`** when shapes match **`field_W_T`**; otherwise recomputes via **`recompute_field_a_eff`** (`native/src/nig_field.cpp`). Qt **`patch_infer_training_snapshot`** and **`memory_train_roundtrip`** (after merge) emit **`field_a_eff`** when a causal field is present. Older v3 files without **`field_a_eff`** still load.
- **Recursive dicts**: dtype `DICT` nests key/value pairs; the tree is the same shape as `save_state()` / `load_state()`.
- **`world.F_field`** (optional in older files; **written by current `CyphaDIF.save_state`**): float64 tensor shape `(feat_dim, field_dim)` — field-conditioned shift for μ₀ (`WorldPrior.F_field`). When present, native loaders may omit external `f_field.json`. Older checkpoints without `F_field` still load via sidecar JSON.

Native loaders should accept **version 3** and reject unknown higher versions.

### Qt shell native save parity (`patch_infer_training_snapshot`)

After native training, **`patch_infer_training_snapshot`** in `native/qt/src/shell_main.cpp` writes all keys that Python `CyphaDIF._save_state()` produces:

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

`Cypha.py` uses `cypha_accel.score_batch.fused_score_llr` for the batched LLR core in `score_matrix` / `generate` (fuses `(H-μ₀)⊙inv_v`, `R @ D.T`, and MDL/context bias). Encoder batch projection uses `project_features`. **CuPy on GPU** when installed and a CUDA device is visible; otherwise **NumPy on CPU**. Numerics must match the reference formula in float64 (`tests/test_accel_cypha_wired.py`, `tests/test_accel_cross_gemm.py`). Native code may implement the same ops on any backend.

## 2. Core inference math (CyphaDIF)

Treat this as the **single spec** shared by `infer`, `batch_infer`, `batch_infer_full`, and the studio `InferenceEngine`.

1. **Latent**: `h = encoder.project(encoder_fn(x))` with `h` shape `(d,)`, `batch_encode` stacks to `(N, d)`.
2. **LLR**: `score_matrix(H, use_field)` — same μ₀ shift when `use_field=True` (field-conditioned prior). Column order = `memory._label_order`. Per-class MDL term uses **`world.v_mean / (n_obs_k + 1)`** (scalar mean of diagonal **variances** `v`), not `mean(inv_v)` — batch `score_matrix` must match `DIFMemory.classify`.
3. **Class probabilities**: `probs = softmax(LLR / (temperature + ε))` with ε = `1e-8` (`_EPS` in code). Batch path uses `_softmax_batch` (must match row-wise softmax of `infer`).
4. **World gate**: GH–NIG gate as in `DIFMemory.classify` / `world_gate_vector(..., gh_chi=1, gh_psi=1)` — **not** the legacy sigmoid-only path when GH is active.
5. **Confidence**: `conf_i = probs[i, argmax_i] * gate_i` (same as returned `(label, confidence)` from `infer` / `batch_infer`).

**Parity rule**: For the same loaded state, `temperature`, `use_field`, and inputs, `batch_infer` and `infer` must agree on **label and confidence** within floating tolerance (`tests/test_parity_fixtures.py`). The batch GH gate calls `_nig_R_eff` row-wise, matching `DIFMemory.classify`. Fixtures assume `use_kernel_llr=False` and deliberation disabled (`deliberation_lo >= deliberation_hi`).

### Optional Python-only inference modifiers (not in native M1–M6)

These features exist in `Cypha.py` but are **not implemented** in the native C++ / `cypha_rest` / Qt shell. Parity fixtures are generated with both disabled.

- **Kernel LLR** (`CyphaDIF(use_kernel_llr=True)`): blends Nyström RBF scores from `KernelMemory` with linear LLRs inside `infer` / `train_step`. `KernelMemory` maintains a reservoir of landmark points and computes kernel-lifted scores at inference time. Not serialised in `.cypha` v3; native `score_matrix_use_field` has no equivalent. Added in commit `1dbfa13` (2026-05-30). Priority-1 architectural upgrade — see [`docs/FUTURE.md §0a`](../FUTURE.md).

- **Deliberation** (`deliberation_lo`, `deliberation_hi`): when `lo < hi` and the inferred confidence falls in `[lo, hi]`, `infer` / `infer_full` return label `__unknown__` and halve the confidence. Default is **`deliberation_lo=1.0`, `deliberation_hi=0.0`** (disabled; `lo >= hi` → no abstention). Not implemented in native REST/Qt/C++. With defaults there is no numeric difference from native.

- **`gh_infer` vs native REST `use_gh`:** FastAPI `InferenceEngine` with `use_gh=True` calls `CyphaDIF.gh_infer` (NIG-adjusted temperature `T_adj`, `ood_sigma=inf` classify path). Native `cypha_rest` with `use_gh=True` applies `world_gate_vector_use_field` at χ=ψ=1 on the standard LLR+softmax path. Label/confidence may differ for the same model and input — treat as a known cross-runtime gap until implementations are aligned.

## 3. REST API (FastAPI)

Base: `cypha_studio.server.api.create_app`. Typical routes:

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | `{ status, model, uptime, n_predictions }` — `n_predictions` matches the engine counter (same as `/metrics` → `n_predictions` on native `cypha_rest` + FastAPI) |
| GET | `/ready` | **`200`** `{ "ready": true, "model_type": str }` when an engine is loaded; **`503`** `{ "ready": false, "reason": "no_model_loaded" }` when not (FastAPI + native `cypha_rest`) |
| GET | `/metrics` | `uptime_seconds`, `model_loaded`, `model_type`, `n_predictions`, `n_corrections`, `registry_model_count`, optional `gh_chi_session` / `gh_psi_session` when `CyphaDIF`, `session` or `null`, **`regression_head_loaded`** (bool — MoE sidecar active for `/predict`) |
| POST | `/predict` | Body: `{ "input": [float, ...], "use_gh": bool, "return_explanation": bool }`; **`503`** `{ "detail": "No model loaded" }` when no engine / native has no model |
| POST | `/update` | Body: `{ "input": [...], "correct_label": str, "use_gh": bool }`; success **`200`** → **`{ "loss": float, "n_corrections": int }`** only (no extra keys); **`503`** `{ "detail": "No model loaded" }` when no model. **Native `cypha_rest` only (optional):** when `regression_head.json` includes an **`mke`** block (see below), you may add **`regression_y`** (number) to run one scalar **`MKERegressor.train_step`**-style update; **`loss`** is then the router **`dif_train_step_vector`** loss. Optional **`router_train_label`** (string) overrides the router training label; optional **`replay_u01`** (array of numbers) fixes priority-replay uniforms (parity-style). Sending **`regression_y`** without an **`mke`** block → **`400`** `{"detail":"regression_y requires mke block in regression_head.json"}`. |
| POST | `/register` | Body `{ "name", "version", "model_cypha", "card_json", "preprocessor_json"?: str \| null, "overwrite"?: bool }` — absolute or relative **host** paths to existing files; copies into **`<registry_root>/<name>/<version>/`**. Success **`200`** → `{ "registered": true, "model_dir": "<path>" }`; **`503`** `{"detail":"No registry configured"}` when no registry (native without **`--registry`**, FastAPI with **`create_app(..., registry=None)`**); failure **`400`** `{"detail":"…"}` (missing sources, destination exists without **`overwrite`**, etc.). Native **`cypha_rest`** refreshes its in-process registry scan cache after success. **FastAPI** default **`uvicorn cypha_studio.server.api:app`** uses **`ModelRegistry(CYPHA_REGISTRY_ROOT)`** (see **`env_config.registry_root`**); same copy semantics when a registry is attached (no in-memory cache refresh beyond the next **`GET /models`** scan). CLI **`registry_register`**. |
| POST | `/adapt_temperature` | Body: `{ "calibration": [ { "input": [...], "correct_label": str }, ... ], "n_grid"?, "T_min"?, "T_max"?, "n_bins"? }` → `{ "temperature", "n_used" }` (ECE grid); **`503`** `{ "detail": "No model loaded" }` when no model |
| GET | `/models` | `{ "models": [ ModelCard dicts ] }` — empty registry → **`{ "models": [] }`** (full and summary); query **`summary=true`** (or `1`) → `{ "models": [ { "name", "version" }, ... ] }` (no full card parse on native) |
| POST | `/load` | Body: `{ "name": str, "version"?: str }` (`version` defaults to **`latest`**). FastAPI returns **`422`** if **`name`** is missing. On success: **`200`** `{ "loaded": <ModelCard dict> }` — **`loaded`** must expose **every** `ModelCard` field (stable key set for Qt/native); **`503`** `{ "detail": "No registry configured" }` if no registry (`cypha_rest` without **`--registry`**, or FastAPI **`create_app(..., registry=None)`** — the default **`api:app`** has a registry from **`CYPHA_REGISTRY_ROOT`**); **`404`** — native `{ "detail": "model not found" }`; FastAPI `{ "detail": "<exception message>" }` (typically a missing card path — not byte-identical to native) |
| GET | `/session` | `n_predictions`, `n_corrections`, `correction_accuracy`, `mean_confidence`, `mean_anomaly`, `n_ood_flagged`, `label_distribution`, `session_duration_s` (same keys in `/metrics` → `session` when a session exists). FastAPI: if `create_app(..., session=None)`, returns **200** with zeros / empty `label_distribution` (no `InferenceSession` attached); native `cypha_rest` always has an in-process session buffer when a model is loaded. |
| DELETE | `/session` | → `{ "cleared": true }` (**200** always on FastAPI); clears prediction history and session GH χ/ψ when an `InferenceSession` exists (native matches `InferenceSession.clear`). If `session=None` on `create_app`, FastAPI treats delete as a no-op but still returns **`cleared: true`**. |
| GET | `/session/rng` | `{ "state": [uint32 × 624], "pos": int }` — snapshot of session RNG for deterministic replay cross-runtime. |
| POST | `/session/rng` | Body `{ "seed": int }` or `{ "state": [...], "pos": int }` — re-seed or restore; returns GET shape. Tested in `tests/test_cypha_rest_smoke.py`. |
| GET | `/classes` | `{ "classes": { label: { "n_obs": float } } }`; **`503`** `{ "detail": "No model loaded" }` when no model |

**Malformed request body:** if the client sends **invalid JSON** on `POST /predict`, `/update`, or `/adapt_temperature`, native `cypha_rest` responds with **`400`** and `{"detail":"bad json"}`. The same applies to **`POST /load`** when a registry is configured (parse fails before lookup). **Note:** with **no** registry, native **`POST /load`** returns **`503`** before parsing the body, so a garbage body still yields **`{"detail":"No registry configured"}`** rather than **`bad json`**. FastAPI parses the body first and typically responds with **`422`** and a structured `detail` (validation / JSON decode) — not byte-identical to native.

**Input dimension:** after optional preprocessor transform, vector length must match model latent dim; otherwise **`POST /predict`**, **`/update`**, and **`/adapt_temperature`** (per calibration row) return **`400`** and `{"detail":"input dim mismatch after preprocessor"}` on both native `cypha_rest` and FastAPI (FastAPI maps encoder **`ValueError` / `TypeError`** when the message contains **`got length`** or **`shape`** + **`mismatch`**).

**Replay on `/update`:** native **`cypha_rest`** drives priority replay from an in-process **`std::mt19937`** session RNG by default. Optional **`replay_u01`** on **`POST /update`** is forwarded through **`TrainStepExtras`** for **classification** (`dif_train_step_vector` / GH) and for **MKE** (`mke_scalar_train_step` when **`regression_y`** + **`mke`**), mirroring parity harness fixed replay uniforms. When **`replay_u01`** is omitted, replay sampling uses the session RNG.

**FastAPI vs native on `/update`:** FastAPI **`UpdateRequest`** allows the same optional keys **`regression_y`**, **`router_train_label`**, **`replay_u01`** (defaults **null** / omitted) so OpenAPI matches native JSON shapes; if any of these is set to a non-null value, FastAPI responds **`501`** with a textual **`detail`** (MKERegressor-style and replay overrides are **`cypha_rest`** + **`mke`** only).

**On-disk registry (native tooling):** pre-built **`model.cypha`** + **`card.json`** can be installed under **`<root>/<name>/<version>/`** with **`native/registry_register`** (see **`native/README.md`**) or **`cypha::registry_register_bundle`** — same tree Python **`ModelRegistry`** scans. CTest **`native_registry_register`**; subprocess pytest **`tests/test_registry_register_native_parity.py`** (env **`CYPHA_REGISTRY_REGISTER_BIN`**).

**Predict response** (`PredictResponse`): `label`, `confidence`, `all_scores`, `anomaly_score`, `is_ood`, `regression_val`, `uncertainty`, optional `explanation`, `latency_ms`. When `return_explanation` is true, native **`cypha_rest`** and FastAPI (via **`InferenceEngine.explain`**) use the same top-level **`explanation`** key set in **`tests/test_cypha_rest_smoke.py`** (`test_cypha_rest_fastapi_json_shape_parity`); nested numeric parity for every leaf is not guaranteed if **`explain()`** gains extra fields beyond the native REST builder.

**Native `cypha_rest` — optional scalar regression head:** with `--regression-json regression_head.json` (or `regression_head.json` beside `model.cypha` on registry **`POST /load`**), JSON shape is (JSON Schema: [`schemas/regression_head.schema.json`](schemas/regression_head.schema.json)):

```json
{ "experts": { "<class_label>": { "mu": <float_or_[d]>, "var_ema": <float> }, ... } }
```

For each loaded class label (same strings as routing / `all_scores` keys), `mu` is the expert target EMA (scalar number or first element of an array for future vector targets). Native fills `regression_val` = Σ_k p_k·μ_k and `uncertainty` = √(Σ_k p_k·var_k) using the same softmax `p` as classification (`LLR / temperature` then `softmax_batch_like_python`). If the sidecar is absent, `regression_val` is JSON `null` and `uncertainty` is `0` (matches classification-only FastAPI). **FastAPI** loads the same file via **`CYPHA_REGRESSION_HEAD`** or `create_app(..., regression_head_path=...)` so `/predict` can match native numerically when the model and inputs are the same.

**Optional `mke` block (native `cypha_rest` — online scalar MKERegressor step):** same file may include **`mke`** with **`d_in`**, **`D_rff`** (must equal classifier latent **`d`**), **`rff_W_rowmajor`**, **`rff_b`**, per-label **`w`** (length **`D_rff`**) and **`P`** (length **`D_rff`²** row-major), **`temperature`**, **`forgetting_factor`**, optional **`pi_floor`** (default **0.02**), optional **`gh_scales`** (length **K**). With **`mke`**, **`/predict`** uses RFF(**`input`**) as routing features (after preprocessor; **`input`** length must be **`d_in`**) and sets **`regression_val`** = Σ_k p_k·(w_k·φ); **`uncertainty`** still uses expert **`var_ema`** mixture when **`experts`** lists **`var_ema`** per label. **`POST /update`** with **`regression_y`** runs **`mke_scalar_train_step`** (see **`native/include/cypha/mke_scalar_train_step.hpp`**).

Qt or C++ clients should treat these JSON shapes as **stable** for v1; add fields additively rather than renaming.

## 4. Parity fixtures (machine-checked)

- Directory: `parity_fixtures/`
- **`manifest.json`**: model geometry, seeds, label order.
- **`reference.cypha`**: binary state after a fixed training schedule. *(Includes **Tier-1** (`ctx_hist_packed`, co-occurrence, last label); native **`CyphaInferModel::from_root`** restores them for **`cypha_parity`** / **`cypha_rest`**.)*
- **`expected.npz`**: `x_input`, `llr`, `probs`, `gates`, `conf_batch`, `pred_idx`, `serial_conf`, plus metadata arrays.
- **`native_parity.bin`** (optional sidecar): `F_field` + the same `x_input` / LLR / probs / gates buffers for the C++ `cypha_parity` tool (`native/`). **Version 2** appends **`batch_infer_full`** per-row **entropy** and **confidence** (argmax prob × gate) so native checks the explanation subset without Python. **Version 1** remains accepted. Regenerated with `generate_parity_fixtures.py` whenever fixtures change.
- **`train_step_vector/sidecar.json`**: one online `train_step` (same row as `expected.npz` `x_input[0]`) — expected loss for native `train_step_vector_parity` (CTest `native_train_step_vector`).
- **`mke_train_step/`** (`before.cypha`, `f_field.json`, `sidecar.json`): one scalar **`MKERegressor.train_step`** vs native **`mke_train_step_parity`** (CTest **`native_mke_train_step`**). Regenerate: **`python scripts/generate_mke_train_step_fixture.py`** (also invoked from **`generate_parity_fixtures.py`**).
- **`mke_train_extended/`**: same layout; sidecar **`fixture_schema` ≥ 2** with **`steps`**, optional **`replay_warmup`** + **`replay_u01`** when **`replay_ratio > 0`** — sequential **`MKERegressor.train_step`** checks vs **`mke_train_step_parity`** (CTest **`native_mke_train_extended`**). Regenerate: **`python scripts/generate_mke_train_extended_fixture.py`** (also invoked from **`generate_parity_fixtures.py`**).
- **`regression_head.json`** (optional): expert `mu` / `var_ema` per class label — native `cypha_rest` `/predict` regression fields (`tests/test_cypha_rest_smoke.py`).

**Staleness note (as of 2026-05-31):** fixtures in `parity_fixtures/` were last committed 2026-03-31. `Cypha.py` received updates on 2026-05-30 (commit `1dbfa13`) that added `KernelMemory` and deliberation. **These additions do not affect fixture numerics** when generated with `use_kernel_llr=False` (the default) and `deliberation_lo >= deliberation_hi` (the default) — both are off in all current fixture generators. However, you should regenerate fixtures after any further changes to `score_matrix`, `train_step_vector`, or world-prior update paths.

Regenerate after intentional numerical changes:

```bash
python scripts/generate_parity_fixtures.py
```

Then run:

```bash
pytest tests/test_parity_fixtures.py -v
```

A future **native** runtime should load `reference.cypha` (or an exported copy), run the same pipeline on `x_input`, and compare to `expected.npz` within agreed tolerances.

## 5. Suggested port order

1. **Read `.cypha` v3** + materialise weights / buffers in native memory.  
2. **`score_matrix` + softmax + GH gate** (batched).  
3. **`batch_encode`** for `VectorEncoder` (GEMM), then RFF path.  
4. **Online training** (`train_step` / world update) only after inference parity passes.  
5. **Qt shell** against the same REST JSON or a thin IPC mirroring it.

---

## 6. Related (full product port)

- **[`PORT_FULL_STACK.md`](PORT_FULL_STACK.md)** — replace Python core + CyphaStudio + REST + Qt (milestones, risks).  
- **[`PREPROCESSOR_CONTRACT.md`](PREPROCESSOR_CONTRACT.md)** — registry `preprocessor.json` beside `.cypha`.
- **Experiments SQLite (M6):** CTests **`native_experiment_db_smoke`** / **`native_experiment_db_file`** + pytest **`tests/test_experiment_db_smoke_native_parity.py`**; CTest **`native_experiment_db_crud`** + pytest **`tests/test_experiment_db_crud_native_parity.py`** (DDL via **`tests/experiment_schema_ddl.py`** — no **`numpy`**); env **`CYPHA_EXPERIMENT_DB_SMOKE_BIN`**, **`CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN`** — **`native/README.md`**.

---

*Update this file when bumping `_CYPHA_VERSION` or changing public API shapes.*
