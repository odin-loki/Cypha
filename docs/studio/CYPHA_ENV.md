# Cypha environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CYPHA_REGISTRY_ROOT` | Model registry directory. Used by **`cypha_rest --registry`** for **`/models`**, **`/load`**, **`/register`**. | `~/.cypha/models` |
| `CYPHA_API_HOST` | REST bind address (`cypha_rest --host`) | `127.0.0.1` |
| `CYPHA_API_PORT` | REST port | `7749` |
| `CYPHA_CORS_ORIGINS` | Comma-separated allowed browser origins, or `*` for all | `*` |
| `CYPHA_CSV_CHUNK_ROWS` | Stream large CSV imports in chunks of this row count (unset = load whole file into memory first) | *(unset)* |
| `CYPHA_REGRESSION_HEAD` | Path to optional `regression_head.json` (same schema as native `cypha_rest --regression-json`) — MoE **`regression_val`** / **`uncertainty`** on **`POST /predict`** | *(unset)* |
| `CYPHA_SEQUENCE_CHECKPOINT` | Path to Cypha sequence checkpoint base (``.json`` + ``.npz``). Native **`cypha_rest`** loads the sequence model for **`/generate`**, **`/sequence/*`**, **`/predict_next`** when built with sequence support. Aliases: **`CYPHALM_CHECKPOINT`**, **`CYPHA_LM_CHECKPOINT`**. Health/metrics report **`sequence_loaded`** (alias **`lm_loaded`**). | *(unset)* |
| `CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD` | Epistemic variance gate for **`POST /route/text`** and **`POST /route/generate`** (abstain → Ollama) | `0.5` |
| `CYPHA_BRANCH_A_N_TRAIN` | 20 Newsgroups samples for lazy Branch A router training on first `/route/*` call | `1200` |
| `CYPHA_BRANCH_A_EMBED_BACKEND` | Text embedder: `auto`, `sentence_transformers`, or `hashing` | `auto` |
| `CYPHA_OLLAMA_URL` | Ollama base URL for OOD **`fallback_llm`** generation | `http://127.0.0.1:11434` |
| `CYPHA_OLLAMA_MODEL` | Ollama model tag (e.g. `mistral`, `llama3`) | `mistral` |
| `CYPHA_OLLAMA_TIMEOUT_S` | HTTP timeout for Ollama generate | `120` |
| `CYPHA_BRANCH_A_CHECKPOINT` | Branch A router checkpoint base (``.json`` + ``.npz``); loaded on first `/route/*` | `~/.cypha/branch_a_router` |
| `CYPHA_BRANCH_A_AUTO_SAVE` | When `1`, save checkpoint after router training | *(off)* |
| `CYPHA_REST_BIN` | *(Dev / CI only.)* Absolute path to a built `cypha_rest` executable. When set, REST smoke tests run subprocess checks instead of skipping. See `native/README.md` and `scripts/wsl_verify.sh` (`RUN_NATIVE=1`). | *(unset)* |
| `CYPHA_QT_STUB_BIN` | Override path for `cypha_qt_stub` (build with `-DCYPHA_BUILD_QT=ON` and Qt6). | *(unset)* |
| `CYPHA_PREPROCESSOR_PARITY_BIN` | Override for `preprocessor_parity` | *(unset)* |
| `CYPHA_PREPROCESSOR_FIT_PARITY_BIN` | Override for `preprocessor_fit_parity` | *(unset)* |
| `CYPHA_CSV_INGEST_PARITY_BIN` | Override for `csv_ingest_parity` | *(unset)* |
| `CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN` | Override for `dif_regressor_train_step_parity` | *(unset)* |
| `CYPHA_BATCH_LLR_PARITY_BIN` | Override for `batch_llr_parity` | *(unset)* |
| `CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN` | Override for `quantile_dif_train_parity` | *(unset)* |
| `CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN` | Override for `preprocess_train_classify_parity` | *(unset)* |
| `CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN` | Override for replay fixture harness | *(unset)* |
| `CYPHA_MKE_TRAIN_STEP_PARITY_BIN` | Override for `mke_train_step_parity` | *(unset)* |
| `CYPHA_REGRESSION_M4_PARITY_BIN` | Override for `regression_m4_parity` | *(unset)* |
| `CYPHA_REGRESSION_RFF_PARITY_BIN` | Override for `regression_rff_parity` | *(unset)* |
| `CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN` | Override for `regression_two_stage_pipeline_parity` | *(unset)* |
| `CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN` | Override for `regression_two_stage_ridge_fit_parity` | *(unset)* |

**CLI:** `cypha_qt_shell` overrides host/port for that run. If `--host` / `--port` are omitted, the environment defaults above apply.

**CORS:** For production behind a known web UI, set e.g. `CYPHA_CORS_ORIGINS=https://app.example.com`. Use `*` only on trusted networks.

**GUI:** Dataset **File → Import** and the Dataset panel **Load** button remember the last browse directory and **File → Recent Datasets** (stored under `QSettings` org `Cypha`, app `Cypha`).

## Health, readiness, metrics (REST)

| Route | Use |
|-------|-----|
| `GET /health` | **Liveness** — process is up; includes model class name (or `none`), uptime, `n_predictions`. Always **200** when the server responds. |
| `GET /ready` | **Readiness** — **200** only when an `InferenceEngine` is loaded; **503** with `{"ready": false, "reason": "no_model_loaded"}` otherwise. Point Kubernetes / load balancer readiness probes here only if you require a model before receiving traffic. |
| `GET /metrics` | **JSON snapshot** for dashboards or scripts: `uptime_seconds`, `model_loaded`, `model_type`, engine `n_predictions` / `n_corrections`, **`registry_model_count`** (pairs with `card.json` under **`CYPHA_REGISTRY_ROOT`** — updates after **`POST /register`**), optional `gh_chi_session` / `gh_psi_session`, `regression_head_loaded` (MoE sidecar), and a short `session` block (or `null`). Not Prometheus text format; scrape and convert if you use Prom. |
| `GET /models` | Full **`ModelCard`** JSON per registered version (reads each `card.json`). Use **`GET /models?summary=true`** for `{name, version}` only (directory scan; faster on large registries). |
| `POST /register` | Copy **`model.cypha`** + **`card.json`** (+ optional **`preprocessor.json`**) from host paths into **`<CYPHA_REGISTRY_ROOT>/<name>/<version>/`** (native **`cypha_rest`** / **`POST /register`** body). |
| `DELETE /session` | Clears in-memory session history (`InferenceSession.clear`); model weights unchanged. |

## Production: `cypha_rest` deployment

Native **`cypha_rest`** keeps the loaded model, session counters, and registry scan in **process memory**.

1. **Run one process per replica** — do not run multiple independent **`cypha_rest`** instances behind a load balancer without treating each as a separate model copy (session state is not shared).

2. **Scale out** by running **multiple single-process instances** behind a load balancer only when clients tolerate independent session state.

3. **Concurrency**: predict/update paths are synchronous C++ work; for high concurrency, front with a proxy queue or dedicated inference worker.

4. **TLS**: Terminate HTTPS at **nginx**, **Caddy**, or a cloud LB; bind **`cypha_rest`** to `127.0.0.1` and forward to `CYPHA_API_PORT`.

5. **Shutdown**: **`cypha_rest`** handles **SIGINT/SIGTERM** for graceful stop.

**Example (single worker, all interfaces, env-driven port):**

```bash
export CYPHA_API_HOST=0.0.0.0
export CYPHA_API_PORT=7749
export CYPHA_CORS_ORIGINS=https://studio.example.com
# Optional: same regression sidecar as native cypha_rest (see PORT_CONTRACT §3)
# export CYPHA_REGRESSION_HEAD=/path/to/regression_head.json
```

**Headless CLI:** `cypha_rest --host … --port …` overrides env defaults for that process.

See also [`CYPHA_STUDIO_MASTER_PLAN.md`](CYPHA_STUDIO_MASTER_PLAN.md) and the [documentation hub](../README.md).
