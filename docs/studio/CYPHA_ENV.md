# CyphaStudio environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CYPHA_REGISTRY_ROOT` | Model registry directory (`ModelRegistry` root). Used by the **default** FastAPI `app` (`uvicorn cypha_studio.server.api:app`) for **`/models`**, **`/load`**, **`/register`**. | `~/.cypha/models` |
| `CYPHA_API_HOST` | REST bind address (headless / `main.py --headless`) | `127.0.0.1` |
| `CYPHA_API_PORT` | REST port | `7749` |
| `CYPHA_CORS_ORIGINS` | Comma-separated allowed browser origins, or `*` for all | `*` |
| `CYPHA_CSV_CHUNK_ROWS` | Stream large CSV imports in chunks of this row count (unset = load whole file into memory first) | *(unset)* |
| `CYPHA_REGRESSION_HEAD` | Path to optional `regression_head.json` (same schema as native `cypha_rest --regression-json`) — FastAPI `POST /predict` fills `regression_val` / `uncertainty` for **classification** models via softmax-routed scalar MoE | *(unset)* |
| `CYPHA_LM_CHECKPOINT` | Path to CyphaLM checkpoint base (``.json`` + ``.npz``). FastAPI-only: auto-loads CyphaLM for **`/generate`**, **`/lm/*`** routes at startup. Requires `pip install -e cypha_lm/`. | *(unset)* |
| `CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD` | Epistemic variance gate for **`POST /route/text`** and **`POST /route/generate`** (abstain → Ollama) | `0.5` |
| `CYPHA_BRANCH_A_N_TRAIN` | 20 Newsgroups samples for lazy Branch A router training on first `/route/*` call | `1200` |
| `CYPHA_BRANCH_A_EMBED_BACKEND` | Text embedder: `auto`, `sentence_transformers`, or `hashing` | `auto` |
| `CYPHA_OLLAMA_URL` | Ollama base URL for OOD **`fallback_llm`** generation | `http://127.0.0.1:11434` |
| `CYPHA_OLLAMA_MODEL` | Ollama model tag (e.g. `mistral`, `llama3`) | `mistral` |
| `CYPHA_OLLAMA_TIMEOUT_S` | HTTP timeout for Ollama generate | `120` |
| `CYPHA_BRANCH_A_CHECKPOINT` | Branch A router checkpoint base (``.json`` + ``.npz``); loaded on first `/route/*` | `~/.cypha/branch_a_router` |
| `CYPHA_BRANCH_A_AUTO_SAVE` | When `1`, save checkpoint after router training | *(off)* |
| `CYPHA_REST_BIN` | *(Dev / CI only.)* Absolute path to a built `cypha_rest` executable (Linux ELF or Windows `.exe`). When set, `pytest tests/test_cypha_rest_smoke.py` runs subprocess REST checks instead of skipping. See `native/README.md` and `scripts/wsl_verify.sh` (`RUN_NATIVE=1`). | *(unset)* |
| `CYPHA_QT_STUB_BIN` | Override path for `cypha_qt_stub` (pytest `tests/test_qt_stub_native.py`; build with `-DCYPHA_BUILD_QT=ON` and Qt6). | *(unset)* |
| `CYPHA_PREPROCESSOR_PARITY_BIN` | Override for `preprocessor_parity` (pytest `test_preprocessor_native_parity`) | *(unset)* |
| `CYPHA_PREPROCESSOR_FIT_PARITY_BIN` | Override for `preprocessor_fit_parity` (pytest `test_preprocessor_fit_native_parity`) | *(unset)* |
| `CYPHA_CSV_INGEST_PARITY_BIN` | Override for `csv_ingest_parity` (pytest `test_csv_ingest_native_parity`) | *(unset)* |
| `CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN` | Override for `dif_regressor_train_step_parity` (pytest `test_dif_regressor_train_step_native_parity`) | *(unset)* |
| `CYPHA_BATCH_LLR_PARITY_BIN` | Override path for native `batch_llr_parity` (pytest `test_batch_llr_native_parity`) | *(unset)* |
| `CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN` | Override path for native `quantile_dif_train_parity` (pytest `test_quantile_dif_train_native_parity`) | *(unset)* |
| `CYPHA_STUDIO_TRAINER_CLASSIFY_HOTPATH_BIN` | Override path for the same binary when testing `parity_fixtures/studio_trainer_classify_hotpath/` (pytest `test_studio_trainer_classify_hotpath_native_parity`) | *(unset)* |
| `CYPHA_STUDIO_TRAINER_GH_CLASSIFY_HOTPATH_BIN` | Override for `parity_fixtures/studio_trainer_gh_classify_hotpath/` (pytest `test_studio_trainer_gh_classify_hotpath_native_parity`) | *(unset)* |
| `CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN` | Override for `preprocess_train_classify_parity` (pytest preprocess + preprocess+GH hotpath modules) | *(unset)* |
| `CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN` | Override path for the same binary when testing `parity_fixtures/dif_train_replay/` (pytest `test_dif_train_replay_native_parity`) | *(unset)* |
| `CYPHA_MKE_TRAIN_STEP_PARITY_BIN` | Override for `mke_train_step_parity` (pytest `test_mke_train_step_native_parity`, including `mke_train_extended`) | *(unset)* |
| `CYPHA_REGRESSION_M4_PARITY_BIN` | Override for `regression_m4_parity` | *(unset)* |
| `CYPHA_REGRESSION_RFF_PARITY_BIN` | Override for `regression_rff_parity` | *(unset)* |
| `CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN` | Override for `regression_two_stage_pipeline_parity` | *(unset)* |
| `CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN` | Override for `regression_two_stage_ridge_fit_parity` | *(unset)* |

**CLI:** `python cypha_studio/main.py --headless --host 0.0.0.0 --port 8800` overrides host/port for that run. If `--host` / `--port` are omitted, the environment defaults above apply.

**CORS:** For production behind a known web UI, set e.g. `CYPHA_CORS_ORIGINS=https://app.example.com`. Use `*` only on trusted networks.

**GUI:** Dataset **File → Import** and the Dataset panel **Load** button remember the last browse directory and **File → Recent Datasets** (stored under `QSettings` org `Cypha`, app `CyphaStudio`).

## Health, readiness, metrics (REST)

| Route | Use |
|-------|-----|
| `GET /health` | **Liveness** — process is up; includes model class name (or `none`), uptime, `n_predictions`. Always **200** when the server responds. |
| `GET /ready` | **Readiness** — **200** only when an `InferenceEngine` is loaded; **503** with `{"ready": false, "reason": "no_model_loaded"}` otherwise. Point Kubernetes / load balancer readiness probes here only if you require a model before receiving traffic. |
| `GET /metrics` | **JSON snapshot** for dashboards or scripts: `uptime_seconds`, `model_loaded`, `model_type`, engine `n_predictions` / `n_corrections`, **`registry_model_count`** (pairs with `card.json` under **`CYPHA_REGISTRY_ROOT`** — updates after **`POST /register`**), optional `gh_chi_session` / `gh_psi_session`, `regression_head_loaded` (MoE sidecar), and a short `session` block (or `null`). Not Prometheus text format; scrape and convert if you use Prom. |
| `GET /models` | Full **`ModelCard`** JSON per registered version (reads each `card.json`). Use **`GET /models?summary=true`** for `{name, version}` only (directory scan; faster on large registries). |
| `POST /register` | Copy **`model.cypha`** + **`card.json`** (+ optional **`preprocessor.json`**) from host paths into **`<CYPHA_REGISTRY_ROOT>/<name>/<version>/`** (same JSON body as native **`cypha_rest`**). **`503`** only if the app was built with **`registry=None`**; the default **`uvicorn cypha_studio.server.api:app`** app uses **`ModelRegistry(registry_root())`**. |
| `DELETE /session` | Clears in-memory session history (`InferenceSession.clear`); model weights unchanged. |

## Production: uvicorn and workers

The reference server keeps **`InferenceEngine`**, **`InferenceSession`**, and **`ModelRegistry`** in **process memory**. The FastAPI `app` object is built once and holds that state on `app.state`.

1. **Use a single worker** (`uvicorn` default is **one** process). Do **not** run `uvicorn --workers 4` against this app unless you accept that **each worker is a separate copy** of the model and session: `/load` and `/predict` on different workers see **different** memory, `/metrics` differs per PID, and RAM use **multiplies** by worker count.

2. **Scale out** by running **multiple single-worker instances** behind a load balancer and treating them as **independent** replicas (sticky sessions not enough for shared mutable session state). For true shared state you need an external store or a single inference worker design.

3. **Threads vs asyncio**: Predict/update paths run **sync** Python and NumPy/Cypha work on the event loop thread. For high concurrency, front the API with a proxy queue or run inference in a **thread pool** / dedicated worker process (future hardening); the current stack is aimed at **moderate** concurrent `POST /predict` from a few clients.

4. **TLS**: Terminate HTTPS at **nginx**, **Caddy**, or a cloud LB; bind uvicorn to `127.0.0.1` on the node and forward to `CYPHA_API_PORT`.

5. **Shutdown**: uvicorn handles **SIGINT/SIGTERM** for graceful stop; in-flight requests may still be cut on hard kill.

**Example (single worker, all interfaces, env-driven port):**

```bash
export CYPHA_API_HOST=0.0.0.0
export CYPHA_API_PORT=7749
export CYPHA_CORS_ORIGINS=https://studio.example.com
# Optional: same regression sidecar as native cypha_rest (see PORT_CONTRACT §3)
# export CYPHA_REGRESSION_HEAD=/path/to/regression_head.json
python -m uvicorn cypha_studio.server.api:app --host "$CYPHA_API_HOST" --port "$CYPHA_API_PORT"
```

**Headless CLI:** `python cypha_studio/main.py --headless --regression-head /path/to/regression_head.json` overrides `CYPHA_REGRESSION_HEAD` for that process.

Embedding via `create_app()` + `start_server()` from `cypha_studio/main.py --headless` is equivalent to one process, one engine.

See also [`CYPHA_STUDIO_MASTER_PLAN.md`](CYPHA_STUDIO_MASTER_PLAN.md), [`env_config.py`](../cypha_studio/env_config.py), and the [documentation hub](../README.md).
