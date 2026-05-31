# CyphaStudio

PySide6 desktop application and FastAPI REST server for CyphaDIF. Provides a full GUI workflow (training, inference, registry, experiments) alongside a headless REST API that matches the native `cypha_rest` C++ server.

---

## Layout

| Path | Role |
|------|------|
| `main.py` | Entry point — launches the PySide6 `MainWindow` |
| `env_config.py` | `CYPHA_*` environment variable parsing |
| `core/` | Dataset, trainer, experiment, registry, inference engine |
| `server/` | FastAPI app (`create_app`), models, routes |
| `gui/` | PySide6 widgets: main window, train/chat/dataset/registry/inference tabs |
| `requirements.txt` | Runtime deps (PySide6, pyqtgraph, fastapi, uvicorn, httpx, pydantic) |
| `test_cypha_studio.py` | 48 integration checks (dataset → trainer → registry → inference) |
| `demo.py` | Headless demo: train a toy CyphaDIF and serve one predict request |

---

## Install

From the repository root:

```bash
pip install -r cypha_studio/requirements.txt
```

Or with the `studio` optional extra from `pyproject.toml`:

```bash
pip install -e ".[studio]"
```

---

## Run

**Desktop GUI:**

```bash
python cypha_studio/main.py
```

**Headless REST server:**

```bash
uvicorn cypha_studio.server.api:app --host 127.0.0.1 --port 7749
```

Or via `make`:

```bash
make studio           # starts uvicorn on port 7749
```

**Environment variables** (see `env_config.py`):

| Variable | Default | Effect |
|----------|---------|--------|
| `CYPHA_REGISTRY_ROOT` | `./registry` | Where model card files live |
| `CYPHA_API_HOST` | `127.0.0.1` | REST bind address |
| `CYPHA_API_PORT` | `7749` | REST port |
| `CYPHA_REGRESSION_HEAD` | *(unset)* | Path to regression sidecar JSON |
| `CYPHA_CORS_ORIGINS` | `*` | Allowed CORS origins |

Full list: [`docs/studio/CYPHA_ENV.md`](../docs/studio/CYPHA_ENV.md).

---

## Test

```bash
python cypha_studio/test_cypha_studio.py      # 48 pipeline checks (headless)
pytest tests/test_gui_smoke.py -v             # Qt offscreen GUI smoke
pytest tests/test_gui_qtbot.py -v             # pytest-qt click tests
```

In CI: `QT_QPA_PLATFORM=offscreen` is set automatically — no display required.

---

## REST API surface

Matches the native `cypha_rest` C++ server. Key routes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + uptime |
| GET | `/ready` | 200 when model loaded, 503 otherwise |
| GET | `/metrics` | Prediction counts, session stats |
| POST | `/predict` | Classify an input vector |
| POST | `/update` | Online training step |
| POST | `/adapt_temperature` | ECE-grid temperature calibration |
| POST | `/load` | Load model from registry |
| POST | `/register` | Register a model into the registry |
| GET | `/models` | List registry contents |
| GET/DELETE | `/session` | Session stats / clear |
| GET/POST | `/session/rng` | RNG snapshot / restore |
| GET | `/classes` | Class observation counts |

### CyphaLM language model (FastAPI only)

Requires `pip install -e cypha_lm/` and a loaded checkpoint (`POST /lm/load` or `CYPHA_LM_CHECKPOINT`).

| Method | Path | Description |
|--------|------|-------------|
| POST | `/lm/load` | Load CyphaLM from checkpoint |
| GET | `/lm/metrics` | LM stats (experts, vocab, generation counts) |
| POST | `/lm/predict_next` | Single token + CyphaDIF routing probs |
| POST | `/generate` | Batch or SSE streaming generation |
| POST | `/generate/stream` | SSE streaming (one JSON chunk per token) |

Sampling strategies: `greedy`, `temperature`, `top_k`, `top_p` (nucleus), `uncertainty_gated`.

Full contract: [`docs/port/PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4.

---

## Architecture

```
main.py
  └── MainWindow (gui/main_window.py)
        ├── TrainTab       → core/trainer.py → CyphaDIF (Cypha.py)
        ├── ChatTab        → core/inference_engine.py
        ├── DatasetTab     → core/dataset.py
        ├── RegistryTab    → core/registry.py (ModelRegistry)
        └── ExperimentTab  → core/experiment.py (ExperimentDB → SQLite)

server/api.py (FastAPI)
  └── create_app(engine, registry, session, lm_engine, regression_head_path)
        ├── /predict       → InferenceEngine.predict (CyphaDIF)
        ├── /update        → InferenceEngine.train_step
        ├── /generate      → LMEngine → CyphaLM (Izaac→SSM→CyphaDIF→GRIA)
        └── /lm/*          → CyphaLM load / predict_next / metrics
```

Threading: all GUI ↔ core communication goes through `QThread` workers and a `SignalBus`. See [`docs/studio/STUDIO_THREADING.md`](../docs/studio/STUDIO_THREADING.md).
