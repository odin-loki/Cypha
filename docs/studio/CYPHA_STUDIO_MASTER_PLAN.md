# Cypha Studio — master plan (profile, test, product)

Single checklist to drive **profiling**, **automation**, **UX**, and **ops**. Work phases in order unless you have a blocking need. Check boxes in Git/PRs as you finish chunks.

---

## Phase 1 — Profiling (close the gaps)

**Done in repo**

- [x] GUI cold start — `scripts/profile_gui_startup.py`
- [x] Core engine on sklearn tabular — `scripts/profile_real_datasets.py`
- [x] E2E download + class/reg/gen + cProfile — `scripts/download_profile_e2e.py`
- [x] GPU micro/full bench — `scripts/gpu_microbench.py`, `scripts/gpu_fullbench.py`
- [x] **Studio GUI hot paths** — `scripts/profile_studio_hotpaths.py` (modes below)

**Modes for `profile_studio_hotpaths.py`**

| Mode | What it stresses |
|------|------------------|
| `training` | `TrainingWidget` + `SignalBus.training_step` × N (pyqtgraph refresh **time-throttled**, ~12 Hz) |
| `chat` | `ChatWidget` + tiny `CyphaDIF`, send × N (inference + bubbles + bus) |
| `dataset` | `DatasetWidget.load_file` on a temp CSV (parse + stats + tables) |
| `registry` | `ModelRegistry.list_models` repeated (disk scan of `~/.cypha/models`) |
| `api` | FastAPI `TestClient` POST `/predict` × N with a loaded engine |

**Still optional / later** (commands / notes: [`OPTIONAL_MEMORY_AND_LOAD.md`](OPTIONAL_MEMORY_AND_LOAD.md))

- [x] Memory profile starter — `scripts/profile_studio_memory.py` (tracemalloc diff); memray / long CSV still manual
- [x] Threading notes — [`STUDIO_THREADING.md`](STUDIO_THREADING.md) (`QThread` + `SignalBus` only)
- [x] **`ab` example** — [`examples/cypha_predict_body.json`](../examples/cypha_predict_body.json) + [`scripts/loadtest_ab_predict_example.sh`](../scripts/loadtest_ab_predict_example.sh) / `.ps1`; Locust still custom ([`OPTIONAL_MEMORY_AND_LOAD.md`](OPTIONAL_MEMORY_AND_LOAD.md))

**Commands**

`requirements-verify.txt` includes **httpx** (needed for FastAPI `TestClient` in `api` mode). One file with the same pins (plus **pytest-qt**): **`pip install -r requirements-pip-merged.txt`** from repo root — useful if **`pip install -r`** fails on encoding; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

```bash
pip install -r requirements-verify.txt
pip install -r cypha_studio/requirements.txt
pip install pytest-qt   # tests/test_gui_qtbot.py only; not listed in requirements-verify.txt

python scripts/profile_gui_startup.py -o artifacts/profiles/gui_startup_cprofile.txt
python scripts/profile_studio_hotpaths.py training --steps 3000 -o artifacts/profiles/profile_gui_training.txt
python scripts/profile_studio_hotpaths.py chat --rounds 200 -o artifacts/profiles/profile_gui_chat.txt
python scripts/profile_studio_hotpaths.py dataset -o artifacts/profiles/profile_gui_dataset.txt
python scripts/profile_studio_hotpaths.py registry --iterations 50 -o artifacts/profiles/profile_registry.txt
python scripts/profile_studio_hotpaths.py api --predicts 400 -o artifacts/profiles/profile_api_predict.txt
python scripts/profile_studio_memory.py --steps 800
```

---

## Phase 2 — Automated tests

- [x] GUI smoke (`tests/test_gui_smoke.py`)
- [x] GUI qtbot (`tests/test_gui_qtbot.py`)
- [x] **Dataset:** `tests/test_studio_data_registry.py` (`CSVDataset.from_file`); `tests/test_gui_training_dataset.py` (`DatasetWidget.load_file` + `AppState._train_ds`)
- [x] **TrainingWidget:** `tests/test_gui_training_dataset.py` — N × `training_step` + deque length / summary text
- [x] **API:** `test_api_contract.py` — `/predict`, `/update`, **`/adapt_temperature`**, `/health`, `/session`+`DELETE`, `/classes`, `/models` + `summary=true`
- [x] **Registry:** `tests/test_studio_data_registry.py` — temp dir `register` → `list_models` → `load` + `save_state` ndarray parity (no GUI)
- [x] Wrap `cypha_studio/test_cypha_studio.py` in pytest (`tests/test_cypha_studio_runner.py`, `@pytest.mark.slow`) + trainer/registry wiring test

---

## Phase 3 — UX / product

- [x] **Empty-state chat tip** — rich-text hint + shortcuts; hidden when a model loads (`chat_widget.py`).
- [x] Persist **geometry** + **dock/toolbar layout** (`QSettings`: `geometry` then `restoreState`; docks/toolbar have `objectName`; `tests/test_gui_window_settings.py` asserts keys on close). Manual **Reset Layout** still available.
- [x] **Recent datasets + last browse dir** — `gui/path_history.py`, **File → Recent Datasets**, `SignalBus.dataset_opened`, `QFileDialog` start directory.
- [x] **Keyboard shortcuts** — **Ctrl+Enter** send chat, **Ctrl+L** focus chat input (`MainWindow._build_shortcuts`; qtbot exercises focus handler).
- [x] In-app **Studio Log** dock (`LogDockWidget`, View → Studio Log)
- [x] **Export test predictions (CSV)** — File menu (classification + regression columns)
- [x] Richer **registry compare** table (task, type, val R², train N, optional **test** metrics when a test split exists) + **Confusion matrix** dialog (Model menu, classification)

---

## Phase 4 — API / ops / security

- [x] **Env defaults** — `CYPHA_REGISTRY_ROOT`, `CYPHA_API_HOST`, `CYPHA_API_PORT`, `CYPHA_CORS_ORIGINS` ([`CYPHA_ENV.md`](CYPHA_ENV.md), `cypha_studio/env_config.py`). Headless `main.py` + `ModelRegistry` + `create_app` CORS use these.
- [ ] If exposed beyond localhost: auth, rate limits (CORS is configurable; `*` remains the default for dev)
- [x] **Health / readiness / metrics** — `GET /health` (liveness), `GET /ready` (model required), `GET /metrics` (JSON snapshot). See [`CYPHA_ENV.md`](CYPHA_ENV.md).
- [x] **Production uvicorn notes** — single-worker rationale, scaling caveats, TLS, shutdown ([`CYPHA_ENV.md`](CYPHA_ENV.md) § *Production: uvicorn and workers*).

---

## Phase 5 — Performance hardening (after profiles)

- [x] Throttle pyqtgraph training curves (~12.5 Hz via `QElapsedTimer`); val series bounded by `MAX_POINTS` deques
- [x] Plot downsample for pyqtgraph — `TrainingWidget._compress_xy` (cap **PLOT_DISPLAY_MAX**); spot-check: `python scripts/profile_studio_hotpaths.py training --steps 50000`
- [x] Lazy / paged UI — **Experiments** “Load more runs…” (`list_runs` **offset**); toolbar + Load dialog use **directory listing** (`list_model_names` / `registered_versions`) instead of parsing every `card.json`
- [x] Chunked CSV load — `CSVDataset.from_file(..., read_chunk_rows=…)` + env `CYPHA_CSV_CHUNK_ROWS` (GUI import respects env)

---

## How we “work through” this

1. Keep this file as the **source of truth** for scope.
2. Each PR: complete a **small vertical slice** (e.g. one profile mode + doc line, or one new test file).
3. Refresh **top cumtime** lines in [`VERIFICATION_STATUS.md`](../verify/VERIFICATION_STATUS.md) when a new profile artifact is worth citing.

When Phase 1–2 are solid, Phase 3–5 are mostly product prioritisation — reorder to match your release goals.
