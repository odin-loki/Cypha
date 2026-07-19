# Multi-model REST — bounded slice (2026-07-17)

**Build:** `native/build_rest_multi` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Bill of Work §5 / FUTURE.md §5 — in-process multi-model map for `cypha_rest`. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight bench scripts.

## Shipped

| Component | API / flag | Notes |
|-----------|------------|-------|
| **In-memory map** | `g_models` keyed `name/version` | `LoadedModelBundle` per slot: model, mem, preprocessor, replay, regression/MKE sidecar, per-slot `std::mutex` |
| **Active model** | `g_active_model_key` + legacy globals | `POST /load` hot-swaps globals; omitting `"model"` on predict/update uses active slot (byte-compatible single-model path) |
| **Startup preload** | `--preload-registry` | Loads every registry bundle into RAM at boot (active model loaded via `--cypha` first) |
| **GET /models** | `loaded`, `active`, `active_model` | Full cards by default; `?summary=true` → `{name, version, loaded, active}` only |
| **POST /predict** | optional `"model": "<name>/<version>"` | `404` `model not loaded` for unknown key; `503` when no active model and field omitted |
| **POST /update** | optional `"model"` | Same routing as predict; per-slot train state (replay, GH χ/ψ, MKE) |
| **POST /load** | `{name, version?}`, `{model}`, or `{}` | Empty body → preload all; single load sets active + map entry |
| **GET /metrics** | `loaded_model_count`, `active_model` | Registry scan count unchanged as `registry_model_count` |

## API shape (native)

```text
cypha_rest --registry <root> [--preload-registry] --cypha <active.cypha> ...

GET  /models
     → { "models": [ { ...ModelCard, "loaded": bool, "active": bool }, ... ],
         "active_model": "name/version" | null }

POST /predict
     { "input": [...], "use_gh": true, "model"?: "name/version" }
     → PredictResponse (unchanged keys)

POST /update
     { "input": [...], "correct_label": "...", "model"?: "name/version" }
     → { "loss": float, "n_corrections": int }

POST /load
     { "name": "...", "version"?: "latest" } | { "model": "name/version" } | {}
```

Model selection uses the JSON body field (same style as other POST routes). Query-param / path-prefix routing deferred.

## Single-model compatibility

When one model is loaded and `"model"` is omitted (or equals `active_model`), requests hit the original `g_model` / `g_mem` globals under `g_mu` — response shapes and status codes match the pre-multi-model server.

## Follow-up (not in this slice)

- **LRU eviction** — cap RAM by evicting least-recently-used map entries (FUTURE.md §5 sketch).
- **Query/path model routing** — e.g. `POST /models/<name>/<version>/predict` for clients that cannot set JSON body fields on GET-adjacent tooling.
- **Per-model session metrics** — `g_predictions` / `g_sess` remain process-global (active-model session only).

## Tests

| CTest | Script / binary | Asserts |
|-------|-----------------|---------|
| `native_rest_multi_model` | `test_cypha_rest_multi_model.ps1` | Two registry models preloaded; `GET /models`; default + named `/predict`; named `/update` on second slot |
| `native_rest_schema_contract` | `rest_schema_contract` + PS1 wrapper | `/models` keys (`loaded`, `active`, `active_model`); `/load`; predict/update shapes |

Run (after build):

```powershell
ctest --test-dir native/build_rest_multi -R "native_rest_multi_model|native_rest_schema_contract" --output-on-failure
```

## Files touched

- `native/apps/cypha_rest.cpp` — `g_models`, `LoadedModelBundle`, `resolve_model_view`, route wiring
- `native/scripts/test_cypha_rest_multi_model.ps1` — dual-model smoke
- `native/tests/rest_schema_contract.cpp` — `/models` contract keys
- `docs/port/PORT_CONTRACT.md` — multi-model table rows + note
