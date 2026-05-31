# Examples

Request/response examples for the Cypha REST API (`cypha_rest` native or FastAPI).

All examples assume the server is running on `http://127.0.0.1:7749`. Start it with:

```bash
# FastAPI (Python)
uvicorn cypha_studio.server.api:app --host 127.0.0.1 --port 7749

# Native cypha_rest (after building native/)
./native/build/cypha_rest --port 7749 --registry /path/to/registry
```

> **Dimension note:** the 4-float `input` arrays in these examples only work if your loaded
> model has a latent dim of 4 (e.g. after a 4-feature preprocessor). Adjust to match your
> model's `feat_dim`. For `parity_fixtures/reference.cypha`, the latent dim differs — these
> examples are illustrative.

---

## Files

| File | Route | Method |
|------|-------|--------|
| `cypha_predict_body.json` | `/predict` | POST |
| `cypha_update_body.json` | `/update` | POST |
| `cypha_adapt_temperature_body.json` | `/adapt_temperature` | POST |
| `cypha_load_body.json` | `/load` | POST |
| `curl_predict.ps1` | `/predict` | one-liner curl (PowerShell) |
| `curl_predict.sh` | `/predict` | one-liner curl (bash) |
| `lm_generate_body.json` | `/generate`, `/generate/stream` | POST (CyphaLM) |
| `curl_lm_generate_stream.ps1` | `/generate/stream` | SSE streaming (PowerShell) |
| `curl_lm_generate_stream.sh` | `/generate/stream` | SSE streaming (bash) |

> **CyphaLM routes** (`/generate`, `/lm/*`) are **FastAPI-only**. Load a checkpoint first:
> `POST /lm/load` with `{"checkpoint_path": "path/to/ckpt"}` or set `CYPHA_LM_CHECKPOINT` before starting uvicorn.

---

## CyphaLM generation (FastAPI)

**Load checkpoint:**
```bash
curl -s -X POST http://127.0.0.1:7749/lm/load \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_path": "/path/to/my_ckpt"}'
```

**Batch generate (top-p nucleus sampling):**
```bash
curl -s -X POST http://127.0.0.1:7749/generate \
  -H "Content-Type: application/json" \
  -d @examples/lm_generate_body.json
```

**Stream tokens (SSE):**
```bash
bash examples/curl_lm_generate_stream.sh
```

Each SSE `data:` line includes `token_id`, `epistemic_var`, `dominant_expert`, and `routing_probs` from CyphaDIF.

---

## Quick reference

**Predict:**
```bash
curl -s -X POST http://127.0.0.1:7749/predict \
  -H "Content-Type: application/json" \
  -d @examples/cypha_predict_body.json
```

**Update (online training):**
```bash
curl -s -X POST http://127.0.0.1:7749/update \
  -H "Content-Type: application/json" \
  -d @examples/cypha_update_body.json
```

**Load a model from the registry:**
```bash
curl -s -X POST http://127.0.0.1:7749/load \
  -H "Content-Type: application/json" \
  -d @examples/cypha_load_body.json
```

**Health check:**
```bash
curl -s http://127.0.0.1:7749/health
```

For load-testing, see `scripts/loadtest_ab_predict_example.sh` / `.ps1` (uses Apache Bench).
