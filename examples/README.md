# Examples

Request/response examples for the native Cypha REST API (`cypha_rest`).

Default listen port is **8099** (override with `CYPHA_REST_PORT` in the curl helpers).

```bash
./native/build/cypha_rest --listen 127.0.0.1:8099 \
  --cypha fixtures/reference.cypha \
  --f-field-json fixtures/f_field.json
```

> **Dimension note:** the 4-float `input` arrays in these examples only work if your loaded
> model has a latent dim of 4 (e.g. after a 4-feature preprocessor). Adjust to match your
> model's `feat_dim`. For `fixtures/reference.cypha`, the latent dim differs — these
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
| `demo_cyphalm/` | CyphaLM demo checkpoint | see `demo_cyphalm/README.md` |

Load CyphaLM at startup with `--cyphalm-checkpoint <base>` or env `CYPHALM_CHECKPOINT` (`.json` + `.npz`).

---

## CyphaLM generation

**Batch generate (top-p nucleus sampling):**
```bash
curl -s -X POST http://127.0.0.1:8099/generate \
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
curl -s -X POST http://127.0.0.1:8099/predict \
  -H "Content-Type: application/json" \
  -d @examples/cypha_predict_body.json
```

**Update (online training):**
```bash
curl -s -X POST http://127.0.0.1:8099/update \
  -H "Content-Type: application/json" \
  -d @examples/cypha_update_body.json
```

**Load a model from the registry:**
```bash
curl -s -X POST http://127.0.0.1:8099/load \
  -H "Content-Type: application/json" \
  -d @examples/cypha_load_body.json
```

**Health check:**
```bash
curl -s http://127.0.0.1:8099/health
```

For load-testing, see `scripts/loadtest_ab_predict_example.sh` / `.ps1` (uses Apache Bench).
