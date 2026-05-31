#!/usr/bin/env bash
# Stream CyphaLM generation (SSE) — FastAPI only.
# Requires: CyphaLM loaded via POST /lm/load
curl -N -X POST http://127.0.0.1:7749/generate/stream \
  -H "Content-Type: application/json" \
  -d @examples/lm_generate_body.json
