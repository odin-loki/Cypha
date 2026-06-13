#!/usr/bin/env bash
# Stream CyphaLM generation (SSE) from native cypha_rest.
# Requires CyphaLM loaded at startup: --cyphalm-checkpoint or CYPHALM_CHECKPOINT.
PORT="${CYPHA_REST_PORT:-8099}"
curl -N -X POST "http://127.0.0.1:${PORT}/generate/stream" \
  -H "Content-Type: application/json" \
  -d @"$(dirname "$0")/lm_generate_body.json"
