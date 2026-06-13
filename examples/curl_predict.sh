#!/usr/bin/env bash
# One-liner curl against the /predict endpoint using cypha_predict_body.json
# Default port matches native cypha_rest (8099). Set CYPHA_REST_PORT to override.
PORT="${CYPHA_REST_PORT:-8099}"
curl -s -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d @"$(dirname "$0")/cypha_predict_body.json" \
  | (command -v jq >/dev/null && jq . || cat)
