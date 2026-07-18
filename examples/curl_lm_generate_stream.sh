#!/usr/bin/env bash
# Stream Cypha sequence generation (SSE) from native cypha_rest.
# Requires Cypha sequence loaded at startup: --sequence-checkpoint or CYPHA_SEQUENCE_CHECKPOINT
# (aliases: --cyphalm-checkpoint, CYPHALM_CHECKPOINT, CYPHA_LM_CHECKPOINT).
PORT="${CYPHA_REST_PORT:-8099}"
curl -N -X POST "http://127.0.0.1:${PORT}/generate/stream" \
  -H "Content-Type: application/json" \
  -d @"$(dirname "$0")/lm_generate_body.json"
