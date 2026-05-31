#!/usr/bin/env bash
# One-liner curl against the /predict endpoint using cypha_predict_body.json
# Adjust --port and the input vector to match your model's latent dim.
curl -s -X POST http://127.0.0.1:7749/predict \
  -H "Content-Type: application/json" \
  -d @"$(dirname "$0")/cypha_predict_body.json" | python3 -m json.tool
