#!/usr/bin/env sh
# Example ApacheBench run against a *live* uvicorn (trusted network only).
# Default CyphaStudio headless: CYPHA_API_HOST / CYPHA_API_PORT
#
#   export CYPHA_API_HOST=127.0.0.1
#   export CYPHA_API_PORT=7749
#   bash scripts/loadtest_ab_predict_example.sh
#
# Requires: ab (apache2-utils / httpd)

HOST="${CYPHA_API_HOST:-127.0.0.1}"
PORT="${CYPHA_API_PORT:-7749}"
URL="http://${HOST}:${PORT}/predict"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ab -n 500 -c 10 -T application/json -p "${ROOT}/examples/cypha_predict_body.json" "$URL"
