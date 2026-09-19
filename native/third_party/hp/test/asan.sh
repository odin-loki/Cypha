#!/usr/bin/env bash
# AddressSanitizer smoke test: build with ASAN=1 and round-trip proxy corpora.
set -euo pipefail
cd "$(dirname "$0")/.."
exec make asan-test
