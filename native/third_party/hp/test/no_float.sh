#!/usr/bin/env bash
# Enforce the hard rule: no floating point anywhere in the codec.
#
# The Hutter Prize verifies by running YOUR decompressor on THEIR machine.
# One ULP of difference in a libm call changes one probability, changes one
# arithmetic-coder interval, and the decoder desynchronises catastrophically.
# This check is the mechanical guarantee that that can never happen.
set -u
cd "$(dirname "$0")/.."
fail=0
hits=$(grep -nE '\b(float|double|long double)\b|<cmath>|<math\.h>|\b(exp|log|log2|pow|sqrt|expf|logf)\s*\(' \
        include/hp/*.hpp src/*.cpp \
        | grep -v 'log2_q16' | grep -v 'nlog2n_q16' | grep -v '^\s*//' | grep -vE '^[^:]+:[0-9]+:\s*//')
if [ -n "$hits" ]; then
  echo "FAIL: floating point found in codec sources:"
  echo "$hits"
  fail=1
else
  echo "PASS: no floating point in codec sources"
fi
exit $fail
