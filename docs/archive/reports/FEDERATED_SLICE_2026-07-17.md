# Federated merge golden parity (2026-07-17)

**Build:** `native/build_fed` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Bill of Work — federated training slice beyond TLS smoke. JSON merge of `world_mu` / `world_v` / `world_n` and per-class `delta_mu` + observation counts via `federated_average_payloads` → `memory_merge_from`. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight processes.

## Shipped

| Item | Purpose |
|------|---------|
| `fixtures/federated_merge/expected_merged.json` | Golden merged state for `worker_a.json` + `worker_b.json` |
| `native/tests/regression/federated_merge_golden.cpp` | Validates world prior merge + Fisher–Rao class `delta_mu` blend + label union (`neg` from worker B only) |
| `native_federated_merge_golden` (CTest) | Regression gate with `PASS_REGULAR_EXPRESSION` |
| `native_federated_coordinator_merge_smoke` (CTest) | Coordinator `--merge` CLI round-trip (no HTTP/Redis/gRPC) |

Existing tools unchanged in contract: `cypha_federated_merge`, `cypha_federated_coordinator` (`--merge`, `--watch-dir`, `--listen`).

## Merge semantics (fixture)

Two workers, `d_latent=2`:

- **Worker A:** `world_n=10`, `world_mu=[0,1]`, class `pos` only (`n_obs=4`, `delta_mu=[0.1,-0.1]`).
- **Worker B:** `world_n=20`, `world_mu=[2,3]`, classes `pos` + `neg`.

**Expected merged:**

- `world_n=30`, `world_mu` / `world_v` = count-weighted average across workers.
- `pos`: `n_obs=10`, `n_correct=8`, `delta_mu` Fisher–Rao blend of overlapping class stats.
- `neg`: copied from worker B (new label union).

Golden test also checks nested `world: {n, mu, v}` JSON parse path (`federated_payload_from_json`).

## Tests

```powershell
cmake -S native -B native/build_fed -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_fed --target federated_merge_golden cypha_federated_merge cypha_federated_coordinator -j
ctest --test-dir native/build_fed -R "native_federated_merge|native_federated_coordinator_merge" --output-on-failure
```

| CTest | Result |
|-------|--------|
| `native_federated_merge_smoke` | PASS (CLI write) |
| `native_federated_merge_golden` | PASS (numeric parity) |
| `native_federated_coordinator_merge_smoke` | PASS (`--merge` CLI) |
| `native_federated_coordinator_smoke` | PASS (`--watch-dir --once`) |
| `native_federated_worker_smoke` | PASS (in-process HTTP loopback) |
| `native_federated_tls_smoke` | SKIP (OpenSSL not built; `SKIP_RETURN_CODE 2`) |

## Files touched

- `fixtures/federated_merge/expected_merged.json` (new)
- `native/tests/regression/federated_merge_golden.cpp` (new)
- `native/cmake/CyphaRegression.cmake` — register golden exe
- `native/CMakeLists.txt` — compile flags + CTests

## Follow-up (not in this slice)

- Golden validation of coordinator HTTP `/submit` output vs merge tool
- TLS smoke in CI with `-DCYPHA_ENABLE_OPENSSL=ON`
- Weighted merge CLI (`--weight-self` / per-worker weights)
- Round-trip: merged JSON → `CyphaDifMemoryState` → `.cypha` save/load
