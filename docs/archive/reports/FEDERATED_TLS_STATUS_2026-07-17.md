# Federated TLS status (2026-07-17)

**Scope:** Bill of Work — federated training TLS smoke slice. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight processes.

## Verdict

| Layer | Status |
|-------|--------|
| **Merge golden parity** (`d1a9bf1`) | **PASS** — blocking CTest `native_federated_merge_golden` |
| **Default native build** | TLS **skipped** (`CYPHA_ENABLE_OPENSSL=OFF`; `native_federated_tls_smoke` exit 2) |
| **OpenSSL-enabled build** | TLS **enabled** — `native_federated_tls_smoke` **PASS** on this host (2026-07-17) |

**BoW closeout:** merge golden is the sole blocking federated cert; TLS remains an optional CI/local job when OpenSSL is available.

## OpenSSL on this host

| Check | Result |
|-------|--------|
| `openssl` CLI | **Present** — OpenSSL 3.3.0 (`C:\Strawberry\c\bin\openssl.exe`) |
| Dev headers/libs | **Present** — `include/openssl/ssl.h`, `lib/libssl.a` under `C:\Strawberry\c` |
| CMake default | **OFF** — `-DCYPHA_ENABLE_OPENSSL=OFF` (no OpenSSL link in blocking gate) |

Without OpenSSL dev libraries **or** the `openssl` CLI on `PATH`, the smoke binary exits **2** and CTest treats it as **Skipped** (`SKIP_RETURN_CODE 2`).

## Golden merge (`d1a9bf1`)

Commit **`d1a9bf1`** — *Add federated merge golden parity CTest beyond TLS smoke.*

| Item | Role |
|------|------|
| `fixtures/federated_merge/expected_merged.json` | Golden merged state (`worker_a` + `worker_b`) |
| `native/tests/regression/federated_merge_golden.cpp` | World prior + Fisher–Rao `delta_mu` + label union |
| `native_federated_merge_golden` (CTest) | Blocking gate; `PASS_REGULAR_EXPRESSION` `federated_merge parity OK` |

Related non-TLS smokes (also blocking in default build): `native_federated_merge_smoke`, `native_federated_coordinator_merge_smoke`, `native_federated_coordinator_smoke`, `native_federated_worker_smoke`.

See [`FEDERATED_SLICE_2026-07-17.md`](FEDERATED_SLICE_2026-07-17.md) for merge semantics.

## TLS smoke (already wired — no new code this slice)

Implementation lives in existing tree (pre-BoW):

| Item | Purpose |
|------|---------|
| `native/tools/federated_tls_smoke.cpp` | HTTPS loopback: self-signed cert via `openssl req`, `httplib::SSLServer` + `SSLClient`, `/submit` merge |
| `native/CMakeLists.txt` | `CYPHA_ENABLE_OPENSSL` option; links `OpenSSL::SSL` / `OpenSSL::Crypto` when ON |
| `native_federated_tls_smoke` (CTest) | Optional; **exit 2 = skip** (`SKIP_RETURN_CODE 2`) without OpenSSL build or CLI |
| `scripts/ci_federated_tls_linux.sh` | Optional CI job mirror (Linux) |
| `scripts/ci_federated_tls_windows.ps1` | Optional CI job mirror (Windows) |

Security notes (unchanged):

- Smoke uses a **1-day self-signed** cert generated at runtime (`CN=localhost`); not for production.
- Loopback client disables cert verification **only** for the smoke (`enable_server_certificate_verification(false)`); production coordinator/worker should use proper trust stores.

## How to enable TLS smoke locally

### Windows (Strawberry OpenSSL example)

```powershell
$env:OPENSSL_ROOT_DIR = "C:\Strawberry\c"
powershell -NoProfile -File scripts/ci_federated_tls_windows.ps1
```

Or manually:

```powershell
cmake -S native -B native/build_fed_tls -DCMAKE_BUILD_TYPE=Release `
  -DCYPHA_ENABLE_OPENSSL=ON -DOPENSSL_ROOT_DIR=C:\Strawberry\c
cmake --build native/build_fed_tls --target federated_tls_smoke
ctest --test-dir native/build_fed_tls -R native_federated_tls_smoke --output-on-failure
```

Use a dedicated tree (`native/build_fed_tls`); do **not** reconfigure overnight or baseline build dirs.

### Linux

```bash
# apt: libssl-dev + openssl
bash scripts/ci_federated_tls_linux.sh
```

### CI

Optional workflow job **`federated_tls`** in `.github/workflows/ci.yml` (`continue-on-error`); blocking gate elsewhere runs `ctest -R native_` without OpenSSL.

## Local run (2026-07-17)

```text
scripts/ci_federated_tls_windows.ps1 -BuildDir native/build_fed_tls
→ Federated TLS: OpenSSL enabled (CPPHTTPLIB_OPENSSL_SUPPORT)
→ native_federated_tls_smoke ....... Passed (~1s)
```

Default `native/build_*` trees without `-DCYPHA_ENABLE_OPENSSL=ON` report `native_federated_tls_smoke` as **Skipped** — expected.
