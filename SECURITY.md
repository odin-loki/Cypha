# Security Policy

## Supported versions

Cypha is a research prototype. Only the current `main` branch is maintained.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| older commits | ❌ |

## Reporting a vulnerability

If you discover a security vulnerability in this project, please **do not open a public GitHub issue**. Instead:

1. Email the repository owner directly (see GitHub profile for contact).
2. Describe the vulnerability, steps to reproduce, and potential impact.
3. Allow up to 14 days for an initial response.

Security issues in third-party dependencies (Qt, OpenSSL where linked, system libc, etc.) should be reported to the respective upstream projects.

## Scope

Cypha is a local research tool. The native C++ runtime (`cypha_core`, `cypha_rest`, optional Qt shell) is the sole maintained attack surface. There is no in-tree Python or FastAPI server.

### Primary surfaces

- **`cypha_rest`** — HTTP REST API. Binds to `127.0.0.1:7749` by default (`CYPHA_API_HOST`, `CYPHA_API_PORT`). No built-in authentication. Do not expose it to untrusted networks without TLS termination, access control, and a deliberate threat model.
- **`cypha_qt_shell`** — Qt Studio GUI. Can spawn `cypha_rest`, import CSV datasets, load registry bundles via `POST /load`, and save `.cypha` checkpoints. Treat the host machine and any configured REST base URL as trusted.
- **`.cypha` model files** — binary checkpoints loaded by `cypha::Cypha` / `cypha_rest --cypha`. Treated as trusted input; parsing is not sandboxed. Do not open untrusted `.cypha` files.
- **Sequence checkpoints** — optional `.json` + `.npz` pairs (`CYPHA_SEQUENCE_CHECKPOINT` and aliases). Same trust model as `.cypha`.
- **Registry paths** — `CYPHA_REGISTRY_ROOT` (or `cypha_rest --registry`) is resolved as a filesystem path. `POST /register` copies host-supplied paths into the registry tree. Do not point the registry at user-controlled directories or pass untrusted paths in register requests.
- **Preprocessor / regression sidecars** — `preprocessor.json`, `regression_head.json`, and related JSON loaded from explicit paths or alongside a model. Validate origin before load.
- **Branch A / Ollama fallback** (optional) — `/route/*` may call a configured Ollama HTTP endpoint (`CYPHA_OLLAMA_URL`). Restrict network egress and treat the Ollama host as a separate trust boundary.

### Deployment notes

- **CORS** — `CYPHA_CORS_ORIGINS` defaults to `*`. Restrict to known origins when a browser UI talks to `cypha_rest` on anything other than a single-user localhost setup.
- **TLS** — Terminate HTTPS at a reverse proxy or load balancer; keep `cypha_rest` on loopback where possible.
- **Session state** — predict/update history lives in process memory; scaling out requires independent replicas or external session design.

### Historical (removed from tree)

Prior releases shipped a Python FastAPI server (`cypha_studio.server.api`) with a parallel REST contract. That stack is **not maintained** after the One Cypha cutover. Do not deploy archived Python wheels or docs as if they were supported; report issues only against the native runtime above.
