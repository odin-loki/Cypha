# Security Policy

## Supported versions

CyphaDIF is a research prototype. Only the current `main` branch is maintained.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| older commits | ❌ |

## Reporting a vulnerability

If you discover a security vulnerability in this project, please **do not open a public GitHub issue**. Instead:

1. Email the repository owner directly (see GitHub profile for contact).
2. Describe the vulnerability, steps to reproduce, and potential impact.
3. Allow up to 14 days for an initial response.

Security issues in third-party dependencies (numpy, scipy, FastAPI, Qt, etc.) should be reported to the respective upstream projects.

## Scope

This project is a local research tool. The attack surface is:

- **FastAPI REST server** (`cypha_studio.server.api`) — binds to `localhost` by default. Do not expose it to the public internet without authentication.
- **Native `cypha_rest`** — same; binds to `localhost:7749` by default.
- **`.cypha` model files** — treated as trusted input. Do not load untrusted `.cypha` files; the binary format is not sandboxed.
- **Registry paths** — `CYPHA_REGISTRY_ROOT` is resolved as a filesystem path; do not set it to user-controlled input.
