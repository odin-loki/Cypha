# Documentation & build-tree organization plan

> **Status:** The layout described below is **implemented** in-tree (see [`docs/README.md`](../README.md)). The phases remain as a record of how we got here.

This file was the **working plan** for consolidating docs and cleaning the repo layout.

---

## Goals

1. **One obvious entry point** for humans: where to read first, and how docs are grouped by audience.
2. **Predictable places** for generated output (profiles, benches, tuning) so the tree stays scannable and `.gitignore` stays honest.
3. **Minimal link rot**: prefer additive steps (new index + redirects in README) before mass renames.

---

## Implemented layout (reference)

| Area | Location |
|------|-----------|
| **Docs** | [`docs/README.md`](../README.md) hub; `docs/studio/`, `docs/port/`, `docs/verify/` (incl. [`MAINTENANCE.md`](MAINTENANCE.md)), `docs/benchmarks/` |
| **Scripts** | [`scripts/README.md`](../../scripts/README.md) index |
| **Examples** | `examples/` |
| **Generated output** | `artifacts/profiles/`, `artifacts/bench/`, `artifacts/tuning/` |
| **Tooling noise** | Ignored via root `.gitignore` (`.venv*/`, `.pytest_cache/`, etc.) |

---

## Target information architecture (documentation)

Introduce a **hub page** that everything else hangs off:

- **`docs/README.md`** (or `docs/INDEX.md`) — short intro + **three lanes**:
  - **Use / run** — Studio GUI, headless API, env vars (`CYPHA_ENV.md`), keyboard shortcuts (pointer to master plan or a future `docs/studio/USER.md`).
  - **Develop / verify** — `VERIFICATION_STATUS.md`, `VERIFY_PLAN.md`, `CONTRIBUTING.md`, test commands, parity fixtures.
  - **Port / native** — `PORT_CONTRACT.md`, `PORT_FULL_STACK.md`, `PREPROCESSOR_CONTRACT.md`, `parity_fixtures/README.md`, `native/README.md`.

**Optional physical grouping** (only after hub exists and links are updated):

```text
docs/
  README.md                 # hub
  studio/
    CYPHA_ENV.md
    CYPHA_STUDIO_MASTER_PLAN.md
    STUDIO_THREADING.md
    OPTIONAL_MEMORY_AND_LOAD.md
  port/
    PORT_CONTRACT.md
    PORT_FULL_STACK.md
    PREPROCESSOR_CONTRACT.md
  verify/
    VERIFICATION_STATUS.md
    VERIFY_PLAN.md
    ROADMAP.md
  benchmarks/
    BENCHMARK_GPU.md
    PROFILE_IMPROVEMENTS_*.md   # or docs/history/ for dated one-offs
```

**Naming:** keep **stable public names** where external links might exist (`PORT_CONTRACT.md`); if you move files, add **stub files** at old paths (one line: “Moved to …”) for one release cycle, or accept a single coordinated rename + global search/replace in-repo.

---

## Target layout (build outputs & “generated” tree)

| Purpose | Suggested location | Git policy |
|--------|---------------------|------------|
| Profile / cProfile text outputs | `artifacts/profiles/` or `var/profiles/` | Ignore `*.txt` / `*.log` here; optionally commit **one** canonical snippet referenced from docs |
| GPU / production bench JSON | `artifacts/bench/` | Same: ignore by default; commit only if used as regression baselines |
| Tuning grid outputs | `artifacts/tuning/` | Ignore; document how to reproduce |
| Parity fixtures | `parity_fixtures/` | **Keep** — versioned test assets |

**Migration:** move existing `bench_runs/` and `tuning_runs/` contents under `artifacts/bench/` and `artifacts/tuning/` (or `var/…`), update any scripts that write there, then delete empty dirs.

---

## Scripts directory

Add **`scripts/README.md`** — a single table:

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `setup_and_test.sh` / `.ps1` | Bootstrap + verify; optional **`FULL_STUDIO_DEPS=1`** / **`-Studio`** for studio + **`pytest-qt`** | console |
| `profile_*.py` | cProfile / tracemalloc | stdout or `-o` path |
| `gpu_*.py`, `bench_*.py` | GPU timing | stdout / JSON |
| `loadtest_ab_predict_example.*` | Live `ab` example | console |

Point to **`docs/README.md`** for narrative “when to run what.”

---

## Repository hygiene

1. **Root `.gitignore`** (if missing or incomplete):  
   `.venv/`, `.venv-*/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, IDE folders, and under `artifacts/` ignore generated files (keep `artifacts/.gitkeep` if you want the directory present).
2. **Single source of truth for verify commands:** shorten **README** to a minimal quickstart + link to **CONTRIBUTING** or **docs/verify/** for the full matrix; avoid duplicating long bash blocks in three places.
3. **`pyproject.toml` / `Makefile`:** ensure targets reference the same paths as the artifact layout (after moves).

---

## Phased execution (recommended order)

| Phase | Work | Risk |
|-------|------|------|
| **A — Hub + hygiene** | Add `docs/README.md` hub; extend root README doc table with missing links; add/ tighten `.gitignore`; add `scripts/README.md` | Low |
| **B — Artifacts** | Create `artifacts/{profiles,bench,tuning}/`, move `bench_runs/` / `tuning_runs/`, update writers + docs | Medium (path updates) |
| **C — Doc folders** | Create `docs/studio/`, `docs/port/`, `docs/verify/`, move markdown, global link replace, optional stubs at old paths | Medium–high |
| **D — Consolidate verify narrative** | Merge overlap between `VERIFY_PLAN` and `VERIFICATION_STATUS` *or* clearly scope one as “checklist” and one as “snapshot” | Low (editorial) |

Stop after any phase if you want a stable checkpoint.

---

## Success criteria

- A new contributor opens **`docs/README.md`** and finds the right doc in **two clicks**.
- Generated files **do not clutter** the repo root; **`git status`** stays readable on a dev machine.
- **README** + **CONTRIBUTING** + **docs hub** do not contradict each other on install and verify commands.

---

## Out of scope (unless you explicitly want them)

- Rewriting technical content of port contracts.
- CI configuration (unless artifact paths are hard-coded there).
- Auth / rate limits for the API (product decision; see master plan Phase 4).

When this plan is fully executed, you can **archive or delete** this file and fold a one-paragraph “repo layout” into `docs/README.md`, or keep this file as **`docs/MAINTENANCE_PLAN.md`** for historical traceability.
