# cypha_som — failed experiment (archived)

**Status:** RETAINED IN TREE, **DEFAULT OFF**. Do not enable in production or `everyday_profile.json`.

## Hypothesis

Self-organising map / GNG / GRIA / Hebbian / temporal hooks would improve online CyphaDIF accuracy on standard tabular classification.

## Result (2026-05-26)

Full evaluation: [`docs/reports/SOM_UPGRADE_REPORT.md`](../../reports/SOM_UPGRADE_REPORT.md)

| Upgrade | Accuracy Δ | Verdict |
|---------|------------|---------|
| U2 SOM encoder | **−3.5%** | REVERT |
| U1 GNG | neutral, **>2× train time** | REVERT |
| U4 discriminative feedback | +1pp, high variance | REVERT |
| U3 / U5 / U6 | neutral on CyphaDIF | CellAI-only research |
| `all` combined | **−3.5%** | REVERT |

## Code locations

| Layer | Path |
|-------|------|
| Python package | `cypha_som/` (7 pytest tests, not in CI) |
| Integration hooks | `Cypha.py` (lazy load when flags ON) |
| CellAI wiring | `cypha_lm/temporal/cellai_ssm.py` |
| Native parity smoke | `native/src/som/`, `native/tools/som_parity.cpp` |
| Config flags | `cypha_som/config.py`, `native/include/cypha/cyphalm/cyphalm_config.hpp` |

## Lessons

- SOM-on-RFF loses too much information for tabular CyphaDIF.
- GNG reservoir overhead is not justified by accuracy gains.
- Combined upgrades interact badly (U2 dominates negative effect).

## What to use instead

- **Nonlinear boundaries (XOR):** `use_kernel_llr=True` + `KernelMemory` — see `scripts/benchmark_xor_kernel_llr.py`.
- **Structural intelligence stats:** `native/include/cypha/intelligence/` (P-space profiler).
