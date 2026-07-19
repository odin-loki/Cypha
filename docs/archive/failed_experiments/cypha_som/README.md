# cypha_som -- failed experiment (archived)

> **Archive banner:** Retained for history only. Not part of the One Cypha product spine (`cypha::Cypha`). Living docs: [`../../../README.md`](../../../README.md), [`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md).

**Status:** RETAINED IN TREE, **DEFAULT OFF**. Do not enable in production or `everyday_profile.json`.

## Hypothesis

Self-organising map / GNG / GRIA / Hebbian / temporal hooks would improve online classifier accuracy on standard tabular classification.

## Result (2026-05-26)

Full evaluation: [`../../reports/SOM_UPGRADE_REPORT.md`](../../reports/SOM_UPGRADE_REPORT.md)

| Upgrade | Accuracy delta | Verdict |
|---------|----------------|---------|
| U2 SOM encoder | **-3.5%** | REVERT |
| U1 GNG | neutral, **>2x train time** | REVERT |
| U4 discriminative feedback | +1pp, high variance | REVERT |
| U3 / U5 / U6 | neutral on DIF classifier | CellAI-only research |
| `all` combined | **-3.5%** | REVERT |

## Code locations

| Layer | Path |
|-------|------|
| Native parity smoke | `native/src/som/`, `native/tools/som_parity.cpp` |
| Config flags | `native/include/cypha/cyphalm/cyphalm_config.hpp` |
| Historical Python package | removed with P7 (`cypha_som/`) |

## Lessons

- SOM-on-RFF loses too much information for tabular DIF routing.
- GNG reservoir overhead is not justified by accuracy gains.
- Combined upgrades interact badly (U2 dominates negative effect).

## What to use instead

- **Nonlinear boundaries (XOR):** kernel LLR + `KernelMemory` -- see `xor_kernel_bench` / [`../../../research/upgrades/NONLINEAR_BOUNDARY.md`](../../../research/upgrades/NONLINEAR_BOUNDARY.md).
- **Structural intelligence stats:** `native/include/cypha/intelligence/` (P-space profiler).
