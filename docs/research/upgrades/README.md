# Possible upgrades — research index

Distilled specs backported from the former `Cypha Possible Upgrades/` folder (2026-06). These are **historical research specs** unless a row below says otherwise. **RPSM and cell-sweep tracks are STOP/closed** per BoW (2026-07-17) — not active product work to beat the canonical **2.873 BPC** pin.

**Canonical status table:** [`docs/RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md) § Possible upgrades.

**Roadmap placement:** [`docs/FUTURE.md`](../../FUTURE.md) §0 (evidence-ranked) and §10 (RPSM / CyphaLM horizon).

---

## Documents

| Doc | Topic | Status |
|-----|--------|--------|
| [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md) | Option A (CyphaDIF matrix refactor) + Option B (RPSM sequence layer), execution order, D17 BPC target | **STOP** — closed per BoW; canonical pin **2.873 BPC** |
| [RPSM_IMPLEMENTATION.md](RPSM_IMPLEMENTATION.md) | RPSM core dynamics, five critical fixes (spectral α, normalised η, orthogonal init, multi-level injection, symmetric W_down) | **STOP** — archived implementation detail |
| [NONLINEAR_BOUNDARY.md](NONLINEAR_BOUNDARY.md) | Nonlinear discriminant fixes (Nyström, RFF, MLP encoder, GRIA-α kernel, spectral mixture) | Partially shipped (Nyström kernel LLR); remainder historical |
| [CELL_HYPOTHESIS_TESTBENCH.md](CELL_HYPOTHESIS_TESTBENCH.md) | 28 recurrent cell hypotheses vs char-LSTM / hybrid_gria_lstm | **STOP** — cell sweep closed per BoW |

---

## Execution order (combined RPSM track) — **STOP/closed**

RPSM Option A/B and the cell hypothesis testbench are **not active**. Kept for audit trail only.

---

## Related docs

- [`docs/FUTURE.md`](../../FUTURE.md) — §0a kernel LLR (shipped), §10 RPSM horizon
- [`docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`](../../native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md) — Option A as future native milestone
- [`docs/port/PORT_FULL_STACK.md`](../../port/PORT_FULL_STACK.md) — port tracker pointer
- [`docs/reports/DIAGNOSTIC_REPORT.md`](../../reports/DIAGNOSTIC_REPORT.md) — XOR ceiling evidence
