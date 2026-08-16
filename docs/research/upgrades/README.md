# Possible upgrades — research index

Distilled specs backported from the former `Cypha Possible Upgrades/` folder (2026-06). These are **historical research specs** unless a row below says otherwise. **RPSM and cell-sweep tracks are STOP/closed** per BoW (2026-07-17). Living sequence work is **Hybrid 2.664 BPC (L2 + Wave2 BPTT) + predictive arithmetic coding**.

**Canonical status table:** [`docs/RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md).

**Roadmap placement:** [`docs/FUTURE.md`](../../FUTURE.md) §0 (evidence-ranked) and §10 (RPSM / CyphaLM horizon).

---

## Documents

| Doc | Topic | Status |
|-----|--------|--------|
| [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md) | Option A (CyphaDIF matrix refactor) + Option B (RPSM sequence layer) | **STOP** — historical target 2.873; living pin **2.664** |
| [RPSM_IMPLEMENTATION.md](RPSM_IMPLEMENTATION.md) | RPSM core dynamics and five critical fixes | **STOP** — archived implementation detail |
| [NONLINEAR_BOUNDARY.md](NONLINEAR_BOUNDARY.md) | Nonlinear discriminant (Nyström, RFF) | **Living default:** latent RFF kernel LLR; xor_pair opt-in |
| [CELL_HYPOTHESIS_TESTBENCH.md](CELL_HYPOTHESIS_TESTBENCH.md) | 36 cell variants (B0–B2, H01–H23, U01–U10) vs hybrid | **STOP** — historical research; living spine is Hybrid 2.664 |
| [PREDICTIVE_ARITHMETIC_CODING.md](PREDICTIVE_ARITHMETIC_CODING.md) | LLMZip-style arithmetic coding on `predict_next` | **Living** — shipped 2026-07-19 |
| [ADAPTIVE_PREDICTOR_MIXER.md](ADAPTIVE_PREDICTOR_MIXER.md) | Codec-owned CMIX-style mixer + match/kNN + online adapt | **Living** — slice-2 shipped |
| [ARCHITECTURE_BPC_LEARNINGS_2026-07-19.md](ARCHITECTURE_BPC_LEARNINGS_2026-07-19.md) | Mixer + depth path toward sub-2.0 BPC | **Living** — architecture notes |

---

## Execution order (combined RPSM track) — **STOP/closed**

RPSM Option A/B and the cell hypothesis testbench are **not active**. Kept for audit trail only.

---

## Related docs

- [`docs/FUTURE.md`](../../FUTURE.md) — §0a kernel LLR (shipped), §10 RPSM horizon
- [`docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`](../../native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md)
- [`docs/port/PORT_FULL_STACK.md`](../../port/PORT_FULL_STACK.md)
- [`docs/archive/reports/DIAGNOSTIC_REPORT.md`](../../archive/reports/DIAGNOSTIC_REPORT.md)
- [`data/README.md`](../../../data/README.md) — historical XOR / D17 traces
