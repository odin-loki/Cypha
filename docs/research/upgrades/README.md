# Possible upgrades — research index

Distilled specs backported from the former `Cypha Possible Upgrades/` folder (2026-06). These are **planned** engineering directions, not shipped product features unless noted.

**Canonical status table:** [`docs/RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md) § Possible upgrades.

**Roadmap placement:** [`docs/FUTURE.md`](../../FUTURE.md) §0 (evidence-ranked) and §10 (RPSM / CyphaLM horizon).

---

## Documents

| Doc | Topic | Target |
|-----|--------|--------|
| [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md) | Option A (CyphaDIF matrix refactor) + Option B (RPSM sequence layer), execution order, D17 BPC target | Beat hybrid @ 300k (**2.873 BPC**) |
| [RPSM_IMPLEMENTATION.md](RPSM_IMPLEMENTATION.md) | RPSM core dynamics, five critical fixes (spectral α, normalised η, orthogonal init, multi-level injection, symmetric W_down) | Option B implementation detail |
| [NONLINEAR_BOUNDARY.md](NONLINEAR_BOUNDARY.md) | Nonlinear discriminant fixes (Nyström, RFF, MLP encoder, GRIA-α kernel, spectral mixture) | Close XOR 32.3 pp gap |
| [CELL_HYPOTHESIS_TESTBENCH.md](CELL_HYPOTHESIS_TESTBENCH.md) | 28 recurrent cell hypotheses vs char-LSTM / hybrid_gria_lstm | Replace or beat LSTM primitive |

---

## Execution order (combined RPSM track)

```
1. Option A — CyphaDIF matrix refactor (Ψ_mu / Ψ_var, batched LLR)
2. Nonlinear boundary — Nyström kernel LLR into A (partially shipped; see NONLINEAR_BOUNDARY.md)
3. Option B — RPSM sequence layer in CyphaLM (CyphaDIF at level 0)
4. Wire Izaac episodic store + working memory + GMM world model
5. Full D17 benchmark vs hybrid_gria_lstm baseline
```

Parallel track: **Cell hypothesis testbench** (Tier 1–3 sweep) — independent of RPSM; may inform Option B cell design.

---

## Related docs

- [`docs/FUTURE.md`](../../FUTURE.md) — §0a kernel LLR (shipped), §10 RPSM horizon
- [`docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`](../../native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md) — Option A as future native milestone
- [`docs/port/PORT_FULL_STACK.md`](../../port/PORT_FULL_STACK.md) — port tracker pointer
- [`docs/reports/DIAGNOSTIC_REPORT.md`](../../reports/DIAGNOSTIC_REPORT.md) — XOR ceiling evidence
