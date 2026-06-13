# Cell hypothesis testbench

**Author:** Odin Loch  
**Purpose:** Systematic sweep of **28 recurrent cell hypotheses** to find a primitive that synergises with CyphaDIF/CyphaLM better than standard LSTM.  
**Status:** Planned — no native bench harness yet  
**Baseline:** `hybrid_gria_lstm` D17 BPC **2.873** @ 300k

Alternative to Option B in [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md); may inform RPSM level-0 design if a Cypha-derived cell wins.

---

## Testbench setup

| Parameter | Value |
|-----------|-------|
| Primary dataset | WikiText-2 char-level (D17) |
| Secondary | Penn Treebank char-level (validation) |
| Train budget | 300k tokens |
| Hidden dim | 256, 1 layer, seq_len 200, batch 64 |
| Optimiser | Adam lr=1e-3, 20 epochs |
| Runs per variant | 5 (report median BPC) |

### Locked baselines

| ID | Model | BPC |
|----|-------|-----|
| B0 | 4-gram | ~3.478 |
| B1 | char-LSTM | ~2.979 |
| B2 | hybrid_gria_lstm | **2.873** |

Logging: `results/variant_NAME_run_N.json` → `summarise.py` → `results/summary.csv`.

---

## Hypothesis tiers

### Tier 1 — highest priority (Cypha-native maths)

| ID | Name | Idea |
|----|------|------|
| H01 | α-gate cell | Forget gate = GRIA α(h) — semantic decay |
| H02 | EML activation cell | Sheffer `eml(x,y)` replaces σ/tanh |
| H03 | CausalField cell | SGEMV recurrence (Cypha SSM primitive) |
| H04 | Pure CyphaDIF LM | No neural recurrence — LLR + TieredContext only |
| H05 | α-fitness aux loss | L_α on LSTM / hybrid (a/b variants) |

### Tier 2 — medium complexity

| ID | Name |
|----|------|
| H06 | NIG-state cell |
| H07 | Differential gate (θ₀ + Δh_k) |
| H08 | TieredContext cell (short/mid/long) |
| H09 | GRIA-gated mixture (ordered vs chaotic) |
| H10 | NMP regularised (spec_alpha → 0.485) |
| H11 | Reversible cell (RevNet-style) |
| H12 | MDL forget (norm projection) |
| H13 | Priority replay recurrence |
| H14 | OOD-branching cell |

### Tier 3 — expensive / SR

| ID | Name | Notes |
|----|------|-------|
| H15 | AXIOM-evolved cell | GP over EML/σ/tanh grammar |
| H16 | SR on trained LSTM gates | **Strongest science** — discover what network learned |
| H17 | Sheffer-only cell | Extreme H02 |
| H18 | CA state cell | Wolfram class IV |
| H19 | Izaac-seeded init | Modifier on top-3 variants |
| H20 | Spectral state cell | FFT-domain recurrence |
| H21 | Free Energy cell | Variational active inference |
| H22 | Algebraic fingerprint cell | Izaac cluster transitions |

---

## Execution phases

```
Phase 1: Lock B0–B2 baselines
Phase 2: Tier 1 (H01–H05) — 5 variants × 5 runs
Phase 3: Tier 2 on best Tier 1 — apply modifiers
Phase 4: Tier 3 (H15 AXIOM, H16 SR) — background / overnight
Phase 5: Combine top-3 compatible variants → hybrid vs B2
```

Run **H04** (Pure CyphaDIF LM) in parallel with Tier 1 — high variance.

---

## Decision signals

| Signal | Criterion |
|--------|-----------|
| Primary | Median BPC **< 2.873** |
| GRIA alignment | α trajectory → ~0.5 |
| Edge-of-chaos | NMP > 0.97 on trained weights |
| Forgetting | Cypha zero-forgetting probe preserved |
| OOD | NIG / Free Energy variants excel on unseen Unicode |

**H16 priority:** If Tier 1 ambiguous, symbolic regression on trained `hybrid_gria_lstm` gates discovers closed-form update laws.

---

## Scaffold interface

All variants implement `BaseCyphaCell.forward(x, h_prev, c_prev) → (h, c)` wrapped by a common `CyphaLM` training loop (embed → cell → linear head). H04 runs standalone via native CyphaDIF path.

Native implementation path: optional `native/tools/cell_hypothesis_bench/` or extend `cyphalm_bench_native` with `--cell-variant` flag — not yet scheduled.

---

## Result table template

| ID | Variant | Median BPC | Δ vs B2 | Notes |
|----|---------|------------|---------|-------|
| B2 | hybrid_gria_lstm | 2.873 | 0.000 | current best |
| H01 | α-gate | | | |
| … | … | | | |

Fill as runs complete; link results into [`RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md) Possible upgrades table.
