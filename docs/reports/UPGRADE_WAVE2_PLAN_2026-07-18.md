# Upgrade wave 2 — 2026-07-18

**Goal:** Attack remaining *capability* ceilings (not free polish / STOP’d RPSM).  
**Prior plan:** [`BACKLOG_EXECUTION_PLAN_2026-07-18.md`](BACKLOG_EXECUTION_PLAN_2026-07-18.md)

| Fork | Bet | FAST gate | Stretch |
|------|-----|-----------|---------|
| 1 RFF regression | Post-warmup γ recalib on D14 | mean R² ≥ linear-FAST | Full 14A ≥ 0.444 |
| 1b Residual RFF | Stage-2 RFF on residuals (if 1 fails) | single-eq R² lift | — |
| 2 Expert util | Soft expert updates + entropy floor + warm `n_experts` | `n_experts≥4` @5k + BPC ≤ Tiny+5% | 300k BPC ≤ 2.873 |
| 3 Routing CL | Task-prefix delta protect in D16B | forgetting ≤ EWC best | — |
| 4 BPE LM | Train BPE assets + d17_bpe profile | BPC @5k < 4.04 | 300k < 2.873 |
| 5 Causal | OnlineCorrelation + lag asymmetry | smoke + fidelity | discovery later |
| 6 Product | Demo script: online + OOD + uncertainty-rank | runs end-to-end | — |

## Non-goals
RPSM BPTT/capacity, GMM XOR default-on, index-reorder multi-view, D17 micro-opt polish.

## Status log
See [`UPGRADE_WAVE2_STATUS_2026-07-18.md`](UPGRADE_WAVE2_STATUS_2026-07-18.md) — forks 1–6 measured; stretch 1b residual RFF **PASS** (`0.527`); real WikiText BPE short/mid **FAIL**; paper PNGs landed.
