# RFF latent auto-gamma — recommended exploratory default

**Date:** 2026-07-17  
**Scope:** Generalizable **latent** XOR/kernel path only. **`xor_pair` production default unchanged.**  
**Evidence source:** Prior validated sweep in [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) Priority 1 (2026-07-11); no fresh re-run this closeout (Nyström M=512 full latent run impractical; RFF numbers already reproduced live in `d03_xor` wiring).

---

## Verdict

**Promoted (exploratory default):** `latent` feature mode + RFF basis + auto-gamma (median heuristic).  
**Not promoted:** `xor_pair` / Nyström M=512 shipped default — remains production path for `d03_xor` and `kernel_llr_profile.json`.

Profile: [`bench/config/latent_rff_auto_gamma.json`](../../bench/config/latent_rff_auto_gamma.json).

---

## Accuracy table (3 seeds × 8 passes unless noted)

Linear baseline (latent): **51.2%** on all rows below.

| Config | Kernel acc | Δ vs linear | Gap to sklearn (~79%) | Notes |
|--------|------------|-------------|------------------------|-------|
| Nyström M=256, γ_scale=1.0 | **62.07%** | +10.83 pp | ~16.9 pp | Prior baseline; practical M ceiling ~384 |
| Nyström M=384 | 62.53% | +11.33 pp | ~16.5 pp | Diminishing returns |
| Nyström M=512, latent | 54.7% | +3.3 pp | — | FAST only (1×2); full 3×8 impractical (`O(M³)`/step) |
| RFF fixed γ=1.0, `rff_dim=512` | 49.4% | **−1.8 pp** | — | Worse than linear — bad fixed γ |
| RFF fixed γ=0.2, `rff_dim=512` | 55.6% | +4.4 pp | — | |
| RFF fixed γ=0.01, `rff_dim=512` | 64.3% | +13.1 pp | ~14.7 pp | |
| **RFF auto-γ, `rff_dim=512`** | **68.0%** | **+16.8 pp** | ~11.0 pp | Beats all fixed-γ at D=512 |
| RFF auto-γ, `rff_dim=2048` | 75.7% | +24.5 pp | ~3.3 pp | |
| **RFF auto-γ, `rff_dim=4096`** | **76.3%** | **+25.1 pp** | **~2.7 pp** | **Best found**; per-seed 0.763/0.743/0.784 |
| `xor_pair` + Nyström M=512 (prod) | **98.3%** | +46.9 pp | exceeds ceiling | **Unchanged default** |

Sklearn RBF SVM ceiling on same splits: ~79%. Returns past `rff_dim≈3072–4096` flatten (+0.56 pp for last +1024 dims).

---

## Recommendation

| Path | Default | Rationale |
|------|---------|-----------|
| **Production XOR / `d03_xor`** | `xor_pair` + Nyström M=512 | Hand-engineered features already exceed sklearn; no RFF benefit |
| **Generalizable latent / research** | RFF + auto-γ, `rff_dim=4096` | Closes ~18 pp Nyström gap to ~2.7 pp; `O(M·d)` tractable |
| **Fast latent smoke** | RFF + auto-γ, `rff_dim=512` | 68.0% in prior sweep; cheaper than 4096 for CI/exploration |

Auto-gamma uses `KernelMemory::auto_gamma_median_heuristic` (calibration batch median pairwise distance). Override with `--rff-fixed-gamma G` or `CYPHA_D03_RFF_GAMMA_SCALE` only when comparing.

---

## How to enable

**Bench domain (`d03_xor`):**

```powershell
$env:CYPHA_D03_KERNEL_FEATURE_MODE = "latent"
$env:CYPHA_D03_KERNEL_BASIS = "rff"
$env:CYPHA_D03_RFF_DIM = "4096"
# auto-γ is default; optional scale:
# $env:CYPHA_D03_RFF_GAMMA_SCALE = "1.0"
cypha_bench_run --domain-tag d03_xor
```

**Standalone tool:**

```text
xor_kernel_bench --kernel-feature-mode latent --kernel-basis rff --rff-dim 4096 --seeds 3 --passes 8
```

Or load profile env block from `bench/config/latent_rff_auto_gamma.json`.

---

## What did not change

- `xor_kernel_bench` CLI defaults: still `xor_pair` + `nystrom`
- `run_d03_xor()` default env: still `xor_pair` + Nyström M=512
- `bench/config/kernel_llr_profile.json`: still Nyström tabular / xor_pair notes
- D14 kernelized routing: negative result; do not default ([`NONLINEAR_BOUNDARY.md`](../research/upgrades/NONLINEAR_BOUNDARY.md))

---

## References

- Prior sweep + fixed-vs-auto-γ: [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) Priority 1 (2026-07-11)
- D03 wiring + live 76.3% reproduction: [`NONLINEAR_BOUNDARY.md`](../research/upgrades/NONLINEAR_BOUNDARY.md) Fix 2 updates
- Phase 5 opt-ins (SORF/leverage): [`OPTIMALITY_PHASE5_PROMOTE_2026-07-17.md`](OPTIMALITY_PHASE5_PROMOTE_2026-07-17.md) — not part of this default
