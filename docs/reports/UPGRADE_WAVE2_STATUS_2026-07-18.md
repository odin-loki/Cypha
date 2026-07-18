# Upgrade wave 2 — status (2026-07-18)

**Plan:** [`UPGRADE_WAVE2_PLAN_2026-07-18.md`](UPGRADE_WAVE2_PLAN_2026-07-18.md)  
**Build:** `native/build_ewc_d16`

| Fork | Shipped | FAST / short gate | Result |
|------|---------|-------------------|--------|
| 1 D14 RFF post-warmup γ | `CYPHA_D14_KERNEL_CALIB_WARMUP_STEPS` | mean R² ≥ linear-FAST | **PASS** — linear `-0.2233` → RFF+warmup `0.0227` |
| 1b Residual RFF | `CYPHA_D14_RESIDUAL_RFF=1` (+ optional dim) | single-eq / mean R² lift | **PASS** — linear `-0.2233` → residual `0.5273`; residual+warmup RFF `0.5445` |
| 2 Expert utilization | soft NIG + entropy floor + `n_experts`; DIF blend into ngram GRIA (15%) | `n_experts≥4` @5k, BPC not exploded | **PARTIAL** — `n_experts=8` live, BPC ≈ char `4.04` |
| 3 Task-sticky CL | prefix-protect D + skip cross-task encoder; `CYPHA_D16_TASK_STICKY=1` | forgetting ≤ EWC | **FAIL** — sticky on/off both `0.0345` @ FAST |
| 4 BPE LM | `scripts/train_wikitext_bpe.py`, profile `d17_bpe` | BPC@5k < 4.04; stretch @300k < 2.873 | **FAIL** — real WikiText m300 vocab427: `5.12`@5k, `4.79`@30k, **`4.154`@300k** vs char pin **2.873**. STOP promoting BPE over char hybrid |
| 5 Causal | maturation→tau OnlineCorrelation + Granger-lite lag | smoke | **PASS** — fidelity `0.9116` |
| 6 Product demo | `scripts/demo_cypha_capabilities.ps1` | script runs | **PASS** (REST optional) |
| Paper PNGs | `scripts/render_native_paper_figures.py` | PNG exists | **PASS** — `native_fig_*.png` |

## Disposition

| Promote | Hold / reject |
|---------|----------------|
| **D14 residual RFF** as opt-in research (largest FAST R² lift) | Sticky CL — no FAST lift |
| D14 warmup recalib as opt-in stack with residual | Expert util — plumbing live, no BPC win @5k |
| Causal maturation edge in fidelity | Real WikiText BPE short-budget — worse than char until longer train |
| Paper native PNG figures | Do not default-on D14 kernel routing alone |

## Env cheat-sheet

```powershell
# Fork 1 + 1b
$env:CYPHA_BENCH_FAST=1
$env:CYPHA_D14_RESIDUAL_RFF='1'
$env:CYPHA_D14_RESIDUAL_RFF_DIM='256'
# optional stack:
$env:CYPHA_D14_KERNEL_BASIS='rff'
$env:CYPHA_D14_RFF_DIM='512'
$env:CYPHA_D14_KERNEL_BLEND='0.1'
$env:CYPHA_D14_KERNEL_CALIB_WARMUP_STEPS='100'

# Fork 2
$env:CYPHA_LM_N_EXPERTS='8'
$env:CYPHA_LM_SOFT_EXPERT_UPDATES='1'
$env:CYPHA_LM_ROUTING_ENTROPY_FLOOR='1'

# Fork 3
$env:CYPHA_D16_TASK_STICKY='1'

# Fork 4 (assets gitignored under bench/data/wikitext2/)
python scripts/train_wikitext_bpe.py --text-file bench/data/wikitext2/wikitext-2/wiki.train.tokens --out-dir bench/data/wikitext2/bpe --merges 300 --max-chars 500000
cyphalm_bench_native --profile d17_bpe --mode hybrid --n-train 5000 --n-eval 256 --threads 1
```

## Still human-gated

- `gh auth login` → `scripts/publish_release.ps1`
