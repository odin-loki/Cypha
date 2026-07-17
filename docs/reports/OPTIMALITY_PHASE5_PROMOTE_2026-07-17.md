# Optimality Phase 5b — Leverage / SORF Latent XOR A/B (2026-07-17)

**Build:** `native/build_perf_p5b` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Product wiring + latent-mode promotion decision. Did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight, or `CYPHA_*.md`.  
**Production default unchanged:** D03 `xor_pair` (~98%) remains the shipped path.

## Wiring (shipped)

| Surface | Flag / env | Effect |
|---------|------------|--------|
| `xor_kernel_bench` | `--nystrom-landmark-sampling {uniform,leverage}` | Opt-in `LandmarkSamplingKind::LeverageScore` + calib `init_leverage_landmarks_from_samples` |
| `xor_kernel_bench` | `--rff-projection {iid,sorf}` | Opt-in `KernelMemory::make_orthogonal_rff` when `--kernel-basis rff` |
| D03 env | `CYPHA_D03_NYSTROM_LANDMARK_SAMPLING=leverage` | Appends leverage flag to subprocess |
| D03 env | `CYPHA_D03_RFF_PROJECTION=sorf` | Appends SORF when `CYPHA_D03_KERNEL_BASIS=rff` |

Defaults remain `uniform` / `iid`. Unset env reproduces prior D03 xor_pair Nyström call.

## Latent-mode A/B (available numbers)

Protocol: `--kernel-feature-mode latent --kernel-m 256 --gamma-scale 1.0`, auto-gamma RFF where applicable.

| Config | Seeds × passes | Linear | Kernel | Δ vs linear | Notes |
|--------|----------------|--------|--------|-------------|-------|
| Nyström uniform (baseline) | 1 × 4 | 0.498 | **0.601** | +10.3 pp | Smoke budget; matches ~60% latent ballpark |
| Nyström leverage | — | — | — | — | Incomplete in this session (same setup queued) |
| RFF iid D=512 | — | — | — | — | Incomplete |
| RFF SORF D=512 | — | — | — | — | Incomplete |
| RFF iid D=1024 | — | — | — | — | Incomplete |
| RFF SORF D=1024 | — | — | — | — | Incomplete |

Phase 5 approximation smoke (prior report, not XOR accuracy): leverage Nyström ‖K̂−K‖ −14.5%, SORF RFF −13.1% vs iid — quality of sketch, not latent XOR accuracy.

Historical latent RFF (3×8, prior work): iid `rff_dim=4096` → **76.3%**; Nyström M=256 latent → ~62%. Those figures predate this promote pass and are not re-run here.

## Promotion decision

**Not promoted.** Available accuracy evidence does not show a ≥1 pp latent win for leverage or SORF over baseline. Approximation gains alone are insufficient without matched latent accuracy.

- Keep leverage / SORF **opt-in** via CLI + D03 env flags above.
- Keep **xor_pair** as production D03 default.
- Re-run full 3-seed × 8-pass latent matrix when budget allows; promote only if a config wins ≥1 pp accuracy (or same acc with clearly better ‖K̂−K‖ and no regression).

## How to reproduce

```bash
cmake -S native -B native/build_perf_p5b -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_perf_p5b -j8 --target xor_kernel_bench

# Latent baselines / opt-in
xor_kernel_bench --kernel-feature-mode latent --kernel-m 256 --gamma-scale 1.0 --seeds 3 --passes 8
xor_kernel_bench --kernel-feature-mode latent --kernel-m 256 --gamma-scale 1.0 --nystrom-landmark-sampling leverage --seeds 3 --passes 8
xor_kernel_bench --kernel-feature-mode latent --kernel-basis rff --rff-dim 512 --seeds 3 --passes 8
xor_kernel_bench --kernel-feature-mode latent --kernel-basis rff --rff-dim 512 --rff-projection sorf --seeds 3 --passes 8
```

## Files

- `native/tools/xor_kernel_bench.cpp` — CLI flags + leverage init / SORF construction
- `native/src/bench/bench_domains.cpp` — D03 env passthrough
- This report
