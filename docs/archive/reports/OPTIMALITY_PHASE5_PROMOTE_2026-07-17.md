# Optimality Phase 5b — Leverage / SORF Latent XOR A/B (2026-07-17)

**Build:** `native/build_perf_p5c` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Product wiring + latent-mode promotion decision. Did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight, or `CYPHA_*.md`.  
**Production default unchanged:** D03 `xor_pair` (~98%) remains the shipped path.  
**Budget:** FAST — primary cells are **1 seed × 4 passes** (RFF + Nyström uniform M=256). Nyström leverage at M=256 is pathologically slow online (per-step ridge-leverage Cholesky ≈ O(M⁴)); matched leverage A/B used **M=64, 1×2**.

## Wiring (shipped)

| Surface | Flag / env | Effect |
|---------|------------|--------|
| `xor_kernel_bench` | `--nystrom-landmark-sampling {uniform,leverage}` | Opt-in `LandmarkSamplingKind::LeverageScore` + calib `init_leverage_landmarks_from_samples` |
| `xor_kernel_bench` | `--rff-projection {iid,sorf}` | Opt-in `KernelMemory::make_orthogonal_rff` when `--kernel-basis rff` |
| D03 env | `CYPHA_D03_NYSTROM_LANDMARK_SAMPLING=leverage` | Appends leverage flag to subprocess |
| D03 env | `CYPHA_D03_RFF_PROJECTION=sorf` | Appends SORF when `CYPHA_D03_KERNEL_BASIS=rff` |

Defaults remain `uniform` / `iid`. Unset env reproduces prior D03 xor_pair Nyström call.

## Latent-mode A/B (accuracy)

Protocol: `--kernel-feature-mode latent`, auto-gamma RFF where applicable. Binary: `native/build_perf_p5c/xor_kernel_bench.exe`.

### Primary (matched 1 seed × 4 passes)

| Config | Seeds × passes | Linear | Kernel | Δ vs linear | vs baseline |
|--------|----------------|--------|--------|-------------|-------------|
| Nyström uniform M=256 γ-scale=1.0 (baseline) | 1 × 4 | 0.498 | **0.601** | +10.3 pp | — |
| Nyström leverage M=256 | 1 × 4 | — | — | — | **Incomplete** — online leverage reservoir too slow at M=256 (no finish in >8 min / pass) |
| RFF iid D=512 | 1 × 4 | 0.498 | **0.672** | +17.4 pp | +7.1 pp vs Nyström uniform |
| RFF SORF D=512 | 1 × 4 | 0.498 | **0.665** | +16.7 pp | **−0.7 pp vs RFF iid** |
| RFF iid / SORF D=1024 | — | — | — | — | Skipped (FAST budget) |

### Nyström leverage matched mini-A/B (M=64, 1×2)

Forced by M=256 leverage wall-clock. Same seed/pass budget for both arms:

| Config | Seeds × passes | Linear | Kernel | Δ vs linear | vs uniform |
|--------|----------------|--------|--------|-------------|------------|
| Nyström uniform M=64 | 1 × 2 | 0.514 | 0.489 | −2.5 pp | — |
| Nyström leverage M=64 | 1 × 2 | 0.514 | 0.490 | −2.4 pp | **+0.1 pp** |

Phase 5 approximation smoke (prior report, not XOR accuracy): leverage Nyström ‖K̂−K‖ −14.5%, SORF RFF −13.1% vs iid — quality of sketch, not latent XOR accuracy.

Historical latent RFF (3×8, prior work): iid `rff_dim=4096` → **76.3%**; Nyström M=256 latent → ~62%. Those figures predate this promote pass.

## Promotion decision

**Not promoted.** No config meets the ≥1 pp latent win bar for changing recommended defaults:

| Candidate | Evidence | ≥1 pp win? |
|-----------|----------|------------|
| Nyström leverage vs uniform | +0.1 pp at M=64 (1×2); M=256 unfinished | **No** |
| RFF SORF vs iid (D=512) | 0.665 vs 0.672 (−0.7 pp) | **No** |

Notes (informational only — not a default flip):

- Latent RFF iid D=512 (0.672) beats latent Nyström uniform M=256 (0.601) by +7.1 pp on this FAST cell, but that is a basis choice already available via `--kernel-basis rff` / D03 env; this promote pass was about leverage/SORF sampling, not flipping the latent Nyström default.
- Keep leverage / SORF **opt-in** via CLI + D03 env flags above.
- Keep **xor_pair** as production D03 default.
- Do **not** change latent recommended defaults in docs/env.

## How to reproduce

```bash
cmake -S native -B native/build_perf_p5c -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_perf_p5c -j8 --target xor_kernel_bench

# Primary FAST cells (1×4)
xor_kernel_bench --kernel-feature-mode latent --kernel-m 256 --gamma-scale 1.0 --seeds 1 --passes 4
xor_kernel_bench --kernel-feature-mode latent --kernel-basis rff --rff-dim 512 --seeds 1 --passes 4
xor_kernel_bench --kernel-feature-mode latent --kernel-basis rff --rff-dim 512 --rff-projection sorf --seeds 1 --passes 4

# Matched leverage mini-A/B (M=64; M=256 leverage online is impractical)
xor_kernel_bench --kernel-feature-mode latent --kernel-m 64 --gamma-scale 1.0 --seeds 1 --passes 2
xor_kernel_bench --kernel-feature-mode latent --kernel-m 64 --gamma-scale 1.0 --nystrom-landmark-sampling leverage --seeds 1 --passes 2
```

## Files

- `native/tools/xor_kernel_bench.cpp` — CLI flags + leverage init / SORF construction
- `native/src/bench/bench_domains.cpp` — D03 env passthrough
- This report
