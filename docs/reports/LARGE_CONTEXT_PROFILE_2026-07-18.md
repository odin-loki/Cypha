# Large-context / large-data Cypha profiling (2026-07-18)

**Status:** Medium suite **in progress**  
**Artifacts:** `bench/results/large_context_profile/Medium_20260718_210114/`  
**Runner:** `scripts/run_large_context_profile.ps1 -Tier Medium -BuildDir native/build-pgm`

## What “larger context” means here

| Axis | Tool | Medium tier |
|------|------|-------------|
| PGM slot capacity `N` | `pgm_cell_bench --scale large` | up to **262144** slots |
| Train data | `cyphalm_sample_efficiency_curve` | **10k / 40k / 100k** (hybrid + H23) |
| Sequence length | `cyphalm_needle_haystack` | **512 / 1024 / 2048** chars |

Larger tiers: `-Tier Large` adds 300k BPC + haystack 4k + `CYPHA_PERF_TRACE` @ 40k; `-Tier XL` adds PGM xl + haystack 8k + hidden=256 spot check.

## Phase 1 — PGM capacity (complete)

`pgm_cell_bench --scale large --steps 20000` (warmup 500):

| Config | d | N | med µs/step | p95 µs | tok/s | occupied |
|--------|--:|--:|------------:|-------:|------:|---------:|
| CharLSTM D17-like | 128 | dense | 160.8 | 175 | 6.1k | — |
| PGM H23-ish (b=8 L=3) | 64 | 512 | **14.2** | 357 | 25.5k | 502 |
| PGM field_dim | 160 | 512 | 36.3 | 1007 | 9.4k | 512 |
| PGM (b=16 L=3) | 64 | 4096 | 31.0 | 872 | 11.4k | 770 |
| PGM (b=8 L=4) | 64 | 4096 | 21.1 | 566 | 16.8k | 793 |
| PGM (b=8 L=5) | 64 | 32768 | 28.5 | 784 | 12.5k | 969 |
| PGM (b=16 L=4) | 64 | 65536 | 44.6 | 1262 | 7.9k | 976 |
| PGM wide+65k | 128 | 65536 | 93.3 | 2695 | 3.7k | 1825 |
| PGM (b=8 L=6) | 64 | **262144** | **35.7** | 1007 | 9.8k | 981 |
| PGM (b=64 L=3) | 64 | **262144** | 108.3 | 3228 | 3.2k | 874 |

### Takeaways (capacity)

1. **PGM stays faster than CharLSTM-128 even at the 262k slot cap** when addressed with deep hierarchy (`b=8 L=6`: ~36 µs vs ~161 µs median).
2. **Addressing shape matters more than N:** same 262k slots via `b=64 L=3` is ~3× slower than `b=8 L=6` (address cost ∝ `levels · n_sub · d`).
3. **p95 ≫ median** on PGM (rehash / sparse growth spikes). Occupancy after 20k steps stays ≪ N — capacity is not saturated at these step counts.
4. Width hurts more than depth: `d=128 @ 65k` (~93 µs) is slower than `d=64 @ 262k` (~36 µs).

## Phase 2–3 — data + sequence length (running)

- Sample-efficiency BPC: hybrid then H23 at `n_train ∈ {10k, 40k, 100k}`, `n_eval=2000`, full WikiText (`CYPHA_BENCH_FULL_CORPUS=1`).
- Needle-haystack: depths 512 / 1024 / 2048.

Prior PGM BPC compare only went to 40k (`PGM_BPC_COMPARE_2026-07-18.md`); this run extends to **100k**.

## How to continue

```powershell
# Medium (this run) — already started
powershell -File scripts\run_large_context_profile.ps1 -Tier Medium -BuildDir native/build-pgm

# After Medium: production-scale data + longer haystacks
powershell -File scripts\run_large_context_profile.ps1 -Tier Large -BuildDir native/build-pgm -SkipBuild

# PGM-only capacity XL (fast)
.\native\build-pgm\pgm_cell_bench.exe --scale xl --steps 20000
```

## Next update

Fill Phase 2 BPC table and Phase 3 recall rates when `suite.log` reports `SUITE COMPLETE`.
