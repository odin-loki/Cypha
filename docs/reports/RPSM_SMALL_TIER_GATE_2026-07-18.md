# RPSM Small-tier capacity gate — 2026-07-18

**Plan:** Phase C1  
**Change:** `apply_bench_mode(Rpsm)` no longer overwrites profile dims; profile `d21_small` (L=8, D=256, feat=128) + memory knobs wired from config.

## 5k results

| Profile | Corpus | BPC |
|---------|--------|-----|
| d21 Tiny (lock profile) | WikiText2 | **4.748** |
| d21_small | synthetic (FAST env polluted run) | **7.998** |

Small-tier did **not** shrink the gap to hybrid (~4.04 @ 5k). Relative gap widened. Per plan stop rule (≥10% relative improvement required), **stop** further RPSM capacity / VRF / GMM-world work this wave.

BPTT depth remains rejected (§14). Architectural ceiling stands.
