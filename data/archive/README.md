# data/archive/

Pinned traces that document a lock or a published experiment. Live runs do **not** write here.

## profiles/

| File | Why it is here |
|------|----------------|
| `push2_L2_bptt_300k.txt` | Living D17 pin source (2.664 BPC, 2026-08-08) |
| `push_L2_300k.txt` | Prior SGD L2 pin (2.816) |
| `push_L1_300k.txt` / `d17_layers1_300k.txt` | Prior L1 pin (2.873) |
| `d17_hybrid_*` / `d17_layers*` / `push*` / `codec_bench_*` | Recipe ladder (5k–300k, Wave2, codec) |
| `xor_kernel_llr*.json` | XOR Nyström / RFF / blend sweeps |
| `benchmark_*_20260719*` | July 19 core/regression bench captures |
| `d01_sorf_profile_full.txt` | D01 SORF profile dump |

## cell_sweep/

| File | Why it is here |
|------|----------------|
| `variant_B2.json` | B2 hybrid control rerun @ 300k / eval 2k (2026-08-16); **3.681 BPC**, math-integration on — not a promote vs Hybrid 2.664 |
| `variant_H06.json` | H06 NIG-state rerun @ 300k / eval 2k (2026-08-16); **3.681 BPC**, same as B2 under this recipe — not a promote |
| `variant_H15.json` | Preserved H15 @ 300k (not a promote; see paper `RESULTS_ATTEST.md`) |

Live cell-sweep checkpoints: `bench/results/cell_sweep/variant_*.json` (gitignored).
