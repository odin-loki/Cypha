# `paper/figures/` — status note (2026-07-12)

The `exp01_embedding_metrics.json` … `exp10_efficiency_metrics.json` files in this directory are
**historical artifacts of the decommissioned Python prototype** (`cypha_lm/`, removed in Phase P7 —
see `CHANGELOG.md`, "[Unreleased] > Removed"). They were generated against a NumPy-based,
gradient-free reference implementation that no longer exists in this repository, on tiny synthetic
tasks (vocab size 512, sequences up to length 100, 1k–20k training steps).

They are **kept for historical reference only** and are cited that way throughout
`paper/CyphaLM_paper.md` (§3, §5) — none of the numbers in these files should be read as current
measurements of the native C++ CyphaLM (`native/src/cyphalm/`, `native/include/cypha/cyphalm/`).

No corresponding PNG figures ever existed in this directory (only the JSON metric files above); the
paper draft's old `Figure N (...png)` references were aspirational placeholders, not real files —
this has been corrected in the rewritten paper to cite the JSON files directly (as historical) or the
current native bench artifacts instead, per subsystem:

| Old figure | Historical JSON (Python prototype, superseded) | Current native equivalent (if any) |
|---|---|---|
| Fig 1 — embedding benchmark | `exp01_embedding_metrics.json` | Not re-measured; collision-rate-zero is re-derivable as a structural guarantee of the native GF(2^n) construction — see paper §3.1 |
| Fig 2 — SSM capacity | `exp02_ssm_metrics.json` | `bench/BASELINE_REPORT.md` D17 `ssm_diagnose` block — see paper §3.2 |
| Fig 3 — expert self-organisation | `exp03_experts_metrics.json` | Not re-measured on an equivalent task; closest analogue is D16A routing ARI in `docs/RESEARCH_STATUS.md` — see paper §3.3 |
| Fig 4 — alpha spectrum | `exp04_alpha_metrics.json` | `docs/RESEARCH_STATUS.md` post-upgrade D17B mean-α reading — see paper §3.4 |
| Fig 5/6 — toy LM / code LM | `exp05_toy_lm_metrics.json`, `exp06_code_lm_metrics.json` | Superseded by D04/D17 native bench — see paper §5.5 |
| Fig 7 — uncertainty calibration | `exp07_calibration_metrics.json` | Not re-measured (ECE); nearest native evidence is OOD AUROC in `docs/RESEARCH_STATUS.md` — see paper §5.6 |
| Fig 8 — online adaptation | `exp08_adaptation_metrics.json` | D17D ΔBPC in `docs/RESEARCH_STATUS.md` — see paper §5.7 |
| Fig 9 — catastrophic forgetting | `exp09_forgetting_metrics.json` | D16B/D16E/D16F forgetting scores in `docs/RESEARCH_STATUS.md` — see paper §5.8 |
| Fig 10 — parameter efficiency | `exp10_efficiency_metrics.json` | Not meaningful even historically (near-zero transformer BPC on a memorised synthetic task); current native efficiency evidence is training throughput in `docs/reports/PERFORMANCE_PROFILE_2026-07-12.md` — see paper §5.9 |

Regenerating these figures from current native `bench/results/` data (rather than only citing tables
in prose) is future work, not performed as part of this 2026-07-12 rewrite.

## Native figure payloads (2026-07-18)

JSON summaries + PNG bars (render: `python scripts/render_native_paper_figures.py`). Sources: production lock + RESEARCH_STATUS:

| File | Content |
|------|---------|
| `native_fig_d17_bpc.json` / `.png` | D17 BPC vs bigram / LSTM / GRIA / RPSM |
| `native_fig_d17b_alpha.json` / `.png` | D17B mean_alpha / n_experts |
| `native_fig_d16_forgetting.json` / `.png` | D16B/EWC/16F forgetting scores |

Submit (2027 Q1) still needs full bibliography; native JSON+PNG figures are landed.
