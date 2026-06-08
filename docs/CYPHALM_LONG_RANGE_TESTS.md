# CyphaLM Long-Range Context Tests

**Maps:** [`Cypha Tests.txt`](../Cypha%20Tests.txt) Phase 1 → CyphaLM empirical verification  
**Runner:** `python cypha_bench/tuning/cyphalm_long_range_suite.py --write --figures`  
**D17:** experiment **17K_long_range_context**

---

## Cypha Tests.txt → CyphaLM mapping

| Cypha Tests experiment | CyphaLM implementation | Pass criterion |
|------------------------|------------------------|----------------|
| **1A** Sequential vs shuffled | `eval_shuffled_stream_bpc` — forward vs block-shuffled eval | Shuffled **worse** (order matters) |
| **1A** Field active vs zeroed | `eval_ssm_ablation_sequential` — `gria_ngram` vs `ablation_no_ssm` | SSM path **lower BPC** than embed-only |
| **1C** Field as context vector | `eval_context_length_extended` — BPC vs warm-up length | BPC **falls** as context window grows (SSM uses history) |
| Long-range memory | `eval_reset_interval_bpc` — reset every N tokens | BPC **rises** when reset often (needs long context) |

Phase 2–3 (Hebbian biochem, next-token head) → [`CYPHALM_UPGRADE_V2.md`](CYPHALM_UPGRADE_V2.md) + [`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md).

---

## Suite outputs

Artifact: `cypha_bench/config/cyphalm_long_range_suite.json`

| Block | Meaning |
|-------|---------|
| `context_length_bpc` | BPC after warming SSM with last `L` tokens before each prediction |
| `reset_interval_bpc` | BPC when memory cleared every N eval steps (`never` = full stream) |
| `sequential_vs_shuffled` | Same tokens, forward vs shuffled 512-blocks |
| `ssm_ablation_sequential` | Retrain per mode; `ssm_contribution_bpc` = no_ssm − gria_ngram |

Figures: `fig17_long_range_context_bpc.png`, `fig17_long_range_reset_interval.png`

---

## Results @ 40k (WikiText-2, profile d17)

| Test | Result | Cypha Tests verdict |
|------|--------|---------------------|
| Held-out BPC | **4.023** | — |
| Context warm-up | **4.14 @ 8 → 3.77 @ 128** (spike 4.51 @ 256) | **1C pass** — SSM state improves prediction with history |
| Reset interval | **4.02 never → 4.18 @ reset=8** | **Pass** — frequent reset hurts (+0.16 BPC) |
| Block shuffle | forward 4.023 vs shuffled 4.022 (−0.001) | **1A fail @ block level** — order insensitive (n-gram head may dominate) |
| SSM ablation | gria 4.023 vs no_ssm 4.019; ssm_only 4.477 | **1A marginal** — SSM adds ~0 at stream eval; n-gram fusion essential |

## Results @ 300k (convergence peak, context-only)

Artifact: `cypha_bench/config/cyphalm_long_range_300k.json`

| Test | Result | vs 40k |
|------|--------|--------|
| Held-out BPC | **3.839** | −0.18 (matches stack validation **3.838**) |
| Context warm-up | **4.06 @ 8 → 3.64 @ 128** | Stronger monotonic drop; best window still **128** |
| Reset interval | **3.84 never → 4.09 @ reset=8** | **+0.26 BPC** when memory cleared every 8 tokens (stronger than @ 40k) |
| Block shuffle | +0.002 BPC | Still flat — sequential structure not captured at block scale |

**Interpretation @ 300k:** The recurrent field **clearly carries long-range context** (warm-up and reset probes). Stream BPC and block-shuffle still barely separate SSM from n-gram path — the **gria_ngram** head dominates standard LM eval. Long-range value shows up in **context-conditioned** probes, not aggregate BPC alone.

---

## Commands

```powershell
# Fast smoke (8k train)
python cypha_bench/tuning/cyphalm_long_range_suite.py --fast --write --figures

# Full (40k train + SSM ablation)
python cypha_bench/tuning/cyphalm_long_range_suite.py --write --figures

# Convergence peak — context only (~40 min train)
python cypha_bench/tuning/cyphalm_long_range_suite.py --n-train 300000 --skip-ablation --write --figures --out cypha_bench/config/cyphalm_long_range_300k.json

# D17 bench (includes 17K when not fast)
python cypha_bench/run_all.py --domain 17
```

Phase 1c full corpus: set `CYPHA_BENCH_FULL_N_TRAIN=300000` (default) with `CYPHA_BENCH_FULL_CORPUS=1`.

---

## Concurrent tracks

Runs in parallel with Phase 1c full corpus and Upgrade V2 / model-class research.
