# CyphaLM Training Findings (2026-05-31 — 2026-06-01)

Canonical log of what we learned from iteration sweeps and convergence search. Related: [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md), [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

---

## Baseline (pre multi-view, 40k cap)

| Metric | D17 WikiText | Notes |
|--------|--------------|-------|
| CyphaLM (`gria_ngram`, `same_order`) | **4.154 BPC** | Beats trigram **4.398** (−0.24) |
| Bigram | **3.914 BPC** | +0.24 gap to close |
| Trigram | 4.398 BPC | ✅ beaten |
| Char-LSTM | **3.589 BPC** | Strongest baseline |

The **40k token train cap** was a bench default, not a convergence study.

---

## Multi-view schedules

| Preset | Views | Meaning |
|--------|-------|---------|
| `same_order` | forward × `train_epochs` | Static left-to-right; 2 epochs = two forward passes |
| `schedule_a` | forward → block_shuffle | Block shuffle only (512-token blocks) |
| `schedule_b` | forward → block_shuffle → rotated | Shuffle + rotate start (~25% offset) |
| `schedule_c` | forward → block_shuffle → backward | Adds reverse pass |

`block_shuffle` preserves within-block char order; shuffles block order only. Fast memory resets per block on shuffle views.

---

## Iteration × view sweep (2k–40k, 32 runs)

Artifact: `cypha_bench/config/cyphalm_view_iteration_sweep.json`

1. **`schedule_b` wins mid-train** — beats `same_order_e2` at every **n_train ≤ 32k**.
2. **One-pass overtrains** — `same_order_e1` crosses above bigram between ~12k–16k.
3. **40k looked optimal only because we stopped there** — `same_order_e2` @ 40k was **4.067 BPC** (best in that grid).

---

## Convergence limit sweep (40k–250k, 10M-token WikiText cap)

Artifact: `cypha_bench/config/cyphalm_convergence_limit.json`  
Script: `python cypha_bench/tuning/cyphalm_convergence_limit.py --write`

| Mode | Converged? | Training limit | Best BPC | @ n_train | vs bigram @ that n |
|------|------------|----------------|----------|-----------|---------------------|
| **same_order_e2** | **Yes** | **~40k** (overtrain by 50k) | **4.094** | 40k | +0.18 |
| **schedule_b** | **Yes** | **~300k** (worse by 400k) | **3.905** | 300k | +0.34 |

### same_order_e2

- Peaks at **40k**, BPC **rises** at 50k (4.112) and 60k (4.124).
- Early stop flagged **overtrain @ ~50k**.
- **Do not train same-order past 40k.**

### schedule_b

- 250k sweep: best **3.936** BPC, looked still improving.
- **Extended 300k–500k** (`cyphalm_convergence_continue.json`): peaks @ **300k (3.905)**, then **regresses** at 400k (3.916) and 500k (3.948).
- **Converged ~300k** — do not train schedule_b past 300k on this corpus.
- Curve remains noisy (local min ~70k @ 3.985; 250k @ 3.936; global min 300k @ 3.905).
- **Not yet beaten bigram** at any checkpoint (+0.34 best @ 300k).

---

## Implications for beat-bigram

| Lever | Status |
|-------|--------|
| Multi-view (`schedule_b`) | ✅ Helps vs same-order; extends useful train budget beyond 40k |
| Same-order + 2 epochs | ❌ Overtrains after 40k — cap or switch schedule |
| More tokens alone | ❌ Peaks @ 300k for schedule_b; not a path to beat bigram |
| Full official WikiText train | Phase 1c — `CYPHA_BENCH_FULL_CORPUS=1` runs **17A/17B/17D/17K** @ 300k |
| Learnable view embeddings | ❌ Neutral @ 300k — keep fixed (`CYPHALM_UPGRADE_V2` Track A) |
| Stronger n-gram fusion (gated) | ❌ Worse @ 40k — keep sum fusion (Track B) |
| **Hybrid GRIA+LSTM (C2)** | ✅ **2.870 BPC @ 300k** — beats bigram & char-LSTM; **default D17 profile** |

## Hybrid dual head @ 300k (2026-06-07)

Artifact: `cypha_bench/config/cyphalm_hybrid_lstm_300k.json`

| Model | BPC @ 300k | vs bigram |
|-------|------------|-----------|
| GRIA-only (`gria_ngram`) | 3.842 | +0.364 |
| **Hybrid (`hybrid_gria_lstm`)** | **2.870** | **−0.608** |
| Char-LSTM baseline | 2.979 | −0.108 vs hybrid |

Blend weight learns ~**99.6% LSTM**. Profile: `cyphalm_d17_wikitext.json` → `context_mode=hybrid_gria_lstm`.

**Phase 1c 17A confirm (2026-06-07):** **2.873 BPC** @ 300k hybrid via streamlined 17A (`train_sequence`, ~50 min).

## Beat-bigram hyperparam sweep (2026-06-01)

Artifact: `cypha_bench/config/cyphalm_beat_bigram_sweep.json`  
Script: `python cypha_bench/tuning/cyphalm_beat_bigram_sweep.py --write`

| n_train | Best BPC | vs bigram | Notes |
|---------|----------|-----------|-------|
| 70k | **3.980** | +0.23 | **Best overall** in this sweep |
| 100k | 4.019 | +0.34 | Noisy spike (matches convergence sweep) |
| 150k | 3.953 | +0.33 | Lower absolute BPC, bigram also stronger |

- **`gria_lr` / `tau_slow` do not matter** — all 9 combos within ~0.01 BPC at each n_train.
- **Did not beat bigram** at any checkpoint.
- **Hyperparam tuning alone is insufficient** — axis sweep @ 70k (`cyphalm_beat_bigram_axes.json`):
  - Best: **`gria_lr_decay=0.3`** → 3.966 BPC (+0.22 vs bigram); no decay (1.0) hurts (4.044)
  - **`ngram_context=3`** slightly beats 2 (3.967 vs 3.985)
  - Laplace / train_ssm: flat at 70k
  - **`schedule_c` fails** (4.540 BPC); schedule_b still best schedule

---

## Commands

```powershell
python cypha_bench/tuning/cyphalm_view_iteration_sweep.py --write
python cypha_bench/tuning/cyphalm_convergence_limit.py --write
python cypha_bench/tuning/cyphalm_sweep.py --corpus wikitext --n-train 8000 --write-profile
```
