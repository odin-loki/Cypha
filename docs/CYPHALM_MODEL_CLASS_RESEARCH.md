# CyphaLM Model-Class Research — Char-LSTM & Hybrids

**Status:** Research started (parallel with Upgrade V2)  
**Last updated:** 2026-06-06  
**Related:** [`CYPHALM_UPGRADE_V2.md`](CYPHALM_UPGRADE_V2.md), [`cypha_bench/adapters/char_lstm_baseline.py`](../cypha_bench/adapters/char_lstm_baseline.py)

---

## Why a second model class?

| Model | D17 @ 40k BPC | vs bigram |
|-------|---------------|-----------|
| CyphaLM (best stack @ 300k) | **3.838** | +0.27 |
| Bigram | ~3.56–3.91 | — |
| Trigram | 4.40 | CyphaLM wins |
| **Char-LSTM** (bench baseline) | **3.589** | **beats bigram** |

CyphaLM **beats trigram** but **char-LSTM beats CyphaLM by ~0.25–0.55 BPC** on the same eval harness. That gap is larger than the bigram gap — closing it may require **recurrent depth** (LSTM/GRU) or **hybrid**, not more GRIA sweeps.

---

## Existing char-LSTM baseline

**Location:** `cypha_bench/adapters/char_lstm_baseline.py`

| Property | Value |
|----------|-------|
| Architecture | 1-layer LSTM, hidden=128 default |
| Training | Online BPTT-1 per step |
| Backend | NumPy only (bench baseline, not production) |
| Integration | D04/D17 ablation block when not `CYPHA_BENCH_FAST` |

**Not in CyphaLM package** — separate code path, not sharing SSM/DIF/GRIA.

---

## Research questions

1. **Why does LSTM win?** Long-range dependency via gated recurrence vs GRIA linear projection + n-gram concat.
2. **Can CyphaLM reuse LSTM as head only?** SSM → LSTM → vocab (drop GRIA or blend).
3. **Can we ensemble?** `log p = α log p_GRIA + (1-α) log p_LSTM` with online α.
4. **Does Cypha online machinery add value on top of LSTM?** DIF experts on LSTM hidden state.
5. **Training budget parity** — char-LSTM baseline uses same `n_train`; confirm fair comparison @ 300k.

---

## Candidate architectures (evaluate in parallel)

### C1 — Char-LSTM head (replace GRIA)

```text
token → IzaacEmbed → SSM → h → CharLSTMCell → logits
```

- Keep CyphaLM training loop; swap `GRIAProjection` for small LSTM head
- **Pros:** Simplest path to char-LSTM parity
- **Cons:** Loses GRIA interpretability / Laplace bias

### C2 — Dual head (GRIA + LSTM)

```text
SSM context → GRIA → log p_g
            ↘ LSTM → log p_l
log p = logaddexp(log p_g + log w_g, log p_l + log w_l)
```

- Online learn `w_g`, `w_l` (2 scalars or per-token)
- **Pros:** Keeps current best path; adds LSTM correction
- **Cons:** 2× compute per step

### C3 — LSTM on embed history only (n-gram path upgrade)

```text
embed history → LSTM → h_n → fuse with field_x → GRIA
```

- Upgrades Track B fusion with recurrence over embeds (not full vocab LSTM)
- **Pros:** Stays close to `gria_ngram` ablation winner
- **Cons:** May not match full LSTM without wider hidden

### C4 — Promote bench LSTM into `cypha_lm`

- Move `_CharLSTM` → `cypha_lm/model/char_lstm_head.py`
- `context_mode=char_lstm` or `hybrid_gria_lstm`
- Unified config, save/load, profiles

---

## Evaluation protocol

| Step | Action | Metric |
|------|--------|--------|
| 1 | Char-LSTM @ **300k** WikiText (match CyphaLM budget) | BPC |
| 2 | Char-LSTM @ **full corpus** (`CYPHA_BENCH_FULL_CORPUS=1`) | BPC |
| 3 | Implement **C2 dual head** minimal prototype | BPC vs GRIA-only |
| 4 | Implement **C3** if C2 too heavy | BPC |
| 5 | Profile size, train_seconds, active_experts | Cost table |

**Success (model-class track):** Match or beat char-LSTM **3.589** @ 40k equivalent budget, or beat bigram with hybrid @ 300k.

---

## Implementation plan

### Phase M1 — Benchmark parity (1–2 days)

- [x] Script `cyphalm_char_lstm_extended.py` — char-LSTM @ 40k/70k/150k/300k
- [x] Results in `cyphalm_char_lstm_extended.json`
- [ ] Confirm char-LSTM @ full corpus (Phase 1c companion)

**M1 results (WikiText valid, 2026-06-06):**

| n_train | Char-LSTM BPC | vs bigram | CyphaLM stack ref |
|---------|---------------|-----------|-------------------|
| 40k | 3.615 | **−0.30** | ~4.04 (frozen α) |
| 70k | 3.390 | **−0.36** | ~3.92 |
| 150k | 3.193 | **−0.42** | — |
| **300k** | **3.098** | **−0.47** | **3.838 (+0.27)** |

Char-LSTM **beats bigram at every budget** and improves monotonically to 300k. Gap vs CyphaLM best @ 300k: **~0.74 BPC** — hybrid/LSTM head justified.

### Phase M2 — Dual head prototype (3–5 days)

- [ ] `CharLSTMHead` in `cypha_lm/model/char_lstm_head.py`
- [ ] `context_mode=hybrid_gria_lstm`
- [ ] D17 **17J_hybrid_lstm**
- [ ] Online blend weights

### Phase M3 — Integration decision

| Outcome | Action |
|---------|--------|
| Hybrid beats GRIA-only | New default profile `hybrid_gria_lstm` |
| LSTM-only wins | `context_mode=char_lstm` profile |
| No gain | Document ceiling; focus on Upgrade V2 only |

---

## Parallel with Upgrade V2

| Track | Focus | Runs on |
|-------|-------|---------|
| **V2** | Learnable views + n-gram fusion | Existing GRIA + `gria_ngram` |
| **Model class** | LSTM / hybrid heads | Same corpora, same `n_train` grid |

Merge point: **dual head (C2)** can combine V2 fusion output with LSTM hidden state.

---

## References

- Bench char-LSTM: `cypha_bench/adapters/char_lstm_baseline.py`
- CyphaLM ablations: `cypha_bench/config/cyphalm_component_ablation.json`
- Stack validation: `cypha_bench/config/cyphalm_stack_validation.json`
