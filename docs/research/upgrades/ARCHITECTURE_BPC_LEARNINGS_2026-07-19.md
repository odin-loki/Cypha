# Architecture BPC learnings (2026-07-19)

Goal: move WikiText-2 char BPC from ~2.88 toward <=2.0 via **architecture**, not knobs.

## What we learned

1. **Arithmetic coding is not a BPC algorithm.** It realizes the predictor's entropy. Neural-only coded BPC ~= eval_bpc.
2. **Production Hybrid is a collapsed 2-way blend** (~99.6% LSTM). Cell sweep (H19 etc.) did not beat 2.873 — local cell variants are exhausted.
3. **SOTA ~0.8–1.0 bpb** (Nacrith/CMIX) is mostly **pretrain + capacity + long context** (~65–75%); stealable codec pieces are ~25–35%.
4. **CMIX-style mixer on the codec path works** after gating (cold ~2.0 vs neural 6.0; warm WikiText gains shrink as the neural warms).
5. **Stacked residual LSTM wins at lock scale.** Lock protocol 300k / eval 2k / seed 42: L1 **2.859**, L2 **2.816** (Δ −0.043). Re-pinned production to **2.816** / `lstm_layers=2`.
6. **Width 128→256 lost at 40k** (L2 h256 **3.392** vs L2 h128 **3.369**). Do not promote.
7. **LSTM memory-attn = KILL.** Residual softmax over compressive/bank keys into `h` before `Wy` (`--lstm-memory-attn`) lost hard at 40k and stayed killed after a gated redesign. Default **OFF**; production recipe forces `use_lstm_memory_attn=false`. Do not promote. See [Verdict: LSTM memory-attn](#verdict-lstm-memory-attn-kill).
8. **Hidden-state kNN** for the codec path (`hidden_knn_log_probs`, `use_hidden_knn`) is on by default for compress; disable for eval_bpc identity checks.
9. Honest ceiling **without** a large pretrained LM: ~**2.0–2.3**. Sub-1.0 needs Nacrith/CMIX-class capacity.

## What landed (push ceiling wave)

| Piece | Role | Default |
|-------|------|---------|
| `AdaptivePredictorMixer` | Neural + gated n-grams + match (vote) + token-hash kNN | On for compress |
| Online adapt | `adapt_after_predict` @ lr_scale 0.25 | On for codec |
| Hidden kNN | `(h, next)` ring blended into neural before mixer | On for codec (`use_hidden_knn`) |
| Stacked LSTM | `lstm_layers` | **2** in D17 profile / lock |
| LSTM memory attn | Residual softmax over compressive slots → `h` before `Wy` | **KILL / Off** |

## Verdict: LSTM memory-attn (KILL)

**Recommendation: leave off forever unless a qualitatively different design beats L2 @40k by a clear margin.** Do not spend further 40k/300k CPU on scale/min_slots knobs.

| Variant @40k / eval 4k / seed 42 | eval_bpc | Δ vs L2 baseline |
|----------------------------------|----------|------------------|
| L2 h128 baseline | **3.369** | — |
| Raw mem-attn (`--lstm-memory-attn`) | 3.634 | **+0.265** (kill) |
| Gated (scale=0.10, min_slots=16, fill ramp) | 3.570 | **+0.201** (still kill) |

- Flag remains opt-in research only: `use_lstm_memory_attn` defaults `false`; CLI `--lstm-memory-attn`.
- `apply_hybrid_production_recipe` explicitly sets `use_lstm_memory_attn = false`.
- Not in D17 production pin / baseline lock.
- Further light blend tweaks deferred (CPU busy; gated already failed). Next memory path should be a different architecture (e.g. bank keys + learned readout, or external prior), not residual scale tuning.

Artifacts: `artifacts/profiles/push_L2_40k.txt`, `push_L2_attn_40k.txt`, `push2_L2_attn_gated_40k.txt`.

## Measured snapshot

### Capacity matrix @ 40k / eval 4k / seed 42

| Config | eval_bpc |
|--------|----------|
| L1 h128 | 3.382 |
| L2 h128 | **3.369** |
| L2 h256 | 3.392 (worse) |
| L2 + memory-attn (raw) | 3.634 (**KILL**) |
| L2 + memory-attn (gated) | 3.570 (**KILL**) |

### Lock protocol @ 300k / eval 2k / seed 42

| Config | eval_bpc |
|--------|----------|
| L1 h128 | 2.859 |
| L2 h128 | **2.816** (new pin) |
| Prior L1 pin | 2.873 |

### Codec (mixer + online adapt + hidden kNN)

| Train/eval | mix / neural / eval |
|------------|---------------------|
| 80k/8k L2 h128 | **2.546** / 2.557 / 2.818 |
| 80k/8k prior L1-ish h64 | 2.612 / 2.694 / 2.929 |
| Cold pattern (smoke) | ~2.02 / 6.00 |

Artifacts: `artifacts/profiles/push_*.txt`.

## Path remaining toward <=2.0

1. ~~Mixer + match/kNN + online adapt~~
2. ~~Stacked LSTM @ 2 layers on 300k + re-pin~~
3. ~~Residual LSTM memory-attn~~ — **KILL** (raw + gated @40k). Do not revisit scale knobs.
4. Qualitatively different long-context path that actually wins on WikiText (bank/learned readout, small from-scratch transformer, or external prior) — not residual mem-attn retunes.

## Commands

```powershell
predictive_codec_smoke
stacked_lstm_smoke
cyphalm_bench_native --profile d17 --mode hybrid --n-train 300000 --n-eval 2000 --bench-seed 42 --lstm-layers 2
predictive_codec_bench --n-train 80000 --n-eval 8000 --epochs 1 --lstm-layers 2 --lstm-hidden 128
# research only (KILL — do not promote):
cyphalm_bench_native ... --lstm-memory-attn
```
