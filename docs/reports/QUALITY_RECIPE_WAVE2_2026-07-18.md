# Quality recipe Wave 2 (2026-07-18)

**Ledger:** [`ROADMAP_EXECUTION_LEDGER_2026-07-18.md`](ROADMAP_EXECUTION_LEDGER_2026-07-18.md)  
**Prior:** [`QUALITY_RECIPE_WAVE1_2026-07-18.md`](QUALITY_RECIPE_WAVE1_2026-07-18.md)  
**Scope:** Wave-2 substrate (AdamW, LR schedule, score_matrix `inv_v` fold) plus FAST/20k recipe A/B and Adam LR sweep. **Defaults unchanged** — D17 pin **2.873** path remains BPTT-1 / SGD / N(0,0.02).

## Wave-2 code landing (opt-in OFF)

| Knob | CLI | Env | Default |
|------|-----|-----|---------|
| AdamW weight decay | `--weight-decay W` | `CYPHA_LSTM_WEIGHT_DECAY` | `0` |
| LR linear warmup | `--lstm-lr-warmup N` | `CYPHA_LSTM_LR_WARMUP` | `0` |
| LR cosine after warmup | `--lstm-lr-cosine N` | `CYPHA_LSTM_LR_COSINE` | `0` |
| LSTM LR override | `--lstm-lr LR` | `CYPHA_LSTM_LR` | profile (`0.05`) |

Also: **score_matrix `inv_v` fold** — CTest `native_score_matrix_inv_v_fold_smoke`.  
Smoke: `native_quality_recipe_wave2_smoke`.

## Measured BPC (hybrid d17, seed=42, n_eval=256)

| Run | Config | n_train | BPC | Δ vs matched baseline |
|-----|--------|---------|-----|------------------------|
| FAST baseline | default | 2000 | **4.728** | — |
| FAST recipe | bptt=8 Adam clip=1 classic **lr=0.05** | 2000 | **5.294** | +0.566 worse |
| 20k baseline | default | 20000 | **3.475** | — |
| 20k recipe | bptt=8 Adam clip=1 classic **lr=0.05** | 20000 | **5.866** | +2.391 worse |
| 20k recipe | … **lr=0.001** | 20000 | **3.365** | **−0.110 better** |
| 20k recipe | … lr=0.005 | 20000 | 3.691 | +0.216 worse |
| 20k recipe | … lr=0.01 | 20000 | 3.820 | +0.345 worse |

Logs: `bench/results/recipe_wave2/`.

## Promote decision

- Profile `lstm_lr=0.05` is **SGD-tuned**; Adam at that LR collapses quality.
- **Candidate (opt-in only):** `bptt=8 --optim adam --grad-clip 1.0 --lstm-init classic --lstm-lr 0.001` — wins at 20k by ~110 mBPC.
- **Do not flip defaults** / do not change D17 lock until a 300k matched run confirms.
- Wave-2 substrate (AdamW / schedule / inv_v fold) stays opt-in OFF.

## STOP reminder

Do not reopen BoW STOPs. Do not promote recipe-as-default before 300k evidence.

## Next steps

1. Matched 300k A/B: baseline vs candidate (`--lstm-lr 0.001`).
2. If 300k wins with margin: consider re-pin + cell re-sweep (§1.7); else keep pin and treat recipe as research opt-in.
3. Optional: warmup/cosine / weight-decay ablations on the candidate LR.
