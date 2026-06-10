# CyphaLM Native — Tier 2 Model Class

**Status:** implemented (2026-06-10)  
**Library target:** `cypha_lm_native`  
**Parity tool:** `cyphalm_model_parity`

## Overview

Tier 2 adds a unified native CyphaLM stack under `native/include/cypha/cyphalm/` with selectable **context modes**, a **Mamba-lite selective SSM** head, **compressive memory** slots, and an inference-only **BPE tokenizer** stub.

```
token ─► EmbedTable ─► CellAISSM ─► field projection ─► CompressiveMemory bias
                              │                              │
                              └► SelectiveSSM (optional) ────┘
                                              │
                                              ▼
                                    LowRankGRIA ─► log_probs
                              CharLSTM (hybrid / char_lstm modes)
```

## Context modes (`ContextMode`)

| Enum | Python alias | Path |
|------|--------------|------|
| `Full` | `full` | SSM → DIF-style epistemic blend → GRIA + SelectiveSSM |
| `GriaNgram` | `gria_ngram` | SSM field + n-gram embed history → GRIA |
| `Hybrid` | `hybrid_gria_lstm` | GRIA + CharLSTM log-prob blend |
| `CharLstm` | `char_lstm` | CharLSTM only |
| `SsmGria` | `ssm_gria` / `ssm_only` | SSM → GRIA |
| `SsmGriaNoLstm` | `ssm_gria_no_lstm` | SelectiveSSM + SSM → GRIA (no LSTM) |

## B1 — SelectiveSSM (`selective_ssm.hpp`)

Mamba-lite diagonal state:

- Learnable log-decay vector `A_log` → per-dim `exp(A)` decay
- Input-dependent gates: `B(x)=σ(W_b x)`, `C(h)=σ(W_c h)`
- Recurrence: `h_i ← A_i h_i + B_i · (W_b row · x)` — **O(d_state)** per token
- Output: `y = C ⊙ (W_c h) + W_d x`
- `pooled_state()` L2-normalized state for compressive memory

## B2 — CyphaLMModel (`cyphalm_model.hpp`)

Public API:

```cpp
CyphaLMModel model(CyphaLMConfig{});
CyphaLMModel::from_json_npz("checkpoint.json");
CyphaLMModel::from_embedded(config, weights);

model.reset_context();
auto pred = model.predict_next(token_id);
auto metrics = model.train_step(token_id, next_token_id);
auto lp = model.forward_log_probs(token_id);
```

Weight loading:

- **JSON + NPZ:** matches Python `CyphaLM.save()` layout (`config`, `gria`, `proj_*` arrays)
- **Embedded struct:** deterministic tiny weights for parity without files

## B3 — CompressiveMemory (`compressive_memory.hpp`)

CyphaDIF-style slot storage (simplified NIG expert means):

- Every `compress_interval` tokens: pool SSM / selective state → running mean in next slot
- `retrieve(query)` → LLR vs prior → softmax weights → bias vector added to GRIA input
- Uses `kappa0`, `alpha0`, `beta0` from config (mirrors `CyphaDIF` NIG hyperparams)

## B4 — BpeTokenizer (`bpe_tokenizer.hpp`)

Inference-only stub:

```cpp
auto tok = BpeTokenizer::load("merges.txt", "vocab.json");
auto ids = tok.encode("hello");
auto text = tok.decode(ids);
```

No training; loads merge rules + vocab id map from disk.

## Build

```bash
cmake --build native/build --target cypha_lm_native cyphalm_model_parity
./native/build/cyphalm_model_parity
./native/build/cyphalm_model_parity --mode hybrid
./native/build/cyphalm_model_parity path/to/checkpoint.json
```

## Parity scaffold

`cyphalm_model_parity` runs a fixed 10-token sequence with train + predict steps, printing per-step loss and top log-probs. Full numeric parity vs Python checkpoints is tracked in the master upgrade doc (`cyphalm_parity` integration item).

## File map

| File | Role |
|------|------|
| `selective_ssm.*` | B1 selective head |
| `cyphalm_model.*` | B2 unified stack |
| `compressive_memory.*` | B3 memory slots |
| `bpe_tokenizer.*` | B4 tokenizer stub |
| `cellai_ssm.*` | CellAI rank-2 SSM (Tier 0 dep) |
| `gria_lowrank.*` | Low-rank GRIA projection |
| `char_lstm.*` | Char LSTM head |
| `embed_table.*` | Fixed embedding lookup |
| `npz_util.*` | Minimal NPZ reader for checkpoints |
