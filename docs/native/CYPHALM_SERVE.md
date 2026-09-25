# CyphaLM serve / generation

gate24 CyphaLM is an online hp context mixer. **Train** and **serve** share the same
`hp::Predictor` state advance (predict + update per bit), but serve avoids train-only
bookkeeping and can use faster scoring paths.

## API split

| Path | Entry points | Scoring | Side effects |
|------|--------------|---------|--------------|
| **Train / BPC** | `train_step`, `observe_stream_bits`, `eval_bpc_compress_equivalent` | Bit-serial observe (8 bits/byte) | Increments `train_step_count`; updates tables |
| **Serve / generate** | `serve_advance`, `serve_predict_next`, `serve_greedy_next`, `generate_decode` | Bit-tree joint log probs (default) or O(8) greedy | No `train_step_count`; no `adapt_after_predict` stash |

### HpSequenceBackend

- `serve_advance_byte` — advance live context (alias of `consume_byte`).
- `serve_next_byte_log_probs` — full-vocab log P(next byte). **Default:** MSB bit-tree with delta undo on the live `pred_`: one `predict()` per node, a re-predict only before the bit-1 update, no update at leaves (382 predicts + 254 updates instead of 510 + 510). Exact against fresh-clone scoring, and scoring leaves the live model untouched (`hp_bit_tree_smoke`; before 2026-09-22 DMC splits and word-match resets leaked, see [`CYPHALM_LOSSY_MIXER_REPORT.md`](../reports/CYPHALM_LOSSY_MIXER_REPORT.md)). Legacy 256-clone path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`. **Serve defaults since 2026-09-23/24:** frozen scoring (`hp_frozen_scoring`: no learning inside the hypothetical byte; +0.003–0.008 bits/byte, 2.1–2.5× faster) and bit-tree pruning below 1e-4 (`hp_tree_prune`; 12.8 → 2.3–2.8 ms per distribution on a slim 95 MB model + ∞-gram). `CYPHA_HP_FROZEN_SCORING=0` and `CYPHA_HP_TREE_PRUNE=0` restore the exact path described here ([LM quality report](../reports/CYPHALM_LM_QUALITY_REPORT.md)).
- `serve_greedy_next_byte` — O(8) argmax via undo on `pred_` (no 256-way fan-out).
- `serve_sample_next_byte` — O(8) temperature bit sampling via undo on `pred_`.

### CyphaLMModel

- `serve_advance(token)` — priming long prompts without vocab fan-out.
- `serve_predict_next(context)` — consume context + bit-tree log probs + top-k fill.
- `serve_greedy_next(context)` — consume context + O(8) greedy byte.
- `serve_observe(token)` — score the byte with the full served distribution, then advance (prompt scoring: the mixing weights learn from it).
- `predict_next` — legacy wrapper that also records state for `adapt_after_predict` / `train_step_count`.

## Generation

High-level decode lives in `cyphalm_generation.hpp`:

- `generate_decode` — greedy, beam, temperature, top-k, top-p, uncertainty-gated.
- `generate_beam` — byte-level beam search on the full served distribution (members, ∞-gram, session cache, neural experts). Each hypothesis is replayed on the live model with learning off and rewound exactly (`hp::StreamRewind`, neural states, session), as word lookahead does; bytes all hypotheses share are committed, learned per `learn_from_output`. It used to rank with the primary predictor alone on a predictor copy and learn from its own output.
- Prompt priming (`prime_serve_context`) starts a new stream on the trained model with `reset_stream(keep_history=true)`, switches serve mode on (mixer at `hp_serve_mixer_lr_scale`, default 1) and feeds the prompt with `serve_advance`. On composite models it scores the last `prompt_score_bytes` (512) context bytes with `serve_observe` instead, so the mixing weights adapt to the prompt (priming alone never scored, so generation served the start weights); every mixing weight is restored when the request ends (`restore_mixing`, default on). Greedy takes the argmax of the full distribution (`exact_greedy`, default on; off = O(8) `serve_greedy_next`); sampling uses `serve_predict_next` + bit-tree log probs.
- Word lookahead (`word_candidates`, default 8) runs for every non-beam strategy while `learn_from_output` is off: K candidate words, exact rewind (`hp::StreamRewind`), best mean log-probability. `--word-candidates 0` gives plain byte decoding.

CLI example:

```bash
cmake --build native/build --target cyphalm_generate -j$(nproc)
./native/build/cyphalm_generate --prompt "Hello " --max-bytes 32 --strategy top_p --top-p 0.9 --temperature 0.8
./native/build/cyphalm_generate --strategy greedy --max-bytes 16
./native/build/cyphalm_generate --strategy beam --beam 4 --max-bytes 16 --latency
./native/build/cyphalm_generate --warmup-file bench/data/canterbury/alice29.txt --warmup-bytes 4096 \
  --ban-last-k 3 --repetition-penalty 1.15 --text-like-prior 0.35 --strategy greedy --max-bytes 32
```

Trained models, ensembles and the ∞-gram expert (flags added 2026-09-23/24):

```bash
./native/build/cyphalm_generate --load ckpt.json --prompt "..." --max-bytes 300          # checkpoint or ensemble manifest
./native/build/cyphalm_generate --load a.json --ensemble b.json:0.5 --infinigram x.igr --prompt "..."
./native/build/cyphalm_generate --load ckpt.json --word-candidates 0 --min-p 0.1 --no-repeat 8 --prompt "..."
```

Decode defaults, serve-time model settings (frozen scoring, bit-tree pruning,
serve mixer rate), env vars and file formats:
[`CYPHALM_LM_QUALITY_REPORT.md`, Reference](../reports/CYPHALM_LM_QUALITY_REPORT.md#reference).

Serve-time decode modifiers (do not change compress/BPC fidelity):

| Flag / JSON field | Role |
|-------------------|------|
| `--warmup-file` / `warmup_ids` | Prime context via `serve_advance` before the prompt |
| `--warmup-bytes` | Cap bytes read from warmup file |
| `--ban-last-k` / `ban_last_k` | Hard-ban bytes in the last k context bytes |
| `--repetition-penalty` / `repetition_penalty` | Down-weight repeats in the repetition window |
| `--text-like-prior` / `text_like_prior` | Soft log-prob bonus for printable ASCII |

Harness: `bash scripts/cyphalm_generation_harness.sh` — cold vs primed before/after samples + `decode_ms`.

REST: `POST /generate` and `POST /generate/stream` (see `cyphalm_rest_routes.cpp`) use the same `generate_decode` path. Body fields: `prompt_ids`, `max_tokens`, `strategy`, `temperature`, `top_k`, `top_p`, `min_p`, `word_candidates`, `word_no_repeat`, `no_repeat_ngram`, `no_repeat_window`, `learn_from_output`, `exact_greedy`, `prompt_score_bytes`, `restore_mixing`, `seed`, plus the modifiers below; unspecified fields take the `DecodeParams` defaults. `POST /sequence/load {"checkpoint_path": ...}` loads a checkpoint or an ensemble manifest.

## RAM note (gate24 mem22, measured 2026-09-20)

| Layout | VmRSS after construct | Notes |
|--------|----------------------|-------|
| Pre-undo checkpoint tree + dual `pred_` | **~4.3 GB** class | CHANGELOG PR #7 baseline |
| Post-undo dual `pred_` + `scratch_` | **~3.1 GB** | PR #7 landed |
| Single-predictor serve | **~1.5 GB** | [`GATE24_POST_UNDO_BENCH.md`](../reports/GATE24_POST_UNDO_BENCH.md) |
| **Demand-zero tables (this tree)** | **~80 MB** at construct, grows to ~1.5 GB as slots fill | construct ~55 ms instead of ~8 s; see lossy mixer report |

Lossy tiers (`apply_hp_lossy_tier`, `CYPHA_HP_LOSSY_TIER`) cut the table footprint further; measured RAM / bpc / speed per tier: [`CYPHALM_LOSSY_MIXER_REPORT.md`](../reports/CYPHALM_LOSSY_MIXER_REPORT.md).

`hp_serve_compact` is retained for API compatibility but is a no-op when only one predictor is allocated.

Served-model RAM after 2026-09-23 (packed 16-bit context slots, `slim` tier,
table folding, tables mapped from the checkpoint with `CYPHA_HP_MMAP`):
[`CYPHALM_LM_QUALITY_REPORT.md`, RAM](../reports/CYPHALM_LM_QUALITY_REPORT.md#ram).

---

## Quality note

**BPC / compression metrics** use `observe_stream_bits` (compress-faithful bit-serial NLL).
**Generation log probs** use the bit-tree joint path; greedy generation can skip the full tree.
These are the same hp math; greedy/top-k sampling is an inference convenience, not a BPC claim.

## Generation quality harness

Qualitative sampling (no invented scores):

```bash
bash scripts/cyphalm_generation_harness.sh
# → bench/results/generation_harness/*.json + docs/reports/CYPHALM_GENERATION_HARNESS.md
```

Runs greedy + temperature completions on built-in prompts via `generate_decode` (cold gate24 hp). For trained models use `cyphalm_lm_quality` / `cyphalm_gen_bench` ([`CYPHALM_LM_QUALITY_REPORT.md`](../reports/CYPHALM_LM_QUALITY_REPORT.md)).

## Tests

- `cyphalm_serve_smoke` — serve generation does not bump `train_step_count`; train_step does.
- `hp_bit_tree_smoke` — bit-tree ≡ legacy fork log probs; on text, a served model and an unserved twin stay identical (no speculative-state leak).
- `hp_undo_smoke` — predictor checkpoint round-trip (when undo headers are present).
- `hp_frozen_smoke`, `hp_stream_rewind_smoke`, `hp_fold_smoke`, `cyphalm_ensemble_smoke`, `cyphalm_infinigram_smoke` — frozen scoring, exact rewind, folding, ensembles, ∞-gram (see the LM quality report's Reference).
