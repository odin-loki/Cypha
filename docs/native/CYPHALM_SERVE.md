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
- `serve_next_byte_log_probs` — full-vocab log P(next byte). **Default:** MSB bit-tree with delta undo on the live `pred_`: one `predict()` per node, a re-predict only before the bit-1 update, no update at leaves (382 predicts + 254 updates instead of 510 + 510). Exact against fresh-clone scoring, and scoring leaves the live model untouched (`hp_bit_tree_smoke`; before 2026-09-22 DMC splits and word-match resets leaked, see [`CYPHALM_LOSSY_MIXER_REPORT.md`](../reports/CYPHALM_LOSSY_MIXER_REPORT.md)). Legacy 256-clone path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`.
- `serve_greedy_next_byte` — O(8) argmax via undo on `pred_` (no 256-way fan-out).
- `serve_sample_next_byte` — O(8) temperature bit sampling via undo on `pred_`.

### CyphaLMModel

- `serve_advance(token)` — priming long prompts without vocab fan-out.
- `serve_predict_next(context)` — consume context + bit-tree log probs + top-k fill.
- `serve_greedy_next(context)` — consume context + O(8) greedy byte.
- `predict_next` — legacy wrapper that also records state for `adapt_after_predict` / `train_step_count`.

## Generation

High-level decode lives in `cyphalm_generation.hpp`:

- `generate_decode` — greedy, beam, temperature, top-k, top-p, uncertainty-gated.
- `generate_beam` — byte-level beam search with bit-tree log probs and `hp::Predictor` snapshots.
- Prompt priming uses `serve_advance`; greedy decode uses `serve_greedy_next`; sampling uses `serve_predict_next` + bit-tree log probs; beam uses `HpSequenceBackend::byte_log_probs_bit_tree` per hypothesis.

CLI example:

```bash
cmake --build native/build --target cyphalm_generate -j$(nproc)
./native/build/cyphalm_generate --prompt "Hello " --max-bytes 32 --strategy top_p --top-p 0.9 --temperature 0.8
./native/build/cyphalm_generate --strategy greedy --max-bytes 16
./native/build/cyphalm_generate --strategy beam --beam 4 --max-bytes 16 --latency
./native/build/cyphalm_generate --warmup-file bench/data/canterbury/alice29.txt --warmup-bytes 4096 \
  --ban-last-k 3 --repetition-penalty 1.15 --text-like-prior 0.35 --strategy greedy --max-bytes 32
```

Serve-time decode modifiers (do not change compress/BPC fidelity):

| Flag / JSON field | Role |
|-------------------|------|
| `--warmup-file` / `warmup_ids` | Prime context via `serve_advance` before the prompt |
| `--warmup-bytes` | Cap bytes read from warmup file |
| `--ban-last-k` / `ban_last_k` | Hard-ban bytes in the last k context bytes |
| `--repetition-penalty` / `repetition_penalty` | Down-weight repeats in the repetition window |
| `--text-like-prior` / `text_like_prior` | Soft log-prob bonus for printable ASCII |

Harness: `bash scripts/cyphalm_generation_harness.sh` — cold vs primed before/after samples + `decode_ms`.

REST: `POST /cyphalm/generate` (see `cyphalm_rest_routes.cpp`) uses the same `generate_decode` path.

## RAM note (gate24 mem22, measured 2026-09-20)

| Layout | VmRSS after construct | Notes |
|--------|----------------------|-------|
| Pre-undo checkpoint tree + dual `pred_` | **~4.3 GB** class | CHANGELOG PR #7 baseline |
| Post-undo dual `pred_` + `scratch_` | **~3.1 GB** | PR #7 landed |
| Single-predictor serve | **~1.5 GB** | [`GATE24_POST_UNDO_BENCH.md`](../reports/GATE24_POST_UNDO_BENCH.md) |
| **Demand-zero tables (this tree)** | **~80 MB** at construct, grows to ~1.5 GB as slots fill | construct ~55 ms instead of ~8 s; see lossy mixer report |

Lossy tiers (`apply_hp_lossy_tier`, `CYPHA_HP_LOSSY_TIER`) cut the table footprint further; measured RAM / bpc / speed per tier: [`CYPHALM_LOSSY_MIXER_REPORT.md`](../reports/CYPHALM_LOSSY_MIXER_REPORT.md).

`hp_serve_compact` is retained for API compatibility but is a no-op when only one predictor is allocated.

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

Runs greedy + temperature completions on built-in prompts via `generate_decode` (cold gate24 hp).

## Tests

- `cyphalm_serve_smoke` — serve generation does not bump `train_step_count`; train_step does.
- `hp_bit_tree_smoke` — bit-tree ≡ legacy fork log probs; on text, a served model and an unserved twin stay identical (no speculative-state leak).
- `hp_undo_smoke` — predictor checkpoint round-trip (when undo headers are present).
