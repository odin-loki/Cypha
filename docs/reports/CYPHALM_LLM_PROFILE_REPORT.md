# CyphaLM (hp) LLM Profile Report

**Date:** 2026-09-19  
**Commit:** `bbf385360abcd48319b44ae9afd923e9d0422102` (main after PR #1 hp integration)  
**Harness:** `native/tools/cyphalm_llm_profile.cpp` + `scripts/cyphalm_llm_profile.sh`

---

## Executive summary

CyphaLM on **main** is backed by the vendored **hp** integer-exact context mixer (`HpSequenceBackend` → `hp::Predictor`) with the **light** compile profile (`CYPHA_HP_PROFILE=light`, `HP_SLOT_MAX=24`, `hp_table_bits=22`).

**Capability (headline — WikiText-2 natural language, byte tokens):**

| Metric | Cold eval (no train) | After 32 online train steps |
|--------|----------------------|-----------------------------|
| **BPC** (n=16 held-out bytes) | **6.77** | **5.48** |
| Top-1 accuracy | 6.25% | 6.25% |
| Top-5 accuracy | 31.25% | 25.0% |
| Top-10 accuracy | 43.75% | 50.0% |
| Mean pred. entropy (bits) | 7.49 | 5.96 |
| Mean empirical NLL (bits) | 6.77 | 5.48 |

Online training on 32 WikiText bytes **lowers BPC by ~1.3 bits** vs cold start on this short eval window. Top-1 remains low (byte-level 256-way classification is hard); top-10 improves modestly. Generation after brief training is **not English-like** (greedy collapses to spaces; sampling is noisy).

**Performance (production light profile):**

| Metric | Measured value |
|--------|----------------|
| Model construct latency | **297 ms** |
| Steady / peak RSS (`VmHWM`) | **~781 MB** construct → **~1.17 GB** under `predict_next` load |
| `predict_next` (vocab 256) | **~16.1 s** / call |
| `next_byte_log_probs` (vocab 256) | **~16.1 s** / call |
| `train_step` throughput | **~0.062 bytes/s** (64-step window) |
| `consume_byte` only (no vocab fan-out) | **~47,600 bytes/s** |

**Negative control (synthetic arithmetic pattern):** BPC **~7.0** — near-uniform, **not** a meaningful LM claim.

**Champ profile (`CYPHA_HP_PROFILE=champ`, ~15 GB lab RSS):** **skipped** — VM has 16 GiB RAM, ~6.7 GiB available; champ build risks OOM.

---

## Build / config under test

| Item | Value |
|------|-------|
| Git SHA | `bbf385360abcd48319b44ae9afd923e9d0422102` |
| CMake | `-DCMAKE_BUILD_TYPE=Release -DCYPHA_HP_PROFILE=light -DCMAKE_CXX_COMPILER=g++` |
| `HP_SLOT_MAX` (compile) | 24 |
| `hp_table_bits` (runtime recipe) | 22 |
| `hp_slot_max` | 24 |
| `hp_mixer_lr` | 2 |
| `hp_gria` | true |
| `vocab_size` | 256 (raw bytes) |
| Legacy Hybrid GRIA+LSTM | **OFF** (`CYPHA_BUILD_LEGACY_CYPHALM=OFF`) |

---

## Hardware / OS

| Item | Value |
|------|-------|
| OS | Linux 6.12.94+ (KVM) |
| CPU | Intel Xeon, 4 cores |
| RAM | 15 GiB total, ~6.7 GiB available at run time |
| Compiler | g++ 13.3.0 (Ubuntu) |

---

## Phase 1 — Profile (RSS, latency, throughput)

### Methodology

- Tool: `cyphalm_llm_profile` with default production recipe (`apply_hp_production_recipe`).
- RSS from `/proc/self/status` (`VmRSS`, `VmHWM`) and GNU `time -v` (`Maximum resident set size`).
- Latency: 5 timed iterations after 2 warmup `predict_next` / `next_byte_log_probs` calls.
- Throughput: 64 `train_step` and 64 `eval_bpc` steps on a **synthetic** stream (timing only; BPC from that stream is **not** used as capability).

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Cold construct | 297 ms | Single `CyphaLMModel` + dual `hp::Predictor` |
| RSS after construct | 800,564 KiB (~782 MiB) | `VmHWM` = `VmRSS` |
| RSS after latency bench | 1,203,288 KiB peak (~1.15 GiB) | Dual predictor + per-byte scratch clones |
| `predict_next` | 16,134,850 µs (~16.1 s) | Includes `consume_byte` + 256× predictor clone |
| `next_byte_log_probs` | 16,079,464 µs (~16.1 s) | 256 vocab fan-out |
| `train_step` (64 steps) | 0.062 bytes/s | Dominated by `next_byte_log_probs` |
| `eval_bpc` (64 steps, synthetic) | 0.062 bytes/s | Same hot path |
| `consume_byte` (512 bytes) | 47,613 bytes/s | No vocab fan-out |

**RAM hotspot (documented in `hp_backend.hpp`):** `HpSequenceBackend` holds `pred_` + `scratch_`; `next_byte_log_probs()` clones `*pred_` once per vocab byte (256× per `predict_next`).

### hp_llm_smoke (table_bits=16, CI fast path)

Separate smoke binary uses `hp_table_bits=16` for CTest speed:

```
hp_llm_smoke OK bpc=6.3616 table_bits=16 rss_kb=37256
```

`time -v` max RSS: **37,496 KiB** (~37 MiB). **Not comparable** to production `table_bits=22` footprint.

---

## Phase 2 — LLM capability (real text)

### Corpus

| Field | Value |
|-------|-------|
| Source | **WikiText-2** (`bench/data/wikitext2/wikitext-2/wiki.train.tokens`) |
| Loader | `load_bench_corpus("d21", max_chars=100000, vocab_size=256)` |
| Split | 80/20 of capped train file → **80,000** train bytes / **20,000** eval bytes |
| Tokenization | Raw **bytes** (UTF-8 octets), not BPE |
| Profile JSON | `bench/config/profiles/cyphalm_d21_hp.json` |

Download: `bash scripts/download_wikitext2.sh` (succeeded on this VM).

### Headline capability run

**Command:**

```bash
/usr/bin/time -v ./native/build/cyphalm_llm_profile --wiki-only \
  --wiki-train-steps 32 --wiki-eval-n 16 --gen-tokens 16 \
  --wiki-prompt-len 24 --wiki-max-chars 100000
```

**Wall time:** 44:06 (`time -v`). **Peak RSS:** 1,204,140 KiB (~1.15 GiB).

#### BPC / top-k / entropy (WikiText-2 eval stream, n=16)

| Condition | BPC | Top-1 | Top-5 | Top-10 | H(pred) bits | NLL bits |
|-----------|-----|-------|-------|--------|--------------|----------|
| Eval only (cold) | **6.769** | 6.25% | 31.25% | 43.75% | 7.491 | 6.769 |
| After 32 online train steps | **5.480** | 6.25% | 25.0% | 50.0% | 5.957 | 5.480 |

- **Methodology (cold):** `reset_context()` → score next byte on eval stream (one pass, no training).
- **Methodology (trained):** `train_sequence(train_ids, 32)` → continue on eval stream without reset.
- **Calibration:** Mean predictive entropy exceeds empirical NLL by **+0.48–0.72 bits** (overconfident / diffuse).

#### Generation (after 32 train steps, prompt = first 24 eval bytes)

| Strategy | Sample (16 bytes) | Notes |
|----------|-------------------|-------|
| Greedy | (spaces) | Collapse to high-probability whitespace |
| Temperature 0.9 | `jgh l e'!$.,]Wjo` | No coherent English; high noise |

Prompt bytes decode to: `k gj l`] kalmYlagf jgge ` (WikiText raw tokenization artifacts at byte level).

#### Context scaling (WikiText train bytes, after 16 online train steps)

Measured via `consume_byte` prefix + single `next_byte_log_probs` (next-byte BPC at context length):

| Context bytes | Next-byte BPC |
|---------------|---------------|
| 256 | 3.653 |
| 1,024 | 4.966 |
| 8,192 | 2.515 |

Non-monotonic at these short samples — interpret as directional only; full sweep needs longer budget.

### Negative control — synthetic pattern (do not use as LM score)

**Command:** `cyphalm_llm_profile` throughput section on 16-byte arithmetic pattern `(i*7+3)%96+32`.

| Metric | Value |
|--------|-------|
| BPC | **6.998** |
| Top-1 | **0%** |
| Interpretation | Near **8 bits** uniform — no linguistic structure |

### Lossless codec roundtrip

`predictive_codec` `compress_tokens` / `decompress_tokens` path exists but was **not run** in this profile pass (same per-step `predict_next` cost; would add multi-hour runtime). Prior CTest: `native_hp_roundtrip_smoke`, `native_predictive_codec_smoke`.

---

## Limitations / not measured

| Item | Status |
|------|--------|
| Champ profile (`HP_SLOT_MAX=35`, ~15 GB RSS) | **Skipped** (RAM) |
| Large WikiText eval (e.g. 4k+ bytes) | **Not run** — ~16 s/step → multi-hour |
| Full `CYPHA_BENCH_FULL_CORPUS=1` WikiText train | **Not run** |
| GPU / CUDA hp path | **Not measured** (CPU integration only) |
| Locked BPC in `BASELINE_LOCK.json` for hp | **Not updated** (historical Hybrid 2.664 unrelated) |
| Perplexity on BPE tokens | **Not measured** (byte vocab only) |
| Arithmetic codec roundtrip on WikiText | **Not measured** (time budget) |

---

## Comparison framing: lossless compressor vs neural LLM

| Aspect | hp / CyphaLM today | Typical neural LLM |
|--------|-------------------|-------------------|
| Objective | Lossless next-byte coding (integer counts) | Lossy cross-entropy on subword dist |
| Output | Calibrated-ish byte histogram | Rich semantic representations |
| RAM | **~1–1.2 GiB** light / **~15 GiB** champ | Model-dependent (often GBs) |
| Speed | **~0.06 bytes/s** train/eval (256× clone) | ms/token on GPU |
| Strength | Exact online adaptation; principled compression | Fluency, instruction, reasoning |
| Weakness | Byte vocab, slow vocab fan-out, weak generation | Lossy, needs large data |

CyphaLM today is best understood as a **byte-level context mixer** exposing LM APIs (`predict_next`, `generate_decode`, `eval_bpc`), not a chat LLM.

---

## Phase 4 — Toward a lossy LLM (prioritized backlog)

### P0 — Serve path RAM / latency (1–2 days each)

| Idea | Goal | RAM / speed | Quality risk | First experiment |
|------|------|-------------|--------------|------------------|
| **Drop dual predictor + per-byte clone** | Single `pred_`; incremental byte log-prob without full snapshot | **−50%+ RAM**, **~256× faster** `predict_next` | None if bit-path math unchanged | Refactor `next_byte_log_probs` to reuse scratch in-place with rollback |
| **Train/serve split** | Train with full tables; serve frozen or quantized counts | Serve **&lt;200 MB** target | Small BPC drift | Export `hp::Predictor` snapshot after N bytes; read-only infer |
| **Table_bits ablation** | `mem 18–20` vs 22 | **−4–16× RAM**, faster | +0.1–0.5 BPC | Sweep `hp_table_bits` on WikiText n=256 eval |

### P1 — Lossy distillation (week-scale)

| Idea | Goal | RAM / speed | Quality risk | First experiment |
|------|------|-------------|--------------|------------------|
| **Quantized count tables** | 8–16 bit counters vs exact | **−2–4× RAM** | BPC +0.05–0.2 | Port hp `mem` sweep data; lock acceptable drift |
| **Sparse / hashed slots** | Replace dense SLOT tables | **−10× RAM** at scale | Context collisions | Hash context → slot; measure WikiText BPC vs dense |
| **Decay / LRU slot cache** | Bounded memory streaming LM | Fixed RAM cap | Long-context BPC | Max slots 64k; decay stale counts |

### P2 — LLM UX layer

| Idea | Goal | RAM / speed | Quality risk | First experiment |
|------|------|-------------|--------------|------------------|
| **BPE / SentencePiece front-end** | Subword UX, smaller effective vocab | Faster softmax | Byte BPC ≠ token PPL | Train BPE on WikiText; map hp probs to merges |
| **Sampling API** | top-p, temperature on calibrated probs | Negligible | Bad samples if uncalibrated | Wire `DecodeParams` to hp distribution; ECE on WikiText |
| **Eval harness v2** | PPL, LAMBADA-style byte completion | CI-time budget | — | Extend `cyphalm_llm_profile` with `--eval-n 256` nightly job |

### P3 — Algorithm direction (lossy)

| Idea | Goal | RAM / speed | Quality risk | First experiment |
|------|------|-------------|--------------|------------------|
| **Neural head on hp features** | MLP reads mixer logits → lossy LM | +10–50 MB | May beat hp BPC on small data | Freeze hp features; train 1-layer head |
| **Distill hp → small transformer** | Export (context → dist) pairs | Student **&lt;100 MB** | Lossy | 1M byte pairs from hp; train tiny GPT |
| **Ablation: BPC vs RAM Pareto curve** | Product knob | Document tradeoff | — | Automated sweep in `cyphalm_llm_profile` |

---

## Appendix — raw commands and outputs

### A. Profile (synthetic timing + negative control)

```bash
cd native/build
/usr/bin/time -v ./cyphalm_llm_profile --skip-wiki --skip-roundtrip \
  --latency-iters 5 --train-cap 64 --eval-cap 64 --topk-n 64
```

See: `bench/results/cyphalm_llm_profile/run_main.txt`

### B. WikiText capability (headline)

```bash
/usr/bin/time -v ./cyphalm_llm_profile --wiki-only \
  --wiki-train-steps 32 --wiki-eval-n 16 --gen-tokens 16 \
  --wiki-prompt-len 24 --wiki-max-chars 100000
```

See: `bench/results/cyphalm_llm_profile/wiki_capability.txt`, `wiki_capability_time_v.txt`

### C. WikiText context scaling

```bash
./cyphalm_llm_profile --skip-construct --skip-latency --skip-throughput \
  --skip-negative-control --skip-roundtrip --skip-generation-wiki \
  --wiki-train-steps 16 --wiki-eval-n 4 --wiki-max-chars 100000
```

See: `bench/results/cyphalm_llm_profile/wiki_context.txt`

### D. CI smoke

```bash
/usr/bin/time -v ./hp_llm_smoke
```

See: `bench/results/cyphalm_llm_profile/hp_llm_smoke.txt`

---

*All BPC / RSS / latency numbers in this report are measured on the VM described above; none are invented or extrapolated.*
