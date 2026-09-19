# CyphaLM (hp) LLM Profile Report

**Date:** 2026-09-19  
**Commit:** `bbf385360abcd48319b44ae9afd923e9d0422102` (main after PR #1 hp integration)  
**Harness:** `native/tools/cyphalm_llm_profile.cpp` + `scripts/cyphalm_llm_profile.sh`

---

## Executive summary

> **BPC gap (2026-09-19):** Headline **5.48 BPC** below is the **256-clone `predict_next` API** on WikiText n=16 — **not** hp archive BPC. Default `eval_bpc` now uses **bit-serial observe NLL**, which matches hp archive on the same corpus/flags (enwik 8 MB light: **1.721** observe vs **1.721** archive). User **~1.610** is **v78/champ** flags, not light. Full analysis: [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md).

CyphaLM on **main** is backed by the vendored **hp** integer-exact context mixer (`HpSequenceBackend` → `hp::Predictor`) with the **light** compile profile (`CYPHA_HP_PROFILE=light`, `HP_SLOT_MAX=24`, `hp_table_bits=22`).

**Capability (headline — WikiText-2 natural language, byte tokens, *clone-API metric*):**

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

## Architecture — living API (Cypha main)

### Adapter: `HpSequenceBackend` (`native/include/cypha/cyphalm/hp_backend.hpp`, `native/src/cyphalm/hp_backend.cpp`)

Wraps vendored `hp::Predictor` (integer-exact context mixer). Key paths:

| Method | Behavior |
|--------|----------|
| `consume_byte` | 8 bit steps: `predict()` → `update(bit)` per bit of the byte; advances **main** `pred_` |
| `next_byte_log_probs(vocab)` | For each candidate byte `b ∈ [0,vocab)`: **clone** live `*pred_` into scratch via `clone_from`, score 8 bits; **do not** advance main. Opt-in bit-tree: `CYPHA_HP_BIT_TREE_LOGPROBS=1` |
| `observe_next_byte` | Score + update main predictor on the observed byte (train / compress-equivalent BPC path) |
| `log_prob_byte` / `sample_next_byte` | Single-clone O(8) bit path (train fast path / bit-serial generation) |

```7:9:native/include/cypha/cyphalm/hp_backend.hpp
/// RAM note: holds two ``hp::Predictor`` instances (``pred_`` + ``scratch_``).
/// ``next_byte_log_probs()`` clones ``*pred_`` once per vocab byte — a known
/// RAM/CPU hotspot at large vocab; not optimized in the default integration.
```

### Model: `CyphaLMModel` (`cyphalm_model.*`)

| API | Path |
|-----|------|
| `predict_next(token)` | `consume_byte(tok)` → `next_byte_log_probs(vocab_size)` → top-k fill |
| `train_step` / `eval_bpc` | Default **`eval_bpc` → `eval_bpc_compress_equivalent`** (bit-serial observe NLL). Legacy clone API still available via `predict_next` scoring |
| `eval_bpc` | Default: mean −log₂ p(bit) via observe (matches hp archive BPC). Clone-API eval requires explicit `predict_next` loop |
| Generation | `cyphalm_generation.*` — `generate_decode` with `DecodeStrategy` greedy / temperature / top-k / top-p |

Recipes (`cyphalm_config.cpp`):

- **Light (production):** `hp_table_bits=22`, `hp_slot_max=24`, `CYPHA_HP_PROFILE=light` → `HP_SLOT_MAX=24` at compile time.
- **Champ (research):** `CYPHA_HP_PROFILE=champ` → `HP_SLOT_MAX=35` + v78 feature flags from `v78_flags.ps1`.

### REST (`cyphalm_rest_routes.cpp`)

| Route | Maps to |
|-------|---------|
| `POST /predict_next` | `CyphaLMModel::predict_next` |
| `POST /generate` | `generate_decode` (JSON) |
| `POST /generate/stream` | `stream_generate` (SSE chunks) |

### hp archive bytes vs Cypha `eval_bpc`

| Metric | What it measures |
|--------|------------------|
| **hp compressed archive** | Arithmetic-coded bitstream size from the hp compressor tool — lossless coding of input |
| **Cypha `eval_bpc` (default)** | **`eval_bpc_compress_equivalent`**: Σ −log₂ p(bit) with update on observed bytes — **matches hp archive BPC** on same corpus/flags (see [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)) |
| **Cypha clone-API BPC** | Mean −log₂ P(next byte) via `predict_next` + 256× `next_byte_log_probs` — **different metric**; pre-fix copy bug inflated error |

These are related in theory (good predictors compress well). **Do not compare** hp archive **1.610** (v78 champ) to WikiText clone-API **5.48** (light, n=16) without relabeling profile and metric.

---

## Documented RAM hotspots (cited → measured)

| # | Hotspot | Source | Measured (light, `table_bits=22`) |
|---|---------|--------|-----------------------------------|
| 1 | **Dual `hp::Predictor`** (`pred_` + `scratch_`) | `hp_backend.hpp` L7–8 | Construct RSS **800,564 KiB** (~782 MiB) — two full table allocations at init |
| 2 | **Per-byte clone in `next_byte_log_probs`** | `hp_backend.hpp` L8–9; `hp_backend.cpp` L59–65 | `predict_next` **~16.1 s**/call ≈ `next_byte_log_probs` **~16.1 s** → ~256× clone dominates; `consume_byte` alone **47,613 B/s** (~750× faster) |
| 3 | **`SLOT_MAX` / `table_bits` axis** | `HpFlags.cmake`, `hp/tools/hp_harness.sh` H34 | Light slot 24: peak **1,203,288 KiB** measured; lab refs **~1.6 GB** @ mem 22 slot 24, **~15 GB** @ slot 35 — champ **not built** on 16 GiB VM |
| — | CI smoke `table_bits=16` | `hp_llm_smoke.cpp` | RSS **37,496 KiB** — not production footprint |

**Conclusion from measurements:** scoring-path clone fan-out (#2) is the binding latency constraint; table footprint (#1, #3) sets RAM floor. Optimizations should attack #2 before wider eval or champ profiling.

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

## Phase 4 — Execution plan (prioritized by measurements)

Measurements show **`next_byte_log_probs` ≈ `predict_next` latency** and **`consume_byte` ~750× faster** — the 256× full-predictor clone per vocab byte is the first target. Items are ordered for a **lossy LLM direction** (probability distribution + minimal data) while preserving optional lossless hp compress paths.

### 1. Prefix-share / bit-trie scoring (attack clone fan-out)

| Field | Detail |
|-------|--------|
| **Status (2026-09-19)** | **`next_byte_log_probs_bit_tree` implemented**; parity OK vs legacy (`hp_bit_tree_smoke`). **Not default** — ~510 full clones @ table_bits=22 → **~3× slower** than fixed 256-clone legacy. Needs undo-stack (#2) |
| **Goal** | Score all 256 byte candidates sharing one bit-prefix walk; branch only where candidate sets diverge |
| **Expected RAM / speed** | **~10–50×** faster full-vocab score; scratch RAM bounded to trie depth, not 256× `Predictor` |
| **Quality risk** | **Low** if bit-path math matches current clone semantics |
| **1–2 day experiment** | Prototype `next_byte_log_probs_trie` beside existing path; assert max Δlog p &lt; 1e−9 vs clone on 1k random contexts; benchmark `predict_next_us` on WikiText prefix |

### 2. Single predictor + reversible undo

| Field | Detail |
|-------|--------|
| **Goal** | One `pred_` only; record bit updates on a stack; undo after each hypothetical byte instead of cloning |
| **Expected RAM / speed** | **−~50% RSS** (drop standing `scratch_` clone target); **~256×** faster score at equal table size |
| **Quality risk** | **Low** if undo is exact inverse of `update` |
| **1–2 day experiment** | Implement undo stack in `hp_backend.cpp`; parity test vs clone on `hp_roundtrip_smoke` + 256-byte grid; measure peak RSS during full-vocab score |

### 3. Freeze tables for serve (train / serve split)

| Field | Detail |
|-------|--------|
| **Goal** | Train-online with mutable counts; export immutable snapshot for REST `/predict_next` and generation |
| **Expected RAM / speed** | Serve path skips dual mutable tables; enables RO layout + sharing; faster init |
| **Quality risk** | **Medium** — stale if serve snapshot not refreshed; BPC drift vs online |
| **1–2 day experiment** | `CyphaLMModel::save` → load read-only `HpSequenceBackend`; compare WikiText BPC online vs frozen after 32 train steps |

### 4. Lower `table_bits` / `slot` LLM SKU

| Field | Detail |
|-------|--------|
| **Goal** | Product tiers: e.g. `table_bits=18–20`, `slot_max=20` for sub-200 MB LM SKU |
| **Expected RAM / speed** | **−4–16× RAM** (hp harness mem axis); proportionally faster clone if #1–2 not yet done |
| **Quality risk** | **Medium** — WikiText BPC regression; measure on real text only |
| **1–2 day experiment** | Sweep `hp_table_bits ∈ {18,20,22}` × `slot_max ∈ {20,24}`; plot RSS vs WikiText BPC (n=64 eval) in `cyphalm_llm_profile` |

### 5. Top-k partial expansion (infer without full 256)

| Field | Detail |
|-------|--------|
| **Goal** | Expand only top-M byte candidates by cheap unigram / cache prior; exact hp score on M≪256 |
| **Expected RAM / speed** | **~256/M×** faster generation and REST infer; lower peak RSS during score |
| **Quality risk** | **Medium–high** — tail bytes missed; lossy for full distribution |
| **1–2 day experiment** | M=16,32 on WikiText: top-1/top-10 vs full vocab; latency `generate_decode` 64 tokens |

### 6. BPE outer / byte inner

| Field | Detail |
|-------|--------|
| **Goal** | User-facing BPE tokens; inner loop stays byte-level hp for compression fidelity |
| **Expected RAM / speed** | Outer vocab ≪256 per step for UX APIs; inner cost unchanged unless #1–2 land |
| **Quality risk** | **Low** for API; byte BPC ≠ BPE perplexity — label separately |
| **1–2 day experiment** | Wire existing `BpeTokenizer` in `CyphaLMModel`; report BPE-token top-1 + byte BPC on same WikiText slice |

### 7. Lossy top-M renormalize at infer

| Field | Detail |
|-------|--------|
| **Goal** | After partial expansion, renormalize over top-M logits only (lossy distribution) for sampling |
| **Expected RAM / speed** | Tiny extra CPU; enables calibrated temperature/top-p on reduced support |
| **Quality risk** | **High** for calibration — entropy vs empirical gap already **+0.48–0.72 bits** on WikiText |
| **1–2 day experiment** | Compare greedy/temp samples M=32 vs full 256; ECE on held-out WikiText bytes |

### 8. mmap read-only shared tables

| Field | Detail |
|-------|--------|
| **Goal** | Post-#3 snapshot backed by `mmap` RO pages; multiple workers share one physical copy |
| **Expected RAM / speed** | **N× RSS reduction** for N REST workers; faster fork/warm start |
| **Quality risk** | **Low** (identical math if snapshot frozen) |
| **1–2 day experiment** | Export predictor blob; mmap in second process; verify BPC match + `VmRSS` with two `cypha_rest` workers |

### Eval harness follow-ups (CI / nightly)

Extend `cyphalm_llm_profile` + `scripts/ci_native_hp_smoke.sh` when #1–2 reduce per-step cost:

| Suite piece | Status this report |
|-------------|-------------------|
| Large-n `eval_bpc` on WikiText | **n=16 only** (budget); need faster score path for n≥256 CI |
| Top-1 / top-5 / top-10 | **Measured** WikiText n=16 |
| Entropy / calibration | **Measured** (+0.48–0.72 bit gap) |
| Generation samples | **Measured** (weak quality noted) |
| Needle / long-context | **Not run** |
| RSS during full-vocab score | **Measured** peak 1.15 GiB |
| `predict_next` vs `next_byte_log_probs` latency | **Measured** ~16.1 s each |
| Synthetic negative control | **Labeled** ~7.0 BPC, not headline |

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
