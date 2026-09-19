# CyphaLM LLM Eval — measured bit-serial BPC (large-n)

**Date:** 2026-09-19  
**Branch:** `cursor/bit-tree-gate24-eval-9d44`  
**Harness:** `native/tools/cyphalm_llm_eval.cpp`, `scripts/cyphalm_llm_eval.sh`  
**Build:** `-DCMAKE_BUILD_TYPE=Release -DCYPHA_HP_PROFILE=light -DCMAKE_CXX_COMPILER=g++`

All numbers below were measured on this VM. None are extrapolated.

---

## Build / SKU under test

| Item | Value |
|------|-------|
| `CYPHA_HP_PROFILE` | **light** (CI default; fast/dev — **not** quality-champ tier) |
| `HP_SLOT_MAX` (compile) | 24 |
| v78 flags matched | **0 / 78** (see [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)) |
| `hp_table_bits` | 22 |
| `hp_mixer_lr` | 2 |
| Metric | **`observe_bit_serial_bpc`** via `CyphaLMModel::eval_bpc()` (compress-equivalent) |

Quality claims require **gate24** or **champ** builds (`-DCYPHA_HP_PROFILE=gate24|champ`). enwik8.8mb was **not available** on this VM; WikiText-2 only for large-n observe here. Prior enwik8MB light observe: **1.721 BPC** @ 8,388,608 bytes ([`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)).

---

## Hardware

| Item | Value |
|------|-------|
| OS | Linux 6.12.94+ (KVM) |
| CPU | Intel Xeon, 4 cores |
| RAM | 15 GiB |
| Compiler | g++ 13.3.0 |

---

## Observe BPC (bit-serial, n = 100,000 bytes)

**Command:**

```bash
./native/build/cyphalm_llm_eval --observe-n 100000 --topk-n 8 --latency-iters 5
```

Corpus: WikiText-2 (`bench/data/wikitext2/wikitext-2/`), loader profile `d21`, 80/20 split capped at 500k chars.

| Corpus slice | Bytes | BPC | Wall (ms) | Throughput (B/s) |
|--------------|-------|-----|-----------|------------------|
| WikiText-2 train | 100,000 | **2.106972** | 3,281.3 | 30,476 |
| WikiText-2 eval | 100,000 | **2.138972** | 2,913.3 | 34,326 |

---

## Observe BPC (bit-serial, n = 10,000 bytes)

**Command:**

```bash
./native/build/cyphalm_llm_eval --observe-n 10000 --skip-topk --latency-iters 3
```

| Corpus slice | Bytes | BPC | Wall (ms) | Throughput (B/s) |
|--------------|-------|-----|-----------|------------------|
| WikiText-2 train | 10,000 | **2.680266** | 887.3 | 11,270 |
| WikiText-2 eval | 10,000 | **2.806720** | 701.4 | 14,258 |

---

## Bit-tree top-k (predict_next path, n = 8 pairs)

Uses default **bit-tree** `next_byte_log_probs` (not legacy 256-clone). Cold context; no online training.

| Corpus | Pairs | BPC (NLL) | Top-1 | Top-5 | Top-10 | Mean H(pred) bits | `predict_next` µs/call |
|--------|-------|-----------|-------|-------|--------|-------------------|------------------------|
| WikiText-2 train | 8 | **7.725** | 0% | 25% | 25% | 7.617 | **37,073,906** |

**Note:** Clone-API BPC is a different metric from observe/archive BPC. Do not compare 7.73 to enwik **1.721** light observe or **1.610** champ archive.

---

## Full-vocab latency — bit-tree (checkpoint DFS) vs legacy 256-clone

**Command:**

```bash
./native/build/bpc_gap_measure --corpus bench/data/wikitext2/wikitext-2/wiki.train.tokens \
  --bytes 65536 --clone-n 3
```

Warm context: 64 bytes consumed; 3 timed iterations.

| Path | Latency (µs/call) | vs legacy |
|------|-------------------|-----------|
| **bit-tree** `next_byte_log_probs` (default) | **40,424,640** | **0.86×** (faster) |
| legacy 256-clone (`CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`) | 46,967,565 | 1.00× |
| `predict_next` (bit-tree default) | 38,826,194 | 0.83× |

**Command (eval harness, 5 iters):**

```bash
./native/build/cyphalm_llm_eval --observe-n 100000 --topk-n 8 --latency-iters 5
```

| Path | Latency (µs/call) | vs legacy |
|------|-------------------|-----------|
| bit-tree (checkpoint DFS) | **38,813,756** | **0.92×** (legacy ~8% faster this run) |
| legacy 256-clone | 35,771,945 | 1.00× |

**Interpretation:** Bit-tree joint logprobs match legacy exactly (`hp_bit_tree_smoke` max Δ = 0). Default path eliminates **256 heap `clone_from` allocations** per full-vocab score; latency is in the same ballpark as legacy (run-to-run variance ±~15%). Further speed requires a true hp `undo` stack (see [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md)).

---

## Parity

```bash
./native/build/hp_bit_tree_smoke
# hp_bit_tree_smoke OK max_tree_legacy_delta=0 single_delta=0
```

---

## Reproduce

```bash
bash scripts/download_wikitext2.sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DCYPHA_HP_PROFILE=light -DCMAKE_CXX_COMPILER=g++
cmake --build native/build --target cyphalm_llm_eval bpc_gap_measure hp_bit_tree_smoke -j$(nproc)
bash scripts/cyphalm_llm_eval.sh 100000 8
```

Gate24 / champ eval: rebuild with `-DCYPHA_HP_PROFILE=gate24` or `champ` and rerun the same harness.
