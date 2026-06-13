# Branch A — CyphaDIF on Frozen Semantic Embeddings

**Maps:** [`Cypha Tests.txt`](../Cypha%20Tests.txt) Branch A → CyphaDIF as NLP routing/classification layer on top of a frozen encoder.

**Runner:** `cypha_tune_run --config bench/config/cypha_branch_a_sweep.py --write`

Optional dependency for semantic embeddings:

```powershell
native build (`cmake --build`) sentence-transformers
```

Without it, the sweep falls back to deterministic hashing+SVD (weaker; for pipeline smoke only).

---

## Architecture

```text
text → [frozen encoder] → vector x → CyphaDIF(VectorEncoder) → label + epistemic variance
                ↑                           ↑
           no LM training              online train (field + optional W_enc)
```

| Component | Role |
|-----------|------|
| `frozen_text_embeddings.py` | `embed_texts()` via MiniLM or hashing fallback |
| `CyphaDIF` | Online classifier, OOD epistemic signal, adversarial GH gate |
| `encoder._frozen=True` | Skip Fisher-Rao encoder updates; DIF field only |

---

## Results @ 2000 Newsgroups (20 classes)

Artifact: `bench/config/cypha_branch_a_sweep.json`  
Encoder: **sentence-transformers/all-MiniLM-L6-v2** (384-d, frozen)

| Method | Accuracy | Notes |
|--------|----------|-------|
| CyphaDIF + TF-IDF/SVD (D09 reference) | **34.0%** | Current D09-style path |
| CyphaDIF + frozen ST, online `W_enc` | **49.3%** | +15.3pp vs TF-IDF |
| LogReg + frozen ST | **60.3%** | Batch baseline |
| **CyphaDIF + frozen ST, frozen `W_enc`** | **62.5%** | **Best** — online field only |

**Verdict:** Frozen semantic embeddings unlock CyphaDIF on text — **+28.5pp vs TF-IDF** at 2k samples. Freezing `EncoderProjection` (**62.5%**) beats online projection tuning (**49.3%**) and matches/beats batch LogReg on this split.

### Gutenberg OOD epistemic (D09 Branch A run)

Artifact: `bench/config/d09_branch_a_summary.json`

| Split | Mean epistemic var |
|-------|-------------------|
| 20news held-out (in-domain) | **0.170** |
| Gutenberg segments (OOD) | **1.077** |

Mann-Whitney **p ≈ 2.8×10⁻¹⁰²** — CyphaDIF uncertainty **rises on out-of-domain book text** when trained on newsgroup MiniLM vectors. Use for routing / abstain before LLM generation.

```powershell
cypha_tune_run --config bench/config/run_d09_branch_a.py
# or full D09 with Branch A block:
$env:CYPHA_BENCH_BRANCH_A="1"; cypha_bench_run --domain 9
```

Hashing fallback @ same protocol: **16–19%** (not semantic — use only for offline CI smoke).

---

## Use cases (from Cypha Tests.txt)

- Intent / topic routing with **online updates** without retraining the LM
- **Calibrated confidence** and OOD detection on embedding inputs
- Adversarial GH gate before queries reach a local LLM
- Sub-ms CyphaDIF classify after embedding is computed (C++ path)

---

## Commands

```powershell
# Semantic (requires sentence-transformers)
cypha_tune_run --config bench/config/cypha_branch_a_sweep.py --n-samples 2000 --backend sentence_transformers --write

# Offline smoke
cypha_tune_run --config bench/config/cypha_branch_a_sweep.py --n-samples 800 --backend hashing --write --out bench/config/cypha_branch_a_hashing.json
```

---

## Next steps

1. ~~Wire frozen-ST path into **D09**~~ — `run_d09_branch_a.py` + `CYPHA_BENCH_BRANCH_A=1`.
2. ~~OOD eval: Gutenberg vs 20news epistemic~~ — **done** (see above).
3. ~~**Local LLM routing**~~ — REST `/route/text`, `/route/generate` + Ollama stub (`cypha_core / cypha_qt_shell ollama_client.py`).
4. ~~Compare **RFF vs VectorEncoder** on 384-d ST inputs~~ — **VectorEncoder 59.5%** vs RFF **4.5%** @ 2k MiniLM (`cypha_branch_a_encoder_sweep.json`). Keep VectorEncoder for Branch A.

---

## Encoder sweep (384-d MiniLM @ 2k)

Artifact: `bench/config/cypha_branch_a_encoder_sweep.json`

| Encoder | Accuracy | Notes |
|---------|----------|-------|
| **VectorEncoder** (frozen W_enc) | **59.5%** | Default Branch A path |
| RFFEncoder D=256 | 4.5% | Wrong tool for 384-d semantic vectors |
| LogReg baseline | 60.3% | Batch reference |

**Verdict:** RFF is for small tabular dims (≤30); frozen **VectorEncoder** is required for sentence-transformer inputs.

---

## REST routing (CyphaStudio)

| Route | Purpose |
|-------|---------|
| `GET /route/health` | Router trained?, Ollama reachable?, CyphaLM loaded? |
| `POST /route/text` | Embed → classify → epistemic gate (no generation) |
| `POST /route/generate` | Route then **CyphaLM** (in-domain) or **Ollama** (OOD abstain) |

Environment (see [`docs/studio/CYPHA_ENV.md`](studio/CYPHA_ENV.md)):

| Variable | Default |
|----------|---------|
| `CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD` | `0.5` |
| `CYPHA_BRANCH_A_N_TRAIN` | `1200` |
| `CYPHA_BRANCH_A_EMBED_BACKEND` | `auto` |
| `CYPHA_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `CYPHA_OLLAMA_MODEL` | `mistral` |
| `CYPHA_BRANCH_A_CHECKPOINT` | `~/.cypha/branch_a_router` (``.json`` + ``.npz``) |
| `CYPHA_BRANCH_A_AUTO_SAVE` | Save checkpoint after train when `1` |
| `CYPHA_LM_CHECKPOINT` | *(optional CyphaLM for in-domain gen)* |

```powershell
uvicorn cypha_qt_shell / cypha_rest.server.api:app --port 7749

curl -s -X POST http://127.0.0.1:7749/route/text `
  -H "Content-Type: application/json" `
  -d '{"text":"How do I compile Linux kernel modules?"}'

curl -s -X POST http://127.0.0.1:7749/route/generate `
  -H "Content-Type: application/json" `
  -d '{"text":"quantum gardening on Mars","max_tokens":64}'


# Pre-train checkpoint (skip ~30s retrain on REST cold-start)
$env:CYPHA_BRANCH_A_CHECKPOINT="$HOME/.cypha/branch_a_router"
```

**Studio GUI:** File → Settings → Inference → enable **Branch A text routing**. Chat embeds queries, shows route label/epistemic, streams CyphaLM in-domain or calls Ollama on abstain.

## References

- Embeddings adapter: `bench/adapters/frozen_text_embeddings.py`
- D09 documents: `bench/domains/d09_documents.py`
- Phase 2 encoder study: [`CYPHA_TESTS_PHASE2.md`](CYPHA_TESTS_PHASE2.md)
