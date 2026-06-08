# Branch A — CyphaDIF on Frozen Semantic Embeddings

**Maps:** [`Cypha Tests.txt`](../Cypha%20Tests.txt) Branch A → CyphaDIF as NLP routing/classification layer on top of a frozen encoder.

**Runner:** `python cypha_bench/tuning/cypha_branch_a_sweep.py --write`

Optional dependency for semantic embeddings:

```powershell
pip install sentence-transformers
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

Artifact: `cypha_bench/config/cypha_branch_a_sweep.json`  
Encoder: **sentence-transformers/all-MiniLM-L6-v2** (384-d, frozen)

| Method | Accuracy | Notes |
|--------|----------|-------|
| CyphaDIF + TF-IDF/SVD (D09 reference) | **34.0%** | Current D09-style path |
| CyphaDIF + frozen ST, online `W_enc` | **49.3%** | +15.3pp vs TF-IDF |
| LogReg + frozen ST | **60.3%** | Batch baseline |
| **CyphaDIF + frozen ST, frozen `W_enc`** | **62.5%** | **Best** — online field only |

**Verdict:** Frozen semantic embeddings unlock CyphaDIF on text — **+28.5pp vs TF-IDF** at 2k samples. Freezing `EncoderProjection` (**62.5%**) beats online projection tuning (**49.3%**) and matches/beats batch LogReg on this split.

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
python cypha_bench/tuning/cypha_branch_a_sweep.py --n-samples 2000 --backend sentence_transformers --write

# Offline smoke
python cypha_bench/tuning/cypha_branch_a_sweep.py --n-samples 800 --backend hashing --write --out cypha_bench/config/cypha_branch_a_hashing.json
```

---

## Next steps

1. Wire frozen-ST path into **D09** as optional experiment (`CYPHA_BENCH_BRANCH_A=1`).
2. OOD eval: Gutenberg segments vs 20news (epistemic separation) on ST vectors.
3. **Local LLM routing:** embed user query → CyphaDIF route → Ollama/Mistral generate (REST stub).
4. Compare **RFF vs VectorEncoder** on 384-d ST inputs for small-dim regime.

---

## References

- Embeddings adapter: `cypha_bench/adapters/frozen_text_embeddings.py`
- D09 documents: `cypha_bench/domains/d09_documents.py`
- Phase 2 encoder study: [`CYPHA_TESTS_PHASE2.md`](CYPHA_TESTS_PHASE2.md)
