# mined/qwen3emb

**Odin Loch** · 2026-10-28 · small outputs of the Qwen3-Embedding dissection ([report](../../docs/reports/QWEN3_EMBEDDING_DISSECTION.md))

Every directory written by `scripts/gguf_mine.py` has a `provenance.json`: model file and SHA-256, licence, versions, command, seed, input hashes. All teacher output is low-trust (EMBEDDING_GEOMETRY §11).

| Directory | What | Report |
|---|---|---|
| `info/<model>/` | `info` on each GGUF: metadata, tensor table, licence (A1, A3) | §3.1 |
| `qe0/` | The pooling gate: 20 sentences, cosines against `llama-embedding` with flash attention off and on, the EOS decision; the Qwen3.5-9B smoke test | §3.2–3.3, §4.3 |
| `timing/` | `embedtexts --limit 1000` before and after the thread fix; `llama-bench` tables | §8 |
| `static_0.6b/` | `vocab` and `embed` summaries, geometry of 20,000 input word vectors, final-norm gains (A1, A2, A7) | §5.1–5.3 |
| `static_0.6b/` (spectra, heads, neurons) | per-layer summaries of `spectra`, `heads` and `neurons` | §5.5–5.7 |
| `static_8b/` | the same on Qwen3-Embedding-8B | §5.8 |
| `wdiff/` | A4: the embedder against its base (F16 and Q8_0), against the instruct model, and 8B against its base; the numerical rank of ΔW; the last-layer change against the output mean | §5.4, §5.8 |
| `provisional_timing1000/` | model-free `geometry` on the 1,000 vectors the timing run had made; provisional only | §7 |

Large files (vectors, full neighbour and neuron tables) stay in `/root/work/teacher/qe/` on the build machine.
