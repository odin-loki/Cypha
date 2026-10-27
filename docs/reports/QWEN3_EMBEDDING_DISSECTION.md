# Qwen3-Embedding dissection: QE0, the probe set and the static dissection

**Odin Loch**, 2026-10-28

Plan: [EMBEDDING_GEOMETRY.md](../EMBEDDING_GEOMETRY.md) §6 (dissection IDs), §9 (logistics), §10 (QE0–QE9, predictions Q1–Q10).
Tool: `scripts/gguf_mine.py` ([GGUF_MINING.md](../GGUF_MINING.md) §14). Small outputs: `mined/qwen3emb/`. Probe set: [mined/probe/README.md](../../mined/probe/README.md).
Machine: the cloud container, 4 shared cores (2 used, `nice -n 5`), 15 GB RAM. Load average 3–8 from other agents during every run.

## In short

- **QE0 gate: hit.** `embedtexts --pooling last` reproduces `llama-embedding --pooling last` on 20 sentences. Every cosine is at least 0.9999999 with both on the same attention path.
- **Qwen3.5-9B loads** in llama-cpp-python 0.3.36, and its chat template renders with thinking off. It runs at 10.6 tokens/s for prompts and 1.91 tokens/s for generation on 2 cores here.
- **Three bugs fixed in `gguf_mine.py`.** The worst one made every batch decode ignore `--threads`. Fixing it made `embedtexts` 3.1× faster here.
- **The probe set is built and hashed:** 135,895 items, 2,455,468 tokens. A rebuild gives the same hashes.
- **A4: the weight change from the base is small and low-rank. Q4 hit**, in all 196 matrices.
  - Contrastive training left the token embeddings and every norm untouched. It changed the linear layers by 3.3%, and the last FFN down-projection by 31%.
  - That last change is almost rank 1, and it shapes the common direction of the output vectors.
- **The dynamic dissection did not run.** Odin stopped all inference in this container on 2026-10-28; it runs on his PC. Section 9 gives the commands, sized from the timing here. Q1–Q3, Q5, Q6 and Q10 are not scored yet. Section 7 has a provisional look at the 1,000 vectors the timing run had already made.

## 1. What ran, and what did not

| Step | Status |
|---|---|
| Setup: venv, llama-cpp-python 0.3.36 built from source, `llama-embedding` and `llama-bench` built from the same vendored llama.cpp | done |
| Downloads, SHA-256 checked against the Hugging Face LFS hashes | done (§2) |
| `info` on every GGUF (A1, A3) | done |
| QE0 pooling gate on 0.6B; EOS decision | done before the stop (§3) |
| Qwen3.5-9B smoke load, template, one answer, speed | done before the stop (§4) |
| Probe set (QE2 inputs) | done (§6) |
| Timing: `embedtexts --limit 1000` on 0.6B | done before the stop (§8) |
| Static dissection of 0.6B: `vocab`, `embed`, token-matrix geometry, `spectra`, `heads`, `neurons`, `wdiff` against the base and the instruct model | done (§5) |
| Static dissection of 8B | see §5.8 |
| QE2 probe vectors, QE3 `geometry`, `sensitivity`, `occlusion`, `instruct`, 8B timing, 0.6B against 8B | **not run: Odin's PC** (§9) |
| QE4 `pairs` with Qwen3.5-9B; teacher Q/A text | not started (next wave; Odin's PC) |

The 0.6B probe run had started and was stopped after 180 s, before its first part was written. No probe-set vectors exist yet.

## 2. Downloads

| File | Source (revision) | Bytes | SHA-256 | Licence |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B-Q8_0.gguf | Qwen/Qwen3-Embedding-0.6B-GGUF (`370f27d7`) | 639,150,592 | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` | Apache-2.0 |
| Qwen3-Embedding-0.6B-f16.gguf | same | 1,197,629,632 | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` | Apache-2.0 |
| Qwen3-0.6B-Base.Q8_0.gguf | mradermacher/Qwen3-0.6B-Base-GGUF (`b8b11a6d`), converted from Qwen/Qwen3-0.6B-Base | 639,447,232 | `4b088f1793f6cba9f0c2f77ab835ef6734f205c2159168698c6e1a51b7df168a` | Apache-2.0 |
| Qwen3-0.6B-Base.f16.gguf | same | 1,198,182,592 | `d7f3a9d59cc8de30bbc9f1feda8d1e1809b2aadd139b72a0342c769a14002b37` | Apache-2.0 |
| Qwen3-0.6B-Q8_0.gguf (instruct) | Qwen/Qwen3-0.6B-GGUF (`23749fef`) | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Apache-2.0 on the card; the GGUF has no `general.license` key, so `info` warns |
| Qwen3.5-9B-Q4_K_M.gguf | unsloth/Qwen3.5-9B-GGUF (`3885219b`) | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | Apache-2.0 |
| Qwen3-Embedding-8B-Q8_0.gguf | Qwen/Qwen3-Embedding-8B-GGUF (`69d0e58a`) | 8,047,105,824 | `d20ddc71e8a5c4344f2343481e242233a997dc5eaff442427a945836c97b4deb` | Apache-2.0 |

The Qwen team publishes no base-model GGUF for Qwen3-0.6B, so the base is mradermacher's conversion of the official `Qwen/Qwen3-0.6B-Base`. Its metadata names that source (`general.source.url`). Every file was deleted after use to keep under the disk budget; the hashes above identify them.

## 3. QE0: metadata, EOS and the pooling gate

### 3.1 What `info` says (A1, A3)

| | Embedding 0.6B | Base 0.6B | Embedding 8B | Qwen3.5-9B |
|---|---|---|---|---|
| Architecture | qwen3 | qwen3 | qwen3 | qwen35 |
| Layers × width (FFN) | 28 × 1,024 (3,072) | 28 × 1,024 (3,072) | 36 × 4,096 (12,288) | 32 × 4,096 (12,288) |
| Heads / KV heads | 16 / 8 | 16 / 8 | 32 / 8 | 16 / 4 (head dim 256) |
| Vocabulary rows | 151,669 | 151,936 (padded) | 151,665 | 248,320 |
| Parameters | 595,776,512 | 596,049,920 | 7,567,295,488 | 8,953,803,264 |
| `add_eos_token` | **true** | not set | **true** | not set |
| EOS id | 151643 `<|endoftext|>` | 151645 | 151643 | 248046 |
| `pooling_type` | 3 (last) | — | 3 (last) | — |
| Output head (A3) | none (tied to `token_embd`) | tied | none | separate |

### 3.2 The EOS decision

`embedtexts` reads `add_eos_token = true` and checks the tokenizer: tokenising with special tokens on appends token 151643 itself. So the plan is `tokenizer`: the script appends nothing, and the input ends in `<|endoftext|>` as the model card asks. `llama-embedding --verbose-prompt` shows the same token at the end of every prompt, and the token counts agree on all 20 sentences.

### 3.3 The gate

20 hand-written sentences (`mined/qwen3emb/qe0/gate20.txt`): short and long, a question, numbers, code, French, a one-word greeting.

| Comparison | Min cosine | Mean | ≥ 0.999 |
|---|---|---|---|
| `embedtexts` against `llama-embedding -fa off` | 0.99999997 | 0.99999998 | 20 / 20 |
| `embedtexts` against `llama-embedding -fa on` | 0.99883 | 0.99959 | 19 / 20 |
| `llama-embedding -fa off` against `-fa on` | 0.99883 | 0.99959 | 19 / 20 |

- **QE0 gate: hit.** With the same attention path the two agree to 7 decimals.
- **Flash attention.** llama-cpp-python turns it off by default. `llama-embedding` defaults to `auto`, which turns it on for CPU. Flash attention alone moves one sentence ("Photosynthesis converts light energy…") to 0.99883. That is llama.cpp's own numerical gap, not the script's.
- Storing the vectors as fp16 costs less than 3 × 10⁻⁸ in cosine.
- The fix in §4.1 changes nothing: a rerun gives bit-identical vectors.

## 4. Bugs found on real models, and Qwen3.5-9B

### 4.1 Fixed in `gguf_mine.py`

1. **`--threads` was ignored for every batch decode.** `Runner` passed `n_threads` but not `n_threads_batch`. llama-cpp-python sets the latter to `cpu_count()`, so embedding and prompt decodes used all 4 cores. That oversubscribes a shared machine. The fix passes both.
   - On the same 1,000 probe items, `embedtexts` went from 19.0 to 58.3 tokens/s.
   - The vectors are bit-identical (all 1,000, maximum difference 0).
2. **`wdiff` skipped vocabulary matrices padded to another size.** The embedder's `token_embd` has 151,669 rows and the base's has 151,936. `wdiff` reported a shape mismatch. Now it compares the shared rows when both token lists agree on them.
3. **`spectra` copied a whole tensor to float64 before sampling rows.** For the 8B vocabulary matrix that is 5 GB on top of 2.5 GB. It now samples first. The same random rows are drawn, so the output is unchanged.

The self-test passes all 50 rows on both vocabularies after these fixes (§10).

### 4.2 Not a bug, but a cost

In embedding mode llama.cpp marks every token as an output. It still runs the tied output head (151,669 × 1,024) on each one. That adds about 35% compute for the 0.6B model and about 9% for the 8B.

### 4.3 Qwen3.5-9B smoke test (`mined/qwen3emb/qe0/smoke_qwen3.5-9b.json`)

The test drives `gguf_mine.py`'s own `Runner`, `PromptFormat` and `judge`.

- **It loads.** The architecture is `qwen35`; loading takes 21.6 s.
- **The template renders.** It goes through `Jinja2ChatFormatter` with no warning. With thinking off, the prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`; the template writes the empty block itself. With thinking on, it ends `<think>\n`.
- **The answer.** "What is the capital of France? Answer in one short sentence." gives, greedily, "The capital of France is Paris." (7 tokens).
- **The judge.** `pairs`' yes/no verification scores +4.64 for "Paris is the capital and largest city of France." against the query "capital of France". It scores −7.65 for "The Loire is the longest river in France."
- **Speed.** `llama-bench` with 2 threads gives 10.60 ± 0.02 tokens/s for a 128-token prompt and 1.91 ± 0.02 tokens/s for generation (load average 4.3–6).
- **Memory.** Peak RSS is 9.2 GB at n_ctx 1,024. The generative `Runner` keeps logits for every position, n_ctx × 248,320 floats, which is 1.0 GB here. Use a small `--ctx` for `pairs`.

## 5. QE1: the static dissection (weights only)

### 5.1 Tokenizer (A1, `vocab`)

- **Size.** 151,669 entries: 151,643 normal, 20 control and 6 user-defined. Byte-level BPE with 151,387 merges and no scores, so the frequency prior is the merge order.
- **Scripts.** Latin 94,626, CJK 25,922, Cyrillic 4,149, Arabic 3,979, Hangul 3,589, Hebrew 3,183, Thai 2,550.
- **Words.** 47,575 whole-word tokens. The mean token is 5.10 characters long, and 35.0% of tokens start a word.

### 5.2 Input embedding matrix (A2, `embed` and `geometry` on 20,000 word vectors)

| Measure | Value |
|---|---|
| Mean cosine of random word pairs | 0.067 (0.000 after centring) |
| Norm of the mean unit vector | 0.261 |
| Effective rank; PCA dimensions for 95% | 780; 817 of 1,024 |
| Top principal component | 1.25% of variance |
| TwoNN intrinsic dimension | 7.9 |
| Hubness (k-occurrence skew at k = 10) | **48.9**, and 1.03 under CSLS |
| Spearman of norm against merge rank | 0.236 (rarer words have slightly larger norms) |
| Prefix agreement at 32 / 256 dimensions | 0.573 / 0.858 (no nesting in the input matrix) |

- **Near-isotropic.** The input matrix is close to isotropic, so the anisotropy of the output (§7) is made by the layers.
- **Hubs.** The worst hub, "eskorte", is among the 10 nearest neighbours of 1,032 of 5,000 sampled words. The next are "prostituerade" (425), "bureaucr" and "Sexe". These look like rarely trained spam and fragment tokens near the centre. CSLS removes them.
- **Relations.** A leave-one-out offset finds the answer first for capital-of 0.67 (12 pairs), female-of 0.70, plural 0.70, past tense 0.70 and comparative 0.90. It finds it in the top 10 for 0.83, 1.0, 1.0, 1.0 and 0.90.
- **Neighbours are sensible.** bank → Bank, banks, banking, lender, loan. Paris → paris, France, London, Madrid, French.

### 5.3 Final RMSNorm gains (A7)

The 1,024 gains have median 3.64. The largest are dimension 1 at 15.3 (4.2× the median), then dimension 7 at 11.7 and dimension 27 at 9.8. They are identical to the base's.

### 5.4 Weight difference from the base (A4, `wdiff`)

**Which model is the parent?** The base. The instruct model is twice as far, and its difference is full rank.

| `wdiff` | Total relative change ‖ΔW‖/‖W‖ | Matrices with top-5% share ≥ 0.5 | Top-5% share, median (min) | `token_embd` change | Norm changes |
|---|---|---|---|---|---|
| F16 embedder − F16 base | **0.0332** | **196 / 196** | 0.980 (0.907) | **0 exactly** | **0 exactly** |
| Q8_0 embedder − Q8_0 base | 0.0334 | 196 / 196 | 0.965 (0.874) | 0 | 0 |
| Q8_0 embedder − Q8_0 instruct | 0.0663 | 11 / 197 | 0.343 (0.247) | 0.133 | up to 0.016 |

Q8_0 noise lowers the share a little (median 0.980 → 0.965) but changes no conclusion.

**What changed (F16).**
- **Untouched:** the token embeddings, every RMSNorm gain (input, attention, Q/K and final) and the final norm. Only the linear layers moved.
- **By layer:** the relative change is 3.8% in layer 0, peaks at 6.0% in layer 3, stays at 4–6% to layer 18, then falls to 1.8% in layer 26. Layer 27 is at 3.4%.
- **By matrix type:** the median relative change is largest for `ffn_gate` (9.4%) and smallest for `attn_v` (4.9%).
- **The largest single changes are in the last three layers:** `ffn_gate` of layers 24–26 (13.5–15.0%), `ffn_up` of layer 27 (12.7%), and **`ffn_down` of layer 27 (31.5%)**.
- **Low rank, but not one small adapter rank.** Stable ranks are 1.1–7. In 35 matrices from layers 0, 3, 10, 20 and 27, 90% of ‖ΔW‖² lies in a median of 22 singular values and 99% in a median of 67 (range 3–104) of 1,024. There is no sharp cut-off at 8, 16, 32 or 64. The largest gap between neighbouring singular values is at most 2.7×, except in layer 27's `ffn_down`.
- **The last down-projection is almost rank 1.** 93.4% of its change lies in the top singular direction and 99.5% in the top three (largest gap 4.3×).
  - Its second direction, scaled by the final norm gains, has |cosine| 0.52 with the mean of the model's output vectors. A random direction has 0.061 at the 95th percentile. (The mean is of the 1,000 timing vectors, §7.)
  - The three directions hold 30% of that mean direction.
  - So the contrastive fine-tune shapes the output's common component in this one matrix.
- **By neuron (Q8_0, `ffn_down` columns).**
  - In layers 0–26, each neuron's write vector changed by a median of 4.7–6.7%. The change is spread out: half of a layer's ‖ΔW‖² takes 670–1,040 of its 3,072 neurons, and 90% takes 2,150–2,490. A low-rank change written by many neurons looks like this.
  - In layer 27 the median neuron changed by 23.5% (largest 195%), and half the change sits in 422 neurons.
  - Read through the tied embedding, the change in the most-changed layer-27 neurons promotes the same odd tokens every time ("生产总", "游戏当中", "getEmail", "删除成功"). It is one geometric direction, not a word-level feature.
  - Output: `mined/qwen3emb/wdiff/ffn_down_neuron_changes_0.6b.json`.

**Q4 (the top 5% of singular values hold at least 50% of ‖ΔW‖² in most matrices): hit.** 196 of 196 matrices, with a median share of 0.980 (F16).

**What it means for Cypha (EMBEDDING_GEOMETRY §6 A4).** The embedder is its base plus a 3% low-rank edit, with identical token embeddings. The knowledge comes from the base; contrastive training mostly rearranged the geometry, and most strongly in the last layers. That supports learning Cypha's geometry last and cheaply, on top of knowledge it already has.

### 5.5 Spectra (`spectra`, all 197 matrices of both models)

| Matrix type | Heavy-tail α, embedder (base) | Stable rank, embedder (base) |
|---|---|---|
| attn_q | 3.42 (3.46) | 51.8 (52.0) |
| attn_k | 3.09 (3.09) | 61.8 (61.6) |
| attn_v | 5.46 (5.34) | 150.8 (150.6) |
| attn_output | 5.58 (5.58) | 135.0 (134.6) |
| ffn_gate | 5.05 (5.27) | 73.8 (75.1) |
| ffn_up | 7.43 (8.04) | 201.3 (213.0) |
| ffn_down | 5.65 (5.67) | 118.1 (125.5) |

The medians are over the 28 layers. The embedder's spectra are the base's almost exactly: median α 5.01 against 5.13, with 72% against 69% of matrices in the "well trained" band 2–6. The only clear shift is in the FFN: ffn_up's α falls from 8.04 to 7.43, and the stable ranks of ffn_up and ffn_down fall by 5–6%. A 3% low-rank edit hardly moves a spectrum.

### 5.6 Attention copying circuits (A6, `heads --tokens 300`, all 448 heads of both models)

Each head's OV circuit is read through the tied embedding: does a word's own token come out on top?

| | Mean copy@1 | Heads with copy@1 > 0.5 | Heads > 0.2 |
|---|---|---|---|
| Embedder | 0.0621 | 23 | 44 |
| Base | 0.0628 | 24 | 44 |

The copying heads are spread over layers 4–27, at most two per layer. The strongest are layer 7 head 12 (copy@1 0.87), layer 10 head 4 (0.86), layer 25 head 0 (0.84) and layer 9 head 12 (0.83). No head's copy score moved by more than 0.023. **Contrastive training did not touch the copying circuits.** For Cypha's context views (A6), the base model's circuits are the embedder's. Each run took 13–17 minutes on 2 cores.

### 5.7 FFN neurons as key–value memories (A5, `neurons --max-neurons 1024`, 28,672 neurons)

Each neuron's write vector (a column of `ffn_down`) is read through the final norm and the tied embedding. The embedder has no output head, and its `token_embd` is the base's, so this is the base model's logit lens.

- **Entropy is uninformative here.** One neuron's logits are small: the median top logit is 0.38 in layer 1 and rises to 0.61 by layer 23. So the write entropy is 11.91–11.93 for every neuron, against a maximum of ln 151,669 = 11.93. A sharper statistic, such as a temperature or the logit margin, would separate neurons; the script does not compute one.
- **What the top tokens say.** Early layers write subword fragments with no theme. From about layer 13, many neurons write one concept, often across languages, and read its trigger:
  - layer 13: writes "again, _REPEAT, 回来, reconnect, rebound"; reads "retro, BACK, revert";
  - layer 19: writes "negatives, minus, Negative, negativity"; reads "Positive, positive";
  - layer 21: writes "addition, +, 加, Adds, .add";
  - layer 22: writes "image, 图片, images, picture, 照片"; reads "description, images, pictured";
  - layer 24: writes "都需要, 都被, 都会" (all, both); reads "these, those, several".
- These are hypotheses for the DISCOVERY firewall, not facts.

The summary per layer is in `mined/qwen3emb/static_0.6b/neurons_summary.json`. The full table (9.3 MB) stays on the build machine. The run took 24 minutes.

EIGHTB_PLACEHOLDER

## 6. QE2: the probe set

Built and hashed; details in [mined/probe/README.md](../../mined/probe/README.md).

| | Items | Qwen3 tokens (EOS counted) |
|---|---|---|
| `probe.txt` (words, phrases, titles, sentences, Dolly instructions and response leads) | 135,895 | 2,455,468 |
| `pos_pairs.tsv` (anchor, positive) | 24,240 pairs | over probe items |
| `classes.tsv` (a, b, class; 18 classes) | 51,073 pairs | over probe items |
| `held_out.txt` (sentences not in the probe) | 4,000 | 95,008 |
| `questions.txt` (Dolly questions) | 2,000 | 24,869 |
| `timing_1000.txt` (= first 1,000 lines of probe.txt) | 1,000 | 18,352 |

- **Exclusions.** No T-QA question is in it, and no test split of STS-B, MRPC, PAWS or SNLI.
- **Splits.** Every item has a dev or test split from a hash of its text (20.1% test).
- **QE2 gate (reproducible to the hash): hit** for the inputs. A second build gave the same SHA-256 for all nine files.
- **Two nested subsets for runs that cannot take the full set**, both from scratch builders recorded in the probe README:
  - the 0.6B pilot: 600 pairs per class, all words, phrases and titles, and 6,000 more wiki sentences. 67,379 items, 745,914 tokens.
  - an 8B subset inside it: 50 pairs per class, 3,000 words and phrases, all titles, 500 more wiki sentences. 10,239 items, 82,063 tokens.

## 7. A provisional look at the 1,000 timing vectors (no new inference)

The timing run had already embedded the first 1,000 probe items (the same random mix of words, phrases and sentences as the full set). `geometry` reads vectors only, so it was run on them. **These numbers are provisional.** They are not the registered measurements, which need the probe set and its pairs (§9).

Outputs are in `mined/qwen3emb/provisional_timing1000/`. The intervals are 95% percentile bootstraps: over items for the pair means, over anchors for MRL.

| Measure (0.6B, 1,000 items: 582 sentences, 142 words, 142 phrases, others) | Value |
|---|---|
| Mean cosine of random pairs, all items | 0.278 [0.270, 0.288] |
| … sentences only | **0.216** [0.210, 0.221] |
| … held-out wiki sentences only (282) | 0.230 [0.222, 0.237] |
| … words only / phrases only | 0.600 [0.590, 0.611] / 0.538 [0.528, 0.551] |
| After centring (all items) | −0.001 |
| Norm of the mean unit vector | 0.529 |
| Effective rank; PCA dimensions for 95% | 246; 379 of 1,024 |
| TwoNN intrinsic dimension | 23.4 |
| Hubness skew at k = 10 | 3.32, and 1.12 under CSLS |
| Spearman of log length against cosine to the centroid | −0.75 |
| Prefix agreement with 1,024 dims, all items: 32 / 64 / 128 / 256 / 512 | 0.884 / 0.926 / 0.954 / 0.972 / 0.988 |
| Prefix agreement, sentences only: 32 / 256 | **0.783** [0.751, 0.810] / **0.937** [0.927, 0.946] |

- **A floor exists.** Random sentence pairs average 0.22, which is inside Q1's band of 0.05–0.5. Q1's registered measure is the "unrelated" class of the probe set, so it is not scored here.
- **Short items crowd together.** Words average 0.60 with each other, against 0.22 for sentences. Shorter items also sit nearer the centroid (Spearman −0.75), so raw cosine thresholds must depend on length (B17).
- **The output is far more anisotropic than the input.** The mean vector's norm is 0.53 here, against 0.26 for the input word vectors, and the effective rank is 246 against 780.
- **Nesting depends on the mix of items.** On mixed items the 256 and 32 prefixes clear Q3's bars (0.972, 0.884). On sentences alone both fall just short (0.937, 0.783), with intervals that exclude 0.95 and 0.80. Q3 is defined at the 8B's 4,096 dimensions and on the probe set, so it waits for §9.

## 8. Timing and job sizes

### 8.1 Measured here (2 threads of 4 shared cores, Q8_0)

| Run | Items | Tokens | Decode time | Items/s | Tokens/s | Load average |
|---|---|---|---|---|---|---|
| 0.6B `embedtexts`, 1,000 probe items, before the thread fix | 1,000 | 18,352 | 965 s | 1.04 | 19.0 | 5.9–8.1 |
| 0.6B `embedtexts`, 1,000 probe items, after the fix (`--batch 256`) | 1,000 | 18,352 | 315 s | 3.18 | **58.3** | 4.2–4.5 |
| 0.6B `llama-bench`, 512-token prompt, flash attention off | | | | | 78.2 ± 1.7 | ~4.6 |
| Qwen3.5-9B Q4_K_M `llama-bench` prompt (128) / generation (32) | | | | | 10.6 / 1.91 | 4.3–6.0 |

### 8.2 Estimates

The 8B embedder does 12.7× the 0.6B's work per token: 7.57B against 0.596B parameters, each including the tied output head that embedding mode runs on every token. The brief's 13× agrees. At this machine's rates:

| Job | Tokens | 0.6B at 58.3 tokens/s | 8B at about 4.6 tokens/s |
|---|---|---|---|
| Full probe set | 2,455,468 | 11.7 h | 148 h (6.2 days) |
| 0.6B pilot subset | 745,914 | 3.6 h | 45 h |
| 8B subset | 82,063 | 23 min | 5.0 h |
| 8B timing, first 200 probe items | 3,981 | 68 s | 14 min |
| `occlusion`, 2,000 held-out sentences (45,054 texts) | 888,387 | 4.2 h | 54 h |
| `instruct`, 2,000 questions × 5 instructions (12,000 texts) | 310,874 | 1.5 h | 19 h |
| `sensitivity`, 200 per kind (about 3,600 texts, estimated) | about 86,000 | 25 min | 5.2 h |

- **Qwen3.5-9B mining (QE4, next wave).** The cost per document is a property of `pairs`, not of this machine. With the default four query types, one query each, four hard negatives and a verification of every row, one document needs about 2,200 prompt tokens and 340 generated tokens, with thinking off. That is about 6.4 minutes per document here (1,000 documents in 107 h). Use Odin's own `llama-bench` rates in the same sum.
- **Odin's PC.** Rates there will be far higher. Time 1,000 items first (§9) and scale this table by the ratio.

## 9. What remains, on Odin's PC

Run from the repository root with llama-cpp-python 0.3.36 or newer, `gguf`, `numpy` and `pyarrow`. The fixed `gguf_mine.py` honours `--threads` (set it to the PC's cores). `PR` is the probe directory (the files in `mined/probe/` plus `probe.txt`, `probe_meta.tsv`, `pos_pairs.tsv` and `classes.tsv`, rebuilt by its README).

```bash
E=Qwen3-Embedding-0.6B-Q8_0.gguf; E8=Qwen3-Embedding-8B-Q8_0.gguf; G=Qwen3.5-9B-Q4_K_M.gguf
PR=probe; O=mined/qwen3emb; W=work; T=8   # T = threads
GM="python scripts/gguf_mine.py"

# QE0 on the 8B (as on the 0.6B: cosine >= 0.999 on every sentence, flash attention off on both sides)
$GM embedtexts --model $E8 --out $W/qe0_8b --input $O/qe0/gate20.txt --threads $T
llama-embedding -m $E8 --pooling last -f $O/qe0/gate20.txt --embd-output-format json --embd-normalize 2 -fa off -t $T

# Timing (size everything below from these)
$GM embedtexts --model $E  --out $O/timing_pc_0.6b --input $PR/probe.txt --limit 1000 --batch 256 --threads $T
$GM embedtexts --model $E8 --out $O/timing_pc_8b   --input $PR/probe.txt --limit 200  --batch 256 --threads $T

# QE2: teacher vectors of the whole probe set (resumable)
$GM embedtexts --model $E  --out $W/qe/probe    --input $PR/probe.txt --batch 256 --chunk 2000 --resume --threads $T
$GM embedtexts --model $E8 --out $W/qe/probe_8b --input $PR/probe.txt --batch 256 --chunk 2000 --resume --threads $T

# QE3 on each model (shown for 0.6B; for 8B use $E8, probe_8b and --prefixes 32,...,4096)
$GM geometry    --vectors $W/qe/probe/embeddings.npy --out $O/geometry_0.6b --key text \
                --pairs $PR/pos_pairs.tsv --labelled-pairs $PR/classes.tsv --lengths index --prefixes 32,64,128,256,512,1024
$GM sensitivity --model $E --out $O/sensitivity_0.6b --sentences $PR/held_out.txt --threads $T
$GM occlusion   --model $E --out $O/occlusion_0.6b --sentences $PR/held_out.txt --freq $PR/word_counts.tsv --threads $T
$GM instruct    --model $E --out $O/instruct_0.6b --queries $PR/questions.txt --threads $T
$GM compare     --a $W/qe/probe_8b/embeddings.npy --b $W/qe/probe/embeddings.npy --key text \
                --pairs $PR/pos_pairs.tsv --out $O/compare_8b_vs_0.6b

# QE4 (next wave): Qwen3.5-9B pairs, thinking off; the audit with thinking
$GM pairs --model $G --out $O/pairs --docs docs.txt --personas personas.txt --embedder $E8 --ctx 1024 --resume --threads $T
$GM pairs --model $G --out $O/audit --audit masked_candidates.tsv --think --ctx 2048 --threads $T
```

- **If the full set is too slow,** use the nested subsets: `probe_pilot.txt` with `pos_pairs_pilot.tsv` and `classes_pilot.tsv`, or `probe_s8b.txt` with its pairs files (probe README).
- **What each run scores.** `geometry` scores Q1–Q3 and gives B1–B7, B13 and B17. `sensitivity` scores Q6, `occlusion` Q5 and `instruct` Q10. `compare` gives what scale adds.
- **Bootstrap intervals.** For these, `/root/work/teacher/bin/analyse.py` (a scratch reader of these outputs, described in §10) gives percentile intervals for Q1, Q2, Q3 and Q6, plus the B2 AUCs and the STS-B Spearman.
- **Keep the vectors.** The 0.6B vectors belong at `work/qe/probe/embeddings.npy` with `embeddings_index.tsv`, so that Cypha's exported vectors can be compared to them with `compare --key text`.

## 10. Predictions

| # | Prediction | Result |
|---|---|---|
| Q1 | Floor: mean cosine of unrelated sentence pairs in 0.05–0.5 | **not scored** (needs the probe pairs; provisional value in §7) |
| Q2 | Effective temperature in 0.01–0.1 | **not scored** (needs probe-set positives) |
| Q3 | MRL: 256 dims keep ≥ 95% of the full pair Spearman, 32 dims keep ≥ 80% | **not scored** (provisional value in §7) |
| Q4 | Weight change concentrated: top 5% of singular values hold ≥ 50% of ‖ΔW‖² in most matrices | **hit** on 0.6B: 196 / 196 matrices, median share 0.980, minimum 0.907 (F16). EIGHTB_Q4 |
| Q5 | Additivity: weighted word sum reaches ≥ 0.8 pair Spearman | **not scored** (`occlusion`, Odin's PC) |
| Q6 | Median cosine of "X is Y" against "X is not Y" ≥ 0.8 | **not scored** (`sensitivity`, Odin's PC) |
| Q10 | One offset per instruction explains ≥ 70% of its shift | **not scored** (`instruct`, Odin's PC) |

Q7–Q9 belong to QE5–QE7 on the Cypha side.

## 11. Reproduce

Setup (scratch directory `/root/work/teacher`):

```bash
python3 -m venv venv && venv/bin/pip install numpy gguf==0.19.0 pyarrow
curl -LO https://files.pythonhosted.org/packages/ec/e9/e7de2b0463ea3ffbf0ede6cb21b58c1258a8f6521aae45ca773a59fe7cf3/llama_cpp_python-0.3.36.tar.gz
# sha256 832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e
tar xzf llama_cpp_python-0.3.36.tar.gz
CMAKE_BUILD_PARALLEL_LEVEL=2 venv/bin/pip install ./llama_cpp_python-0.3.36
cmake -S llama_cpp_python-0.3.36/vendor/llama.cpp -B llama_build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build llama_build --target llama-embedding llama-bench -j2
```

Models: `curl -L https://huggingface.co/<repo>/resolve/main/<file>` for each row of §2, then check the hashes.

Static dissection (from the repository root; `M` holds the GGUFs):

```bash
GM="python scripts/gguf_mine.py"; O=mined/qwen3emb
$GM info    --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out $O/info/qwen3-embedding-0.6b-q8_0   # and each GGUF
$GM vocab   --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/vocab_0.6b
$GM embed   --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/embed_0.6b
$GM geometry --vectors work/embed_0.6b/vectors.npy --index work/embed_0.6b/vectors_index.tsv \
             --out work/geometry_static_0.6b --prefixes 32,64,128,256,512,1024 --no-knn-graph
$GM spectra --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/spectra_emb_0.6b     # and the base
$GM heads   --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/heads_emb_0.6b --tokens 300   # and the base
$GM neurons --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/neurons_emb_0.6b --max-neurons 1024
$GM wdiff   --model $M/Qwen3-Embedding-0.6B-f16.gguf --base $M/Qwen3-0.6B-Base.f16.gguf --out $O/wdiff/f16_vs_base_f16
$GM wdiff   --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --base $M/Qwen3-0.6B-Base.Q8_0.gguf --out $O/wdiff/q8_vs_base_q8
$GM wdiff   --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --base $M/Qwen3-0.6B-Q8_0.gguf --out $O/wdiff/q8_vs_instruct_q8
```

Done before the stop:

```bash
$GM embedtexts --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/qe0/embedtexts --input $O/qe0/gate20.txt --threads 2
llama-embedding -m $M/Qwen3-Embedding-0.6B-Q8_0.gguf --pooling last -f $O/qe0/gate20.txt --embd-output-format json \
                --embd-normalize 2 -t 2 -c 512 -b 512 -ub 512 -fa off      # and -fa on
$GM embedtexts --model $M/Qwen3-Embedding-0.6B-Q8_0.gguf --out work/qe/timing --input probe/probe.txt --limit 1000 \
               --threads 2 --ctx 512 --batch 256 --chunk 250
llama-bench -m $M/Qwen3.5-9B-Q4_K_M.gguf -t 2 -p 128 -n 32 -fa 0 -r 2
```

Self-test: `python scripts/gguf_mine.py selftest --vocab <vendored llama.cpp>/models/ggml-vocab-qwen2.gguf --out /tmp/st` (and `ggml-vocab-llama-spm.gguf`). It builds tiny random models, not Qwen ones.

Scratch helpers in `/root/work/teacher/bin/`, outside the repository:
- `build_probe.py` and `pilot_subset.py` (the probe set and its subsets);
- `count_tokens.py` (tokenizer-only counts);
- `smoke_9b.py` (the §4.3 test);
- `drank.py` (the numerical rank of ΔW);
- `analyse.py` (bootstrap intervals of `geometry` and `sensitivity` outputs).
