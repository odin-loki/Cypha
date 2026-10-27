# Copying Qwen3-Embedding-8B's geometry into Cypha

**Odin Loch** · 2026-10-27 · Track C of [PLAN.md](PLAN.md), extends [SEMANTICS.md](SEMANTICS.md)

**In one line.** The goal is for Cypha's vectors to attract and repel the way Qwen3-Embedding-8B's do. We take the shape of its space, not its network. First we take the model apart to learn what that shape is and what made it. Then we rebuild the same forces in Cypha's own statistical machinery and measure how close we get. Qwen3.5-9B is the reasoning teacher that makes the training pairs.

**What stays true.**
- **No neural nets at runtime.** Both models are used offline, as measuring instruments and data sources, like a dictionary or a corpus.
- **Rule 10 still holds.** CyphaLM does every task. The vectors are a support tool that CyphaLM's experts read (SEMANTICS §5).
- **Teacher output is low-trust.** Everything the teachers give enters as low-trust data with provenance (ECS Beta(1, 3)). Cypha's own evidence can overrule it.

---

## 1. Why this model

Qwen3-Embedding-8B was first on the multilingual MTEB leaderboard at release: 70.58 mean over tasks, against 68.37 for Gemini-Embedding. On MTEB English v2 it scores 75.22, and 80.68 on MTEB Code (technical report, Tables 2–3). It is Apache-2.0 and ships as an official GGUF. That makes it the best open map of "which texts mean the same thing" that we can take apart with the tools we already have (`scripts/gguf_mine.py`).

**What we want from it, precisely:**
1. **Its similarity structure.** Which words, phrases, sentences and facts sit near which, how near, and along which directions.
2. **The forces that produced that structure.** What pulls two texts together, what pushes them apart, and how hard each one acts.
3. **Anything else the dissection turns up** that Cypha can use (§8).

---

## 2. The teachers (checked 2026-10-27)

### 2.1 Qwen3-Embedding (the geometry teacher)

| | 0.6B | 4B | **8B** |
|---|---|---|---|
| Layers | 28 | 36 | **36** |
| Output dimensions | 1,024 | 2,560 | **4,096** |
| Matryoshka (MRL) output sizes | yes | yes | **32 to 4,096** |
| Context | 32K | 32K | **32K** |
| Official GGUF | yes | yes | Q4_K_M 4.68 GB · Q5_K_M 5.42 GB · Q6_K 6.21 GB · **Q8_0 8.05 GB** · F16 15.1 GB |

- **Base.** The dense Qwen3 foundation models, with causal attention.
- **Pooling.** An `[EOS]` (`<|endoftext|>`) token is appended. The embedding is the last layer's hidden state at that token: last-token pooling. With llama.cpp, use `--pooling last`.
- **Instructions.**
  - The instruction goes on the query side only: `{Instruction} {Query}<|endoftext|>`. Documents get no instruction.
  - The model card says query instructions are worth about 1–5% on most retrieval tasks, and recommends writing them in English even for multilingual use.
  - The card's code examples use the `Instruct: …\nQuery:…` form. That should be checked against the card before the instruction runs (QE3).
- **Similarity.** Cosine.
- **Licence.** Apache-2.0.
- **Siblings.** Qwen3-Reranker-0.6B, 4B and 8B score a (query, document) pair directly. They are an optional second teacher (§6, B20).

**Training (technical report):**

| Stage | Data | What it does |
|---|---|---|
| 1. Weakly supervised pre-training | About 150M synthetic pairs. Qwen3-32B wrote queries for documents from the multilingual Qwen3 pre-training corpus. A persona was picked per document from Persona Hub (top five retrieved by a retrieval model). The prompts set query type, length, difficulty and language | Broad geometry |
| 2. Supervised fine-tuning | About 7M labelled pairs (MS MARCO, NQ, HotpotQA, NLI, DuReader, T2-Ranking, SimCLUE, MIRACL, MLDR, Mr.TyDi, Multi-CPR, CodeSearchNet), plus about 12M high-quality synthetic pairs kept only if their cosine was above 0.7 | Sharpens relevance |
| 3. Model merging | Spherical linear interpolation (slerp) of several fine-tuning checkpoints | Smooths the final space |

### 2.2 Qwen3.5-9B (the reasoning teacher, for mining)

- **Size and licence.** About 9B parameters; the 10B in the page metadata includes the vision encoder. Apache-2.0.
- **Layers.** 32, with hidden size 4,096. The layout is hybrid: each block has three Gated DeltaNet (linear-attention) layers, then one gated-attention layer. So only 8 layers have ordinary attention. The gated-attention layers have 16 query heads, 4 key/value heads and head dimension 256. The feed-forward width is 12,288.
- **Context.** 262,144 tokens natively, extendable to about 1M with YaRN.
- **Inputs.** Images and video as well as text (we use text only).
- **Thinking.** On by default. Turn it off with `enable_thinking: False` in the chat template; the `/think` and `/nothink` switches are not supported.
- **GGUF.**
  - The Qwen page lists no official GGUF. Community quants exist, for example Q4_K_M 5.63 GB and Q8_0 9.53 GB.
  - Those need a recent llama.cpp with `--jinja`, or the chat template comes out malformed.
  - llama-cpp-python support for this architecture must be confirmed before `gguf_mine.py` relies on it (QE0).
- **Weight mining.** `gguf_mine.py heads` assumes ordinary attention, so on this model it can only read the 8 attention layers.

---

## 3. How Qwen3-Embedding attracts and pushes away

### 3.1 The loss

Write `s(x, y)` for cosine similarity and `τ` for temperature. The report does not give τ's value, so we estimate it in B3. For query `qᵢ` with positive document `dᵢ⁺` and K hard negatives `dᵢ,ₖ⁻`, the per-instance loss is

```
Lᵢ = −log( e^{s(qᵢ,dᵢ⁺)/τ} / Zᵢ )

Zᵢ = e^{s(qᵢ,dᵢ⁺)/τ}                       the positive
   + Σₖ   mᵢₖ e^{s(qᵢ, dᵢ,ₖ⁻)/τ}           K hard negatives
   + Σⱼ≠ᵢ mᵢⱼ e^{s(qᵢ, qⱼ)/τ}              other queries in the batch
   + Σⱼ≠ᵢ mᵢⱼ e^{s(dᵢ⁺, dⱼ)/τ}             document against other documents
   + Σⱼ≠ᵢ mᵢⱼ e^{s(qᵢ, dⱼ)/τ}              query against other documents
```

with the false-negative mask

```
mᵢⱼ = 0  if  sᵢⱼ > s(qᵢ, dᵢ⁺) + 0.1   or   dⱼ is the same document as dᵢ⁺
mᵢⱼ = 1  otherwise
```

The report lists these negative groups. The grouping into five terms written above is our reading of its normaliser.

### 3.2 The forces

Write `p⁺ = e^{s⁺/τ}/Z` for the positive and `pₙ = mₙ e^{sₙ/τ}/Z` for each negative. One gradient step on `q` moves it by

```
Δq = (η/τ) · [ (1 − p⁺) · ∇s(q, d⁺)  −  Σₙ pₙ · ∇s(q, n) ]
```

For unit vectors, `∇s(q, y) = y − (q·y) q`. This is the part of `y` at right angles to `q`.

That one line is the whole of "attract and push away".

| Force | Strength | What it means |
|---|---|---|
| Pull toward the positive | `(1 − p⁺)/τ` | Strong while the positive is not yet the clear winner. Once it is, the pull stops. Learning is error-driven |
| Push from each negative | `pₙ/τ`, a softmax weight | The push is shared out by closeness. The nearest negatives take almost all of it, and far ones take almost none. **The hard negatives do the shaping** |
| Temperature τ | Sets how sharp that sharing is | Small τ means only the very nearest few are pushed, which gives fine local structure. Large τ spreads a gentle push over everything |
| Mask | Zero push | A "negative" that sits closer than the positive plus 0.1 is assumed to be an unlabelled paraphrase, and is left alone. Without this, near-duplicates get torn apart |
| Query–query and document–document terms | Push within each side | Queries spread out among queries and documents among documents. This is the *uniformity* half of contrastive learning (Wang & Isola 2020); alignment of positives is the other half |
| Cosine and right-angle gradients | Rotation only | Every force turns a vector; none rescales it. Length carries no meaning, only direction |
| Instruction on the query side | Moves the query, not the document | The same words land in a task-specific region. Queries and documents live in one space but are not treated the same |
| Matryoshka training | The loss is applied at several prefix lengths | Most of the information is packed into the leading dimensions, so a prefix such as 256 of the 4,096 still works |
| Slerp merging | Averages checkpoints along the sphere | Settles on a smoother, flatter solution than any single checkpoint |

### 3.3 The geometry this should produce

These are the facts the dissection (§6) has to confirm or refute, with numbers.

1. **Alignment.** Paraphrases sit close, but not at cosine 1.
2. **A similarity floor.** Unrelated texts usually still have positive cosine (anisotropy). Raw thresholds mean nothing until the floor is measured.
3. **Relevance, not just topic.** Hard negatives ("same subject, wrong answer") are pushed apart. Same-topic and actually-relevant separate. **This is what Cypha's count-based vectors lack.** PPMI-SVD knows that "Paris" and "France" go together, but not that "capital of France?" is answered by one sentence and not another.
4. **Task regions.** Instructions shift queries.
5. **Nested dimensions.** The leading dimensions carry the most information.
6. **A "what comes next" summary.** With last-token pooling, the vector is the model's state after reading everything, shaped by contrastive training into a summary.

---

## 4. The same forces in Cypha

Core Cypha already pulls an observation toward its true class and pushes it from one competitor (SEMANTICS §2). The upgrade is to make that push **softmax-weighted over several sampled competitors, with a temperature, a mask and uniformity terms, in Cypha's own Fisher geometry.**

### 4.1 Items

An item is anything with meaning: a word, a sense, a phrase, a sentence or a fact. Each item `a` is a diagonal Gaussian `N(μₐ, Σₐ)` in `d` dimensions, with an evidence count `nₐ`. This is the words-as-classes model of SEMANTICS §2, extended past words.

### 4.2 The Cypha pull–push rule

Each training example is an anchor `a`, a positive `b` and a candidate set `C`. `C` is drawn from three sources:
- **(i) Cypha's own confusions:** the nearest current neighbours, plus the classes the DIF classifier currently mixes up with `b`;
- **(ii) random items**, drawn by unigram^0.75;
- **(iii) mined hard negatives** (§6 C2).

The update is:

```
p_c = m_c · e^{s(a,c)/τ} / Z          (over {b} ∪ C)

m_c = 0  if  s(a,c) > s(a,b) + δ      (δ = 0.1, copied)
         or  c is a known synonym, alias or paraphrase of b (lexical pack, ECS ≥ moderate)

μₐ ← μₐ + η · Kₐ · [ (1 − p_b) ∇s(a,b) − Σ_c p_c ∇s(a,c) ]
μ_b, μ_c receive the mirror-image updates
Kₐ = Σₐ / (nₐ + κ₀)                   (natural-gradient step: the posterior covariance of the mean)
```

**What Cypha adds that Qwen does not have.**
- **Confidence controls movement.** `Kₐ` makes a new, rarely seen item move a lot and a well-evidenced item move little. This is a Kalman-like step, and it is why online learning does not wreck what is already known.
- **Rare items stay put.** MDL shrinkage pulls items with little evidence toward the world prior `θ₀` instead of letting them scatter.
- **Every vector carries its uncertainty.** The variance becomes the confidence of a similarity and the input to the don't-know tag (DISCOVERY §3.5). Qwen's vectors have no confidence at all.

### 4.3 Where the pairs come from

| Source | Positives | Stage |
|---|---|---|
| Cypha's own text (no teacher) | Word and context (S-1). Adjacent sentences. Article title and first sentence. Dolly question and answer. Infobox subject and the sentence that states the fact | Cypha's version of Qwen's weak supervision |
| Mined by Qwen3.5-9B (§6 C) | Generated queries for documents, with set type, length, difficulty and persona. Paraphrases. Definitions. Lexical relations | Cypha's version of Qwen's synthetic pairs |
| The geometry teacher (§5) | Its own similarities, used as soft targets | Distillation |

### 4.4 The text encoder (no transformer)

Cypha needs a way to turn a text into a vector. Each stage below is kept only if it raises geometry agreement (§7) enough to pay for its cost, under the TOOL_PROFILE keep criteria.

| Stage | Encoder | What it can capture | Known reference point |
|---|---|---|---|
| E0 | **Static.** A weighted mean of item means, minus the common component (SIF, Arora et al. 2017). The weights are learned from the teacher (B15) | Topic, synonymy | Model2Vec distils a sentence model into static token vectors (PCA to 256 dimensions, Zipf weighting). It reaches MTEB 45.34 against 56.09 for all-MiniLM-L6-v2. Static models trained contrastively with Matryoshka reach about 87% of all-mpnet-base-v2 on NanoBEIR (0.503 vs 0.576) and run about 400× faster on a CPU |
| E1 | **Plus n-grams.** Hashed word bigrams and character n-grams | Morphology, unseen words, short phrases such as "not good" | |
| E2 | **Plus order.** Random-index permutations by offset (Sahlgren, Holst & Kanerva 2008), or the CausalField recurrent state | Word order and role ("dog bites man" against "man bites dog") | |
| E3 | **Plus structure.** The E1 tagger and chunker give heads and slots (subject, verb, object, negation, number), composed by slot | Negation, role swaps, quantities: symbolic composition | |
| E4 | **Plus tasks.** A per-instruction diagonal or low-rank map, fitted with Cypha's regressor | Qwen's instruction-aware queries | Depends on B8 |

**Honest limit.** A shallow, mostly additive encoder cannot reproduce everything a 36-layer model composes. B16 measures that ceiling before E2–E4 are built, so we know what is reachable.

### 4.5 Dimensions and RAM

- **Nested dimensions, as in MRL.** Dimensions are ordered by the teacher's variance (its PCA), so every prefix is meaningful. Cypha's loss is applied at d ∈ {32, 64, 128, 256}.
- **RAM.** The default is 256 dimensions in fp16. For 200k items that is 200k × 256 × 2 B ≈ 100 MB of means, and the same again for variances. A 32-dimension prefix is the coarse first pass of a search.
- **Disk.** Teacher vectors live on disk for training and evaluation only. They are never part of the runtime.

---

## 5. Distilling the geometry: copying the forces directly

The teacher has already decided how strongly each pair should attract or repel. We make Cypha's similarities match those decisions.

### 5.1 Relational softmax distillation (the main loss)

For anchor `a` and candidate set `C`:

```
p_T(c) = softmax_c( s_T(a,c) / τ_T )      teacher
p_S(c) = softmax_c( s_S(a,c) / τ_S )      Cypha

L = KL( p_T ‖ p_S )

Δ on Cypha's similarity to each c  ∝  ( p_T(c) − p_S(c) ) / τ_S
```

- **If the teacher puts `c` closer than Cypha does,** the difference is positive and Cypha pulls them together.
- **If Cypha rates `c` closer than the teacher does,** the difference is negative and Cypha pushes them apart.
- **The forces are copied pair by pair.** The size of each one is exactly the disagreement. This is relational and contrastive distillation (Park et al. 2019; Tian, Krishnan & Isola 2020), run through §4.2's natural-gradient update.
- **Combined loss.** The hard-label loss from §4.2 is added on the mined pairs: `L = α·KL + (1 − α)·InfoNCE`, with α chosen on the dev split.

### 5.2 Other distillation routes, compared

| Route | Teacher data kept | Use |
|---|---|---|
| D-rel (§5.1) | Teacher similarities for each anchor's candidate set | Main route |
| D-rank | The teacher's top-k neighbours and their similarities: a sparse kNN graph | Cheaper to store; a listwise loss |
| D-feat | Teacher vectors, cut to an MRL prefix or reduced by PCA to 256 | Fit a map from Cypha's space to the teacher's: orthogonal Procrustes, then Cypha's regressor with variance. Lets us score Cypha's vectors in the teacher's own space |
| D-static | Teacher vectors for every word and common phrase, each embedded alone | Starting means for Cypha's items (Model2Vec route; SEMANTICS §6). Then a corpus pass fits Cypha's sentence vectors to the teacher's sentence vectors (Tokenlearn-style) |

### 5.3 Trust

- **Priors, not ground truth.** Teacher geometry enters as priors at ECS Beta(1, 3). Cypha's own pull–push from text keeps running and can overrule it.
- **Always compared with from-scratch.** Teacher-initialised and from-scratch Cypha are run side by side (SEMANTICS S-5), so the teacher's contribution is measured, not assumed.

---

## 6. The full dissection

`gguf_mine.py` already has `info`, `vocab`, `embed`, `neurons`, `heads`, `spectra` and `vectors` (GGUF_MINING.md). The new analyses below become subcommands of the same script (§9), not a new tool. Each item lists what it measures and what Cypha does with the answer.

### A. Static: weights only, no inference

| ID | Analysis | Cypha use |
|---|---|---|
| A1 | **Metadata and tokenizer.** Vocabulary of about 152k byte-level BPE tokens (`info` gives the exact count). Merges, scripts covered, special tokens. Whether the GGUF adds `<|endoftext|>` itself (`tokenizer.ggml.add_eos_token`) | Pooling must be right before anything else (QE0) |
| A2 | **Input embedding matrix.** Norm against frequency rank. Anisotropy (mean cosine). PCA spectrum and effective rank. *Rogue dimensions*: a few huge-magnitude dimensions that dominate cosine (Timkey & van Schijndel 2021). Clusters for scripts, code, numbers and subword families. Morphology offsets (plural, past tense) | Starting vectors; which dimensions to whiten or drop |
| A3 | **Output head.** Is there one? Is it tied to the input embedding? | Decides how to read neurons (A5) |
| A4 | **Weight difference from the base model.** `ΔW = W_embedding − W_Qwen3-8B` for every matrix: Frobenius norm by layer, the singular-value spectrum of ΔW, and how much of it the top singular values explain. Try both Qwen3-8B and Qwen3-8B-Base; the one with smaller ‖ΔW‖ is the parent. Add the heavy-tail exponent of each layer's spectrum (Martin & Mahoney 2021) | **The most telling single test.** It shows where and how much contrastive training changed a language model into an embedder. If ΔW is small and low-rank, the knowledge came from the base, and contrastive training only rearranged the geometry. For Cypha, that would mean knowledge (counts, trie, facts) and geometry (vectors) are separable, and the geometry can be learned last and cheaply |
| A5 | **Feed-forward neurons as key–value memories.** Each down-projection column projected onto token embeddings (through the base model's output head if A3 finds none) | Concept groups as hypotheses (DISCOVERY firewall) |
| A6 | **Attention circuits.** QK and OV matrices read through the embeddings (Elhage et al. 2021): which token pairs attend, and which heads copy | Which context views matter. Cypha's multi-view fusion copies the useful ones |
| A7 | **Final RMSNorm gains.** Whether the final norm amplifies the rogue dimensions | Whitening plan |

### B. Dynamic: needs the model running in embedding mode

| ID | Analysis | Cypha use |
|---|---|---|
| B1 | **Similarity floor and anisotropy of outputs.** Cosine distribution of random unrelated sentence pairs; the mean vector; the effect of whitening (Su et al. 2021) | Thresholds relative to the floor |
| B2 | **Calibration by relation class.** Similarity distributions for duplicates, paraphrases (MRPC, PAWS, QQP), entailment, neutral and contradiction (SNLI, MNLI), translations, same-topic-different-fact, and unrelated pairs. Overlap and AUC for each pair of classes | Cypha's calibrated `P(paraphrase \| s)` and `P(relevant \| s)`. The floor below which "nothing stored is relevant" gives `UNKNOWN_ENTITY` or `OUT_OF_SCOPE` |
| B3 | **Effective temperature.** The τ that best predicts the positive among in-batch candidates on held-out pairs, by maximum likelihood | Cypha's τ |
| B4 | **Alignment and uniformity** (Wang & Isola 2020). `L_align = E‖f(x) − f(y)‖²` over positives; `L_unif = log E e^{−2‖f(x) − f(y)‖²}` | Two target numbers Cypha is tracked against |
| B5 | **Hubness.** Skew of k-occurrence: some vectors are everyone's neighbour (Radovanović et al. 2010). Test CSLS (Conneau et al. 2018) and mutual-proximity corrections | Fix for Cypha's nearest-neighbour search |
| B6 | **Intrinsic dimension.** Local (TwoNN, Facco et al. 2017) and global (PCA at 90/95/99% of variance), at each MRL prefix | How many dimensions Cypha really needs |
| B7 | **MRL truncation curves.** Geometry agreement and task scores at d = 32 … 4,096 | Chooses Cypha's d |
| B8 | **Instruction effects.** The same query under N instructions. Is the change an added offset, a projection, or a rotation? Fit a per-instruction affine map and measure the rank of `A − I`. Also with no instruction at all | If it is an offset, Cypha copies it with a tiny per-task table (E4) |
| B9 | **Query/document asymmetry.** The same text embedded as a query and as a document | Whether Cypha needs separate query and document sides |
| B10 | **Sensitivity to minimal edits.** Negation, role swap, a changed number, a swapped entity, shuffled word order, typos, synonym paraphrase, padding, translation. Each gives Δs | Which encoder stage (E0–E3) each effect needs. Where Qwen itself is insensitive, Cypha's vectors need not be. CyphaLM and the symbolic layer must still get those facts right |
| B11 | **Probes.** POS, entity type, sentiment, tense, number, language, topic, and relation facts, read from the output vectors. **Cypha's DIF classifier is the probe**, which also tests the classifier | What is linearly readable, and so what Cypha's regressors can map |
| B12 | **Concept geometry.** Offset consistency (parallelograms); whether hierarchical concepts are orthogonal and categorical ones form simplices (Park et al. 2024) | Compared against Cypha's own way of representing is-a: variance ordering and KL asymmetry (S-2) |
| B13 | **kNN graph.** Over about 100k items; communities compared with WordNet supersenses and Wikidata types | Concept formation (DISCOVERY G9) seeds |
| B14 | **Layer by layer** (optional; needs hidden states, GGUF_MINING §9). CKA between layers; the layer where paraphrase similarity appears | Whether a few "stages" suffice |
| B15 | **Word importance by occlusion.** Drop each word and measure how far the embedding moves; compare with frequency weights | **A learned word-weight table** for E0, distilled from the teacher |
| B16 | **Additivity ceiling.** The best weighted sum of word vectors fitted to each sentence vector (R² and Spearman agreement). The residual is the non-additive part | **How far a static Cypha encoder can get**, measured before E2–E4 are built |
| B17 | **Length and position bias.** Similarity against the length difference; the truncation effect | Normalisation |
| B18 | **Multilingual alignment.** Cosine of translation pairs; a language-identity direction and whether it can be removed | Later multilingual support |
| B19 | **Numbers and dates.** Are they ordered along a line? | Whether numbers need symbolic handling (E3) |
| B20 | **Reranker against embedder** (optional). Where Qwen3-Reranker's P(yes) disagrees with the embedder's cosine | What a two-vector comparison misses. The case for a second check stage |
| B21 | **Near-duplicate robustness.** Adversarial near-copies | Test cases |
| B22 | **Base against embedder, dynamically.** Qwen3-8B's own last-token vectors against the embedder's: anisotropy, hubness, calibration | What contrastive training bought, with the same measurements we use on Cypha |

### C. Mining with Qwen3.5-9B

The reasoning teacher makes the material Qwen3-Embedding was trained on, at Cypha's scale, and judges borderline cases. All of it goes to the hypothesis store under the DISCOVERY firewall, with provenance.

| ID | Mining | Kept when |
|---|---|---|
| C1 | **Synthetic pairs, Qwen style.** For each document (an enwik sentence or paragraph, a Dolly answer), generate queries with a set type (keyword, factual, summary, judgement), length, difficulty and persona | Three checks pass: the embedder's cosine is above 0.7 (Qwen's own filter); Qwen3.5-9B confirms that the document answers the query; Cypha's evidence does not contradict it |
| C2 | **Hard negatives.** Same topic but wrong answer, negated, roles swapped, wrong entity | Embedder similarity is close to the positive's, but Qwen3.5-9B labels it not relevant |
| C3 | **False-negative audit.** Pairs the mask would zero (s > s⁺ + 0.1): are they really paraphrases or relevant? | Gives Cypha's mask rule an estimated precision instead of an assumption |
| C4 | **Lexical and relational material.** Definitions, synonyms, antonyms, hypernyms, meronyms, relation facts. These are `gguf_mine.py lexical`, `taxonomy`, `paraphrase` and `facts`, pointed at Qwen3.5-9B | The usual verification in those commands |
| C5 | **Naming clusters and directions.** Give it a B13 cluster or a B12 direction and ask what links the members, with "unknown" allowed | Agreement over several samples; otherwise unknown |
| C6 | **Minimal-pair suites for B10**, generated at scale | Spot-checked |

**Running it.**
- **Thinking off for bulk generation (C1, C2, C4, C6).** This keeps cost down.
- **Thinking on, with a token cap, for judging (C3, C5).**
- **Qwen3.5-9B is never the judge of Cypha's own output.** SCRIBE and the held-out Q/A suites stay the measures.
- **Bigger teachers later.** A bigger reasoning model can replace it with the same commands.

---

## 7. How "runs like Qwen3-Embedding" is measured

### 7.1 Geometry agreement (the main score)

**Probe set.** It is fixed and hashed:
- 20k words;
- 20k phrases from the E1 chunker;
- 50k sentences from held-out enwik8, Dolly and the STS sets.

It is split into dev and frozen test, and nothing is tuned on test.

| Measure | Definition |
|---|---|
| Pair Spearman | Rank correlation of teacher and Cypha similarities over sampled pairs, stratified into near neighbours, middle and random. Unstratified random pairs would swamp the result |
| kNN@10 overlap | Jaccard overlap of the two neighbour sets, plus the recall of the teacher's top 10 within Cypha's top 50 |
| Linear CKA | Kornblith et al. 2019, between the two sets of vectors |
| Force match | `KL(p_T ‖ p_S)` at the teacher's τ: the exact quantity being distilled |
| Calibration match | AUC by relation class (B2), against the teacher's |
| Confidence check | Spearman of Cypha's per-item variance against its disagreement with the teacher. This is "knows what it doesn't know" for vectors, and the teacher has nothing like it |

### 7.2 Task scores (what the geometry is for)

- **Word similarity:** WordSim-353, SimLex-999 and MEN, which we already have.
- **Sentence and pair tasks:** STS-B, MRPC and PAWS, from GitHub copies.
- **Retrieval:**
  - each T-QA question against its source sentence in the corpus, scored by recall@10;
  - Dolly questions against their answers.
- **kNN classification on TREC.** This is a direct comparison with the E4 result.
- **Hypernym direction** (S-2).
- **A small MTEB subset**, run on Odin's machine, because the `mteb` package downloads its data from Hugging Face, which is blocked here. Published scores for the teacher and for the static models are the reference points.

**Every table reports the same seven methods:**
1. random indexing;
2. PPMI-SVD (C1);
3. words-as-classes (S-1);
4. the cheapest copy, a Model2Vec-style static distillation of Qwen (QE5);
5. Cypha's own pull–push without the teacher (QE6);
6. Cypha distilled (QE7);
7. the teacher itself.

### 7.3 Cost

- RAM, and bytes per item;
- encoding time per sentence on one core;
- size on disk;
- time per online update.

### 7.4 Into CyphaLM

The geometry only counts once CyphaLM uses it: the similar-context and class-context experts (SEMANTICS §5, S-4). The results are reported per the standing rule, all together:
- held-out answers right, abstained and wrong, at a 2% wrong rate;
- the surface score;
- BPC.

---

## 8. What else there is to learn from it

| Finding from Qwen3-Embedding | What Cypha does with it |
|---|---|
| **The hard negatives do the shaping** (softmax-weighted push) | Cypha's competitor sampler aims at its own confusions and at mined near-misses, not at random words |
| **The false-negative mask** (positive + 0.1) | Cypha never pushes a probable paraphrase away. The mask is checked by ECS and by the C3 audit, not just assumed |
| **Synthetic pairs with controls and a consistency filter** (cosine > 0.7) | A data-pack process: generate, cross-check against several sources, admit only what agrees. This is the "picky about what it learns" rule, written down as a process (add to `processes.yaml` as `synthetic_pair_mining`) |
| **Personas and controls give diverse queries** | T-QA and CyphaLM's class-context expert need many phrasings of each question, not one |
| **Instructions as task regions** | Per-task similarity in Cypha: a small offset or reweighting for each task (E4) |
| **Matryoshka nesting** | RAM tiers: search on 32 dimensions, confirm on 256 |
| **Slerp merging of checkpoints** | Cypha's `merge_from` combines runs by precision-weighted Gaussian averaging, which is the principled version of slerp for Gaussians. Test merged runs against single runs |
| **Last-token pooling** | Option: CausalField's final state as Cypha's sentence vector. CyphaLM's own context state is the same kind of object |
| **Anisotropy and rogue dimensions** | Centre and whiten Cypha's vectors. It is cheap and often a free gain |
| **Hubness** | CSLS in Cypha's neighbour search |
| **The effective temperature** | Cypha's τ, and a scale for turning similarity into evidence |
| **The reranker is a calibrated yes/no judge** | Two stages: vector recall first, then a precise check (CyphaLM's own yes/no scoring or a DIF pair classifier). A low "yes" becomes the don't-know tag, not a guess |
| **The size of the weight change from the base** (A4) | Whether knowledge and geometry can be learned separately in Cypha |
| **Additivity** (B16) and **sensitivity** (B10) | Exactly which composition Cypha has to add, and which it can skip |

---

## 9. Logistics

### 9.1 Where it runs

- **On Odin's machine.** Hugging Face is blocked in the Cypha work container, and its 2 cores are far too slow for an 8B model. All teacher passes run on Odin's machine, and only their outputs come back. Those are vectors as fp16 `.npy` files, `.tsv` and `.jsonl` files, and provenance with SHA-256. They come back through the linked folder or a GitHub release.
- **Size the jobs by timing them.** Time 1,000 probe sentences on Odin's hardware, then size the jobs from that. The full probe set is about 3M tokens.
- **Pilot first.** Every script is piloted on Qwen3-Embedding-0.6B (1,024 dimensions, 28 layers) before the 8B run.
- **Comparing 0.6B with 8B is itself a dissection.** It shows what scale adds to the geometry.

### 9.2 Downloads (Odin)

| File | Why |
|---|---|
| `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) | Pilot |
| `Qwen/Qwen3-Embedding-8B-GGUF` (Q8_0, 8.05 GB; F16 for A4 if disk allows) | The geometry teacher |
| Qwen3-8B, base and instruct, GGUF at the same quantisation | A4 weight difference, B22 |
| Qwen3.5-9B GGUF (Q8_0 about 9.5 GB, or Q4_K_M 5.6 GB), plus a recent llama.cpp | The mining teacher |
| Qwen3-Reranker-0.6B (optional) | B20 |

### 9.3 Tools (keep code debt low)

**Teacher side: new subcommands in `scripts/gguf_mine.py`, no new script.**

| Subcommand | Does |
|---|---|
| `embedtexts` | Batched last-token embeddings, with an optional instruction, an MRL prefix and resume. It checks that `<|endoftext|>` was appended and appends it if not |
| `geometry` | B1, B4–B7, B13 and B17 on one set of vectors; exports the kNN graph |
| `compare` | Two sets of vectors over the same items: pair Spearman, kNN overlap, CKA, Procrustes. Used for teacher against Cypha, 0.6B against 8B, and base against embedder |
| `sensitivity` | The B10 minimal-pair suites |
| `occlusion` | B15 word importance and the B16 additivity fit |
| `instruct` | B8 and B9 |
| `wdiff` | A4, the weight difference between two GGUFs |
| `pairs` | C1–C3 with Qwen3.5-9B: controls, filters, audit |

**Fix in `vectors`.** The existing command mean-pools. That is wrong for Qwen3-Embedding, so it gets `--pooling last`.

**Cypha side.**
- One library header, `native/include/cypha/semantics/geometry.hpp`, holds the §4.2 rule, §5.1 distillation, MRL ordering, whitening and CSLS.
- The S-1 tool (`cypha_word_classes.cpp`) grows into the one semantics tool, with `train`, `distill`, `eval` and `export`. There is no third vector tool. If S-1 shows the PPMI-SVD tool is no longer needed, it is cut (TOOL_PROFILE_PLAN).
- Probes (B11) run in C++ with Cypha's DIF classifier on exported vectors, so Cypha does the probing.

---

## 10. Work items, gates and predictions

| ID | Work | Gate |
|---|---|---|
| QE0 | Downloads; `info` and licence check on every GGUF; pooling and EOS check; a smoke run of llama-cpp-python on Qwen3.5-9B | 0.6B `embedtexts` reproduces llama.cpp's `llama-embedding --pooling last` output to a cosine ≥ 0.999 |
| QE1 | Static dissection A1–A7, on 0.6B and then 8B, including the A4 weight difference | Report written |
| QE2 | Probe set built and hashed; teacher vectors at 4,096 and the MRL prefixes | Reproducible to the hash |
| QE3 | Dynamic dissection B1–B22 | Report `QWEN3_EMBEDDING_DISSECTION.md` |
| QE4 | Qwen3.5-9B mining C1–C6 | Pair set with provenance; C3 mask precision estimated |
| QE5 | **The cheapest copy:** D-static into Cypha items, plus E0 with B15 weights | Above PPMI-SVD on STS-B and on retrieval recall@10 (bootstrap interval) |
| QE6 | **Cypha's own forces:** the §4.2 rule on Cypha's own pairs only, no teacher | Reported as the from-scratch line, against QE5 |
| QE7 | **Distilled:** §5.1 relational KL plus InfoNCE on QE4 pairs. Encoder stages E1–E4 added one at a time | Each stage kept only if agreement rises by a registered margin at an LCI of at least 0.95, within the RAM budget |
| QE8 | MRL ordering, whitening, CSLS, variance as confidence | Each kept on its own measured gain |
| QE9 | Into CyphaLM through S-4's similar-context and class-context experts | TOOL_PROFILE gates: held-out answers at a 2% wrong rate, surface score and BPC, reported together |

**Predictions, registered before any run (PLAN rule 3):**

| # | Prediction |
|---|---|
| Q1 | The floor exists. The mean cosine of unrelated sentence pairs is between 0.05 and 0.5 |
| Q2 | The effective temperature (B3) is between 0.01 and 0.1 |
| Q3 | At 256 dimensions the MRL prefix keeps at least 95% of the 4,096-dimension pair Spearman; at 32 dimensions it keeps at least 80% |
| Q4 | The weight change from the base is concentrated: in most matrices, the top 5% of singular values hold at least 50% of ‖ΔW‖²_F |
| Q5 | Additivity: a weighted sum of word vectors gets at least 0.8 pair Spearman with the teacher's sentence vectors (B16) |
| Q6 | The teacher mostly ignores negation: the median cosine of "X is Y" against "X is not Y" is at least 0.8 |
| Q7 | The cheapest copy (QE5) reaches 0.6–0.8 pair Spearman with the teacher on sentences, and STS-B at 0.6–0.75 |
| Q8 | Cypha's own forces alone (QE6) stay below QE5. The teacher's pairs carry information that Cypha's corpus pairs do not |
| Q9 | Cypha's variance predicts disagreement with the teacher, with a Spearman of at least 0.3 |
| Q10 | Instructions act mostly as an offset: one translation per instruction explains at least 70% of the change it causes |

A miss is a result. It is recorded with the number and the reason, and the plan changes accordingly.

---

## 11. Rules for this track

1. **No neural network at runtime.** The teachers run offline only.
2. **Provenance on every file.** It records which model, which file hash, which command and the licence (`gguf_mine.py` already writes this). Both teachers are Apache-2.0.
3. **Teacher output is a hypothesis** (DISCOVERY firewall, ECS Beta(1, 3)) until Cypha's own evidence agrees.
4. **CyphaLM does the task** (rule 10). These vectors never answer anything by themselves.
5. **Held-out discipline.** The dev split is for choices, and the frozen test is read once per change.
6. **No teacher as judge** of Cypha's text. SCRIBE and the Q/A suites stay the measures.

---

## 12. References

- Zhang et al. (2025). Qwen3 Embedding: advancing text embedding and reranking through foundation models. Technical report, arXiv:2506.05176.
- Qwen team (2026). Qwen3.5-9B model card, huggingface.co/Qwen/Qwen3.5-9B.
- van den Oord, Li & Vinyals (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748. (InfoNCE.)
- Wang & Isola (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. *ICML*.
- Kusupati et al. (2022). Matryoshka representation learning. *NeurIPS*.
- Chan et al. (2024). Scaling synthetic data creation with 1,000,000,000 personas. arXiv:2406.20094.
- Moreira et al. (2024). NV-Retriever: improving text embedding models with effective hard-negative mining. arXiv:2407.15831.
- Shoemake (1985). Animating rotation with quaternion curves. *SIGGRAPH*. (Slerp.)
- Wortsman et al. (2022). Model soups. *ICML*.
- Park, Kim, Lu & Cho (2019). Relational knowledge distillation. *CVPR*.
- Tian, Krishnan & Isola (2020). Contrastive representation distillation. *ICLR*.
- Kornblith, Norouzi, Lee & Hinton (2019). Similarity of neural network representations revisited. *ICML*. (CKA.)
- Arora, Liang & Ma (2017). A simple but tough-to-beat baseline for sentence embeddings. *ICLR*. (SIF.)
- Su, Cao, Liu & Ou (2021). Whitening sentence representations for better semantics and faster retrieval. arXiv:2103.15316.
- Ethayarajh (2019). How contextual are contextualized word representations? *EMNLP*. (Anisotropy.)
- Timkey & van Schijndel (2021). All bark and no bite: rogue dimensions in transformer language models obscure representational quality. *EMNLP*.
- Radovanović, Nanopoulos & Ivanović (2010). Hubs in space. *JMLR*.
- Conneau, Lample, Ranzato, Denoyer & Jégou (2018). Word translation without parallel data. *ICLR*. (CSLS.)
- Facco, d'Errico, Rodriguez & Laio (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*. (TwoNN.)
- Sahlgren, Holst & Kanerva (2008). Permutations as a means to encode order in word space. *CogSci*.
- Elhage et al. (2021). A mathematical framework for transformer circuits. Anthropic.
- Park, Choe, Jiang & Veitch (2024). The geometry of categorical and hierarchical concepts in large language models. arXiv:2406.01506.
- Martin & Mahoney (2021). Implicit self-regularization in deep neural networks. *JMLR*.
- Bills et al. (2023). Language models can explain neurons in language models. OpenAI.
- Minish Lab (2024). Model2Vec: distill a small fast model from any Sentence Transformer. huggingface.co/blog/Pringled/model2vec.
- Aarsen (2025). Train 400x faster static embedding models with Sentence Transformers. huggingface.co/blog/static-embeddings.
- Further references (Levy & Goldberg, Vilnis & McCallum, Geva et al., Hernandez et al., Hinton et al.) are listed in [SEMANTICS.md](SEMANTICS.md) §8.
