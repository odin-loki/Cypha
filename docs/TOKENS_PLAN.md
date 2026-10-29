# CyphaLM over tokens: plan (Track T)

**Odin Loch** · 2026-10-29 · **started 2026-10-29** (Odin: "You pick for the questions"; PLAN ledger 74) · from the discussion of 2026-10-29 ("Modern NLP uses tokens, which naturally map to words. Ours doesn't.")

This plan moves CyphaLM from bytes to tokens, and then above tokens. Nothing here is built yet. Each step is a registered experiment: rules and predictions first, dev for choices, the frozen test read once, intervals, and Odin's veto.

---

## 1. Why tokens matter more for CyphaLM than for a neural model

**Measured on CyphaLM's 8 MiB training slice (2026-10-29):**

| Read | Value |
|---|---|
| Bytes per word (raw wiki XML) | 7.06 |
| Words | 1,187,592 |
| Distinct words | 66,939 (32,092 seen once) |
| Share of word occurrences covered by the 16,000 most common words | 0.930 |

- **Reach.** CyphaLM predicts the next byte from the 4 bytes before it. That is about 0.6 of a word. The long-match expert stops helping at about 16 bytes (PLAN §3.2), about 2 words. A transformer sees thousands of tokens, so for it tokens are mostly about speed. A count model only knows what its context covers.
- **Invented words.** Every byte is a separate random choice. So CyphaLM builds words letter by letter ("Bahnaric acid", "abbrevision").
- **More chances to drift.** One word is 5–7 sampling decisions instead of 1–2.
- **Backing off mid-word.** When a context is unseen, a byte model drops one letter. A token model drops one word, which is where language actually breaks.
- **The experts fit tokens.** A word-level expert added as a byte expert fell to weight 0.0005 (CURRENT). The sentence gate and the content-plan expert (GEN track) speak in words and facts, not letters.

**What tokens will not do alone.** Word n-gram text is locally grammatical but still wanders: nothing plans past a few words. The honest expectation for tokens alone is toward SCRIBE's "sentences shuffled across docs" rung (0.344), not human text (0.923). Today's default is 0.063. Paragraph coherence needs the GEN track on top (gate, topic, content plan). Tokens make those steps work better.

---

## 2. The rules this plan keeps

- **Rule 10.** CyphaLM does every task. A tokenizer and a token-level count model are CyphaLM itself. Tools may only add training text, add next-byte experts, or gate.
- **No neural network at runtime.** Every tokenizer here is learned by counting (pair merges, likelihood over counts, entropy from counts, MDL).
- **Byte fallback.** All 256 bytes stay in every vocabulary. Any string can still be written and read.
- **Comparable scores.** Token models are scored in bits per byte: total bits divided by bytes. They line up with every BPC number in CURRENT.
- **Metrics together.** Answers, SCRIBE v2 with surface S, and bits per byte, every time.
- **Track A is untouched.** The 1 TB byte run continues. A token run is separate (T11).
- **Add before replace.** The token level is built beside the byte model. Defaults change only when a registered rule says so.
- **Own files first.** It is built as its own library and tool (`cyphalm_tok`), then wired into `cyphalm_trie` when it wins. That keeps it clear of the GEN agents' work.

---

## 3. What the field has learned (research, 2026-10-29)

| Method | Finding | What CyphaLM takes |
|---|---|---|
| **BPE** (merge the most frequent pair, repeat) | The standard. Deterministic, fast, learned by counting | Baseline tokenizer |
| **Unigram LM** (Kudo 2018) | A likelihood model over segmentations, pruned by counts. Beats BPE on morphology-rich languages; tokenizer family mattered more than any intrinsic score ([2508.08424](https://arxiv.org/pdf/2508.08424)) | A second arm in T1 |
| **"Tokenization is more than compression"** (Schmidt et al., EMNLP 2024) | Fewest tokens is not best. 64 models: pre-tokenization (word boundaries) mattered most; starting from BPE helps ([2402.18376](https://arxiv.org/html/2402.18376v2)) | Keep word boundaries in stage 1; do not pick by compression |
| **Intrinsic metrics** (Rényi efficiency and others) | Two counterexamples break Rényi efficiency ([2402.14614](https://arxiv.org/pdf/2402.14614)); a 2026 study of about 200 models finds no single tokenizer feature predicts downstream results ([EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.96/)) | Choose only by end-to-end reads: SCRIBE, bits per byte, answers |
| **SuperBPE** (2025) | Two stages: subwords first, then merges across spaces ("by the way"). Up to 33% fewer tokens, +4.0% over 30 tasks at a fixed vocabulary ([2503.13423](https://arxiv.org/abs/2503.13423v2)) | Superwords: more reach per context order (T5) |
| **BoundlessBPE** (COLM 2025) | Merges whole pretokens into superwords (" of" + " the"); about 20% more bytes per token, a more even token distribution ([2504.00178](https://arxiv.org/pdf/2504.00178)); a 2026 follow-up trains on 1 GB in about 600 s ([2604.05192](https://huggingface.co/papers/2604.05192)) | A second superword arm (T5) |
| **Byte Latent Transformer** (2024) | Patch boundaries where next-byte entropy is high; matches token models at scale ([2412.09871](https://huggingface.co/papers/2412.09871)) | CyphaLM already computes next-byte entropy from counts: entropy tokens are nearly free (T1, §6) |
| **H-Net** (2025) | Learned chunking lands near BPE's 4.5–5 bytes per chunk and beats a BPE transformer at matched compute ([2507.07955](https://arxiv.org/html/2507.07955v1)) | Boundaries can be learned from the data, not fixed |
| **SpaceByte** (NeurIPS 2024) | Breaking after spaces roughly matches subwords; the first letter of a word is the hardest byte ([2404.14408](https://arxiv.org/abs/2404.14408)) | A simple arm: word-start boundaries |
| **Branching entropy / Harris** (Jin & Tanaka-Ishii 2006; Zhikov et al. 2010 with MDL) | Word boundaries where the entropy of what follows stops falling; unsupervised, from counts ([P06-2056](https://aclanthology.org/P06-2056.bib), [D10-1081](https://preview.aclanthology.org/setup/D10-1081)) | The count-native ancestor of entropy patching (T1) |
| **Nested Pitman–Yor LM** (Mochihashi et al., ACL 2009) | Word segmentation and a word model learned together, with a spelling model inside ([P09-1012](https://aclanthology.org/P09-1012/)) | The Bayesian form of PLAN B3 (word model + spelling model); a stretch arm |
| **Sequence Memoizer** (Wood et al., ICML 2009) | Unbounded-context hierarchical Pitman–Yor smoothing; beats PPM as a compressor ([DCC 2010](https://www.cs.ubc.ca/~fwood/assets/pdf/Gasthaus-DCC-2010.pdf)) | Smoothing for an unbounded token context (T2) |
| **Multigrams** (Deligne & Bimbot 1995) | Text as variable-length units chosen by EM on counts; phrase models near trigram perplexity with fewer entries ([ATR TR-IT-0270](https://results.atr.jp/35th/techreport/TR-IT-0270/index.html)) | Phrase units from counts (T7) |
| **Infini-gram** (2024) | An ∞-gram over tokens with a suffix array, up to 5 trillion tokens; 47% next-token accuracy ([2401.17377](https://arxiv.org/html/2401.17377v4)) | CyphaLM's FM-index over token IDs instead of bytes (T2) |
| **Exact byte-level probabilities** (Phan et al., ICLR 2025) | Every token model has an equivalent byte process; "tokenization bias" when a prompt ends mid-token; fixed at O(1) model runs ([2410.09303](https://arxiv.org/abs/2410.09303v2)) | The bridge: token experts mix with byte experts and score in bits per byte (T3) |
| **BPE-dropout / subword regularization** (2020) | Train over several segmentations; up to +3 BLEU ([1910.13267](https://arxiv.org/pdf/1910.13267)) | Count over several segmentations to smooth sparse counts (T8) |
| **Copy Is All You Need** (ICLR 2023); **Generation with Dynamic Vocabulary** (EMNLP 2024) | Generate by whole phrases taken from a text collection; MAUVE +25%, 20% faster ([2307.06962](https://arxiv.org/pdf/2307.06962), [2410.08481](https://arxiv.org/abs/2410.08481v1)) | Phrase-level continuation drafts from the ∞-gram, under SCRIBE's copy limit (T7) |
| **Class-based n-grams** (Brown et al. 1992; Kneser–Ney classes) | Classes cut sparsity; up to 25% lower perplexity reported ([J92-4003](https://wiki.eecs.yorku.ca/course_archive/2014-15/W/6339/_media/j92-4003.pdf)) | Classes over tokens (T6) |
| **Strong n-gram baselines** (Shareghi et al., NAACL 2019) | Modified Kneser–Ney beat an LSTM on 42 of 50 languages at small data ([N19-1417](https://aclanthology.org/N19-1417/)) | Word-level counts are competitive at our data size |
| **N-gram drafting** (speculative decoding) | n-gram lookups propose several tokens, a model accepts them ([vLLM](https://docs.vllm.ai/projects/ascend/en/v0.22.1rc/user_guide/feature_guide/speculative_decoding.html)) | Draft phrases, accept by the mixture (T7) |

---

## 4. The design: levels above the byte

This is PLAN §3.2's context hierarchy, built.

| Level | Unit | Count model | Status |
|---|---|---|---|
| 0 | byte | packed trie, FM-index ∞-gram, class, copy, similar-context | Have |
| 1 | token (subword, with word boundaries) | packed trie over token IDs, FM-index over token IDs, modified Kneser–Ney | T1–T4 |
| 2 | superword (multi-word unit) | the same, over superword IDs | T5 |
| 3 | class and topic over tokens | class backoff, topic cache | T6 |
| 4 | phrase draft | ∞-gram continuations of several tokens | T7 |
| 5 | content plan | facts and patterns (GEN-E) | GEN track |

**How the levels meet.** Each token-level model gives P(next token | tokens so far). The bridge (T3) turns that into exact next-byte probabilities. So token experts sit in the same mixture as the byte experts, and everything is scored in bits per byte. Generation samples whole tokens. The byte model still spells anything the vocabulary lacks.

---

## 5. Work items

| # | Item | Measure | Gate (registered before the run) |
|---|---|---|---|
| T0 | **Done 2026-10-29** ([report](reports/TOKENS_T0_T2.md)). **Harness.** GEN-A's frozen generation benchmark, plus bits per byte on clean held-out text, plus tokenizer statistics (bytes per token, share of word starts at token starts) reported only | — | Baseline: today's byte model |
| T1 | **Done 2026-10-29** (five arms; Unigram and BPE at 4k best). **Tokenizers learned by counting.** BPE with word-boundary pre-tokenization; Unigram LM; word-start boundaries (SpaceByte); entropy boundaries from CyphaLM's own trie (BLT); branching entropy with MDL (Harris). Vocabularies of 4k, 8k, 16k and 32k. Byte fallback, deterministic | Statistics only | None: choices come from T2 and T4 |
| T2 | **Done 2026-10-29** (dev 1.904 against the byte model's 2.073; byte-level Kneser–Ney matches). **Token count models.** Packed trie over token IDs (orders 1–5), modified Kneser–Ney; FM-index ∞-gram over token IDs; the same mixture learner | Bits per byte on dev, per tokenizer and size | Report; the best two go to T4 |
| T3 | **Done 2026-10-29** ([report](reports/TOKENS_T3_T4.md); exact to 5e-16; real-text gap 0.0017). **Byte–token bridge.** Exact byte-level probabilities (Phan et al.); prompt boundaries that end mid-token; token experts inside the byte mixture | Bits per byte equal both ways on a planted Markov chain | Exact to 1e-9 on the planted chain |
| T4 | **Done 2026-10-29, not adopted** ([report](reports/TOKENS_T3_T4.md), "T4"): test SCRIBE 0.112 against 0.106, each model at its best dev temperature; the intervals overlap. The token model is 0.10 bits per byte better through the bridge. Low temperatures copy. T4b 2026-10-29: the cap and the byte sampler's rules settled on dev; not adopted (test 0.115 against 0.106). The nucleus lifts SCRIBE but copies. T4c 2026-10-29: nucleus 0.9 + repeat ban over Kneser–Ney alone meets T4's gate on test against T4's byte default (0.187 against 0.106, disjoint; copy 0.065); not yet compared with BKN2's byte default; no default changed. **Generation at the token level.** Sample tokens; SCRIBE v2, surface S, copy share, invented-word rate | GEN-A's benchmark, test read once | SCRIBE rises over the byte model with disjoint intervals; copy share +0.02 at most |
| T5 | **Superwords.** A second stage of merges across spaces (SuperBPE, BoundlessBPE) | As T2 and T4 | SCRIBE rises over plain tokens with disjoint intervals |
| T6 | **Classes and topics over tokens.** GEN-C's cache and triggers (built at the byte level, not adopted there: topic up, sentence form down) carry over here, where a cached word is written whole. Brown classes over tokens with class backoff; the class and similar-context experts at the token level; a document/topic cache (PLAN B5) | Bits per byte, SCRIBE | LCI ≥ 0.95 |
| T7 | **Phrase drafts.** The token ∞-gram proposes several tokens at once; the mixture accepts or rejects each one (speculative style) | SCRIBE, copy share, speed | SCRIBE up, copy share +0.02 at most, speed not worse |
| T8 | **Counting over several segmentations** (BPE-dropout, Unigram sampling) to smooth sparse counts | Bits per byte | Lower with a bootstrap interval below 0 |
| T9 | **The GEN steps at the token level.** The sentence gate (GEN-B) and the content-plan expert (GEN-E) over tokens; entity names as single tokens linked to the fact store | SCRIBE, factual precision | As GEN-B and GEN-E |
| T10 | **Answers.** `qa` with token experts in the mixture; the don't-know message as real tokens | Frozen Q/A test, read once | Right answers not fewer at 2% wrong; don't-know precision not lower |
| T11 | **Scale, on Odin's PC.** Token models built while the 1 TB corpus streams, beside Track A, with predictions registered (bits per byte, distinct token n-grams, RAM) | Checkpoints | Registered before the run |
| T12 | **Decide.** The token level becomes CyphaLM's default unit for generation only if T4 passes, and for answers only if T10 passes. The byte model stays as the speller and as the compressor | — | Odin's veto |

---

## 6. Further ideas (each one a measured experiment, after T4)

- **CyphaLM tokenizes itself.** Put boundaries where CyphaLM's own next-byte entropy jumps. Build the token model. Measure its entropy and re-cut. Each round is a self-made change, so it passes the LCI referee (PLAN §3.4). Tokens become "what CyphaLM finds predictable as one unit".
- **The unit chosen per position.** Experts at several unit sizes (byte, token, superword, phrase) vote at each step. The decoder writes a whole phrase where it is confident and letter by letter where it is not.
- **Entity tokens.** Every entity in the fact store gets one token linked to its entity ID ("Agamemnon"). The content-plan expert and the store then speak the model's own units.
- **Concept tokens.** Tokens mapped to senses or concept IDs from the data pack. A count model over concepts is the statistics-to-semantics bridge (PLAN B4) at the token level.
- **Segmentation as a Core Cypha decision.** At ambiguous points, Cypha's classifier picks the segmentation, learning from reward (rule 10: gating).
- **Hierarchy all the way up.** Bytes, tokens, superwords, clause chunks, sentence plans: each level a count model over the IDs of the level below, with backoff down the hierarchy.

---

## 7. Predictions (registered here, scored when the runs happen)

| # | Prediction |
|---|---|
| P-T1 | On the 8 MiB slice, the best token model's bits per byte is within ±0.05 of the byte model's (counts are sparse at this size); at 64 MiB and above, tokens are lower |
| P-T2 | Token-level samples score SCRIBE v2 ≥ 0.15 (byte default 0.063), and the invented-word rate falls below 1% |
| P-T3 | Superwords add at least +0.02 SCRIBE over plain tokens |
| P-T4 | Entropy boundaries from CyphaLM's own trie come within 0.02 bits per byte of BPE at the same vocabulary size |
| P-T5 | Unigram LM and BPE with word boundaries are the top two in T2; fewest-tokens choices are not |
| P-T6 | Answers on the frozen Q/A test are unchanged or better with token experts in the mixture |
| P-T7 | A 16k token trie on 8 MiB stays under 50 MB resident |

---

## 8. Order and dependencies

1. **Built as its own library and tool** (`native/include/cypha/cyphalm/tokenizer.hpp`, `token_model.hpp`, tool `cyphalm_tok`), clear of the GEN agents' work in `cyphalm_trie.cpp`.
2. T0 → T1 → T2 → T3 → T4 is the spine. T5–T8 follow T4. T9 needs the GEN track. T10 needs T3. T11 is on Odin's PC.
3. Agents can split it: one for T1 (tokenizers), one for T2 and T3 (models and bridge), then T4 onward.

---

## 9. Decisions (2026-10-29, Odin: "You pick for the questions"; PLAN ledger 74)

1. **Subwords with word boundaries, then superwords.** Whole words with a byte fallback stay as one arm of T1.
2. **Add beside the byte model.** Defaults switch only on T4 (generation) and T10 (answers).
3. **Start now, as its own library and tool** (`cyphalm_tok`), clear of the GEN agents' files. Two agents: TOK-A (T0–T2: tokenizers and token count models) and TOK-B (T3: the byte–token bridge, then T4 when TOK-A and GEN-A's harness are ready).
4. **T11 on Odin's PC** after T4 passes on 8 MiB, run when the 1 TB run has finished or has spare capacity. It never competes with Track A.
