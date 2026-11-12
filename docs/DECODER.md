# CyphaLM's decoder

**Odin Loch** · 2026-11-12 · design, registered before any arm runs · branch `wip/k5-mixdec` (from main 4629f68c) · report and results: [reports/K5_MIXDEC.md](reports/K5_MIXDEC.md) · research: [research/DECODER_RESEARCH.md](research/DECODER_RESEARCH.md) (`wip/decoder-research` 7b5baf8b)

**Odin:** "Lets make a mixer decoder and an architecture like a LLM architecture. Currently we have a learner/encoder and we need a decoder. We have done enough testing to know what experts work well and what does what. We should put all that together in a decoder and generate with it. We should do the reverse of compression."

## 1. The architecture, in LLM terms

| LLM part | CyphaLM part |
|---|---|
| Weights learned in pretraining | The counts: the 1 TB byte trie, byte Kneser–Ney, K5's word Kneser–Ney on 337 MB, the fact store |
| The encoder that reads the context | **The learner**: CyphaLM's online byte stack. It reads the retrieved article, the prompt and the text so far, and adapts its online experts and mixer weights to them |
| The context window | What the learner has read (its copy window is 16 KiB; its n-gram, LSTM and mixer weights carry the rest) |
| Retrieval-augmented context | K1's retrieval (§13.66): the BM25-nearest article of the 337 MB text |
| The decoder's vocabulary and softmax | K5's word distribution: the words K5 proposes, with their probabilities |
| Sampling (temperature, nucleus) | K5's nucleus 0.8 at T 1.0, the repeat ban and the 12-word copy cap, then **arithmetic decoding** |
| A reranker / guidance | The learner's score of each candidate word, and of each gate draft |
| Planning | The planner's fact sentences (§13.37–13.63), untouched |

`context8k` is the encoder at full strength (1.4900 bits a byte, §13.55). It has never written a byte (DECODER_RESEARCH 1b). `sample` used an older byte stack.

## 2. Generation as the reverse of compression

- A compressor codes each symbol with the model's probability. An arithmetic decoder does the reverse. Fed random bytes, it splits its interval in proportion to the model's probabilities, so each symbol comes out with exactly the model's probability.
- So "decompress random bits" is sampling. It adds exactness, not quality (DECODER_RESEARCH item 2, Delétang et al. 2023).
- Truncation is what made every CyphaLM writer work: the byte sampler 0.174 at T 0.6 against 0.073 at T 1.0 (§13.16); the token nucleus 0.119 → 0.192 (T4c); K5's nucleus 0.9 → 0.8 (K5_UPGRADES). So the decoder decodes **a truncated distribution**: K5's nucleus, after the repeat ban and the cap. Temperature and nucleus reshape the distribution before the decode.
- The coder (`lm/decoder/arith.hpp`): a 64-bit range coder, LZMA's scheme widened. Each distribution is quantised to integer frequencies summing to 2^24, so the draw is exact to 2^-24 a word. The selftest encodes the decoded words and decodes the encoder's bytes back to the same words.

## 3. The unit is the word

- Bytes alone wrote word salad (§2–3: 0.147 at best). Words on clean prose made the largest jump (§13.11, §13.19).
- So the decoder decodes words. K5 proposes them. The learner scores each candidate's spelling exactly: log₂ P(bytes of the word, spacing included | article, prompt, text so far), plus the probability that the word ends there (the next byte is not a letter or digit). The end term is the T3 bridge's "cover" (TOKENS_T3_T4): without it "the" also counts "then" and "there" (DECODER_RESEARCH item 6).
- Scoring shares prefixes. The candidates' spellings form a small tree; each node's 256-way distribution is computed once.

## 4. Where the learner enters, and why not as a plain product

The record says, without exception, that a predictor blended into the sampling distribution makes text worse (§13.11; DECODER_RESEARCH 1b):
- linear blends spread the mass the sampler draws from (§13.11: ALL predicts best, writes worst);
- the plain product p_K5^(1−w) × p_bytes^w was already run, with the byte LSTM (§13.24): SCRIBE 0.353 → 0.149 (w 0.25), 0.018 (w 0.5). A product of byte probabilities is a whole word probability, so it counts K5's word prior twice and favours short common words (4.31 → 3.48 letters; distinct words halved);
- the ratio form p_K5 × (p_LSTM(w | context) / p_LSTM(w | " "))^w kept word length but still lost (0.221).

A product of experts sharpens where both agree; a linear blend averages. Neither is safe at full weight. So the learner enters in three registered ways, each one arm:

1. **The context ratio** (classifier-free guidance's form, Sanchez et al. 2023; DExperts): p_K5(w) × (p_mix(w | context) / p_mix(w | neutral))^r, normalised over K5's nucleus, small r. The neutral learner is the base: primed on generic prose, never on the article or the prompt. The ratio keeps only what the context adds; K5 keeps its own word prior.
2. **The plausibility veto** (contrastive decoding's plausibility constraint, Li et al. 2022): drop the nucleus words whose context probability is more than V bits below the best candidate's, then decode K5's own distribution on the rest. It uses the learner where it is strong: spelling, names, broken clauses (DECODER_RESEARCH item 4).
3. **The judge**, in the sentence gate: each of the 16 drafts is scored by (log₂ P_mix(draft | context) − log₂ P_mix(draft | neutral)) per byte, z-scored over the drafts like the gate's other terms, weight 1. This is what the context explains, not what is probable: a generic sentence scores near 0, so the term does not reward blandness. The gate never picks by likelihood (DECODER_RESEARCH item 3; §13.57: lower surprise is a screen, not a target).

Exact draft-and-verify (speculative sampling) is not used: it returns the verifier's own distribution and K5's part vanishes (DECODER_RESEARCH 2.2d).

## 5. Priming is conditioning; then the learner is frozen

- Before writing, each continuation's learner reads K1's retrieved article (its first 8,192 bytes, as K5 holds it: words outside K5's vocabulary are left out) and the prompt, **learning** them: the mixer's weights, the n-gram's counts, the LSTM. This is the compression analogue of a context window, and of the retrieval that moved readers most after the 337 MB counts (§13.66).
- While writing, the learner is **frozen**. It reads its own text, so its contexts move (copy, trie, Kneser–Ney, LSTM state), but it learns nothing from it. Learning from its own output looped every time (GEN-C; the cache, §13.65; DECODER_RESEARCH item 5).
- Rewind is exact. The frozen learner's state is the text read, the LSTM's recurrent state and the n-gram's 64-byte history; a mark and a rewind restore all three (selftest: bit for bit).

## 6. What each measured result contributes

Kept, unchanged, under the decoder:
- K5's counts from 337 MB (§13.48, readers 0.940);
- the 16-draft sentence gate (§13.51, 0.670);
- the LDA topic term in the gate (§13.53, 0.610 / 0.700);
- the honest fill and the planner's facts as hard constraints (§13.63, fact precision 0.935); the decoder never touches a fact sentence;
- K1's retrieval overlay in K5's counts (§13.66, 0.860 / 0.880); the decoder's priming is extra, not a replacement;
- the nucleus 0.8, the repeat ban and the 12-word copy cap: applied before the decoder sees a word.

Left out, because readers did not reward them: the word cache and dynamic counts (§13.65), the paragraph-plan term (§13.62), web prose counts (§13.67), topic stuffing and rescaling (§13.34), the link-to-previous-sentence term (§13.53 R2). Mean surprise is a screen, never a target (§13.57).

## 7. The learner: a lighter configuration, and its cost

The full stack cannot serve a decoder as it stands. Its bitwise mixers give one byte's probability along one path, so a whole 256-way distribution costs 255 node evaluations. Its similar-context expert is the full run's serial bottleneck (680 s, §13.54). None of its parts can rewind a hypothetical byte. DECODER_RESEARCH estimates 2.4 ms a distribution for the serving stack and days for 250 prompts with the gate (item 8).

So the decoder's learner (`lm/decoder/learner.hpp`) is lighter:

| Expert | As in context8k | Here |
|---|---|---|
| 1 TB trie, 5 orders + match, and their mixture | yes | yes (the 992 MB file, shared by every thread) |
| Byte Kneser–Ney 6:4:2:1 | on wiki train | on K5's 7 MB clean prose |
| Copy | yes | over the article, the prompt and the text |
| Online n-gram | orders 1–12, 24-bit tables | orders 1, 2, 3, 4, 6, 18-bit tables; learns only while primed |
| Byte LSTM | hidden 256, from scratch | hidden 64, pretrained on K5's clean prose (`lstm-train`), learns while primed |
| InfiniGram caps, class, similar-context, document cache | yes | no |
| Mixer | online chain, final layer, full mix, meta mix | the full mix's arm B: a geometric mixer, 40 weight sets (copy bucket × byte class), learned online |

- Rule 10: every expert is CyphaLM's own. The byte LSTM is CyphaLM's own online learner, which `context8k` already runs; there is no neural network beyond it at runtime.
- Cost per continuation: one copy of the learner (n-gram tables 6.5 MB, LSTM 1 MB, 32 KiB of text) for the primed state and one for the neutral state; one copy per gate draft. Six continuations run at once on 6 threads (`K5_THREADS=6`). The trie (1 GB) and Kneser–Ney are shared. Measured costs are in K5_MIXDEC.md.

## 8. Arms, screen, judge, predictions

Registered in [reports/K5_MIXDEC.md](reports/K5_MIXDEC.md). In short:
- **Arms** (K1 plus): MJ (the judge), MR (the context ratio, r 0.3), MV (the veto, V 10 bits), MVJ (veto and judge). A check, not an arm: MA (arithmetic decoding alone, no learner term), which should read like K1.
- **Screen:** verbatim13 0; copy32 ≤ 0.30; fact precision ≥ 0.874; SCRIBE v2 ≥ K1 − 0.03; mean surprise reported (a drop of more than 0.5 bits a word below K1 fails: the guard against blandness); wall time; my strict reading. At most 2 candidates.
- **Judge:** blind pairwise raters against K1 at 460 characters, §13.38's rule. The winner, if any, then goes to the 1–10 round of §13.69.
