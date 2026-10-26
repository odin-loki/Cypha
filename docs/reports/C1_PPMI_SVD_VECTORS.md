# C1: PPMI-SVD word vectors and random indexing (baseline for Track C)

Odin Loch, 2026-10-27

Tool: `native/tools/ppmi_svd.cpp` (standalone C++20, no external library).
Plan items: Track C row C1; suite T-words (section 9).

## Summary

Word vectors built from counts alone, on the enwik8 Wikipedia text, score
**WordSim-353 0.672, SimLex-999 0.255, MEN 0.637** (Spearman rho, PPMI-SVD,
300 dimensions, p = 0.5). The whole build takes 152 seconds on one thread and
peaks at 640 MB of RAM.

- WordSim-353 lands inside the published PPMI-SVD range (about 0.65 to 0.70),
  even though our text is roughly 90 times smaller than the billion-word corpora
  behind those figures.
- SimLex-999 is below the published range (about 0.3 to 0.4), as expected for a
  corpus this small.
- Keeping pairs seen once matters: dropping them (as the plan row said) lowers
  WordSim-353 by 0.08 and SimLex-999 by 0.035.
- Random indexing is far weaker (best 0.404 / 0.131 / 0.374).

These are the numbers C2 (predictive vectors), C4 (retrofitting) and C5 (DIF
senses) have to beat.

## Protocol

**Corpus.** `/root/cdata/enwik8` (100 MB of 2006 Wikipedia XML). 12,343 pages;
3,663 `#REDIRECT` pages skipped. Cleaning: HTML entities decoded (two rounds, for
double escaping); `<ref>`, `<math>`, `<gallery>` and similar tags removed with
their content; other tags and comments removed; `{{...}}` templates and
`{| ... |}` tables removed (nested); `[[a|b]]` becomes b, `[[a]]` becomes a,
`[[Category:..]]`, `[[Image:..]]` and interlanguage links dropped with their
captions; `''` and `'''` dropped; URLs dropped. Lowercased. A token is a run of
letters with single internal apostrophes or hyphens; a token touching a digit is
dropped. A newline ends a context unit, so windows never cross a line or article.
Result: **11,226,546 tokens, 265,565 types**, about 67 MB of running text (the
rest of the 100 MB is markup, templates, tables, numbers and redirects). The
"100 MB" corpus is therefore about 11 million words.

**Vocabulary.** The top 50,000 words (the 50,000th has 7 occurrences), used for
both targets and contexts. They cover **96.56%** of tokens (10,839,960).
Out-of-vocabulary tokens are deleted before windowing (word2vec convention).

**Co-occurrence.** Symmetric window of 5 on each side, weight 1/distance. 97.1
million ordered (target, context) updates were packed into uint64 keys per chunk
of 1.5 million tokens, sorted, reduced and merged into one sorted run, then
stored as CSR (uint16 column, float weight, saturating uint8 raw count).
**19,596,850 nonzero ordered pairs** (9,806,607 unordered, since the matrix is
symmetric); 6,665,030 of them have raw count at least 2.

**PPMI.** Context-distribution smoothing alpha = 0.75, negative values set to
zero, marginals taken from the retained pairs. Two settings: min_count 1 (keep
every pair; 15.1 million positive entries) and min_count 2 (drop pairs seen once;
4.9 million positive entries).

**Randomised SVD (written from scratch on the sparse CSR matrix).** Gaussian test
matrix with 310 columns (300 plus oversampling 10), 3 power iterations with
modified Gram-Schmidt (applied twice) between products, small matrix B = Q^T A,
exact eigendecomposition of B B^T by cyclic Jacobi in double, top 300 kept.
Vectors are U * S^p with rows L2-normalised, for p = 0, 0.5 and 1. One seed
(12345).

**Random indexing.** 2,000 dimensions, 10 nonzero index entries (five +1, five
-1) per word, same window and 1/distance weight, one pass, 50,000 x 2,000 floats
(400 MB, fixed). Four context weightings (none, log(N/f), f^-0.5, f^-1; f is the
corpus frequency) and, for each, cosine on unit rows with and without
mean-centring.

**Evaluation.** Spearman rho between cosine similarity and the human scores, on
the copies in github.com/mfaruqui/eval-word-vectors (`data/word-sim/`, fetched
from raw.githubusercontent.com): WordSim-353 (`EN-WS-353-ALL`, combined set),
SimLex-999 and MEN (3k). Pairs with a word outside the vocabulary are excluded.
The Spearman code was checked against `scipy.stats.spearmanr` on the written
vectors and agrees to four decimals.

| Set | Pairs in file | Covered | Excluded (out of vocabulary) | Examples of missing words |
|---|---|---|---|---|
| WordSim-353 | 353 | 350 | 3 | madhouse, tableware, seepage |
| SimLex-999 | 999 | 982 | 17 | frigid, hallway, suds, sweater, crib, orthodontist, drizzle |
| MEN | 3,000 | 2,800 | 200 | stair, twig, daffodil, dandelion, swimsuit, stork, pug, puddle |

The excluded words are everyday objects that an encyclopedia rarely mentions, so
the covered subset leans toward encyclopedic vocabulary. This is a small bias
against comparing with published full-set numbers.

## Results

Spearman rho. Each column's pair count is in the header.

### PPMI-SVD, 300 dimensions

| Setting | WS353 (n=350) | SimLex-999 (n=982) | MEN (n=2800) |
|---|---|---|---|
| min_count 1, p = 0 | 0.681 | 0.262 | 0.633 |
| **min_count 1, p = 0.5** | **0.672** | **0.255** | **0.637** |
| min_count 1, p = 1 | 0.592 | 0.243 | 0.581 |
| min_count 2, p = 0 | 0.594 | 0.216 | 0.543 |
| min_count 2, p = 0.5 | 0.589 | 0.220 | 0.548 |
| min_count 2, p = 1 | 0.524 | 0.221 | 0.501 |

p = 0.5 was the specified setting. min_count 1 became the headline after both
settings were run, so there is a small amount of selection; all rows are shown.
Sampling error on rho is about 0.03 for WS353 and SimLex-999 and 0.01 for MEN,
so differences between p = 0 and p = 0.5 are within noise; p = 1 is clearly worse
on WS353 and MEN, and the min_count gap is larger than noise.

### Random indexing, 2,000 dimensions

| Context weighting | Rows | WS353 | SimLex-999 | MEN |
|---|---|---|---|---|
| none | cosine | 0.156 | 0.089 | 0.153 |
| none | centred | 0.218 | 0.092 | 0.187 |
| log(N/f) | cosine | 0.109 | 0.102 | 0.155 |
| log(N/f) | centred | 0.195 | 0.121 | 0.237 |
| f^-0.5 | cosine | 0.343 | 0.109 | 0.334 |
| **f^-0.5** | **centred** | **0.404** | **0.131** | **0.374** |
| f^-1 | cosine | 0.345 | 0.071 | 0.257 |
| f^-1 | centred | 0.351 | 0.077 | 0.261 |

Plain random indexing is dominated by function words. Down-weighting frequent
contexts and subtracting the mean unit vector helps a lot, but the best variant
is still 0.27 below PPMI-SVD on WS353. The weightings were not tuned on
held-out data; all four are shown. I added the f^-0.5 and f^-1 forms after plain
random indexing did badly on an 8 MB prefix run, where f^-0.5 was also best, and
it is best again on the full run. The best-of-four figure is therefore mildly
selected on the evaluation sets.

### Is the randomised SVD accurate enough?

Singular values fall slowly: sigma_1 = 1744, sigma_100 = 180, sigma_300 = 132
(min_count 1). With 3 power iterations the top triplets are tight and the tail is
loose. The relative residual |A v - s u| / s is 1.5e-6 (1st), 0.0018 (10th), 0.11
(100th), 0.19 (300th). A control run with 6 power iterations (98 s instead of 55
s) cuts those to 0.029 (100th) and 0.099 (300th) and moves rho by at most 0.01
(WS353 0.674, SimLex 0.256, MEN 0.644 at p = 0.5). The 3-iteration baseline is
adequate: the loose tail does not change the conclusions here.

## Nearest neighbours (cosine, top 8, probe words)

PPMI-SVD, min_count 1, p = 0.5:

| Word | Neighbours |
|---|---|
| king | son kings ii iii reign throne iv prince |
| paris | du le les france sur des nantes brussels |
| computer | computers hardware machines computing graphics systems pc ibm |
| music | musical jazz dance pop musicians folk blues instrumental |
| river | rivers lake valley tributaries basin canal navigable tributary |
| war | wars battle during civil conflict occupation fought sino-japanese |
| january | november december march february october april july september |
| red | blue white yellow black green orange purple colored |
| run | runs running ran operating home hit inning batted |
| good | bad that's better excellent you it's doing your |
| france | belgium paris switzerland luxembourg french spain italy netherlands |
| physics | mechanics quantum chemistry mathematics relativity cosmology electrodynamics theoretical |
| dog | dogs cat breed hound breeds cats horse wolves |
| money | funds pay credit cash payments profit profits paying |
| bank | banks fund monetary funds debt investment financial loans |

These are sensible. The weak spots are typical of the method: "paris" pulls in
French function words (enwik8 has French text), "run" shows its baseball sense
(home, hit, inning), "bank" shows only the money sense, and "good" lands on
other evaluative and conversational words.

Random indexing (f^-0.5, centred), same probes:

| Word | Neighbours |
|---|---|
| king | iii iv son the kings his was and |
| paris | france montparnasse the versailles gare cimetière père de |
| computer | computers video hardware fiction a using application the |
| music | songs dance folk musicians style and etymology culture |
| river | valley rivers ganges basin danube tributaries brahmaputra tributary |
| war | liberties during ww diablo post-world second wars world |
| january | november february december october march july april june |
| red | white charing societies blue cruised cross crescent boston |
| run | runs the and hit used which use a |
| good | bad it but or one only a of |
| france | italy spain the in germany england belgium and |
| physics | accelerators mechanics mathematics accelerator gravity impinging antiparticle physicists |
| dog | crunch cur catahoula bonzo doo leopard dah hermas |
| money | laundering conduit drug-related trafficking vulnerable even and a |
| bank | indies inter-american strip banks nfc development afc bundesbank |

Random indexing finds the right neighbourhood for frequent words (months,
rivers, countries) but leaks function words, and its rare-word neighbours are
noise ("dog": crunch, cur, catahoula).

## Cost

Single thread, shared 2-core VM (other agents were working on the same machine),
wall-clock, so timings are noisy.

| Stage | Time |
|---|---|
| Read, clean and tokenise 100 MB | 4 s |
| Co-occurrence counting (97.1M updates) | 9 s |
| PPMI, min_count 1 | under 1 s |
| Randomised SVD, 300 dimensions, 3 power iterations | 55 s |
| (min_count 2 SVD, ablation) | 47 s |
| Random indexing, per variant | 6 to 9 s |
| **Whole run, all settings** | **152 s** |
| Headline vectors only (clean, count, PPMI, SVD) | about 70 s |

Peak RAM **640 MB**, reached while merging count runs; the SVD and random
indexing stages did not exceed it. The sparse store is 9 bytes per distinct pair
(176 MB for 19.6 million), the vectors are 60 MB, and random indexing holds 400
MB, so fixed memory is not an advantage over exact counts at this vocabulary size.

Output (outside the repo): `<prefix>.vectors.f32` (50,000 x 300 float32,
unit-length rows, p = 0.5, min_count 1) and `<prefix>.vocab.tsv` (word and count,
one line per row).

## Comparison with published ranges

| Set | Published PPMI-SVD (billion-word corpora, as quoted in the plan) | This run |
|---|---|---|
| WordSim-353 | about 0.65 to 0.70 | 0.672 (inside) |
| SimLex-999 | about 0.3 to 0.4 | 0.255 (below) |
| MEN | not given in the plan | 0.637 |

The plan's T-words gate is "within published PPMI-SVD ranges". It is met on
WordSim-353 and missed on SimLex-999. A caution before reading too much into the
WordSim-353 result: that set rewards topical relatedness (money, bank, credit),
and an encyclopedia supplies exactly that; SimLex-999 measures strict similarity
(synonyms against merely related words), which needs more text. The shortfall on
SimLex-999 is what a 90-fold smaller corpus would lead us to expect. I did not
run a larger corpus, so I cannot say how much of the gap is size and how much is
our cleaning or tokenisation.

Other limits: one seed for the SVD and one for random indexing (seed variance not
measured); no dynamic window, subsampling or other Levy-style hyperparameter
search, which was out of scope; the evaluation files are the mfaruqui copies, so
tokenisation of multi-word items is as in those files.

## What this means for Track C

1. **Baseline to beat.** Any later word encoder is compared to PPMI-SVD
   min_count 1, p = 0.5: WS353 0.672, SimLex-999 0.255, MEN 0.637 (same pairs,
   same 50,000-word vocabulary).
2. **Do not drop singletons at this scale.** Plan row C1 says "dropping
   singletons". On this corpus that costs 0.08 on WS353, 0.035 on SimLex-999 and
   0.09 on MEN, and saves only memory we do not need (19.6 million pairs fit in
   176 MB). Keep them, or revisit when the corpus is 10 times larger.
3. **Random indexing is not a substitute for the exact counts here.** It is
   online and fixed-size, but 0.27 behind on WS353 and 0.12 behind on SimLex-999,
   and sensitive to the context weighting. Keep it only as the incremental
   fallback for when counts cannot be held.
4. **Room for C4.** SimLex-999 is the weak set, and it is the one retrofitting to
   WordNet synonyms and hypernyms (C4) is designed to lift.
5. **Not done here.** The C1 topic gate (20 Newsgroups at 2k, target 55%) was not
   measured; this report covers word similarity only.

## Next steps

1. Pool these vectors into document vectors (mean, then SIF weighting) and run
   T-topic on 20 Newsgroups at 2k, to test the C1 gate.
2. Rebuild on a larger text (enwik9 or more) to separate the effect of corpus
   size from the effect of our pipeline, especially for SimLex-999.
3. Run C4 retrofitting on these vectors with the same pairs and report the gain.
4. Once C2 exists, compare it on exactly these pair sets; report covered counts
   alongside every number.
5. Optional: seed variance (three SVD seeds), and a window of 2 with subsampling
   as in Levy, Goldberg and Dagan (2015), since a smaller window usually favours
   similarity over relatedness.

## Reproduce

```
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
./ppmi_svd /root/cdata/enwik8 EN-WS-353-ALL.txt EN-SIMLEX-999.txt EN-MEN-TR-3k.txt /path/outside/repo/enwik8
```

Add `--power 6` for the SVD accuracy control, `--min-count 2` to make the other
setting the written one, `--no-ri` to skip random indexing, and `--dump-clean
file` to write the first 300 KB of cleaned text for inspection.
