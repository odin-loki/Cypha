# J2: how real paragraphs move, mined offline (CT3.3, CT2.2)

**Odin Loch** · registered 2026-11-12, before any mining · branch `wip/para-plan` (from main fb79cc93) · GENERATION_PLAN §7 Phase 2, §13.48, §13.51, §13.53

## Why

- The writer's main weakness is drift. Its paragraphs leave the subject after one or two sentences (§13.48, §13.51, §13.53).
- No sampling setting (T2) and no sentence chooser (T3) fixed it. In half the read prompts the first sentence is already off the subject (§13.53).
- A paragraph planner needs to know how real paragraphs move. This job counts that from real Wikipedia text. It changes no writer code.

## Data

- **Mined:** `/mnt/d/cypha_llm/k5scale/clean/train_all.txt`, 329.6 MB of clean enwik9 past enwik8, one document per line. GEN-A and SCRIBE overlap is already removed.
- **Compared:** GEN-A **dev** only (`bench/eval/gen_prompts_dev.tsv`, 250 prompts): the human continuations, and today's writer, L1 (`bench/k5_coherence/runs/L1.gen.txt`). G16 (`bench/tune_write/runs/G16.gen.txt`, the default before L1) is shown beside it. The long reading uses `L1_long.gen.txt` (1,800 bytes).
- GEN-A test is never read.

## The tool

`scripts/para_plan_mine.py` (new). Every source goes through the same functions.

- **Sentences:** cut after `.`, `!` or `?` before a capital, a digit or an opening mark. Not after an abbreviation or an initial. A piece under 24 bytes joins the next (K5's gate rule).
- **The cut:** each continuation is cut near 460 characters at a word, as `blind_rate.py` cuts it for the raters (1,500 for the long reading). A last piece cut mid-sentence is dropped.
- **Wikipedia windows** have GEN-A's shape: in each document, a window starts at sentence 0, 8, 16, …. The prompt is that sentence (and the next if the first is under 100 characters). The continuation is the text after it.
- **Capitalisation:** as K5_STRUCTURE's `caps_`, counted on the first 30 MB. A sentence's first word is a name if it is capitalised more often than not elsewhere.

### Labels (fixed now)

- **open**, how a sentence starts: K5_STRUCTURE's six opening types, ported from `structure/experts.hpp` `open_type`. Connective, pronoun subject, "The/A/An …", date or time phrase, named subject, other.
- **func**, what a sentence does. The first match in this order:
  - DEF: a copula definition ("is a", "was the", "is one of") in the first 14 tokens;
  - DEATH (died, killed, buried …), BIRTH (born, childhood, grew up …), EDUC (studied, graduated …), AWARD (won, prize, medal …), WORK (joined, became, founded, wrote, released …);
  - NAME (named, called, known as, derived …), PLACE (located, population, river, census …);
  - EVENT: holds a year or a month;
  - OTHER: description.
- **subject** of a window: the prompt's main name, a run of name words, the most frequent head in the prompt (ties to the first). With no name, the prompt's most frequent content word (a TERM subject).
- **ref**, how a sentence refers to the subject:
  - NAME: one of the name's words (or the term, singular or plural);
  - DEFNP: the name's head in lower case ("the society"), or "The/This/These/Such" + a content word of the prompt or the sentence before, in the first four tokens;
  - PRON: a third-person pronoun in the first six tokens;
  - NONE.
  - A sentence "keeps the subject" if its ref is not NONE.
- **Content words:** as `coh_measures.py` (lower case, letters only, 4+ letters, not a function word).

These are rough, rule-based labels. They are not a parser. They are the same for every source, so a difference between sources is a difference in the text.

### Smoke check

The script was run on the first 30 documents in `debug` mode to read each sentence's labels (for bugs), and once on 3 MB for time. No count or share was read before this registration. Three fixes came from that reading: non-ASCII letters in tokens, possessives ("Spoiler's"), and too-broad NAME cues ("means", "word", "term").

## Measures

**Mined from Wikipedia** (whole documents), written to `bench/para_plan/tables/` as TSV counts:
- `open2.tsv`: (opening type two back, one back, this one), the layout of K5_STRUCTURE's `Stats::open_`, so the existing expert can read it;
- `func1.tsv`: function → next function, from the document start;
- `state1.tsv`: (function, ref) → next (function, ref) in windows of 1,500 characters. This is the planner's table;
- `lead_func.tsv`, `lead_ref.tsv`: function by sentence index, and ref to the lead sentence's subject by sentence index (0–15);
- `persist.tsv`: whether sentence i and i + d share a content word (d = 1–10), and whether sentence d of a window shares one with the window's prompt (d = 1–12).

**Compared per window** (Wikipedia windows, GEN-A dev human, L1, G16), in `bench/para_plan/compare.json`:
- kept_s1 … kept_s4 (s4 = 4th and later): the share of sentences at that position that keep the subject; name_s*, pron_s*;
- run: the mean number of leading sentences that keep the subject, before the first that does not;
- on_prompt_s1 … s4 and on_prev: as `coh_measures.py`;
- new_share: the mean share of a sentence's content words not seen in the prompt or earlier sentences (the drift rate);
- same_func: the share of sentences with the same function as the one before;
- the opening-type and function distributions, and their Jensen–Shannon divergence (bits) from the human text;
- ntok: mean tokens per sentence.

**The writer against humans:** each L1 − human difference with a 95% paired bootstrap interval over the 250 prompts (1,000 resamples, seed 20261112). **"Where the writer differs most"** is the share measure with the largest absolute difference whose interval excludes 0, with its z (difference over the bootstrap SD) beside it. ntok and run are reported but not ranked (they are not shares).

**A check of the tool:** GEN-A dev human continuations come from enwik8 Wikipedia, so they should look like the mined windows. If they do not, the window shape is wrong, not the writer.

## Predictions

- **The check:** human dev against Wikipedia windows at 460 characters: kept_s1–s3 within ±0.08 each; on_prev within ±0.06; the opening-type JSD under 0.02 bits.
- **Wikipedia, 460:** kept_s1 0.65–0.85, kept_s2 0.55–0.80; pron_s2 0.15–0.30; run 1.5–2.5 sentences.
- **L1, 460:** kept_s1 0.25–0.45; kept_s2 and kept_s3 0.15–0.35; run 0.3–0.8.
- **The largest gap is subject continuity:** a kept_s* measure, by 0.25 or more. on_prev or on_prompt comes next. Opening types differ less: L1 opens with a pronoun less often (human about 0.15–0.25, L1 at most 0.12).
- **Functions:** the function JSD (L1 against human) is above 0.03 bits but below the ref JSD. same_func is lower for L1 than for humans.
- **Persistence (Wikipedia):** sentences i and i + 1 share a content word 0.45–0.60; at d = 5, 0.25–0.40; it falls slowly, not to 0. Window prompt to sentence 8: 0.30–0.45.
- **Leads:** sentence 0 is DEF in at least 45% of documents. BIRTH peaks at sentences 0–2. DEATH peaks later than BIRTH.
- **Long (1,500 characters):** L1_long kept_s4 at most 0.15; Wikipedia's at least 0.45.
- **Cost:** under 15 minutes on 4 processes, under 3 GB RAM.

## Results

(Added after the mining, under this line.)

### The run (2026-11-12)

- The first run stopped on a crash: `year_of` took "10¹²" for digits (Python's `isdigit` accepts superscripts). The fix: ASCII digits only, as `is_digits` in `experts.hpp`. Nothing had been written; no label rule changed. The rerun is the only result.
- `para_plan_mine.py mine`, 4 processes: **89,026 documents, 2,417,861 sentences, 5:55 wall**, 0.15 GB peak in the parent process (each worker holds only its counts). 327,345 windows at 460 characters (88,914 leads, 238,431 inner).
- `compare`: 18 s. Files: `bench/para_plan/tables/*.tsv`, `mined.json`, `wiki_*.scalars.json`, `compare.json`, `caps.json` (the capitalisation table, so `compare` reruns without the 330 MB text).

### The check: human dev against Wikipedia windows (460 characters)

| | Wikipedia | human dev | L1 | G16 |
|---|---:|---:|---:|---:|
| kept_s1 | 0.515 | 0.476 | 0.392 | 0.396 |
| kept_s2 | 0.507 | 0.411 | 0.407 | 0.341 |
| kept_s3 | 0.488 | 0.511 | 0.410 | 0.423 |
| kept_s4+ | 0.473 | 0.426 | 0.339 | 0.351 |
| run (sentences) | 1.01 | 0.92 | 0.78 | 0.78 |
| pron_s2 | 0.186 | 0.138 | 0.262 | 0.209 |
| on_prompt s1 / s2 / s3 / s4+ | 0.652 / 0.576 / 0.516 / 0.443 | 0.644 / 0.492 / 0.517 / 0.351 | 0.372 / 0.294 / 0.247 / 0.130 | 0.388 / 0.317 / 0.272 / 0.118 |
| on_prev | 0.493 | 0.508 | 0.204 | 0.201 |
| new_share | 0.840 | 0.843 | 0.907 | 0.908 |
| same_func | 0.481 | 0.519 | 0.423 | 0.436 |
| tokens a sentence | 23.0 | 23.3 | 16.1 | 15.8 |
| sentences in 460 characters | 2.98 | 3.07 | 5.25 | 5.37 |
| JSD to human: open / func / ref (bits) | 0.002 / 0.005 / 0.002 | – | 0.035 / 0.031 / 0.045 | 0.034 / 0.029 / 0.045 |

- The human dev text looks like the mined windows (JSD 0.002–0.005 bits; on_prev 0.508 against 0.493). So the window shape is right, and the mined tables describe the kind of text the raters compare with.
- The ref labels are strict: even Wikipedia keeps the subject (by these rules) in only about half its sentences. Read kept_s* as a comparison between sources, not as a true rate.

### Where the writer differs most (L1 − human, paired, 95% interval over the 250 prompts)

| measure | human | L1 | L1 − human | z |
|---|---:|---:|---:|---:|
| **on_prev** (shares a content word with the sentence before) | 0.508 | 0.204 | **−0.304 [−0.366, −0.240]** | −9.6 |
| on_prompt s1 | 0.644 | 0.372 | −0.272 [−0.344, −0.192] | −6.7 |
| on_prompt s3 | 0.517 | 0.247 | −0.270 [−0.362, −0.179] | −5.9 |
| on_prompt s4+ | 0.351 | 0.130 | −0.221 [−0.334, −0.102] | −3.7 |
| on_prompt s2 | 0.492 | 0.294 | −0.198 [−0.278, −0.115] | −4.7 |
| opening "other" | 0.371 | 0.210 | −0.161 [−0.207, −0.116] | −6.9 |
| pron_s2 | 0.138 | 0.262 | +0.124 [+0.058, +0.191] | +3.6 |
| opening with a pronoun | 0.177 | 0.289 | +0.112 [+0.078, +0.149] | +5.9 |
| func OTHER | 0.629 | 0.522 | −0.107 [−0.164, −0.055] | −3.8 |
| kept_s3 | 0.511 | 0.410 | −0.101 [−0.190, −0.013] | −2.3 |
| same_func | 0.519 | 0.423 | −0.096 [−0.156, −0.033] | −3.1 |
| new_share | 0.843 | 0.907 | +0.065 [+0.046, +0.082] | +7.0 |
| func DEF | 0.059 | 0.111 | +0.053 [+0.028, +0.080] | +3.9 |
| func BIRTH | 0.003 | 0.043 | +0.041 [+0.031, +0.053] | +7.2 |
| kept_s1, kept_s2, kept_s4+ | | | −0.084, −0.003, −0.087 (intervals touch 0) | |

G16 gives the same picture (`compare.json`). L1's LDA term moved none of these.

**What this says:**
1. **The largest gap is lexical, not the subject's name.** A writer's sentence shares a content word with the sentence before in 20% of cases; a human's in 51%. Each of its sentences is mostly new words (91% new, against 84%). The paragraph has no thread from one sentence to the next.
2. **The writer re-uses the subject's name about as often as humans** (kept_s2 0.407 against 0.411). But it **opens with a pronoun far more often** (0.289 against 0.177), and its pronouns often point at nothing. After a sentence that dropped the subject, the next sentence is a pronoun in 29% of cases (human 12%, Wikipedia 15%). These **orphan pronouns** ("He also wrote …" after a sentence about something else) make the subject look kept when it is not. This step (P(PRON | NONE)) was not a registered measure; it is read from the registered transition counts.
3. **It writes lead sentences in the middle of a paragraph.** Definitions ("X is a …") are 11.1% of its sentences against 5.9%, and birth sentences 4.3% against 0.3%. In Wikipedia DEF is 53% of sentence 0 and about 5% after it; BIRTH peaks at sentence 1 (6.5%) and falls under 1%. The writer keeps starting a new article: a new name, a definition, a birth date.
4. **Its sentences are short:** 16 tokens against 23, so 5.3 sentences fill 460 characters instead of 3.1. Each break is a chance to drift.
5. **Functions do not chain.** Humans repeat the function of the sentence before in 52% of cases (in Wikipedia WORK → WORK 0.19, EVENT → EVENT 0.24, PLACE → PLACE 0.20); the writer in 42%.

### Long text (1,500 characters; Wikipedia windows against L1_long and G16_long)

| | Wikipedia | L1 | G16 |
|---|---:|---:|---:|
| kept_s4+ | 0.446 | 0.319 | 0.332 |
| run | 1.47 | 0.78 | 0.81 |
| on_prompt s4+ | 0.465 | 0.111 | 0.114 |
| on_prev | 0.534 | 0.116 | 0.113 |
| ref JSD to Wikipedia (bits) | – | 0.076 | 0.075 |

The gap grows with length. In long text the writer's link to the sentence before falls to 0.12 (Wikipedia 0.53).

### What Wikipedia does (the tables)

- **Topic words persist** (`persist.tsv`). Sentences i and i + d share a content word: d = 1 **0.551**, 2 0.478, 3 0.437, 5 0.394, 10 0.351. The window's prompt and its sentence d: d = 1 0.652, 4 0.515, 8 0.463, 12 0.438. It falls fast for two sentences, then slowly. A paragraph keeps a set of words for a long time.
- **The subject fades slowly** (`lead_ref.tsv`, the lead sentence's subject). Sentence 1: NAME 0.31, PRON 0.31, DEFNP 0.09, NONE 0.29. Sentence 5: 0.29 / 0.21 / 0.11 / 0.40. Sentence 15: 0.28 / 0.18 / 0.09 / 0.46. The name stays near 0.28 throughout; the pronoun slowly gives way.
- **References chain** (window transitions, P(next | previous)): NAME → NAME 0.40, PRON → PRON 0.34, NONE → NONE 0.62. After NONE the subject comes back by name in 12% and by pronoun in 15%.
- **Functions** (`func1.tsv`, `lead_func.tsv`): a document starts DEF 0.53, EVENT 0.09, BIRTH 0.05. Each function tends to repeat (the diagonal above). DEATH rises slowly, from 1.0% at sentence 0 to 2.7% at sentence 11.
- **Openings** (`open2.tsv`, K5_STRUCTURE's layout): the opening types repeat weakly (pron → pron 0.24, the → the 0.29, name → name 0.19). 38% of Wikipedia's sentences open "other".

### Predictions

- The check (human against Wikipedia): kept_s1 within ±0.08 **hit** (−0.04), kept_s2 **miss** (−0.096), kept_s3 **hit** (+0.02); on_prev within ±0.06 **hit** (+0.015); opening JSD under 0.02 bits **hit** (0.002).
- Wikipedia: kept_s1 0.65–0.85 **miss** (0.515); kept_s2 0.55–0.80 **miss** (0.507); pron_s2 0.15–0.30 **hit** (0.186); run 1.5–2.5 **miss** (1.01). The ref labels are stricter than I expected.
- L1: kept_s1 0.25–0.45 **hit** (0.392); kept_s2 and s3 0.15–0.35 **miss** (0.407, 0.410); run 0.3–0.8 **hit** (0.78).
- The largest gap is a kept_s* measure, by 0.25 or more: **miss**. It is on_prev (−0.30), then on_prompt. Opening types differ less: **hit** in the ranking. But L1 opens with a pronoun *more* often (0.289), not less: **miss** (predicted at most 0.12).
- Function JSD above 0.03 and below the ref JSD: **hit** (0.031 < 0.045). same_func lower for L1: **hit** (0.423 against 0.519).
- Persistence: d = 1 0.45–0.60 **hit** (0.551); d = 5 0.25–0.40 **hit** (0.394); prompt to sentence 8 0.30–0.45 **miss** (0.463).
- Leads: DEF at least 45% of sentence 0 **hit** (0.526); BIRTH peaks at 0–2 **hit** (sentence 1); DEATH peaks later **hit**.
- Long: L1_long kept_s4+ at most 0.15 **miss** (0.319); Wikipedia at least 0.45 **miss** (0.446, just under).
- Cost under 15 minutes and 3 GB: **hit** (5:55).

## The proposed hook (registered 2026-11-12, before any code or run of it)

**What it targets**, from the measures above: orphan pronouns, lead sentences in mid-paragraph, and functions that do not chain. It does not push the lexical link directly. T3's R2 rewarded the link to the previous sentence, and the raters marked it down (0.39, §13.53). Lexical cohesion is read as a check, not pushed.

**The code** (new files only; another job owns `planner/` and `word_kn.hpp` now):
- `native/include/cypha/lm/para_plan.hpp`: a `ParaPlan` class. It loads `bench/para_plan/tables/state1.tsv` (and `open2.tsv`). It ports this report's labels (open, func, subject, ref) to C++ on K5's tokens.
- `term(prompt, sentences kept so far, draft)` returns log₂ P(state(draft) | state(previous sentence)), add-0.5 smoothed, where a state is (func, ref). For the first draft, the previous state is `START:func(prompt's last sentence)`.
- Two penalties in the same term, weighed in the spec: `orph=W` subtracts W when the draft opens with a pronoun and the previous sentence's ref is NONE; `lead=W` subtracts W when a draft after the first sentence is DEF or BIRTH about a name that is not the subject.
- Selftest: the labels on fixed sentences equal `para_plan_mine.py`'s; the table loads; the flag parses and is off by default.

**The flag:** `--k5-para-plan TABLEDIR[,w=W,orph=W,lead=W]`, **off by default**. With it off, nothing is loaded or computed.

**The hook point** (two lines at each site, made by the owner of those files once they are free): the term joins the z-scored sentence gate as one more weighed term, in `word_kn.hpp`'s gate (prompts with no plan) and in `discourse.hpp`'s free sentences. With the flag off, `cmp` on all 250 dev prompts must give L1's GEN file, MODEL line aside.

**Arms** (dev, seed 0, 250 prompts; each is L1 plus): PP1 `w=1`; PP2 `w=2`; PP3 `w=1,orph=2,lead=2`.

**Screen:** verbatim13 0; copy32 at most 0.30; SCRIBE v2 at least L1's minus 0.03; by `para_plan_mine.py compare`, the orphan rate (NONE → PRON) and the DEF + BIRTH share both fall; by `coh_measures.py`, stuffing (3rd+ use) up at most 0.01; my reading of T3's 12 dev prompts beside L1. At most two candidates, chosen by reading, then by the orphan rate.

**The deciding judge** (as T3): `scripts/blind_rate.py make --mode pair --baseline L1 --ids bench/blind/llm_rating/prompts.json --seed 20261117` (50 prompts, four fresh blind Claude raters), and the long reading at `--cut 1500` on `bench/blind/q3_long/prompts.json` (seed 20261118). Packets, `key.b64` and `RATER_PROMPT.md` are committed before the raters run. The §13.38 rule: win share minus 0.5 above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign. Adopted if better by the rule on either reading and at least 0.5 on the other. Then GEN-A test is read once, for it only. Odin can veto.

**Predictions for the hook:**
- off: `cmp` equal to L1: hit expected;
- PP3: NONE → PRON from 0.29 to at most 0.20; DEF + BIRTH share from 0.154 to at most 0.11; on_prev up at most 0.03; SCRIBE v2 within ±0.03 of L1;
- PP1 and PP2 move these by half as much as PP3;
- raters: PP3 about 0.55–0.60 against L1 at 460 characters, about 0.60 on the long reading. It does not fix drift by itself. It removes the clearest false signs of a kept subject.
- What it cannot fix: the missing lexical thread (on_prev −0.30). That needs content to write about (the fact plan, CT3.1) or a draft source that carries words from sentence to sentence, not a chooser.
