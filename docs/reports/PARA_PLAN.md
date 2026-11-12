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
