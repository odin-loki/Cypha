# C1: a decoding controller that steers toward human text

**Odin Loch** · registered 2026-11-13, before any arm runs · branch `wip/k5-control` (from `wip/mixdec-t8` 42206a76) · GENERATION_PLAN §13.76; §13.57, §13.59, §13.62, §13.65, §13.66, §13.71, §13.74

(The properties, the targets, the knobs, the control law, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Odin chose this job from four: learning while it writes, guided by a target taken from real human text.
- Three earlier guides failed, so none is used here:
  - the model's own output (GEN-C, the word cache §13.65): it loops;
  - lower surprise (WRITING_SIGNAL, §13.57): human text is far more surprising, so lower surprise points away from it;
  - real against draft (§13.31–13.36): the discriminator picked odd drafts.
- The idea: while K5 writes, measure a few properties of the text so far. Steer them into the range human continuations show, by moving decoding knobs sentence by sentence. This is a feedback loop, as Mirostat holds surprise at a target, but on properties of whole sentences.
- The gap is largest in long text. Today's writer links a sentence to the one before 0.31 of the time at 460 characters, and 0.22 at 1,500. Human text: 0.51 at both (§13.59 and the table below).

## The base and the baseline

- **Base:** D5's branch `wip/mixdec-t8` (42206a76): main deda93c3 (T8) with D4's decoder (`--mixdec`) merged, off by default. Its default equals T8 (MIXDEC_T8's checks). The branch appeared at 02:09, after I had waited about two hours, as the job says.
- **Baseline T8**, the base's default: `scripts/gen_eval.sh dev T8 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.
- **The long baseline T8_long:** the same with `--bytes 1800 --k5-max-words 600` (K5_COHERENCE's long setup), `cyphalm_trie sample` only, my build with the flag off. No T8_long file existed, so it was run before this was written: 21:20 wall, 11.3 GB, copy32 0.483, and **verbatim13 1 of 250**. Today's default already breaks the 12-word cap once in 250 long texts. So the long screen asks for no more 13-word copies than T8_long has, not for 0.
- **Checks, done before this was written,** on a build of main deda93c3 with the same hooks (made while waiting for the base): my build with `--k5-control` off gives a GEN file equal to `bench/k5_decode/runs/T8.gen.txt` under `cmp`, MODEL line aside (12:35 wall, 11.3 GB, copy32 0.487, verbatim13 0). The selftest passes. Both are repeated on the base branch's build before the arms: the short default against `T8.gen.txt`, and the long default against the T8_long above, each under `cmp`.

## The properties, and their human targets

Measured by `bench/k5_control/ctl_measures.py` (new) on the 250 GEN-A **dev** human continuations, never on test. Each text is cut as the raters read it (`blind_rate.py`'s cut near 460 or 1,500 characters; a last piece cut mid-sentence dropped), with `para_plan_mine.py`'s sentence split:
- **link:** the share of sentences 2+ that share a content word with the sentence before (`para_plan_mine.content`: 4+ letters, not on its stop list). This is `hook_screen.py`'s `on_prev`.
- **words:** words a sentence (tokens with a letter or digit), pooled over sentences.
- **short:** the share of sentences under 8 words. A proxy for verbless fragments, reported, not steered.

| Source (dev, 250 prompts) | words | link | link of sentence 1 to the prompt | short | sentences | words per text (quartiles) |
|---|---:|---:|---:|---:|---:|---|
| **human, cut 460** | **20.4** | **0.508** | 0.580 | 0.046 | 3.1 | 17.7, 21.0, 25.7 |
| human, cut 1,500 (the whole continuation, about 4 sentences) | 21.2 | 0.513 | 0.580 | 0.039 | 4.0 | 18.8, 21.6, 25.8 |
| T8 (today), cut 460 | 13.6 | 0.310 | 0.482 | 0.204 | 5.3 | 11.6, 13.3, 16.6 |
| T8_long, cut 1,500 | 14.2 | 0.222 | 0.480 | 0.215 | 18.3 | 12.6, 14.2, 16.2 |

**The registered targets:** link **0.508**, words **20.4** (the human values at 460 characters). The human values at 1,500 are the same within 0.01 and 0.8 words, so one target serves both lengths.

### Why these two

- **link** is the largest gap J2 found (§13.59: −0.30, the 9.6 SD gap), and the one that grows with length: 0.11–0.16 against 0.51 in the long runs below. That growth is the drift readers see.
- **words** is the second gap (20.4 against 13.6, and 0.20 of T8's sentences are under 8 words against 0.05). No setting tried so far moves it: every dev file below lies between 12.8 and 13.8 words. So it needs its own lever.

### Properties not chosen

- **Surprise, mean and spread.** Human text reads 8.92 bits a word under WRITING_SIGNAL's model, the writers 6.8–7.8. Steering toward it means sampling warmer. Readers chose the colder sampler (T8 against K1 0.855, §13.71), and between variants of one writer lower surprise agrees with the raters 0.738. There is also a reason it cannot work: a sample's expected surprise under its own model is that model's entropy; human text's is the cross-entropy, higher by however wrong the model is. Better sampling cannot close that gap. So no surprise target, and no Mirostat on surprise. Mean surprise stays a guard against bland text (below).
- **Verbless fragments.** No tagger is on the PC (K5_DECODE: the E1 tagger's model is not here). The short-sentence share stands in for it, and the words loop acts on it.
- **Keyword repetition** (stuff3). Read as a guard in the screen, not steered: pushed on, it is the §13.34 failure.

## The knobs, and how they move the properties

From dev GEN files already committed (cut 460, `ctl_measures.py`; no new run):

| Arm | What differs | words | link | short |
|---|---|---:|---:|---:|
| HLT | no retrieval, T 1.0 | 13.52 | 0.200 | 0.234 |
| K1 | retrieval k 1, rel 0.5, T 1.0 | 13.21 | 0.232 | 0.252 |
| T9 | K1 at T 0.9 | 13.75 | 0.299 | 0.222 |
| **T8** | K1 at T 0.8 (today) | 13.55 | 0.310 | 0.204 |
| K4L | k 4, rel 0.2, T 1.0 | 13.53 | 0.213 | 0.214 |
| K4 | k 4, rel 0.5 | 13.21 | 0.252 | 0.235 |
| K4H | k 4, rel 1.5 | 13.34 | 0.324 | 0.203 |
| F1, F2 | K1 with the fluency weight wf 1, 2 | 13.48, 13.57 | 0.238, 0.226 | 0.218, 0.212 |
| GR | K1 with the grammar term | 12.83 | 0.213 | 0.264 |

Long runs (cut 1,500): HLT_long 14.5 words, link 0.109; K1_long 14.7, 0.160; K4_long 14.6, 0.153; human 21.2, 0.513.

**The knobs chosen:**
- **Temperature** for link. T 1.0 → 0.8 raised link by 0.078, and readers preferred the colder text (0.855).
- **The retrieved documents' weight `rel`** for link. rel 0.2 → 1.5 raised link by 0.11 (k 4); retrieval itself +0.03 (HLT → K1), and readers preferred it (0.860 / 0.880). rel 1.5 copied too much (copy32 0.501), so the controller stays at or under 1.0.
- **A gate length weight `wl`** for words: a new term in the sentence gate, wl × z(words in the draft) over the 16 drafts. A draft that does not end with . ! or ? (cut at the gate's 240 bytes) counts as the shortest, so the term never favours a run-on cut short. No existing knob moves length (table above), so the gate's choice among K5's own drafts is the lever. This is one of "the gate's weights".

**Knobs not chosen:**
- **Nucleus.** At T 0.8 the temperature and the nucleus both trim the tail; N7 read the same as G16 once combined (NG 0.510, §13.51). One tail knob is enough.
- **The gate's prompt association (wp), LDA and fluency (wf) weights.** They do not move link (F1 and F2: −0.01 to +0.01), and the gate term on the link itself (R2, `wr`) raised it and read worse (0.390, §13.53). So link is moved through content (temperature, retrieval), never by picking drafts for the link.
- **The decoder's ratio exponent** (D4, `--mixdec ratio=R`). MR read better than K1 (0.750), but it does not move link (on_prev +0.015 [−0.016, +0.045], K5_MIXDEC), and it makes sentences shorter (6.7 sentences in 460 characters against 5.3), against the words target. It also needs the learner primed for every prompt (about twice the wall time). And D5 is testing it on T8 now; a controller arm that turned it on would mix the two questions. So the controller runs with `--mixdec` off.

## The control law

Code: `native/include/cypha/lm/control.hpp` (new), `sample --k5 … --k5-control SPEC`. Off by default; with it off, nothing new is computed.

- **When:** before each sentence K5 writes itself: each gate sentence (`gate_z`, prompts with no plan) and each free sentence of the discourse writer (`free_sentence`). The planner's fact sentences are never touched; the controller reads them as text.
- **What it reads:** every sentence finished since the last call (the same split and content words as `ctl_measures.py`, in C++). For each sentence: link (1 if it shares a content word with the sentence before; the prompt's last sentence before the first) and its words.
- **Running means:** x ← (1 − λ) x + λ v, λ = 0.5, so the last sentence weighs half and a drift shows within two sentences. Words are averaged in log2. Both start at the target, so **the first sentence is written with T8's own knobs.**
- **Error, with a dead band:** e = target − x, less the band; no action inside it. Link band ±0.15 (0.358–0.658). Words band ±0.3 in log2 (16.6–25.1 words, near the human texts' own quartiles 17.7–25.7).
- **Leaky integral:** I ← clamp(0.7 I + e, −2, 2), once a sentence. It adds an error that stays; the leak stops windup.
- **Signal:** u = clamp(kp e + ki I, −1, 1). Link: kp 4, ki 1 (0.25 past the band is full action). Words: kp 2, ki 0.5.
- **Knobs for the next sentence:**
  - temperature = 0.8 − 0.1 u_link, in [0.70, 0.90];
  - rel = 0.5 × 2^(u_link), in [0.25, 1.0] (a copy of that continuation's overlay; the shared counts never change);
  - wl = u_words, in [−1, 1] (the gate's other terms have weight 1).
- When link runs above the band (a text that keeps repeating a word), the loop warms the sampler and lowers rel. When words run long, wl turns negative.
- **It learns nothing.** No count, weight or model changes from the text it writes (that looped in GEN-C). It only moves knobs.
- A line per controlled sentence goes to stderr (`K5CTL`: sentences read, temperature, rel, wl, the running link and words), so the knobs' path can be reported.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus (each also long, `--bytes 1800 --k5-max-words 600`):

| Arm | Flags | Loop |
|---|---|---|
| CL | `--k5-control link=1` | link → temperature and rel |
| CW | `--k5-control words=1` | words → the gate's length weight |
| CLW | `--k5-control link=1,words=1` | both |

Three arms; the fourth slot is left empty. One arm at a time, at most 6 threads, started only with at least 16 GB available.

## Measures (dev)

- verbatim13 (Odin's 12-word cap) and copy32, short and long;
- fact precision: `python3 -I scripts/gen_fact_check.py run --v2`, with its 95% interval;
- the properties (`ctl_measures.py`, cut 460 on the short run and 1,500 on the long run): words, link, short, each arm minus T8 with a 95% paired bootstrap;
- `hook_screen.py` at 460: mean surprise, on_prev, stuff3;
- SCRIBE v2 and v3.1 (short runs);
- the knobs' path from the `K5CTL` lines: mean, spread, share at each clamp;
- wall time and peak RAM; samples unlike T8.

## The screen

An arm passes if all hold:
1. **verbatim13 is 0** on the short run (Odin's 12-word cap), and on the long run no more than T8_long's 1 of 250.
2. **Fact precision at least 0.874.**
3. **Distance to the human targets.** For each property the arm steers, |arm − human| is at least 10% smaller than T8's (T8_long's on the long run) on at least one of the two readings, and not larger on the other. No other of words, link and short moves away from human by more than 10% of T8's gap.
4. **Not bland:** mean surprise no more than 0.5 bits a word below T8's.
5. **My strict reading.** The 12 unrated dev prompts read before (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13) at 460 characters, and 0–3 at 1,500 (none is in the long packet), each beside T8. I count whole sentences as K5_DECODE did (an English clause with a subject and a finite verb that does not break off). The arm passes if it has no fewer whole sentences than T8, and no more run-ons, keyword repetition or patchwork, pasted training text, or drift from the subject.

**Candidates:** at most 2 that pass, one per loop where possible (CLW counts as both). Ranked by how much of T8's gap to the human targets the arm closes on its steered properties at 1,500 characters (the mean of the relative shrinks), then by whole sentences. If none passes, nothing goes to the raters and I say so.

## The deciding judge

- **460 characters:** `python3 scripts/blind_rate.py make --out bench/blind/ctl_pair --ids bench/blind/llm_rating/prompts.json --arm T8=bench/k5_decode/runs/T8.gen.txt --arm X=… [--arm Y=…] --mode pair --baseline T8 --seed 20261206`: the 50 prompts, four fresh blind raters.
- **1,500 characters:** `--out bench/blind/ctl_long --ids bench/blind/coh_long/prompts.json --arm T8=…/T8_long.gen.txt --arm X=…_long.gen.txt … --mode pair --baseline T8 --seed 20261207 --cut 1500`: the 25 prompts of `coh_long`, four more fresh raters.
- The packets, `key.b64`, `RATER_PROMPT.md`, the GEN files, summaries and fact checks are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The rule:** an arm reads better on a reading if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign (§13.38). An arm is **adopted** if it reads better on either reading and its win share on the other is at least 0.5. The best adopted arm (higher lower bound of the two readings' weaker, then lower copy32) is proposed as the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

| # | Prediction |
|---|---|
| C1 | My build with the flag off equals T8 (`cmp`) |
| C2 | CL: link at 460 0.310 → 0.33–0.37; at 1,500 0.222 → 0.25–0.29; words within ±1 |
| C3 | CW: words at 460 13.6 → 15.5–17.5 (long 14.2 → 15.5–17.5); short 0.204 → 0.12–0.16; link within ±0.03 |
| C4 | CLW: about the sum of C2 and C3 |
| C5 | The knobs: in CL, temperature below 0.8 on at least 60% of controlled sentences, at the 0.70 clamp on 25–50%; in CW, wl above 0 on at least 70% |
| C6 | copy32 (T8 0.487): CL and CLW +0.02 to +0.08; CW within ±0.03 |
| C7 | verbatim13 0 on the short runs of CW; in CL and CLW 0 is expected, but one 13-word copy in 250 is possible (TF broke the cap that way, §13.71); long runs 0–2 of 250 |
| C8 | Fact precision within ±0.03 of T8's 0.924; every arm at least 0.874 |
| C9 | Mean surprise: CL and CLW 0.1–0.3 bits a word below T8; CW within ±0.1 |
| C10 | The screen: CW and CLW pass the numbers; my reading fails CW about half the time (longer drafts may be run-ons) |
| C11 | Raters: the best candidate 0.45–0.60 at 460 characters (the first sentence is T8's own), 0.50–0.65 at 1,500; no arm reaches 0.70; adoption by the rule is less likely than not |
| C12 | Wall time within +15% of T8's (12:35 short, 21:20 long); peak RAM about T8's 11.3 GB |

## Before this was written (stated so the reader can weigh it)

- **A smoke run.** To check that the loop runs and to time it, I ran CLW (`link=1,words=1`) with `cyphalm_trie sample` on dev prompts 0–11 (no SCRIBE), 6 threads, on a shared PC (load about 80 on 64 cores). I did not read its text. 5:57 wall with the 140 s build, 11.1 GB. The knobs over its 68 controlled sentences: temperature mean 0.767 (at 0.70 on 35%, at 0.90 on 9%), rel mean 0.69, wl mean 0.64 (at 1 on 53%, at 0 on 16%). So both loops act, and neither sits at a clamp all the time.
- After it, only a selftest's expected word count was fixed. The law, the targets and the arms were not changed.
- **The measures** in the tables above were made on committed dev files and the human dev continuations, before any arm ran, to choose the properties and the knobs.

## Results

(Added after the runs, under this line.)
