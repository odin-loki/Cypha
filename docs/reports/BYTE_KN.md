# BKN: Kneser–Ney over bytes for CyphaLM's byte model

**Odin Loch** · 2026-10-29 · HANDOFF §5.22 · follows [TOKENS_T0_T2.md](TOKENS_T0_T2.md) · decided by measurement under PLAN ledger 55, open to Odin's veto

(The design, the reads, the rules and the predictions below were written and committed before any BKN run. Results are added under them.)

**Update, BKN2 (2026-10-29).** Kneser–Ney is now built inside `qa` under 200 MB (built 196.8 / 197.1 MB, served 196.4), with the same tables. By BKN2's registered reads, 6:4:2:1, way (b), is the default of `qa`, `sample` and `context8k` (`--no-bkn` turns it off). See the BKN2 sections at the end. The "Decision (rule 3)" below is the earlier one, before BKN2.

**Question.** TOK-A found that interpolated modified Kneser–Ney over single bytes (V = 256) at order 8–12, mixed with the ∞-gram, beats CyphaLM's byte trie + InfiniGram by 0.18–0.19 bits per byte on the dev holdout (order 8 1.8951, order 12 1.8831, against 2.0725). Its tables were unpacked: 526 MB at order 8, 1.3 GB at order 12. Can a packed, pruned Kneser–Ney keep most of that gain inside Odin's RAM budget (built and served `qa` under 200 MB)? Does it hold on fresh text, and on top of the class, similar-context and copy experts?

**Rule 10.** Kneser–Ney is an expert inside CyphaLM. Nothing here answers a question.

## The model (fixed before any run)

`native/include/cypha/cyphalm/byte_kn.hpp`, class `ByteKN`.

- **Estimate:** the same as `token_model.hpp`'s `KneserNey` with V = 256 (Chen & Goodman 1998): orders 1..K; raw counts at order K, continuation counts (distinct left extensions) below it; three discounts per order from the count of counts, each clamped to [0, j]; a uniform base 1/256. P_k(w | c) = max(n(c w) − D, 0) / n(c) + γ(c) P_{k−1}(w | c′), with γ(c) = (D₁ n₁(c) + D₂ n₂(c) + D₃₊ n₃₊(c)) / n(c). A context not stored ends the recursion at the shorter one.
- **Packed form:** per order, the contexts sorted by a 64-bit key (the key of `token_model.hpp`: a hash grown to the left, one step per byte), a 32-bit offset into the children and γ as a float (16 bytes a context); per child, its byte (1 byte) and its discounted value max(n − D, 0) / n(c) as a float (4 bytes). So the lookup is p ← value(c, w) + γ(c) p, order by order.
- **Pruning** (P[m, c]): at orders ≥ m, contexts seen fewer than c times are dropped (the walk backs off), and children seen once are dropped with their discounted value added to γ, so every stored context still sums to 1 exactly.
- **Build:** order by order, in 256 partitions by the context's last byte, so the build holds one partition's pairs at a time.

## The ways in (fixed before any run)

- **(a) Replace:** Kneser–Ney in place of the trie's six-expert mixture; InfiniGram mixed over it by λ per (match length × count) bucket, exactly as today.
- **(b) Add:** Kneser–Ney mixed over the trie mixture by λ per bucket (bucket = the deepest order found, 1..K, × three count bands of that context: < 4, < 64, ≥ 64); then InfiniGram over that, as today.
- **The stack:** each way also with the defaults on top (class `shape`, similar-context `default`, copy, as `qa` mixes them: class, then similar-context, then copy, each by λ per its own bucket, fit on the first half).

## The reads

A new subcommand, `cyphalm_trie bkn <wiki> <SPEC>...`, on `cls`'s protocol (wiki-only training text, as TOK-A's baseline):

- **Text:** train [0, 7,340,032) of enwik8; holdout [7,340,032, 8,388,608); fresh [8,388,608, 9,437,184).
- **Dev (holdout2):** every λ fit on the holdout's first half; bits per byte on its second half. Every choice is made here.
- **Fresh (once):** λ refit on the whole holdout; bits per byte on the fresh MiB; paired gains with 4 KiB block bootstraps (4,000 resamples). Read once, for the chosen configuration only.
- **RAM:** the packed tables' bytes; the build's peak RSS (`/root/work/r2/peak.sh`, one configuration per run); build time.
- **Configurations on dev:** orders 5, 6, 8, 10, 12 unpruned (12 if the machine allows); pruned P[5, 2], P[4, 2] at orders 6, 8, 10.

**Amendment 1 (registered before any BKN run on the reads above).** A unit check on 1.8 MB of enwik8 (`token_model.hpp` against `byte_kn.hpp`, built while writing the header) showed that dropping singleton children at order 8 costs much more than its size saving suggests (2.254 → 2.395 bits per byte there). So the dev configurations also include context-only pruning, written m:c:0 (from order m, contexts seen fewer than c times dropped; children kept): 6:5:2:0, 8:5:2:0, 8:4:2:0, 10:5:2:0, 10:4:2:0, 12:5:2:0. They are judged by the same rules. Specs are written K:m:c:d (d = 1: children seen once dropped), so P[m, c] above is m:c:1.

## Rules (fixed before any run)

1. **Dev choice.** Among configurations whose packed tables are at most **14 MB** (`qa`'s heaviest built configuration, C1L + defaults, peaks at 185 MB), take the lowest dev bits per byte of way (a) or (b) with InfiniGram. If none gains at least 0.02 bits per byte over trie + InfiniGram on dev, stop: report, keep the flag, no fresh read.
2. **Fresh read** (once). The chosen configuration against trie + InfiniGram, and its stack against today's stack: the gain's 95% block-bootstrap interval must lie below 0 (P(gain > 0) ≥ 0.95, LCI ≥ 0.95).
3. **Default.** Kneser–Ney becomes CyphaLM's default byte expert only if rule 2 holds **and** in `qa`: built and served peaks under 200 MB, answers on dev not fewer right at the 2%-wrong point (then the frozen Q/A test read once), and `sample`'s SCRIBE v2 on dev not below the current default's 95% lower bound. Until all of these are measured, it stays behind the flag.
4. Every default output of `cyphalm_trie` stays byte-identical without the flag.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| K1 | Unpruned packed Kneser–Ney reproduces TOK-A's KN and KN + INF bits per byte at orders 5 and 8 within 0.0005 (the same counts; float values) |
| K2 | Unpruned packed tables at order 8 on the 7 MiB train text take 60–150 MB |
| K3 | Pruned P[5, 2] at order 8 fits in 25 MB or less and keeps at least 70% of the unpruned order-8 gain over trie + InfiniGram on dev |
| K4 | Way (b) beats way (a) by less than 0.01 bits per byte on dev at every configuration |
| K5 | With the defaults on top, the gain is at most half of the gain over trie + InfiniGram |
| K6 | Some configuration of 14 MB or less passes rule 1 (gains at least 0.02 on dev) |
| K7 | The fresh read passes rule 2 for trie + InfiniGram |

## The dev choice (rule 1), recorded before the fresh read

Dev (holdout2) read of 18 configurations: `cyphalm_trie bkn` at commit 95f1b43 (`/root/work/bkn/runs/dev.out`). Of those whose packed tables are at most 14 MB, two remain: order 5 (12.5 MB; (b) 1.9944) and **6:4:2:1** (13.8 MB; (b) **1.9324**, gain over trie + InfiniGram −0.1401 [−0.1454, −0.1350]). Rule 1 takes **6:4:2:1, way (b)**. Its gain passes the 0.02 bar, so the fresh text is read once for it, with the defaults on top (`--bkn-fresh --bkn-stack`).

## Results: dev (holdout2, 2026-10-29)

`cyphalm_trie bkn` at commit 95f1b43 on the bpc model (wiki-only training text, as `cls` and TOK-A's baseline); `/root/work/bkn/runs/dev.out`. The base reproduces B4 and TOK-A exactly: **trie 2.278066, trie + InfiniGram 2.072500**. Bits per byte on the holdout's second half; λ fit on its first half. Tables are the packed bytes (MB = 10⁶ bytes). The gain is way (b) with InfiniGram against trie + InfiniGram, with its 95% block-bootstrap interval (4 KiB blocks, 2,000 resamples).

| Spec | Tables (MB) | Contexts | Build (s) | KN alone | (a) KN + IG | **(b) trie + KN + IG** | (b) gain |
|---|---:|---:|---:|---:|---:|---:|---|
| 5 | 12.5 | 404,600 | 28 | 2.1611 | 1.9957 | 1.9944 | −0.0781 [−0.0830, −0.0735] |
| 6 | 31.3 | 1,130,385 | 34 | 1.9984 | 1.9255 | 1.9224 | −0.1501 [−0.1558, −0.1446] |
| 8 | 110.2 | 4,394,035 | 59 | 1.9168 | 1.8949 | 1.8870 | −0.1855 [−0.1916, −0.1796] |
| 10 | 251.2 | 10,493,776 | 79 | 1.8971 | 1.8875 | 1.8783 | −0.1942 [−0.2007, −0.1879] |
| 12 | 447.4 | 19,177,232 | 99 | 1.8880 | 1.8827 | 1.8735 | −0.1990 [−0.2055, −0.1929] |
| 6:5:2:1 | 15.4 | 575,020 | 39 | 2.0471 | 1.9398 | 1.9307 | −0.1418 [−0.1472, −0.1366] |
| **6:4:2:1** | **13.8** | 539,319 | 40 | 2.0672 | 1.9438 | **1.9324** | **−0.1401 [−0.1454, −0.1353]** |
| 8:5:2:1 | 36.1 | 1,541,336 | 54 | 2.0520 | 1.9443 | 1.9272 | −0.1453 [−0.1505, −0.1401] |
| 8:4:2:1 | 34.5 | 1,505,635 | 55 | 2.0760 | 1.9483 | 1.9290 | −0.1435 [−0.1487, −0.1384] |
| 10:5:2:1 | 58.4 | 2,644,218 | 72 | 2.0630 | 1.9495 | 1.9324 | −0.1401 [−0.1455, −0.1350] |
| 10:4:2:1 | 56.8 | 2,608,517 | 79 | 2.0884 | 1.9535 | 1.9341 | −0.1384 [−0.1437, −0.1335] |
| 6:5:2:0 | 19.1 | 575,020 | 43 | 2.0406 | 1.9418 | 1.9319 | −0.1406 [−0.1458, −0.1356] |
| 8:5:2:0 | 47.4 | 1,541,336 | 58 | 2.0401 | 1.9552 | 1.9336 | −0.1389 [−0.1441, −0.1339] |
| 8:4:2:0 | 46.6 | 1,505,635 | 54 | 2.0545 | 1.9582 | 1.9345 | −0.1380 [−0.1431, −0.1330] |
| 10:5:2:0 | 78.5 | 2,644,218 | 60 | 2.0501 | 1.9684 | 1.9435 | −0.1290 [−0.1343, −0.1240] |
| 10:4:2:0 | 77.7 | 2,608,517 | 63 | 2.0645 | 1.9714 | 1.9444 | −0.1281 [−0.1335, −0.1231] |
| 12:5:2:0 | 103.8 | 3,559,616 | 74 | 2.0607 | 1.9749 | 1.9490 | −0.1235 [−0.1289, −0.1185] |

(Build times were taken on a machine at load 17–30 with about twenty other jobs; scoring was 2.6–26 µs a byte.)

**What dev shows.**
- **Unpruned Kneser–Ney reproduces TOK-A** to the printed digit: order 5 KN 2.1611 and KN + IG 1.9957 (TOK-A 2.1611, 1.9929 with its own λ fit), order 8 1.9168 / 1.8949 (TOK-A 1.9168 / 1.8951), order 12 1.8880 / 1.8827 (TOK-A 1.8880 / 1.8831).
- **Packing cuts the cost about five times** against TOK-A's unpacked tables (order 8: 526 MB peak there; 110 MB of tables and a 189 MB build peak here), but long orders still do not fit Odin's budget.
- **Pruning removes almost all of what long orders add.** Every pruned order 8, 10 or 12 lands at 1.927–1.949, no better than order 6, because the contexts of rare long strings are exactly what makes long orders win. Dropping contexts seen once from order 4 on costs order 6 only 0.010 (1.9224 → 1.9324) for 17.5 MB less.
- **Way (b) is a little better than way (a) everywhere** (0.002–0.026): the trie's mixture and its 4-byte match still add something over Kneser–Ney.
- **The choice (rule 1):** 6:4:2:1, way (b), 13.8 MB, dev −0.1401.

## Results: the fresh read (once, 2026-10-29)

`cyphalm_trie bkn /root/cdata/enwik8 6:4:2:1 --bkn-fresh --bkn-stack` at commit d4a87c0's tool (`/root/work/bkn/runs/fresh.out`). λ refit on the whole holdout; bits per byte on [8,388,608, 9,437,184). The defaults on top are `qa`'s order: class `shape`, similar-context `default` (A4d), copy (min 8), each by λ per its own bucket, here all built on the wiki training bytes (the `cls` protocol).

| Read | Base | With Kneser–Ney 6:4:2:1 | Gain, 95% interval (4,000 resamples) |
|---|---:|---:|---|
| Fresh, trie + IG → (a) KN + IG | 2.1660 | 2.0210 | −0.1450 [−0.1520, −0.1385] |
| Fresh, trie + IG → **(b) trie + KN + IG** | 2.1660 | **2.0115** | **−0.1545 [−0.1607, −0.1486]** |
| Fresh, the stack (trie + IG + cls + sim + copy) → (a) | 1.8069 | 1.7438 | −0.0630 [−0.0663, −0.0598] |
| Fresh, the stack → **(b)** | 1.8069 | **1.7447** | **−0.0621 [−0.0650, −0.0594]** |
| Holdout2 (dev), the stack → (b) | 1.8022 | 1.7434 | −0.0588 [−0.0624, −0.0554] |

- **Rule 2 holds** for both: every resample gains (P(gain > 0) = 0, so LCI ≥ 0.95).
- **With the defaults on top the gain is 40% of the gain over trie + InfiniGram** (fresh −0.062 against −0.155): the class, similar-context and copy experts already cover part of what Kneser–Ney smoothing adds.

## Results: `qa` and `sample` with `--bkn 6:4:2:1` (way b), 2026-10-29

Rule 3's reads. The tool is commit ce24f6a's (`/root/work/bkn/bin/cyphalm_trie_bkn`). `qa` is the default model (C1L as A4d ran it, with `--sim default,online=1,probe=128`), built with `--save` and then served with `--model`. In `qa`, Kneser–Ney is built on T (wiki train + chat), so its tables are bigger than on the wiki bytes alone.

| Read | Without the flag | With `--bkn 6:4:2:1` |
|---|---:|---:|
| `qa` holdout2, trie mixture | 2.3224 | 2.0323 (KN mixed in) |
| `qa` holdout2, trie + IG | 2.0729 | 1.9207 |
| `qa` holdout2, trie + IG + cls + sim | 1.8398 | 1.7743 |
| **`qa` holdout2, the default stack (+ copy)** | **1.7895** | **1.7273** |
| Kneser–Ney tables in `qa` (on T) | — | 24.3 MB (`kn.bkn` 24.3 MB; the store 79 → 102 MB) |
| **`qa` built peak, with `--save`** | **186.7 MB** | **286.5 MB** |
| `qa` served peak (`--model`) | 172.3 MB | 196.0 MB |
| Dev answers, chat form, trie + IG: cross-validated gate | 2 right, 0 wrong | 3 right, 2 wrong (wrong rate 0.015) |
| Dev answers, full-dev gate | 3 right | 4 right |
| Dev answers, raw (no gate) | 7 right, 57 wrong | 6 right, 56 wrong |
| `sample` SCRIBE v2, pooled (A4d's dev prompts) | 0.063 [0.051, 0.078] | **0.103 [0.087, 0.120]** |
| `sample` SCRIBE v2 at T 0.6 / 0.8 / 1.0 | 0.081 / 0.070 / 0.037 | 0.137 / 0.110 / 0.059 |
| `sample` surface S, pooled | 0.860 | 0.893 |
| `sample` copy share (bytes inside a 32-byte span found verbatim in training) | 0.0429 | 0.0436 |
| `sample` peak | 79.1 MB (A4d) | 97.9 MB |

- **Parity.** Without the flag, `qa`'s rows are byte-identical to A4d's run (`/root/work/a4d/runs/QO.rows.tsv`), and its output differs only in timings. Served rows equal built rows, with and without the flag, so the stored Kneser–Ney loses nothing.
- **The gain holds inside `qa`'s full stack:** −0.062 bits per byte on the holdout, the same size as the fresh read's −0.062.
- **Text.** SCRIBE v2 rises with disjoint intervals (0.063 → 0.103) without more copying (+0.0007). It is still far below shuffled sentences (0.344): word salad with fewer broken words. Two samples on the same prompts (T 0.8):
  - Prompt "The requirement is that a person be the subject of the attack or".
    - Before: `&quot;save&quot; percentages, which in considerable [[Algiers]], [[Strained the continent defeat all be real in arithmetic which were subsequent formed`
    - After: `''Apollo '''Article by means the [[Soviet States at the election about an alternate that all options form an inter of exploration countries like and the surrounding`
  - Prompt "The two Emperors met at Tilsit on the [[25 June]], [[1807]]. Ale".
    - Before: `xand station on the contrary, the and [[Messier 1980s and African compared and started Spain|The Bierce the [[comment of had her of later in [[1995]]`
    - After: `xandria]].\n\n=== Accuracy===\n[[Image:Adelaide Crows and promoting possible to the right characteria corner.\n\n==External links==`
- **Answers.** Dev answers are few (130 questions, 2–4 right). With the flag the gate keeps one more right answer and two more wrong ones, both inside the 2% point. They do not fall.
- **The built peak fails.** Building Kneser–Ney inside `qa` adds about 100 MB at the peak (186.7 → 286.5 MB). The build runs after the index, the class table and the lowered index are resident, on a second copy of T taken from the index (about 21 MB), with its own scratch (the 16 MB position chunk and the largest group's triples). Served, it costs only its tables (+23.7 MB, 172.3 → 196.0 MB).

## Decision (rule 3)

| Condition | Result |
|---|---|
| Fresh BPC gain, interval below 0 (rule 2) | **Pass:** −0.1545 [−0.1607, −0.1486]; with the defaults −0.0621 [−0.0650, −0.0594] |
| Dev answers not fewer right at the 2%-wrong point | **Pass:** 2 → 3 right (wrong 0 → 2, rate 0.015) |
| SCRIBE v2 not below the default's 95% lower bound | **Pass:** 0.103 against 0.051 (it rises, disjoint) |
| Built `qa` under 200 MB | **Fail:** 286.5 MB |
| Served `qa` under 200 MB | Pass: 196.0 MB |

**Byte Kneser–Ney is not made a default.** It stays behind `--bkn K[:m:c:d]` (`--bkn-way a|b`) in `qa` and `sample`, and `cyphalm_trie bkn` gives its reads. The frozen Q/A test was not read, because the default did not change. Every default output is byte-identical to before.

**What would let it pass** (the next change, with its own registered read): build Kneser–Ney in `qa` without the extra 100 MB. Build it before the lowered index and the class table are resident, or from T before the index takes it (no second copy), and split the largest groups so the triples stay small. The served cost (+24 MB, 196 MB) shows the tables themselves fit. On `sample`'s wiki-only text it already fits (97.9 MB).

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| K1 | Unpruned KN and KN + INF reproduce TOK-A within 0.0005 at orders 5 and 8 | **Partial:** KN alone exact at both (2.1611, 1.9168); KN + INF exact at order 8 (1.8949 against 1.8951), 0.0028 apart at order 5 (1.9957 against 1.9929): `cyphalm_trie`'s byte buckets for InfiniGram's λ, not TOK-A's |
| K2 | Unpruned packed order 8: 60–150 MB | **Hit:** 110.2 MB |
| K3 | 8:5:2:1 at most 25 MB and keeps at least 70% of order 8's gain | **Miss:** 36.1 MB (it keeps 78% of the gain) |
| K4 | Way (b) beats (a) by less than 0.01 everywhere | **Miss:** under 0.01 unpruned and at orders 5–6; 0.011–0.026 for the pruned long orders |
| K5 | With the defaults on top, at most half the gain | **Hit:** 40% fresh (0.062 of 0.155), 42% dev |
| K6 | Some configuration of 14 MB or less gains at least 0.02 on dev | **Hit:** 6:4:2:1, −0.140 |
| K7 | The fresh read passes rule 2 | **Hit** |

4 hits, 1 partial, 2 misses.

## Checks

- `byte_kn.hpp` equals `token_model.hpp`'s Kneser–Ney over bytes: the same bits per byte at orders 1, 2, 5 and 8 on 1.8 MB of enwik8 (probabilities within 5e-8, from float storage). Every distribution sums to one, pruned or not. The chunked build gives the same numbers as the first build.
- `cyphalm_trie selftest` gains these checks. Three planted bugs each fail it: raw counts below the top order, a pruned singleton's value not moved into γ, and the wrong group in the lookup.
- The nine CTests of HANDOFF §4.4 plus `native_cyphalm_tok_selftest` and `native_token_bridge_test` pass (11 of 11).
- Without the flag, `qa`'s rows and output equal A4d's (timings aside), and `sample`'s file (10 prompts, 200 bytes, the MODEL line included) is byte-identical to A4d's tool's. Served equals built, with and without the flag.

## How to reproduce

```bash
B=build/cyphalm_trie; E=/root/cdata/enwik8
$B bkn $E 5 6 8 10 12 6:5:2:1 6:4:2:1 8:5:2:1 8:4:2:1 10:5:2:1 10:4:2:1 6:5:2:0 8:5:2:0 8:4:2:0 10:5:2:0 10:4:2:0 12:5:2:0   # dev
$B bkn $E 6:4:2:1 --bkn-fresh --bkn-stack                                                                                     # the fresh read (once)
# qa as A4d ran it (C1LD = A4d's flags), without and with the flag: built with --save, then served with --model
$B qa $E dolly.jsonl $C1LD --sim default,online=1,probe=128 [--bkn 6:4:2:1 --bkn-way b] --save DIR --out rows.tsv
$B qa $E dolly.jsonl $C1LD --sim default,online=1,probe=128 [--bkn 6:4:2:1 --bkn-way b] --model DIR --out rows.tsv
$B sample $E dev.txt fresh.txt --sim default,online=1,probe=128 --bkn 6:4:2:1 --bkn-way b    # then scribe ladder as A4d
```

Runs are in `/root/work/bkn/runs` (dev, fresh, `qa/`, `sb/`).

## BKN2: Kneser–Ney built inside `qa` under 200 MB (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · follows the decision above

**Question.** Building Kneser–Ney inside `qa` raised the built peak from 186.7 to 286.5 MB. Served, it costs only its tables (+23.7 MB, 196.0 MB). Can the build fit under 200 MB with the same tables? If it can, does Kneser–Ney pass the rest of rule 3 and become the default?

**Where the 100 MB comes from** (read from the code; not yet placed by a trace).
- The default similar-context expert keys the wiki bytes only (`src=wiki`), so the index takes T and T is not resident. Kneser–Ney is built on a second copy of T from the index (about 21 MB).
- It is built after the index, the class table and the lowered index (about 26 MB, kept for the store) are resident.
- One group is still large. In T, the group of positions after a space holds about 3–4 million positions. Its entries take 24 bytes each (context key, left-extension key, byte): about 80–100 MB.
- The pairs made from them grow by doubling. The output arrays also grow by doubling, and `shrink_to_fit` copies them at the end of each order.
- Positions are gathered 4 Mi at a time (16 MB).

**The change to the build** (`byte_kn.hpp`; the tables must come out the same).
1. **Parts by key.** Each group is cut into P parts by its context key: part p holds the contexts whose key k has ⌊k·P / 2⁶⁴⌋ = p, with P = ⌈group size / 2¹⁹⌉. Parts run in key order, so contexts come out in today's order. A context never spans two parts.
2. **Slices inside a part.** A part's positions are read in slices of 2¹⁹. Each slice's entries are sorted and merged into the part's running set, which keeps only distinct entries: (context, byte, left byte) below the top order, (context, byte) with summed counts at the top order. A part holds its distinct entries, not one per position. This keeps orders 2–3 small, where a few contexts fill a whole group.
3. **The left byte, not its key.** Below the top order an entry keeps the left extension's byte in place of its 64-bit key. For a fixed context the key step is one-to-one in the byte, so the distinct count is the same. Entries shrink from 24 to 16 bytes.
4. **Positions.** Gathered 1 Mi at a time (4 MB). A group bigger than that is not gathered: each of its parts scans the text.
5. **Exact sizes.** Pass 1 also counts the contexts and children that pruning keeps. Pass 2 reserves the arrays at that size, so nothing doubles and nothing is copied at the end.
- The part and slice size is a parameter of `build` (default 2¹⁹), so the self-test can force many parts and slices.

**The change to `qa`** (`cyphalm_trie.cpp`, only when Kneser–Ney is built).
- **The lowered index leaves early.** With `--save`, under M17's condition (an entity statistic in the token, no `--cards`), the lowered index is written to `low.fm` and freed at M14's point, before the chat text is trained. `save_qa` then lists `low.fm` without writing it again. Nothing after M14's point reads it (M14's argument).
- **One copy of T.** Kneser–Ney is built right after the class table, from the copy of T the class build already takes from the index, before that copy is freed. When the similar-context expert keeps T (`src=all`), from T itself, as before.
- It is not moved before the index. Its 24 MB of tables would then sit through the index build (M16: 178.5 MB in C1L + defaults).

**Checks (lossless).**
- In the same `qa` configuration, `kn.bkn` from the new build is byte-identical to the old build's (`/root/work/bkn/runs/qa/store_on/kn.bkn`), and so is every other store file.
- Harness (`ByteKN` alone, 6:4:2:1 on 21 MB of enwik8): old and new tables byte-identical; each one's peak and time.
- `cyphalm_trie bkn $E 6:4:2:1` prints the same numbers as before (1.932358 for (b) + InfiniGram).
- Self-test: the build with parts and slices of 64 and of 1,000 entries gives the same tables as one part, at orders 1, 3 and 6, pruned and not.
- Without `--bkn`, `qa`'s rows and output (timings aside) equal the merge commit's (a1f218a).

**Reads, in this order.**
0. The old tool (a1f218a), `qa` with `--save --bkn 6:4:2:1`, under M15's trace (`/root/work/ram2/prof.sh`: `VmRSS` every 0.25 s beside the progress lines), once: where its 286.5 MB peak is, and the level after the build with the tables resident.
1. The new tool, the same command under the trace. If its peak is within 5 MB of 200, a second run.
2. The same model served (`--model`).

`qa` is C1L as A4d ran it (the merge check's flags) with the defaults (class, similar-context, copy).

**If the built and served peaks are both under 200 MB,** 6:4:2:1, way (b), is wired as the default of `qa`, `sample` and `context8k`; `--no-bkn` turns it off. Then:

3. Dev answers (from read 1): the cross-validated gate keeps at least as many right answers as without (2), at a wrong rate of 2% or less.
4. The frozen Q/A test, read once with the new default, in A1d's test configuration (`--test bench/eval/t_qa_wiki.tsv --parse parse_all_span.tsv`, `ent_title`, the low-only token). Gated: wrong rate 2% or less, and at least as many right as A1d's TEST_C1L (0). Logged in A1D_A2B's table of test reads.
5. `sample` on GEN-A's dev benchmark (`scripts/gen_eval.sh dev`): pooled SCRIBE v2 at least the default's 95% lower bound (default 0.079 [0.064, 0.096]), and the copy share at most the default's + 0.02 (GEN-A's rule; default 0.0414).
6. `context8k`: its lines and peak, recorded. Its peak is not under the 200 MB rule: it is 202.3 MB without Kneser–Ney (M16).
7. With `--no-bkn`, `qa`'s rows, `sample`'s file and `context8k`'s lines equal the tool's before the switch.

**Rule.** The default stays switched only if reads 1–5 and 7 hold. If any fails, Kneser–Ney stays behind `--bkn`, and the report says which read failed. If the built peak is over 200 MB and lies after the Kneser–Ney build (the tables resident through the later phases), nothing more is tried here.

**Predictions.**

| # | Prediction |
|---|---|
| K8 | Every store file and the harness tables are byte-identical to the old build's |
| K9 | Read 0 peaks inside the Kneser–Ney build, 270–300 MB |
| K10 | In read 1, the Kneser–Ney build adds at most 40 MB above what is resident when it starts (its tables included) |
| K11 | Read 1 peaks at 190–199 MB, after the Kneser–Ney build (the similar-context build or the holdout arrays), not inside it |
| K12 | The harness build takes at most twice its old time |
| K13 | Served `qa` peaks at 196 ± 1 MB, as before |
| K14 | The frozen test, gated: 0–2 right, wrong rate 2% or less |
| K15 | GEN-A's dev: pooled SCRIBE v2 0.10–0.14, the copy share within ±0.01 of the default's |
| K16 | `context8k`'s last chain line falls by 0.04–0.08 bits per byte; its peak rises by 20–40 MB |

### Results: BKN2 (2026-10-29)

- **Tools.**
  - Old: the merge commit a1f218a.
  - New build: 0b3c017.
  - Default switch: 50fa5f4. Its binary is identical to the one the reads after the gate used (same md5 before and after the commit).
- **Units.** Peaks are `VmHWM` in kB / 1000.
- **Traces.** M15's wrapper polled `VmRSS` every 0.25 s beside the progress lines.
- **Times are rough.** Other tracks shared the machine (load 14–22 on 4 cores).
- **`qa` configuration.** C1L as A4d ran it, with the defaults (class, similar-context, copy).

#### The build is lossless

| Check | Result |
|---|---|
| Harness, 6:4:2:1 on 21 MiB of enwik8 | Tables byte-identical (sha256 `e25012f7…`). Build peak **144.7 → 66.0 MB**; time 85.6 → 145.4 s |
| Harness on 2 MB: order 3, order 6, 6:4:2:1 | Tables byte-identical; peaks 19.1 / 34.5 / 23.9 → 13.7 / 27.5 / 20.0 MB |
| Self-test | Parts and slices of 64 and of 1,000 entries give the one-part tables at K = 1, 3, 6, pruned and not. A planted bug fails it (the merge drops earlier slices) |
| `qa` store, read 1 against read 0 | All six files byte-identical; `kn.bkn` also equals the pre-merge store's. Rows and stdout (timings masked) identical |
| `cyphalm_trie bkn $E 6:4:2:1`, new build | Every line identical (timings aside): (b) + InfiniGram **1.932358**. Peak 207.5 → 194.5 MB |

#### Where the peak was, and where it is now

Phases of `qa --save --bkn 6:4:2:1` (MB). Each phase runs from one progress line to the next.

| Phase | Read 0 (old tool) | Read 1 | Read 1, second run |
|---|---:|---:|---:|
| Wiki trained → chat trained (the lowered index's build) | 184.9 | 184.9 | 184.6 |
| Chat trained → index built (T and the index) | 180.3 | 178.9 | 178.8 |
| Index built → class table built | 169.2 | 151.3 | 155.9 |
| **Class table built → Kneser–Ney built** | **286.9** (+117.7) | **160.5** (+9.2) | **160.6** (+4.8) |
| Store written | 217.2 | 176.1 | 176.2 |
| Similar-context build, holdout, answers (to the end) | 219.4 | 196.8 | 197.1 |
| **Peak** | **286.9** | **196.8** | **197.1** |

- **The old peak** was inside the Kneser–Ney build: +117.7 MB over what was resident when it began. After it, the freed scratch stayed in the heap, so the later phases ran at 217–219 MB.
- **The new build** adds 5–9 MB, its 24.3 MB of tables included. Its scratch fits in heap the class build had freed.
- **Freeing the lowered index early** lowers the class build's phase by 13–18 MB.
- **The new peak** comes at the very end of the run, while the answers are written. It is not at the similar-context build or the holdout arrays. The tables (24.3 MB) are resident from their build on.
- **Kneser–Ney build time inside `qa`:** 153–160 s, against 170 s for the old build.

#### Reads 1–7

| Read | Without Kneser–Ney | With Kneser–Ney 6:4:2:1 (b) | Rule | Holds |
|---|---:|---:|---|---|
| 1. Built `qa`, `--save` | 186.7 MB (M17) | **196.8 / 197.1 MB** (old build: 286.9) | under 200 | Yes |
| 2. Served `qa` | 172.3 MB | **196.4 MB** (pre-merge 196.0); rows equal the built rows | under 200 | Yes |
| 3. Dev answers, cross-validated gate (chat, trie + IG) | 2 right, 0 wrong | **3 right, 2 wrong** (wrong rate 0.0154); full-dev gate 3 → 4 right | not fewer right; wrong ≤ 2% | Yes |
| 4. **Frozen test, once**, gated | 0 right, 0 wrong (TEST_C1L) | **0 right, 3 wrong** (wrong rate 0.015, Wilson upper 0.043) | wrong ≤ 2%; right ≥ 0 | Yes |
| 5. GEN-A dev, SCRIBE v2 pooled | 0.079 [0.064, 0.096] | **0.125 [0.107, 0.143]** | ≥ 0.064 | Yes |
| 5. GEN-A dev, copy share (copy32, pooled) | 0.0414 | **0.0428** | ≤ 0.0614 | Yes |
| 6. `context8k`, last chain line (+ class + similar-context + copy) | 1.7897 | **1.7276** | recorded | — |
| 6. `context8k` peak | 202.5 MB | **218.9 MB** | recorded (not under the rule) | — |
| 7. `--no-bkn` equals the tool before the switch | — | `qa` rows and stdout (timings masked) = the merge check's; `sample` 10 prompts = the merge tool's file; `sample` on GEN-A's dev prompts = GEN-A's D0 file byte for byte; `context8k` every line | identical | Yes |

**More numbers.**

- **The frozen test, without the gate** (trie + IG):
  - 2 right, 96 wrong, 102 abstained, against 2 / 101 / 97 (TEST_C1L).
  - Abstentions: 48 of 50 not-in-slice questions, precision 0.471 (was 0.495). Of the in-slice questions the trie would get wrong, 54 of 150 abstain (was 49).
  - Tags 73 (was 72).
  - Test-configuration peak: 199.9 MB, against 184.8 without Kneser–Ney (ANS's TBOLD and TDKC). This configuration is not under the rule, but it is at the edge.
- **The three gated wrong answers on the test** sit right at the gate's cut (min_lp −1.279, −1.301 and −1.304 against τ = −1.3039):
  - "The creation of metal is often attributed to the band Black Sabbath."
  - "Augustus was a leader of the Roman Empire"
  - "It is the problem of your geography."
  - The dev gate's cut moved from −0.895 to −1.304, because Kneser–Ney changes the scale of the answers' log-probabilities.
- **GEN-A dev:**
  - SCRIBE v2 at T 0.6 / 0.8 / 1.0: 0.101 / 0.088 / 0.047 → 0.157 / 0.136 / 0.080.
  - Surface S, pooled: 0.855 → 0.895.
  - BPC of the human continuations under sampling's mixture: 2.1339 [2.1016, 2.1659] → 2.0145 [1.9846, 2.0441].
  - 624 of 750 samples kept after SCRIBE's cleaning (was 618).
- **`context8k`:**
  - trie 2.3199 → 2.0313 with Kneser–Ney; trie + InfiniGram (8192) 2.0715 → 1.9200; + class 1.9940 → 1.8952; + similar-context 1.8404 → 1.7748; + copy 1.7897 → 1.7276.
  - Its answers use Kneser–Ney (it is part of the byte model). Some answer texts change; every tally is the same (seen 3/3, first letter changed 3/3, held-out Dolly 1/30, the rest 0).
- **The `qa` holdout stack** is unchanged from BKN's read: 1.7895 → 1.7273.

#### Decision

Reads 1–5 and 7 hold, so **byte Kneser–Ney 6:4:2:1, way (b), is now the default** of `qa`, `sample` and `context8k`.
- `--no-bkn` turns it off, and each subcommand then prints exactly what it printed before.
- `--bkn SPEC` and `--bkn-way a|b` choose another.
- A store built without it is served with `--no-bkn`: the store's record names `bkn=`.

**For Odin's veto.**
- **The frozen test.** Without Kneser–Ney, the gate let nothing through (0 right, 0 wrong). With it, the gate lets 3 wrong answers through and still no right one. That is inside the registered rule (wrong rate 0.015 ≤ 0.02), but it gains no right answer on the test. On dev it gains one right and two wrong.
- **RAM is close to the limit.** Built `qa` is at 197 MB, and the frozen-test configuration at 199.9 MB. `context8k` (not under the rule) rises from 202.5 to 218.9 MB.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| K8 | Store files and harness tables byte-identical | **Hit** |
| K9 | Read 0 peaks inside the Kneser–Ney build, 270–300 MB | **Hit:** 286.9 |
| K10 | The new build adds at most 40 MB over what is resident when it starts | **Hit:** +9.2 / +4.8 (old: +117.7) |
| K11 | Read 1 peaks at 190–199 MB, after the Kneser–Ney build | **Hit:** 196.8 / 197.1. But the place is the run's end (answers), not the similar-context build or the holdout arrays named in the prediction |
| K12 | The harness build takes at most twice its old time | **Hit:** 1.70× (85.6 → 145.4 s) |
| K13 | Served `qa` at 196 ± 1 MB | **Hit:** 196.4 |
| K14 | Frozen test, gated: 0–2 right, wrong rate at most 2% | **Hit:** 0 right, 3 wrong (0.015) |
| K15 | GEN-A dev SCRIBE v2 0.10–0.14; copy share within ±0.01 | **Hit:** 0.125; +0.0014 |
| K16 | `context8k` falls 0.04–0.08; peak +20–40 MB | **Partial:** −0.062; peak +16.4 MB |

8 hits, 1 partial.

#### Checks

- **Self-test:** `cyphalm_trie selftest` passes. It now checks the build's parts and slices; a planted merge bug fails it.
- **Lossless and parity:** every lossless and `--no-bkn` parity check in the tables above holds.
- **Frozen test:** read once. Its row for A1D_A2B's table of test reads:

  | Date | Change | Run | Result |
  |---|---|---|---|
  | 2026-10-29 | byte Kneser–Ney default (BKN2) | T4 (`/root/work/bkn/runs/bkn2`) | gated 0 right / 3 wrong (was 0 / 0); trie + IG raw 2 / 96 / 102 abstained |

#### How to reproduce (BKN2)

```bash
# the harness (tables old vs new): kn_build FILE BYTES SPEC OUT, compiled against each byte_kn.hpp
kn_build_old $E 22020096 6:4:2:1 old.bkn; kn_build_new $E 22020096 6:4:2:1 new.bkn; cmp old.bkn new.bkn
# reads 0-1 (old and new tool), traced; read 2 served from read 1's store
/root/work/ram2/prof.sh R1 $B qa $E dolly.jsonl $C1LD --bkn 6:4:2:1 --bkn-way b --save DIR --out rows.tsv
/root/work/ram2/prof.sh R2 $B qa $E dolly.jsonl $C1LD --bkn 6:4:2:1 --bkn-way b --model DIR --out rows.tsv
# after the switch (Kneser-Ney by default): the frozen test once, GEN-A dev, context8k, and --no-bkn parity
$B qa $E dolly.jsonl --dev DEV --test bench/eval/t_qa_wiki.tsv --parse parse_all_span.tsv --dk dk_mention.tsv \
   --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard shard_facts.tsv --base a0_rows.tsv --out T4.rows.tsv
CYPHALM_TRIE=$B scripts/gen_eval.sh dev BKN OUTDIR
$B context8k $E dolly.jsonl [--no-bkn]
$B qa $E dolly.jsonl $C1LD --no-bkn --out rows.tsv; $B sample $E dev.txt --prompts 10 --no-bkn
```

Runs are in `/root/work/bkn/runs/bkn2` (traces `R0`, `R1`, `R1b`, `R2`; `T4`; `gen/`; `C8`, `C8M`, `C87`; `Q7`, `S7`, `S7F`; `H_*`; `BK2`).
