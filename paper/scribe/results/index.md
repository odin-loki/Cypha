# SCRIBE paper results

Written by `scripts/scribe_eval/report.py` on 2026-10-27 23:55 (197 s). Inputs under SP = `/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad`. Statistics: `scripts/scribe_eval/meta.py`, 1000 bootstrap resamples, seed 20261028. Each analysis has a Markdown file, a CSV of every table (column `table` names the table) and a JSON file with the same tables. Rerun the script as inputs land; analyses listed under Skipped run once their inputs exist.

## Outputs

| analysis | section | what | units | metrics | not available yet |
|---|---|---|---|---|---|
| [wiki_fresh_scribe](wiki_fresh_scribe.md) | 1 | SCRIBE v2, v1, surface S and the v2 deviance parts; 500 texts per rung | 4746 | 9 | — |
| [wiki_fresh_W_all](wiki_fresh_W_all.md) | 1 | All per-text metrics on subset W (150 texts per rung) | 1650 | 18 | R/wiki_fresh_W/ppl_gpt2.jsonl (partial), R/wiki_fresh_W/detector.jsonl (partial) |
| [wiki_fresh_trie](wiki_fresh_trie.md) | 1 | Where the order-4 trie falls: P(trie > control rung), SCRIBE on all texts | 4746 | 9 | — |
| [wiki_fresh_W_trie](wiki_fresh_W_trie.md) | 1 | Where the trie falls, all per-text metrics on W | 1650 | 18 | R/wiki_fresh_W/ppl_gpt2.jsonl (partial), R/wiki_fresh_W/detector.jsonl (partial) |
| [genre_web](genre_web.md) | 2 | Web ladder, each genre with its own pack and reference, SCRIBE only | 3500 | 3 | — |
| [genre_books](genre_books.md) | 2 | Books ladder, each genre with its own pack and reference, SCRIBE only | 3500 | 3 | — |
| [dose_scribe](dose_scribe.md) | 3 | SCRIBE on all corruptions (9 families x 6 strengths x 200 texts) | 10800 | 3 | — |
| [dose_S20_all](dose_S20_all.md) | 3 | All per-text metrics on S20 (every family and strength of the first 20 sources) | 1080 | 14 | R/corrupt_S20/ppl_gpt2.jsonl, R/corrupt_S20/cola.jsonl, R/corrupt_S20/coherence.jsonl, R/corrupt_S20/detector.jsonl |
| [mauve_he_scribe](mauve_he_scribe.md) | 4 | SCRIBE on all 3,240 judgments | 6116 | 5 | — |
| [mauve_he_scribe_te25](mauve_he_scribe_te25.md) | 4 | SCRIBE, judgments with an estimated answer time over 25 s only (the MAUVE paper's filter; times from the raw CSV) | 6116 | 5 | — |
| [mauve_he_M300_all](mauve_he_M300_all.md) | 4 | All per-text metrics on M300 (first 300 pairs) | 596 | 16 | R/mauve_he_M300/ppl_gpt2.jsonl, R/mauve_he_M300/detector.jsonl |
| [mauve_he_M_all](mauve_he_M_all.md) | 4 | All per-text metrics on M (first 1,000 pairs; only stats under the plan in force) | 1962 | 16 | — |
| [hanna_corr](hanna_corr.md) | 5 | HANNA, 6 criteria, with and without human stories: SCRIBE, our baselines, 13 shipped automatic metrics and 4 shipped LLM judges | 1056 | 33 | R/hanna/ppl_gpt2.jsonl |
| [hanna_corr_full](hanna_corr_full.md) | 5 | HANNA, SCRIBE against all 72 shipped automatic metrics | 1056 | 75 | — |
| [mans_roc_corr](mans_roc_corr.md) | 5 | OpenMEVA MANS ROC, overall quality (5 ratings per story) | 1000 | 16 | R/mans_roc/ppl_gpt2.jsonl |
| [mans_wp_corr](mans_wp_corr.md) | 5 | OpenMEVA MANS WP, overall quality (5 ratings per story) | 1000 | 16 | R/mans_wp/ppl_gpt2.jsonl |
| [gpt2out_pure_scribe](gpt2out_pure_scribe.md) | 6 | GPT-2 output dataset, pure sampling: human > xl > large > medium > small, SCRIBE on 1,000 texts per system | 9000 | 4 | — |
| [gpt2out_pure_200_all](gpt2out_pure_200_all.md) | 6 | GPT-2 output dataset, pure sampling, all per-text metrics on gpt2out_200 | 1800 | 15 | R/gpt2out_200/ppl_gpt2.jsonl |
| [gpt2out_pure_500_all](gpt2out_pure_500_all.md) | 6 | GPT-2 output dataset, pure sampling, per-text metrics on gpt2out_500 (stats only under the plan in force) | 4500 | 15 | — |
| [gpt2out_k40_scribe](gpt2out_k40_scribe.md) | 6 | GPT-2 output dataset, top-k 40: human > xl > large > medium > small, SCRIBE on 1,000 texts per system | 9000 | 4 | — |
| [gpt2out_k40_200_all](gpt2out_k40_200_all.md) | 6 | GPT-2 output dataset, top-k 40, all per-text metrics on gpt2out_200 | 1800 | 15 | R/gpt2out_200/ppl_gpt2.jsonl |
| [gpt2out_k40_500_all](gpt2out_k40_500_all.md) | 6 | GPT-2 output dataset, top-k 40, per-text metrics on gpt2out_500 (stats only under the plan in force) | 4500 | 15 | — |
| [ngram_ladder](ngram_ladder.md) | 6 | KN word n-grams: human > 5 > 4 > 3 > 2 > 1 | 1491 | 3 | — |
| [cyphalm_cls](cyphalm_cls.md) | 6 | CyphaLM: human, trie, trie + class expert (no known order: means and pairwise accuracies) | 493 | 4 | — |
| [attacks](attacks.md) | 7 | Mean per attack and parity with human text (mean / mean of human); P(human > attack) | 2500 | 5 | — |
| [attacks_50_all](attacks_50_all.md) | 7 | The same on attacks_50 (first 50 prompts) with every baseline | 500 | 16 | R/attacks_50/ppl_gpt2.jsonl, R/attacks_50/detector.jsonl |
| [length_wiki_scribe](length_wiki_scribe.md) | 8 | Held-out human text, standard (human) and short (human_short) buckets and pooled, SCRIBE | 4746 | 3 | — |
| [length_wiki_W_all](length_wiki_W_all.md) | 8 | Held-out human text on W (150 per bucket), all per-text metrics | 1650 | 20 | — |
| [qa_ladder](qa_ladder.md) | 9 | correct > other answer > prompt echo > shuffled words | 2400 | 4 | — |
| [profiling](profiling.md) | 10 | Ablation ladder table (as explore/abl.py), TIMING and peak RAM of SCRIBE, throughput of every baseline |  |  | — |

## Skipped

| analysis | section | what | why |
|---|---|---|---|
| wiki_fresh_W60_judge | 1 | The LLM judge and every other metric on W60 (60 texts per rung) | skipped: missing: R/wiki_fresh_W60/judge_*.jsonl |
| wiki_fresh_W_setlevel | 1 | Set-level metrics on W, one set per rung (q: 100 texts per rung; MAUVE with GPT-2 small features against 100 other human texts) | skipped: missing: SP/run/baselines/wiki_fresh_W_mauve/setlevel.jsonl |
| wiki_fresh_setlevel | 1 | Set-level metrics, one set of 250 texts per rung (MAUVE with GPT-2-large features, reference: 250 other human texts) | skipped: missing: SP/run/baselines/wiki_fresh_mauve/setlevel.jsonl |
| wiki_fresh_setlevel_K5 | 1 | Set-level metrics on 5 disjoint subsets of 50 texts per rung | skipped: missing: SP/run/baselines/wiki_fresh_mauve/setlevel_K5.jsonl |
| dose_S40_all | 3 | All per-text metrics on S40 (earlier plan; runs if its files appear) | skipped: no baseline output yet: R/corrupt_S40/* |
| dose_S10_judge | 3 | The LLM judge and every other metric on S10 (first 10 sources) | skipped: missing: R/corrupt_S10/judge_*.jsonl |
| mauve_he_M300_judge | 4 | The LLM judge and every other metric on M300 (first 300 pairs) | skipped: missing: R/mauve_he_M300/judge_*.jsonl |
| gpt2out_pure_setlevel | 6 | GPT-2 output dataset, pure sampling, set-level metrics (one set per system: self-BLEU and Zipf on gpt2out_200; MAUVE with GPT-2 small features if run) | skipped: missing: SP/run/baselines/gpt2out_200/setlevel.jsonl, SP/run/baselines/gpt2out_mauve/setlevel.jsonl |
| gpt2out_k40_setlevel | 6 | GPT-2 output dataset, top-k 40, set-level metrics (one set per system: self-BLEU and Zipf on gpt2out_200; MAUVE with GPT-2 small features if run) | skipped: missing: SP/run/baselines/gpt2out_200/setlevel.jsonl, SP/run/baselines/gpt2out_mauve/setlevel.jsonl |
| pythia_steps | 6 | Pythia-410M training checkpoints: later > earlier | skipped: missing: SP/run/models/pythia_steps.jsonl, SP/run/models/pythia_steps.scribe.jsonl |
| pythia_size | 6 | Pythia sizes: larger > smaller | skipped: missing: SP/run/models/pythia_size.jsonl, SP/run/models/pythia_size.scribe.jsonl |
| gpt2_decoding | 6 | GPT-2-large decoding (no known order): means, pairwise accuracies and the share of texts flagged degenerate (D_I > 0) | skipped: missing: SP/run/models/gpt2_decoding.jsonl, SP/run/models/gpt2_decoding.scribe.jsonl |

## Checks against known numbers

| analysis | metric | pair | expected | got | match |
|---|---|---|---|---|---|
| wiki_fresh_scribe | scribe_v2 | human > sent_shuffled | 0.975 | 0.975 | yes |
| wiki_fresh_scribe | scribe_v2 | sent_shuffled > word_shuffled | 0.999 | 0.999 | yes |
| wiki_fresh_scribe | scribe_v2 | word_shuffled > random_lexicon | 0.998 | 0.998 | yes |
| wiki_fresh_scribe | scribe_v2 | random_lexicon > random_letters | 0.998 | 0.998 | yes |
| wiki_fresh_scribe | scribe_v2 | random_letters > repeated_phrase | 1.000 | 1.000 | yes |
| wiki_fresh_scribe | surface_s | human > sent_shuffled | 0.627 | 0.627 | yes |
| wiki_fresh_scribe | surface_s | sent_shuffled > word_shuffled | 0.341 | 0.341 | yes |
| wiki_fresh_scribe | surface_s | word_shuffled > random_lexicon | 0.750 | 0.750 | yes |
| wiki_fresh_scribe | surface_s | random_lexicon > random_letters | 1.000 | 1.000 | yes |
| wiki_fresh_scribe | surface_s | random_letters > repeated_phrase | 0.559 | 0.559 | yes |
