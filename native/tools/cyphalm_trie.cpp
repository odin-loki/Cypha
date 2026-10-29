// cyphalm_trie: the one tool for CyphaLM's packed byte trie (PLAN Track B1).
//
// Replaces ten tools: trie_fold_lm, trie_online_mix (-> bpc), trie_chat (-> chat),
// trie_qa_probe (-> probe), trie_context_8k (-> context8k), trie_sample (-> sample),
// and the experiments trie_together, trie_adaptive, trie_gria_gate and
// trie_online_holdout (retired; results and recovery commits are in
// docs/reports/B1_TRIE_CONSOLIDATION.md). The model itself is
// native/include/cypha/cyphalm/packed_trie.hpp (namespace cypha::cyphalm::trie).
// trie_stream_train.cpp still carries its own copy of the model.
// After training, every subcommand serves from PackedModel (packed_trie.hpp; docs/reports/RAM_PROFILE.md M4),
// which gives the training Model's Synth distributions bit for bit, and frees the training form (bpc keeps it
// for --negative). InfiniGram is FmInfiniGram (fm_infinigram.hpp, M5: the same answers from a compressed index),
// queried with a cursor where text is read one byte at a time; it takes its training text (M6, M9).
//
// Backoff orders 2..4 as a byte trie (alpha 0.4) + a 4-byte match, mixed by six
// experts (unigram, order 1, orders 2, 3, 4, match) whose weights learn on the
// training stream and are then frozen. Train [0, 7,340,032), score the frozen
// holdout [7,340,032, 8,388,608) of the first 8 MiB of enwik9. enwik8's first
// 8 MiB are the same bytes. CyphaLM answers every task: there is no string-search
// answering here (PLAN rule 10).
//
// The similar-context expert (A4, similar_context.hpp; docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md "A4b") is a default in
// qa, sample and context8k: the spec `default` (W=2, equal word weights, 2048 lists, keys over wiki train; A4c:
// distinct keys stored once, the k=8192 nearest each pooling its occurrences; A4d: 128 lists probed, and keys from the
// text being read too) with the projected PPMI-SVD file data/cyphalm/simvec_ppmi32.bin, read from $CYPHA_SIMVEC, else from
// CYPHA_SIMVEC_DEFAULT (a compile-time path), else from that path under the current directory. It is off inside every
// answer (in_answer), so answers are written without it. --no-sim leaves it out; without the file those subcommands
// stop and say so. bpc, chat and probe never build it: bpc's lines are the trie's reference numbers, and chat and
// probe write only answers.
// B4C (docs/reports/B4C_ECHO_GATE.md): the class-context expert (`shape` = shape:6:3:cp2, or --cls SPEC) and the copy
// expert (m = --copy-min, default 8; a 16,384-byte window) are defaults in qa and context8k, in the order class,
// similar-context, copy; in sample only the copy expert is (SCRIBE v2 falls with the class expert; --cls adds it). Like
// the similar-context expert they are off inside every answer, so answers are written without them. --no-cls and
// --no-copy leave them out; with --no-cls --no-copy --no-sim each subcommand prints what it printed before them.
// BKN2 (docs/reports/BYTE_KN.md): packed byte Kneser-Ney (byte_kn.hpp; spec 6:4:2:1, way b: mixed into the trie
// mixture by lambda per bucket, before InfiniGram) is a default in qa, sample and context8k, built on each one's
// training bytes. Unlike the experts above it is on inside answers: it is part of the byte model. --no-bkn leaves it
// out, and each subcommand then prints what it printed before it; --bkn SPEC and --bkn-way a|b choose another.
// BKN3: --bkn-answers off turns it off inside answers (in_answer, the experts' gate) in qa and context8k, where
// InfiniGram then uses lambdas fit without it, so the answers are those of the model without Kneser-Ney.
//
// Subcommands (stdout carries the numbers, stderr the progress):
//   bpc [wiki] [--negative] [--no-guard]
//        fold_lm + online_mix. Prints the context counts, hash accounting, the
//        full / drop_ctx / synth stored forms (synth_stored is the packed layout),
//        synth_bpc (fixed backoff), backoff_bpc, mix_bpc (learned, frozen), the six
//        weights, mix_stored. --negative adds the PPMD / chains / inheritance rows
//        (CURRENT.md "~2.42, worse than backoff"). The guard stops the run if the
//        backoff is outside 2.36..2.41 on the default split.
//   chat [wiki] [dolly] [--prompt TEXT] [--no-repeat] [--save FILE | --model FILE]
//        trie_chat: train on wiki, then two passes over Dolly after a <|doc|> break;
//        prints the wiki BPC after chat, the weights, mix_stored and one sample after
//        a question the chat never saw.
//   probe [wiki] [dolly]
//        trie_qa_probe: what the mixture writes after a seen / first-letter-changed /
//        Wikipedia-fact question (greedy and nucleus 0.9, T = 0.8).
//   context8k [wiki] [dolly]
//        trie_context_8k: InfiniGram (order up to 8192) mixed with the trie, BPC at five
//        caps, matched-length histogram, 50 questions with 30 Dolly turns held out. A4: trie_plus_context_8192_sim_bpc
//        adds the similar-context expert (lambdas on the first half; --no-sim leaves the line out). B4C: the class
//        expert (on InfiniGram's bytes) and the copy expert add lines too, each name the chain so far
//        (trie_plus_context_8192_cls_bpc, ..._cls_sim_bpc, ..._cls_sim_copy_bpc), each expert's lambdas fit on the
//        first half over the distribution it is mixed into.
//   sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600] [--save FILE | --model FILE]
//        trie_sample: wiki-only trie, 3 temperatures, no-repeat rule, files for `scribe ladder --gen-trie`.
//        A4: with the similar-context expert (its lambdas over the trie on the first half of the holdout) the files
//        start with a MODEL line and keep the GEN kinds; --no-sim writes the trie alone (B1's files). B4C: the copy
//        expert joins it, fit over trie + sim (--no-copy --no-sim writes B1's files). The class expert is not a default
//        here, because SCRIBE v2 falls with it; --cls SPEC adds it (built on the train bytes, before the others).
//        GEN-A: --prompts-file F (bench/eval/gen_prompts_{dev,test}.tsv; one out file) takes the prompts from F in
//        its order, and after sampling prints GEN_EVAL lines: the BPC of each human continuation under sampling's
//        mixture (written per row to OUT.cont.tsv) and the share of generated bytes in 32-byte spans that occur in
//        the training bytes, raw or cleaned (the same reference with --clean). scripts/gen_eval.sh runs it with SCRIBE v2 (docs/reports/GEN_TRACK.md). --clean trains
//        on SCRIBE's cleaned prose of [0, train) (clean_wiki_documents, a page per line) and fits the lambdas and draws
//        holdout prompts on the cleaned [train, sample).
//        GEN-E: --plan STOREDIR[,RAWFRAMES] adds the content-plan expert (content_plan.hpp,
//        docs/reports/GEN_E_CONTENT_PLAN.md), mixed last: the store (pattern_learn's v2 files) plans the facts of the
//        prompt's entity, and the expert follows their realisations (F2's frames, filled). Facts read from the holdout
//        are dropped, and from --plan-drop LO:HI (raw bytes; the benchmark's source range for --prompts-file). The
//        files gain PLAN lines (the plan of each prompt; SCRIBE reads GEN lines only). Off by default.
//        chat and sample: --save FILE stores the trained packed trie with its weights and split (RAM_PROFILE M7);
//        --model FILE serves that file without training (chat then prints no mix_stored: it measures the training form).
//   bkn <wiki> K[:m:c:d] [more specs] [--bkn-fresh] [--bkn-stack]
//        BKN (docs/reports/BYTE_KN.md): packed byte Kneser-Ney (byte_kn.hpp) replacing or added to the trie, on cls's
//        protocol (holdout2; fresh with --bkn-fresh; the defaults on top with --bkn-stack). See cmd_bkn.
//   cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs] [--sim SPEC ...] [--sim-fresh] [--sim-surface]
//        B4 (docs/reports/B4_CLASS_CONTEXT_EXPERT.md): the class-context expert (class_context.hpp) over the
//        bpc model, each spec paired against the model without it: holdout second-half BPC (lambdas on the
//        first half), fresh-text BPC on [sample, sample + 1 MiB) (lambdas on the whole holdout), and surface S
//        of 400-byte samples from 200 holdout prompts, with block-bootstrap intervals; later specs are also
//        compared with the first. TABLE is word<TAB>class (e.g. E1 tags); "shape" is shape classes only. The
//        default table is three orders, compact, contexts seen once dropped (shape = shape:6:3:cp2, Odin
//        2026-10-28); B4's two-order full table, which its report's numbers come from, is shape:6:2:f.
//        A4 (docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md): --sim adds the similar-context expert (similar_context.hpp) instead
//        of the class loop: each SPEC over trie + InfiniGram and over trie + InfiniGram + the first class spec, holdout
//        second half (lambdas on the first), the index's recall against an exact search, build time, table bytes and
//        query latency; --sim-fresh adds the fresh text (lambdas on the whole holdout), --sim-surface surface S of
//        trie + sim and trie + class + sim. SPEC is vec=PREFIX (PREFIX.vectors.f32 and PREFIX.vocab.tsv, as ppmi_svd
//        writes them) or a projected file (simvec), plus W, k, T, min, sif, dim, drop, lists, probe, tail, src, gate (see
//        parse_sim); the token `default` is the default spec.
//        A4e: --sim-stream FROM:LEN reads the corpus bytes [FROM, FROM + LEN) after the holdout in place of the fresh
//        MiB (implies --sim-fresh) and adds `simstream` lines: the expert at each MiB of the stream (online keys held,
//        their MB, search p50/p99, scoring time) and the continuation's BPC by MiB; --sim-bits PREFIX writes each
//        spec's continuation bits for `simpair`; --sim-growth 1,4,16 reads only the expert (no trie, InfiniGram or class)
//        and reports its online keys' memory and search time at those stream lengths in MiB.
//        GEN-C2: --doc V1;V2;... reads each variant of the document experts (GEN-C's --doc specs) over qa's default
//        chain (trie + InfiniGram, class, similar-context, copy): the holdout's second half (lambdas on its first half),
//        and with --sim-stream FROM:LEN the stream after the holdout (lambdas on the whole holdout). `doc2` lines.
//   simpair [--margin M] REF.f64 FILE.f64 ...  (A4e) pairs --sim-bits files against REF: whole continuation, each MiB,
//        last 2 MiB; with a margin also P(change < M).
//        --out FILE writes the samples (systems trie, trie+cls:SPEC) and the human continuations (human) as JSON
//        lines for the SCRIBE model ladder.
//   qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--leak TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]
//      [--dk DK.tsv --dk-ratio R] [--fam none|before|after] [--parse E1.tsv] [--fam-stat NAME]
//      [--fam-train stat|outcome|low] [--fam-only] [--copy] [--copy-min N] [--r3 PATTERNS.tsv]
//      [--fam-ent FRAMES] [--copy-dk] [--fam-read] [--shard F.tsv] [--shard2 F.tsv] [--cls TABLE|shape[:tail[:orders[:flags]]]] [--sim SPEC] [--cards FILE]
//      [--dk-plain P] [--fam-ent-at all|answer|start] [--copy-frame]
//        A0/A1 (docs/CYPHALM_ANSWERING.md §4; TOOL_PROFILE_PLAN §2): trains as chat (wiki, then
//        two Dolly passes), optionally with don't-know turns spread at share R of all turns and a
//        familiarity token <|fam=low|mid|high|> before (or after) "Assistant: ". CyphaLM answers
//        each T-QA question greedily (no-repeat rule; stop at newline / sentence end; 80 bytes;
//        a don't-know answer runs on to the end of its tag) in two prompt forms, chat
//        "User: <q>\nAssistant: " and wiki "<q> ", with the trie mixture and with trie +
//        InfiniGram (context8k lambdas, cap 8192). Confidence is CyphaLM's own: answer mean and
//        min log2 p, first-byte entropy, InfiniGram match length, question log2 p. Grades right /
//        abstained / wrong (T-QA normalised containment; the message or tag = abstained).
//        Abstain gate for wrong <= 2% by 5-fold CV in dev; risk-coverage curves; A1 abstain
//        precision / recall and reason codes. --test applies the full-dev gate to the frozen test.
//        A1c / A2 / A3 (docs/reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md, A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md):
//        --parse reads E1's question parse (pos_tagger question --tsv) for the entity. With it, dev AUROC of
//        seven familiarity statistics (question log2 p; the entity's count, link, bold-lead and title counts
//        in a suffix array over the lowered training bytes; its longest piece found; its trie log2 p), cuts
//        fit on training-side entities only (wiki-train titles against the --dk entities), and --fam-stat
//        picks the token's statistic (--fam-train outcome: training turns carry the decision instead).
//        The copy expert (B4C, a default): a match over the current context, lambdas per match length fit on the
//        holdout's first half (BPC and LCI gain on the second half), off inside answers. --copy adds two more models
//        whose copy expert is also on inside answers (A1c's answer copy, for comparison; with the same lambdas, so
//        --copy-min 0 gives A1c's m 0). --no-copy leaves the default copy out. --r3 runs statement
//        scoring with F2's patterns as frames. --shard / --shard2 add question-form fact turns (rows from the
//        BPC holdout dropped; exits if a shard question equals a dev or test question, from --test or --leak).
//        The class-context expert (B4; TOOL_PROFILE P2; a default since B4C, --cls SPEC sets its table, --no-cls
//        leaves it out), built on the same bytes as InfiniGram, is in every model and off inside answers (lambdas on
//        the holdout's first half); the copy expert goes on after it, its lambdas fit
//        over the class-mixed distribution, and cls_given_copy_lci gives the class expert's gain with copy present.
//        --copy-min N (B4B) sets the copy lambdas of matches shorter than N bytes to 0 after fitting.
//        A1d / A2b (docs/reports/A1D_A2B_ANSWERING_VARIANTS.md): the --parse entity column may hold several
//        candidates ("a|b|c", E1's first): per statistic the best candidate's value counts, and the token's entity is
//        the best by --fam-stat (ENTITY: its accuracy against the source title, grading only). ent_tb is the larger of
//        the bold-lead and title counts; FAMSTAT adds the paired-bootstrap difference from ent_bold, FAMTOK the
//        token's implied precision / recall. --fam-train low is outcome, but answered turns and high questions carry
//        no token. --r3 reads a 12th column `frame` (F2R's raw-byte frames, markup kept) when present.
//        ANS (the same report, "ANS"): --fam-ent FRAMES writes the entity into every token as frames before it
//        (e.g. entity="{E}" gives <|entity="Napoleon"|><|fam=low|>; a don't-know or shard row's own entity, else the
//        chosen span). --copy-dk turns the copy expert on inside a don't-know answer (once it begins "I don't know"),
//        in every model, with its own lambdas fit on the don't-know turns held out of training; ordinary answers and
//        text are unchanged. --fam-read prints FAMREAD (each dev question's best span and count per statistic) and
//        FAMDOLLY (the token each statistic gives the Dolly instructions), report only.
//        DK2 (docs/reports/DK2_DONT_KNOW_TRAINING.md): --dk-plain P writes share P of the don't-know training turns with
//        no token (turn j when floor((j+1)P) > floor(jP)), so CyphaLM's own "I don't know" is plain text, not an echoed
//        token. --fam-ent-at answer|start keeps --fam-ent's frames out of all training text: only prompts it answers (dev,
//        test, the held-out --copy-dk turns) carry them; answer puts them with the token, start before "User: " on every
//        question (high too). --copy-frame (with --copy-dk), inside a don't-know answer: the copy expert never
//        predicts the '|' of a source "|>" while the answer has no open "<|" (a frame's entity is copied without the
//        frame's end), and inside the open tag the no-repeat rule looks at the tag only (the tag may repeat the message).
//        A4: --sim SPEC adds the similar-context expert to every model, after the class expert and before the copy
//        expert (src=all: keys over InfiniGram's bytes, wiki train + chat; src=wiki: wiki train only), lambdas on the
//        holdout's first half; SIM_NB prints the three nearest stored contexts at each dev chat prompt, LATENCY the
//        decode time a question. The expert is a default here (spec `default`; --no-sim leaves it out); gate=0 in a
//        spec lets it into answers, for comparison only.
//        M11 (docs/reports/RAM_PROFILE.md): --save DIR stores the built model (packed trie, InfiniGram, the lowered
//        index, the class table and a record of the options, each file checked); --model DIR serves it without training
//        or building (the similar-context expert is built from its vector file). The build's calibration lines
//        (FAMSTAT, ENTITY, FAMCAL, FAMTOK, fam_dolly, chat_turns) print at build time only; every other line and the rows
//        file are the build's. A served model refuses other options or data, --fam-only, the token statistics that read
//        the wiki-stage trie (question_lp, ent_lp), --sim src=all, and --cards (S1's cards map T's bytes to the chat
//        turns, which a served model does not rebuild).
//        S1 (docs/reports/S1_REASON_CODES_AND_EVIDENCE.md): --cards FILE writes what the system shows for each answer
//        (a kept answer, or a don't-know with its reason code: CyphaLM's own, or the gate's on its behalf) with its
//        evidence (where in the training text its bytes, or the question's entity, were seen: FmInfiniGram locate),
//        REASON lines (reason codes against the reference, grading only) and FILE.audit.tsv (100 dev rows for Odin).
//        Made after the answers and gates; the rows file is the same with and without it.
//   simvec [wiki] --sim vec=PREFIX[,...] --out FILE
//        A4: builds the similar-context expert of the spec over the train bytes and writes its projected word rows (int8,
//        a scale a row) as the file the default reads.
//   dktext [wiki] [dolly] --dev DEV.tsv --test TEST.tsv --out DK.tsv [--from 8388608]
//        A1 training text: T-QA relation templates filled with page titles from enwik8 after
//        --from whose lowered text never occurs in the lowered training bytes (wiki train +
//        Dolly; exact suffix-array check, audited by a plain scan), paired with the DISCOVERY
//        §3.5 message and tag. Rejects any entity or answer variant found in dev or test, and
//        fails if the final leakage check or the audit finds anything. --mention also keeps titles that occur
//        in the training bytes only as a link ([[title): named, but with no article in the slice.
//   selftest
//        packed_trie.hpp self_test plus a golden BPC on a 40,000-byte built-in corpus, plus the
//        qa grading, tag parsing, Wilson bounds, abstain gate, copy expert, AUROC / Youden cut, frames,
//        candidates, statement choice, question relation, shard leakage check, class-context and similar-context experts
//        on fixed inputs
//        (CTest native_cyphalm_trie_selftest; needs no data files; about 0.1 second).
// Every subcommand takes --wiki P, --dolly P, --train N, --sample N in place of the
// positional paths; the paths default to $CYPHA_WIKI / $CYPHA_DOLLY, then to the paths
// the old tools used (kDefaultWiki, kDefaultDolly below).
//
// Measured 2026-10-27 on enwik8's first 8 MiB and Dolly 15,015 records. Every number line
// equals the old tool's output to the printed digit (old-vs-new table in
// docs/reports/B1_TRIE_CONSOLIDATION.md):
//   bpc        fixed backoff 2.386293 BPC; learned mixture, weights frozen, 2.297794
//              (w4 0.930724, match 0.061085); packed layout synth_stored 5,385,965 bytes
//              (full 9,112,775, drop_ctx 6,623,887); --negative: PPMD 2.415026,
//              + chains 2.417500, + inheritance 2.416818. About 100 s.
//   chat       wiki BPC after the Dolly passes 2.344571 (w4 0.960948, match 0.038609),
//              mix_stored 11,181,225. CURRENT.md's 2.341 was a 15,011-record Dolly copy. About 5 min.
//   probe      model_contains_expect 0 on all 14 questions. About 4 min.
//   context8k  trie only 2.3199 BPC; + InfiniGram 2.0637 (cap 16) to 2.0715 (cap 8192);
//              TALLY seen 3/3, change_first_letter 3/3, add_Thanks_at_end 0/3, paraphrase 0/3,
//              heldout_dolly 1/30, wiki_fact 0/8. About 4.5 min.
//   sample     two 250-prompt files, byte-identical to trie_sample's. About 1 min.
// Measured 2026-10-28 (docs/reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md):
//   qa (A0)    holdout BPC 2.344571 (= chat), trie 2.319989 / + InfiniGram 2.071574 (= context8k).
//              Dev 130: raw right 0 (trie) and 2 (+ InfiniGram, both luck); coverage at 2% wrong
//              0.0% (5-fold CV) in every form and model. Test 200: dev gate keeps 3, all wrong.
//   qa (A1)    20% don't-know turns, no token: abstains on 130/130 (trie), precision on
//              not-in-slice 0.231 = base rate. Token after "Assistant: ": the trie abstains on
//              exactly the `low` questions, dev precision 0.50 / recall 0.43, test 1.00 / 0.30.
//              BPC +0.0003 to +0.0033. About 6 min a run, peak RSS <= 265 MB.
//   dktext     3,963 rows from 3,274 absent entities; leakage 0; plain-scan audit 0/300. 6 s.
//              --mention: + 2,231 linked-only entities, 6,583 rows, leakage 0.
// Measured 2026-10-28 (docs/reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md, A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md):
//   qa A1c     dev AUROC (not-in-slice vs in-slice): ent_title 0.795 (boot95 0.745-0.847), ent_lp 0.771,
//              ent_bold 0.765, question_lp 0.693, ent_count 0.535, ent_link 0.517, ent_igfrac 0.495.
//              ent_title token, --fam-train outcome: every model abstains on exactly the low/mid
//              questions; precision / recall dev 0.435 / 1.00, test 0.371 / 0.98. On every turn
//              (stat): 97% of Dolly turns get low and the trie abstains on all 130 (0.231 / 1.00).
//   --copy     holdout BPC (second half) trie 2.319989 -> 2.113702, trie + InfiniGram 2.071574 ->
//              1.926101; LCI P_gain(> 0.005) 1.000 on 256 blocks. No tag names the question's entity.
//   --r3       runs on 25-32 dev questions, raw 1 right; 0 right at 2% wrong out of fold.
//   --shard    1,488 F2 fact rows (513 holdout rows dropped, leakage 0): dev raw right 2 -> 7
//              (trie + InfiniGram), 2 right 0 wrong at 2% wrong out of fold (1.5%); BPC +0.0014.
//              Test: covers 3 questions, adds 0; with --r3 the gate keeps 1 right, 0 wrong.
//              About 7.5-8.5 min a run.
// Measured 2026-10-28 (docs/reports/A1D_A2B_ANSWERING_VARIANTS.md):
//   qa A1d     span candidates (every 1-6-word span of the question), best by ent_title: entity = source title
//              76/100 in-slice (E1 63); dev AUROC ent_title 0.900 (boot95 0.862-0.940; E1 0.795), ent_bold 0.869.
//              Token (ent_title, --fam-train low): dev precision / recall 0.600 / 1.00 (trie), test 0.635 / 0.94
//              (A1c: 0.371 / 0.98); with D4's shard trie + InfiniGram keeps 2 right 0 wrong at 2% wrong out of
//              fold (a high token keeps none). stat: trie + IG 0.313 / 0.50. Entity in the token: 0 of 65 tags.
//   --cls + --copy  holdout2 trie + IG 2.071574, + shape:6:2:f 1.995847, + copy(m 8) 1.985451, both 1.914210;
//              with the shard, class + copy echoes the question (raw right 6 -> 2).
//   --r3 raw   F2R's frames (column 12) + raw-form shard: dev 5 raw right, 2 right 1 wrong out of fold; test 1 / 0.
//
// Build (from the repo root):
//   gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o
//   g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais
//       native/tools/cyphalm_trie.cpp native/src/cyphalm/fm_infinigram.cpp libsais.o -o cyphalm_trie
//   (one command line; the two lines above are split for width)
// Usage:
//   ./cyphalm_trie bpc /root/cdata/enwik8
//   ./cyphalm_trie chat /root/cdata/enwik8 /root/cdata/dolly.jsonl
//   ./cyphalm_trie sample /root/cdata/enwik8 gen_dev.txt gen_fresh.txt
//   ./cyphalm_trie selftest

#include "cypha/cyphalm/byte_kn.hpp"
#include "cypha/cyphalm/class_context.hpp"
#include "cypha/cyphalm/content_plan.hpp"
#include "cypha/cyphalm/doc_cache.hpp"
#include "cypha/cyphalm/evidence.hpp"
#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/cyphalm/packed_trie.hpp"
#include "cypha/cyphalm/similar_context.hpp"
#include "cypha/cyphalm/token_model.hpp"  // BKN selftest: the reference Kneser-Ney
#include "cypha/eval/lci.hpp"
#include "cypha/eval/scribe.hpp"
#include "cypha/eval/unknown_signals.hpp"

#include <cctype>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>

using namespace cypha::cyphalm::trie;
using cypha::cyphalm::ClassContext;
using cypha::cyphalm::ContentPlanStore;
using cypha::cyphalm::DocTopic;
using cypha::cyphalm::FmInfiniGram;
using cypha::cyphalm::SimilarContext;
namespace evidence = cypha::cyphalm::evidence;

namespace {

// What the old tools opened. Q2 asks for no hardcoded paths in new code: these are only the
// fallbacks, behind the arguments and $CYPHA_WIKI / $CYPHA_DOLLY.
constexpr const char* kDefaultWiki = "/mnt/d/cypha_llm/enwik9";
constexpr const char* kDefaultDolly = "/mnt/d/cypha_llm/chat/dolly.jsonl";
constexpr std::uint32_t kSeed = 42;
constexpr const char* kUnseenPrompt = "User: What colour is Cypha's packed trie?\nAssistant: ";

struct Args {
    std::vector<std::string> pos;
    std::string wiki, dolly, prompt = kUnseenPrompt;
    Config cfg;
    int prompts = 250, bytes = 600;
    bool negative = false, no_guard = false, no_repeat = false;
    bool ok = true;
    std::vector<std::string> outs;  // sample: the output files
    // qa / dktext
    std::string dev, test, out, base, dk, fam = "none";
    double dk_ratio = 0;
    std::size_t from = 8'388'608, max_rows = 0;
    std::string parse, fam_stat = "question_lp", fam_train = "stat", r3, shard, shard2, leak;  // A1c, A3, A2
    bool copy = false, fam_only = false, mention = false;
    std::string cls;      // B4: class-context expert spec, TABLE|shape[:tail[:orders[:flags]]] (make_class_context)
    int copy_min = 8;     // B4B: copy lambdas of match lengths below this set to 0 after fitting (B4C: 8, was 0)
    std::string model, save;  // M7: serve a stored packed trie (chat, sample); store the one just trained
    std::vector<std::string> sims;               // A4: similar-context expert specs (cls: each; qa: the first)
    bool sim_fresh = false, sim_surface = false;  // A4, cls: also the fresh text / surface S
    std::string sim_stream, sim_bits, sim_growth;  // A4e, cls: a longer stream after the holdout; per-byte bits; growth
    bool no_sim = false;                          // A4: the similar-context expert off where it is a default
    std::string cards;                            // S1: qa's evidence cards and reason codes (--cards FILE)
    bool no_cls = false, no_copy = false;         // B4C: the class and copy experts off where they are defaults
    std::string plant, part;                      // S4b surprise: the planted pages (D), --part K/N
    bool bkn_fresh = false, bkn_stack = false;    // BKN: also the fresh text / the defaults on top
    std::string bkn, bkn_way = "b";               // BKN: --bkn K[:m:c:d] in qa, sample, context8k; --bkn-way a (replace) | b (add)
    bool no_bkn = false;                          // BKN2: byte Kneser-Ney off where it is a default
    std::string bkn_answers = "on";               // BKN3: Kneser-Ney inside answers (on: BKN2) or off (the experts' gate)
    std::string plan;                             // GEN-E: sample's content-plan expert, --plan STOREDIR[,RAWFRAMES]
    std::string plan_drop;                        // GEN-E: --plan-drop LO:HI, the prompts' source bytes (facts there dropped)
    std::string prompts_file;                     // GEN-A: sample's prompts and human continuations
    bool clean = false;                           // GEN-A: sample trains on SCRIBE's cleaned prose of its bytes
    std::string fam_ent;                          // ANS: the entity's frames before the token (--fam-ent)
    bool copy_dk = false, fam_read = false;       // ANS: copy inside don't-know answers; FAMREAD / FAMDOLLY
    std::string doc;                              // GEN-C: document experts in sample (--doc SPEC; off by default)
    bool no_doc = false;                          // GEN-C2: the document experts' cache off in context8k and qa's holdout
    double dk_plain = 0;                          // DK2: share of don't-know training turns with no token
    std::string fam_ent_at = "all";               // DK2: where --fam-ent's frames go (all | answer | start)
    bool copy_frame = false;                      // DK2: copy a frame's entity, not its end; the tag may repeat the message
    std::string gate;                             // GEN-B: sample's sentence gate, K[,wf=..][,wp=..][,wm=..][,min=..][,max=..]
    std::string temps;                            // TEMPS: sample's temperatures, T[,T][,T] (default 0.6,0.8,1.0)
};

std::string env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

Args parse(int argc, char** argv, int from) {
    Args a;
    std::string wiki_flag, dolly_flag;
    for (int i = from; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : (a.ok = false, ""); };
        if (s == "--wiki") wiki_flag = next();
        else if (s == "--dolly") dolly_flag = next();
        else if (s == "--prompt") a.prompt = next();
        else if (s == "--train") a.cfg.train = std::strtoull(next(), nullptr, 10);
        else if (s == "--sample") a.cfg.sample = std::strtoull(next(), nullptr, 10);
        else if (s == "--prompts") a.prompts = std::atoi(next());
        else if (s == "--bytes") a.bytes = std::atoi(next());
        else if (s == "--negative") a.negative = true;
        else if (s == "--no-guard") a.no_guard = true;
        else if (s == "--no-repeat") a.no_repeat = true;
        else if (s == "--dev") a.dev = next();
        else if (s == "--test") a.test = next();
        else if (s == "--out") a.out = next();
        else if (s == "--base") a.base = next();
        else if (s == "--dk") a.dk = next();
        else if (s == "--dk-ratio") a.dk_ratio = std::atof(next());
        else if (s == "--fam") a.fam = next();
        else if (s == "--from") a.from = std::strtoull(next(), nullptr, 10);
        else if (s == "--max-rows") a.max_rows = std::strtoull(next(), nullptr, 10);
        else if (s == "--parse") a.parse = next();
        else if (s == "--fam-stat") a.fam_stat = next();
        else if (s == "--fam-train") a.fam_train = next();
        else if (s == "--r3") a.r3 = next();
        else if (s == "--shard") a.shard = next();
        else if (s == "--shard2") a.shard2 = next();
        else if (s == "--leak") a.leak = next();
        else if (s == "--copy") a.copy = true;
        else if (s == "--cls") a.cls = next();
        else if (s == "--copy-min") a.copy_min = std::atoi(next());
        else if (s == "--sim") a.sims.push_back(next());
        else if (s == "--sim-fresh") a.sim_fresh = true;
        else if (s == "--bkn-fresh") a.bkn_fresh = true;  // BKN
        else if (s == "--bkn") a.bkn = next();             // BKN
        else if (s == "--bkn-way") a.bkn_way = next();     // BKN
        else if (s == "--bkn-stack") a.bkn_stack = true;  // BKN
        else if (s == "--no-bkn") a.no_bkn = true;        // BKN2
        else if (s == "--bkn-answers") a.bkn_answers = next();  // BKN3
        else if (s == "--sim-surface") a.sim_surface = true;
        else if (s == "--sim-stream") a.sim_stream = next();  // A4e
        else if (s == "--sim-bits") a.sim_bits = next();
        else if (s == "--sim-growth") a.sim_growth = next();
        else if (s == "--no-sim") a.no_sim = true;
        else if (s == "--no-cls") a.no_cls = true;
        else if (s == "--no-copy") a.no_copy = true;
        else if (s == "--gate") a.gate = next();
        else if (s == "--temps") a.temps = next();  // TEMPS
        else if (s == "--fam-only") a.fam_only = true;
        else if (s == "--model") a.model = next();
        else if (s == "--plan") a.plan = next();
        else if (s == "--plan-drop") a.plan_drop = next();
        else if (s == "--save") a.save = next();
        else if (s == "--mention") a.mention = true;
        else if (s == "--cards") a.cards = next();
        else if (s == "--plant") a.plant = next();
        else if (s == "--part") a.part = next();
        else if (s == "--prompts-file") a.prompts_file = next();
        else if (s == "--clean") a.clean = true;
        else if (s == "--fam-ent") a.fam_ent = next();
        else if (s == "--copy-dk") a.copy_dk = true;
        else if (s == "--fam-read") a.fam_read = true;
        else if (s == "--doc") a.doc = next();
        else if (s == "--no-doc") a.no_doc = true;  // GEN-C2
        else if (s == "--dk-plain") a.dk_plain = std::atof(next());  // DK2
        else if (s == "--fam-ent-at") a.fam_ent_at = next();         // DK2
        else if (s == "--copy-frame") a.copy_frame = true;           // DK2
        else if (s.rfind("--", 0) == 0) {
            std::cerr << "unknown option " << s << "\n";
            a.ok = false;
        } else a.pos.push_back(s);
    }
    // sample <wiki> <out_dev> [out_fresh]: the wiki is positional unless --wiki was given.
    a.outs = a.pos;
    if (wiki_flag.empty() && !a.outs.empty()) a.outs.erase(a.outs.begin());
    a.wiki = !wiki_flag.empty() ? wiki_flag : (!a.pos.empty() ? a.pos[0] : env_or("CYPHA_WIKI", kDefaultWiki));
    a.dolly = !dolly_flag.empty() ? dolly_flag : (a.pos.size() > 1 ? a.pos[1] : env_or("CYPHA_DOLLY", kDefaultDolly));
    if (a.cfg.train < 5 || a.cfg.sample <= a.cfg.train) {
        std::cerr << "need 5 <= train < sample\n";
        a.ok = false;
    }
    return a;
}

bool load_wiki(const Args& a, std::vector<std::uint8_t>& data) {
    const std::string err = read_prefix(a.wiki, a.cfg.sample, data);
    if (!err.empty()) std::cerr << err << "\n";
    return err.empty();
}

// The training text of InfiniGram and the class expert: wiki train + the <|doc|> boundary + chat, never the wiki
// holdout. The chat bytes then leave data, which keeps [0, sample) for the holdout contexts (the packed trie never
// reads the training text). Returns data's size before: a follower limit that admits every follower.
std::size_t training_text(std::vector<std::uint8_t>& data, const Config& cfg, std::vector<std::uint8_t>& T) {
    const std::size_t all = data.size();
    T.reserve(cfg.train + (all - cfg.sample));
    T.assign(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.train));
    T.insert(T.end(), data.begin() + static_cast<std::ptrdiff_t>(cfg.sample), data.end());
    std::vector<std::uint8_t>(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.sample)).swap(data);
    return all;
}

// M7: --save FILE stores the packed trie just trained; --model FILE serves one without training. The file holds the
// mixture weights and the split, so cfg's train and sample come from it, and only the wiki text is read.
constexpr std::size_t kAllFollowers = std::numeric_limits<std::size_t>::max();

bool save_model(const std::string& path, const PackedModel& pm, const double w[6], const Config& cfg) {
    try {
        pm.save(path, w, cfg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return false;
    }
    return true;
}

bool load_model(const Args& a, PackedModel& pm, double w[6], Config& cfg, std::vector<std::uint8_t>& data) {
    try {
        pm = PackedModel::load(a.model, w, cfg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return false;
    }
    std::cerr << "served from " << a.model << " (" << pm.bytes() << " bytes), no training\n";
    const std::string err = read_prefix(a.wiki, cfg.sample, data);
    if (!err.empty()) std::cerr << err << "\n";
    return err.empty();
}

std::string lowered(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// ---------------------------------------------------------------- bpc

void dump(const char* name, const Layout& L) {
    std::cout << name << " nodes " << L.nodes << " path_only " << L.path_only << " one_child " << L.one_child
              << " leaves_dropped " << L.leaves_dropped << " followers " << L.followers << " trie_bytes " << L.bytes
              << "\n";
}

int cmd_bpc(const Args& a) {
    const Config& cfg = a.cfg;
    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    TrainLog log;
    log.progress = true;
    train_span(model, data, w, 0, cfg.train, cfg, &log);
    model.compact_match();
    std::cerr << "train_match_rate " << (static_cast<double>(log.matches) / static_cast<double>(cfg.train)) << "\n";

    const TrieStats ts = trie_stats(model);
    const Layout full = measure(model, Mode::Full);
    const Layout drop = measure(model, Mode::DropCtx);
    const Layout synth = measure(model, Mode::Synth);
    const std::uint64_t o1_bytes = order1_bytes(model);
    const std::uint64_t mt_bytes = match_bytes(model);
    const std::uint64_t build_bytes = model.nodes.capacity() * sizeof(Node) + model.fols.capacity() * sizeof(Fol);
    const std::uint64_t mix_stored = synth.bytes + o1_bytes + mt_bytes + 6ull * sizeof(double);
    // M4: the packed form scores. The training form is freed unless --negative still reads it.
    const PackedModel pm = pack(model);
    const std::uint64_t h_res = hash_reserved(model), h_pay = hash_payload(model);
    if (!a.negative) model.release();

    std::cerr << "scoring synth\n";
    const Score s_synth = heldout(pm, data.data(), Mode::Synth, cfg);
    std::cerr << "scoring mixture\n";
    const Holdout h = score_holdout(pm, data.data(), w, cfg);
    if (!h.ok) return 1;

    Score s_ppmd, s_chain, s_inherit;
    std::uint64_t deep = 0;
    Layout chain_layout, inherit_layout;
    if (a.negative) {
        std::cerr << "scoring ppmd\n";
        s_ppmd = heldout(model, data.data(), Mode::Synth, cfg, true, 4);
        std::cerr << "extending chains\n";
        model.extend_chains(cfg.train, 12);
        for (const Node& n : model.nodes)
            if (n.depth > 4 && n.total >= 2) ++deep;
        chain_layout = measure(model, Mode::Synth);
        std::cerr << "scoring ppmd+chain\n";
        s_chain = heldout(model, data.data(), Mode::Synth, cfg, true, 12);
        std::cerr << "inheriting\n";
        model.inherit_singletons();
        inherit_layout = measure(model, Mode::Synth);
        std::cerr << "scoring ppmd+chain+inherit\n";
        s_inherit = heldout(model, data.data(), Mode::Synth, cfg, true, 12);
    }

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "depth2 " << ts.by_depth[2] << " depth3 " << ts.by_depth[3] << " depth4 " << ts.by_depth[4] << "\n";
    std::cout << "contexts " << ts.ctx << " singletons " << ts.singletons << " singleton_frac "
              << (static_cast<double>(ts.singletons) / static_cast<double>(ts.ctx)) << "\n";
    std::cout << "order2_nodes " << ts.by_depth[2] << " mean_children "
              << (static_cast<double>(ts.sum_child) / static_cast<double>(ts.by_depth[2])) << " max_children "
              << ts.max_child << " one_child " << ts.one_child << "\n";
    std::cout << "hash_reserved " << (a.negative ? hash_reserved(model) : h_res) << " hash_payload "
              << (a.negative ? hash_payload(model) : h_pay) << "\n";
    std::cout << "build_heap " << build_bytes << " order1_bytes " << o1_bytes << " match_dense " << mt_bytes << "\n";
    dump("full", full);
    dump("drop_ctx", drop);
    dump("synth", synth);
    std::cout << "full_stored " << (full.bytes + o1_bytes + mt_bytes) << "\n";
    std::cout << "drop_ctx_stored " << (drop.bytes + o1_bytes + mt_bytes) << "\n";
    std::cout << "synth_stored " << (synth.bytes + o1_bytes + mt_bytes) << "\n";
    if (a.negative) {
        std::cout << "deep_nodes_total_ge2 " << deep << "\n";
        dump("chain", chain_layout);
        dump("inherit", inherit_layout);
        std::cout << "chain_stored " << (chain_layout.bytes + o1_bytes + mt_bytes) << "\n";
        std::cout << "inherit_stored " << (inherit_layout.bytes + o1_bytes + mt_bytes) << "\n";
    }
    std::cout << "synth_bpc " << s_synth.bpc << " mean_u " << s_synth.mean_u << " match_rate " << s_synth.match_rate
              << "\n";
    if (a.negative) {
        std::cout << "ppmd_bpc " << s_ppmd.bpc << " mean_u " << s_ppmd.mean_u << " match_rate " << s_ppmd.match_rate
                  << "\n";
        std::cout << "chain_bpc " << s_chain.bpc << " mean_u " << s_chain.mean_u << " match_rate " << s_chain.match_rate
                  << "\n";
        std::cout << "inherit_bpc " << s_inherit.bpc << " mean_u " << s_inherit.mean_u << " match_rate "
                  << s_inherit.match_rate << "\n";
    }

    const Config def;
    const bool default_split = cfg.train == def.train && cfg.sample == def.sample;
    if (default_split && !a.no_guard && !(h.backoff_bpc >= 2.36 && h.backoff_bpc <= 2.41)) {
        std::cout << "backoff_bpc " << h.backoff_bpc << "\n";
        std::cout << "copy is wrong\n";
        return 1;
    }
    std::cout << "backoff_bpc " << h.backoff_bpc << "\n";
    std::cout << "mix_bpc " << h.mix_bpc << "\n";
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";
    std::cout << "mix_stored " << mix_stored << "\n";
    std::cout << "mix_beats_backoff " << (h.mix_bpc < h.backoff_bpc ? 1 : 0) << "\n";
    return 0;
}

// ---------------------------------------------------------------- chat

int cmd_chat(const Args& a) {
    Config cfg = a.cfg;
    std::vector<std::uint8_t> data;
    double w[6];
    PackedModel pm;
    std::uint64_t mix_stored = 0;  // the training form's Synth accounting (not printed when served from --model)
    if (!a.model.empty()) {
        if (!load_model(a, pm, w, cfg, data)) return 1;
    } else {
        if (!load_wiki(a, data)) return 1;
        Model model;
        model.data = data.data();
        model.init(cfg);
        init_weights(w, cfg);
        TrainLog log;
        log.progress = true;
        train_span(model, data, w, 0, cfg.train, cfg, &log);

        std::vector<Turn> turns;
        const std::string chat = load_dolly(a.dolly, turns);
        if (turns.empty()) {
            std::cerr << "dolly missing " << a.dolly << "\n";
            return 1;
        }
        std::cerr << "dolly_turns " << turns.size() << " chat_bytes " << chat.size() << "\n";
        const std::size_t chat_at = append_chat(model, data, chat);
        for (int pass = 0; pass < 2; ++pass) {
            train_span(model, data, w, chat_at, data.size(), cfg);
            std::cerr << "chat_pass " << pass << " w4 " << w[4] << " wmatch " << w[5] << "\n";
        }
        model.compact_match();
        std::cerr << "train_match_rate " << (static_cast<double>(log.matches) / static_cast<double>(cfg.train)) << "\n";

        const Layout synth = measure(model, Mode::Synth);
        mix_stored = synth.bytes + order1_bytes(model) + match_bytes(model) + 6ull * sizeof(double);
        pm = pack(model);  // M4: the packed form serves; the training form is freed
        model.release();
        if (!a.save.empty() && !save_model(a.save, pm, w, cfg)) return 1;
    }

    std::cerr << "scoring\n";
    const Holdout h = score_holdout(pm, data.data(), w, cfg);
    if (!h.ok) return 1;

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "wiki_bpc_after_chat " << h.mix_bpc << "\n";
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";
    if (a.model.empty()) std::cout << "mix_stored " << mix_stored << "\n";
    std::cout << "mix_beats_backoff " << (h.mix_bpc < h.backoff_bpc ? 1 : 0) << "\n";

    // Answers come from CyphaLM generation only (PLAN rule 10).
    SampleOptions opt;
    opt.no_repeat = a.no_repeat;
    const std::string sample = sample_continuation(pm, w, a.prompt, kAllFollowers, 160, kSeed, opt);
    std::cout << "unseen_sample " << printable(sample) << "\n";
    return 0;
}

// ---------------------------------------------------------------- probe

// wiki [0, train), then two Dolly passes after a <|doc|> break.
bool train_wiki_and_chat(const Args& a, std::vector<std::uint8_t>& data, Model& model, double w[6],
                         std::vector<Turn>& turns) {
    const Config& cfg = a.cfg;
    if (!load_wiki(a, data)) return false;
    model.data = data.data();
    model.init(cfg);
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);

    const std::string chat = load_dolly(a.dolly, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return false;
    }
    const std::size_t chat_at = append_chat(model, data, chat);
    train_span(model, data, w, chat_at, data.size(), cfg);
    train_span(model, data, w, chat_at, data.size(), cfg);
    model.compact_match();
    return true;
}

int cmd_probe(const Args& a) {
    std::vector<std::uint8_t> data;
    Model model;
    double w[6];
    std::vector<Turn> turns;
    if (!train_wiki_and_chat(a, data, model, w, turns)) return 1;
    std::cerr << "trained turns " << turns.size() << " w4 " << w[4] << " wmatch " << w[5] << "\n";
    const PackedModel pm = pack(model);  // M4
    model.release();

    struct Q {
        std::string kind, prompt, expect;
    };
    std::vector<Q> qs;
    const std::size_t picks[3] = {0, turns.size() / 2, turns.size() - 1};
    for (auto p : picks) qs.push_back({"seen", turns[p].prompt, turns[p].answer.substr(0, 40)});
    for (auto p : picks) {
        std::string s = turns[p].prompt;
        s[6] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[6])));
        qs.push_back({"change_first_letter", s, turns[p].answer.substr(0, 40)});
    }
    const std::pair<const char*, const char*> wiki[] = {
        {"Who wrote Atlas Shrugged?", "Rand"},
        {"What is the capital of Algeria?", "Algiers"},
        {"Who was Aristotle's teacher?", "Plato"},
        {"What is the capital of Andorra?", "Andorra la Vella"},
        {"Who composed An American in Paris?", "Gershwin"},
        {"What is the capital of Albania?", "Tirana"},
        {"Who taught Alexander the Great?", "Aristotle"},
        {"What does an aardvark eat?", "ant"},
    };
    for (auto& [q, ans] : wiki) qs.push_back({"wiki_fact", std::string("User: ") + q + "\nAssistant: ", ans});

    SampleOptions greedy;
    greedy.greedy = true;
    for (auto& q : qs) {
        const std::string g = sample_continuation(pm, w, q.prompt, data.size(), 120, 0, greedy);
        const std::string s = sample_continuation(pm, w, q.prompt, data.size(), 160, kSeed);
        const bool hit = lowered(g + s).find(lowered(q.expect)) != std::string::npos;
        std::cout << "== " << q.kind << " | " << printable(q.prompt.substr(0, 110)) << "\n";
        std::cout << "   expect: " << printable(q.expect) << "\n";
        std::cout << "   model_contains_expect: " << (hit ? 1 : 0) << "\n";
        std::cout << "   greedy: " << printable(g.substr(0, 120)) << "\n";
        std::cout << "   sample: " << printable(s.substr(0, 160)) << "\n";
        std::cout << "GEN\tqa_probe_nucleus\t" << printable(s) << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------- context8k

int len_bucket(int n) { return n < 4 ? 0 : n < 8 ? 1 : n < 16 ? 2 : n < 32 ? 3 : n < 64 ? 4 : n < 128 ? 5 : 6; }
int count_bucket(std::uint64_t t) { return t <= 1 ? 0 : t <= 3 ? 1 : t <= 15 ? 2 : 3; }
int bucket(int n, std::uint64_t t) { return len_bucket(n) * 4 + count_bucket(t); }
constexpr int kBuckets = 28;

// A4, defined below with the similar-context expert's spec: the expert where it is a default, and its use in sampling.
void fit_lambdas(const std::vector<double>& pe, const std::vector<double>& pb, const std::vector<int>& bk,
                 std::size_t upto, int nb, double* lam);
std::unique_ptr<SimilarContext> default_sim(const Args& a, const std::uint8_t* text, std::size_t n, int& rc);
// GEN-C (PLAN B5), defined below with the similar-context expert: the document experts of --doc SPEC, and their
// lambdas fit on the first half of the holdout over cur (cur is then mixed with them; its second half is the read).
std::unique_ptr<DocTopic> make_doc(const std::string& doc, const std::uint8_t* text, std::size_t n, int& rc);
std::vector<std::vector<double>> fit_doc(const DocTopic& dt, const std::uint8_t* data, std::size_t train, std::size_t half,
                                         std::vector<double>& cur);
void mix_doc_into(const DocTopic& dt, const std::uint8_t* data, std::size_t train, std::size_t half, std::vector<double>& cur);
std::string sample_with_class(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                              const double* lam, const std::string& prompt, int n, std::uint32_t seed,
                              const SimilarContext* sc, const double* lsm, double temperature, const double* lcp,
                              const DocTopic* dt = nullptr, const std::vector<std::vector<double>>* ldoc = nullptr,
                              const ContentPlanStore::Plan* pl = nullptr, const double* lpl = nullptr,
                              const cypha::cyphalm::bkn::ByteKN* kn = nullptr, const double* lkn = nullptr,
                              bool kn_replace = false);
void sample_mix(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc, const double* lam,
                const std::vector<std::uint8_t>& seq, const SimilarContext* sc, const double* lsm, const double* lcp,
                double M[256], const cypha::cyphalm::bkn::ByteKN* kn = nullptr, const double* lkn = nullptr,
                bool kn_replace = false);
// BKN (docs/reports/BYTE_KN.md), defined below: --bkn K[:m:c:d] [--bkn-way a|b], packed byte Kneser-Ney replacing (a)
// or added to (b, the default) the trie mixture in qa, sample and context8k; on by default (BKN2), off with --no-bkn.
std::vector<int> bkn_spec(const std::string& s);
constexpr const char* kBknDefault = "6:4:2:1";  // BKN2: the default of qa, sample and context8k (way b)
std::unique_ptr<cypha::cyphalm::bkn::ByteKN> make_bkn(const std::string& spec, const std::uint8_t* t, std::size_t n);
std::vector<double> mix_bkn_into(const cypha::cyphalm::bkn::ByteKN& kn, const std::uint8_t* data, std::size_t train,
                                 std::size_t half, bool replace, std::vector<double>& cur);
void bkn_mix(const cypha::cyphalm::bkn::ByteKN& kn, const double* lkn, bool replace, const std::uint8_t* seq,
             std::size_t len, double* P);
// GEN-B, defined below with sample_with_class: the sentence gate.
struct GateSpec {
    int k = 0;                       // candidates per sentence (0: no gate)
    double wf = 1, wp = 0, wm = 0;   // weights of fluency, pmi and markup
    int min_len = 24, max_len = 240;  // a candidate's bytes before a sentence end may stop it; at most
};
bool parse_gate(const std::string& spec, GateSpec& g);
struct GateStats {
    double fluency = 0, pmi = 0, markup = 0, changed = 0;  // sums over chosen candidates; changed: not candidate 0
    std::size_t sentences = 0;
};
std::string sample_gated(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                         const double* lam, const SimilarContext* sc, const double* lsm, const double* lcp,
                         const std::string& prompt, int n, std::uint32_t seed, double temperature, const GateSpec& g,
                         GateStats& st, const cypha::cyphalm::bkn::ByteKN* kn = nullptr, const double* lkn = nullptr,
                         bool kn_replace = false);
struct GateCand {
    std::vector<std::uint8_t> bytes;
    double f = 0, p = 0, m = 0, score = 0;  // fluency, pmi, markup, score
};
std::uint32_t gate_seed(std::uint32_t seed, std::uint32_t j, int k);
GateCand gate_candidate(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                        const double* lam, const SimilarContext* sc, const double* lsm, const double* lcp,
                        const std::vector<std::uint8_t>& seq, std::size_t p0, int budget, std::uint32_t rs,
                        double temperature, const GateSpec& g, const cypha::cyphalm::bkn::ByteKN* kn = nullptr,
                        const double* lkn = nullptr, bool kn_replace = false);
// B4C, defined below with the copy and class experts: the class and copy experts where they are defaults.
std::string default_cls_spec(const Args& a);
std::unique_ptr<ClassContext> default_cls(const Args& a, const std::uint8_t* text, std::size_t n, int& rc);
std::vector<double> mix_cls_into(const ClassContext& cc, const std::uint8_t* data, std::size_t train, std::size_t half,
                                 std::vector<double>& cur);
std::vector<double> mix_copy_into(const std::uint8_t* data, std::size_t train, std::size_t half, int m,
                                  std::vector<double>& cur);
// GEN-E, defined below with the copy expert: A3's frames, and the content-plan expert mixed into cur.
std::map<std::string, std::vector<std::string>> load_frames(const std::string& path);
bool good_start(const std::string& l);
std::vector<double> mix_plan_into(const ContentPlanStore& st, const std::map<std::string, std::vector<std::string>>& frames,
                                  const std::uint8_t* data, std::size_t train, std::size_t half, std::vector<double>& cur);

int cmd_context8k(const Args& a) {
    const Config& cfg = a.cfg;
    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);

    std::vector<Turn> turns;
    load_dolly(a.dolly, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return 1;
    }
    std::vector<char> held(turns.size(), 0);
    std::vector<std::size_t> held_idx;
    for (std::size_t i = 0; i < turns.size(); ++i)
        if (i % 500 == 250) {
            held[i] = 1;
            held_idx.push_back(i);
        }
    std::string chat;
    for (std::size_t i = 0; i < turns.size(); ++i)
        if (!held[i]) chat += turns[i].prompt + turns[i].answer + "\n\n";
    const std::size_t chat_at = append_chat(model, data, chat);
    train_span(model, data, w, chat_at, data.size(), cfg);
    train_span(model, data, w, chat_at, data.size(), cfg);
    model.compact_match();
    std::cerr << "trie trained, held out " << held_idx.size() << " dolly turns\n";
    const PackedModel pm = pack(model);  // M4
    model.release();

    // Index the training bytes only: wiki train + boundary + chat. No wiki holdout. The index takes T, then the class
    // expert (B4C, a default; --no-cls leaves it out) is built on the same bytes, which the index gives back (M16: so
    // the class table is not resident during the index's scratch), as in qa.
    std::vector<std::uint8_t> T;
    const std::size_t all = training_text(data, cfg, T), tn = T.size();
    const FmInfiniGram ig(std::move(T));
    std::cerr << "index built over " << tn << " bytes\n";
    int crc = 0;
    std::unique_ptr<ClassContext> cc;
    std::unique_ptr<cypha::cyphalm::bkn::ByteKN> kn;  // BKN2: byte Kneser-Ney on the same bytes, from the same copy
    if (!default_cls_spec(a).empty() || !a.bkn.empty()) {
        const std::vector<std::uint8_t> t = ig.text();
        if (!default_cls_spec(a).empty()) cc = default_cls(a, t.data(), t.size(), crc);
        if (!crc && !a.bkn.empty() && !(kn = make_bkn(a.bkn, t.data(), t.size()))) return 2;
    }
    if (crc) return crc;

    const std::size_t kTrain = cfg.train, kSample = cfg.sample;
    const std::size_t H = kSample - kTrain, half = H / 2;
    std::vector<double> pt(H);
    for (std::size_t k = 0; k < H; ++k) {
        const std::size_t i = kTrain + k;
        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        pm.fill_experts(data.data(), i, kTrain, Mode::Synth, e, em, pred, false);
        double M[256];
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
    }
    double bits_trie = 0;
    for (std::size_t k = half; k < H; ++k) bits_trie += -std::log2(pt[k]);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "trie_only_bpc " << bits_trie / static_cast<double>(H - half) << "\n";
    // BKN2: Kneser-Ney into the trie mixture, first (as qa); the lines below are over trie + Kneser-Ney.
    const bool kn_rep = a.bkn_way == "a";
    // BKN3: with Kneser-Ney off inside answers, the answers (gen, below) use InfiniGram's lambdas fit without it (lam8k0,
    // over this copy of the trie mixture).
    const bool kn_off_ans = kn && a.bkn_answers == "off";
    std::vector<double> pt0;
    if (kn_off_ans) pt0 = pt;
    std::vector<double> lkn;
    if (kn) {
        lkn = mix_bkn_into(*kn, data.data(), kTrain, half, kn_rep, pt);
        double b = 0;
        for (std::size_t k = half; k < H; ++k) b += -std::log2(pt[k]);
        std::cout << "trie_plus_bkn_bpc " << b / static_cast<double>(H - half) << "\n";
    }

    const int caps[] = {4, 16, 64, 512, 8192};
    double lam8k[kBuckets] = {0}, lam8k0[kBuckets] = {0};
    std::vector<int> n8k(H), bk8k;
    std::vector<double> pinf8k;
    for (int cap : caps) {
        std::vector<double> pinf(H);
        std::vector<int> bk(H);
        int prev = -1;
        FmInfiniGram::Cursor cur;
        for (std::size_t k = 0; k < H; ++k) {
            const std::size_t i = kTrain + k;
            const std::size_t len = std::min<std::size_t>(i, static_cast<std::size_t>(cap));
            const int hint = prev < 0 ? -1 : std::min(prev + 1, cap);
            const auto r = ig.query(data.data() + (i - len), len, cap, hint, 1, cur);
            prev = r.n;
            pinf[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
            bk[k] = bucket(r.n, r.total);
            if (cap == 8192) n8k[k] = r.n;
        }
        double lam[kBuckets];
        for (int b = 0; b < kBuckets; ++b) {
            double best = -1e300;
            lam[b] = 0;
            for (int g = 0; g < 200; ++g) {
                const double l = g * 0.005;
                double ll = 0;
                bool any = false;
                for (std::size_t k = 0; k < half; ++k)
                    if (bk[k] == b) {
                        ll += std::log(l * pinf[k] + (1 - l) * pt[k]);
                        any = true;
                    }
                if (!any) break;
                if (ll > best) {
                    best = ll;
                    lam[b] = l;
                }
            }
        }
        double bits = 0;
        for (std::size_t k = half; k < H; ++k) bits += -std::log2(lam[bk[k]] * pinf[k] + (1 - lam[bk[k]]) * pt[k]);
        std::cout << "trie_plus_context_" << cap << "_bpc " << bits / static_cast<double>(H - half) << "\n";
        if (cap == 8192 && kn_off_ans) {  // BKN3: the same grid fit over the trie mixture without Kneser-Ney
            for (int b = 0; b < kBuckets; ++b) {
                double best = -1e300;
                lam8k0[b] = 0;
                for (int g = 0; g < 200; ++g) {
                    const double l = g * 0.005;
                    double ll = 0;
                    bool any = false;
                    for (std::size_t k = 0; k < half; ++k)
                        if (bk[k] == b) {
                            ll += std::log(l * pinf[k] + (1 - l) * pt0[k]);
                            any = true;
                        }
                    if (!any) break;
                    if (ll > best) {
                        best = ll;
                        lam8k0[b] = l;
                    }
                }
            }
            std::vector<double>().swap(pt0);
        }
        if (cap == 8192) {
            std::copy(lam, lam + kBuckets, lam8k);
            pinf8k = std::move(pinf);
            bk8k = std::move(bk);
        }
    }
    // A4, B4C: the default experts over trie + InfiniGram (cap 8192), in qa's order: the class expert (built above),
    // the similar-context expert (keys over the wiki train bytes), the copy expert (m = --copy-min, over the holdout
    // stream). Each one's lambdas are fit on the first half over the distribution it is mixed into; its line reads the
    // second half. --no-cls, --no-sim and --no-copy leave them out. The answers below are not touched: the experts are
    // off inside answers (in_answer), and they are written without them. Kneser-Ney (BKN2) is part of the byte model
    // above, so the answers use it.
    std::vector<double> base(H);
    for (std::size_t k = 0; k < H; ++k) base[k] = lam8k[bk8k[k]] * pinf8k[k] + (1 - lam8k[bk8k[k]]) * pt[k];
    std::string chain = "trie_plus_context_8192";
    auto chain_line = [&](const char* part) {
        double bits = 0;
        for (std::size_t k = half; k < H; ++k) bits += -std::log2(base[k]);
        chain += part;
        std::cout << chain << "_bpc " << bits / static_cast<double>(H - half) << "\n";
    };
    if (cc) {
        mix_cls_into(*cc, data.data(), kTrain, half, base);
        chain_line("_cls");
    }
    int rc = 0;
    if (const auto sc = default_sim(a, data.data(), kTrain, rc)) {
        std::vector<double> ps(H);
        std::vector<int> sb(H);
        for (std::size_t k = 0; k < H; ++k) {
            double P[256];
            sb[k] = SimilarContext::bucket(sc->predict(data.data(), kTrain + k, P));
            ps[k] = P[data[kTrain + k]];
        }
        double ls[SimilarContext::kBuckets];
        fit_lambdas(ps, base, sb, half, SimilarContext::kBuckets, ls);
        for (std::size_t k = 0; k < H; ++k) base[k] = ls[sb[k]] * ps[k] + (1 - ls[sb[k]]) * base[k];
        chain_line("_sim");
    } else if (rc)
        return rc;
    if (!a.no_copy) {
        mix_copy_into(data.data(), kTrain, half, a.copy_min, base);
        chain_line("_copy");
    }
    // GEN-C2 (docs/reports/GEN_C_TOPIC_CACHE.md, "GEN-C2"): the document experts' cache (--doc SPEC to read another
    // setting), last, over wiki train's words; its lambdas on the first half, its line on the second. --no-doc leaves
    // it out. The answers below never read it.
    if (!a.no_doc) {
        int drc = 0;
        const auto dt = make_doc(a.doc.empty() ? std::string("cache") : a.doc, data.data(), kTrain, drc);
        if (!dt) return drc ? drc : 1;
        mix_doc_into(*dt, data.data(), kTrain, half, base);
        chain_line("_doc");
    }

    const int edges[] = {4, 8, 16, 32, 64, 256, 1024, 8193};
    const char* names[] = {"<4", "4-7", "8-15", "16-31", "32-63", "64-255", "256-1023", "1024-8192"};
    std::size_t cnt[8] = {0};
    for (int n : n8k)
        for (int e = 0; e < 8; ++e)
            if (n < edges[e]) {
                ++cnt[e];
                break;
            }
    std::vector<int> sorted = n8k;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(H / 2), sorted.end());
    std::cout << "median_matched_context_bytes " << sorted[H / 2] << "\n";
    for (int e = 0; e < 8; ++e) std::cout << "matched " << names[e] << " " << 100.0 * cnt[e] / H << "%\n";

    auto gen = [&](const std::string& prompt, int& n0) {
        std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
        int prev = -1;
        FmInfiniGram::Cursor cur;
        n0 = -1;
        for (int t = 0; t < 150; ++t) {
            double e[5][256];
            bool m = false;
            unsigned pred = 0;
            pm.fill_experts(seq.data(), seq.size(), all, Mode::Synth, e, m, pred, false);
            double M[256];
            mix_dist(w, e, m, pred, M);
            if (kn && !kn_off_ans) bkn_mix(*kn, lkn.empty() ? nullptr : lkn.data(), kn_rep, seq.data(), seq.size(), M);  // BKN2; BKN3
            const std::size_t len = std::min<std::size_t>(seq.size(), 8192);
            const auto r = ig.query(seq.data() + (seq.size() - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192),
                                    1, cur);
            prev = r.n;
            if (t == 0) n0 = r.n;
            const double l = (kn_off_ans ? lam8k0 : lam8k)[bucket(r.n, r.total)];  // BKN3: answers without Kneser-Ney
            int best = 0;
            double bp = -1;
            for (int s = 0; s < 256; ++s) {
                const double pc = r.total ? static_cast<double>(r.count[s]) / static_cast<double>(r.total) : 0.0;
                const double p = l * pc + (1 - l) * M[s];
                if (p > bp) {
                    bp = p;
                    best = s;
                }
            }
            seq.push_back(static_cast<std::uint8_t>(best));
        }
        return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
    };

    struct Q {
        std::string kind, prompt, expect;
    };
    std::vector<Q> qs;
    const std::size_t picks[3] = {0, turns.size() / 2, turns.size() - 1};
    for (auto p : picks) qs.push_back({"seen", turns[p].prompt, turns[p].answer.substr(0, 30)});
    for (auto p : picks) {
        std::string s = turns[p].prompt;
        s[6] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[6])));
        qs.push_back({"change_first_letter", s, turns[p].answer.substr(0, 30)});
    }
    for (auto p : picks) {
        const std::string& s = turns[p].prompt;
        qs.push_back({"add_Thanks_at_end", s.substr(0, s.size() - 12) + " Thanks.\nAssistant: ",
                      turns[p].answer.substr(0, 30)});
    }
    qs.push_back({"paraphrase", "User: When did Virgin Blue begin flying?\nAssistant: ", "2000"});
    qs.push_back({"paraphrase", "User: Who are the cast members of Southern Charm?\nAssistant: ", "Conover"});
    qs.push_back({"paraphrase", "User: Tell me about the Masters golf tournament.\nAssistant: ", "Augusta"});
    for (auto j : held_idx) qs.push_back({"heldout_dolly", turns[j].prompt, turns[j].answer.substr(0, 30)});
    const std::pair<const char*, const char*> wiki[] = {
        {"Who wrote Atlas Shrugged?", "Rand"},          {"What is the capital of Algeria?", "Algiers"},
        {"Who was Aristotle's teacher?", "Plato"},      {"What is the capital of Andorra?", "Andorra la Vella"},
        {"Who composed An American in Paris?", "Gershwin"}, {"What is the capital of Albania?", "Tirana"},
        {"Who taught Alexander the Great?", "Aristotle"}, {"What does an aardvark eat?", "ant"}};
    for (auto& [q, ans] : wiki) qs.push_back({"wiki_fact", std::string("User: ") + q + "\nAssistant: ", ans});

    std::map<std::string, std::pair<int, int>> tally;
    for (auto& q : qs) {
        int n0 = 0;
        const std::string g = gen(q.prompt, n0);
        const bool ok = lowered(g).find(lowered(q.expect)) != std::string::npos;
        tally[q.kind].first += ok;
        tally[q.kind].second += 1;
        std::cout << "== " << q.kind << " ok=" << ok << " matched_ctx=" << n0 << " | Q: "
                  << printable(q.prompt.substr(0, 90)) << "\n";
        std::cout << "   want: " << printable(q.expect) << "\n   got:  " << printable(g.substr(0, 130)) << "\n";
        std::cout << "GEN\t" << q.kind << "\t" << printable(g) << "\n";
    }
    for (auto& [k, v] : tally) std::cout << "TALLY " << k << " " << v.first << "/" << v.second << "\n";
    return 0;
}

// ---------------------------------------------------------------- sample

// GEN-A: bench/eval/gen_prompts_{dev,test}.tsv, a header row, then id, doc, prompt, continuation with tab, newline and
// backslash escaped (scripts/gen_bench.py). The continuation is scored after a space, as it follows the prompt's last
// sentence in its page.
std::vector<std::string> split(const std::string& s, char d);

std::string unescape_tsv(const std::string& s) {
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o += c == 't' ? '\t' : c == 'n' ? '\n' : c;
        } else o += s[i];
    }
    return o;
}

bool read_gen_prompts(const std::string& path, std::vector<std::string>& prompts, std::vector<std::string>& conts) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot read " << path << "\n";
        return false;
    }
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 4) continue;
        prompts.push_back(unescape_tsv(f[2]));
        conts.push_back(" " + unescape_tsv(f[3]));
    }
    return !prompts.empty();
}

// GEN-A: the share of the texts' bytes that lie in a 32-byte span occurring in the given texts (a sorted table of
// FNV-1a 64 hashes of every 32-byte window of each).
struct Spans32 {
    static constexpr std::size_t K = 32;
    std::vector<std::uint64_t> h;
    static std::uint64_t hash(const std::uint8_t* p) {
        std::uint64_t x = 0xcbf29ce484222325ull;
        for (std::size_t i = 0; i < K; ++i) x = (x ^ p[i]) * 0x100000001b3ull;
        return x;
    }
    void add(const std::uint8_t* t, std::size_t n) {
        for (std::size_t i = 0; i + K <= n; ++i) h.push_back(hash(t + i));
    }
    void finish() {
        std::sort(h.begin(), h.end());
        h.erase(std::unique(h.begin(), h.end()), h.end());
    }
    // bytes covered, bytes in all
    std::pair<std::size_t, std::size_t> covered(const std::string& s) const {
        const auto* b = reinterpret_cast<const std::uint8_t*>(s.data());
        std::vector<char> cov(s.size(), 0);
        for (std::size_t i = 0; i + K <= s.size(); ++i)
            if (std::binary_search(h.begin(), h.end(), hash(b + i))) std::fill(cov.begin() + static_cast<std::ptrdiff_t>(i), cov.begin() + static_cast<std::ptrdiff_t>(i + K), 1);
        return {static_cast<std::size_t>(std::count(cov.begin(), cov.end(), 1)), s.size()};
    }
};

bool good_start(const std::string& l) {
    if (l.size() < 100) return false;
    const unsigned char c = static_cast<unsigned char>(l[0]);
    if (l.compare(0, 3, "'''") == 0) return true;
    return c >= 'A' && c <= 'Z';
}

// Real generated text from the wiki-only trie for the SCRIBE control ladder (PLAN 6.4).
// Prompts are the first 64 bytes (cut back to a UTF-8 boundary) of paragraph lines in the
// holdout; the candidates are shuffled with a fixed seed and dealt to the output files in
// order, so the files use DISJOINT prompts. Every prompt is sampled at T = 0.6, 0.8, 1.0
// with nucleus 0.9 and the no-repeat rule. The match expert may only copy from bytes before
// cfg.train, so a prompt's own paragraph can never be copied back.
int cmd_sample(const Args& a) {
    Config cfg = a.cfg;
    if (a.outs.empty() || a.outs.size() > 2 || a.prompts < 1 || a.bytes < 100) {
        std::fprintf(stderr,
                     "usage: cyphalm_trie sample <enwik8> <out_dev.txt> [out_fresh.txt] [--prompts 250] [--bytes 600]\n"
                     "       cyphalm_trie sample <enwik8> <out.txt> --prompts-file bench/eval/gen_prompts_dev.tsv\n"
                     "       [--temps T[,T][,T]] (default 0.6,0.8,1.0)\n");
        return 2;
    }
    // TEMPS: one to three temperatures; each keeps the seed of its position, so --temps 0.4 pairs with 0.6's draws.
    std::vector<double> temps = {0.6, 0.8, 1.0};
    if (!a.temps.empty()) {
        temps.clear();
        for (const auto& t : split(a.temps, ',')) {
            char* end = nullptr;
            const double v = std::strtod(t.c_str(), &end);
            if (t.empty() || end == t.c_str() || *end != '\0' || !(v > 0.0) || v > 4.0) {
                std::cerr << "bad --temps: " << a.temps << " (one to three temperatures in (0, 4], e.g. 0.4,0.6,0.8)\n";
                return 2;
            }
            temps.push_back(v);
        }
        if (temps.empty() || temps.size() > 3) {
            std::cerr << "--temps takes one to three temperatures\n";
            return 2;
        }
    }
    const std::size_t nt = temps.size();
    const auto temp_label = [](double t) {  // trie_T0.6 for one decimal (as before), else two
        char b[32];
        std::snprintf(b, sizeof b, std::fabs(t * 10.0 - std::round(t * 10.0)) < 1e-9 ? "%.1f" : "%.2f", t);
        return std::string(b);
    };
    const std::size_t nfiles = a.outs.size();
    GateSpec gate;  // GEN-B
    if (!a.gate.empty() && !parse_gate(a.gate, gate)) {
        std::cerr << "bad --gate spec: " << a.gate << " (K[,wf=..][,wp=..][,wm=..][,min=..][,max=..])\n";
        return 2;
    }
    if (gate.k > 0 && (!a.doc.empty() || !a.plan.empty())) {  // GEN-B: candidates use sample_mix's experts (BKN included)
        std::cerr << "--gate does not take --doc or --plan\n";
        return 2;
    }
    std::vector<std::uint8_t> data;
    double w[6];
    PackedModel pm;
    if (!a.model.empty() ? !load_model(a, pm, w, cfg, data) : !load_wiki(a, data)) return 1;
    std::vector<std::uint8_t> raw_train;  // GEN-A: the copy share's reference is the raw and the cleaned training bytes
    if (!a.prompts_file.empty()) raw_train.assign(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.train));
    const std::size_t plan_lo = cfg.train, plan_hi = cfg.sample;  // GEN-E: the holdout in raw bytes, before --clean
    if (a.clean) {  // GEN-A: [0, train) and [train, sample) each become SCRIBE's cleaned pages, one per line
        if (!a.model.empty()) {
            std::cerr << "--clean trains; a stored model (--model) keeps the text it was trained on\n";
            return 2;
        }
        std::vector<std::uint8_t> c;
        for (const auto& d : cypha::eval::clean_wiki_documents(reinterpret_cast<const char*>(data.data()), cfg.train)) {
            c.insert(c.end(), d.begin(), d.end());
            c.push_back('\n');
        }
        const std::size_t ntrain = c.size();
        for (const auto& d : cypha::eval::clean_wiki_documents(reinterpret_cast<const char*>(data.data()) + cfg.train,
                                                               cfg.sample - cfg.train)) {
            c.insert(c.end(), d.begin(), d.end());
            c.push_back('\n');
        }
        std::cerr << "clean text: train " << cfg.train << " -> " << ntrain << " bytes, holdout " << cfg.sample - cfg.train
                  << " -> " << c.size() - ntrain << "\n";
        cfg.train = ntrain;
        cfg.sample = c.size();
        data.swap(c);
    }

    std::vector<std::string> prompts, conts;  // GEN-A: conts only with --prompts-file
    if (!a.prompts_file.empty()) {
        if (nfiles != 1) {
            std::cerr << "--prompts-file takes one output file\n";
            return 2;
        }
        if (!read_gen_prompts(a.prompts_file, prompts, conts)) return 1;
    } else {
        const std::string hold(reinterpret_cast<const char*>(data.data()) + cfg.train, cfg.sample - cfg.train);
        std::size_t pos = hold.find('\n');  // the first line is cut by the range
        pos = pos == std::string::npos ? hold.size() : pos + 1;
        while (pos < hold.size()) {
            std::size_t e = hold.find('\n', pos);
            if (e == std::string::npos) break;  // the last line is cut by the range
            const std::string line = hold.substr(pos, e - pos);
            pos = e + 1;
            if (!good_start(line)) continue;
            std::size_t cut = 64;
            while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;  // character boundary
            prompts.push_back(line.substr(0, cut));
        }
    }
    if (a.prompts_file.empty()) {
        std::mt19937_64 prng(20261027);
        std::shuffle(prompts.begin(), prompts.end(), prng);
    }
    const std::size_t need = static_cast<std::size_t>(a.prompts) * nfiles;
    std::cerr << (a.prompts_file.empty() ? "holdout prompts " : "file prompts ") << prompts.size() << ", using " << need << "\n";
    if (prompts.size() < need) {
        std::cerr << "too few holdout paragraphs for " << need << " prompts\n";
        return 1;
    }

    if (a.model.empty()) {
        Model model;
        model.data = data.data();
        model.init(cfg);
        init_weights(w, cfg);
        TrainLog log;
        log.progress = true;
        train_span(model, data, w, 0, cfg.train, cfg, &log);
        model.compact_match();
        std::cerr << "trained on [0, " << cfg.train << ") nodes " << model.nodes.size() << " w4 " << w[4] << " wmatch "
                  << w[5] << "\n";
        pm = pack(model);  // M4
        model.release();
        if (!a.save.empty() && !save_model(a.save, pm, w, cfg)) return 1;
    }

    // A4, B4C: the default experts (--no-sim, --no-copy leave them out; with none, the trie alone writes B1's files) in
    // qa's order: the similar-context expert with keys over the train bytes, then the copy expert (m = --copy-min). The
    // class expert is not a default here (B4C: SCRIBE v2 falls with it); --cls SPEC adds it first, built on the train
    // bytes. Each one's lambdas are fit on the first half of the holdout over the trie with the experts before it. The
    // files then start with a MODEL line.
    int rc = 0;
    // BKN (--bkn): byte Kneser-Ney on the train bytes, first in the order (it replaces or joins the trie mixture).
    std::unique_ptr<cypha::cyphalm::bkn::ByteKN> kn;
    if (!a.bkn.empty() && !(kn = make_bkn(a.bkn, data.data(), cfg.train))) return 2;
    const bool kn_rep = a.bkn_way == "a";
    std::vector<double> lkn;
    std::unique_ptr<ClassContext> cc;
    if (!a.cls.empty()) cc = default_cls(a, data.data(), cfg.train, rc);
    if (rc) return rc;
    const auto sc = default_sim(a, data.data(), cfg.train, rc);
    if (rc) return rc;
    // GEN-E: --plan STOREDIR[,RAWFRAMES], the content-plan expert (docs/reports/GEN_E_CONTENT_PLAN.md). The store drops
    // facts read from the holdout (raw bytes, so also under --clean) and from --plan-drop's range; its frames are A3's,
    // from STOREDIR/patterns.tsv and F2R's raw frames.
    std::unique_ptr<ContentPlanStore> cps;
    std::map<std::string, std::vector<std::string>> frames;
    if (!a.plan.empty()) {
        const std::size_t comma = a.plan.find(',');
        const std::vector<std::string> parts =
            comma == std::string::npos ? std::vector<std::string>{a.plan}
                                       : std::vector<std::string>{a.plan.substr(0, comma), a.plan.substr(comma + 1)};
        cps = std::make_unique<ContentPlanStore>();
        std::string err;
        std::size_t lo2 = 0, hi2 = 0;  // GEN-E: --plan-drop LO:HI (raw bytes), e.g. the benchmark's source range
        if (!a.plan_drop.empty()) {
            const std::size_t c = a.plan_drop.find(':');
            if (c == std::string::npos) {
                std::cerr << "--plan-drop wants LO:HI\n";
                return 2;
            }
            lo2 = std::strtoull(a.plan_drop.substr(0, c).c_str(), nullptr, 10);
            hi2 = std::strtoull(a.plan_drop.substr(c + 1).c_str(), nullptr, 10);
        }
        if (!cps->load(parts[0], plan_lo, plan_hi, err, lo2, hi2)) {
            std::cerr << err << "\n";
            return 1;
        }
        frames = load_frames(parts[0] + "/patterns.tsv");
        if (parts.size() > 1)
            for (auto& [r, v] : load_frames(parts[1])) {
                auto& o = frames[r];
                for (auto& f : v)
                    if (std::find(o.begin(), o.end(), f) == o.end()) o.insert(o.begin(), f);  // raw frames first
            }
        std::size_t nf = 0;
        for (const auto& [r, v] : frames) nf += v.size();
        std::cerr << "plan store subjects " << cps->subjects() << " relations with frames " << frames.size() << " frames " << nf
                  << "\n";
    }
    const auto dt = make_doc(a.doc, data.data(), cfg.train, rc);  // GEN-C: nullptr without --doc
    if (rc) return rc;
    double lsm[SimilarContext::kBuckets] = {0};
    std::vector<double> lcl, lcp, lpl;
    std::vector<std::vector<double>> ldoc;
    if (cc || sc || !a.no_copy || dt || cps || kn) {
        const std::size_t half = (cfg.sample - cfg.train) / 2;
        // GEN-C, GEN-E: with --doc or --plan every expert is read over the whole holdout (fit on the first half, as
        // without them), so their BPC can be read on the second half.
        const std::size_t span = dt || cps ? cfg.sample - cfg.train : half;
        std::vector<double> pt(span);
        for (std::size_t k = 0; k < span; ++k) {
            const std::size_t i = cfg.train + k;
            double e[5][256], M[256];
            bool em = false;
            unsigned pred = 0;
            pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
            mix_dist(w, e, em, pred, M);
            pt[k] = M[data[i]];
        }
        if (kn) lkn = mix_bkn_into(*kn, data.data(), cfg.train, half, kn_rep, pt);  // BKN
        if (cc) lcl = mix_cls_into(*cc, data.data(), cfg.train, half, pt);
        if (sc) {
            std::vector<double> ps(span);
            std::vector<int> sb(span);
            for (std::size_t k = 0; k < span; ++k) {
                double P[256];
                sb[k] = SimilarContext::bucket(sc->predict(data.data(), cfg.train + k, P));
                ps[k] = P[data[cfg.train + k]];
            }
            fit_lambdas(ps, pt, sb, half, SimilarContext::kBuckets, lsm);
            for (std::size_t k = 0; k < span; ++k) pt[k] = lsm[sb[k]] * ps[k] + (1 - lsm[sb[k]]) * pt[k];
        }
        if (!a.no_copy) lcp = mix_copy_into(data.data(), cfg.train, half, a.copy_min, pt);
        if (dt) ldoc = fit_doc(*dt, data.data(), cfg.train, half, pt);
        if (cps) lpl = mix_plan_into(*cps, frames, data.data(), cfg.train, half, pt);  // GEN-E: mixed last
    }
    const bool experts = cc || sc || !lcp.empty() || dt || cps || kn;
    std::string model_line = std::string("MODEL\ttrie") + (kn ? (kn_rep ? "=bkn" : "+bkn") : "") + (cc ? "+cls" : "") + (sc ? "+sim" : "") + (lcp.empty() ? "" : "+copy") +
                             "\t" + (sc ? (a.sims.empty() ? std::string("default") : a.sims[0]) : std::string("-"));
    if (cc) model_line += "\tcls=" + default_cls_spec(a);
    if (!lcp.empty()) model_line += "\tcopy_min=" + std::to_string(a.copy_min);
    if (dt) model_line += "\tdoc=" + a.doc;
    if (cps) model_line += "\tplan=" + a.plan;
    if (kn) model_line += "\tbkn=" + a.bkn + ":" + a.bkn_way;  // BKN
    if (gate.k > 0) model_line += "\tgate=" + a.gate;  // GEN-B
    if (!a.temps.empty()) model_line += "\ttemps=" + a.temps;  // TEMPS
    std::vector<std::vector<std::string>> gens(nt);  // GEN-A: the generated texts by temperature (--prompts-file)
    std::vector<int> planned(nfiles, 0);
    std::vector<GateStats> gst(nt);
    const auto t_gen = std::chrono::steady_clock::now();
    const std::clock_t c_gen = std::clock();  // GEN-B: CPU time too (wall time varies with the machine's load)
    std::size_t gen_bytes = 0;
    for (std::size_t f = 0; f < nfiles; ++f) {
        std::ofstream out(a.outs[f]);
        if (!out) {
            std::cerr << "cannot write " << a.outs[f] << "\n";
            return 1;
        }
        if (experts || gate.k > 0 || !a.temps.empty()) out << model_line << "\n";
        for (int k = 0; k < a.prompts; ++k) {
            const std::size_t id = f * static_cast<std::size_t>(a.prompts) + static_cast<std::size_t>(k);
            const std::string& p = prompts[id];
            out << "PROMPT\t" << id << "\t" << printable(p) << "\n";
            ContentPlanStore::Plan plan;
            if (cps) {  // GEN-E: the prompt's plan, for the factual read (SCRIBE reads GEN lines only)
                plan = cps->plan(p, frames);
                out << "PLAN\t" << id << "\t" << printable(plan.entity) << "\t";
                for (std::size_t q = 0; q < plan.facts.size(); ++q)
                    out << (q ? "|" : "") << plan.facts[q].rel << "=" << printable(plan.facts[q].value);
                out << "\n";
                planned[f] += !plan.facts.empty();
            }
            for (std::size_t ti = 0; ti < nt; ++ti) {
                const std::uint64_t seed = 1000003ull * (id + 1) + static_cast<std::uint64_t>(ti) * 7919ull;
                SampleOptions opt;
                opt.temperature = temps[ti];
                opt.no_repeat = true;
                const auto seed32 = static_cast<std::uint32_t>(seed ^ (seed >> 32));
                const std::string text =
                    gate.k > 0 ? sample_gated(pm, w, cfg.train, cc.get(), lcl.empty() ? nullptr : lcl.data(), sc.get(), lsm,
                                              lcp.empty() ? nullptr : lcp.data(), p, a.bytes, seed32, temps[ti], gate, gst[ti],
                                              kn.get(), lkn.empty() ? nullptr : lkn.data(), kn_rep)
                    : experts ? sample_with_class(pm, w, cfg.train, cc.get(), lcl.empty() ? nullptr : lcl.data(), p, a.bytes,
                                                  seed32, sc.get(), lsm, temps[ti], lcp.empty() ? nullptr : lcp.data(),
                                                  dt.get(), dt ? &ldoc : nullptr, plan.facts.empty() ? nullptr : &plan,
                                                  lpl.empty() ? nullptr : lpl.data(), kn.get(), lkn.empty() ? nullptr : lkn.data(),
                                                  kn_rep)
                              : sample_continuation(pm, w, p, cfg.train, a.bytes, seed32, opt);
                gen_bytes += text.size();
                out << "GEN\ttrie_T" << temp_label(temps[ti]) << "\t" << printable(text) << "\n";
                if (!a.prompts_file.empty()) gens[ti].push_back(text);
            }
            if ((k + 1) % 50 == 0) std::cerr << "file " << f << " prompt " << (k + 1) << "/" << a.prompts << "\n";
        }
        if (cps) std::cerr << "plan coverage file " << f << ": " << planned[f] << " of " << a.prompts << " prompts\n";
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen).count();
    const double cpu = static_cast<double>(std::clock() - c_gen) / CLOCKS_PER_SEC;
    std::cerr << "generated " << gen_bytes << " bytes in " << secs << " s (" << 1e6 * secs / std::max<std::size_t>(gen_bytes, 1)
              << " us per byte); cpu " << cpu << " s (" << 1e6 * cpu / std::max<std::size_t>(gen_bytes, 1) << " us per byte)\n";
    for (std::size_t ti = 0; ti < nt && gate.k > 0; ++ti) {  // GEN-B: what the gate kept
        const double ns = static_cast<double>(std::max<std::size_t>(gst[ti].sentences, 1));
        std::fprintf(stderr, "gate T%s sentences %zu mean fluency %.4f pmi %.4f markup %.4f not-first %.3f\n", temp_label(temps[ti]).c_str(),
                     gst[ti].sentences, gst[ti].fluency / ns, gst[ti].pmi / ns, gst[ti].markup / ns, gst[ti].changed / ns);
    }
    if (a.prompts_file.empty()) return 0;

    // GEN-A: each human continuation's bits under sampling's mixture (no temperature, no nucleus), and the copy share.
    std::ofstream ct(a.outs[0] + ".cont.tsv");
    ct << "row\tbytes\tbits\n";
    const auto np = static_cast<std::size_t>(a.prompts);
    std::vector<double> bits(np), nb(np);
    for (std::size_t k = 0; k < np; ++k) {
        std::vector<std::uint8_t> seq(prompts[k].begin(), prompts[k].end());
        std::unique_ptr<DocTopic::Stream> ds;  // GEN-C's document experts, as in sample_with_class
        if (dt) {
            ds = std::make_unique<DocTopic::Stream>(*dt);
            ds->feed(seq.data(), seq.size());
            if (dt->params().frozen) ds->freeze();
        }
        ContentPlanStore::Plan plan;  // GEN-E: the content-plan expert, as in sample_with_class
        if (cps && !lpl.empty()) plan = cps->plan(prompts[k], frames);
        std::string folded;
        std::vector<int> active;
        double b = 0;
        for (const char ch : conts[k]) {
            double M[256];
            sample_mix(pm, w, cfg.train, cc.get(), lcl.empty() ? nullptr : lcl.data(), seq, sc.get(), lsm,
                       lcp.empty() ? nullptr : lcp.data(), M, kn.get(), lkn.empty() ? nullptr : lkn.data(), kn_rep);
            if (ds)
                for (int part = 0; part < DocTopic::kParts; ++part) {
                    const auto& lp = ldoc[static_cast<std::size_t>(part)];
                    if (lp.empty()) continue;
                    double P[256];
                    const double l = lp[static_cast<std::size_t>(ds->predict(part, P))];
                    for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
                }
            if (!plan.facts.empty()) {
                active.clear();
                for (std::size_t q = 0; q < plan.facts.size(); ++q)
                    if (!ContentPlanStore::expressed(plan, q, folded)) active.push_back(static_cast<int>(q));
                double P[256];
                const int L = ContentPlanStore::predict(plan, active, seq.data(), seq.size(), P);
                if (L > 0) {
                    const double l = lpl[static_cast<std::size_t>(ContentPlanStore::bucket(L))];
                    for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
                }
            }
            const auto u = static_cast<std::uint8_t>(ch);
            b -= std::log2(std::max(M[u], 1e-300));
            seq.push_back(u);
            if (ds) ds->feed(u);
            ContentPlanStore::fold_into(folded, u);
        }
        bits[k] = b;
        nb[k] = static_cast<double>(conts[k].size());
        ct << k << "\t" << conts[k].size() << "\t" << b << "\n";
    }
    const double sb = std::accumulate(bits.begin(), bits.end(), 0.0), sn = std::accumulate(nb.begin(), nb.end(), 0.0);
    std::vector<double> boot;
    std::mt19937_64 br(20261029);
    std::uniform_int_distribution<std::size_t> pick(0, np - 1);
    for (int r = 0; r < 2000; ++r) {
        double x = 0, y = 0;
        for (std::size_t k = 0; k < np; ++k) {
            const std::size_t j = pick(br);
            x += bits[j];
            y += nb[j];
        }
        boot.push_back(x / y);
    }
    std::sort(boot.begin(), boot.end());
    std::printf("GEN_EVAL cont_bpc %.4f ci95 %.4f %.4f bytes %.0f prompts %zu\n", sb / sn, boot[49], boot[1949], sn, np);
    Spans32 sp;  // the same reference with and without --clean
    sp.add(raw_train.data(), raw_train.size());
    for (const auto& d : cypha::eval::clean_wiki_documents(reinterpret_cast<const char*>(raw_train.data()), raw_train.size()))
        sp.add(reinterpret_cast<const std::uint8_t*>(d.data()), d.size());
    sp.finish();
    std::size_t all_c = 0, all_n = 0;
    std::printf("GEN_EVAL copy32");
    for (std::size_t ti = 0; ti < nt; ++ti) {
        std::size_t cv = 0, n = 0;
        for (const auto& g : gens[ti]) {
            const auto [x, y] = sp.covered(g);
            cv += x;
            n += y;
        }
        all_c += cv;
        all_n += n;
        std::printf(" T%s %.4f", temp_label(temps[ti]).c_str(), n ? static_cast<double>(cv) / static_cast<double>(n) : 0.0);
    }
    std::printf(" pooled %.4f\n", all_n ? static_cast<double>(all_c) / static_cast<double>(all_n) : 0.0);
    return 0;
}

// ---------------------------------------------------------------- qa and dktext (A0, A1)
//
// PLAN rule 10: CyphaLM writes every answer byte. Nothing below searches for an answer. The
// string functions grade what the model wrote (as probe does), or prepare training text
// (dktext: which entities are absent from the training bytes, and a leakage check).

struct QaRow {
    std::string q, title;
    std::vector<std::string> gold;
    int in_slice = 0;
};

std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find(d, a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) return out;
        a = b + 1;
    }
}

// question \t answers (| separated) \t source_title \t in_slice, one header row (T-QA format).
std::vector<QaRow> load_tqa(const std::string& path) {
    std::vector<QaRow> rows;
    std::ifstream in(path);
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty()) continue;
        const auto f = split(line, '\t');
        if (f.size() < 4) continue;
        QaRow r;
        r.q = f[0];
        if (!f[1].empty()) r.gold = split(f[1], '|');
        r.title = f[2];
        r.in_slice = std::atoi(f[3].c_str());
        rows.push_back(r);
    }
    return rows;
}

// The T-QA normaliser: case, accents (Latin-1 range) and punctuation ignored, spaces collapsed.
std::string norm_text(const std::string& s) {
    static const char* const latin1 =  // U+00C0..U+00FF folded to ASCII ('.' = drop)
        "aaaaaaaceeeeiiiidnooooo.ouuuuyts"
        "aaaaaaaceeeeiiiidnooooo.ouuuuyty";
    std::string o = " ";
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char t = ' ';
        if (c < 128) t = std::isalnum(c) ? static_cast<char>(std::tolower(c)) : ' ';
        else if (c == 0xC3 && i + 1 < s.size()) {
            const unsigned char d = static_cast<unsigned char>(s[++i]);
            if (d >= 0x80 && d <= 0xBF) t = latin1[d - 0x80];
            if (t == '.') t = ' ';
            if (d == 0x9F) {  // sharp s
                o += "ss";
                continue;
            }
        } else if ((c & 0xC0) == 0x80) continue;  // other multi-byte characters: dropped
        if (t == ' ' && o.back() == ' ') continue;
        o += t;
    }
    if (o.back() != ' ') o += ' ';
    return o;  // " word word "
}

std::size_t nwords(const std::string& n) { return static_cast<std::size_t>(std::count(n.begin(), n.end(), ' ')) - 1; }

// Right if equal to an accepted variant, or either contains the other when at least one of the
// two is multi-word (T_QA_WIKI_200.md). Word-bounded, on normalised text.
bool gold_match(const std::string& answer, const std::vector<std::string>& gold) {
    const std::string a = norm_text(answer);
    if (a.size() <= 1) return false;
    for (const auto& g0 : gold) {
        const std::string g = norm_text(g0);
        if (g.size() <= 1) continue;
        if (a == g) return true;
        if (a.find(g) != std::string::npos) return true;  // gold inside the answer
        if (nwords(g) >= 2 && nwords(a) >= 1 && g.find(a) != std::string::npos) return true;
    }
    return false;
}

bool is_dont_know(const std::string& a) {
    return a.find("I don't know") != std::string::npos || a.find("<|unknown") != std::string::npos;
}

// Tag field: reason=CODE / entity="..." / relation=REL.
std::string tag_field(const std::string& a, const std::string& key) {
    const std::size_t t = a.find("<|unknown");
    if (t == std::string::npos) return {};
    std::size_t p = a.find(key + "=", t);
    if (p == std::string::npos) return {};
    p += key.size() + 1;
    if (p < a.size() && a[p] == '"') {
        const std::size_t e = a.find('"', p + 1);
        return a.substr(p + 1, e == std::string::npos ? std::string::npos : e - p - 1);
    }
    std::size_t e = p;
    while (e < a.size() && a[e] != ' ' && a[e] != '|') ++e;
    return a.substr(p, e - p);
}

double wilson_lo(double k, double n, double z = 1.96) {
    if (n <= 0) return 0;
    const double p = k / n, z2 = z * z;
    return (p + z2 / (2 * n) - z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n);
}
double wilson_hi(double k, double n, double z = 1.96) {
    if (n <= 0) return 1;
    const double p = k / n, z2 = z * z;
    return (p + z2 / (2 * n) + z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n);
}

std::string gap_id(const std::string& entity, const std::string& rel) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : entity + "|" + rel) h = (h ^ c) * 16777619u;
    char b[16];
    std::snprintf(b, sizeof b, "G-%04X", h & 0xffffu);
    return b;
}

// DISCOVERY 3.5: the message and the tag.
std::string dont_know_text(const std::string& entity, const std::string& rel, const std::string& reason) {
    return "I don't know. I have nothing about " + entity + " in what I've read. <|unknown reason=" + reason +
           " entity=\"" + entity + "\" relation=" + rel + " gap=" + gap_id(entity, rel) + "|>";
}

// S1 (DISCOVERY 3.1, 3.5, 12 I3; docs/reports/S1_REASON_CODES_AND_EVIDENCE.md): the codes, the gate's message, and what
// the system shows for one answer.
const char* const kReasonCodes[] = {"OUT_OF_SCOPE",     "UNPARSEABLE", "AMBIGUOUS_QUESTION", "UNKNOWN_ENTITY",
                                    "AMBIGUOUS_ENTITY", "NO_EVIDENCE", "CONFLICT",           "LOW_CERTAINTY",
                                    "STALE",            "BEYOND_DEPTH", "HYPOTHESIS_ONLY"};
bool known_code(const std::string& c) {
    return std::find(std::begin(kReasonCodes), std::end(kReasonCodes), c) != std::end(kReasonCodes);
}

// The gate's message and tag, written on CyphaLM's behalf (DISCOVERY 3.5). UNKNOWN_ENTITY's is the training rows' text.
std::string gate_message(const std::string& entity, const std::string& rel, const std::string& code) {
    if (code == "UNKNOWN_ENTITY") return dont_know_text(entity, rel, code);
    const std::string why =
        code == "NO_EVIDENCE" ? "I have read about " + entity + ", but not this." : std::string("I am not sure enough to answer.");
    return "I don't know. " + why + " <|unknown reason=" + code + " entity=\"" + entity + "\" relation=" + rel +
           " gap=" + gap_id(entity, rel) + "|>";
}

// What the system shows for one answer. A kept answer stands. Anything else says it does not know, with a code:
// CyphaLM's own message keeps its own code; when that message has no readable code, the gate's code completes it
// (by cyphalm+gate); an answer the gate does not keep, or no answer, becomes the gate's message (by gate). I3: whenever
// the gate does not keep an answer, the card is a don't-know with a known code (given a known gate_code).
struct Card {
    std::string kind, by, code, text;
};
Card final_card(const std::string& answer, bool kept, const std::string& gate_code, const std::string& entity,
                const std::string& rel) {
    if (is_dont_know(answer)) {
        const std::string c = tag_field(answer, "reason");
        if (known_code(c)) return {"dont_know", "cyphalm", c, answer};
        return {"dont_know", "cyphalm+gate", gate_code, answer};
    }
    if (kept && !answer.empty()) return {"answer", "cyphalm", "", answer};
    return {"dont_know", "gate", gate_code, gate_message(entity, rel, gate_code)};
}

// The reference code (grading only; slice meaning): not in the slice, UNKNOWN_ENTITY; in it, LOW_CERTAINTY when the fact
// is in what it read (a gold answer in the source article, or the shard's entity and relation), else NO_EVIDENCE.
std::string reference_code(int in_slice, int prose, int covered) {
    return !in_slice ? "UNKNOWN_ENTITY" : (prose == 1 || covered == 2) ? "LOW_CERTAINTY" : "NO_EVIDENCE";
}

// Copy expert (A1c, PLAN B9): the longest suffix of s[0..i), 3 to 64 bytes, that also ends at an earlier j inside the
// last 16,384 bytes of the current context (the prompt and the answer so far, or the stream being scored). It predicts
// s[j], from the most recent of the longest matches. Returns -1 (n = 0) when there is none. The window was 4,096 bytes
// until B4C (B4B chose 16,384; a Q/A prompt and its answer fit inside either).
constexpr std::size_t kCopyWin = 16384;
constexpr int kCopyBuckets = 8;
int copy_pred(const std::uint8_t* s, std::size_t i, int& n, std::size_t* at = nullptr) {  // DK2: at = the source j
    n = 0;
    int pred = -1;
    const std::size_t lo = i > kCopyWin ? i - kCopyWin : 0;
    for (std::size_t j = i; j-- > lo + 1;) {
        int L = 0;
        while (L < 64 && j - lo > static_cast<std::size_t>(L) && s[j - 1 - L] == s[i - 1 - L]) ++L;
        if (L > n) {
            n = L;
            pred = s[j];
            if (at) *at = j;
        }
    }
    if (n < 3) n = 0, pred = -1;
    return pred;
}
// DK2 (--copy-frame): the copy source j is the '|' of a "|>" and the answer s[answer_at, i) has no open "<|" (none, or
// the last one already closed). Copy then predicts nothing: a frame's entity is copied without the frame's end, which
// ended A1d's E1c messages ("I have nothing about character Spider-Man|>").
// DK2: where the answer s[answer_at, i) has an open "<|" (its last "<|", not closed by a later "|>"), else npos.
std::size_t open_token_at(const std::uint8_t* s, std::size_t i, std::size_t answer_at) {
    std::size_t open = std::string::npos, close = std::string::npos;
    for (std::size_t k = answer_at; k + 1 < i; ++k) {
        if (s[k] == '<' && s[k + 1] == '|') open = k;
        if (s[k] == '|' && s[k + 1] == '>') close = k;
    }
    return open != std::string::npos && (close == std::string::npos || close < open) ? open : std::string::npos;
}
bool copy_closes_unopened(const std::uint8_t* s, std::size_t i, std::size_t j, std::size_t answer_at) {
    return j + 1 < i && s[j] == '|' && s[j + 1] == '>' && open_token_at(s, i, answer_at) == std::string::npos;
}
int copy_bucket(int n) { return n == 0 ? 0 : n < 4 ? 1 : n < 6 ? 2 : n < 8 ? 3 : n < 12 ? 4 : n < 16 ? 5 : n < 32 ? 6 : 7; }
// --copy-min m (B4B): copy only from matches of at least m bytes; the lambdas of buckets starting below m become 0.
void copy_clip(double* lam, int m) {
    static const int lo[kCopyBuckets] = {0, 3, 4, 6, 8, 12, 16, 32};
    for (int b = 0; b < kCopyBuckets; ++b)
        if (lo[b] < m) lam[b] = 0;
}

// lam[b] in {0, 0.005, ..., 0.995} maximising the log-likelihood of lam pe + (1 - lam) pb over the items k < upto in
// bucket b (context8k's InfiniGram fit).
void fit_lambdas(const std::vector<double>& pe, const std::vector<double>& pb, const std::vector<int>& bk,
                 std::size_t upto, int nb, double* lam) {
    for (int b = 0; b < nb; ++b) {
        double best = -1e300;
        lam[b] = 0;
        for (int g = 0; g < 200; ++g) {
            const double l = g * 0.005;
            double ll = 0;
            bool any = false;
            for (std::size_t k = 0; k < upto; ++k)
                if (bk[k] == b) {
                    ll += std::log(l * pe[k] + (1 - l) * pb[k]);
                    any = true;
                }
            if (!any) break;
            if (ll > best) {
                best = ll;
                lam[b] = l;
            }
        }
    }
}

// B4C: one default expert mixed into cur, where cur[k] is a distribution's probability of the true byte data[train + k]:
// the expert's lambdas are fit on k < half over cur (the copy expert's then zeroed below a match of m bytes), and cur
// becomes lam pe + (1 - lam) cur. Returns the lambdas, one a bucket.
std::vector<double> mix_into(const std::vector<double>& pe, const std::vector<int>& bk, std::size_t half, int nb,
                             std::vector<double>& cur, int copy_m = -1) {
    std::vector<double> lam(static_cast<std::size_t>(nb), 0.0);
    fit_lambdas(pe, cur, bk, half, nb, lam.data());
    if (copy_m >= 0) copy_clip(lam.data(), copy_m);
    for (std::size_t k = 0; k < cur.size(); ++k) {
        const double l = lam[static_cast<std::size_t>(bk[k])];
        cur[k] = l * pe[k] + (1 - l) * cur[k];
    }
    return lam;
}
std::vector<double> mix_cls_into(const ClassContext& cc, const std::uint8_t* data, std::size_t train, std::size_t half,
                                 std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) {
        double P[256];
        bk[k] = ClassContext::bucket(cc.predict(data, train + k, P));
        pe[k] = P[data[train + k]];
    }
    return mix_into(pe, bk, half, ClassContext::kBuckets, cur);
}
std::vector<double> mix_copy_into(const std::uint8_t* data, std::size_t train, std::size_t half, int m,
                                  std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) {
        int n = 0;
        pe[k] = copy_pred(data, train + k, n) == data[train + k] ? 1.0 : 0.0;
        bk[k] = copy_bucket(n);
    }
    return mix_into(pe, bk, half, kCopyBuckets, cur, m);
}
// GEN-E: the content-plan expert's predictions over the holdout text[train, train + cur.size()): every paragraph line
// that could be a prompt (good_start) is planned from its first 64 bytes, as cmd_sample plans a prompt, and the expert
// reads the rest of the line; elsewhere it predicts nothing (bucket 0). Lambdas fit on k < half; cur becomes the mix.
// Prints the plan's coverage of those lines, the lambdas, and the bits per byte on k >= half before and after.
std::vector<double> mix_plan_into(const ContentPlanStore& st, const std::map<std::string, std::vector<std::string>>& frames,
                                  const std::uint8_t* data, std::size_t train, std::size_t half, std::vector<double>& cur) {
    std::vector<double> pe(cur.size(), 0.0);
    std::vector<int> bk(cur.size(), 0);
    const std::string hold(reinterpret_cast<const char*>(data) + train, cur.size());
    std::size_t lines = 0, planned_lines = 0, aligned = 0, right = 0;
    std::size_t pos = hold.find('\n');
    pos = pos == std::string::npos ? hold.size() : pos + 1;
    while (pos < hold.size()) {
        const std::size_t e = hold.find('\n', pos);
        if (e == std::string::npos) break;
        const std::string line = hold.substr(pos, e - pos);
        const std::size_t ls = pos;
        pos = e + 1;
        if (!good_start(line)) continue;
        ++lines;
        std::size_t cut = 64;
        while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;
        const auto pl = st.plan(line.substr(0, cut), frames);
        if (pl.facts.empty()) continue;
        ++planned_lines;
        std::string folded;
        std::vector<int> active;
        for (std::size_t q = ls + cut; q < e; ++q) {
            active.clear();
            for (std::size_t f = 0; f < pl.facts.size(); ++f)
                if (!ContentPlanStore::expressed(pl, f, folded)) active.push_back(static_cast<int>(f));
            double P[256];
            const int L = ContentPlanStore::predict(pl, active, data, train + q, P);
            bk[q] = ContentPlanStore::bucket(L);
            pe[q] = P[data[train + q]];
            aligned += L > 0;
            right += L > 0 && P[data[train + q]] > 0.5;
            ContentPlanStore::fold_into(folded, data[train + q]);
        }
    }
    auto bpc = [&](std::size_t from) {
        double s = 0;
        for (std::size_t k = from; k < cur.size(); ++k) s -= std::log2(cur[k]);
        return cur.size() > from ? s / static_cast<double>(cur.size() - from) : 0.0;
    };
    const double before = bpc(half);
    const auto lam = mix_into(pe, bk, half, ContentPlanStore::kBuckets, cur);
    std::cerr << "plan holdout paragraph lines " << lines << " planned " << planned_lines << " aligned bytes " << aligned
              << " right " << right << "\nplan lambdas";
    for (double l : lam) std::cerr << " " << l;
    std::cerr << "\nplan holdout second-half bpc without " << before << " with " << bpc(half) << "\n";
    return lam;
}

// BKN: the packed byte Kneser-Ney from a spec (bkn_spec), built on t[0, n); nullptr (and a message) for a bad spec.
std::unique_ptr<cypha::cyphalm::bkn::ByteKN> make_bkn(const std::string& spec, const std::uint8_t* t, std::size_t n) {
    const std::vector<int> v = bkn_spec(spec);
    if (v.empty()) {
        std::cerr << "bad --bkn spec " << spec << " (K[:m:c:d])\n";
        return nullptr;
    }
    cypha::cyphalm::bkn::Prune pr;
    pr.from_order = v[1], pr.min_ctx = static_cast<std::uint32_t>(v[2]), pr.drop_singletons = v[3] != 0;
    return std::make_unique<cypha::cyphalm::bkn::ByteKN>(t, n, v[0], pr);
}
// BKN into cur (the trie mixture's probability of each holdout byte): (a) cur becomes Kneser-Ney's; (b) Kneser-Ney is
// mixed in by lambda per (deepest order found x count band), fit on k < half. Returns (b)'s lambdas (empty for (a)).
std::vector<double> mix_bkn_into(const cypha::cyphalm::bkn::ByteKN& kn, const std::uint8_t* data, std::size_t train,
                                 std::size_t half, bool replace, std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) pe[k] = kn.prob_bucket(data, train + k, data[train + k], bk[k]);
    if (replace) {
        cur.swap(pe);
        return {};
    }
    return mix_into(pe, bk, half, kn.buckets(), cur);
}
// BKN into a next-byte distribution P (the trie mixture's), as mix_bkn_into fit it.
void bkn_mix(const cypha::cyphalm::bkn::ByteKN& kn, const double* lkn, bool replace, const std::uint8_t* seq,
             std::size_t len, double* P) {
    double Pk[256];
    int band = 0;
    const int d = kn.dist(seq, len, Pk, &band);
    if (replace) {
        std::memcpy(P, Pk, sizeof Pk);
        return;
    }
    const double l = lkn[d == 0 ? 0 : (d - 1) * 3 + band];
    for (int s = 0; s < 256; ++s) P[s] = l * Pk[s] + (1 - l) * P[s];
}

// B4: the class-context expert from "TABLE[:tail[:orders[:flags]]]": TABLE a word<TAB>class file, or "shape" (shape
// classes only). Defaults (Odin, 2026-10-28): tail 6, orders 3, a compact table with contexts seen once dropped, so
// "shape" is shape:6:3:cp2. A flags field states the storage in full: c (compact table), pN (contexts seen fewer
// than N times dropped), f (neither: the full table); the earlier default is shape:6:2:f. The fields are read from the
// right, so a Windows path keeps its drive colon. Not built yet (call build()). nullptr (and a message) for a bad spec
// or table.
std::unique_ptr<ClassContext> make_class_context(const std::string& spec) {
    const auto f = split(spec, ':');
    auto digits = [](const std::string& x) {
        return !x.empty() && x.size() < 6 && std::all_of(x.begin(), x.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    ClassContext::Options o;
    auto flags = [&](const std::string& x) {  // a flags field states the storage in full, from the full table up
        bool any = false;
        o.compact = false, o.min_total = 1;
        for (std::size_t i = 0; i < x.size();) {
            if (x[i] == 'c' && !o.compact) o.compact = true, ++i;
            else if (x[i] == 'f' && i == 0 && x.size() == 1) ++i;
            else if (x[i] == 'p' && i + 1 < x.size() && std::isdigit(static_cast<unsigned char>(x[i + 1]))) {
                std::size_t j = i + 1;
                while (j < x.size() && std::isdigit(static_cast<unsigned char>(x[j]))) ++j;
                o.min_total = static_cast<std::uint32_t>(std::atoi(x.substr(i + 1, j - i - 1).c_str()));
                i = j;
            } else return false;
            any = true;
        }
        return any;
    };
    std::size_t k = f.size();  // f[0..k) is the table
    if (k >= 4 && digits(f[k - 3]) && digits(f[k - 2]) && flags(f[k - 1])) {
        o.tail = std::atoi(f[k - 3].c_str()), o.orders = std::atoi(f[k - 2].c_str()), k -= 3;
    } else {
        o = ClassContext::Options{};
        if (k >= 3 && digits(f[k - 2]) && digits(f[k - 1])) o.tail = std::atoi(f[k - 2].c_str()), o.orders = std::atoi(f[k - 1].c_str()), k -= 2;
        else if (k >= 2 && digits(f[k - 1])) o.tail = std::atoi(f[k - 1].c_str()), k -= 1;
    }
    std::string table = f[0];
    for (std::size_t i = 1; i < k; ++i) table += ":" + f[i];
    if (o.orders != 2 && o.orders != 3) {
        std::cerr << "class spec " << spec << ": orders must be 2 or 3\n";
        return nullptr;
    }
    std::unordered_map<std::string, std::uint16_t> cls;
    if (table != "shape") {
        std::string err;
        cls = ClassContext::load_tsv(table, err);
        if (!err.empty()) {
            std::cerr << err << "\n";
            return nullptr;
        }
    }
    return std::make_unique<ClassContext>(std::move(cls), o);
}

// B4C: the class-context expert where it is a default (qa, context8k; sample only with --cls): --cls SPEC, else
// `shape`. Built on text[0, n). nullptr with rc 0 under --no-cls; nullptr with rc 1 for a bad spec.
std::string default_cls_spec(const Args& a) { return a.no_cls ? std::string() : a.cls.empty() ? std::string("shape") : a.cls; }
std::unique_ptr<ClassContext> default_cls(const Args& a, const std::uint8_t* text, std::size_t n, int& rc) {
    rc = 0;
    const std::string spec = default_cls_spec(a);
    if (spec.empty()) return nullptr;
    auto cc = make_class_context(spec);
    if (!cc) {
        rc = 1;
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    cc->build(text, n);
    std::cerr << "class expert " << spec << " contexts " << cc->contexts() << " table_bytes " << cc->bytes() << " build_s "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return cc;
}

// A4: the similar-context expert from "vec=PREFIX[,W=8][,k=32][,T=0.05][,min=-1][,sif=1][,dim=64][,drop=1][,lists=512]
// [,probe=8][,tail=32][,src=all][,gate=1]". PREFIX.vectors.f32 and PREFIX.vocab.tsv (ppmi_svd's output), or a projected
// file (simvec's output). qa's src: all = wiki train + chat, wiki = wiki train only. gate=1 (default): the expert is off
// inside an answer (in_answer). The token `default` stands for kSimDefault; with no vec= the shipped file is read
// (sim_vectors_path). `build` names what the index depends on: specs that differ only in k, T, min, probe, tail or gate
// share it. A4c: [,mode=knn|dedupe] (similar_context.hpp; in dedupe the tail depth is fixed at the build). A4d: [,online=1]
// (keys from the text being read; a query setting).
#ifndef CYPHA_SIMVEC_DEFAULT
#define CYPHA_SIMVEC_DEFAULT "data/cyphalm/simvec_ppmi32.bin"
#endif
constexpr const char* kSimDefault = "W=2,sif=0,k=8192,drop=0,lists=2048,probe=128,src=wiki,mode=dedupe,online=1";  // A4c, A4d
std::string sim_vectors_path() { return env_or("CYPHA_SIMVEC", CYPHA_SIMVEC_DEFAULT); }
struct SimSpec {
    SimilarContext::Params p;
    std::string vec, src = "all", build;
    bool gate = true, ok = false;
};
SimSpec parse_sim(const std::string& spec) {
    SimSpec s;
    std::vector<std::string> kvs;
    for (const auto& kv : split(spec, ','))
        if (kv == "default")
            for (const auto& d : split(kSimDefault, ',')) kvs.push_back(d);
        else
            kvs.push_back(kv);
    for (const auto& kv : kvs) {
        const std::size_t eq = kv.find('=');
        if (eq == std::string::npos) return s;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        const double x = std::atof(v.c_str());
        if (k == "vec") s.vec = v;
        else if (k == "src") s.src = v;
        else if (k == "W") s.p.W = static_cast<int>(x);
        else if (k == "k") s.p.k = static_cast<int>(x);
        else if (k == "T") s.p.temp = x;
        else if (k == "min") s.p.min_sim = x;
        else if (k == "sif") s.p.sif = x != 0;
        else if (k == "dim") s.p.dim = static_cast<int>(x);
        else if (k == "drop") s.p.drop = static_cast<int>(x);
        else if (k == "lists") s.p.lists = static_cast<int>(x);
        else if (k == "probe") s.p.probe = static_cast<int>(x);
        else if (k == "tail") s.p.max_tail = static_cast<int>(x);
        else if (k == "gate") s.gate = x != 0;
        else if (k == "mode" && (v == "knn" || v == "dedupe")) s.p.mode = v == "dedupe";  // A4c
        else if (k == "online") s.p.online = x != 0;  // A4d: keys from the text being read; a query setting
        else return s;
    }
    if (s.vec.empty()) s.vec = sim_vectors_path();
    s.build = s.vec + "|" + s.src + "|" + std::to_string(s.p.W) + "|" + std::to_string(s.p.sif) + "|" +
              std::to_string(s.p.dim) + "|" + std::to_string(s.p.drop) + "|" + std::to_string(s.p.lists) + "|" +
              std::to_string(s.p.mode) + "|" + std::to_string(s.p.mode ? s.p.max_tail : 0);  // A4c: groups sort by the tail
    s.ok = !s.vec.empty() && (s.src == "all" || s.src == "wiki");
    return s;
}
std::unique_ptr<SimilarContext> make_similar_context(const SimSpec& s) {
    std::string err;
    auto V = std::ifstream(s.vec, std::ios::binary) ? SimilarContext::load_projected(s.vec, err)
                                                     : SimilarContext::load(s.vec + ".vectors.f32", s.vec + ".vocab.tsv", err);
    if (!err.empty()) {
        std::cerr << err << "\n";
        return nullptr;
    }
    return std::make_unique<SimilarContext>(std::move(V), s.p);
}

// A4: the similar-context expert where it is a default (qa builds its own; sample, context8k): the first --sim spec or
// `default`, keys over text[0, n). nullptr with rc 0 under --no-sim; nullptr with rc 1 when it cannot be built.
std::unique_ptr<SimilarContext> default_sim(const Args& a, const std::uint8_t* text, std::size_t n, int& rc) {
    rc = 0;
    if (a.no_sim) return nullptr;
    const std::string spec = a.sims.empty() ? "default" : a.sims[0];
    const SimSpec ss = parse_sim(spec);
    std::unique_ptr<SimilarContext> sc = ss.ok ? make_similar_context(ss) : nullptr;
    if (!sc) {
        std::cerr << "similar-context expert: cannot build '" << spec << "' (vectors " << ss.vec
                  << "): run from the repository root, set CYPHA_SIMVEC, or pass --no-sim\n";
        rc = 1;
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    sc->build(text, n);
    std::cerr << "sim " << spec << " keys " << sc->keys() << " bytes " << sc->bytes() << " build_s "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return sc;
}

// GEN-C (PLAN B5): --doc SPEC, the document experts in sample. SPEC is "default" (trig,cache: the setting chosen on
// dev, docs/reports/GEN_TRACK.md "GEN-C") or a comma list of parts (trig, cache, topic), frozen (in sample the document
// state is the prompt's alone; the holdout fit is unchanged) and settings: cand=N (candidate
// words a part), per=N (triggers a word), stop=N (stop list size), mindf=N, minco=N, words=N (content words a training
// document gives to pairs), topk=N, topict=X (topic temperature), every=N (topic refresh). The parts are built over
// the train bytes; the topic part reads the similar-context expert's projected vectors (sim_vectors_path()).
std::unique_ptr<DocTopic> make_doc(const std::string& doc, const std::uint8_t* text, std::size_t n, int& rc) {
    rc = 0;
    if (doc.empty()) return nullptr;
    DocTopic::Params p;
    p.trig = p.cache = p.topic = false;
    const std::string spec = doc == "default" ? std::string("trig,cache") : doc;
    for (const auto& f : split(spec, ',')) {
        const std::size_t eq = f.find('=');
        const std::string k = f.substr(0, eq), v = eq == std::string::npos ? "" : f.substr(eq + 1);
        if (k == "trig") p.trig = true;
        else if (k == "cache") p.cache = true;
        else if (k == "topic") p.topic = true;
        else if (k == "frozen") p.frozen = true;
        else if (k == "cand") p.cand = std::atoi(v.c_str());
        else if (k == "per") p.per_source = std::atoi(v.c_str());
        else if (k == "stop") p.stop_top = std::atoi(v.c_str());
        else if (k == "mindf") p.min_df = std::atoi(v.c_str());
        else if (k == "minco") p.min_co = std::atoi(v.c_str());
        else if (k == "words") p.max_doc_words = std::atoi(v.c_str());
        else if (k == "topk") p.topic_k = std::atoi(v.c_str());
        else if (k == "topict") p.topic_temp = std::atof(v.c_str());
        else if (k == "every") p.topic_every = std::atoi(v.c_str());
        else {
            std::cerr << "--doc: unknown field '" << f << "'\n";
            rc = 2;
            return nullptr;
        }
    }
    if (!(p.trig || p.cache || p.topic) || p.cand < 1 || p.per_source < 1 || p.topic_k < 1 || !(p.topic_temp > 0) ||
        p.topic_every < 1) {
        std::cerr << "--doc " << doc << ": no part, or a bad setting\n";
        rc = 2;
        return nullptr;
    }
    auto dt = std::make_unique<DocTopic>(p);
    if (p.topic) {
        std::string err;
        const SimilarContext::Vectors V = SimilarContext::load_projected(sim_vectors_path(), err);
        if (!err.empty() || !V.projected()) {
            std::cerr << "--doc topic: " << (err.empty() ? std::string("no projected vectors") : err) << "\n";
            rc = 1;
            return nullptr;
        }
        std::unordered_map<std::string, std::vector<float>> rows;
        for (const auto& [word, r] : V.id) {
            std::vector<float> x(static_cast<std::size_t>(V.d));
            for (int d = 0; d < V.d; ++d)
                x[static_cast<std::size_t>(d)] = V.scale[static_cast<std::size_t>(r)] * V.q[static_cast<std::size_t>(r) * V.d + d];
            rows.emplace(DocTopic::lower(word), std::move(x));
        }
        dt->set_vectors(V.d, std::move(rows));
    }
    const auto t0 = std::chrono::steady_clock::now();
    dt->build(text, n);
    std::cerr << "doc " << spec << " docs " << dt->docs() << " vocab " << dt->vocab() << " trigger_pairs "
              << dt->trigger_pairs() << " topic_words " << dt->topic_words() << " build_s "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return dt;
}

// GEN-C: each part's next-byte distribution over the holdout (one stream from the train end, reset at "<page>"),
// its lambdas fit on k < half over cur, cur mixed with it, in DocTopic's part order. The read: bits a byte on
// [half, cur.size()) before and after, with a bootstrap over 16 KiB blocks, and split by the byte's offset in its
// document (under and over 4 KiB). Printed to stderr with a DOC prefix.
std::vector<std::vector<double>> fit_doc(const DocTopic& dt, const std::uint8_t* data, std::size_t train, std::size_t half,
                                         std::vector<double>& cur) {
    const std::size_t span = cur.size();
    const auto& p = dt.params();
    const bool on[DocTopic::kParts] = {p.trig, p.cache, p.topic};
    std::vector<double> pe[DocTopic::kParts];
    std::vector<int> bk[DocTopic::kParts];
    for (int q = 0; q < DocTopic::kParts; ++q)
        if (on[q]) pe[q].resize(span), bk[q].resize(span);
    std::vector<std::uint32_t> off(span);
    DocTopic::Stream st(dt);
    std::size_t page = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < span; ++k) {
        const std::size_t i = train + k;
        for (int q = 0; q < DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                bk[q][k] = st.predict(q, P);
                pe[q][k] = P[data[i]];
            }
        if (k >= 6 && std::memcmp(data + i - 6, "<page>", 6) == 0) page = k - 6;
        off[k] = static_cast<std::uint32_t>(std::min<std::size_t>(k - page, 0xffffffffu));
        st.feed(data[i]);
    }
    const std::vector<double> before = cur;
    std::vector<std::vector<double>> lam(DocTopic::kParts);
    for (int q = 0; q < DocTopic::kParts; ++q)
        if (on[q]) lam[static_cast<std::size_t>(q)] = mix_into(pe[q], bk[q], half, DocTopic::kBuckets, cur);
    static const char* const name[DocTopic::kParts] = {"trig", "cache", "topic"};
    for (int q = 0; q < DocTopic::kParts; ++q) {
        if (!on[q]) continue;
        std::cerr << "DOC lambdas " << name[q];
        for (double l : lam[static_cast<std::size_t>(q)]) std::cerr << " " << l;
        std::cerr << "\n";
    }
    if (span > half) {
        const std::size_t B = 16384;
        std::vector<double> diff;  // mean bits a byte, after minus before, a block
        double sb = 0, sa = 0, lb[2] = {0, 0}, la[2] = {0, 0}, ln[2] = {0, 0};
        for (std::size_t s0 = half; s0 < span; s0 += B) {
            const std::size_t s1 = std::min(span, s0 + B);
            double d = 0;
            for (std::size_t k = s0; k < s1; ++k) {
                const double b0 = -std::log2(before[k]), b1 = -std::log2(cur[k]);
                sb += b0, sa += b1, d += b1 - b0;
                const int g = off[k] >= 4096;
                lb[g] += b0, la[g] += b1, ln[g] += 1;
            }
            diff.push_back(d / static_cast<double>(s1 - s0));
        }
        const double n = static_cast<double>(span - half);
        std::mt19937_64 rng(20261029);
        std::vector<double> bs(2000);
        for (double& x : bs) {
            double m = 0;
            for (std::size_t j = 0; j < diff.size(); ++j) m += diff[rng() % diff.size()];
            x = m / static_cast<double>(diff.size());
        }
        std::sort(bs.begin(), bs.end());
        double mean = 0;
        for (double d : diff) mean += d;
        mean /= static_cast<double>(diff.size());
        std::fprintf(stderr, "DOC read [half, end) bytes %.0f bpc before %.6f after %.6f diff %.6f ci95 %.6f %.6f blocks %zu\n",
                     n, sb / n, sa / n, mean, bs[50], bs[1949], diff.size());
        for (int g = 0; g < 2; ++g)
            if (ln[g] > 0)
                std::fprintf(stderr, "DOC read offset %s 4KiB bytes %.0f bpc before %.6f after %.6f diff %.6f\n",
                             g ? ">=" : "<", ln[g], lb[g] / ln[g], la[g] / ln[g], (la[g] - lb[g]) / ln[g]);
    }
    std::cerr << "DOC fit_s " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return lam;
}

// GEN-C2: fit_doc's mixture in less memory (context8k, qa's holdout). One stream from the train end; each part's
// predictions are kept over the first half only, where its lambdas are fit (over cur, in DocTopic's part order) and it
// is mixed; the second half is mixed as the stream reads it. The same numbers as fit_doc; no DOC lines.
void mix_doc_into(const DocTopic& dt, const std::uint8_t* data, std::size_t train, std::size_t half, std::vector<double>& cur) {
    const std::size_t span = cur.size(), h = std::min(half, span);
    const auto& p = dt.params();
    const bool on[DocTopic::kParts] = {p.trig, p.cache, p.topic};
    std::vector<double> pe[DocTopic::kParts];
    std::vector<int> bk[DocTopic::kParts];
    for (int q = 0; q < DocTopic::kParts; ++q)
        if (on[q]) pe[q].resize(h), bk[q].resize(h);
    DocTopic::Stream st(dt);
    for (std::size_t k = 0; k < h; ++k) {
        for (int q = 0; q < DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                bk[q][k] = st.predict(q, P);
                pe[q][k] = P[data[train + k]];
            }
        st.feed(data[train + k]);
    }
    double lam[DocTopic::kParts][DocTopic::kBuckets] = {};
    for (int q = 0; q < DocTopic::kParts; ++q) {
        if (!on[q]) continue;
        fit_lambdas(pe[q], cur, bk[q], h, DocTopic::kBuckets, lam[q]);
        for (std::size_t k = 0; k < h; ++k) cur[k] = lam[q][bk[q][k]] * pe[q][k] + (1 - lam[q][bk[q][k]]) * cur[k];
        std::vector<double>().swap(pe[q]);
        std::vector<int>().swap(bk[q]);
    }
    for (std::size_t k = h; k < span; ++k) {
        for (int q = 0; q < DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                const int b = st.predict(q, P);
                cur[k] = lam[q][b] * P[data[train + k]] + (1 - lam[q][b]) * cur[k];
            }
        st.feed(data[train + k]);
    }
}

// simvec [wiki] --sim SPEC --out FILE (A4): the expert of SPEC built over the train bytes; its projected word rows
// written as the file the default reads.
int cmd_simvec(const Args& a) {
    if (a.out.empty() || a.sims.empty()) {
        std::cerr << "usage: cyphalm_trie simvec <wiki> --sim vec=PREFIX[,...] --out FILE\n";
        return 2;
    }
    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;
    const SimSpec ss = parse_sim(a.sims[0]);
    auto sc = ss.ok ? make_similar_context(ss) : nullptr;
    if (!sc) {
        std::cerr << "bad --sim spec " << a.sims[0] << "\n";
        return 1;
    }
    sc->build(data.data(), a.cfg.train);
    std::ofstream out(a.out, std::ios::binary);
    if (!out || !sc->save_projected(out)) {
        std::cerr << "cannot write " << a.out << "\n";
        return 1;
    }
    out.close();
    std::cout << "simvec " << a.sims[0] << " keys " << sc->keys() << " dim " << sc->dim() << " file_bytes "
              << std::ifstream(a.out, std::ios::binary | std::ios::ate).tellg() << "\n";
    return 0;
}

// A4's gate: is the next byte after seq inside an answer? Yes from `answer_at` on (the end of decode's prompt: the
// question in either prompt form), and after the latest "Assistant: " when no "User: " follows it (a chat turn runs
// until the next user turn). The look back stops at 8 KiB.
bool in_answer(const std::vector<std::uint8_t>& seq, std::size_t answer_at) {
    if (seq.size() >= answer_at) return true;
    const std::size_t lo = seq.size() > 8192 ? seq.size() - 8192 : 0;
    for (std::size_t i = seq.size(); i-- > lo + 1;) {
        if (seq[i] != ' ' || seq[i - 1] != ':') continue;
        if (i >= lo + 10 && std::memcmp(&seq[i - 10], "Assistant: ", 11) == 0) return true;
        if (i >= lo + 5 && std::memcmp(&seq[i - 5], "User: ", 6) == 0) return false;
    }
    return false;
}

// CyphaLM's next-byte distribution: the frozen trie mixture, optionally mixed with InfiniGram
// (context8k: P = lam[b] P_ctx + (1 - lam[b]) P_trie, one lam per match-length x count bucket),
// then optionally with the copy expert (P = lc[b] [pred] + (1 - lc[b]) P, one lc per match-length bucket).
// B4C: the class expert (cls_gate) and the default copy expert (lc_out) are off inside an answer, as the
// similar-context expert is (sim_gate). lc is the copy expert inside answers too (qa's --copy models, A1c); decode's
// no-repeat rule reads lc, so the gated copy has its own pointer.
struct Lm {
    const Model* model = nullptr;  // the training form (qa's wiki stage); not read when packed is set
    const double* w = nullptr;
    std::size_t follower_limit = 0;
    bool online_match = false;
    const FmInfiniGram* ig = nullptr;
    const double* lam = nullptr;
    const double* lc = nullptr;
    int cap = 8192;
    const ClassContext* cc = nullptr;  // B4: class-context expert, P = lcls[b] P_cls + (1 - lcls[b]) P, before the copy
    const double* lcls = nullptr;
    const SimilarContext* sc = nullptr;  // A4: similar-context expert, P = lsim[b] P_sim + (1 - lsim[b]) P, after the class
    const double* lsim = nullptr;        // expert and before the copy
    const PackedModel* packed = nullptr;  // M4: the serving form
    mutable FmInfiniGram::Cursor cur{};   // M5: InfiniGram's last match, for the next byte of the same text
    bool sim_gate = false;                // A4: the similar-context expert off inside an answer (in_answer)
    std::size_t answer_at = std::string::npos;  // A4: where the answer starts (decode sets it to the prompt's end)
    bool cls_gate = false;                // B4C: the class expert off inside an answer
    const double* lc_out = nullptr;       // B4C: the copy expert's lambdas, outside answers only (when lc is null)
    const cypha::cyphalm::bkn::ByteKN* kn = nullptr;  // BKN: byte Kneser-Ney, (a) replacing or (b) mixed into the trie
    const double* lkn = nullptr;                      // mixture by lkn per bucket (mix_bkn_into)
    bool kn_replace = false;
    bool kn_gate = false;                 // BKN3: Kneser-Ney off inside an answer (in_answer), as the other experts
    const double* lam_ans = nullptr;      // BKN3: InfiniGram's lambdas fit without Kneser-Ney, used where it is off
    const double* lc_dk = nullptr;        // ANS: the copy expert's lambdas inside a don't-know answer (--copy-dk)
    bool copy_frame = false;              // DK2: --copy-frame inside a don't-know answer (copy_closes_unopened; decode)
    // ANS: the next byte is inside a don't-know answer: the answer (from answer_at) begins "I don't know", the switch
    // decode already makes to its longer limit. Only with lc_dk; never outside an answer.
    bool in_dont_know(const std::vector<std::uint8_t>& seq) const {
        return lc_dk && answer_at != std::string::npos && seq.size() >= answer_at + 12 &&
               std::memcmp(seq.data() + answer_at, "I don't know", 12) == 0;
    }
    // hint: the previous step's match length (-1 at a new sequence); n_out: this step's.
    void dist(const std::vector<std::uint8_t>& seq, int& hint, double* P, int* n_out = nullptr) const {
        double e[5][256];
        bool m = false;
        unsigned pred = 0;
        if (packed) packed->fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, m, pred, online_match);
        else model->fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, m, pred, online_match);
        mix_dist(w, e, m, pred, P);
        const bool kn_off = kn && kn_gate && in_answer(seq, answer_at);  // BKN3: the experts' gate
        if (kn && !kn_off) bkn_mix(*kn, lkn, kn_replace, seq.data(), seq.size(), P);  // BKN
        if (ig) {
            const std::size_t len = std::min<std::size_t>(seq.size(), static_cast<std::size_t>(cap));
            const auto r =
                ig->query(seq.data() + (seq.size() - len), len, cap, hint < 0 ? -1 : std::min(hint + 1, cap), 1, cur);
            hint = r.n;
            if (n_out) *n_out = r.n;
            const double l = (kn_off && lam_ans ? lam_ans : lam)[bucket(r.n, r.total)];  // BKN3
            for (int s = 0; s < 256; ++s) {
                const double pc = r.total ? static_cast<double>(r.count[s]) / static_cast<double>(r.total) : 0.0;
                P[s] = l * pc + (1 - l) * P[s];
            }
        }
        const bool ans = ((cc && cls_gate) || (sc && sim_gate) || (lc_out && !lc)) && in_answer(seq, answer_at);
        if (cc && !(cls_gate && ans)) {
            double Pc[256];
            const double l = lcls[ClassContext::bucket(cc->predict(seq.data(), seq.size(), Pc))];
            for (int s = 0; s < 256; ++s) P[s] = l * Pc[s] + (1 - l) * P[s];
        }
        if (sc && !(sim_gate && ans)) {
            double Ps[256];
            const double l = lsim[SimilarContext::bucket(sc->predict(seq.data(), seq.size(), Ps))];
            for (int s = 0; s < 256; ++s) P[s] = l * Ps[s] + (1 - l) * P[s];
        }
        const bool idk = in_dont_know(seq);
        const double* lcp = idk ? lc_dk : lc ? lc : ans ? nullptr : lc_out;
        int n = 0;
        std::size_t cj = 0;
        int c = lcp ? copy_pred(seq.data(), seq.size(), n, &cj) : -1;
        if (c >= 0 && idk && copy_frame && copy_closes_unopened(seq.data(), seq.size(), cj, answer_at)) c = -1;  // DK2
        if (c < 0) return;
        const double l = lcp[copy_bucket(n)];
        for (int s = 0; s < 256; ++s) P[s] *= 1 - l;
        P[c] += l;
    }
};

// ---------------------------------------------------------------- cls (B4: the class-context expert)
//
// The bpc model (train bytes only) and InfiniGram over the train bytes; then, for each spec, the class-context expert
// built on the train bytes. Three reads, every one paired against the same model without the expert:
//   holdout  second half of the holdout, lambdas (InfiniGram's and the expert's) fit on the first half;
//   fresh    [sample, sample + 1 MiB) of the corpus, lambdas refit on the whole holdout; the read that decides;
//   surface  S = H*D*L (scribe.hpp) of 400 bytes from each of 200 holdout prompts (sample's prompt rule and seed),
//            T 0.8, nucleus 0.9, no-repeat, the same seed for every model; D and L referenced to the human
//            continuations; the trie alone and the trie + expert (lambdas on the first holdout half).
// Intervals are block bootstraps of the mean gain: 4 KiB blocks (2,000 resamples on the holdout, 4,000 on fresh text),
// prompts on the surface score (4,000).

struct Boot {
    double mean, lo, hi, p_neg, p_pos;
};
Boot block_boot(const std::vector<double>& d, std::size_t block, int reps, std::uint64_t seed) {
    const std::size_t nb = d.size() / block;
    std::vector<double> bm(nb), bs;
    for (std::size_t b = 0; b < nb; ++b)
        bm[b] = std::accumulate(d.begin() + static_cast<std::ptrdiff_t>(b * block),
                                d.begin() + static_cast<std::ptrdiff_t>((b + 1) * block), 0.0) / static_cast<double>(block);
    std::mt19937_64 rng(seed);
    for (int r = 0; r < reps; ++r) {
        double t = 0;
        for (std::size_t b = 0; b < nb; ++b) t += bm[rng() % nb];
        bs.push_back(t / static_cast<double>(nb));
    }
    std::sort(bs.begin(), bs.end());
    double neg = 0, pos = 0;
    for (double x : bs) neg += x < 0, pos += x > 0;
    return {std::accumulate(d.begin(), d.end(), 0.0) / static_cast<double>(d.size()), bs[reps / 40],
            bs[reps - reps / 40 - 1], neg / reps, pos / reps};
}

void print_boot(const std::string& name, const Boot& b) {
    std::printf("%s mean %.6f ci95 %.6f %.6f p_lt0 %.4f p_gt0 %.4f\n", name.c_str(), b.mean, b.lo, b.hi, b.p_neg, b.p_pos);
}

// The next-byte distribution of sampling's mixture after seq: trie, then byte Kneser-Ney (BKN), class, similar-context
// and copy where given (GEN-A: shared by sample_with_class and the scoring of human continuations).
void sample_mix(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc, const double* lam,
                const std::vector<std::uint8_t>& seq, const SimilarContext* sc, const double* lsm, const double* lcp,
                double M[256], const cypha::cyphalm::bkn::ByteKN* kn, const double* lkn, bool kn_replace) {
    double e[5][256];
    bool matched = false;
    unsigned pred = 0;
    model.fill_experts(seq.data(), seq.size(), limit, Mode::Synth, e, matched, pred, false);
    mix_dist(w, e, matched, pred, M);
    if (kn) bkn_mix(*kn, lkn, kn_replace, seq.data(), seq.size(), M);  // BKN: first after the trie, as qa and the fit
    if (cc) {
        double P[256];
        const double l = lam[ClassContext::bucket(cc->predict(seq.data(), seq.size(), P))];
        for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
    }
    if (sc) {  // A4
        double P[256];
        const double l = lsm[SimilarContext::bucket(sc->predict(seq.data(), seq.size(), P))];
        for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
    }
    int cn = 0;
    const int c = lcp ? copy_pred(seq.data(), seq.size(), cn) : -1;
    if (c >= 0) {  // B4C: the copy expert, mixed last
        const double l = lcp[copy_bucket(cn)];
        for (int s = 0; s < 256; ++s) M[s] *= 1 - l;
        M[c] += l;
    }
}

// One sampled byte from M: no 11-byte repeats (unless every byte is banned), temperature, nucleus 0.9, one draw.
int pick_byte(const double M[256], const std::vector<std::uint8_t>& seq, double temperature, std::mt19937& rng,
              std::uniform_real_distribution<double>& U) {
    bool banned[256];
    mark_banned(seq, banned);
    double p[256], mass = 0;
    for (int pass = 0; pass < 2 && !(mass > 0); ++pass) {
        mass = 0;
        for (int s = 0; s < 256; ++s) mass += p[s] = (pass == 0 && banned[s]) ? 0.0 : std::pow(M[s], 1.0 / temperature);
    }
    int order[256];
    std::iota(order, order + 256, 0);
    std::sort(order, order + 256, [&](int x, int y) { return p[x] > p[y]; });
    double acc = 0, km = 0;
    int keep = 1;
    for (int k = 0; k < 256; ++k) {
        acc += p[order[k]] / mass;
        keep = k + 1;
        if (acc >= 0.9) break;
    }
    for (int k = 0; k < keep; ++k) km += p[order[k]];
    double r = U(rng) * km;
    int pick = order[keep - 1];
    for (int k = 0; k < keep; ++k)
        if ((r -= p[order[k]]) <= 0) {
            pick = order[k];
            break;
        }
    return pick;
}

std::string sample_with_class(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                              const double* lam, const std::string& prompt, int n, std::uint32_t seed,
                              const SimilarContext* sc = nullptr, const double* lsm = nullptr, double temperature = 0.8,
                              const double* lcp = nullptr, const DocTopic* dt,
                              const std::vector<std::vector<double>>* ldoc, const ContentPlanStore::Plan* pl,
                              const double* lpl, const cypha::cyphalm::bkn::ByteKN* kn, const double* lkn,
                              bool kn_replace) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::string gen_folded;  // GEN-E: the generated text, folded, for the plan's "expressed" test
    std::vector<int> active;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::unique_ptr<DocTopic::Stream> ds;  // GEN-C: the document experts read the prompt and what is written
    if (dt && ldoc) {
        ds = std::make_unique<DocTopic::Stream>(*dt);
        ds->feed(seq.data(), seq.size());
        if (dt->params().frozen) ds->freeze();  // GEN-C amendment: the prompt alone is the document state
    }
    for (int t = 0; t < n; ++t) {
        double M[256];
        sample_mix(model, w, limit, cc, lam, seq, sc, lsm, lcp, M, kn, lkn, kn_replace);  // BKN: Kneser-Ney inside
        if (ds)  // GEN-C: the document experts, mixed after the copy expert, in DocTopic's part order
            for (int part = 0; part < DocTopic::kParts; ++part) {
                const auto& lp = (*ldoc)[static_cast<std::size_t>(part)];
                if (lp.empty()) continue;
                double P[256];
                const double l = lp[static_cast<std::size_t>(ds->predict(part, P))];
                for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
            }
        if (pl && lpl) {  // GEN-E: the content-plan expert, mixed last (after the document experts)
            active.clear();
            for (std::size_t q = 0; q < pl->facts.size(); ++q)
                if (!ContentPlanStore::expressed(*pl, q, gen_folded)) active.push_back(static_cast<int>(q));
            double P[256];
            const int L = ContentPlanStore::predict(*pl, active, seq.data(), seq.size(), P);
            if (L > 0) {
                const double l = lpl[ContentPlanStore::bucket(L)];
                for (int s = 0; s < 256; ++s) M[s] = l * P[s] + (1 - l) * M[s];
            }
        }
        const int pick = pick_byte(M, seq, temperature, rng, U);
        seq.push_back(static_cast<std::uint8_t>(pick));
        ContentPlanStore::fold_into(gen_folded, static_cast<unsigned char>(pick));
        if (ds) ds->feed(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

// GEN-B (docs/reports/GEN_B_SENTENCE_GATE.md): the sentence gate. CyphaLM writes K candidates for the next sentence,
// each a sample from the same mixture and sampler as sample_with_class; the gate keeps the one with the highest score
// and writes nothing itself (PLAN rule 10: a tool may gate). Every feature is CyphaLM's own; none is SCRIBE's.
//   fluency  mean log2 M[b] over the candidate's bytes, under the mixture that wrote it (before temperature and ban)
//   pmi      fluency minus the same mean with the prompt replaced by a line break (what the prompt adds)
//   markup   share of the candidate's bytes that are wiki markup: []{}|=<>&*# and the second quote of each ''
// score = wf * fluency + wp * pmi - wm * markup. A candidate ends after a line break, after ". ", "! " or "? " once it
// has min_len bytes, at max_len bytes, or at the byte budget, so the text is n bytes long, as without the gate.
// Candidate k of sentence j draws from its own mt19937 (gate_seed), so no candidate depends on another; ties keep the
// lower k.
bool parse_gate(const std::string& spec, GateSpec& g) {
    std::stringstream ss(spec);
    std::string tok;
    bool first = true;
    while (std::getline(ss, tok, ',')) {
        if (first) {
            g.k = std::atoi(tok.c_str());
            first = false;
            continue;
        }
        const std::size_t eq = tok.find('=');
        if (eq == std::string::npos) return false;
        const std::string key = tok.substr(0, eq);
        const double v = std::atof(tok.c_str() + eq + 1);
        if (key == "wf") g.wf = v;
        else if (key == "wp") g.wp = v;
        else if (key == "wm") g.wm = v;
        else if (key == "min") g.min_len = static_cast<int>(v);
        else if (key == "max") g.max_len = static_cast<int>(v);
        else return false;
    }
    return g.k >= 1 && g.min_len >= 1 && g.max_len >= g.min_len;
}

std::uint32_t gate_seed(std::uint32_t seed, std::uint32_t j, int k) {
    return seed ^ (0x9E3779B9u * (j + 1)) ^ (0x85EBCA6Bu * static_cast<std::uint32_t>(k + 1));
}

// One candidate sentence after seq (whose first p0 bytes are the prompt), at most budget bytes, and its features.
GateCand gate_candidate(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                        const double* lam, const SimilarContext* sc, const double* lsm, const double* lcp,
                        const std::vector<std::uint8_t>& seq, std::size_t p0, int budget, std::uint32_t rs,
                        double temperature, const GateSpec& g, const cypha::cyphalm::bkn::ByteKN* kn, const double* lkn,
                        bool kn_replace) {
    std::mt19937 rng(rs);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<std::uint8_t> s = seq;
    double lp = 0;
    int len = 0, mk = 0;
    while (len < budget) {
        double M[256];
        sample_mix(model, w, limit, cc, lam, s, sc, lsm, lcp, M, kn, lkn, kn_replace);
        const int b = pick_byte(M, s, temperature, rng, U);
        lp += std::log2(std::max(M[b], 1e-12));
        mk += (b != 0 && std::strchr("[]{}|=<>&*#", b) != nullptr) || (b == '\'' && !s.empty() && s.back() == '\'') ? 1 : 0;
        s.push_back(static_cast<std::uint8_t>(b));
        ++len;
        if (b == '\n') break;
        if (len >= g.min_len && b == ' ' && std::strchr(".!?", s[s.size() - 2]) != nullptr) break;
    }
    GateCand c;
    c.bytes.assign(s.begin() + static_cast<std::ptrdiff_t>(seq.size()), s.end());
    c.f = lp / len;
    c.m = static_cast<double>(mk) / len;
    if (g.wp != 0) {  // the same bytes after a line break and the text written so far, without the prompt
        std::vector<std::uint8_t> c0(1, '\n');
        c0.insert(c0.end(), seq.begin() + static_cast<std::ptrdiff_t>(p0), seq.end());
        double lp0 = 0;
        for (const std::uint8_t b : c.bytes) {
            double M[256];
            sample_mix(model, w, limit, cc, lam, c0, sc, lsm, lcp, M, kn, lkn, kn_replace);
            lp0 += std::log2(std::max(M[b], 1e-12));
            c0.push_back(b);
        }
        c.p = c.f - lp0 / len;
    }
    c.score = g.wf * c.f + g.wp * c.p - g.wm * c.m;
    return c;
}

std::string sample_gated(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                         const double* lam, const SimilarContext* sc, const double* lsm, const double* lcp,
                         const std::string& prompt, int n, std::uint32_t seed, double temperature, const GateSpec& g,
                         GateStats& st, const cypha::cyphalm::bkn::ByteKN* kn, const double* lkn, bool kn_replace) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    const std::size_t p0 = seq.size();
    for (std::uint32_t j = 0; seq.size() - p0 < static_cast<std::size_t>(n); ++j) {
        const int budget = std::min(g.max_len, n - static_cast<int>(seq.size() - p0));
        GateCand best;
        int best_k = -1;
        for (int k = 0; k < g.k; ++k) {
            GateCand c = gate_candidate(model, w, limit, cc, lam, sc, lsm, lcp, seq, p0, budget, gate_seed(seed, j, k),
                                        temperature, g, kn, lkn, kn_replace);
            if (best_k < 0 || c.score > best.score) {
                best_k = k;
                best = std::move(c);
            }
        }
        seq.insert(seq.end(), best.bytes.begin(), best.bytes.end());
        st.fluency += best.f;
        st.pmi += best.p;
        st.markup += best.m;
        st.changed += best_k != 0 ? 1 : 0;
        ++st.sentences;
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(p0), seq.end());
}

// A4e: simpair [--margin M] REF.f64 FILE.f64 ... : cls --sim-bits files (doubles: bits per byte of trie + IG + class +
// sim over a --sim-stream continuation), each paired against REF over the whole continuation (4 KiB blocks, 4,000
// resamples), each MiB and the last 2 MiB (2,000), as cls pairs specs in one run. With a margin, also the whole
// continuation's change minus M (its p_lt0 is P(change < M)).
int cmd_simpair(std::vector<std::string> files) {
    double margin = 0;
    const bool has_margin = files.size() >= 2 && files[0] == "--margin";
    if (has_margin) margin = std::atof(files[1].c_str()), files.erase(files.begin(), files.begin() + 2);
    auto load = [](const std::string& f, std::vector<double>& v) {
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        if (!in) return false;
        v.resize(static_cast<std::size_t>(in.tellg()) / sizeof(double));
        in.seekg(0);
        return static_cast<bool>(in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double))));
    };
    std::vector<double> ref;
    if (files.size() < 2 || !load(files[0], ref) || ref.empty()) {
        std::fprintf(stderr, "usage: cyphalm_trie simpair REF.f64 FILE.f64 ...\n");
        return 2;
    }
    constexpr std::size_t kMiB = std::size_t{1} << 20;
    const std::size_t C = ref.size();
    auto part = [](const std::vector<double>& v, std::size_t from, std::size_t to) {
        return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(from), v.begin() + static_cast<std::ptrdiff_t>(to));
    };
    auto diff = [](const std::vector<double>& x, const std::vector<double>& y) {
        std::vector<double> d(x.size());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] = x[k] - y[k];
        return d;
    };
    auto mean = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    for (std::size_t f = 1; f < files.size(); ++f) {
        std::vector<double> x;
        if (!load(files[f], x) || x.size() != C) {
            std::fprintf(stderr, "%s: not %zu doubles\n", files[f].c_str(), C);
            return 1;
        }
        std::printf("simpair %s bits %.6f ref %.6f\n", files[f].c_str(), mean(x), mean(ref));
        print_boot("simpair " + files[f] + " all vs_ref", block_boot(diff(x, ref), 4096, 4000, 11));
        if (has_margin) {
            std::vector<double> d = diff(x, ref);
            for (double& v : d) v -= margin;
            print_boot("simpair " + files[f] + " all vs_ref_minus_" + std::to_string(margin), block_boot(d, 4096, 4000, 11));
        }
        for (std::size_t g0 = 0; g0 < C; g0 += kMiB) {
            const std::size_t g1 = std::min(C, g0 + kMiB);
            print_boot("simpair " + files[f] + " cont_mib " + std::to_string(g0 / kMiB + 1) + " vs_ref",
                       block_boot(diff(part(x, g0, g1), part(ref, g0, g1)), 4096, 2000, 11));
        }
        if (C >= 2 * kMiB)
            print_boot("simpair " + files[f] + " last_2mib vs_ref", block_boot(diff(part(x, C - 2 * kMiB, C), part(ref, C - 2 * kMiB, C)), 4096, 2000, 11));
    }
    return 0;
}

// A4e: --sim-growth LIST (cls, with --sim-stream): for each --sim spec, keys over the train bytes; at each stream length L
// in LIST (MiB from the holdout's start), the expert reads the 64 KiB just before L, after one untimed read that takes in
// every word end before them, and prints the online keys held, their bytes (counted and allocated), the
// process's resident MB, the searches' p50 and p99, the time per byte, and how many held keys equal a stored key or an
// earlier held one (what merging them would save). No trie, InfiniGram or class expert is built.
int sim_growth(const Args& a, const std::vector<std::uint8_t>& data, std::size_t train) {
    auto rss_mb = [] {
        std::ifstream st("/proc/self/status");
        std::string line;
        while (std::getline(st, line))
            if (line.rfind("VmRSS:", 0) == 0) return std::atof(line.c_str() + 6) / 1000.0;
        return 0.0;
    };
    std::unique_ptr<SimilarContext> sc;
    std::string built;
    for (const auto& spec : a.sims) {
        const SimSpec ss = parse_sim(spec);
        if (!ss.ok) {
            std::cerr << "bad --sim spec " << spec << "\n";
            return 2;
        }
        if (ss.build != built) {
            sc.reset();
            if (!(sc = make_similar_context(ss))) return 1;
            sc->build(data.data(), train);
            built = ss.build;
        } else
            sc->set_query(ss.p);
        const std::size_t index = sc->bytes();
        std::printf("simgrowth %s index_mb %.3f rss_mb %.1f\n", spec.c_str(), static_cast<double>(index) / 1e6, rss_mb());
        for (const auto& x : split(a.sim_growth, ',')) {
            const double L = std::atof(x.c_str());
            const std::size_t end = train + static_cast<std::size_t>(L * 1048576.0), from = end - 65536;
            if (end > data.size() || from <= train) {
                std::cerr << "--sim-growth " << x << ": outside the stream\n";
                return 2;
            }
            double P[256];
            sc->predict(data.data(), from - 1, P);  // takes in the word ends before the timed bytes
            const std::size_t q0 = sc->queries();
            const auto t0 = std::chrono::steady_clock::now();
            for (std::size_t n = from; n < end; ++n) sc->predict(data.data(), n, P);
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto dup = sc->online_duplicates();
            std::printf("simgrowth %s mib %s held %zu online_mb %.3f alloc_mb %.3f rss_mb %.1f searches %zu p50_us %.1f "
                        "p99_us %.1f us_per_byte %.2f dup_stored %zu dup_online %zu\n",
                        spec.c_str(), x.c_str(), sc->online_keys(), static_cast<double>(sc->bytes() - index) / 1e6,
                        static_cast<double>(sc->online_alloc_bytes()) / 1e6, rss_mb(), sc->queries() - q0,
                        sc->latency_us(0.5, q0, sc->queries()), sc->latency_us(0.99, q0, sc->queries()), t * 1e6 / 65536.0,
                        dup.first, dup.second);
            std::fflush(stdout);
        }
    }
    return 0;
}

int cmd_cls(const Args& a) {
    const Config& cfg = a.cfg;
    if (a.pos.size() < 2) {
        std::fprintf(stderr, "usage: cyphalm_trie cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs]\n");
        return 2;
    }
    // A4e: --sim-stream FROM:LEN puts the corpus bytes [FROM, FROM + LEN) after the holdout in place of the fresh MiB
    // [sample, sample + 1 MiB) (sizes in bytes, or with K or M); it implies --sim-fresh.
    constexpr std::size_t kMiB = std::size_t{1} << 20;
    std::size_t sfrom = cfg.sample, slen = kMiB;
    if (!a.sim_stream.empty()) {
        auto size_of = [](const std::string& x) {
            const char u = x.empty() ? 0 : x.back();
            const double v = std::atof(x.c_str());
            return static_cast<std::size_t>(v * (u == 'M' ? 1048576.0 : u == 'K' ? 1024.0 : 1.0));
        };
        const std::size_t c = a.sim_stream.find(':');
        if (c != std::string::npos) sfrom = size_of(a.sim_stream.substr(0, c)), slen = size_of(a.sim_stream.substr(c + 1));
        if (c == std::string::npos || sfrom < cfg.sample || slen < kMiB) {
            std::fprintf(stderr, "--sim-stream FROM:LEN needs FROM >= %zu and LEN >= 1M\n", cfg.sample);
            return 2;
        }
    }
    const std::size_t H = cfg.sample - cfg.train, half = H / 2, N = H + slen;
    std::vector<std::uint8_t> data;
    if (const std::string e = read_prefix(a.wiki, sfrom + slen, data); !e.empty()) {
        std::cerr << e << "\n";
        return 1;
    }
    if (sfrom > cfg.sample)
        data.erase(data.begin() + static_cast<std::ptrdiff_t>(cfg.sample), data.begin() + static_cast<std::ptrdiff_t>(sfrom));
    if (!a.sim_growth.empty()) return sim_growth(a, data, cfg.train);  // A4e: the online keys' memory and search time only
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);
    model.compact_match();
    const PackedModel pm = pack(model);  // M4
    model.release();
    const FmInfiniGram ig(data.data(), cfg.train);
    std::vector<double> pt(N), pi(N);
    std::vector<int> bi(N);
    FmInfiniGram::Cursor cur;
    for (std::size_t k = 0, prev = 0; k < N; ++k) {
        const std::size_t i = cfg.train + k;
        double e[5][256], M[256];
        bool em = false;
        unsigned pred = 0;
        pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
        const std::size_t len = std::min<std::size_t>(i, 8192);
        const auto r = ig.query(data.data() + (i - len), len, 8192, k == 0 ? -1 : std::min<int>(static_cast<int>(prev) + 1, 8192),
                                1, cur);
        prev = static_cast<std::size_t>(r.n);
        pi[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
        bi[k] = bucket(r.n, r.total);
    }
    // InfiniGram mixed over the trie: lambdas on the first holdout half (holdout read) and on the whole holdout (fresh).
    double lam_h[kBuckets], lam_f[kBuckets];
    fit_lambdas(pi, pt, bi, half, kBuckets, lam_h);
    fit_lambdas(pi, pt, bi, H, kBuckets, lam_f);
    std::vector<double> pih(N), pif(N);
    for (std::size_t k = 0; k < N; ++k) {
        pih[k] = lam_h[bi[k]] * pi[k] + (1 - lam_h[bi[k]]) * pt[k];
        pif[k] = lam_f[bi[k]] * pi[k] + (1 - lam_f[bi[k]]) * pt[k];
    }
    auto bits = [](const std::vector<double>& p, std::size_t from, std::size_t to) {
        std::vector<double> b;
        for (std::size_t k = from; k < to; ++k) b.push_back(-std::log2(p[k]));
        return b;
    };
    auto mean = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    const auto h_t = bits(pt, half, H), h_i = bits(pih, half, H), f_t = bits(pt, H, N), f_i = bits(pif, H, N);
    std::printf("holdout2 trie %.6f trie+ig %.6f | fresh trie %.6f trie+ig %.6f\n", mean(h_t), mean(h_i), mean(f_t), mean(f_i));

    // GEN-C2 (docs/reports/GEN_C_TOPIC_CACHE.md, "GEN-C2"): --doc V1;V2;... reads each variant of the document experts
    // over qa's default chain: trie + InfiniGram, the class expert pos[1], the similar-context expert (default), copy.
    // Dev: the holdout's second half, every lambda fit on its first half. With --sim-stream the stream after the
    // holdout is the fresh read: every lambda fit on the whole holdout, read as one stream. The other cls reads are skipped.
    if (!a.doc.empty()) {
        const bool fresh = !a.sim_stream.empty();
        const std::size_t M = fresh ? N : H;
        auto cc = make_class_context(a.pos[1]);
        if (!cc) return 1;
        cc->build(data.data(), cfg.train);
        int rc = 0;
        const auto sc = default_sim(a, data.data(), cfg.train, rc);
        if (rc) return rc;
        std::vector<double> dev(pih.begin(), pih.begin() + static_cast<std::ptrdiff_t>(H)),
            fr(pif.begin(), pif.begin() + static_cast<std::ptrdiff_t>(M));
        auto both = [&](const std::vector<double>& pe, const std::vector<int>& bk, int nb, int copy_m) {
            mix_into(pe, bk, half, nb, dev, copy_m);
            if (fresh) mix_into(pe, bk, H, nb, fr, copy_m);
        };
        std::vector<double> pe(M);
        std::vector<int> bk(M);
        for (std::size_t k = 0; k < M; ++k) {
            double P[256];
            bk[k] = ClassContext::bucket(cc->predict(data.data(), cfg.train + k, P));
            pe[k] = P[data[cfg.train + k]];
        }
        both(pe, bk, ClassContext::kBuckets, -1);
        if (sc) {
            for (std::size_t k = 0; k < M; ++k) {
                double P[256];
                bk[k] = SimilarContext::bucket(sc->predict(data.data(), cfg.train + k, P));
                pe[k] = P[data[cfg.train + k]];
            }
            both(pe, bk, SimilarContext::kBuckets, -1);
        }
        if (!a.no_copy) {
            for (std::size_t k = 0; k < M; ++k) {
                int n = 0;
                pe[k] = copy_pred(data.data(), cfg.train + k, n) == data[cfg.train + k] ? 1.0 : 0.0;
                bk[k] = copy_bucket(n);
            }
            both(pe, bk, kCopyBuckets, a.copy_min);
        }
        auto change = [](const std::vector<double>& with, const std::vector<double>& without) {
            std::vector<double> d(with.size());
            for (std::size_t k = 0; k < d.size(); ++k) d[k] = with[k] - without[k];
            return d;
        };
        const auto h0 = bits(dev, half, H);
        const auto f0 = fresh ? bits(fr, H, M) : std::vector<double>{};
        std::printf("doc2 chain trie+ig+cls+%s%s holdout2 %.6f", sc ? "sim+" : "", a.no_copy ? "" : "copy", mean(h0));
        if (fresh) std::printf(" | fresh %.6f (%zu bytes)", mean(f0), M - H);
        std::printf("\n");
        for (const auto& v : split(a.doc, ';')) {
            const auto dt = make_doc(v, data.data(), cfg.train, rc);
            if (!dt) return rc ? rc : 1;
            std::vector<double> d1 = dev;
            fit_doc(*dt, data.data(), cfg.train, half, d1);
            const auto h1 = bits(d1, half, H);
            std::printf("doc2 %s holdout2 %.6f -> %.6f\n", v.c_str(), mean(h0), mean(h1));
            print_boot("doc2 " + v + " holdout2_change", block_boot(change(h1, h0), 4096, 2000, 7));
            if (fresh) {
                std::vector<double> f1 = fr;
                fit_doc(*dt, data.data(), cfg.train, H, f1);
                const auto fb = bits(f1, H, M);
                std::printf("doc2 %s fresh %.6f -> %.6f\n", v.c_str(), mean(f0), mean(fb));
                print_boot("doc2 " + v + " fresh_change", block_boot(change(fb, f0), 4096, 4000, 11));
            }
            std::fflush(stdout);
        }
        return 0;
    }

    // Surface prompts and their human continuations.
    const int np = 200, nbytes = 400;
    std::vector<std::pair<std::string, std::string>> pr;
    {
        const std::string hold(reinterpret_cast<const char*>(data.data()) + cfg.train, H);
        std::size_t pos = hold.find('\n') + 1;
        while (pos < hold.size()) {
            const std::size_t e = hold.find('\n', pos);
            if (e == std::string::npos) break;
            const std::string line = hold.substr(pos, e - pos);
            const std::size_t at = pos;
            pos = e + 1;
            if (!good_start(line)) continue;
            std::size_t cut = 64;
            while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;
            if (at + cut + nbytes > hold.size()) continue;
            pr.push_back({line.substr(0, cut), hold.substr(at + cut, nbytes)});
        }
        std::mt19937_64 prng(20261027);
        std::shuffle(pr.begin(), pr.end(), prng);
        pr.resize(std::min<std::size_t>(pr.size(), np));
    }
    cypha::eval::SurfaceRef ref;
    cypha::eval::surface_add_vocab(ref, std::string(reinterpret_cast<const char*>(data.data()), cfg.train));
    double d4 = 0, lp8 = 0;
    for (auto& ph : pr) d4 += cypha::eval::scribe_detail::distinct4_bytes(ph.second), lp8 += cypha::eval::scribe_detail::loop8_bytes(ph.second);
    ref.ref_d4[1] = d4 / static_cast<double>(pr.size());
    ref.ref_loop[1] = lp8 / static_cast<double>(pr.size());
    // --out FILE: the samples (and the human continuations) as JSON lines, for `scribe prepare --mode wikiraw`.
    std::ofstream samples_out;
    if (!a.out.empty()) samples_out.open(a.out);
    auto json_str = [](const std::string& x) {
        std::string o = "\"";
        for (std::size_t i = 0; i < x.size();) {
            const unsigned char c = static_cast<unsigned char>(x[i]);
            if (c >= 0x80) {
                const std::size_t L = cypha::eval::scribe_detail::utf8_seq_len(x, i);
                if (L == 0) {
                    o += "\\ufffd";
                    ++i;
                } else {
                    o.append(x, i, L);
                    i += L;
                }
                continue;
            }
            if (c == '"' || c == '\\') {
                o.push_back('\\');
                o.push_back(static_cast<char>(c));
            } else if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                o += b;
            } else {
                o.push_back(static_cast<char>(c));
            }
            ++i;
        }
        return o + "\"";
    };
    auto write_sample = [&](const std::string& system, std::size_t i, const std::string& text) {
        if (!samples_out) return;
        samples_out << "{\"id\":" << json_str("cls/" + system + "/" + std::to_string(i)) << ",\"set\":\"cls\",\"system\":" << json_str(system)
                    << ",\"prompt\":" << json_str(pr[i].first) << ",\"text\":" << json_str(text) << "}\n";
    };
    auto surface = [&](const ClassContext* cc, const double* lam, const std::string& system, const SimilarContext* sc = nullptr,
                       const double* lsm = nullptr) {
        std::vector<double> v;
        for (std::size_t i = 0; i < pr.size(); ++i) {
            const std::uint64_t seed = 1000003ull * (i + 1) + 2ull * 7919ull;
            const std::string smp = sample_with_class(pm, w, cfg.train, cc, lam, pr[i].first, nbytes,
                                                      static_cast<std::uint32_t>(seed ^ (seed >> 32)), sc, lsm);
            write_sample(system, i, smp);
            v.push_back(cypha::eval::surface_S(ref, smp));
        }
        return v;
    };
    std::vector<double> s_human;
    for (std::size_t i = 0; i < pr.size(); ++i) {
        s_human.push_back(cypha::eval::surface_S(ref, pr[i].second));
        write_sample("human", i, pr[i].second);
    }
    const auto s_trie = surface(nullptr, nullptr, "trie");
    std::printf("surface prompts %zu human %.4f trie %.4f\n", pr.size(), mean(s_human), mean(s_trie));
    auto gain = [](const std::vector<double>& with, const std::vector<double>& without) {
        std::vector<double> d(with.size());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] = with[k] - without[k];
        return d;
    };

    // A4 (docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md): with --sim, each similar-context spec (keys over the train bytes)
    // over trie + InfiniGram and over trie + InfiniGram + the class expert of the first spec, lambdas on the first holdout
    // half; the holdout read, the index's recall against an exact search on 200 holdout positions, and its costs. The
    // fresh text (lambdas on the whole holdout) only with --sim-fresh; surface S (trie + sim, trie + class + sim) only
    // with --sim-surface. The class specs' own loop is skipped.
    if (!a.sims.empty()) {
        auto cc = make_class_context(a.pos[1]);
        if (!cc) return 1;
        cc->build(data.data(), cfg.train);
        constexpr int kC = ClassContext::kBuckets, kS = SimilarContext::kBuckets;
        std::vector<double> pc(N), ptc(N), pch(N), pcf(N);
        std::vector<int> bc(N);
        for (std::size_t k = 0; k < N; ++k) {
            double P[256];
            bc[k] = ClassContext::bucket(cc->predict(data.data(), cfg.train + k, P));
            pc[k] = P[data[cfg.train + k]];
        }
        double lt_h[kC], li_h[kC], li_f[kC];
        fit_lambdas(pc, pt, bc, half, kC, lt_h);
        fit_lambdas(pc, pih, bc, half, kC, li_h);
        fit_lambdas(pc, pif, bc, H, kC, li_f);
        for (std::size_t k = 0; k < N; ++k) {
            ptc[k] = lt_h[bc[k]] * pc[k] + (1 - lt_h[bc[k]]) * pt[k];
            pch[k] = li_h[bc[k]] * pc[k] + (1 - li_h[bc[k]]) * pih[k];
            pcf[k] = li_f[bc[k]] * pc[k] + (1 - li_f[bc[k]]) * pif[k];
        }
        const auto h_c = bits(pch, half, H), f_c = bits(pcf, H, N);
        std::printf("cls %s holdout2 trie+ig+cls %.6f | fresh trie+ig+cls %.6f\n", a.pos[1].c_str(), mean(h_c), mean(f_c));
        std::vector<double> s_cls;
        if (a.sim_surface) {
            s_cls = surface(cc.get(), lt_h, "trie+cls:" + a.pos[1]);
            std::printf("cls %s surface %.4f\n", a.pos[1].c_str(), mean(s_cls));
        }
        std::unique_ptr<SimilarContext> sc;
        std::string built;
        std::vector<double> first_hc, first_fc, first_sv, first_svt;  // the first spec's reads, for paired comparisons
        const bool stream = !a.sim_stream.empty(), fresh = a.sim_fresh || stream;  // A4e
        if (!a.sim_bits.empty()) std::ofstream(a.sim_bits + ".base.f64", std::ios::binary)
                                     .write(reinterpret_cast<const char*>(f_c.data()), static_cast<std::streamsize>(f_c.size() * sizeof(double)));
        std::size_t spec_no = 0;
        for (const auto& spec : a.sims) {
            const SimSpec ss = parse_sim(spec);
            if (!ss.ok) {
                std::cerr << "bad --sim spec " << spec << "\n";
                return 2;
            }
            double build_s = 0;
            if (ss.build != built) {
                sc.reset();
                if (!(sc = make_similar_context(ss))) return 1;
                const auto t0 = std::chrono::steady_clock::now();
                sc->build(data.data(), cfg.train);
                build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                built = ss.build;
            } else
                sc->set_query(ss.p);
            const std::size_t M = fresh ? N : H;
            std::vector<double> ps(N, 0.0);
            std::vector<int> sb(N, 0);
            std::size_t live = 0;
            const auto t1 = std::chrono::steady_clock::now();
            struct Seg { std::size_t held, bytes, alloc, q; double s; };
            std::vector<Seg> segs;  // A4e (--sim-stream): the expert at the end of each MiB of the stream
            const std::size_t index_bytes = sc->bytes();
            auto seg_now = [&] {
                segs.push_back({sc->online_keys(), sc->bytes(), sc->online_alloc_bytes(), sc->queries(),
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count()});
            };
            for (std::size_t k = 0; k < M; ++k) {
                if (stream && k > 0 && k % kMiB == 0) seg_now();
                double P[256];
                const auto hit = sc->predict(data.data(), cfg.train + k, P);
                sb[k] = SimilarContext::bucket(hit);
                ps[k] = P[data[cfg.train + k]];
                live += hit.live > 0;
            }
            if (stream) seg_now();
            const double score_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            double ls_i[kS], ls_c[kS];
            fit_lambdas(ps, pih, sb, half, kS, ls_i);
            fit_lambdas(ps, pch, sb, half, kS, ls_c);
            auto mixed = [&](const std::vector<double>& base, const double* l, std::size_t from, std::size_t to) {
                std::vector<double> b;
                for (std::size_t k = from; k < to; ++k) b.push_back(-std::log2(l[sb[k]] * ps[k] + (1 - l[sb[k]]) * base[k]));
                return b;
            };
            const auto hi = mixed(pih, ls_i, half, H), hc = mixed(pch, ls_c, half, H);
            // Recall of the inverted file against an exact scan, on 200 holdout positions.
            double rec = 0;
            int nq = 0;
            for (std::size_t r = 0; r < 200; ++r) {
                const std::size_t i = cfg.train + 1 + r * (H / 200);
                const auto ex = sc->neighbours(data.data(), i, true), iv = sc->neighbours(data.data(), i, false);
                if (ex.empty()) continue;
                std::set<std::uint32_t> e;
                for (const auto& x : ex) e.insert(x.first);
                int hitn = 0;
                for (const auto& x : iv) hitn += static_cast<int>(e.count(x.first));
                rec += static_cast<double>(hitn) / static_cast<double>(ex.size());
                ++nq;
            }
            std::printf("sim %s keys %zu dim %d lists %d table_bytes %zu build_s %.1f score_s %.1f queries %zu p50_us %.1f p99_us "
                        "%.1f live_share %.4f recall %.4f\n",
                        spec.c_str(), sc->keys(), sc->dim(), sc->lists(), sc->bytes(), build_s, score_s, sc->queries(),
                        sc->latency_us(0.5), sc->latency_us(0.99), static_cast<double>(live) / static_cast<double>(M),
                        nq ? rec / nq : 0.0);
            std::printf("sim %s holdout2 trie+ig+sim %.6f trie+ig+cls+sim %.6f\n", spec.c_str(), mean(hi), mean(hc));
            print_boot("sim " + spec + " holdout2_gain_over_trie_ig", block_boot(gain(hi, h_i), 4096, 2000, 7));
            print_boot("sim " + spec + " holdout2_gain_over_trie_ig_cls", block_boot(gain(hc, h_c), 4096, 2000, 7));
            if (first_hc.empty()) first_hc = hc;
            else print_boot("sim " + spec + " holdout2_trie_ig_cls_vs_first_sim", block_boot(gain(hc, first_hc), 4096, 2000, 7));
            {  // where the gain over trie + InfiniGram + class falls: bytes where the expert is silent, at a word's start, inside it
                double n3[3] = {0, 0, 0}, g3[3] = {0, 0, 0};
                for (std::size_t k = half; k < H; ++k) {
                    const int c = sb[k] == 0 ? 0 : (sb[k] - 1) % 2 == 0 ? 1 : 2;
                    ++n3[c];
                    g3[c] += hc[k - half] - h_c[k - half];
                }
                std::printf("sim %s holdout2_split silent %.0f %.6f word_start %.0f %.6f in_word %.0f %.6f (bytes, summed bits / all bytes)\n",
                            spec.c_str(), n3[0], g3[0] / static_cast<double>(H - half), n3[1], g3[1] / static_cast<double>(H - half),
                            n3[2], g3[2] / static_cast<double>(H - half));
            }
            if (fresh) {
                double fs_i[kS], fs_c[kS];
                fit_lambdas(ps, pif, sb, H, kS, fs_i);
                fit_lambdas(ps, pcf, sb, H, kS, fs_c);
                const auto fi = mixed(pif, fs_i, H, N), fc = mixed(pcf, fs_c, H, N);
                std::printf("sim %s fresh trie+ig+sim %.6f trie+ig+cls+sim %.6f\n", spec.c_str(), mean(fi), mean(fc));
                print_boot("sim " + spec + " fresh_gain_over_trie_ig", block_boot(gain(fi, f_i), 4096, 4000, 11));
                print_boot("sim " + spec + " fresh_gain_over_trie_ig_cls", block_boot(gain(fc, f_c), 4096, 4000, 11));
                const bool first = first_fc.empty();
                if (first) first_fc = fc;
                else print_boot("sim " + spec + " fresh_trie_ig_cls_vs_first_sim", block_boot(gain(fc, first_fc), 4096, 4000, 11));
                if (stream) {  // A4e: the expert at each MiB of the stream (MiB 1 is the holdout), and the continuation by MiB
                    for (std::size_t g = 0; g < segs.size(); ++g) {
                        const std::size_t q0 = g ? segs[g - 1].q : 0;
                        std::printf("simstream %s mib %zu held %zu online_mb %.3f alloc_mb %.3f searches %zu p50_us %.1f p99_us %.1f "
                                    "score_s %.1f\n", spec.c_str(), g + 1, segs[g].held,
                                    static_cast<double>(segs[g].bytes - index_bytes) / 1e6, static_cast<double>(segs[g].alloc) / 1e6,
                                    segs[g].q - q0, sc->latency_us(0.5, q0, segs[g].q), sc->latency_us(0.99, q0, segs[g].q),
                                    segs[g].s - (g ? segs[g - 1].s : 0.0));
                    }
                    const std::size_t C = N - H;
                    auto part = [](const std::vector<double>& v, std::size_t from, std::size_t to) {
                        return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(from), v.begin() + static_cast<std::ptrdiff_t>(to));
                    };
                    for (std::size_t g0 = 0; g0 < C; g0 += kMiB) {
                        const std::size_t g1 = std::min(C, g0 + kMiB), no = g0 / kMiB + 1;
                        std::printf("simstream %s cont_mib %zu trie+ig+cls %.6f trie+ig+cls+sim %.6f\n", spec.c_str(), no,
                                    mean(part(f_c, g0, g1)), mean(part(fc, g0, g1)));
                        if (!first)
                            print_boot("simstream " + spec + " cont_mib " + std::to_string(no) + " vs_first_sim",
                                       block_boot(gain(part(fc, g0, g1), part(first_fc, g0, g1)), 4096, 2000, 11));
                    }
                    if (C >= 2 * kMiB && !first)
                        print_boot("simstream " + spec + " last_2mib vs_first_sim",
                                   block_boot(gain(part(fc, C - 2 * kMiB, C), part(first_fc, C - 2 * kMiB, C)), 4096, 2000, 11));
                }
                if (!a.sim_bits.empty())  // A4e: this spec's continuation bits, for simpair
                    std::ofstream(a.sim_bits + "." + std::to_string(spec_no) + ".f64", std::ios::binary)
                        .write(reinterpret_cast<const char*>(fc.data()), static_cast<std::streamsize>(fc.size() * sizeof(double)));
            }
            ++spec_no;
            if (a.sim_surface) {
                double ls_t[kS], ls_tc[kS];
                fit_lambdas(ps, pt, sb, half, kS, ls_t);
                fit_lambdas(ps, ptc, sb, half, kS, ls_tc);
                const auto sv = surface(nullptr, nullptr, "trie+sim:" + spec, sc.get(), ls_t),
                           svc = surface(cc.get(), lt_h, "trie+cls+sim:" + spec, sc.get(), ls_tc);
                std::printf("sim %s surface trie+sim %.4f trie+cls+sim %.4f\n", spec.c_str(), mean(sv), mean(svc));
                print_boot("sim " + spec + " surface_gain_over_trie", block_boot(gain(sv, s_trie), 1, 4000, 5));
                print_boot("sim " + spec + " surface_gain_over_trie_cls", block_boot(gain(svc, s_cls), 1, 4000, 5));
                if (first_sv.empty()) first_sv = svc, first_svt = sv;
                else {
                    print_boot("sim " + spec + " surface_trie_vs_first_sim", block_boot(gain(sv, first_svt), 1, 4000, 5));
                    print_boot("sim " + spec + " surface_trie_cls_vs_first_sim", block_boot(gain(svc, first_sv), 1, 4000, 5));
                }
            }
            std::fflush(stdout);
        }
        return 0;
    }

    std::vector<double> first_f, first_s;
    for (std::size_t sp = 1; sp < a.pos.size(); ++sp) {
        const std::string& spec = a.pos[sp];
        auto cc = make_class_context(spec);
        if (!cc) return 1;
        const auto t0 = std::chrono::steady_clock::now();
        cc->build(data.data(), cfg.train);
        const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<double> pc(N);
        std::vector<int> bc(N);
        for (std::size_t k = 0; k < N; ++k) {
            double P[256];
            bc[k] = ClassContext::bucket(cc->predict(data.data(), cfg.train + k, P));
            pc[k] = P[data[cfg.train + k]];
        }
        double lt_h[ClassContext::kBuckets], li_h[ClassContext::kBuckets], lt_f[ClassContext::kBuckets], li_f[ClassContext::kBuckets];
        fit_lambdas(pc, pt, bc, half, ClassContext::kBuckets, lt_h);
        fit_lambdas(pc, pih, bc, half, ClassContext::kBuckets, li_h);
        fit_lambdas(pc, pt, bc, H, ClassContext::kBuckets, lt_f);
        fit_lambdas(pc, pif, bc, H, ClassContext::kBuckets, li_f);
        auto mixed = [&](const std::vector<double>& base, const double* l, std::size_t from, std::size_t to) {
            std::vector<double> b;
            for (std::size_t k = from; k < to; ++k) b.push_back(-std::log2(l[bc[k]] * pc[k] + (1 - l[bc[k]]) * base[k]));
            return b;
        };
        const auto ht = mixed(pt, lt_h, half, H), hi = mixed(pih, li_h, half, H);
        const auto ft = mixed(pt, lt_f, H, N), fi = mixed(pif, li_f, H, N);
        std::printf("spec %s classes %zu contexts %zu followers %zu table_bytes %zu build_s %.1f\n", spec.c_str(),
                    cc->classes(), cc->contexts(), cc->followers(), cc->bytes(), build_s);
        std::printf("spec %s holdout2 trie+cls %.6f trie+ig+cls %.6f | fresh trie+cls %.6f trie+ig+cls %.6f\n", spec.c_str(),
                    mean(ht), mean(hi), mean(ft), mean(fi));
        print_boot("spec " + spec + " holdout2_gain_over_trie", block_boot(gain(ht, h_t), 4096, 2000, 7));
        print_boot("spec " + spec + " holdout2_gain_over_trie_ig", block_boot(gain(hi, h_i), 4096, 2000, 7));
        print_boot("spec " + spec + " fresh_gain_over_trie", block_boot(gain(ft, f_t), 4096, 4000, 11));
        print_boot("spec " + spec + " fresh_gain_over_trie_ig", block_boot(gain(fi, f_i), 4096, 4000, 11));
        const auto sv = surface(cc.get(), lt_h, "trie+cls:" + spec);
        std::printf("spec %s surface %.4f\n", spec.c_str(), mean(sv));
        print_boot("spec " + spec + " surface_gain_over_trie", block_boot(gain(sv, s_trie), 1, 4000, 5));
        if (sp == 1) {
            first_f = fi;
            first_s = sv;
        } else {
            print_boot("spec " + spec + " fresh_trie_ig_vs_" + a.pos[1], block_boot(gain(fi, first_f), 4096, 4000, 11));
            print_boot("spec " + spec + " surface_vs_" + a.pos[1], block_boot(gain(sv, first_s), 1, 4000, 5));
        }
        std::fflush(stdout);
    }
    return 0;
}

struct R3Score {
    double bpb = 1e300, min_lp = 0;
    int ig_n = 0;
};

// Bits per byte of `text` from byte a on, read after `prefix`; the min log2 p over its bytes [a, b) (A3: the object
// inside a statement, which the candidates change) and InfiniGram's match length when it predicts byte a (how much
// of the frame and subject it found).
R3Score score_statement(const Lm& lm, const std::string& prefix, const std::string& text, std::size_t a = 0,
                        std::size_t b = 0) {
    R3Score r;
    std::vector<std::uint8_t> seq(prefix.begin(), prefix.end());
    int hint = -1;
    double bits = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        double P[256];
        int n = 0;
        lm.dist(seq, hint, P, &n);
        const double lp = std::log2(std::max(P[static_cast<unsigned char>(text[i])], 1e-300));
        if (i >= a) bits -= lp;
        if (i >= a && i < b) r.min_lp = i == a ? lp : std::min(r.min_lp, lp);
        if (i == a) r.ig_n = n;
        seq.push_back(static_cast<std::uint8_t>(text[i]));
    }
    r.bpb = text.size() <= a ? 0.0 : bits / static_cast<double>(text.size() - a);
    return r;
}

// Mean log2 p per byte of `text` read after `prefix` (the familiarity measure).
double text_logprob(const Lm& lm, const std::string& prefix, const std::string& text) {
    return -score_statement(lm, prefix, text).bpb;
}

struct Answer {
    std::string text;
    double mean_lp = 0, min_lp = 0, ent0 = 0;
    int ig_n0 = 0;
};

// The no-repeat rule (mark_banned), counting only earlier copies that start at or after `from`. The copy models pass
// the prompt's length: copying from the question is their point, looping on their own answer is still banned.
void ban_repeats(const std::vector<std::uint8_t>& seq, std::size_t from, bool banned[256]) {
    std::memset(banned, 0, 256);
    const std::size_t n = seq.size();
    if (n < 11) return;
    for (std::size_t s = from; s + 12 <= n; ++s)
        if (std::memcmp(seq.data() + s, seq.data() + (n - 11), 11) == 0) banned[seq[s + 11]] = true;
}

// Greedy (seed 0) or sampled (T = 0.8, nucleus 0.9, mt19937(seed), as sample_continuation), no-repeat rule, until a
// newline or a sentence end, at most max_bytes. Once the answer begins with the don't-know message, decoding runs on
// to the end of the tag (at most 240 bytes) so the tag can be read; that is the only change to the stopping rule.
Answer decode(const Lm& lm_in, const std::string& prompt, int max_bytes = 80, std::uint32_t seed = 0) {
    Lm lm = lm_in;  // A4: the answer starts after the prompt, whatever its form
    lm.answer_at = prompt.size();
    Answer A;
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int hint = -1;
    double lp_sum = 0;
    int nlp = 0;
    A.min_lp = 0;
    int limit = max_bytes;
    for (int t = 0; t < limit; ++t) {
        double P[256];
        int n = 0;
        lm.dist(seq, hint, P, &n);
        if (t == 0) {
            A.ig_n0 = n;
            double h = 0;
            for (int s = 0; s < 256; ++s)
                if (P[s] > 0) h -= P[s] * std::log2(P[s]);
            A.ent0 = h;
        }
        bool banned[256];
        // ANS: inside a don't-know answer with --copy-dk, as for the copy models: copying from the prompt is allowed.
        // DK2 (--copy-frame): inside the tag of a don't-know answer the rule looks at the tag only, so the tag may
        // repeat the message's entity (an entity longer than 11 bytes would otherwise be cut in the tag).
        const std::size_t tag_at = lm.copy_frame && lm.in_dont_know(seq) ? open_token_at(seq.data(), seq.size(), prompt.size())
                                                                          : std::string::npos;
        ban_repeats(seq, tag_at != std::string::npos ? tag_at : lm.lc || lm.in_dont_know(seq) ? prompt.size() : 0, banned);
        bool all = true;
        for (int s = 0; s < 256; ++s)
            if (!banned[s]) all = false;
        std::vector<std::pair<double, int>> it;  // (p^(1/T), byte), most likely first
        double mass = 0;
        for (int s = 0; s < 256; ++s)
            if (!banned[s] || all) {
                it.push_back({seed ? std::pow(P[s], 1 / 0.8) : P[s], s});
                mass += it.back().first;
            }
        std::stable_sort(it.begin(), it.end(), [](auto& x, auto& y) { return x.first > y.first; });
        int pick = it[0].second;
        if (seed) {
            double acc = 0, km = 0;
            std::size_t keep = 0;
            while (keep < it.size() && acc < 0.9) acc += it[keep].first / mass, km += it[keep++].first;
            double r = U(rng) * km;
            pick = it[keep - 1].second;
            for (std::size_t k = 0; k < keep; ++k)
                if ((r -= it[k].first) <= 0) {
                    pick = it[k].second;
                    break;
                }
        }
        const double lp = std::log2(std::max(P[pick], 1e-300));
        if (pick == '\n') break;
        const bool dk = A.text.rfind("I don't know", 0) == 0;
        if (dk) limit = 240;
        if (!dk && pick == ' ' && !A.text.empty()) {
            const char c = A.text.back();
            const bool initial = A.text.size() >= 2 && std::isupper(static_cast<unsigned char>(A.text[A.text.size() - 2])) &&
                                 (A.text.size() == 2 || A.text[A.text.size() - 3] == ' ');
            if ((c == '.' || c == '!' || c == '?') && !initial) break;
        }
        A.text += static_cast<char>(pick);
        seq.push_back(static_cast<std::uint8_t>(pick));
        lp_sum += lp;
        ++nlp;
        A.min_lp = nlp == 1 ? lp : std::min(A.min_lp, lp);
        if (dk && A.text.size() >= 2 && A.text.compare(A.text.size() - 2, 2, "|>") == 0) break;
    }
    A.mean_lp = nlp ? lp_sum / nlp : -30.0;
    if (!nlp) A.min_lp = -30.0;
    return A;
}

// ---------------------------------------------------------------- dktext (A1 training text)

struct DkRow {
    std::string q, entity, rel, reason, response;
};

// T-QA relation templates (from the dev set's question forms). dktext picks them by the page's categories;
// A2 and A3 read a question's relation from the one that, filled with its E1 entity, is the question.
const std::pair<std::string, std::string> kForms[] = {
    {"birth_date", "When was {E} born?"},       {"birth_place", "Where was {E} born?"},
    {"death_date", "When did {E} die?"},        {"identity", "Who was {E}?"},
    {"author", "Who wrote {E}?"},               {"director", "Who directed {E}?"},
    {"composer", "Who composed {E}?"},          {"painter", "Who painted {E}?"},
    {"capital", "What is the capital of {E}?"}, {"language", "What is the official language of {E}?"},
    {"founder", "Who founded {E}?"},            {"location", "Where is {E}?"},
    {"country", "In which country is {E}?"},    {"inventor", "Who invented {E}?"},
    {"definition", "What is {E}?"}};

std::vector<std::pair<std::string, std::string>> templates_for(const std::string& cats_lower) {
    auto has = [&](const char* k) { return cats_lower.find(k) != std::string::npos; };
    auto pick = [](std::initializer_list<int> ix) {
        std::vector<std::pair<std::string, std::string>> v;
        for (int i : ix) v.push_back(kForms[i]);
        return v;
    };
    if (has("births") || has("deaths")) return pick({0, 1, 2, 3});
    if (has("novels") || has("books") || has("plays") || has("poems")) return pick({4});
    if (has("films")) return pick({5});
    if (has("operas") || has("songs") || has("compositions") || has("symphon") || has("ballets")) return pick({6});
    if (has("paintings")) return pick({7});
    if (has("countries") || has("states of") || has("provinces") || has("territories")) return pick({8, 9});
    if (has("companies") || has("organizations") || has("organisations") || has("bands") || has("musical groups"))
        return pick({10});
    if (has("cities") || has("towns") || has("villages") || has("rivers") || has("islands") ||
        has("municipalities") || has("counties") || has("mountains"))
        return pick({11, 12});
    if (has("inventions") || has("technology")) return pick({13});
    return pick({14});
}

// The question's relation: the template that, filled with the E1 entity, equals the question (normalised).
std::string question_relation(const std::string& q, const std::string& entity) {
    if (entity.empty()) return {};
    for (const auto& [rel, form] : kForms) {
        std::string t = form;
        t.replace(t.find("{E}"), 3, entity);
        if (norm_text(t) == norm_text(q)) return rel;
    }
    return {};
}

// fn(title, body) for every main-namespace, non-redirect page of enwik8 XML text whose title has the dktext shape:
// 4-40 ASCII bytes, no brackets or quotes, not "List of", capitalised. Returns {pages, namespace, redirect, shape}.
std::array<std::size_t, 4> for_pages(const std::string& hay,
                                     const std::function<void(const std::string&, const std::string&)>& fn) {
    std::array<std::size_t, 4> n{};
    for (std::size_t pos = 0;;) {
        const std::size_t t0 = hay.find("<title>", pos);
        if (t0 == std::string::npos) break;
        const std::size_t t1 = hay.find("</title>", t0);
        const std::size_t pe = hay.find("</page>", t0);
        if (t1 == std::string::npos || pe == std::string::npos) break;
        pos = pe + 7;
        ++n[0];
        const std::string title = hay.substr(t0 + 7, t1 - t0 - 7);
        const std::string body = hay.substr(t1, pe - t1);
        if (title.find(':') != std::string::npos) {
            ++n[1];
            continue;
        }
        const std::size_t tx = body.find("<text");
        const std::size_t tx_end = tx == std::string::npos ? std::string::npos : body.find('>', tx);
        if (tx_end == std::string::npos) continue;
        if (lowered(body.substr(tx_end + 1, 12)).rfind("#redirect", 0) == 0) {
            ++n[2];
            continue;
        }
        bool shape = title.size() >= 4 && title.size() <= 40 && title.find_first_of("(),&;\"'") == std::string::npos &&
                     title.rfind("List of", 0) != 0 && std::isupper(static_cast<unsigned char>(title[0]));
        for (unsigned char c : title)
            if (c >= 128) shape = false;
        if (!shape) {
            ++n[3];
            continue;
        }
        fn(title, body);
    }
    return n;
}

int cmd_dktext(const Args& a, const std::string& dev_path, const std::string& test_path, const std::string& out_path,
               std::size_t from, std::size_t max_rows, bool mention) {
    const Config& cfg = a.cfg;
    std::vector<std::uint8_t> all;
    {
        std::ifstream in(a.wiki, std::ios::binary);
        if (!in) {
            std::cerr << "cannot open " << a.wiki << "\n";
            return 1;
        }
        all.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (all.size() <= from) {
        std::cerr << "corpus shorter than --from\n";
        return 1;
    }
    std::vector<Turn> turns;
    const std::string chat = load_dolly(a.dolly, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return 1;
    }
    // The training bytes, lowered: wiki [0, train) + the Dolly chat text.
    std::string train(reinterpret_cast<const char*>(all.data()), cfg.train);
    train += chat;
    for (auto& c : train) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const FmInfiniGram occ(reinterpret_cast<const std::uint8_t*>(train.data()), train.size());
    auto occurs = [&](const std::string& lower) {
        std::size_t cnt = 0;
        return occ.match_prefix(reinterpret_cast<const std::uint8_t*>(lower.data()), lower.size(), lower.size(), cnt) ==
               lower.size();
    };
    // Dev and test, lowered, for the leakage check.
    std::string evaltext;
    std::vector<std::string> eval_q, eval_alias;
    for (const auto& p : {dev_path, test_path}) {
        const auto rows = load_tqa(p);
        if (rows.empty()) {
            std::cerr << "cannot read " << p << "\n";
            return 1;
        }
        for (const auto& r : rows) {
            eval_q.push_back(lowered(r.q));
            evaltext += lowered(r.q) + "\n" + lowered(r.title) + "\n";
            for (const auto& g : r.gold) {
                evaltext += lowered(g) + "\n";
                if (g.size() >= 4) eval_alias.push_back(norm_text(g));
            }
        }
    }
    const std::string hay(reinterpret_cast<const char*>(all.data()) + from, all.size() - from);
    std::size_t n_in_train = 0, n_in_eval = 0, n_alias = 0;
    std::vector<DkRow> rows;
    std::vector<std::string> kept_entities, absent;
    const auto pages = for_pages(hay, [&](const std::string& title, const std::string& body) {
        const std::string el = lowered(title);
        // In the training bytes: rejected, unless --mention and it occurs as a link (named, with no article in the slice).
        const bool in_train = occurs(el);
        if (in_train && !(mention && occurs("[[" + el))) {
            ++n_in_train;
            return;
        }
        if (evaltext.find(el) != std::string::npos) {  // leakage: the entity is in dev or test
            ++n_in_eval;
            return;
        }
        const std::string en = norm_text(title);
        for (const auto& g : eval_alias)
            if (en.find(g) != std::string::npos) {
                ++n_alias;
                return;
            }
        std::string cats;
        for (std::size_t c = body.find("[[Category:"); c != std::string::npos; c = body.find("[[Category:", c + 1))
            cats += lowered(body.substr(c + 11, body.find("]]", c) - c - 11)) + "|";
        const auto tpl = templates_for(cats);
        // At most two relations per entity, picked by a hash so the set is the same on every run.
        const std::uint32_t h = static_cast<std::uint32_t>(std::hash<std::string>{}(title));
        for (std::size_t k = 0; k < std::min<std::size_t>(2, tpl.size()); ++k) {
            const auto& [rel, form] = tpl[(h + k) % tpl.size()];
            std::string q = form;
            q.replace(q.find("{E}"), 3, title);
            rows.push_back({q, title, rel, "UNKNOWN_ENTITY", dont_know_text(title, rel, "UNKNOWN_ENTITY")});
        }
        kept_entities.push_back(title);
        if (!in_train) absent.push_back(title);
    });
    std::mt19937_64 prng(20261028);
    std::shuffle(rows.begin(), rows.end(), prng);
    if (max_rows && rows.size() > max_rows) rows.resize(max_rows);

    // Leakage check on the final rows: no dev/test question, no dev/test answer variant (>= 4 bytes).
    std::size_t leak_q = 0, leak_alias = 0;
    for (const auto& r : rows) {
        const std::string ql = lowered(r.q);
        for (const auto& e : eval_q)
            if (e == ql) ++leak_q;
        const std::string n = norm_text(r.q + " " + r.entity);
        for (const auto& g : eval_alias)
            if (n.find(g) != std::string::npos) ++leak_alias;
    }
    // Audit the absence check with a plain scan (std::string::find) over the lowered training bytes.
    std::size_t audit = 0, audit_found = 0;
    for (std::size_t i = 0; i < absent.size() && audit < 300; i += std::max<std::size_t>(1, absent.size() / 300)) {
        ++audit;
        if (train.find(lowered(absent[i])) != std::string::npos) ++audit_found;
    }
    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "cannot write " << out_path << "\n";
        return 1;
    }
    out << "question\tentity\trelation\treason\tresponse\n";
    std::map<std::string, int> by_rel;
    for (const auto& r : rows) {
        out << r.q << "\t" << r.entity << "\t" << r.rel << "\t" << r.reason << "\t" << r.response << "\n";
        ++by_rel[r.rel];
    }
    std::cout << "dk_pages_scanned " << pages[0] << " from_byte " << from << "\n";
    std::cout << "dk_rejected namespace " << pages[1] << " redirect " << pages[2] << " shape " << pages[3]
              << " in_training_bytes " << n_in_train << " in_dev_or_test " << n_in_eval << " dev_test_alias " << n_alias
              << "\n";
    std::cout << "dk_entities_kept " << kept_entities.size() << " absent " << absent.size() << " mentioned "
              << kept_entities.size() - absent.size() << " dk_rows " << rows.size() << "\n";
    for (const auto& [k, v] : by_rel) std::cout << "dk_relation " << k << " " << v << "\n";
    std::cout << "leak_check question_equal " << leak_q << " answer_variant_in_text " << leak_alias << "\n";
    std::cout << "absence_audit_plain_scan " << audit << " found " << audit_found << "\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, rows.size()); ++i)
        std::cout << "dk_example " << rows[i].q << " => " << rows[i].response << "\n";
    return (leak_q || leak_alias || audit_found) ? 1 : 0;
}

// ---------------------------------------------------------------- qa (A0 harness, A1 variants, A1c, A2, A3)

struct QaOpts {
    std::string dev, test, out, base, dk;
    double dk_ratio = 0;
    std::string fam = "none";  // none | before | after: where the <|fam=..|> token goes
    std::string parse, fam_stat = "question_lp", fam_train = "stat", r3, shard, shard2, leak;
    bool copy = false, fam_only = false;
    std::string cls;
    int copy_min = 0;
    std::string sim;  // A4: similar-context expert spec
    bool copy_out = false;  // B4C: the default copy expert, off inside answers
    std::string fam_ent;    // ANS: frames before the token, e.g. entity="{E}"
    bool copy_dk = false, fam_read = false;  // ANS
    double dk_plain = 0;                     // DK2: --dk-plain
    std::string fam_ent_at = "all";          // DK2: --fam-ent-at
    bool copy_frame = false;                 // DK2: --copy-frame
};

constexpr int kFeat = 5;
const char* const kFeatName[kFeat] = {"mean_lp", "min_lp", "neg_entropy0", "ig_match0", "question_lp"};
const char* const kR3Name[kFeat] = {"margin", "agree", "neg_bpb", "min_lp", "ig_match_at_object"};

struct Graded {
    int row = 0, in_slice = 0, form = 0, lm = 0;  // form 0 chat, 1 wiki; lm 0 trie, 1 + InfiniGram, 2 / 3 + copy
    int grade = 0;                                // 0 right, 1 abstained (message), 2 wrong
    Answer ans;
    double qlp = 0;  // final model: mean log2 p of the question (familiarity)
    std::string fam_tok;
    double fv[kFeat] = {0, 0, 0, 0, 0};  // the gate's features: kFeatName, or kR3Name for A3 rows
    double ms = 0;                       // A4: decode time of the answer
    int prose = -1, covered = -1;        // in-slice: a gold answer is in its article's text; A2: 2 entity and
                                         // relation in a shard, 1 the entity only, 0 neither
};

double feat(const Graded& g, int f) { return g.fv[f]; }

struct Gate {
    int f = -1;
    double tau = 1e300;  // answer iff feat >= tau (tau = +inf: answer nothing)
};

// Among thresholds with wrong / n <= 2%, the one with the most right answers; ties go to fewer
// wrong, then fewer answered. Questions the model already abstained on stay abstained.
Gate choose_gate(const std::vector<const Graded*>& v, bool have_ig, int* right_out = nullptr) {
    Gate best;
    int best_r = 0, best_w = 0, best_ans = 0;
    const double n = static_cast<double>(v.size());
    for (int f = 0; f < kFeat; ++f) {
        if (f == 3 && !have_ig) continue;
        std::vector<std::pair<double, int>> s;  // (feature, grade) of the questions it answered
        for (auto* g : v)
            if (g->grade != 1) s.push_back({feat(*g, f), g->grade});
        std::sort(s.begin(), s.end(), [](auto& x, auto& y) { return x.first > y.first; });
        int r = 0, w = 0;
        for (std::size_t k = 0; k < s.size(); ++k) {
            (s[k].second == 0 ? r : w)++;
            if (k + 1 < s.size() && s[k + 1].first == s[k].first) continue;  // a threshold cannot split ties
            if (w > 0.02 * n) break;
            const int ans = static_cast<int>(k + 1);
            if (r > best_r || (r == best_r && r > 0 && (w < best_w || (w == best_w && ans < best_ans)))) {
                best_r = r;
                best_w = w;
                best_ans = ans;
                best.f = f;
                best.tau = s[k].first;
            }
        }
    }
    if (right_out) *right_out = best_r;
    return best;
}

bool gated_answer(const Graded& g, const Gate& gate) { return g.grade != 1 && gate.f >= 0 && feat(g, gate.f) >= gate.tau; }

struct Tally {
    int n = 0, right = 0, abst = 0, wrong = 0;
    void add(int grade) {
        ++n;
        if (grade == 0) ++right;
        else if (grade == 1) ++abst;
        else ++wrong;
    }
};

void print_tally(const std::string& label, const Tally& t) {
    const int ans = t.right + t.wrong;
    char b[512];
    std::snprintf(b, sizeof b,
                  "%s n %d right %d abstained %d wrong %d coverage %.4f sel_acc %.4f sel_acc_wilson_lo %.4f "
                  "wrong_rate %.4f wrong_rate_wilson_hi %.4f\n",
                  label.c_str(), t.n, t.right, t.abst, t.wrong, t.n ? static_cast<double>(ans) / t.n : 0.0,
                  ans ? static_cast<double>(t.right) / ans : 0.0, wilson_lo(t.right, ans),
                  t.n ? static_cast<double>(t.wrong) / t.n : 0.0, wilson_hi(t.wrong, t.n));
    std::cout << b;
}

// A1c / A2 / A3. Strings below parse the question (E1's entity, the dktext templates), build training text, grade,
// or check leakage. CyphaLM still writes every answer byte.

// question -> E1 entity ("" when it found none), from `pos_tagger question --tsv` rows
// (question wh wh_phrase wh_head aux verb focus entity entity_via).
std::map<std::string, std::string> load_parse(const std::string& path) {
    std::map<std::string, std::string> m;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() >= 9 && f[0] != "question") m[f[0]] = f[7] == "-" ? "" : f[7];
    }
    return m;
}

std::string strip_article(const std::string& e) {
    for (const char* a : {"the ", "The ", "a ", "an "})
        if (e.rfind(a, 0) == 0) return e.substr(std::strlen(a));
    return e;
}

constexpr int kFam = 8;
const char* const kFamName[kFam] = {"question_lp", "ent_count", "ent_link", "ent_bold",
                                    "ent_title",   "ent_igfrac", "ent_lp",  "ent_tb"};

// A1c: how familiar CyphaLM is with a question's subject, from its own model. question_lp is A1's statistic (the
// question's mean log2 p under the wiki trie). For the E1 entity (article dropped, lowered): log2(1 + its count) in
// the lowered training bytes (wiki train + Dolly, `low`), the same after "[[" (a link), after "'''" (a bold lead) and
// as "<title>entity</title>"; the longest piece of it found there, as a share of its length; its mean log2 p under the
// wiki trie, read in the question; ent_tb, the larger of the bold-lead and title counts (A1d). NaN when there is no
// entity. Several candidates ("a|b|c", A1d): per statistic, the best candidate's value.
std::array<double, kFam> fam_stats(const Lm& wiki_lm, const FmInfiniGram& low, const std::string& q,
                                   const std::string& entity) {
    std::array<double, kFam> v;
    v.fill(std::nan(""));
    if (entity.find('|') != std::string::npos) {
        for (const auto& c : split(entity, '|')) {
            const auto x = fam_stats(wiki_lm, low, q, c);
            for (std::size_t k = 0; k < kFam; ++k) v[k] = std::isnan(v[k]) || x[k] > v[k] ? x[k] : v[k];
        }
        return v;
    }
    v[0] = text_logprob(wiki_lm, "User: ", q);
    const std::string raw = strip_article(entity), e = lowered(raw);
    if (e.empty()) return v;
    auto longest = [&](const std::string& s, std::size_t from, std::size_t& n) {
        return low.match_prefix(reinterpret_cast<const std::uint8_t*>(s.data()) + from, s.size() - from,
                                s.size() - from, n);
    };
    auto count = [&](const std::string& s) {
        std::size_t n = 0;
        return std::log2(1.0 + (longest(s, 0, n) == s.size() ? static_cast<double>(n) : 0.0));
    };
    v[1] = count(e);
    v[2] = count("[[" + e);
    v[3] = count("'''" + e);
    v[4] = count("<title>" + e + "</title>");
    std::size_t best = 0, n = 0;
    for (std::size_t i = 0; i < e.size(); ++i) best = std::max(best, longest(e, i, n));
    v[5] = static_cast<double>(best) / static_cast<double>(e.size());
    const std::size_t at = lowered(q).find(e);
    v[6] = at == std::string::npos ? text_logprob(wiki_lm, "User: ", raw)
                                   : text_logprob(wiki_lm, "User: " + q.substr(0, at), q.substr(at, e.size()));
    v[7] = std::max(v[3], v[4]);
    return v;
}

// ANS (A1d's --fam-ent, restored): the familiarity token with the entity's frames before it. FRAMES is ';'-separated,
// each with {E} for the entity (a leading article dropped): entity="{E}" gives <|entity="Napoleon"|><|fam=low|>. No
// frame is written when there is no entity, and with no FRAMES this is the plain <|fam=LEVEL|>.
std::string fam_token(const std::string& level, const std::string& frames, const std::string& entity) {
    const std::string e = strip_article(entity);
    std::string t;
    for (std::string f : split(frames, ';'))
        if (!e.empty() && f.find("{E}") != std::string::npos) t += "<|" + f.replace(f.find("{E}"), 3, e) + "|>";
    return t + "<|fam=" + level + "|>";
}
// DK2: the frames alone (--fam-ent-at start writes them before "User: ").
std::string fam_frames(const std::string& frames, const std::string& entity) {
    const std::string t = fam_token("x", frames, entity);
    return t.substr(0, t.size() - 9);  // "<|fam=x|>"
}
// DK2 (--dk-plain P): don't-know training turn j carries no token; floor(nP) of the first n turns, spread evenly.
bool dk_plain_turn(std::size_t j, double p) {
    return std::floor(static_cast<double>(j + 1) * p) > std::floor(static_cast<double>(j) * p);
}

// A1d (grading only): the entity of the message "... nothing about X in what I've read.", and whether a name is a
// span of a question (normalised, word-bounded, at least 3 bytes; a leading article dropped).
std::string msg_entity(const std::string& a) {
    const std::size_t p = a.find("nothing about ");
    const std::size_t e = p == std::string::npos ? p : a.find(" in what", p + 14);
    return e == std::string::npos ? std::string() : a.substr(p + 14, e - p - 14);
}
bool names_in(const std::string& name, const std::string& q) {
    const std::string n = norm_text(strip_article(name));
    return n.size() >= 5 && norm_text(q).find(n) != std::string::npos;
}

// AUROC of `pos` above `neg` (ties count half; NaN sits below every number).
double auroc(const std::vector<double>& pos, const std::vector<double>& neg) {
    auto k = [](double x) { return std::isnan(x) ? -1e300 : x; };
    double s = 0;
    for (double p : pos)
        for (double n : neg) s += k(p) > k(n) ? 1.0 : k(p) == k(n) ? 0.5 : 0.0;
    return pos.empty() || neg.empty() ? 0.5 : s / static_cast<double>(pos.size() * neg.size());
}

// The cut t (low iff x < t) that best separates `neg` (low) from `pos` (Youden J); NaN is never low.
double youden(const std::vector<double>& pos, const std::vector<double>& neg, double& j) {
    std::vector<double> c;
    for (const auto* v : {&pos, &neg})
        for (double x : *v)
            if (!std::isnan(x)) c.push_back(x);
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    c.push_back(c.empty() ? 0.0 : c.back() + 1);
    double t = c[0];
    j = -2;
    for (double x : c) {
        double a = 0, b = 0;
        for (double y : neg) a += y < x;
        for (double y : pos) b += y < x;
        const double jj = a / static_cast<double>(neg.size()) - b / static_cast<double>(pos.size());
        if (jj > j) j = jj, t = x;
    }
    return t;
}

// F2 pattern tokens as text: "^" dropped; no space before , . ; : ) ' nor after ( nor inside "' s".
std::string render_frame(const std::string& pat, const std::string& subj, const std::string& obj) {
    std::string out, prev;
    for (std::string t : split(pat, ' ')) {
        if (t.empty() || t == "^") continue;
        const bool glue = (t.size() == 1 && std::strchr(",.;:)'", t[0])) || prev == "(" || (prev == "'" && t == "s");
        prev = t;
        if (t == "[S]" || t == "[P]") t = subj;
        else if (t == "[O]") t = obj;
        if (!out.empty() && !glue) out += ' ';
        out += t;
    }
    return out;
}

// A frame with its subject (\x01) and object (\x02) filled.
std::string fill_frame(std::string f, const std::string& subj, const std::string& obj) {
    for (std::size_t p; (p = f.find('\x01')) != std::string::npos;) f.replace(p, 1, subj);
    const std::size_t p = f.find('\x02');
    return p == std::string::npos ? f : f.replace(p, 1, obj);
}

// A3 statement frames from F2's patterns.tsv, per relation: kept patterns first, then candidates by training
// subjects, at most 3 distinct ones. Skipped: slots CyphaLM cannot fill (<M>, #), and the object before the subject
// (nothing would tie the object to the subject). A 12th column `frame` (F2R's patterns_raw.tsv: the raw bytes with
// their markup, e.g. '''[S]''' (born [[[O]]]) is used as it stands, [S] and [O] filled in place.
std::map<std::string, std::vector<std::string>> load_frames(const std::string& path) {
    std::map<std::string, std::vector<std::tuple<int, int, std::string>>> by;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 11 || f[0] == "relation") continue;
        const bool raw = f.size() >= 12 && !f[11].empty();
        const std::string& p = raw ? f[11] : f[1];
        const std::size_t subj = std::min(p.find("[S]"), p.find("[P]")), obj = p.find("[O]");
        if (obj == std::string::npos || subj == std::string::npos || obj < subj || p.find("<M>") != std::string::npos ||
            p.find('#') != std::string::npos)
            continue;
        std::string r = p;
        for (const char* t : {"[S]", "[P]", "[O]"})
            for (std::size_t q; (q = r.find(t)) != std::string::npos;) r.replace(q, 3, t[1] == 'O' ? "\x02" : "\x01");
        by[f[0]].push_back({-std::atoi(f[10].c_str()), -std::atoi(f[2].c_str()), raw ? r : render_frame(p, "\x01", "\x02")});
    }
    std::map<std::string, std::vector<std::string>> out;
    for (auto& [rel, v] : by) {
        std::sort(v.begin(), v.end());
        auto& o = out[rel];
        for (const auto& t : v)
            if (o.size() < 3 && std::find(o.begin(), o.end(), std::get<2>(t)) == o.end()) o.push_back(std::get<2>(t));
    }
    return out;
}

// F2's relations for a template relation (a date question takes the year patterns too).
std::vector<std::string> f2_relations(const std::string& rel) {
    if (rel == "birth_date") return {"birth_date", "birth_year"};
    if (rel == "death_date") return {"death_date", "death_year"};
    return {rel};
}

// Markup dropped: [[ ]] ''' '' gone, | a space.
std::string strip_markup(const std::string& text) {
    std::string t;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const std::size_t skip = text.compare(i, 3, "'''") == 0 ? 3
                                 : (text.compare(i, 2, "''") == 0 || text.compare(i, 2, "[[") == 0 ||
                                    text.compare(i, 2, "]]") == 0)
                                     ? 2
                                     : 0;
        if (skip) i += skip - 1;
        else t += text[i] == '|' ? ' ' : text[i];
    }
    return t;
}

// A3 candidate objects inside CyphaLM's own text, as (the bytes it wrote, the value with the markup dropped): every
// year (3-4 digits) for a date relation; otherwise every run of capitalised words ("of" allowed inside) and, with
// head (a frame continuation, which fills the slot directly), its bytes before the first , . ; ( ) | or newline (and
// `close`, a raw frame's markup after its object, e.g. ]]) when they are a name: at most 5 words, one capitalised.
std::vector<std::pair<std::string, std::string>> r3_candidates(const std::string& text, bool date, bool head,
                                                               const std::string& close = "") {
    std::vector<std::pair<std::string, std::string>> out;
    auto trim = [](std::string h) {
        while (!h.empty() && h.back() == ' ') h.pop_back();
        while (!h.empty() && h[0] == ' ') h.erase(0, 1);
        return h;
    };
    if (head && !date) {
        const std::size_t cut = std::min(text.find_first_of(",.;()\n|"), close.empty() ? text.npos : text.find(close));
        const std::string sf = trim(text.substr(0, cut)), v = trim(strip_markup(sf));
        const auto ws = split(v, ' ');
        if (!v.empty() && ws.size() <= 5 &&
            std::any_of(ws.begin(), ws.end(), [](const std::string& x) { return !x.empty() && std::isupper(static_cast<unsigned char>(x[0])); }))
            out.push_back({sf, v});
    }
    std::vector<std::string> w;
    for (std::string x : split(strip_markup(text), ' ')) {
        while (!x.empty() && std::strchr(",.;:()\"!?\n", x.back())) x.pop_back();
        while (!x.empty() && std::strchr(",.;:()\"!?\n", x[0])) x.erase(0, 1);
        w.push_back(x);
    }
    auto cap = [&](std::size_t k) { return k < w.size() && !w[k].empty() && std::isupper(static_cast<unsigned char>(w[k][0])); };
    std::string run;
    for (std::size_t k = 0; k <= w.size(); ++k) {
        const std::string x = k < w.size() ? w[k] : "";
        if (date) {
            if (x.size() >= 3 && x.size() <= 4 && std::all_of(x.begin(), x.end(), [](char c) { return c >= '0' && c <= '9'; }))
                out.push_back({x, x});
            continue;
        }
        if (cap(k) || (x == "of" && !run.empty() && cap(k + 1))) {
            run += (run.empty() ? "" : " ") + x;
            continue;
        }
        if (run.size() >= 3 && run != "The") out.push_back({run, run});
        run.clear();
    }
    return out;
}

// A3: the winner is the candidate with the lowest statement bits per byte. Returns its index (-1: none) and the
// margin to the second (0 with one candidate).
int r3_choose(const std::vector<double>& bpb, double& margin) {
    int best = -1;
    for (int i = 0; i < static_cast<int>(bpb.size()); ++i)
        if (best < 0 || bpb[static_cast<std::size_t>(i)] < bpb[static_cast<std::size_t>(best)]) best = i;
    double second = 1e300;
    for (int i = 0; i < static_cast<int>(bpb.size()); ++i)
        if (i != best) second = std::min(second, bpb[static_cast<std::size_t>(i)]);
    margin = bpb.size() > 1 ? second - bpb[static_cast<std::size_t>(best)] : 0.0;
    return best;
}

// dktext rows, or A2 shard rows: question entity relation source response, one header row. Rows whose source ends in
// "_holdout" (from pages inside the BPC holdout) are dropped when `dropped` is given, and counted.
std::vector<DkRow> load_rows(const std::string& path, std::size_t* dropped = nullptr) {
    std::vector<DkRow> v;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 5) continue;
        if (dropped && f[3].size() >= 8 && f[3].compare(f[3].size() - 8, 8, "_holdout") == 0) {
            ++*dropped;
            continue;
        }
        v.push_back({f[0], f[1], f[2], f[3], f[4]});
    }
    return v;
}

// A2 leakage check: shard questions equal to a dev or test question after normalisation.
std::size_t shard_leaks(const std::vector<DkRow>& rows, const std::vector<std::string>& eval_q) {
    std::set<std::string> e;
    for (const auto& q : eval_q) e.insert(norm_text(q));
    std::size_t n = 0;
    for (const auto& r : rows) n += e.count(norm_text(r.q));
    return n;
}

// M11 (docs/reports/RAM_PROFILE.md): a stored qa model, so serving never trains or builds. `qa --save DIR` writes
// model.pk (the packed chat-stage trie, its weights and split, M7), ig.fm (InfiniGram over T, M8), low.fm (A1c's
// lowered index, with --parse), cls.cc (the class table, M10, with --cls) and qa.meta: the record of what the model
// was built from, the wiki-stage weights, the follower limit and the familiarity cuts, as text (exact hex doubles)
// with an FNV-1a 64 checksum. `qa --model DIR` loads them, and refuses a store whose record differs.
struct QaStore {
    double wf[6] = {0, 0, 0, 0, 0, 0};
    std::size_t all_followers = 0;
    std::array<double, kFam> fam_cut{};
    std::unique_ptr<FmInfiniGram> ig, low;
    std::unique_ptr<ClassContext> cc;
    std::unique_ptr<cypha::cyphalm::bkn::ByteKN> kn;  // BKN
};

std::uint64_t fnv64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ull;
    for (const char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    return h;
}

// What a qa model is built from: the split, the wiki bytes it reads, and the options that shape the training text,
// the tokens and the class table.
std::string qa_record(const Args& a, const QaOpts& o, const std::vector<std::uint8_t>& data) {
    std::ostringstream s;
    s << "train=" << a.cfg.train << " sample=" << a.cfg.sample << " wiki_fnv=" << std::hex
      << fnv64(std::string_view(reinterpret_cast<const char*>(data.data()), std::min(data.size(), a.cfg.sample))) << std::dec
      << " dolly=" << a.dolly << " dk=" << o.dk << " dk_ratio=" << o.dk_ratio << " fam=" << o.fam << " fam_stat=" << o.fam_stat
      << " fam_train=" << o.fam_train << " parse=" << o.parse << " shard=" << o.shard << " shard2=" << o.shard2 << " cls=" << o.cls;
    if (!a.bkn.empty()) s << " bkn=" << a.bkn;  // BKN (the way is chosen at serving, its lambdas fit on the holdout)
    return s.str();
}

bool save_qa(const std::string& dir, const std::string& record, const PackedModel& pm, const double w[6], const Config& cfg,
             const double wf[6], std::size_t all_followers, const std::array<double, kFam>& fam_cut, const FmInfiniGram& ig,
             const FmInfiniGram* low, const ClassContext* cc, const cypha::cyphalm::bkn::ByteKN* kn = nullptr,
             bool low_written = false) {  // BKN2: low.fm already written (and the index freed) before the build
    try {
        std::filesystem::create_directories(dir);
        pm.save(dir + "/model.pk", w, cfg);
        ig.save(dir + "/ig.fm");
        if (low) low->save(dir + "/low.fm");
        if (cc) cc->save(dir + "/cls.cc");
        if (kn) kn->save(dir + "/kn.bkn");  // BKN
        std::ostringstream m;
        auto reals = [&m](const double* x, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                char b[48];
                std::snprintf(b, sizeof b, " %a", x[i]);
                m << (std::isnan(x[i]) ? " nan" : b);
            }
        };
        m << "cypha-qa-store 1\nrecord " << record << "\nwiki_weights";
        reals(wf, 6);
        m << "\nfollower_limit " << all_followers << "\nfam_cut";
        reals(fam_cut.data(), fam_cut.size());
        m << "\nparts model.pk ig.fm" << (low || low_written ? " low.fm" : "") << (cc ? " cls.cc" : "") << (kn ? " kn.bkn" : "")
          << "\n";
        const std::string body = m.str();
        std::ofstream f(dir + "/qa.meta", std::ios::binary);
        f << body << "fnv " << std::hex << fnv64(body) << "\n";
        f.close();
        if (!f) throw std::runtime_error("cannot write " + dir + "/qa.meta");
    } catch (const std::exception& e) {
        std::cerr << "qa --save: " << e.what() << "\n";
        return false;
    }
    std::cerr << "stored the qa model in " << dir << "\n";
    return true;
}

bool load_qa(const std::string& dir, const std::string& record, const QaOpts& o, const Config& cfg, PackedModel& pm,
             double w[6], QaStore& st, bool want_kn = false) {
    try {
        std::ifstream f(dir + "/qa.meta", std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + dir + "/qa.meta");
        const std::string text((std::istreambuf_iterator<char>(f)), {});
        const std::size_t at = text.rfind("fnv ");
        if (text.rfind("cypha-qa-store 1\n", 0) != 0 || at == std::string::npos ||
            std::strtoull(text.c_str() + at + 4, nullptr, 16) != fnv64(std::string_view(text).substr(0, at)))
            throw std::runtime_error(dir + "/qa.meta is not a stored qa model, or it is damaged");
        std::map<std::string, std::string> kv;
        std::istringstream in(text.substr(0, at));
        for (std::string line; std::getline(in, line);)
            if (const std::size_t sp = line.find(' '); sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        if (kv["record"] != record)
            throw std::runtime_error("the store was built from other options or data\n  stored: " + kv["record"] +
                                     "\n  given:  " + record);
        auto reals = [&](const std::string& k, double* x, std::size_t n) {
            std::istringstream s(kv[k]);
            std::size_t i = 0;
            for (std::string t; i < n && s >> t; ++i) x[i] = t == "nan" ? std::nan("") : std::strtod(t.c_str(), nullptr);
            if (i != n) throw std::runtime_error(dir + "/qa.meta: bad " + k);
        };
        reals("wiki_weights", st.wf, 6);
        reals("fam_cut", st.fam_cut.data(), st.fam_cut.size());
        st.all_followers = static_cast<std::size_t>(std::strtoull(kv["follower_limit"].c_str(), nullptr, 10));
        Config c = cfg;
        pm = PackedModel::load(dir + "/model.pk", w, c);
        if (c.train != cfg.train || c.sample != cfg.sample) throw std::runtime_error(dir + "/model.pk has another split");
        st.ig = std::make_unique<FmInfiniGram>(FmInfiniGram::load(dir + "/ig.fm"));
        if (!o.parse.empty()) st.low = std::make_unique<FmInfiniGram>(FmInfiniGram::load(dir + "/low.fm"));
        if (!o.cls.empty()) st.cc = std::make_unique<ClassContext>(ClassContext::load(dir + "/cls.cc"));
        if (want_kn) st.kn = std::make_unique<cypha::cyphalm::bkn::ByteKN>(cypha::cyphalm::bkn::ByteKN::load(dir + "/kn.bkn"));  // BKN
    } catch (const std::exception& e) {
        std::cerr << "qa --model: " << e.what() << "\n";
        return false;
    }
    std::cerr << "served from " << dir << ": no training, no index or class build\n";
    return true;
}

// ---------------------------------------------------------------- S1: evidence cards (qa --cards FILE)
//
// Made after a set's answers and gates, from those decisions only: nothing here feeds an answer, a gate or a tag. One
// card a question, form and model: what the system shows (final_card) and its evidence (evidence.hpp). For an answer:
// the longest suffix of prompt + answer in qa's training text T, its count and its places. For a don't-know: the
// question's entity (E1's choice) in A1c's lowered training bytes, its page-title and mention counts and places.
// Gates: on dev each question's 5-fold CV gate, as qa's CV lines; on test the gate chosen on all of dev.
// Gate codes: G0 LOW_CERTAINTY; G1 UNKNOWN_ENTITY when the familiarity token is low, else LOW_CERTAINTY; G2
// UNKNOWN_ENTITY when the entity has no page title and fewer than c mentions, else LOW_CERTAINTY, c by 5-fold CV on the
// audit rows. G1x and G2x (reported only) also replace CyphaLM's own code with G1's or G2's.
// REASON lines (stdout): per rule, the don't-know cards whose code is the reference (reference_code, grading only).
// Dev also writes FILE.audit.tsv: 100 dev questions drawn before any run, chat form, trie + InfiniGram, for Odin.
// The rule the cards carry, chosen on dev (2026-10-28): G0. On the audit rows G0 and G2 are right on 82 of 100 and G1
// on 81; G2's CV estimate is 81, and G0 has no parameter.
constexpr const char* kCardRule = "G0";
constexpr int kAuditLm = 1;              // trie_ig
const std::size_t kG2Cuts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, std::numeric_limits<std::size_t>::max()};

struct CardContext {
    FmInfiniGram* ig = nullptr;   // T (wiki train + chat); locate is built on first use
    FmInfiniGram* low = nullptr;  // A1c's lowered training bytes (wiki train + Dolly); null without --parse
    evidence::Sources t_src, low_src;
    std::function<std::string(const std::string&)> entity_of;
    std::function<std::string(int, const QaRow&, const std::string&)> prompt;  // (form, row, token): decode's prompt
    const char* const* lm_name = nullptr;
    int nlm = 0;
    std::string path;
    std::size_t g2_cut = std::numeric_limits<std::size_t>::max();  // chosen on dev
    double locate_s = 0;
};

// T's sources: wiki pages, the document boundary, then the chat turns (kind shard, dk or dolly).
void card_sources_t(CardContext& cx, const std::vector<std::uint8_t>& data, std::size_t train, const std::string& chat,
                    const std::vector<DkRow>& sh1, const std::vector<DkRow>& sh2) {
    std::set<std::string> shard_q;
    for (const auto* v : {&sh1, &sh2})
        for (const auto& r : *v) shard_q.insert(r.q);
    cx.t_src.add_pages(data.data(), train);
    cx.t_src.add(train, "boundary");
    cx.t_src.add_turns(chat, train + std::strlen(kDocBoundary), [&](const std::string& turn) {
        const std::size_t nl = turn.find('\n');
        if (shard_q.count(turn.substr(6, nl == std::string::npos ? std::string::npos : nl - 6))) return std::string("shard");
        const std::size_t as = turn.find("Assistant: ");
        return std::string(as != std::string::npos && turn.find("I don't know.", as) != std::string::npos ? "dk" : "dolly");
    });
}

// The lowered index's sources (A1c: wiki train, a newline, then the Dolly text, which is read again here).
void card_sources_low(CardContext& cx, const Args& a, const std::vector<std::uint8_t>& data, std::size_t train) {
    std::vector<Turn> turns;
    const std::string dolly = load_dolly(a.dolly, turns);
    cx.low_src.add_pages(data.data(), train);
    cx.low_src.add_turns(dolly, train + 1, [](const std::string&) { return std::string("dolly"); });  // same offsets lowered
}

// The audit rows: 100 of n, drawn with mt19937_64(20261028) (Fisher-Yates, the same on every platform), in row order.
std::vector<std::size_t> audit_rows(std::size_t n) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(20261028);
    for (std::size_t i = n; i > 1; --i) std::swap(idx[i - 1], idx[rng() % i]);
    idx.resize(std::min<std::size_t>(100, n));
    std::sort(idx.begin(), idx.end());
    return idx;
}

void write_cards(CardContext& cx, const char* set, const std::vector<QaRow>& rows, const std::vector<Graded>& gs,
                 const std::vector<std::string>& toks, const std::vector<Gate>* full_gates) {
    const bool dev = full_gates == nullptr;
    auto t0 = std::chrono::steady_clock::now();
    for (FmInfiniGram* ix : {cx.ig, cx.low})
        if (ix && !ix->can_locate()) ix->build_locate();
    cx.locate_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    std::map<std::tuple<int, int, int>, Gate> cv;  // dev: (form, model, fold) -> the gate chosen on the other folds
    for (int form = 0; dev && form < 2; ++form)
        for (int l = 0; l < cx.nlm; ++l)
            for (int fold = 0; fold < 5; ++fold) {
                std::vector<const Graded*> tr;
                for (const auto& g : gs)
                    if (g.form == form && g.lm == l && g.row % 5 != fold) tr.push_back(&g);
                cv[{form, l, fold}] = choose_gate(tr, l % 2 == 1);
            }
    auto gate_of = [&](const Graded& g) {
        return dev ? cv[{g.form, g.lm, g.row % 5}] : (*full_gates)[static_cast<std::size_t>(g.form * cx.nlm + g.lm)];
    };
    struct Q {
        std::string e, rel, ev;
        std::size_t titles = 0, mentions = 0;
        bool low_tok = false;
    };
    std::vector<Q> qs(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        Q& q = qs[r];
        q.e = cx.entity_of(rows[r].q);
        q.rel = question_relation(rows[r].q, q.e);
        q.low_tok = toks[r].find("fam=low") != std::string::npos;
        q.ev = "entity=\"" + q.e + "\"";
        const std::string el = lowered(strip_article(q.e));
        if (cx.low && !el.empty()) {
            const auto m = evidence::occurrences(*cx.low, cx.low_src, el);
            q.titles = evidence::occurrences(*cx.low, cx.low_src, "<title>" + el + "</title>", 0).count;
            q.mentions = m.count;
            q.ev += " titles " + std::to_string(q.titles) + " mentions " + std::to_string(q.mentions);
            for (const auto& p : m.places) q.ev += " | " + p;
        }
    }
    auto rule_code = [&](const Q& q, const std::string& rule, std::size_t cut) -> std::string {
        if (rule == "G0") return "LOW_CERTAINTY";
        const bool unknown = rule == "G2" && cx.low ? q.titles == 0 && q.mentions < cut : q.low_tok;
        return unknown ? "UNKNOWN_ENTITY" : "LOW_CERTAINTY";
    };
    // The card under a rule (G1x, G2x: G1, G2, and CyphaLM's own code replaced too; reported only).
    auto card_of = [&](const Graded& g, const std::string& rule, std::size_t cut) {
        const Q& q = qs[static_cast<std::size_t>(g.row)];
        const bool over = rule.back() == 'x';
        const std::string gc = rule_code(q, over ? rule.substr(0, 2) : rule, cut);
        Card c = final_card(g.ans.text, gated_answer(g, gate_of(g)), gc, q.e, q.rel);
        if (over && c.kind == "dont_know") c.code = gc;
        return c;
    };
    // Tally of don't-know cards: n, right (the reference code), lenient right (in the slice: NO_EVIDENCE or
    // LOW_CERTAINTY), and the same for CyphaLM's own codes and the gate's.
    struct T {
        int n = 0, right = 0, lenient = 0, own = 0, own_right = 0, gate = 0, gate_right = 0;
    };
    auto tally = [&](const std::vector<const Graded*>& v, const std::string& rule, std::size_t cut) {
        T t;
        for (auto* g : v) {
            const Card c = card_of(*g, rule, cut);
            if (c.kind != "dont_know") continue;
            const QaRow& r = rows[static_cast<std::size_t>(g->row)];
            const bool ok = c.code == reference_code(r.in_slice, g->prose, g->covered);
            ++t.n, t.right += ok;
            t.lenient += r.in_slice ? (c.code == "NO_EVIDENCE" || c.code == "LOW_CERTAINTY") : c.code == "UNKNOWN_ENTITY";
            (c.by == "cyphalm" ? t.own : t.gate)++;
            (c.by == "cyphalm" ? t.own_right : t.gate_right) += ok;
        }
        return t;
    };
    auto print_t = [](const std::string& label, const T& t) {  // std::cout only: qa runs without stdio sync
        char b[512];
        std::snprintf(b, sizeof b,
                      " dont_know %d right %d rate %.4f wilson95 %.4f %.4f lenient %d | cyphalm_code %d right %d | "
                      "gate_code %d right %d\n",
                      t.n, t.right, t.n ? static_cast<double>(t.right) / t.n : 0.0, wilson_lo(t.right, t.n),
                      wilson_hi(t.right, t.n), t.lenient, t.own, t.own_right, t.gate, t.gate_right);
        std::cout << label << b;
    };
    auto cut_name = [](std::size_t c) { return c == kG2Cuts[11] ? std::string("inf") : std::to_string(c); };
    auto group = [&](int form, int l, const std::vector<std::size_t>* only) {
        std::vector<const Graded*> v;
        for (const auto& g : gs)
            if (g.form == form && g.lm == l &&
                (!only || std::binary_search(only->begin(), only->end(), static_cast<std::size_t>(g.row))))
                v.push_back(&g);
        return v;
    };
    const auto audit = audit_rows(rows.size());
    if (dev) {  // G2's cut: 5-fold CV over the audit rows (fold = row % 5), then chosen on all of them; ties: the larger
        const auto v = group(0, kAuditLm, &audit);
        auto best_cut = [&](int skip) {
            std::size_t best = kG2Cuts[0];
            int br = -1;
            for (std::size_t c : kG2Cuts) {
                std::vector<const Graded*> tr;
                for (auto* g : v)
                    if (g->row % 5 != skip) tr.push_back(g);
                const int r = tally(tr, "G2", c).right;
                if (r >= br) br = r, best = c;
            }
            return best;
        };
        T cvt;
        std::cout << "REASON_G2_CV cuts";
        for (int fold = 0; fold < 5; ++fold) {
            const std::size_t c = best_cut(fold);
            std::vector<const Graded*> te;
            for (auto* g : v)
                if (g->row % 5 == fold) te.push_back(g);
            const T t = tally(te, "G2", c);
            cvt.n += t.n, cvt.right += t.right, cvt.lenient += t.lenient, cvt.own += t.own, cvt.own_right += t.own_right,
                cvt.gate += t.gate, cvt.gate_right += t.gate_right;
            std::cout << " " << cut_name(c);
        }
        std::cout << "\n";
        print_t("REASON_AUDIT dev chat/trie_ig G2_cv", cvt);
        cx.g2_cut = best_cut(-1);
        std::cout << "REASON_G2 cut " << cut_name(cx.g2_cut) << " (all audit rows)\n";
    }
    for (const char* rule : {"G0", "G1", "G2", "G1x", "G2x"}) {
        if (dev) print_t(std::string("REASON_AUDIT dev chat/trie_ig ") + rule, tally(group(0, kAuditLm, &audit), rule, cx.g2_cut));
        for (int form = 0; form < 2; ++form)
            for (int l = 0; l < cx.nlm; ++l)
                print_t(std::string("REASON ") + set + " " + (form ? "wiki/" : "chat/") + cx.lm_name[l] + " " + rule,
                        tally(group(form, l, nullptr), rule, cx.g2_cut));
    }
    // The cards, under kCardRule.
    std::ofstream out(cx.path, dev ? std::ios::trunc : std::ios::app);
    if (dev) out << "set\trow\tform\tlm\tkind\tby\tcode\tgate\tentity\trelation\ttext\tevidence\n";
    int n_ans = 0, n_dk = 0;
    std::map<std::size_t, std::string> audit_line;
    for (const auto& g : gs) {
        const QaRow& r = rows[static_cast<std::size_t>(g.row)];
        const Q& q = qs[static_cast<std::size_t>(g.row)];
        const Card c = card_of(g, kCardRule, cx.g2_cut);
        const Gate gt = gate_of(g);
        std::string ev = q.ev;
        if (c.kind == "answer") {
            ++n_ans;
            const std::string pr = cx.prompt(g.form, r, toks[static_cast<std::size_t>(g.row)]);
            const auto f = evidence::longest(*cx.ig, cx.t_src, pr + c.text, 8192);
            ev = "match " + std::to_string(f.n) + " answer_bytes " + std::to_string(std::min(f.n, c.text.size())) + "/" +
                 std::to_string(c.text.size()) + " count " + std::to_string(f.count);
            for (const auto& p : f.places) ev += " | " + p;
        } else ++n_dk;
        const std::string gate = gt.f < 0 ? "none" : std::string(kFeatName[gt.f]) + " " + std::to_string(g.fv[gt.f]) +
                                                         (g.fv[gt.f] >= gt.tau ? " >= " : " < ") + std::to_string(gt.tau);
        out << set << "\t" << g.row << "\t" << (g.form ? "wiki" : "chat") << "\t" << cx.lm_name[g.lm] << "\t" << c.kind
            << "\t" << c.by << "\t" << c.code << "\t" << gate << "\t" << q.e << "\t" << q.rel << "\t" << printable(c.text)
            << "\t" << printable(ev) << "\n";
        if (dev && g.form == 0 && g.lm == kAuditLm && std::binary_search(audit.begin(), audit.end(), static_cast<std::size_t>(g.row))) {
            const std::string ref = reference_code(r.in_slice, g.prose, g.covered);
            const std::string why = !r.in_slice ? "not in the slice: its source article is outside wiki train"
                                    : g.prose == 1 ? "in the slice; a gold answer is in the source article's text"
                                    : g.covered == 2 ? "in the slice; the shard has its entity and relation"
                                                     : "in the slice; no gold answer in the source article, not in the shard";
            std::ostringstream l;
            l << g.row << "\t" << r.q << "\t" << r.in_slice << "\t" << g.prose << "\t" << g.covered << "\t" << ref << "\t" << why
              << "\t" << c.kind << "\t" << c.by << "\t" << c.code << "\t"
              << (c.kind == "dont_know" ? (c.code == ref ? "1" : "0") : "-") << "\t" << q.e << "\t" << q.titles << "\t"
              << q.mentions << "\t" << printable(c.text.substr(0, 240)) << "\t\n";
            audit_line[static_cast<std::size_t>(g.row)] = l.str();
        }
    }
    if (dev) {
        std::ofstream au(cx.path + ".audit.tsv");
        au << "row\tquestion\tin_slice\tfact_in_prose\tshard\treference\twhy\tkind\tby\tcode\tright\tentity\ttitles\tmentions"
              "\toutput\todin\n";
        for (const auto& [row, l] : audit_line) au << l;
    }
    const double card_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char b[512];
    std::snprintf(b, sizeof b,
                  "CARDS %s cards %zu answers %d dont_know %d rule %s locate_build_s %.3f locate_bytes ig %zu low %zu "
                  "sources_bytes %zu ms_per_card %.4f\n",
                  set, gs.size(), n_ans, n_dk, kCardRule, cx.locate_s, cx.ig->locate_bytes(),
                  cx.low ? cx.low->locate_bytes() : std::size_t{0}, cx.t_src.bytes() + cx.low_src.bytes(),
                  1000.0 * card_s / static_cast<double>(std::max<std::size_t>(1, gs.size())));
    std::cout << b;
}

int cmd_qa(const Args& a, const QaOpts& o) {
    const Config& cfg = a.cfg;
    // M11: --model DIR serves a stored model; --save DIR stores the one built here. A served model has no wiki-stage
    // training form, so the token statistics that read it (question_lp, ent_lp) and --fam-only are refused, and the
    // similar-context expert must key on wiki train (src=wiki): T is not kept.
    const bool serve = !a.model.empty();
    {
        const int fk = static_cast<int>(std::find(kFamName, kFamName + kFam, o.fam_stat) - kFamName);
        if ((serve || !a.save.empty()) &&
            (o.fam_only || (serve && !a.save.empty()) ||
             (serve && ((o.fam != "none" && (fk == 0 || fk == 6)) || (!o.sim.empty() && parse_sim(o.sim).src != "wiki") ||
                        !a.cards.empty())))) {
            std::cerr << "qa --model / --save: not with --fam-only, not both, and a served model takes neither a token "
                         "statistic that reads the wiki-stage trie (question_lp, ent_lp), nor --sim src=all, nor --cards "
                         "(S1's cards map T's bytes to the chat turns, which a served model does not rebuild)\n";
            return 2;
        }
    }
    const std::vector<QaRow> dev = load_tqa(o.dev);
    const std::vector<QaRow> test = o.test.empty() ? std::vector<QaRow>{} : load_tqa(o.test);
    const std::vector<QaRow> leak = o.leak.empty() ? std::vector<QaRow>{} : load_tqa(o.leak);
    if (dev.empty() || (!o.test.empty() && test.empty()) || (!o.leak.empty() && leak.empty())) {
        std::cerr << "cannot read the T-QA files\n";
        return 1;
    }
    std::vector<DkRow> dk;
    if (!o.dk.empty() && (dk = load_rows(o.dk)).empty()) {
        std::cerr << "no don't-know rows in " << o.dk << "\n";
        return 1;
    }
    // A2 shards: rows from the BPC holdout dropped; no shard question may equal a dev or test question.
    std::size_t drop1 = 0, drop2 = 0;
    const std::vector<DkRow> sh1 = o.shard.empty() ? std::vector<DkRow>{} : load_rows(o.shard, &drop1);
    const std::vector<DkRow> sh2 = o.shard2.empty() ? std::vector<DkRow>{} : load_rows(o.shard2, &drop2);
    std::vector<std::string> eval_q;
    std::string evaltext;
    for (const auto* set : {&dev, &test, &leak})
        for (const auto& r : *set) {
            eval_q.push_back(r.q);
            evaltext += lowered(r.q) + "\n" + lowered(r.title) + "\n";
        }
    if (!o.shard.empty() || !o.shard2.empty()) {
        const std::size_t l1 = shard_leaks(sh1, eval_q), l2 = shard_leaks(sh2, eval_q);
        std::cout << "shard rows " << sh1.size() << " dropped_holdout " << drop1 << " leak " << l1 << " | shard2 rows "
                  << sh2.size() << " dropped_holdout " << drop2 << " leak " << l2 << " | eval_questions "
                  << eval_q.size() << "\n";
        if (l1 || l2 || (test.empty() && leak.empty()) || (sh1.empty() && sh2.empty())) {
            std::cerr << "shard leakage, no shard rows, or no test questions to check against (--leak)\n";
            return 1;
        }
    }
    const auto parse = o.parse.empty() ? std::map<std::string, std::string>{} : load_parse(o.parse);
    if (!o.parse.empty() && parse.empty()) {
        std::cerr << "cannot read the parse " << o.parse << "\n";
        return 1;
    }
    const int fam_k = static_cast<int>(std::find(kFamName, kFamName + kFam, o.fam_stat) - kFamName);
    if (fam_k == kFam || (fam_k > 0 && (parse.empty() || dk.empty()))) {
        std::cerr << "--fam-stat: one of question_lp ent_count ent_link ent_bold ent_title ent_igfrac ent_lp ent_tb; an "
                     "entity statistic needs --parse and --dk\n";
        return 1;
    }
    std::unique_ptr<FmInfiniGram> low;  // A1c: the lowered training bytes, built below with --parse
    const Lm* wiki_lmp = nullptr;
    std::map<std::string, std::string> chosen;
    // E1's entity for a question. With several candidates (A1d), the best by the --fam-stat statistic (ties: the
    // earlier; the first when the statistic is question_lp).
    auto entity_of = [&](const std::string& q) -> std::string {
        const auto it = parse.find(q);
        if (it == parse.end() || it->second.find('|') == std::string::npos) return it == parse.end() ? "" : it->second;
        const auto c = split(it->second, '|');
        if (const auto ch = chosen.find(q); ch != chosen.end() && !ch->second.empty()) return ch->second;  // M14
        if (fam_k == 0 || !low) return c[0];
        std::string& best = chosen[q];
        if (best.empty()) {
            double bv = std::nan("");
            best = c[0];
            for (const auto& e : c) {
                const double x = fam_stats(*wiki_lmp, *low, q, e)[static_cast<std::size_t>(fam_k)];
                if (!std::isnan(x) && (std::isnan(bv) || x > bv)) bv = x, best = e;
            }
        }
        return best;
    };
    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;
    const std::string record = qa_record(a, o, data);  // M11
    QaStore st;
    PackedModel pm;
    double w[6];
    if (serve && !load_qa(a.model, record, o, cfg, pm, w, st, !a.bkn.empty())) return 1;
    Model model;
    if (!serve) {
        model.data = data.data();
        model.init(cfg);
        init_weights(w, cfg);
        train_span(model, data, w, 0, cfg.train, cfg);
        std::cerr << "wiki trained, nodes " << model.nodes.size() << "\n";
    }

    std::vector<Turn> turns;
    std::string dolly_text = serve ? std::string() : load_dolly(a.dolly, turns);
    if (!serve && turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return 1;
    }
    auto instruction = [&](std::size_t i) {  // "User: <instruction>\n..."
        const std::string& p = turns[i].prompt;
        return p.substr(6, p.find('\n') - 6);
    };
    // The model state after the wiki pass, before any chat (the match expert does not fire on a separate buffer).
    double wf[6];
    std::copy(serve ? st.wf : w, (serve ? st.wf : w) + 6, wf);
    Lm wiki_lm{&model, wf, cfg.train, true};
    if (serve) wiki_lm.model = nullptr, wiki_lm.packed = &pm;  // as after the chat passes below
    wiki_lmp = &wiki_lm;

    // A1c: entity familiarity, from CyphaLM's wiki trie and its lowered training bytes (wiki train + Dolly).
    std::array<double, kFam> fam_cut;
    fam_cut.fill(std::nan(""));
    if (serve) fam_cut = st.fam_cut, low = std::move(st.low);  // M11: the build's cuts and index
    if (parse.empty()) std::string().swap(dolly_text);  // only A1c's lowered text reads it
    if (!serve && !parse.empty()) {
        std::vector<std::uint8_t> lt(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.train));
        lt.push_back('\n');
        lt.insert(lt.end(), dolly_text.begin(), dolly_text.end());
        std::string().swap(dolly_text);
        for (auto& c : lt) c = static_cast<std::uint8_t>(std::tolower(c));
        low = std::make_unique<FmInfiniGram>(std::move(lt));  // the index takes the lowered bytes
        std::vector<std::array<double, kFam>> dv;
        for (const auto& r : dev) {
            const auto it = parse.find(r.q);
            dv.push_back(fam_stats(wiki_lm, *low, r.q, it == parse.end() ? "" : it->second));
        }
        // Dev AUROC (in-slice above not-in-slice) with a bootstrap 95% interval over the dev questions, and the paired
        // bootstrap interval of its difference from ent_bold's (A1d: the statistic that reads no XML markup).
        std::mt19937_64 rng(20261028);
        std::vector<std::vector<std::size_t>> boot(2000, std::vector<std::size_t>(dev.size()));
        for (auto& b : boot)
            for (auto& i : b) i = rng() % dev.size();
        std::vector<std::size_t> all(dev.size());
        std::iota(all.begin(), all.end(), 0);
        std::vector<std::vector<double>> bsk(kFam);
        for (int k = 0; k < kFam; ++k) {
            auto au = [&](const std::vector<std::size_t>& idx) {
                std::vector<double> p, n;
                for (std::size_t i : idx) (dev[i].in_slice ? p : n).push_back(dv[i][static_cast<std::size_t>(k)]);
                return auroc(p, n);
            };
            for (const auto& b : boot) bsk[static_cast<std::size_t>(k)].push_back(au(b));
            std::vector<double> bs = bsk[static_cast<std::size_t>(k)], d;
            std::sort(bs.begin(), bs.end());
            for (std::size_t b = 0; k > 3 && b < boot.size(); ++b) d.push_back(bsk[static_cast<std::size_t>(k)][b] - bsk[3][b]);
            std::sort(d.begin(), d.end());
            int miss = 0;
            for (const auto& v : dv) miss += std::isnan(v[static_cast<std::size_t>(k)]);
            std::cout << "FAMSTAT " << kFamName[k] << " dev_auroc " << au(all) << " boot95 " << bs[49] << " "
                      << bs[1949] << " missing " << miss;
            if (!d.empty()) std::cout << " minus_ent_bold boot95 " << d[49] << " " << d[1949];
            std::cout << "\n";
        }
        // Entity choice (grading only): the chosen candidate against the in-slice question's source title.
        {
            int n = 0, first = 0, any = 0, by[kFam] = {0};
            auto same = [](const std::string& x, const std::string& t) {
                return norm_text(strip_article(x)) == norm_text(strip_article(t));
            };
            for (const auto& r : dev) {
                const auto it = parse.find(r.q);
                if (!r.in_slice || it == parse.end()) continue;
                ++n;
                const auto c = split(it->second, '|');
                first += same(c[0], r.title);
                any += std::any_of(c.begin(), c.end(), [&](const std::string& e) { return same(e, r.title); });
                for (int k = 1; k < kFam; ++k) {
                    std::string best = c[0];
                    double bv = std::nan("");
                    for (const auto& e : c) {
                        const double x = fam_stats(wiki_lm, *low, r.q, e)[static_cast<std::size_t>(k)];
                        if (!std::isnan(x) && (std::isnan(bv) || x > bv)) bv = x, best = e;
                    }
                    by[k] += same(best, r.title);
                }
            }
            std::cout << "ENTITY in_slice " << n << " first_eq_title " << first << " any_candidate_eq_title " << any;
            for (int k = 1; k < kFam; ++k) std::cout << " " << kFamName[k] << " " << by[k];
            std::cout << "\n";
        }
        // Cuts from training-side entities only: titles of the wiki-train pages (they have an article) against the
        // don't-know entities (no article in the slice); neither may occur in dev or test text.
        if (!dk.empty()) {
            std::vector<double> P[kFam], N[kFam];
            for_pages(std::string(reinterpret_cast<const char*>(data.data()), cfg.train),
                      [&](const std::string& title, const std::string&) {
                          if (evaltext.find(lowered(title)) != std::string::npos) return;
                          const auto v = fam_stats(wiki_lm, *low, title, title);
                          for (int k = 0; k < kFam; ++k) P[k].push_back(v[static_cast<std::size_t>(k)]);
                      });
            std::set<std::string> seen;
            for (const auto& r : dk)
                if (seen.insert(r.entity).second && evaltext.find(lowered(r.entity)) == std::string::npos) {
                    const auto v = fam_stats(wiki_lm, *low, r.entity, r.entity);
                    for (int k = 0; k < kFam; ++k) N[k].push_back(v[static_cast<std::size_t>(k)]);
                }
            for (int k = 0; k < kFam; ++k) {
                double j = 0;
                fam_cut[static_cast<std::size_t>(k)] = youden(P[k], N[k], j);
                std::cout << "FAMCAL " << kFamName[k] << " train_titles " << P[k].size() << " dk_entities " << N[k].size()
                          << " auroc " << auroc(P[k], N[k]) << " cut " << fam_cut[static_cast<std::size_t>(k)]
                          << " youden_j " << j << "\n";
                // A1d: the token this cut gives the dev questions, and its precision and recall on not-in-slice if
                // CyphaLM abstains exactly on low and mid (as it did in every A1c model).
                int c[3][2] = {{0, 0}, {0, 0}, {0, 0}};  // low, mid, high x (not-in-slice, in-slice)
                for (std::size_t i = 0; i < dev.size(); ++i) {
                    const double x = dv[i][static_cast<std::size_t>(k)];
                    ++c[std::isnan(x) ? 1 : x < fam_cut[static_cast<std::size_t>(k)] ? 0 : 2][dev[i].in_slice];
                }
                const double ab = c[0][0] + c[0][1] + c[1][0] + c[1][1], tp = c[0][0] + c[1][0], nis = tp + c[2][0];
                std::cout << "FAMTOK " << kFamName[k] << " low " << c[0][1] << "/" << c[0][0] << " mid " << c[1][1] << "/"
                          << c[1][0] << " high " << c[2][1] << "/" << c[2][0] << " (in/not) precision "
                          << (ab ? tp / ab : 0.0) << " " << wilson_lo(tp, ab) << " " << wilson_hi(tp, ab) << " recall "
                          << (nis ? tp / nis : 0.0) << " " << wilson_lo(tp, nis) << " " << wilson_hi(tp, nis) << "\n";
            }
        }
        // ANS (--fam-read, report only): what each statistic reads. Neither reads anything in a question but its spans.
        // FAMREAD: each dev question's best span and its value under ent_title and ent_bold (log2(1 + count)).
        // FAMDOLLY: the token each statistic gives the Dolly instructions (real user questions, span candidates) at the
        // training-side cuts above (low below the cut, mid with no entity, high above).
        if (o.fam_read && !dk.empty()) {
            auto best_of = [&](const std::string& q, int k, double& bv) {
                std::string best;
                bv = std::nan("");
                const auto it = parse.find(q);
                if (it == parse.end()) return best;
                for (const auto& e : split(it->second, '|')) {
                    const double x = fam_stats(wiki_lm, *low, q, e)[static_cast<std::size_t>(k)];
                    if (!std::isnan(x) && (std::isnan(bv) || x > bv)) bv = x, best = e;
                }
                return best;
            };
            for (const auto& r : dev) {
                double vt = 0, vb = 0;
                const std::string et = best_of(r.q, 4, vt), eb = best_of(r.q, 3, vb);  // ent_title, ent_bold
                std::cout << "FAMREAD " << (r.in_slice ? "in" : "not") << "\t" << r.q << "\ttitle\t" << et << "\t" << vt
                          << "\tbold\t" << eb << "\t" << vb << "\n";
            }
            std::array<std::array<int, 3>, kFam> c{};
            for (std::size_t i = 0; i < turns.size(); ++i) {
                const std::string q = instruction(i);
                const auto it = parse.find(q);
                const auto v = fam_stats(wiki_lm, *low, q, it == parse.end() ? "" : it->second);
                for (std::size_t k = 1; k < kFam; ++k)
                    ++c[k][std::isnan(v[k]) ? 1 : v[k] < fam_cut[k] ? 0 : 2];
            }
            for (std::size_t k = 1; k < kFam; ++k) {
                const double n = static_cast<double>(turns.size());
                std::cout << "FAMDOLLY " << kFamName[k] << " low " << c[k][0] << " mid " << c[k][1] << " high " << c[k][2]
                          << " low_share " << c[k][0] / n << " " << wilson_lo(c[k][0], n) << " " << wilson_hi(c[k][0], n)
                          << "\n";
            }
        }
        if (o.fam_only) return 0;
    }

    // Familiarity token: A1b (question_lp; cuts from the Dolly and don't-know questions) or A1c (an entity statistic;
    // low below the calibrated cut, high above, mid when E1 found no entity).
    const bool fam_on = o.fam != "none";
    std::size_t n_dk = 0;
    if (!dk.empty() && o.dk_ratio > 0)
        n_dk = std::min(dk.size(), static_cast<std::size_t>(std::llround(o.dk_ratio / (1 - o.dk_ratio) * turns.size())));
    if (o.copy_dk && n_dk >= dk.size()) {
        std::cerr << "--copy-dk fits its lambdas on the don't-know rows held out of training: needs --dk with more rows "
                     "than --dk-ratio uses\n";
        return 1;
    }
    std::vector<std::string> tok_dolly(turns.size()), tok_dk(dk.size()), tok_dev(dev.size()), tok_test(test.size()),
        tok_sh1(sh1.size()), tok_sh2(sh2.size());
    std::vector<std::string> pre_dk(dk.size()), pre_dev(dev.size()), pre_test(test.size());  // DK2: start frames
    std::function<std::string(const std::string&)> tokq;
    if (fam_on && fam_k == 0) {
        std::vector<double> fd(turns.size()), fk(dk.size());
        for (std::size_t i = 0; i < turns.size(); ++i) fd[i] = text_logprob(wiki_lm, "User: ", instruction(i));
        for (std::size_t i = 0; i < dk.size(); ++i) fk[i] = text_logprob(wiki_lm, "User: ", dk[i].q);
        // t1: the cut that best separates don't-know questions (below) from Dolly ones (Youden J);
        // t2: the median Dolly familiarity above t1.
        double t1 = 0, t2 = 0;
        std::vector<double> cand = fd;
        cand.insert(cand.end(), fk.begin(), fk.end());
        std::sort(cand.begin(), cand.end());
        double bestj = -1;
        for (std::size_t c = 0; c < cand.size(); c += 25) {
            const double t = cand[c];
            double tp = 0, fp = 0;
            for (double x : fk) tp += x < t;
            for (double x : fd) fp += x < t;
            const double j = tp / fk.size() - fp / fd.size();
            if (j > bestj) {
                bestj = j;
                t1 = t;
            }
        }
        std::vector<double> above;
        for (double x : fd)
            if (x >= t1) above.push_back(x);
        std::sort(above.begin(), above.end());
        t2 = above.empty() ? t1 : above[above.size() / 2];
        std::cout << "fam_thresholds t1 " << t1 << " t2 " << t2 << " youden_j " << bestj << "\n";
        tokq = [&wiki_lm, t1, t2](const std::string& q) {
            const double x = text_logprob(wiki_lm, "User: ", q);
            return std::string(x < t1 ? "low" : x < t2 ? "mid" : "high");
        };
    } else if (fam_on) {
        const double cut = fam_cut[static_cast<std::size_t>(fam_k)];
        std::cout << "fam_thresholds " << o.fam_stat << " cut " << cut << "\n";
        tokq = [&, cut](const std::string& q) {
            const double x = fam_stats(wiki_lm, *low, q, entity_of(q))[static_cast<std::size_t>(fam_k)];
            return std::string(std::isnan(x) ? "mid" : x < cut ? "low" : "high");
        };
    }
    if (fam_on) {
        // --fam-train outcome: the training turns carry the gate's decision instead (high on answered turns, low on
        // don't-know turns); dev and test still get the statistic's token. --fam-train low (A1d): the same, but answered
        // turns and high questions carry no token at all.
        const bool lo = o.fam_train == "low", oc = o.fam_train == "outcome" || lo;
        // ANS: --fam-ent writes the entity's frames before the level, in every token that is written (A1d's form):
        // a don't-know or shard row's own entity, else the chosen span of the question (computed only when needed).
        // DK2: --fam-ent-at answer|start: no frame in any training turn (fr_train); answer keeps them in the token of
        // the prompts the model answers, start writes them before "User: " instead (pre_*). framed: a prompt it answers.
        const bool fr_train = o.fam_ent_at == "all", fr_tok = o.fam_ent_at != "start";
        auto tok = [&](const std::string& l, const std::function<std::string()>& ent, bool framed) {
            if (lo && l == "high") return std::string();
            const bool f = framed && fr_tok && !o.fam_ent.empty();
            return fam_token(l, f ? o.fam_ent : std::string(), f ? ent() : std::string());
        };
        std::map<std::string, int> cd, ck;
        std::map<std::string, std::pair<int, int>> cv;
        for (std::size_t i = 0; i < turns.size(); ++i) {
            const std::string l = oc ? "high" : tokq(instruction(i));
            ++cd["<|fam=" + l + "|>"];
            tok_dolly[i] = tok(l, [&] { return entity_of(instruction(i)); }, fr_train);
        }
        for (std::size_t i = 0; i < dk.size(); ++i) {
            const std::string l = oc ? "low" : tokq(dk[i].q);
            ++ck["<|fam=" + l + "|>"];
            // DK2: rows below n_dk are training turns (--dk-plain drops their token); the rest are the held-out turns
            // --copy-dk fits on, prompts the model answers.
            const bool train = i < n_dk;
            tok_dk[i] = train && o.dk_plain > 0 && dk_plain_turn(i, o.dk_plain) ? std::string()
                                                                                 : tok(l, [&] { return dk[i].entity; }, !train || fr_train);
            if (!train && !fr_tok && !o.fam_ent.empty()) pre_dk[i] = fam_frames(o.fam_ent, dk[i].entity);
        }
        for (std::size_t i = 0; i < dev.size(); ++i) {
            const std::string l = tokq(dev[i].q);
            (dev[i].in_slice ? cv["<|fam=" + l + "|>"].first : cv["<|fam=" + l + "|>"].second)++;
            tok_dev[i] = tok(l, [&] { return entity_of(dev[i].q); }, true);
            if (!fr_tok && !o.fam_ent.empty()) pre_dev[i] = fam_frames(o.fam_ent, entity_of(dev[i].q));  // DK2
        }
        for (std::size_t i = 0; i < test.size(); ++i) {
            tok_test[i] = tok(tokq(test[i].q), [&] { return entity_of(test[i].q); }, true);
            if (!fr_tok && !o.fam_ent.empty()) pre_test[i] = fam_frames(o.fam_ent, entity_of(test[i].q));  // DK2
        }
        for (std::size_t i = 0; i < sh1.size(); ++i)
            tok_sh1[i] = tok(oc ? "high" : tokq(sh1[i].q), [&] { return sh1[i].entity; }, fr_train);
        for (std::size_t i = 0; i < sh2.size(); ++i)
            tok_sh2[i] = tok(oc ? "high" : tokq(sh2[i].q), [&] { return sh2[i].entity; }, fr_train);
        for (auto& [k, v] : cd) std::cout << "fam_dolly " << k << " " << v << "\n";
        for (auto& [k, v] : ck) std::cout << "fam_dk " << k << " " << v << "\n";
        for (auto& [k, v] : cv) std::cout << "fam_dev " << k << " in_slice " << v.first << " not_in_slice " << v.second << "\n";
        if (o.dk_plain > 0 || o.fam_ent_at != "all") {  // DK2
            std::size_t plain = 0, framed_train = 0, framed_ans = 0;
            for (std::size_t i = 0; i < dk.size(); ++i) {
                plain += i < n_dk && tok_dk[i].empty();
                const bool fr = tok_dk[i].find("<|fam=") != 0 && !tok_dk[i].empty();
                (i < n_dk ? framed_train : framed_ans) += fr || !pre_dk[i].empty();
            }
            std::cout << "dk2_text dk_plain " << o.dk_plain << " plain_turns " << plain << " of " << n_dk << " fam_ent_at "
                      << o.fam_ent_at << " framed_training_turns " << framed_train << " framed_heldout_turns " << framed_ans
                      << " copy_frame " << o.copy_frame << "\n";
        }
    }
    // M14: with an entity statistic in the token, every dev and test entity was chosen above and is kept (`chosen`), and
    // nothing below reads the lowered index: free it (a store writes it later, so --save keeps it).
    if (fam_on && fam_k > 0 && a.save.empty() && a.cards.empty()) low.reset();  // --cards reads it (S1)
    // BKN2: when Kneser-Ney is built below, --save writes the store's low.fm here and frees the index (M17's condition),
    // so it is not resident through the chat training, the index, the class table and Kneser-Ney's build.
    bool low_written = false;
    if (!serve && !a.bkn.empty() && !a.save.empty() && fam_on && fam_k > 0 && a.cards.empty() && low) {
        try {
            std::filesystem::create_directories(a.save);
            low->save(a.save + "/low.fm");
        } catch (const std::exception& e) {
            std::cerr << "qa --save: " << e.what() << "\n";
            return 1;
        }
        low_written = true;
        low.reset();
    }
    auto with_tok = [&](const std::string& prompt, const std::string& tok) {
        // prompt ends in "Assistant: "; "before" puts the token in front of it, "after" behind it.
        if (!fam_on) return prompt;
        const std::size_t at = prompt.size() - 11;
        return o.fam == "before" ? prompt.substr(0, at) + tok + prompt.substr(at) : prompt + tok;
    };

    // Chat text: Dolly, with the extra turns (don't-know pairs at dk_ratio of all turns, then every shard row), each
    // list spread evenly through it. A served model (M11) has it trained in already.
    CardContext cards;  // S1: where T's bytes come from (wiki pages, then chat turns), for --cards
    if (!serve) {
        std::vector<std::pair<std::size_t, std::string>> extra;  // (after which Dolly turn, text)
        auto spread = [&](const std::vector<DkRow>& rows, std::size_t n, const std::vector<std::string>& toks) {
            for (std::size_t j = 0; j < n; ++j)
                extra.push_back({(j + 1) * turns.size() / (n + 1),
                                 with_tok("User: " + rows[j].q + "\nAssistant: ", toks[j]) + rows[j].response + "\n\n"});
        };
        spread(dk, n_dk, tok_dk);
        spread(sh1, sh1.size(), tok_sh1);
        spread(sh2, sh2.size(), tok_sh2);
        std::stable_sort(extra.begin(), extra.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        std::string chat;
        for (std::size_t i = 0, j = 0; i < turns.size(); ++i) {
            chat += with_tok(turns[i].prompt, tok_dolly[i]) + turns[i].answer + "\n\n";
            while (j < extra.size() && extra[j].first <= i) chat += extra[j++].second;
        }
        std::cout << "chat_turns dolly " << turns.size() << " dont_know " << n_dk << " share "
                  << static_cast<double>(n_dk) / static_cast<double>(turns.size() + n_dk) << " fam " << o.fam << " shard "
                  << sh1.size() << " shard2 " << sh2.size() << "\n";
        if (!a.cards.empty()) card_sources_t(cards, data, cfg.train, chat, sh1, sh2);  // S1: where T's bytes come from
        const std::size_t chat_at = append_chat(model, data, chat);
        std::string().swap(chat);  // data holds the chat bytes now, and nothing below reads the turns
        std::vector<Turn>().swap(turns);
        train_span(model, data, w, chat_at, data.size(), cfg);
        train_span(model, data, w, chat_at, data.size(), cfg);
        model.compact_match();
        std::cerr << "chat trained\n";
        // M4: the packed form serves from here and the training form is freed. A later wiki_lm read (entity_of) sees the
        // chat-trained counts with no online match in either form (compact_match freed the training hash).
        pm = pack(model);
        model.release();
        wiki_lm.model = nullptr;
        wiki_lm.packed = &pm;
    }

    // T, the training text (wiki train + chat): InfiniGram takes it (M6 order, after the trie: A1c's tokens need the wiki
    // trie before the chat text exists), then the class expert is built from the text the index gives back (M16: so the
    // class table is not resident during the index's scratch). Both builds come before the holdout arrays below, which
    // they do not read. A served model (M11) loads both instead, and T is not made.
    std::vector<std::uint8_t> T;
    const std::size_t all_followers = serve ? st.all_followers : training_text(data, cfg, T);
    // InfiniGram takes T, unless the similar-context expert reads it (A4, src=all: its keys and its neighbours are T's).
    const bool sim_reads_t = !o.sim.empty() && parse_sim(o.sim).src != "wiki";
    FmInfiniGram ig = serve ? std::move(*st.ig) : sim_reads_t ? FmInfiniGram(T.data(), T.size()) : FmInfiniGram(std::move(T));
    std::cerr << (serve ? "served" : "index built") << "\n";  // S1: not const, --cards builds locate on it
    std::unique_ptr<ClassContext> cc = std::move(st.cc);
    // BKN (--bkn): byte Kneser-Ney on T (wiki train + chat), after the index and the class table, from the same copy
    // of T the class build takes from the index (BKN2: one copy; T itself when the similar-context expert keeps it).
    // A served model loads it.
    std::unique_ptr<cypha::cyphalm::bkn::ByteKN> kn = std::move(st.kn);
    const bool kn_rep = a.bkn_way == "a";
    const bool kn_build = !serve && !a.bkn.empty();
    if (!serve && (!o.cls.empty() || kn_build)) {
        if (!sim_reads_t) T = ig.text();  // M16: one walk through the index, so T was never copied
        if (!o.cls.empty()) {
            cc = make_class_context(o.cls);
            if (!cc) return 1;
            cc->build(T.data(), T.size());
            std::cerr << "class table built\n";
        }
        if (kn_build) {
            kn = make_bkn(a.bkn, T.data(), T.size());
            if (!kn) return 2;
            std::cerr << "byte Kneser-Ney built: " << kn->bytes() << " bytes\n";
        }
        if (!sim_reads_t) std::vector<std::uint8_t>().swap(T);
    }
    if (!a.save.empty() &&
        !save_qa(a.save, record, pm, w, cfg, wf, all_followers, fam_cut, ig, low.get(), cc.get(), kn.get(), low_written))
        return 1;
    if (!a.save.empty() && fam_on && fam_k > 0 && a.cards.empty()) low.reset();  // M17: written; nothing below reads it (M14)
    // A4: the similar-context expert, keys over wiki train + chat (src=all: InfiniGram's bytes) or wiki train only
    // (src=wiki). M13: built here, before the per-byte holdout arrays, so its build does not sit on them; its lambdas
    // and lines come after the class expert's, below.
    std::unique_ptr<SimilarContext> sc;
    const std::uint8_t* sc_text = nullptr;
    double build_s = 0;
    if (!o.sim.empty()) {
        const SimSpec ss = parse_sim(o.sim);
        if (!ss.ok) {
            std::cerr << "bad --sim spec " << o.sim << "\n";
            return 2;
        }
        if (!(sc = make_similar_context(ss))) return 1;
        const auto t0 = std::chrono::steady_clock::now();
        sc_text = ss.src == "wiki" ? data.data() : T.data();
        sc->build(sc_text, ss.src == "wiki" ? cfg.train : T.size());
        build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cerr << "similar-context expert built\n";
    }
    // Holdout: the trie mixture's BPC (= chat's wiki_bpc_after_chat), then InfiniGram's lambdas on the first half and
    // the trie + InfiniGram BPC on the second half (as context8k, cap 8192); with --copy, the copy expert's lambdas
    // on the first half (its context: the holdout stream itself) and its BPC and LCI gain on the second half.
    const std::size_t H = cfg.sample - cfg.train, half = H / 2;
    const bool copy_fit = o.copy || o.copy_out;  // B4C: the default copy expert reads the copy arrays too
    std::vector<double> pt(H), pinf(H), pig(H), pcp(copy_fit ? H : 0);  // M13: each array only when it is read
    std::vector<int> bk(H), cbk(copy_fit ? H : 0);
    double bits_all = 0;
    for (std::size_t k = 0; k < H; ++k) {
        const std::size_t i = cfg.train + k;
        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
        double M[256];
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
        bits_all += -std::log2(pt[k]);
    }
    // InfiniGram over the holdout (cap 8192). BKN3: queried before Kneser-Ney is mixed into pt (the queries do not read
    // pt), so lam0, InfiniGram's lambdas over the trie mixture without Kneser-Ney, can be fit for the answers.
    double lam[kBuckets] = {0}, lam0[kBuckets] = {0}, lc_t[kCopyBuckets] = {0}, lc_i[kCopyBuckets] = {0};
    FmInfiniGram::Cursor cur;
    for (std::size_t k = 0, prev = 0; k < H; ++k) {
        const std::size_t i = cfg.train + k;
        const std::size_t len = std::min<std::size_t>(i, 8192);
        const auto r = ig.query(data.data() + (i - len), len, 8192, k == 0 ? -1 : std::min<int>(static_cast<int>(prev) + 1, 8192),
                                1, cur);
        prev = static_cast<std::size_t>(r.n);
        pinf[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
        bk[k] = bucket(r.n, r.total);
    }
    const bool kn_off_ans = kn && a.bkn_answers == "off";  // BKN3: Kneser-Ney off inside answers
    if (kn_off_ans) fit_lambdas(pinf, pt, bk, half, kBuckets, lam0);
    std::vector<double> lkn;  // BKN: Kneser-Ney into the trie mixture, before InfiniGram and the other experts
    if (kn) {
        double b0 = 0, b1 = 0;
        for (std::size_t k = half; k < H; ++k) b0 -= std::log2(pt[k]);
        lkn = mix_bkn_into(*kn, data.data(), cfg.train, half, kn_rep, pt);
        for (std::size_t k = half; k < H; ++k) b1 -= std::log2(pt[k]);
        std::cout << "bkn " << a.bkn << " way " << a.bkn_way << " table_bytes " << kn->bytes() << " holdout2_trie_bpc "
                  << b0 / static_cast<double>(H - half) << " holdout2_trie_bkn_bpc " << b1 / static_cast<double>(H - half) << "\n";
        if (kn_off_ans) std::cout << "bkn_answers off (Kneser-Ney off inside answers; InfiniGram's lambdas there fit without it)\n";
    }
    fit_lambdas(pinf, pt, bk, half, kBuckets, lam);
    double bits_trie2 = 0, bits_ig2 = 0;
    for (std::size_t k = 0; k < H; ++k) {
        pig[k] = lam[bk[k]] * pinf[k] + (1 - lam[bk[k]]) * pt[k];
        if (k >= half) bits_trie2 += -std::log2(pt[k]), bits_ig2 += -std::log2(pig[k]);
    }
    std::vector<double>().swap(pinf);  // M13: not read again
    std::vector<int>().swap(bk);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "holdout_mix_bpc " << bits_all / static_cast<double>(H) << "\n";
    std::cout << "holdout2_trie_bpc " << bits_trie2 / static_cast<double>(H - half) << " holdout2_trie_ig_bpc "
              << bits_ig2 / static_cast<double>(H - half) << "\n";
    // B4: the class-context expert over the same training bytes as InfiniGram (wiki train + chat), built above, lambdas
    // on the first half of the holdout over the trie (lcl_t) and over trie + InfiniGram (lcl_i).
    double lcl_t[ClassContext::kBuckets] = {0}, lcl_i[ClassContext::kBuckets] = {0};
    std::vector<double> pcl(cc ? H : 0);
    std::vector<int> clb(cc ? H : 0);
    if (cc) {
        for (std::size_t k = 0; k < H; ++k) {
            double P[256];
            clb[k] = ClassContext::bucket(cc->predict(data.data(), cfg.train + k, P));
            pcl[k] = P[data[cfg.train + k]];
        }
        fit_lambdas(pcl, pt, clb, half, ClassContext::kBuckets, lcl_t);
        fit_lambdas(pcl, pig, clb, half, ClassContext::kBuckets, lcl_i);
        double bt = 0, bi = 0;
        for (std::size_t k = half; k < H; ++k) {
            bt += -std::log2(lcl_t[clb[k]] * pcl[k] + (1 - lcl_t[clb[k]]) * pt[k]);
            bi += -std::log2(lcl_i[clb[k]] * pcl[k] + (1 - lcl_i[clb[k]]) * pig[k]);
        }
        std::cout << "cls " << o.cls << " contexts " << cc->contexts() << " table_bytes " << cc->bytes()
                  << " holdout2_trie_cls_bpc " << bt / static_cast<double>(H - half) << " holdout2_trie_ig_cls_bpc "
                  << bi / static_cast<double>(H - half) << "\n";
    }
    // The distribution the next experts are mixed into: trie (+ cls) and trie + InfiniGram (+ cls), then (+ sim).
    std::vector<double> bt0 = pt, bi0 = pig;
    for (std::size_t k = 0; cc && k < H; ++k) {
        bt0[k] = lcl_t[clb[k]] * pcl[k] + (1 - lcl_t[clb[k]]) * pt[k];
        bi0[k] = lcl_i[clb[k]] * pcl[k] + (1 - lcl_i[clb[k]]) * pig[k];
    }
    std::vector<double>().swap(pcl);  // M13: not read again
    std::vector<int>().swap(clb);
    // A4: the similar-context expert's lambdas on the first half of the holdout over trie (+ cls) (lsm_t) and trie +
    // InfiniGram (+ cls) (lsm_i); its gain on the second half (LCI over 256 blocks). The copy expert, if on, is fit over it.
    double lsm_t[SimilarContext::kBuckets] = {0}, lsm_i[SimilarContext::kBuckets] = {0};
    if (sc) {
        std::vector<double> ps(H);
        std::vector<int> sb(H);
        for (std::size_t k = 0; k < H; ++k) {
            double P[256];
            sb[k] = SimilarContext::bucket(sc->predict(data.data(), cfg.train + k, P));
            ps[k] = P[data[cfg.train + k]];
        }
        fit_lambdas(ps, bt0, sb, half, SimilarContext::kBuckets, lsm_t);
        fit_lambdas(ps, bi0, sb, half, SimilarContext::kBuckets, lsm_i);
        const std::size_t nb = 256, bl = (H - half) / nb;
        std::vector<double> d_t(nb, 0.0), d_i(nb, 0.0);
        double b[4] = {0, 0, 0, 0};
        for (std::size_t k = 0; k < H; ++k) {
            const double xt = lsm_t[sb[k]] * ps[k] + (1 - lsm_t[sb[k]]) * bt0[k], xi = lsm_i[sb[k]] * ps[k] + (1 - lsm_i[sb[k]]) * bi0[k];
            if (k >= half) {
                b[0] -= std::log2(bt0[k]), b[1] -= std::log2(xt), b[2] -= std::log2(bi0[k]), b[3] -= std::log2(xi);
                if (k < half + nb * bl) {
                    d_t[(k - half) / bl] += (std::log2(xt) - std::log2(bt0[k])) / static_cast<double>(bl);
                    d_i[(k - half) / bl] += (std::log2(xi) - std::log2(bi0[k])) / static_cast<double>(bl);
                }
            }
            bt0[k] = xt;
            bi0[k] = xi;
        }
        const std::string cl = cc ? "_cls" : "";
        const double nh = static_cast<double>(H - half);
        std::cout << "sim " << o.sim << " keys " << sc->keys() << " table_bytes " << sc->bytes() << " build_s " << build_s
                  << " holdout2_trie" << cl << "_bpc " << b[0] / nh << " holdout2_trie" << cl << "_sim_bpc " << b[1] / nh
                  << " holdout2_trie_ig" << cl << "_bpc " << b[2] / nh << " holdout2_trie_ig" << cl << "_sim_bpc " << b[3] / nh << "\n";
        for (auto [name, d] : {std::pair<const char*, std::vector<double>*>{"trie", &d_t}, {"trie_ig", &d_i}}) {
            const auto g = cypha::eval::bayesian_bootstrap(*d, 0.005), g0 = cypha::eval::bayesian_bootstrap(*d, 0.0);
            std::cout << "sim_lci " << name << cl << " gain_bpc " << g.mean << " lo90 " << g.lo90 << " hi90 " << g.hi90
                      << " p_gain_gt_0.005 " << g.p_gt << " p_gain_gt_0 " << g0.p_gt << " blocks " << g.n_eff << "\n";
        }
    }
    if (copy_fit) {  // B4C: the default copy expert and --copy's models share these lambdas
        for (std::size_t k = 0; k < H; ++k) {
            int n = 0;
            pcp[k] = copy_pred(data.data(), cfg.train + k, n) == data[cfg.train + k] ? 1.0 : 0.0;
            cbk[k] = copy_bucket(n);
        }
        // With --cls the copy expert goes on after the class expert (B4, then copy, A1d): its lambdas are fit over
        // trie + cls and trie + InfiniGram + cls (bt, bi), and its gain is over those.
        const std::vector<double>&bt = bt0, &bi = bi0;  // A4: with --sim, trie (+ IG) (+ cls) + sim
        fit_lambdas(pcp, bt, cbk, half, kCopyBuckets, lc_t);
        fit_lambdas(pcp, bi, cbk, half, kCopyBuckets, lc_i);
        copy_clip(lc_t, o.copy_min);
        copy_clip(lc_i, o.copy_min);
        // 256 blocks of the second half: per block, bits per byte of b with copy mixed in at lam (lam 0: without it).
        const std::size_t nb = 256, bl = (H - half) / nb;
        auto blocks = [&](const double* lam, const std::vector<double>& b, std::vector<double>& out) {
            out.assign(nb, 0.0);
            double sum = 0;
            for (std::size_t k = half; k < half + nb * bl; ++k) {
                const double x = -std::log2(lam[cbk[k]] * pcp[k] + (1 - lam[cbk[k]]) * b[k]);
                out[(k - half) / bl] += x / static_cast<double>(bl);
                sum += x;
            }
            return sum / static_cast<double>(nb * bl);
        };
        auto lci = [](const std::string& name, const std::vector<double>& x, const std::vector<double>& y) {
            std::vector<double> d(x.size());
            for (std::size_t b = 0; b < d.size(); ++b) d[b] = x[b] - y[b];  // bits per byte saved
            const auto g = cypha::eval::bayesian_bootstrap(d, 0.005), g0 = cypha::eval::bayesian_bootstrap(d, 0.0);
            std::cout << name << " gain_bpc " << g.mean << " lo90 " << g.lo90 << " hi90 " << g.hi90 << " p_gain_gt_0.005 "
                      << g.p_gt << " p_gain_gt_0 " << g0.p_gt << " blocks " << g.n_eff << "\n";
        };
        const double none[kCopyBuckets] = {0};
        std::vector<double> xt, xi, yt, yi;
        const double ct = blocks(lc_t, bt, xt), ci = blocks(lc_i, bi, xi);
        blocks(none, bt, yt);
        blocks(none, bi, yi);
        const std::string cl = std::string(cc ? "_cls" : "") + (sc ? "_sim" : "");
        std::cout << "holdout2_trie" << cl << "_copy_bpc " << ct << " holdout2_trie_ig" << cl << "_copy_bpc " << ci
                  << " (first " << nb * bl << " bytes of the second half)\n";
        lci("copy_lci trie" + cl, yt, xt);
        lci("copy_lci trie_ig" + cl, yi, xi);
        if (cc) {  // what the class expert adds with copy present: copy alone (lambdas over pt, pig) against cls + copy
            double l0t[kCopyBuckets], l0i[kCopyBuckets];
            fit_lambdas(pcp, pt, cbk, half, kCopyBuckets, l0t);
            fit_lambdas(pcp, pig, cbk, half, kCopyBuckets, l0i);
            copy_clip(l0t, o.copy_min);
            copy_clip(l0i, o.copy_min);
            std::vector<double> zt, zi;
            const double c0t = blocks(l0t, pt, zt), c0i = blocks(l0i, pig, zi);
            std::cout << "holdout2_trie_copy_bpc " << c0t << " holdout2_trie_ig_copy_bpc " << c0i << "\n";
            lci("cls_given_copy_lci trie", zt, xt);
            lci("cls_given_copy_lci trie_ig", zi, xi);
        }
        std::cout << "copy_lambdas trie";
        for (double l : lc_t) std::cout << " " << l;
        std::cout << " | trie_ig";
        for (double l : lc_i) std::cout << " " << l;
        std::cout << "\n";
    }
    // GEN-C2 (docs/reports/GEN_C_TOPIC_CACHE.md, "GEN-C2"): the document experts' cache (--doc SPEC to read another
    // setting) over trie + InfiniGram (+ cls) (+ sim) (+ copy), last: its lambdas on the first half, its line and LCI on
    // the second half's 256 blocks (as copy's). It reads only these lines; no answer mixer is given it. The per-byte
    // arrays above are not read again: they go first, then the cache is built over wiki train's words in their memory.
    // --no-doc leaves it out.
    if (!a.no_doc) {
        std::vector<double>().swap(pt);
        std::vector<double>().swap(pig);
        std::vector<double>().swap(bt0);
        if (copy_fit) {
            for (std::size_t k = 0; k < H; ++k) bi0[k] = lc_i[cbk[k]] * pcp[k] + (1 - lc_i[cbk[k]]) * bi0[k];
            std::vector<double>().swap(pcp);
            std::vector<int>().swap(cbk);
        }
        const std::size_t nb = 256, bl = (H - half) / nb;
        auto blocks = [&](std::vector<double>& out) {
            out.assign(nb, 0.0);
            double sum = 0;
            for (std::size_t k = half; k < half + nb * bl; ++k) {
                const double x = -std::log2(bi0[k]);
                out[(k - half) / bl] += x / static_cast<double>(bl);
                sum += x;
            }
            return sum / static_cast<double>(nb * bl);
        };
        std::vector<double> before, after;
        const double b0 = blocks(before);
        {
            int drc = 0;
            const auto dt = make_doc(a.doc.empty() ? std::string("cache") : a.doc, data.data(), cfg.train, drc);
            if (!dt) return drc ? drc : 1;
            mix_doc_into(*dt, data.data(), cfg.train, half, bi0);
        }
        const double b1 = blocks(after);
        const std::string cl = std::string(cc ? "_cls" : "") + (sc ? "_sim" : "") + (copy_fit ? "_copy" : "");
        std::cout << "holdout2_trie_ig" << cl << "_bpc " << b0 << " holdout2_trie_ig" << cl << "_doc_bpc " << b1 << " (first "
                  << nb * bl << " bytes of the second half)\n";
        std::vector<double> d(nb);
        for (std::size_t b = 0; b < nb; ++b) d[b] = before[b] - after[b];  // bits per byte saved
        const auto g = cypha::eval::bayesian_bootstrap(d, 0.005), g0 = cypha::eval::bayesian_bootstrap(d, 0.0);
        std::cout << "doc_lci trie_ig" << cl << " gain_bpc " << g.mean << " lo90 " << g.lo90 << " hi90 " << g.hi90
                  << " p_gain_gt_0.005 " << g.p_gt << " p_gain_gt_0 " << g0.p_gt << " blocks " << g.n_eff << "\n";
    }
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";

    // ANS (--copy-dk): the copy expert's lambdas inside a don't-know answer, fit on the don't-know turns held out of
    // training (rows n_dk.. of --dk, at most 1,000), over the distributions answers use: the trie, and trie +
    // InfiniGram (class, similar-context and the default copy are off inside answers). Report: per match-length bucket
    // the bytes and copy's hit rate; the held-out answers' bits per byte without and with it (in-sample fit).
    double ldk_t[kCopyBuckets] = {0}, ldk_i[kCopyBuckets] = {0};
    if (o.copy_dk) {
        Lm base[2] = {{nullptr, w, all_followers, false, nullptr, nullptr, nullptr, 8192, nullptr, nullptr, nullptr,
                       nullptr, &pm},
                      {nullptr, w, all_followers, false, &ig, kn_off_ans ? lam0 : lam, nullptr, 8192, nullptr, nullptr, nullptr,
                       nullptr, &pm}};  // BKN3: inside answers InfiniGram's lambdas are lam0 when Kneser-Ney is off there
        std::vector<double> pe, pb[2];
        std::vector<int> bk;
        std::size_t used = 0;
        for (std::size_t j = n_dk; j < dk.size() && used < 1000; ++j, ++used) {
            const std::string prompt = pre_dk[j] + with_tok("User: " + dk[j].q + "\nAssistant: ", tok_dk[j]);  // DK2: pre
            const std::string& resp = dk[j].response;
            for (int m = 0; m < 2; ++m) {
                Lm lm = base[m];
                lm.answer_at = prompt.size();
                std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
                int hint = -1;
                for (char ch : resp) {
                    double P[256];
                    lm.dist(seq, hint, P);
                    const auto y = static_cast<std::uint8_t>(ch);
                    pb[m].push_back(P[y]);
                    if (m == 0) {
                        int n = 0;
                        std::size_t cj = 0;
                        int c = copy_pred(seq.data(), seq.size(), n, &cj);
                        if (c >= 0 && o.copy_frame && copy_closes_unopened(seq.data(), seq.size(), cj, prompt.size()))
                            c = -1, n = 0;  // DK2: as decode will see it
                        pe.push_back(c == y ? 1.0 : 0.0);
                        bk.push_back(copy_bucket(n));
                    }
                    seq.push_back(y);
                }
            }
        }
        fit_lambdas(pe, pb[0], bk, pe.size(), kCopyBuckets, ldk_t);
        fit_lambdas(pe, pb[1], bk, pe.size(), kCopyBuckets, ldk_i);
        std::cout << "copy_dk_fit heldout_turns " << used << " bytes " << pe.size();
        for (int b = 0; b < kCopyBuckets; ++b) {
            double nb = 0, hit = 0;
            for (std::size_t k = 0; k < pe.size(); ++k)
                if (bk[k] == b) ++nb, hit += pe[k];
            std::cout << " | b" << b << " n " << nb << " hit " << (nb ? hit / nb : 0.0) << " lam_trie " << ldk_t[b]
                      << " lam_trie_ig " << ldk_i[b];
        }
        for (int m = 0; m < 2; ++m) {
            const double* L = m ? ldk_i : ldk_t;
            double without = 0, with = 0;
            for (std::size_t k = 0; k < pe.size(); ++k) {
                without -= std::log2(std::max(pb[m][k], 1e-300));
                with -= std::log2(std::max(L[bk[k]] * pe[k] + (1 - L[bk[k]]) * pb[m][k], 1e-300));
            }
            std::cout << (m ? " | trie_ig" : " | trie") << " bpb_without " << without / static_cast<double>(pe.size())
                      << " bpb_with " << with / static_cast<double>(pe.size());
        }
        std::cout << "\n";
    }

    const int nlm = o.copy ? 4 : 2;
    const char* const lm_name[4] = {"trie", "trie_ig", "trie_cp", "trie_ig_cp"};
    const std::size_t lim = all_followers;
    Lm lms[4] = {{nullptr, w, lim, false, nullptr, nullptr, nullptr, 8192, cc.get(), lcl_t, sc.get(), lsm_t, &pm},
                 {nullptr, w, lim, false, &ig, lam, nullptr, 8192, cc.get(), lcl_i, sc.get(), lsm_i, &pm},
                 {nullptr, w, lim, false, nullptr, nullptr, lc_t, 8192, cc.get(), lcl_t, sc.get(), lsm_t, &pm},
                 {nullptr, w, lim, false, &ig, lam, lc_i, 8192, cc.get(), lcl_i, sc.get(), lsm_i, &pm}};
    for (auto& l : lms) l.sim_gate = sc && parse_sim(o.sim).gate;  // A4: the expert off inside answers
    for (auto& l : lms) l.kn = kn.get(), l.lkn = lkn.empty() ? nullptr : lkn.data(), l.kn_replace = kn_rep;  // BKN
    for (auto& l : lms) l.kn_gate = kn_off_ans, l.lam_ans = kn_off_ans ? lam0 : nullptr;  // BKN3
    for (int l = 0; l < 4; ++l) lms[l].cls_gate = true, lms[l].lc_out = o.copy_out ? (l % 2 ? lc_i : lc_t) : nullptr;  // B4C
    if (o.copy_dk)
        for (int l = 0; l < 4; ++l) lms[l].lc_dk = l % 2 ? ldk_i : ldk_t, lms[l].copy_frame = o.copy_frame;  // ANS; DK2

    // Grading only: is a gold answer in the source article's text (wiki train)? Splits A3's results.
    const std::string_view wiki(reinterpret_cast<const char*>(data.data()), cfg.train);
    auto in_prose = [&](const QaRow& r) {
        const std::size_t p = r.in_slice ? wiki.find("<title>" + r.title + "</title>") : std::string::npos;
        if (p == std::string::npos) return r.in_slice ? 0 : -1;
        const std::string page = norm_text(std::string(wiki.substr(p, wiki.find("</page>", p) - p)));
        for (const auto& g : r.gold)
            if (norm_text(g).size() > 3 && page.find(norm_text(g)) != std::string::npos) return 1;
        return 0;
    };
    // A2 coverage (analysis only): a shard row with the question's E1 entity (2: and its relation).
    std::set<std::string> cov_e, cov_er;
    auto rel_key = [](const std::string& r) {
        return r == "birth_year" ? std::string("birth_date") : r == "death_year" ? std::string("death_date") : r;
    };
    for (const auto* v : {&sh1, &sh2})
        for (const auto& r : *v) {
            cov_e.insert(lowered(strip_article(r.entity)));
            cov_er.insert(lowered(strip_article(r.entity)) + "|" + rel_key(r.rel));
        }
    auto covered = [&](const QaRow& r) {
        const std::string e = entity_of(r.q), k = lowered(strip_article(e));
        return cov_e.empty() || e.empty() ? -1 : cov_er.count(k + "|" + question_relation(r.q, e)) ? 2 : cov_e.count(k) ? 1 : 0;
    };

    auto run_set = [&](const std::vector<QaRow>& rows, const std::vector<std::string>& toks,
                       const std::vector<std::string>& pres) {  // DK2: pres, the start frames (empty unless start)
        std::vector<Graded> out;
        for (std::size_t r = 0; r < rows.size(); ++r)
            for (int form = 0; form < 2; ++form)
                for (int l = 0; l < nlm; ++l) {
                    const Lm& lm = lms[l];
                    Graded g;
                    g.row = static_cast<int>(r);
                    g.in_slice = rows[r].in_slice;
                    g.form = form;
                    g.lm = l;
                    g.fam_tok = fam_on ? toks[r] : "";
                    const std::string prompt =
                        form == 0 ? pres[r] + with_tok("User: " + rows[r].q + "\nAssistant: ", g.fam_tok) : rows[r].q + " ";
                    const auto t0 = std::chrono::steady_clock::now();
                    g.ans = decode(lm, prompt);
                    g.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    g.qlp = text_logprob(lm, form == 0 ? "User: " : "", rows[r].q);
                    const double fv[kFeat] = {g.ans.mean_lp, g.ans.min_lp, -g.ans.ent0, static_cast<double>(g.ans.ig_n0), g.qlp};
                    std::copy(fv, fv + kFeat, g.fv);
                    g.prose = in_prose(rows[r]);
                    g.covered = covered(rows[r]);
                    if (is_dont_know(g.ans.text) || g.ans.text.empty()) g.grade = 1;
                    else g.grade = (rows[r].in_slice && gold_match(g.ans.text, rows[r].gold)) ? 0 : 2;
                    out.push_back(g);
                }
        return out;
    };

    // A3 (R3): CyphaLM's own candidate objects (its greedy and 5 seeded chat answers; its greedy and 2 seeded
    // continuations of each frame's prefix), each scored by the bits per byte of its filled statements after the
    // chat prompt (the best frame counts); the lowest wins. R3 runs only where the question's relation has a frame.
    const auto frames = o.r3.empty() ? std::map<std::string, std::vector<std::string>>{} : load_frames(o.r3);
    if (!o.r3.empty() && frames.empty()) {
        std::cerr << "no usable frames in " << o.r3 << "\n";
        return 1;
    }
    for (const auto& [rel, fs] : frames)
        for (const auto& f : fs) std::cout << "R3_FRAME " << rel << " | " << printable(fill_frame(f, "[S]", "[O]")) << "\n";
    const Lm& r3lm = lms[nlm - 1];
    auto r3_set = [&](const std::vector<QaRow>& rows, std::map<std::string, int>& why) {
        std::vector<Graded> out;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const QaRow& row = rows[r];
            Graded g;
            g.row = static_cast<int>(r);
            g.in_slice = row.in_slice;
            g.lm = nlm - 1;
            g.grade = 1;
            g.prose = in_prose(row);
            g.covered = covered(row);
            const std::string e = entity_of(row.q), rel = question_relation(row.q, e);
            std::vector<std::string> fr;
            for (const auto& r2 : f2_relations(rel))
                if (frames.count(r2)) fr.insert(fr.end(), frames.at(r2).begin(), frames.at(r2).end());
            ++why[rel.empty() ? "no_relation" : fr.empty() ? "no_frame:" + rel : "ran:" + rel];
            if (fr.empty()) {
                out.push_back(g);
                continue;
            }
            const bool date = rel == "birth_date" || rel == "death_date";
            const std::string subj = strip_article(e), prompt = "User: " + row.q + "\nAssistant: ";
            std::vector<std::vector<std::pair<std::string, std::string>>> src;
            for (std::uint32_t s = 0; s < 6; ++s) src.push_back(r3_candidates(decode(r3lm, prompt, 80, s).text, date, false));
            for (const auto& f : fr) {
                const std::string pre = fill_frame(f.substr(0, f.find('\x02')), subj, "");
                if (f.find('\x02') == 0) continue;  // the object comes first: nothing to continue
                const std::string close = f.compare(f.find('\x02') + 1, 2, "]]") == 0 ? "]]" : "";
                for (std::uint32_t s = 0; s < 3; ++s)
                    src.push_back(r3_candidates(decode(r3lm, prompt + pre, 40, s).text, date, true, close));
            }
            std::vector<std::pair<std::string, std::string>> cand;  // (surface, value): the first of each value
            std::set<std::string> seen;
            const std::string nq = norm_text(row.q);
            for (const auto& v : src)
                for (const auto& c : v) {
                    const std::string n = norm_text(c.second);
                    if (n.size() > 2 && nq.find(n) == std::string::npos && seen.insert(n).second) cand.push_back(c);
                }
            if (cand.empty()) {
                ++why["no_candidate"];
                out.push_back(g);
                continue;
            }
            std::vector<double> bpb;
            std::vector<R3Score> sc;
            for (const auto& c : cand) {
                R3Score best;
                for (const auto& f : fr) {
                    const std::size_t p = f.find('\x02');
                    const std::string pre = fill_frame(f.substr(0, p), subj, "");
                    const R3Score s = score_statement(r3lm, prompt, pre + c.first + fill_frame(f.substr(p + 1), subj, ""),
                                                      pre.size(), pre.size() + c.first.size());
                    if (s.bpb < best.bpb) best = s;
                }
                bpb.push_back(best.bpb);
                sc.push_back(best);
            }
            double margin = 0;
            const std::size_t win = static_cast<std::size_t>(r3_choose(bpb, margin));
            int agree = 0;
            for (const auto& v : src) {
                bool has = false;
                for (const auto& c : v) has = has || norm_text(c.second) == norm_text(cand[win].second);
                agree += has;
            }
            g.ans.text = cand[win].second;
            const double fv[kFeat] = {margin, static_cast<double>(agree) / static_cast<double>(src.size()), -bpb[win],
                                      sc[win].min_lp, static_cast<double>(sc[win].ig_n)};
            std::copy(fv, fv + kFeat, g.fv);
            g.grade = row.in_slice && gold_match(cand[win].second, row.gold) ? 0 : 2;
            out.push_back(g);
        }
        return out;
    };

    const std::vector<Graded> gd = run_set(dev, tok_dev, pre_dev);
    std::map<std::string, int> why_dev;
    const std::vector<Graded> gr3 = o.r3.empty() ? std::vector<Graded>{} : r3_set(dev, why_dev);
    const char* const form_name[2] = {"chat", "wiki"};

    // The A0 base run's grades, for A1's "in-slice questions it gets wrong" (a copy model uses its base model's).
    std::map<std::tuple<int, int, int>, int> base;
    if (!o.base.empty()) {
        std::ifstream in(o.base);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            const auto f = split(line, '\t');
            if (f.size() >= 6 && f[0] == "dev" && (f[4] == "trie" || f[4] == "trie_ig"))
                base[{std::atoi(f[1].c_str()), f[3] == "wiki", f[4] == "trie_ig"}] = std::atoi(f[5].c_str());
        }
    }
    // A3 rows carry lm "r3" (cypha_gate reads these rows and splits them by set, then by --where).
    auto write_rows = [&](const char* set, const std::vector<Graded>& gs, bool append, bool r3 = false) {
        if (o.out.empty()) return;
        std::ofstream f(o.out, append ? std::ios::app : std::ios::trunc);
        if (!append)
            f << "set\trow\tin_slice\tform\tlm\tgrade\tmean_lp\tmin_lp\tentropy0\tig_n0\tquestion_lp\tfam\tanswer\n";
        for (const auto& g : gs)
            f << set << "\t" << g.row << "\t" << g.in_slice << "\t" << form_name[g.form] << "\t" << (r3 ? "r3" : lm_name[g.lm]) << "\t"
              << g.grade << "\t" << g.ans.mean_lp << "\t" << g.ans.min_lp << "\t" << g.ans.ent0 << "\t" << g.ans.ig_n0
              << "\t" << g.qlp << "\t" << g.fam_tok << "\t" << printable(g.ans.text) << "\n";
    };
    write_rows("dev", gd, false);
    write_rows("dev", gr3, true, true);

    std::cout.precision(4);
    // RAW (everything it wrote counts; only its own message abstains), CV (the 2%-wrong gate chosen on four folds,
    // fold = row % 5, applied to the fifth; split by prose and shard coverage when known), the full-dev gate and the
    // risk-coverage curves.
    auto report = [&](const std::string& tag, const std::vector<const Graded*>& v, bool have_ig, const char* const* names) {
        Tally raw, raw_in, raw_out;
        for (auto* g : v) {
            raw.add(g->grade);
            (g->in_slice ? raw_in : raw_out).add(g->grade);
        }
        print_tally("RAW " + tag + " all", raw);
        print_tally("RAW " + tag + " in_slice", raw_in);
        print_tally("RAW " + tag + " not_in_slice", raw_out);
        Tally cvt, cv_in, cv_out, pr[2], cov[2];
        std::map<int, int> feat_pick;
        for (int fold = 0; fold < 5; ++fold) {
            std::vector<const Graded*> tr;
            for (auto* g : v)
                if (g->row % 5 != fold) tr.push_back(g);
            const Gate gate = choose_gate(tr, have_ig);
            ++feat_pick[gate.f];
            for (auto* g : v)
                if (g->row % 5 == fold) {
                    const int gr = gated_answer(*g, gate) ? g->grade : 1;
                    cvt.add(gr);
                    (g->in_slice ? cv_in : cv_out).add(gr);
                    if (g->prose >= 0) pr[g->prose].add(gr);
                    if (g->covered >= 0) cov[g->covered == 2].add(gr);
                }
        }
        print_tally("CV " + tag + " all", cvt);
        print_tally("CV " + tag + " in_slice", cv_in);
        print_tally("CV " + tag + " not_in_slice", cv_out);
        for (int b = 1; b >= 0; --b) {
            if (pr[b].n) print_tally("CV " + tag + (b ? " in_slice_fact_in_prose" : " in_slice_fact_not_in_prose"), pr[b]);
            if (cov[b].n) print_tally("CV " + tag + (b ? " shard_covered" : " shard_not_covered"), cov[b]);
        }
        std::cout << "CV " << tag << " gate_features";
        for (auto& [f, c] : feat_pick) std::cout << " " << (f < 0 ? "none" : names[f]) << "x" << c;
        std::cout << "\n";
        int rr = 0;
        const Gate gate = choose_gate(v, have_ig, &rr);
        std::cout << "FULLDEV_GATE " << tag << " feature " << (gate.f < 0 ? "none" : names[gate.f]) << " tau " << gate.tau
                  << " right " << rr << "\n";
        for (int f = 0; f < kFeat; ++f) {
            if (f == 3 && !have_ig) continue;
            std::vector<std::pair<double, int>> s;
            for (auto* g : v)
                if (g->grade != 1) s.push_back({feat(*g, f), g->grade});
            std::sort(s.begin(), s.end(), [](auto& x, auto& y) { return x.first > y.first; });
            std::cout << "RC " << tag << " " << names[f] << " (answered:right:wrong)";
            int r = 0, wr = 0;
            for (std::size_t k = 0; k < s.size(); ++k) {
                (s[k].second == 0 ? r : wr)++;
                const std::size_t m = k + 1;
                if (m == 1 || m == 2 || m == 3 || m == 5 || m % 10 == 0 || m == s.size())
                    std::cout << " " << m << ":" << r << ":" << wr;
            }
            std::cout << "\n";
        }
        return gate;
    };
    // A1: abstention by CyphaLM's own message, its precision and recall on not-in-slice questions (and, with an A0 base
    // run, on in-slice questions that run got wrong), its tags, reason codes, and whether the tag (A1d: and the message)
    // names the question's entity: a span of the question, the chosen E1 entity, the in-slice source title.
    auto abstain_line = [&](const std::vector<QaRow>& rows, const std::vector<const Graded*>& v,
                            const std::map<std::tuple<int, int, int>, int>* bs, int form, int l) {
        int abst = 0, abst_out = 0, n_out = 0, abst_inw = 0, n_inw = 0, tags = 0, code_ok = 0, ent_ok = 0, ent_eq = 0,
            complete = 0, ent_title = 0, msg_ok = 0;
        for (auto* g : v) {
            const bool ab = g->grade == 1 && !g->ans.text.empty();
            const QaRow& r = rows[static_cast<std::size_t>(g->row)];
            abst += ab;
            if (!g->in_slice) {
                ++n_out;
                abst_out += ab;
            } else {
                auto it = bs ? bs->find({g->row, form, l % 2}) : std::map<std::tuple<int, int, int>, int>::const_iterator{};
                const bool base_wrong = !bs || it == bs->end() || it->second != 0;
                if (base_wrong) {
                    ++n_inw;
                    abst_inw += ab;
                }
            }
            msg_ok += ab && names_in(msg_entity(g->ans.text), r.q);
            const std::string code = tag_field(g->ans.text, "reason");
            if (code.empty()) continue;
            ++tags;
            // Expected: UNKNOWN_ENTITY for a subject with no article in the slice (not-in-slice),
            // NO_EVIDENCE or LOW_CERTAINTY when the article is there (in-slice).
            code_ok += g->in_slice ? (code == "NO_EVIDENCE" || code == "LOW_CERTAINTY") : code == "UNKNOWN_ENTITY";
            const std::string ent = tag_field(g->ans.text, "entity");
            ent_ok += !ent.empty() && lowered(r.q).find(lowered(ent)) != std::string::npos;
            ent_eq += !ent.empty() && lowered(strip_article(ent)) == lowered(strip_article(entity_of(r.q)));
            ent_title += g->in_slice && !ent.empty() && norm_text(strip_article(ent)) == norm_text(strip_article(r.title));
            complete += g->ans.text.find("gap=G-") != std::string::npos &&
                        g->ans.text.compare(g->ans.text.size() - 2, 2, "|>") == 0;
        }
        std::ostringstream b;
        b.setf(std::ios::fixed);
        b.precision(4);
        const double base_rate = static_cast<double>(n_out) / static_cast<double>(v.size());
        b << " abstained " << abst << " not_in_slice " << abst_out << "/" << n_out << " precision_nis "
          << (abst ? static_cast<double>(abst_out) / abst : 0.0) << " recall_nis "
          << (n_out ? static_cast<double>(abst_out) / n_out : 0.0) << " base_rate_nis " << base_rate
          << " | in_slice_base_wrong " << abst_inw << "/" << n_inw << " precision_should "
          << (abst ? static_cast<double>(abst_out + abst_inw) / abst : 0.0) << " recall_should "
          << (n_out + n_inw ? static_cast<double>(abst_out + abst_inw) / (n_out + n_inw) : 0.0) << " base_rate_should "
          << static_cast<double>(n_out + n_inw) / static_cast<double>(v.size()) << " | tags " << tags << " complete "
          << complete << " reason_ok " << code_ok << " entity_in_question " << ent_ok << " entity_eq_parse " << ent_eq
          << " entity_eq_title " << ent_title << " message_entity_in_question " << msg_ok;
        return b.str();
    };
    // The test's gates: chosen once on all of dev.
    std::vector<Gate> final_gate(static_cast<std::size_t>(2 * nlm));
    for (int form = 0; form < 2; ++form)
        for (int l = 0; l < nlm; ++l) {
            std::vector<const Graded*> v;
            for (const auto& g : gd)
                if (g.form == form && g.lm == l) v.push_back(&g);
            const std::string tag = std::string(form_name[form]) + "/" + lm_name[l];
            final_gate[static_cast<std::size_t>(form * nlm + l)] = report(tag, v, l % 2 == 1, kFeatName);
            std::cout << "ABSTAIN " << tag << abstain_line(dev, v, &base, form, l) << "\n";
            std::vector<double> ms;  // A4: decode time a question (PLAN 10: p50 <= 200 ms)
            for (auto* g : v) ms.push_back(g->ms);
            std::sort(ms.begin(), ms.end());
            std::cout << "LATENCY " << tag << " decode_ms p50 " << ms[ms.size() / 2] << " p99 "
                      << ms[std::min(ms.size() - 1, ms.size() * 99 / 100)] << " max " << ms.back() << "\n";
        }
    if (sc) {  // A4: the expert's search latency, and the three nearest stored contexts at each dev question's chat prompt
        std::cout << "SIM_LAT queries " << sc->queries() << " p50_us " << sc->latency_us(0.5) << " p99_us " << sc->latency_us(0.99) << "\n";
        const std::size_t tn = sc_text == data.data() ? cfg.train : T.size();
        for (std::size_t r = 0; r < dev.size(); ++r) {
            const std::string pr = "User: " + dev[r].q + "\nAssistant: ";
            const auto nb = sc->neighbours(reinterpret_cast<const std::uint8_t*>(pr.data()), pr.size());
            for (std::size_t j = 0; j < std::min<std::size_t>(3, nb.size()); ++j) {
                const std::size_t p = nb[j].first, from = p > 60 ? p - 60 : 0;
                std::cout << "SIM_NB " << r << " " << dev[r].q << " | " << nb[j].second << " | "
                          << printable(std::string(reinterpret_cast<const char*>(sc_text) + from, p - from)) << " >> "
                          << printable(std::string(reinterpret_cast<const char*>(sc_text) + p, std::min<std::size_t>(40, tn - p))) << "\n";
            }
        }
    }
    Gate r3_gate;
    if (!o.r3.empty()) {
        for (const auto& [k, v] : why_dev) std::cout << "R3_RUNS " << k << " " << v << "\n";
        std::vector<const Graded*> v;
        for (const auto& g : gr3) v.push_back(&g);
        r3_gate = report("R3", v, true, kR3Name);
        for (const auto& g : gr3)
            if (!g.ans.text.empty())
                std::cout << "R3_EX grade " << g.grade << " in_slice " << g.in_slice << " prose " << g.prose << " | "
                          << dev[static_cast<std::size_t>(g.row)].q << " => " << printable(g.ans.text) << " | margin "
                          << g.fv[0] << " agree " << g.fv[1] << " neg_bpb " << g.fv[2] << " min_lp " << g.fv[3]
                          << " ig " << g.fv[4] << "\n";
    }
    if (!cov_e.empty()) {
        int c[3] = {0, 0, 0}, unknown = 0;
        for (const auto& r : dev) {
            const int k = covered(r);
            if (k < 0) ++unknown;
            else ++c[k];
            if (k == 2) std::cout << "SHARD_COVERED " << r.q << "\n";
        }
        std::cout << "SHARD dev_covered_entity_relation " << c[2] << " entity_only " << c[1] << " neither " << c[0]
                  << " no_entity " << unknown << "\n";
    }
    {
        int p[2] = {0, 0};
        for (const auto& r : dev)
            if (r.in_slice) ++p[in_prose(r)];
        std::cout << "PROSE dev_in_slice fact_in_prose " << p[1] << " not " << p[0] << "\n";
    }
    // Examples: the first 12 dev rows, chat form.
    for (const auto& g : gd)
        if (g.form == 0 && g.row < 12)
            std::cout << "EX " << lm_name[g.lm] << " grade " << g.grade << " | " << dev[static_cast<std::size_t>(g.row)].q
                      << " " << g.fam_tok << " => " << printable(g.ans.text) << "\n";
    for (const auto& g : gd)
        if (g.grade == 0)
            std::cout << "RIGHT " << form_name[g.form] << "/" << lm_name[g.lm] << " | "
                      << dev[static_cast<std::size_t>(g.row)].q << " => " << printable(g.ans.text) << "\n";

    if (!a.cards.empty()) {  // S1: cards after dev's answers and gates; nothing below reads them
        cards.ig = &ig;
        cards.low = low.get();
        cards.entity_of = entity_of;
        cards.prompt = [&](int form, const QaRow& r, const std::string& tok) {
            return form == 0 ? with_tok("User: " + r.q + "\nAssistant: ", tok) : r.q + " ";
        };
        cards.lm_name = lm_name;
        cards.nlm = nlm;
        cards.path = a.cards;
        if (low) card_sources_low(cards, a, data, cfg.train);
        write_cards(cards, "dev", dev, gd, tok_dev, nullptr);
    }
    // The frozen test, once: gates chosen on all of dev, applied unchanged.
    if (!test.empty()) {
        const std::vector<Graded> gt = run_set(test, tok_test, pre_test);
        std::map<std::string, int> why_test;
        const std::vector<Graded> gt3 = o.r3.empty() ? std::vector<Graded>{} : r3_set(test, why_test);
        write_rows("test", gt, true);
        write_rows("test", gt3, true, true);
        auto test_report = [&](const std::string& tag, const std::vector<const Graded*>& v, const Gate& gate, int form = -1,
                               int l = 0) {
            Tally raw, gated, gin, gout, pr[2], cov[2];
            for (auto* g : v) {
                raw.add(g->grade);
                const int gr = gated_answer(*g, gate) ? g->grade : 1;
                gated.add(gr);
                (g->in_slice ? gin : gout).add(gr);
                if (g->prose >= 0) pr[g->prose].add(gr);
                if (g->covered >= 0) cov[g->covered == 2].add(gr);
            }
            print_tally("TEST_RAW " + tag + " all", raw);
            print_tally("TEST_GATED " + tag + " all", gated);
            print_tally("TEST_GATED " + tag + " in_slice", gin);
            print_tally("TEST_GATED " + tag + " not_in_slice", gout);
            for (int b = 1; b >= 0; --b) {
                if (pr[b].n) print_tally("TEST_GATED " + tag + (b ? " in_slice_fact_in_prose" : " in_slice_fact_not_in_prose"), pr[b]);
                if (cov[b].n) print_tally("TEST_GATED " + tag + (b ? " shard_covered" : " shard_not_covered"), cov[b]);
            }
            if (form >= 0) std::cout << "TEST_ABSTAIN " << tag << abstain_line(test, v, nullptr, form, l) << "\n";
        };
        for (int form = 0; form < 2; ++form)
            for (int l = 0; l < nlm; ++l) {
                std::vector<const Graded*> v;
                for (const auto& g : gt)
                    if (g.form == form && g.lm == l) v.push_back(&g);
                test_report(std::string(form_name[form]) + "/" + lm_name[l], v, final_gate[static_cast<std::size_t>(form * nlm + l)],
                            form, l);
            }
        if (!o.r3.empty()) {
            for (const auto& [k, v] : why_test) std::cout << "TEST_R3_RUNS " << k << " " << v << "\n";
            std::vector<const Graded*> v;
            for (const auto& g : gt3) v.push_back(&g);
            test_report("R3", v, r3_gate);
            for (const auto& g : gt3)
                if (g.grade == 0 || gated_answer(g, r3_gate))
                    std::cout << "TEST_R3 grade " << g.grade << " gated " << gated_answer(g, r3_gate) << " | "
                              << test[static_cast<std::size_t>(g.row)].q << " => " << printable(g.ans.text) << "\n";
        }
        for (const auto& g : gt)
            if (g.grade == 0)
                std::cout << "TEST_RIGHT " << form_name[g.form] << "/" << lm_name[g.lm] << " | "
                          << test[static_cast<std::size_t>(g.row)].q << " => " << printable(g.ans.text) << "\n";
        if (!a.cards.empty()) write_cards(cards, "test", test, gt, tok_test, &final_gate);  // S1
    }
    return 0;
}

// ---------------------------------------------------------------- selftest

// 40,000 bytes of words picked by a fixed LCG: the same bytes on every platform.
std::vector<std::uint8_t> tiny_corpus(std::size_t n) {
    static const char* const words[] = {"the",  "of",     "and",   "trie",  "byte",   "context", "model", "count",
                                        "back", "off",    "match", "order", "four",   "wiki",    "[[",    "]]",
                                        "is",   "a",      "node",  "mix",   "weight", "frozen",  "hold",  "out"};
    std::uint32_t s = 12345u;
    std::string t;
    while (t.size() < n) {
        s = s * 1664525u + 1013904223u;
        t += words[(s >> 16) % (sizeof words / sizeof *words)];
        t += ((s >> 8) & 7u) == 0 ? ".\n" : " ";
    }
    t.resize(n);
    return std::vector<std::uint8_t>(t.begin(), t.end());
}

// Golden values on the built-in corpus (train 36,000, holdout 4,000), recorded 2026-10-27.
constexpr double kGoldenBackoff = 1.222293339;
constexpr double kGoldenMix = 1.130916116;
constexpr double kGoldenTol = 1e-6;

// ---------------------------------------------------------------- bkn (BKN: packed byte Kneser-Ney)
//
// docs/reports/BYTE_KN.md (HANDOFF 5.22). cls's protocol: the bpc model (train bytes only), InfiniGram over the train
// bytes; then, for each SPEC, byte_kn.hpp's packed Kneser-Ney built on the train bytes, read two ways:
//   (a) replace: Kneser-Ney in place of the trie mixture, InfiniGram mixed over it by lambda per (match length x count);
//   (b) add: Kneser-Ney mixed over the trie mixture by lambda per (deepest order found x count band), then InfiniGram.
// holdout2: lambdas on the holdout's first half, bits per byte on its second half (every choice is made here);
// --bkn-fresh: also the fresh text [sample, sample + 1 MiB), lambdas refit on the whole holdout (the read that decides);
// --bkn-stack: also with the defaults on top (class shape, then similar-context default, then copy, as qa mixes them).
// SPEC is K[:m:c:d]: order K; from order m on, contexts seen fewer than c times dropped and, with d = 1, children
// seen once dropped (their value into gamma). Intervals: 4 KiB block bootstraps of the paired gain.
std::vector<int> bkn_spec(const std::string& s) {  // {K, m, c, d}; empty on a bad spec
    std::vector<int> v;
    std::stringstream ss(s);
    std::string f;
    while (std::getline(ss, f, ':')) {
        if (f.empty() || f.find_first_not_of("0123456789") != std::string::npos) return {};
        v.push_back(std::atoi(f.c_str()));
    }
    if (v.size() == 1) v = {v[0], 1 << 30, 1, 0};
    if (v.size() != 4 || v[0] < 1 || v[0] > 32) return {};
    return v;
}

int cmd_bkn(const Args& a) {
    const Config& cfg = a.cfg;
    if (a.pos.size() < 2) {
        std::fprintf(stderr, "usage: cyphalm_trie bkn <wiki> K[:m:c:d] [more specs] [--bkn-fresh] [--bkn-stack]\n");
        return 2;
    }
    for (std::size_t s = 1; s < a.pos.size(); ++s)
        if (bkn_spec(a.pos[s]).empty()) {
            std::cerr << "bad bkn spec " << a.pos[s] << "\n";
            return 2;
        }
    const std::size_t H = cfg.sample - cfg.train, half = H / 2, N = H + (std::size_t{1} << 20);
    std::vector<std::uint8_t> data;
    if (const std::string e = read_prefix(a.wiki, cfg.sample + (std::size_t{1} << 20), data); !e.empty()) {
        std::cerr << e << "\n";
        return 1;
    }
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);
    model.compact_match();
    const PackedModel pm = pack(model);
    model.release();
    const FmInfiniGram ig(data.data(), cfg.train);
    std::vector<double> pt(N), pi(N);
    std::vector<int> bi(N);
    FmInfiniGram::Cursor cur;
    for (std::size_t k = 0, prev = 0; k < N; ++k) {
        const std::size_t i = cfg.train + k;
        double e[5][256], M[256];
        bool em = false;
        unsigned pred = 0;
        pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
        const std::size_t len = std::min<std::size_t>(i, 8192);
        const auto r = ig.query(data.data() + (i - len), len, 8192, k == 0 ? -1 : std::min<int>(static_cast<int>(prev) + 1, 8192),
                                1, cur);
        prev = static_cast<std::size_t>(r.n);
        pi[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
        bi[k] = bucket(r.n, r.total);
    }
    auto bits = [](const std::vector<double>& p, std::size_t from, std::size_t to) {
        std::vector<double> b;
        b.reserve(to - from);
        for (std::size_t k = from; k < to; ++k) b.push_back(-std::log2(p[k]));
        return b;
    };
    auto mean = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    auto gain = [](const std::vector<double>& with, const std::vector<double>& without) {
        std::vector<double> d(with.size());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] = with[k] - without[k];
        return d;
    };
    // cur = lam pe + (1 - lam) base, lam per bucket fit on k < upto (half: the holdout read; H: the fresh read).
    auto mixed = [&](const std::vector<double>& pe, const std::vector<int>& bk, int nb, const std::vector<double>& base,
                     std::size_t upto) {
        std::vector<double> lam(static_cast<std::size_t>(nb), 0.0), out(base.size());
        fit_lambdas(pe, base, bk, upto, nb, lam.data());
        for (std::size_t k = 0; k < base.size(); ++k) out[k] = lam[static_cast<std::size_t>(bk[k])] * pe[k] + (1 - lam[static_cast<std::size_t>(bk[k])]) * base[k];
        return out;
    };
    const std::vector<double> pih = mixed(pi, bi, kBuckets, pt, half), pif = mixed(pi, bi, kBuckets, pt, H);
    std::printf("bkn base holdout2 trie %.6f trie+ig %.6f", mean(bits(pt, half, H)), mean(bits(pih, half, H)));
    if (a.bkn_fresh) std::printf(" | fresh trie %.6f trie+ig %.6f", mean(bits(pt, H, N)), mean(bits(pif, H, N)));
    std::printf("\n");

    // The defaults on top (--bkn-stack): class shape, similar-context default, copy (min 8), each mixed by lambda per its
    // bucket over what came before, as qa mixes them.
    std::vector<double> pc, ps, pcp;
    std::vector<int> bc, sb, cb;
    if (a.bkn_stack) {
        auto cc = make_class_context("shape");
        if (!cc) return 1;
        cc->build(data.data(), cfg.train);
        pc.resize(N), bc.resize(N);
        for (std::size_t k = 0; k < N; ++k) {
            double P[256];
            bc[k] = ClassContext::bucket(cc->predict(data.data(), cfg.train + k, P));
            pc[k] = P[data[cfg.train + k]];
        }
        cc.reset();
        const SimSpec ss = parse_sim("default");
        auto sc = make_similar_context(ss);
        if (!ss.ok || !sc) return 1;
        sc->build(data.data(), cfg.train);
        ps.resize(N), sb.resize(N);
        for (std::size_t k = 0; k < N; ++k) {
            double P[256];
            sb[k] = SimilarContext::bucket(sc->predict(data.data(), cfg.train + k, P));
            ps[k] = P[data[cfg.train + k]];
        }
        sc.reset();
        pcp.resize(N), cb.resize(N);
        for (std::size_t k = 0; k < N; ++k) {
            int n = 0;
            pcp[k] = copy_pred(data.data(), cfg.train + k, n) == data[cfg.train + k] ? 1.0 : 0.0;
            cb[k] = copy_bucket(n);
        }
        std::cerr << "bkn: class, similar-context and copy predictions done\n";
    }
    auto stack = [&](const std::vector<double>& base, std::size_t upto) {
        std::vector<double> x = mixed(pc, bc, ClassContext::kBuckets, base, upto);
        x = mixed(ps, sb, SimilarContext::kBuckets, x, upto);
        std::vector<double> lam(kCopyBuckets, 0.0);
        fit_lambdas(pcp, x, cb, upto, kCopyBuckets, lam.data());
        copy_clip(lam.data(), a.copy_min);
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = lam[static_cast<std::size_t>(cb[k])] * pcp[k] + (1 - lam[static_cast<std::size_t>(cb[k])]) * x[k];
        return x;
    };
    std::vector<double> sth, stf;
    if (a.bkn_stack) {
        sth = stack(pih, half);
        if (a.bkn_fresh) stf = stack(pif, H);
        std::printf("bkn base_stack holdout2 trie+ig+cls+sim+copy %.6f", mean(bits(sth, half, H)));
        if (a.bkn_fresh) std::printf(" | fresh %.6f", mean(bits(stf, H, N)));
        std::printf("\n");
    }

    for (std::size_t s = 1; s < a.pos.size(); ++s) {
        const std::vector<int> v = bkn_spec(a.pos[s]);
        cypha::cyphalm::bkn::Prune pr;
        pr.from_order = v[1], pr.min_ctx = static_cast<std::uint32_t>(v[2]), pr.drop_singletons = v[3] != 0;
        const auto t0 = std::chrono::steady_clock::now();
        const cypha::cyphalm::bkn::ByteKN kn(data.data(), cfg.train, v[0], pr);
        const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<double> pk(N);
        std::vector<int> kb(N);
        const auto t1 = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < N; ++k) pk[k] = kn.prob_bucket(data.data(), cfg.train + k, data[cfg.train + k], kb[k]);
        const double score_us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() * 1e6 / static_cast<double>(N);
        const std::string sp = a.pos[s];
        std::printf("bkn spec %s table_bytes %zu contexts %zu children %zu build_s %.1f score_us_per_byte %.3f\n", sp.c_str(), kn.bytes(),
                    kn.contexts(), kn.children(), build_s, score_us);
        for (int k = 1; k <= v[0]; ++k) {
            const auto& o = kn.table(k);
            std::printf("bkn spec %s order %d D %.4f %.4f %.4f contexts %llu->%zu children %llu->%zu\n", sp.c_str(), k, o.D[1], o.D[2], o.D[3],
                        static_cast<unsigned long long>(o.ctx_before), o.key.size(), static_cast<unsigned long long>(o.child_before), o.sym.size());
        }
        // (a) replace; (b) add.
        const std::vector<double> ah = mixed(pi, bi, kBuckets, pk, half);
        const std::vector<double> bh0 = mixed(pk, kb, kn.buckets(), pt, half);
        const std::vector<double> bh = mixed(pi, bi, kBuckets, bh0, half);
        const auto h_ti = bits(pih, half, H), h_a = bits(ah, half, H), h_b = bits(bh, half, H);
        std::printf("bkn spec %s holdout2 kn %.6f kn+ig(a) %.6f trie+kn(b) %.6f trie+kn+ig(b) %.6f\n", sp.c_str(), mean(bits(pk, half, H)),
                    mean(h_a), mean(bits(bh0, half, H)), mean(h_b));
        print_boot("bkn spec " + sp + " holdout2 gain_a_over_trie_ig", block_boot(gain(h_a, h_ti), 4096, 2000, 11));
        print_boot("bkn spec " + sp + " holdout2 gain_b_over_trie_ig", block_boot(gain(h_b, h_ti), 4096, 2000, 11));
        if (a.bkn_stack) {
            const auto s_base = bits(sth, half, H), s_a = bits(stack(ah, half), half, H), s_b = bits(stack(bh, half), half, H);
            std::printf("bkn spec %s holdout2 stack(a) %.6f stack(b) %.6f\n", sp.c_str(), mean(s_a), mean(s_b));
            print_boot("bkn spec " + sp + " holdout2 stack_gain_a", block_boot(gain(s_a, s_base), 4096, 2000, 11));
            print_boot("bkn spec " + sp + " holdout2 stack_gain_b", block_boot(gain(s_b, s_base), 4096, 2000, 11));
        }
        if (a.bkn_fresh) {
            const std::vector<double> af = mixed(pi, bi, kBuckets, pk, H);
            const std::vector<double> bf = mixed(pi, bi, kBuckets, mixed(pk, kb, kn.buckets(), pt, H), H);
            const auto f_ti = bits(pif, H, N), f_a = bits(af, H, N), f_b = bits(bf, H, N);
            std::printf("bkn spec %s fresh kn %.6f kn+ig(a) %.6f trie+kn+ig(b) %.6f\n", sp.c_str(), mean(bits(pk, H, N)), mean(f_a), mean(f_b));
            print_boot("bkn spec " + sp + " fresh gain_a_over_trie_ig", block_boot(gain(f_a, f_ti), 4096, 4000, 11));
            print_boot("bkn spec " + sp + " fresh gain_b_over_trie_ig", block_boot(gain(f_b, f_ti), 4096, 4000, 11));
            if (a.bkn_stack) {
                const auto s_base = bits(stf, H, N), s_a = bits(stack(af, H), H, N), s_b = bits(stack(bf, H), H, N);
                std::printf("bkn spec %s fresh stack(a) %.6f stack(b) %.6f\n", sp.c_str(), mean(s_a), mean(s_b));
                print_boot("bkn spec " + sp + " fresh stack_gain_a", block_boot(gain(s_a, s_base), 4096, 4000, 11));
                print_boot("bkn spec " + sp + " fresh stack_gain_b", block_boot(gain(s_b, s_base), 4096, 4000, 11));
            }
        }
        std::fflush(stdout);
    }
    return 0;
}

int cmd_selftest() {
    if (!self_test()) {
        std::cerr << "selftest: packed_trie self_test failed\n";
        return 1;
    }
    Config cfg;
    cfg.sample = 40'000;
    cfg.train = 36'000;
    cfg.reserve_nodes = std::size_t{1} << 16;
    cfg.reserve_fols = std::size_t{1} << 17;
    cfg.match_bits = 12;
    std::vector<std::uint8_t> data = tiny_corpus(cfg.sample);
    Model model;
    model.verbose = false;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);
    model.compact_match();
    const Holdout h = score_holdout(model, data.data(), w, cfg, false);
    const Holdout hp = score_holdout(pack(model), data.data(), w, cfg, false);  // M4: bit-identical
    if (!h.ok || !hp.ok || std::memcmp(&h.backoff_bpc, &hp.backoff_bpc, sizeof(double)) != 0 ||
        std::memcmp(&h.mix_bpc, &hp.mix_bpc, sizeof(double)) != 0) {
        std::cerr << "selftest: holdout scoring failed, or the packed form differs\n";
        return 1;
    }
    double ws = 0;
    for (int i = 0; i < 6; ++i) ws += w[i];
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);
    std::cout << "tiny_backoff_bpc " << h.backoff_bpc << "\n";
    std::cout << "tiny_mix_bpc " << h.mix_bpc << "\n";
    if (std::fabs(ws - 1.0) > 1e-9) {
        std::cerr << "selftest: weights sum " << ws << "\n";
        return 1;
    }
    if (std::fabs(h.backoff_bpc - kGoldenBackoff) > kGoldenTol || std::fabs(h.mix_bpc - kGoldenMix) > kGoldenTol) {
        std::cerr << "selftest: golden BPC mismatch (want " << kGoldenBackoff << " and " << kGoldenMix << ")\n";
        return 1;
    }
    // B4C: the gates, on the tiny model with a class expert and copy lambdas. Inside an answer the gated distribution
    // is the trie's, bit for bit; outside one it is not; qa's --copy models (lc) keep copying inside answers.
    {
        const PackedModel pk = pack(model);
        ClassContext tc({}, ClassContext::Options{});
        tc.build(data.data(), cfg.train);
        double lcl[ClassContext::kBuckets], lcp[kCopyBuckets];
        std::fill(lcl, lcl + ClassContext::kBuckets, 0.5);
        std::fill(lcp, lcp + kCopyBuckets, 0.5);
        const Lm plain{nullptr, w, cfg.train, false, nullptr, nullptr, nullptr, 8192, nullptr, nullptr, nullptr, nullptr, &pk};
        Lm gated = plain, open = plain;
        gated.cc = open.cc = &tc;
        gated.lcls = open.lcls = lcl;
        gated.cls_gate = open.cls_gate = true;
        gated.lc_out = lcp;
        open.lc = lcp;
        auto dist = [](const Lm& lm, const std::string& t, double* P) {
            const std::vector<std::uint8_t> seq(t.begin(), t.end());
            int hint = -1;
            lm.dist(seq, hint, P);
            return seq;
        };
        const std::string inside = "User: the trie model is a byte context?\nAssistant: the trie model is a",
                          outside(data.begin() + static_cast<std::ptrdiff_t>(cfg.train), data.begin() + static_cast<std::ptrdiff_t>(cfg.train) + 300);
        double A[256], B[256], C[256], D[256], E[256];
        dist(plain, inside, A);
        const auto si = dist(gated, inside, B);
        dist(open, inside, C);
        const auto so = dist(plain, outside, D);
        dist(gated, outside, E);
        int n = 0;
        const bool ok = in_answer(si, std::string::npos) && !in_answer(so, std::string::npos) &&
                        std::memcmp(A, B, sizeof A) == 0 && std::memcmp(B, C, sizeof B) != 0 &&
                        copy_pred(si.data(), si.size(), n) == ' ' && std::memcmp(D, E, sizeof D) != 0;
        if (!ok) {
            std::cerr << "selftest: class / copy gate checks failed\n";
            return 1;
        }
        // BKN3: Kneser-Ney off inside an answer (kn_gate) through the same gate: inside one, the trie + InfiniGram
        // distribution with lam_ans, as a model without Kneser-Ney gives it; outside one, Kneser-Ney and lam as before.
        const cypha::cyphalm::bkn::ByteKN tkn(data.data(), cfg.train, 4);
        const FmInfiniGram tig(data.data(), cfg.train);
        std::vector<double> lk(static_cast<std::size_t>(tkn.buckets()), 0.5);
        double la[kBuckets], lb[kBuckets];
        std::fill(la, la + kBuckets, 0.3);
        std::fill(lb, lb + kBuckets, 0.7);
        Lm ig_a = plain, ig_b = plain;
        ig_a.ig = ig_b.ig = &tig;
        ig_a.lam = la;
        ig_b.lam = lb;
        Lm kn_on = ig_a, kn_off = ig_a;
        kn_on.kn = kn_off.kn = &tkn;
        kn_on.lkn = kn_off.lkn = lk.data();
        kn_off.kn_gate = true;
        kn_off.lam_ans = lb;
        double F[256], G[256], H[256], I[256], J[256], K[256];
        dist(ig_b, inside, F);    // no Kneser-Ney, InfiniGram at lb
        dist(kn_off, inside, G);  // gated: must be F
        dist(kn_on, inside, H);   // BKN2: Kneser-Ney inside the answer
        dist(kn_on, outside, I);
        dist(kn_off, outside, J);  // outside an answer the gate changes nothing: must be I
        dist(ig_a, outside, K);    // and Kneser-Ney is in it: not K
        if (std::memcmp(F, G, sizeof F) != 0 || std::memcmp(G, H, sizeof G) == 0 || std::memcmp(I, J, sizeof I) != 0 ||
            std::memcmp(J, K, sizeof J) == 0) {
            std::cerr << "selftest: Kneser-Ney answer gate checks failed (BKN3)\n";
            return 1;
        }
    }
    // GEN-B: the sentence gate on the tiny model with copy lambdas. Specs parse or are refused; the text is n bytes and
    // the same for the same seed; the first sentence kept is the best-scoring of the K candidates (ties: the lower k);
    // after the prompt "\n" the pmi context is the real one, so every pmi is exactly 0.
    {
        const PackedModel pk = pack(model);
        double lcp[kCopyBuckets];
        std::fill(lcp, lcp + kCopyBuckets, 0.5);
        GateSpec g4, gm, gp;
        bool ok = parse_gate("4", g4) && g4.k == 4 && g4.wf == 1 && g4.wp == 0 && g4.wm == 0 && g4.min_len == 24 &&
                  parse_gate("6,wf=0,wm=1,min=10,max=60", gm) && gm.k == 6 && gm.wf == 0 && gm.wm == 1 && gm.min_len == 10 &&
                  gm.max_len == 60 && parse_gate("3,wp=1,wm=2", gp) && gp.wp == 1 && gp.wm == 2;
        for (const char* s : {"", "0", "4,x=1", "4,wp", "4,min=50,max=10", "4,min=0"}) {
            GateSpec bad;
            ok = ok && !parse_gate(s, bad);
        }
        bool varied = false, marked = false;
        auto run = [&](const std::string& p, int n, const GateSpec& g, GateStats& s) {
            return sample_gated(pk, w, cfg.train, nullptr, nullptr, nullptr, nullptr, lcp, p, n, 17, 0.8, g, s);
        };
        for (const GateSpec* g : {&g4, &gm, &gp})
            for (std::size_t q = 0; q < 8; ++q) {  // eight prompts from the tiny holdout
                const std::string prompt(data.begin() + static_cast<std::ptrdiff_t>(cfg.train + 300 * q),
                                         data.begin() + static_cast<std::ptrdiff_t>(cfg.train + 300 * q + 64));
                GateStats st, st2;
                const std::string a = run(prompt, 300, *g, st);
                if (q == 0) {
                    const std::string b = run(prompt, 300, *g, st2), c = run(prompt, 37, *g, st2);
                    ok = ok && a.size() == 300 && a == b && c.size() == 37 && st.sentences >= 2;
                }
                const std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
                std::vector<GateCand> cand;
                for (int k = 0; k < g->k; ++k)
                    cand.push_back(gate_candidate(pk, w, cfg.train, nullptr, nullptr, nullptr, nullptr, lcp, seq, seq.size(),
                                                  std::min(g->max_len, 300), gate_seed(17, 0, k), 0.8, *g));
                std::size_t best = 0;
                double mmin = cand[0].m;
                for (std::size_t k = 1; k < cand.size(); ++k) {
                    if (cand[k].score > cand[best].score) best = k;
                    mmin = std::min(mmin, cand[k].m);
                    varied = varied || cand[k].bytes != cand[0].bytes;
                    marked = marked || (g == &gm && cand[k].m != cand[0].m);
                }
                ok = ok && a.compare(0, cand[best].bytes.size(), std::string(cand[best].bytes.begin(), cand[best].bytes.end())) == 0;
                if (g == &gm) ok = ok && cand[best].m == mmin;
                if (g == &gp && q == 0) {
                    const GateCand z = gate_candidate(pk, w, cfg.train, nullptr, nullptr, nullptr, nullptr, lcp,
                                                      std::vector<std::uint8_t>(1, '\n'), 1, 120, gate_seed(17, 0, 0), 0.8, gp);
                    bool some = false;
                    for (const GateCand& x : cand) some = some || x.p != 0;
                    ok = ok && z.p == 0 && z.bytes.size() > 0 && some;
                }
            }
        if (!ok || !varied || !marked) {
            std::cerr << "selftest: sentence gate (GEN-B) checks failed\n";
            return 1;
        }
    }
    // qa grading, the don't-know message and tag, Wilson bounds and the abstain gate.
    {
        const std::string dk = dont_know_text("Zorbia", "capital", "UNKNOWN_ENTITY");
        std::vector<Graded> gs(4);
        const double mlp[4] = {-0.5, -1.0, -2.0, -3.0};
        const int grd[4] = {0, 0, 2, 2};
        for (int i = 0; i < 4; ++i) {
            gs[static_cast<std::size_t>(i)].fv[0] = mlp[i];
            gs[static_cast<std::size_t>(i)].grade = grd[i];
        }
        std::vector<const Graded*> v;
        for (const auto& g : gs) v.push_back(&g);
        int right = 0;
        const Gate gate = choose_gate(v, false, &right);
        const bool ok = gold_match("Antoni Gaud\xc3\xad designed it.", {"Gaudi"}) &&
                        !gold_match("Parish council", {"Paris"}) && gold_match("Gaudi", {"Antoni Gaudi"}) &&
                        !gold_match("the", {"Antoni Gaudi"}) && is_dont_know(dk) &&
                        tag_field(dk, "reason") == "UNKNOWN_ENTITY" && tag_field(dk, "entity") == "Zorbia" &&
                        tag_field(dk, "relation") == "capital" && std::fabs(wilson_lo(0, 10)) < 1e-12 &&
                        std::fabs(wilson_hi(0, 130) - 0.028703) < 1e-5 && gate.f == 0 && gate.tau == -1.0 && right == 2;
        if (!ok) {
            std::cerr << "selftest: qa grading / tag / gate checks failed\n";
            return 1;
        }
    }
    // S1, I3: whenever the gate does not keep an answer, the card is a don't-know with a known code. CyphaLM's own code
    // stands; its message without a code takes the gate's; the gate's message parses back; the reference rule; the
    // audit draw; evidence places on a tiny index.
    {
        const Gate gate{0, -1.0};
        const std::string own = dont_know_text("Zorbia", "capital", "NO_EVIDENCE");
        const char* const outs[] = {"Paris", "", "I don't know.", "<|fam=low|>I don't know.", own.c_str(),
                                    "x reason=UNKNOWN_ENTI"};
        bool i3 = true;
        std::mt19937 rng(7);
        for (int k = 0; k < 600; ++k) {
            const std::string ans = outs[k % 6];
            Graded g;
            g.fv[0] = -3.0 + 4.0 * std::uniform_real_distribution<double>(0, 1)(rng);
            g.grade = is_dont_know(ans) || ans.empty() ? 1 : 2;
            const bool kept = gated_answer(g, gate);
            for (const char* gc : {"LOW_CERTAINTY", "UNKNOWN_ENTITY", "NO_EVIDENCE"}) {
                const Card c = final_card(ans, kept, gc, "Zorbia", "capital");
                if (!kept) i3 = i3 && c.kind == "dont_know" && known_code(c.code) && is_dont_know(c.text);
                else i3 = i3 && c.kind == "answer" && c.text == ans && g.fv[0] >= gate.tau;
            }
        }
        const Card c_own = final_card(own, false, "LOW_CERTAINTY", "Zorbia", "capital");
        const Card c_bare = final_card("<|fam=low|>I don't know.", false, "LOW_CERTAINTY", "Zorbia", "capital");
        const Card c_gate = final_card("Paris", false, "UNKNOWN_ENTITY", "Zorbia", "capital");
        bool msgs = true;
        for (const char* code : {"LOW_CERTAINTY", "UNKNOWN_ENTITY", "NO_EVIDENCE"}) {
            const std::string m = gate_message("Zorbia", "capital", code);
            msgs = msgs && is_dont_know(m) && tag_field(m, "reason") == code && tag_field(m, "entity") == "Zorbia" &&
                   tag_field(m, "relation") == "capital" && m.compare(m.size() - 2, 2, "|>") == 0;
        }
        const auto au = audit_rows(130);
        bool draw = au.size() == 100 && std::is_sorted(au.begin(), au.end()) &&
                    std::adjacent_find(au.begin(), au.end()) == au.end() && au.back() < 130 && audit_rows(5).size() == 5;
        // Evidence: a tiny text of two pages, a boundary and two chat turns.
        const std::string w = "<page><title>Alpha</title> one two</page><page><title>Beta</title> three</page>";
        const std::string chat = "User: who?\nAssistant: Beta.\n\nUser: where?\nAssistant: I don't know.\n\n";
        evidence::Sources src;
        src.add_pages(reinterpret_cast<const std::uint8_t*>(w.data()), w.size());
        src.add(w.size(), "boundary");
        src.add_turns(chat, w.size() + 1, [](const std::string& t) {
            return std::string(t.find("I don't know.") != std::string::npos ? "dk" : "dolly");
        });
        const std::string text = w + "\n" + chat;
        FmInfiniGram ix(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
        ix.build_locate(4);
        const auto f = evidence::longest(ix, src, "zzz Assistant: Beta.", 100);
        const auto m = evidence::occurrences(ix, src, "three");
        const auto d = evidence::occurrences(ix, src, "Assistant: I");
        const bool ev = f.n == 16 && f.count == 1 && f.places.size() == 1 && f.places[0] == "dolly:User: who?+11" &&
                        m.count == 1 && m.places[0] == "wiki:Beta+" + std::to_string(w.find("three") - w.find("<title>Beta")) &&
                        d.count == 1 && d.places[0] == "dk:User: where?+13" && src.where(w.size()) == "boundary+0";
        const bool ok = i3 && c_own.by == "cyphalm" && c_own.code == "NO_EVIDENCE" && c_bare.by == "cyphalm+gate" &&
                        c_bare.code == "LOW_CERTAINTY" && c_gate.by == "gate" && c_gate.code == "UNKNOWN_ENTITY" && msgs &&
                        reference_code(0, -1, -1) == "UNKNOWN_ENTITY" && reference_code(1, 1, 0) == "LOW_CERTAINTY" &&
                        reference_code(1, 0, 2) == "LOW_CERTAINTY" && reference_code(1, 0, 1) == "NO_EVIDENCE" && draw && ev;
        if (!ok) {
            std::cerr << "selftest: S1 cards / I3 checks failed (i3 " << i3 << " messages " << msgs << " draw " << draw
                      << " evidence " << ev << ")\n";
            return 1;
        }
    }
    // A1c / A2 / A3: copy expert, familiarity statistics' AUROC and cut, frames, candidates, the statement choice,
    // the question's relation and the shard leakage check.
    {
        const std::string c = "the cat sat. the cat ";
        int n = 0, n0 = 0;
        const int p = copy_pred(reinterpret_cast<const std::uint8_t*>(c.data()), c.size(), n);
        const int p0 = copy_pred(reinterpret_cast<const std::uint8_t*>(c.data()), 5, n0);
        double j = 0, margin = 0;
        const double cut = youden({2, 3, 4}, {0, 0, 1}, j);
        const std::string f = render_frame("^ [S] ( born [O] ) ,", "\x01", "\x02");
        const auto cy = r3_candidates("Andrei Tarkovsky ([[April 4]], [[1932]] - 1986) was", true, false);
        const auto cp = r3_candidates("the village of [[Zavrazhye]], in Ivanovo Oblast", false, true);
        const int win = r3_choose({2.5, 1.25, 1.5}, margin);
        const std::vector<DkRow> sh = {{"When was Al Gore born?", "Al Gore", "birth_date", "seed", "1948"},
                                       {"In what year was Al Gore born?", "Al Gore", "birth_year", "seed", "1948"}};
        const bool ok = p == 's' && n == 8 && p0 == -1 && n0 == 0 && copy_bucket(8) == 4 &&
                        std::fabs(auroc({2, 3}, {1, 2}) - 0.875) < 1e-12 && cut == 2 && std::fabs(j - 1) < 1e-12 &&
                        fill_frame(f, "Ann Lee", "1900") == "Ann Lee (born 1900)," &&
                        render_frame("[S] ' s capital , [O]", "X", "Y") == "X's capital, Y" &&
                        cy.size() == 2 && cy[0].second == "1932" && cy[1].first == "1986" && cp.size() == 3 &&
                        cp[0] == std::pair<std::string, std::string>{"the village of [[Zavrazhye]]", "the village of Zavrazhye"} &&
                        cp[1].second == "Zavrazhye" && cp[2].second == "Ivanovo Oblast" &&
                        win == 1 && std::fabs(margin - 0.25) < 1e-12 && r3_candidates("met with in the 13th century", false, true).empty() &&
                        r3_candidates("Ryazan]], Russia", false, true, "]]")[0].first == "Ryazan" &&
                        question_relation("when was al gore born", "Al Gore") == "birth_date" &&
                        question_relation("Who directed the film Jaws?", "the film Jaws") == "director" &&
                        question_relation("Who was the second Prime Minister of Canada?", "Canada").empty() &&
                        strip_article("the Ashes") == "Ashes" &&
                        shard_leaks(sh, {"When was Al Gore born", "Who wrote Ulysses?"}) == 1;
        if (!ok) {
            std::cerr << "selftest: copy / familiarity / frame / candidate / shard checks failed\n";
            return 1;
        }
    }
    {
        // A1d: the message's entity, and --copy-min 8 (B4B) zeroing the copy buckets below 8 bytes.
        const std::string m = dont_know_text("Zorbia", "capital", "UNKNOWN_ENTITY");
        double lam[kCopyBuckets] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
        copy_clip(lam, 8);
        const bool ok = lam[3] == 0 && lam[1] == 0 && lam[4] == 0.4 && lam[7] == 0.7 && msg_entity(m) == "Zorbia" &&
                        msg_entity("I don't know.").empty() && names_in("the Zorbia", "What is the capital of Zorbia?") &&
                        !names_in("or", "Zorbia or not?");
        if (!ok) {
            std::cerr << "selftest: message entity / copy-min checks failed\n";
            return 1;
        }
    }
    {
        // ANS: the entity frames in the token, and the don't-know context of --copy-dk (only with its lambdas, only
        // once the answer begins "I don't know").
        auto b = [](const char* x) { return std::vector<std::uint8_t>(x, x + std::strlen(x)); };
        const double l8[kCopyBuckets] = {0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9};
        Lm d;
        d.lc_dk = l8;
        d.answer_at = 13;  // the answer starts after "Who? x\nAsst: " (13 bytes)
        Lm none = d;
        none.lc_dk = nullptr;
        const bool ok = fam_token("low", "entity=\"{E}\"", "the Napoleon") == "<|entity=\"Napoleon\"|><|fam=low|>" &&
                        fam_token("low", "entity=\"{E}\"", "") == "<|fam=low|>" && fam_token("high", "", "X") == "<|fam=high|>" &&
                        fam_token("low", "about {E};entity=\"{E}\"", "Zorbia") == "<|about Zorbia|><|entity=\"Zorbia\"|><|fam=low|>" &&
                        d.in_dont_know(b("Who? x\nAsst: I don't know. I")) && !d.in_dont_know(b("Who? x\nAsst: I don't kno")) &&
                        !d.in_dont_know(b("Who? x\nAsst: It is Paris")) && !none.in_dont_know(b("Who? x\nAsst: I don't know. I")) &&
                        !d.in_dont_know(b("I don't know. I"));
        if (!ok) {
            std::cerr << "selftest: ANS frame / don't-know context checks failed\n";
            return 1;
        }
    }
    {
        // DK2: the frames alone; the plain-turn spread of --dk-plain; --copy-frame: a frame's end is not copied into the
        // message (no "<|" open in the answer), but a tag the answer opened may be closed.
        auto b = [](const std::string& x) { return std::vector<std::uint8_t>(x.begin(), x.end()); };
        int p50 = 0, p75 = 0, p0 = 0;
        for (std::size_t j = 0; j < 8; ++j) p50 += dk_plain_turn(j, 0.5), p75 += dk_plain_turn(j, 0.75), p0 += dk_plain_turn(j, 0);
        const std::string pr = "<|about Zorbia|><|entity=\"Zorbia\"|>User: Who? x\nAssistant: ";
        const auto m = b(pr + "I don't know. I have nothing about Zorbia");
        const auto t = b(pr + "I don't know. I have nothing about Zorbia in it. <|unknown entity=\"Zorbia\"");
        const auto u = b(pr + "I don't know. <|unknown|> about Zorbia");
        int n1 = 0, n2 = 0, n3 = 0;
        std::size_t j1 = 0, j2 = 0, j3 = 0;
        const int c1 = copy_pred(m.data(), m.size(), n1, &j1), c2 = copy_pred(t.data(), t.size(), n2, &j2),
                  c3 = copy_pred(u.data(), u.size(), n3, &j3);
        const bool ok = fam_frames("about {E};entity=\"{E}\"", "the Zorbia") == "<|about Zorbia|><|entity=\"Zorbia\"|>" &&
                        fam_frames("entity=\"{E}\"", "").empty() && p50 == 4 && p75 == 6 && p0 == 0 && !dk_plain_turn(0, 0.5) &&
                        dk_plain_turn(1, 0.5) && c1 == '|' && n1 >= 12 && copy_closes_unopened(m.data(), m.size(), j1, pr.size()) &&
                        c2 == '|' && !copy_closes_unopened(t.data(), t.size(), j2, pr.size()) && c3 == '|' &&
                        copy_closes_unopened(u.data(), u.size(), j3, pr.size()) &&
                        !copy_closes_unopened(m.data(), m.size(), 1, pr.size()) &&  // "<|" opens, it does not close
                        open_token_at(m.data(), m.size(), pr.size()) == std::string::npos &&
                        open_token_at(t.data(), t.size(), pr.size()) == t.size() - 25 &&  // at `<|unknown entity="Zorbia"`
                        open_token_at(u.data(), u.size(), pr.size()) == std::string::npos;
        if (!ok) {
            std::cerr << "selftest: DK2 frames / plain turns / copy-frame checks failed\n";
            return 1;
        }
    }
    {
        // B4: the class-context expert on a tiny corpus. A verb of the same class reuses what "wrote" learned; an
        // unknown word falls to the uniform floor; P sums to 1; the bucket packs order and count.
        ClassContext cc({{"wrote", 10}, {"composed", 10}, {"who", 11}, {"the", 12}}, ClassContext::Options{6, 2, false, 1});
        const std::string t = "who wrote the song? who wrote the play? who wrote the book? ";
        cc.build(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
        auto ask = [&](const std::string& h, double* P) {
            return cc.predict(reinterpret_cast<const std::uint8_t*>(h.data()), h.size(), P);
        };
        double P1[256], P2[256], P3[256];
        const auto h1 = ask("who wrote t", P1), h2 = ask("who composed t", P2), h3 = ask("who painted t", P3);
        const double sum = std::accumulate(P1, P1 + 256, 0.0), sum3 = std::accumulate(P3, P3 + 256, 0.0);
        const bool ok = h1.order == 2 && h1.total == 3 && P1['h'] > 0.9 && std::fabs(sum - 1) < 1e-9 && h2.order == 2 &&
                        std::fabs(P2['h'] - P1['h']) < 1e-12 && h3.order == 0 && std::fabs(P3['h'] - 1.0 / 256) < 1e-12 &&
                        std::fabs(sum3 - 1) < 1e-9 && ClassContext::bucket({2, 3}) == 9 && cc.contexts() == 68 &&
                        make_class_context("shape:4") != nullptr;
        // Spec fields are read from the right (M2/M3 flags). The default is three orders, compact, contexts seen
        // once dropped (Odin, 2026-10-28); a flags field states the storage in full, and f is the full table.
        auto built = [&](const std::string& spec) {
            const auto c = make_class_context(spec);
            if (c) c->build(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
            return c ? c->contexts() : std::size_t{0};
        };
        const std::size_t sd = built("shape"), s2f = built("shape:6:2:f"), s3f = built("shape:6:3:f"),
                          s3p = built("shape:6:3:p2"), s3cp = built("shape:6:3:cp2");
        const bool flags_ok = sd == built("shape:6") && sd == built("shape:6:3") && sd == s3cp && s2f == 64 &&
                              s3f == 103 && s3p == 42 && s3cp == 42;
        if (!ok || !flags_ok) {
            std::cerr << "selftest: class-context expert checks failed (spec sizes " << sd << " " << s2f << " " << s3f
                      << " " << s3p << " " << s3cp << ")\n";
            return 1;
        }
    }
    {
        // A4: the similar-context expert on a tiny corpus. A verb with the same vector reaches the stored contexts of
        // "who wrote the", whose next byte after the tail " s" is 'o'; a tail no neighbour shares kills them all; the
        // inverted file finds the exact neighbours; the eigenvectors and the bucket packing.
        SimilarContext::Vectors V;
        V.d = 4;
        const char* const wd[] = {"who", "wrote", "composed", "the", "song", "cat", "sat", "on", "mat", "a", "dog", "ran"};
        const float vv[12][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0},
                                 {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 1, 0}, {1, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 1, 0}};
        for (int i = 0; i < 12; ++i) {
            V.id[wd[i]] = i;
            V.v.insert(V.v.end(), vv[i], vv[i] + 4);
        }
        SimilarContext::Params sp;
        sp.W = 3, sp.k = 4, sp.dim = 0, sp.drop = 0, sp.sif = false, sp.lists = 2, sp.probe = 1;
        const std::string t = "who wrote the song. the cat sat on the mat. who wrote the song. a dog ran. ";
        SimilarContext sc(V, sp);
        sc.build(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
        auto ask = [&](const std::string& h, double* P) {
            return sc.predict(reinterpret_cast<const std::uint8_t*>(h.data()), h.size(), P);
        };
        double P1[256], P2[256], P3[256];
        const auto h1 = ask("who composed the s", P1), h2 = ask("who wrote the s", P2), h3 = ask("who composed the x", P3);
        const std::string q = "who composed the s";
        const auto ex = sc.neighbours(reinterpret_cast<const std::uint8_t*>(q.data()), q.size(), true),
                   iv = sc.neighbours(reinterpret_cast<const std::uint8_t*>(q.data()), q.size(), false);
        std::vector<double> E;
        SimilarContext::eigen_sym({2, 1, 0, 1, 2, 0, 0, 0, 1}, 3, E);
        const bool ok = sc.keys() == 17 && h1.live == 2 && h1.word == 1 && h1.top > 0.99 && P1['o'] == 1.0 &&
                        std::fabs(std::accumulate(P1, P1 + 256, 0.0) - 1) < 1e-12 && h2.live == 2 && P2['o'] == 1.0 &&
                        h3.live == 0 && std::accumulate(P3, P3 + 256, 0.0) == 0 && SimilarContext::bucket(h3) == 0 &&
                        SimilarContext::bucket(h1) == 118 && ex.size() == 4 && ex == iv && ex[0].first == 13 &&
                        std::fabs(std::fabs(E[0]) - std::sqrt(0.5)) < 1e-9 && std::fabs(E[6]) < 1e-9 &&
                        parse_sim("vec=x,W=4,k=8,src=wiki").ok && !parse_sim("vec=x,src=dolly").ok && !parse_sim("W4").ok &&
                        parse_sim("vec=x,k=8").build == parse_sim("vec=x,k=128,T=0").build &&
                        parse_sim("vec=x,W=4").build != parse_sim("vec=x,W=8").build;
        if (!ok) {
            std::cerr << "selftest: similar-context expert checks failed\n";
            return 1;
        }
        // A4 defaults: the gate's rule (chat turns and decode's prompt end), the default spec, and a projected file:
        // written, read back (a flipped byte is refused) and built on the same text, it gives the same prediction.
        auto bytes = [](const std::string& x) { return std::vector<std::uint8_t>(x.begin(), x.end()); };
        const auto dflt = parse_sim("default"), dk8 = parse_sim("default,k=8"), ng = parse_sim("vec=x,gate=0");
        std::stringstream file;
        sc.save_projected(file);
        std::string bad_file = file.str(), err, err2;
        bad_file[bad_file.size() / 2] ^= 1;
        std::stringstream bad(bad_file);
        const auto PV = SimilarContext::load_projected(file, err);
        SimilarContext::load_projected(bad, err2);
        SimilarContext sx(PV, sp);
        sx.build(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
        double Q1[256];
        const auto hq = sx.predict(reinterpret_cast<const std::uint8_t*>(q.data()), q.size(), Q1);
        const bool ok2 = in_answer(bytes("User: who?\nAssistant: "), std::string::npos) &&
                         in_answer(bytes("User: who?\nAssistant: it is "), std::string::npos) &&
                         !in_answer(bytes("User: who?\nAssistant: it.\n\nUser: and "), std::string::npos) &&
                         !in_answer(bytes("the wiki text"), std::string::npos) && in_answer(bytes("Who? "), 5) &&
                         !in_answer(bytes("Who?"), 5) && dflt.ok && dflt.p.W == 2 && !dflt.p.sif && dflt.p.k == 8192 && dflt.p.mode == 1 && dflt.p.probe == 128 && dflt.p.online &&
                         dflt.src == "wiki" && dflt.gate && dflt.vec == sim_vectors_path() && dk8.p.k == 8 && !ng.gate &&
                         err.empty() && !err2.empty() && PV.projected() && PV.rows() == 12 && hq.live == 2 && Q1['o'] == 1.0;
        if (!ok2) {
            std::cerr << "selftest: similar-context gate / default / projected file checks failed\n";
            return 1;
        }
        // A4c: with every occurrence pooled at equal weight (k above the key count, every list probed, T 0), knn and dedupe
        // give the same distribution, bit for bit, at every position of a random text over the same words.
        std::mt19937 rw(20261028);
        auto words = [&](int n) {
            std::string x;
            for (int i = 0; i < n; ++i) x += std::string(wd[rw() % 12]) + (rw() % 7 == 0 ? ". " : " ");
            return x;
        };
        const std::string big = words(3000), probe_text = words(80);
        SimilarContext::Params ap;
        ap.W = 2, ap.k = 1 << 30, ap.temp = 0, ap.dim = 0, ap.drop = 0, ap.sif = false, ap.lists = 8, ap.probe = 8;
        std::vector<std::unique_ptr<SimilarContext>> m3;
        for (int mode = 0; mode < 2; ++mode) {
            ap.mode = mode;
            m3.push_back(std::make_unique<SimilarContext>(V, ap));
            m3.back()->build(reinterpret_cast<const std::uint8_t*>(big.data()), big.size());
        }
        bool same = m3[1]->contexts() < m3[0]->contexts() && m3[1]->contexts() > 0 && m3[0]->keys() == m3[1]->keys() &&
                    parse_sim("default").p.mode == 1 && parse_sim("default,mode=knn").p.mode == 0 && !parse_sim("default,mode=x").ok &&
                    parse_sim("default,mode=knn").build != parse_sim("default").build;
        int live_pos = 0;
        for (std::size_t n = 1; same && n <= probe_text.size(); ++n) {
            double A[2][256];
            SimilarContext::Hit hh[2];
            for (int mode = 0; mode < 2; ++mode)
                hh[mode] = m3[static_cast<std::size_t>(mode)]->predict(reinterpret_cast<const std::uint8_t*>(probe_text.data()), n, A[mode]);
            live_pos += hh[0].live > 0;
            for (int mode = 1; mode < 2; ++mode)
                same = same && hh[mode].live == hh[0].live && hh[mode].word == hh[0].word && std::equal(A[0], A[0] + 256, A[mode]);
        }
        if (!same || live_pos < 100) {
            std::cerr << "selftest: similar-context dedupe mode (A4c) disagrees with knn\n";
            return 1;
        }
        // A4d, online keys read only bytes already scored: an expert that read a text up to m and one that read the same
        // text with every byte from m on replaced give the same distribution at m, bit for bit (both modes); and the
        // online keys are used (some position's distribution differs from the expert's without them).
        const std::string stream = words(400);
        bool causal = parse_sim("default").p.online && !parse_sim("default,online=0").p.online &&
                      parse_sim("default,online=0").build == parse_sim("default").build;
        int differs = 0;
        for (int mode = 0; mode < 2 && causal; ++mode) {
            SimilarContext::Params op = ap;
            op.mode = mode, op.k = 64, op.temp = 0.05, op.lists = 8, op.probe = 2, op.online = true;
            SimilarContext full(V, op);
            SimilarContext::Params np = op;
            np.online = false;
            SimilarContext none(V, np);
            for (auto* e : {&full, &none}) e->build(reinterpret_cast<const std::uint8_t*>(big.data()), 600);
            const auto* sb = reinterpret_cast<const std::uint8_t*>(stream.data());
            for (std::size_t m = 1; causal && m <= stream.size(); m += 7) {
                double Pf[256], Pc[256], Pn[256];
                for (std::size_t n = 1; n <= m; ++n) full.predict(sb, n, Pf);
                std::string other = stream.substr(0, m);
                for (std::size_t i = m; i < stream.size(); ++i) other += static_cast<char>('a' + (i * 7 + m) % 26);
                SimilarContext fresh(V, op);
                fresh.build(reinterpret_cast<const std::uint8_t*>(big.data()), 600);
                for (std::size_t n = 1; n <= m; ++n) fresh.predict(reinterpret_cast<const std::uint8_t*>(other.data()), n, Pc);
                none.predict(sb, m, Pn);
                causal = std::equal(Pf, Pf + 256, Pc);
                differs += !std::equal(Pf, Pf + 256, Pn);
            }
        }
        if (!causal || differs < 10) {
            std::cerr << "selftest: similar-context online keys (A4d) read a byte not yet scored, or are never used\n";
            return 1;
        }
    }
    {  // BKN (docs/reports/BYTE_KN.md): byte_kn.hpp equals token_model.hpp's Kneser-Ney over byte ids (to float
       // storage); prob() equals dist(); every distribution sums to one, pruned or not; the build's parts (BKN2) do not
       // change the tables.
        namespace bk = cypha::cyphalm::bkn;
        const std::vector<std::uint8_t> t = tiny_corpus(40'000);
        const std::size_t T = 36'000;
        const std::vector<std::uint32_t> ids(t.begin(), t.end());
        bool same = true, norm = true;
        for (int K : {1, 3, 6}) {
            const bk::ByteKN kn(t.data(), T, K);
            const cypha::cyphalm::tok::KneserNey ref(ids.data(), T, 256, K);
            for (std::size_t i = T; i < t.size(); i += 13) {
                const double x = kn.prob(t.data(), i, t[i]), y = ref.prob(ids.data(), i, ids[i]);
                if (std::fabs(x - y) > 1e-5 * y) same = false;
            }
        }
        bk::Prune p0, p1, p2, p3;
        p1.from_order = 3, p1.min_ctx = 2;
        p2.from_order = 3, p2.min_ctx = 2, p2.drop_singletons = true;
        p3.from_order = 2, p3.min_ctx = 3, p3.drop_singletons = true;
        for (const bk::Prune& pr : {p0, p1, p2, p3}) {
            const bk::ByteKN kn(t.data(), T, 6, pr);
            for (std::size_t i = T; i < t.size(); i += 101) {
                double P[256], z = 0;
                kn.dist(t.data(), i, P);
                for (double x : P) z += x;
                if (std::fabs(z - 1.0) > 1e-5 || std::fabs(P[t[i]] - kn.prob(t.data(), i, t[i])) > 1e-9) norm = false;
            }
        }
        bool parts = true;  // BKN2: a build in parts and slices of 64 or 1,000 entries gives the one-part tables
        for (int K : {1, 3, 6})
            for (const bk::Prune& pr : {p0, p2}) {
                bk::ByteKN one, a, b;
                one.build(t.data(), T, K, pr, 1u << 30);
                a.build(t.data(), T, K, pr, 64);
                b.build(t.data(), T, K, pr, 1000);
                parts = parts && a.same(one) && b.same(one) && one.contexts() > 0;
            }
        if (!same || !norm || !parts) {
            std::cerr << "selftest: byte Kneser-Ney (BKN) differs from token_model.hpp's, does not sum to one, or its parts "
                         "change its tables\n";
            return 1;
        }
    }
    // GEN-E: the content-plan expert (content_plan.hpp). The entity rule's order, dates to years, the holdout and
    // list exclusions, the relation order, alignment and its tie rule, and the "expressed" test.
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / ("cyphalm_plan_selftest_" + std::to_string(std::random_device{}()));
        fs::create_directories(dir);
        {
            std::ofstream(dir / "pages.tsv") << "Agamemnon\t1000\nMycenae\t5000\n";
            std::ofstream(dir / "facts_f2.tsv")
                << "subject\trelation\tobject\tP\tgrade\tn_evidence\tbest_pattern\tsource_page\tbyte_offset\n"
                << "Agamemnon\tbirth_place\tMycenae\t0.9\tLow\t1\tp\tAgamemnon\t1010\n"
                << "Agamemnon\tdeath_date\t1150-03-02\t0.9\tLow\t1\tp\tAgamemnon\t1005\n";
            std::ofstream(dir / "seeds.tsv") << "subject\trelation\tobject\tsource\ttemplate\tbyte_offset\n"
                                             << "Agamemnon\tspouse\tClytemnestra\twikidata\twikidata\t0\n"
                                             << "Agamemnon\tspouse\tOther\twikidata\twikidata\t0\n"
                                             << "Agamemnon\tspouse\tThird\twikidata\twikidata\t0\n"
                                             << "Agamemnon\tcountry\tGreece\tinfobox\tx\t20500\n"
                                             << "Agamemnon\tfounded\t0944\twikidata\twikidata\t0\n";
        }
        ContentPlanStore st;
        std::string err;
        bool ok = st.load(dir.string(), 20000, 30000, err) && err.empty();
        fs::remove_all(dir);
        const std::map<std::string, std::vector<std::string>> fr = {
            {"birth_place", {"\x01 was born in [[\x02]],"}}, {"death_year", {"\x01 (died \x02)"}},
            {"spouse", {"\x01 married \x02"}}, {"country", {"\x01 is in \x02"}}, {"founded", {"founded in \x02"}}};
        const auto cand = ContentPlanStore::candidates("'''Ahmed III''' (in [[Arabic language|Arabic]]) ruled the Ottoman Empire.");
        ok = ok && cand.size() >= 4 && cand[0] == "Ahmed III" && cand[1] == "Arabic language" &&
             std::find(cand.begin(), cand.end(), "Ottoman Empire") != cand.end();
        ok = ok && ContentPlanStore::canon("birth_date", "1673-12-30") == std::make_pair(std::string("birth_year"), std::string("1673")) &&
             ContentPlanStore::canon("founded", "0944").second == "944";
        const auto pl = st.plan("Agamemnon gathered together the Greek forces to sail for Troy.", fr);
        // country (offset 20500) is in the holdout [20000, 30000); spouse has 3 values; birth_place and death_year come
        // from the text (F2), by their position in the page: death_year (5 bytes in) before birth_place (10 bytes in)
        ok = ok && pl.entity == "Agamemnon" && pl.facts.size() == 3 && pl.facts[0].rel == "death_year" &&
             pl.facts[0].value == "1150" && pl.facts[1].rel == "birth_place" && pl.facts[2].rel == "founded" &&
             pl.facts[2].value == "944" && pl.real[1].size() == 1 && pl.real[1][0] == "Agamemnon was born in [[Mycenae]],";
        auto pred = [&](const std::string& text, const std::vector<int>& act, double* P) {
            return ContentPlanStore::predict(pl, act, reinterpret_cast<const std::uint8_t*>(text.data()), text.size(), P);
        };
        double P[256];
        const int L = pred("And then he was born in [[", {0, 1, 2}, P);
        ok = ok && L == 15 && P['M'] == 1.0;
        ok = ok && pred("nothing like it xyz", {0, 1, 2}, P) == 0 && *std::max_element(P, P + 256) == 0.0;
        ok = ok && pred("he was born in [[", {0, 2}, P) == 0;  // birth_place expressed: not active
        // tie: both facts align " in " (4 bytes); the earlier fact in the plan predicts
        ContentPlanStore::Plan tp;
        tp.facts = {{"a", "x", true}, {"b", "y", true}};
        tp.real = {{"go in Q"}, {"be in R"}};
        const std::string t = "xx in ";
        ok = ok && ContentPlanStore::predict(tp, {0, 1}, reinterpret_cast<const std::uint8_t*>(t.data()), t.size(), P) == 4 &&
             P['Q'] == 1.0 && P['R'] == 0.0;
        ok = ok && ContentPlanStore::predict(tp, {1, 0}, reinterpret_cast<const std::uint8_t*>(t.data()), t.size(), P) == 4 &&
             P['R'] == 1.0;
        ok = ok && ContentPlanStore::expressed(pl, 1, ContentPlanStore::fold("he was born in [[Mycenae]] and")) &&
             !ContentPlanStore::expressed(pl, 1, ContentPlanStore::fold("he was born in [[Myc"));
        if (!ok) {
            std::cerr << "selftest: content-plan expert (GEN-E) checks failed " << err << "\n";
            return 1;
        }
    }
    {  // GEN-C: the document experts on a toy corpus (they print nothing unless they fail)
        std::string corpus;
        for (int d = 0; d < 40; ++d) {
            corpus += "<page> ";
            corpus += d % 2 == 0 ? "Agamemnon sailed to Troy with Achilles and the Greek army. Troy fell. "
                                 : "The pianist played Chopin and Liszt in Vienna. Vienna loved Chopin. ";
            corpus += "the of and a to in is ";
        }
        DocTopic::Params dp;
        dp.stop_top = 8, dp.min_df = 2, dp.min_co = 2;
        DocTopic toy(dp);
        toy.build(reinterpret_cast<const std::uint8_t*>(corpus.data()), corpus.size());
        bool ach = false, cho = false;
        for (const auto& [wd, x] : toy.triggers("troy")) ach |= wd == "achilles", cho |= wd == "chopin";
        bool back = false;  // a trigger pair works both ways
        for (const auto& [wd, x] : toy.triggers("achilles")) back |= wd == "troy";
        ach = ach && back;
        DocTopic::Stream st(toy);
        double P[256];
        bool okd = toy.docs() == 40 && ach && !cho && st.predict(DocTopic::kTrig, P) == 0;
        const std::string pr = "<page> Agamemnon ";
        st.feed(reinterpret_cast<const std::uint8_t*>(pr.data()), pr.size());
        const int b = st.predict(DocTopic::kTrig, P);
        double sum = 0;
        for (double x : P) sum += x;
        okd = okd && b > 0 && std::fabs(sum - 1) < 1e-9 && P['T'] > 0 && P['C'] == 0;  // Troy, never Chopin
        st.feed(reinterpret_cast<const std::uint8_t*>("Tr"), 2);
        okd = okd && st.predict(DocTopic::kTrig, P) > 0 && P['o'] > 0.99;  // "Tr" goes on as "Troy"
        st.feed(reinterpret_cast<const std::uint8_t*>(" <page> "), 8);
        okd = okd && st.predict(DocTopic::kTrig, P) == 0 && st.predict(DocTopic::kCache, P) == 0;  // a new page forgets
        DocTopic::Stream fz(toy);  // frozen: words after freeze() change nothing
        const std::string p2 = "<page> Agamemnon ", p3 = "Chopin Liszt Vienna ";
        fz.feed(reinterpret_cast<const std::uint8_t*>(p2.data()), p2.size());
        fz.freeze();
        fz.feed(reinterpret_cast<const std::uint8_t*>(p3.data()), p3.size());
        okd = okd && fz.predict(DocTopic::kTrig, P) > 0 && P['L'] == 0 && P['T'] > 0 &&
              fz.predict(DocTopic::kCache, P) > 0 && P['C'] == 0 && P['A'] > 0.99;
        {  // GEN-C2: mix_doc_into (context8k, qa) gives fit_doc's numbers, trig and cache on, over a toy mixture
            const auto* cb = reinterpret_cast<const std::uint8_t*>(corpus.data());
            auto base = [](std::size_t k) { return 0.02 + 0.4 * static_cast<double>((k * 2654435761u) % 1000) / 1000.0; };
            std::vector<double> c1(corpus.size());
            for (std::size_t k = 0; k < c1.size(); ++k) c1[k] = base(k);
            std::vector<double> c2 = c1;
            const std::size_t hf = c1.size() / 2;
            fit_doc(toy, cb, 0, hf, c1);
            mix_doc_into(toy, cb, 0, hf, c2);
            double dmax = 0;
            std::size_t moved = 0;
            for (std::size_t k = 0; k < c1.size(); ++k) {
                dmax = std::max(dmax, std::fabs(c1[k] - c2[k]));
                moved += c2[k] != base(k);
            }
            okd = okd && dmax < 1e-15 && moved > 0;
        }
        if (!okd) {
            std::cerr << "selftest: document experts (GEN-C) wrong on the toy corpus\n";
            return 1;
        }
    }
    std::cout << "selftest ok\n";
    return 0;
}

// ---------------------------------------------------------------- surprise (S4b)

// surprise <wiki> --out SCORES.tsv [--plant PAGES.xml] [--part K/N] (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b"):
// CyphaLM's surprise per page, for Cypha's unknown-unknown signals (Rule 10: it watches what Cypha reads; nothing here
// answers a question). The default expert stack trained on the wiki train bytes only (no chat): the packed trie,
// InfiniGram (cap 8,192), the class expert, the similar-context expert and copy (m = --copy-min), each one's lambdas
// fit on the whole holdout [train, sample) in qa's order. Each page is read in a fresh context (only the dump's
// `<text xml:space="preserve">` before it); its first <= 2,048 body bytes are scored. Pages (pattern_learn's list_pages
// rule: main namespace, not a redirect, body >= 256 bytes): C = bodies in [9,437,184, 30,000,000) and S = [30,000,000,
// 50,000,000) without the "List of" and disambiguation pages, L = those pages from 9,437,184 on, D = every page of
// --plant. --part K/N scores every N-th page from the K-th. Rows: set, body offset, title, body bytes, scored bytes,
// bits, bits per byte.
struct SignalPage {
    std::string set, title;
    std::size_t tb = 0, te = 0;
    const std::string* src = nullptr;
};
void signal_pages(const std::string& d, bool plant, std::vector<SignalPage>& out) {
    for (std::size_t pos = 0;;) {
        const std::size_t pb = d.find("<page>", pos);
        if (pb == std::string::npos) break;
        const std::size_t pe = d.find("</page>", pb);
        if (pe == std::string::npos) break;
        pos = pe + 7;
        const std::size_t tb = d.find("<title>", pb), te = d.find("</title>", tb);
        if (tb == std::string::npos || te == std::string::npos || te > pe) continue;
        const std::string title = d.substr(tb + 7, te - tb - 7);
        const std::size_t xb = d.find("<text", te);
        if (xb == std::string::npos || xb > pe) continue;
        const std::size_t xg = d.find('>', xb);
        if (xg == std::string::npos || xg > pe || d[xg - 1] == '/') continue;
        const std::size_t xe = d.find("</text>", xg);
        if (xe == std::string::npos || xe > pe) continue;
        std::string head = d.substr(xg + 1, std::min<std::size_t>(40, xe - xg - 1));
        head.erase(0, head.find_first_not_of(" \t\r\n"));
        if (lowered(head.substr(0, 9)) == "#redirect") continue;
        const std::size_t colon = title.find(':');
        if (colon != std::string::npos && colon + 1 < title.size() && title[colon + 1] != ' ') continue;  // namespace page
        if (xe - (xg + 1) < 256) continue;
        const std::size_t b = xg + 1;
        const bool skip = cypha::eval::signal_store_skips(title);
        std::string set = plant ? "D"
                          : skip ? (b >= cypha::eval::kSignalCalibFrom ? "L" : "")
                          : b >= cypha::eval::kSignalCalibFrom && b < cypha::eval::kSignalSplit ? "C"
                          : b >= cypha::eval::kSignalSplit && b < cypha::eval::kSignalTestTo ? "S" : "";
        if (!set.empty()) out.push_back({set, title, b, xe, &d});
    }
}

int cmd_surprise(const Args& a) {
    const Config& cfg = a.cfg;
    int part_k = 0, part_n = 1;
    if (a.out.empty() || (!a.part.empty() && (std::sscanf(a.part.c_str(), "%d/%d", &part_k, &part_n) != 2 || part_n < 1 ||
                                               part_k < 0 || part_k >= part_n))) {
        std::cerr << "usage: cyphalm_trie surprise <wiki> --out SCORES.tsv [--plant PAGES.xml] [--part K/N]\n";
        return 2;
    }
    std::string wiki, plant;
    auto slurp = [](const std::string& p, std::string& s) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        return true;
    };
    if (!slurp(a.wiki, wiki) || (!a.plant.empty() && !slurp(a.plant, plant))) {
        std::cerr << "cannot read " << a.wiki << (a.plant.empty() ? "" : " or " + a.plant) << "\n";
        return 1;
    }
    std::vector<SignalPage> pages;
    signal_pages(wiki, false, pages);
    if (!a.plant.empty()) signal_pages(plant, true, pages);
    std::map<std::string, int> per_set;
    for (const auto& p : pages) ++per_set[p.set];
    for (const auto& [s, n] : per_set) std::cerr << "surprise pages " << s << " " << n << "\n";

    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);
    model.compact_match();
    const PackedModel pm = pack(model);  // M4
    model.release();
    const FmInfiniGram ig(data.data(), cfg.train);
    int rc = 0;
    const std::unique_ptr<ClassContext> cc = default_cls(a, data.data(), cfg.train, rc);
    if (rc) return rc;
    const std::unique_ptr<SimilarContext> sc = default_sim(a, data.data(), cfg.train, rc);
    if (rc) return rc;

    // The lambdas, on the whole holdout read as one stream (qa's order: InfiniGram, class, similar-context, copy).
    const std::size_t H = cfg.sample - cfg.train;
    std::vector<double> pt(H), pi(H);
    std::vector<int> bi(H);
    {
        FmInfiniGram::Cursor cur;
        int prev = -1;
        for (std::size_t k = 0; k < H; ++k) {
            const std::size_t i = cfg.train + k;
            double e[5][256], M[256];
            bool em = false;
            unsigned pred = 0;
            pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
            mix_dist(w, e, em, pred, M);
            pt[k] = M[data[i]];
            const std::size_t len = std::min<std::size_t>(i, 8192);
            const auto r = ig.query(data.data() + (i - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192), 1, cur);
            prev = r.n;
            pi[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
            bi[k] = bucket(r.n, r.total);
        }
    }
    double lig[kBuckets];
    fit_lambdas(pi, pt, bi, H, kBuckets, lig);
    std::vector<double> base(H);
    for (std::size_t k = 0; k < H; ++k) base[k] = lig[bi[k]] * pi[k] + (1 - lig[bi[k]]) * pt[k];
    auto line = [&](const char* name) {
        double bits = 0;
        for (double p : base) bits -= std::log2(p);
        std::cout << "surprise_holdout_" << name << "_bpc " << bits / static_cast<double>(H) << "  (lambdas fit on it)\n";
    };
    line("trie_ig");
    std::vector<double> lcl, lsm, lcp;
    if (cc) lcl = mix_cls_into(*cc, data.data(), cfg.train, H, base), line("cls");
    if (sc) {
        std::vector<double> ps(H);
        std::vector<int> sb(H);
        for (std::size_t k = 0; k < H; ++k) {
            double P[256];
            sb[k] = SimilarContext::bucket(sc->predict(data.data(), cfg.train + k, P));
            ps[k] = P[data[cfg.train + k]];
        }
        lsm = mix_into(ps, sb, H, SimilarContext::kBuckets, base);
        line("sim");
    }
    if (!a.no_copy) lcp = mix_copy_into(data.data(), cfg.train, H, a.copy_min, base), line("copy");
    std::cout.flush();

    std::ofstream out(a.out);
    if (!out) {
        std::cerr << "cannot write " << a.out << "\n";
        return 1;
    }
    out << "set\tbody_offset\ttitle\tbody_bytes\tscored_bytes\tbits\tbpb\n";
    static const std::string kPrefix = "<text xml:space=\"preserve\">";
    std::size_t scored = 0, bytes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t u = static_cast<std::size_t>(part_k); u < pages.size(); u += static_cast<std::size_t>(part_n)) {
        const SignalPage& pg = pages[u];
        const std::size_t n = std::min<std::size_t>(pg.te - pg.tb, 2048);
        std::vector<std::uint8_t> buf(kPrefix.begin(), kPrefix.end());
        buf.insert(buf.end(), pg.src->begin() + static_cast<std::ptrdiff_t>(pg.tb),
                   pg.src->begin() + static_cast<std::ptrdiff_t>(pg.tb + n));
        FmInfiniGram::Cursor cur;
        int prev = -1;
        double bits = 0;
        for (std::size_t i = kPrefix.size(); i < buf.size(); ++i) {
            const std::uint8_t y = buf[i];
            double e[5][256], M[256], P[256];
            bool em = false;
            unsigned pred = 0;
            pm.fill_experts(buf.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
            mix_dist(w, e, em, pred, M);
            double p = M[y];
            const std::size_t len = std::min<std::size_t>(i, 8192);
            const auto r = ig.query(buf.data() + (i - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192), 1, cur);
            prev = r.n;
            const int b = bucket(r.n, r.total);
            p = lig[b] * (r.total ? static_cast<double>(r.count[y]) / static_cast<double>(r.total) : 0.0) + (1 - lig[b]) * p;
            if (cc) {
                const auto c = static_cast<std::size_t>(ClassContext::bucket(cc->predict(buf.data(), i, P)));
                p = lcl[c] * P[y] + (1 - lcl[c]) * p;
            }
            if (sc) {
                const auto s = static_cast<std::size_t>(SimilarContext::bucket(sc->predict(buf.data(), i, P)));
                p = lsm[s] * P[y] + (1 - lsm[s]) * p;
            }
            if (!a.no_copy) {
                int m = 0;
                const double pc = copy_pred(buf.data(), i, m) == y ? 1.0 : 0.0;
                const auto c = static_cast<std::size_t>(copy_bucket(m));
                p = lcp[c] * pc + (1 - lcp[c]) * p;
            }
            bits -= std::log2(p);
        }
        char row[96];
        std::snprintf(row, sizeof row, "\t%zu\t%zu\t%.3f\t%.6f\n", pg.te - pg.tb, n, bits, bits / static_cast<double>(n));
        out << pg.set << '\t' << pg.tb << '\t' << pg.title << row;
        ++scored, bytes += n;
        if (scored % 200 == 0) {
            out.flush();
            std::cerr << "surprise scored " << scored << " pages " << bytes << " bytes "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
        }
    }
    std::cout << "surprise pages_scored " << scored << " bytes " << bytes << " part " << part_k << "/" << part_n << "\n";
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: cyphalm_trie <bpc|chat|probe|context8k|sample|cls|qa|dktext|simvec|surprise|selftest> [wiki] [dolly] [options]\n"
                 "  bpc [wiki] [--negative] [--no-guard]\n"
                 "  chat [wiki] [dolly] [--prompt TEXT] [--no-repeat] [--save FILE | --model FILE]\n"
                 "  probe [wiki] [dolly]\n"
                 "  context8k [wiki] [dolly]\n"
                 "  sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600] [--save FILE | --model FILE]\n"
                 "  cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs] [--sim SPEC ...] [--sim-fresh] [--sim-surface]\n"
                 "      [--sim-stream FROM:LEN] [--sim-bits PREFIX] [--sim-growth MIB,MIB,...]   (A4e)\n"
                 "      cls <wiki> shape --doc V1;V2;... [--sim-stream FROM:LEN]   (GEN-C2)\n"
                 "  bkn <wiki> K[:m:c:d] [more specs] [--bkn-fresh] [--bkn-stack]   (BKN: packed byte Kneser-Ney)\n"
                 "  simpair [--margin M] REF.f64 FILE.f64 ...   (A4e: cls --sim-bits files paired against REF)\n"
                 "  SPEC: [default,][vec=PREFIX|FILE][,W=8][,k=32][,T=0.05][,min=-1][,sif=1][,dim=64][,drop=1][,lists=512][,probe=8][,tail=32]\n"
                 "        [,src=all|wiki][,gate=1][,mode=knn|dedupe][,online=0]   (default = %s; vectors from $CYPHA_SIMVEC, else %s)\n"
                 "  qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--leak TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]\n"
                 "     [--dk DK.tsv --dk-ratio R] [--fam none|before|after] [--parse E1.tsv] [--fam-stat NAME]\n"
                 "     [--fam-train stat|outcome|low] [--fam-only] [--copy] [--copy-min N] [--r3 PATTERNS.tsv]\n"
                 "     [--fam-ent FRAMES] [--copy-dk] [--fam-read] [--dk-plain P] [--fam-ent-at all|answer|start] [--copy-frame]\n"
                 "     [--shard F.tsv] [--shard2 F.tsv] [--cls TABLE|shape[:tail[:orders[:flags]]]] [--sim SPEC] [--cards FILE]\n"
                 "     [--save DIR | --model DIR]\n"
                 "     (class, similar-context and copy experts are defaults, off inside answers: --no-cls --no-sim --no-copy)\n"
                 "  dktext [wiki] [dolly] --dev DEV.tsv --test TEST.tsv --out DK.tsv [--from 8388608] [--max-rows N] [--mention]\n"
                 "  simvec [wiki] --sim SPEC --out FILE\n"
                 "  surprise <wiki> --out SCORES.tsv [--plant PAGES.xml] [--part K/N]   (S4b: per-page bits per byte)\n"
                 "  selftest\n"
                 "  options: --wiki P --dolly P --train N --sample N; qa, sample, context8k: --sim SPEC | --no-sim,\n"
                 "           --cls SPEC | --no-cls (sample: no class expert unless --cls), --copy-min N (default 8) | --no-copy,\n"
                 "           --bkn K[:m:c:d] [--bkn-way a|b] (byte Kneser-Ney; default 6:4:2:1, way b) | --no-bkn,\n"
                 "           --bkn-answers on|off (qa, context8k: Kneser-Ney inside answers, or off there as the experts)\n"
                 "  paths default to $CYPHA_WIKI / $CYPHA_DOLLY, then %s and %s\n",
                 kSimDefault, CYPHA_SIMVEC_DEFAULT, kDefaultWiki, kDefaultDolly);
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "selftest") return cmd_selftest();
    if (cmd == "simpair") return cmd_simpair(std::vector<std::string>(argv + 2, argv + argc));  // A4e
    Args a = parse(argc, argv, 2);
    if (!a.ok) {
        usage();
        return 2;
    }
    if (cmd != "bpc" && cmd != "chat" && cmd != "probe" && cmd != "context8k" && cmd != "sample" && cmd != "qa" &&
        cmd != "dktext" && cmd != "cls" && cmd != "simvec" && cmd != "surprise" && cmd != "bkn") {
        usage();
        return 2;
    }
    // BKN2: byte Kneser-Ney is a default of qa, sample and context8k; --no-bkn leaves it out.
    const bool kn_cmd = cmd == "qa" || cmd == "sample" || cmd == "context8k";
    if (a.no_bkn && (!kn_cmd || !a.bkn.empty())) {
        std::cerr << "--no-bkn is for qa, sample and context8k, without --bkn\n";
        return 2;
    }
    if (kn_cmd && !a.no_bkn && a.bkn.empty()) a.bkn = kBknDefault;
    if (!a.bkn.empty() && (!kn_cmd || (a.bkn_way != "a" && a.bkn_way != "b") || bkn_spec(a.bkn).empty())) {
        std::cerr << "--bkn K[:m:c:d] is for qa, sample and context8k, with --bkn-way a (replace) or b (add)\n";  // BKN
        return 2;
    }
    if (a.bkn_answers != "on" && a.bkn_answers != "off") {  // BKN3
        std::cerr << "--bkn-answers on|off (Kneser-Ney inside answers: on, BKN2; off, the experts' gate)\n";
        return 2;
    }
    if ((!a.model.empty() || !a.save.empty()) && cmd != "chat" && cmd != "sample" && cmd != "qa") {
        std::cerr << "--model and --save are for chat, sample and qa\n";
        return 2;
    }
    if (!self_test()) return 1;
    if ((cmd == "qa" || cmd == "sample" || cmd == "context8k" || cmd == "surprise") && !a.no_sim) {  // A4: find the vectors first
        const SimSpec ss = parse_sim(a.sims.empty() ? "default" : a.sims[0]);
        if (!ss.ok || !(std::ifstream(ss.vec, std::ios::binary) || std::ifstream(ss.vec + ".vectors.f32", std::ios::binary))) {
            std::cerr << "similar-context expert: no vectors at " << ss.vec
                      << " (run from the repository root, set CYPHA_SIMVEC, or pass --no-sim)\n";
            return 1;
        }
    }
    if (cmd == "qa" || cmd == "sample" || cmd == "context8k" || cmd == "surprise") {  // B4C: the class spec is read first
        if (a.no_cls && !a.cls.empty()) {
            std::cerr << "--cls and --no-cls together\n";
            return 2;
        }
        if (!default_cls_spec(a).empty() && !make_class_context(default_cls_spec(a))) return 2;
    }
    const auto t0 = std::chrono::steady_clock::now();
    int rc = 2;
    if (cmd == "bpc") rc = cmd_bpc(a);
    else if (cmd == "chat") rc = cmd_chat(a);
    else if (cmd == "probe") rc = cmd_probe(a);
    else if (cmd == "context8k") rc = cmd_context8k(a);
    else if (cmd == "sample") rc = cmd_sample(a);
    else if (cmd == "cls") rc = cmd_cls(a);
    else if (cmd == "bkn") rc = cmd_bkn(a);  // BKN
    else if (cmd == "simvec") rc = cmd_simvec(a);
    else if (cmd == "surprise") rc = cmd_surprise(a);  // S4b
    else if (cmd == "qa") {
        if (a.dev.empty() || (a.fam != "none" && a.fam != "before" && a.fam != "after") || a.dk_ratio < 0 ||
            a.dk_ratio >= 1 || (a.fam_train != "stat" && a.fam_train != "outcome" && a.fam_train != "low") ||
            ((!a.fam_ent.empty() || a.copy_dk) && (!a.model.empty() || !a.save.empty()))) {  // ANS: not stored
            usage();
            return 2;
        }
        if (a.dk_plain < 0 || a.dk_plain >= 1 || (a.fam_ent_at != "all" && a.fam_ent_at != "answer" && a.fam_ent_at != "start") ||
            (a.fam_ent_at != "all" && a.fam_ent.empty()) || (a.copy_frame && !a.copy_dk) || (a.dk_plain > 0 && a.fam == "none") ||
            ((a.dk_plain > 0 || a.fam_ent_at != "all" || a.copy_frame) && (!a.model.empty() || !a.save.empty())) ||
            (a.fam_ent_at == "start" && !a.cards.empty())) {  // DK2: training text, not stored; start frames not in cards
            usage();
            return 2;
        }
        rc = cmd_qa(a, QaOpts{a.dev, a.test, a.out, a.base, a.dk, a.dk_ratio, a.fam, a.parse, a.fam_stat, a.fam_train,
                              a.r3, a.shard, a.shard2, a.leak, a.copy, a.fam_only, default_cls_spec(a), a.copy_min,
                              a.no_sim ? std::string() : a.sims.empty() ? std::string("default") : a.sims[0], !a.no_copy,
                              a.fam_ent, a.copy_dk, a.fam_read, a.dk_plain, a.fam_ent_at, a.copy_frame});
    } else if (cmd == "dktext") {
        if (a.dev.empty() || a.test.empty() || a.out.empty()) {
            usage();
            return 2;
        }
        rc = cmd_dktext(a, a.dev, a.test, a.out, a.from, a.max_rows, a.mention);
    }
    std::cerr << "seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return rc;
}
