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
//        caps, matched-length histogram, 50 questions with 30 Dolly turns held out.
//   sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600] [--save FILE | --model FILE]
//        trie_sample: wiki-only trie, 3 temperatures, no-repeat rule, files for `scribe ladder --gen-trie`.
//        chat and sample: --save FILE stores the trained packed trie with its weights and split (RAM_PROFILE M7);
//        --model FILE serves that file without training (chat then prints no mix_stored: it measures the training form).
//   cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs] [--sim SPEC ...] [--sim-fresh] [--sim-surface]
//        B4 (docs/reports/B4_CLASS_CONTEXT_EXPERT.md): the class-context expert (class_context.hpp) over the
//        bpc model, each spec paired against the model without it: holdout second-half BPC (lambdas on the
//        first half), fresh-text BPC on [sample, sample + 1 MiB) (lambdas on the whole holdout), and surface S
//        of 400-byte samples from 200 holdout prompts, with block-bootstrap intervals; later specs are also
//        compared with the first. TABLE is word<TAB>class (e.g. E1 tags); "shape" is shape classes only.
//        A4 (docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md): --sim adds the similar-context expert (similar_context.hpp) instead
//        of the class loop: each SPEC over trie + InfiniGram and over trie + InfiniGram + the first class spec, holdout
//        second half (lambdas on the first), the index's recall against an exact search, build time, table bytes and
//        query latency; --sim-fresh adds the fresh text (lambdas on the whole holdout), --sim-surface surface S of
//        trie + sim and trie + class + sim. SPEC is vec=PREFIX (PREFIX.vectors.f32 and PREFIX.vocab.tsv, as ppmi_svd
//        writes them) plus W, k, T, min, sif, dim, drop, lists, probe, tail (see parse_sim).
//        --out FILE writes the samples (systems trie, trie+cls:SPEC) and the human continuations (human) as JSON
//        lines for the SCRIBE model ladder.
//   qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--leak TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]
//      [--dk DK.tsv --dk-ratio R] [--fam none|before|after] [--parse E1.tsv] [--fam-stat NAME]
//      [--fam-train stat|outcome|low] [--fam-only] [--copy] [--copy-min N] [--r3 PATTERNS.tsv]
//      [--shard F.tsv] [--shard2 F.tsv] [--cls TABLE|shape[:tail[:orders[:flags]]]] [--sim SPEC]
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
//        --copy adds the copy expert (a match over the current context, lambdas per match length fit on the
//        holdout's first half; BPC and LCI gain on the second half) as two more models. --r3 runs statement
//        scoring with F2's patterns as frames. --shard / --shard2 add question-form fact turns (rows from the
//        BPC holdout dropped; exits if a shard question equals a dev or test question, from --test or --leak).
//        --cls (B4; TOOL_PROFILE P2) adds the class-context expert, built on the same bytes as InfiniGram, to every
//        model (lambdas on the holdout's first half); with --copy the copy expert goes on after it, its lambdas fit
//        over the class-mixed distribution, and cls_given_copy_lci gives the class expert's gain with copy present.
//        --copy-min N (B4B) sets the copy lambdas of matches shorter than N bytes to 0 after fitting.
//        A1d / A2b (docs/reports/A1D_A2B_ANSWERING_VARIANTS.md): the --parse entity column may hold several
//        candidates ("a|b|c", E1's first): per statistic the best candidate's value counts, and the token's entity is
//        the best by --fam-stat (ENTITY: its accuracy against the source title, grading only). ent_tb is the larger of
//        the bold-lead and title counts; FAMSTAT adds the paired-bootstrap difference from ent_bold, FAMTOK the
//        token's implied precision / recall. --fam-train low is outcome, but answered turns and high questions carry
//        no token. --r3 reads a 12th column `frame` (F2R's raw-byte frames, markup kept) when present.
//        A4: --sim SPEC adds the similar-context expert to every model, after the class expert and before the copy
//        expert (src=all: keys over InfiniGram's bytes, wiki train + chat; src=wiki: wiki train only), lambdas on the
//        holdout's first half; SIM_NB prints the three nearest stored contexts at each dev chat prompt, LATENCY the
//        decode time a question.
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
//   --cls + --copy  holdout2 trie + IG 2.071574, + shape:6 1.995847, + copy(m 8) 1.985451, both 1.914210;
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

#include "cypha/cyphalm/class_context.hpp"
#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/cyphalm/packed_trie.hpp"
#include "cypha/cyphalm/similar_context.hpp"
#include "cypha/eval/lci.hpp"
#include "cypha/eval/scribe.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
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
using cypha::cyphalm::FmInfiniGram;
using cypha::cyphalm::SimilarContext;

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
    int copy_min = 0;     // B4B: copy lambdas of match lengths below this set to 0 after fitting
    std::string model, save;  // M7: serve a stored packed trie (chat, sample); store the one just trained
    std::vector<std::string> sims;               // A4: similar-context expert specs (cls: each; qa: the first)
    bool sim_fresh = false, sim_surface = false;  // A4, cls: also the fresh text / surface S
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
        else if (s == "--sim-surface") a.sim_surface = true;
        else if (s == "--fam-only") a.fam_only = true;
        else if (s == "--model") a.model = next();
        else if (s == "--save") a.save = next();
        else if (s == "--mention") a.mention = true;
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

    // Index the training bytes only: wiki train + boundary + chat. No wiki holdout. The index takes T.
    std::vector<std::uint8_t> T;
    const std::size_t all = training_text(data, cfg, T), tn = T.size();
    const FmInfiniGram ig(std::move(T));
    std::cerr << "index built over " << tn << " bytes\n";

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

    const int caps[] = {4, 16, 64, 512, 8192};
    double lam8k[kBuckets] = {0};
    std::vector<int> n8k(H);
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
        if (cap == 8192) std::copy(lam, lam + kBuckets, lam8k);
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
            const std::size_t len = std::min<std::size_t>(seq.size(), 8192);
            const auto r = ig.query(seq.data() + (seq.size() - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192),
                                    1, cur);
            prev = r.n;
            if (t == 0) n0 = r.n;
            const double l = lam8k[bucket(r.n, r.total)];
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
                     "usage: cyphalm_trie sample <enwik8> <out_dev.txt> [out_fresh.txt] [--prompts 250] [--bytes 600]\n");
        return 2;
    }
    const std::size_t nfiles = a.outs.size();
    std::vector<std::uint8_t> data;
    double w[6];
    PackedModel pm;
    if (!a.model.empty() ? !load_model(a, pm, w, cfg, data) : !load_wiki(a, data)) return 1;

    std::vector<std::string> prompts;
    {
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
    std::mt19937_64 prng(20261027);
    std::shuffle(prompts.begin(), prompts.end(), prng);
    const std::size_t need = static_cast<std::size_t>(a.prompts) * nfiles;
    std::cerr << "holdout prompts " << prompts.size() << ", using " << need << "\n";
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

    const double temps[3] = {0.6, 0.8, 1.0};
    for (std::size_t f = 0; f < nfiles; ++f) {
        std::ofstream out(a.outs[f]);
        if (!out) {
            std::cerr << "cannot write " << a.outs[f] << "\n";
            return 1;
        }
        for (int k = 0; k < a.prompts; ++k) {
            const std::size_t id = f * static_cast<std::size_t>(a.prompts) + static_cast<std::size_t>(k);
            const std::string& p = prompts[id];
            out << "PROMPT\t" << id << "\t" << printable(p) << "\n";
            for (int ti = 0; ti < 3; ++ti) {
                const std::uint64_t seed = 1000003ull * (id + 1) + static_cast<std::uint64_t>(ti) * 7919ull;
                SampleOptions opt;
                opt.temperature = temps[ti];
                opt.no_repeat = true;
                const std::string text = sample_continuation(pm, w, p, cfg.train, a.bytes,
                                                              static_cast<std::uint32_t>(seed ^ (seed >> 32)), opt);
                char kind[32];
                std::snprintf(kind, sizeof kind, "trie_T%.1f", temps[ti]);
                out << "GEN\t" << kind << "\t" << printable(text) << "\n";
            }
            if ((k + 1) % 50 == 0) std::cerr << "file " << f << " prompt " << (k + 1) << "/" << a.prompts << "\n";
        }
    }
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

// Copy expert (A1c, PLAN B9): the longest suffix of s[0..i), 3 to 64 bytes, that also ends at an earlier j inside the
// last 4,096 bytes of the current context (the prompt and the answer so far, or the stream being scored). It predicts
// s[j], from the most recent of the longest matches. Returns -1 (n = 0) when there is none.
constexpr std::size_t kCopyWin = 4096;
constexpr int kCopyBuckets = 8;
int copy_pred(const std::uint8_t* s, std::size_t i, int& n) {
    n = 0;
    int pred = -1;
    const std::size_t lo = i > kCopyWin ? i - kCopyWin : 0;
    for (std::size_t j = i; j-- > lo + 1;) {
        int L = 0;
        while (L < 64 && j - lo > static_cast<std::size_t>(L) && s[j - 1 - L] == s[i - 1 - L]) ++L;
        if (L > n) {
            n = L;
            pred = s[j];
        }
    }
    if (n < 3) n = 0, pred = -1;
    return pred;
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

// B4: the class-context expert from "TABLE[:tail[:orders[:flags]]]": TABLE a word<TAB>class file, or "shape" (shape
// classes only); tail defaults to 6, orders to 2 (2 or 3); flags (RAM_PROFILE M2/M3) are c (compact table) and pN
// (contexts seen fewer than N times dropped), e.g. shape:6:3:cp2. The fields are read from the right, so a Windows path
// keeps its drive colon. Not built yet (call build()). nullptr (and a message) for a bad spec or table.
std::unique_ptr<ClassContext> make_class_context(const std::string& spec) {
    const auto f = split(spec, ':');
    auto digits = [](const std::string& x) {
        return !x.empty() && x.size() < 6 && std::all_of(x.begin(), x.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    ClassContext::Options o;
    auto flags = [&](const std::string& x) {
        bool any = false;
        for (std::size_t i = 0; i < x.size();) {
            if (x[i] == 'c' && !o.compact) o.compact = true, ++i;
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

// A4: the similar-context expert from "vec=PREFIX[,W=8][,k=32][,T=0.05][,min=-1][,sif=1][,dim=64][,drop=1][,lists=512]
// [,probe=8][,tail=32][,src=all]" (PREFIX.vectors.f32 and PREFIX.vocab.tsv; qa's src: all = wiki train + chat, wiki =
// wiki train only). `build` names what the index depends on: specs that differ only in k, T, min, probe or tail share it.
struct SimSpec {
    SimilarContext::Params p;
    std::string vec, src = "all", build;
    bool ok = false;
};
SimSpec parse_sim(const std::string& spec) {
    SimSpec s;
    for (const auto& kv : split(spec, ',')) {
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
        else return s;
    }
    s.build = s.vec + "|" + s.src + "|" + std::to_string(s.p.W) + "|" + std::to_string(s.p.sif) + "|" +
              std::to_string(s.p.dim) + "|" + std::to_string(s.p.drop) + "|" + std::to_string(s.p.lists);
    s.ok = !s.vec.empty() && (s.src == "all" || s.src == "wiki");
    return s;
}
std::unique_ptr<SimilarContext> make_similar_context(const SimSpec& s) {
    std::string err;
    auto V = SimilarContext::load(s.vec + ".vectors.f32", s.vec + ".vocab.tsv", err);
    if (!err.empty()) {
        std::cerr << err << "\n";
        return nullptr;
    }
    return std::make_unique<SimilarContext>(std::move(V), s.p);
}

// CyphaLM's next-byte distribution: the frozen trie mixture, optionally mixed with InfiniGram
// (context8k: P = lam[b] P_ctx + (1 - lam[b]) P_trie, one lam per match-length x count bucket),
// then optionally with the copy expert (P = lc[b] [pred] + (1 - lc[b]) P, one lc per match-length bucket).
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
    // hint: the previous step's match length (-1 at a new sequence); n_out: this step's.
    void dist(const std::vector<std::uint8_t>& seq, int& hint, double* P, int* n_out = nullptr) const {
        double e[5][256];
        bool m = false;
        unsigned pred = 0;
        if (packed) packed->fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, m, pred, online_match);
        else model->fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, m, pred, online_match);
        mix_dist(w, e, m, pred, P);
        if (ig) {
            const std::size_t len = std::min<std::size_t>(seq.size(), static_cast<std::size_t>(cap));
            const auto r =
                ig->query(seq.data() + (seq.size() - len), len, cap, hint < 0 ? -1 : std::min(hint + 1, cap), 1, cur);
            hint = r.n;
            if (n_out) *n_out = r.n;
            const double l = lam[bucket(r.n, r.total)];
            for (int s = 0; s < 256; ++s) {
                const double pc = r.total ? static_cast<double>(r.count[s]) / static_cast<double>(r.total) : 0.0;
                P[s] = l * pc + (1 - l) * P[s];
            }
        }
        if (cc) {
            double Pc[256];
            const double l = lcls[ClassContext::bucket(cc->predict(seq.data(), seq.size(), Pc))];
            for (int s = 0; s < 256; ++s) P[s] = l * Pc[s] + (1 - l) * P[s];
        }
        if (sc) {
            double Ps[256];
            const double l = lsim[SimilarContext::bucket(sc->predict(seq.data(), seq.size(), Ps))];
            for (int s = 0; s < 256; ++s) P[s] = l * Ps[s] + (1 - l) * P[s];
        }
        int n = 0;
        const int c = lc ? copy_pred(seq.data(), seq.size(), n) : -1;
        if (c < 0) return;
        const double l = lc[copy_bucket(n)];
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

std::string sample_with_class(const PackedModel& model, const double w[6], std::size_t limit, const ClassContext* cc,
                              const double* lam, const std::string& prompt, int n, std::uint32_t seed,
                              const SimilarContext* sc = nullptr, const double* lsm = nullptr) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < n; ++t) {
        double e[5][256], M[256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(seq.data(), seq.size(), limit, Mode::Synth, e, matched, pred, false);
        mix_dist(w, e, matched, pred, M);
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
        bool banned[256];
        mark_banned(seq, banned);
        double p[256], mass = 0;
        for (int pass = 0; pass < 2 && !(mass > 0); ++pass) {
            mass = 0;
            for (int s = 0; s < 256; ++s) mass += p[s] = (pass == 0 && banned[s]) ? 0.0 : std::pow(M[s], 1.0 / 0.8);
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
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

int cmd_cls(const Args& a) {
    const Config& cfg = a.cfg;
    if (a.pos.size() < 2) {
        std::fprintf(stderr, "usage: cyphalm_trie cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs]\n");
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
        std::vector<double> first_hc, first_fc;  // the first spec's reads, for paired comparisons
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
            const std::size_t M = a.sim_fresh ? N : H;
            std::vector<double> ps(N, 0.0);
            std::vector<int> sb(N, 0);
            std::size_t live = 0;
            const auto t1 = std::chrono::steady_clock::now();
            for (std::size_t k = 0; k < M; ++k) {
                double P[256];
                const auto hit = sc->predict(data.data(), cfg.train + k, P);
                sb[k] = SimilarContext::bucket(hit);
                ps[k] = P[data[cfg.train + k]];
                live += hit.live > 0;
            }
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
            if (a.sim_fresh) {
                double fs_i[kS], fs_c[kS];
                fit_lambdas(ps, pif, sb, H, kS, fs_i);
                fit_lambdas(ps, pcf, sb, H, kS, fs_c);
                const auto fi = mixed(pif, fs_i, H, N), fc = mixed(pcf, fs_c, H, N);
                std::printf("sim %s fresh trie+ig+sim %.6f trie+ig+cls+sim %.6f\n", spec.c_str(), mean(fi), mean(fc));
                print_boot("sim " + spec + " fresh_gain_over_trie_ig", block_boot(gain(fi, f_i), 4096, 4000, 11));
                print_boot("sim " + spec + " fresh_gain_over_trie_ig_cls", block_boot(gain(fc, f_c), 4096, 4000, 11));
                if (first_fc.empty()) first_fc = fc;
                else print_boot("sim " + spec + " fresh_trie_ig_cls_vs_first_sim", block_boot(gain(fc, first_fc), 4096, 4000, 11));
            }
            if (a.sim_surface) {
                double ls_t[kS], ls_tc[kS];
                fit_lambdas(ps, pt, sb, half, kS, ls_t);
                fit_lambdas(ps, ptc, sb, half, kS, ls_tc);
                const auto sv = surface(nullptr, nullptr, "trie+sim:" + spec, sc.get(), ls_t),
                           svc = surface(cc.get(), lt_h, "trie+cls+sim:" + spec, sc.get(), ls_tc);
                std::printf("sim %s surface trie+sim %.4f trie+cls+sim %.4f\n", spec.c_str(), mean(sv), mean(svc));
                print_boot("sim " + spec + " surface_gain_over_trie", block_boot(gain(sv, s_trie), 1, 4000, 5));
                print_boot("sim " + spec + " surface_gain_over_trie_cls", block_boot(gain(svc, s_cls), 1, 4000, 5));
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
Answer decode(const Lm& lm, const std::string& prompt, int max_bytes = 80, std::uint32_t seed = 0) {
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
        ban_repeats(seq, lm.lc ? prompt.size() : 0, banned);
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

int cmd_qa(const Args& a, const QaOpts& o) {
    const Config& cfg = a.cfg;
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
    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    train_span(model, data, w, 0, cfg.train, cfg);
    std::cerr << "wiki trained, nodes " << model.nodes.size() << "\n";

    std::vector<Turn> turns;
    std::string dolly_text = load_dolly(a.dolly, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return 1;
    }
    auto instruction = [&](std::size_t i) {  // "User: <instruction>\n..."
        const std::string& p = turns[i].prompt;
        return p.substr(6, p.find('\n') - 6);
    };
    // The model state after the wiki pass, before any chat (the match expert does not fire on a separate buffer).
    double wf[6];
    std::copy(w, w + 6, wf);
    Lm wiki_lm{&model, wf, cfg.train, true};
    wiki_lmp = &wiki_lm;

    // A1c: entity familiarity, from CyphaLM's wiki trie and its lowered training bytes (wiki train + Dolly).
    std::array<double, kFam> fam_cut;
    fam_cut.fill(std::nan(""));
    if (parse.empty()) std::string().swap(dolly_text);  // only A1c's lowered text reads it
    if (!parse.empty()) {
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
        if (o.fam_only) return 0;
    }

    // Familiarity token: A1b (question_lp; cuts from the Dolly and don't-know questions) or A1c (an entity statistic;
    // low below the calibrated cut, high above, mid when E1 found no entity).
    const bool fam_on = o.fam != "none";
    std::size_t n_dk = 0;
    if (!dk.empty() && o.dk_ratio > 0)
        n_dk = std::min(dk.size(), static_cast<std::size_t>(std::llround(o.dk_ratio / (1 - o.dk_ratio) * turns.size())));
    std::vector<std::string> tok_dolly(turns.size()), tok_dk(dk.size()), tok_dev(dev.size()), tok_test(test.size()),
        tok_sh1(sh1.size()), tok_sh2(sh2.size());
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
        auto tok = [lo](const std::string& l) { return lo && l == "high" ? std::string() : "<|fam=" + l + "|>"; };
        std::map<std::string, int> cd, ck;
        std::map<std::string, std::pair<int, int>> cv;
        for (std::size_t i = 0; i < turns.size(); ++i) {
            const std::string l = oc ? "high" : tokq(instruction(i));
            ++cd["<|fam=" + l + "|>"];
            tok_dolly[i] = tok(l);
        }
        for (std::size_t i = 0; i < dk.size(); ++i) {
            const std::string l = oc ? "low" : tokq(dk[i].q);
            ++ck["<|fam=" + l + "|>"];
            tok_dk[i] = tok(l);
        }
        for (std::size_t i = 0; i < dev.size(); ++i) {
            const std::string l = tokq(dev[i].q);
            (dev[i].in_slice ? cv["<|fam=" + l + "|>"].first : cv["<|fam=" + l + "|>"].second)++;
            tok_dev[i] = tok(l);
        }
        for (std::size_t i = 0; i < test.size(); ++i) tok_test[i] = tok(tokq(test[i].q));
        for (std::size_t i = 0; i < sh1.size(); ++i) tok_sh1[i] = tok(oc ? "high" : tokq(sh1[i].q));
        for (std::size_t i = 0; i < sh2.size(); ++i) tok_sh2[i] = tok(oc ? "high" : tokq(sh2[i].q));
        for (auto& [k, v] : cd) std::cout << "fam_dolly " << k << " " << v << "\n";
        for (auto& [k, v] : ck) std::cout << "fam_dk " << k << " " << v << "\n";
        for (auto& [k, v] : cv) std::cout << "fam_dev " << k << " in_slice " << v.first << " not_in_slice " << v.second << "\n";
    }
    auto with_tok = [&](const std::string& prompt, const std::string& tok) {
        // prompt ends in "Assistant: "; "before" puts the token in front of it, "after" behind it.
        if (!fam_on) return prompt;
        const std::size_t at = prompt.size() - 11;
        return o.fam == "before" ? prompt.substr(0, at) + tok + prompt.substr(at) : prompt + tok;
    };

    // Chat text: Dolly, with the extra turns (don't-know pairs at dk_ratio of all turns, then every shard row), each
    // list spread evenly through it.
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
    const std::size_t chat_at = append_chat(model, data, chat);
    std::string().swap(chat);  // data holds the chat bytes now, and nothing below reads the turns
    std::vector<Turn>().swap(turns);
    train_span(model, data, w, chat_at, data.size(), cfg);
    train_span(model, data, w, chat_at, data.size(), cfg);
    model.compact_match();
    std::cerr << "chat trained\n";
    // M4: the packed form serves from here and the training form is freed. A later wiki_lm read (entity_of) sees the
    // chat-trained counts with no online match in either form (compact_match freed the training hash).
    const PackedModel pm = pack(model);
    model.release();
    wiki_lm.model = nullptr;
    wiki_lm.packed = &pm;

    // T, the training text (wiki train + chat): the class expert is built from it first (B4), then InfiniGram takes it
    // (M6 order, after the trie: A1c's tokens need the wiki trie before the chat text exists). Both builds come before
    // the holdout arrays below, which they do not read.
    std::vector<std::uint8_t> T;
    const std::size_t all_followers = training_text(data, cfg, T);
    std::unique_ptr<ClassContext> cc;
    if (!o.cls.empty()) {
        cc = make_class_context(o.cls);
        if (!cc) return 1;
        cc->build(T.data(), T.size());
    }
    // InfiniGram takes T, unless the similar-context expert reads it (A4, src=all: its keys and its neighbours are T's).
    const bool sim_reads_t = !o.sim.empty() && parse_sim(o.sim).src != "wiki";
    const FmInfiniGram ig = sim_reads_t ? FmInfiniGram(T.data(), T.size()) : FmInfiniGram(std::move(T));
    // Holdout: the trie mixture's BPC (= chat's wiki_bpc_after_chat), then InfiniGram's lambdas on the first half and
    // the trie + InfiniGram BPC on the second half (as context8k, cap 8192); with --copy, the copy expert's lambdas
    // on the first half (its context: the holdout stream itself) and its BPC and LCI gain on the second half.
    const std::size_t H = cfg.sample - cfg.train, half = H / 2;
    std::vector<double> pt(H), pinf(H), pig(H), pcp(H);
    std::vector<int> bk(H), cbk(H);
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
    double lam[kBuckets] = {0}, lc_t[kCopyBuckets] = {0}, lc_i[kCopyBuckets] = {0};
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
    fit_lambdas(pinf, pt, bk, half, kBuckets, lam);
    double bits_trie2 = 0, bits_ig2 = 0;
    for (std::size_t k = 0; k < H; ++k) {
        pig[k] = lam[bk[k]] * pinf[k] + (1 - lam[bk[k]]) * pt[k];
        if (k >= half) bits_trie2 += -std::log2(pt[k]), bits_ig2 += -std::log2(pig[k]);
    }
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "holdout_mix_bpc " << bits_all / static_cast<double>(H) << "\n";
    std::cout << "holdout2_trie_bpc " << bits_trie2 / static_cast<double>(H - half) << " holdout2_trie_ig_bpc "
              << bits_ig2 / static_cast<double>(H - half) << "\n";
    // B4: the class-context expert over the same training bytes as InfiniGram (wiki train + chat), built above, lambdas
    // on the first half of the holdout over the trie (lcl_t) and over trie + InfiniGram (lcl_i).
    double lcl_t[ClassContext::kBuckets] = {0}, lcl_i[ClassContext::kBuckets] = {0};
    std::vector<double> pcl(H);
    std::vector<int> clb(H);
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
    // A4: the similar-context expert, keys over wiki train + chat (src=all: InfiniGram's bytes) or wiki train only
    // (src=wiki); lambdas on the first half of the holdout over trie (+ cls) (lsm_t) and trie + InfiniGram (+ cls)
    // (lsm_i); its gain on the second half (LCI over 256 blocks). The copy expert, if on, is fit over it.
    std::unique_ptr<SimilarContext> sc;
    double lsm_t[SimilarContext::kBuckets] = {0}, lsm_i[SimilarContext::kBuckets] = {0};
    const std::uint8_t* sc_text = nullptr;
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
        const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
    if (o.copy) {
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
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";

    const int nlm = o.copy ? 4 : 2;
    const char* const lm_name[4] = {"trie", "trie_ig", "trie_cp", "trie_ig_cp"};
    const std::size_t lim = all_followers;
    const Lm lms[4] = {{nullptr, w, lim, false, nullptr, nullptr, nullptr, 8192, cc.get(), lcl_t, sc.get(), lsm_t, &pm},
                       {nullptr, w, lim, false, &ig, lam, nullptr, 8192, cc.get(), lcl_i, sc.get(), lsm_i, &pm},
                       {nullptr, w, lim, false, nullptr, nullptr, lc_t, 8192, cc.get(), lcl_t, sc.get(), lsm_t, &pm},
                       {nullptr, w, lim, false, &ig, lam, lc_i, 8192, cc.get(), lcl_i, sc.get(), lsm_i, &pm}};

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

    auto run_set = [&](const std::vector<QaRow>& rows, const std::vector<std::string>& toks) {
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
                        form == 0 ? with_tok("User: " + rows[r].q + "\nAssistant: ", g.fam_tok) : rows[r].q + " ";
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

    const std::vector<Graded> gd = run_set(dev, tok_dev);
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

    // The frozen test, once: gates chosen on all of dev, applied unchanged.
    if (!test.empty()) {
        const std::vector<Graded> gt = run_set(test, tok_test);
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
        // B4: the class-context expert on a tiny corpus. A verb of the same class reuses what "wrote" learned; an
        // unknown word falls to the uniform floor; P sums to 1; the bucket packs order and count.
        ClassContext cc({{"wrote", 10}, {"composed", 10}, {"who", 11}, {"the", 12}}, 6);
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
        // Spec fields are read from the right (M2/M3 flags): a third order and dropping contexts change the table,
        // and the plain forms parse as before.
        auto built = [&](const std::string& spec) {
            const auto c = make_class_context(spec);
            if (c) c->build(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
            return c ? c->contexts() : std::size_t{0};
        };
        const std::size_t s2 = built("shape"), s3 = built("shape:6:3"), s3p = built("shape:6:3:p2"),
                          s3cp = built("shape:6:3:cp2");
        const bool flags_ok = s2 == built("shape:6") && s2 == 64 && s3 == 103 && s3p == 42 && s3cp == 42;
        if (!ok || !flags_ok) {
            std::cerr << "selftest: class-context expert checks failed (spec sizes " << s2 << " " << s3 << " " << s3p
                      << " " << s3cp << ")\n";
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
                        parse_sim("vec=x,W=4,k=8,src=wiki").ok && !parse_sim("vec=x,src=dolly").ok && !parse_sim("W=4").ok &&
                        parse_sim("vec=x,k=8").build == parse_sim("vec=x,k=128,T=0").build &&
                        parse_sim("vec=x,W=4").build != parse_sim("vec=x,W=8").build;
        if (!ok) {
            std::cerr << "selftest: similar-context expert checks failed\n";
            return 1;
        }
    }
    std::cout << "selftest ok\n";
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: cyphalm_trie <bpc|chat|probe|context8k|sample|cls|qa|dktext|selftest> [wiki] [dolly] [options]\n"
                 "  bpc [wiki] [--negative] [--no-guard]\n"
                 "  chat [wiki] [dolly] [--prompt TEXT] [--no-repeat] [--save FILE | --model FILE]\n"
                 "  probe [wiki] [dolly]\n"
                 "  context8k [wiki] [dolly]\n"
                 "  sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600] [--save FILE | --model FILE]\n"
                 "  cls <wiki> <TABLE|shape>[:tail[:orders[:flags]]] [more specs] [--sim SPEC ...] [--sim-fresh] [--sim-surface]\n"
                 "  SPEC: vec=PREFIX[,W=8][,k=32][,T=0.05][,min=-1][,sif=1][,dim=64][,drop=1][,lists=512][,probe=8][,tail=32][,src=all|wiki]\n"
                 "  qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--leak TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]\n"
                 "     [--dk DK.tsv --dk-ratio R] [--fam none|before|after] [--parse E1.tsv] [--fam-stat NAME]\n"
                 "     [--fam-train stat|outcome|low] [--fam-only] [--copy] [--copy-min N] [--r3 PATTERNS.tsv]\n"
                 "     [--shard F.tsv] [--shard2 F.tsv] [--cls TABLE|shape[:tail[:orders[:flags]]]] [--sim SPEC]\n"
                 "  dktext [wiki] [dolly] --dev DEV.tsv --test TEST.tsv --out DK.tsv [--from 8388608] [--max-rows N] [--mention]\n"
                 "  selftest\n"
                 "  options: --wiki P --dolly P --train N --sample N\n"
                 "  paths default to $CYPHA_WIKI / $CYPHA_DOLLY, then %s and %s\n",
                 kDefaultWiki, kDefaultDolly);
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
    const Args a = parse(argc, argv, 2);
    if (!a.ok) {
        usage();
        return 2;
    }
    if (cmd != "bpc" && cmd != "chat" && cmd != "probe" && cmd != "context8k" && cmd != "sample" && cmd != "qa" &&
        cmd != "dktext" && cmd != "cls") {
        usage();
        return 2;
    }
    if ((!a.model.empty() || !a.save.empty()) && cmd != "chat" && cmd != "sample") {
        std::cerr << "--model and --save are for chat and sample\n";
        return 2;
    }
    if (!self_test()) return 1;
    const auto t0 = std::chrono::steady_clock::now();
    int rc = 2;
    if (cmd == "bpc") rc = cmd_bpc(a);
    else if (cmd == "chat") rc = cmd_chat(a);
    else if (cmd == "probe") rc = cmd_probe(a);
    else if (cmd == "context8k") rc = cmd_context8k(a);
    else if (cmd == "sample") rc = cmd_sample(a);
    else if (cmd == "cls") rc = cmd_cls(a);
    else if (cmd == "qa") {
        if (a.dev.empty() || (a.fam != "none" && a.fam != "before" && a.fam != "after") || a.dk_ratio < 0 ||
            a.dk_ratio >= 1 || (a.fam_train != "stat" && a.fam_train != "outcome" && a.fam_train != "low")) {
            usage();
            return 2;
        }
        rc = cmd_qa(a, QaOpts{a.dev, a.test, a.out, a.base, a.dk, a.dk_ratio, a.fam, a.parse, a.fam_stat, a.fam_train,
                              a.r3, a.shard, a.shard2, a.leak, a.copy, a.fam_only, a.cls, a.copy_min,
                              a.sims.empty() ? std::string() : a.sims[0]});
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
