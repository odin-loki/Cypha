// cyphalm_trie: the one tool for CyphaLM's packed byte trie (PLAN Track B1).
//
// Replaces ten tools: trie_fold_lm, trie_online_mix (-> bpc), trie_chat (-> chat),
// trie_qa_probe (-> probe), trie_context_8k (-> context8k), trie_sample (-> sample),
// and the experiments trie_together, trie_adaptive, trie_gria_gate and
// trie_online_holdout (retired; results and recovery commits are in
// docs/reports/B1_TRIE_CONSOLIDATION.md). The model itself is
// native/include/cypha/cyphalm/packed_trie.hpp (namespace cypha::cyphalm::trie).
// trie_stream_train.cpp still carries its own copy of the model.
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
//   chat [wiki] [dolly] [--prompt TEXT] [--no-repeat]
//        trie_chat: train on wiki, then two passes over Dolly after a <|doc|> break;
//        prints the wiki BPC after chat, the weights, mix_stored and one sample after
//        a question the chat never saw.
//   probe [wiki] [dolly]
//        trie_qa_probe: what the mixture writes after a seen / first-letter-changed /
//        Wikipedia-fact question (greedy and nucleus 0.9, T = 0.8).
//   context8k [wiki] [dolly]
//        trie_context_8k: InfiniGram (order up to 8192) mixed with the trie, BPC at five
//        caps, matched-length histogram, 50 questions with 30 Dolly turns held out.
//   sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600]
//        trie_sample: wiki-only trie, 3 temperatures, no-repeat rule, files for scribe_ladder.
//   qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]
//      [--dk DK.tsv --dk-ratio R] [--fam none|before|after]
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
//   dktext [wiki] [dolly] --dev DEV.tsv --test TEST.tsv --out DK.tsv [--from 8388608]
//        A1 training text: T-QA relation templates filled with page titles from enwik8 after
//        --from whose lowered text never occurs in the lowered training bytes (wiki train +
//        Dolly; exact suffix-array check, audited by a plain scan), paired with the DISCOVERY
//        §3.5 message and tag. Rejects any entity or answer variant found in dev or test, and
//        fails if the final leakage check or the audit finds anything.
//   selftest
//        packed_trie.hpp self_test plus a golden BPC on a 40,000-byte built-in corpus, plus the
//        qa grading, tag parsing, Wilson bounds and abstain gate on fixed inputs
//        (CTest native_cyphalm_trie_selftest; needs no data files; about 1 second).
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
//
// Build (from the repo root):
//   gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o
//   g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais
//       native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie
//   (one command line; the two lines above are split for width)
// Usage:
//   ./cyphalm_trie bpc /root/cdata/enwik8
//   ./cyphalm_trie chat /root/cdata/enwik8 /root/cdata/dolly.jsonl
//   ./cyphalm_trie sample /root/cdata/enwik8 gen_dev.txt gen_fresh.txt
//   ./cyphalm_trie selftest

#include "cypha/cyphalm/infinigram.hpp"
#include "cypha/cyphalm/packed_trie.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iterator>
#include <map>
#include <tuple>

using namespace cypha::cyphalm::trie;
using cypha::cyphalm::InfiniGram;

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

    std::cerr << "scoring synth\n";
    const Score s_synth = heldout(model, data.data(), Mode::Synth, cfg);
    std::cerr << "scoring mixture\n";
    const Holdout h = score_holdout(model, data.data(), w, cfg);
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
    std::cout << "hash_reserved " << hash_reserved(model) << " hash_payload " << hash_payload(model) << "\n";
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
    const std::uint64_t mix_stored = synth.bytes + order1_bytes(model) + match_bytes(model) + 6ull * sizeof(double);

    std::cerr << "scoring\n";
    const Holdout h = score_holdout(model, data.data(), w, cfg);
    if (!h.ok) return 1;

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "wiki_bpc_after_chat " << h.mix_bpc << "\n";
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";
    std::cout << "mix_stored " << mix_stored << "\n";
    std::cout << "mix_beats_backoff " << (h.mix_bpc < h.backoff_bpc ? 1 : 0) << "\n";

    // Answers come from CyphaLM generation only (PLAN rule 10).
    SampleOptions opt;
    opt.no_repeat = a.no_repeat;
    const std::string sample = sample_continuation(model, w, a.prompt, data.size(), 160, kSeed, opt);
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
        const std::string g = sample_continuation(model, w, q.prompt, data.size(), 120, 0, greedy);
        const std::string s = sample_continuation(model, w, q.prompt, data.size(), 160, kSeed);
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

    // Index the training bytes only: wiki train + boundary + chat. No wiki holdout.
    std::vector<std::uint8_t> T(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.train));
    T.insert(T.end(), data.begin() + static_cast<std::ptrdiff_t>(cfg.sample), data.end());
    InfiniGram ig(T.data(), T.size());
    std::cerr << "index built over " << T.size() << " bytes\n";

    const std::size_t kTrain = cfg.train, kSample = cfg.sample;
    const std::size_t H = kSample - kTrain, half = H / 2;
    std::vector<double> pt(H);
    for (std::size_t k = 0; k < H; ++k) {
        const std::size_t i = kTrain + k;
        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, kTrain, Mode::Synth, e, em, pred, false);
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
        for (std::size_t k = 0; k < H; ++k) {
            const std::size_t i = kTrain + k;
            const std::size_t len = std::min<std::size_t>(i, static_cast<std::size_t>(cap));
            const int hint = prev < 0 ? -1 : std::min(prev + 1, cap);
            const auto r = ig.query(data.data() + (i - len), len, cap, hint);
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
        n0 = -1;
        for (int t = 0; t < 150; ++t) {
            double e[5][256];
            bool m = false;
            unsigned pred = 0;
            model.fill_experts(seq.data(), seq.size(), data.size(), Mode::Synth, e, m, pred, false);
            double M[256];
            mix_dist(w, e, m, pred, M);
            const std::size_t len = std::min<std::size_t>(seq.size(), 8192);
            const auto r =
                ig.query(seq.data() + (seq.size() - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192));
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
    const Config& cfg = a.cfg;
    if (a.outs.empty() || a.outs.size() > 2 || a.prompts < 1 || a.bytes < 100) {
        std::fprintf(stderr,
                     "usage: cyphalm_trie sample <enwik8> <out_dev.txt> [out_fresh.txt] [--prompts 250] [--bytes 600]\n");
        return 2;
    }
    const std::size_t nfiles = a.outs.size();
    std::vector<std::uint8_t> data;
    if (!load_wiki(a, data)) return 1;

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

    Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    init_weights(w, cfg);
    TrainLog log;
    log.progress = true;
    train_span(model, data, w, 0, cfg.train, cfg, &log);
    model.compact_match();
    std::cerr << "trained on [0, " << cfg.train << ") nodes " << model.nodes.size() << " w4 " << w[4] << " wmatch "
              << w[5] << "\n";

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
                const std::string text = sample_continuation(model, w, p, cfg.train, a.bytes,
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

// CyphaLM's next-byte distribution: the frozen trie mixture, optionally mixed with InfiniGram
// (context8k: P = lam[b] P_ctx + (1 - lam[b]) P_trie, one lam per match-length x count bucket).
struct Lm {
    const Model* model = nullptr;
    const double* w = nullptr;
    std::size_t follower_limit = 0;
    bool online_match = false;
    const InfiniGram* ig = nullptr;
    const double* lam = nullptr;
    int cap = 8192;
    // hint: the previous step's match length (-1 at a new sequence); n_out: this step's.
    void dist(const std::vector<std::uint8_t>& seq, int& hint, double* P, int* n_out = nullptr) const {
        double e[5][256];
        bool m = false;
        unsigned pred = 0;
        model->fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, m, pred, online_match);
        mix_dist(w, e, m, pred, P);
        if (!ig) return;
        const std::size_t len = std::min<std::size_t>(seq.size(), static_cast<std::size_t>(cap));
        const auto r = ig->query(seq.data() + (seq.size() - len), len, cap, hint < 0 ? -1 : std::min(hint + 1, cap));
        hint = r.n;
        if (n_out) *n_out = r.n;
        const double l = lam[bucket(r.n, r.total)];
        for (int s = 0; s < 256; ++s) {
            const double pc = r.total ? static_cast<double>(r.count[s]) / static_cast<double>(r.total) : 0.0;
            P[s] = l * pc + (1 - l) * P[s];
        }
    }
};

// Mean log2 p per byte of `text` read after `prefix` (the familiarity measure).
double text_logprob(const Lm& lm, const std::string& prefix, const std::string& text) {
    std::vector<std::uint8_t> seq(prefix.begin(), prefix.end());
    int hint = -1;
    double lp = 0;
    for (unsigned char b : text) {
        double P[256];
        lm.dist(seq, hint, P);
        lp += std::log2(std::max(P[b], 1e-300));
        seq.push_back(b);
    }
    return text.empty() ? 0.0 : lp / static_cast<double>(text.size());
}

struct Answer {
    std::string text;
    double mean_lp = 0, min_lp = 0, ent0 = 0;
    int ig_n0 = 0;
};

// Greedy, no-repeat rule, until a newline or a sentence end, at most max_bytes. Once the answer
// begins with the don't-know message, decoding runs on to the end of the tag (at most 240 bytes)
// so the tag can be read; that is the only change to the stopping rule.
Answer decode(const Lm& lm, const std::string& prompt, int max_bytes = 80) {
    Answer A;
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
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
        mark_banned(seq, banned);
        bool all = true;
        for (int s = 0; s < 256; ++s)
            if (!banned[s]) all = false;
        int pick = -1;
        for (int s = 0; s < 256; ++s) {
            if (banned[s] && !all) continue;
            if (pick < 0 || P[s] > P[pick]) pick = s;
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

// T-QA relation templates (from the dev set's question forms), chosen by the page's categories.
std::vector<std::pair<std::string, std::string>> templates_for(const std::string& cats_lower) {
    auto has = [&](const char* k) { return cats_lower.find(k) != std::string::npos; };
    if (has("births") || has("deaths"))
        return {{"birth_date", "When was {E} born?"}, {"birth_place", "Where was {E} born?"},
                {"death_date", "When did {E} die?"},  {"identity", "Who was {E}?"}};
    if (has("novels") || has("books") || has("plays") || has("poems")) return {{"author", "Who wrote {E}?"}};
    if (has("films")) return {{"director", "Who directed {E}?"}};
    if (has("operas") || has("songs") || has("compositions") || has("symphon") || has("ballets"))
        return {{"composer", "Who composed {E}?"}};
    if (has("paintings")) return {{"painter", "Who painted {E}?"}};
    if (has("countries") || has("states of") || has("provinces") || has("territories"))
        return {{"capital", "What is the capital of {E}?"}, {"language", "What is the official language of {E}?"}};
    if (has("companies") || has("organizations") || has("organisations") || has("bands") || has("musical groups"))
        return {{"founder", "Who founded {E}?"}};
    if (has("cities") || has("towns") || has("villages") || has("rivers") || has("islands") ||
        has("municipalities") || has("counties") || has("mountains"))
        return {{"location", "Where is {E}?"}, {"country", "In which country is {E}?"}};
    if (has("inventions") || has("technology")) return {{"inventor", "Who invented {E}?"}};
    return {{"definition", "What is {E}?"}};
}

int cmd_dktext(const Args& a, const std::string& dev_path, const std::string& test_path, const std::string& out_path,
               std::size_t from, std::size_t max_rows) {
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
    InfiniGram occ(reinterpret_cast<const std::uint8_t*>(train.data()), train.size());
    auto occurs = [&](const std::string& lower) {
        std::size_t pos = 0, cnt = 0;
        return occ.match_prefix(reinterpret_cast<const std::uint8_t*>(lower.data()), lower.size(), lower.size(), 0,
                                pos, cnt) == lower.size();
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
    std::size_t pos = 0, pages = 0, n_ns = 0, n_redirect = 0, n_shape = 0, n_in_train = 0, n_in_eval = 0, n_alias = 0;
    std::vector<DkRow> rows;
    std::vector<std::string> kept_entities;
    while (true) {
        const std::size_t t0 = hay.find("<title>", pos);
        if (t0 == std::string::npos) break;
        const std::size_t t1 = hay.find("</title>", t0);
        const std::size_t pe = hay.find("</page>", t0);
        if (t1 == std::string::npos || pe == std::string::npos) break;
        pos = pe + 7;
        ++pages;
        const std::string title = hay.substr(t0 + 7, t1 - t0 - 7);
        const std::string body = hay.substr(t1, pe - t1);
        if (title.find(':') != std::string::npos) {
            ++n_ns;
            continue;
        }
        const std::size_t tx = body.find("<text");
        const std::size_t tx_end = tx == std::string::npos ? std::string::npos : body.find('>', tx);
        if (tx_end == std::string::npos) continue;
        if (lowered(body.substr(tx_end + 1, 12)).rfind("#redirect", 0) == 0) {
            ++n_redirect;
            continue;
        }
        bool shape = title.size() >= 4 && title.size() <= 40 && title.find_first_of("(),&;\"'") == std::string::npos &&
                     title.rfind("List of", 0) != 0 && std::isupper(static_cast<unsigned char>(title[0]));
        for (unsigned char c : title)
            if (c >= 128) shape = false;
        if (!shape) {
            ++n_shape;
            continue;
        }
        const std::string el = lowered(title);
        if (occurs(el)) {  // the entity occurs somewhere in the training bytes
            ++n_in_train;
            continue;
        }
        if (evaltext.find(el) != std::string::npos) {  // leakage: the entity is in dev or test
            ++n_in_eval;
            continue;
        }
        const std::string en = norm_text(title);
        bool alias_hit = false;
        for (const auto& g : eval_alias)
            if (en.find(g) != std::string::npos) alias_hit = true;
        if (alias_hit) {
            ++n_alias;
            continue;
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
    }
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
    for (std::size_t i = 0; i < kept_entities.size() && audit < 300; i += std::max<std::size_t>(1, kept_entities.size() / 300)) {
        ++audit;
        if (train.find(lowered(kept_entities[i])) != std::string::npos) ++audit_found;
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
    std::cout << "dk_pages_scanned " << pages << " from_byte " << from << "\n";
    std::cout << "dk_rejected namespace " << n_ns << " redirect " << n_redirect << " shape " << n_shape
              << " in_training_bytes " << n_in_train << " in_dev_or_test " << n_in_eval << " dev_test_alias " << n_alias
              << "\n";
    std::cout << "dk_entities_kept " << kept_entities.size() << " dk_rows " << rows.size() << "\n";
    for (const auto& [k, v] : by_rel) std::cout << "dk_relation " << k << " " << v << "\n";
    std::cout << "leak_check question_equal " << leak_q << " answer_variant_in_text " << leak_alias << "\n";
    std::cout << "absence_audit_plain_scan " << audit << " found " << audit_found << "\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, rows.size()); ++i)
        std::cout << "dk_example " << rows[i].q << " => " << rows[i].response << "\n";
    return (leak_q || leak_alias || audit_found) ? 1 : 0;
}

// ---------------------------------------------------------------- qa (A0 harness, A1 variants)

struct QaOpts {
    std::string dev, test, out, base, dk;
    double dk_ratio = 0;
    std::string fam = "none";  // none | before | after: where the <|fam=..|> token goes
};

struct Graded {
    int row = 0, in_slice = 0, form = 0, lm = 0;  // form 0 chat, 1 wiki; lm 0 trie, 1 trie+InfiniGram
    int grade = 0;                                // 0 right, 1 abstained (message), 2 wrong
    Answer ans;
    double qlp = 0;  // final model: mean log2 p of the question (familiarity)
    std::string fam_tok;
};

constexpr int kFeat = 5;
const char* const kFeatName[kFeat] = {"mean_lp", "min_lp", "neg_entropy0", "ig_match0", "question_lp"};
double feat(const Graded& g, int f) {
    switch (f) {
        case 0: return g.ans.mean_lp;
        case 1: return g.ans.min_lp;
        case 2: return -g.ans.ent0;
        case 3: return g.ans.ig_n0;
        default: return g.qlp;
    }
}

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

int cmd_qa(const Args& a, const QaOpts& o) {
    const Config& cfg = a.cfg;
    const std::vector<QaRow> dev = load_tqa(o.dev);
    const std::vector<QaRow> test = o.test.empty() ? std::vector<QaRow>{} : load_tqa(o.test);
    if (dev.empty() || (!o.test.empty() && test.empty())) {
        std::cerr << "cannot read the T-QA files\n";
        return 1;
    }
    std::vector<DkRow> dk;
    if (!o.dk.empty()) {
        std::ifstream in(o.dk);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            const auto f = split(line, '\t');
            if (f.size() >= 5) dk.push_back({f[0], f[1], f[2], f[3], f[4]});
        }
        if (dk.empty()) {
            std::cerr << "no don't-know rows in " << o.dk << "\n";
            return 1;
        }
    }
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
    load_dolly(a.dolly, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << a.dolly << "\n";
        return 1;
    }

    // Familiarity token (A1b): CyphaLM's own mean log2 p of the question after reading the wiki
    // slice (the model state before any chat; the match expert does not fire on a separate buffer).
    // Thresholds come from the training questions only (Dolly instructions and don't-know questions).
    const bool fam_on = o.fam != "none";
    double t1 = 0, t2 = 0;
    std::vector<std::string> tok_dolly(turns.size()), tok_dk(dk.size()), tok_dev(dev.size()), tok_test(test.size());
    if (fam_on) {
        double wf[6];
        std::copy(w, w + 6, wf);
        Lm wiki_lm{&model, wf, cfg.train, true};
        std::vector<double> fd(turns.size()), fk(dk.size());
        for (std::size_t i = 0; i < turns.size(); ++i) {
            const std::string& p = turns[i].prompt;  // "User: <instruction>\n..."
            const std::size_t nl = p.find('\n');
            fd[i] = text_logprob(wiki_lm, "User: ", p.substr(6, nl - 6));
        }
        for (std::size_t i = 0; i < dk.size(); ++i) fk[i] = text_logprob(wiki_lm, "User: ", dk[i].q);
        // t1: the cut that best separates don't-know questions (below) from Dolly ones (Youden J);
        // t2: the median Dolly familiarity above t1.
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
        auto tok = [&](double x) { return std::string(x < t1 ? "<|fam=low|>" : x < t2 ? "<|fam=mid|>" : "<|fam=high|>"); };
        std::map<std::string, int> cd, ck;
        for (std::size_t i = 0; i < turns.size(); ++i) ++cd[tok_dolly[i] = tok(fd[i])];
        for (std::size_t i = 0; i < dk.size(); ++i) ++ck[tok_dk[i] = tok(fk[i])];
        for (std::size_t i = 0; i < dev.size(); ++i) tok_dev[i] = tok(text_logprob(wiki_lm, "User: ", dev[i].q));
        for (std::size_t i = 0; i < test.size(); ++i) tok_test[i] = tok(text_logprob(wiki_lm, "User: ", test[i].q));
        std::cout << "fam_thresholds t1 " << t1 << " t2 " << t2 << " youden_j " << bestj << "\n";
        for (auto& [k, v] : cd) std::cout << "fam_dolly " << k << " " << v << "\n";
        for (auto& [k, v] : ck) std::cout << "fam_dk " << k << " " << v << "\n";
        std::map<std::string, std::pair<int, int>> cv;
        for (std::size_t i = 0; i < dev.size(); ++i) (dev[i].in_slice ? cv[tok_dev[i]].first : cv[tok_dev[i]].second)++;
        for (auto& [k, v] : cv) std::cout << "fam_dev " << k << " in_slice " << v.first << " not_in_slice " << v.second << "\n";
    }
    auto with_tok = [&](const std::string& prompt, const std::string& tok) {
        // prompt ends in "Assistant: "; "before" puts the token in front of it, "after" behind it.
        if (!fam_on) return prompt;
        const std::size_t at = prompt.size() - 11;
        return o.fam == "before" ? prompt.substr(0, at) + tok + prompt.substr(at) : prompt + tok;
    };

    // Chat text: Dolly, with don't-know pairs spread evenly at dk_ratio of all turns.
    std::size_t n_dk = 0;
    if (!dk.empty() && o.dk_ratio > 0)
        n_dk = std::min(dk.size(), static_cast<std::size_t>(std::llround(o.dk_ratio / (1 - o.dk_ratio) * turns.size())));
    std::string chat;
    std::size_t j = 0;
    for (std::size_t i = 0; i < turns.size(); ++i) {
        chat += with_tok(turns[i].prompt, tok_dolly[i]) + turns[i].answer + "\n\n";
        while (j < n_dk && (j + 1) * turns.size() / (n_dk + 1) <= i) {
            chat += with_tok("User: " + dk[j].q + "\nAssistant: ", tok_dk[j]) + dk[j].response + "\n\n";
            ++j;
        }
    }
    std::cout << "chat_turns dolly " << turns.size() << " dont_know " << n_dk << " share "
              << static_cast<double>(n_dk) / static_cast<double>(turns.size() + n_dk) << " fam " << o.fam << "\n";
    const std::size_t chat_at = append_chat(model, data, chat);
    train_span(model, data, w, chat_at, data.size(), cfg);
    train_span(model, data, w, chat_at, data.size(), cfg);
    model.compact_match();
    std::cerr << "chat trained\n";

    // Holdout: the trie mixture's BPC (= chat's wiki_bpc_after_chat), then InfiniGram's lambdas on
    // the first half and the trie + InfiniGram BPC on the second half (as context8k, cap 8192).
    const std::size_t H = cfg.sample - cfg.train, half = H / 2;
    std::vector<double> pt(H);
    double bits_all = 0;
    for (std::size_t k = 0; k < H; ++k) {
        const std::size_t i = cfg.train + k;
        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
        double M[256];
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
        bits_all += -std::log2(pt[k]);
    }
    std::vector<std::uint8_t> T(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cfg.train));
    T.insert(T.end(), data.begin() + static_cast<std::ptrdiff_t>(cfg.sample), data.end());
    InfiniGram ig(T.data(), T.size());
    double lam[kBuckets] = {0};
    double bits_trie2 = 0, bits_ig2 = 0;
    {
        std::vector<double> pinf(H);
        std::vector<int> bk(H);
        int prev = -1;
        for (std::size_t k = 0; k < H; ++k) {
            const std::size_t i = cfg.train + k;
            const std::size_t len = std::min<std::size_t>(i, 8192);
            const auto r = ig.query(data.data() + (i - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192));
            prev = r.n;
            pinf[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
            bk[k] = bucket(r.n, r.total);
        }
        for (int b = 0; b < kBuckets; ++b) {
            double best = -1e300;
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
        for (std::size_t k = half; k < H; ++k) {
            bits_trie2 += -std::log2(pt[k]);
            bits_ig2 += -std::log2(lam[bk[k]] * pinf[k] + (1 - lam[bk[k]]) * pt[k]);
        }
    }
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "holdout_mix_bpc " << bits_all / static_cast<double>(H) << "\n";
    std::cout << "holdout2_trie_bpc " << bits_trie2 / static_cast<double>(H - half) << " holdout2_trie_ig_bpc "
              << bits_ig2 / static_cast<double>(H - half) << "\n";
    for (int i = 0; i < 5; ++i) std::cout << "w" << i << " " << w[i] << "\n";
    std::cout << "wmatch " << w[5] << "\n";

    const Lm lm_trie{&model, w, data.size(), false, nullptr, nullptr};
    const Lm lm_ig{&model, w, data.size(), false, &ig, lam};

    auto run_set = [&](const std::vector<QaRow>& rows, const std::vector<std::string>& toks) {
        std::vector<Graded> out;
        for (std::size_t r = 0; r < rows.size(); ++r)
            for (int form = 0; form < 2; ++form)
                for (int l = 0; l < 2; ++l) {
                    const Lm& lm = l ? lm_ig : lm_trie;
                    Graded g;
                    g.row = static_cast<int>(r);
                    g.in_slice = rows[r].in_slice;
                    g.form = form;
                    g.lm = l;
                    g.fam_tok = fam_on ? toks[r] : "";
                    const std::string prompt =
                        form == 0 ? with_tok("User: " + rows[r].q + "\nAssistant: ", g.fam_tok) : rows[r].q + " ";
                    g.ans = decode(lm, prompt);
                    g.qlp = text_logprob(lm, form == 0 ? "User: " : "", rows[r].q);
                    if (is_dont_know(g.ans.text) || g.ans.text.empty()) g.grade = 1;
                    else g.grade = (rows[r].in_slice && gold_match(g.ans.text, rows[r].gold)) ? 0 : 2;
                    out.push_back(g);
                }
        return out;
    };
    const std::vector<Graded> gd = run_set(dev, tok_dev);
    const char* const form_name[2] = {"chat", "wiki"};
    const char* const lm_name[2] = {"trie", "trie_ig"};

    // The A0 base run's grades, for A1's "in-slice questions it gets wrong".
    std::map<std::tuple<int, int, int>, int> base;
    if (!o.base.empty()) {
        std::ifstream in(o.base);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            const auto f = split(line, '\t');
            if (f.size() >= 6 && f[0] == "dev")
                base[{std::atoi(f[1].c_str()), f[3] == "wiki", f[4] == "trie_ig"}] = std::atoi(f[5].c_str());
        }
    }

    if (!o.out.empty()) {
        std::ofstream f(o.out);
        f << "set\trow\tin_slice\tform\tlm\tgrade\tmean_lp\tmin_lp\tentropy0\tig_n0\tquestion_lp\tfam\tanswer\n";
        for (const auto& g : gd)
            f << "dev\t" << g.row << "\t" << g.in_slice << "\t" << form_name[g.form] << "\t" << lm_name[g.lm] << "\t"
              << g.grade << "\t" << g.ans.mean_lp << "\t" << g.ans.min_lp << "\t" << g.ans.ent0 << "\t" << g.ans.ig_n0
              << "\t" << g.qlp << "\t" << g.fam_tok << "\t" << printable(g.ans.text) << "\n";
    }

    std::cout.precision(4);
    Gate final_gate[2][2];
    for (int form = 0; form < 2; ++form)
        for (int l = 0; l < 2; ++l) {
            std::vector<const Graded*> v;
            for (const auto& g : gd)
                if (g.form == form && g.lm == l) v.push_back(&g);
            const std::string tag = std::string(form_name[form]) + "/" + lm_name[l];
            // Raw: everything it wrote counts; only its own message abstains.
            Tally raw, raw_in, raw_out;
            for (auto* g : v) {
                raw.add(g->grade);
                (g->in_slice ? raw_in : raw_out).add(g->grade);
            }
            print_tally("RAW " + tag + " all", raw);
            print_tally("RAW " + tag + " in_slice", raw_in);
            print_tally("RAW " + tag + " not_in_slice", raw_out);
            // 5-fold CV inside dev (fold = row % 5): gate chosen on four folds, applied to the fifth.
            Tally cvt, cv_in, cv_out;
            std::map<int, int> feat_pick;
            for (int fold = 0; fold < 5; ++fold) {
                std::vector<const Graded*> tr;
                for (auto* g : v)
                    if (g->row % 5 != fold) tr.push_back(g);
                const Gate gate = choose_gate(tr, l == 1);
                ++feat_pick[gate.f];
                for (auto* g : v)
                    if (g->row % 5 == fold) {
                        const int gr = gated_answer(*g, gate) ? g->grade : 1;
                        cvt.add(gr);
                        (g->in_slice ? cv_in : cv_out).add(gr);
                    }
            }
            print_tally("CV " + tag + " all", cvt);
            print_tally("CV " + tag + " in_slice", cv_in);
            print_tally("CV " + tag + " not_in_slice", cv_out);
            std::cout << "CV " << tag << " gate_features";
            for (auto& [f, c] : feat_pick) std::cout << " " << (f < 0 ? "none" : kFeatName[f]) << "x" << c;
            std::cout << "\n";
            int rr = 0;
            final_gate[form][l] = choose_gate(v, l == 1, &rr);
            std::cout << "FULLDEV_GATE " << tag << " feature "
                      << (final_gate[form][l].f < 0 ? "none" : kFeatName[final_gate[form][l].f]) << " tau "
                      << final_gate[form][l].tau << " right " << rr << "\n";
            // Risk-coverage curve, in-sample on all of dev, for each feature.
            for (int f = 0; f < kFeat; ++f) {
                if (f == 3 && l == 0) continue;
                std::vector<std::pair<double, int>> s;
                for (auto* g : v)
                    if (g->grade != 1) s.push_back({feat(*g, f), g->grade});
                std::sort(s.begin(), s.end(), [](auto& x, auto& y) { return x.first > y.first; });
                std::cout << "RC " << tag << " " << kFeatName[f] << " (answered:right:wrong)";
                int r = 0, wr = 0;
                for (std::size_t k = 0; k < s.size(); ++k) {
                    (s[k].second == 0 ? r : wr)++;
                    const std::size_t m = k + 1;
                    if (m == 1 || m == 2 || m == 3 || m == 5 || m % 10 == 0 || m == s.size())
                        std::cout << " " << m << ":" << r << ":" << wr;
                }
                std::cout << "\n";
            }
            // A1: abstention by CyphaLM's own message.
            int abst = 0, abst_out = 0, n_out = 0, abst_inw = 0, n_inw = 0, tags = 0, code_ok = 0, ent_ok = 0;
            for (auto* g : v) {
                const bool ab = g->grade == 1 && !g->ans.text.empty();
                abst += ab;
                if (!g->in_slice) {
                    ++n_out;
                    abst_out += ab;
                } else {
                    auto it = base.find({g->row, form, l});
                    const bool base_wrong = it == base.end() ? true : it->second != 0;
                    if (base_wrong) {
                        ++n_inw;
                        abst_inw += ab;
                    }
                }
                const std::string code = tag_field(g->ans.text, "reason");
                if (!code.empty()) {
                    ++tags;
                    // Expected: UNKNOWN_ENTITY for a subject with no article in the slice (not-in-slice),
                    // NO_EVIDENCE or LOW_CERTAINTY when the article is there (in-slice).
                    code_ok += g->in_slice ? (code == "NO_EVIDENCE" || code == "LOW_CERTAINTY") : code == "UNKNOWN_ENTITY";
                    const std::string ent = tag_field(g->ans.text, "entity");
                    ent_ok += !ent.empty() && lowered(dev[static_cast<std::size_t>(g->row)].q).find(lowered(ent)) != std::string::npos;
                }
            }
            const double base_rate = static_cast<double>(n_out) / static_cast<double>(v.size());
            std::cout << "ABSTAIN " << tag << " abstained " << abst << " not_in_slice " << abst_out << "/" << n_out
                      << " precision_nis " << (abst ? static_cast<double>(abst_out) / abst : 0.0) << " recall_nis "
                      << (n_out ? static_cast<double>(abst_out) / n_out : 0.0) << " base_rate_nis " << base_rate
                      << " | in_slice_base_wrong " << abst_inw << "/" << n_inw << " precision_should "
                      << (abst ? static_cast<double>(abst_out + abst_inw) / abst : 0.0) << " recall_should "
                      << (n_out + n_inw ? static_cast<double>(abst_out + abst_inw) / (n_out + n_inw) : 0.0)
                      << " base_rate_should " << static_cast<double>(n_out + n_inw) / static_cast<double>(v.size())
                      << " | tags " << tags << " reason_ok " << code_ok << " entity_in_question " << ent_ok << "\n";
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

    // The frozen test, once: gate chosen on all of dev, applied unchanged.
    if (!test.empty()) {
        const std::vector<Graded> gt = run_set(test, tok_test);
        if (!o.out.empty()) {
            std::ofstream f(o.out, std::ios::app);
            for (const auto& g : gt)
                f << "test\t" << g.row << "\t" << g.in_slice << "\t" << form_name[g.form] << "\t" << lm_name[g.lm] << "\t"
                  << g.grade << "\t" << g.ans.mean_lp << "\t" << g.ans.min_lp << "\t" << g.ans.ent0 << "\t" << g.ans.ig_n0
                  << "\t" << g.qlp << "\t" << g.fam_tok << "\t" << printable(g.ans.text) << "\n";
        }
        for (int form = 0; form < 2; ++form)
            for (int l = 0; l < 2; ++l) {
                const std::string tag = std::string(form_name[form]) + "/" + lm_name[l];
                Tally raw, gated, gin, gout;
                int abst = 0, abst_out = 0, n_out = 0;
                for (const auto& g : gt)
                    if (g.form == form && g.lm == l) {
                        raw.add(g.grade);
                        const int gr = gated_answer(g, final_gate[form][l]) ? g.grade : 1;
                        gated.add(gr);
                        (g.in_slice ? gin : gout).add(gr);
                        const bool ab = g.grade == 1 && !g.ans.text.empty();
                        abst += ab;
                        if (!g.in_slice) {
                            ++n_out;
                            abst_out += ab;
                        }
                    }
                print_tally("TEST_RAW " + tag + " all", raw);
                print_tally("TEST_GATED " + tag + " all", gated);
                print_tally("TEST_GATED " + tag + " in_slice", gin);
                print_tally("TEST_GATED " + tag + " not_in_slice", gout);
                std::cout << "TEST_ABSTAIN " << tag << " abstained " << abst << " not_in_slice " << abst_out << "/" << n_out
                          << " precision_nis " << (abst ? static_cast<double>(abst_out) / abst : 0.0) << " recall_nis "
                          << (n_out ? static_cast<double>(abst_out) / n_out : 0.0) << "\n";
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
    if (!h.ok) {
        std::cerr << "selftest: holdout scoring failed\n";
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
            gs[static_cast<std::size_t>(i)].ans.mean_lp = mlp[i];
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
    std::cout << "selftest ok\n";
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: cyphalm_trie <bpc|chat|probe|context8k|sample|selftest> [wiki] [dolly] [options]\n"
                 "  bpc [wiki] [--negative] [--no-guard]\n"
                 "  chat [wiki] [dolly] [--prompt TEXT] [--no-repeat]\n"
                 "  probe [wiki] [dolly]\n"
                 "  context8k [wiki] [dolly]\n"
                 "  sample <wiki> <out_dev> [out_fresh] [--prompts 250] [--bytes 600]\n"
                 "  qa [wiki] [dolly] --dev DEV.tsv [--test TEST.tsv] [--out ROWS.tsv] [--base A0_ROWS.tsv]\n"
                 "     [--dk DK.tsv --dk-ratio R] [--fam none|before|after]\n"
                 "  dktext [wiki] [dolly] --dev DEV.tsv --test TEST.tsv --out DK.tsv [--from 8388608] [--max-rows N]\n"
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
        cmd != "dktext") {
        usage();
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
    else if (cmd == "qa") {
        if (a.dev.empty() || (a.fam != "none" && a.fam != "before" && a.fam != "after") || a.dk_ratio < 0 ||
            a.dk_ratio >= 1) {
            usage();
            return 2;
        }
        rc = cmd_qa(a, QaOpts{a.dev, a.test, a.out, a.base, a.dk, a.dk_ratio, a.fam});
    } else if (cmd == "dktext") {
        if (a.dev.empty() || a.test.empty() || a.out.empty()) {
            usage();
            return 2;
        }
        rc = cmd_dktext(a, a.dev, a.test, a.out, a.from, a.max_rows);
    }
    std::cerr << "seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return rc;
}
