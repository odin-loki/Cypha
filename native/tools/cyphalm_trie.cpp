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
//   selftest
//        packed_trie.hpp self_test plus a golden BPC on a 40,000-byte built-in corpus
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
#include <map>

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
    if (cmd != "bpc" && cmd != "chat" && cmd != "probe" && cmd != "context8k" && cmd != "sample") {
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
    std::cerr << "seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return rc;
}
