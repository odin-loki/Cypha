// k5_probe: where today's writer breaks (docs/reports/WHERE_IT_BREAKS.md). Odin Loch, 2026-11-13.
//
// Reads K5 exactly as `cyphalm_trie sample --k5` builds today's default T8 (GENERATION_PLAN §13.71): K5's counts from
// fuzzy/train.txt plus the 337 MB more-text (vocabulary and classes from the first 7 MB), K1's retrieval overlay
// (k=1, rel=0.5), the LDA topic term in the gate (100 topics, weight 1), the 16-draft gate, T 0.8, nucleus 0.8, the
// repeat ban and the 12-word copy cap. It changes nothing in the writer; it only reads the model.
//
// Three modes, one model build:
//   map   per word of a given text after its prompt: log2 P under K5 with the overlay (what is sampled from) and
//         without it, which order answered and how far back the context matched (main counts and overlay), the
//         word's rank, whether it was inside T8's nucleus (T 0.8, nucleus 0.8, after the bans), and, when the
//         decoder's learner is built (K5_MIXDEC, wip/k5-mixdec), the learner's log2 P of the word's bytes, primed on
//         the retrieved article and the prompt, and its context-free prior.
//   gate  for T8's own text: at each sentence, the 16 drafts the writer drew (K5::draft with the writer's seed and
//         stream numbers), which of them T8 kept, the plain gate's choice recomputed (prompts the planner leaves to
//         K5), and the kept draft's fluency rank among the 16.
//   beam  the most probable output: word-level beam search under K5 with the overlay (widths given; width 1 is
//         greedy), with the repeat ban and the copy cap, no sampling; with the learner, a beam under K5 x the
//         learner's context ratio (MR's form: p_K5 (p_mix(w | context) / p_mix(w))^r over the 16 most probable words).
//
// Input (--items): TSV, one text per line: item, kind, prompt, text (\n, \t and \\ escaped). The prompt is read as
// sample reads GEN-A's; the text is the continuation exactly as written (a GEN line, or " " + the human text).
//
// Build (no learner): g++ -std=c++23 -O2 -I native/include -I native/third_party native/tools/lm/k5_probe.cpp -o k5_probe
// With the learner, put wip/k5-mixdec's lm/decoder/learner.hpp and online/online_ngram.hpp on an include path before
// native/include (it is unmerged); the probe sees them by __has_include.

#include "cypha/lm/word_kn.hpp"

#if __has_include("cypha/lm/decoder/learner.hpp")
#define K5P_LEARNER 1
#include "cypha/lm/decoder/learner.hpp"
#include "cypha/lm/packed_trie.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace wkn = cypha::cyphalm::wkn;

namespace {

struct Item {
    std::string id, kind, prompt, text;
};

std::string unesc(const std::string& s) {
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
        } else {
            o += s[i];
        }
    }
    return o;
}
std::string esc(std::string_view s) {
    std::string o;
    for (const char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\\') o += "\\\\";
        else o += c;
    }
    return o;
}
std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> f;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find('\t', a);
        f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return f;
}

bool is_stop(std::string_view t) { return t == "." || t == "!" || t == "?" || t == "\n"; }

struct Args {
    std::string k5 = "/mnt/d/cypha_llm/fuzzy/train.txt";
    std::string more = "/mnt/d/cypha_llm/k5scale/clean/train_all.txt";
    std::string disc_cache = "/mnt/d/cypha_llm/disc/cache";
    std::string items, out = "probe";
    std::string modes = "map,gate,beam";
    std::vector<int> widths = {1, 8, 16};
    int mix_width = 8;
    double mix_r = 0.3;
    std::size_t beam_bytes = 460;
    int beam_max_words = 160;
    std::uint64_t seed0 = wkn::kSeed;  // sample's seed: kSeed + the prompt's index (temperature index 0)
    unsigned threads = 6;
    double temperature = 0.8;
    // the learner (K5_MIXDEC's files)
    std::string trie = "/mnt/d/cypha_llm/corpus/trie/trie_1tb.pk";
    std::string lstm = "/mnt/d/cypha_llm/mixdec/clean64.lstm";
    std::string generic = "/mnt/d/cypha_llm/k5scale/clean/train_all.txt";
    std::size_t generic_bytes = 524288, art_bytes = 8192;
    bool learner = true;
};

#ifdef K5P_LEARNER
namespace dc = cypha::cyphalm::decoder;
struct LearnerKit {
    dc::Models M;
    cypha::cyphalm::trie::PackedModel pm;
    std::unique_ptr<cypha::cyphalm::bkn::ByteKN> kn;
    std::unique_ptr<cypha::online::ByteLSTM> lstm0;
    std::unique_ptr<dc::Learner> base;
};
bool word_byte(std::uint8_t c) { return c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
double end_lp(const double* P) {
    double e = 0;
    for (int s = 0; s < 256; ++s)
        if (!word_byte(static_cast<std::uint8_t>(s))) e += P[s];
    return std::log2(std::max(e, 1e-300));
}
// log2 P(piece's bytes [+ the word ends]) after L's text; L is rewound.
double piece_lp(dc::Learner& L, std::string_view piece, bool with_end) {
    const auto mk = L.mark();
    double lb = 0, P[256];
    for (const char c : piece) {
        L.dist(P);
        lb += std::log2(std::max(P[static_cast<std::uint8_t>(c)], 1e-300));
        L.read(static_cast<std::uint8_t>(c));
    }
    if (with_end && !piece.empty() && word_byte(static_cast<std::uint8_t>(piece.back()))) {
        L.dist(P);
        lb += end_lp(P);
    }
    L.rewind(mk);
    return lb;
}
#endif

// The retrieved article's text as the decoder primes it (K5_MIXDEC: K5's words, <unk> left out).
std::string article(const wkn::K5& m, const wkn::K5::State& s, const wkn::Options& o, std::string_view prompt) {
    std::string art;
    if (!o.ret.on() || !m.retrieval()) return art;
    for (const std::uint32_t d : m.retrieval()->top(s.ctx, o.ret, m.kn().index(), prompt)) {
        const auto [a, b] = m.retrieval()->span(d);
        const auto& ids = m.kn().index().text();
        std::string prev;
        bool hp = false;
        for (std::size_t i = a; i < b; ++i) {
            if (ids[i] == wkn::kUnk || ids[i] == wkn::kBos) continue;
            art += wkn::piece(m.vocab()[ids[i]], prev, hp);
            prev = m.vocab()[ids[i]], hp = true;
        }
        art += "\n\n";
    }
    return art;
}

// The text's tokens with the bytes each adds (from the end of the token before to its own end), so a replay writes
// exactly the text's bytes.
struct Tok {
    std::string tok, bytes;
    std::size_t end = 0;  // offset in the text after it
};
std::vector<Tok> text_tokens(const std::string& text) {
    std::vector<Tok> v;
    std::size_t last = 0;
    wkn::tokenize(text, [&](std::string_view t) {
        const std::size_t at = static_cast<std::size_t>(t.data() - text.data());
        Tok k;
        k.tok.assign(t);
        k.end = at + t.size();
        k.bytes = text.substr(last, k.end - last);
        last = k.end;
        v.push_back(std::move(k));
    });
    return v;
}

// One step of a replay: the state reads token k as the writer's step_word would have written it.
void replay_step(const wkn::K5& m, const wkn::Options& o, wkn::K5::State& s, const Tok& k) {
    const std::uint32_t y = m.id(k.tok);
    s.out += k.bytes;
    s.rep.add(k.bytes);
    wkn::K5::learn(s, y, o);
    s.prev = k.tok;
    s.has_prev = true;
    ++s.step;
}

struct WordRow {
    std::string tok;
    std::uint32_t id = 0;
    double lp_full = 0, lp_k5 = 0, lp_t = 0;
    int ctx_ord = 1, ans_ord = 1, ov_ctx = 0, ov_ans = 0;
    std::size_t rank = 0, ncand = 0, nuc_k = 0;
    int in_cand = 0, rep_b = 0, cap_b = 0, nuc_in = 0;
    double nuc_before = 0;  // the kept mass (normalised, after temperature) above the word
    std::string top1;
    double lp_top1 = 0;
    double lp_mix = 0, lp_prior = 0;
    int has_mix = 0;
    int sent = 0;
};

// Per word: the measures of WHERE_IT_BREAKS.md (map).
void map_item(const wkn::K5& m, const wkn::Options& o, const Item& it, std::vector<WordRow>& rows, std::string& info, const Args& A,
              void* learner_kit) {
    const std::size_t V = m.vocab().size();
    wkn::K5::State s = m.start(it.prompt, o);
    std::vector<double> p(V), pk(V), tmp, pt(V);
    std::vector<std::uint32_t> ord(V);
    const auto toks = text_tokens(it.text);
    const wkn::NgramSet cs = m.cap_set();
    const wkn::Overlay* rov = s.rov.get();
    const std::string art = article(m, s, o, it.prompt);
#ifdef K5P_LEARNER
    auto* kit = static_cast<LearnerKit*>(learner_kit);
    std::unique_ptr<dc::Learner> com, neu;
    std::map<std::string, double> prior_cache;
    if (kit) {
        com = std::make_unique<dc::Learner>(*kit->base);
        neu = std::make_unique<dc::Learner>(*kit->base);
        com->own(), neu->own();
        com->learn("\n\n");
        com->learn(std::string_view(art).substr(0, std::min(art.size(), A.art_bytes)));
        com->learn("\n\n");
        com->learn(it.prompt);
        com->trim();
    }
#else
    (void)learner_kit;
#endif
    int sent = 0;
    for (const Tok& k : toks) {
        WordRow r;
        r.tok = k.tok;
        r.sent = sent;
        const std::uint32_t y = m.id(k.tok);
        r.id = y;
        m.model_dist(s, o, p.data(), tmp);
        // K5 without the overlay (the class floor kept): tmp holds the class distribution after model_dist.
        m.kn().dist(s.ctx.data(), s.ctx.size(), pk.data(), nullptr, m.classes().on() ? tmp.data() : nullptr);
        const double zf = 1 - p[wkn::kUnk] - p[wkn::kBos], zk = 1 - pk[wkn::kUnk] - pk[wkn::kBos];
        // an unknown word: <unk>'s own mass (as cont_bits counts it)
        r.lp_full = std::log2(std::max(y == wkn::kUnk ? p[y] : p[y] / zf, 1e-300));
        r.lp_k5 = std::log2(std::max(y == wkn::kUnk ? pk[y] : pk[y] / zk, 1e-300));
        // which order answered: the main counts' walk (orders 2 up), and the overlay's contexts
        {
            int found = 0, ans = 1;
            m.kn().walk(s.ctx.data(), s.ctx.size(), [&](double, const std::uint16_t* ids, const double*, std::size_t mm) {
                ++found;
                const std::uint16_t* f = std::lower_bound(ids, ids + mm, static_cast<std::uint16_t>(y));
                if (f != ids + mm && *f == y) ans = found + 1;
            });
            r.ctx_ord = found + 1, r.ans_ord = y == wkn::kUnk ? 0 : ans;
            if (rov) {
                for (int kk = 2; kk <= rov->order; ++kk) {
                    const std::size_t L = static_cast<std::size_t>(kk - 1);
                    if (s.ctx.size() < L) break;
                    wkn::Key key = 0;
                    for (std::size_t i = 0; i < L; ++i) key |= wkn::Key{s.ctx[s.ctx.size() - 1 - i]} << (16 * i);
                    const auto* oc = rov->find(kk, key);
                    if (!oc) continue;
                    r.ov_ctx = kk;
                    for (const auto& [w, d] : oc->e)
                        if (w == y) r.ov_ans = kk;
                }
            }
        }
        // rank and the most probable word
        {
            std::size_t above = 0;
            std::uint32_t best = 2;
            for (std::uint32_t x = 2; x < V; ++x) {
                if (p[x] > p[y] && x != y) ++above;
                if (p[x] > p[best]) best = x;
            }
            r.rank = above + 1;
            r.top1 = m.vocab()[best];
            r.lp_top1 = std::log2(std::max(p[best] / zf, 1e-300));
        }
        // T8's nucleus at this step (full_pick: temperature, candidates to cand_mass, the bans, the nucleus)
        {
            double z = 0;
            for (std::size_t x = 0; x < V; ++x) z += pt[x] = (x < 2) ? 0.0 : std::pow(p[x], 1.0 / A.temperature);
            for (double& v : pt) v /= z;
            std::iota(ord.begin(), ord.end(), 0u);
            const std::size_t mc = std::min(o.cand_max, V - 2);
            std::partial_sort(ord.begin(), ord.begin() + static_cast<std::ptrdiff_t>(mc), ord.end(),
                              [&](std::uint32_t a, std::uint32_t b) { return pt[a] != pt[b] ? pt[a] > pt[b] : a < b; });
            std::vector<std::uint32_t> cand;
            double cum = 0;
            for (std::size_t i = 0; i < mc; ++i) {
                cand.push_back(ord[i]);
                cum += pt[ord[i]];
                if (cum >= o.cand_mass) break;
            }
            wkn::NgramSet::Index::Range hp;
            const bool capped = o.cap && cs.prefix(s.ctx, hp);
            std::vector<char> rb(cand.size()), cb(cand.size());
            bool any_kept = false;
            for (std::size_t i = 0; i < cand.size(); ++i) {
                const std::string pc = wkn::piece(m.vocab()[cand[i]], s.prev, s.has_prev);
                cb[i] = capped && cs.completes(hp, cand[i]);
                rb[i] = s.rep.banned(pc);
                any_kept = any_kept || (!rb[i] && !cb[i]);
                if (cand[i] == y) r.in_cand = 1, r.rep_b = rb[i], r.cap_b = cb[i];
            }
            if (!any_kept) std::fill(rb.begin(), rb.end(), 0);
            double zkp = 0;
            for (std::size_t i = 0; i < cand.size(); ++i)
                if (!rb[i] && !cb[i]) zkp += pt[cand[i]];
            double c2 = 0, zn = 0;
            std::size_t kk = 0;
            for (std::size_t i = 0; i < cand.size(); ++i) {
                if (rb[i] || cb[i]) continue;
                if (cand[i] == y) r.nuc_before = c2, r.nuc_in = 1;
                c2 += pt[cand[i]] / zkp;
                zn += pt[cand[i]];
                ++kk;
                if (c2 >= o.nucleus) break;
            }
            r.ncand = cand.size(), r.nuc_k = kk;
            if (!r.nuc_in) r.nuc_before = c2;  // outside: at least the whole nucleus is above it
            r.lp_t = std::log2(std::max(r.nuc_in ? pt[y] / zn : pt[y], 1e-300));
        }
#ifdef K5P_LEARNER
        if (com) {
            double P[256], lb = 0;
            for (const char c : k.bytes) {
                com->dist(P);
                lb += std::log2(std::max(P[static_cast<std::uint8_t>(c)], 1e-300));
                com->read(static_cast<std::uint8_t>(c));
            }
            com->trim();
            r.lp_mix = lb;
            auto itp = prior_cache.find(k.bytes);
            if (itp == prior_cache.end()) itp = prior_cache.emplace(k.bytes, piece_lp(*neu, k.bytes, false)).first;
            r.lp_prior = itp->second;
            r.has_mix = 1;
        }
#endif
        rows.push_back(r);
        replay_step(m, o, s, k);
        if (is_stop(k.tok)) ++sent;
    }
    char b[256];
    std::snprintf(b, sizeof b, "%zu\t%d\t%d", toks.size(), s.out == it.text ? 1 : 0, rov ? 1 : 0);
    info = it.id + "\t" + it.kind + "\t" + b + "\t" + esc(art.substr(0, 160));
}

// The gate: the 16 drafts at each sentence of T8's text, drawn as the writer drew them.
struct GateRow {
    int sent = 0;
    std::string path;  // k5 (plain gate) | disc (the discourse writer's free sentence) | fact (no draft matched)
    std::uint64_t no = 0;
    int match = -1, pick = -1, rank_flu = 0, n = 0;
    double flu_kept = 0, flu_best = 0, flu_mean = 0, minlp_kept = 0, zp_kept = 0, zt_kept = 0, words_kept = 0;
    std::string kept, most_fluent;
};
void gate_item(const wkn::K5& m, const wkn::Options& o, const Item& it, std::uint64_t seed, std::vector<GateRow>& rows) {
    wkn::K5::State s = m.start(it.prompt, o);
    const auto toks = text_tokens(it.text);
    const std::vector<double> assoc = m.prompt_assoc(s.ctx);
    const std::vector<double> tlr = m.topic_lr(it.prompt);
    const int K = o.gate_k;
    std::vector<wkn::K5::State> dr(static_cast<std::size_t>(K));
    std::vector<std::vector<std::uint32_t>> words(dr.size());
    std::vector<std::vector<double>> lps(dr.size());
    std::size_t ti = 0;  // the next token of the text
    std::string path;    // decided at the first sentence
    std::uint64_t sent_k5 = 0, free_no = 1000;
    int sent = 0;
    const auto zs = [](std::vector<double> x) {
        double mu = 0, v = 0;
        for (const double a : x) mu += a;
        mu /= static_cast<double>(x.size());
        for (const double a : x) v += (a - mu) * (a - mu);
        const double sd = std::sqrt(v / static_cast<double>(x.size()));
        for (double& a : x) a = sd > 1e-12 ? (a - mu) / sd : 0.0;
        return x;
    };
    while (ti < toks.size()) {
        const std::string rest = it.text.substr(s.out.size());
        // try the stream numbers the writer would use here
        std::vector<std::pair<std::string, std::uint64_t>> tries;
        if (path.empty() || path == "k5") tries.push_back({"k5", sent_k5});
        if (path.empty() || path == "disc") tries.push_back({"disc", free_no});
        int match = -1;
        std::string used;
        std::uint64_t no = 0;
        for (const auto& [pth, num] : tries) {
            for (int k = 0; k < K; ++k) {
                wkn::GenStats g;
                m.draft(s, seed, num, k, o, dr[static_cast<std::size_t>(k)], words[static_cast<std::size_t>(k)], lps[static_cast<std::size_t>(k)], g);
                const std::string d = dr[static_cast<std::size_t>(k)].out.substr(s.out.size());
                if (match < 0 && !d.empty() && (rest.compare(0, d.size(), d) == 0 || (rest.size() < d.size() && d.compare(0, rest.size(), rest) == 0)))
                    match = k;
            }
            if (match >= 0) {
                used = pth, no = num;
                break;
            }
        }
        GateRow r;
        r.sent = sent;
        if (match < 0) {
            // no draft matched: a planner's fact sentence (or a replay that left the writer's state); read to the next stop
            if (path.empty()) path = "disc";
            r.path = "fact";
            const std::size_t a = s.out.size();
            while (ti < toks.size()) {
                replay_step(m, o, s, toks[ti]);
                const bool stop = is_stop(toks[ti].tok);
                ++ti;
                if (stop) break;
            }
            r.kept = it.text.substr(a, s.out.size() - a);
            rows.push_back(r);
            ++sent;
            continue;
        }
        if (path.empty()) path = used;
        r.path = used, r.no = no, r.match = match, r.n = K;
        // the drafts' fluency (the gate's wf term, which is 0 by default) and the plain gate's terms
        std::vector<double> tf(dr.size()), tp(dr.size()), tt(dr.size()), mk(dr.size());
        for (std::size_t k = 0; k < dr.size(); ++k) {
            double a = 0, pm = 0, to = 0;
            for (std::size_t i = 0; i < words[k].size(); ++i) a += lps[k][i], pm += assoc[words[k][i]], to += tlr.empty() ? 0 : tlr[words[k][i]];
            const double n = static_cast<double>(std::max<std::size_t>(words[k].size(), 1));
            tf[k] = a / n, tp[k] = pm / n, tt[k] = to / n;
            const std::string d = dr[k].out.substr(s.out.size());
            std::size_t q = 0;
            for (const char c : d) q += std::string_view("[]{}|=<>&*#").find(c) != std::string_view::npos;
            mk[k] = static_cast<double>(q) / static_cast<double>(std::max<std::size_t>(d.size(), 1));
        }
        const auto ztp = zs(tp), ztt = zs(tt), ztf = zs(tf);
        if (used == "k5") {
            int best = 0;
            double bs = 0;
            for (int k = 0; k < K; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const double sc = o.gate_wf * ztf[ku] + o.gate_wp * ztp[ku] + o.lda_w * ztt[ku] - o.gate_wm * mk[ku];
                if (k == 0 || sc > bs) bs = sc, best = k;
            }
            r.pick = best;
        }
        const auto mu = static_cast<std::size_t>(match);
        int rank = 1;
        std::size_t bestf = 0;
        double mean = 0;
        for (std::size_t k = 0; k < dr.size(); ++k) {
            if (tf[k] > tf[mu]) ++rank;
            if (tf[k] > tf[bestf]) bestf = k;
            mean += tf[k];
        }
        r.rank_flu = rank;
        r.flu_kept = tf[mu], r.flu_best = tf[bestf], r.flu_mean = mean / static_cast<double>(dr.size());
        r.minlp_kept = lps[mu].empty() ? 0 : *std::min_element(lps[mu].begin(), lps[mu].end());
        r.zp_kept = ztp[mu], r.zt_kept = ztt[mu], r.words_kept = static_cast<double>(words[mu].size());
        r.kept = dr[mu].out.substr(s.out.size());
        r.most_fluent = dr[bestf].out.substr(s.out.size());
        rows.push_back(r);
        // the state after the kept draft (exactly the writer's); the tokens it wrote are consumed
        const std::size_t nb = dr[mu].out.size();
        s = dr[mu];
        while (ti < toks.size() && toks[ti].end <= nb) ++ti;
        if (used == "k5") ++sent_k5;
        else ++free_no;
        ++sent;
        if (s.out.size() >= it.text.size()) break;
    }
}

// Beam search: the most probable continuation under K5 with the overlay (no sampling, no temperature, no nucleus),
// with the repeat ban and the copy cap; it stops when the best hypothesis has written beam_bytes bytes.
struct Hyp {
    wkn::K5::State s;
    double score = 0, lp = 0;  // score: what the beam ranks by; lp: K5's log2 P of the words
    std::vector<double> lps;
#ifdef K5P_LEARNER
    std::shared_ptr<dc::Learner> L;
#endif
};
struct BeamOut {
    std::string mode, text;
    double lp = 0;
    std::size_t words = 0;
    double min_lp = 0;
};
BeamOut beam_item(const wkn::K5& m, const wkn::Options& o, const Item& it, int W, const Args& A, void* learner_kit, bool mix) {
    const std::size_t V = m.vocab().size();
    std::vector<double> p(V), tmp;
    std::vector<std::uint32_t> ord(V);
    const wkn::NgramSet cs = m.cap_set();
    std::vector<Hyp> beam(1);
    beam[0].s = m.start(it.prompt, o);
#ifdef K5P_LEARNER
    auto* kit = static_cast<LearnerKit*>(learner_kit);
    std::unique_ptr<dc::Learner> neu;
    std::map<std::string, double> prior_cache;
    if (mix && kit) {
        auto L = std::make_shared<dc::Learner>(*kit->base);
        L->own();
        const std::string art = article(m, beam[0].s, o, it.prompt);
        L->learn("\n\n");
        L->learn(std::string_view(art).substr(0, std::min(art.size(), A.art_bytes)));
        L->learn("\n\n");
        L->learn(it.prompt);
        L->trim();
        beam[0].L = L;
        neu = std::make_unique<dc::Learner>(*kit->base);
        neu->own();
    }
#else
    (void)learner_kit;
    (void)mix;
#endif
    const std::size_t expand = mix ? 16 : static_cast<std::size_t>(std::max(2 * W, 4));
    for (int step = 0; step < A.beam_max_words; ++step) {
        if (beam[0].s.out.size() >= A.beam_bytes) break;
        struct Ext {
            std::size_t h;
            std::uint32_t w;
            double score, lp;
            std::string piece;
        };
        std::vector<Ext> ext;
        for (std::size_t h = 0; h < beam.size(); ++h) {
            const wkn::K5::State& s = beam[h].s;
            m.model_dist(s, o, p.data(), tmp);
            const double z = 1 - p[wkn::kUnk] - p[wkn::kBos];
            std::iota(ord.begin(), ord.end(), 0u);
            const std::size_t top = std::min<std::size_t>(400, V);
            std::partial_sort(ord.begin(), ord.begin() + static_cast<std::ptrdiff_t>(top), ord.end(),
                              [&](std::uint32_t a, std::uint32_t b) { return p[a] != p[b] ? p[a] > p[b] : a < b; });
            wkn::NgramSet::Index::Range hp;
            const bool capped = o.cap && cs.prefix(s.ctx, hp);
            std::size_t got = 0;
            for (std::size_t i = 0; i < top && got < expand; ++i) {
                const std::uint32_t w = ord[i];
                if (w < 2) continue;
                std::string pc = wkn::piece(m.vocab()[w], s.prev, s.has_prev);
                if (s.rep.banned(pc) || (capped && cs.completes(hp, w))) continue;
                const double lp = std::log2(std::max(p[w] / z, 1e-300));
                double sc = beam[h].score + lp;
#ifdef K5P_LEARNER
                if (mix && beam[h].L) {
                    auto itp = prior_cache.find(pc);
                    if (itp == prior_cache.end()) itp = prior_cache.emplace(pc, piece_lp(*neu, pc, true)).first;
                    sc += A.mix_r * (piece_lp(*beam[h].L, pc, true) - itp->second);
                }
#endif
                ext.push_back({h, w, sc, lp, std::move(pc)});
                ++got;
            }
        }
        if (ext.empty()) break;
        std::stable_sort(ext.begin(), ext.end(), [](const Ext& a, const Ext& b) { return a.score > b.score; });
        std::vector<Hyp> nb;
        for (const Ext& e : ext) {
            if (static_cast<int>(nb.size()) >= W) break;
            Hyp hy;
            hy.s = beam[e.h].s;
            hy.score = e.score, hy.lp = beam[e.h].lp + e.lp;
            hy.lps = beam[e.h].lps;
            hy.lps.push_back(e.lp);
            hy.s.out += e.piece;
            hy.s.rep.add(e.piece);
            wkn::K5::learn(hy.s, e.w, o);
            hy.s.prev = m.vocab()[e.w];
            hy.s.has_prev = true;
            ++hy.s.step;
#ifdef K5P_LEARNER
            if (beam[e.h].L) {
                hy.L = std::make_shared<dc::Learner>(*beam[e.h].L);
                hy.L->read(e.piece);
                hy.L->trim();
            }
#endif
            nb.push_back(std::move(hy));
        }
        beam = std::move(nb);
    }
    BeamOut r;
    r.text = beam[0].s.out;
    r.lp = beam[0].lp;
    r.words = beam[0].lps.size();
    r.min_lp = beam[0].lps.empty() ? 0 : *std::min_element(beam[0].lps.begin(), beam[0].lps.end());
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    Args A;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << k << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--k5") A.k5 = next();
        else if (k == "--more") A.more = next();
        else if (k == "--disc-cache") A.disc_cache = next();
        else if (k == "--items") A.items = next();
        else if (k == "--out") A.out = next();
        else if (k == "--modes") A.modes = next();
        else if (k == "--widths") {
            A.widths.clear();
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) A.widths.push_back(std::atoi(t.c_str()));
        } else if (k == "--mix-width") A.mix_width = std::atoi(next().c_str());
        else if (k == "--mix-r") A.mix_r = std::atof(next().c_str());
        else if (k == "--beam-bytes") A.beam_bytes = std::strtoull(next().c_str(), nullptr, 10);
        else if (k == "--threads") A.threads = static_cast<unsigned>(std::max(1, std::atoi(next().c_str())));
        else if (k == "--no-learner") A.learner = false;
        else if (k == "--trie") A.trie = next();
        else if (k == "--lstm") A.lstm = next();
        else {
            std::cerr << "unknown flag " << k << "\n";
            return 2;
        }
    }
    if (A.items.empty()) {
        std::cerr << "usage: k5_probe --items ITEMS.tsv --out PREFIX [--modes map,gate,beam] [--widths 1,8,16] [--threads N]\n";
        return 2;
    }
    std::vector<Item> items;
    {
        std::ifstream in(A.items);
        std::string line;
        while (std::getline(in, line)) {
            const auto f = split_tab(line);
            if (f.size() < 4) continue;
            items.push_back({f[0], f[1], unesc(f[2]), unesc(f[3])});
        }
    }
    std::cerr << "k5_probe: " << items.size() << " texts\n";
    const auto has = [&](const char* mo) { return A.modes.find(mo) != std::string::npos; };

    // today's default, as cyphalm_trie sample --k5 builds it (GENERATION_PLAN §13.48, §13.53, §13.66, §13.71)
    wkn::Options o;
    o.more = A.more;
    std::string err;
    if (!wkn::apply_flag("--k5-retrieve", "k=1,rel=0.5", o, err)) {
        std::cerr << err << "\n";
        return 2;
    }
    o.lda_k = 100, o.lda_w = 1, o.disc_text = o.more;
    if (std::ifstream(A.disc_cache + "/lda_train_all.txt_K100_it200_t1.bin")) o.disc_cache = A.disc_cache;
    o.temperature = A.temperature;
    o.bytes = 600;
    wkn::K5 m;
    if (!m.load(A.k5, 12, err, &o)) {
        std::cerr << "k5: " << err << "\n";
        return 1;
    }
    std::fprintf(stderr, "k5_probe: K5 built (vocab %zu, %zu tokens, unk %.4f) in %.1f s; options%s\n", m.vocab().size(), m.tokens(),
                 static_cast<double>(m.unk()) / static_cast<double>(std::max<std::size_t>(m.tokens(), 1)), m.build_seconds(), o.describe().c_str());

    void* kitp = nullptr;
#ifdef K5P_LEARNER
    LearnerKit kit;
    if (A.learner) {
        const auto t0 = std::chrono::steady_clock::now();
        kit.M.copy = false;  // as MR and MRJ (K5_MIXDEC: copy=0 in the ratio and the judge)
        cypha::cyphalm::trie::Config tc;
        kit.pm = cypha::cyphalm::trie::PackedModel::load(A.trie, kit.M.tw, tc);
        kit.M.trie = &kit.pm;
        {
            std::ifstream kin(A.k5, std::ios::binary);
            const std::string kt((std::istreambuf_iterator<char>(kin)), std::istreambuf_iterator<char>());
            cypha::cyphalm::bkn::Prune pr;  // kBknDefault "6:4:2:1"
            pr.from_order = 4, pr.min_ctx = 2, pr.drop_singletons = true;
            kit.kn = std::make_unique<cypha::cyphalm::bkn::ByteKN>(reinterpret_cast<const std::uint8_t*>(kt.data()), kt.size(), 6, pr);
            kit.M.kn = kit.kn.get();
        }
        kit.lstm0 = cypha::online::ByteLSTM::load_file(A.lstm);
        if (!kit.lstm0) {
            std::cerr << "cannot read " << A.lstm << "\n";
            return 1;
        }
        std::string gen(A.generic_bytes, '\0');
        {
            std::ifstream gin(A.generic, std::ios::binary);
            gin.read(gen.data(), static_cast<std::streamsize>(gen.size()));
            gen.resize(static_cast<std::size_t>(gin.gcount()));
        }
        dc::Learner L(kit.M, kit.lstm0.get());
        const double b = L.learn(gen);
        kit.base = std::make_unique<dc::Learner>(L);
        kit.base->trim();
        std::fprintf(stderr, "k5_probe: learner ready (copy off), base primed on %zu bytes at %.4f bits a byte, %.1f s\n", gen.size(),
                     gen.empty() ? 0.0 : -b / static_cast<double>(gen.size()),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        kitp = &kit;
    }
#endif

    std::vector<std::vector<WordRow>> maps(items.size());
    std::vector<std::string> infos(items.size());
    std::vector<std::vector<GateRow>> gates(items.size());
    std::vector<std::vector<BeamOut>> beams(items.size());
    std::atomic<std::size_t> next{0};
    std::mutex mu;
    const auto t0 = std::chrono::steady_clock::now();
    const auto work = [&] {
        for (std::size_t j; (j = next++) < items.size();) {
            const Item& it = items[j];
            if (has("map")) map_item(m, o, it, maps[j], infos[j], A, kitp);
            const std::uint64_t seed = A.seed0 + std::strtoull(it.id.c_str(), nullptr, 10);
            if (has("gate") && it.kind == "T8") gate_item(m, o, it, seed, gates[j]);
            if (has("beam") && it.kind == "human") {  // one beam set per prompt (on the human item: the prompt is the same)
                for (const int W : A.widths) {
                    BeamOut b = beam_item(m, o, it, W, A, kitp, false);
                    b.mode = W == 1 ? "greedy" : "beam" + std::to_string(W);
                    beams[j].push_back(std::move(b));
                }
#ifdef K5P_LEARNER
                if (kitp && A.mix_width > 0) {
                    BeamOut b = beam_item(m, o, it, A.mix_width, A, kitp, true);
                    b.mode = "beam" + std::to_string(A.mix_width) + "_ratio";
                    beams[j].push_back(std::move(b));
                }
#endif
            }
            std::lock_guard<std::mutex> lk(mu);
            std::fprintf(stderr, "k5_probe: %zu/%zu %s %s (%.0f s)\n", j + 1, items.size(), it.id.c_str(), it.kind.c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < A.threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    if (has("map")) {
        std::ofstream f(A.out + ".map.tsv");
        f << "item\tkind\twi\tsent\ttok\tunk\tlp_full\tlp_k5\tctx_ord\tans_ord\tov_ctx\tov_ans\trank\ttop1\tlp_top1\tncand\tin_cand\trep_b\tcap_b\tnuc_k\tnuc_in\tnuc_before\tlp_t\thas_mix\tlp_mix\tlp_prior\n";
        for (std::size_t j = 0; j < items.size(); ++j)
            for (std::size_t i = 0; i < maps[j].size(); ++i) {
                const WordRow& r = maps[j][i];
                char b[512];
                std::snprintf(b, sizeof b, "%d\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t%zu", r.id == wkn::kUnk ? 1 : 0, r.lp_full, r.lp_k5, r.ctx_ord, r.ans_ord, r.ov_ctx,
                              r.ov_ans, r.rank);
                char c[512];
                std::snprintf(c, sizeof c, "%.4f\t%zu\t%d\t%d\t%d\t%zu\t%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f", r.lp_top1, r.ncand, r.in_cand, r.rep_b, r.cap_b, r.nuc_k,
                              r.nuc_in, r.nuc_before, r.lp_t, r.has_mix, r.lp_mix, r.lp_prior);
                f << items[j].id << "\t" << items[j].kind << "\t" << i << "\t" << r.sent << "\t" << esc(r.tok) << "\t" << b << "\t" << esc(r.top1) << "\t" << c
                  << "\n";
            }
        std::ofstream g(A.out + ".info.tsv");
        g << "item\tkind\twords\treplay_ok\toverlay\tarticle_head\n";
        for (std::size_t j = 0; j < items.size(); ++j) g << infos[j] << "\n";
    }
    if (has("gate")) {
        std::ofstream f(A.out + ".gate.tsv");
        f << "item\tsent\tpath\tno\tmatch\tpick\tn\trank_flu\tflu_kept\tflu_best\tflu_mean\tminlp_kept\tzp_kept\tzt_kept\twords\tkept\tmost_fluent\n";
        for (std::size_t j = 0; j < items.size(); ++j)
            for (const GateRow& r : gates[j]) {
                char b[512];
                std::snprintf(b, sizeof b, "%d\t%s\t%llu\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.0f", r.sent, r.path.c_str(),
                              static_cast<unsigned long long>(r.no), r.match, r.pick, r.n, r.rank_flu, r.flu_kept, r.flu_best, r.flu_mean, r.minlp_kept,
                              r.zp_kept, r.zt_kept, r.words_kept);
                f << items[j].id << "\t" << b << "\t" << esc(r.kept) << "\t" << esc(r.most_fluent) << "\n";
            }
    }
    if (has("beam")) {
        std::ofstream f(A.out + ".beam.tsv");
        f << "item\tmode\twords\tlp\tbits_per_word\tmin_lp\ttext\n";
        for (std::size_t j = 0; j < items.size(); ++j)
            for (const BeamOut& b : beams[j]) {
                char c[256];
                std::snprintf(c, sizeof c, "%zu\t%.3f\t%.4f\t%.3f", b.words, b.lp, b.words ? -b.lp / static_cast<double>(b.words) : 0.0, b.min_lp);
                f << items[j].id << "\t" << b.mode << "\t" << c << "\t" << esc(b.text) << "\n";
            }
    }
    std::fprintf(stderr, "k5_probe: done in %.0f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
