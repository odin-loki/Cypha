// k5_structure: K5's structure experts and their meta-mixer (docs/reports/K5_STRUCTURE.md, GENERATION_PLAN §13.33).
//
//   k5_structure stats    --k5 TRAIN --text FILE --from MB --to MB --out STATS
//       The experts' statistics from the documents of FILE (one per line) that start in [from, to) MB.
//   k5_structure discourse --text FILE --from MB --to MB --out STATS [--lda-mb 50] [--threads 4]
//       wip/k5-discourse's LDA (100 topics, on at most the first --lda-mb MB) and entity grid, as STATS.lda, STATS.grid.
//   k5_structure drafts   --k5 TRAIN --text FILE --from MB --to MB --out DRAFTS [--n 8] [--max B] [--threads 8]
//       Boundaries (a 1-sentence prompt, then up to 6 sentences) and, at each, the real next sentence and N K5 drafts.
//       Targets must be in K5's vocabulary and pass K5's repeat ban and copy cap. Lines: B, then C (context sentences),
//       then X (candidates, the real one first): "X k5lp pmi tokens...".
//   k5_structure features --k5 TRAIN --stats STATS --in DRAFTS --out FEAT
//       Every candidate's features (experts.hpp), the weight-set selectors and the pseudo target (P signal).
//   k5_structure run      --train FEAT --test FEAT --stats STATS --out PREFIX [--signal R|P]
//       Every arm: each expert alone, uniform, M1-M4, L0, the ablations. Sentence bits [95%], top-1. Writes
//       PREFIX.M1.model ... PREFIX.M4.model (for sample --k5-struct) and PREFIX.tsv (every arm's held-out losses' mean).
//       --drop EXPERT[,EXPERT]: the mix arms leave those experts out. --max-train N: the first N shuffled items.
//   k5_structure selftest
//
// Author: Odin Loch

#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/structure/mixer.hpp"
#include "cypha/lm/structure/writer.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace st = cypha::cyphalm::structure;
namespace wkn = cypha::cyphalm::wkn;

namespace {

std::map<std::string, std::string> parse(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    return a;
}
std::string get(const std::map<std::string, std::string>& a, const std::string& k, const std::string& d = "") {
    const auto it = a.find(k);
    return it == a.end() ? d : it->second;
}
double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
long peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string l;
    while (std::getline(f, l))
        if (l.rfind("VmHWM:", 0) == 0) return std::atol(l.c_str() + 6) / 1024;
    return 0;
}

// The documents of path that start in [from, to) bytes.
std::string read_docs(const std::string& path, double from_mb, double to_mb, std::size_t* ndocs = nullptr) {
    std::ifstream in(path, std::ios::binary);
    const std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto lo = static_cast<std::size_t>(from_mb * 1e6), hi = static_cast<std::size_t>(to_mb * 1e6);
    std::string out;
    std::size_t n = 0;
    for (std::size_t a = 0; a < all.size();) {
        std::size_t b = all.find('\n', a);
        if (b == std::string::npos) b = all.size();
        if (a >= lo && a < hi && b > a) out.append(all, a, b - a), out += '\n', ++n;
        if (a >= hi) break;
        a = b + 1;
    }
    if (ndocs) *ndocs = n;
    return out;
}

bool load_k5(const std::string& path, wkn::K5& m, const wkn::Options& o) {
    std::string err;
    if (!m.load(path, 12, err, &o)) {
        std::cerr << "--k5: " << err << "\n";
        return false;
    }
    std::fprintf(stderr, "k5: %zu docs, %zu tokens, vocab %zu, built in %.1f s\n", m.docs(), m.tokens(), m.vocab().size(), m.build_seconds());
    return true;
}

std::string join_toks(const std::vector<std::string>& t) {
    std::string o;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (i) o += ' ';
        o += t[i];
    }
    return o;
}
std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> v;
    std::istringstream in(s);
    for (std::string t; in >> t;) v.push_back(t);
    return v;
}

// ---------------------------------------------------------------- stats

int cmd_stats(const std::map<std::string, std::string>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    wkn::K5 m;
    if (!load_k5(get(a, "--k5"), m, wkn::Options{})) return 1;
    std::size_t nd = 0;
    const std::string text = read_docs(get(a, "--text"), std::atof(get(a, "--from", "0").c_str()), std::atof(get(a, "--to", "1e9").c_str()), &nd);
    std::fprintf(stderr, "stats: %zu docs, %.1f MB from %s\n", nd, text.size() / 1e6, get(a, "--text").c_str());
    st::Stats s;
    s.build(text, m);
    if (!s.save(get(a, "--out"))) {
        std::cerr << "cannot write " << get(a, "--out") << "\n";
        return 1;
    }
    std::fprintf(stderr, "stats: wrote %s (%.0f MB est.) in %.1f s, peak RSS %ld MB\n", get(a, "--out").c_str(), s.bytes() / 1e6, secs_since(t0), peak_rss_mb());
    return 0;
}

// ---------------------------------------------------------------- discourse (wip/k5-discourse's models)

int cmd_discourse(const std::map<std::string, std::string>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    const double from = std::atof(get(a, "--from", "0").c_str()), to = std::atof(get(a, "--to", "1e9").c_str());
    const unsigned th = static_cast<unsigned>(std::atoi(get(a, "--threads", "4").c_str()));
    const std::string out = get(a, "--out");
    const std::string text = read_docs(get(a, "--text"), from, to);
    const auto tok = [](std::string_view line, auto&& emit) { wkn::tokenize(line, emit); };
    cypha::cyphalm::eg::EntityGrid g;
    g.train_lines(text, tok, th);
    g.save(out + ".grid");
    std::fprintf(stderr, "discourse: grid on %zu docs, %zu sentences (%.0f s)\n", g.docs(), g.sentences(), secs_since(t0));
    const double lda_mb = std::atof(get(a, "--lda-mb", "50").c_str());
    const std::string lt = read_docs(get(a, "--text"), from, std::min(to, from + lda_mb));
    cypha::cyphalm::lda::LDA l;
    cypha::cyphalm::lda::Params p;
    p.threads = th;
    l.train_lines(lt, tok, p);
    l.save(out + ".lda");
    std::fprintf(stderr, "discourse: LDA %zu topics on %.1f MB, %zu content words, %zu tokens, %d sweeps, %u threads; total %.0f s, peak RSS %ld MB\n",
                 l.topics(), lt.size() / 1e6, l.vocab_size(), l.corpus_tokens(), p.iters, th, secs_since(t0), peak_rss_mb());
    return 0;
}

// ---------------------------------------------------------------- drafts

struct Boundary {
    std::size_t uid = 0;
    int j = 0;
    std::size_t prompt_chars = 0;
    int n_prompt = 1;
    std::vector<st::Sent> ctx;
    st::Sent target;
};

int cmd_drafts(const std::map<std::string, std::string>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    wkn::K5 m;
    const wkn::Options o;
    if (!load_k5(get(a, "--k5"), m, o)) return 1;
    std::size_t nd = 0;
    const std::string text = read_docs(get(a, "--text"), std::atof(get(a, "--from", "0").c_str()), std::atof(get(a, "--to", "1e9").c_str()), &nd);
    const int N = std::atoi(get(a, "--n", "8").c_str());
    const std::size_t maxb = std::strtoull(get(a, "--max", "1000000000").c_str(), nullptr, 10);
    const unsigned nth = static_cast<unsigned>(std::atoi(get(a, "--threads", "8").c_str()));
    // Windows and boundaries.
    std::vector<Boundary> bs;
    std::size_t uid = 0, cand_targets = 0, drop_unk = 0;
    for (std::size_t p = 0; p < text.size() && bs.size() < maxb;) {
        std::size_t q = text.find('\n', p);
        if (q == std::string::npos) q = text.size();
        const auto ss = st::split(wkn::tokens(std::string_view(text).substr(p, q - p)));
        p = q + 1;
        for (std::size_t w = 0; w < ss.size() && bs.size() < maxb;) {
            std::vector<st::Sent> prompt = {ss[w]};
            std::size_t k = w + 1;
            if (st::render(prompt).size() < 100 && k < ss.size()) prompt.push_back(ss[k++]);
            const std::size_t pc = st::render(prompt).size();
            for (int j = 0; j < 6 && k < ss.size() && bs.size() < maxb; ++j, ++k) {
                ++cand_targets;
                bool unk = false;
                for (const auto& t : ss[k].tok) unk = unk || m.id(t) == wkn::kUnk;
                if (unk) {
                    ++drop_unk, ++uid;
                    continue;
                }
                Boundary b;
                b.uid = uid++;
                b.j = j;
                b.prompt_chars = pc;
                b.n_prompt = static_cast<int>(prompt.size());
                b.ctx = prompt;
                for (std::size_t i = w + prompt.size(); i < k; ++i) b.ctx.push_back(ss[i]);
                b.target = ss[k];
                bs.push_back(std::move(b));
            }
            w = k;
        }
    }
    std::fprintf(stderr, "drafts: %zu docs, %zu targets, %zu with an unknown word dropped, %zu kept before the bans\n", nd, cand_targets, drop_unk, bs.size());
    // Drafts, in parallel; the bans on the real target.
    std::vector<std::string> out(bs.size());
    std::vector<char> keep(bs.size(), 0);
    std::atomic<std::size_t> next{0}, done{0}, ban_rep{0}, ban_cap{0};
    const auto work = [&] {
        std::vector<std::uint32_t> words;
        std::vector<double> lp;
        std::vector<double> p(m.vocab().size()), tmp;
        for (std::size_t i; (i = next++) < bs.size();) {
            const Boundary& b = bs[i];
            const std::string ctx_text = st::render(b.ctx);
            std::vector<st::Sent> pr(b.ctx.begin(), b.ctx.begin() + b.n_prompt);
            const wkn::K5::State sp = m.start(st::render(pr), o);
            const std::vector<double> assoc = m.prompt_assoc(sp.ctx);
            const wkn::K5::State s0 = m.start(ctx_text, o);
            // The real target: K5's numbers and the bans.
            wkn::K5::State s = s0;
            std::vector<std::uint32_t> ids;
            double tl = 0;
            bool ok = true;
            const wkn::NgramSet& cs = m.cap_set();
            for (const auto& t : b.target.tok) {
                const std::uint32_t w = m.id(t);
                const std::string pc = wkn::piece(t, s.prev, s.has_prev);
                if (s.rep.banned(pc)) {
                    ok = false, ++ban_rep;
                    break;
                }
                std::uint64_t hp = 0;
                if (cs.prefix(s.ctx, hp) && cs.has(wkn::fnv_step(hp, w))) {
                    ok = false, ++ban_cap;
                    break;
                }
                m.model_dist(s, o, p.data(), tmp);
                tl += std::log2(std::max(p[w] / std::max(1 - p[wkn::kUnk] - p[wkn::kBos], 1e-300), 1e-300));
                s.rep.add(pc);
                wkn::K5::learn(s, w, o);
                s.prev = t, s.has_prev = true;
                ids.push_back(w);
            }
            if (!ok) continue;
            std::ostringstream os;
            os.precision(7);
            os << "B " << b.uid << " " << b.j << " " << b.prompt_chars << " " << b.n_prompt << " " << b.ctx.size() << " " << N + 1 << "\n";
            for (const auto& c : b.ctx) os << "C " << (c.ended ? 1 : 0) << " " << join_toks(c.tok) << "\n";
            os << "X " << tl / static_cast<double>(std::max<std::size_t>(ids.size(), 1)) << " " << st::mean_assoc(assoc, ids) << " " << (b.target.ended ? 1 : 0) << " "
               << join_toks(b.target.tok) << "\n";
            wkn::GenStats g;
            const std::uint64_t seed = wkn::mix64(20261105ull ^ wkn::mix64(b.uid + 0xD4AF7));
            for (int k = 0; k < N; ++k) {
                wkn::K5::State c;
                m.draft(s0, seed, static_cast<std::uint64_t>(b.j), k, o, c, words, lp, g);
                double t = 0;
                for (const double x : lp) t += x;
                std::vector<std::string> tk;
                for (const auto w : words) tk.push_back(m.vocab()[w]);
                const bool ended = !tk.empty() && (tk.back() == "." || tk.back() == "!" || tk.back() == "?");
                os << "X " << t / static_cast<double>(std::max<std::size_t>(lp.size(), 1)) << " " << st::mean_assoc(assoc, words) << " " << (ended ? 1 : 0) << " "
                   << join_toks(tk) << "\n";
            }
            out[i] = os.str();
            keep[i] = 1;
            const std::size_t d = ++done;
            if (d % 5000 == 0) std::fprintf(stderr, "drafts: %zu boundaries in %.0f s\n", d, secs_since(t0));
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < nth; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    std::ofstream f(get(a, "--out"));
    std::size_t kept = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
        if (keep[i]) f << out[i], ++kept;
    std::fprintf(stderr, "drafts: %zu boundaries kept (%zu repeat-banned, %zu over the cap), %d drafts each, in %.1f s on %u threads, peak RSS %ld MB\n", kept,
                 ban_rep.load(), ban_cap.load(), N, secs_since(t0), nth, peak_rss_mb());
    return 0;
}

// ---------------------------------------------------------------- features

struct RawItem {
    int n = 0, target_p = 1;  // the pseudo target (a draft, 1..n-1)
    int sel[3] = {0, 0, 0};
    int j = 0;
    std::vector<float> f;  // n x kNFeat, raw
};

// Content words for the pseudo target.
std::unordered_set<std::string> content(const std::vector<std::string>& t) {
    static const std::unordered_set<std::string> stop = {
        "the", "a", "an", "of", "in", "and", "to", "is", "was", "for", "on", "as", "by", "with", "at", "from", "that", "which", "it", "its",
        "he", "she", "his", "her", "they", "their", "this", "these", "be", "been", "were", "are", "or", "but", "not", "also", "has", "had",
        "have", "who", "after", "into", "than", "one", "other", "first", "can", "may", "such", "most", "more", "all", "s", "there", "when",
        "would", "some", "only", "two", "new", "about", "over", "under", "between", "during", "up", "out", "him", "them", "then", "so"};
    std::unordered_set<std::string> c;
    for (const auto& w : t) {
        if (w.empty() || !wkn::az(w[0])) continue;
        const std::string l = st::lower(w);
        if (!stop.count(l)) c.insert(l);
    }
    return c;
}
double f1(const std::unordered_set<std::string>& a, const std::unordered_set<std::string>& b) {
    if (a.empty() || b.empty()) return 0;
    std::size_t i = 0;
    for (const auto& x : a) i += b.count(x);
    return 2.0 * static_cast<double>(i) / static_cast<double>(a.size() + b.size());
}

int cmd_features(const std::map<std::string, std::string>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    wkn::K5 m;
    if (!load_k5(get(a, "--k5"), m, wkn::Options{})) return 1;
    st::Stats s;
    std::string err;
    if (!s.load(get(a, "--stats"), err)) {
        std::cerr << err << "\n";
        return 1;
    }
    st::TrainRuns tr;
    tr.build(m.train_ids());
    cypha::cyphalm::lda::LDA topics;
    cypha::cyphalm::eg::EntityGrid grid;
    if (std::ifstream(get(a, "--stats") + ".lda")) topics.load(get(a, "--stats") + ".lda");
    if (std::ifstream(get(a, "--stats") + ".grid")) grid.load(get(a, "--stats") + ".grid");
    std::fprintf(stderr, "features: topics %s, grid %s\n", topics.trained() ? "on" : "off", grid.trained() ? "on" : "off");
    const st::Experts ex(s, m, tr, &topics, &grid);
    std::ifstream in(get(a, "--in"));
    std::ofstream out(get(a, "--out"), std::ios::binary);
    std::string line;
    std::size_t n_items = 0;
    // Read all boundaries' text first, then compute in parallel.
    struct Raw {
        std::string block;
    };
    std::vector<std::string> blocks;
    std::string cur;
    while (std::getline(in, line)) {
        if (line.rfind("B ", 0) == 0 && !cur.empty()) blocks.push_back(std::move(cur)), cur.clear();
        cur += line, cur += '\n';
    }
    if (!cur.empty()) blocks.push_back(std::move(cur));
    std::vector<RawItem> items(blocks.size());
    std::atomic<std::size_t> next{0};
    const unsigned nth = static_cast<unsigned>(std::atoi(get(a, "--threads", "8").c_str()));
    const auto work = [&] {
        std::string line;  // each thread its own
        for (std::size_t bi; (bi = next++) < blocks.size();) {
            std::istringstream bs(blocks[bi]);
            std::string tag;
            std::size_t uid, pc, nctx;
            int j, np, n;
            std::getline(bs, line);
            std::istringstream hl(line);
            hl >> tag >> uid >> j >> pc >> np >> nctx >> n;
            st::Context c;
            c.n_prompt = static_cast<std::size_t>(np);
            c.prompt_chars = pc;
            for (std::size_t k = 0; k < nctx; ++k) {
                std::getline(bs, line);
                st::Sent z;
                z.ended = line.size() > 2 && line[2] == '1';
                z.tok = split_ws(line.substr(std::min<std::size_t>(4, line.size())));
                c.sents.push_back(std::move(z));
            }
            ex.prepare(c);
            RawItem& it = items[bi];
            it.n = n, it.j = j;
            st::selectors(c, it.sel);
            it.f.resize(static_cast<std::size_t>(n * st::kNFeat));
            std::unordered_set<std::string> real_c;
            double best = -1;
            for (int k = 0; k < n; ++k) {
                std::getline(bs, line);
                std::istringstream xl(line);
                double lp, pm;
                int ended;
                xl >> tag >> lp >> pm >> ended;
                std::string rest;
                std::getline(xl, rest);
                st::Sent z;
                z.ended = ended != 0;
                z.tok = split_ws(rest);
                std::vector<std::uint32_t> ids;
                for (const auto& t : z.tok) ids.push_back(m.id(t));
                double f[st::kNFeat];
                ex.features(c, z, ids, lp, pm, f);
                for (int q = 0; q < st::kNFeat; ++q) it.f[static_cast<std::size_t>(k * st::kNFeat + q)] = static_cast<float>(f[q]);
                const auto cw = content(z.tok);
                if (k == 0) {
                    real_c = cw;
                } else {
                    const double x = f1(cw, real_c);
                    if (x > best) best = x, it.target_p = k;
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < nth; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    const int J = st::kNFeat;
    out.write("K5FT0001", 8);
    out.write(reinterpret_cast<const char*>(&J), 4);
    for (const auto& it : items) {
        const int hd[6] = {it.n, it.target_p, it.sel[0], it.sel[1], it.sel[2], it.j};
        out.write(reinterpret_cast<const char*>(hd), sizeof hd);
        out.write(reinterpret_cast<const char*>(it.f.data()), static_cast<std::streamsize>(it.f.size() * 4));
        ++n_items;
    }
    std::fprintf(stderr, "features: %zu items in %.1f s, peak RSS %ld MB\n", n_items, secs_since(t0), peak_rss_mb());
    return 0;
}

// ---------------------------------------------------------------- run: the arms

std::vector<RawItem> read_feat(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    char mg[8];
    int J = 0;
    in.read(mg, 8);
    in.read(reinterpret_cast<char*>(&J), 4);
    std::vector<RawItem> v;
    if (std::string(mg, 8) != "K5FT0001" || J != st::kNFeat) {
        std::cerr << path << ": not a features file of " << st::kNFeat << " features\n";
        return v;
    }
    for (;;) {
        int hd[6];
        if (!in.read(reinterpret_cast<char*>(hd), sizeof hd)) break;
        RawItem it;
        it.n = hd[0], it.target_p = hd[1], it.sel[0] = hd[2], it.sel[1] = hd[3], it.sel[2] = hd[4], it.j = hd[5];
        it.f.resize(static_cast<std::size_t>(it.n * J));
        in.read(reinterpret_cast<char*>(it.f.data()), static_cast<std::streamsize>(it.f.size() * 4));
        v.push_back(std::move(it));
    }
    return v;
}

// Items for signal R (the real one is candidate 0) or P (drafts only, the pseudo target), scaled by sd.
std::vector<st::Item> make_items(const std::vector<RawItem>& raw, bool pseudo, const std::vector<double>& sd) {
    std::vector<st::Item> v;
    v.reserve(raw.size());
    const int J = st::kNFeat;
    for (const auto& r : raw) {
        st::Item it;
        const int lo = pseudo ? 1 : 0;
        it.n = r.n - lo;
        it.target = pseudo ? r.target_p - 1 : 0;
        for (int k = 0; k < 3; ++k) it.sel[k] = r.sel[k];
        it.z.resize(static_cast<std::size_t>(it.n * J));
        for (int i = 0; i < it.n; ++i)
            for (int j = 0; j < J; ++j)
                it.z[static_cast<std::size_t>(i * J + j)] = static_cast<float>(r.f[static_cast<std::size_t>((i + lo) * J + j)] / sd[static_cast<std::size_t>(j)]);
        v.push_back(std::move(it));
    }
    return v;
}

struct ArmResult {
    std::string name;
    double preq = 0;           // M's prequential bits (online arms)
    std::vector<double> loss;  // H's per-item bits (frozen)
    double top1 = 0;
    st::MetaMixer mx;
};

double mean(const std::vector<double>& v) { return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
std::pair<double, double> boot(const std::vector<double>& v, const std::vector<double>* ref = nullptr) {
    std::mt19937_64 rng(20261105);
    std::vector<double> ms;
    const std::size_t n = v.size();
    if (!n) return {0, 0};
    std::uniform_int_distribution<std::size_t> u(0, n - 1);
    for (int b = 0; b < 1000; ++b) {
        double t = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t k = u(rng);
            t += v[k] - (ref ? (*ref)[k] : 0.0);
        }
        ms.push_back(t / static_cast<double>(n));
    }
    std::sort(ms.begin(), ms.end());
    return {ms[25], ms[974]};
}

void eval_frozen(ArmResult& r, const std::vector<st::Item>& test) {
    r.loss.clear();
    std::size_t hit = 0;
    std::vector<double> P;
    for (const auto& it : test) {
        r.mx.dist(it, P);
        r.loss.push_back(-std::log2(std::max(P[static_cast<std::size_t>(it.target)], 1e-300)));
        int b = 0;
        for (int i = 1; i < it.n; ++i)
            if (P[static_cast<std::size_t>(i)] > P[static_cast<std::size_t>(b)]) b = i;
        hit += b == it.target;
    }
    r.top1 = test.empty() ? 0 : static_cast<double>(hit) / static_cast<double>(test.size());
}

// One online pass over train; returns prequential bits per item.
double train_online(st::MetaMixer& mx, const std::vector<st::Item>& train) {
    double t = 0;
    for (const auto& it : train) t += mx.learn(it);
    return train.empty() ? 0 : t / static_cast<double>(train.size());
}

const double kUs[] = {2, 4, 8, 16};
const double kEtas[] = {0.002, 0.005, 0.01, 0.02, 0.05};

// M1 (one set) or one selector, U and eta chosen on train's prequential loss.
ArmResult best_single(const std::string& name, const std::vector<st::Item>& train, const std::vector<char>& mask, int selector) {
    ArmResult best;
    best.preq = 1e300;
    for (const double U : kUs)
        for (const double eta : kEtas) {
            st::MetaMixer mx;
            mx.r.emplace_back(st::kNFeat, selector, U, eta, mask);
            const double p = train_online(mx, train);
            if (p < best.preq) best.preq = p, best.mx = mx;
        }
    best.name = name;
    return best;
}

int cmd_run(const std::map<std::string, std::string>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool pseudo = get(a, "--signal", "R") == "P";
    auto rtrain = read_feat(get(a, "--train")), rtest = read_feat(get(a, "--test"));
    if (rtrain.empty() || rtest.empty()) return 1;
    const std::size_t maxtrain = std::strtoull(get(a, "--max-train", "1000000000").c_str(), nullptr, 10);
    {  // the shuffle (seed 20261105), then the cap
        std::mt19937_64 rng(20261105);
        std::shuffle(rtrain.begin(), rtrain.end(), rng);
        if (rtrain.size() > maxtrain) rtrain.resize(maxtrain);
    }
    const int J = st::kNFeat;
    std::vector<double> sd(J, 1.0);
    {
        std::vector<double> s1(J, 0), s2(J, 0);
        double n = 0;
        for (const auto& r : rtrain)
            for (int i = 0; i < r.n; ++i, n += 1)
                for (int j = 0; j < J; ++j) {
                    const double x = r.f[static_cast<std::size_t>(i * J + j)];
                    s1[j] += x, s2[j] += x * x;
                }
        for (int j = 0; j < J; ++j) {
            const double v = s2[j] / n - (s1[j] / n) * (s1[j] / n);
            sd[j] = v > 1e-12 ? std::sqrt(v) : 1.0;
        }
    }
    const auto train = make_items(rtrain, pseudo, sd), test = make_items(rtest, pseudo, sd);
    const double chance = test.empty() ? 0 : std::log2(static_cast<double>(test[0].n));
    std::fprintf(stderr, "run: %zu train items, %zu test items, signal %s, chance %.3f bits\n", train.size(), test.size(), pseudo ? "P" : "R", chance);
    // Feature means: real vs drafts (on train, raw), for the report.
    {
        std::printf("## Feature means (train, raw): target vs other candidates\n\n| feature | sd | target | others |\n|---|---:|---:|---:|\n");
        for (int j = 0; j < J; ++j) {
            double tr = 0, ot = 0, nt = 0, no = 0;
            for (std::size_t k = 0; k < train.size(); ++k)
                for (int i = 0; i < train[k].n; ++i) {
                    const double x = train[k].z[static_cast<std::size_t>(i * J + j)] * sd[j];
                    if (i == train[k].target) tr += x, nt += 1;
                    else ot += x, no += 1;
                }
            std::printf("| %s | %.4f | %.4f | %.4f |\n", st::feat_name(j), sd[j], tr / nt, ot / no);
        }
        std::printf("\n");
    }
    std::vector<ArmResult> arms;
    std::vector<char> all(J, 1);
    // --drop EXPERT[,EXPERT]: the mix arms (M1-M4, L0, ablations) leave those experts out.
    for (const auto& name : split_ws([&] { std::string d = get(a, "--drop"); std::replace(d.begin(), d.end(), ',', ' '); return d; }()))
        for (const auto& e : st::experts())
            if (name == e.name)
                for (const int f : e.feats) all[f] = 0;
    // Each expert alone.
    for (const auto& e : st::experts()) {
        std::vector<char> mask(J, 0);
        for (const int f : e.feats) mask[f] = 1;
        auto r = best_single(std::string("alone: ") + e.name, train, mask, 0);
        eval_frozen(r, test);
        arms.push_back(std::move(r));
    }
    // Uniform: natural signs, one scale fit on train.
    {
        ArmResult best;
        best.preq = 1e300;
        for (double c = 0.02; c <= 3.0; c *= 1.25) {
            st::MetaMixer mx;
            mx.r.emplace_back(J, 0, 1.0, 0.0, all);
            // a fixed ranker: write its weights through read/write
            std::stringstream ss;
            ss << "ranker 0 0 " << c * 2 * J << " " << J;
            for (int j = 0; j < J; ++j) ss << " 1";
            ss << "\n";
            for (int j = 0; j < J; ++j) ss << (st::natural_sign(j) > 0 ? c + 1e-12 : 1e-12) << " " << (st::natural_sign(j) < 0 ? c + 1e-12 : 1e-12) << " ";
            mx.r[0].read(ss);
            double t = 0;
            for (const auto& it : train) t += mx.loss(it);
            t /= static_cast<double>(train.size());
            if (t < best.preq) best.preq = t, best.mx = mx;
        }
        best.name = "uniform";
        eval_frozen(best, test);
        arms.push_back(std::move(best));
    }
    // M1-M4.
    auto m1 = best_single("M1 one set", train, all, 0);
    auto m2 = best_single("M2 12 sets (index x prompt)", train, all, 1);
    const double U1 = m1.mx.r[0].U(), eta1 = m1.mx.r[0].eta();
    auto layer2 = [&](const std::string& name, bool rates) {
        ArmResult best;
        best.preq = 1e300;
        for (const double e2 : {0.005, 0.02, 0.1}) {
            st::MetaMixer mx;
            for (int sel = 0; sel < 3; ++sel) {
                if (rates)
                    for (const double eta : kEtas) mx.r.emplace_back(J, sel, U1, eta, all);
                else
                    mx.r.emplace_back(J, sel, U1, eta1, all);
            }
            mx.eta2 = e2;
            mx.init_layer2();
            const double p = train_online(mx, train);
            if (p < best.preq) best.preq = p, best.mx = mx;
        }
        best.name = name;
        return best;
    };
    auto m3 = layer2("M3 second layer over 3 selectors", false);
    auto m4 = layer2("M4 M3 x 5 learning rates", true);
    for (auto* r : {&m1, &m2, &m3, &m4}) eval_frozen(*r, test), arms.push_back(*r);
    // L0: full-batch gradient descent, unconstrained weights, L2 1e-3.
    {
        std::vector<double> w(J, 0), mo(J, 0), ve(J, 0);
        ArmResult r;
        std::vector<double> P;
        for (int itn = 1; itn <= 300; ++itn) {
            std::vector<double> g(J, 0);
            double L = 0;
            for (const auto& it : train) {
                P.resize(static_cast<std::size_t>(it.n));
                double mx = -1e300;
                for (int i = 0; i < it.n; ++i) {
                    double s = 0;
                    for (int j = 0; j < J; ++j) s += w[j] * it.z[static_cast<std::size_t>(i * J + j)];
                    P[i] = s, mx = std::max(mx, s);
                }
                double z = 0;
                for (double& p : P) z += (p = std::exp(p - mx));
                for (double& p : P) p /= z;
                L -= std::log2(P[it.target]);
                for (int i = 0; i < it.n; ++i)
                    for (int j = 0; j < J; ++j) g[j] += (P[i] - (i == it.target)) * it.z[static_cast<std::size_t>(i * J + j)];
            }
            for (int j = 0; j < J; ++j) {  // Adam, step 0.05
                const double gj = all[j] ? g[j] / static_cast<double>(train.size()) + 1e-3 * w[j] : 0.0;
                mo[j] = 0.9 * mo[j] + 0.1 * gj, ve[j] = 0.999 * ve[j] + 0.001 * gj * gj;
                const double mh = mo[j] / (1 - std::pow(0.9, itn)), vh = ve[j] / (1 - std::pow(0.999, itn));
                w[j] -= 0.05 * mh / (std::sqrt(vh) + 1e-8);
            }
            r.preq = L / static_cast<double>(train.size());
        }
        std::stringstream ss;
        double U = 0;
        for (const double x : w) U += std::fabs(x);
        ss << "ranker 0 0 " << U << " " << J;
        for (int j = 0; j < J; ++j) ss << " 1";
        ss << "\n";
        for (int j = 0; j < J; ++j) ss << (w[j] > 0 ? w[j] : 0) + 1e-12 << " " << (w[j] < 0 ? -w[j] : 0) + 1e-12 << " ";
        r.mx.r.emplace_back();
        r.mx.r[0].read(ss);
        r.name = "L0 batch (train fit)";
        eval_frozen(r, test);
        arms.push_back(std::move(r));
    }
    // The chosen mix: the lowest prequential loss among M1-M4.
    const ArmResult* chosen = &m1;
    for (const auto* r : {&m2, &m3, &m4})
        if (r->preq < chosen->preq) chosen = r;
    // Ablations (M1 without each expert).
    std::vector<ArmResult> abl;
    for (const auto& e : st::experts()) {
        std::vector<char> mask = all;
        for (const int f : e.feats) mask[f] = 0;
        auto r = best_single(std::string("M1 without ") + e.name, train, mask, 0);
        eval_frozen(r, test);
        abl.push_back(std::move(r));
    }
    // The table.
    const ArmResult* best_alone = &arms[0];
    for (std::size_t k = 0; k < st::experts().size(); ++k)
        if (mean(arms[k].loss) < mean(best_alone->loss)) best_alone = &arms[k];
    const ArmResult* uni = &arms[st::experts().size()];
    std::printf("## Sentence bits (held out; chance %.3f; signal %s; %zu train, %zu held-out items)\n\n", chance, pseudo ? "P" : "R", train.size(), test.size());
    std::printf("| arm | train prequential | held-out bits [95%%] | top-1 |\n|---|---:|---:|---:|\n");
    for (const auto& r : arms) {
        const auto ci = boot(r.loss);
        std::printf("| %s | %.4f | %.4f [%.4f, %.4f] | %.3f |\n", r.name.c_str(), r.preq, mean(r.loss), ci.first, ci.second, r.top1);
    }
    std::printf("\nchosen mix (lowest train prequential): **%s**\n\n", chosen->name.c_str());
    {
        const auto d1 = boot(chosen->loss, &best_alone->loss), d2 = boot(chosen->loss, &uni->loss);
        std::printf("- vs best single expert (%s): %+.4f [%.4f, %.4f]\n", best_alone->name.c_str(), mean(chosen->loss) - mean(best_alone->loss), d1.first, d1.second);
        std::printf("- vs uniform: %+.4f [%.4f, %.4f]\n", mean(chosen->loss) - mean(uni->loss), d2.first, d2.second);
        std::printf("- rule 1 (both intervals below 0): %s\n\n", d1.second < 0 && d2.second < 0 ? "holds" : "fails");
    }
    std::printf("## M1's weights (z units; U %.3g, eta %.3g)\n\n| feature | weight |\n|---|---:|\n", m1.mx.r[0].U(), m1.mx.r[0].eta());
    for (int j = 0; j < J; ++j) std::printf("| %s | %+.4f |\n", st::feat_name(j), m1.mx.r[0].w(0, j));
    std::printf("\n## Ablations (M1 without each expert; held-out bits, change vs M1 [95%%])\n\n| without | bits | change |\n|---|---:|---:|\n");
    for (const auto& r : abl) {
        const auto ci = boot(r.loss, &m1.loss);
        std::printf("| %s | %.4f | %+.4f [%.4f, %.4f] |\n", r.name.c_str() + 11, mean(r.loss), mean(r.loss) - mean(m1.loss), ci.first, ci.second);
    }
    // By sentence index.
    std::printf("\n## By sentence index (held out; chosen mix vs pmi alone)\n\n| index | n | chosen | pmi alone |\n|---|---:|---:|---:|\n");
    for (int j = 0; j < 6; ++j) {
        double a1 = 0, a2 = 0, n = 0;
        for (std::size_t k = 0; k < rtest.size(); ++k)
            if (rtest[k].j == j) a1 += chosen->loss[k], a2 += arms[1].loss[k], n += 1;
        if (n) std::printf("| %d | %.0f | %.4f | %.4f |\n", j, n, a1 / n, a2 / n);
    }
    // Models.
    const std::string pre = get(a, "--out"), stats = get(a, "--stats");
    std::ofstream tsv(pre + ".tsv");
    for (const auto& r : arms) tsv << r.name << "\t" << r.preq << "\t" << mean(r.loss) << "\t" << r.top1 << "\n";
    for (const auto& r : abl) tsv << r.name << "\t" << r.preq << "\t" << mean(r.loss) << "\t" << r.top1 << "\n";
    int k = 1;
    for (const auto* r : {&m1, &m2, &m3, &m4}) {
        std::ofstream f(pre + ".M" + std::to_string(k++) + ".model");
        st::MetaMixer mx = r->mx;
        mx.sd = sd;
        f << "K5STRUCT 1\nstats " << stats << "\n";
        mx.write(f);
    }
    {
        std::ofstream f(pre + ".chosen.model");
        st::MetaMixer mx = chosen->mx;
        mx.sd = sd;
        f << "K5STRUCT 1\nstats " << stats << "\n";
        mx.write(f);
    }
    std::fprintf(stderr, "run: done in %.1f s, peak RSS %ld MB\n", secs_since(t0), peak_rss_mb());
    return 0;
}

// ---------------------------------------------------------------- selftest

int cmd_selftest() {
    int bad = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) std::fprintf(stderr, "FAIL %s\n", what), ++bad;
    };
    // Sentences cut as the gate cuts them.
    {
        const auto ss = st::split(wkn::tokens("He was born in Paris in 1950. He moved to London. It rained."));
        check(ss.size() == 2 && ss[0].ended && ss[1].ended && ss[1].tok.front() == "He", "split");
        check(st::render(ss) == "He was born in Paris in 1950. He moved to London. It rained.", "render");
    }
    check(st::year_of("1950") == 1950 && !st::year_of("950") && !st::year_of("3000") && st::is_number("12"), "years");
    check(st::pron_class("She") == 1 && st::pron_class("their") == 3 && st::pron_class("the") == -1, "pronouns");
    check(st::gap_bucket(1990, 0) == 9 && st::gap_bucket(1990, 1990) == 4 && st::gap_bucket(1995, 1990) == 5 && st::gap_bucket(1800, 1990) == 0, "gaps");
    // A ranker learns to prefer the candidate with the larger feature.
    {
        std::vector<char> mask(2, 1);
        st::MetaMixer mx;
        mx.r.emplace_back(2, 0, 4.0, 0.05, mask);
        double first = 0, last = 0;
        std::mt19937_64 rng(1);
        std::normal_distribution<double> nd;
        for (int t = 0; t < 4000; ++t) {
            st::Item it;
            it.n = 4;
            it.z.resize(8);
            for (auto& x : it.z) x = static_cast<float>(nd(rng));
            it.target = 0;
            for (int i = 1; i < 4; ++i)
                if (it.z[static_cast<std::size_t>(i * 2)] - it.z[static_cast<std::size_t>(i * 2 + 1)] > it.z[static_cast<std::size_t>(it.target * 2)] - it.z[static_cast<std::size_t>(it.target * 2 + 1)]) it.target = i;
            const double l = mx.learn(it);
            if (t < 200) first += l;
            if (t >= 3800) last += l;
        }
        check(last < first * 0.85 && mx.r[0].w(0, 0) > 0 && mx.r[0].w(0, 1) < 0, "ranker learns signs");
        // second layer and write/read round trip
        st::MetaMixer m2;
        m2.r.emplace_back(2, 0, 4.0, 0.05, mask);
        m2.r.emplace_back(2, 1, 4.0, 0.01, mask);
        m2.init_layer2();
        m2.sd = {1.0, 2.0};
        std::stringstream ss;
        m2.write(ss);
        st::MetaMixer m3;
        check(m3.read(ss, 2) && m3.r.size() == 2 && m3.v.size() == 2 && m3.sd[1] == 2.0 && m3.r[1].selector() == 1, "mixer round trip");
    }
    // The class n-gram's distribution sums to one, and the experts run, on a toy K5.
    {
        std::string text;
        for (int i = 0; i < 60; ++i)
            text += "John Smith was born in Paris in 1950. He moved to London in 1975. The city was large. However, it rained. "
                    "Mary Jones lived in Berlin. She wrote books in 1980.\n";
        wkn::K5 m;
        wkn::Options o;
        o.classes = 8;
        m.build(text, 12, 5, 8);
        st::Stats s;
        s.build(text, m, nullptr);
        double z = 0;
        const std::uint32_t h[2] = {s.cls(m.id("was")), s.cls(m.id("born"))};
        for (std::uint32_t c = 0; c < 10; ++c) z += std::exp2(s.class_lp(h, 2, c));
        check(std::fabs(z - 1) < 1e-6, "class n-gram sums to one");
        const auto* js = s.name("John Smith");
        check(js && js->type() == 1 && s.type_of_name("Paris") == 2, "name types");
        check(s.pron_score(js, 0) > 0 && s.pron_score(js, 1) < 0, "pronoun agreement");
        st::TrainRuns tr;
        tr.build(m.train_ids());
        const st::Experts ex(s, m, tr);
        st::Context c;
        c.sents = st::split(wkn::tokens("John Smith was born in Paris in 1950."));
        c.prompt_chars = 37;
        ex.prepare(c);
        const auto cand = st::split(wkn::tokens("He moved to London in 1975."))[0];
        std::vector<std::uint32_t> ids;
        for (const auto& t : cand.tok) ids.push_back(m.id(t));
        double f[st::kNFeat];
        ex.features(c, cand, ids, -3, 0.5, f);
        check(f[st::kPronAgree] > 0 && f[st::kPronOrphan] == 0 && f[st::kNYears] > 0 && f[st::kMatchTrain] >= 6 && f[st::kBalanced] == 1, "features");
        check(f[st::kTopic] == 0 && f[st::kGridFull] == 0, "no topic or grid without their models");
        {  // with wip/k5-discourse's models
            const auto tok = [](std::string_view line, auto&& emit) { wkn::tokenize(line, emit); };
            cypha::cyphalm::eg::EntityGrid g;
            g.train_lines(text, tok, 1);
            cypha::cyphalm::lda::LDA l;
            cypha::cyphalm::lda::Params lp;
            lp.K = 4, lp.iters = 20, lp.min_count = 1;
            l.train_lines(text, tok, lp);
            const st::Experts ex2(s, m, tr, &l, &g);
            st::Context c2 = c;
            ex2.prepare(c2);
            double f2[st::kNFeat];
            ex2.features(c2, cand, ids, -3, 0.5, f2);
            check(!c2.theta.empty() && std::isfinite(f2[st::kTopic]) && f2[st::kGridFull] < 0 && std::isfinite(f2[st::kGridCont]), "topic and grid features");
            for (int j = 0; j < st::kGridFull; ++j) check(j == st::kTopic || f2[j] == f[j], "other features unchanged by topic and grid");
        }
        // Scorer and Writer run.
        std::string err;
        auto md = std::make_shared<st::Model>();
        md->st = s;
        md->mx.r.emplace_back(st::kNFeat, 1, 4.0, 0.01, std::vector<char>(st::kNFeat, 1));
        md->mx.sd.assign(st::kNFeat, 1.0);
        const st::Scorer sc(md, m);
        const auto r = sc.score("John Smith was born in Paris in 1950.", {}, "He moved to London in 1975.");
        check(r.f[st::kNYears] > 0 && std::isfinite(r.mixed), "scorer");
        wkn::Options wo;
        wo.gate_k = 4;
        wo.bytes = 200;
        wo.structure = std::make_shared<st::Writer>(md, 0.0);
        wkn::GenStats g;
        const std::string a1 = m.generate("John Smith was born in Paris.", 7, wo, &g), a2 = m.generate("John Smith was born in Paris.", 7, wo);
        check(!a1.empty() && a1 == a2 && a1.size() <= 200 && g.gate_cands == 4 * g.gate_sents && !m.has_run13(a1), "writer deterministic, in budget");
        // With every weight 0 the writer keeps candidate 0, which is K5's gate with pmi weight 0 (wf 0, wp 0): the same text.
        auto md0 = std::make_shared<st::Model>();
        md0->st = s;
        md0->mx.r.emplace_back(st::kNFeat, 0, 1e-9, 0.0, std::vector<char>(st::kNFeat, 1));
        md0->mx.sd.assign(st::kNFeat, 1.0);
        wkn::Options w0 = wo;
        w0.structure = std::make_shared<st::Writer>(md0, 0.0);
        wkn::Options gz;
        gz.gate_k = 4, gz.bytes = 200, gz.gate_wf = 0, gz.gate_wp = 0, gz.gate_wm = 0;
        check(m.generate("John Smith was born in Paris.", 7, w0) == m.generate("John Smith was born in Paris.", 7, gz), "writer = gate on ties");
    }
    std::fprintf(stderr, "k5_structure selftest: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: k5_structure stats|drafts|features|run|selftest [flags]\n";
        return 2;
    }
    const std::string c = argv[1];
    const auto a = parse(argc, argv, 2);
    if (c == "stats") return cmd_stats(a);
    if (c == "drafts") return cmd_drafts(a);
    if (c == "discourse") return cmd_discourse(a);
    if (c == "features") return cmd_features(a);
    if (c == "run") return cmd_run(a);
    if (c == "selftest") return cmd_selftest();
    std::cerr << "unknown command " << c << "\n";
    return 2;
}
