// cyphalm_tok: CyphaLM over tokens, items T0-T2 of docs/TOKENS_PLAN.md (report: docs/reports/TOKENS_T0_T2.md).
//
// Tokenizers learned by counting (tokenizer.hpp) and count models over token IDs (token_model.hpp), beside the byte
// model; nothing in cyphalm_trie changes. Rule 10: this is CyphaLM itself; nothing here answers a question.
//
//   selftest                                  the CTest: round trips, BPE's merge order, Kneser-Ney against a brute-force
//                                             reference and summing to 1, the ∞-gram against a scan
//   entropy  <enwik8> <out.f32>               the byte model's next-byte entropy over [0, train), each byte scored
//                                             before it is seen (as in training), float32 bits
//   train    <enwik8> --arm bpe|uni|spb|ent|wrd --sizes 4096,8192,... --out PREFIX [--entropy FILE] [--mean-len 4.5]
//                                             learns on [0, train) and writes PREFIX-<arm>-<size>.tok
//   baseline <enwik8> --out FILE              the byte model as `cyphalm_trie cls` builds it (wiki only): per-byte bits
//                                             of the holdout for the trie and for trie + InfiniGram (λ on the first half)
//   eval     <enwik8> <TOK> --baseline FILE [--order 5] [--cap 64] [--boot 2000] [--dump FILE]
//                                             token models (KN, KN + ∞-gram) on the holdout's second half, bits per byte
//                                             (canonical segmentation), against the byte model over the same bytes
//
// The text: train [0, 7,340,032), holdout [7,340,032, 8,388,608) of enwik8 (packed_trie.hpp Config). Mixture weights
// on the holdout's first half; every number on its second half. The fresh text [8 MiB, 9 MiB) is not read.

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/cyphalm/packed_trie.hpp"
#include "cypha/cyphalm/token_model.hpp"
#include "cypha/cyphalm/tokenizer.hpp"

using namespace cypha::cyphalm;
using namespace cypha::cyphalm::tok;
using trie::Config;
using trie::Mode;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
double peak_mb() {  // 0 on Windows (no getrusage); the reports are measured on Linux
#ifndef _WIN32
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#else
    return 0.0;
#endif
}

int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
        }                                                                        \
    } while (0)

std::string arg_value(const std::vector<std::string>& a, const std::string& key, const std::string& def = "") {
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == key) return a[i + 1];
    return def;
}

std::vector<std::size_t> parse_sizes(const std::string& s) {
    std::vector<std::size_t> v;
    std::size_t p = 0;
    while (p < s.size()) {
        std::size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        v.push_back(std::stoull(s.substr(p, q - p)));
        p = q + 1;
    }
    return v;
}

// ---------------------------------------------------------------- selftest

// BPE by definition: every merge in training order, all occurrences left to right.
std::vector<std::uint32_t> bpe_in_order(const Tokenizer& t, const std::string& s) {
    std::vector<std::uint32_t> w(s.begin(), s.end());
    for (auto& x : w) x &= 0xff;
    std::vector<std::uint32_t> nw;
    for (std::size_t r = 0; r < t.merges.size(); ++r) {
        const auto [a, b] = t.merges[r];
        nw.clear();
        for (std::size_t k = 0; k < w.size();) {
            if (k + 1 < w.size() && w[k] == a && w[k + 1] == b) {
                nw.push_back(static_cast<std::uint32_t>(256 + r));
                k += 2;
            } else {
                nw.push_back(w[k++]);
            }
        }
        w.swap(nw);
    }
    return w;
}

// Brute-force interpolated modified Kneser-Ney over explicit token vectors (the reference for the hashed tables).
struct RefKN {
    int K;
    std::uint32_t V;
    std::vector<std::map<std::vector<std::uint32_t>, double>> c;  // c[k][ngram] (raw at K, continuation below)
    std::vector<double> D1, D2, D3;
    RefKN(const std::vector<std::uint32_t>& t, std::uint32_t V_, int K_) : K(K_), V(V_), c(K_ + 1), D1(K_ + 1), D2(K_ + 1), D3(K_ + 1) {
        std::map<std::vector<std::uint32_t>, double> raw;
        for (std::size_t i = 0; i + K <= t.size(); ++i) raw[std::vector<std::uint32_t>(t.begin() + i, t.begin() + i + K)] += 1;
        c[K] = raw;
        for (int k = K - 1; k >= 1; --k) {
            std::map<std::vector<std::uint32_t>, std::set<std::uint32_t>> ext;
            for (std::size_t i = 1; i + k <= t.size(); ++i)
                ext[std::vector<std::uint32_t>(t.begin() + i, t.begin() + i + k)].insert(t[i - 1]);
            for (const auto& [g, s] : ext) c[k][g] = static_cast<double>(s.size());
        }
        for (int k = 1; k <= K; ++k) {
            double n[5] = {0, 0, 0, 0, 0};
            for (const auto& [g, x] : c[k])
                if (x <= 4) n[static_cast<int>(x)] += 1;
            KneserNey::Order o;
            for (int j = 1; j <= 4; ++j) o.nc[j] = static_cast<std::uint64_t>(n[j]);
            KneserNey::set_discounts(o);
            D1[k] = o.D[1];
            D2[k] = o.D[2];
            D3[k] = o.D[3];
        }
    }
    double prob(const std::vector<std::uint32_t>& h, std::uint32_t w) const {
        double p = 1.0 / V;
        for (int k = 1; k <= K; ++k) {
            if (h.size() < static_cast<std::size_t>(k - 1)) break;
            const std::vector<std::uint32_t> ctx(h.end() - (k - 1), h.end());
            double sum = 0, n1 = 0, n2 = 0, n3 = 0, cw = 0;
            for (const auto& [g, x] : c[k]) {
                if (!std::equal(ctx.begin(), ctx.end(), g.begin())) continue;
                sum += x;
                if (x == 1) ++n1;
                else if (x == 2) ++n2;
                else ++n3;
                if (g.back() == w) cw = x;
            }
            if (sum == 0) break;
            const double d = cw == 0 ? 0 : cw == 1 ? D1[k] : cw == 2 ? D2[k] : D3[k];
            p = std::max(cw - d, 0.0) / sum + (D1[k] * n1 + D2[k] * n2 + D3[k] * n3) / sum * p;
        }
        return p;
    }
};

int cmd_selftest() {
    std::mt19937 rng(20261029);
    // A small corpus: sentences from a few templates, so merges and contexts repeat.
    const char* words[] = {"the", "cat", "sat", "on", "mat", "dog", "ran", "to", "a", "park", "[[link]]", "1984", "of", "and"};
    std::string corpus;
    for (int i = 0; i < 4000; ++i) {
        corpus += words[rng() % 14];
        corpus += (rng() % 9 == 0) ? ",\n" : " ";
    }
    const auto* cb = reinterpret_cast<const std::uint8_t*>(corpus.data());

    // Pre-tokens cover the text exactly, in order.
    {
        std::size_t at = 0;
        bool ok = true;
        pretokens(cb, corpus.size(), [&](std::size_t a, std::size_t b) {
            if (a != at || b <= a) ok = false;
            at = b;
        });
        CHECK(ok && at == corpus.size());
        std::vector<std::string> pt;
        const std::string s = "Hello  world, 42 times\n\n x";
        pretokens(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), [&](std::size_t a, std::size_t b) { pt.push_back(s.substr(a, b - a)); });
        CHECK((pt == std::vector<std::string>{"Hello", " ", " world", ",", " 42", " times", "\n\n", " x"}));
    }
    const auto chunks = count_chunks(cb, corpus.size(), false);
    std::vector<Tokenizer> toks;
    const Tokenizer bpe = train_bpe(chunks, 320);
    CHECK(bpe.size() > 256 && bpe.size() <= 320);
    // Encoding by rank gives the segmentation of training order on every training pre-token.
    bool same = true;
    for (const auto& [s, c] : chunks)
        if (bpe.encode(s) != bpe_in_order(bpe, s)) same = false;
    CHECK(same);
    toks.push_back(bpe);
    toks.push_back(bpe_prefix(bpe, 290));
    for (auto& t : train_unigram(chunks, {300, 280}, 2000)) toks.push_back(t);
    toks.push_back(top_chunks(Kind::Words, sorted_chunks(chunks), 300));
    toks.push_back(top_chunks(Kind::SpaceByte, sorted_chunks(count_chunks(cb, corpus.size(), true)), 300));
    {
        std::vector<float> ent(corpus.size());
        for (std::size_t i = 0; i < ent.size(); ++i) ent[i] = (i == 0 || corpus[i - 1] == ' ') ? 3.0f : 1.0f;
        const double th = entropy_threshold(ent, 4.0);
        Tokenizer e = top_chunks(Kind::Entropy, sorted_chunks(count_entropy_segments(cb, corpus.size(), ent, th)), 300);
        toks.push_back(e);
    }
    // Every tokenizer: the 256 bytes are tokens; round trips on the corpus, random bytes and edge strings; save/load.
    std::string rnd;
    for (int i = 0; i < 5000; ++i) rnd.push_back(static_cast<char>(rng() & 0xff));
    const std::string edge = std::string("\0\xff\x80 ", 4) + "  \n\t x" + std::string(200, 'a');
    int idx = 0;
    for (const auto& t : toks) {
        bool bytes_ok = true;
        for (std::uint32_t b = 0; b < 256; ++b)
            if (t.pieces[b] != std::string(1, static_cast<char>(b))) bytes_ok = false;
        CHECK(bytes_ok);
        for (const std::string* s : std::initializer_list<const std::string*>{&corpus, &rnd, &edge}) CHECK(t.decode(t.encode(*s)) == *s);
        const std::string path = "/tmp/cyphalm_tok_selftest_" + std::to_string(idx++) + ".tok";
        t.save(path);
        const Tokenizer u = Tokenizer::load(path);
        std::remove(path.c_str());
        CHECK(u.encode(corpus) == t.encode(corpus) && u.encode(rnd) == t.encode(rnd));
        CHECK(t.encode(corpus).size() <= corpus.size());
    }
    // The tokens shorten the text.
    CHECK(bpe.encode(corpus).size() * 2 < corpus.size());

    // Kneser-Ney: against the brute-force reference, and every distribution sums to 1.
    {
        const std::vector<std::uint32_t> t = bpe.encode(corpus);
        const std::uint32_t V = static_cast<std::uint32_t>(bpe.size());
        for (int K : {1, 2, 3}) {
            const KneserNey kn(t.data(), t.size(), V, K);
            const RefKN ref(t, V, K);
            double maxdiff = 0, maxsum = 0;
            std::vector<double> P;
            for (int q = 0; q < 40; ++q) {
                const std::size_t at = 3 + rng() % (t.size() - 3);
                const std::vector<std::uint32_t> h(t.begin() + static_cast<std::ptrdiff_t>(at) - 3, t.begin() + static_cast<std::ptrdiff_t>(at));
                for (std::uint32_t w : {t[at], static_cast<std::uint32_t>(rng() % V)})
                    maxdiff = std::max(maxdiff, std::fabs(kn.prob(h.data(), h.size(), w) - ref.prob(h, w)));
                kn.dist(h.data(), h.size(), P);
                maxsum = std::max(maxsum, std::fabs(std::accumulate(P.begin(), P.end(), 0.0) - 1.0));
                CHECK(std::fabs(P[t[at]] - kn.prob(h.data(), h.size(), t[at])) < 1e-12);
            }
            CHECK(maxdiff < 1e-12);
            CHECK(maxsum < 1e-9);
            for (int k = 1; k <= K; ++k) {
                const auto& o = kn.order_table(k);
                CHECK(o.D[1] >= 0 && o.D[1] <= 1 && o.D[2] <= 2 && o.D[3] <= 3);
            }
        }
    }
    // ∞-gram over tokens: against a scan.
    {
        std::vector<std::uint32_t> t(3000);
        for (auto& x : t) x = rng() % 6;
        const TokenInfini inf(t.data(), t.size(), 6);
        bool ok = true;
        for (int q = 0; q < 200; ++q) {
            std::vector<std::uint32_t> h(1 + rng() % 12);
            for (auto& x : h) x = rng() % 6;
            const std::uint32_t w = rng() % 6;
            const auto r = inf.query(h.data(), h.size(), w, 64);
            // Scan: the longest suffix of h with an occurrence followed by a token.
            int best = 0;
            std::uint64_t tot = 0, cnt = 0;
            for (int m = static_cast<int>(h.size()); m >= 1 && best == 0; --m) {
                std::uint64_t tt = 0, cc = 0;
                for (std::size_t i = 0; i + static_cast<std::size_t>(m) < t.size(); ++i)
                    if (std::equal(h.end() - m, h.end(), t.begin() + static_cast<std::ptrdiff_t>(i))) {
                        ++tt;
                        if (t[i + static_cast<std::size_t>(m)] == w) ++cc;
                    }
                if (tt > 0) {
                    best = m;
                    tot = tt;
                    cnt = cc;
                }
            }
            if (r.n != best || r.total != tot || r.count != cnt) ok = false;
        }
        CHECK(ok);
    }
    std::printf("cyphalm_tok selftest: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- the byte model

bool read_corpus(const std::string& path, std::size_t n, std::vector<std::uint8_t>& data) {
    const std::string e = trie::read_prefix(path, n, data);
    if (!e.empty()) {
        std::cerr << e << "\n";
        return false;
    }
    return true;
}

int cmd_entropy(const std::vector<std::string>& a) {
    if (a.size() < 3) {
        std::cerr << "usage: cyphalm_tok entropy <enwik8> <out.f32>\n";
        return 2;
    }
    const Config cfg;
    std::vector<std::uint8_t> data;
    if (!read_corpus(a[1], cfg.sample, data)) return 1;
    const double t0 = now_s();
    trie::Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    trie::init_weights(w, cfg);
    std::vector<float> ent(cfg.train);
    // train_span with the mixture's entropy taken before each byte is observed.
    for (std::size_t i = 0; i < cfg.train; ++i) {
        double e[5][256], M[256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, i, Mode::Synth, e, matched, pred, true);
        trie::mix_dist(w, e, matched, pred, M);
        double h = 0;
        for (int s = 0; s < 256; ++s)
            if (M[s] > 0) h -= M[s] * std::log2(M[s]);
        ent[i] = static_cast<float>(h);
        double py[6];
        const unsigned y = data[i];
        for (int k = 0; k < 5; ++k) py[k] = e[k][y];
        py[5] = (matched && pred == y) ? 1.0 : 0.0;
        trie::eg_update(w, py, matched, cfg);
        model.observe(i);
    }
    std::ofstream os(a[2], std::ios::binary);
    os.write(reinterpret_cast<const char*>(ent.data()), static_cast<std::streamsize>(ent.size() * sizeof(float)));
    double mean = 0;
    for (float x : ent) mean += x;
    std::printf("ENTROPY bytes %zu mean %.4f bits time %.1f s peak %.0f MB\n", ent.size(), mean / static_cast<double>(ent.size()), now_s() - t0, peak_mb());
    return 0;
}

// Per-byte bits of the holdout, two models: the trie and the trie + InfiniGram, exactly as cmd_cls computes them.
int cmd_baseline(const std::vector<std::string>& a) {
    const std::string out = arg_value(a, "--out");
    if (a.size() < 2 || out.empty()) {
        std::cerr << "usage: cyphalm_tok baseline <enwik8> --out FILE\n";
        return 2;
    }
    const Config cfg;
    const std::size_t H = cfg.sample - cfg.train, half = H / 2;
    std::vector<std::uint8_t> data;
    if (!read_corpus(a[1], cfg.sample, data)) return 1;
    const double t0 = now_s();
    trie::Model model;
    model.data = data.data();
    model.init(cfg);
    double w[6];
    trie::init_weights(w, cfg);
    trie::train_span(model, data, w, 0, cfg.train, cfg);
    model.compact_match();
    const trie::PackedModel pm = trie::pack(model);
    model.release();
    const FmInfiniGram ig(data.data(), cfg.train);
    std::vector<double> pt(H), pi(H);
    std::vector<int> bi(H);
    FmInfiniGram::Cursor cur;
    auto len_bucket = [](int n) { return n < 4 ? 0 : n < 8 ? 1 : n < 16 ? 2 : n < 32 ? 3 : n < 64 ? 4 : n < 128 ? 5 : 6; };
    auto count_bucket = [](std::uint64_t t) { return t <= 1 ? 0 : t <= 3 ? 1 : t <= 15 ? 2 : 3; };
    for (std::size_t k = 0, prev = 0; k < H; ++k) {
        const std::size_t i = cfg.train + k;
        double e[5][256], M[256];
        bool em = false;
        unsigned pred = 0;
        pm.fill_experts(data.data(), i, cfg.train, Mode::Synth, e, em, pred, false);
        trie::mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
        const std::size_t len = std::min<std::size_t>(i, 8192);
        const auto r = ig.query(data.data() + (i - len), len, 8192, k == 0 ? -1 : std::min<int>(static_cast<int>(prev) + 1, 8192), 1, cur);
        prev = static_cast<std::size_t>(r.n);
        pi[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
        bi[k] = len_bucket(r.n) * 4 + count_bucket(r.total);
    }
    double lam[28];
    fit_lambdas(pi, pt, bi, half, 28, lam);
    std::vector<double> bt(H), bg(H);
    double st = 0, sg = 0;
    for (std::size_t k = 0; k < H; ++k) {
        bt[k] = -std::log2(pt[k]);
        bg[k] = -std::log2(lam[bi[k]] * pi[k] + (1 - lam[bi[k]]) * pt[k]);
        if (k >= half) {
            st += bt[k];
            sg += bg[k];
        }
    }
    std::ofstream os(out, std::ios::binary);
    const std::uint64_t n = H;
    os.write(reinterpret_cast<const char*>(&n), sizeof n);
    os.write(reinterpret_cast<const char*>(bt.data()), static_cast<std::streamsize>(H * sizeof(double)));
    os.write(reinterpret_cast<const char*>(bg.data()), static_cast<std::streamsize>(H * sizeof(double)));
    std::printf("BASELINE holdout2 trie %.6f trie+ig %.6f | time %.1f s peak %.0f MB\n", st / static_cast<double>(H - half),
                sg / static_cast<double>(H - half), now_s() - t0, peak_mb());
    return 0;
}

// ---------------------------------------------------------------- tokenizers

int cmd_train(const std::vector<std::string>& a) {
    const std::string arm = arg_value(a, "--arm"), prefix = arg_value(a, "--out");
    const auto sizes = parse_sizes(arg_value(a, "--sizes", "4096,8192,16384,32768"));
    if (a.size() < 2 || arm.empty() || prefix.empty()) {
        std::cerr << "usage: cyphalm_tok train <enwik8> --arm bpe|uni|spb|ent|wrd --sizes LIST --out PREFIX [--entropy FILE]\n";
        return 2;
    }
    const Config cfg;
    std::vector<std::uint8_t> data;
    if (!read_corpus(a[1], cfg.train, data)) return 1;
    const double t0 = now_s();
    const Kind k = kind_from(arm);
    std::vector<Tokenizer> out;
    std::size_t biggest = 0;
    for (auto s : sizes) biggest = std::max(biggest, s);
    if (k == Kind::BPE) {
        const Tokenizer full = train_bpe(count_chunks(data.data(), data.size(), false), biggest, true);
        for (auto s : sizes) out.push_back(bpe_prefix(full, s));
    } else if (k == Kind::Unigram) {
        out = train_unigram(count_chunks(data.data(), data.size(), false), sizes, std::max<std::size_t>(200000, 6 * biggest), true);
    } else if (k == Kind::Words) {
        const auto sc = sorted_chunks(count_chunks(data.data(), data.size(), false));
        for (auto s : sizes) out.push_back(top_chunks(k, sc, s));
    } else if (k == Kind::SpaceByte) {
        const auto sc = sorted_chunks(count_chunks(data.data(), data.size(), true));
        for (auto s : sizes) out.push_back(top_chunks(k, sc, s));
    } else {
        const std::string ef = arg_value(a, "--entropy");
        std::ifstream is(ef, std::ios::binary);
        if (!is) {
            std::cerr << "need --entropy FILE (cyphalm_tok entropy)\n";
            return 1;
        }
        std::vector<float> ent(cfg.train);
        is.read(reinterpret_cast<char*>(ent.data()), static_cast<std::streamsize>(ent.size() * sizeof(float)));
        const double mean_len = std::stod(arg_value(a, "--mean-len", "4.5"));
        const double th = entropy_threshold(ent, mean_len);
        const auto segs = count_entropy_segments(data.data(), data.size(), ent, th);
        std::uint64_t nseg = 0;
        for (const auto& [s, c] : segs) nseg += c;
        std::printf("ENT threshold %.4f bits, segments %llu, mean %.3f bytes, distinct %zu\n", th, static_cast<unsigned long long>(nseg),
                    static_cast<double>(data.size()) / static_cast<double>(nseg), segs.size());
        const auto sc = sorted_chunks(segs);
        for (auto s : sizes) out.push_back(top_chunks(k, sc, s));
    }
    std::sort(out.begin(), out.end(), [](const Tokenizer& x, const Tokenizer& y) { return x.size() < y.size(); });
    for (const auto& t : out) {
        std::size_t want = 0;
        for (auto s : sizes)
            if (s >= t.size() && (want == 0 || s < want)) want = s;
        const std::string path = prefix + "-" + arm + "-" + std::to_string(want ? want : t.size()) + ".tok";
        t.save(path);
        std::printf("TRAIN %s size %zu (asked %zu) -> %s\n", arm.c_str(), t.size(), want, path.c_str());
    }
    std::printf("TRAIN %s time %.1f s peak %.0f MB\n", arm.c_str(), now_s() - t0, peak_mb());
    return 0;
}

// ---------------------------------------------------------------- eval

struct Ratio {
    double mean = 0, lo = 0, hi = 0;
};

int cmd_eval(const std::vector<std::string>& a) {
    const std::string base = arg_value(a, "--baseline");
    if (a.size() < 3 || base.empty()) {
        std::cerr << "usage: cyphalm_tok eval <enwik8> <TOK> --baseline FILE [--order 5] [--cap 64] [--boot 2000]\n";
        return 2;
    }
    const int K = std::stoi(arg_value(a, "--order", "5"));
    const std::size_t cap = std::stoull(arg_value(a, "--cap", "64"));
    const int boots = std::stoi(arg_value(a, "--boot", "2000"));
    const Config cfg;
    const std::size_t H = cfg.sample - cfg.train, half_byte = cfg.train + H / 2;
    std::vector<std::uint8_t> data;
    if (!read_corpus(a[1], cfg.sample, data)) return 1;
    std::vector<double> bt, bg;
    {
        std::ifstream is(base, std::ios::binary);
        std::uint64_t n = 0;
        is.read(reinterpret_cast<char*>(&n), sizeof n);
        if (n != H) {
            std::cerr << "baseline file does not match the holdout\n";
            return 1;
        }
        bt.resize(H);
        bg.resize(H);
        is.read(reinterpret_cast<char*>(bt.data()), static_cast<std::streamsize>(H * sizeof(double)));
        is.read(reinterpret_cast<char*>(bg.data()), static_cast<std::streamsize>(H * sizeof(double)));
        if (!is) {
            std::cerr << "short baseline file\n";
            return 1;
        }
    }
    const double t0 = now_s();
    const Tokenizer tk = Tokenizer::load(a[2]);
    const std::uint32_t V = static_cast<std::uint32_t>(tk.size());
    std::vector<std::uint32_t> toks;
    tk.encode(data.data(), cfg.train, toks);
    const std::size_t ntrain = toks.size();
    tk.encode(data.data() + cfg.train, H, toks);
    // Byte offset of each holdout token, and the round trip.
    std::vector<std::size_t> at(toks.size() - ntrain + 1);
    {
        std::size_t pos = cfg.train;
        for (std::size_t j = ntrain; j < toks.size(); ++j) {
            at[j - ntrain] = pos;
            pos += tk.pieces[toks[j]].size();
        }
        at.back() = pos;
        if (pos != cfg.sample || tk.decode(toks.data() + ntrain, toks.size() - ntrain) !=
                                     std::string(reinterpret_cast<const char*>(data.data()) + cfg.train, H)) {
            std::cerr << "holdout round trip failed\n";
            return 1;
        }
    }
    const double t_enc = now_s() - t0;
    const KneserNey kn(toks.data(), ntrain, V, K);
    const TokenInfini inf(toks.data(), ntrain, V);
    const double t_build = now_s() - t0 - t_enc;
    const std::size_t nh = toks.size() - ntrain;
    std::vector<double> pk(nh), pi(nh);
    std::vector<int> bk(nh);
    int prev = -1;
    for (std::size_t j = 0; j < nh; ++j) {
        const std::size_t i = ntrain + j;
        pk[j] = kn.prob(toks.data(), i, toks[i]);
        const auto r = inf.query(toks.data(), i, toks[i], cap, prev < 0 ? -1 : prev + 1);
        prev = r.n;
        pi[j] = r.total ? static_cast<double>(r.count) / static_cast<double>(r.total) : 0.0;
        bk[j] = tok_bucket(r.n, r.total);
    }
    const double t_score = now_s() - t0 - t_enc - t_build;
    std::size_t split = 0;
    while (split < nh && at[split] < half_byte) ++split;
    double lam[kTokBuckets];
    fit_lambdas(pi, pk, bk, split, kTokBuckets, lam);
    // Documents of the second half: split at "<page>".
    const std::size_t b0 = at[split];
    std::vector<std::size_t> doc_of(cfg.sample - b0, 0);
    {
        const std::string hold(reinterpret_cast<const char*>(data.data()) + b0, cfg.sample - b0);
        std::size_t d = 0, p = 0;
        for (std::size_t q = hold.find("<page>"); q != std::string::npos; q = hold.find("<page>", q + 1)) {
            for (; p < q; ++p) doc_of[p] = d;
            if (q > 0) ++d;
        }
        for (; p < hold.size(); ++p) doc_of[p] = d;
    }
    const std::size_t ndoc = doc_of.back() + 1;
    // Per document: bits of KN, KN+INF, byte trie, byte trie+IG, and bytes.
    std::vector<std::array<double, 5>> D(ndoc, {0, 0, 0, 0, 0});
    for (std::size_t j = split; j < nh; ++j) {
        const std::size_t d = doc_of[at[j] - b0];
        const double l = lam[bk[j]];
        D[d][0] += -std::log2(pk[j]);
        D[d][1] += -std::log2(l * pi[j] + (1 - l) * pk[j]);
    }
    for (std::size_t p = b0; p < cfg.sample; ++p) {
        const std::size_t d = doc_of[p - b0];
        D[d][2] += bt[p - cfg.train];
        D[d][3] += bg[p - cfg.train];
        D[d][4] += 1;
    }
    auto ratio = [&](const std::vector<std::size_t>& pick, int col) {
        double b = 0, n = 0;
        for (auto d : pick) {
            b += D[d][col];
            n += D[d][4];
        }
        return b / n;
    };
    std::vector<std::size_t> all(ndoc);
    std::iota(all.begin(), all.end(), 0);
    double point[4];
    for (int c = 0; c < 4; ++c) point[c] = ratio(all, c);
    std::mt19937_64 rng(20261029);
    std::vector<std::vector<double>> diffs(3);  // KN+INF - byte trie+IG, KN+INF - byte trie, KN - byte trie
    std::vector<std::size_t> pick(ndoc);
    for (int b = 0; b < boots; ++b) {
        for (auto& x : pick) x = rng() % ndoc;
        const double knI = ratio(pick, 1), kn0 = ratio(pick, 0), bT = ratio(pick, 2), bG = ratio(pick, 3);
        diffs[0].push_back(knI - bG);
        diffs[1].push_back(knI - bT);
        diffs[2].push_back(kn0 - bT);
    }
    auto ci = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        return std::make_pair(v[static_cast<std::size_t>(0.025 * static_cast<double>(n))], v[std::min(n - 1, static_cast<std::size_t>(0.975 * static_cast<double>(n)))]);
    };
    // Statistics (reported only): bytes per token, word starts that are token starts, vocabulary used.
    std::vector<char> starts(H + 1, 0);
    for (std::size_t j = 0; j < nh; ++j) starts[at[j] - cfg.train] = 1;
    std::uint64_t ws = 0, wst = 0;
    for (std::size_t p = cfg.train; p < cfg.sample; ++p)
        if (is_letter(data[p]) && !is_letter(data[p - 1])) {
            ++ws;  // a word start counts as aligned when a token starts at it or at the space just before it
            if (starts[p - cfg.train] || (data[p - 1] == ' ' && p - 1 >= cfg.train && starts[p - 1 - cfg.train])) ++wst;
        }
    std::vector<char> used(V, 0);
    std::size_t nused = 0;
    for (std::size_t j = 0; j < nh; ++j)
        if (!used[toks[ntrain + j]]) {
            used[toks[ntrain + j]] = 1;
            ++nused;
        }
    // --dump FILE: one line a document (KN bits, KN+INF bits, byte trie bits, byte trie+IG bits, bytes) for paired
    // comparisons between runs that score the same bytes.
    if (const std::string dump = arg_value(a, "--dump"); !dump.empty()) {
        std::ofstream os(dump);
        os << "# b0 " << b0 << "\n";
        char buf[160];
        for (const auto& d : D) {
            std::snprintf(buf, sizeof buf, "%.6f\t%.6f\t%.6f\t%.6f\t%.0f\n", d[0], d[1], d[2], d[3], d[4]);
            os << buf;
        }
    }
    const auto c0 = ci(diffs[0]), c1 = ci(diffs[1]), c2 = ci(diffs[2]);
    std::printf("EVAL %s V %u K %d | holdout2 bytes %zu docs %zu tokens %zu | KN %.6f KN+INF %.6f | byte trie %.6f trie+ig %.6f | "
                "KN+INF-trie+ig %+.4f [%+.4f, %+.4f] KN+INF-trie %+.4f [%+.4f, %+.4f] KN-trie %+.4f [%+.4f, %+.4f] | bytes/token %.3f "
                "wordstart %.3f vocab_used %.3f | train tokens %zu | KN %.1f MB INF %.1f MB | enc %.1f s build %.1f s score %.1f s peak %.0f MB\n",
                a[2].c_str(), V, K, cfg.sample - b0, ndoc, nh - split, point[0], point[1], point[2], point[3], point[1] - point[3], c0.first,
                c0.second, point[1] - point[2], c1.first, c1.second, point[0] - point[2], c2.first, c2.second,
                static_cast<double>(H) / static_cast<double>(nh), static_cast<double>(wst) / static_cast<double>(ws),
                static_cast<double>(nused) / static_cast<double>(V), ntrain, static_cast<double>(kn.bytes()) / 1e6,
                static_cast<double>(inf.bytes()) / 1e6, t_enc, t_build, t_score, peak_mb());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty()) {
        std::cerr << "usage: cyphalm_tok selftest | entropy | train | baseline | eval (see the file header)\n";
        return 2;
    }
    try {
        if (a[0] == "selftest") return cmd_selftest();
        if (a[0] == "entropy") return cmd_entropy(a);
        if (a[0] == "train") return cmd_train(a);
        if (a[0] == "baseline") return cmd_baseline(a);
        if (a[0] == "eval") return cmd_eval(a);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown subcommand " << a[0] << "\n";
    return 2;
}
