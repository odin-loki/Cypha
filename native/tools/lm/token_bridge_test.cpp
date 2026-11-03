// native_token_bridge_test: the exact byte-token bridge (TOKENS_PLAN T3,
// docs/reports/TOKENS_T3_T4.md). A byte-level Markov chain is tokenized four ways; the exact
// token model of each is built by enumeration; the bridge, given only that token model and the
// tokenizer, must recover the chain's next-byte probabilities to 1e-9.
#include "cypha/cyphalm/token_bridge.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace br = cypha::cyphalm::bridge;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(c)) {                                                      \
            ++g_failed;                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Toy tokenizers. Every vocabulary holds all 256 single bytes (id = byte) plus extra tokens.
// ---------------------------------------------------------------------------------------------

static int cls(unsigned char c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) return 0;
    if (c >= '0' && c <= '9') return 1;
    if (c == ' ') return 2;
    if (c == '\n' || c == '\t' || c == '\r') return 3;
    return 4;
}
// Word pretokens: a pretoken starts at j when j == 0, s[j] is a space, or the byte class changes
// and the previous byte is not a space (" word" keeps its leading space). Appending bytes never
// moves an earlier boundary.
static bool pretoken_boundary(std::string_view s, std::size_t j) {
    if (j == 0) return true;
    const auto a = static_cast<unsigned char>(s[j - 1]), b = static_cast<unsigned char>(s[j]);
    return b == ' ' || (cls(a) != cls(b) && a != ' ');
}

class ToyBase : public br::Tokenizer {
public:
    ToyBase() {
        for (int b = 0; b < 256; ++b) add(std::string(1, static_cast<char>(b)));
    }
    int vocab_size() const override { return static_cast<int>(tok_.size()); }
    std::string_view token(int id) const override { return tok_[static_cast<std::size_t>(id)]; }
    int add(const std::string& s) {
        auto it = id_.find(s);
        if (it != id_.end()) return it->second;
        tok_.push_back(s);
        lmax_ = std::max(lmax_, s.size());
        return id_[s] = static_cast<int>(tok_.size()) - 1;
    }
    int id(const std::string& s) const {
        auto it = id_.find(s);
        return it == id_.end() ? -1 : it->second;
    }
    std::size_t lmax() const { return lmax_; }

protected:
    std::vector<std::string> tok_;
    std::map<std::string, int> id_;
    std::size_t lmax_ = 1;
};

// Tokenizers that encode each word pretoken on its own.
class PretokBase : public ToyBase {
public:
    std::vector<int> encode(std::string_view s) const override {
        std::vector<int> out;
        std::size_t a = 0;
        for (std::size_t j = 1; j <= s.size(); ++j)
            if (j == s.size() || pretoken_boundary(s, j)) {
                const std::vector<int> e = encode_pretoken(s.substr(a, j - a));
                out.insert(out.end(), e.begin(), e.end());
                a = j;
            }
        return out;
    }
    std::size_t stable_prefix(std::string_view s, std::size_t) const override {
        if (s.empty()) return 0;
        std::size_t j = s.size() - 1;
        while (j > 0 && !pretoken_boundary(s, j)) --j;
        return j;
    }
    virtual std::vector<int> encode_pretoken(std::string_view w) const = 0;
};

// BPE: ranked merges, applied lowest rank first, all non-overlapping occurrences left to right.
class ToyBPE : public PretokBase {
public:
    void merge(const std::string& a, const std::string& b) {
        const int ia = id(a), ib = id(b);
        const int m = add(a + b);
        rank_[{ia, ib}] = {static_cast<int>(rank_.size()), m};
    }
    std::vector<int> encode_pretoken(std::string_view w) const override {
        std::vector<int> sym;
        for (char c : w) sym.push_back(static_cast<unsigned char>(c));
        for (;;) {
            int best = -1, br_ = 1 << 30, bm = -1;
            for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
                auto it = rank_.find({sym[i], sym[i + 1]});
                if (it != rank_.end() && it->second.first < br_) {
                    br_ = it->second.first;
                    best = static_cast<int>(i);
                    bm = it->second.second;
                }
            }
            if (best < 0) break;
            const int a = sym[static_cast<std::size_t>(best)], b = sym[static_cast<std::size_t>(best) + 1];
            std::vector<int> nx;
            for (std::size_t i = 0; i < sym.size(); ++i) {
                if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
                    nx.push_back(bm);
                    ++i;
                } else {
                    nx.push_back(sym[i]);
                }
            }
            sym.swap(nx);
        }
        return sym;
    }

private:
    std::map<std::pair<int, int>, std::pair<int, int>> rank_;  // pair -> (rank, merged id)
};

// Longest match (MPE) inside each pretoken.
class ToyMPE : public PretokBase {
public:
    std::vector<int> encode_pretoken(std::string_view w) const override {
        std::vector<int> out;
        std::size_t i = 0;
        while (i < w.size()) {
            for (std::size_t L = std::min(lmax(), w.size() - i); L >= 1; --L) {
                const int t = id(std::string(w.substr(i, L)));
                if (t >= 0) {
                    out.push_back(t);
                    i += L;
                    break;
                }
            }
        }
        return out;
    }
};

// Unigram: the highest-scoring segmentation (Viterbi) inside each pretoken; ties go to the longer
// last token.
class ToyUnigram : public PretokBase {
public:
    void score(const std::string& s, double v) { sc_[add(s)] = v; }
    double sc(int id) const {
        auto it = sc_.find(id);
        return it == sc_.end() ? -10.0 : it->second;
    }
    std::vector<int> encode_pretoken(std::string_view w) const override {
        const std::size_t m = w.size();
        std::vector<double> best(m + 1, -1e300);
        std::vector<int> back(m + 1, -1), bt(m + 1, -1);
        best[0] = 0;
        for (std::size_t j = 1; j <= m; ++j)
            for (std::size_t i = (j > lmax() ? j - lmax() : 0); i < j; ++i) {
                const int t = id(std::string(w.substr(i, j - i)));
                if (t < 0) continue;
                const double c = best[i] + sc(t);
                if (c > best[j]) {
                    best[j] = c;
                    back[j] = static_cast<int>(i);
                    bt[j] = t;
                }
            }
        std::vector<int> out;
        for (std::size_t j = m; j > 0; j = static_cast<std::size_t>(back[j])) out.push_back(bt[j]);
        std::reverse(out.begin(), out.end());
        return out;
    }

private:
    std::map<int, double> sc_;
};

// Longest match over the raw stream, no pretokens; tokens may cross spaces. The token at q is
// fixed once bytes q .. q + lmax - 1 are known.
class ToyRawMPE : public ToyBase {
public:
    int longest(std::string_view s, std::size_t q) const {
        for (std::size_t L = std::min(lmax(), s.size() - q); L >= 1; --L) {
            const int t = id(std::string(s.substr(q, L)));
            if (t >= 0) return t;
        }
        return static_cast<unsigned char>(s[q]);
    }
    std::vector<int> encode(std::string_view s) const override {
        std::vector<int> out;
        for (std::size_t q = 0; q < s.size();) {
            const int t = longest(s, q);
            out.push_back(t);
            q += token(t).size();
        }
        return out;
    }
    std::size_t stable_prefix(std::string_view s, std::size_t hint) const override {
        std::size_t q = hint;
        while (q < s.size() && q + lmax() <= s.size()) q += token(longest(s, q)).size();
        return q;
    }
};

// ---------------------------------------------------------------------------------------------
// The planted byte-level Markov chain: {a, b, c, space}; state = last byte and word length;
// no space after a space; a space is forced after kMaxWord letters.
// ---------------------------------------------------------------------------------------------

constexpr int kMaxWord = 4;
const std::string kAlpha = "abc ";

struct Chain {
    double T[4][4];  // T[prev][next] over kAlpha
    explicit Chain(std::uint64_t seed) {
        std::mt19937_64 r(seed);
        std::uniform_real_distribution<double> U(0.2, 1.0);
        for (int a = 0; a < 4; ++a) {
            double z = 0;
            for (int b = 0; b < 4; ++b) {
                T[a][b] = (a == 3 && b == 3) ? 0.0 : U(r);
                z += T[a][b];
            }
            for (int b = 0; b < 4; ++b) T[a][b] /= z;
        }
    }
    static int sym(unsigned char c) {
        const auto k = kAlpha.find(static_cast<char>(c));
        return k == std::string::npos ? -1 : static_cast<int>(k);
    }
    struct State {
        int prev = 3, len = 0;
    };
    // P(next = c | state)
    double step(const State& s, unsigned char c) const {
        const int b = sym(c);
        if (b < 0) return 0;
        if (s.len == kMaxWord) return b == 3 ? 1.0 : 0.0;
        if (s.prev == 3) {
            if (b == 3) return 0;
            const double z = T[3][0] + T[3][1] + T[3][2];
            return T[3][b] / z;
        }
        return T[s.prev][b];
    }
    static State advance(State s, unsigned char c) {
        const int b = sym(c);
        s.prev = b;
        s.len = (b == 3) ? 0 : s.len + 1;
        return s;
    }
    // P(the stream starts with y), and the state after it.
    double prob(std::string_view y, State* out = nullptr) const {
        State s;
        double p = 1;
        for (char c : y) {
            p *= step(s, static_cast<unsigned char>(c));
            if (p == 0) break;
            s = advance(s, static_cast<unsigned char>(c));
        }
        if (out) *out = s;
        return p;
    }
    std::array<double, 256> next(std::string_view x) const {
        State s;
        prob(x, &s);
        std::array<double, 256> d{};
        for (char c : kAlpha) d[static_cast<unsigned char>(c)] = step(s, static_cast<unsigned char>(c));
        return d;
    }
    std::string sample(std::size_t n, std::mt19937_64& r) const {
        std::string y;
        State s;
        std::uniform_real_distribution<double> U(0, 1);
        while (y.size() < n) {
            double u = U(r);
            char pick = ' ';
            for (char c : kAlpha) {
                u -= step(s, static_cast<unsigned char>(c));
                if (u <= 0) {
                    pick = c;
                    break;
                }
            }
            if (step(s, static_cast<unsigned char>(pick)) == 0)
                for (char c : kAlpha)
                    if (step(s, static_cast<unsigned char>(c)) > 0) pick = c;
            y.push_back(pick);
            s = advance(s, static_cast<unsigned char>(pick));
        }
        return y;
    }
};

// ---------------------------------------------------------------------------------------------
// The exact token model of the tokenized chain: P(t) = P(the stream's canonical encoding starts
// with t), by enumerating the K bytes after decode(t) that decide its last tokens.
// ---------------------------------------------------------------------------------------------

class TrueModel : public br::Model {
public:
    TrueModel(const Chain& ch, const ToyBase& tok, int K) : ch_(ch), tok_(tok), K_(K) {}
    void next(std::span<const int> h, std::vector<double>& out) const override {
        const std::vector<int> hv(h.begin(), h.end());
        auto it = memo_.find(hv);
        if (it != memo_.end()) {
            out = it->second;
            return;
        }
        out.assign(static_cast<std::size_t>(tok_.vocab_size()), 0.0);
        const double base = ptok(hv);
        if (!(base > 0)) {
            for (double& x : out) x = 1.0 / tok_.vocab_size();
        } else {
            std::vector<int> t(hv);
            t.push_back(0);
            for (int v = 0; v < tok_.vocab_size(); ++v) {
                bool in = true;
                for (char c : tok_.token(v)) in = in && Chain::sym(static_cast<unsigned char>(c)) >= 0;
                if (!in) continue;
                t.back() = v;
                out[static_cast<std::size_t>(v)] = ptok(t) / base;
            }
        }
        memo_[hv] = out;
    }
    // P(the canonical encoding starts with t)
    double ptok(const std::vector<int>& t) const {
        auto it = pmemo_.find(t);
        if (it != pmemo_.end()) return it->second;
        std::string tb;
        for (int x : t) tb.append(tok_.token(x));
        Chain::State st;
        const double p0 = ch_.prob(tb, &st);
        double total = 0;
        if (p0 > 0) {
            std::string y = tb;
            std::function<void(Chain::State, double, int)> rec = [&](Chain::State s, double p, int left) {
                if (left == 0) {
                    const std::vector<int> e = tok_.encode(y);
                    if (e.size() >= t.size() && std::equal(t.begin(), t.end(), e.begin())) total += p;
                    return;
                }
                for (char c : kAlpha) {
                    const double q = ch_.step(s, static_cast<unsigned char>(c));
                    if (q == 0) continue;
                    y.push_back(c);
                    rec(Chain::advance(s, static_cast<unsigned char>(c)), p * q, left - 1);
                    y.pop_back();
                }
            };
            rec(st, p0, K_);
        }
        return pmemo_[t] = total;
    }

private:
    const Chain& ch_;
    const ToyBase& tok_;
    int K_;
    mutable std::map<std::vector<int>, std::vector<double>> memo_;
    mutable std::map<std::vector<int>, double> pmemo_;
};

// A model that leaks: 0.7 x the true model + 0.3 x uniform over every token (valid or not).
class LeakyModel : public br::Model {
public:
    explicit LeakyModel(const br::Model& m) : m_(m) {}
    void next(std::span<const int> h, std::vector<double>& out) const override {
        m_.next(h, out);
        const double u = 1.0 / static_cast<double>(out.size());
        for (double& x : out) x = 0.7 * x + 0.3 * u;
    }

private:
    const br::Model& m_;
};

// ---------------------------------------------------------------------------------------------
// An independent brute force: every segmentation of x into tokens (the last one may overhang),
// validity by full re-encoding, each next-token distribution truncated and renormalised.
// ---------------------------------------------------------------------------------------------

class BruteForce {
public:
    BruteForce(const br::Tokenizer& tok, const br::Model& m) : tok_(tok), m_(m) {}
    bool valid(const std::vector<int>& t) const {
        std::string s;
        for (int x : t) s.append(tok_.token(x));
        return tok_.encode(s) == t;
    }
    const std::vector<double>& cond(const std::vector<int>& h) {
        auto it = memo_.find(h);
        if (it != memo_.end()) return it->second;
        std::vector<double> q;
        m_.next(h, q);
        std::vector<int> t(h);
        t.push_back(0);
        double z = 0;
        for (int v = 0; v < tok_.vocab_size(); ++v) {
            t.back() = v;
            if (q[static_cast<std::size_t>(v)] > 0 && !valid(t)) q[static_cast<std::size_t>(v)] = 0;
            z += q[static_cast<std::size_t>(v)];
        }
        if (z > 0)
            for (double& x : q) x /= z;
        return memo_[h] = q;
    }
    double Z(std::string_view x) {
        if (x.empty()) return 1;
        double tot = 0;
        std::vector<int> h;
        std::function<void(std::size_t, double)> rec = [&](std::size_t pos, double w) {
            if (pos >= x.size()) {
                tot += w;
                return;
            }
            const std::vector<double> q = cond(h);
            for (int v = 0; v < tok_.vocab_size(); ++v) {
                if (!(q[static_cast<std::size_t>(v)] > 0)) continue;
                const std::string_view b = tok_.token(v);
                const std::size_t k = std::min(b.size(), x.size() - pos);
                if (b.compare(0, k, x.substr(pos, k)) != 0) continue;
                h.push_back(v);
                rec(pos + b.size(), w * q[static_cast<std::size_t>(v)]);
                h.pop_back();
            }
        };
        rec(0, 1.0);
        return tot;
    }

private:
    const br::Tokenizer& tok_;
    const br::Model& m_;
    std::map<std::vector<int>, std::vector<double>> memo_;
};

// ---------------------------------------------------------------------------------------------

static ToyBPE make_bpe() {
    ToyBPE t;
    t.merge("a", "b");
    t.merge(" ", "a");
    t.merge("b", "c");
    t.merge("c", "a");
    t.merge("ab", "c");
    t.merge(" ", "ab");
    t.merge(" ", "c");
    t.merge(" ", "ca");
    t.merge("b", "b");
    t.merge("a", "a");
    t.merge(" a", "c");
    t.merge("bc", "a");
    return t;
}
static const std::vector<std::string> kExtra = {"ab", " a", "bc", "ca", "abc", " ab", " c", " ca", "bb", "aa", " ac", "bca"};
static ToyMPE make_mpe() {
    ToyMPE t;
    for (const auto& s : kExtra) t.add(s);
    return t;
}
static ToyUnigram make_unigram() {
    ToyUnigram t;
    t.score("a", -1.5);
    t.score("b", -1.5);
    t.score("c", -1.5);
    t.score(" ", -1.6);
    const std::vector<double> sc = {-2.0, -2.5, -2.6, -2.4, -3.0, -2.8, -2.7, -3.1, -3.5, -3.3, -3.4, -3.2};
    for (std::size_t k = 0; k < kExtra.size(); ++k) t.score(kExtra[k], sc[k]);
    return t;
}
static ToyRawMPE make_raw() {
    ToyRawMPE t;
    for (const char* s : {"ab", "bc", "ca", "abc", "a ", "b a", " c", "c b", "a b", " ab"}) t.add(s);
    return t;
}

static std::vector<int> ids(const ToyBase& t, std::initializer_list<const char*> ss) {
    std::vector<int> v;
    for (const char* s : ss) v.push_back(t.id(s));
    return v;
}

struct Summary {
    double max_err = 0, max_rel = 0, naive_tv_sum = 0, gap_sum = 0, max_gap = 0;
    int naive_n = 0, gap_n = 0, prompts = 0;
    double tv_space = 0, tv_letter = 0;
    int n_space = 0, n_letter = 0;
};

static Summary exactness(const char* name, const ToyBase& tok, int K, const Chain& ch,
                         const std::vector<std::string>& prompts) {
    TrueModel tm(ch, tok, K);
    br::Bridge b(tok, tm);
    Summary S;
    for (const std::string& x : prompts) {
        // incremental stream: every prefix, checked against the chain and the from-scratch bridge
        br::Stream st(b);
        double sum_log2 = 0;
        for (std::size_t k = 0; k <= x.size(); ++k) {
            const std::array<double, 256> got = st.next_byte();
            const std::array<double, 256> want = ch.next(std::string_view(x).substr(0, k));
            double e = 0;
            for (int a = 0; a < 256; ++a) e = std::max(e, std::abs(got[static_cast<std::size_t>(a)] - want[static_cast<std::size_t>(a)]));
            S.max_err = std::max(S.max_err, e);
            CHECK(e < 1e-9);
            if (k < x.size()) {
                sum_log2 += std::log2(got[static_cast<unsigned char>(x[k])]);
                st.push(std::string_view(x).substr(k, 1));
            }
        }
        // the prefix identity
        const double lp = st.log2_prefix();
        const double truth = std::log2(ch.prob(x));
        const double rel = std::abs(lp - truth) / std::max(1.0, std::abs(truth));
        const double rel2 = std::abs(sum_log2 - truth) / std::max(1.0, std::abs(truth));
        S.max_rel = std::max({S.max_rel, rel, rel2});
        CHECK(rel < 1e-9);
        CHECK(rel2 < 1e-9);
        // canonical segmentation: one cover only, so never more probable than the marginal
        const double lc = st.log2_canonical();
        CHECK(lc <= lp + 1e-9);
        S.gap_sum += lp - lc;
        S.max_gap = std::max(S.max_gap, lp - lc);
        ++S.gap_n;
        // the naive estimate: the first byte of the next token after encode(x)
        if (!x.empty()) {
            std::vector<double> q;
            tm.next(tok.encode(x), q);
            std::array<double, 256> nv{};
            double z = 0;
            for (int v = 0; v < tok.vocab_size(); ++v) {
                const std::string_view tb = tok.token(v);
                nv[static_cast<unsigned char>(tb[0])] += q[static_cast<std::size_t>(v)];
                z += q[static_cast<std::size_t>(v)];
            }
            const std::array<double, 256> want = ch.next(x);
            double tv = 0;
            for (int a = 0; a < 256; ++a) tv += std::abs(nv[static_cast<std::size_t>(a)] / z - want[static_cast<std::size_t>(a)]);
            S.naive_tv_sum += tv / 2;
            ++S.naive_n;
            if (x.back() == ' ') {
                S.tv_space += tv / 2;
                ++S.n_space;
            } else {
                S.tv_letter += tv / 2;
                ++S.n_letter;
            }
        }
        ++S.prompts;
    }
    std::printf("%-8s prompts %d | max |bridge - chain| %.2e | prefix identity rel err %.2e | naive TV %.4f | canonical gap mean %.4f max %.4f bits\n",
                name, S.prompts, S.max_err, S.max_rel, S.naive_n ? S.naive_tv_sum / S.naive_n : 0.0,
                S.gap_n ? S.gap_sum / S.gap_n : 0.0, S.max_gap);
    std::printf("%-8s naive TV after a letter %.4f (n %d), after a space %.4f (n %d)\n", name,
                S.n_letter ? S.tv_letter / S.n_letter : 0.0, S.n_letter, S.n_space ? S.tv_space / S.n_space : 0.0, S.n_space);
    return S;
}

static void leaky(const char* name, const ToyBase& tok, int K, const Chain& ch, const std::vector<std::string>& prompts) {
    TrueModel tm(ch, tok, K);
    LeakyModel lm(tm);
    br::Bridge b(tok, lm);
    BruteForce bf(tok, lm);
    double max_err = 0, max_mass = 0;
    for (const std::string& x : prompts) {
        br::Stream st(b);
        st.push(x);
        const std::array<double, 256> got = st.next_byte();
        std::array<double, 256> want{};
        double z = 0;
        for (int a = 0; a < 256; ++a) {
            std::string y(x);
            y.push_back(static_cast<char>(a));
            want[static_cast<std::size_t>(a)] = bf.Z(y);
            z += want[static_cast<std::size_t>(a)];
        }
        for (int a = 0; a < 256; ++a) max_err = std::max(max_err, std::abs(got[static_cast<std::size_t>(a)] - want[static_cast<std::size_t>(a)] / z));
        // a proper process: the extensions' prefix probabilities add up to the prefix's
        const double zx = bf.Z(x);
        max_mass = std::max(max_mass, std::abs(z - zx) / zx);
        CHECK(std::abs(st.log2_prefix() - std::log2(zx)) < 1e-9 * std::max(1.0, std::abs(std::log2(zx))));
    }
    CHECK(max_err < 1e-9);
    CHECK(max_mass < 1e-9);
    std::printf("%-8s leaky model: max |bridge - brute force| %.2e | mass identity %.2e\n", name, max_err, max_mass);
}

static void sampling(const ToyBase& tok, int K, const Chain& ch, const std::string& prompt) {
    TrueModel tm(ch, tok, K);
    br::Bridge b(tok, tm);
    std::mt19937_64 r(7);
    const std::array<double, 256> want = ch.next(prompt);
    std::array<double, 256> cnt{};
    const int N = 4000;
    for (int k = 0; k < N; ++k) {
        const std::string s = br::sample_after(b, prompt, 1, 1.0, r);
        CHECK(!s.empty());
        if (!s.empty()) cnt[static_cast<unsigned char>(s[0])] += 1;
    }
    double worst = 0;
    for (int a = 0; a < 256; ++a) {
        const double p = want[static_cast<std::size_t>(a)];
        const double sd = std::sqrt(std::max(p * (1 - p), 1e-12) / N);
        worst = std::max(worst, std::abs(cnt[static_cast<std::size_t>(a)] / N - p) / sd);
    }
    CHECK(worst < 4.5);
    std::printf("sampling after \"%s\": worst |freq - p| / sd = %.2f over %d draws\n", prompt.c_str(), worst, N);
}

int main() {
    const Chain ch(20261029);
    ToyBPE bpe = make_bpe();
    ToyMPE mpe = make_mpe();
    ToyUnigram uni = make_unigram();
    ToyRawMPE raw = make_raw();

    // Known answers: BPE applies merges by rank (a merge-order error changes these).
    CHECK(bpe.encode("cab") == ids(bpe, {"c", "ab"}));
    CHECK(bpe.encode("abc") == ids(bpe, {"abc"}));
    CHECK(bpe.encode("bca") == ids(bpe, {"bca"}));
    CHECK(bpe.encode(" ab") == ids(bpe, {" ab"}));
    CHECK(bpe.encode(" ac") == ids(bpe, {" ac"}));
    CHECK(bpe.encode("aab") == ids(bpe, {"a", "ab"}));
    CHECK(bpe.encode("ca") == ids(bpe, {"ca"}));
    CHECK(bpe.encode("ab cab") == ids(bpe, {"ab", " c", "ab"}));
    CHECK(mpe.encode("cab") == ids(mpe, {"ca", "b"}));
    CHECK(uni.encode("cab") == ids(uni, {"c", "ab"}));
    CHECK(raw.encode("ab cab") == ids(raw, {"ab", " c", "ab"}));
    CHECK(raw.encode("cb ab a") == ids(raw, {"c", "b a", "b a"}));
    // stable prefixes
    CHECK(bpe.stable_prefix("ab ca", 0) == 2);
    CHECK(raw.stable_prefix("abcab", 0) == 3);

    // Samples from the chain.
    std::mt19937_64 r(11);
    std::vector<std::string> samples, prompts, short_prompts;
    for (int k = 0; k < 30; ++k) samples.push_back(ch.sample(24, r));
    for (int k = 0; k < 24; ++k) prompts.push_back(ch.sample(4 + (k % 12), r));
    for (int k = 0; k < 6; ++k) short_prompts.push_back(ch.sample(3 + (k % 4), r));

    // The tokenizer contract the bridge relies on.
    const std::vector<std::pair<const char*, const br::Tokenizer*>> toks = {
        {"bpe", &bpe}, {"mpe", &mpe}, {"unigram", &uni}, {"raw-mpe", &raw}};
    for (const auto& [name, t] : toks) {
        std::string why;
        const int bad = br::check_contract(*t, samples, &why);
        if (bad) std::printf("%s contract: %d violations, first: %s\n", name, bad, why.c_str());
        CHECK(bad == 0);
    }

    // Rule 1, 2: exactness on the planted chain. K = bytes that decide the last tokens.
    const int Kw = kMaxWord + 1, Kr = static_cast<int>(raw.lmax()) - 1;
    const Summary sb = exactness("bpe", bpe, Kw, ch, prompts);
    const Summary sm = exactness("mpe", mpe, Kw, ch, prompts);
    const Summary su = exactness("unigram", uni, Kw, ch, prompts);
    const Summary sr = exactness("raw-mpe", raw, Kr, ch, prompts);
    CHECK(sr.prompts == static_cast<int>(prompts.size()));
    // The naive estimate is biased (otherwise the check above would be trivial).
    CHECK(sb.naive_tv_sum / sb.naive_n > 1e-3);
    CHECK(sm.naive_tv_sum / sm.naive_n > 1e-3);
    CHECK(su.naive_tv_sum / su.naive_n > 1e-3);

    // Rule 3: a leaky model, truncate-renormalised, against brute force.
    leaky("bpe", bpe, Kw, ch, short_prompts);
    leaky("mpe", mpe, Kw, ch, short_prompts);
    leaky("unigram", uni, Kw, ch, short_prompts);
    leaky("raw-mpe", raw, Kr, ch, short_prompts);

    // Sampling whole tokens after a prompt that ends inside a word is exact too.
    sampling(bpe, Kw, ch, "ab c");
    sampling(mpe, Kw, ch, "ab ca");

    std::printf("token_bridge_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
