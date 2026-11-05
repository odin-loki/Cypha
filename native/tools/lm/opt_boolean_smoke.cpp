// O3 (GENERATION_PLAN §13.41; docs/reports/OPT_BOOLEAN.md): every decision O3 rewrote, against its old form.
//   copy_pred      the SSE2 scan that skips j it cannot use, against the old loop over every j (n, prediction, source)
//   copy_bucket    the table, against the chain of compares, for n in [-1000, 1000]
//   trie::quantize, trie::used_count  counted compares, against the binary search (every count up to 2^26, and more)
//   similar_top_k  packed keys (radix sort, nth_element and sort, or partial_sort by k), against std::partial_sort
//   upper_index    the branchless binary search, against std::upper_bound
//   SimilarContext::bucket, ClassContext::bucket  sums of compare results, against the chains (NaN included)
//   gate_class.hpp     the byte tables, against the library calls and string compares they replace, on all 256 bytes
//   mix::byte_class, mix::path_bits  (online/mix_core.hpp) the table and the masked step, against the chain and the branch
// Exit 0 when all agree.
//
// Author: Odin Loch

#include "cypha/lm/class_context.hpp"
#include "cypha/lm/copy_expert.hpp"
#include "cypha/lm/gate_class.hpp"
#include "cypha/lm/packed_trie.hpp"
#include "cypha/lm/similar_context.hpp"
#include "cypha/online/mix_core.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace cl = cypha::cyphalm;

// The copy expert before O3, unchanged.
int copy_pred_old(const std::uint8_t* s, std::size_t i, int& n, std::size_t* at) {
    n = 0;
    int pred = -1;
    const std::size_t lo = i > cl::kCopyWin ? i - cl::kCopyWin : 0;
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
int copy_bucket_old(int n) {
    return n == 0 ? 0 : n < 4 ? 1 : n < 6 ? 2 : n < 8 ? 3 : n < 12 ? 4 : n < 16 ? 5 : n < 32 ? 6 : 7;
}
int sim_bucket_old(const cl::SimilarContext::Hit& h) {
    if (h.live == 0) return 0;
    const int sb = h.top < 0.5 ? 0 : h.top < 0.65 ? 1 : h.top < 0.8 ? 2 : h.top < 0.9 ? 3 : 4;
    const int ab = h.agree < 0.5 ? 0 : h.agree < 0.8 ? 1 : h.agree < 0.99 ? 2 : 3;
    const int lb = h.live == 1 ? 0 : h.live < 8 ? 1 : 2;
    return 1 + ((sb * 4 + ab) * 3 + lb) * 2 + (h.word > 0);
}
int cls_bucket_old(const cl::ClassContext::Hit& h) {
    const int cb = h.total <= 1 ? 0 : h.total <= 3 ? 1 : h.total <= 15 ? 2 : 3;
    return h.order * 4 + cb;
}
// mix_core.hpp's path_bits before O3, unchanged.
void path_bits_old(const double* D, unsigned y, double out[8]) {
    unsigned lo = 0, size = 256;
    for (int j = 0; j < 8; ++j) {
        const unsigned h = size / 2;
        double s0 = 0, s1 = 0;
        for (unsigned s = lo; s < lo + h; ++s) s0 += D[s];
        for (unsigned s = lo + h; s < lo + size; ++s) s1 += D[s];
        out[j] = s0 + s1 > 0 ? s1 / (s0 + s1) : 0.5;
        if ((y >> (7 - j)) & 1) lo += h;
        size = h;
    }
}

int fails = 0;
void fail(const char* what, long long a, long long b, long long c) {
    if (++fails <= 20) std::fprintf(stderr, "opt_boolean_smoke: %s differs (%lld %lld %lld)\n", what, a, b, c);
}

// Every position i of s (or every step-th): the new copy_pred equals the old one.
std::size_t check_copy(const std::vector<std::uint8_t>& s, std::size_t step) {
    std::size_t checked = 0;
    for (std::size_t i = 0; i <= s.size(); i += step) {
        int n0 = -7, n1 = -7;
        std::size_t a0 = 12345, a1 = 12345;
        const int p0 = copy_pred_old(s.data(), i, n0, &a0);
        const int p1 = cl::copy_pred(s.data(), i, n1, &a1);
        if (p0 != p1 || n0 != n1 || a0 != a1) fail("copy_pred", static_cast<long long>(i), n0, n1);
        int n2 = -7;
        if (cl::copy_pred(s.data(), i, n2) != p0 || n2 != n0) fail("copy_pred (no source)", static_cast<long long>(i), n0, n2);
        ++checked;
    }
    return checked;
}

}  // namespace

int main() {
    std::mt19937_64 rng(20261105);
    std::size_t positions = 0;
    // Random texts over alphabets of 2, 3, 4, 8, 26 and 256 bytes: short matches everywhere, many ties.
    for (const int k : {2, 3, 4, 8, 26, 256}) {
        std::vector<std::uint8_t> s(20000 + 1000 * static_cast<std::size_t>(k % 7));
        for (auto& c : s) c = static_cast<std::uint8_t>('a' + rng() % static_cast<unsigned>(k));
        positions += check_copy(s, k <= 4 ? 7 : 3);
    }
    // Repeats: long matches (past 64 bytes), matches that reach the window's start, a text longer than the window.
    {
        std::string base;
        for (int w = 0; w < 400; ++w) base += std::string("the ") + static_cast<char>('a' + rng() % 26) + "lpha beta gamma ";
        std::string t;
        while (t.size() < 40000) {
            const std::size_t a = rng() % (base.size() - 200), l = 20 + rng() % 180;
            t += base.substr(a, l);
            if (rng() % 3 == 0) t += static_cast<char>('A' + rng() % 26);
        }
        const std::vector<std::uint8_t> s(t.begin(), t.end());
        positions += check_copy(s, 5);
        // around the window's edge, every position
        std::vector<std::uint8_t> s2(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(cl::kCopyWin + 300));
        for (std::size_t i = cl::kCopyWin - 200; i <= s2.size(); ++i) {
            int n0 = 0, n1 = 0;
            std::size_t a0 = 0, a1 = 0;
            if (copy_pred_old(s2.data(), i, n0, &a0) != cl::copy_pred(s2.data(), i, n1, &a1) || n0 != n1 || a0 != a1)
                fail("copy_pred (window edge)", static_cast<long long>(i), n0, n1);
            ++positions;
        }
    }
    // One byte repeated (every j matches; n reaches 64 at once), and short texts, every length.
    {
        const std::vector<std::uint8_t> s(5000, 'x');
        positions += check_copy(s, 1);
        for (std::size_t len = 0; len < 200; ++len) {
            std::vector<std::uint8_t> t(len);
            for (auto& c : t) c = static_cast<std::uint8_t>("ab"[rng() % 2]);
            positions += check_copy(t, 1);
        }
    }
    // copy_bucket: every n in [-1000, 1000] and the int limits.
    for (int n = -1000; n <= 1000; ++n)
        if (cl::copy_bucket(n) != copy_bucket_old(n)) fail("copy_bucket", n, copy_bucket_old(n), cl::copy_bucket(n));
    for (const int n : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()})
        if (cl::copy_bucket(n) != copy_bucket_old(n)) fail("copy_bucket", n, copy_bucket_old(n), cl::copy_bucket(n));
    // SimilarContext::bucket: the thresholds, either side of each, NaN and infinities.
    {
        const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
        std::vector<double> xs = {nan, -inf, inf, -1.0, 0.0, 1.0, 2.0};
        for (const double t : {0.5, 0.65, 0.8, 0.9, 0.99})
            xs.push_back(t), xs.push_back(std::nextafter(t, 0.0)), xs.push_back(std::nextafter(t, 1.0));
        for (int r = 0; r < 200; ++r) xs.push_back(static_cast<double>(rng() % 100001) / 100000.0);
        std::size_t nb = 0;
        for (const double top : xs)
            for (const double agree : xs)
                for (const int live : {0, 1, 2, 7, 8, 9, 1000, -1})
                    for (const int word : {0, 1, 5, -1}) {
                        cl::SimilarContext::Hit h;
                        h.top = top, h.agree = agree, h.live = live, h.word = word;
                        if (cl::SimilarContext::bucket(h) != sim_bucket_old(h)) fail("SimilarContext::bucket", live, sim_bucket_old(h), cl::SimilarContext::bucket(h));
                        ++nb;
                    }
        std::fprintf(stderr, "opt_boolean_smoke: SimilarContext::bucket on %zu hits\n", nb);
    }
    // ClassContext::bucket: orders 0-3, totals 0-100 and large ones.
    for (int o = 0; o < 4; ++o)
        for (std::uint64_t t : {0ull, 1ull, 2ull, 3ull, 4ull, 15ull, 16ull, 17ull, 100ull, 1ull << 40, ~0ull}) {
            cl::ClassContext::Hit h;
            h.order = o, h.total = t;
            if (cl::ClassContext::bucket(h) != cls_bucket_old(h)) fail("ClassContext::bucket", o, static_cast<long long>(t), 0);
        }
    // gate_class.hpp: every byte, against the library calls they replace (the C locale: this program sets none).
    for (int c = 0; c < 256; ++c) {
        const char ch = static_cast<char>(c);
        const auto u = static_cast<unsigned char>(c);
        const int cls_old = std::isspace(u) ? 0 : std::isalpha(u) ? 1 : std::isdigit(u) ? 2 : std::strchr("[]{}<>|&=", u) && u ? 3 : 4;
        if (cl::gate_class(u) != cls_old) fail("gate_class", c, cls_old, cl::gate_class(u));
        const bool mk_old = std::string_view("[]{}|=<>&*#").find(ch) != std::string_view::npos;
        if (cl::is_gate_markup(ch) != mk_old) fail("is_gate_markup", c, mk_old, 0);
        const std::string t1(1, ch), t2 = std::string(1, ch) + ".";
        for (const std::string& t : {t1, t2, std::string()}) {
            const bool end_old = t == "." || t == "!" || t == "?" || t == "\n";
            if (cl::is_sentence_end(t) != end_old) fail("is_sentence_end", c, end_old, static_cast<long long>(t.size()));
        }
    }
    // packed_trie.hpp: quantize (the ladder by counted compares) against the binary search, every c up to 2^26, near
    // every ladder value, and 2^24 random c above; used_count in each mode. Then that no octave holds more ladder steps
    // than the window counts.
    {
        namespace tr = cl::trie;
        std::size_t nq = 0;
        const auto one = [&](std::uint32_t c) {
            if (tr::quantize(c) != tr::quantize_search(c)) fail("quantize", c, tr::quantize_search(c), tr::quantize(c));
            const std::uint32_t ref = c == 0 ? 0 : (c < 2 ? 0 : tr::qdec[tr::quantize_search(c)]);
            if (tr::used_count(c, tr::Mode::Synth) != ref) fail("used_count Synth", c, ref, tr::used_count(c, tr::Mode::Synth));
            if (tr::used_count(c, tr::Mode::Full) != c || tr::used_count(c, tr::Mode::DropCtx) != c) fail("used_count Full", c, 0, 0);
            ++nq;
        };
        for (std::uint32_t c = 0; c <= (1u << 26); ++c) one(c);
        for (int s = 0; s < 256; ++s)
            for (std::int64_t d = -3; d <= 3; ++d) {
                const std::int64_t c = static_cast<std::int64_t>(tr::qdec[static_cast<std::size_t>(s)]) + d;
                if (c >= 0 && c <= 0xffffffffll) one(static_cast<std::uint32_t>(c));
            }
        for (int r = 0; r < (1 << 24); ++r) one(static_cast<std::uint32_t>(rng()));
        for (std::uint32_t c : {0xffffffffu, 0xfffffffeu, 0x80000000u, 0x7fffffffu, 0x88d6e0f0u, 0x88d6e0f1u}) one(c);
        for (int w = 7; w <= 32; ++w) {
            const std::uint64_t lo = std::uint64_t{1} << (w - 1), hi = std::uint64_t{1} << w;
            int in = 0;
            for (int s = 65; s < 255; ++s) in += tr::qdec[static_cast<std::size_t>(s)] >= lo && tr::qdec[static_cast<std::size_t>(s)] < hi;
            if (in > tr::QuantWindow::kSpan) fail("ladder steps in an octave", w, in, tr::QuantWindow::kSpan);
        }
        std::fprintf(stderr, "opt_boolean_smoke: quantize and used_count on %zu counts\n", nq);
    }
    // similar_context.hpp: similar_top_k against std::partial_sort, with many equal dots (and repeated pairs).
    {
        std::size_t nt = 0;
        // k from 0 to n and past it, so every path is taken (partial_sort below 256; nth_element and sort from 256; the
        // radix sort when k is at least a quarter of n), and the default expert's size (k 8192 of 12,000).
        for (int t = 0; t < 4000; ++t) {
            const std::size_t n = t % 400 == 7 ? 12000 : rng() % 3000;
            const std::size_t kr = t % 400 == 7 ? 8192 : t % 3 == 0 ? rng() % 70 : t % 3 == 1 ? rng() % (n + 1) : n / 4 + rng() % (n + 9);
            const std::size_t k = std::min<std::size_t>(n, kr);
            const int span = 1 + static_cast<int>(rng() % 200);
            std::vector<std::pair<int, std::uint32_t>> a(n);
            for (auto& x : a) {
                x.first = static_cast<int>(rng() % static_cast<unsigned>(span)) - span / 2;
                x.second = static_cast<std::uint32_t>(rng() % (t % 2 ? 100000u : 0x100000000ull));
                if (rng() % 16 == 0) x.first = rng() % 2 ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
            }
            if (n > 1 && t % 5 == 0) a[n - 1] = a[0];  // a repeated pair
            auto b = a;
            std::partial_sort(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(k), b.end(),
                              [](auto& x, auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
            cl::similar_top_k(a, kr);  // a k past the list's size is cut to it, as search does before the call
            if (!std::equal(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(k), b.begin())) fail("similar_top_k", t, static_cast<long long>(n), static_cast<long long>(k));
            ++nt;
        }
        std::fprintf(stderr, "opt_boolean_smoke: similar_top_k on %zu lists\n", nt);
        // upper_index against std::upper_bound: sorted arrays with repeats (empty lists repeat a start), every x near them
        std::size_t nu = 0;
        for (int t = 0; t < 3000; ++t) {
            std::vector<std::uint32_t> a(rng() % 600);
            std::uint32_t v = static_cast<std::uint32_t>(rng() % 5);
            for (auto& x : a) x = v, v += static_cast<std::uint32_t>(rng() % 4 == 0 ? 0 : rng() % 300);
            for (int q = 0; q < 50; ++q) {
                const std::uint32_t x = static_cast<std::uint32_t>(rng() % (static_cast<std::uint64_t>(v) + 10));
                const auto ref = static_cast<std::size_t>(std::upper_bound(a.begin(), a.end(), x) - a.begin());
                if (cl::upper_index(a, x) != ref) fail("upper_index", t, static_cast<long long>(ref), static_cast<long long>(cl::upper_index(a, x)));
                ++nu;
            }
            if (!a.empty()) {
                for (const std::uint32_t x : {a.front(), a.back(), 0xffffffffu}) {
                    const auto ref = static_cast<std::size_t>(std::upper_bound(a.begin(), a.end(), x) - a.begin());
                    if (cl::upper_index(a, x) != ref) fail("upper_index (ends)", t, static_cast<long long>(ref), 0);
                }
            }
        }
        std::fprintf(stderr, "opt_boolean_smoke: upper_index on %zu searches\n", nu);
    }
    // The expressions of the patches for files other jobs own (docs/reports/opt_boolean_patches/), and of one change made
    // in place inside a class, against the code they replace: InfiniGram's length and count buckets (cyphalm_trie.cpp,
    // patch), the meta mix's match log (meta_mixer.hpp, in place), the n-gram's bit position and slot claim
    // (online_ngram.hpp, patch), path_bits' step (mix_core.hpp, in place; also checked whole below).
    for (int n = -100; n <= 20000; ++n) {
        const int o = n < 4 ? 0 : n < 8 ? 1 : n < 16 ? 2 : n < 32 ? 3 : n < 64 ? 4 : n < 128 ? 5 : 6;
        const int b = n < 4 ? 0 : std::min(6, static_cast<int>(std::bit_width(static_cast<unsigned>(n))) - 2);
        if (o != b) fail("len_bucket patch", n, o, b);
        int lg = 0;
        for (int m = n + 1; m > 1 && lg < 15; m >>= 1) ++lg;
        const int n1 = n + 1;
        const int lg2 = n1 > 1 ? std::min(15, static_cast<int>(std::bit_width(static_cast<unsigned>(n1))) - 1) : 0;
        if (lg != lg2) fail("meta mix match log patch", n, lg, lg2);
    }
    for (std::uint64_t t : {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 14ull, 15ull, 16ull, 17ull, 1000ull, ~0ull}) {
        const int o = t <= 1 ? 0 : t <= 3 ? 1 : t <= 15 ? 2 : 3;
        if (o != (t > 1) + (t > 3) + (t > 15)) fail("count_bucket patch", static_cast<long long>(t), o, 0);
    }
    for (unsigned c0 = 1; c0 < 256; ++c0) {
        int j = 0;
        while ((c0 >> (j + 1)) != 0) ++j;
        if (j != static_cast<int>(std::bit_width(c0)) - 1) fail("bit position patch", c0, j, 0);
    }
    for (int ck = 0; ck < 256; ++ck)
        for (int c = 0; c < 256; c += 5) {
            const std::uint16_t n1 = static_cast<std::uint16_t>(rng()), nn = static_cast<std::uint16_t>(rng());
            std::uint16_t a1 = n1, an = nn, b1 = n1, bn = nn;
            std::uint8_t ack = static_cast<std::uint8_t>(ck), bck = static_cast<std::uint8_t>(ck);
            if (ack != c) ack = static_cast<std::uint8_t>(c), a1 = an = 0;  // the old claim
            const auto keep = static_cast<std::uint16_t>(0u - static_cast<unsigned>(bck == c));
            b1 = static_cast<std::uint16_t>(b1 & keep), bn = static_cast<std::uint16_t>(bn & keep), bck = static_cast<std::uint8_t>(c);
            if (a1 != b1 || an != bn || ack != bck) fail("claim patch", ck, c, 0);
        }
    // mix_core.hpp (changed in place): byte_class on every byte against the chain of library calls; path_bits on random
    // distributions (some with empty halves, so the 0.5 case is taken) against the old loop, compared bit for bit.
    for (int c = 0; c < 256; ++c) {
        const auto u = static_cast<unsigned char>(c);
        const int old = std::isspace(u) ? 0 : std::isalpha(u) ? 1 : std::isdigit(u) ? 2 : std::strchr("[]{}<>|&=", u) && u ? 3 : 4;
        if (cypha::online::mix::byte_class(u) != old) fail("mix::byte_class", c, old, cypha::online::mix::byte_class(u));
    }
    {
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (int t = 0; t < 2000; ++t) {
            double D[256];
            for (double& d : D) d = (t % 3 == 0 && rng() % 4 == 0) ? 0.0 : ud(rng);
            if (t % 7 == 0) std::fill(D + 128, D + 256, 0.0);
            for (unsigned y = 0; y < 256; ++y) {
                double a[8], b[8];
                path_bits_old(D, y, a);
                cypha::online::mix::path_bits(D, y, b);
                if (std::memcmp(a, b, sizeof a) != 0) fail("mix::path_bits", t, y, 0);
            }
        }
    }
    std::fprintf(stderr, "opt_boolean_smoke: copy_pred on %zu positions; %s\n", positions, fails ? "FAIL" : "pass");
    return fails ? 1 : 0;
}
