// Known-answer test for cypha/cyphalm/fm_infinigram.hpp (RAM_PROFILE M5): the
// FM-index must give InfiniGram's answer (n, total, all 256 counts) on every
// query. Random texts over small and full alphabets, byte 0 included; queries
// with and without hints, min_total above 1, and cursors used right, reset and
// misused. Prints the count of checks; exits 1 on any difference.
//
// fm_infinigram_test [--bench FILE BYTES]   (--bench: time both indexes on a file's prefix)

#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/cyphalm/infinigram.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using cypha::cyphalm::FmInfiniGram;
using cypha::cyphalm::InfiniGram;

namespace {

std::size_t checks = 0, failures = 0;

void same(const InfiniGram::Result& a, const InfiniGram::Result& b, const char* what, std::size_t at) {
    ++checks;
    if (a.n == b.n && a.total == b.total && a.count == b.count) return;
    if (failures++ < 10)
        std::printf("FAIL %s at %zu: n %d/%d total %llu/%llu\n", what, at, a.n, b.n,
                    static_cast<unsigned long long>(a.total), static_cast<unsigned long long>(b.total));
}

std::vector<std::uint8_t> random_text(std::mt19937_64& g, std::size_t n, int sigma, bool repeats) {
    std::vector<std::uint8_t> t(n);
    std::uniform_int_distribution<int> sym(0, sigma - 1);
    for (std::size_t i = 0; i < n; ++i) {
        // Copy an earlier span now and then, so long matches occur.
        if (repeats && i > 20 && g() % 50 == 0) {
            const std::size_t from = g() % (i - 10), len = std::min<std::size_t>(n - i, 5 + g() % 60);
            for (std::size_t k = 0; k < len && from + k < i; ++k, ++i) t[i] = t[from + k];
            if (i >= n) break;
        }
        t[i] = static_cast<std::uint8_t>(sigma == 256 ? sym(g) : (sym(g) * 37 + (sigma > 4 ? 'a' : 0)) % 256);
    }
    return t;
}

void one_text(std::mt19937_64& g, std::size_t n, int sigma, bool repeats) {
    const auto t = random_text(g, n, sigma, repeats);
    const InfiniGram sa(t.data(), t.size());
    const FmInfiniGram fm(t.data(), t.size());
    // Queries over a second text drawn the same way (unseen bytes possible), plus the text itself.
    const auto q = random_text(g, n / 2 + 50, sigma == 256 ? 256 : sigma + 1, repeats);
    for (const auto* s : {&t, &q}) {
        const std::size_t L = s->size();
        // 1) every position, hint chain, cap 64, stateless and with a cursor
        FmInfiniGram::Cursor cur;
        int prev_a = -1, prev_b = -1, prev_c = -1;
        for (std::size_t i = 1; i <= L; ++i) {
            const std::size_t len = std::min<std::size_t>(i, 64);
            const auto a = sa.query(s->data() + (i - len), len, 64, prev_a < 0 ? -1 : std::min(prev_a + 1, 64));
            const auto b = fm.query(s->data() + (i - len), len, 64, prev_b < 0 ? -1 : std::min(prev_b + 1, 64));
            const auto c = fm.query(s->data() + (i - len), len, 64, prev_c < 0 ? -1 : std::min(prev_c + 1, 64), 1, cur);
            same(a, b, "chain", i);
            same(a, c, "chain cursor", i);
            prev_a = a.n, prev_b = b.n, prev_c = c.n;
        }
        // 2) random contexts, caps, hints and min_total; one cursor reused across unrelated queries
        cur = {};
        for (int k = 0; k < 300; ++k) {
            const std::size_t i = 1 + g() % L, len = 1 + g() % std::min<std::size_t>(i, 80);
            const int cap = static_cast<int>(g() % 90), hint = g() % 3 == 0 ? -1 : static_cast<int>(g() % 90);
            const std::uint64_t mt = 1 + g() % 4;
            const auto a = sa.query(s->data() + (i - len), len, cap, hint, mt);
            same(a, fm.query(s->data() + (i - len), len, cap, hint, mt), "random", i);
            same(a, fm.query(s->data() + (i - len), len, cap, hint, mt, cur), "random cursor", i);
        }
    }
    // 3) match_prefix: longest prefix and its count, against InfiniGram's
    for (const auto* s : {&t, &q})
        for (int k = 0; k < 200; ++k) {
            const std::size_t at = g() % s->size(), len = std::min<std::size_t>(s->size() - at, g() % 90);
            const std::size_t cap = g() % 4 == 0 ? g() % 20 : len;
            std::size_t pos = 0, ca = 0, cb = 0;
            const std::size_t la = sa.match_prefix(s->data() + at, len, cap, 0, pos, ca);
            const std::size_t lb = fm.match_prefix(s->data() + at, len, cap, cb);
            ++checks;
            if (la != lb || ca != cb) {
                if (failures++ < 10) std::printf("FAIL match_prefix at %zu: length %zu/%zu count %zu/%zu\n", at, la, lb, ca, cb);
            }
        }
    // 4) the empty context and a context longer than the text
    same(sa.query(t.data(), 0, 10), fm.query(t.data(), 0, 10), "empty", 0);
    same(sa.query(q.data(), q.size(), 1 << 20), fm.query(q.data(), q.size(), 1 << 20), "long", 0);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int bench(const char* path, std::size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::uint8_t> t(bytes);
    in.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(bytes));
    t.resize(static_cast<std::size_t>(in.gcount()));
    const std::size_t train = t.size() * 7 / 8;
    double t0 = now();
    const InfiniGram sa(t.data(), train);
    const double bsa = now() - t0;
    t0 = now();
    const FmInfiniGram fm(t.data(), train);
    const double bfm = now() - t0;
    double qa = 0, qb = 0, qc = 0;
    FmInfiniGram::Cursor cur;
    int pa = -1, pb = -1, pc = -1;
    std::size_t diff = 0;
    for (std::size_t i = train; i < t.size(); ++i) {
        const std::size_t len = std::min<std::size_t>(i, 8192);
        const std::uint8_t* c = t.data() + (i - len);
        double a0 = now();
        const auto a = sa.query(c, len, 8192, pa < 0 ? -1 : std::min(pa + 1, 8192));
        double a1 = now();
        const auto b = fm.query(c, len, 8192, pb < 0 ? -1 : std::min(pb + 1, 8192));
        double a2 = now();
        const auto d = fm.query(c, len, 8192, pc < 0 ? -1 : std::min(pc + 1, 8192), 1, cur);
        double a3 = now();
        qa += a1 - a0, qb += a2 - a1, qc += a3 - a2;
        pa = a.n, pb = b.n, pc = d.n;
        diff += !(a.n == b.n && a.count == b.count) + !(a.n == d.n && a.count == d.count);
    }
    const double q = static_cast<double>(t.size() - train);
    std::printf("bench %zu training bytes: build s sa %.2f fm %.2f; index MB sa %.1f (text + suffix array) fm %.1f; "
                "us/query sa %.2f fm %.2f fm+cursor %.2f; differ %zu\n",
                train, bsa, bfm, (train + (train * 25 + 7) / 8) / 1048576.0, fm.bytes() / 1048576.0, qa / q * 1e6,
                qb / q * 1e6, qc / q * 1e6, diff);
    return diff ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--bench") == 0) return bench(argv[2], std::strtoull(argv[3], nullptr, 10));
    std::mt19937_64 g(20261028);
    for (int sigma : {1, 2, 4, 26, 256})
        for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{100}, std::size_t{3000}})
            for (bool rep : {false, true}) one_text(g, n, sigma, rep);
    std::printf("fm_infinigram_test: %zu checks, %zu failed\n", checks, failures);
    return failures ? 1 : 0;
}
