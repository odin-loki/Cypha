// Known-answer test for cypha/cyphalm/fm_infinigram.hpp (RAM_PROFILE M5): the
// FM-index must give InfiniGram's answer (n, total, all 256 counts) on every
// query. Random texts over small and full alphabets, byte 0 included; queries
// with and without hints, min_total above 1, and cursors used right, reset and
// misused. S1: locate, rows and longest_suffix against a brute-force search on
// the same texts (their own random stream, so the query checks stay the same),
// and stored files of version 2 (with locate) and version 1. Prints the count
// of checks; exits 1 on any difference.
//
// fm_infinigram_test [--bench FILE BYTES]   (--bench: time both indexes on a file's prefix)

#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/cyphalm/infinigram.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using cypha::cyphalm::FmInfiniGram;
using cypha::cyphalm::InfiniGram;

namespace {

std::size_t checks = 0, failures = 0, locate_checks = 0;

void expect(bool ok, const char* what, std::size_t at) {
    ++checks, ++locate_checks;
    if (!ok && failures++ < 10) std::printf("FAIL %s at %zu\n", what, at);
}

// Every position where p[0..len) starts in t (brute force).
std::vector<std::size_t> occurrences(const std::vector<std::uint8_t>& t, const std::uint8_t* p, std::size_t len) {
    std::vector<std::size_t> v;
    for (std::size_t i = 0; len && i + len <= t.size(); ++i)
        if (std::equal(p, p + len, t.begin() + static_cast<std::ptrdiff_t>(i))) v.push_back(i);
    return v;
}

// A version 1 file of the same index: version 2 without locate is version 1's body plus a 4-byte zero rate.
void write_v1(const std::string& v2_path, const std::string& v1_path) {
    std::ifstream in(v2_path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t head = 4 + 4 + 8 + 8;
    b.resize(b.size() - 4);
    std::uint64_t h = 1469598103934665603ull, size = b.size() - head;
    for (std::size_t i = head; i < b.size(); ++i) h = (h ^ static_cast<unsigned char>(b[i])) * 1099511628211ull;
    const std::uint32_t v = 1;
    std::memcpy(b.data() + 4, &v, 4);
    std::memcpy(b.data() + 8, &size, 8);
    std::memcpy(b.data() + 16, &h, 8);
    std::ofstream(v1_path, std::ios::binary).write(b.data(), static_cast<std::streamsize>(b.size()));
}

// S1: locate gives the suffix array (a permutation of the positions, suffixes in order); rows and locate find exactly
// the brute-force occurrences; longest_suffix finds the longest occurring suffix; stored files keep locate.
void locate_text(const std::vector<std::uint8_t>& t, const std::vector<std::uint8_t>& q, const FmInfiniGram& fm,
                 std::uint64_t seed) {
    std::mt19937_64 g(seed);
    const std::size_t n = t.size();
    for (std::uint32_t rate : {1u, 3u, 32u}) {
        FmInfiniGram f = fm;
        f.build_locate(rate);
        std::vector<std::size_t> sa(n + 1, 0);
        std::vector<bool> seen(n, false);
        bool perm = true, order = true;
        for (std::size_t r = 1; r <= n; ++r) {
            sa[r] = f.locate(r);
            if (sa[r] >= n || seen[sa[r]]) perm = false;
            else seen[sa[r]] = true;
            if (perm && r > 1)  // suffix sa[r-1] < suffix sa[r], a shorter prefix first
                order = order && std::lexicographical_compare(t.begin() + static_cast<std::ptrdiff_t>(sa[r - 1]), t.end(),
                                                              t.begin() + static_cast<std::ptrdiff_t>(sa[r]), t.end());
        }
        expect(perm && order, "locate gives the suffix array", rate);
        for (int k = 0; k < 60; ++k) {
            const auto& s = g() % 2 ? t : q;
            const std::size_t at = g() % s.size(), len = 1 + g() % std::min<std::size_t>(12, s.size() - at);
            const auto want = occurrences(t, s.data() + at, len);
            const auto rr = f.rows(s.data() + at, len);
            std::vector<std::size_t> got;
            for (std::size_t r = rr.lo; r < rr.hi; ++r) got.push_back(f.locate(r));
            std::sort(got.begin(), got.end());
            expect(got == want, "rows + locate = brute force", at);
        }
        if (rate == 32) {  // a stored index keeps locate; a version 1 file loads without it
            const std::string p2 = "fm_infinigram_test_v2.fmig", p1 = "fm_infinigram_test_v1.fmig";
            f.save(p2);
            const FmInfiniGram back = FmInfiniGram::load(p2);
            bool same_sa = back.can_locate() && back.locate_bytes() == f.locate_bytes();
            for (std::size_t r = 1; same_sa && r <= n; ++r) same_sa = back.locate(r) == sa[r];
            expect(same_sa, "stored locate", n);
            fm.save(p2);
            write_v1(p2, p1);
            const FmInfiniGram v1 = FmInfiniGram::load(p1);
            bool same_q = !v1.can_locate() && v1.bytes() == fm.bytes();
            for (std::size_t i = 1; same_q && i <= q.size(); i += 7) {
                const std::size_t len = std::min<std::size_t>(i, 40);
                const auto x = fm.query(q.data() + (i - len), len, 40), y = v1.query(q.data() + (i - len), len, 40);
                same_q = x.n == y.n && x.count == y.count;
            }
            expect(same_q, "version 1 file", n);
            std::remove(p1.c_str());
            std::remove(p2.c_str());
        }
    }
    // longest_suffix: the longest suffix of a context (at most cap bytes) that occurs in t, and its count
    for (int k = 0; k < 80; ++k) {
        const auto& s = g() % 2 ? t : q;
        const std::size_t end = 1 + g() % s.size(), len = std::min<std::size_t>(end, 1 + g() % 40);
        const std::size_t cap = g() % 4 == 0 ? g() % 10 : len;
        const std::uint8_t* ctx = s.data() + (end - len);
        std::size_t want = 0;
        for (std::size_t m = std::min(len, cap); m > 0 && !want; --m)
            if (!occurrences(t, ctx + len - m, m).empty()) want = m;
        std::size_t got = 0;
        const auto rr = fm.longest_suffix(ctx, len, cap, got);
        const std::size_t cnt = want ? occurrences(t, ctx + len - want, want).size() : 0;
        expect(got == want && rr.hi - rr.lo == cnt, "longest_suffix", end);
    }
}

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
    // 4) a stored index gives the same answers (every tenth text)
    static int stored = 0;
    if (stored++ % 10 == 0) {
        const std::string path = "fm_infinigram_test.fmig";
        fm.save(path);
        const FmInfiniGram back = FmInfiniGram::load(path);
        std::remove(path.c_str());
        ++checks;
        if (back.bytes() != fm.bytes() || back.size() != fm.size()) {
            if (failures++ < 10) std::printf("FAIL stored index: bytes %zu/%zu\n", back.bytes(), fm.bytes());
        }
        int pb = -1, pc = -1;
        for (std::size_t i = 1; i <= q.size(); ++i) {
            const std::size_t len = std::min<std::size_t>(i, 64);
            const auto x = fm.query(q.data() + (i - len), len, 64, pb < 0 ? -1 : std::min(pb + 1, 64));
            const auto y = back.query(q.data() + (i - len), len, 64, pc < 0 ? -1 : std::min(pc + 1, 64));
            same(x, y, "stored", i);
            pb = x.n, pc = y.n;
        }
    }
    // 5) the empty context and a context longer than the text
    same(sa.query(t.data(), 0, 10), fm.query(t.data(), 0, 10), "empty", 0);
    same(sa.query(q.data(), q.size(), 1 << 20), fm.query(q.data(), q.size(), 1 << 20), "long", 0);
    // 6) S1: locate, on its own random stream
    locate_text(t, q, fm, n * 1000003u + static_cast<std::uint64_t>(sigma) * 101u + (repeats ? 7u : 0u));
}

// M16: the stored index of fixed texts, hashed (FNV-1a 64 of the whole file). The values were written by the build
// before M16 (both BWTs from copies, trees from per-node copies), so the build's every path must give the same bytes.
struct GoldenText {
    std::size_t n;
    int sigma;
    bool zeros;  // about half the bytes 0 (the $ row also holds 0)
};
const GoldenText kGolden[] = {{1, 1, false},    {2, 2, true},       {3, 256, false},     {64, 2, false},
                              {1000, 26, false}, {1000, 256, true},  {4096, 4, false},    {65537, 26, true},
                              {65537, 256, false}, {300000, 26, false}, {300000, 256, true}, {300000, 3, false}};
std::vector<std::uint8_t> golden_text(const GoldenText& gt, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::vector<std::uint8_t> t(gt.n);
    for (auto& c : t) {
        // Skewed over the alphabet (a common byte, then rarer ones), as text is.
        const std::uint64_t r = g();
        int s = static_cast<int>((r % 1000) * (r % 1000) / 1000000.0 * gt.sigma);
        if (s >= gt.sigma) s = gt.sigma - 1;
        c = static_cast<std::uint8_t>(gt.zeros && (r >> 40) % 2 == 0 ? 0 : (gt.sigma == 256 ? s : 'a' + s));
    }
    return t;
}
std::uint64_t file_hash(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::uint64_t h = 1469598103934665603ull;
    for (std::istreambuf_iterator<char> it(in), end; it != end; ++it) h = (h ^ static_cast<unsigned char>(*it)) * 1099511628211ull;
    return h;
}
const std::uint64_t kGoldenHash[std::size(kGolden)] = {  // --golden on the build before M16 (wip/integ2 95b5883)
    0xc34d6c70ab37eb3eull, 0x028b29520a4ce81aull, 0xcb629568b366ab5eull, 0xa7e674806d853eeaull,
    0xcf6a57e93b576d74ull, 0x73b69cdf7e11b997ull, 0x195a3db83f72551cull, 0xdc6188804005d67eull,
    0xbfbdb39297955f30ull, 0xe2dc266da68403aaull, 0x8ea33b3301ce9429ull, 0x4d1b693bcae09a32ull};
int golden(bool print) {
    const std::string path = "fm_infinigram_golden.fmig";
    for (std::size_t i = 0; i < std::size(kGolden); ++i) {
        std::vector<std::uint8_t> t = golden_text(kGolden[i], 1000 + i);
        const FmInfiniGram a(t.data(), t.size());  // the copying constructor
        a.save(path);
        const std::uint64_t ha = file_hash(path);
        std::vector<std::uint8_t> moved = t;
        const FmInfiniGram b(std::move(moved));    // the consuming one (qa's)
        b.save(path);
        const std::uint64_t hb = file_hash(path);
        if (print) std::printf("golden %zu n %zu sigma %d zeros %d hash %016llx %016llx\n", i, kGolden[i].n, kGolden[i].sigma,
                               kGolden[i].zeros ? 1 : 0, static_cast<unsigned long long>(ha), static_cast<unsigned long long>(hb));
        else {
            checks += 2;  // M16: the index gives its text back
            if ((a.text() != t || b.text() != t) && failures++ < 10) std::printf("FAIL golden %zu: text() differs\n", i);
            if ((ha != kGoldenHash[i] || hb != kGoldenHash[i]) && failures++ < 10)
                std::printf("FAIL golden %zu: hash %016llx %016llx, want %016llx\n", i, static_cast<unsigned long long>(ha),
                            static_cast<unsigned long long>(hb), static_cast<unsigned long long>(kGoldenHash[i]));
        }
    }
    std::remove(path.c_str());
    return 0;
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
    if (argc == 2 && std::strcmp(argv[1], "--golden") == 0) return golden(true);
    std::mt19937_64 g(20261028);
    for (int sigma : {1, 2, 4, 26, 256})
        for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{100}, std::size_t{3000}})
            for (bool rep : {false, true}) one_text(g, n, sigma, rep);
    golden(false);  // M16
    std::printf("fm_infinigram_test: %zu checks (queries and stored files %zu, locate %zu), %zu failed\n", checks,
                checks - locate_checks, locate_checks, failures);
    return failures ? 1 : 0;
}
