#include "cypha/cyphalm/fm_infinigram.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "libsais.h"

namespace cypha::cyphalm {

// Stored form: plain little-endian integers and arrays, each array after its length.
namespace {
template <class T>
void put(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof v); }
template <class T>
void put_vec(std::ostream& o, const std::vector<T>& v) {
    put(o, static_cast<std::uint64_t>(v.size()));
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
template <class T>
void get(std::istream& in, T& v) {
    if (!in.read(reinterpret_cast<char*>(&v), sizeof v)) throw std::runtime_error("FmInfiniGram: stored index cut short");
}
template <class T>
void get_vec(std::istream& in, std::vector<T>& v, std::uint64_t most) {
    std::uint64_t n = 0;
    get(in, n);
    if (n > most) throw std::runtime_error("FmInfiniGram: stored index damaged");
    v.resize(static_cast<std::size_t>(n));
    if (!in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T))))
        throw std::runtime_error("FmInfiniGram: stored index cut short");
}
}  // namespace

namespace fm_detail {

void RankBits::reset(std::size_t nbits) {
    w.assign((nbits + 63) / 64 + 8, 0);
    cum.assign(w.size() / 8 + 2, 0);
}

void RankBits::finish() {
    std::uint32_t c = 0;
    for (std::size_t b = 0; b * 8 < w.size(); ++b) {
        cum[b] = c;
        for (std::size_t k = 0; k < 8 && b * 8 + k < w.size(); ++k) c += static_cast<std::uint32_t>(std::popcount(w[b * 8 + k]));
    }
    cum[(w.size() + 7) / 8] = c;
}

void AlphaTree::build(std::vector<std::uint8_t> seq) {
    std::fill(std::begin(freq_), std::end(freq_), 0);
    for (auto c : seq) ++freq_[c];
    std::vector<int> syms;
    for (int c = 0; c < 256; ++c)
        if (freq_[c]) syms.push_back(c);
    nodes_.clear();
    if (syms.size() <= 1) {
        root_ = syms.empty() ? -1 : -(syms[0] + 1);
        return;
    }
    nodes_.reserve(syms.size() - 1);
    root_ = make_(syms, 0, syms.size(), seq);
}

int AlphaTree::make_(const std::vector<int>& syms, std::size_t a, std::size_t b, std::vector<std::uint8_t>& seq) {
    if (b - a == 1) return -(syms[a] + 1);
    // Split where the two sides' counts are closest.
    std::uint64_t W = 0, acc = 0, best = ~std::uint64_t{0};
    for (std::size_t i = a; i < b; ++i) W += freq_[syms[i]];
    std::size_t cut = a + 1;
    for (std::size_t i = a + 1; i < b; ++i) {
        acc += freq_[syms[i - 1]];
        const std::uint64_t d = 2 * acc > W ? 2 * acc - W : W - 2 * acc;
        if (d < best) best = d, cut = i;
    }
    const int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    const int mid = syms[cut - 1];
    nodes_[static_cast<std::size_t>(id)].mid = mid;
    std::size_t nl = 0;
    {
        RankBits& rb = nodes_[static_cast<std::size_t>(id)].bits;
        rb.reset(seq.size());
        for (std::size_t i = 0; i < seq.size(); ++i)
            if (seq[i] > mid) rb.w[i >> 6] |= std::uint64_t{1} << (i & 63);
            else ++nl;
        rb.finish();
    }
    std::vector<std::uint8_t> L, R;
    L.reserve(nl);
    R.reserve(seq.size() - nl);
    for (auto c : seq) (c <= mid ? L : R).push_back(c);
    seq.clear();
    seq.shrink_to_fit();
    const int l = make_(syms, a, cut, L);
    const int r = make_(syms, cut, b, R);
    nodes_[static_cast<std::size_t>(id)].left = l;
    nodes_[static_cast<std::size_t>(id)].right = r;
    return id;
}

void AlphaTree::counts_(int v, std::size_t lo, std::size_t hi, std::uint32_t* out) const {
    if (lo >= hi) return;
    if (v < 0) {
        out[-v - 1] += static_cast<std::uint32_t>(hi - lo);
        return;
    }
    const Node& nd = nodes_[static_cast<std::size_t>(v)];
    const std::size_t o1 = nd.bits.rank1(lo), h1 = nd.bits.rank1(hi);
    counts_(nd.left, lo - o1, hi - h1, out);
    counts_(nd.right, o1, h1, out);
}

std::size_t AlphaTree::bytes() const {
    std::size_t b = sizeof(*this);
    for (const auto& nd : nodes_) b += nd.bits.bytes() + sizeof(Node);
    return b;
}


void AlphaTree::write(std::ostream& out) const {
    for (auto f : freq_) put(out, f);
    put(out, static_cast<std::int32_t>(root_));
    put(out, static_cast<std::uint64_t>(nodes_.size()));
    for (const auto& nd : nodes_) {
        put(out, static_cast<std::int32_t>(nd.mid));
        put(out, static_cast<std::int32_t>(nd.left));
        put(out, static_cast<std::int32_t>(nd.right));
        put_vec(out, nd.bits.w);
        put_vec(out, nd.bits.cum);
    }
}

void AlphaTree::read(std::istream& in) {
    for (auto& f : freq_) get(in, f);
    std::int32_t root = 0;
    get(in, root);
    root_ = root;
    std::uint64_t nn = 0;
    get(in, nn);
    if (nn > 255) throw std::runtime_error("FmInfiniGram: stored index damaged");
    nodes_.assign(static_cast<std::size_t>(nn), Node{});
    for (auto& nd : nodes_) {
        std::int32_t mid = 0, l = 0, r = 0;
        get(in, mid), get(in, l), get(in, r);
        nd.mid = mid, nd.left = l, nd.right = r;
        get_vec(in, nd.bits.w, std::uint64_t{1} << 32);
        get_vec(in, nd.bits.cum, std::uint64_t{1} << 32);
        if (nd.bits.cum.size() != nd.bits.w.size() / 8 + 2) throw std::runtime_error("FmInfiniGram: stored index damaged");
    }
    const auto ok = [&](int v) { return v < 0 ? v >= -256 : static_cast<std::size_t>(v) < nodes_.size(); };
    if (!ok(root_)) throw std::runtime_error("FmInfiniGram: stored index damaged");
    for (const auto& nd : nodes_)
        if (!ok(nd.left) || !ok(nd.right) || nd.mid < 0 || nd.mid > 255) throw std::runtime_error("FmInfiniGram: stored index damaged");
}

// BWT of t$ as n + 1 bytes, the $ row holding 0 (libsais gives the n others and
// the $ row). With in_place the text's own buffer becomes the BWT.
static std::vector<std::uint8_t> bwt(std::vector<std::uint8_t>& t, bool in_place, std::size_t& dollar) {
    const std::size_t n = t.size();
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t>& out = in_place ? t : u;
    if (!in_place) u.resize(n + 1);
    std::int32_t p = 0;
    {
        std::vector<std::int32_t> A(n);
        p = libsais_bwt(t.data(), out.data(), A.data(), static_cast<std::int32_t>(n), 0, nullptr);
    }
    if (p < 0) throw std::runtime_error("FmInfiniGram: libsais_bwt failed");
    out.resize(n + 1);
    std::memmove(out.data() + p + 1, out.data() + p, n - static_cast<std::size_t>(p));
    out[static_cast<std::size_t>(p)] = 0;
    dollar = static_cast<std::size_t>(p);
    return in_place ? std::move(t) : u;
}

}  // namespace fm_detail

FmInfiniGram::FmInfiniGram(const std::uint8_t* text, std::size_t n)
    : FmInfiniGram(std::vector<std::uint8_t>(text, text + n)) {}

FmInfiniGram::FmInfiniGram(std::vector<std::uint8_t>&& text) : n_(text.size()) {
    if (n_ >= (std::size_t{1} << 31)) throw std::runtime_error("FmInfiniGram: corpus over 2 GiB");
    std::size_t freq[256] = {};
    for (auto c : text) ++freq[c];
    C_[0] = 1;
    for (int c = 1; c < 256; ++c) C_[c] = C_[c - 1] + freq[c - 1];
    fwd_.build(fm_detail::bwt(text, false, dT_));
    std::reverse(text.begin(), text.end());
    rev_.build(fm_detail::bwt(text, true, dR_));
}

namespace {
constexpr char kMagic[4] = {'F', 'M', 'I', 'G'};
constexpr std::uint32_t kVersion = 1;
// FNV-1a over the stored bytes after the header, checked on load.
struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void add(const char* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) h = (h ^ static_cast<unsigned char>(p[i])) * 1099511628211ull;
    }
};
}  // namespace

void FmInfiniGram::save(const std::string& path) const {
    // Header (magic, version, body size, checksum) with the last two filled in after the body is written.
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("FmInfiniGram: cannot write " + path);
    const std::streamoff head = 4 + sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t);
    out.write(kMagic, 4);
    put(out, kVersion);
    put(out, std::uint64_t{0});
    put(out, std::uint64_t{0});
    put(out, static_cast<std::uint64_t>(n_));
    for (auto c : C_) put(out, static_cast<std::uint64_t>(c));
    put(out, static_cast<std::uint64_t>(dT_));
    put(out, static_cast<std::uint64_t>(dR_));
    fwd_.write(out);
    rev_.write(out);
    const std::uint64_t size = static_cast<std::uint64_t>(static_cast<std::streamoff>(out.tellp()) - head);
    Fnv f;
    out.seekg(head);
    std::vector<char> buf(1 << 20);
    for (std::uint64_t left = size; left;) {
        const std::size_t k = static_cast<std::size_t>(std::min<std::uint64_t>(left, buf.size()));
        if (!out.read(buf.data(), static_cast<std::streamsize>(k))) throw std::runtime_error("FmInfiniGram: cannot reread " + path);
        f.add(buf.data(), k);
        left -= k;
    }
    out.seekp(4 + sizeof(std::uint32_t));
    put(out, size);
    put(out, f.h);
    if (!out) throw std::runtime_error("FmInfiniGram: cannot write " + path);
}

FmInfiniGram FmInfiniGram::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("FmInfiniGram: cannot open " + path);
    char magic[4] = {};
    std::uint32_t version = 0;
    std::uint64_t size = 0, sum = 0;
    in.read(magic, 4);
    get(in, version), get(in, size), get(in, sum);
    if (std::memcmp(magic, kMagic, 4) != 0 || version != kVersion) throw std::runtime_error("FmInfiniGram: not a stored index: " + path);
    // Check the checksum while reading the body in blocks, then parse it from the start.
    {
        Fnv f;
        std::vector<char> buf(1 << 20);
        std::uint64_t left = size;
        while (left) {
            const std::size_t k = static_cast<std::size_t>(std::min<std::uint64_t>(left, buf.size()));
            if (!in.read(buf.data(), static_cast<std::streamsize>(k))) throw std::runtime_error("FmInfiniGram: stored index cut short");
            f.add(buf.data(), k);
            left -= k;
        }
        if (f.h != sum) throw std::runtime_error("FmInfiniGram: stored index damaged (checksum)");
    }
    in.clear();
    in.seekg(4 + sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t));
    FmInfiniGram g;
    std::uint64_t v = 0;
    get(in, v);
    g.n_ = static_cast<std::size_t>(v);
    for (auto& c : g.C_) get(in, v), c = static_cast<std::size_t>(v);
    get(in, v);
    g.dT_ = static_cast<std::size_t>(v);
    get(in, v);
    g.dR_ = static_cast<std::size_t>(v);
    g.fwd_.read(in);
    g.rev_.read(in);
    return g;
}

std::size_t FmInfiniGram::cap_(std::size_t len, int max_n, int hint) {
    std::size_t cap = std::min<std::size_t>(len, static_cast<std::size_t>(std::max(0, max_n)));
    if (hint >= 0) cap = std::min<std::size_t>(cap, static_cast<std::size_t>(hint));
    return cap;
}

FmInfiniGram::Found FmInfiniGram::search_(const std::uint8_t* ctx, std::size_t len, std::size_t cap,
                                          std::size_t need) const {
    // The longest suffix of ctx (at most cap bytes) followed by a byte at least
    // need times. That count never grows with the length, so stop at the first miss.
    Found f;
    f.sz = n_ + 1;
    std::size_t lo = 0, hi = n_ + 1, loR = 0;
    for (std::size_t k = 1; k <= cap; ++k) {
        const int c = ctx[len - k];
        std::size_t rlo = 0, rhi = 0, less = 0;
        fwd_.step(c, lo, hi, rlo, rhi, less);
        const bool dol = dT_ >= lo && dT_ < hi;  // the occurrence at the text's start
        if (c == 0) rlo -= (dT_ < lo), rhi -= (dT_ < hi);  // the $ row stores 0 ...
        else if (dol) --less;                              // ... and counted below c
        const std::size_t nlo = C_[c] + rlo, sz = rhi - rlo;
        const std::size_t nloR = loR + (dol ? 1 : 0) + less;
        const std::size_t total = sz - ((dR_ >= nloR && dR_ < nloR + sz) ? 1 : 0);
        if (sz == 0 || total < need) break;
        lo = nlo, hi = nlo + sz, loR = nloR;
        f.n = static_cast<int>(k), f.loR = loR, f.sz = sz;
    }
    return f;
}

InfiniGram::Result FmInfiniGram::counts_(int n, std::size_t loR, std::size_t sz) const {
    InfiniGram::Result r;
    r.n = n;
    rev_.counts(loR, loR + sz, r.count.data());
    if (dR_ >= loR && dR_ < loR + sz) --r.count[0];  // the occurrence at the text's end: no next byte
    std::uint64_t t = 0;
    for (auto v : r.count) t += v;
    r.total = t;
    return r;
}

std::size_t FmInfiniGram::match_prefix(const std::uint8_t* s, std::size_t len, std::size_t cap,
                                      std::size_t& count) const {
    const std::size_t m = std::min(len, cap);
    std::size_t loR = 0, sz = n_ + 1, good = 0;
    for (std::size_t k = 0; k < m; ++k) {
        const int b = s[k];
        std::size_t rlo = 0, rhi = 0, less = 0;
        rev_.step(b, loR, loR + sz, rlo, rhi, less);
        if (b == 0) rlo -= (dR_ < loR), rhi -= (dR_ < loR + sz);
        if (rhi == rlo) break;
        loR = C_[b] + rlo, sz = rhi - rlo, good = k + 1;
    }
    count = good ? sz : n_;
    return good;
}

InfiniGram::Result FmInfiniGram::query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint,
                                       std::uint64_t min_total) const {
    const std::size_t need = static_cast<std::size_t>(std::max<std::uint64_t>(1, min_total));
    const Found f = search_(ctx, len, cap_(len, max_n, hint), need);
    return counts_(f.n, f.loR, f.sz);
}

InfiniGram::Result FmInfiniGram::query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint,
                                       std::uint64_t min_total, Cursor& cur) const {
    const std::size_t cap = cap_(len, max_n, hint);
    const std::size_t need = static_cast<std::size_t>(std::max<std::uint64_t>(1, min_total));
    const std::size_t m = cur.n < 0 ? 0 : static_cast<std::size_t>(cur.n);
    // The last query found m bytes, the longest under its bound cur.cap. If a suffix of
    // L + 1 bytes is usable now, its first L bytes are usable and end where the last
    // context ended. When the cap - 1 bytes before the new byte are the last context's
    // own last bytes, and cap <= cur.cap + 1, those L bytes were a candidate last time,
    // so L <= m: nothing is longer than m + 1. Then if (match, b) is usable it is the
    // answer, and extending a match to the right is one step on the reversed BWT.
    if (cur.n >= 0 && need == cur.need && cap >= m + 1 && cap <= cur.cap + 1 &&
        std::memcmp(ctx + len - cap, cur.tail.data() + (cur.tail.size() - (cap - 1)), cap - 1) == 0) {
        const int b = ctx[len - 1];
        std::size_t rlo = 0, rhi = 0, less = 0;
        rev_.step(b, cur.loR, cur.loR + cur.sz, rlo, rhi, less);
        if (b == 0) rlo -= (dR_ < cur.loR), rhi -= (dR_ < cur.loR + cur.sz);
        const std::size_t nloR = C_[b] + rlo, sz = rhi - rlo;
        const std::size_t total = sz - ((dR_ >= nloR && dR_ < nloR + sz) ? 1 : 0);
        if (sz > 0 && total >= need) {
            cur.n = static_cast<int>(m + 1), cur.loR = nloR, cur.sz = sz, cur.cap = cap;
            cur.tail.assign(ctx + len - cap, ctx + len);
            return counts_(cur.n, nloR, sz);
        }
    }
    const Found f = search_(ctx, len, cap, need);
    cur.n = f.n, cur.loR = f.loR, cur.sz = f.sz, cur.need = need, cur.cap = cap;
    cur.tail.assign(ctx + len - cap, ctx + len);
    return counts_(f.n, f.loR, f.sz);
}

}  // namespace cypha::cyphalm
