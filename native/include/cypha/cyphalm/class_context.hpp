#pragma once

/// Class-context expert for CyphaLM (PLAN B4; TOOL_PROFILE_PLAN P1, the one adapter shared by the
/// part-of-speech tagger and the word classes). It predicts the next byte from the *classes* of the
/// last two completed words plus the bytes typed since the last one ended (at most ``tail``, default 6),
/// so a context learned after "Who wrote" also serves "Who composed" when the two verbs share a class.
///
/// - A word is a run of ASCII letters, digits or apostrophes. Its class comes from a word -> class
///   table (E1 tags, or k-means word classes), looked up in lower case. A word not in the table gets
///   one of three shape classes (lower case, capitalised, number).
/// - Three orders by default, (c1, tail), (c2, c1, tail) and (c3, c2, c1, tail), or two, interpolated
///   Witten-Bell style over a uniform floor.
///   The caller mixes the result into CyphaLM's distribution with one lambda per bucket(), as
///   cyphalm_trie mixes InfiniGram, so the trie stays the main expert.
/// - Counts are built once over the training bytes, so this expert is batch-built, not online. The
///   build sorts the (context, byte) events chunk by chunk and merges the counted runs, each order on its
///   own, so its peak is about one chunk of events, the distinct (context, byte) pairs, and one order's
///   pairs again, not every event at once.
/// - Storage (docs/reports/RAM_PROFILE.md): by default 16 bytes per distinct context (56-bit hash key,
///   first follower, total) and 5 per distinct follower (byte, 32-bit count). ``compact`` keeps a 32-bit
///   key, 1-byte counts on the trie's quantising ladder and no stored total: 8 + 2 bytes. ``min_total``
///   drops contexts seen fewer times than that.
/// - save() / load() store a built table in a checked file (RAM_PROFILE M10), so serving never builds it.
/// - Hashes are 56 bits; with ~10M contexts a collision is about a 1-in-1,000 event per table. Compact
///   keys are 32 bits: contexts that share one have their counts merged (a few hundred of 2M).

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm {

class ClassContext {
 public:
    static constexpr int kMaxTail = 7;   ///< most bytes since the last completed word that can be kept
    static constexpr int kScan = 256;    ///< how far back a context is looked for
    static constexpr int kBuckets = 16;  ///< bucket(): deepest order found (0..3) x count bucket (0..3)

    struct Hit {
        int order = 0;            ///< 0 nothing seen, 1 (c1, tail), 2 (c2, c1, tail), 3 (c3, c2, c1, tail)
        std::uint64_t total = 0;  ///< followers counted at the deepest order found
    };

    /// The defaults are the smaller table Odin chose on 2026-10-28 (RAM_PROFILE M2/M3, B4B): three word orders,
    /// compact storage, contexts seen once dropped. Fresh text 2.0853 BPC against 2.0879 for the earlier
    /// two-order full table, in about a third of its memory. The earlier table is {6, 2, false, 1}.
    struct Options {
        int tail = 6;                ///< bytes since the last completed word kept in the context (1..kMaxTail)
        int orders = 3;              ///< 2: the last two words; 3: also the third
        bool compact = true;         ///< 32-bit keys, 1-byte quantised counts, totals summed at lookup
        std::uint32_t min_total = 2; ///< contexts seen fewer times are dropped
        std::size_t chunk = std::size_t{1} << 17;  ///< positions per build chunk (RAM_PROFILE: 2^17 gave the lowest peak)
    };

    /// Read a ``word<TAB>class`` table. Class names are arbitrary strings; ids are assigned in order
    /// of first appearance. Words are lower-cased. Returns an empty map and sets ``err`` on failure.
    static std::unordered_map<std::string, std::uint16_t> load_tsv(const std::string& path, std::string& err) {
        std::unordered_map<std::string, std::uint16_t> out;
        std::unordered_map<std::string, std::uint16_t> names;
        std::ifstream in(path);
        if (!in) {
            err = "cannot open " + path;
            return out;
        }
        std::string line;
        while (std::getline(in, line)) {
            const std::size_t t = line.find('\t');
            if (t == std::string::npos || t == 0) continue;
            std::string w = lower(line.substr(0, t));
            std::string c = line.substr(t + 1);
            while (!c.empty() && (c.back() == '\r' || c.back() == '\n')) c.pop_back();
            auto it = names.find(c);
            if (it == names.end()) {
                if (names.size() >= 65535 - kShapes) {
                    err = "too many classes in " + path;
                    return {};
                }
                it = names.emplace(c, static_cast<std::uint16_t>(kShapes + names.size())).first;
            }
            out.emplace(std::move(w), it->second);
        }
        if (out.empty()) err = "no rows in " + path;
        return out;
    }

    ClassContext(std::unordered_map<std::string, std::uint16_t> classes, const Options& o)
        : cls_(std::move(classes)), opt_(o) {
        opt_.tail = std::clamp(opt_.tail, 1, kMaxTail);
        opt_.orders = std::clamp(opt_.orders, 2, 3);
        opt_.chunk = std::max<std::size_t>(opt_.chunk, 1);
    }

    /// Count every (context, next byte) event in text[0..n).
    void build(const std::uint8_t* text, std::size_t n) {
        // Counted runs per order, sorted by (key << 8 | byte), merged chunk by chunk. Each order is
        // merged on its own, so only one order's runs are ever held twice.
        const int no = opt_.orders;
        std::vector<std::uint64_t> kb[3], ev[3], kb2;
        std::vector<std::uint32_t> cnt[3], cnt2;
        for (std::size_t a = 1; a < n; a += opt_.chunk) {
            const std::size_t b = std::min(n, a + opt_.chunk);
            for (int o = 0; o < no; ++o) ev[o].clear(), ev[o].reserve(b - a);
            for (std::size_t i = a; i < b; ++i) {
                std::uint64_t k[3];
                keys(text, i, k);
                for (int o = 0; o < no; ++o) ev[o].push_back(stored_key(k[o]) << 8 | text[i]);
            }
            for (int o = 0; o < no; ++o) {
                std::sort(ev[o].begin(), ev[o].end());
                std::vector<std::uint64_t>& K = kb[o];
                std::vector<std::uint32_t>& C = cnt[o];
                const std::vector<std::uint64_t>& E = ev[o];
                kb2.clear();
                cnt2.clear();
                kb2.reserve(K.size() + E.size());  // an upper bound: the merge never regrows
                cnt2.reserve(K.size() + E.size());
                std::size_t x = 0, y = 0;
                while (x < K.size() || y < E.size()) {  // merge the old runs with this chunk's sorted events
                    std::uint64_t key;
                    std::uint32_t c = 0;
                    if (y >= E.size() || (x < K.size() && K[x] <= E[y])) {
                        key = K[x];
                        c = C[x++];
                    } else {
                        key = E[y];
                    }
                    while (y < E.size() && E[y] == key) ++c, ++y;
                    if (!kb2.empty() && kb2.back() == key) cnt2.back() += c;
                    else kb2.push_back(key), cnt2.push_back(c);
                }
                K.swap(kb2);
                C.swap(cnt2);
            }
        }
        for (auto& e : ev) std::vector<std::uint64_t>().swap(e);
        std::vector<std::uint64_t>().swap(kb2);
        std::vector<std::uint32_t>().swap(cnt2);
        // The table holds every order's contexts in one sorted array: merge the orders' runs (two
        // runs with the same key and byte, a hash collision across orders, are added together).
        auto each = [&](auto&& visit) {
            std::size_t at[3] = {0, 0, 0};
            for (;;) {
                std::uint64_t key = ~std::uint64_t{0};
                bool any = false;
                for (int o = 0; o < no; ++o)
                    if (at[o] < kb[o].size() && (!any || kb[o][at[o]] < key)) key = kb[o][at[o]], any = true;
                if (!any) return;
                std::uint64_t c = 0;
                for (int o = 0; o < no; ++o)
                    if (at[o] < kb[o].size() && kb[o][at[o]] == key) c += cnt[o][at[o]++];
                visit(key, static_cast<std::uint32_t>(c));
            }
        };
        // Pass 1: the table's exact size, so its vectors are allocated once. Pass 2: fill it.
        std::size_t nctx = 0, nfol = 0;
        {
            std::uint64_t cur = ~std::uint64_t{0};
            std::uint64_t tot = 0, fol = 0;
            auto close = [&] {
                if (fol && tot >= opt_.min_total) ++nctx, nfol += fol;
            };
            each([&](std::uint64_t k, std::uint32_t c) {
                if (k >> 8 != cur) close(), cur = k >> 8, tot = 0, fol = 0;
                tot += c, ++fol;
            });
            close();
        }
        key_.clear();
        key32_.clear();
        first_.clear();
        total_.clear();
        byte_.clear();
        count_.clear();
        q_.clear();
        if (opt_.compact) key32_.reserve(nctx), q_.reserve(nfol);
        else key_.reserve(nctx), total_.reserve(nctx), count_.reserve(nfol);
        first_.reserve(nctx + 1);
        byte_.reserve(nfol);
        {
            std::vector<std::pair<std::uint8_t, std::uint32_t>> run;  // one context's followers
            std::uint64_t cur = ~std::uint64_t{0}, tot = 0;
            auto close = [&] {
                if (run.empty() || tot < opt_.min_total) return;
                if (opt_.compact) key32_.push_back(static_cast<std::uint32_t>(cur));
                else key_.push_back(cur), total_.push_back(static_cast<std::uint32_t>(tot));
                first_.push_back(static_cast<std::uint32_t>(byte_.size()));
                for (const auto& f : run) {
                    byte_.push_back(f.first);
                    if (opt_.compact) q_.push_back(quantise(f.second));
                    else count_.push_back(f.second);
                }
            };
            each([&](std::uint64_t k, std::uint32_t c) {
                if (k >> 8 != cur) close(), cur = k >> 8, tot = 0, run.clear();
                tot += c;
                run.emplace_back(static_cast<std::uint8_t>(k & 0xff), c);
            });
            close();
        }
        first_.push_back(static_cast<std::uint32_t>(byte_.size()));
    }

    /// The next-byte distribution after h[0..n) (oldest byte first). P sums to 1.
    Hit predict(const std::uint8_t* h, std::size_t n, double P[256]) const {
        for (int s = 0; s < 256; ++s) P[s] = 1.0 / 256;
        Hit hit;
        std::uint64_t k[3];
        keys(h, n, k);
        for (int o = 0; o < opt_.orders; ++o) {  // lower order first: (c1, tail), (c2, c1, tail), (c3, c2, c1, tail)
            const std::size_t c = find(stored_key(k[o]));
            if (c == kNone) break;
            double N = 0;
            if (opt_.compact)
                for (std::uint32_t f = first_[c]; f < first_[c + 1]; ++f) N += qdec()[q_[f]];
            else
                N = total_[c];
            const double T = first_[c + 1] - first_[c];
            for (int s = 0; s < 256; ++s) P[s] *= T / (N + T);
            for (std::uint32_t f = first_[c]; f < first_[c + 1]; ++f)
                P[byte_[f]] += (opt_.compact ? qdec()[q_[f]] : count_[f]) / (N + T);
            hit.order = o + 1;
            hit.total = static_cast<std::uint64_t>(N);
        }
        return hit;
    }

    static int bucket(const Hit& h) {
        const int cb = h.total <= 1 ? 0 : h.total <= 3 ? 1 : h.total <= 15 ? 2 : 3;
        return h.order * 4 + cb;
    }

    std::size_t classes() const { return cls_.size(); }
    std::size_t contexts() const { return first_.empty() ? 0 : first_.size() - 1; }
    std::size_t followers() const { return byte_.size(); }
    std::size_t bytes() const {
        return opt_.compact ? contexts() * 8 + followers() * 2 : contexts() * 16 + followers() * 5;
    }

    /// A stored table (docs/reports/RAM_PROFILE.md, M10), so a served model never builds it. Little-endian, no
    /// padding: "CYPHACC" and a zero byte, u32 version 1, u32 tail, orders, compact and min_total, the word-class
    /// map (u64 count, then each word as a u32 length, its bytes and a u16 class, sorted by word), then key_,
    /// key32_, first_, total_, byte_, count_ and q_, each a u64 count and its elements, and last the FNV-1a 64-bit
    /// hash of every byte before it. A file that is not one, another version, cut short, damaged or inconsistent
    /// throws std::runtime_error. The build's chunk size is not stored.
    void save(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        Out o{os};
        o.raw(kMagic, sizeof kMagic);
        for (const std::uint32_t v : {kVersion, static_cast<std::uint32_t>(opt_.tail), static_cast<std::uint32_t>(opt_.orders),
                                      static_cast<std::uint32_t>(opt_.compact), opt_.min_total})
            o.num(v);
        std::vector<std::pair<std::string, std::uint16_t>> m(cls_.begin(), cls_.end());
        std::sort(m.begin(), m.end());
        o.num(std::uint64_t{m.size()});
        for (const auto& [w, c] : m) {
            o.num(static_cast<std::uint32_t>(w.size()));
            o.raw(w.data(), w.size());
            o.num(c);
        }
        o.vec(key_), o.vec(key32_), o.vec(first_), o.vec(total_), o.vec(byte_), o.vec(count_), o.vec(q_);
        o.num(o.h);
        os.close();
        if (!os) throw std::runtime_error("ClassContext: cannot write " + path);
    }

    static ClassContext load(const std::string& path) {
        std::ifstream is(path, std::ios::binary | std::ios::ate);
        if (!is) throw std::runtime_error("ClassContext: cannot open " + path);
        In in{is, static_cast<std::uint64_t>(is.tellg())};
        is.seekg(0);
        char magic[sizeof kMagic];
        in.raw(magic, sizeof magic);
        if (std::memcmp(magic, kMagic, sizeof magic) != 0 || in.num<std::uint32_t>() != kVersion)
            throw std::runtime_error("ClassContext: " + path + " is not a stored class table (or another version)");
        Options o;
        o.tail = static_cast<int>(in.num<std::uint32_t>());
        o.orders = static_cast<int>(in.num<std::uint32_t>());
        const std::uint32_t compact = in.num<std::uint32_t>();
        o.compact = compact != 0;
        o.min_total = in.num<std::uint32_t>();
        std::unordered_map<std::string, std::uint16_t> m;
        const auto nm = in.num<std::uint64_t>();
        if (nm > in.left / 6) throw std::runtime_error("ClassContext: stored file damaged");
        for (std::uint64_t i = 0; i < nm; ++i) {
            const auto len = in.num<std::uint32_t>();
            if (len > in.left) throw std::runtime_error("ClassContext: stored file damaged");
            std::string w(len, '\0');
            in.raw(w.data(), w.size());
            m.emplace(std::move(w), in.num<std::uint16_t>());
        }
        ClassContext c(std::move(m), o);
        in.vec(c.key_), in.vec(c.key32_), in.vec(c.first_), in.vec(c.total_), in.vec(c.byte_), in.vec(c.count_), in.vec(c.q_);
        const std::uint64_t h = in.h;
        if (in.num<std::uint64_t>() != h) throw std::runtime_error("ClassContext: stored file damaged (checksum)");
        if (in.left != 0 || compact > 1 || c.opt_.tail != o.tail || c.opt_.orders != o.orders || c.cls_.size() != nm ||
            !c.consistent())
            throw std::runtime_error("ClassContext: stored file damaged");
        return c;
    }

    /// The same options (the chunk aside), word classes and table.
    bool operator==(const ClassContext& b) const {
        return opt_.tail == b.opt_.tail && opt_.orders == b.opt_.orders && opt_.compact == b.opt_.compact &&
               opt_.min_total == b.opt_.min_total && cls_ == b.cls_ && key_ == b.key_ && key32_ == b.key32_ &&
               first_ == b.first_ && total_ == b.total_ && byte_ == b.byte_ && count_ == b.count_ && q_ == b.q_;
    }

 private:
    static constexpr char kMagic[8] = {'C', 'Y', 'P', 'H', 'A', 'C', 'C', '\0'};
    static constexpr std::uint32_t kVersion = 1;

    // The stored file's writer and reader: little-endian numbers, FNV-1a 64 over every byte.
    struct Out {
        std::ostream& os;
        std::uint64_t h = 14695981039346656037ull;
        void raw(const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
            os.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        }
        template <class T>
        void num(T v) {
            unsigned char b[sizeof(T)];
            for (std::size_t i = 0; i < sizeof(T); ++i) b[i] = static_cast<unsigned char>(std::uint64_t{v} >> (8 * i));
            raw(b, sizeof b);
        }
        template <class T>
        void vec(const std::vector<T>& v) {
            num(std::uint64_t{v.size()});
            if constexpr (std::endian::native == std::endian::little) raw(v.data(), v.size() * sizeof(T));
            else
                for (const T x : v) num(x);
        }
    };
    struct In {
        std::istream& is;
        std::uint64_t left;  // bytes the file still holds
        std::uint64_t h = 14695981039346656037ull;
        void raw(void* p, std::size_t n) {
            if (n > left || !is.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
                throw std::runtime_error("ClassContext: stored file cut short");
            left -= n;
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
        }
        template <class T>
        T num() {
            unsigned char b[sizeof(T)];
            raw(b, sizeof b);
            std::uint64_t v = 0;
            for (std::size_t i = 0; i < sizeof(T); ++i) v |= std::uint64_t{b[i]} << (8 * i);
            return static_cast<T>(v);
        }
        template <class T>
        void vec(std::vector<T>& v) {
            const auto n = num<std::uint64_t>();
            if (n > left / sizeof(T)) throw std::runtime_error("ClassContext: stored file damaged");
            v.resize(static_cast<std::size_t>(n));
            if constexpr (std::endian::native == std::endian::little) raw(v.data(), v.size() * sizeof(T));
            else
                for (T& x : v) x = num<T>();
        }
    };

    // The arrays fit together, so every lookup stays inside them: one storage's arrays only, the keys strictly rising,
    // and each context's followers a non-empty run that ends where the next begins.
    bool consistent() const {
        const std::size_t nk = opt_.compact ? key32_.size() : key_.size();
        if (opt_.compact ? !(key_.empty() && total_.empty() && count_.empty() && q_.size() == byte_.size())
                         : !(key32_.empty() && q_.empty() && total_.size() == nk && count_.size() == byte_.size()))
            return false;
        if (first_.empty()) return nk == 0 && byte_.empty();
        if (first_.size() != nk + 1 || first_.front() != 0 || first_.back() != byte_.size()) return false;
        for (std::size_t c = 0; c < nk; ++c)
            if (first_[c + 1] <= first_[c] ||
                (c > 0 && (opt_.compact ? key32_[c] <= key32_[c - 1] : key_[c] <= key_[c - 1])))
                return false;
        return true;
    }

    static constexpr std::uint16_t kShapes = 4;  // 0 none, 1 lower, 2 capitalised, 3 number

    static constexpr std::size_t kNone = ~std::size_t{0};

    // The trie's count ladder (packed_trie.hpp): exact to 64, then one byte on a 1.08 geometric ladder.
    static const std::array<std::uint32_t, 256>& qdec() {
        static const std::array<std::uint32_t, 256> q = [] {
            std::array<std::uint32_t, 256> t{};
            for (int s = 1; s <= 64; ++s) t[static_cast<std::size_t>(s)] = static_cast<std::uint32_t>(s);
            double v = 64.0;
            for (int s = 65; s < 256; ++s) {
                v *= 1.08;
                auto c = static_cast<std::uint32_t>(std::llround(v));
                if (c <= t[static_cast<std::size_t>(s - 1)]) c = t[static_cast<std::size_t>(s - 1)] + 1;
                t[static_cast<std::size_t>(s)] = c;
            }
            return t;
        }();
        return q;
    }
    static std::uint8_t quantise(std::uint32_t c) {
        if (c <= 64) return static_cast<std::uint8_t>(c);
        const auto& q = qdec();
        int lo = 65, hi = 255;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (q[static_cast<std::size_t>(mid)] < c) lo = mid + 1;
            else hi = mid;
        }
        if (lo > 65 && c - q[static_cast<std::size_t>(lo - 1)] <= q[static_cast<std::size_t>(lo)] - c) --lo;
        return static_cast<std::uint8_t>(lo);
    }

    static bool is_word(std::uint8_t b) { return std::isalnum(b) || b == '\''; }

    static std::string lower(std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static std::uint64_t mix(std::uint64_t x) {  // splitmix64 finaliser
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    std::uint64_t stored_key(std::uint64_t k56) const { return opt_.compact ? k56 >> 24 : k56; }

    std::uint16_t word_class(const std::uint8_t* w, std::size_t len) const {
        if (len == 0) return 0;
        const auto it = cls_.find(lower(std::string(reinterpret_cast<const char*>(w), len)));
        if (it != cls_.end()) return it->second;
        return std::isdigit(w[0]) ? 3 : std::isupper(w[0]) ? 2 : 1;
    }

    // k[0] = (c1, tail), k[1] = (c2, c1, tail), k[2] = (c3, c2, c1, tail), as 56-bit hashes, for the history h[0..n).
    void keys(const std::uint8_t* h, std::size_t n, std::uint64_t k[3]) const {
        const std::size_t stop = n > static_cast<std::size_t>(kScan) ? n - kScan : 0;
        std::size_t i = n;
        while (i > stop && is_word(h[i - 1])) --i;  // the word being typed
        while (i > stop && !is_word(h[i - 1])) --i;  // and the bytes before it
        const std::size_t tail_end = n,
                          tail_beg = std::max(i, n >= static_cast<std::size_t>(opt_.tail) ? n - opt_.tail : 0);
        std::uint16_t c[3] = {0, 0, 0};
        for (int w = 0; w < opt_.orders && i > stop; ++w) {  // the last two (or three) completed words
            std::size_t j = i;
            while (j > stop && is_word(h[j - 1])) --j;
            c[w] = (j == stop && stop > 0) ? 0 : word_class(h + j, i - j);
            i = j;
            while (i > stop && !is_word(h[i - 1])) --i;
        }
        std::uint64_t t = tail_end - tail_beg;
        for (std::size_t p = tail_beg; p < tail_end; ++p) t = t << 8 | h[p];
        const std::uint64_t base = mix(t ^ mix(tail_end - tail_beg));
        k[0] = mix(base ^ (std::uint64_t{c[0]} << 32 | 1)) >> 8;
        k[1] = mix(base ^ (std::uint64_t{c[0]} << 32 | std::uint64_t{c[1]} << 16 | 2)) >> 8;
        k[2] = mix(base ^ (std::uint64_t{c[2]} << 48 | std::uint64_t{c[0]} << 32 | std::uint64_t{c[1]} << 16 | 3)) >> 8;
    }

    std::size_t find(std::uint64_t key) const {
        if (opt_.compact) {
            const auto it = std::lower_bound(key32_.begin(), key32_.end(), static_cast<std::uint32_t>(key));
            return it != key32_.end() && *it == key ? static_cast<std::size_t>(it - key32_.begin()) : kNone;
        }
        const auto it = std::lower_bound(key_.begin(), key_.end(), key);
        return it != key_.end() && *it == key ? static_cast<std::size_t>(it - key_.begin()) : kNone;
    }

    std::unordered_map<std::string, std::uint16_t> cls_;
    Options opt_;
    std::vector<std::uint64_t> key_;    // sorted context hashes (default storage)
    std::vector<std::uint32_t> key32_;  // sorted 32-bit context hashes (compact storage)
    std::vector<std::uint32_t> first_;  // followers of context c: [first_[c], first_[c + 1])
    std::vector<std::uint32_t> total_;  // their summed counts (default storage)
    std::vector<std::uint8_t> byte_;
    std::vector<std::uint32_t> count_;  // default storage
    std::vector<std::uint8_t> q_;       // compact storage: quantised counts
};

}  // namespace cypha::cyphalm
