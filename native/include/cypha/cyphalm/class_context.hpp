#pragma once

/// Class-context expert for CyphaLM (PLAN B4; TOOL_PROFILE_PLAN P1, the one adapter shared by the
/// part-of-speech tagger and the word classes). It predicts the next byte from the *classes* of the
/// last two completed words plus the bytes typed since the last one ended (at most ``tail``, default 6), so a
/// context learned after "Who wrote" also serves "Who composed" when the two verbs share a class.
///
/// - A word is a run of ASCII letters, digits or apostrophes. Its class comes from a word -> class
///   table (E1 tags, or k-means word classes), looked up in lower case. A word not in the table gets
///   one of three shape classes (lower case, capitalised, number).
/// - Two orders, (c2, c1, tail) and (c1, tail), interpolated Witten-Bell style over a uniform floor.
///   The caller mixes the result into CyphaLM's distribution with one lambda per bucket(), as
///   cyphalm_trie mixes InfiniGram, so the trie stays the main expert.
/// - Counts are built once over the training bytes: (context, byte) keys are sorted and packed into
///   one table. So this expert is batch-built, not online. Peak build memory is 16 bytes a training
///   byte; the packed table is 16 bytes per distinct context plus 5 per distinct follower.
/// - Hashes are 56 bits; with ~10M contexts a collision is about a 1-in-1,000 event per table.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm {

class ClassContext {
 public:
    static constexpr int kMaxTail = 7;   ///< most bytes since the last completed word that can be kept
    static constexpr int kScan = 256;    ///< how far back a context is looked for
    static constexpr int kBuckets = 12;  ///< bucket(): deepest order found (0..2) x count bucket (0..3)

    struct Hit {
        int order = 0;            ///< 0 nothing seen, 1 (c1, tail), 2 (c2, c1, tail)
        std::uint64_t total = 0;  ///< followers counted at the deepest order found
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

    /// ``tail``: bytes since the last completed word kept in the context (1..kMaxTail).
    explicit ClassContext(std::unordered_map<std::string, std::uint16_t> classes, int tail = 6)
        : cls_(std::move(classes)), tail_(std::clamp(tail, 1, kMaxTail)) {}

    /// Count every (context, next byte) event in text[0..n).
    void build(const std::uint8_t* text, std::size_t n) {
        std::vector<std::uint64_t> ev;
        ev.reserve(2 * n);
        for (std::size_t i = 1; i < n; ++i) {
            std::uint64_t k[2];
            keys(text, i, k);
            ev.push_back(k[0] << 8 | text[i]);
            ev.push_back(k[1] << 8 | text[i]);
        }
        std::sort(ev.begin(), ev.end());
        key_.clear();
        first_.clear();
        total_.clear();
        byte_.clear();
        count_.clear();
        for (std::size_t a = 0; a < ev.size();) {
            const std::uint64_t key = ev[a] >> 8;
            key_.push_back(key);
            first_.push_back(static_cast<std::uint32_t>(byte_.size()));
            std::uint32_t tot = 0;
            while (a < ev.size() && ev[a] >> 8 == key) {
                std::size_t b = a;
                while (b < ev.size() && ev[b] == ev[a]) ++b;
                byte_.push_back(static_cast<std::uint8_t>(ev[a] & 0xff));
                count_.push_back(static_cast<std::uint32_t>(b - a));
                tot += static_cast<std::uint32_t>(b - a);
                a = b;
            }
            total_.push_back(tot);
        }
        first_.push_back(static_cast<std::uint32_t>(byte_.size()));
    }

    /// The next-byte distribution after h[0..n) (oldest byte first). P sums to 1.
    Hit predict(const std::uint8_t* h, std::size_t n, double P[256]) const {
        for (int s = 0; s < 256; ++s) P[s] = 1.0 / 256;
        Hit hit;
        std::uint64_t k[2];
        keys(h, n, k);
        for (int o = 1; o >= 0; --o) {  // lower order first: (c1, tail), then (c2, c1, tail)
            const std::size_t c = find(k[o]);
            if (c == kNone) break;
            const double N = total_[c], T = first_[c + 1] - first_[c];
            for (int s = 0; s < 256; ++s) P[s] *= T / (N + T);
            for (std::uint32_t f = first_[c]; f < first_[c + 1]; ++f) P[byte_[f]] += count_[f] / (N + T);
            hit.order = 2 - o;
            hit.total = total_[c];
        }
        return hit;
    }

    static int bucket(const Hit& h) {
        const int cb = h.total <= 1 ? 0 : h.total <= 3 ? 1 : h.total <= 15 ? 2 : 3;
        return h.order * 4 + cb;
    }

    std::size_t classes() const { return cls_.size(); }
    std::size_t contexts() const { return key_.size(); }
    std::size_t followers() const { return byte_.size(); }
    std::size_t bytes() const { return key_.size() * 16 + byte_.size() * 5; }

 private:
    static constexpr std::uint16_t kShapes = 4;  // 0 none, 1 lower, 2 capitalised, 3 number

    static constexpr std::size_t kNone = ~std::size_t{0};

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

    std::uint16_t word_class(const std::uint8_t* w, std::size_t len) const {
        if (len == 0) return 0;
        const auto it = cls_.find(lower(std::string(reinterpret_cast<const char*>(w), len)));
        if (it != cls_.end()) return it->second;
        return std::isdigit(w[0]) ? 3 : std::isupper(w[0]) ? 2 : 1;
    }

    // k[0] = (c2, c1, tail), k[1] = (c1, tail), as 56-bit hashes, for the history h[0..n).
    void keys(const std::uint8_t* h, std::size_t n, std::uint64_t k[2]) const {
        const std::size_t stop = n > static_cast<std::size_t>(kScan) ? n - kScan : 0;
        std::size_t i = n;
        while (i > stop && is_word(h[i - 1])) --i;  // the word being typed
        while (i > stop && !is_word(h[i - 1])) --i;  // and the bytes before it
        const std::size_t tail_end = n, tail_beg = std::max(i, n >= static_cast<std::size_t>(tail_) ? n - tail_ : 0);
        std::uint16_t c[2] = {0, 0};
        for (int w = 0; w < 2 && i > stop; ++w) {  // the last two completed words
            std::size_t j = i;
            while (j > stop && is_word(h[j - 1])) --j;
            c[w] = (j == stop && stop > 0) ? 0 : word_class(h + j, i - j);
            i = j;
            while (i > stop && !is_word(h[i - 1])) --i;
        }
        std::uint64_t t = tail_end - tail_beg;
        for (std::size_t p = tail_beg; p < tail_end; ++p) t = t << 8 | h[p];
        const std::uint64_t base = mix(t ^ mix(tail_end - tail_beg));
        k[1] = mix(base ^ (std::uint64_t{c[0]} << 32 | 1)) >> 8;
        k[0] = mix(base ^ (std::uint64_t{c[0]} << 32 | std::uint64_t{c[1]} << 16 | 2)) >> 8;
    }

    std::size_t find(std::uint64_t key) const {
        const auto it = std::lower_bound(key_.begin(), key_.end(), key);
        return it != key_.end() && *it == key ? static_cast<std::size_t>(it - key_.begin()) : kNone;
    }

    std::unordered_map<std::string, std::uint16_t> cls_;
    int tail_;
    std::vector<std::uint64_t> key_;    // sorted context hashes
    std::vector<std::uint32_t> first_;  // followers of context c: [first_[c], first_[c + 1])
    std::vector<std::uint32_t> total_;  // their summed counts
    std::vector<std::uint8_t> byte_;
    std::vector<std::uint32_t> count_;
};

}  // namespace cypha::cyphalm
