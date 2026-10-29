// CyphaLM tokenizers learned by counting (docs/TOKENS_PLAN.md T1; docs/reports/TOKENS_T0_T2.md).
//
// Every tokenizer keeps the 256 single bytes as tokens 0..255, so every byte string encodes and decodes
// exactly. Nothing here is a neural network: vocabularies come from pair counts (BPE), a likelihood over
// counts (Unigram LM), chunk counts (SpaceByte, whole words) or CyphaLM's own next-byte entropy (Entropy).
//
//   BPE      GPT-style pre-tokens (a space joins the word after it), merges by pair frequency
//   Unigram  the same pre-tokens, Kudo's unigram LM: frequent substrings, EM, pruning by loss, Viterbi
//   SpaceByte chunks end after a space-like byte not preceded by another; the most frequent chunks
//   Entropy  segments cut where the byte model's next-byte entropy passes a threshold; the most frequent
//   Words    the same pre-tokens as BPE; the most frequent whole pre-tokens, others as bytes
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cypha::cyphalm::tok {

enum class Kind : int { BPE = 1, Unigram = 2, SpaceByte = 3, Entropy = 4, Words = 5 };

inline const char* kind_name(Kind k) {
    switch (k) {
        case Kind::BPE: return "bpe";
        case Kind::Unigram: return "uni";
        case Kind::SpaceByte: return "spb";
        case Kind::Entropy: return "ent";
        case Kind::Words: return "wrd";
    }
    return "?";
}

inline Kind kind_from(const std::string& s) {
    if (s == "bpe") return Kind::BPE;
    if (s == "uni") return Kind::Unigram;
    if (s == "spb") return Kind::SpaceByte;
    if (s == "ent") return Kind::Entropy;
    if (s == "wrd") return Kind::Words;
    throw std::runtime_error("unknown tokenizer arm " + s);
}

// ---- byte classes and pre-tokens

inline bool is_letter(std::uint8_t c) { return ((c | 32) >= 'a' && (c | 32) <= 'z') || c >= 0x80; }
inline bool is_digit(std::uint8_t c) { return c >= '0' && c <= '9'; }

// 0 letter (or any byte >= 0x80), 1 digit, 2 other symbol, 3 space, 4 other whitespace.
inline int byte_class(std::uint8_t c) {
    if (is_letter(c)) return 0;
    if (is_digit(c)) return 1;
    if (c == ' ') return 3;
    if (c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') return 4;
    return 2;
}

// GPT-2's split without the contractions: " ?letters | ?digits | ?symbols | whitespace", where a whitespace
// run leaves its last space to the pre-token after it. Calls f(begin, end) for each pre-token in order.
template <class F>
void pretokens(const std::uint8_t* s, std::size_t n, F&& f) {
    std::size_t i = 0;
    while (i < n) {
        const int c = byte_class(s[i]);
        std::size_t j;
        if (c == 3 && i + 1 < n && byte_class(s[i + 1]) <= 2) {
            const int c2 = byte_class(s[i + 1]);
            j = i + 2;
            while (j < n && byte_class(s[j]) == c2) ++j;
        } else if (c <= 2) {
            j = i + 1;
            while (j < n && byte_class(s[j]) == c) ++j;
        } else {
            j = i + 1;
            while (j < n && byte_class(s[j]) >= 3) ++j;
            if (j < n && j - i > 1 && s[j - 1] == ' ') --j;
        }
        f(i, j);
        i = j;
    }
}

// SpaceByte (Slagle 2024): a byte is space-like unless it is a letter, a digit or a UTF-8 continuation byte;
// a chunk ends after a space-like byte that is not preceded by another space-like byte.
inline bool spacelike(std::uint8_t c) {
    const bool letter = (c | 32) >= 'a' && (c | 32) <= 'z';
    return !(letter || is_digit(c) || (c >= 0x80 && c < 0xC0));
}

template <class F>
void spacebyte_chunks(const std::uint8_t* s, std::size_t n, F&& f) {
    std::size_t start = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (spacelike(s[i]) && (i == 0 || !spacelike(s[i - 1]))) {
            f(start, i + 1);
            start = i + 1;
        }
    }
    if (start < n) f(start, n);
}

struct SvHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};
using PieceIndex = std::unordered_map<std::string, std::uint32_t, SvHash, std::equal_to<>>;

inline std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) { return (std::uint64_t{a} << 32) | b; }

// ---- the tokenizer

class Tokenizer {
 public:
    Kind kind = Kind::BPE;
    std::vector<std::string> pieces;                             // id -> bytes; 0..255 are the single bytes
    std::vector<std::pair<std::uint32_t, std::uint32_t>> merges;  // BPE: merge r makes token 256 + r
    std::vector<double> logp;                                    // Unigram: log probability of each piece
    std::size_t max_len = 1;                                     // longest piece in bytes

    std::size_t size() const { return pieces.size(); }

    // Rebuilds the lookup tables from pieces / merges. Call after building or loading.
    void finish() {
        index_.clear();
        rank_.clear();
        cache_.clear();
        max_len = 1;
        for (std::uint32_t i = 0; i < pieces.size(); ++i) {
            index_.emplace(pieces[i], i);
            max_len = std::max(max_len, pieces[i].size());
        }
        for (std::uint32_t r = 0; r < merges.size(); ++r) rank_.emplace(pair_key(merges[r].first, merges[r].second), r);
    }

    std::uint32_t find(std::string_view s) const {
        const auto it = index_.find(s);
        return it == index_.end() ? kNone : it->second;
    }

    // Appends the tokens of s[0..n) to out.
    void encode(const std::uint8_t* s, std::size_t n, std::vector<std::uint32_t>& out) const {
        switch (kind) {
            case Kind::BPE:
            case Kind::Unigram:
            case Kind::Words:
                pretokens(s, n, [&](std::size_t a, std::size_t b) { encode_chunk(s + a, b - a, out); });
                break;
            case Kind::SpaceByte:
                spacebyte_chunks(s, n, [&](std::size_t a, std::size_t b) { encode_chunk(s + a, b - a, out); });
                break;
            case Kind::Entropy: longest_match(s, n, out); break;
        }
    }
    std::vector<std::uint32_t> encode(std::string_view s) const {
        std::vector<std::uint32_t> out;
        encode(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), out);
        return out;
    }

    std::string decode(const std::uint32_t* ids, std::size_t n) const {
        std::string out;
        for (std::size_t i = 0; i < n; ++i) out += pieces.at(ids[i]);
        return out;
    }
    std::string decode(const std::vector<std::uint32_t>& ids) const { return decode(ids.data(), ids.size()); }

    // Text file: "CYPHATOK 1 <kind> <size> <merges>", then one hex piece a line (with its log probability for
    // Unigram), then one "a b" a line for each BPE merge.
    void save(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) throw std::runtime_error("cannot write " + path);
        os << "CYPHATOK 1 " << kind_name(kind) << ' ' << pieces.size() << ' ' << merges.size() << '\n';
        char hex[3];
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            for (unsigned char c : pieces[i]) {
                std::snprintf(hex, sizeof hex, "%02x", c);
                os << hex;
            }
            if (kind == Kind::Unigram) {
                char buf[40];
                std::snprintf(buf, sizeof buf, " %.17g", logp[i]);
                os << buf;
            }
            os << '\n';
        }
        for (const auto& [a, b] : merges) os << a << ' ' << b << '\n';
        if (!os) throw std::runtime_error("write failed " + path);
    }

    static Tokenizer load(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) throw std::runtime_error("cannot open " + path);
        std::string magic, kname;
        int version = 0;
        std::size_t np = 0, nm = 0;
        is >> magic >> version >> kname >> np >> nm;
        if (magic != "CYPHATOK" || version != 1) throw std::runtime_error("not a tokenizer file " + path);
        Tokenizer t;
        t.kind = kind_from(kname);
        std::string line;
        std::getline(is, line);
        t.pieces.resize(np);
        if (t.kind == Kind::Unigram) t.logp.resize(np);
        for (std::size_t i = 0; i < np; ++i) {
            if (!std::getline(is, line)) throw std::runtime_error("short tokenizer file " + path);
            const std::size_t sp = line.find(' ');
            const std::string hex = line.substr(0, sp);
            std::string piece;
            for (std::size_t k = 0; k + 1 < hex.size(); k += 2)
                piece.push_back(static_cast<char>(std::stoi(hex.substr(k, 2), nullptr, 16)));
            t.pieces[i] = piece;
            if (t.kind == Kind::Unigram) t.logp[i] = std::stod(line.substr(sp + 1));
        }
        t.merges.resize(nm);
        for (std::size_t r = 0; r < nm; ++r) is >> t.merges[r].first >> t.merges[r].second;
        for (std::uint32_t b = 0; b < 256; ++b)
            if (t.pieces[b] != std::string(1, static_cast<char>(b))) throw std::runtime_error("byte fallback missing");
        t.finish();
        return t;
    }

    static constexpr std::uint32_t kNone = 0xffffffffu;

 private:
    void encode_chunk(const std::uint8_t* s, std::size_t n, std::vector<std::uint32_t>& out) const {
        const std::string_view key(reinterpret_cast<const char*>(s), n);
        if (n <= 64) {
            const auto it = cache_.find(key);
            if (it != cache_.end()) {
                out.insert(out.end(), it->second.begin(), it->second.end());
                return;
            }
        }
        std::vector<std::uint32_t> ids;
        switch (kind) {
            case Kind::BPE: bpe(s, n, ids); break;
            case Kind::Unigram: viterbi(s, n, ids); break;
            case Kind::Words: {
                const std::uint32_t id = find(key);
                if (id != kNone) ids.push_back(id);
                else
                    for (std::size_t i = 0; i < n; ++i) ids.push_back(s[i]);
                break;
            }
            case Kind::SpaceByte: {
                const std::uint32_t id = find(key);
                if (id != kNone) ids.push_back(id);
                else longest_match(s, n, ids);
                break;
            }
            case Kind::Entropy: longest_match(s, n, ids); break;
        }
        if (n <= 64 && cache_.size() < (std::size_t{1} << 20)) cache_.emplace(std::string(key), ids);
        out.insert(out.end(), ids.begin(), ids.end());
    }

    // GPT-2's rule: merge the pair of lowest rank, all its occurrences left to right, until none is left.
    void bpe(const std::uint8_t* s, std::size_t n, std::vector<std::uint32_t>& w) const {
        w.assign(s, s + n);
        std::vector<std::uint32_t> nw;
        while (w.size() > 1) {
            std::uint32_t best = kNone;
            for (std::size_t k = 0; k + 1 < w.size(); ++k) {
                const auto it = rank_.find(pair_key(w[k], w[k + 1]));
                if (it != rank_.end() && it->second < best) best = it->second;
            }
            if (best == kNone) break;
            const auto [a, b] = merges[best];
            const std::uint32_t nt = 256 + best;
            nw.clear();
            for (std::size_t k = 0; k < w.size();) {
                if (k + 1 < w.size() && w[k] == a && w[k + 1] == b) {
                    nw.push_back(nt);
                    k += 2;
                } else {
                    nw.push_back(w[k++]);
                }
            }
            w.swap(nw);
        }
    }

    void viterbi(const std::uint8_t* s, std::size_t n, std::vector<std::uint32_t>& ids) const {
        std::vector<double> best(n + 1, -std::numeric_limits<double>::infinity());
        std::vector<std::uint32_t> from(n + 1, 0), piece(n + 1, 0);
        best[0] = 0;
        const char* c = reinterpret_cast<const char*>(s);
        for (std::size_t i = 0; i < n; ++i) {
            if (best[i] == -std::numeric_limits<double>::infinity()) continue;
            const std::size_t lim = std::min(max_len, n - i);
            for (std::size_t l = 1; l <= lim; ++l) {
                const std::uint32_t id = find(std::string_view(c + i, l));
                if (id == kNone) continue;
                const double v = best[i] + logp[id];
                if (v > best[i + l]) {
                    best[i + l] = v;
                    from[i + l] = static_cast<std::uint32_t>(i);
                    piece[i + l] = id;
                }
            }
        }
        std::vector<std::uint32_t> rev;
        for (std::size_t j = n; j > 0; j = from[j]) rev.push_back(piece[j]);
        ids.insert(ids.end(), rev.rbegin(), rev.rend());
    }

    void longest_match(const std::uint8_t* s, std::size_t n, std::vector<std::uint32_t>& ids) const {
        const char* c = reinterpret_cast<const char*>(s);
        for (std::size_t i = 0; i < n;) {
            std::size_t l = std::min(max_len, n - i);
            for (; l > 1; --l)
                if (find(std::string_view(c + i, l)) != kNone) break;
            ids.push_back(l == 1 ? static_cast<std::uint32_t>(s[i]) : find(std::string_view(c + i, l)));
            i += l;
        }
    }

    PieceIndex index_;
    std::unordered_map<std::uint64_t, std::uint32_t> rank_;
    mutable std::unordered_map<std::string, std::vector<std::uint32_t>, SvHash, std::equal_to<>> cache_;
};

inline Tokenizer bytes_only(Kind k) {
    Tokenizer t;
    t.kind = k;
    for (int b = 0; b < 256; ++b) t.pieces.emplace_back(1, static_cast<char>(b));
    return t;
}

// Counts of the pre-tokens (or SpaceByte chunks) of a text.
inline std::unordered_map<std::string, std::uint64_t> count_chunks(const std::uint8_t* s, std::size_t n, bool spacebyte) {
    std::unordered_map<std::string, std::uint64_t> c;
    auto add = [&](std::size_t a, std::size_t b) { ++c[std::string(reinterpret_cast<const char*>(s + a), b - a)]; };
    if (spacebyte) spacebyte_chunks(s, n, add);
    else pretokens(s, n, add);
    return c;
}

// The chunks sorted by count (descending), ties by bytes, so every run is the same.
inline std::vector<std::pair<std::string, std::uint64_t>> sorted_chunks(const std::unordered_map<std::string, std::uint64_t>& c) {
    std::vector<std::pair<std::string, std::uint64_t>> v(c.begin(), c.end());
    std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second != y.second ? x.second > y.second : x.first < y.first; });
    return v;
}

// ---- BPE training: one run to the largest size; a smaller vocabulary is the first merges (a prefix).

inline Tokenizer train_bpe(const std::unordered_map<std::string, std::uint64_t>& chunks, std::size_t vocab, bool log = false) {
    Tokenizer t = bytes_only(Kind::BPE);
    const auto sorted = sorted_chunks(chunks);
    std::vector<std::vector<std::uint32_t>> words;
    std::vector<std::int64_t> wc;
    words.reserve(sorted.size());
    for (const auto& [s, c] : sorted) {
        if (s.size() < 2) continue;
        words.emplace_back(s.begin(), s.end());
        for (auto& x : words.back()) x &= 0xff;
        wc.push_back(static_cast<std::int64_t>(c));
    }
    std::unordered_map<std::uint64_t, std::int64_t> pc;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> where;
    for (std::uint32_t w = 0; w < words.size(); ++w) {
        const auto& v = words[w];
        for (std::size_t k = 0; k + 1 < v.size(); ++k) {
            const std::uint64_t key = pair_key(v[k], v[k + 1]);
            pc[key] += wc[w];
            auto& wl = where[key];
            if (wl.empty() || wl.back() != w) wl.push_back(w);
        }
    }
    // Max-heap of (count, ~pair): the most frequent pair first, ties by the lowest pair key.
    std::priority_queue<std::pair<std::int64_t, std::uint64_t>> heap;
    for (const auto& [k, c] : pc) heap.emplace(c, ~k);
    std::vector<std::uint32_t> stamp(words.size(), 0);
    std::uint32_t round = 0;
    std::vector<std::pair<std::uint64_t, std::int64_t>> delta;
    std::vector<std::uint32_t> nw;
    while (t.pieces.size() < vocab && !heap.empty()) {
        const auto [c, nk] = heap.top();
        heap.pop();
        const std::uint64_t key = ~nk;
        const auto it = pc.find(key);
        const std::int64_t cur = it == pc.end() ? 0 : it->second;
        if (cur != c) {
            if (cur > 0 && cur < c) heap.emplace(cur, nk);  // stale: put it back with its count now
            continue;
        }
        if (c < 2) break;
        const std::uint32_t a = static_cast<std::uint32_t>(key >> 32), b = static_cast<std::uint32_t>(key & 0xffffffffu);
        const std::uint32_t nt = static_cast<std::uint32_t>(t.pieces.size());
        t.merges.emplace_back(a, b);
        t.pieces.push_back(t.pieces[a] + t.pieces[b]);
        ++round;
        std::vector<std::uint32_t> ws = std::move(where[key]);
        where.erase(key);
        for (const std::uint32_t w : ws) {
            if (stamp[w] == round) continue;
            stamp[w] = round;
            auto& v = words[w];
            bool has = false;
            for (std::size_t k = 0; k + 1 < v.size(); ++k)
                if (v[k] == a && v[k + 1] == b) {
                    has = true;
                    break;
                }
            if (!has) continue;
            delta.clear();
            for (std::size_t k = 0; k + 1 < v.size(); ++k) delta.emplace_back(pair_key(v[k], v[k + 1]), -wc[w]);
            nw.clear();
            for (std::size_t k = 0; k < v.size();) {
                if (k + 1 < v.size() && v[k] == a && v[k + 1] == b) {
                    nw.push_back(nt);
                    k += 2;
                } else {
                    nw.push_back(v[k++]);
                }
            }
            v.swap(nw);
            for (std::size_t k = 0; k + 1 < v.size(); ++k) delta.emplace_back(pair_key(v[k], v[k + 1]), wc[w]);
            std::sort(delta.begin(), delta.end());
            for (std::size_t i = 0; i < delta.size();) {
                std::size_t j = i;
                std::int64_t d = 0;
                while (j < delta.size() && delta[j].first == delta[i].first) d += delta[j++].second;
                const std::uint64_t pk = delta[i].first;
                bool present = false;
                for (std::size_t q = i; q < j; ++q)
                    if (delta[q].second > 0) present = true;
                if (d != 0) {
                    std::int64_t& cnt = pc[pk];
                    cnt += d;
                    if (d > 0) heap.emplace(cnt, ~pk);
                    if (cnt <= 0) pc.erase(pk);
                }
                if (present && pk != key) {
                    auto& wl = where[pk];
                    if (wl.empty() || wl.back() != w) wl.push_back(w);
                }
                i = j;
            }
        }
        pc.erase(key);
        if (log && (t.pieces.size() & 4095) == 0) std::fprintf(stderr, "bpe vocab %zu\n", t.pieces.size());
    }
    t.finish();
    return t;
}

// The first `vocab` tokens of a trained BPE (its first vocab - 256 merges).
inline Tokenizer bpe_prefix(const Tokenizer& full, std::size_t vocab) {
    Tokenizer t;
    t.kind = Kind::BPE;
    vocab = std::min(vocab, full.pieces.size());
    t.pieces.assign(full.pieces.begin(), full.pieces.begin() + static_cast<std::ptrdiff_t>(vocab));
    t.merges.assign(full.merges.begin(), full.merges.begin() + static_cast<std::ptrdiff_t>(vocab - 256));
    t.finish();
    return t;
}

// ---- Most-frequent-chunk vocabularies (Words, SpaceByte, Entropy segments)

inline Tokenizer top_chunks(Kind kind, const std::vector<std::pair<std::string, std::uint64_t>>& sorted, std::size_t vocab) {
    Tokenizer t = bytes_only(kind);
    for (const auto& [s, c] : sorted) {
        if (t.pieces.size() >= vocab) break;
        if (s.size() < 2 || c < 2) continue;
        t.pieces.push_back(s);
    }
    t.finish();
    return t;
}

// ---- Unigram LM (Kudo 2018)

struct UnigramTrainer {
    std::vector<std::string> chunk;   // unique pre-tokens
    std::vector<double> weight;       // their counts
    std::size_t max_piece = 16;

    // Expected counts and the corpus log likelihood under t (forward-backward over each chunk's lattice).
    double em_step(const Tokenizer& t, std::vector<double>& expected) const {
        expected.assign(t.pieces.size(), 0.0);
        double ll = 0;
        const double ninf = -std::numeric_limits<double>::infinity();
        std::vector<double> alpha, beta;
        struct Edge {
            std::uint32_t i, l, id;
        };
        std::vector<Edge> edges;
        auto lse = [&](double a, double b) {
            if (a == ninf) return b;
            if (b == ninf) return a;
            const double m = std::max(a, b);
            return m + std::log(std::exp(a - m) + std::exp(b - m));
        };
        for (std::size_t c = 0; c < chunk.size(); ++c) {
            const std::string& s = chunk[c];
            const std::size_t n = s.size();
            edges.clear();
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t lim = std::min(max_piece, n - i);
                for (std::size_t l = 1; l <= lim; ++l) {
                    const std::uint32_t id = t.find(std::string_view(s.data() + i, l));
                    if (id != Tokenizer::kNone) edges.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(l), id});
                }
            }
            alpha.assign(n + 1, ninf);
            beta.assign(n + 1, ninf);
            alpha[0] = 0;
            for (const auto& e : edges) alpha[e.i + e.l] = lse(alpha[e.i + e.l], alpha[e.i] + t.logp[e.id]);
            beta[n] = 0;
            for (std::size_t k = edges.size(); k-- > 0;) {
                const auto& e = edges[k];
                beta[e.i] = lse(beta[e.i], beta[e.i + e.l] + t.logp[e.id]);
            }
            const double z = alpha[n];
            ll += weight[c] * z;
            for (const auto& e : edges) {
                const double post = std::exp(alpha[e.i] + t.logp[e.id] + beta[e.i + e.l] - z);
                expected[e.id] += weight[c] * post;
            }
        }
        return ll;
    }

    static void m_step(Tokenizer& t, const std::vector<double>& expected) {
        double total = 0;
        for (double x : expected) total += x;
        for (std::size_t i = 0; i < t.pieces.size(); ++i)
            t.logp[i] = std::log(std::max(expected[i], 1e-10) / total);
    }

    // The best segmentation of a piece's bytes without the piece itself, as a log probability.
    static double alternative(const Tokenizer& t, std::uint32_t id) {
        const std::string& s = t.pieces[id];
        const std::size_t n = s.size();
        std::vector<double> best(n + 1, -std::numeric_limits<double>::infinity());
        best[0] = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (best[i] == -std::numeric_limits<double>::infinity()) continue;
            for (std::size_t l = 1; l <= n - i && l <= t.max_len; ++l) {
                if (i == 0 && l == n) continue;
                const std::uint32_t j = t.find(std::string_view(s.data() + i, l));
                if (j == Tokenizer::kNone) continue;
                best[i + l] = std::max(best[i + l], best[i] + t.logp[j]);
            }
        }
        return best[n];
    }
};

// Trains Unigram LMs and returns one for each requested size (largest first in the schedule).
inline std::vector<Tokenizer> train_unigram(const std::unordered_map<std::string, std::uint64_t>& chunks,
                                            std::vector<std::size_t> sizes, std::size_t seed_size, bool log = false) {
    UnigramTrainer tr;
    const auto sorted = sorted_chunks(chunks);
    for (const auto& [s, c] : sorted) {
        tr.chunk.push_back(s);
        tr.weight.push_back(static_cast<double>(c));
    }
    // Seed: frequent substrings of 2..16 bytes of chunks seen at least twice, scored by frequency x length.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> hs;  // (hash with length, weight)
    auto hsub = [](const char* p, std::size_t l) {
        return std::hash<std::string_view>{}(std::string_view(p, l)) * 31 + l;
    };
    for (std::size_t c = 0; c < tr.chunk.size(); ++c) {
        if (tr.weight[c] < 2) continue;
        const std::string& s = tr.chunk[c];
        for (std::size_t i = 0; i < s.size(); ++i)
            for (std::size_t l = 2; l <= tr.max_piece && i + l <= s.size(); ++l)
                hs.emplace_back(hsub(s.data() + i, l), static_cast<std::uint64_t>(tr.weight[c]));
    }
    std::sort(hs.begin(), hs.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> agg;
    for (std::size_t i = 0; i < hs.size();) {
        std::size_t j = i;
        std::uint64_t w = 0;
        while (j < hs.size() && hs[j].first == hs[i].first) w += hs[j++].second;
        if (w >= 2) agg.emplace_back(hs[i].first, w);
        i = j;
    }
    hs.clear();
    hs.shrink_to_fit();
    std::unordered_map<std::uint64_t, std::uint64_t> freq(agg.begin(), agg.end());
    std::unordered_map<std::uint64_t, std::string> text;
    std::vector<std::pair<double, std::string>> cand;
    for (std::size_t c = 0; c < tr.chunk.size(); ++c) {
        if (tr.weight[c] < 2) continue;
        const std::string& s = tr.chunk[c];
        for (std::size_t i = 0; i < s.size(); ++i)
            for (std::size_t l = 2; l <= tr.max_piece && i + l <= s.size(); ++l) {
                const std::uint64_t h = hsub(s.data() + i, l);
                const auto it = freq.find(h);
                if (it == freq.end()) continue;
                cand.emplace_back(static_cast<double>(it->second) * static_cast<double>(l), s.substr(i, l));
                freq.erase(it);
            }
    }
    std::sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    if (cand.size() > seed_size) cand.resize(seed_size);
    Tokenizer t = bytes_only(Kind::Unigram);
    for (const auto& [score, s] : cand) t.pieces.push_back(s);
    // Initial log probabilities from the seed scores; bytes from their counts.
    std::vector<double> init(t.pieces.size(), 1.0);
    for (std::size_t c = 0; c < tr.chunk.size(); ++c)
        for (unsigned char b : tr.chunk[c]) init[b] += tr.weight[c];
    for (std::size_t i = 256; i < t.pieces.size(); ++i) init[i] = cand[i - 256].first;
    t.logp.resize(t.pieces.size());
    UnigramTrainer::m_step(t, init);
    t.finish();

    std::sort(sizes.begin(), sizes.end(), std::greater<>());
    std::vector<Tokenizer> out;
    std::vector<double> expected;
    for (const std::size_t target : sizes) {
        for (int round = 0; round < 64; ++round) {
            for (int it = 0; it < 2; ++it) {
                const double ll = tr.em_step(t, expected);
                UnigramTrainer::m_step(t, expected);
                if (log) std::fprintf(stderr, "uni size %zu ll %.6g\n", t.pieces.size(), ll);
            }
            if (t.pieces.size() <= target) break;
            // Loss of removing each piece: its expected count times the drop to its best alternative.
            tr.em_step(t, expected);
            std::vector<std::pair<double, std::uint32_t>> loss;
            for (std::uint32_t i = 256; i < t.pieces.size(); ++i) {
                const double alt = UnigramTrainer::alternative(t, i);
                loss.emplace_back(expected[i] * (t.logp[i] - alt), i);
            }
            std::sort(loss.begin(), loss.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
            const std::size_t keep_extra = std::max(target - 256, static_cast<std::size_t>(0.75 * static_cast<double>(t.pieces.size() - 256)));
            std::vector<std::uint32_t> kept;
            for (std::size_t k = 0; k < loss.size() && kept.size() < keep_extra; ++k) kept.push_back(loss[k].second);
            std::sort(kept.begin(), kept.end());
            Tokenizer nt = bytes_only(Kind::Unigram);
            nt.logp.assign(t.logp.begin(), t.logp.begin() + 256);
            for (const std::uint32_t i : kept) {
                nt.pieces.push_back(t.pieces[i]);
                nt.logp.push_back(t.logp[i]);
            }
            nt.finish();
            t = std::move(nt);
        }
        for (int it = 0; it < 2; ++it) {
            tr.em_step(t, expected);
            UnigramTrainer::m_step(t, expected);
        }
        t.finish();
        out.push_back(t);
    }
    return out;
}

// ---- Entropy segments (BLT's global-threshold rule on CyphaLM's own counts)

// Cuts before byte i when ent[i] > theta. theta is the quantile that gives a mean segment of mean_len bytes.
inline double entropy_threshold(const std::vector<float>& ent, double mean_len) {
    std::vector<float> v(ent.begin(), ent.end());
    const double q = 1.0 - 1.0 / mean_len;
    const std::size_t k = static_cast<std::size_t>(q * static_cast<double>(v.size()));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return v[k];
}

inline std::unordered_map<std::string, std::uint64_t> count_entropy_segments(const std::uint8_t* s, std::size_t n,
                                                                             const std::vector<float>& ent, double theta,
                                                                             std::size_t max_len = 24) {
    std::unordered_map<std::string, std::uint64_t> c;
    std::size_t start = 0;
    for (std::size_t i = 1; i <= n; ++i) {
        if (i == n || ent[i] > theta || i - start >= max_len) {
            ++c[std::string(reinterpret_cast<const char*>(s + start), i - start)];
            start = i;
        }
    }
    return c;
}

}  // namespace cypha::cyphalm::tok
