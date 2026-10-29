#pragma once
// Exact next-byte probabilities from a token-level CyphaLM (docs/TOKENS_PLAN.md item T3;
// docs/reports/TOKENS_T3_T4.md). Follows Phan et al., "Exact Byte-Level Probabilities from
// Tokenized Language Models" (ICLR 2025, arXiv 2410.09303):
//   - a token sequence t is valid when encode(decode(t)) == t; for BPE, longest-match and
//     Viterbi tokenizers, only valid sequences are prefixes of canonical encodings;
//   - the covers of a byte prefix x are the valid sequences whose last token covers the end of
//     x; they partition the continuations of x, so P(x) = sum over covers of P(t);
//   - a valid sequence that decodes to exactly s is encode(s), so every cover is
//     encode(x[0:i]) followed by one token that starts with x[i:n];
//   - a count model gives invalid tokens some probability; each next-token distribution is
//     truncated to the valid tokens and renormalised (their Proposition 2).
// Nothing here answers a question (PLAN rule 10): it converts a token model's predictions into
// byte predictions, and samples whole tokens after a byte prompt.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::cyphalm::bridge {

// ---------------------------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------------------------

// A deterministic tokenizer, as the bridge needs it. All 256 single bytes should be tokens.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual int vocab_size() const = 0;
    virtual std::string_view token(int id) const = 0;
    // The canonical encoding of s.
    virtual std::vector<int> encode(std::string_view s) const = 0;
    // A stable position p <= s.size() of s. The contract (checked by check_contract):
    //   S1: for every w, encode(s + w) == encode(s[0:p]) ++ encode(s[p:] + w);
    //   S2: for every i in [p, |s|], encode(s[0:i]) == encode(s[0:p]) ++ encode(s[p:i]).
    // With word pretokens, p is the start of the last pretoken. `hint` is a stable position of s
    // already known to the caller (0 is always allowed); scanning may start there.
    virtual std::size_t stable_prefix(std::string_view s, std::size_t hint) const = 0;
};

// A token-level model: next-token probabilities after a token history (out.size() ==
// vocab_size(), summing to 1).
class Model {
public:
    virtual ~Model() = default;
    virtual void next(std::span<const int> history, std::vector<double>& out) const = 0;
};

struct Options {
    bool renormalize = true;         // truncate each next-token distribution to valid tokens
    std::size_t valid_cache = 4096;  // tails whose valid-next-token sets are kept
};

// ---------------------------------------------------------------------------------------------
// The bridge
// ---------------------------------------------------------------------------------------------

class Bridge {
public:
    Bridge(const Tokenizer& tok, const Model& model, Options opt = {})
        : tok_(tok), model_(model), opt_(opt), V_(tok.vocab_size()) {
        sorted_.resize(static_cast<std::size_t>(V_));
        canon_.assign(static_cast<std::size_t>(V_), 0);
        for (int v = 0; v < V_; ++v) {
            sorted_[static_cast<std::size_t>(v)] = v;
            const std::string_view b = tok_.token(v);
            lmax_ = std::max(lmax_, b.size());
            const std::vector<int> e = tok_.encode(b);
            canon_[static_cast<std::size_t>(v)] = (e.size() == 1 && e[0] == v) ? 1 : 0;
        }
        std::sort(sorted_.begin(), sorted_.end(),
                  [&](int a, int b) { return tok_.token(a) < tok_.token(b); });
    }

    const Tokenizer& tokenizer() const { return tok_; }
    const Model& model() const { return model_; }
    const Options& options() const { return opt_; }
    std::size_t max_token_bytes() const { return lmax_; }
    int vocab_size() const { return V_; }

    // Token ids whose bytes start with q (a contiguous range of the sorted vocabulary).
    std::pair<std::size_t, std::size_t> prefix_range(std::string_view q) const {
        const auto lo = std::lower_bound(sorted_.begin(), sorted_.end(), q,
                                         [&](int a, std::string_view k) { return tok_.token(a) < k; });
        const auto hi = std::partition_point(lo, sorted_.end(), [&](int a) {
            const std::string_view b = tok_.token(a);
            return b.size() >= q.size() && b.compare(0, q.size(), q) == 0;
        });
        return {static_cast<std::size_t>(lo - sorted_.begin()), static_cast<std::size_t>(hi - sorted_.begin())};
    }
    int sorted_at(std::size_t k) const { return sorted_[k]; }

    // valid[v] == 1 when history ++ [v] is valid, i.e. encode(hbytes + token(v)) == history ++ [v].
    // hbytes = decode(history); history must itself be valid. `hint` is a stable position of hbytes
    // (a stable position of a string is stable in every extension of it, so callers carry it along).
    const std::vector<char>& valid_next(std::span<const int> history, std::string_view hbytes, std::size_t hint = 0) {
        const std::size_t p = tok_.stable_prefix(hbytes, hint);
        const std::string_view tail = hbytes.substr(p);
        std::string key(tail);
        auto it = valid_.find(key);
        if (it != valid_.end()) return it->second;
        if (valid_.size() >= opt_.valid_cache) valid_.clear();
        // The tokens of history that cover the tail (S2: encode(hbytes) = encode(hbytes[0:p]) ++ encode(tail)).
        std::size_t k = history.size(), cov = 0;
        while (cov < tail.size() && k > 0) cov += tok_.token(history[--k]).size();
        const std::span<const int> tail_toks = history.subspan(k);
        std::vector<char> ok(static_cast<std::size_t>(V_), 0);
        std::string buf(tail);
        for (int v = 0; v < V_; ++v) {
            if (!canon_[static_cast<std::size_t>(v)]) continue;
            buf.resize(tail.size());
            buf.append(tok_.token(v));
            const std::vector<int> e = tok_.encode(buf);
            if (e.size() != tail_toks.size() + 1 || e.back() != v) continue;
            ok[static_cast<std::size_t>(v)] = std::equal(tail_toks.begin(), tail_toks.end(), e.begin()) ? 1 : 0;
        }
        return valid_.emplace(std::move(key), std::move(ok)).first->second;
    }

    // The model's next-token distribution after a valid history, masked to valid tokens, and
    // renormalised over them when options().renormalize.
    void conditional(std::span<const int> history, std::string_view hbytes, std::vector<double>& q,
                     std::size_t hint = 0) {
        model_.next(history, q);
        const std::vector<char>& ok = valid_next(history, hbytes, hint);
        double z = 0;
        for (int v = 0; v < V_; ++v) {
            if (!ok[static_cast<std::size_t>(v)]) q[static_cast<std::size_t>(v)] = 0;
            z += q[static_cast<std::size_t>(v)];
        }
        if (opt_.renormalize && z > 0)
            for (double& x : q) x /= z;
    }

private:
    const Tokenizer& tok_;
    const Model& model_;
    Options opt_;
    int V_;
    std::size_t lmax_ = 1;
    std::vector<int> sorted_;
    std::vector<char> canon_;
    std::unordered_map<std::string, std::vector<char>> valid_;
};

// ---------------------------------------------------------------------------------------------
// A byte stream read through the bridge
// ---------------------------------------------------------------------------------------------

// Bytes are pushed one at a time (or in runs). The committed tokens (before the stable position)
// and their log-probability are kept, so each next-byte distribution costs at most
// max_token_bytes() + 1 model calls on short open histories.
class Stream {
public:
    explicit Stream(Bridge& b) : b_(b), q_(static_cast<std::size_t>(b.vocab_size())) {}

    const std::string& bytes() const { return x_; }
    std::size_t stable() const { return p_; }
    // A stable position of x[0:stable()] for any extension (a hint for histories past it).
    std::size_t stable_hint() const { return hs_; }
    const std::vector<int>& committed() const { return C_; }

    void push(std::string_view s) {
        x_.append(s);
        const std::size_t np = b_.tokenizer().stable_prefix(x_, p_);
        if (np <= p_) return;
        // S2 with the old stable position: encode(x[0:np]) = encode(x[0:p]) ++ encode(x[p:np]).
        const std::vector<int> nt = b_.tokenizer().encode(std::string_view(x_).substr(p_, np - p_));
        std::size_t end = p_;
        for (int t : nt) {
            const std::string_view hb = std::string_view(x_).substr(0, end);
            hs_ = b_.tokenizer().stable_prefix(hb, hs_);
            b_.conditional(C_, hb, q_, hs_);
            lpC_ += std::log(q_[static_cast<std::size_t>(t)]);
            C_.push_back(t);
            end += b_.tokenizer().token(t).size();
        }
        hs_ = b_.tokenizer().stable_prefix(std::string_view(x_).substr(0, np), hs_);
        p_ = np;
    }

    // Exact next-byte distribution after bytes() (all zero if bytes() has probability 0).
    std::array<double, 256> next_byte() {
        std::array<double, 256> z{};
        const std::size_t n = x_.size();
        const std::size_t L = b_.max_token_bytes();
        const std::size_t lo = std::max(p_, n + 1 > L ? n + 1 - L : std::size_t{0});
        for (std::size_t i = lo; i <= n; ++i) {
            double lw = 0;  // log weight of encode(x[0:i]) relative to the committed tokens
            if (!open_path(i, lw)) continue;
            const std::string_view rest = std::string_view(x_).substr(i, n - i);
            const auto [a, e] = b_.prefix_range(rest);
            const double w = std::exp(lw);
            for (std::size_t k = a; k < e; ++k) {
                const int v = b_.sorted_at(k);
                const std::string_view tb = b_.tokenizer().token(v);
                if (tb.size() <= rest.size()) continue;
                z[static_cast<unsigned char>(tb[rest.size()])] += w * q_[static_cast<std::size_t>(v)];
            }
        }
        double s = 0;
        for (double v : z) s += v;
        if (s > 0)
            for (double& v : z) v /= s;
        return z;
    }

    // log2 P(bytes() as a prefix): the sum over covers (exact marginal over encodings).
    double log2_prefix() {
        const std::size_t n = x_.size();
        if (n == 0) return 0;
        // Every byte committed: the last committed token ends at n and is the only cover.
        if (p_ == n) return lpC_ / std::log(2.0);
        const std::size_t L = b_.max_token_bytes();
        const std::size_t lo = std::max(p_, n > L ? n - L : std::size_t{0});
        double z = 0;
        for (std::size_t i = lo; i < n; ++i) {
            double lw = 0;
            if (!open_path(i, lw)) continue;
            const std::string_view rest = std::string_view(x_).substr(i, n - i);
            const auto [a, e] = b_.prefix_range(rest);
            const double w = std::exp(lw);
            for (std::size_t k = a; k < e; ++k) z += w * q_[static_cast<std::size_t>(b_.sorted_at(k))];
        }
        if (!(z > 0)) return -std::numeric_limits<double>::infinity();
        return (lpC_ + std::log(z)) / std::log(2.0);
    }

    // log2 P(encode(bytes()) as a token prefix): the canonical segmentation alone.
    double log2_canonical() {
        const std::size_t n = x_.size();
        const std::vector<int> open = b_.tokenizer().encode(std::string_view(x_).substr(p_, n - p_));
        std::vector<int> h(C_);
        double lp = lpC_;
        std::size_t end = p_;
        for (int t : open) {
            b_.conditional(h, std::string_view(x_).substr(0, end), q_, hs_);
            lp += std::log(q_[static_cast<std::size_t>(t)]);
            h.push_back(t);
            end += b_.tokenizer().token(t).size();
        }
        return lp / std::log(2.0);
    }

    // The covers of bytes() that reach past its end, with their weights (relative to the
    // committed tokens): each is (split i, token v). Used to heal a prompt before sampling.
    struct Cover {
        std::size_t split;
        int token;
        double weight;
    };
    std::vector<Cover> covers_past_end() {
        std::vector<Cover> out;
        const std::size_t n = x_.size();
        const std::size_t L = b_.max_token_bytes();
        const std::size_t lo = std::max(p_, n + 1 > L ? n + 1 - L : std::size_t{0});
        for (std::size_t i = lo; i <= n; ++i) {
            double lw = 0;
            if (!open_path(i, lw)) continue;
            const std::string_view rest = std::string_view(x_).substr(i, n - i);
            const auto [a, e] = b_.prefix_range(rest);
            const double w = std::exp(lw);
            for (std::size_t k = a; k < e; ++k) {
                const int v = b_.sorted_at(k);
                if (b_.tokenizer().token(v).size() <= rest.size()) continue;
                const double c = w * q_[static_cast<std::size_t>(v)];
                if (c > 0) out.push_back({i, v, c});
            }
        }
        return out;
    }

    // The canonical tokens of x[0:i] for a split i >= stable(): committed ++ encode(x[p:i]).
    std::vector<int> split_tokens(std::size_t i) const {
        std::vector<int> h(C_);
        const std::vector<int> open = b_.tokenizer().encode(std::string_view(x_).substr(p_, i - p_));
        h.insert(h.end(), open.begin(), open.end());
        return h;
    }

private:
    // Fills q_ with the conditional after encode(x[0:i]) and lw with log P(encode(x[p:i]) | C).
    // Returns false when that path has probability 0.
    bool open_path(std::size_t i, double& lw) {
        const std::vector<int> open = b_.tokenizer().encode(std::string_view(x_).substr(p_, i - p_));
        hist_.assign(C_.begin(), C_.end());
        std::size_t end = p_;
        lw = 0;
        for (int t : open) {
            b_.conditional(hist_, std::string_view(x_).substr(0, end), q_, hs_);
            const double pt = q_[static_cast<std::size_t>(t)];
            if (!(pt > 0)) return false;
            lw += std::log(pt);
            hist_.push_back(t);
            end += b_.tokenizer().token(t).size();
        }
        b_.conditional(hist_, std::string_view(x_).substr(0, i), q_, hs_);
        return true;
    }

    Bridge& b_;
    std::string x_;
    std::size_t p_ = 0;
    std::size_t hs_ = 0;
    std::vector<int> C_;
    double lpC_ = 0;
    std::vector<double> q_;
    std::vector<int> hist_;
};

// Convenience: the exact next-byte distribution after x, from scratch.
inline std::array<double, 256> next_byte(Bridge& b, std::string_view x) {
    Stream s(b);
    s.push(x);
    return s.next_byte();
}

// ---------------------------------------------------------------------------------------------
// Sampling whole tokens after a byte prompt (TOKENS_PLAN T4)
// ---------------------------------------------------------------------------------------------

// Draws a continuation of `prompt` token by token. The prompt is healed first: one cover of the
// prompt that reaches past its end is drawn by weight, so a prompt that ends inside a word
// continues as the token model would have written it. Then whole tokens are drawn from the
// renormalised model at temperature T. Returns the bytes after the prompt (at least max_bytes
// unless the model stops).
inline std::string sample_after(Bridge& b, std::string_view prompt, std::size_t max_bytes, double T,
                                std::mt19937_64& rng) {
    Stream s(b);
    s.push(prompt);
    const std::vector<Stream::Cover> cov = s.covers_past_end();
    if (cov.empty()) return {};
    double tot = 0;
    for (const auto& c : cov) tot += c.weight;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng) * tot;
    std::size_t pick = cov.size() - 1;
    for (std::size_t k = 0; k < cov.size(); ++k) {
        r -= cov[k].weight;
        if (r <= 0) {
            pick = k;
            break;
        }
    }
    std::vector<int> h = s.split_tokens(cov[pick].split);
    std::string text(prompt.substr(0, cov[pick].split));
    text.append(b.tokenizer().token(cov[pick].token));
    h.push_back(cov[pick].token);
    std::size_t hint = s.stable_hint();
    std::vector<double> q(static_cast<std::size_t>(b.vocab_size()));
    while (text.size() < prompt.size() + max_bytes) {
        hint = b.tokenizer().stable_prefix(text, hint);
        b.conditional(h, text, q, hint);
        double z = 0;
        for (double& x : q) {
            if (x > 0 && T != 1.0) x = std::pow(x, 1.0 / T);
            z += x;
        }
        if (!(z > 0)) break;
        double u = U(rng) * z;
        int v = -1;
        for (int k = 0; k < b.vocab_size(); ++k) {
            if (!(q[static_cast<std::size_t>(k)] > 0)) continue;
            v = k;  // the last token with probability, in case rounding leaves u above 0
            u -= q[static_cast<std::size_t>(k)];
            if (u <= 0) break;
        }
        h.push_back(v);
        text.append(b.tokenizer().token(v));
    }
    return text.substr(prompt.size());
}

// ---------------------------------------------------------------------------------------------
// Checking a tokenizer against the contract the bridge relies on
// ---------------------------------------------------------------------------------------------

// For each sample s and each cut c: S1 (encode(s) = encode(s[0:p]) ++ encode(s[p:]) with
// p = stable_prefix(s[0:c])), S2 (every prefix of s[0:c] past p), and prefix closure (every
// prefix of encode(s) re-encodes to itself: validity is a property of prefixes). Returns the
// number of violations; the first is described in *why.
inline int check_contract(const Tokenizer& tok, std::span<const std::string> samples, std::string* why = nullptr) {
    int bad = 0;
    auto fail = [&](const std::string& m) {
        if (bad++ == 0 && why) *why = m;
    };
    auto cat = [](std::vector<int> a, const std::vector<int>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    for (const std::string& s : samples) {
        const std::string_view sv(s);
        const std::vector<int> full = tok.encode(sv);
        std::string dec;
        for (int t : full) dec.append(tok.token(t));
        if (dec != s) fail("decode(encode(s)) != s for \"" + s + "\"");
        for (std::size_t k = 1; k <= full.size(); ++k) {
            std::string pre;
            for (std::size_t j = 0; j < k; ++j) pre.append(tok.token(full[j]));
            const std::vector<int> re = tok.encode(pre);
            if (!std::equal(re.begin(), re.end(), full.begin(), full.begin() + static_cast<std::ptrdiff_t>(k)) || re.size() != k)
                fail("prefix closure fails at token " + std::to_string(k) + " of \"" + s + "\"");
        }
        for (std::size_t c = 0; c <= s.size(); ++c) {
            const std::string_view head = sv.substr(0, c);
            const std::size_t p = tok.stable_prefix(head, 0);
            if (p > c) {
                fail("stable_prefix beyond the string");
                continue;
            }
            const std::vector<int> base = tok.encode(head.substr(0, p));
            if (cat(base, tok.encode(sv.substr(p))) != full)
                fail("S1 fails: \"" + s + "\" cut " + std::to_string(c) + " p " + std::to_string(p));
            for (std::size_t i = p; i <= c; ++i)
                if (cat(base, tok.encode(head.substr(p, i - p))) != tok.encode(head.substr(0, i)))
                    fail("S2 fails: \"" + s + "\" cut " + std::to_string(c) + " i " + std::to_string(i));
            if (c > 0) {
                const std::size_t p2 = tok.stable_prefix(head, p);  // the hint must not change the answer's validity
                if (p2 > c || cat(tok.encode(head.substr(0, p2)), tok.encode(sv.substr(p2))) != full)
                    fail("S1 fails with a hint: \"" + s + "\" cut " + std::to_string(c));
            }
        }
    }
    return bad;
}

}  // namespace cypha::cyphalm::bridge
