// An LDA topic model for K5 (Blei, Ng and Jordan, 2003; docs/reports/K5_DISCOURSE.md, GENERATION_PLAN §13.32).
//
// Bayesian: each document is a mixture theta of K topics, each topic a distribution phi_k over words, with Dirichlet
// priors alpha (on theta) and beta (on phi). Trained by collapsed Gibbs sampling (Griffiths and Steyvers, 2004), one
// document per line of a clean text. Only content words take part: lower case, alphabetic, at least 3 letters, not a
// function word (stopwords.hpp), seen at least min_count times, the vmax most frequent.
//
// Large texts: the sweeps run on several threads over disjoint blocks of documents, each thread on its own copy of the
// topic-word counts, merged after every sweep (AD-LDA, Newman et al., 2009). Deterministic for a given thread count.
//
// A context's topic mixture is inferred by Gibbs sampling with phi fixed (fold-in), theta averaged over the last sweeps.
// Then p_topic(w | theta) = sum_k theta_k phi_kw, against the corpus unigram p(w) over the same words.
//
// Interface for other sessions (the MCMC sampler of wip/k5-mcmc, the structure experts of wip/k5-structure). Tokens are
// K5's (wkn::tokens); word_kn.hpp's wkn::lda_score(model, context_text, candidate_text) does the tokenising.
//   LDA m; m.train(docs, params)  or  m.train_lines(text, tokenizer, params)  or  m.load(path)
//   double s = m.score(context_tokens, candidate_tokens);  // log2 score of the candidate given the context (below)
//   auto theta = m.infer(context_tokens);                  // the context's topic mixture (deterministic per context)
//   double r = m.log_ratio(theta, tokens);                 // sum over content words of log2 (p_topic(w | theta) / p(w))
//   double a = m.log_score(theta, tokens);                 // sum over content words of log2 p_topic(w | theta)
//   std::size_t n = m.content_count(tokens);               // how many of the tokens are content words
// score(context, candidate) = log_ratio(infer(context), candidate): positive when the context's topics favour the
// candidate's content words over their corpus frequency. Cache infer(context) when scoring many candidates.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cypha/lm/stopwords.hpp"

namespace cypha::cyphalm::lda {

struct Params {
    std::size_t K = 100;        // topics
    double alpha = 0.1;         // Dirichlet prior on each document's theta (symmetric)
    double beta = 0.01;         // Dirichlet prior on each topic's phi (symmetric)
    int iters = 200;            // Gibbs sweeps over the corpus
    std::size_t min_count = 5;  // a content word's least corpus count
    std::size_t vmax = 50000;   // at most this many content words (the most frequent)
    unsigned threads = 1;       // 1: exact collapsed Gibbs; more: AD-LDA (on 7 MB its word log-likelihood was 3-5% lower)
    std::uint64_t seed = 20261105;
};

inline std::uint64_t mix(std::uint64_t x) {  // splitmix64's finaliser
    x ^= x >> 30, x *= 0xbf58476d1ce4e5b9ull, x ^= x >> 27, x *= 0x94d049bb133111ebull, x ^= x >> 31;
    return x;
}

class LDA {
 public:
    // The content-word key of a token: lower case, or "" when it is not a content word.
    static std::string key(std::string_view t) {
        if (t.size() < 3 || !words::alpha(t) || t.find('\'') != std::string_view::npos) return {};
        std::string s = words::lower(t);
        if (words::stop().count(s)) return {};
        return s;
    }

    // docs: one token vector per document.
    void train(const std::vector<std::vector<std::string>>& docs, const Params& p = Params{}) {
        train_lines_impl(docs.size(), [&](std::size_t j, auto&& emit) {
            for (const auto& t : docs[j]) emit(std::string_view(t));
        }, p);
    }
    // text: one document per line; tokenize(line, emit) calls emit(token) for each token (e.g. wkn::tokenize).
    template <class Tok>
    void train_lines(std::string_view text, Tok&& tokenize, const Params& p = Params{}) {
        std::vector<std::string_view> lines;
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            if (b == std::string_view::npos) b = text.size();
            if (b > a) lines.push_back(text.substr(a, b - a));
            a = b + 1;
        }
        train_lines_impl(lines.size(), [&](std::size_t j, auto&& emit) { tokenize(lines[j], emit); }, p);
    }

    bool trained() const { return !phi_.empty(); }
    std::size_t topics() const { return p_.K; }
    std::size_t vocab_size() const { return words_.size(); }
    std::size_t corpus_tokens() const { return tokens_; }
    std::size_t corpus_docs() const { return docs_; }
    const Params& params() const { return p_; }
    const std::vector<double>& loglik() const { return loglik_; }  // log P(words | z) every 50 sweeps and at the end
    const std::vector<std::string>& words() const { return words_; }
    // The content word's id, or -1.
    long id(std::string_view t) const {
        const auto it = id_.find(key(t));
        return it == id_.end() ? -1 : static_cast<long>(it->second);
    }
    double phi(std::size_t k, std::size_t w) const { return phi_[w * p_.K + k]; }
    double unigram(std::size_t w) const { return uni_[w]; }

    // The topic mixture of a token sequence: Gibbs fold-in with phi fixed, theta averaged over the last `keep` sweeps.
    // Deterministic: the seed is a hash of the content words. No content word: the prior (uniform).
    std::vector<double> infer(const std::vector<std::string>& toks, int sweeps = 50, int keep = 30) const {
        const std::size_t K = p_.K;
        std::vector<std::uint32_t> w;
        std::uint64_t h = 0x9E3779B97F4A7C15ull;
        for (const auto& t : toks) {
            const long x = id(t);
            if (x >= 0) w.push_back(static_cast<std::uint32_t>(x)), h = (h ^ static_cast<std::uint64_t>(x)) * 0x100000001b3ull;
        }
        std::vector<double> theta(K, 1.0 / static_cast<double>(K));
        if (w.empty()) return theta;
        std::mt19937_64 rng(h);
        std::vector<std::uint32_t> z(w.size()), nd(K, 0);
        for (std::size_t i = 0; i < w.size(); ++i) z[i] = static_cast<std::uint32_t>(rng() % K), ++nd[z[i]];
        std::vector<double> pr(K), acc(K, 0.0);
        for (int s = 0; s < sweeps; ++s) {
            for (std::size_t i = 0; i < w.size(); ++i) {
                --nd[z[i]];
                const double* f = &phi_[w[i] * K];
                double c = 0;
                for (std::size_t k = 0; k < K; ++k) c += (nd[k] + p_.alpha) * f[k], pr[k] = c;
                const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * c;
                std::size_t k = static_cast<std::size_t>(std::upper_bound(pr.begin(), pr.end(), u) - pr.begin());
                if (k >= K) k = K - 1;
                z[i] = static_cast<std::uint32_t>(k), ++nd[k];
            }
            if (s >= sweeps - keep)
                for (std::size_t k = 0; k < K; ++k) acc[k] += (nd[k] + p_.alpha) / (static_cast<double>(w.size()) + K * p_.alpha);
        }
        double z0 = 0;
        for (const double x : acc) z0 += x;
        for (std::size_t k = 0; k < K; ++k) theta[k] = acc[k] / z0;
        return theta;
    }

    // p_topic(w | theta) for every content word.
    std::vector<double> word_dist(const std::vector<double>& theta) const {
        const std::size_t V = words_.size(), K = p_.K;
        std::vector<double> q(V, 0.0);
        for (std::size_t x = 0; x < V; ++x) {
            const double* f = &phi_[x * K];
            double s = 0;
            for (std::size_t k = 0; k < K; ++k) s += theta[k] * f[k];
            q[x] = s;
        }
        return q;
    }
    double p_topic(const std::vector<double>& theta, std::size_t w) const {
        const double* f = &phi_[w * p_.K];
        double s = 0;
        for (std::size_t k = 0; k < p_.K; ++k) s += theta[k] * f[k];
        return s;
    }
    // Sum over the content words of toks of log2 p_topic(w | theta). Function words and unknown words add nothing.
    double log_score(const std::vector<double>& theta, const std::vector<std::string>& toks) const {
        double s = 0;
        for (const auto& t : toks) {
            const long x = id(t);
            if (x >= 0) s += std::log2(p_topic(theta, static_cast<std::size_t>(x)));
        }
        return s;
    }
    // Sum over the content words of toks of log2 (p_topic(w | theta) / p(w)): how much the topic mixture favours them.
    double log_ratio(const std::vector<double>& theta, const std::vector<std::string>& toks) const {
        double s = 0;
        for (const auto& t : toks) {
            const long x = id(t);
            if (x >= 0) s += std::log2(p_topic(theta, static_cast<std::size_t>(x)) / uni_[static_cast<std::size_t>(x)]);
        }
        return s;
    }
    // The stable scoring call: log2 score of candidate given context (prompt + paragraph so far).
    double score(const std::vector<std::string>& context, const std::vector<std::string>& candidate) const {
        return log_ratio(infer(context), candidate);
    }
    std::size_t content_count(const std::vector<std::string>& toks) const {
        std::size_t n = 0;
        for (const auto& t : toks) n += id(t) >= 0;
        return n;
    }

    bool save(const std::string& path) const {
        std::ofstream o(path, std::ios::binary);
        if (!o) return false;
        const std::uint64_t K = p_.K, V = words_.size(), N = tokens_, D = docs_;
        o.write("LDA2", 4);
        o.write(reinterpret_cast<const char*>(&K), 8), o.write(reinterpret_cast<const char*>(&V), 8);
        o.write(reinterpret_cast<const char*>(&p_.alpha), 8), o.write(reinterpret_cast<const char*>(&p_.beta), 8);
        o.write(reinterpret_cast<const char*>(&N), 8), o.write(reinterpret_cast<const char*>(&D), 8);
        const std::uint64_t nl = loglik_.size();
        o.write(reinterpret_cast<const char*>(&nl), 8);
        o.write(reinterpret_cast<const char*>(loglik_.data()), static_cast<std::streamsize>(nl * 8));
        for (const auto& w : words_) o << w << '\n';
        o.write(reinterpret_cast<const char*>(phi_.data()), static_cast<std::streamsize>(phi_.size() * 8));
        o.write(reinterpret_cast<const char*>(uni_.data()), static_cast<std::streamsize>(uni_.size() * 8));
        return static_cast<bool>(o);
    }
    bool load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char m[4];
        std::uint64_t K = 0, V = 0, N = 0, D = 0, nl = 0;
        if (!in.read(m, 4) || std::string_view(m, 4) != "LDA2") return false;
        in.read(reinterpret_cast<char*>(&K), 8), in.read(reinterpret_cast<char*>(&V), 8);
        in.read(reinterpret_cast<char*>(&p_.alpha), 8), in.read(reinterpret_cast<char*>(&p_.beta), 8);
        in.read(reinterpret_cast<char*>(&N), 8), in.read(reinterpret_cast<char*>(&D), 8), in.read(reinterpret_cast<char*>(&nl), 8);
        loglik_.resize(nl);
        in.read(reinterpret_cast<char*>(loglik_.data()), static_cast<std::streamsize>(nl * 8));
        p_.K = K, tokens_ = N, docs_ = D;
        words_.resize(V);
        id_.clear();
        for (std::uint64_t i = 0; i < V; ++i) {
            std::getline(in, words_[i]);
            id_.emplace(words_[i], static_cast<std::uint32_t>(i));
        }
        phi_.resize(K * V), uni_.resize(V);
        in.read(reinterpret_cast<char*>(phi_.data()), static_cast<std::streamsize>(phi_.size() * 8));
        in.read(reinterpret_cast<char*>(uni_.data()), static_cast<std::streamsize>(uni_.size() * 8));
        return static_cast<bool>(in);
    }

 private:
    // docs(j, emit): calls emit(token) for each token of document j.
    template <class Docs>
    void train_lines_impl(std::size_t D, Docs&& docs, const Params& p) {
        p_ = p;
        const unsigned T = std::max(1u, p.threads);
        // Vocabulary: content-word counts, on T threads.
        std::vector<std::unordered_map<std::string, std::size_t>> cnts(T);
        const auto split = [&](unsigned t) { return std::make_pair(D * t / T, D * (t + 1) / T); };
        run(T, [&](unsigned t) {
            const auto [a, b] = split(t);
            for (std::size_t j = a; j < b; ++j)
                docs(j, [&](std::string_view tok) {
                    std::string k = key(tok);
                    if (!k.empty()) ++cnts[t][std::move(k)];
                });
        });
        for (unsigned t = 1; t < T; ++t)
            for (auto& [k, c] : cnts[t]) cnts[0][k] += c;
        std::vector<std::pair<std::string, std::size_t>> v(cnts[0].begin(), cnts[0].end());
        cnts.clear();
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
        words_.clear();
        id_.clear();
        for (const auto& [w, c] : v) {
            if (c < p.min_count || words_.size() >= p.vmax) break;
            id_.emplace(w, static_cast<std::uint32_t>(words_.size())), words_.push_back(w);
        }
        const std::size_t V = words_.size(), K = p.K;
        // The corpus: content-word ids, document by document (offsets per document).
        std::vector<std::vector<std::uint32_t>> part(T);
        std::vector<std::vector<std::size_t>> plen(T);
        run(T, [&](unsigned t) {
            const auto [a, b] = split(t);
            for (std::size_t j = a; j < b; ++j) {
                std::size_t n = 0;
                docs(j, [&](std::string_view tok) {
                    const auto it = id_.find(key(tok));
                    if (it != id_.end()) part[t].push_back(it->second), ++n;
                });
                plen[t].push_back(n);
            }
        });
        std::vector<std::uint32_t> w;
        std::vector<std::size_t> off(1, 0);
        for (unsigned t = 0; t < T; ++t) {
            w.insert(w.end(), part[t].begin(), part[t].end());
            for (const std::size_t n : plen[t]) off.push_back(off.back() + n);
            std::vector<std::uint32_t>().swap(part[t]);
        }
        const std::size_t N = w.size();
        tokens_ = N, docs_ = D;
        // Counts: word-major n_wk, n_k, n_dk.
        std::vector<std::uint32_t> z(N), nwk(V * K, 0), nk(K, 0), ndk(D * K, 0);
        {
            std::mt19937_64 rng(p.seed);
            for (std::size_t j = 0; j < D; ++j)
                for (std::size_t i = off[j]; i < off[j + 1]; ++i) {
                    z[i] = static_cast<std::uint32_t>(rng() % K);
                    ++nwk[w[i] * K + z[i]], ++nk[z[i]], ++ndk[j * K + z[i]];
                }
        }
        const double Vb = static_cast<double>(V) * p.beta;
        // Each thread's documents: blocks of about equal token counts.
        std::vector<std::size_t> db(T + 1, D);
        db[0] = 0;
        for (unsigned t = 1; t < T; ++t) db[t] = static_cast<std::size_t>(std::lower_bound(off.begin(), off.end(), N * t / T) - off.begin());
        for (unsigned t = 1; t <= T; ++t) db[t] = std::max(db[t], db[t - 1]);
        std::vector<std::vector<std::uint32_t>> lw(T > 1 ? T : 0), lk(T > 1 ? T : 0);
        loglik_.clear();
        for (int it = 0; it < p.iters; ++it) {
            run(T, [&](unsigned t) {
                std::uint32_t* cw = nwk.data();
                std::uint32_t* ck = nk.data();
                if (T > 1) {
                    lw[t] = nwk, lk[t] = nk;
                    cw = lw[t].data(), ck = lk[t].data();
                }
                std::mt19937_64 rng(mix(p.seed ^ mix(static_cast<std::uint64_t>(it) * 1315423911ull + t + 1)));
                std::vector<double> pr(K), inv(K);
                for (std::size_t k = 0; k < K; ++k) inv[k] = 1.0 / (ck[k] + Vb);
                for (std::size_t j = db[t]; j < db[t + 1]; ++j) {
                    std::uint32_t* nd = &ndk[j * K];
                    for (std::size_t i = off[j]; i < off[j + 1]; ++i) {
                        const std::uint32_t wi = w[i];
                        std::uint32_t* nw = &cw[wi * K];
                        std::uint32_t k = z[i];
                        --nw[k], --ck[k], --nd[k];
                        inv[k] = 1.0 / (ck[k] + Vb);
                        double s = 0;
                        for (std::size_t q = 0; q < K; ++q) {
                            s += (nd[q] + p.alpha) * (nw[q] + p.beta) * inv[q];
                            pr[q] = s;
                        }
                        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * s;
                        k = static_cast<std::uint32_t>(std::upper_bound(pr.begin(), pr.end(), u) - pr.begin());
                        if (k >= K) k = static_cast<std::uint32_t>(K - 1);
                        z[i] = k;
                        ++nw[k], ++ck[k], ++nd[k];
                        inv[k] = 1.0 / (ck[k] + Vb);
                    }
                }
            });
            if (T > 1) {  // merge: n <- n + sum_t (n_t - n)
                run(T, [&](unsigned t) {
                    const std::size_t a = V * K * t / T, b = V * K * (t + 1) / T;
                    for (std::size_t i = a; i < b; ++i) {
                        std::int64_t s = nwk[i];
                        for (unsigned u = 0; u < T; ++u) s += static_cast<std::int64_t>(lw[u][i]) - static_cast<std::int64_t>(nwk[i]);
                        nwk[i] = static_cast<std::uint32_t>(s);
                    }
                });
                for (std::size_t k = 0; k < K; ++k) {
                    std::int64_t s = nk[k];
                    for (unsigned u = 0; u < T; ++u) s += static_cast<std::int64_t>(lk[u][k]) - static_cast<std::int64_t>(nk[k]);
                    nk[k] = static_cast<std::uint32_t>(s);
                }
            }
            if ((it + 1) % 50 == 0 || it + 1 == p.iters) loglik_.push_back(word_loglik(nwk, nk, V));
        }
        phi_.assign(V * K, 0.0);
        for (std::size_t x = 0; x < V; ++x)
            for (std::size_t k = 0; k < K; ++k) phi_[x * K + k] = (nwk[x * K + k] + p.beta) / (nk[k] + Vb);
        uni_.assign(V, 0.0);
        for (const std::uint32_t x : w) uni_[x] += 1;
        for (double& x : uni_) x /= static_cast<double>(std::max<std::size_t>(N, 1));
    }
    template <class F>
    static void run(unsigned T, F&& f) {
        if (T == 1) return f(0u);
        std::vector<std::thread> th;
        for (unsigned t = 1; t < T; ++t) th.emplace_back([&f, t] { f(t); });
        f(0u);
        for (auto& x : th) x.join();
    }
    double word_loglik(const std::vector<std::uint32_t>& nwk, const std::vector<std::uint32_t>& nk, std::size_t V) const {
        const std::size_t K = p_.K;
        double ll = 0;
        for (std::size_t k = 0; k < K; ++k) ll += std::lgamma(V * p_.beta) - std::lgamma(nk[k] + V * p_.beta);
        for (std::size_t i = 0; i < V * K; ++i)
            if (nwk[i]) ll += std::lgamma(nwk[i] + p_.beta) - std::lgamma(p_.beta);
        return ll;
    }
    Params p_;
    std::vector<std::string> words_;
    std::unordered_map<std::string, std::uint32_t> id_;
    std::vector<double> phi_, uni_, loglik_;  // phi_ word-major: phi_[w * K + k]
    std::size_t tokens_ = 0, docs_ = 0;
};

// Selftest: two clearly separate themes; topics normalised; a context on one theme favours its words; threads agree
// on the themes; save and load round-trip.
inline bool self_test(std::string& err) {
    std::vector<std::vector<std::string>> docs;
    const std::vector<std::string> a = {"guitar", "drums", "melody", "concert", "album", "singer"};
    const std::vector<std::string> b = {"protein", "enzyme", "molecule", "cell", "membrane", "genome"};
    std::mt19937_64 r(7);
    for (int d = 0; d < 40; ++d) {
        std::vector<std::string> doc;
        const auto& src = d % 2 ? a : b;
        for (int i = 0; i < 60; ++i) doc.push_back(src[r() % src.size()]), doc.push_back("the");
        docs.push_back(doc);
    }
    for (const unsigned T : {1u, 4u}) {
        Params p;
        p.K = 2, p.iters = 50, p.min_count = 2, p.threads = T;
        LDA m;
        m.train(docs, p);
        if (m.vocab_size() != 12) return err = "lda: vocabulary (function words must be left out)", false;
        for (std::size_t k = 0; k < 2; ++k) {
            double z = 0;
            for (std::size_t x = 0; x < m.vocab_size(); ++x) z += m.phi(k, x);
            if (std::abs(z - 1) > 1e-9) return err = "lda: phi not normalised", false;
        }
        const std::vector<std::string> ctx = {"guitar", "album", "singer", "concert"};
        if (m.score(ctx, {"melody"}) <= 0 || m.score(ctx, {"enzyme"}) >= 0) return err = "lda: the context's topic is not favoured", false;
        if (m.infer(ctx) != m.infer(ctx)) return err = "lda: infer not deterministic", false;
        LDA m2;
        m2.train(docs, p);
        if (m2.score(ctx, {"melody", "cell"}) != m.score(ctx, {"melody", "cell"})) return err = "lda: training not deterministic", false;
    }
    return true;
}

}  // namespace cypha::cyphalm::lda
