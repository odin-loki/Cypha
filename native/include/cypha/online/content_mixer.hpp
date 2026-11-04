#pragma once

/// The content mixer (docs/GENERATION_PLAN.md §13.10-13.12; docs/MODULES.md, module online): CyphaLM's next-byte
/// experts mixed into the trie mixture one stage at a time, and the online final layer over every expert and stage.
/// Moved here from tools/lm/cyphalm_trie.cpp unchanged (2026-11-03); cyphalm_trie's context8k, qa, sample, cls and
/// selftest call it.
///
/// A stage: cur[k] is the chain's probability of the true byte data[train + k] of a holdout; an expert gives its own
/// probability pe[k] and a bucket bk[k] in [0, nb). mix_into fits one lambda a bucket on k < half (fit_lambdas, then
/// frozen) and makes cur[k] = lam[bk[k]] pe[k] + (1 - lam[bk[k]]) cur[k]; with g_online_mix the lambdas learn online
/// instead (online_mix). Any expert that gives pe and bk joins the chain through mix_into; mix_bkn_into, mix_cls_into,
/// mix_copy_into and mix_doc_into (fit_doc prints its read) are the chain's own stages. bkn_mix mixes Kneser-Ney into
/// one next-byte distribution, for the samplers.
///
/// The online final layer (OnlineFinal, --online-final): one gated linear mixer over Inputs probabilities of the true
/// byte (the experts and the chain stages, the chain last), a weight set chosen by the caller's context, learned online
/// by exponentiated gradient on log loss. Another expert joins as one more input.

#include "cypha/lm/byte_kn.hpp"
#include "cypha/lm/class_context.hpp"
#include "cypha/lm/copy_expert.hpp"
#include "cypha/lm/doc_cache.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace cypha::online {

// lam[b] in {0, 0.005, ..., 0.995} maximising the log-likelihood of lam pe + (1 - lam) pb over the items k < upto in
// bucket b (context8k's InfiniGram fit).
inline void fit_lambdas(const std::vector<double>& pe, const std::vector<double>& pb, const std::vector<int>& bk,
                        std::size_t upto, int nb, double* lam) {
    for (int b = 0; b < nb; ++b) {
        double best = -1e300;
        lam[b] = 0;
        for (int g = 0; g < 200; ++g) {
            const double l = g * 0.005;
            double ll = 0;
            bool any = false;
            for (std::size_t k = 0; k < upto; ++k)
                if (bk[k] == b) {
                    ll += std::log(l * pe[k] + (1 - l) * pb[k]);
                    any = true;
                }
            if (!any) break;
            if (ll > best) {
                best = ll;
                lam[b] = l;
            }
        }
    }
}

// B4C: one default expert mixed into cur, where cur[k] is a distribution's probability of the true byte data[train + k]:
// the expert's lambdas are fit on k < half over cur (the copy expert's then zeroed below a match of m bytes), and cur
// becomes lam pe + (1 - lam) cur. Returns the lambdas, one a bucket.
// GLNX (docs/GENERATION_PLAN.md §13.12; Odin: "don't freeze the mixer"): with g_online_mix the same chain stage learns
// its lambda per bucket online instead of fitting it on the first half and freezing it. Each byte is predicted with the
// current lambda, then the lambda learns from that byte. cur becomes the online mix.
// §13.15 (Odin: "fix the spike in the learning on the content mixer; it's overfiltered"): the update is exponentiated
// gradient on the pair (lambda, 1 - lambda), step 0.02, the exponent capped at 30, a floor of 1e-4 on each weight.
// AdaGrad was used before; its step shrank by the square root of every squared gradient seen, and a few bytes the mixture
// gave almost nothing to (gradients up to 29,088) set that sum, so a bucket stopped learning after one spike. A spike
// moves exponentiated gradient's weights once and never shrinks later steps (-0.0074 bits per byte on the dump).
// copy_m >= 0 holds the copy stage's short-match buckets at 0 (B4B, for writing); the online path only scores, so
// mix_into passes -1 and the online chain takes short copies too (-0.023 on the dump). The frozen fit mix_into returns
// to answers and samplers keeps B4B's clip.
inline bool g_online_mix = false;
// §13.22 (sample --online-lambdas): mix_into returns the online lambdas' state after the whole holdout, not the frozen
// fit, so the sampler writes with what the online mixer has learned (copy's short matches still held at 0, B4B).
inline bool g_online_lambdas_out = false;
// §13.21 (two learning rates, as PAQ runs mixers at several rates): with g_rate_chains set, every online stage also
// runs a slow (0.005) and a fast (0.08) copy of the chain, each over its own running mixture, so the final layer can
// read all three and pick. The main chain (0.02) is unchanged by them.
inline std::vector<std::vector<double>>* g_rate_chains = nullptr;
inline constexpr double kRateSteps[2] = {0.005, 0.08};

// One stage by exponentiated gradient at step eta over cur (see online_mix); lam gets the last weights when given.
inline void eg_stage(const std::vector<double>& pe, const std::vector<int>& bk, int nb, std::vector<double>& cur,
                     double eta, const std::vector<char>& dead, std::vector<double>* lam) {
    std::vector<double> we(static_cast<std::size_t>(nb), 0.05), wc(static_cast<std::size_t>(nb), 0.95);
    for (std::size_t k = 0; k < cur.size(); ++k) {
        const std::size_t b = static_cast<std::size_t>(bk[k]);
        if (dead[b]) continue;  // lambda 0: cur stays
        const double prev = cur[k], l = we[b], p = l * pe[k] + (1 - l) * prev;
        cur[k] = p;
        const double pp = std::max(p, 1e-300);
        double e = we[b] * std::exp(eta * std::min(pe[k] / pp, 30.0)), c = wc[b] * std::exp(eta * std::min(prev / pp, 30.0));
        double z = e + c;
        e = std::max(e / z, 1e-4);
        c = std::max(c / z, 1e-4);
        z = e + c;
        we[b] = e / z;
        wc[b] = c / z;
    }
    if (lam)
        for (std::size_t b = 0; b < lam->size(); ++b) (*lam)[b] = dead[b] ? 0.0 : we[b];
}

inline std::vector<double> online_mix(const std::vector<double>& pe, const std::vector<int>& bk, int nb,
                                      std::vector<double>& cur, int copy_m = -1) {
    std::vector<char> dead(static_cast<std::size_t>(nb), 0);
    if (copy_m >= 0) {
        std::vector<double> one(static_cast<std::size_t>(nb), 1.0);
        cyphalm::copy_clip(one.data(), copy_m);
        for (std::size_t b = 0; b < one.size(); ++b) dead[b] = one[b] == 0.0;
    }
    std::vector<double> lam(static_cast<std::size_t>(nb));
    eg_stage(pe, bk, nb, cur, 0.02, dead, &lam);
    if (g_rate_chains)
        for (std::size_t r = 0; r < g_rate_chains->size() && r < 2; ++r) eg_stage(pe, bk, nb, (*g_rate_chains)[r], kRateSteps[r], dead, nullptr);
    return lam;
}
inline std::vector<double> mix_into(const std::vector<double>& pe, const std::vector<int>& bk, std::size_t half, int nb,
                                    std::vector<double>& cur, int copy_m = -1) {
    std::vector<double> lam(static_cast<std::size_t>(nb), 0.0);
    if (g_online_mix) {  // the frozen fit is still returned, so answers and samplers that read it are unchanged
        fit_lambdas(pe, cur, bk, half, nb, lam.data());
        if (copy_m >= 0) cyphalm::copy_clip(lam.data(), copy_m);
        std::vector<double> on = online_mix(pe, bk, nb, cur);  // §13.15: the online path takes short copies
        if (g_online_lambdas_out) {
            if (copy_m >= 0) cyphalm::copy_clip(on.data(), copy_m);
            return on;
        }
        return lam;
    }
    fit_lambdas(pe, cur, bk, half, nb, lam.data());
    if (copy_m >= 0) cyphalm::copy_clip(lam.data(), copy_m);
    for (std::size_t k = 0; k < cur.size(); ++k) {
        const double l = lam[static_cast<std::size_t>(bk[k])];
        cur[k] = l * pe[k] + (1 - l) * cur[k];
    }
    return lam;
}
inline std::vector<double> mix_cls_into(const cyphalm::ClassContext& cc, const std::uint8_t* data, std::size_t train,
                                        std::size_t half, std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) {
        double P[256];
        bk[k] = cyphalm::ClassContext::bucket(cc.predict(data, train + k, P));
        pe[k] = P[data[train + k]];
    }
    return mix_into(pe, bk, half, cyphalm::ClassContext::kBuckets, cur);
}
inline std::vector<double> mix_copy_into(const std::uint8_t* data, std::size_t train, std::size_t half, int m,
                                         std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) {
        int n = 0;
        pe[k] = cyphalm::copy_pred(data, train + k, n) == data[train + k] ? 1.0 : 0.0;
        bk[k] = cyphalm::copy_bucket(n);
    }
    return mix_into(pe, bk, half, cyphalm::kCopyBuckets, cur, m);
}
// BKN into cur (the trie mixture's probability of each holdout byte): (a) cur becomes Kneser-Ney's; (b) Kneser-Ney is
// mixed in by lambda per (deepest order found x count band), fit on k < half. Returns (b)'s lambdas (empty for (a)).
inline std::vector<double> mix_bkn_into(const cyphalm::bkn::ByteKN& kn, const std::uint8_t* data, std::size_t train,
                                        std::size_t half, bool replace, std::vector<double>& cur) {
    std::vector<double> pe(cur.size());
    std::vector<int> bk(cur.size());
    for (std::size_t k = 0; k < cur.size(); ++k) pe[k] = kn.prob_bucket(data, train + k, data[train + k], bk[k]);
    if (replace) {
        if (g_rate_chains)
            for (auto& c : *g_rate_chains) c = pe;  // §13.21: the slow and fast chains start where the main one does
        cur.swap(pe);
        return {};
    }
    return mix_into(pe, bk, half, kn.buckets(), cur);
}
// BKN into a next-byte distribution P (the trie mixture's), as mix_bkn_into fit it.
inline void bkn_mix(const cyphalm::bkn::ByteKN& kn, const double* lkn, bool replace, const std::uint8_t* seq,
                    std::size_t len, double* P) {
    double Pk[256];
    int band = 0;
    const int d = kn.dist(seq, len, Pk, &band);
    if (replace) {
        std::memcpy(P, Pk, sizeof Pk);
        return;
    }
    const double l = lkn[d == 0 ? 0 : (d - 1) * 3 + band];
    for (int s = 0; s < 256; ++s) P[s] = l * Pk[s] + (1 - l) * P[s];
}
// GEN-C: each part's next-byte distribution over the holdout (one stream from the train end, reset at "<page>"),
// its lambdas fit on k < half over cur, cur mixed with it, in DocTopic's part order. The read: bits a byte on
// [half, cur.size()) before and after, with a bootstrap over 16 KiB blocks, and split by the byte's offset in its
// document (under and over 4 KiB). Printed to stderr with a DOC prefix.
inline std::vector<std::vector<double>> fit_doc(const cyphalm::DocTopic& dt, const std::uint8_t* data, std::size_t train,
                                                std::size_t half, std::vector<double>& cur) {
    const std::size_t span = cur.size();
    const auto& p = dt.params();
    const bool on[cyphalm::DocTopic::kParts] = {p.trig, p.cache, p.topic};
    std::vector<double> pe[cyphalm::DocTopic::kParts];
    std::vector<int> bk[cyphalm::DocTopic::kParts];
    for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
        if (on[q]) pe[q].resize(span), bk[q].resize(span);
    std::vector<std::uint32_t> off(span);
    cyphalm::DocTopic::Stream st(dt);
    std::size_t page = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < span; ++k) {
        const std::size_t i = train + k;
        for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                bk[q][k] = st.predict(q, P);
                pe[q][k] = P[data[i]];
            }
        if (k >= 6 && std::memcmp(data + i - 6, "<page>", 6) == 0) page = k - 6;
        off[k] = static_cast<std::uint32_t>(std::min<std::size_t>(k - page, 0xffffffffu));
        st.feed(data[i]);
    }
    const std::vector<double> before = cur;
    std::vector<std::vector<double>> lam(cyphalm::DocTopic::kParts);
    for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
        if (on[q]) lam[static_cast<std::size_t>(q)] = mix_into(pe[q], bk[q], half, cyphalm::DocTopic::kBuckets, cur);
    static const char* const name[cyphalm::DocTopic::kParts] = {"trig", "cache", "topic"};
    for (int q = 0; q < cyphalm::DocTopic::kParts; ++q) {
        if (!on[q]) continue;
        std::cerr << "DOC lambdas " << name[q];
        for (double l : lam[static_cast<std::size_t>(q)]) std::cerr << " " << l;
        std::cerr << "\n";
    }
    if (span > half) {
        const std::size_t B = 16384;
        std::vector<double> diff;  // mean bits a byte, after minus before, a block
        double sb = 0, sa = 0, lb[2] = {0, 0}, la[2] = {0, 0}, ln[2] = {0, 0};
        for (std::size_t s0 = half; s0 < span; s0 += B) {
            const std::size_t s1 = std::min(span, s0 + B);
            double d = 0;
            for (std::size_t k = s0; k < s1; ++k) {
                const double b0 = -std::log2(before[k]), b1 = -std::log2(cur[k]);
                sb += b0, sa += b1, d += b1 - b0;
                const int g = off[k] >= 4096;
                lb[g] += b0, la[g] += b1, ln[g] += 1;
            }
            diff.push_back(d / static_cast<double>(s1 - s0));
        }
        const double n = static_cast<double>(span - half);
        std::mt19937_64 rng(20261029);
        std::vector<double> bs(2000);
        for (double& x : bs) {
            double m = 0;
            for (std::size_t j = 0; j < diff.size(); ++j) m += diff[rng() % diff.size()];
            x = m / static_cast<double>(diff.size());
        }
        std::sort(bs.begin(), bs.end());
        double mean = 0;
        for (double d : diff) mean += d;
        mean /= static_cast<double>(diff.size());
        std::fprintf(stderr, "DOC read [half, end) bytes %.0f bpc before %.6f after %.6f diff %.6f ci95 %.6f %.6f blocks %zu\n",
                     n, sb / n, sa / n, mean, bs[50], bs[1949], diff.size());
        for (int g = 0; g < 2; ++g)
            if (ln[g] > 0)
                std::fprintf(stderr, "DOC read offset %s 4KiB bytes %.0f bpc before %.6f after %.6f diff %.6f\n",
                             g ? ">=" : "<", ln[g], lb[g] / ln[g], la[g] / ln[g], (la[g] - lb[g]) / ln[g]);
    }
    std::cerr << "DOC fit_s " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return lam;
}

// GEN-C2: fit_doc's mixture in less memory (context8k, qa's holdout). One stream from the train end; each part's
// predictions are kept over the first half only, where its lambdas are fit (over cur, in DocTopic's part order) and it
// is mixed; the second half is mixed as the stream reads it. The same numbers as fit_doc; no DOC lines.
// lam_out (FULL_DISTRIBUTION_MIXING.md, context8k --full-mix): when given, gets the fitted lambdas, [part * kBuckets + bucket].
inline void mix_doc_into(const cyphalm::DocTopic& dt, const std::uint8_t* data, std::size_t train, std::size_t half,
                         std::vector<double>& cur, std::vector<double>* lam_out = nullptr) {
    const std::size_t span = cur.size(), h = std::min(half, span);
    const auto& p = dt.params();
    const bool on[cyphalm::DocTopic::kParts] = {p.trig, p.cache, p.topic};
    std::vector<double> pe[cyphalm::DocTopic::kParts];
    std::vector<int> bk[cyphalm::DocTopic::kParts];
    for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
        if (on[q]) pe[q].resize(h), bk[q].resize(h);
    cyphalm::DocTopic::Stream st(dt);
    for (std::size_t k = 0; k < h; ++k) {
        for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                bk[q][k] = st.predict(q, P);
                pe[q][k] = P[data[train + k]];
            }
        st.feed(data[train + k]);
    }
    double lam[cyphalm::DocTopic::kParts][cyphalm::DocTopic::kBuckets] = {};
    for (int q = 0; q < cyphalm::DocTopic::kParts; ++q) {
        if (!on[q]) continue;
        fit_lambdas(pe[q], cur, bk[q], h, cyphalm::DocTopic::kBuckets, lam[q]);
        for (std::size_t k = 0; k < h; ++k) cur[k] = lam[q][bk[q][k]] * pe[q][k] + (1 - lam[q][bk[q][k]]) * cur[k];
        std::vector<double>().swap(pe[q]);
        std::vector<int>().swap(bk[q]);
    }
    if (lam_out) lam_out->assign(&lam[0][0], &lam[0][0] + cyphalm::DocTopic::kParts * cyphalm::DocTopic::kBuckets);
    for (std::size_t k = h; k < span; ++k) {
        for (int q = 0; q < cyphalm::DocTopic::kParts; ++q)
            if (on[q]) {
                double P[256];
                const int b = st.predict(q, P);
                cur[k] = lam[q][b] * P[data[train + k]] + (1 - lam[q][b]) * cur[k];
            }
        st.feed(data[train + k]);
    }
}

/// --online-final (GENERATION_PLAN §13.12): one gated linear mixer over Inputs values (every expert and every chain
/// stage, the chain last), weights on the simplex chosen by a set (context8k: InfiniGram's bucket x the previous byte's
/// class, 80 sets, as gate24 gates by a small context), learned online by exponentiated gradient on log loss, every set
/// starting at the chain. predict gives the mixture's probability of the true byte from the inputs' probabilities q
/// (each at least 1e-9); update takes one step on -log p. Another expert joins as one more input (Inputs + 1).
class OnlineFinal {
 public:
    /// sets weight sets of `inputs` weights each; every set starts at the chain (the last input), the others at 0.01.
    OnlineFinal(int sets, int inputs) : n_(inputs), w_(static_cast<std::size_t>(sets) * inputs, 0.01 / (1.0 + 0.01 * (inputs - 1))) {
        for (int g = 0; g < sets; ++g) w_[static_cast<std::size_t>(g) * inputs + (inputs - 1)] = 1.0 / (1.0 + 0.01 * (inputs - 1));
    }

    /// The previous byte's class for a gate: 0 space, 1 letter, 2 digit, 3 markup ([]{}<>|&=), 4 other.
    static int byte_class(unsigned char pb) {
        return std::isspace(pb) ? 0 : std::isalpha(pb) ? 1 : std::isdigit(pb) ? 2
               : std::strchr("[]{}<>|&=", pb) && pb ? 3 : 4;
    }

    double predict(int set, const double* q) const {
        const double* w = &w_[static_cast<std::size_t>(set) * n_];
        double p = 0;
        for (int j = 0; j < n_; ++j) p += w[j] * q[j];
        return p;
    }

    void update(int set, const double* q, double p) {
        double* w = &w_[static_cast<std::size_t>(set) * n_];
        double z = 0;
        for (int j = 0; j < n_; ++j) z += (w[j] *= std::exp(0.05 * std::min(q[j] / std::max(p, 1e-300), 30.0)));
        double z2 = 0;
        for (int j = 0; j < n_; ++j) z2 += (w[j] = std::max(w[j] / z, 1e-7));
        for (int j = 0; j < n_; ++j) w[j] /= z2;
    }

 private:
    int n_;
    std::vector<double> w_;
};

}  // namespace cypha::online
