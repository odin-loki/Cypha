#pragma once

/// Whole-distribution mixing for the content mixer, the PAQ way (docs/reports/FULL_DISTRIBUTION_MIXING.md).
///
/// context8k's chain and final layer (content_mixer.hpp) mix only each expert's probability of the TRUE byte. That
/// scores a linear mixture, but it rules out logistic mixing and adaptive probability maps (APM/SSE), which are
/// nonlinear and need whole distributions to stay normalised. Here every expert gives its whole next-byte distribution
/// for each holdout byte (ByteIn), and FullMixer:
///   1. replays the online chain and the online final layer on whole distributions. Their weights learn from the true
///      byte exactly as content_mixer.hpp's do, so the replica's probability of the true byte equals the chain's and
///      the final layer's (checked in context8k). The replica's distribution F does not sum to 1 (zero experts and the
///      1e-9 floors); Fn is F normalised.
///   2. mixes (a) bitwise: each distribution as the byte's 8 binary decisions, most significant bit first, each bit's
///      probability from the distribution restricted to the partial byte; the experts' stretched bit probabilities go
///      into gated logistic mixers (weight sets chosen by small contexts), a second layer mixes those, and adaptive
///      probability maps (SSE) refine the bit; P(byte) is the product of the 8 bit probabilities, exactly normalised.
///      (b) byte-level geometric mixing: p(s) proportional to prod_j D_j(s)^w_j, weights per context set learned by
///      gradient on log loss, then the same bitwise SSE.
///   Every part learns online from the first byte; every byte (every bit) is predicted before it is learned.
/// All arms run side by side on the same inputs; each prints its bits per byte on the holdout's first and second half.
/// The replay (full-mix-replay) reads a dump of the inputs (context8k --full-mix-dump) and runs the same code, for tuning.

#include "cypha/lm/copy_expert.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cypha::online::fullmix {

// ------------------------------------------------------------------------------------------------ one byte's inputs

/// One holdout byte's experts, each a whole next-byte distribution (nullptr: the expert is absent or has nothing,
/// which reads as all zeros, as the chain reads it). y is the true byte: FullMixer reads it only to learn and score,
/// after every prediction of the byte is made.
struct ByteIn {
    unsigned y = 0, prev = 32, prev2 = 32;   // the true byte; the two bytes before it
    const double* trie[5] = {};              // the trie's five experts (PackedModel::fill_experts)
    bool em = false;                         // the trie's match fired ...
    unsigned pred = 0;                       // ... predicting this byte
    const double* kn = nullptr;              // byte Kneser-Ney (ByteKN::dist)
    int kn_b = 0;                            // its bucket, (deepest order - 1) * 3 + band
    const double* ig[5] = {};                // InfiniGram at caps 4, 16, 64, 512, 8192: count / total (nullptr: total 0)
    int ig_b = 0, ig_n = 0;                  // cap 8192's bucket and matched length
    const double* cls = nullptr;             // the class expert
    int cls_b = 0;
    const double* sim = nullptr;             // the similar-context expert
    int sim_b = 0;
    int cpred = -1, cn = 0;                  // the copy expert: its predicted byte (-1: none) and match length
    const double* doc[3] = {};               // the document parts trig, cache, topic (nullptr: part off)
    int doc_b[3] = {};
    const double* lstm = nullptr;            // the byte LSTM (ByteLSTM::dist)
    const double* ngram = nullptr;           // the online n-gram, mixed (OnlineNgram::dist; nullptr: off)
    const double* ngram_o = nullptr;         // ... and each order's, [order * 256 + byte]
};

/// What the chain holds: which stages are on, their bucket counts, the doc stage's frozen lambdas (fit on the first
/// half by mix_doc_into, as the baseline chain uses them), and the final layer's gate.
struct ChainSpec {
    bool kn = true, cls = true, sim = true, copy = true, doc = true, lstm = true, two_rates = true;
    int kn_buckets = 1, ig_buckets = 28, cls_buckets = 16, sim_buckets = 121, copy_buckets = 8, doc_buckets = 21;
    std::vector<double> doc_lam;  // [part * doc_buckets + bucket]
    bool doc_on[3] = {false, true, false};
    std::string final_gate = "copy";  // copy | ig
    int ngram_orders = 0;             // the online n-gram's orders (0: no n-gram stage; ONLINE_LEARNING_UPGRADES)
};

/// Tuning knobs (k=v,k=v). Defaults are the registered ones (FULL_DISTRIBUTION_MIXING.md).
struct Params {
    // Chosen on the holdout's first half by replay (FULL_DISTRIBUTION_MIXING.md, "Tuning"); X0 ran with lr1 0.002,
    // lr2 0.005, lrb 0.002, apm 0.02, ssef 1, headline As.
    double lr1 = 0.001, lr2 = 0.0005, lrb = 0.0002, apm_rate = 0.03, eps = 1e-6, floor = 1e-6;
    double lr_decay = 0;      // lr_t = lr / (1 + t * lr_decay) per weight set, t = updates of that set
    int sse_final = 0;        // 0: fixed average (mixer + APMs) / n in the probability domain; 1: learned in stretch
    int a2_bits = 18;         // the order-2 APM: (two previous bytes, partial byte) hashed into 2^a2_bits contexts
    std::string headline = "ABs";
    bool parse(const std::string& s) {
        std::size_t at = 0;
        while (at < s.size()) {
            std::size_t c = s.find(',', at);
            if (c == std::string::npos) c = s.size();
            const std::string f = s.substr(at, c - at);
            at = c + 1;
            const std::size_t eq = f.find('=');
            if (eq == std::string::npos) return false;
            const std::string k = f.substr(0, eq), v = f.substr(eq + 1);
            if (k == "headline") { headline = v; continue; }
            const double x = std::atof(v.c_str());
            if (k == "lr1") lr1 = x;
            else if (k == "lr2") lr2 = x;
            else if (k == "lrb") lrb = x;
            else if (k == "apm") apm_rate = x;
            else if (k == "eps") eps = x;
            else if (k == "floor") floor = x;
            else if (k == "decay") lr_decay = x;
            else if (k == "ssef") sse_final = static_cast<int>(x);
            else if (k == "a2bits") a2_bits = static_cast<int>(x);
            else return false;
        }
        return true;
    }
};

inline double squash(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -40.0, 40.0))); }
inline double stretch_eps(double p, double eps) {
    p = std::clamp(p, eps, 1.0 - eps);
    return std::log(p / (1.0 - p));
}
inline int byte_class(unsigned char pb) {  // as OnlineFinal::byte_class
    return std::isspace(pb) ? 0 : std::isalpha(pb) ? 1 : std::isdigit(pb) ? 2 : std::strchr("[]{}<>|&=", pb) && pb ? 3 : 4;
}

// ------------------------------------------------------------------------------------------------ the mixers' parts

/// The probability of 1 for each of the byte's 8 bits along the true byte's path, from a distribution D (any scale):
/// bit j's probability is the mass of the partial byte's 1-branch over the partial byte's mass. 0.5 when the partial
/// byte has no mass.
inline void path_bits(const double* D, unsigned y, double out[8]) {
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

/// A gated logistic mixer, PAQ8's: several weight sets, each chosen by its own small context, each mixing the inputs'
/// stretched probabilities into squash(w . x); a second layer (weights chosen by bit position) mixes their stretched
/// outputs. Each layer-1 set learns from its own output's error, the second layer from its own: w += lr err x.
class BitMixer {
 public:
    BitMixer(int inputs, std::vector<int> set_sizes, int init_input, double lr1, double lr2, double decay, double eps)
        : n_(inputs), sizes_(std::move(set_sizes)), lr1_(lr1), lr2_(lr2), decay_(decay), eps_(eps) {
        const int m = static_cast<int>(sizes_.size());
        for (int z : sizes_) {
            std::vector<double> w(static_cast<std::size_t>(z) * n_, 0.0);
            for (int c = 0; c < z; ++c) w[static_cast<std::size_t>(c) * n_ + init_input] = 1.0;
            w1_.push_back(std::move(w));
            t1_.emplace_back(static_cast<std::size_t>(z), 0u);
        }
        w2_.assign(8 * static_cast<std::size_t>(m), 1.0 / m);
        t2_.assign(8, 0u);
        ctx_.resize(static_cast<std::size_t>(m));
        p1_.resize(static_cast<std::size_t>(m));
        s1_.resize(static_cast<std::size_t>(m));
    }
    int inputs() const { return n_; }
    /// x: n stretched inputs; ctx: one context per set; returns P(bit = 1).
    double p(const double* x, const int* ctx, int bitpos) {
        x_ = x;
        bp_ = bitpos;
        const std::size_t m = sizes_.size();
        double dot2 = 0;
        for (std::size_t k = 0; k < m; ++k) {
            ctx_[k] = ctx[k];
            const double* w = &w1_[k][static_cast<std::size_t>(ctx[k]) * n_];
            double d = 0;
            for (int i = 0; i < n_; ++i) d += w[i] * x[i];
            p1_[k] = squash(d);
            s1_[k] = stretch_eps(p1_[k], eps_);
            dot2 += w2_[static_cast<std::size_t>(bitpos) * m + k] * s1_[k];
        }
        return p2_ = squash(dot2);
    }
    void update(int bit) {
        const std::size_t m = sizes_.size();
        for (std::size_t k = 0; k < m; ++k) {
            unsigned& t = t1_[k][static_cast<std::size_t>(ctx_[k])];
            const double lr = lr1_ / (1.0 + decay_ * t);
            ++t;
            const double err = (bit - p1_[k]) * lr;
            double* w = &w1_[k][static_cast<std::size_t>(ctx_[k]) * n_];
            for (int i = 0; i < n_; ++i) w[i] += err * x_[i];
        }
        unsigned& t = t2_[static_cast<std::size_t>(bp_)];
        const double lr = lr2_ / (1.0 + decay_ * t);
        ++t;
        const double err = (bit - p2_) * lr;
        for (std::size_t k = 0; k < m; ++k) w2_[static_cast<std::size_t>(bp_) * m + k] += err * s1_[k];
    }
    std::size_t bytes() const {
        std::size_t b = w2_.size() * 8;
        for (const auto& w : w1_) b += w.size() * 8;
        return b;
    }

 private:
    int n_;
    std::vector<int> sizes_;
    double lr1_, lr2_, decay_, eps_;
    std::vector<std::vector<double>> w1_;
    std::vector<std::vector<unsigned>> t1_;
    std::vector<double> w2_;
    std::vector<unsigned> t2_;
    std::vector<int> ctx_;
    std::vector<double> p1_, s1_;
    const double* x_ = nullptr;
    int bp_ = 0;
    double p2_ = 0.5;
};

/// An adaptive probability map (PAQ's APM): per context, 33 values over the stretched input probability in [-16, 16];
/// the output interpolates the two nearest, and both move toward the bit by their share of the rate. Each starts as
/// the identity.
class Apm {
 public:
    explicit Apm(std::size_t contexts) : t_(contexts * 33) {
        for (std::size_t c = 0; c < contexts; ++c)
            for (int j = 0; j < 33; ++j) t_[c * 33 + j] = static_cast<float>(squash((j - 16) * 1.0));
    }
    double p(double st, std::size_t ctx) {
        const double pos = std::clamp(st, -15.999, 15.999) + 16.0;
        const int j = static_cast<int>(pos);
        f_ = pos - j;
        at_ = ctx * 33 + static_cast<std::size_t>(j);
        return t_[at_] * (1 - f_) + t_[at_ + 1] * f_;
    }
    void update(int bit, double rate) {
        t_[at_] += static_cast<float>(rate * (1 - f_) * (bit - t_[at_]));
        t_[at_ + 1] += static_cast<float>(rate * f_ * (bit - t_[at_ + 1]));
    }
    std::size_t bytes() const { return t_.size() * 4; }

 private:
    std::vector<float> t_;
    std::size_t at_ = 0;
    double f_ = 0;
};

/// The SSE stage after a mixer: four APMs keyed by the partial byte (order 0), the previous byte and the partial byte
/// (order 1), a hash of the two previous bytes and the partial byte (order 2), and InfiniGram's bucket and the partial
/// byte; then either a fixed average with the mixer's probability, or a small learned mixer over the five stretched
/// probabilities (weights by bit position, starting at the mixer's own).
class Sse {
 public:
    Sse(const Params& P)
        : a0_(256), a1_(65536), a2_(std::size_t(1) << P.a2_bits), a3_(32 * 256), rate_(P.apm_rate), eps_(P.eps),
          mode_(P.sse_final), bits2_(P.a2_bits), fin_(5, {8}, 0, P.lr1, P.lr2, 0, P.eps) {}
    double p(double pin, const ByteIn& in, unsigned c0, int bitpos) {
        const double st = stretch_eps(pin, eps_);
        const std::uint32_t key = (in.prev2 & 255u) << 16 | (in.prev & 255u) << 8 | (c0 & 255u);
        const std::size_t h2 = static_cast<std::uint32_t>(key * 2654435761u) >> (32 - bits2_);  // Fibonacci hashing
        q_[0] = pin;
        q_[1] = a0_.p(st, c0);
        q_[2] = a1_.p(st, in.prev * 256 + c0);
        q_[3] = a2_.p(st, h2);
        q_[4] = a3_.p(st, static_cast<std::size_t>(std::min(in.ig_b, 31)) * 256 + c0);
        if (mode_ == 0) return std::clamp((q_[0] + q_[1] + q_[2] + q_[3] + q_[4]) / 5, eps_, 1 - eps_);
        for (int k = 0; k < 5; ++k) x_[k] = stretch_eps(q_[k], eps_);
        const int z = 0;
        return std::clamp(fin_.p(x_, &z, bitpos), eps_, 1 - eps_);
    }
    void update(int bit) {
        a0_.update(bit, rate_);
        a1_.update(bit, rate_);
        a2_.update(bit, rate_);
        a3_.update(bit, rate_);
        if (mode_ != 0) fin_.update(bit);
    }
    std::size_t bytes() const { return a0_.bytes() + a1_.bytes() + a2_.bytes() + a3_.bytes() + fin_.bytes(); }

 private:
    Apm a0_, a1_, a2_, a3_;
    double rate_, eps_;
    int mode_, bits2_;
    BitMixer fin_;
    double q_[5] = {}, x_[5] = {};
};

/// Byte-level geometric mixing: p(s) proportional to exp(sum_j w_j log D_j(s)), D_j normalised with a floor; one weight
/// vector per context set, learned by gradient on -log p(y): dw_j = -lr (E_p[log D_j] - log D_j(y)). Starts at the
/// input init (weight 1, the rest 0).
class GeoMix {
 public:
    GeoMix(int inputs, int sets, int init, double lr, double decay)
        : n_(inputs), w_(static_cast<std::size_t>(sets) * inputs, 0.0), t_(static_cast<std::size_t>(sets), 0u), lr_(lr), decay_(decay) {
        for (int g = 0; g < sets; ++g) w_[static_cast<std::size_t>(g) * inputs + init] = 1.0;
    }
    /// L[j]: log D_j over 256 bytes (nullptr: the input has nothing this byte and is skipped); P gets the mixture.
    void dist(const double* const* L, int set, double* P) {
        set_ = set;
        L_ = L;
        const double* w = &w_[static_cast<std::size_t>(set) * n_];
        double z[256];
        std::fill(z, z + 256, 0.0);
        for (int j = 0; j < n_; ++j)
            if (L[j] && w[j] != 0)
                for (int s = 0; s < 256; ++s) z[s] += w[j] * L[j][s];
        double mx = -1e300;
        for (int s = 0; s < 256; ++s) mx = std::max(mx, z[s]);
        double sum = 0;
        for (int s = 0; s < 256; ++s) sum += (P[s] = std::exp(z[s] - mx));
        for (int s = 0; s < 256; ++s) P[s] /= sum;
        P_ = P;
    }
    void update(unsigned y) {
        double* w = &w_[static_cast<std::size_t>(set_) * n_];
        unsigned& t = t_[static_cast<std::size_t>(set_)];
        const double lr = lr_ / (1.0 + decay_ * t);
        ++t;
        for (int j = 0; j < n_; ++j) {
            if (!L_[j]) continue;
            double e = 0;
            for (int s = 0; s < 256; ++s) e += P_[s] * L_[j][s];
            w[j] -= lr * (e - L_[j][y]);
        }
    }
    std::size_t bytes() const { return w_.size() * 8; }

 private:
    int n_;
    std::vector<double> w_;
    std::vector<unsigned> t_;
    double lr_, decay_;
    int set_ = 0;
    const double* const* L_ = nullptr;
    const double* P_ = nullptr;
};

// ------------------------------------------------------------------------------------------------ the whole mixer

/// One online stage of the chain by exponentiated gradient (content_mixer.hpp's eg_stage), on whole distributions.
struct EgState {
    std::vector<double> we, wc;
    double eta = 0.02;
    void init(int nb, double e) {
        we.assign(static_cast<std::size_t>(nb), 0.05);
        wc.assign(static_cast<std::size_t>(nb), 0.95);
        eta = e;
    }
    /// cur := lambda pe + (1 - lambda) cur over all 256 bytes; then the weights learn from the true byte y, with the same
    /// arithmetic as eg_stage, so cur[y] stays equal to the scalar chain's.
    void apply(const double* pe, int b, double* cur, unsigned y) {
        const std::size_t u = static_cast<std::size_t>(b);
        const double l = we[u], prev = cur[y];
        for (int s = 0; s < 256; ++s) cur[s] = l * pe[s] + (1 - l) * cur[s];
        const double p = cur[y], pp = std::max(p, 1e-300);
        double e = we[u] * std::exp(eta * std::min(pe[y] / pp, 30.0)), c = wc[u] * std::exp(eta * std::min(prev / pp, 30.0));
        double z = e + c;
        e = std::max(e / z, 1e-4);
        c = std::max(c / z, 1e-4);
        z = e + c;
        we[u] = e / z;
        wc[u] = c / z;
    }
};

class FullMixer {
 public:
    // The arms, side by side (FULL_DISTRIBUTION_MIXING.md): F the final layer's replica at the true byte (it must equal
    // online_final_bpc); Fn the replica normalised; Fs Fn then bitwise SSE; A the bitwise logistic mixer over every input
    // (the chain and the final replica included); As A then SSE; Rs the bitwise mixer over the raw experts only, then
    // SSE; B the byte-level geometric mixer; Bs B then SSE; ABs As and Bs mixed per bit.
    static constexpr int kArms = 9;
    static constexpr const char* kArmName[kArms] = {"F", "Fn", "Fs", "A", "As", "Rs", "B", "Bs", "ABs"};

    FullMixer(const ChainSpec& cs, const Params& P, std::size_t half) : cs_(cs), P_(P), half_(half) {
        const double rates[3] = {0.02, 0.005, 0.08};
        nr_ = cs.two_rates ? 3 : 1;
        for (int r = 0; r < nr_; ++r) {
            kn_[r].init(cs.kn_buckets, rates[r]);
            ig_[r].init(cs.ig_buckets, rates[r]);
            cls_[r].init(cs.cls_buckets, rates[r]);
            sim_[r].init(cs.sim_buckets, rates[r]);
            copy_[r].init(cs.copy_buckets, rates[r]);
            lstm_[r].init(cs.ig_buckets, rates[r]);
            ngram_[r].init(cs.ig_buckets, rates[r]);
        }
        // The final layer: 23 row values, the LSTM, the slow and fast chains, then the chain (cyphalm_trie context8k).
        // With the online n-gram (main since ONLINE_LEARNING_UPGRADES): its mixed probability and each order's come after
        // the slow and fast chains, before the chain.
        const int ng = cs.ngram_orders > 0 ? 1 + cs.ngram_orders : 0;
        nfin_ = 24 + (cs.lstm ? 1 : 0) + (cs.two_rates ? 2 : 0) + ng;
        kRaw_ = 16 + ng;
        kAll_ = kRaw_ + 8;
        fw_.assign(80 * static_cast<std::size_t>(nfin_), 0.01 / (1.0 + 0.01 * (nfin_ - 1)));
        for (int g = 0; g < 80; ++g) fw_[static_cast<std::size_t>(g) * nfin_ + (nfin_ - 1)] = 1.0 / (1.0 + 0.01 * (nfin_ - 1));
        // Bitwise inputs: the "all" list (see kAll_) plus copy's adaptive input and a bias; the raw list likewise.
        nall_ = kAll_ + 2;
        nraw_ = kRaw_ + 2;
        const std::vector<int> sets = {256, 32 * 8, 8 * 4 * 8, 5 * 256};
        mixA_ = std::make_unique<BitMixer>(nall_, sets, kAll_ - 1, P.lr1, P.lr2, P.lr_decay, P.eps);  // starts at Fn
        mixR_ = std::make_unique<BitMixer>(nraw_, sets, kRawTM, P.lr1, P.lr2, P.lr_decay, P.eps);    // starts at the trie mixture
        geo_ = std::make_unique<GeoMix>(kAll_ + 1, 80, kAll_ - 1, P.lrb, P.lr_decay);                   // + copy's one-hot
        abmix_ = std::make_unique<BitMixer>(2, std::vector<int>{8}, 0, P.lr1, P.lr2, 0, P.eps);
        for (auto* s : {&sseF_, &sseA_, &sseR_, &sseB_}) *s = std::make_unique<Sse>(P);
        copy_ok_.assign(8 * 8, {1.0, 2.0});
    }

    std::size_t bytes() const {
        return mixA_->bytes() + mixR_->bytes() + geo_->bytes() + sseF_->bytes() * 4 + fw_.size() * 8;
    }

    /// One holdout byte k: every arm predicts it, is scored, then learns it.
    void step(const ByteIn& in, std::size_t k) {
        const unsigned y = in.y;
        static const double zero[256] = {};
        // ---- the chain on whole distributions (cyphalm_trie context8k's order; content_mixer.hpp's arithmetic)
        double TM[256];
        {
            double z = 0.0;  // mix_dist
            for (int i = 0; i < 5; ++i) z += tw_[i];
            if (in.em) z += tw_[5];
            const double inv = 1.0 / z;
            std::fill(TM, TM + 256, 0.0);
            for (int i = 0; i < 5; ++i) {
                const double a = tw_[i] * inv;
                for (int s = 0; s < 256; ++s) TM[s] += a * in.trie[i][s];
            }
            if (in.em) TM[in.pred] += tw_[5] * inv;
        }
        double ch[3][256];
        for (int r = 0; r < nr_; ++r) std::copy(TM, TM + 256, ch[r]);
        double copy1[256] = {};
        if (in.cpred >= 0) copy1[in.cpred] = 1.0;
        double lst[256];
        if (cs_.lstm)
            for (int s = 0; s < 256; ++s) lst[s] = std::max(in.lstm[s], 1e-12);
        auto stage = [&](EgState* st, const double* pe, int b) {
            for (int r = 0; r < nr_; ++r) st[r].apply(pe ? pe : zero, b, ch[r], y);
        };
        if (cs_.kn) stage(kn_, in.kn, in.kn_b);
        stage(ig_, in.ig[4], in.ig_b);
        std::copy(ch[0], ch[0] + 256, snap_[0]);
        if (cs_.cls) stage(cls_, in.cls, in.cls_b);
        std::copy(ch[0], ch[0] + 256, snap_[1]);
        if (cs_.sim) stage(sim_, in.sim, in.sim_b);
        std::copy(ch[0], ch[0] + 256, snap_[2]);
        if (cs_.copy) stage(copy_, copy1, cyphalm::copy_bucket(in.cn));
        std::copy(ch[0], ch[0] + 256, snap_[3]);
        if (cs_.doc)  // frozen lambdas (mix_doc_into's fit on the first half), the main chain only
            for (int q = 0; q < 3; ++q)
                if (cs_.doc_on[q]) {
                    const double l = cs_.doc_lam[static_cast<std::size_t>(q * cs_.doc_buckets + in.doc_b[q])];
                    for (int s = 0; s < 256; ++s) ch[0][s] = l * in.doc[q][s] + (1 - l) * ch[0][s];
                }
        double ngp[256];
        if (cs_.ngram_orders > 0) {  // pngram = max(P(y), 1e-12), a lambda per InfiniGram bucket, before the LSTM
            for (int s = 0; s < 256; ++s) ngp[s] = std::max(in.ngram[s], 1e-12);
            stage(ngram_, ngp, in.ig_b);
        }
        if (cs_.lstm) stage(lstm_, lst, in.ig_b);
        chain_y_ = ch[0][y];
        snap3_y_ = snap_[3][y];

        // ---- the final layer on whole distributions (OnlineFinal's arithmetic; the row's values as float)
        auto fl = [](double v) { return static_cast<double>(static_cast<float>(v)); };
        std::vector<std::array<double, 256>>& q = q_;
        q.resize(static_cast<std::size_t>(nfin_));
        int nq = 0;
        for (int j = 0; j < 5; ++j, ++nq)
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, fl(in.trie[j][s]));
        for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, in.em && in.pred == static_cast<unsigned>(s) ? 1.0 : 0.0);
        ++nq;
        for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, fl(TM[s]));
        ++nq;
        for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, in.kn ? fl(in.kn[s]) : 0.0);
        ++nq;
        for (int c = 0; c < 5; ++c, ++nq)
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, in.ig[c] ? fl(in.ig[c][s]) : 0.0);
        for (const double* d : {in.cls, in.sim}) {
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, d ? fl(d[s]) : 0.0);
            ++nq;
        }
        for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, in.cpred == s && in.cn > 0 ? 1.0 : 0.0);
        ++nq;
        for (int p = 0; p < 3; ++p, ++nq)
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, in.doc[p] ? fl(in.doc[p][s]) : 0.0);
        for (int p = 0; p < 4; ++p, ++nq)
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, fl(snap_[p][s]));
        if (cs_.lstm) {
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, lst[s]);
            ++nq;
        }
        if (cs_.two_rates)
            for (int r = 1; r < 3; ++r, ++nq)
                for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, ch[r][s]);
        if (cs_.ngram_orders > 0) {
            for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, ngp[s]);
            ++nq;
            for (int o = 0; o < cs_.ngram_orders; ++o, ++nq)
                for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, fl(in.ngram_o[o * 256 + s]));
        }
        for (int s = 0; s < 256; ++s) q[nq][s] = std::max(1e-9, fl(ch[0][s]));
        ++nq;
        const int bc = byte_class(static_cast<unsigned char>(in.prev));
        const int g = cs_.final_gate == "copy" ? cyphalm::copy_bucket(in.cn) * 10 + bc * 2 + (in.ig_n >= 16 ? 1 : 0)
                                               : std::min(in.ig_b, 15) * 5 + bc;
        double F[256];
        double* w = &fw_[static_cast<std::size_t>(g) * nfin_];
        for (int s = 0; s < 256; ++s) {
            double p = 0;
            for (int j = 0; j < nfin_; ++j) p += w[j] * q[j][s];
            F[s] = p;
        }
        const double pf = F[y];
        double fsum = 0;
        for (int s = 0; s < 256; ++s) fsum += F[s];
        double Fn[256];
        for (int s = 0; s < 256; ++s) Fn[s] = F[s] / fsum;
        fsum_acc_ += fsum;

        // ---- the inputs of the PAQ mixers (unnormalised is fine for bits; geo normalises)
        const double* all[kMax];
        {
            int n = 0;
            for (const double* d : {in.trie[0], in.trie[1], in.trie[2], in.trie[3], in.trie[4], static_cast<const double*>(TM), in.kn,
                                    in.ig[0], in.ig[1], in.ig[2], in.ig[3], in.ig[4], in.cls, in.sim,
                                    in.doc[1] ? in.doc[1] : (in.doc[0] ? in.doc[0] : in.doc[2]),
                                    cs_.lstm ? static_cast<const double*>(lst) : nullptr})
                all[n++] = d;
            if (cs_.ngram_orders > 0) {
                all[n++] = in.ngram;
                for (int o = 0; o < cs_.ngram_orders; ++o) all[n++] = in.ngram_o + o * 256;
            }
            for (const double* d : {static_cast<const double*>(snap_[0]), static_cast<const double*>(snap_[1]),
                                    static_cast<const double*>(snap_[2]), static_cast<const double*>(snap_[3]),
                                    static_cast<const double*>(ch[0]), cs_.two_rates ? static_cast<const double*>(ch[1]) : nullptr,
                                    cs_.two_rates ? static_cast<const double*>(ch[2]) : nullptr, static_cast<const double*>(Fn)})
                all[n++] = d;
        }
        double bits[kMax][8];
        for (int j = 0; j < kAll_; ++j) {
            if (all[j]) path_bits(all[j], y, bits[j]);
            else std::fill(bits[j], bits[j] + 8, 0.5);
        }

        // ---- B: geometric mixing of the normalised, floored distributions (and copy's one-hot)
        double Lb[kMax + 1][256];
        const double* L[kMax + 1];
        for (int j = 0; j <= kAll_; ++j) {
            const double* d = j < kAll_ ? all[j] : (in.cpred >= 0 ? copy1 : nullptr);
            if (!d) {
                L[j] = nullptr;
                continue;
            }
            double z = 0;
            for (int s = 0; s < 256; ++s) z += d[s];
            if (!(z > 0)) {
                L[j] = nullptr;
                continue;
            }
            for (int s = 0; s < 256; ++s) Lb[j][s] = std::log(std::max(d[s] / z, P_.floor));
            L[j] = Lb[j];
        }
        double PB[256];
        geo_->dist(L, g, PB);
        double bitsB[8];
        path_bits(PB, y, bitsB);

        // ---- bit by bit along the true byte: every arm predicts the bit, then learns it
        double lp[kArms] = {};
        lp[0] = std::log2(std::max(pf, 1e-12));
        lp[1] = std::log2(std::max(Fn[y], 1e-300));
        lp[6] = std::log2(std::max(PB[y], 1e-300));
        unsigned c0 = 1;
        const int cb = cyphalm::copy_bucket(in.cn);
        for (int j = 0; j < 8; ++j) {
            const int bit = (y >> (7 - j)) & 1;
            // copy's input: the predicted bit, if the partial byte still agrees with copy's byte, at its learned confidence
            int cst = 0;  // 0 none, 1 disagrees, 2 expects 0, 3 expects 1
            double xc = 0;
            if (in.cpred >= 0) {
                const unsigned cp = static_cast<unsigned>(in.cpred);
                if ((cp + 256) >> (8 - j) == c0) {
                    const int eb = (cp >> (7 - j)) & 1;
                    cst = 2 + eb;
                    const auto& ok = copy_ok_[static_cast<std::size_t>(cb * 8 + j)];
                    const double st = stretch_eps(ok[0] / ok[1], P_.eps);
                    xc = eb ? st : -st;
                } else
                    cst = 1;
            }
            const int ctx[4] = {static_cast<int>(c0), std::min(in.ig_b, 31) * 8 + j, (cb * 4 + cst) * 8 + j,
                                bc * 256 + static_cast<int>(c0)};
            double xa[kMax + 2], xr[kMax + 2];
            for (int i = 0; i < kAll_; ++i) xa[i] = stretch_eps(bits[i][j], P_.eps);
            xa[kAll_] = xc;
            xa[kAll_ + 1] = 0.25;
            for (int i = 0; i < kRaw_; ++i) xr[i] = xa[i];
            xr[kRaw_] = xc;
            xr[kRaw_ + 1] = 0.25;
            const double pA = std::clamp(mixA_->p(xa, ctx, j), P_.eps, 1 - P_.eps);
            const double pR = std::clamp(mixR_->p(xr, ctx, j), P_.eps, 1 - P_.eps);
            const double pFs = sseF_->p(std::clamp(bits[kAll_ - 1][j], P_.eps, 1 - P_.eps), in, c0, j);
            const double pAs = sseA_->p(pA, in, c0, j);
            const double pRs = sseR_->p(pR, in, c0, j);
            const double pBs = sseB_->p(std::clamp(bitsB[j], P_.eps, 1 - P_.eps), in, c0, j);
            const double xab[2] = {stretch_eps(pAs, P_.eps), stretch_eps(pBs, P_.eps)};
            const int z0 = 0;
            const double pAB = std::clamp(abmix_->p(xab, &z0, j), P_.eps, 1 - P_.eps);
            auto lg = [bit](double p) { return std::log2(bit ? p : 1 - p); };
            lp[2] += lg(pFs);
            lp[3] += lg(pA);
            lp[4] += lg(pAs);
            lp[5] += lg(pRs);
            lp[7] += lg(pBs);
            lp[8] += lg(pAB);
            mixA_->update(bit);
            mixR_->update(bit);
            sseF_->update(bit);
            sseA_->update(bit);
            sseR_->update(bit);
            sseB_->update(bit);
            abmix_->update(bit);
            if (cst >= 2) {
                auto& ok = copy_ok_[static_cast<std::size_t>(cb * 8 + j)];
                ok[0] += (bit == cst - 2) ? 1.0 : 0.0;
                ok[1] += 1.0;
                if (ok[1] > 1024) ok[0] *= 0.5, ok[1] *= 0.5;
            }
            c0 = c0 * 2 + static_cast<unsigned>(bit);
        }
        const int h = k >= half_ ? 1 : 0;
        for (int a = 0; a < kArms; ++a) bits_[h][a] -= lp[a];
        ++n_[h];

        // ---- learning from y: the geometric mixer, then the final layer (OnlineFinal::update)
        geo_->update(y);
        {
            double z = 0;
            for (int j = 0; j < nfin_; ++j) z += (w[j] *= std::exp(0.05 * std::min(q[j][y] / std::max(pf, 1e-300), 30.0)));
            double z2 = 0;
            for (int j = 0; j < nfin_; ++j) z2 += (w[j] = std::max(w[j] / z, 1e-7));
            for (int j = 0; j < nfin_; ++j) w[j] /= z2;
        }
    }

    void set_trie_weights(const double w[6]) { std::copy(w, w + 6, tw_); }
    double chain_y() const { return chain_y_; }    // the replica chain's probability of the last true byte
    double snap3_y() const { return snap3_y_; }    // ... after copy
    double bpc(int half, int arm) const { return n_[half] ? bits_[half][arm] / static_cast<double>(n_[half]) : 0; }
    double mean_fsum() const { return fsum_acc_ / static_cast<double>(n_[0] + n_[1]); }
    int arm(const std::string& name) const {
        for (int a = 0; a < kArms; ++a)
            if (name == kArmName[a]) return a;
        return -1;
    }
    void report(std::ostream& o) const {
        char buf[200];
        for (int a = 0; a < kArms; ++a) {
            std::snprintf(buf, sizeof buf, "full_mix_arm %-3s first_half %.4f second_half %.4f\n", kArmName[a], bpc(0, a), bpc(1, a));
            o << buf;
        }
        std::snprintf(buf, sizeof buf, "full_mix_replica_mass %.6f (the final layer's whole distribution's mean sum)\n", mean_fsum());
        o << buf;
    }

 private:
    // The "all" list: 0-4 trie experts, 5 trie mixture, 6 KN, 7-11 InfiniGram caps, 12 class, 13 similar, 14 doc,
    // 15 LSTM, then (with the n-gram) the mixed n-gram and each order, then the chain after InfiniGram/class/similar/
    // copy, the chain, the slow and fast chains, and Fn last. The raw list is everything before the chain (kRaw_).
    static constexpr int kMax = 48, kRawTM = 5;
    int kAll_ = 24, kRaw_ = 16;
    ChainSpec cs_;
    Params P_;
    std::size_t half_;
    int nr_ = 1, nfin_ = 0, nall_ = 0, nraw_ = 0;
    double tw_[6] = {};
    EgState kn_[3], ig_[3], cls_[3], sim_[3], copy_[3], lstm_[3], ngram_[3];
    double snap_[4][256] = {};
    std::vector<double> fw_;
    std::vector<std::array<double, 256>> q_;
    std::unique_ptr<BitMixer> mixA_, mixR_, abmix_;
    std::unique_ptr<GeoMix> geo_;
    std::unique_ptr<Sse> sseF_, sseA_, sseR_, sseB_;
    std::vector<std::array<double, 2>> copy_ok_;
    double chain_y_ = 0, snap3_y_ = 0, fsum_acc_ = 0;
    double bits_[2][kArms] = {};
    std::size_t n_[2] = {0, 0};
};

// ------------------------------------------------------------------------------------------------ dump and replay

/// The dump (context8k --full-mix-dump FILE): a header, then one record per holdout byte with its side values and its
/// distributions as float32, in ByteIn's order (trie 5, KN, InfiniGram 5, class, similar, the doc parts that are on,
/// LSTM). Absent distributions are written as zeros. full-mix-replay reads it into the same FullMixer.
struct DumpHeader {
    char magic[4] = {'F', 'M', 'X', '2'};  // FMX2: then a uint32, the n-gram's orders; FMX1 (X0's dump) has no n-gram
    std::uint64_t H = 0, half = 0;
    std::uint8_t kn = 0, cls = 0, sim = 0, copy = 0, doc = 0, lstm = 0, two_rates = 0, gate_copy = 1;
    std::uint8_t doc_on[3] = {};
    std::int32_t kn_buckets = 1, doc_buckets = 21;
    double tw[6] = {};
    double doc_lam[3 * 21] = {};
};
struct DumpSide {
    std::uint8_t y, prev, prev2, em, pred, kn_b, ig_b, cls_b, sim_b, doc_b[3];
    std::int16_t cpred;
    std::int32_t cn, ig_n;
    std::uint8_t has;  // bit c: InfiniGram cap c has a distribution
};
inline constexpr int kDumpDists = 5 + 1 + 5 + 1 + 1 + 3 + 1;

inline void dump_byte(std::ofstream& f, const ByteIn& in, int ngram_orders = 0) {
    DumpSide s{};
    s.y = static_cast<std::uint8_t>(in.y), s.prev = static_cast<std::uint8_t>(in.prev), s.prev2 = static_cast<std::uint8_t>(in.prev2);
    s.em = in.em, s.pred = static_cast<std::uint8_t>(in.pred), s.kn_b = static_cast<std::uint8_t>(in.kn_b);
    s.ig_b = static_cast<std::uint8_t>(in.ig_b), s.cls_b = static_cast<std::uint8_t>(in.cls_b), s.sim_b = static_cast<std::uint8_t>(in.sim_b);
    for (int q = 0; q < 3; ++q) s.doc_b[q] = static_cast<std::uint8_t>(in.doc_b[q]);
    s.cpred = static_cast<std::int16_t>(in.cpred), s.cn = in.cn, s.ig_n = in.ig_n;
    s.has = 0;
    for (int c = 0; c < 5; ++c) s.has |= in.ig[c] ? (1 << c) : 0;
    f.write(reinterpret_cast<const char*>(&s), sizeof s);
    float buf[256];
    auto put = [&](const double* d) {
        for (int i = 0; i < 256; ++i) buf[i] = d ? static_cast<float>(d[i]) : 0.0f;
        f.write(reinterpret_cast<const char*>(buf), sizeof buf);
    };
    for (int j = 0; j < 5; ++j) put(in.trie[j]);
    put(in.kn);
    for (int c = 0; c < 5; ++c) put(in.ig[c]);
    put(in.cls);
    put(in.sim);
    for (int q = 0; q < 3; ++q) put(in.doc[q]);
    put(in.lstm);
    if (ngram_orders > 0) {
        put(in.ngram);
        for (int o = 0; o < ngram_orders; ++o) put(in.ngram_o + o * 256);
    }
}

/// cyphalm_trie full-mix-replay FILE [PARAMS]: every arm over the dump; prints full_mix_arm lines.
inline int replay(const std::string& path, const std::string& params) {
    std::ifstream f(path, std::ios::binary);
    DumpHeader hd;
    std::uint32_t ngo = 0;
    if (!f.read(reinterpret_cast<char*>(&hd), sizeof hd) ||
        (std::memcmp(hd.magic, "FMX1", 4) != 0 && std::memcmp(hd.magic, "FMX2", 4) != 0) ||
        (hd.magic[3] == '2' && !f.read(reinterpret_cast<char*>(&ngo), sizeof ngo)) || ngo > 16) {
        std::cerr << "full-mix-replay: cannot read " << path << "\n";
        return 2;
    }
    Params P;
    if (!P.parse(params)) {
        std::cerr << "full-mix-replay: bad params '" << params << "'\n";
        return 2;
    }
    ChainSpec cs;
    cs.kn = hd.kn, cs.cls = hd.cls, cs.sim = hd.sim, cs.copy = hd.copy, cs.doc = hd.doc, cs.lstm = hd.lstm;
    cs.two_rates = hd.two_rates;
    cs.final_gate = hd.gate_copy ? "copy" : "ig";
    for (int q = 0; q < 3; ++q) cs.doc_on[q] = hd.doc_on[q];
    cs.kn_buckets = hd.kn_buckets;
    cs.doc_buckets = hd.doc_buckets;
    cs.doc_lam.assign(hd.doc_lam, hd.doc_lam + 3 * 21);
    cs.ngram_orders = static_cast<int>(ngo);
    const int nd = kDumpDists + (ngo ? 1 + static_cast<int>(ngo) : 0);
    FullMixer fm(cs, P, hd.half);
    fm.set_trie_weights(hd.tw);
    std::vector<float> fb(static_cast<std::size_t>(nd) * 256);
    std::vector<double> db(static_cast<std::size_t>(nd) * 256);
    for (std::uint64_t k = 0; k < hd.H; ++k) {
        DumpSide s;
        if (!f.read(reinterpret_cast<char*>(&s), sizeof s) ||
            !f.read(reinterpret_cast<char*>(fb.data()), static_cast<std::streamsize>(fb.size() * 4))) {
            std::cerr << "full-mix-replay: short file at byte " << k << "\n";
            return 2;
        }
        for (std::size_t i = 0; i < fb.size(); ++i) db[i] = fb[i];
        ByteIn in;
        in.y = s.y, in.prev = s.prev, in.prev2 = s.prev2, in.em = s.em, in.pred = s.pred;
        const double* d = db.data();
        for (int j = 0; j < 5; ++j) in.trie[j] = d + 256 * j;
        in.kn = hd.kn ? d + 256 * 5 : nullptr;
        in.kn_b = s.kn_b;
        for (int c = 0; c < 5; ++c) in.ig[c] = (s.has >> c) & 1 ? d + 256 * (6 + c) : nullptr;
        in.ig_b = s.ig_b, in.ig_n = s.ig_n;
        in.cls = hd.cls ? d + 256 * 11 : nullptr;
        in.cls_b = s.cls_b;
        in.sim = hd.sim ? d + 256 * 12 : nullptr;
        in.sim_b = s.sim_b;
        in.cpred = s.cpred, in.cn = s.cn;
        for (int q = 0; q < 3; ++q) {
            in.doc[q] = hd.doc && hd.doc_on[q] ? d + 256 * (13 + q) : nullptr;
            in.doc_b[q] = s.doc_b[q];
        }
        in.lstm = hd.lstm ? d + 256 * 16 : nullptr;
        if (ngo) in.ngram = d + 256 * 17, in.ngram_o = d + 256 * 18;
        fm.step(in, k);
        if ((k + 1) % 262144 == 0) std::cerr << "replay " << (k + 1) << " bytes\n";
    }
    fm.report(std::cout);
    std::cout << "full_mix_state_bytes " << fm.bytes() << "\n";
    return 0;
}

}  // namespace cypha::online::fullmix
