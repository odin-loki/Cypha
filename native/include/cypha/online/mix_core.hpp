#pragma once

/// One library for the maths CyphaLM's mixers share (docs/reports/MERGE_MIXERS.md, GENERATION_PLAN §13.40 M2).
///
/// PAQ's logistic mixing is geometric mixing over a binary alphabet: a product of experts with a convex log loss
/// (Mattern 2012-13). The content mixer (content_mixer.hpp), the full mix (full_mixer.hpp), the meta mix
/// (meta_mixer.hpp) and K5's structure ranker (lm/structure/mixer.hpp) are all built from these parts:
///   - exponentiated gradient: the multiplicative step (reweight), on the simplex with a floor (eg_step), as one
///     weight set per context (EgSets) or one pair per bucket (EgLambdas), and with signed weights (eg_pm);
///   - the logistic domain: squash and stretch, a distribution as its 8 binary decisions (path_bits), PAQ's two-layer
///     gated logistic mixer (LogisticMixer) and adaptive probability maps (Apm);
///   - geometric mixing: softmax over log-linear scores, its log-loss gradient (expect), the byte-level geometric
///     mixer (GeoMixer);
///   - mixtures over learning rates and over arms: exponential weights with fixed share (FixedShare).
/// Weight sets are chosen by small contexts throughout (byte_class is one). Each part keeps the floating-point order
/// of the code it replaced, so every mixer's output is unchanged bit for bit.
///
/// Author: Odin Loch

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cypha/online/hyper_adapt.hpp"  // T1b: settings that move while the model runs (off by default)

namespace cypha::online::mix {

// ------------------------------------------------------------------------------------------------ small pieces

inline double squash(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -40.0, 40.0))); }
inline double stretch_eps(double p, double eps) {
    p = std::clamp(p, eps, 1.0 - eps);
    return std::log(p / (1.0 - p));
}
/// l a + (1 - l) b: one linear stage of the content mixer's chain.
inline double lerp(double l, double a, double b) { return l * a + (1 - l) * b; }
/// The previous byte's class, a context for weight sets: 0 space, 1 letter, 2 digit, 3 markup ([]{}<>|&=), 4 other.
inline int byte_class(unsigned char pb) {
    return std::isspace(pb) ? 0 : std::isalpha(pb) ? 1 : std::isdigit(pb) ? 2 : std::strchr("[]{}<>|&=", pb) && pb ? 3 : 4;
}
/// A mixer's settings "k=v,k=v": set(k, v) for each pair; false if a pair has no '=' or set refuses it.
template <class Set> inline bool parse_kv(const std::string& s, Set set) {
    for (std::size_t at = 0; at < s.size();) {
        std::size_t c = s.find(',', at);
        if (c == std::string::npos) c = s.size();
        const std::string f = s.substr(at, c - at);
        at = c + 1;
        const std::size_t eq = f.find('=');
        if (eq == std::string::npos || !set(f.substr(0, eq), f.substr(eq + 1))) return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ exponentiated gradient

/// The multiplicative step of every exponentiated-gradient and exponential-weights mixer: w_j *= f(j) for j < n, in
/// order. Returns the new weights' sum, added in the same order.
template <class F> inline double reweight(double* w, int n, F f) {
    double z = 0;
    for (int j = 0; j < n; ++j) z += (w[j] *= f(j));
    return z;
}
/// Back onto the simplex after reweight: w_j = max(w_j / z, floor); then, if renorm, w_j divided by their new sum.
inline void eg_floor(double* w, int n, double z, double floor, bool renorm = true) {
    double z2 = 0;
    for (int j = 0; j < n; ++j) z2 += (w[j] = std::max(w[j] / z, floor));
    if (renorm)
        for (int j = 0; j < n; ++j) w[j] /= z2;
}
/// One step of exponentiated gradient on -log p for the linear mixture p = sum_j w_j q(j) (q(j): input j's probability
/// of what happened): w_j *= exp(eta min(q(j) / p, 30)), then eg_floor. The cap keeps one spike from moving a weight
/// too far (§13.15's fix of AdaGrad's spike).
template <class Q> inline void eg_step(double* w, int n, Q q, double p, double eta, double floor) {
    const double pp = std::max(p, 1e-300);
    eg_floor(w, n, reweight(w, n, [&](int j) { return std::exp(eta * std::min(q(j) / pp, 30.0)); }), floor);
}

/// Weight sets on the simplex, `sets` of `inputs` each, every set starting at the last input (the content mixer's
/// online final layer, OnlineFinal, and its replica in the full mix). predict gives sum_j w_j q(j); learn takes eg_step.
class EgSets {
 public:
    EgSets() = default;
    EgSets(int sets, int inputs) : n_(inputs), w_(static_cast<std::size_t>(sets) * inputs, 0.01 / (1.0 + 0.01 * (inputs - 1))) {
        for (int g = 0; g < sets; ++g) w_[static_cast<std::size_t>(g) * inputs + (inputs - 1)] = 1.0 / (1.0 + 0.01 * (inputs - 1));
    }
    template <class Q> double predict(int set, Q q) const {
        const double* w = &w_[static_cast<std::size_t>(set) * n_];
        double p = 0;
        for (int j = 0; j < n_; ++j) p += w[j] * q(j);
        return p;
    }
    template <class Q> void learn(int set, Q q, double p, double eta) { eg_step(&w_[static_cast<std::size_t>(set) * n_], n_, q, p, eta, 1e-7); }
    std::size_t bytes() const { return w_.size() * 8; }

 private:
    int n_ = 0;
    std::vector<double> w_;
};

/// The chain's stages: one pair (lambda, 1 - lambda) a bucket, starting at (0.05, 0.95). The stage mixes
/// lerp(lambda, pe, prev); learn takes eg_step on that mixture with a floor of 1e-4.
struct EgLambdas {
    std::vector<double> w;  // [2 b]: lambda, [2 b + 1]: 1 - lambda
    double eta = 0.02;
    void init(int nb, double e = 0.02) {
        w.resize(2 * static_cast<std::size_t>(nb));
        for (std::size_t b = 0; b < w.size(); b += 2) w[b] = 0.05, w[b + 1] = 0.95;
        eta = e;
    }
    double lam(std::size_t b) const { return w[2 * b]; }
    /// pe, prev: the expert's and the chain's probability of what happened; p: the mixture's.
    void learn(std::size_t b, double pe, double prev, double p, double et) {
        eg_step(&w[2 * b], 2, [&](int j) { return j ? prev : pe; }, p, et, 1e-4);
    }
};

/// Exponentiated gradient with signed weights (EG+-, Kivinen and Warmuth): w = w+ - w-, sum (w+ + w-) = U. For each j
/// with mask[j]: e = clamp(eta grad(j), -30, 30), w+_j *= exp(-e), w-_j *= exp(e); then every weight is scaled back to
/// the total U, with a floor of 1e-12.
template <class G> inline void eg_pm(double* wp, double* wm, int J, const char* mask, double eta, double U, G grad) {
    double tot = 0;
    for (int j = 0; j < J; ++j) {
        if (!mask[j]) continue;
        const double e = std::clamp(eta * grad(j), -30.0, 30.0);
        wp[j] *= std::exp(-e), wm[j] *= std::exp(e);
        tot += wp[j] + wm[j];
    }
    if (tot <= 0) return;
    const double f = U / tot;
    for (int j = 0; j < J; ++j) wp[j] = std::max(wp[j] * f, 1e-12), wm[j] = std::max(wm[j] * f, 1e-12);
}

/// Exponential weights over n predictors with Herbster-Warmuth fixed share: p = sum w_i p_i; then
/// w_i *= p_i(outcome)^eta, normalise, w := (1 - a) w + a / n. A mixture over learning rates, or over arms.
struct FixedShare {
    std::vector<double> w;
    double eta = 1, a = 0;
    void init(int n, double e, double share) {
        w.assign(static_cast<std::size_t>(n), 1.0 / n);
        eta = e, a = share;
    }
    double mix(const double* p) const {
        double s = 0;
        for (std::size_t i = 0; i < w.size(); ++i) s += w[i] * p[i];
        return s;
    }
    hyper::Setting *heta = nullptr, *ha = nullptr;  // T1b: eta and the share read from moving settings (nullptr: fixed)
    /// q: each predictor's probability of what happened.
    void update(const double* q) {
        const double e = heta ? heta->value() : eta, s = ha ? ha->value() : a;
        const double z = reweight(w.data(), static_cast<int>(w.size()), [&](int i) { return std::pow(std::max(q[i], 1e-300), e); });
        const double n = static_cast<double>(w.size());
        for (double& v : w) v = (1 - s) * std::max(v / z, 1e-300) + s / n;
    }
};

// ------------------------------------------------------------------------------------------------ geometric mixing

/// P_i = exp(s_i - max s) / sum: the normaliser of geometric (log-linear) mixing. s and P may be the same array.
inline void softmax(const double* s, int n, double* P) {
    double mx = -1e300;
    for (int i = 0; i < n; ++i) mx = std::max(mx, s[i]);
    double z = 0;
    for (int i = 0; i < n; ++i) z += (P[i] = std::exp(s[i] - mx));
    for (int i = 0; i < n; ++i) P[i] /= z;
}
/// init + sum_i P_i f(i), added in order. E_P[f] - f(target) is the log-loss gradient of a geometric mixture's weight
/// on the input f.
template <class F> inline double expect(const double* P, int n, F f, double init = 0) {
    double e = init;
    for (int i = 0; i < n; ++i) e += P[i] * f(i);
    return e;
}

/// Byte-level geometric mixing: p(s) proportional to exp(sum_j w_j log D_j(s)); one weight vector per context set,
/// learned by gradient on -log p(y): dw_j = -lr (E_p[log D_j] - log D_j(y)), lr = lr0 / (1 + decay t) with t the set's
/// updates. Starts at the input init (weight 1, the rest 0).
class GeoMixer {
 public:
    GeoMixer(int inputs, int sets, int init, double lr, double decay)
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
        softmax(z, 256, P);
        P_ = P;
    }
    void update(unsigned y) {
        double* w = &w_[static_cast<std::size_t>(set_) * n_];
        unsigned& t = t_[static_cast<std::size_t>(set_)];
        const double lr = (h_ ? s_ * h_->value() : lr_) / (1.0 + decay_ * t);
        ++t;
        if (h_) {  // T1b: the one-step hypergradient, d loss / d log lr = -lr (d_t . d_last) over this set's weights
            double* g = &g_[static_cast<std::size_t>(set_) * n_];
            double dot = 0;
            for (int j = 0; j < n_; ++j) {
                const double d = L_[j] ? expect(P_, 256, [&](int s) { return L_[j][s]; }) - L_[j][y] : 0.0;
                dot += d * g[j];
                g[j] = d;
                if (L_[j]) w[j] -= lr * d;
            }
            h_->hg(-lr * dot);
            return;
        }
        for (int j = 0; j < n_; ++j)
            if (L_[j]) w[j] -= lr * (expect(P_, 256, [&](int s) { return L_[j][s]; }) - L_[j][y]);
    }
    /// T1b (hyper_adapt.hpp): the step read from a moving setting times s (nullptr: the fixed step).
    void set_hyper(hyper::Setting* h, double s) {
        h_ = h, s_ = s;
        if (h_) g_.assign(w_.size(), 0.0);
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
    hyper::Setting* h_ = nullptr;  // T1b
    double s_ = 1;
    std::vector<double> g_;  // T1b: each set's last gradient
};

// ------------------------------------------------------------------------------------------------ logistic mixing

/// The probability of 1 for each of the byte's 8 bits along the path of y, from a distribution D (any scale): bit j's
/// probability is the mass of the partial byte's 1-branch over the partial byte's mass; 0.5 when it has no mass.
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

/// PAQ8's gated logistic mixer, two layers. Layer 1: several weight sets (of W: double, or float to halve the memory),
/// each chosen by its own small context, each mixing the stretched inputs into squash(w . x). Layer 2: l2_sets weight
/// sets, one chosen by its own context (the bit position, or the partial byte), mixing layer 1's stretched outputs.
/// Each layer learns from its own error: w += lr err x, lr = lr0 / (1 + decay t), t the set's updates (decay 0: lr0).
/// Layer-1 sets start with weight 1 on init_input, layer 2 at 1/m.
template <class W> class LogisticMixer {
 public:
    LogisticMixer(int inputs, std::vector<int> set_sizes, int init_input, double lr1, double lr2, double decay, double eps,
                  int l2_sets = 8)
        : n_(inputs), m_(static_cast<int>(set_sizes.size())), lr1_(lr1), lr2_(lr2), decay_(decay), eps_(eps) {
        for (int z : set_sizes) {
            std::vector<W> w(static_cast<std::size_t>(z) * n_, W(0));
            for (int c = 0; c < z; ++c) w[static_cast<std::size_t>(c) * n_ + init_input] = W(1);
            w1_.push_back(std::move(w));
            if (decay_ != 0) t1_.emplace_back(static_cast<std::size_t>(z), 0u);
        }
        w2_.assign(static_cast<std::size_t>(l2_sets) * m_, 1.0 / m_);
        if (decay_ != 0) t2_.assign(static_cast<std::size_t>(l2_sets), 0u);
        ctx_.resize(static_cast<std::size_t>(m_));
        p1_.resize(static_cast<std::size_t>(m_));
        s1_.resize(static_cast<std::size_t>(m_));
    }
    int inputs() const { return n_; }
    /// x: n stretched inputs; ctx: one context per layer-1 set; c2: layer 2's context. Returns P(bit = 1).
    double p(const double* x, const int* ctx, int c2) {
        x_ = x;
        c2_ = c2;
        double dot2 = 0;
        for (int k = 0; k < m_; ++k) {
            ctx_[k] = ctx[k];
            const W* w = &w1_[k][static_cast<std::size_t>(ctx[k]) * n_];
            double d = 0;
            for (int i = 0; i < n_; ++i) d += w[i] * x[i];
            p1_[k] = squash(d);
            s1_[k] = stretch_eps(p1_[k], eps_);
            dot2 += w2_[static_cast<std::size_t>(c2) * m_ + k] * s1_[k];
        }
        return p2_ = squash(dot2);
    }
    void update(int bit) {
        if (h1_ || h2_) return update_hyper(bit);
        for (int k = 0; k < m_; ++k) {
            const double err = (bit - p1_[k]) * rate(lr1_, t1_.empty() ? nullptr : &t1_[k][static_cast<std::size_t>(ctx_[k])]);
            W* w = &w1_[k][static_cast<std::size_t>(ctx_[k]) * n_];
            for (int i = 0; i < n_; ++i) w[i] += static_cast<W>(err * x_[i]);
        }
        const double err = (bit - p2_) * rate(lr2_, t2_.empty() ? nullptr : &t2_[static_cast<std::size_t>(c2_)]);
        for (int k = 0; k < m_; ++k) w2_[static_cast<std::size_t>(c2_) * m_ + k] += err * s1_[k];
    }
    /// T1b (hyper_adapt.hpp): the two steps read from moving settings times s1 and s2 (nullptr: that step stays fixed).
    /// Each weight set then keeps its last gradient for the one-step hypergradient.
    void set_hyper(hyper::Setting* h1, double s1, hyper::Setting* h2, double s2) {
        h1_ = h1, hs1_ = s1, h2_ = h2, hs2_ = s2;
        if (h1_)
            for (const auto& w : w1_) g1_.emplace_back(w.size(), W(0));
        if (h2_) g2_.assign(w2_.size(), 0.0);
    }
    std::size_t bytes() const {
        std::size_t b = w2_.size() * 8;
        for (const auto& w : w1_) b += w.size() * sizeof(W);
        return b;
    }

 private:
    // T1b: update() with moving steps; the hypergradient of a step is -rate (g_t . g_last), g = err x, per weight set.
    void update_hyper(int bit) {
        for (int k = 0; k < m_; ++k) {
            const double r = rate(h1_ ? hs1_ * h1_->value() : lr1_, t1_.empty() ? nullptr : &t1_[k][static_cast<std::size_t>(ctx_[k])]);
            const double e = bit - p1_[k], err = e * r;
            const std::size_t at = static_cast<std::size_t>(ctx_[k]) * n_;
            W* w = &w1_[k][at];
            if (h1_) {
                W* g = &g1_[static_cast<std::size_t>(k)][at];
                double dot = 0;
                for (int i = 0; i < n_; ++i) {
                    const double gi = e * x_[i];
                    dot += gi * static_cast<double>(g[i]);
                    g[i] = static_cast<W>(gi);
                }
                h1_->hg(-r * dot);
            }
            for (int i = 0; i < n_; ++i) w[i] += static_cast<W>(err * x_[i]);
        }
        const double r = rate(h2_ ? hs2_ * h2_->value() : lr2_, t2_.empty() ? nullptr : &t2_[static_cast<std::size_t>(c2_)]);
        const double e = bit - p2_, err = e * r;
        double* w2 = &w2_[static_cast<std::size_t>(c2_) * m_];
        if (h2_) {
            double* g = &g2_[static_cast<std::size_t>(c2_) * m_];
            double dot = 0;
            for (int k = 0; k < m_; ++k) {
                const double gk = e * s1_[k];
                dot += gk * g[k];
                g[k] = gk;
            }
            h2_->hg(-r * dot);
        }
        for (int k = 0; k < m_; ++k) w2[k] += err * s1_[k];
    }
    // lr0 / (1 + decay t), then t + 1; with decay 0 there are no counters and the rate is lr0 (lr0 / 1, exactly).
    double rate(double lr0, unsigned* t) const {
        if (!t) return lr0;
        const double lr = lr0 / (1.0 + decay_ * *t);
        ++*t;
        return lr;
    }
    int n_, m_;
    double lr1_, lr2_, decay_, eps_;
    std::vector<std::vector<W>> w1_;
    std::vector<std::vector<unsigned>> t1_;
    std::vector<double> w2_;
    std::vector<unsigned> t2_;
    std::vector<int> ctx_;
    std::vector<double> p1_, s1_;
    const double* x_ = nullptr;
    int c2_ = 0;
    double p2_ = 0.5;
    hyper::Setting *h1_ = nullptr, *h2_ = nullptr;  // T1b
    double hs1_ = 1, hs2_ = 1;
    std::vector<std::vector<W>> g1_;  // T1b: each layer-1 set's last gradient
    std::vector<double> g2_;
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

}  // namespace cypha::online::mix
