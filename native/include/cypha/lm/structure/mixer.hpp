#pragma once

/// The meta-mixer over K5's structure experts (docs/reports/K5_STRUCTURE.md). A softmax ranker over the candidates of
/// one sentence: s_i = sum_j w_j z_ij (z: features over their standard deviation), P(i) = softmax(s). Learned online by
/// exponentiated gradient with signed weights (EG+-: w = w+ - w-, sum (w+ + w-) = U) on log loss, like the content
/// mixer's final layer. A weight set is chosen by a small context (PAQ): none, the sentence's index x the prompt's
/// length, or the previous sentence's opening type. A second layer mixes several rankers' candidate distributions
/// linearly, its weights learned by exponentiated gradient on log loss; with rankers at several learning rates it is a
/// mixture over learning rates (MetaGrad style).
///
/// Author: Odin Loch

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cypha::cyphalm::structure {

// One sentence's candidates: z is n x J (row-major), target the index of the one to predict, sel the weight-set index
// under each selector (0: one set; 1: index x prompt length, 12 sets; 2: previous opening type, 7 sets).
struct Item {
    int n = 0;
    std::vector<float> z;
    int target = 0;
    int sel[3] = {0, 0, 0};
};
inline constexpr int kSelSets[3] = {1, 12, 7};

class Ranker {
 public:
    Ranker() = default;
    Ranker(int J, int selector, double U, double eta, std::vector<char> mask)
        : J_(J), sel_(selector), U_(U), eta_(eta), mask_(std::move(mask)) {
        const int S = kSelSets[sel_];
        wp_.assign(static_cast<std::size_t>(S * J_), 0), wm_.assign(static_cast<std::size_t>(S * J_), 0);
        int on = 0;
        for (const char m : mask_) on += m != 0;
        for (int s = 0; s < S; ++s)
            for (int j = 0; j < J_; ++j)
                if (mask_[static_cast<std::size_t>(j)]) wp_[idx(s, j)] = wm_[idx(s, j)] = U_ / (2.0 * std::max(on, 1));
    }
    int selector() const { return sel_; }
    double eta() const { return eta_; }
    double U() const { return U_; }
    double w(int set, int j) const { return wp_[idx(set, j)] - wm_[idx(set, j)]; }
    // A candidate's score (z: its J features).
    double score(int set, const float* z) const {
        double s = 0;
        for (int j = 0; j < J_; ++j) s += w(set, j) * z[j];
        return s;
    }
    void dist(const Item& it, std::vector<double>& P) const {
        const int set = it.sel[sel_];
        P.resize(static_cast<std::size_t>(it.n));
        double mx = -1e300;
        for (int i = 0; i < it.n; ++i) mx = std::max(mx, P[static_cast<std::size_t>(i)] = score(set, it.z.data() + static_cast<std::size_t>(i * J_)));
        double z = 0;
        for (double& p : P) z += (p = std::exp(p - mx));
        for (double& p : P) p /= z;
    }
    void update(const Item& it, const std::vector<double>& P) {
        const int set = it.sel[sel_];
        double tot = 0;
        for (int j = 0; j < J_; ++j) {
            if (!mask_[static_cast<std::size_t>(j)]) continue;
            double g = -it.z[static_cast<std::size_t>(it.target * J_ + j)];
            for (int i = 0; i < it.n; ++i) g += P[static_cast<std::size_t>(i)] * it.z[static_cast<std::size_t>(i * J_ + j)];
            const double e = std::clamp(eta_ * g, -30.0, 30.0);
            wp_[idx(set, j)] *= std::exp(-e), wm_[idx(set, j)] *= std::exp(e);
            tot += wp_[idx(set, j)] + wm_[idx(set, j)];
        }
        if (tot <= 0) return;
        const double f = U_ / tot;
        for (int j = 0; j < J_; ++j) wp_[idx(set, j)] = std::max(wp_[idx(set, j)] * f, 1e-12), wm_[idx(set, j)] = std::max(wm_[idx(set, j)] * f, 1e-12);
    }
    void write(std::ostream& o) const {
        o << "ranker " << sel_ << " " << eta_ << " " << U_ << " " << J_;
        for (const char m : mask_) o << " " << int(m);
        o << "\n";
        o.precision(17);
        for (std::size_t k = 0; k < wp_.size(); ++k) o << wp_[k] << " " << wm_[k] << (k + 1 == wp_.size() ? "\n" : " ");
    }
    bool read(std::istream& in) {
        std::string tag;
        if (!(in >> tag) || tag != "ranker" || !(in >> sel_ >> eta_ >> U_ >> J_) || sel_ < 0 || sel_ > 2 || J_ <= 0) return false;
        mask_.resize(static_cast<std::size_t>(J_));
        for (auto& m : mask_) {
            int x;
            in >> x;
            m = static_cast<char>(x);
        }
        const std::size_t n = static_cast<std::size_t>(kSelSets[sel_] * J_);
        wp_.resize(n), wm_.resize(n);
        for (std::size_t k = 0; k < n; ++k) in >> wp_[k] >> wm_[k];
        return static_cast<bool>(in);
    }

 private:
    std::size_t idx(int s, int j) const { return static_cast<std::size_t>(s * J_ + j); }
    int J_ = 0, sel_ = 0;
    double U_ = 4, eta_ = 0.01;
    std::vector<char> mask_;
    std::vector<double> wp_, wm_;
};

// One ranker, or several mixed by a second layer.
class MetaMixer {
 public:
    std::vector<Ranker> r;
    std::vector<double> v;  // second-layer weights (empty: one ranker)
    double eta2 = 0.02;
    std::vector<double> sd;  // each feature's standard deviation (z = f / sd)

    void init_layer2() { v.assign(r.size(), 1.0 / static_cast<double>(r.size())); }
    void dist(const Item& it, std::vector<double>& P) const {
        if (r.size() == 1) return r[0].dist(it, P);
        P.assign(static_cast<std::size_t>(it.n), 0.0);
        std::vector<double> q;
        for (std::size_t m = 0; m < r.size(); ++m) {
            r[m].dist(it, q);
            for (int i = 0; i < it.n; ++i) P[static_cast<std::size_t>(i)] += v[m] * q[static_cast<std::size_t>(i)];
        }
    }
    // -log2 P(target) before learning from it (the prequential loss), then learns.
    double learn(const Item& it) {
        std::vector<double> P;
        if (r.size() == 1) {
            r[0].dist(it, P);
            const double l = -std::log2(std::max(P[static_cast<std::size_t>(it.target)], 1e-300));
            r[0].update(it, P);
            return l;
        }
        std::vector<std::vector<double>> qs(r.size());
        double pt = 0;
        for (std::size_t m = 0; m < r.size(); ++m) r[m].dist(it, qs[m]), pt += v[m] * qs[m][static_cast<std::size_t>(it.target)];
        pt = std::max(pt, 1e-300);
        double z = 0;
        for (std::size_t m = 0; m < r.size(); ++m) z += (v[m] *= std::exp(std::min(30.0, eta2 * qs[m][static_cast<std::size_t>(it.target)] / pt)));
        for (double& x : v) x = std::max(x / z, 1e-6);
        for (std::size_t m = 0; m < r.size(); ++m) r[m].update(it, qs[m]);
        return -std::log2(pt);
    }
    double loss(const Item& it) const {
        std::vector<double> P;
        dist(it, P);
        return -std::log2(std::max(P[static_cast<std::size_t>(it.target)], 1e-300));
    }
    void write(std::ostream& o) const {
        o.precision(17);
        o << "sd";
        for (const double x : sd) o << " " << x;
        o << "\nlayer2 " << eta2 << " " << r.size() << "\n";
        for (const auto& x : r) x.write(o);
        o << "v";
        for (const double x : v) o << " " << x;
        o << "\n";
    }
    bool read(std::istream& in, int J) {
        std::string tag;
        sd.assign(static_cast<std::size_t>(J), 1.0);
        if (!(in >> tag) || tag != "sd") return false;
        for (double& x : sd) in >> x;
        std::size_t n = 0;
        if (!(in >> tag) || tag != "layer2" || !(in >> eta2 >> n) || n == 0 || n > 1000) return false;
        r.resize(n);
        for (auto& x : r)
            if (!x.read(in)) return false;
        if (!(in >> tag) || tag != "v") return false;
        v.clear();
        std::string line;
        std::getline(in, line);
        std::istringstream ls(line);
        for (double x; ls >> x;) v.push_back(x);
        return n == 1 || v.size() == n;
    }
};

}  // namespace cypha::cyphalm::structure
