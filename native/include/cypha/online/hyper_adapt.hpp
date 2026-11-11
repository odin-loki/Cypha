#pragma once

/// Tuned, but not pinned (GENERATION_PLAN §13.43, job T1b; docs/reports/TUNE_ADAPT.md).
///
/// Odin: "After you have tuned the experts set them to that but dont pin them to it. Set them an optimal amount back and
/// let the model teter to and fro from optimal to noise so that it learns and we get beneficial stochastics."
///
/// A continuous setting (a learning rate, a count limit, an SSE or mixing rate, a temperature) is held here in log
/// units, x = log theta, and its users read theta = exp(x) each time they use it:
///   - start: "an optimal amount back", a fraction b of the way from T1's tuned value theta* toward the old default
///     (in log units);
///   - to and fro (noise on): x follows a mean-reverting random walk (Ornstein-Uhlenbeck) around its centre c,
///     x += kappa (c - x) + sigma sqrt(kappa (2 - kappa)) N(0, 1) each byte, so it wobbles by about sigma and never
///     settles; without noise x = c;
///   - learning where to sit (adapt on): every block of bytes the centre moves against the loss's gradient with respect
///     to x, normalised by its running size (c -= beta g / rms(g)). For learning rates g is the hypergradient
///     (Baydin et al. 2018): the step's own derivative, -rate (g_t . g_last), where g_last is the gradient of the last
///     update of the same weight set (the one-step truncated hypergradient). For the others g is a one-sided finite
///     difference on the recent loss: a base block at c, then a probe block at c + delta, g = (L_probe - L_base) / delta.
///     The centre stays within theta* x/÷ 8.
/// A Group belongs to one thread (the mixers', the n-gram's, the similar-context expert's) and is stepped by it once a
/// byte with that thread's own loss, so runs are deterministic for a fixed seed. Off (no --hyper-adapt), nothing here
/// is made and every user reads its fixed value as before.
///
/// Author: Odin Loch

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cypha::online::hyper {

/// --hyper-adapt k=v,...: mode=adapt|noise|both (A1, A2, A3), b, beta, kappa, delta, block, seed, cap (log units of the
/// centre's range), only=name:name (the settings that move; default all), sigma.NAME=S (log units; default 0.2).
struct Spec {
    bool adapt = false, noise = false;
    double b = 0, beta = 0.01, kappa = 1.0 / 1024, delta = 0.2, cap = 2.0794415416798357;  // log 8
    int block = 1024;
    std::uint64_t seed = 1;
    std::vector<std::string> only;
    std::vector<std::pair<std::string, double>> sigma;
    bool on() const { return adapt || noise; }
    bool parse(const std::string& s) {
        for (std::size_t at = 0; at < s.size();) {
            std::size_t c = s.find(',', at);
            if (c == std::string::npos) c = s.size();
            const std::string f = s.substr(at, c - at);
            at = c + 1;
            const std::size_t eq = f.find('=');
            if (eq == std::string::npos) return false;
            const std::string k = f.substr(0, eq), v = f.substr(eq + 1);
            const double x = std::atof(v.c_str());
            if (k == "mode") {
                if (v == "adapt") adapt = true, noise = false;
                else if (v == "noise") adapt = false, noise = true;
                else if (v == "both") adapt = noise = true;
                else return false;
            } else if (k == "b") b = x;
            else if (k == "beta") beta = x;
            else if (k == "kappa") kappa = x;
            else if (k == "delta") delta = x;
            else if (k == "cap") cap = x;
            else if (k == "block") block = std::max(16, static_cast<int>(x));
            else if (k == "seed") seed = std::strtoull(v.c_str(), nullptr, 10);
            else if (k == "only") {
                for (std::size_t p = 0; p <= v.size();) {
                    std::size_t q = v.find(':', p);
                    if (q == std::string::npos) q = v.size();
                    if (q > p) only.push_back(v.substr(p, q - p));
                    p = q + 1;
                }
            } else if (k.rfind("sigma.", 0) == 0) sigma.emplace_back(k.substr(6), x);
            else return false;
        }
        return true;
    }
    bool wants(const std::string& name) const {
        return on() && (only.empty() || std::find(only.begin(), only.end(), name) != only.end());
    }
    double sigma_of(const std::string& name) const {
        for (const auto& [n, v] : sigma)
            if (n == name) return v;
        return 0.2;
    }
};

/// One setting. Users read value(); learning-rate users add their hypergradient terms with hg().
class Setting {
 public:
    Setting(std::string name, bool hypergrad, double star, double def, const Spec& s)
        : name_(std::move(name)), hg_(hypergrad), star_(std::log(star)), sigma_(s.sigma_of(name_)) {
        const double d = std::log(def > 0 ? def : star);
        x_ = star_ + s.b * (d - star_);  // the start: an optimal amount back
        c_ = s.adapt ? x_ : star_;       // the centre: learned from the start (A1, A3), or theta* fixed (A2)
        val_ = std::exp(x_);
    }
    double value() const { return val_; }
    /// A learning rate's one-step hypergradient term, d loss / d log(theta) (see the header).
    void hg(double d) { acc_ += d; }
    const std::string& name() const { return name_; }

 private:
    friend class Group;
    std::string name_;
    bool hg_;
    double star_, sigma_, c_, x_, val_;
    double acc_ = 0, v_ = 0, moved_ = 0;
    int updates_ = 0;
    double cmin_ = 1e300, cmax_ = -1e300, xmin_ = 1e300, xmax_ = -1e300;
};

class Group {
 public:
    Group(const Spec& s, std::string name, std::uint64_t id) : s_(s), name_(std::move(name)), rng_(s.seed * 1000003u + id) {}
    /// A setting this group moves, or nullptr when the spec leaves it fixed (then its user keeps its own value).
    Setting* add(const std::string& name, bool hypergrad, double star, double def) {
        if (!s_.wants(name) || !(star > 0)) return nullptr;
        for (const auto& h : set_)
            if (h->name_ == name) return h.get();  // shared by every user of the same setting
        set_.push_back(std::make_unique<Setting>(name, hypergrad, star, def, s_));
        Setting* h = set_.back().get();
        if (!hypergrad) fd_.push_back(h);
        return h;
    }
    bool empty() const { return set_.empty(); }
    /// After each byte, by the owning thread: loss is that byte's bits under this group's own loss.
    void byte(double loss) {
        if (set_.empty()) return;
        blk_loss_ += loss;
        if (++in_blk_ == s_.block) end_block();
        for (auto& h : set_) {
            if (s_.noise) h->x_ += s_.kappa * (h->c_ - h->x_) + h->sigma_ * std::sqrt(s_.kappa * (2 - s_.kappa)) * nd_(rng_);
            else h->x_ = h->c_;
            const double x = h->x_ + (probe_ == h.get() ? s_.delta : 0.0);  // a probe block reads c + delta (+ noise)
            h->val_ = std::exp(x);
            h->xmin_ = std::min(h->xmin_, x), h->xmax_ = std::max(h->xmax_, x);
        }
    }
    /// One line a setting: tuned, start, final centre, centre's range, value's range, centre updates (theta, not logs).
    void report(std::FILE* f) const {
        for (const auto& h : set_)
            std::fprintf(f,
                         "hyper %s %s %s star %.6g centre %.6g (range %.6g..%.6g) value %.6g (range %.6g..%.6g) moved %.4g updates "
                         "%d sigma %.3g\n",
                         name_.c_str(), h->name_.c_str(), h->hg_ ? "hypergrad" : "fd", std::exp(h->star_), std::exp(h->c_),
                         std::exp(std::min(h->cmin_, h->c_)), std::exp(std::max(h->cmax_, h->c_)), h->val_, std::exp(h->xmin_),
                         std::exp(h->xmax_), h->moved_, h->updates_, h->sigma_);
    }

 private:
    void step_centre(Setting& h, double g) {
        if (!s_.adapt || !std::isfinite(g)) return;
        h.v_ = h.updates_ ? 0.99 * h.v_ + 0.01 * g * g : g * g;
        ++h.updates_;
        if (h.v_ <= 0) return;
        const double d = -s_.beta * g / std::sqrt(h.v_);
        const double c0 = h.c_;
        h.c_ = std::clamp(h.c_ + d, h.star_ - s_.cap, h.star_ + s_.cap);
        h.moved_ += std::abs(h.c_ - c0);
        h.cmin_ = std::min(h.cmin_, h.c_), h.cmax_ = std::max(h.cmax_, h.c_);
    }
    void end_block() {
        const double L = blk_loss_ / s_.block;
        blk_loss_ = 0, in_blk_ = 0;
        for (auto& h : set_)
            if (h->hg_) {
                step_centre(*h, h->acc_ / s_.block);
                h->acc_ = 0;
            }
        if (fd_.empty()) return;
        // the finite differences: settings take turns, a base block then a probe block each
        if (!probe_) {
            base_ = L;
            probe_ = fd_[turn_ % fd_.size()];
        } else {
            step_centre(*probe_, (L - base_) / s_.delta);
            probe_ = nullptr;
            ++turn_;
        }
    }
    const Spec& s_;
    std::string name_;
    std::vector<std::unique_ptr<Setting>> set_;
    std::vector<Setting*> fd_;
    Setting* probe_ = nullptr;
    std::size_t turn_ = 0;
    double base_ = 0, blk_loss_ = 0;
    int in_blk_ = 0;
    std::mt19937_64 rng_;
    std::normal_distribution<double> nd_{0.0, 1.0};
};

/// The run's spec (set once by context8k before any group is made; off by default).
inline Spec& spec() {
    static Spec s;
    return s;
}
/// The mixers' group while context8k builds its full mix and meta mix (nullptr otherwise: replays, other commands).
inline Group*& mixers() {
    static Group* g = nullptr;
    return g;
}

}  // namespace cypha::online::hyper
