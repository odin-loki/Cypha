#pragma once

/// Meta mixing over the full mix (docs/reports/META_MIXING.md, GENERATION_PLAN §13.32).
///
/// A read-only hook (fullmix::StepHook) on FullMixer: it sees every byte's inputs and every full-mix arm's prediction of
/// each bit before any arm learns it, and runs three kinds of meta mixer beside the arms. Nothing here changes a
/// full-mix arm; with the hook off, context8k's output is main's, line for line.
///   M1  PAQ8's two layers, wider: many first-layer weight sets over the bitwise mixer's inputs, each chosen by its
///       own small context (bit position x order-1 byte, x match length, x InfiniGram bucket, x byte class, x the LSTM's
///       confidence, ...), and a second layer joining their outputs; then SSE (M1s), then mixed per bit with Bs (M1Bs),
///       as ABs mixes As and Bs.
///   M2  A mixture over learning rates (MetaGrad / Squint): the bitwise mixer A, and the geometric mixer B, each run at
///       eta x {1/4, 1/2, 1, 2, 4} (the 1x members are the full mix's own A and B); a master combines each family
///       (exponential weights with fixed share, or a small logistic mixer); then SSE (M2As, M2Gs), mixed per bit (M2ABs).
///   M3  A master over the arms (Fs, As, Bs, ABs, M1s, M1Bs, M2As, M2Gs, M2ABs): Herbster-Warmuth fixed share, which
///       tracks the arm that is best now (M3), and a logistic mixer over the same arms (M3L).
/// Every part learns online from the first byte; every bit is predicted before it is learned. The mixers are
/// mix_core.hpp's (MERGE_MIXERS.md): M1 is its LogisticMixer with float32 layer-1 weights and a second layer chosen by
/// its own context, the masters its FixedShare.

#include "cypha/online/full_mixer.hpp"
#include "cypha/online/mix_core.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace cypha::online::metamix {

namespace fx = cypha::online::fullmix;
using Mixer2 = mix::LogisticMixer<float>;  // PAQ8's mixer, layer-1 weights as float32 (half the memory of A's)
using mix::FixedShare;

/// k=v,k=v (lists inside a value use ':'). Defaults are the registered ones, chosen on the holdout's first half by
/// replay (META_MIXING.md, "Tuning"; T33). Before tuning: sel 0-10, l2 0, m1lr1 = lr1, m1r 0, m3a 1e-3, headline M3L.
struct Params {
    std::string sel = "0:1:2:3:4:5:6:7:8:9:10:11:12:13";  // M1's first-layer selectors (MetaMixer::kSelName)
    int l2 = 1;                                   // M1's second layer: 0 weights by bit position, 1 by partial byte
    double m1lr1 = 0.002, m1lr2 = -1;             // M1's rates (-1: the full mix's lr1, lr2)
    int m1r = 1;                                  // 1: M1 at each of M2's rate multiples, joined by M2A's kind of master
    std::string scales = "0.25:0.5:1:2:4";        // M2's rate multiples (1 must be present: the full mix's own A and B)
    int m2m = 0;                                  // M2A's master: 0 exponential weights with fixed share, 1 logistic
    double m2eta = 1, m2share = 1e-3;             // M2A's exponential-weights master (per bit)
    double m2geta = 1, m2gshare = 1e-3;           // M2G's master (per byte)
    double m3a = 1e-4, m3eta = 1;                 // M3: fixed share per bit
    int m3lctx = 0;                               // M3L: weight sets by 0 bit position, 1 partial byte
    double m3llr1 = -1, m3llr2 = -1;              // M3L's rates (-1: the full mix's lr1, lr2)
    int m3linit = 3;                              // M3L starts at this arm of M3's list (3: ABs)
    std::string headline = "M1Gs";
    bool parse(const std::string& s) {
        return mix::parse_kv(s, [&](const std::string& k, const std::string& v) {
            const double x = std::atof(v.c_str());
            if (k == "sel") sel = v;
            else if (k == "scales") scales = v;
            else if (k == "headline") headline = v;
            else if (k == "l2") l2 = static_cast<int>(x);
            else if (k == "m1lr1") m1lr1 = x;
            else if (k == "m1lr2") m1lr2 = x;
            else if (k == "m1r") m1r = static_cast<int>(x);
            else if (k == "m2m") m2m = static_cast<int>(x);
            else if (k == "m2eta") m2eta = x;
            else if (k == "m2share") m2share = x;
            else if (k == "m2geta") m2geta = x;
            else if (k == "m2gshare") m2gshare = x;
            else if (k == "m3a") m3a = x;
            else if (k == "m3eta") m3eta = x;
            else if (k == "m3lctx") m3lctx = static_cast<int>(x);
            else if (k == "m3llr1") m3llr1 = x;
            else if (k == "m3llr2") m3llr2 = x;
            else if (k == "m3linit") m3linit = static_cast<int>(x);
            else return false;
            return true;
        });
    }
    /// The settings' sense, before any run.
    bool check(std::string& err) const {
        bool one = false;
        for (double v : list(scales)) one |= v == 1.0;
        if (!one || list(scales).size() > 8) return err = "scales: up to 8, one of them 1", false;
        const auto s = list(sel);
        if (s.empty()) return err = "sel: at least one selector", false;
        for (double v : s)
            if (v < 0 || v > 13) return err = "sel: selectors are 0-13", false;
        static const char* arms[] = {"M1", "M1s", "M1Bs", "M2A", "M2As", "M2Gs", "M2ABs", "M3", "M3L", "M3Ls", "M3s", "M1Gs"};
        if (std::find(std::begin(arms), std::end(arms), headline) == std::end(arms))
            return err = "headline: one of M1 M1s M1Bs M2A M2As M2Gs M2ABs M3 M3L M3Ls M3s M1Gs", false;
        return true;
    }
    static std::vector<double> list(const std::string& v) {
        std::vector<double> o;
        for (std::size_t p0 = 0; p0 < v.size();) {
            std::size_t p1 = v.find(':', p0);
            if (p1 == std::string::npos) p1 = v.size();
            o.push_back(std::atof(v.substr(p0, p1 - p0).c_str()));
            p0 = p1 + 1;
        }
        return o;
    }
};

class MetaMixer : public fx::StepHook {
 public:
    static constexpr int kArms = 12;
    static constexpr const char* kArmName[kArms] = {"M1", "M1s", "M1Bs", "M2A", "M2As", "M2Gs", "M2ABs", "M3", "M3L", "M3Ls", "M3s", "M1Gs"};
    static constexpr int kSel = 14;
    static constexpr const char* kSelName[kSel] = {
        "partial byte",           "InfiniGram bucket x bit",  "copy bucket x copy state x bit", "byte class x partial byte",
        "order-1 x partial byte", "match length x copy x bit", "InfiniGram bucket x partial byte", "class x class2 x bit",
        "LSTM confidence x bit",  "similar bucket x bit",     "KN bucket x bit",                "order-2 hash x partial byte",
        "order-3 hash x partial byte", "word hash x partial byte"};
    // M3's arms: the full mix's Fs, As, Bs, ABs, then M1s, M1Bs, M2As, M2Gs, M2ABs.
    static constexpr int kM3 = 9;

    MetaMixer(const fx::ChainSpec& cs, const fx::Params& P, const Params& M, std::size_t half) : P_(P), M_(M), half_(half) {
        const int ng = cs.ngram_orders > 0 ? 1 + cs.ngram_orders : 0;
        kAll_ = 16 + ng + 8;
        nall_ = kAll_ + 2;
        // ---- M1
        for (double v : Params::list(M.sel)) sel_.push_back(static_cast<int>(v));
        static const int sizes[kSel] = {256, 256, 256, 1280, 65536, 1024, 8192, 200, 128, 968, 512, 65536, 65536, 65536};
        std::vector<int> sz;
        for (int s : sel_) sz.push_back(sizes[s]);
        const double l1 = M.m1lr1 > 0 ? M.m1lr1 : P.lr1, l2r = M.m1lr2 > 0 ? M.m1lr2 : P.lr2;
        m1_ = std::make_unique<Mixer2>(nall_, sz, kAll_ - 1, l1, l2r, 0, P.eps, M.l2 ? 256 : 8);
        if (M.m1r)
            for (double sc : Params::list(M.scales))
                m1r_.push_back(sc == 1.0 ? nullptr : std::make_unique<Mixer2>(nall_, sz, kAll_ - 1, l1 * sc, l2r * sc, 0, P.eps, M.l2 ? 256 : 8));
        m1s_ = std::make_unique<fx::Sse>(P);
        m1b_ = std::make_unique<fx::BitMixer>(2, std::vector<int>{8}, 0, P.lr1, P.lr2, 0, P.eps);
        m1g_ = std::make_unique<fx::BitMixer>(2, std::vector<int>{8}, 0, P.lr1, P.lr2, 0, P.eps);
        // ---- M2
        scales_ = Params::list(M.scales);
        fsM1_.init(static_cast<int>(scales_.size()), M.m2eta, M.m2share);
        for (std::size_t i = 0; i < scales_.size(); ++i) {
            if (scales_[i] == 1.0) {
                center_ = static_cast<int>(i);
                m2a_.emplace_back(nullptr);
                m2g_.emplace_back(nullptr);
                continue;
            }
            const std::vector<int> sets = {256, 32 * 8, 8 * 4 * 8, 5 * 256};
            m2a_.push_back(std::make_unique<fx::BitMixer>(nall_, sets, kAll_ - 1, P.lr1 * scales_[i], P.lr2 * scales_[i], P.lr_decay, P.eps));
            m2g_.push_back(std::make_unique<mix::GeoMixer>(kAll_ + 1, 80, kAll_ - 1, P.lrb * scales_[i], P.lr_decay));
        }
        const int nr = static_cast<int>(scales_.size());
        fsA_.init(nr, M.m2eta, M.m2share);
        fsG_.init(nr, M.m2geta, M.m2gshare);
        m2am_ = std::make_unique<fx::BitMixer>(nr, std::vector<int>{8}, std::max(center_, 0), P.lr1, P.lr2, 0, P.eps);
        PG_.assign(static_cast<std::size_t>(nr), {});
        m2as_ = std::make_unique<fx::Sse>(P);
        m2gs_ = std::make_unique<fx::Sse>(P);
        m2ab_ = std::make_unique<fx::BitMixer>(2, std::vector<int>{8}, 0, P.lr1, P.lr2, 0, P.eps);
        // ---- M3
        fs3_.init(kM3, M.m3eta, M.m3a);
        m3l_ = std::make_unique<fx::BitMixer>(kM3, std::vector<int>{M.m3lctx ? 256 : 8}, std::clamp(M.m3linit, 0, kM3 - 1), M.m3llr1 > 0 ? M.m3llr1 : P.lr1,
                                              M.m3llr2 > 0 ? M.m3llr2 : P.lr2, 0, P.eps);  // starts at ABs
        m3ls_ = std::make_unique<fx::Sse>(P);
        m3s_ = std::make_unique<fx::Sse>(P);
    }
    void byte_begin(const fx::ByteIn& in, const double* const* L, int nL, int g, const double* PB, const double (*bits)[8],
                    int nb, int lstm_at) override {
        in_ = &in;
        lstm_at_ = lstm_at;
        (void)nb;
        (void)nL;
        bc1_ = mix::byte_class(static_cast<unsigned char>(in.prev));
        bc2_ = mix::byte_class(static_cast<unsigned char>(in.prev2));
        cb_ = cyphalm::copy_bucket(in.cn);
        int lg = 0;
        for (int n = in.ig_n + 1; n > 1 && lg < 15; n >>= 1) ++lg;
        mlen_ = lg;
        // M2G: the geometric mixer at each rate (the 1x member is the full mix's own B), mixed per byte
        const int nr = static_cast<int>(scales_.size());
        double PM[256] = {};
        for (int r = 0; r < nr; ++r) {
            double* Pr = PG_[static_cast<std::size_t>(r)].data();
            if (r == center_) std::copy(PB, PB + 256, Pr);
            else m2g_[static_cast<std::size_t>(r)]->dist(L, g, Pr);
            const double w = fsG_.w[static_cast<std::size_t>(r)];
            for (int s = 0; s < 256; ++s) PM[s] += w * Pr[s];
        }
        mix::path_bits(PM, in.y, bitsG_);
        lpG_ = std::log2(std::max(PM[in.y], 1e-300));
        (void)bits;
    }

    void bit(int j, int bit, unsigned c0, const double* xa, int nxa, const int* ctx, const double* arm) override {
        const fx::ByteIn& in = *in_;
        const double eps = P_.eps;
        auto cl = [eps](double p) { return std::clamp(p, eps, 1 - eps); };
        auto st = [eps](double p) { return mix::stretch_eps(p, eps); };
        (void)nxa;
        const double pFs = arm[0], pA = arm[1], pAs = arm[2], pBs = arm[5], pAB = arm[6];
        // ---- M1: the selectors' contexts
        int c1[kSel];
        for (std::size_t k = 0; k < sel_.size(); ++k) {
            int v = 0;
            switch (sel_[k]) {
                case 0: v = static_cast<int>(c0); break;
                case 1: v = ctx[1]; break;
                case 2: v = ctx[2]; break;
                case 3: v = ctx[3]; break;
                case 4: v = static_cast<int>(in.prev & 255u) * 256 + static_cast<int>(c0); break;
                case 5: v = (mlen_ * 8 + cb_) * 8 + j; break;
                case 6: v = std::min(in.ig_b, 31) * 256 + static_cast<int>(c0); break;
                case 7: v = (bc1_ * 5 + bc2_) * 8 + j; break;
                case 8: {
                    int q = 0, sg = 0;
                    if (lstm_at_ >= 0) {
                        const double x = xa[lstm_at_];
                        sg = x >= 0 ? 1 : 0;
                        q = std::min(static_cast<int>(std::abs(x) * 2), 7);
                    }
                    v = (sg * 8 + q) * 8 + j;
                    break;
                }
                case 9: v = std::min(in.sim_b, 120) * 8 + j; break;
                case 10: v = std::min(in.kn_b, 63) * 8 + j; break;
                case 11: {
                    const std::uint32_t key = (in.prev2 & 255u) << 16 | (in.prev & 255u) << 8 | (c0 & 255u);
                    v = static_cast<int>(static_cast<std::uint32_t>(key * 2654435761u) >> 16);
                    break;
                }
                case 12:  // the three previous bytes (the hook's own history) and the partial byte
                    v = static_cast<int>(static_cast<std::uint32_t>((h3_ * 0x2f0b3c49u) ^ ((c0 & 255u) * 0x9E3779B1u)) >> 16);
                    break;
                case 13:  // the current word's letters so far (0 after a non-letter) and the partial byte
                    v = static_cast<int>(static_cast<std::uint32_t>((hw_ * 0x6b43a9b5u) ^ ((c0 & 255u) * 0x9E3779B1u) ^ 0x5bd1e995u) >> 16);
                    break;
            }
            c1[k] = v;
        }
        const int c2 = M_.l2 ? static_cast<int>(c0) : j;
        double pM1 = cl(m1_->p(xa, c1, c2));
        double p1r[8];
        if (!m1r_.empty()) {
            for (std::size_t r = 0; r < m1r_.size(); ++r) p1r[r] = m1r_[r] ? cl(m1r_[r]->p(xa, c1, c2)) : pM1;
            pM1 = cl(fsM1_.mix(p1r));
        }
        const double pM1s = m1s_->p(pM1, in, c0, j);
        const int z0 = 0;
        const double x1b[2] = {st(pM1s), st(pBs)};
        const double pM1B = cl(m1b_->p(x1b, &z0, j));
        // ---- M2A: A at each rate (1x: the full mix's own), mixed by the master
        const int nr = static_cast<int>(scales_.size());
        double pr[8], xr[8];
        for (int r = 0; r < nr; ++r) {
            pr[r] = r == center_ ? pA : cl(m2a_[static_cast<std::size_t>(r)]->p(xa, ctx, j));
            xr[r] = st(pr[r]);
        }
        const double pM2A = cl(M_.m2m ? m2am_->p(xr, &z0, j) : fsA_.mix(pr));
        const double pM2As = m2as_->p(pM2A, in, c0, j);
        const double pM2Gs = m2gs_->p(cl(bitsG_[j]), in, c0, j);
        const double x2[2] = {st(pM2As), st(pM2Gs)};
        const double pM2AB = cl(m2ab_->p(x2, &z0, j));
        const double x1g[2] = {st(pM1s), st(pM2Gs)};
        const double pM1G = cl(m1g_->p(x1g, &z0, j));
        // ---- M3: over the arms
        const double a3[kM3] = {pFs, pAs, pBs, pAB, pM1s, pM1B, pM2As, pM2Gs, pM2AB};
        double x3[kM3];
        for (int i = 0; i < kM3; ++i) x3[i] = st(a3[i]);
        const double pM3 = cl(fs3_.mix(a3));
        const int c3 = M_.m3lctx ? static_cast<int>(c0) : j;
        const double pM3L = cl(m3l_->p(x3, &c3, j));
        const double pM3Ls = m3ls_->p(pM3L, in, c0, j);
        const double pM3s = m3s_->p(pM3, in, c0, j);
        // ---- score, then learn
        auto lg = [bit](double p) { return std::log2(bit ? p : 1 - p); };
        lp_[0] += lg(pM1), lp_[1] += lg(pM1s), lp_[2] += lg(pM1B), lp_[3] += lg(pM2A), lp_[4] += lg(pM2As);
        lp_[6] += lg(pM2AB), lp_[7] += lg(pM3), lp_[8] += lg(pM3L), lp_[9] += lg(pM3Ls), lp_[10] += lg(pM3s), lp_[11] += lg(pM1G);
        lpGs_ += lg(pM2Gs);
        m1_->update(bit);
        if (!m1r_.empty()) {
            double q1[8];
            for (std::size_t r = 0; r < m1r_.size(); ++r) {
                if (m1r_[r]) m1r_[r]->update(bit);
                q1[r] = bit ? p1r[r] : 1 - p1r[r];
            }
            fsM1_.update(q1);
        }
        m1s_->update(bit);
        m1b_->update(bit);
        m1g_->update(bit);
        double q[8];
        for (int r = 0; r < nr; ++r) {
            if (r != center_) m2a_[static_cast<std::size_t>(r)]->update(bit);
            q[r] = bit ? pr[r] : 1 - pr[r];
        }
        if (M_.m2m) m2am_->update(bit);
        else fsA_.update(q);
        m2as_->update(bit);
        m2gs_->update(bit);
        m2ab_->update(bit);
        double q3[kM3];
        for (int i = 0; i < kM3; ++i) q3[i] = bit ? a3[i] : 1 - a3[i];
        fs3_.update(q3);
        m3l_->update(bit);
        m3ls_->update(bit);
        m3s_->update(bit);
        (void)pFs;
    }

    void byte_end(unsigned y, std::size_t k) override {
        const int nr = static_cast<int>(scales_.size());
        double q[8];
        for (int r = 0; r < nr; ++r) {
            q[r] = PG_[static_cast<std::size_t>(r)][y];
            if (r != center_) m2g_[static_cast<std::size_t>(r)]->update(y);
        }
        fsG_.update(q);
        // the hook's own history: the last three bytes, and the current word's letters
        h3_ = ((h3_ << 8) | (y & 255u)) & 0xffffffu;
        const int lc = std::tolower(static_cast<int>(y));
        hw_ = std::isalpha(lc) ? (hw_ + static_cast<std::uint32_t>(lc) + 1) * 0x01000193u : 0;
        const int h = k >= half_ ? 1 : 0;
        lp_[5] = lpGs_;
        for (int a = 0; a < kArms; ++a) bits_[h][a] -= lp_[a];
        g_bits_[h] -= lpG_;
        std::fill(lp_, lp_ + kArms, 0.0);
        lpGs_ = 0;
        ++n_[h];
    }

    int arm(const std::string& name) const {
        for (int a = 0; a < kArms; ++a)
            if (name == kArmName[a]) return a;
        return -1;
    }
    double bpc(int half, int a) const { return n_[half] ? bits_[half][a] / static_cast<double>(n_[half]) : 0; }
    std::size_t bytes() const {
        std::size_t b = m1_->bytes() + m1s_->bytes() * 6 + m1b_->bytes() + m2am_->bytes() + m2ab_->bytes() + m3l_->bytes();
        for (const auto& m : m2a_)
            if (m) b += m->bytes();
        for (const auto& m : m1r_)
            if (m) b += m->bytes();
        for (const auto& m : m2g_)
            if (m) b += m->bytes();
        return b;
    }
    void report(std::ostream& o) const override {
        char buf[200];
        for (int a = 0; a < kArms; ++a) {
            std::snprintf(buf, sizeof buf, "meta_mix_arm %-5s first_half %.4f second_half %.4f\n", kArmName[a], bpc(0, a), bpc(1, a));
            o << buf;
        }
        std::snprintf(buf, sizeof buf, "meta_mix_arm %-5s first_half %.4f second_half %.4f (M2G before SSE)\n", "M2G",
                      n_[0] ? g_bits_[0] / static_cast<double>(n_[0]) : 0, n_[1] ? g_bits_[1] / static_cast<double>(n_[1]) : 0);
        o << buf;
        std::string w = "meta_mix_weights M3";
        for (double v : fs3_.w) std::snprintf(buf, sizeof buf, " %.3f", v), w += buf;
        if (!m1r_.empty()) {
            w += " | M1";
            for (double v : fsM1_.w) std::snprintf(buf, sizeof buf, " %.3f", v), w += buf;
        }
        w += " | M2A";
        for (double v : fsA_.w) std::snprintf(buf, sizeof buf, " %.3f", v), w += buf;
        w += " | M2G";
        for (double v : fsG_.w) std::snprintf(buf, sizeof buf, " %.3f", v), w += buf;
        o << w << "\n";
    }

 private:
    fx::Params P_;
    Params M_;
    std::size_t half_;
    int kAll_ = 0, nall_ = 0;
    const fx::ByteIn* in_ = nullptr;
    int lstm_at_ = -1, bc1_ = 0, bc2_ = 0, cb_ = 0, mlen_ = 0;
    std::uint32_t h3_ = 0, hw_ = 0;
    std::vector<int> sel_;
    std::unique_ptr<Mixer2> m1_;
    std::vector<std::unique_ptr<Mixer2>> m1r_;  // m1r=1: M1 at each rate (the 1x member is m1_)
    std::unique_ptr<fx::Sse> m1s_, m2as_, m2gs_, m3ls_, m3s_;
    std::unique_ptr<fx::BitMixer> m1b_, m1g_, m2am_, m2ab_, m3l_;
    std::vector<double> scales_;
    int center_ = -1;
    std::vector<std::unique_ptr<fx::BitMixer>> m2a_;
    std::vector<std::unique_ptr<mix::GeoMixer>> m2g_;
    std::vector<std::array<double, 256>> PG_;
    FixedShare fsA_, fsG_, fs3_, fsM1_;
    double bitsG_[8] = {}, lpG_ = 0, lpGs_ = 0;
    double lp_[kArms] = {};
    double bits_[2][kArms] = {}, g_bits_[2] = {};
    std::size_t n_[2] = {0, 0};
};

/// cyphalm_trie meta-mix-replay FILE [FULL_MIX_PARAMS] [META_PARAMS] [first-half]: the full mix and the meta mixers
/// over a context8k --full-mix-dump; with "first-half" it stops at the half and never reads the second.
inline int replay(const std::string& path, const std::string& fparams, const std::string& mparams, bool first_half) {
    Params M;
    std::string err;
    if (!M.parse(mparams) || !M.check(err)) {
        std::cerr << "meta-mix-replay: bad meta params '" << mparams << "' " << err << "\n";
        return 2;
    }
    return fx::replay(
        path, fparams,
        [&](const fx::ChainSpec& cs, const fx::Params& P, std::size_t half) -> fx::StepHook* { return new MetaMixer(cs, P, M, half); },
        first_half);
}

}  // namespace cypha::online::metamix
