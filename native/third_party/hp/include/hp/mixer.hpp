#pragma once
//
// hp/mixer.hpp -- two-layer gated logistic mixer network + APM (SSE) stage.

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "hp/blob_io.hpp"
#include "hp/features.hpp"
#include "hp/hash_table.hpp"
#include "hp/int_math.hpp"
#include "hp/mixer_weights.hpp"
#include "hp/simd_dot.hpp"
#include "hp/undo.hpp"

namespace hp {

class MixerNet {
 public:
    MixerNet(int n_inputs, const std::vector<int>& ctx_sizes, int ctx2_size, int lr,
             const std::vector<int>& lrs = {})
        : n_(n_inputs), k_(static_cast<int>(ctx_sizes.size())), lr_(lr),
          ctx_sizes_(ctx_sizes), ctx_(ctx_sizes.size(), 0),
          st_(n_inputs, 0), dot_(ctx_sizes.size(), 0),
          pr_(ctx_sizes.size(), 2048),
          lr1_(ctx_sizes.size(), lr),
          v_(static_cast<std::size_t>(ctx2_size) * ctx_sizes.size(), mixer_wt_pack(0)) {
        w_.resize(ctx_sizes.size());
        const int w0 = (1 << 16) / (n_inputs > 0 ? n_inputs : 1);
        const MixerWt w0p = mixer_wt_pack(w0);
        for (std::size_t j = 0; j < ctx_sizes.size(); ++j) {
            w_[j].assign(static_cast<std::size_t>(ctx_sizes[j]) * n_inputs, w0p);
            if (j < lrs.size()) lr1_[j] = lrs[j];
        }
        const int v0 = (1 << 16) / (k_ > 0 ? k_ : 1);
        const MixerWt v0p = mixer_wt_pack(v0);
        for (auto& x : v_) x = v0p;
    }

    // Lossy knobs (Config::mixer_skip / Config::gate_drop). Defaults keep gate24.
    /// Serve-time adaptation: scale every learning rate by num/den (min 1).
    void scale_rates(int num, int den) {
        auto sc = [&](int r) { const int v = r * num / den; return v < 1 ? 1 : v; };
        lr_ = sc(lr_);
        for (int& r : lr1_) r = sc(r);
    }
    void set_skip(int skip) { skip_ = skip; }
    int skip() const { return skip_; }

    void set_lossy(int skip, std::uint32_t gate_drop) {
        if (skip > 0) skip_ = skip;
        gate_drop_ = gate_drop;
    }

    void reset_inputs() { m_ = 0; }
    void add(int stretched) {
        if (m_ < n_) st_[m_++] = static_cast<MixerSt>(stretched);
    }

    void set_ctx(int j, int c) { ctx_[j] = c % ctx_sizes_[j]; }
    void set_ctx2(int c) { ctx2_ = c; }

    int mix() {
        for (int j = 0; j < k_; ++j) {
            if ((gate_drop_ >> j) & 1u) {
                dot_[j] = 0;
                pr_[j] = 2048;
                continue;
            }
            const MixerWt* w = &w_[j][static_cast<std::size_t>(ctx_[j]) * n_];
            const std::int64_t sum = dot_mixer_wt(w, st_.data(), m_);
            dot_[j] = clamp_int(static_cast<int>(sum >> 16), -2047, 2047);
            pr_[j] = squash(dot_[j]);
        }
        const MixerWt* v = &v_[static_cast<std::size_t>(ctx2_) * k_];
        std::int64_t sum = 0;
        for (int j = 0; j < k_; ++j)
            sum += static_cast<std::int64_t>(mixer_wt_expand(v[j])) * dot_[j];
        final_dot_ = clamp_int(static_cast<int>(sum >> 16), -2047, 2047);
        final_pr_ = squash(final_dot_);
        return final_pr_;
    }

    void update(int y) {
        const int t = y << 12;
        const int err2 = t - final_pr_;
        {
            const int ae = err2 < 0 ? -err2 : err2;
            if (ae < skip_) return;
        }

        MixerWt* v = &v_[static_cast<std::size_t>(ctx2_) * k_];
        for (int j = 0; j < k_; ++j) {
            const std::int32_t dv = static_cast<std::int32_t>(
                (static_cast<std::int64_t>(dot_[j]) * err2 * lr_) >> 14);
            hp_undo_note(v[j]);
            v[j] = mixer_wt_pack(
                clamp_int(mixer_wt_expand(v[j]) + dv, -kMixerClamp, kMixerClamp));
        }

        for (int j = 0; j < k_; ++j) {
            if ((gate_drop_ >> j) & 1u) continue;
            const int err = t - pr_[j];
            const int l1 = lr1_[static_cast<std::size_t>(j)];
            MixerWt* w = &w_[j][static_cast<std::size_t>(ctx_[j]) * n_];
            axpy_mixer_wt(w, st_.data(), m_, err, l1, energy_);
        }
    }

    int num_layer1() const { return k_; }
    std::uint64_t learned_digest(std::uint64_t h) const {
        for (const auto& row : w_) h = fnv_bytes(h, row.data(), row.size() * sizeof(MixerWt));
        return fnv_bytes(h, v_.data(), v_.size() * sizeof(MixerWt));
    }
    int layer1_p(int j) const { return pr_[static_cast<std::size_t>(j)]; }
    int layer1_dot(int j) const { return dot_[static_cast<std::size_t>(j)]; }

    static MixerWt merge_mixer_wt(MixerWt dst, MixerWt src, std::uint64_t src_weight,
                                  std::uint64_t dst_weight) {
        const std::uint64_t total = dst_weight + src_weight;
        if (total == 0) {
            return dst;
        }
        const int merged = static_cast<int>(
            (static_cast<std::uint64_t>(mixer_wt_expand(dst)) * dst_weight +
             static_cast<std::uint64_t>(mixer_wt_expand(src)) * src_weight) /
            total);
        return mixer_wt_pack(clamp_int(merged, -kMixerClamp, kMixerClamp));
    }

    void merge_from(const MixerNet& src, std::uint64_t src_weight, std::uint64_t dst_weight) {
        if (src.n_ != n_ || src.k_ != k_ || src.w_.size() != w_.size() ||
            src.v_.size() != v_.size()) {
            return;
        }
        for (std::size_t j = 0; j < w_.size(); ++j) {
            for (std::size_t i = 0; i < w_[j].size(); ++i) {
                w_[j][i] = merge_mixer_wt(w_[j][i], src.w_[j][i], src_weight, dst_weight);
            }
        }
        for (std::size_t i = 0; i < v_.size(); ++i) {
            v_[i] = merge_mixer_wt(v_[i], src.v_[i], src_weight, dst_weight);
        }
    }

    void copy_from(const MixerNet& src) {
        if (src.n_ != n_ || src.k_ != k_) {
            return;
        }
        w_ = src.w_;
        v_ = src.v_;
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, n_);
        blob::write_pod(os, k_);
        blob::write_pod(os, lr_);
        blob::write_vec(os, ctx_sizes_);
        blob::write_vec(os, ctx_);
        blob::write_vec(os, st_);
        blob::write_vec(os, dot_);
        blob::write_vec(os, pr_);
        blob::write_vec(os, lr1_);
        for (const auto& row : w_) {
            blob::write_vec(os, row);
        }
        blob::write_vec(os, v_);
        blob::write_pod(os, energy_);
        blob::write_pod(os, m_);
        blob::write_pod(os, ctx2_);
        blob::write_pod(os, final_dot_);
        blob::write_pod(os, final_pr_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, n_);
        blob::read_pod(is, k_);
        blob::read_pod(is, lr_);
        blob::read_vec(is, ctx_sizes_);
        blob::read_vec(is, ctx_);
        blob::read_vec(is, st_);
        blob::read_vec(is, dot_);
        blob::read_vec(is, pr_);
        blob::read_vec(is, lr1_);
        w_.resize(ctx_sizes_.size());
        for (auto& row : w_) {
            blob::read_vec(is, row);
        }
        blob::read_vec(is, v_);
        blob::read_pod(is, energy_);
        blob::read_pod(is, m_);
        blob::read_pod(is, ctx2_);
        blob::read_pod(is, final_dot_);
        blob::read_pod(is, final_pr_);
    }

 private:
    int n_, k_, lr_;
    std::vector<int> ctx_sizes_;
    std::vector<int> ctx_;
    std::vector<MixerSt> st_;
    std::vector<int> dot_, pr_;
    std::vector<int> lr1_;
    std::vector<std::vector<MixerWt>> w_;
    std::vector<MixerWt> v_;
    int skip_ = 32;                 // gate24 HP_MIXER_SKIP
    std::uint32_t gate_drop_ = 0;
    std::int64_t energy_ = 0;
    int m_ = 0;
    int ctx2_ = 0;
    int final_dot_ = 0;
    int final_pr_ = 2048;
};

class APM {
 public:
    explicit APM(int n_ctx) : t_(static_cast<std::size_t>(n_ctx) * 33) {
        for (int i = 0; i < n_ctx; ++i) {
            for (int j = 0; j < 33; ++j) {
                t_[static_cast<std::size_t>(i) * 33 + j] =
                    static_cast<std::uint16_t>(squash((j - 16) * 128) * 16);
            }
        }
    }

    std::uint64_t learned_digest(std::uint64_t h) const {
        return fnv_bytes(h, t_.data(), t_.size() * sizeof(t_[0]));
    }

    int refine(int pr, int ctx) {
        const int s = stretch(pr) + 2048;
        const int w = s & 127;
        idx_ = (s >> 7) + ctx * 33;
        return (t_[idx_] * (128 - w) + t_[idx_ + 1] * w) >> 11;
    }

    void update(int y, int rate = 7) {
        hp_undo_note(t_[idx_]);
        hp_undo_note(t_[idx_ + 1]);
        const int g = (y << 16) + (y << rate) - y - y;
        t_[idx_] = static_cast<std::uint16_t>(
            t_[idx_] + ((g - static_cast<int>(t_[idx_])) >> rate));
        t_[idx_ + 1] = static_cast<std::uint16_t>(
            t_[idx_ + 1] + ((g - static_cast<int>(t_[idx_ + 1])) >> rate));
    }

    void merge_from(const APM& src, std::uint64_t src_weight, std::uint64_t dst_weight) {
        if (src.t_.size() != t_.size()) {
            return;
        }
        const std::uint64_t total = dst_weight + src_weight;
        if (total == 0) {
            return;
        }
        for (std::size_t i = 0; i < t_.size(); ++i) {
            t_[i] = static_cast<std::uint16_t>(
                (static_cast<std::uint64_t>(t_[i]) * dst_weight +
                 static_cast<std::uint64_t>(src.t_[i]) * src_weight) /
                total);
        }
    }

    void copy_from(const APM& src) {
        if (src.t_.size() == t_.size()) {
            t_ = src.t_;
        }
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_vec(os, t_);
        blob::write_pod(os, idx_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_vec(is, t_);
        blob::read_pod(is, idx_);
    }

 private:
    std::vector<std::uint16_t> t_;
    int idx_ = 0;
};

}  // namespace hp
