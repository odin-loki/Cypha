#include "cypha/cyphalm/ngram_fusion.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

thread_local std::vector<double> g_bilinear_h_e_cache;

}  // namespace

NgramFusion::NgramFusion(int field_dim, int field_in, int embed_in, const std::string& mode,
                         int n_positions, bool position_weights, bool bilinear_fusion,
                         int bilinear_rank, std::uint64_t seed)
    : field_dim_(field_dim),
      field_in_(field_in),
      embed_in_(embed_in),
      mode_(mode),
      n_positions_(n_positions),
      bilinear_fusion_(bilinear_fusion && mode == "sum"),
      bilinear_rank_(bilinear_rank > 0 ? bilinear_rank : 32) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.02);
    W_field_.assign(static_cast<std::size_t>(field_dim_ * field_in_), 0.0);
    W_embed_.assign(static_cast<std::size_t>(field_dim_ * embed_in_), 0.0);
    for (auto& v : W_field_) v = nd(rng);
    for (auto& v : W_embed_) v = nd(rng);
    if (mode_ == "gated") {
        const int gate_in = field_in_ + embed_in_;
        W_gate_.assign(static_cast<std::size_t>(field_dim_ * gate_in), 0.0);
        for (auto& v : W_gate_) v = nd(rng);
    }
    if (position_weights && n_positions_ > 0) {
        pos_weights_.assign(static_cast<std::size_t>(n_positions_), 1.0);
    }
    if (bilinear_fusion_ && field_in_ > 0 && embed_in_ > 0) {
        const int r = bilinear_rank_;
        W_b_u_.assign(static_cast<std::size_t>(field_dim_ * r), 0.0);
        W_b_vf_.assign(static_cast<std::size_t>(r * field_in_), 0.0);
        W_b_ve_.assign(static_cast<std::size_t>(r * embed_in_), 0.0);
        for (auto& v : W_b_u_) v = nd(rng);
        for (auto& v : W_b_vf_) v = nd(rng);
        for (auto& v : W_b_ve_) v = nd(rng);
    }
}

void NgramFusion::matvec(const std::vector<double>& m, int rows, int cols,
                         const std::vector<double>& x, std::vector<double>& out) {
    if (out.size() != static_cast<std::size_t>(rows)) out.resize(static_cast<std::size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
}

std::vector<double> NgramFusion::matvec(const std::vector<double>& m, int rows, int cols,
                                        const std::vector<double>& x) {
    std::vector<double> out;
    matvec(m, rows, cols, x, out);
    return out;
}

std::vector<double> NgramFusion::apply_position_weights(const std::vector<double>& embeds) const {
    if (pos_weights_.empty() || n_positions_ <= 0) return embeds;
    const int d = embed_in_ / n_positions_;
    if (d <= 0 || static_cast<int>(embeds.size()) != embed_in_) return embeds;
    std::vector<double> out(static_cast<std::size_t>(embed_in_), 0.0);
    for (int i = 0; i < n_positions_; ++i) {
        const double w = pos_weights_[static_cast<std::size_t>(i)];
        for (int j = 0; j < d; ++j) {
            out[static_cast<std::size_t>(i * d + j)] =
                w * embeds[static_cast<std::size_t>(i * d + j)];
        }
    }
    return out;
}

std::vector<double> NgramFusion::forward(const std::vector<double>& field_x,
                                         const std::vector<double>& embeds) const {
    if (static_cast<int>(field_x.size()) != field_in_) {
        throw std::runtime_error("NgramFusion: field_x dim mismatch");
    }
    // Perf (2026-07-12, part 3, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up (part
    // 3)"): `apply_position_weights(embeds)` unconditionally returned a fresh-copy vector even
    // when `pos_weights_` is empty (the default -- `use_ngram_position_weights` defaults false),
    // in which case it was *provably* just an identity copy of `embeds` (see its own early
    // `return embeds;`). Inlining that exact no-op condition here lets the default-config path
    // reference `embeds` directly via a pointer with zero extra allocation/copy, while the
    // position-weighted path (unchanged arithmetic, still computed into a reused thread_local
    // buffer below) behaves identically to before.
    const bool needs_pos_weights =
        !pos_weights_.empty() && n_positions_ > 0 && (embed_in_ / n_positions_) > 0 &&
        static_cast<int>(embeds.size()) == embed_in_;
    const std::vector<double>* em_ptr = &embeds;
    thread_local std::vector<double> pos_weighted_scratch;
    if (needs_pos_weights) {
        const int d = embed_in_ / n_positions_;
        if (pos_weighted_scratch.size() != static_cast<std::size_t>(embed_in_)) {
            pos_weighted_scratch.resize(static_cast<std::size_t>(embed_in_));
        }
        for (int i = 0; i < n_positions_; ++i) {
            const double w = pos_weights_[static_cast<std::size_t>(i)];
            for (int j = 0; j < d; ++j) {
                pos_weighted_scratch[static_cast<std::size_t>(i * d + j)] =
                    w * embeds[static_cast<std::size_t>(i * d + j)];
            }
        }
        em_ptr = &pos_weighted_scratch;
    }
    const auto& em = *em_ptr;
    if (static_cast<int>(em.size()) != embed_in_) {
        throw std::runtime_error("NgramFusion: embeds dim mismatch");
    }
    // `embed_part` is purely an internal accumulator (never returned), unlike `field_part` below
    // (which either *is* the return value or is combined into it in place) -- thread_local reuse
    // via the out-param matvec overload above removes its allocation entirely. `field_part`
    // itself is left as a `matvec`-returned local: it's what NRVO can construct directly in the
    // caller's storage for this function's own return-by-value contract, so a scratch buffer
    // there would just add a second copy back out, not remove one.
    auto field_part = matvec(W_field_, field_dim_, field_in_, field_x);
    thread_local std::vector<double> embed_part_scratch;
    matvec(W_embed_, field_dim_, embed_in_, em, embed_part_scratch);
    const auto& embed_part = embed_part_scratch;
    if (mode_ == "gated") {
        thread_local std::vector<double> gate_in_scratch;
        const std::size_t gate_in_size = static_cast<std::size_t>(field_in_ + embed_in_);
        if (gate_in_scratch.size() != gate_in_size) gate_in_scratch.resize(gate_in_size);
        for (int i = 0; i < field_in_; ++i) gate_in_scratch[static_cast<std::size_t>(i)] = field_x[static_cast<std::size_t>(i)];
        for (int i = 0; i < embed_in_; ++i) {
            gate_in_scratch[static_cast<std::size_t>(field_in_ + i)] = em[static_cast<std::size_t>(i)];
        }
        thread_local std::vector<double> gate_logits_scratch;
        matvec(W_gate_, field_dim_, field_in_ + embed_in_, gate_in_scratch, gate_logits_scratch);
        for (int i = 0; i < field_dim_; ++i) {
            const double g = sigmoid(gate_logits_scratch[static_cast<std::size_t>(i)]);
            field_part[static_cast<std::size_t>(i)] =
                g * field_part[static_cast<std::size_t>(i)] +
                (1.0 - g) * embed_part[static_cast<std::size_t>(i)];
        }
        return field_part;
    }
    for (int i = 0; i < field_dim_; ++i) {
        field_part[static_cast<std::size_t>(i)] += embed_part[static_cast<std::size_t>(i)];
    }
    if (bilinear_fusion_ && !W_b_u_.empty()) {
        const int r = bilinear_rank_;
        thread_local std::vector<double> h_f_scratch;
        thread_local std::vector<double> h_e_scratch;
        if (h_f_scratch.size() != static_cast<std::size_t>(r)) {
            h_f_scratch.assign(static_cast<std::size_t>(r), 0.0);
            h_e_scratch.assign(static_cast<std::size_t>(r), 0.0);
        } else {
            std::fill(h_f_scratch.begin(), h_f_scratch.end(), 0.0);
            std::fill(h_e_scratch.begin(), h_e_scratch.end(), 0.0);
        }
        matvec(W_b_vf_, r, field_in_, field_x, h_f_scratch);
        matvec(W_b_ve_, r, embed_in_, em, h_e_scratch);
        g_bilinear_h_e_cache = h_e_scratch;
        for (int ri = 0; ri < r; ++ri) {
            const double had = h_f_scratch[static_cast<std::size_t>(ri)] *
                               h_e_scratch[static_cast<std::size_t>(ri)];
            for (int i = 0; i < field_dim_; ++i) {
                field_part[static_cast<std::size_t>(i)] +=
                    W_b_u_[static_cast<std::size_t>(i * r + ri)] * had;
            }
        }
    }
    return field_part;
}

std::vector<double> NgramFusion::grad_field_x(const std::vector<double>& grad_v) const {
    std::vector<double> out(static_cast<std::size_t>(field_dim_), 0.0);
    if (static_cast<int>(grad_v.size()) < field_dim_) {
        return out;
    }
    for (int r = 0; r < field_dim_; ++r) {
        double acc = 0.0;
        for (int c = 0; c < field_in_; ++c) {
            acc += W_field_[static_cast<std::size_t>(r * field_in_ + c)] *
                   grad_v[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
    if (bilinear_fusion_ && !W_b_u_.empty() && field_in_ == field_dim_) {
        const int rank = bilinear_rank_;
        if (static_cast<int>(g_bilinear_h_e_cache.size()) != rank) {
            return out;
        }
        thread_local std::vector<double> grad_had_scratch;
        if (grad_had_scratch.size() != static_cast<std::size_t>(rank)) {
            grad_had_scratch.assign(static_cast<std::size_t>(rank), 0.0);
        } else {
            std::fill(grad_had_scratch.begin(), grad_had_scratch.end(), 0.0);
        }
        for (int ri = 0; ri < rank; ++ri) {
            double acc = 0.0;
            for (int i = 0; i < field_dim_; ++i) {
                acc += W_b_u_[static_cast<std::size_t>(i * rank + ri)] *
                       grad_v[static_cast<std::size_t>(i)];
            }
            grad_had_scratch[static_cast<std::size_t>(ri)] = acc;
        }
        for (int ri = 0; ri < rank; ++ri) {
            const double grad_h_f =
                grad_had_scratch[static_cast<std::size_t>(ri)] *
                g_bilinear_h_e_cache[static_cast<std::size_t>(ri)];
            for (int c = 0; c < field_in_; ++c) {
                out[static_cast<std::size_t>(c)] +=
                    W_b_vf_[static_cast<std::size_t>(ri * field_in_ + c)] * grad_h_f;
            }
        }
    }
    return out;
}

void NgramFusion::load_weights(const std::vector<double>& w_field, const std::vector<double>& w_embed,
                               const std::vector<double>& w_gate, const std::vector<double>& pos_weights,
                               const std::vector<double>& w_b_u, const std::vector<double>& w_b_vf,
                               const std::vector<double>& w_b_ve) {
    if (!w_field.empty()) W_field_ = w_field;
    if (!w_embed.empty()) W_embed_ = w_embed;
    if (!w_gate.empty()) W_gate_ = w_gate;
    if (!pos_weights.empty()) pos_weights_ = pos_weights;
    if (!w_b_u.empty()) W_b_u_ = w_b_u;
    if (!w_b_vf.empty()) W_b_vf_ = w_b_vf;
    if (!w_b_ve.empty()) W_b_ve_ = w_b_ve;
}

void NgramFusion::update_position_weights(const std::vector<double>& grad_v,
                                          const std::vector<double>& embeds, double lr) {
    if (pos_weights_.empty() || n_positions_ <= 0 || lr <= 0.0) {
        return;
    }
    const int d = embed_in_ / n_positions_;
    if (d <= 0 || static_cast<int>(embeds.size()) != embed_in_ ||
        static_cast<int>(grad_v.size()) < field_dim_) {
        return;
    }
    if (mode_ != "sum") {
        return;
    }
    thread_local std::vector<double> grad_em_scratch;
    if (grad_em_scratch.size() != static_cast<std::size_t>(embed_in_)) {
        grad_em_scratch.assign(static_cast<std::size_t>(embed_in_), 0.0);
    } else {
        std::fill(grad_em_scratch.begin(), grad_em_scratch.end(), 0.0);
    }
    for (int r = 0; r < field_dim_; ++r) {
        const double gv = grad_v[static_cast<std::size_t>(r)];
        for (int c = 0; c < embed_in_; ++c) {
            grad_em_scratch[static_cast<std::size_t>(c)] +=
                W_embed_[static_cast<std::size_t>(r * embed_in_ + c)] * gv;
        }
    }
    for (int i = 0; i < n_positions_; ++i) {
        double dw = 0.0;
        for (int j = 0; j < d; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i * d + j);
            dw += grad_em_scratch[idx] * embeds[idx];
        }
        pos_weights_[static_cast<std::size_t>(i)] -= lr * dw;
    }
}

}  // namespace cypha::cyphalm
