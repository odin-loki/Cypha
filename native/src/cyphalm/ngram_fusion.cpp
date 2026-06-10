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

}  // namespace

NgramFusion::NgramFusion(int field_dim, int field_in, int embed_in, const std::string& mode,
                         int n_positions, bool position_weights, std::uint64_t seed)
    : field_dim_(field_dim),
      field_in_(field_in),
      embed_in_(embed_in),
      mode_(mode),
      n_positions_(n_positions) {
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
}

std::vector<double> NgramFusion::matvec(const std::vector<double>& m, int rows, int cols,
                                        const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
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
    const auto em = apply_position_weights(embeds);
    if (static_cast<int>(em.size()) != embed_in_) {
        throw std::runtime_error("NgramFusion: embeds dim mismatch");
    }
    auto field_part = matvec(W_field_, field_dim_, field_in_, field_x);
    auto embed_part = matvec(W_embed_, field_dim_, embed_in_, em);
    if (mode_ == "gated") {
        std::vector<double> gate_in(static_cast<std::size_t>(field_in_ + embed_in_));
        for (int i = 0; i < field_in_; ++i) gate_in[static_cast<std::size_t>(i)] = field_x[static_cast<std::size_t>(i)];
        for (int i = 0; i < embed_in_; ++i) {
            gate_in[static_cast<std::size_t>(field_in_ + i)] = em[static_cast<std::size_t>(i)];
        }
        const auto gate_logits = matvec(W_gate_, field_dim_, field_in_ + embed_in_, gate_in);
        for (int i = 0; i < field_dim_; ++i) {
            const double g = sigmoid(gate_logits[static_cast<std::size_t>(i)]);
            field_part[static_cast<std::size_t>(i)] =
                g * field_part[static_cast<std::size_t>(i)] +
                (1.0 - g) * embed_part[static_cast<std::size_t>(i)];
        }
        return field_part;
    }
    for (int i = 0; i < field_dim_; ++i) {
        field_part[static_cast<std::size_t>(i)] += embed_part[static_cast<std::size_t>(i)];
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
    return out;
}

void NgramFusion::load_weights(const std::vector<double>& w_field, const std::vector<double>& w_embed,
                               const std::vector<double>& w_gate, const std::vector<double>& pos_weights) {
    if (!w_field.empty()) W_field_ = w_field;
    if (!w_embed.empty()) W_embed_ = w_embed;
    if (!w_gate.empty()) W_gate_ = w_gate;
    if (!pos_weights.empty()) pos_weights_ = pos_weights;
}

}  // namespace cypha::cyphalm
