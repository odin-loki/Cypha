#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// Fuse SSM field with n-gram embed projections (``NgramFusion`` sum/gated mode).
class NgramFusion {
 public:
    NgramFusion(int field_dim, int field_in, int embed_in, const std::string& mode,
                int n_positions, bool position_weights, bool bilinear_fusion, int bilinear_rank,
                std::uint64_t seed);

    std::vector<double> forward(const std::vector<double>& field_x,
                                const std::vector<double>& embeds) const;

    /// Gradient w.r.t. field_x for truncated BPTT (sum mode).
    std::vector<double> grad_field_x(const std::vector<double>& grad_v) const;

    void load_weights(const std::vector<double>& w_field, const std::vector<double>& w_embed,
                      const std::vector<double>& w_gate, const std::vector<double>& pos_weights,
                      const std::vector<double>& w_b_u, const std::vector<double>& w_b_vf,
                      const std::vector<double>& w_b_ve);

    /// Online SGD on learnable per-position scalars (B3). ``grad_v`` is dL/dv at the fusion
    /// output (same shape as ``field_dim_``); ``embeds`` is the raw history vector before scaling.
    void update_position_weights(const std::vector<double>& grad_v, const std::vector<double>& embeds,
                                 double lr);

    const std::vector<double>& w_field() const { return W_field_; }
    const std::vector<double>& w_embed() const { return W_embed_; }
    const std::vector<double>& w_gate() const { return W_gate_; }
    const std::vector<double>& pos_weights() const { return pos_weights_; }
    const std::vector<double>& w_b_u() const { return W_b_u_; }
    const std::vector<double>& w_b_vf() const { return W_b_vf_; }
    const std::vector<double>& w_b_ve() const { return W_b_ve_; }
    int embed_in() const { return embed_in_; }

 private:
    int field_dim_;
    int field_in_;
    int embed_in_;
    std::string mode_;
    int n_positions_;
    std::vector<double> W_field_;
    std::vector<double> W_embed_;
    std::vector<double> W_gate_;
    std::vector<double> pos_weights_;
    bool bilinear_fusion_;
    int bilinear_rank_;
    std::vector<double> W_b_u_;
    std::vector<double> W_b_vf_;
    std::vector<double> W_b_ve_;

    std::vector<double> apply_position_weights(const std::vector<double>& embeds) const;
    static std::vector<double> matvec(const std::vector<double>& m, int rows, int cols,
                                      const std::vector<double>& x);
    static void matvec(const std::vector<double>& m, int rows, int cols,
                       const std::vector<double>& x, std::vector<double>& out);
};

}  // namespace cypha::cyphalm
