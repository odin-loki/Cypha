#pragma once

#include <cstdint>
#include <vector>

namespace cypha::cyphalm {

/// Mamba-lite selective SSM: diagonal A, input-dependent B/C gates, O(d_state) per token.
class SelectiveSSM {
public:
    SelectiveSSM(std::uint32_t d_input, std::uint32_t d_state, std::uint32_t d_output,
                 bool learnable_a, double decay_init, std::uint32_t seed);

    std::uint32_t d_input() const { return d_input_; }
    std::uint32_t d_state() const { return d_state_; }
    std::uint32_t d_output() const { return d_output_; }

    void reset();
    std::vector<double> step(const double* x, std::uint32_t x_len);

    const std::vector<double>& state() const { return h_; }
    std::vector<double> pooled_state() const;

    std::vector<double>& A_log() { return A_log_; }
    std::vector<double>& W_b() { return W_b_; }
    std::vector<double>& W_c() { return W_c_; }
    std::vector<double>& W_d() { return W_d_; }

private:
    std::uint32_t d_input_;
    std::uint32_t d_state_;
    std::uint32_t d_output_;
    bool learnable_a_;
    std::vector<double> A_log_;  // log decay per state dim -> exp on use
    std::vector<double> W_b_;    // [d_state, d_input]
    std::vector<double> W_c_;    // [d_output, d_state]
    std::vector<double> W_d_;    // [d_output, d_input]
    std::vector<double> h_;

    static void matvec(const std::vector<double>& W, std::uint32_t rows, std::uint32_t cols,
                       const double* x, std::vector<double>& out);
    static double sigmoid(double x);
};

}  // namespace cypha::cyphalm
