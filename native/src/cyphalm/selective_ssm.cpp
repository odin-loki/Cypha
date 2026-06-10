#include "cypha/cyphalm/selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

struct Rng {
    std::uint32_t state;
    explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}
    double normal() {
        double u1 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
        double u2 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
        if (u1 < 1e-12) u1 = 1e-12;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
    }
    std::uint32_t next() {
        state = static_cast<std::uint32_t>((static_cast<std::uint64_t>(state) * 48271u) % 2147483647u);
        return state;
    }
};

}  // namespace

SelectiveSSM::SelectiveSSM(std::uint32_t d_input, std::uint32_t d_state, std::uint32_t d_output,
                           bool learnable_a, double decay_init, std::uint32_t seed)
    : d_input_(d_input),
      d_state_(d_state),
      d_output_(d_output),
      learnable_a_(learnable_a) {
    if (d_input < 1 || d_state < 1 || d_output < 1) throw std::invalid_argument("invalid selective SSM dims");
    Rng rng(seed + 7);
    const double scale = 0.02;
    A_log_.assign(d_state_, std::log(std::clamp(decay_init, 0.01, 0.999)));
    W_b_.resize(static_cast<std::size_t>(d_state_) * d_input_);
    W_c_.resize(static_cast<std::size_t>(d_output_) * d_state_);
    W_d_.resize(static_cast<std::size_t>(d_output_) * d_input_);
    for (auto& v : W_b_) v = rng.normal() * scale;
    for (auto& v : W_c_) v = rng.normal() * scale;
    for (auto& v : W_d_) v = rng.normal() * scale;
    reset();
}

void SelectiveSSM::reset() { h_.assign(d_state_, 0.0); }

double SelectiveSSM::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -40.0, 40.0))); }

void SelectiveSSM::matvec(const std::vector<double>& W, std::uint32_t rows, std::uint32_t cols,
                          const double* x, std::vector<double>& out) {
    out.assign(rows, 0.0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::uint32_t c = 0; c < cols; ++c) acc += W[static_cast<std::size_t>(r) * cols + c] * x[c];
        out[r] = acc;
    }
}

std::vector<double> SelectiveSSM::step(const double* x, std::uint32_t x_len) {
    if (x_len != d_input_) throw std::invalid_argument("SelectiveSSM: input dim mismatch");

    std::vector<double> b_gate;
    matvec(W_b_, d_state_, d_input_, x, b_gate);
    for (std::uint32_t i = 0; i < d_state_; ++i) b_gate[i] = sigmoid(b_gate[i]);

    for (std::uint32_t i = 0; i < d_state_; ++i) {
        double a = learnable_a_ ? std::exp(A_log_[i]) : std::exp(A_log_[i]);
        a = std::clamp(a, 0.01, 0.999);
        double bx = 0.0;
        for (std::uint32_t d = 0; d < d_input_; ++d)
            bx += W_b_[static_cast<std::size_t>(i) * d_input_ + d] * x[d];
        h_[i] = a * h_[i] + b_gate[i] * bx;
    }

    std::vector<double> c_gate;
    matvec(W_c_, d_output_, d_state_, h_.data(), c_gate);
    for (std::uint32_t o = 0; o < d_output_; ++o) c_gate[o] = sigmoid(c_gate[o]);

    std::vector<double> y(d_output_, 0.0);
    for (std::uint32_t o = 0; o < d_output_; ++o) {
        double acc = 0.0;
        for (std::uint32_t i = 0; i < d_state_; ++i)
            acc += W_c_[static_cast<std::size_t>(o) * d_state_ + i] * h_[i] * c_gate[o];
        for (std::uint32_t d = 0; d < d_input_; ++d)
            acc += W_d_[static_cast<std::size_t>(o) * d_input_ + d] * x[d];
        y[o] = acc;
    }
    return y;
}

std::vector<double> SelectiveSSM::pooled_state() const {
    std::vector<double> out = h_;
    if (out.empty()) return out;
    double norm = 0.0;
    for (double v : out) norm += v * v;
    norm = std::sqrt(norm + 1e-12);
    for (double& v : out) v /= norm;
    return out;
}

}  // namespace cypha::cyphalm
