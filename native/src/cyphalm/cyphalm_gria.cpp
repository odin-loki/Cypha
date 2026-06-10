#include "cypha/cyphalm/cyphalm_gria.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace cypha::cyphalm {

GriaHead::GriaHead(int in_dim, int vocab_size, double alpha_init, double laplace)
    : in_dim_(in_dim),
      vocab_size_(vocab_size),
      alpha_(alpha_init),
      laplace_(laplace),
      W_(static_cast<std::size_t>(vocab_size * in_dim), 0.0),
      b_(static_cast<std::size_t>(vocab_size), laplace),
      token_counts_(static_cast<std::size_t>(vocab_size), laplace) {}

void GriaHead::reset() {
    std::fill(W_.begin(), W_.end(), 0.0);
    std::fill(b_.begin(), b_.end(), laplace_);
    std::fill(token_counts_.begin(), token_counts_.end(), laplace_);
}

std::vector<double> GriaHead::softmax(const std::vector<double>& logits) {
    double m = *std::max_element(logits.begin(), logits.end());
    std::vector<double> out(logits.size());
    double s = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] - m);
        s += out[i];
    }
    for (auto& v : out) v /= s;
    return out;
}

std::vector<double> GriaHead::forward(const std::vector<double>& v) {
    if (static_cast<int>(v.size()) != in_dim_) {
        throw std::invalid_argument("GriaHead::forward dim mismatch");
    }
    std::vector<double> logits(vocab_size_);
    for (int k = 0; k < vocab_size_; ++k) {
        double acc = b_[static_cast<std::size_t>(k)];
        for (int j = 0; j < in_dim_; ++j) {
            acc += W_[static_cast<std::size_t>(k * in_dim_ + j)] * v[static_cast<std::size_t>(j)];
        }
        logits[static_cast<std::size_t>(k)] = acc;
    }
    auto probs = softmax(logits);
    std::vector<double> log_probs(vocab_size_);
    for (int k = 0; k < vocab_size_; ++k) {
        log_probs[static_cast<std::size_t>(k)] = std::log(probs[static_cast<std::size_t>(k)] + 1e-12);
    }
    return log_probs;
}

void GriaHead::train_step(const std::vector<double>& v, int target_id, double lr) {
    std::vector<double> logits(vocab_size_);
    for (int k = 0; k < vocab_size_; ++k) {
        double acc = b_[static_cast<std::size_t>(k)];
        for (int j = 0; j < in_dim_; ++j) {
            acc += W_[static_cast<std::size_t>(k * in_dim_ + j)] * v[static_cast<std::size_t>(j)];
        }
        logits[static_cast<std::size_t>(k)] = acc;
    }
    auto probs = softmax(logits);
    for (int k = 0; k < vocab_size_; ++k) {
        double grad = probs[static_cast<std::size_t>(k)];
        if (k == target_id) grad -= 1.0;
        for (int j = 0; j < in_dim_; ++j) {
            W_[static_cast<std::size_t>(k * in_dim_ + j)] -= lr * grad * v[static_cast<std::size_t>(j)];
        }
        b_[static_cast<std::size_t>(k)] -= lr * grad;
    }
    if (target_id >= 0 && target_id < vocab_size_) {
        token_counts_[static_cast<std::size_t>(target_id)] += 1.0;
    }
}

}  // namespace cypha::cyphalm
