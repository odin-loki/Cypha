#pragma once

#include "cypha/preprocessor.hpp"
#include "cypha/rff_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cypha {

/// Drop-in orthogonal RFF encoder (ORF / SORF) with the same layout as legacy ``RFFEncoder``.
/// Used by the forecasting plan Phase 1 encoder upgrade.
class OrthogonalRffEncoder {
 public:
  OrthogonalRffEncoder() = default;
  OrthogonalRffEncoder(int input_dim, int output_dim, double gamma, RffProjectionKind kind, int seed = 42)
      : input_dim_(input_dim), output_dim_(output_dim), gamma_(gamma), kind_(kind), seed_(seed) {}

  [[nodiscard]] int input_dim() const { return input_dim_; }
  [[nodiscard]] int output_dim() const { return output_dim_; }
  [[nodiscard]] RffProjectionKind kind() const { return kind_; }

  void fit(int input_dim, int output_dim, double gamma) {
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    gamma_ = gamma;
    std::mt19937 rng(static_cast<std::uint32_t>(seed_ & 0xffffffffu));
    init_rff_weights(kind_, rng, gamma_, output_dim_, input_dim_, w_flat_, b_, false);
    fitted_ = true;
  }

  [[nodiscard]] std::vector<double> transform_one(const std::vector<double>& x) const {
    if (!fitted_ || static_cast<int>(x.size()) != input_dim_) {
      return {};
    }
    std::vector<double> phi(static_cast<std::size_t>(output_dim_), 0.0);
    const double scale = std::sqrt(2.0 / static_cast<double>(std::max(output_dim_, 1)));
    for (int r = 0; r < output_dim_; ++r) {
      double dot = 0.0;
      for (int c = 0; c < input_dim_; ++c) {
        dot += x[static_cast<std::size_t>(c)] * w_flat_[static_cast<std::size_t>(r * input_dim_ + c)];
      }
      phi[static_cast<std::size_t>(r)] = scale * std::cos(dot + b_[static_cast<std::size_t>(r)]);
    }
    return phi;
  }

  void export_to_preprocessor(PreprocessorState& pre) const {
    pre.rff_dim = output_dim_;
    pre.rff_gamma = gamma_;
    pre.rff_sorf = (kind_ == RffProjectionKind::Sorf);
    pre.rff_orf = (kind_ == RffProjectionKind::Orf);
    pre.rff_w.assign(static_cast<std::size_t>(output_dim_));
    for (int r = 0; r < output_dim_; ++r) {
      pre.rff_w[static_cast<std::size_t>(r)].resize(static_cast<std::size_t>(input_dim_));
      for (int c = 0; c < input_dim_; ++c) {
        pre.rff_w[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
            w_flat_[static_cast<std::size_t>(r * input_dim_ + c)];
      }
    }
    pre.rff_b = b_;
  }

 private:
  int input_dim_{0};
  int output_dim_{256};
  double gamma_{1.0};
  RffProjectionKind kind_{RffProjectionKind::Orf};
  int seed_{42};
  bool fitted_{false};
  std::vector<double> w_flat_;
  std::vector<double> b_;
};

}  // namespace cypha
