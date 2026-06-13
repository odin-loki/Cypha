#pragma once

/// Autocorrelation-based fast/slow decay scaling for SSM (U6; off by default).

#include <cstdint>
#include <tuple>
#include <vector>

namespace cypha::som {

struct TemporalSOMConfig {
  int M{8};
  int L_max{16};
  double eta_ts{0.05};
  std::uint64_t seed{42};
};

class TemporalSOM {
 public:
  explicit TemporalSOM(TemporalSOMConfig cfg = {});

  int M() const { return M_; }
  int L_max() const { return L_max_; }

  /// Returns (BMU index, fast decay multiplier, slow decay multiplier).
  std::tuple<int, double, double> step(const std::vector<double>& x, bool train = true);

 private:
  std::vector<double> autocorr_features(const std::vector<double>& x);

  int M_;
  int L_max_;
  double eta_ts_;
  std::vector<std::vector<double>> centroids_;
  std::vector<std::vector<double>> lambda_;  // [M][2] fast/slow multipliers
  std::vector<std::vector<double>> x_hist_;
};

}  // namespace cypha::som
