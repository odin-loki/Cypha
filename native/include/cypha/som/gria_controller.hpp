#pragma once

/// GRIA alpha live topology controller (U3; off by default).

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace cypha::som {

class GNGExpertManager;

struct GRIAControllerConfig {
  int window{200};
  double low{0.35};
  double high{0.65};
  double delta_ssm{0.01};
  int control_interval{50};
};

class GRIAController {
 public:
  explicit GRIAController(GRIAControllerConfig cfg = {});

  void push(const std::vector<double>& x, const std::vector<double>& activations);

  /// Entropy-based structural readiness in [0, 1] (0.5 until buffer warm).
  double alpha() const;

  /// Periodic GNG topology action: "skip", "hold", "split", or "merge".
  std::string act(int node_id, GNGExpertManager& gng,
                  std::function<void(double)> ssm_adjust = nullptr);

 private:
  static double entropy_hist(const std::deque<double>& buf, int n_bins = 16);
  static double std_dev(const std::vector<double>& v);

  int window_;
  double low_;
  double high_;
  double delta_ssm_;
  int control_interval_;
  std::deque<double> inp_buf_;
  std::deque<double> act_buf_;
  int step_{0};
};

}  // namespace cypha::som
