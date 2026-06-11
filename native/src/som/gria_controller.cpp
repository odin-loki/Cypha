#include "cypha/som/gria_controller.hpp"

#include "cypha/som/gng_expert.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cypha::som {

namespace {

constexpr double kEps = 1e-9;
constexpr double kHistEps = 1e-12;

}  // namespace

GRIAController::GRIAController(GRIAControllerConfig cfg)
    : window_(std::max(1, cfg.window)),
      low_(cfg.low),
      high_(cfg.high),
      delta_ssm_(cfg.delta_ssm),
      control_interval_(std::max(1, cfg.control_interval)) {}

double GRIAController::std_dev(const std::vector<double>& v) {
  if (v.empty()) {
    return 0.0;
  }
  const double mean =
      std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  double var = 0.0;
  for (double x : v) {
    const double d = x - mean;
    var += d * d;
  }
  var /= static_cast<double>(v.size());
  return std::sqrt(var);
}

double GRIAController::entropy_hist(const std::deque<double>& buf, int n_bins) {
  if (buf.empty()) {
    return 0.0;
  }
  const int bins = std::max(1, n_bins);
  double lo = buf.front();
  double hi = buf.front();
  for (double v : buf) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  std::vector<double> hist(static_cast<std::size_t>(bins), 0.0);
  const double width = (hi - lo) / static_cast<double>(bins);
  for (double v : buf) {
    int idx = static_cast<int>((v - lo) / width);
    if (idx >= bins) {
      idx = bins - 1;
    }
    if (idx < 0) {
      idx = 0;
    }
    hist[static_cast<std::size_t>(idx)] += 1.0;
  }
  double ent = 0.0;
  for (double c : hist) {
    const double p = (c + kHistEps) /
                     (static_cast<double>(buf.size()) + kHistEps * static_cast<double>(bins));
    ent -= p * std::log(p);
  }
  return ent;
}

void GRIAController::push(const std::vector<double>& x,
                          const std::vector<double>& activations) {
  const double sx = std_dev(x) + kEps;
  const double sa = std_dev(activations) + kEps;
  inp_buf_.push_back(sx);
  act_buf_.push_back(sa);
  while (static_cast<int>(inp_buf_.size()) > window_) {
    inp_buf_.pop_front();
  }
  while (static_cast<int>(act_buf_.size()) > window_) {
    act_buf_.pop_front();
  }
}

double GRIAController::alpha() const {
  const int warm = std::min(32, window_);
  if (static_cast<int>(inp_buf_.size()) < warm) {
    return 0.5;
  }
  const double h_x = entropy_hist(inp_buf_);
  const double h_f = entropy_hist(act_buf_);
  const double a = 1.0 - h_f / (h_x + kEps);
  return std::clamp(a, 0.0, 1.0);
}

std::string GRIAController::act(int node_id, GNGExpertManager& gng,
                                std::function<void(double)> ssm_adjust) {
  ++step_;
  if (step_ % control_interval_ != 0) {
    return "skip";
  }
  const double a = alpha();
  std::string action = "hold";
  if (a < low_) {
    gng.force_insert(node_id);
    action = "split";
    if (ssm_adjust) {
      ssm_adjust(+delta_ssm_);
    }
  } else if (a > high_) {
    gng.merge_with_nearest(node_id);
    action = "merge";
    if (ssm_adjust) {
      ssm_adjust(-delta_ssm_);
    }
  }
  return action;
}

}  // namespace cypha::som
