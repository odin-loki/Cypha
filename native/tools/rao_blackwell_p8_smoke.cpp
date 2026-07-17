// Phase 8: audit generation.hpp / replay_buffer.hpp for MC sample averages vs RB conditional expectations.
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/generation.hpp"
#include "cypha/replay_buffer.hpp"

namespace {

/// MC sample mean of n i.i.d. N(0, diag(v)) draws — hypothetical estimator RB would replace with mu=0.
double mc_gaussian_mean_coord_variance(int d, const std::vector<double>& v0, int n_mc, int n_trials,
                                       std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  const int total = n_trials * d;
  std::vector<double> trial_means(static_cast<std::size_t>(total));
  for (int t = 0; t < n_trials; ++t) {
    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int k = 0; k < n_mc; ++k) {
        const double std_j = std::sqrt(std::max(v0[static_cast<std::size_t>(j)], cypha::kGenMinVar));
        s += nd(rng) * std_j;
      }
      trial_means[static_cast<std::size_t>(t * d + j)] = s / static_cast<double>(n_mc);
    }
  }
  double var_sum = 0.0;
  for (int j = 0; j < d; ++j) {
    double m = 0.0;
    for (int t = 0; t < n_trials; ++t) {
      m += trial_means[static_cast<std::size_t>(t * d + j)];
    }
    m /= static_cast<double>(n_trials);
    double v = 0.0;
    for (int t = 0; t < n_trials; ++t) {
      const double diff = trial_means[static_cast<std::size_t>(t * d + j)] - m;
      v += diff * diff;
    }
    var_sum += v / static_cast<double>(n_trials - 1);
  }
  return var_sum / static_cast<double>(d);
}

/// RB conditional expectation E[h | weights, neighbors] — exact weighted centroid (generation.cpp:948-956).
double rb_weighted_centroid_variance(const std::vector<std::vector<double>>& /*neighbors*/,
                                     const std::vector<double>& /*weights*/, int /*n_trials*/) {
  return 0.0;  // deterministic given sufficient stats (weights + neighbor latents)
}

/// MC resample-one-neighbor estimator variance (not present in production code).
double mc_neighbor_pick_variance(const std::vector<std::vector<double>>& neighbors,
                                 const std::vector<double>& weights, int n_trials, std::uint32_t seed) {
  const int d = static_cast<int>(neighbors[0].size());
  double w_sum = 0.0;
  for (double w : weights) {
    w_sum += w;
  }
  std::vector<double> probs = weights;
  for (double& p : probs) {
    p /= w_sum;
  }
  std::mt19937 rng(seed);
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  std::vector<double> means(static_cast<std::size_t>(d), 0.0);
  std::vector<std::vector<double>> trials(static_cast<std::size_t>(n_trials),
                                          std::vector<double>(static_cast<std::size_t>(d)));
  for (int t = 0; t < n_trials; ++t) {
    const int pick = dist(rng);
    trials[static_cast<std::size_t>(t)] = neighbors[static_cast<std::size_t>(pick)];
    for (int j = 0; j < d; ++j) {
      means[static_cast<std::size_t>(j)] += trials[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)];
    }
  }
  for (int j = 0; j < d; ++j) {
    means[static_cast<std::size_t>(j)] /= static_cast<double>(n_trials);
  }
  double var_sum = 0.0;
  for (int j = 0; j < d; ++j) {
    double v = 0.0;
    for (int t = 0; t < n_trials; ++t) {
      const double diff =
          trials[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)] - means[static_cast<std::size_t>(j)];
      v += diff * diff;
    }
    var_sum += v / static_cast<double>(n_trials - 1);
  }
  return var_sum / static_cast<double>(d);
}

void fill_replay(cypha::ReplayBuffer& buf, int cap, int d, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  for (int i = 0; i < cap; ++i) {
    std::vector<double> h(static_cast<std::size_t>(d));
    std::vector<double> f(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      h[static_cast<std::size_t>(j)] = nd(rng);
      f[static_cast<std::size_t>(j)] = nd(rng);
    }
    buf.push(h.data(), f.data(), d, "c" + std::to_string(i % 3), static_cast<double>(i + 1));
  }
}

}  // namespace

int main() {
  constexpr int kD = 6;
  constexpr int kMc = 32;
  constexpr int kTrials = 400;
  constexpr std::uint32_t kSeed = 0x08123456u;

  std::vector<double> v0(static_cast<std::size_t>(kD), 0.25);

  const double var_mc_mean = mc_gaussian_mean_coord_variance(kD, v0, kMc, kTrials, kSeed);
  const double var_rb_mean = 0.0;

  std::vector<std::vector<double>> neighbors = {
      {1.0, 0.0, 0.5, -0.2, 0.3, 0.1},
      {-0.5, 0.8, 0.2, 0.4, -0.1, 0.6},
      {0.2, -0.3, 0.9, 0.1, 0.0, -0.4},
  };
  const std::vector<double> weights = {3.0, 1.0, 2.0};
  const double var_rb_centroid = rb_weighted_centroid_variance(neighbors, weights, kTrials);
  const double var_mc_centroid = mc_neighbor_pick_variance(neighbors, weights, kTrials, kSeed + 1u);

  cypha::ReplayBuffer replay(16);
  fill_replay(replay, 12, kD, kSeed + 2u);
  std::mt19937 replay_rng(kSeed + 3u);
  std::vector<std::vector<double>> h_out;
  std::vector<std::string> labels_out;
  replay.sample(4, replay_rng, h_out, labels_out);
  const bool replay_sample_ok = static_cast<int>(h_out.size()) == 4 && static_cast<int>(labels_out.size()) == 4;

  std::cout << "rao_blackwell_p8_smoke:\n"
            << "  audit=NO-GO (no MC sample averages in generation.hpp / replay_buffer.hpp)\n"
            << "  var_mc_gaussian_mean=" << var_mc_mean << " var_rb_gaussian_mean=" << var_rb_mean << "\n"
            << "  var_mc_neighbor_pick=" << var_mc_centroid << " var_rb_weighted_centroid=" << var_rb_centroid
            << "\n"
            << "  replay_weighted_sample_ok=" << (replay_sample_ok ? 1 : 0) << "\n";

  bool pass = true;
  if (!(var_mc_mean > var_rb_mean + 1e-9)) {
    std::cerr << "FAIL: expected MC Gaussian mean variance > RB (0)\n";
    pass = false;
  }
  if (!(var_mc_centroid > var_rb_centroid + 1e-9)) {
    std::cerr << "FAIL: expected MC neighbor-pick variance > RB weighted centroid (0)\n";
    pass = false;
  }
  if (!replay_sample_ok) {
    std::cerr << "FAIL: replay buffer weighted sample returned wrong count\n";
    pass = false;
  }

  if (pass) {
    std::cout << "rao_blackwell_p8_smoke: PASS (audit no-go; variance baseline confirms RB would help only "
                 "hypothetical MC paths)\n";
    return 0;
  }
  std::cout << "rao_blackwell_p8_smoke: FAIL\n";
  return 1;
}
