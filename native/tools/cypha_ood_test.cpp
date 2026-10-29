/// Cypha's learned OOD threshold (docs/reports/CYPHA_OOD_THRESHOLD.md).
///
///   cypha_ood_test selftest DIR   CTest ``native_cypha_ood_test``: ``update`` keeps each row's latent (encoded just
///                                 before learning it) in a ring of the last kOodWindow rows; the threshold is the
///                                 (1 - target) quantile of the current model's scores on them, recomputed after a
///                                 change; ``predict`` decides with it; save / load round-trips it; a file without it
///                                 falls back to the fixed 3.0. A small calibration check on synthetic data.
///   cypha_ood_test measure DIR [--seeds N] [--dims 16,64,384] [--rows R] [--passes P] [--held H] [--planted M]
///                                 The registered read (M1, M3): three synthetic families x dimensions; false OOD
///                                 rate on held-out in-distribution rows and true OOD rate on four planted sets,
///                                 for the fixed 3.0, the learned threshold (B) and a window of scores taken before
///                                 each update (A, the first registered design); the score's AUROC; cost.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace fs = std::filesystem;

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failed;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

std::string unique_name(const std::string& stem) {
  static int counter = 0;
  const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
  return stem + "_" + std::to_string(t) + "_" + std::to_string(counter++);
}

/// A fresh model (identity encoder, latent = input dimension), loaded through ``Cypha::load``.
bool fresh_model(cypha::Cypha& c, int d, const fs::path& dir) {
  cypha::FreshModelParams fp;
  fp.input_dim = d;
  fp.field_dim = 16;
  const fs::path path = dir / (unique_name("fresh") + ".cypha");
  cypha::save_cypha_file(path.string().c_str(), cypha::create_fresh_model_root(fp));
  const bool ok = c.load(path.string());
  std::error_code ec;
  fs::remove(path, ec);
  return ok;
}

// ---------------------------------------------------------------- synthetic data (M1)

/// F1 unit per dimension, F2 unit norm (F1 / sqrt(d)), F3 correlated (a d/4 latent drawn as F1, mapped by A, plus
/// small noise). K = 5 classes. Planted sets: P1 noise doubled, P2 a far cluster, P3 uniform, P4 stretched sqrt(10).
class Synth {
 public:
  static constexpr int kK = 5;
  Synth(int family, int d, unsigned seed) : fam_(family), d_(d), r_(family == 3 ? std::max(1, d / 4) : d), rng_(seed) {
    std::normal_distribution<double> n01(0.0, 1.0);
    for (int k = 0; k < kK; ++k) {
      std::vector<double> m(static_cast<std::size_t>(r_));
      for (double& v : m) v = 1.5 * n01(rng_);
      means_.push_back(std::move(m));
    }
    far_.resize(static_cast<std::size_t>(r_));
    for (double& v : far_) v = 4.5 * n01(rng_);
    if (fam_ == 3) {
      A_.resize(static_cast<std::size_t>(d_) * static_cast<std::size_t>(r_));
      const double sd = 2.0 / std::sqrt(static_cast<double>(d_));
      for (double& v : A_) v = sd * n01(rng_);
    }
  }

  /// One row from latent mean ``m`` with noise scale ``noise``.
  std::vector<double> row(const std::vector<double>& m, double noise) {
    std::normal_distribution<double> n01(0.0, 1.0);
    std::vector<double> z(static_cast<std::size_t>(r_));
    for (int j = 0; j < r_; ++j) z[static_cast<std::size_t>(j)] = m[static_cast<std::size_t>(j)] + noise * n01(rng_);
    return map(z);
  }
  std::vector<double> in_row(int k) { return row(means_[static_cast<std::size_t>(k)], 1.0); }
  std::vector<double> p1_row(int k) { return row(means_[static_cast<std::size_t>(k)], 2.0); }
  std::vector<double> p2_row() { return row(far_, 1.0); }
  std::mt19937& rng() { return rng_; }

 private:
  std::vector<double> map(const std::vector<double>& z) {
    std::vector<double> x(static_cast<std::size_t>(d_));
    if (fam_ == 1) {
      x = z;
    } else if (fam_ == 2) {
      const double s = 1.0 / std::sqrt(static_cast<double>(d_));
      for (int j = 0; j < d_; ++j) x[static_cast<std::size_t>(j)] = s * z[static_cast<std::size_t>(j)];
    } else {
      std::normal_distribution<double> n01(0.0, 1.0);
      for (int i = 0; i < d_; ++i) {
        double acc = 0.0;
        for (int j = 0; j < r_; ++j) acc += A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(r_) + static_cast<std::size_t>(j)] * z[static_cast<std::size_t>(j)];
        x[static_cast<std::size_t>(i)] = acc + 0.1 * n01(rng_);
      }
    }
    return x;
  }

  int fam_, d_, r_;
  std::mt19937 rng_;
  std::vector<std::vector<double>> means_;
  std::vector<double> far_;
  std::vector<double> A_;
};

struct Rows {
  std::vector<std::vector<double>> x;
  std::vector<std::string> y;
};

Rows train_rows(Synth& s, int n) {
  Rows r;
  std::uniform_int_distribution<int> pick(0, Synth::kK - 1);
  for (int i = 0; i < n; ++i) {
    const int k = pick(s.rng());
    r.x.push_back(s.in_row(k));
    r.y.push_back("c" + std::to_string(k));
  }
  return r;
}

void train(cypha::Cypha& c, const Rows& r, int passes) {
  cypha::UpdateOpts uo;
  uo.use_gh = true;
  for (int p = 0; p < passes; ++p) {
    for (std::size_t i = 0; i < r.x.size(); ++i) {
      const auto u = c.update(r.x[i].data(), static_cast<int>(r.x[i].size()), &r.y[i], nullptr, uo);
      if (!u.detail.empty()) {
        std::fprintf(stderr, "update: %s\n", u.detail.c_str());
        std::exit(2);
      }
    }
  }
}

struct Scored {
  std::vector<double> score;
  int flagged_new{0};
  int flagged_old{0};
  int mismatch{0};  // is_ood != (score > ood_threshold())
};

Scored score_rows(cypha::Cypha& c, const std::vector<std::vector<double>>& xs) {
  Scored s;
  const double tau = c.ood_threshold();
  for (const auto& x : xs) {
    const cypha::PredictOut p = c.predict(x.data(), static_cast<int>(x.size()));
    s.score.push_back(p.anomaly_score);
    s.flagged_new += p.is_ood ? 1 : 0;
    s.flagged_old += p.anomaly_score > cypha::Cypha::kOodThreshold ? 1 : 0;
    s.mismatch += p.is_ood != (p.anomaly_score > tau) ? 1 : 0;
  }
  return s;
}

/// P(score of an OOD row > score of an in-distribution row), ties counted half.
double auroc(std::vector<double> in, const std::vector<double>& out) {
  std::sort(in.begin(), in.end());
  double acc = 0.0;
  for (double o : out) {
    const auto lo = std::lower_bound(in.begin(), in.end(), o);
    const auto hi = std::upper_bound(in.begin(), in.end(), o);
    acc += static_cast<double>(lo - in.begin()) + 0.5 * static_cast<double>(hi - lo);
  }
  return out.empty() || in.empty() ? 0.5 : acc / (static_cast<double>(in.size()) * static_cast<double>(out.size()));
}

void wilson(int k, int n, double& lo, double& hi) {
  if (n <= 0) {
    lo = 0;
    hi = 1;
    return;
  }
  const double z = 1.959963984540054, p = static_cast<double>(k) / n, z2 = z * z / n;
  const double c = (p + z2 / 2) / (1 + z2), h = z * std::sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / (1 + z2);
  lo = std::max(0.0, c - h);
  hi = std::min(1.0, c + h);
}

/// The independent check of the threshold: s[ceil((1 - a) n) - 1] of the sorted scores.
double quantile_ref(std::vector<double> v, double target) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  std::size_t k = static_cast<std::size_t>(std::ceil((1.0 - target) * static_cast<double>(n)));
  k = std::min(n, std::max<std::size_t>(k, 1)) - 1;
  return v[k];
}

// ---------------------------------------------------------------- selftest

int selftest(const fs::path& dir) {
  fs::create_directories(dir);
  // (1)-(4) the bookkeeping, d = 16, unit norm (scores vary there; in F1 most in-distribution rows score 0).
  {
    Synth s(2, 16, 7);
    const Rows r = train_rows(s, 1100);
    cypha::Cypha c;
    check(fresh_model(c, 16, dir), "fresh model loads");
    std::vector<std::vector<double>> expect;  // the latents update should keep, encoded just before each update
    cypha::UpdateOpts uo;
    uo.use_gh = true;
    for (std::size_t i = 0; i < r.x.size(); ++i) {
      if (!c.infer()->labels.empty()) {
        std::vector<double> h;
        cypha::batch_encode(*c.infer(), r.x[i].data(), 16, h);
        expect.push_back(std::move(h));
      }
      c.update(r.x[i].data(), 16, &r.y[i], nullptr, uo);
      if (expect.size() == cypha::Cypha::kOodMinScores - 1) {
        check(c.ood_threshold() == cypha::Cypha::kOodThreshold, "fewer than kOodMinScores rows: the fixed 3.0");
      }
    }
    // The ring holds the latents of the last kOodWindow rows, bit for bit, in ring order.
    const std::size_t W = cypha::Cypha::kOodWindow;
    check(expect.size() > W, "more rows than the window");
    const auto& ring = c.ood_latents();
    check(c.ood_rows_held() == W && ring.size() == W * 16, "the ring is full");
    check(c.ood_head() == expect.size() % W, "the ring's head");
    bool same = ring.size() == W * 16;
    for (std::size_t j = 0; same && j < W; ++j) {
      // slot j holds the latent of the last row written there
      const std::size_t last = ((expect.size() - 1 - j) / W) * W + j;
      for (std::size_t t = 0; same && t < 16; ++t) same = same_bits(ring[j * 16 + t], expect[last][t]);
    }
    check(same, "update keeps each row's latent, encoded before learning it, in a ring of the last kOodWindow");
    auto current_scores = [&](cypha::Cypha& m) {
      std::vector<double> v;
      const auto& lat = m.ood_latents();
      for (std::size_t r0 = 0; r0 < m.ood_rows_held(); ++r0)
        v.push_back(cypha::gh_anomaly_score_at_h(*m.infer(), lat.data() + r0 * 16, m.gh_chi(), m.gh_psi()));
      return v;
    };
    const double tau = c.ood_threshold();
    check(same_bits(tau, quantile_ref(current_scores(c), 0.05)),
          "the threshold is the 95% quantile of the current model's scores on the held latents");
    // (2) predict decides with it, and its score is the shared helper's.
    int bad = 0;
    for (int i = 0; i < 200; ++i) {
      const std::vector<double> x = i % 2 == 0 ? s.in_row(i % Synth::kK) : s.p1_row(i % Synth::kK);
      const cypha::PredictOut p = c.predict(x.data(), 16);
      std::vector<double> h;
      cypha::batch_encode(*c.infer(), x.data(), 1, h);
      const double ref = cypha::gh_anomaly_score_at_h(*c.infer(), h.data(), c.gh_chi(), c.gh_psi());
      bad += (!same_bits(p.anomaly_score, ref) || p.is_ood != (p.anomaly_score > tau)) ? 1 : 0;
    }
    check(bad == 0, "predict's anomaly score is the helper's and is_ood compares it with the learned threshold");
    // (7) the target rate.
    c.set_ood_target_rate(0.2);
    check(same_bits(c.ood_threshold(), quantile_ref(current_scores(c), 0.2)), "a 20% target gives the 80% quantile");
    // A change to the GH session re-scores (the cache is keyed on it).
    const double chi0 = c.gh_chi();
    c.gh_chi() = chi0 * 1.5;
    check(same_bits(c.ood_threshold(), quantile_ref(current_scores(c), 0.2)), "a changed GH session re-scores");
    c.gh_chi() = chi0;
    bool threw = false;
    try {
      c.set_ood_target_rate(1.5);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "a target outside (0, 1) is refused");
    c.set_ood_target_rate(0.05);

    // (6) save / load.
    const std::string fa = (dir / "ood_a.cypha").string();
    c.save(fa);
    cypha::Cypha b;
    check(b.load(fa), "reload");
    check(b.ood_rows_held() == c.ood_rows_held() && b.ood_head() == c.ood_head() &&
              same_bits(b.ood_threshold(), c.ood_threshold()) && same_bits(b.ood_target_rate(), c.ood_target_rate()),
          "the ring, head, target and threshold reload");
    bool same_lat = b.ood_latents().size() == ring.size();
    for (std::size_t j = 0; same_lat && j < ring.size(); ++j) same_lat = same_bits(b.ood_latents()[j], ring[j]);
    check(same_lat, "the ring's latents reload bit for bit");
    int diff = 0;
    for (int i = 0; i < 100; ++i) {
      const std::vector<double> x = s.p1_row(i % Synth::kK);
      const auto pa = c.predict(x.data(), 16);
      const auto pb = b.predict(x.data(), 16);
      diff += (pa.is_ood != pb.is_ood || !same_bits(pa.anomaly_score, pb.anomaly_score)) ? 1 : 0;
    }
    check(diff == 0, "predictions agree after a reload");
    for (int i = 0; i < 60; ++i) {
      c.update(r.x[static_cast<std::size_t>(i)].data(), 16, &r.y[static_cast<std::size_t>(i)], nullptr, uo);
      b.update(r.x[static_cast<std::size_t>(i)].data(), 16, &r.y[static_cast<std::size_t>(i)], nullptr, uo);
    }
    check(c.ood_head() == b.ood_head() && same_bits(c.ood_threshold(), b.ood_threshold()),
          "the threshold keeps agreeing while both keep learning");

    // A file without the keys (older format 2, or format 1) falls back to the fixed 3.0.
    cypha::CNode root = cypha::load_cypha_file(fa.c_str());
    for (auto& kv : root.map) {
      if (kv.first != "cypha_save_state") continue;
      for (auto& sv : kv.second.map) {
        if (sv.first != "session") continue;
        auto& m = sv.second.map;
        m.erase(std::remove_if(m.begin(), m.end(),
                               [](const auto& e) {
                                 return e.first == "ood_target" || e.first == "ood_count" || e.first == "ood_head" ||
                                        e.first == "ood_latents";
                               }),
                m.end());
      }
    }
    const std::string fo = (dir / "ood_old.cypha").string();
    cypha::save_cypha_file(fo.c_str(), root);
    cypha::Cypha o;
    check(o.load(fo), "a file without the OOD keys loads");
    check(o.ood_rows_held() == 0 && o.ood_threshold() == cypha::Cypha::kOodThreshold && o.ood_target_rate() == 0.05,
          "a file without the OOD keys has the fixed 3.0 until it learns");
    // Loading into an object that had learned resets the ring.
    check(c.load(fo) && c.ood_rows_held() == 0, "load resets a previous ring");
    std::error_code ec;
    fs::remove(fa, ec);
    fs::remove(fo, ec);
  }
  // (5) calibration, d = 64, unit norm (the fixed 3.0's bad case): held-out false rate near 5%, P4 seen.
  {
    Synth s(2, 64, 11);
    const Rows r = train_rows(s, 700);
    cypha::Cypha c;
    check(fresh_model(c, 64, dir), "fresh model loads (d 64)");
    train(c, r, 1);
    std::vector<std::vector<double>> held, p4;
    std::uniform_int_distribution<int> pick(0, Synth::kK - 1);
    std::vector<double> mean(64, 0.0);
    for (const auto& x : r.x)
      for (int j = 0; j < 64; ++j) mean[static_cast<std::size_t>(j)] += x[static_cast<std::size_t>(j)] / static_cast<double>(r.x.size());
    for (int i = 0; i < 800; ++i) held.push_back(s.in_row(pick(s.rng())));
    for (int i = 0; i < 300; ++i) {
      std::vector<double> x = s.in_row(pick(s.rng()));
      for (int j = 0; j < 64; ++j) x[static_cast<std::size_t>(j)] = mean[static_cast<std::size_t>(j)] + std::sqrt(10.0) * (x[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)]);
      p4.push_back(std::move(x));
    }
    const Scored h = score_rows(c, held), q = score_rows(c, p4);
    const double fr = static_cast<double>(h.flagged_new) / static_cast<double>(held.size());
    const double tr = static_cast<double>(q.flagged_new) / static_cast<double>(p4.size());
    std::printf("selftest calibration (F2, d 64): threshold %.4g, held-out false OOD %.3f (fixed 3.0: %.3f), P4 %.3f (fixed 3.0: %.3f)\n",
                c.ood_threshold(), fr, static_cast<double>(h.flagged_old) / static_cast<double>(held.size()), tr,
                static_cast<double>(q.flagged_old) / static_cast<double>(p4.size()));
    check(fr >= 0.02 && fr <= 0.10, "held-out false OOD rate near the 5% target");
    check(tr >= 0.90, "variance x 10 rows are flagged");
    check(h.mismatch == 0 && q.mismatch == 0, "is_ood follows the threshold");
  }
  std::printf("cypha_ood_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- measure (M1, M3)

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
  return v;
}

int measure(const fs::path& dir, int argc, char** argv) {
  fs::create_directories(dir);
  int seeds = 3, rows = 1500, passes = 2, held_n = 2000, planted_n = 1000;
  std::vector<int> dims = {16, 64, 384};
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--seeds") seeds = std::atoi(nx().c_str());
    else if (a == "--dims") dims = parse_ints(nx());
    else if (a == "--rows") rows = std::atoi(nx().c_str());
    else if (a == "--passes") passes = std::atoi(nx().c_str());
    else if (a == "--held") held_n = std::atoi(nx().c_str());
    else if (a == "--planted") planted_n = std::atoi(nx().c_str());
  }
  const char* fam_name[] = {"", "F1 unit per dim", "F2 unit norm", "F3 correlated"};
  const char* set_name[] = {"held-out", "P1 noise x2", "P2 far cluster", "P3 uniform", "P4 stretched"};
  std::printf("family\td\tset\tn\tfixed3_flagged\tfixed3_rate\tfixed3_lo\tfixed3_hi\tlearnedB_flagged\tlearnedB_rate\tlearnedB_lo\tlearnedB_hi\twindowA_flagged\twindowA_rate\tauroc\ttauB_median\ttauA_median\tscore_median_heldout\tmismatch\n");
  for (int fam = 1; fam <= 3; ++fam) {
    for (int d : dims) {
      int old_k[5] = {0, 0, 0, 0, 0}, new_k[5] = {0, 0, 0, 0, 0}, n_k[5] = {0, 0, 0, 0, 0}, mism = 0;
      std::vector<double> au[5], taus, med;
      double upd_s = 0.0, extra_s = 0.0, rescore_s = 0.0;
      long n_upd = 0, n_rescore = 0;
      int a_k[5] = {0, 0, 0, 0, 0};
      std::vector<double> taus_a;
      for (int seed = 1; seed <= seeds; ++seed) {
        Synth s(fam, d, static_cast<unsigned>(1000 * fam + 10 * d + seed));
        const Rows r = train_rows(s, rows);
        cypha::Cypha c;
        if (!fresh_model(c, d, dir)) return 2;
        // Train; arm A keeps the last kOodWindow scores predict gives each row just before its update.
        std::vector<double> win;
        std::size_t win_head = 0;
        cypha::UpdateOpts uo;
        uo.use_gh = true;
        for (int p = 0; p < passes; ++p)
          for (std::size_t i = 0; i < r.x.size(); ++i) {
            if (!c.infer()->labels.empty()) {
              const double sc0 = c.predict(r.x[i].data(), d).anomaly_score;
              if (win.size() < cypha::Cypha::kOodWindow) win.push_back(sc0);
              else {
                win[win_head] = sc0;
                win_head = (win_head + 1) % cypha::Cypha::kOodWindow;
              }
            }
            const auto t0 = std::chrono::steady_clock::now();
            const auto u = c.update(r.x[i].data(), d, &r.y[i], nullptr, uo);
            upd_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!u.detail.empty()) {
              std::fprintf(stderr, "update: %s\n", u.detail.c_str());
              return 2;
            }
          }
        n_upd += static_cast<long>(rows) * passes;
        const double tau_a = quantile_ref(win, 0.05);
        // M3: the work update adds (the training step's latent copied out, then into the ring), timed alone.
        {
          std::vector<std::vector<double>> hs;
          for (const auto& x : r.x) {
            std::vector<double> h;
            cypha::batch_encode(*c.infer(), x.data(), 1, h);
            hs.push_back(std::move(h));
          }
          std::vector<double> ring_copy;
          ring_copy.reserve(cypha::Cypha::kOodWindow * static_cast<std::size_t>(d));
          std::size_t head = 0, count = 0;
          const auto t1 = std::chrono::steady_clock::now();
          double sink = 0.0;
          for (int p = 0; p < passes; ++p)
            for (const auto& h : hs) {
              std::vector<double> out = h;  // the copy the training step writes
              if (count < cypha::Cypha::kOodWindow) {
                ring_copy.insert(ring_copy.end(), out.begin(), out.end());
                ++count;
              } else {
                std::copy(out.begin(), out.end(), ring_copy.begin() + static_cast<std::ptrdiff_t>(head * static_cast<std::size_t>(d)));
                head = (head + 1) % cypha::Cypha::kOodWindow;
              }
              sink += out[0];
            }
          extra_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
          if (sink == -1.0) std::printf("#\n");
        }
        taus_a.push_back(tau_a);
        // Training statistics for P3 and P4.
        std::vector<double> mean(static_cast<std::size_t>(d), 0.0), sd(static_cast<std::size_t>(d), 0.0);
        for (const auto& x : r.x)
          for (int j = 0; j < d; ++j) mean[static_cast<std::size_t>(j)] += x[static_cast<std::size_t>(j)];
        for (double& m : mean) m /= static_cast<double>(r.x.size());
        for (const auto& x : r.x)
          for (int j = 0; j < d; ++j) {
            const double e = x[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)];
            sd[static_cast<std::size_t>(j)] += e * e;
          }
        for (double& v : sd) v = std::sqrt(v / static_cast<double>(r.x.size()));
        std::uniform_int_distribution<int> pick(0, Synth::kK - 1);
        std::vector<std::vector<double>> sets[5];
        for (int i = 0; i < held_n; ++i) sets[0].push_back(s.in_row(pick(s.rng())));
        for (int i = 0; i < planted_n; ++i) sets[1].push_back(s.p1_row(pick(s.rng())));
        for (int i = 0; i < planted_n; ++i) sets[2].push_back(s.p2_row());
        std::uniform_real_distribution<double> u(-3.0, 3.0);
        for (int i = 0; i < planted_n; ++i) {
          std::vector<double> x(static_cast<std::size_t>(d));
          for (int j = 0; j < d; ++j) x[static_cast<std::size_t>(j)] = mean[static_cast<std::size_t>(j)] + u(s.rng()) * sd[static_cast<std::size_t>(j)];
          sets[3].push_back(std::move(x));
        }
        for (int i = 0; i < planted_n; ++i) {
          std::vector<double> x = s.in_row(pick(s.rng()));
          for (int j = 0; j < d; ++j) x[static_cast<std::size_t>(j)] = mean[static_cast<std::size_t>(j)] + std::sqrt(10.0) * (x[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)]);
          sets[4].push_back(std::move(x));
        }
        Scored sc[5];
        for (int t = 0; t < 5; ++t) {
          sc[t] = score_rows(c, sets[t]);
          old_k[t] += sc[t].flagged_old;
          new_k[t] += sc[t].flagged_new;
          n_k[t] += static_cast<int>(sets[t].size());
          mism += sc[t].mismatch;
        }
        for (int t = 0; t < 5; ++t)
          for (double v : sc[t].score) a_k[t] += v > taus_a.back() ? 1 : 0;
        for (int t = 1; t < 5; ++t) au[t].push_back(auroc(sc[0].score, sc[t].score));
        taus.push_back(c.ood_threshold());
        std::printf("#tau\t%s\td %d\tseed %d\ttauB %.6g\ttauA %.6g\n", fam_name[fam], d, seed, taus.back(), taus_a.back());
        {  // M3: the re-score after one more row (after every set was scored, so the read is unaffected)
          c.update(r.x[0].data(), d, &r.y[0], nullptr, uo);
          const auto t2 = std::chrono::steady_clock::now();
          const double t_new = c.ood_threshold();
          rescore_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
          ++n_rescore;
          if (t_new == -1.0) std::printf("#\n");
        }
        std::vector<double> hs = sc[0].score;
        std::nth_element(hs.begin(), hs.begin() + static_cast<std::ptrdiff_t>(hs.size() / 2), hs.end());
        med.push_back(hs[hs.size() / 2]);
      }
      std::sort(taus.begin(), taus.end());
      std::sort(med.begin(), med.end());
      std::vector<double> taus_a_sorted = taus_a;
      std::sort(taus_a_sorted.begin(), taus_a_sorted.end());
      for (int t = 0; t < 5; ++t) {
        double ol, oh, nl, nh;
        wilson(old_k[t], n_k[t], ol, oh);
        wilson(new_k[t], n_k[t], nl, nh);
        double a = 0.0;
        for (double v : au[t]) a += v;
        a = au[t].empty() ? NAN : a / static_cast<double>(au[t].size());
        std::printf("%s\t%d\t%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4g\t%.4g\t%.4g\t%d\n", fam_name[fam],
                    d, set_name[t], n_k[t], old_k[t], static_cast<double>(old_k[t]) / n_k[t], ol, oh, new_k[t],
                    static_cast<double>(new_k[t]) / n_k[t], nl, nh, a_k[t], static_cast<double>(a_k[t]) / n_k[t], a,
                    taus[taus.size() / 2], taus_a_sorted[taus_a_sorted.size() / 2], med[med.size() / 2], mism);
      }
      std::printf("#cost\t%s\td %d\tupdate_us_per_row %.2f\tadded_us_per_row %.2f\tadded_share %.4f\trescore_ms %.3f\n",
                  fam_name[fam], d, 1e6 * upd_s / static_cast<double>(n_upd), 1e6 * extra_s / static_cast<double>(n_upd),
                  extra_s / std::max(upd_s - extra_s, 1e-12), 1e3 * rescore_s / static_cast<double>(std::max<long>(n_rescore, 1)));
      std::fflush(stdout);
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: cypha_ood_test selftest DIR | measure DIR [--seeds N] [--dims 16,64,384] [--rows R] [--passes P] [--held H] [--planted M]\n");
    return 2;
  }
  const std::string mode = argv[1];
  const fs::path dir = argv[2];
  try {
    if (mode == "selftest") return selftest(dir);
    if (mode == "measure") return measure(dir, argc - 3, argv + 3);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "cypha_ood_test: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
