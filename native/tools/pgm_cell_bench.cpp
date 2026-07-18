/// Microbench: native PGMCell::step vs CharLSTMHead::forward_step latency.
/// Supports larger slot counts (up to kMaxSlots=262144) via --scale / --config.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/pgm_cell.hpp"

namespace {

using clock = std::chrono::steady_clock;

double median_us(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void fill_unit(std::vector<double>& x, int t, std::uint64_t seed) {
  double n2 = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double u = std::sin(0.137 * (t + 1) * (static_cast<int>(i) + 1) + 0.01 * seed);
    x[i] = u;
    n2 += u * u;
  }
  const double inv = 1.0 / std::sqrt(std::max(n2, 1e-12));
  for (double& v : x) v *= inv;
}

struct Row {
  const char* name;
  int d;
  int n_slots;
  int steps;
  double med_us;
  double p95_us;
  double tok_per_s;
  std::size_t edges;
  std::size_t occupied;
};

Row bench_pgm(int d, int n_sub, int levels, int steps, int warmup) {
  using cypha::cyphalm::PGMCell;
  using cypha::cyphalm::PGMCellConfig;
  PGMCellConfig cfg;
  cfg.d_input = d;
  cfg.hidden = d;
  cfg.n_sub = n_sub;
  cfg.levels = levels;
  cfg.chunk_len = 16;
  cfg.topk = 4;
  cfg.beam = 2;
  cfg.rehash_t = 16;
  cfg.hops = 2;
  cfg.seed = 42;
  PGMCell cell(cfg);

  std::vector<double> x(static_cast<std::size_t>(d), 0.0);
  for (int t = 0; t < warmup; ++t) {
    fill_unit(x, t, 1);
    (void)cell.step(x);
  }
  cell.reset();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(steps));
  const auto t0 = clock::now();
  for (int t = 0; t < steps; ++t) {
    fill_unit(x, t, 2);
    const auto a = clock::now();
    (void)cell.step(x);
    const auto b = clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(b - a).count());
  }
  const auto t1 = clock::now();
  const double wall_s = std::chrono::duration<double>(t1 - t0).count();

  std::vector<double> sorted = samples;
  const double med = median_us(sorted);
  const double p95 = sorted[static_cast<std::size_t>(0.95 * (sorted.size() - 1))];

  Row r;
  r.name = "PGM";
  r.d = d;
  r.n_slots = cell.n_slots();
  r.steps = steps;
  r.med_us = med;
  r.p95_us = p95;
  r.tok_per_s = steps / std::max(wall_s, 1e-12);
  r.edges = cell.edge_count();
  r.occupied = cell.occupied_count();
  return r;
}

Row bench_lstm(int hidden, int vocab, int steps, int warmup) {
  using cypha::cyphalm::CharLSTMHead;
  CharLSTMHead lstm(vocab, hidden, /*seed=*/42);
  std::vector<double> h(static_cast<std::size_t>(hidden), 0.0);
  std::vector<double> c(static_cast<std::size_t>(hidden), 0.0);
  std::vector<double> log_probs(static_cast<std::size_t>(vocab), 0.0);
  std::vector<double> h_out, c_out;

  for (int t = 0; t < warmup; ++t) {
    lstm.forward_step(t % vocab, h.data(), c.data(), log_probs.data(), h_out, c_out);
    h.swap(h_out);
    c.swap(c_out);
  }
  h.assign(static_cast<std::size_t>(hidden), 0.0);
  c.assign(static_cast<std::size_t>(hidden), 0.0);

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(steps));
  const auto t0 = clock::now();
  for (int t = 0; t < steps; ++t) {
    const auto a = clock::now();
    lstm.forward_step(t % vocab, h.data(), c.data(), log_probs.data(), h_out, c_out);
    const auto b = clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    h.swap(h_out);
    c.swap(c_out);
  }
  const auto t1 = clock::now();
  const double wall_s = std::chrono::duration<double>(t1 - t0).count();
  std::vector<double> sorted = samples;
  const double med = median_us(sorted);
  const double p95 = sorted[static_cast<std::size_t>(0.95 * (sorted.size() - 1))];

  Row r;
  r.name = "LSTM";
  r.d = hidden;
  r.n_slots = 0;
  r.steps = steps;
  r.med_us = med;
  r.p95_us = p95;
  r.tok_per_s = steps / std::max(wall_s, 1e-12);
  r.edges = 0;
  r.occupied = 0;
  return r;
}

void print_row(const char* label, const Row& r) {
  if (r.n_slots > 0) {
    std::printf("%-36s  d=%4d  N=%7d  steps=%6d  med=%8.2f us  p95=%8.2f us  "
                "thru=%10.0f tok/s  edges=%7zu  occ=%7zu\n",
                label, r.d, r.n_slots, r.steps, r.med_us, r.p95_us, r.tok_per_s, r.edges,
                r.occupied);
  } else {
    std::printf("%-36s  d=%4d  %-10s  steps=%6d  med=%8.2f us  p95=%8.2f us  "
                "thru=%10.0f tok/s\n",
                label, r.d, "(dense)", r.steps, r.med_us, r.p95_us, r.tok_per_s);
  }
}

struct PgmCfg {
  const char* label;
  int d;
  int n_sub;
  int levels;
  int steps_override;  // 0 = use global --steps
};

void usage() {
  std::puts(
      "usage: pgm_cell_bench [--steps N] [--warmup N] [--scale default|large|xl]\n"
      "                      [--config d,n_sub,levels[:steps]]...\n"
      "  --scale default : H23-ish through 65k slots (original set)\n"
      "  --scale large   : + 4k / 16k / 65k / 262k slot configs\n"
      "  --scale xl      : large + wider d (128/160) at high N (fewer steps)\n"
      "  --config        : ad-hoc PGM config; repeatable. Optional :steps suffix.\n");
}

bool parse_config(const std::string& s, PgmCfg& out) {
  // d,n_sub,levels or d,n_sub,levels:steps
  int d = 0, n_sub = 0, levels = 0, steps = 0;
  const std::size_t colon = s.find(':');
  const std::string head = colon == std::string::npos ? s : s.substr(0, colon);
  if (colon != std::string::npos) {
    steps = std::atoi(s.c_str() + static_cast<std::ptrdiff_t>(colon) + 1);
  }
  if (std::sscanf(head.c_str(), "%d,%d,%d", &d, &n_sub, &levels) != 3) {
    return false;
  }
  if (d < 1 || n_sub < 2 || levels < 1) return false;
  out.label = "PGM custom";
  out.d = d;
  out.n_sub = n_sub;
  out.levels = levels;
  out.steps_override = steps;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int steps = 20000;
  int warmup = 500;
  std::string scale = "default";
  std::vector<PgmCfg> custom;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "--steps" || a == "-n") && i + 1 < argc) {
      steps = std::atoi(argv[++i]);
    } else if (a == "--warmup" && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (a == "--scale" && i + 1 < argc) {
      scale = argv[++i];
    } else if (a == "--config" && i + 1 < argc) {
      PgmCfg cfg{};
      if (!parse_config(argv[++i], cfg)) {
        std::fprintf(stderr, "pgm_cell_bench: bad --config (want d,n_sub,levels[:steps])\n");
        usage();
        return 2;
      }
      custom.push_back(cfg);
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "pgm_cell_bench: unknown arg %s\n", a.c_str());
      usage();
      return 2;
    }
  }

  if (scale != "default" && scale != "large" && scale != "xl") {
    std::fprintf(stderr, "pgm_cell_bench: --scale must be default|large|xl\n");
    return 2;
  }

  std::puts("=================================================================");
  std::puts("PGMCell native microbench (median / p95 per step, steady_clock)");
  std::puts("=================================================================");
  std::printf("steps=%d warmup=%d scale=%s  (H23 defaults: topk=4 beam=2 rehash_t=16 hops=2)\n\n",
              steps, warmup, scale.c_str());

  // Cypha-typical widths
  print_row("CharLSTM (D17-like)", bench_lstm(/*hidden=*/128, /*vocab=*/128, steps, warmup));
  print_row("CharLSTM (field-like)", bench_lstm(/*hidden=*/64, /*vocab=*/128, steps, warmup));
  print_row("CharLSTM (tiny)", bench_lstm(/*hidden=*/32, /*vocab=*/64, steps, warmup));
  if (scale == "xl") {
    print_row("CharLSTM (wide 256)", bench_lstm(/*hidden=*/256, /*vocab=*/128, steps, warmup));
    print_row("CharLSTM (wide 512)", bench_lstm(/*hidden=*/512, /*vocab=*/128,
                                               std::max(2000, steps / 4), warmup));
  }

  std::puts("");

  std::vector<PgmCfg> suite;
  if (custom.empty()) {
    // Default set (original)
    suite.push_back({"PGM H23-ish (b=8 L=3)", 64, 8, 3, 0});
    suite.push_back({"PGM field_dim (b=8 L=3)", 160, 8, 3, 0});
    suite.push_back({"PGM large N (b=16 L=3)", 64, 16, 3, 0});
    suite.push_back({"PGM 65k slots (b=16 L=4)", 64, 16, 4, 0});
    suite.push_back({"PGM wide+65k", 128, 16, 4, 0});
    if (scale == "large" || scale == "xl") {
      // Larger addressing capacity (N = b^L, capped at 262144)
      suite.push_back({"PGM 4k slots (b=8 L=4)", 64, 8, 4, 0});
      suite.push_back({"PGM 16k slots (b=8 L=5)", 64, 8, 5, 0});
      suite.push_back({"PGM 262k cap (b=8 L=6)", 64, 8, 6, std::max(2000, steps / 5)});
      suite.push_back({"PGM 262k cap (b=64 L=3)", 64, 64, 3, std::max(2000, steps / 5)});
    }
    if (scale == "xl") {
      suite.push_back({"PGM wide d=128 @262k", 128, 8, 6, std::max(1000, steps / 10)});
      suite.push_back({"PGM field d=160 @65k", 160, 16, 4, std::max(2000, steps / 4)});
      suite.push_back({"PGM field d=160 @262k", 160, 8, 6, std::max(1000, steps / 10)});
    }
  } else {
    suite = custom;
  }

  for (const PgmCfg& c : suite) {
    const int n = c.steps_override > 0 ? c.steps_override : steps;
    char label[96];
    if (custom.empty()) {
      std::snprintf(label, sizeof(label), "%s", c.label);
    } else {
      std::snprintf(label, sizeof(label), "PGM d=%d b=%d L=%d", c.d, c.n_sub, c.levels);
    }
    print_row(label, bench_pgm(c.d, c.n_sub, c.levels, n, warmup));
  }

  std::puts("\ndone.");
  return 0;
}
