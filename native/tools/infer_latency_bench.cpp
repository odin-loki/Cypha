/// Product infer latency: DIF REST /predict hot path vs batch score; CyphaLM predict_next.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/parallel_rows.hpp"
#include "cypha/rpsm/psi_matrices.hpp"

#include <cstdlib>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point t0, int iters) {
  const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
  return (sec / static_cast<double>(iters)) * 1e6;
}

std::vector<double> load_ff_json(const std::string& path, int d, int fd) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open f_field.json: " + path);
  }
  std::stringstream b;
  b << f.rdbuf();
  auto j = nlohmann::json::parse(b.str());
  std::vector<double> out;
  for (const auto& row : j) {
    for (const auto& v : row) {
      out.push_back(v.get<double>());
    }
  }
  if (static_cast<int>(out.size()) != d * fd) {
    throw std::runtime_error("f_field.json size mismatch");
  }
  return out;
}

cypha::CyphaInferModel load_gh_fixture_model() {
  const auto repo = cypha::bench::repo_root();
  const auto dir = repo / "fixtures" / "gh_infer_deliberation";
  const auto side_path = dir / "sidecar.json";
  std::ifstream sf(side_path);
  if (!sf) {
    throw std::runtime_error("missing gh_infer_deliberation sidecar");
  }
  std::stringstream sb;
  sb << sf.rdbuf();
  nlohmann::json side = nlohmann::json::parse(sb.str());

  const auto cypha_path = dir / side.value("reference_cypha", "reference.cypha");
  cypha::CNode root = cypha::load_cypha_file(cypha_path.string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  std::vector<double> fflat =
      load_ff_json((dir / side.value("f_field_json", "f_field.json")).string(), d, fd);
  return cypha::CyphaInferModel::from_root(root, fflat.data(), fd);
}

void bench_dif(cypha::CyphaInferModel& model, const double* x, int d) {
  constexpr int kWarm = 500;
  constexpr int kIters = 20000;

  std::vector<double> h;
  cypha::batch_encode(model, x, 1, h);

  for (int i = 0; i < kWarm; ++i) {
    (void)cypha::gh_infer_at_h(model, h.data(), 1.0, 1.0, 0.98, nullptr);
  }

  const auto t0 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    (void)cypha::gh_infer_at_h(model, h.data(), 1.0, 1.0, 0.98, nullptr);
  }
  std::printf("  gh_infer_at_h_single_us=%.2f (iters=%d)\n", elapsed_us(t0, kIters), kIters);

  const auto t1 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    std::vector<double> hi;
    cypha::batch_encode(model, x, 1, hi);
    (void)cypha::gh_infer_at_h(model, hi.data(), 1.0, 1.0, 0.98, nullptr);
  }
  std::printf("  encode_plus_gh_infer_us=%.2f (iters=%d)\n", elapsed_us(t1, kIters), kIters);

  std::vector<double> llr1;
  for (int i = 0; i < kWarm; ++i) {
    cypha::score_matrix_use_field(model, h.data(), 1, llr1);
  }
  const auto t2 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    cypha::score_matrix_use_field(model, h.data(), 1, llr1);
  }
  std::printf("  score_matrix_n1_us=%.2f (iters=%d)\n", elapsed_us(t2, kIters), kIters);

  constexpr int kBatch = 32;
  std::vector<double> xb(static_cast<std::size_t>(kBatch * d));
  std::vector<double> hb;
  for (int i = 0; i < kBatch; ++i) {
    for (int j = 0; j < d; ++j) {
      xb[static_cast<std::size_t>(i * d + j)] = x[j] + 0.001 * static_cast<double>(i);
    }
  }
  cypha::batch_encode(model, xb.data(), kBatch, hb);
  std::vector<double> llr_b;
  for (int i = 0; i < kWarm; ++i) {
    cypha::score_matrix_use_field(model, hb.data(), kBatch, llr_b);
  }
  const auto t3 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    cypha::score_matrix_use_field(model, hb.data(), kBatch, llr_b);
  }
  const double batch_us = elapsed_us(t3, kIters);
  std::printf("  score_matrix_n%d_us=%.2f per_batch (%.2f per_row)\n", kBatch, batch_us,
              batch_us / static_cast<double>(kBatch));

  constexpr int kBatch256 = 256;
  std::vector<double> x256(static_cast<std::size_t>(kBatch256 * d));
  std::vector<double> h256;
  for (int i = 0; i < kBatch256; ++i) {
    for (int j = 0; j < d; ++j) {
      x256[static_cast<std::size_t>(i * d + j)] = x[j] + 0.0007 * static_cast<double>(i);
    }
  }
  cypha::batch_encode(model, x256.data(), kBatch256, h256);
  std::vector<double> llr_256;
  for (int i = 0; i < kWarm; ++i) {
    cypha::score_matrix_use_field(model, h256.data(), kBatch256, llr_256);
  }
  const auto t4 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    cypha::score_matrix_use_field(model, h256.data(), kBatch256, llr_256);
  }
  const double batch256_us = elapsed_us(t4, kIters);
  std::printf("  score_matrix_n%d_us=%.2f per_batch (%.2f per_row)\n", kBatch256, batch256_us,
              batch256_us / static_cast<double>(kBatch256));
}

void set_env_parallel_rows(const char* value) {
#if defined(_WIN32)
  std::string assign = std::string("CYPHA_SCORE_PARALLEL_ROWS=") + value;
  _putenv(assign.c_str());
#else
  setenv("CYPHA_SCORE_PARALLEL_ROWS", value, 1);
#endif
}

void bench_synth_parallel_gemm() {
  constexpr int kD = 256;
  constexpr int kK = 32;
  constexpr int kWarm = 50;
  constexpr int kIters = 400;

  cypha::rpsm::PsiMatrices psi;
  psi.feat_dim = kD;
  psi.n_classes = kK;
  psi.mu.assign(static_cast<std::size_t>((1 + kK) * kD), 0.0);
  psi.inv_var.assign(static_cast<std::size_t>(kD), 1.0);
  psi.counts.assign(static_cast<std::size_t>(kK), 10.0);
  psi.v_mean = 1.0;
  for (int j = 0; j < kD; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = 0.01 * static_cast<double>(j);
  }
  for (int k = 0; k < kK; ++k) {
    for (int j = 0; j < kD; ++j) {
      psi.mu[static_cast<std::size_t>((1 + k) * kD + j)] =
          0.1 * static_cast<double>(k + 1) + 0.001 * static_cast<double>(j);
    }
  }
  std::vector<double> ctx(static_cast<std::size_t>(kK), 0.0);

  std::printf("infer_latency_bench (synth RPSM GEMM d=%d K=%d, work-gated OpenMP):\n", kD, kK);

  for (int n : {32, 256}) {
    std::vector<double> H(static_cast<std::size_t>(n * kD));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < kD; ++j) {
        H[static_cast<std::size_t>(i * kD + j)] =
            0.02 * static_cast<double>(i) + 0.003 * static_cast<double>(j);
      }
    }
    std::vector<double> llr(static_cast<std::size_t>(n * kK));

    set_env_parallel_rows("0");
    for (int i = 0; i < kWarm; ++i) {
      cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr.data());
    }
    const auto t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
      cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr.data());
    }
    const double ser_us = elapsed_us(t0, kIters);

    set_env_parallel_rows("1");
    if (!cypha::should_parallel_score_rows(n, kD, kK)) {
      std::printf("  gemm_n%d: work gate OFF unexpectedly\n", n);
      continue;
    }
    for (int i = 0; i < kWarm; ++i) {
      cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr.data());
    }
    const auto t1 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
      cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr.data());
    }
    const double par_us = elapsed_us(t1, kIters);
    const double speedup = ser_us / std::max(par_us, 1e-9);
    std::printf("  gemm_n%d_serial_us=%.2f parallel_us=%.2f speedup=%.2fx (per_row ser=%.3f par=%.3f)\n",
                n, ser_us, par_us, speedup, ser_us / n, par_us / n);
  }
  set_env_parallel_rows("1");
}

void bench_cyphalm() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cypha::cyphalm::apply_bench_profile("d17", cfg);
  cypha::cyphalm::apply_bench_mode(cypha::cyphalm::BenchMode::Hybrid, cfg);
  cfg.seed = 42;
  if (cfg.vocab_size < 256) {
    cfg.vocab_size = 256;
  }

  cypha::cyphalm::CyphaLMModel model(cfg);
  const auto corpus =
      cypha::cyphalm::synthetic_corpus(512, cfg.vocab_size, cfg.seed);
  model.train_sequence(corpus, 256, 1, nullptr);

  constexpr int kWarm = 100;
  constexpr int kIters = 2000;
  const std::uint32_t tok = 42;

  for (int i = 0; i < kWarm; ++i) {
    (void)model.predict_next(tok);
  }

  const auto t0 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    (void)model.predict_next(tok);
  }
  std::printf("  predict_next_us=%.2f (iters=%d)\n", elapsed_us(t0, kIters), kIters);

  cypha::cyphalm::DecodeParams dp;
  dp.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
  dp.seed = 42;
  const std::vector<int> prompt{1, 2, 3};
  for (int i = 0; i < kWarm; ++i) {
    (void)cypha::cyphalm::generate_decode(model, prompt, 1, dp, nullptr);
  }
  const auto t1 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    (void)cypha::cyphalm::generate_decode(model, prompt, 1, dp, nullptr);
  }
  std::printf("  generate_1tok_us=%.2f (iters=%d)\n", elapsed_us(t1, kIters), kIters);
}

}  // namespace

int main() {
  try {
    cypha::CyphaInferModel model = load_gh_fixture_model();
    const auto repo = cypha::bench::repo_root();
    const auto side_path = repo / "fixtures" / "gh_infer_deliberation" / "sidecar.json";
    std::ifstream sf(side_path);
    std::stringstream sb;
    sb << sf.rdbuf();
    nlohmann::json side = nlohmann::json::parse(sb.str());
    std::vector<double> x = side.at("cases").at(0).at("x").get<std::vector<double>>();

    std::printf("infer_latency_bench (DIF gh_infer fixture d=%d K=%zu):\n", model.d_latent,
                model.labels.size());
    bench_dif(model, x.data(), model.d_latent);

    bench_synth_parallel_gemm();

    std::printf("infer_latency_bench (Cypha d17 hybrid synthetic):\n");
    bench_cyphalm();

    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "infer_latency_bench: %s\n", ex.what());
    return 1;
  }
}
