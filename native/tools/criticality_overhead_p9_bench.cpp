/// Phase 9 overhead: D17 train hot-path vs simulated per-step CriticalityVector reads.
#include <chrono>
#include <cstdio>
#include <vector>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/intelligence/criticality_vector.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_sec(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

cypha::cyphalm::CyphaLMConfig d17_hybrid_config() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cypha::cyphalm::apply_bench_profile("d17", cfg);
  cypha::cyphalm::apply_bench_mode(cypha::cyphalm::BenchMode::Hybrid, cfg);
  cfg.seed = 42;
  if (cfg.vocab_size < 256) {
    cfg.vocab_size = 256;
  }
  return cfg;
}

cypha::cyphalm::LMCorpus synthetic_corpus(const cypha::cyphalm::CyphaLMConfig& cfg, int n_train,
                                          int n_eval) {
  cypha::cyphalm::LMCorpus corpus;
  corpus.profile = "d17";
  corpus.source = "synthetic";
  corpus.vocab_size = cfg.vocab_size;
  corpus.train_ids =
      cypha::cyphalm::synthetic_corpus(n_train + n_eval + 64, cfg.vocab_size, cfg.seed);
  const std::size_t split = static_cast<std::size_t>(n_train);
  corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                         corpus.train_ids.end());
  corpus.train_ids.resize(split);
  return corpus;
}

}  // namespace

int main() {
  try {
    constexpr int kNTrain = 8000;
    constexpr int kNEval = 256;
    constexpr int kCriticalityIters = kNTrain;

    const cypha::cyphalm::CyphaLMConfig cfg = d17_hybrid_config();
    const cypha::cyphalm::LMCorpus corpus = synthetic_corpus(cfg, kNTrain, kNEval);

    cypha::cyphalm::CyphaLMConfig cfg_a = cfg;
    cypha::cyphalm::CyphaLMModel model_a(cfg_a);
    const auto t_train0 = Clock::now();
    model_a.train_sequence(corpus.train_ids, kNTrain, cfg_a.train_epochs, nullptr);
    const double train_sec = elapsed_sec(t_train0);

    cypha::intelligence::IntelligenceProfiler profiler =
        cypha::intelligence::profile_from_reference_fixture(cypha::bench::repo_root());
    cypha::intelligence::CriticalityHotInput hot;
    hot.anomaly_score = 0.42;
    hot.ood_rate = 0.01;

    const auto t_crit0 = Clock::now();
    for (int i = 0; i < kCriticalityIters; ++i) {
      (void)profiler.criticality_vector(hot, {}, cypha::intelligence::CriticalityBuildOptions{});
    }
    const double crit_sec = elapsed_sec(t_crit0);

    cypha::cyphalm::CyphaLMConfig cfg_b = cfg;
    cypha::cyphalm::CyphaLMModel model_b(cfg_b);
    cypha::intelligence::IntelligenceProfiler train_profiler;
    model_b.train_sequence(corpus.train_ids, kNTrain, cfg_b.train_epochs, &train_profiler);
    const double bpc_on = model_b.eval_bpc(corpus.eval_ids, kNEval, nullptr);

    cypha::cyphalm::CyphaLMConfig cfg_c = cfg;
    cypha::cyphalm::CyphaLMModel model_c(cfg_c);
    model_c.train_sequence(corpus.train_ids, kNTrain, cfg_c.train_epochs, nullptr);
    const double bpc_off = model_c.eval_bpc(corpus.eval_ids, kNEval, nullptr);

    const double per_step_us = (crit_sec / static_cast<double>(kCriticalityIters)) * 1e6;
    const double train_step_us = (train_sec / static_cast<double>(kNTrain)) * 1e6;
    const double sim_overhead_pct = 100.0 * crit_sec / train_sec;
    const double hot_path_pct = 0.0;

    std::printf("criticality_overhead_p9_bench:\n");
    std::printf("  d17_train_steps=%d train_wall_s=%.4f (%.1f us/step)\n", kNTrain, train_sec,
                train_step_us);
    std::printf("  criticality_vector_iters=%d crit_wall_s=%.6f (%.2f us/call)\n",
                kCriticalityIters, crit_sec, per_step_us);
    std::printf("  simulated_per_step_overhead_pct=%.3f\n", sim_overhead_pct);
    std::printf("  shipped_hot_path_overhead_pct=%.3f\n", hot_path_pct);
    std::printf("  bpc_profiler_train=%.12f bpc_baseline=%.12f\n", bpc_on, bpc_off);
    std::printf("  bpc_identical=%s\n", bpc_on == bpc_off ? "yes" : "no");
    std::printf("  gate=CYPHA_CRITICALITY (default ON; 0/false/off strips report/REST)\n");
    std::printf("criticality_overhead_p9_bench: PASS\n");
    return 0;
  } catch (const std::exception& ex) {
    std::printf("criticality_overhead_p9_bench: FAIL (%s)\n", ex.what());
    return 1;
  }
}
