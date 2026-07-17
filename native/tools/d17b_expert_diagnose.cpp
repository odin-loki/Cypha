// d17b_expert_diagnose — diagnostic for D17B's "1 active expert" / low mean_alpha reporting.
// See docs/RESEARCH_STATUS.md Priority 3 roadmap Step 5 and
// docs/reports/D17B_EXPERT_REPORTING_2026-07-12.md.
//
// Trains the exact D17 hybrid config (apply_bench_profile("d17") + apply_bench_mode(Hybrid))
// on a synthetic corpus (no WikiText/Gutenberg dependency needed) and prints
// CyphaDIF expert_count() + per-expert alpha + GRIA alpha stats at several checkpoints
// spanning the training run, so growth (or lack of it) is directly observable.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

using cypha::cyphalm::apply_bench_mode;
using cypha::cyphalm::apply_bench_profile;
using cypha::cyphalm::BenchMode;
using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;

namespace {

void print_checkpoint(const char* label, const CyphaLMModel& model) {
  const auto profile = model.compression_profile();
  std::printf("[%s] n_experts=%d cfg_n_experts=%d max_experts=%d mean_alpha=%.6f mean_expert_alpha=%.6f edge_frac=%.6f\n",
              label, profile.value("n_experts", 0), profile.value("cfg_n_experts", 0),
              profile.value("max_experts", 0), profile.value("mean_alpha", 0.0),
              profile.value("mean_expert_alpha", 0.0), profile.value("fraction_near_edge_of_chaos", 0.0));
  const auto& expert_alphas = profile.at("expert_alpha_spectrum");
  std::printf("         expert_alpha_spectrum = [");
  for (const auto& v : expert_alphas) {
    std::printf("%.4f ", v.get<double>());
  }
  std::printf("]\n");
  const auto& gria_spec = profile.at("gria_alpha_spectrum");
  std::printf("         gria_alpha_spectrum = %s\n", gria_spec.dump().c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const int n_train = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int chunk = argc > 2 ? std::atoi(argv[2]) : 2000;
  const int warm_start_n_experts = argc > 3 ? std::atoi(argv[3]) : 0;
  const bool use_real_corpus = argc > 4 && std::string(argv[4]) == "real";

  CyphaLMConfig cfg;
  apply_bench_profile("d17", cfg);
  apply_bench_mode(BenchMode::Hybrid, cfg);
  if (cfg.vocab_size < 256) cfg.vocab_size = 256;
  if (warm_start_n_experts > 0) cfg.n_experts = warm_start_n_experts;

  std::printf(
      "config: max_experts=%d n_experts=%d field_dim=%d vocab_size=%d context_mode=%s "
      "alpha_learnable=%d corpus=%s\n",
      cfg.max_experts, cfg.n_experts, cfg.field_dim, cfg.vocab_size,
      cypha::cyphalm::context_mode_name(cfg.context_mode).c_str(), cfg.alpha_learnable ? 1 : 0,
      use_real_corpus ? "wikitext2/gutenberg" : "synthetic");

  std::vector<int> ids;
  if (use_real_corpus) {
    auto corpus = cypha::cyphalm::load_bench_corpus("d17", 2'000'000, cfg.vocab_size, cfg.bpe_merges_path,
                                                     cfg.bpe_vocab_path);
    cfg.vocab_size = corpus.vocab_size;
    ids = std::move(corpus.train_ids);
    std::printf("corpus source=%s n_tokens=%d vocab_size=%d\n", corpus.source.c_str(),
                static_cast<int>(ids.size()), corpus.vocab_size);
    if (static_cast<int>(ids.size()) < n_train + 1) {
      std::fprintf(stderr, "not enough real corpus tokens (%d) for n_train=%d\n",
                   static_cast<int>(ids.size()), n_train);
      return 1;
    }
  } else {
    ids = cypha::cyphalm::synthetic_corpus(n_train + 64, cfg.vocab_size, cfg.seed);
  }

  CyphaLMModel model(cfg);
  model.reset_context();

  int done = 0;
  int max_active_experts_seen = 0;
  long long steps_with_multi_active = 0;
  while (done < n_train) {
    const int step = std::min(chunk, n_train - done);
    for (int i = 0; i < step; ++i) {
      const int idx = done + i;
      const auto m = model.train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(idx)]),
                                       static_cast<std::uint32_t>(ids[static_cast<std::size_t>(idx + 1)]));
      max_active_experts_seen = std::max(max_active_experts_seen, m.active_experts);
      if (m.active_experts > 1) ++steps_with_multi_active;
    }
    done += step;
    const std::string label = "step=" + std::to_string(done);
    print_checkpoint(label.c_str(), model);
    std::printf("         train_step.active_experts: max_seen=%d steps_with_multi_active=%lld/%d\n",
                max_active_experts_seen, steps_with_multi_active, done);
  }

  std::printf("d17b_expert_diagnose: done, n_train=%d\n", n_train);
  return 0;
}
