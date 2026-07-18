// U06 PGM cell + Wy checkpoint round-trip smoke.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
  namespace fs = std::filesystem;
  cypha::cyphalm::CyphaLMConfig cfg;
  cypha::cyphalm::apply_cell_variant("U06", cfg);
  cfg.vocab_size = 64;
  cfg.d_embed = 32;
  cfg.d_state = 32;
  cfg.field_dim = 32;

  cypha::cyphalm::CyphaLMModel model(cfg);
  for (int i = 0; i < 40; ++i) {
    (void)model.train_step(static_cast<std::uint32_t>(i % 17),
                           static_cast<std::uint32_t>((i + 1) % 17));
  }
  const auto* pgm = model.pgm_cell();
  if (pgm == nullptr) {
    std::cerr << "pgm_cell missing\n";
    return 1;
  }
  const std::size_t edges_before = pgm->edge_count();
  const std::size_t occ_before = pgm->occupied_count();

  // Save *before* predict: predict_next mutates SSM/PGM, so live vs reload must
  // start from the same checkpointed state.
  const fs::path base = fs::temp_directory_path() / "cypha_pgm_ckpt_rt";
  cypha::cyphalm::save_cyphalm_model(model, base.string());
  cypha::cyphalm::CyphaLMModel loaded =
      cypha::cyphalm::load_cyphalm_model((base.string() + ".json"));
  if (!loaded.config().use_unified_context || !loaded.config().use_pgm_cell) {
    std::cerr << "unified/pgm flags not restored\n";
    return 2;
  }
  if (loaded.pgm_cell() == nullptr) {
    std::cerr << "loaded pgm_cell missing\n";
    return 3;
  }
  if (loaded.pgm_cell()->edge_count() != edges_before ||
      loaded.pgm_cell()->occupied_count() != occ_before) {
    std::cerr << "pgm memory mismatch edges " << loaded.pgm_cell()->edge_count() << " vs "
              << edges_before << " occ " << loaded.pgm_cell()->occupied_count() << " vs "
              << occ_before << "\n";
    return 4;
  }

  auto before = model.predict_next(3);
  auto after = loaded.predict_next(3);
  if (before.log_probs.size() != after.log_probs.size()) {
    std::cerr << "log_probs size mismatch\n";
    return 5;
  }
  double max_abs = 0.0;
  for (std::size_t i = 0; i < before.log_probs.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(before.log_probs[i] - after.log_probs[i]));
  }
  if (max_abs > 1e-9) {
    std::cerr << "logits drift max_abs=" << max_abs << "\n";
    return 6;
  }
  std::cout << "pgm_checkpoint_roundtrip_smoke OK edges=" << edges_before
            << " occupied=" << occ_before << "\n";
  return 0;
}
