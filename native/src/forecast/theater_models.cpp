#include "cypha/forecast/theater_models.hpp"

#include "cypha/classify/create_model.hpp"

#include <algorithm>
#include <filesystem>

namespace cypha::forecast {

TheaterModelBank::TheaterModelBank(TheaterModelConfig cfg) : cfg_(cfg) {}

Cypha* TheaterModelBank::model_for(Theater t) {
  const int key = static_cast<int>(t);
  auto it = models_.find(key);
  if (it == models_.end()) {
    auto cy = std::make_unique<Cypha>();
    FreshModelParams fp;
    fp.input_dim = 64;
    fp.field_dim = 64;
    const auto tmp =
        (std::filesystem::temp_directory_path() / ("cypha_theater_" + std::string(theater_name(t)) + ".cypha"))
            .string();
    create_and_save_fresh_model(tmp.c_str(), fp);
    cy->load(tmp);
    cy->ewc_lambda() = cfg_.ewc_lambda;
    cy->replay_owned() = std::make_unique<ReplayBuffer>(cfg_.replay_cap);
    auto* ptr = cy.get();
    models_[key] = std::move(cy);
    ewc_[key] = std::make_unique<EwcRegularizer>();
    return ptr;
  }
  return it->second.get();
}

const Cypha* TheaterModelBank::model_for(Theater t) const {
  const int key = static_cast<int>(t);
  auto it = models_.find(key);
  return it == models_.end() ? nullptr : it->second.get();
}

void TheaterModelBank::train_step(Theater t, const double* x, int n, const std::string& label,
                                  bool rare_event) {
  Cypha* m = model_for(t);
  if (!m) return;
  if (rare_event && x != nullptr && n > 0) {
    rare_replay_[static_cast<int>(t)].emplace_back(x, x + n);
    const auto& buf = rare_replay_[static_cast<int>(t)];
    const int replay_n = std::min(3, static_cast<int>(buf.size()));
    cypha::UpdateOpts replay_opts;
    for (int i = 0; i < replay_n; ++i) {
      const auto& row = buf[static_cast<std::size_t>(buf.size() - 1 - static_cast<std::size_t>(i))];
      m->update(row.data(), static_cast<int>(row.size()), &label, nullptr, replay_opts);
    }
  }
  cypha::UpdateOpts o;
  o.ewc_snapshot = rare_event;
  m->update(x, n, &label, nullptr, o);
}

void TheaterModelBank::ewc_snapshot_all() {
  for (auto& [key, m] : models_) {
    if (!ewc_[key]) {
      ewc_[key] = std::make_unique<EwcRegularizer>();
    }
    ewc_[key]->snapshot(*m->mem(), *m->infer());
  }
}

}  // namespace cypha::forecast
