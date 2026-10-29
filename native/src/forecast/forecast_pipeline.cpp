#include "cypha/forecast/forecast_pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

namespace cypha::forecast {

ForecastPipeline::ForecastPipeline(ForecastPipelineConfig cfg)
    : cfg_(cfg), theaters_(cfg.theater_cfg), monitor_() {}

ForecastPipelineResult ForecastPipeline::run(const std::filesystem::path& data_dir) {
  ForecastPipelineResult out;
  const DisputeDataset data = load_dispute_data(data_dir);

  if (data.mid.empty()) {
    out.detail = "no MID data in " + data_dir.string();
    return out;
  }

  NodeEstimator estimator(cfg_.node_cfg);
  out.node_result = estimator.train_on_mid(data.mid, 0.2);

  // Every event id, 256 and above included (FORECAST_EVENT_VOCAB.md; before, ids of 256 or more threw in hp).
  seq_ = make_event_model(vocab_, cfg_.model_cfg);

  const auto tokens = gdelt_to_token_sequence(data.gdelt, vocab_);
  if (tokens.size() >= 4) {
    seq_->train(tokens, cfg_.sequence_epochs);
    seq_->set_learning(false);
    seq_->begin_stream();
    double bits = 0.0;
    for (const auto id : tokens) {
      bits += seq_->observe(id);
    }
    out.sequence_eval_bpc = bits / static_cast<double>(tokens.size());
    seq_->set_learning(true);
    seq_->begin_stream();
  }

  std::mt19937 rng(42);
  const std::vector<std::uint32_t> seed(tokens.begin(),
                                        tokens.begin() + std::min(tokens.size(), std::size_t{8}));
  out.scenario_tree = generate_rollout_tree(*seq_, seed, cfg_.tree_cfg, rng);
  out.paths = extract_interpretable_paths(out.scenario_tree, vocab_);

  for (const auto& rec : data.mid) {
    if (rec.great_power && rec.escalated) {
      std::vector<double> x(6);
      x[0] = static_cast<double>(rec.hostility);
      x[1] = static_cast<double>(rec.prev_hostility);
      x[2] = static_cast<double>(rec.great_power);
      x[3] = static_cast<double>(rec.duration);
      x[4] = rec.gdp_ratio;
      x[5] = static_cast<double>(theater_from_string(rec.theater));
      theaters_.train_step(theater_from_string(rec.theater), x.data(), 6, "escalate", true);
    }
  }
  theaters_.ewc_snapshot_all();

  std::vector<ViewsObservation> held_out;
  const auto views_path = resolve_views_csv_path(data_dir);
  if (!views_path.empty() && std::filesystem::exists(views_path)) {
    const auto train = load_views_csv(views_path, "train");
    held_out = load_views_csv(views_path, "holdout");
    if (held_out.empty()) {
      held_out = load_views_csv(views_path, "");
    }
    if (!train.empty() && !held_out.empty()) {
      out.views_validation = validate_views_holdout(train, held_out);
    } else {
      std::vector<FatalityForecast> forecasts;
      for (const auto& obs : held_out) {
        forecasts.push_back(forecast_from_mean(50.0, obs.country, obs.year, obs.month));
      }
      out.views_validation = validate_against_views(forecasts, held_out);
    }
  }

  if (estimator.model().mem()) {
    for (const auto& ev : data.gdelt) {
      const auto alarm =
          monitor_.ingest(ev, estimator.model().drift_score(), estimator.model().drift_score());
      if (alarm.fired) {
        ++out.drift_alarms;
      }
    }
  }

  return out;
}

}  // namespace cypha::forecast
