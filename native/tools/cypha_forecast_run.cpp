/// End-to-end forecasting framework CLI (Phases 1-9).
#include "cypha/forecast/forecast_pipeline.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "bench/data/forecast";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    }
  }

  const char* candidates[] = {
      "bench/data/forecast",
      "../bench/data/forecast",
      "../../bench/data/forecast",
  };
  if (!std::filesystem::exists(data_dir)) {
    for (const char* c : candidates) {
      if (std::filesystem::exists(c)) {
        data_dir = c;
        break;
      }
    }
  }

  cypha::forecast::ForecastPipeline pipe;
  const auto result = pipe.run(data_dir);

  if (!result.detail.empty()) {
    std::cerr << "cypha_forecast_run: " << result.detail << "\n";
    return 2;
  }

  std::cout << "data_dir=" << data_dir << "\n";
  std::cout << "node_eval_acc=" << result.node_result.eval_accuracy << "\n";
  std::cout << "sequence_bits_per_event=" << result.sequence_eval_bpc << "\n";
  std::cout << "tree_nodes=" << result.scenario_tree.nodes.size() << "\n";
  std::cout << "views_crps=" << result.views_validation.mean_crps << "\n";
  std::cout << "cypha_forecast_run OK\n";
  return 0;
}
