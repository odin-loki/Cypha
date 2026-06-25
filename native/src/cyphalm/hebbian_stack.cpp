#include "cypha/cyphalm/hebbian_stack.hpp"

namespace cypha::cyphalm {

void HebbianStack::configure(const HebbianStackConfig& config_in) {
  cfg = config_in;
  ssm.resize(cfg.d_state, cfg.n_layers);
  if (cfg.use_hebb_graph) {
    HebbianGraphConfig gc = cfg.graph;
    if (gc.n <= 0) {
      gc.n = 2 * cfg.d_state;
    }
    graph = std::make_unique<HebbianGraph>(gc);
  } else {
    graph.reset();
  }
}

void HebbianStack::on_ssm_layer_context(std::vector<double>& ctx, int layer, const double* fast_state,
                                        const double* slow_state) {
  if (graph && !ctx.empty()) {
    const std::vector<double> diffused = graph->diffuse(ctx);
    if (diffused.size() == ctx.size()) {
      ctx = diffused;
    }
    if (static_cast<int>(ctx.size()) >= graph->config().n) {
      graph->update(ctx.data());
    }
  }
  if (cfg.use_sparse_hebbian) {
    sparse_hebbian_update(ssm, fast_state, slow_state, cfg.ssm_hebb_lr, layer);
  }
}

void HebbianStack::encoder_train_step(const double* f, const double* h, const std::string& true_label,
                                      const std::string& pred_label, double lr, double weight) {
  encoder.competitor_label = pred_label;
  encoder.update(f, h, true_label, lr, weight);
}

}  // namespace cypha::cyphalm
