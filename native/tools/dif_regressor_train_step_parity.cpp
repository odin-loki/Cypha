// DIFRegressor-shaped online step: native dif_train_step_vector + expert_target_ema vs Python sidecar.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace fs = std::filesystem;

namespace {

constexpr double kTol = 1e-9;
constexpr double kPredTol = 1e-8;

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::vector<double> flatten_f_field(const nlohmann::json& j) {
  std::vector<double> o;
  for (const auto& row : j) {
    for (const auto& v : row) {
      o.push_back(v.get<double>());
    }
  }
  return o;
}

struct ExpertStat {
  std::vector<double> mu;
  double var_ema{0.0};
  int n_updates{0};
};

/// Matches Python ``DIFRegressor.train_step`` expert selection (cold hash vs ``CyphaDIF.infer`` argmax LLR).
std::string pick_dif_regressor_expert(int python_step, int n_existing, int k_target, cypha::CyphaInferModel& infer,
                                      const double* x, int /*d*/) {
  if (n_existing < k_target && python_step <= k_target * 20) {
    return "_e" + std::to_string(python_step % k_target);
  }
  if (n_existing == 0) {
    return "_e0";
  }
  std::vector<double> H;
  cypha::batch_encode(infer, x, 1, H);
  std::vector<double> llr;
  cypha::score_matrix_use_field(infer, H.data(), 1, llr);
  int K = static_cast<int>(infer.labels.size());
  if (K == 0) {
    return "_e0";
  }
  int bi = 0;
  for (int k = 1; k < K; ++k) {
    if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(bi)]) {
      bi = k;
    }
  }
  return infer.labels[static_cast<std::size_t>(bi)];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: dif_regressor_train_step_parity <parity_fixtures/dif_regressor_train_step>\n";
      return 2;
    }
    fs::path dir = fs::path(argv[1]);
    std::ifstream sf(dir / "sidecar.json");
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    std::ifstream jf(dir / "f_field.json");
    if (!jf) {
      throw std::runtime_error("cannot open f_field.json");
    }
    std::stringstream fj;
    fj << jf.rdbuf();
    std::vector<double> fflat = flatten_f_field(nlohmann::json::parse(fj.str()));

    cypha::CNode root_node = cypha::load_cypha_file((dir / "before.cypha").string().c_str());
    const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
    int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
    int d = static_cast<int>(enc.shape[0]);
    if (d != j.at("d_latent").get<int>() || fd != j.at("field_dim").get<int>()) {
      throw std::runtime_error("sidecar d_latent/field_dim mismatch vs before.cypha");
    }
    if (static_cast<int>(fflat.size()) != d * fd) {
      throw std::runtime_error("f_field size mismatch");
    }

    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
    cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root_node, fflat.data(), fd);

    cypha::TrainStepParams tsp;
    tsp.enc_lr = j.value("enc_lr", 0.002);
    tsp.replay_ratio = j.value("replay_ratio", 0.0);
    tsp.replay_cap = j.value("replay_cap", 10000);
    tsp.align_every = j.value("align_every", 500);
    tsp.temp_recalib_every = j.value("temp_recalib_every", 0);

    double world_lr = j.at("world_lr").get<double>();
    double delta_lr = j.at("delta_lr").get<double>();
    double ood_sigma = j.at("ood_sigma").get<double>();
    int n_experts_cap = j.at("n_experts").get<int>();
    double target_lr = j.at("target_lr").get<double>();
    int target_dim = j.at("target_dim").get<int>();
    if (target_dim != 1) {
      throw std::runtime_error("fixture currently targets target_dim==1 only");
    }

    cypha::ReplayBuffer replay(tsp.replay_cap);
    std::vector<double> replay_u01_storage;
    if (j.contains("replay_u01")) {
      replay_u01_storage = j.at("replay_u01").get<std::vector<double>>();
    }
    std::size_t replay_u01_pos = 0;
    std::mt19937 rng{424242};
    int enc_updates = 0;
    int total_steps = 0;
    cypha::TrainStepExtras extras{};
    extras.total_steps = &total_steps;
    extras.ood_sigma = nullptr;
    extras.llr_ema = nullptr;
    if (!replay_u01_storage.empty()) {
      extras.replay_u01 = replay_u01_storage.data();
      extras.replay_u01_len = replay_u01_storage.size();
      extras.replay_u01_pos = &replay_u01_pos;
    }

    std::unordered_map<std::string, ExpertStat> experts;

    for (const auto& st : j.at("steps")) {
      std::vector<double> x;
      for (const auto& v : st.at("x")) {
        x.push_back(v.get<double>());
      }
      if (static_cast<int>(x.size()) != d) {
        throw std::runtime_error("x dim mismatch");
      }
      double y = st.at("y").get<double>();
      double exp_loss = st.at("expected_loss").get<double>();
      std::string exp_expert = st.at("expected_expert").get<std::string>();

      int k_target = std::max(n_experts_cap, 4);
      int python_step = total_steps + 1;
      int n_existing = static_cast<int>(mem.labels.size());
      std::string expert = pick_dif_regressor_expert(python_step, n_existing, k_target, infer, x.data(), d);
      if (expert != exp_expert) {
        std::cerr << "expert mismatch: got " << expert << " expected " << exp_expert << "\n";
        return 1;
      }

      double loss = cypha::dif_train_step_vector(infer, mem, replay, x.data(), d, expert, world_lr, delta_lr,
                                                 world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr,
                                                 &extras);

      if (!near_eq(loss, exp_loss, kTol)) {
        std::cerr << "loss mismatch: got " << loss << " expected " << exp_loss << "\n";
        return 1;
      }

      ExpertStat& es = experts[expert];
      cypha::regression::expert_target_ema_step(es.mu, es.var_ema, es.n_updates, &y, target_dim, target_lr);
    }

    const auto& j_mu = j.at("final_expert_mu");
    const auto& j_var = j.at("final_expert_var");
    const auto& j_n = j.at("final_expert_n");
    for (auto it = j_mu.begin(); it != j_mu.end(); ++it) {
      const std::string key = it.key();
      auto ge = experts.find(key);
      if (ge == experts.end()) {
        std::cerr << "missing expert " << key << "\n";
        return 1;
      }
      const auto& arr = it.value();
      if (ge->second.mu.size() != arr.size()) {
        std::cerr << "mu len mismatch " << key << "\n";
        return 1;
      }
      for (std::size_t i = 0; i < ge->second.mu.size(); ++i) {
        if (!near_eq(ge->second.mu[i], arr[i].get<double>(), kTol)) {
          std::cerr << "mu mismatch " << key << " at " << i << "\n";
          return 1;
        }
      }
      if (!near_eq(ge->second.var_ema, j_var.at(key).get<double>(), kTol)) {
        std::cerr << "var mismatch " << key << "\n";
        return 1;
      }
      if (ge->second.n_updates != j_n.at(key).get<int>()) {
        std::cerr << "n_updates mismatch " << key << "\n";
        return 1;
      }
    }
    if (experts.size() != j_mu.size()) {
      std::cerr << "expert count mismatch\n";
      return 1;
    }

    std::vector<double> qx;
    for (const auto& v : j.at("predict_x")) {
      qx.push_back(v.get<double>());
    }
    if (static_cast<int>(qx.size()) != d) {
      throw std::runtime_error("predict_x dim mismatch");
    }
    std::vector<double> Hq;
    cypha::batch_encode(infer, qx.data(), 1, Hq);
    std::vector<double> llr;
    cypha::score_matrix_use_field(infer, Hq.data(), 1, llr);
    int K = static_cast<int>(infer.labels.size());
    if (K < 1) {
      throw std::runtime_error("no labels after train");
    }
    std::vector<double> z(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
      z[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] / (infer.temperature + 1e-8);
    }
    std::vector<double> probs;
    cypha::softmax_batch_like_python(z.data(), 1, K, 1e-8, probs);

    std::vector<double> mu_k(static_cast<std::size_t>(K), 0.0);
    std::vector<double> var_k(static_cast<std::size_t>(K), 0.0);
    for (int i = 0; i < K; ++i) {
      const std::string& lb = infer.labels[static_cast<std::size_t>(i)];
      auto it = experts.find(lb);
      if (it != experts.end() && !it->second.mu.empty()) {
        mu_k[static_cast<std::size_t>(i)] = it->second.mu[0];
        var_k[static_cast<std::size_t>(i)] = it->second.var_ema;
      }
    }
    double yhat = 0.0;
    double unc = 0.0;
    cypha::regression::predict_mixture_scalar(probs.data(), mu_k.data(), var_k.data(), static_cast<std::size_t>(K),
                                              yhat, unc);

    double exp_y = j.at("expected_y_pred").get<double>();
    double exp_u = j.at("expected_uncertainty").get<double>();
    if (!near_eq(yhat, exp_y, kPredTol)) {
      std::cerr << "predict y mismatch: got " << yhat << " expected " << exp_y << "\n";
      return 1;
    }
    if (!near_eq(unc, exp_u, kPredTol)) {
      std::cerr << "predict unc mismatch: got " << unc << " expected " << exp_u << "\n";
      return 1;
    }

    if (!replay_u01_storage.empty() && replay_u01_pos != replay_u01_storage.size()) {
      std::cerr << "replay_u01 consumption mismatch: pos=" << replay_u01_pos << " len=" << replay_u01_storage.size()
                << "\n";
      return 1;
    }

    std::cout << "dif_regressor_train_step parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dif_regressor_train_step_parity: " << e.what() << "\n";
    return 1;
  }
}
