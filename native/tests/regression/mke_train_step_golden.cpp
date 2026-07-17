// CTest: MKERegressor.train_step vs fixtures/mke_train_step/ (single) or mke_train_extended/ (multi).
// RFF φ from sidecar x + enc; score_matrix_use_field(φ); expert RLS; dif_train_step_vector(pred).
// Extended: refresh_world_log_norm_from_v each step; optional replay_warmup + TrainStepExtras.replay_u01.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace fs = std::filesystem;

namespace {

constexpr double kPriorityLossEps = 1e-3;

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

std::unordered_map<std::string, std::vector<double>> load_vec_map(const nlohmann::json& j) {
  std::unordered_map<std::string, std::vector<double>> m;
  for (const auto& [k, v] : j.items()) {
    m[k] = v.get<std::vector<double>>();
  }
  return m;
}

std::unordered_map<std::string, std::vector<double>> load_p_map(const nlohmann::json& j, int d) {
  std::unordered_map<std::string, std::vector<double>> m;
  const std::size_t expect = static_cast<std::size_t>(d * d);
  for (const auto& [k, arr] : j.items()) {
    auto row = arr.get<std::vector<double>>();
    if (row.size() != expect) {
      throw std::runtime_error("P size mismatch for " + k);
    }
    m[k] = std::move(row);
  }
  return m;
}

struct LoadedInfer {
  int d_latent{0};
  cypha::CyphaInferModel infer{};
  cypha::CyphaDifMemoryState mem{};
};

LoadedInfer load_infer_from_fixture_dir(const fs::path& dir) {
  fs::path cypha_path = dir / "before.cypha";
  fs::path ff_path = dir / "f_field.json";
  cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
  int d_latent = static_cast<int>(enc.shape[0]);

  std::ifstream jf(ff_path);
  if (!jf) {
    throw std::runtime_error("cannot open f_field.json");
  }
  std::stringstream fj;
  fj << jf.rdbuf();
  std::vector<double> fflat = flatten_f_field(nlohmann::json::parse(fj.str()));
  if (static_cast<int>(fflat.size()) != d_latent * fd) {
    throw std::runtime_error("f_field size mismatch");
  }

  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root_node, fflat.data(), fd);
  (void)fd;
  return LoadedInfer{d_latent, std::move(infer), std::move(mem)};
}

int run_legacy_single(const nlohmann::json& j, const fs::path& dir) {
  LoadedInfer L = load_infer_from_fixture_dir(dir);
  int d_latent = L.d_latent;
  cypha::CyphaInferModel& infer = L.infer;
  cypha::CyphaDifMemoryState& mem = L.mem;

  const int d_in = j.at("d_in").get<int>();
  const int d_rff = j.at("D_rff").get<int>();
  if (d_rff != d_latent) {
    std::cerr << "D_rff mismatch enc_W\n";
    return 1;
  }

  std::vector<double> x;
  for (const auto& v : j.at("x")) {
    x.push_back(v.get<double>());
  }
  if (static_cast<int>(x.size()) != d_in) {
    throw std::runtime_error("x dim mismatch");
  }

  std::vector<double> W_rff = j.at("rff_W_rowmajor").get<std::vector<double>>();
  std::vector<double> b_rff = j.at("rff_b").get<std::vector<double>>();
  if (static_cast<int>(W_rff.size()) != d_rff * d_in || static_cast<int>(b_rff.size()) != d_rff) {
    std::cerr << "rff W/b size mismatch\n";
    return 1;
  }

  std::vector<double> phi(static_cast<std::size_t>(d_rff));
  cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, d_in, W_rff.data(), b_rff.data(), d_rff, phi.data());

  const auto& exp_phi = j.at("expected_phi");
  constexpr double kPhiTol = 1e-9;
  for (int t = 0; t < d_rff; ++t) {
    double ev = exp_phi[t].get<double>();
    if (!near_eq(phi[static_cast<std::size_t>(t)], ev, kPhiTol)) {
      std::cerr << "phi mismatch at " << t << "\n";
      return 1;
    }
  }

  const int K = static_cast<int>(infer.labels.size());
  const auto& rl = j.at("routing_labs");
  if (static_cast<int>(rl.size()) != K) {
    std::cerr << "routing_labs length mismatch\n";
    return 1;
  }
  for (int k = 0; k < K; ++k) {
    if (infer.labels[static_cast<std::size_t>(k)] != rl[k].get<std::string>()) {
      std::cerr << "label order mismatch at " << k << "\n";
      return 1;
    }
  }

  double temperature = j.at("temperature").get<double>();
  constexpr double kSoftmaxEps = 1e-8;
  auto w_work = load_vec_map(j.at("w_before"));
  auto p_work = load_p_map(j.at("P_before"), d_rff);
  double y = j.at("y").get<double>();
  const auto& ghj = j.at("gh_scales");
  double ff = j.at("forgetting_factor").get<double>();
  std::vector<double> gh_k(static_cast<std::size_t>(K));
  for (int i = 0; i < K; ++i) {
    gh_k[static_cast<std::size_t>(i)] = ghj[i].get<double>();
  }

  cypha::TrainStepParams tsp;
  tsp.enc_lr = j.value("enc_lr", 0.0);
  tsp.replay_ratio = j.value("replay_ratio", 0.0);
  tsp.replay_cap = j.value("replay_cap", 10000);
  tsp.align_every = j.value("align_every", 500);
  tsp.temp_recalib_every = j.value("temp_recalib_every", 0);

  double world_lr = j.value("world_lr", 0.008);
  double delta_lr = j.value("delta_lr", 0.05);
  double ood_sigma = j.value("ood_sigma", 15.0);

  cypha::ReplayBuffer replay(tsp.replay_cap);
  unsigned rseed = static_cast<unsigned>(j.value("rng_seed", 42));
  std::mt19937 rng{rseed};
  int enc_updates = j.value("enc_update_count_start", 0);
  int total_steps = j.at("total_steps_start").get<int>();
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  extras.ood_sigma = nullptr;
  extras.llr_ema = nullptr;

  std::string router_label = j.at("router_train_label").get<std::string>();
  cypha::regression::MkeScalarTrainStepOutputs step_out{};
  constexpr double kPiFloor = 0.02;
  cypha::regression::mke_scalar_train_step_from_phi(infer, mem, replay, phi.data(), d_rff, y, w_work, p_work,
                                                    gh_k.data(), temperature, ff, kPiFloor, tsp, world_lr, delta_lr,
                                                    ood_sigma, rng, enc_updates, &extras, &router_label, kSoftmaxEps,
                                                    &step_out);

  double exp_err_sq = j.at("expected_err_sq").get<double>();
  constexpr double kErrTol = 1e-8;
  if (!near_eq(step_out.err_sq, exp_err_sq, kErrTol)) {
    std::cerr << "err_sq mismatch: got " << step_out.err_sq << " exp " << exp_err_sq << "\n";
    return 1;
  }

  const auto& w_after_j = j.at("w_after");
  const auto& p_after_j = j.at("P_after");
  constexpr double kWTol = 1e-7;
  constexpr double kPTol = 1e-6;
  for (const auto& [lbl, exp_w] : w_after_j.items()) {
    auto it = w_work.find(lbl);
    if (it == w_work.end()) {
      std::cerr << "w_after key missing in work: " << lbl << "\n";
      return 1;
    }
    auto ew = exp_w.get<std::vector<double>>();
    if (ew.size() != it->second.size()) {
      std::cerr << "w length mismatch " << lbl << "\n";
      return 1;
    }
    for (std::size_t t = 0; t < ew.size(); ++t) {
      if (!near_eq(it->second[t], ew[t], kWTol)) {
        std::cerr << "w mismatch " << lbl << " at " << t << "\n";
        return 1;
      }
    }
  }
  for (const auto& [lbl, exp_p] : p_after_j.items()) {
    auto it = p_work.find(lbl);
    if (it == p_work.end()) {
      std::cerr << "P_after key missing in work: " << lbl << "\n";
      return 1;
    }
    auto ep = exp_p.get<std::vector<double>>();
    if (ep.size() != it->second.size()) {
      std::cerr << "P length mismatch " << lbl << "\n";
      return 1;
    }
    for (std::size_t t = 0; t < ep.size(); ++t) {
      if (!near_eq(it->second[t], ep[t], kPTol)) {
        std::cerr << "P mismatch " << lbl << " at " << t << "\n";
        return 1;
      }
    }
  }

  double exp_loss = j.at("expected_router_loss").get<double>();
  constexpr double kLossTol = 1e-8;
  if (!near_eq(step_out.router_loss, exp_loss, kLossTol)) {
    std::cerr << "router loss mismatch: got " << step_out.router_loss << " exp " << exp_loss << "\n";
    return 1;
  }

  std::cout << "mke_train_step parity OK\n";
  return 0;
}

int run_extended_mke(const nlohmann::json& j, const fs::path& dir) {
  LoadedInfer L = load_infer_from_fixture_dir(dir);
  int d_latent = L.d_latent;
  cypha::CyphaInferModel& infer = L.infer;
  cypha::CyphaDifMemoryState& mem = L.mem;

  const int d_in = j.at("d_in").get<int>();
  const int d_rff = j.at("D_rff").get<int>();
  if (d_rff != d_latent) {
    std::cerr << "D_rff mismatch enc_W\n";
    return 1;
  }

  std::vector<double> W_rff = j.at("rff_W_rowmajor").get<std::vector<double>>();
  std::vector<double> b_rff = j.at("rff_b").get<std::vector<double>>();
  if (static_cast<int>(W_rff.size()) != d_rff * d_in || static_cast<int>(b_rff.size()) != d_rff) {
    std::cerr << "rff W/b size mismatch\n";
    return 1;
  }

  cypha::TrainStepParams tsp;
  tsp.enc_lr = j.value("enc_lr", 0.002);
  tsp.replay_ratio = j.value("replay_ratio", 0.30);
  tsp.replay_cap = j.value("replay_cap", 10000);
  tsp.align_every = j.value("align_every", 500);
  tsp.temp_recalib_every = j.value("temp_recalib_every", 0);

  double world_lr = j.value("world_lr", 0.008);
  double delta_lr = j.value("delta_lr", 0.05);
  double ood_sigma = j.value("ood_sigma", 15.0);

  cypha::ReplayBuffer replay(tsp.replay_cap);
  for (const auto& e : j.at("replay_warmup")) {
    std::vector<double> h = e.at("h").get<std::vector<double>>();
    std::vector<double> f = e.at("f").get<std::vector<double>>();
    std::string label = e.at("label").get<std::string>();
    double loss_v = e.at("loss_v").get<double>();
    double loss_arg = loss_v - kPriorityLossEps;
    if (loss_arg < 0.0) {
      loss_arg = 0.0;
    }
    if (static_cast<int>(h.size()) != d_latent || static_cast<int>(f.size()) != d_latent) {
      throw std::runtime_error("replay_warmup h/f dim mismatch");
    }
    replay.push(h.data(), f.data(), d_latent, label, loss_arg);
  }

  std::vector<double> replay_u01_storage;
  if (j.contains("replay_u01") && j.at("replay_u01").is_array() && !j.at("replay_u01").empty()) {
    replay_u01_storage = j.at("replay_u01").get<std::vector<double>>();
  }
  std::size_t replay_u01_pos = 0;
  unsigned rseed = static_cast<unsigned>(j.value("rng_seed", 42));
  std::mt19937 rng{rseed};
  int enc_updates = j.at("enc_update_count_start").get<int>();
  int total_steps = j.at("total_steps_start").get<int>();
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  extras.ood_sigma = nullptr;
  extras.llr_ema = nullptr;
  if (!replay_u01_storage.empty()) {
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  const auto& steps = j.at("steps");
  double temperature = j.at("temperature").get<double>();
  double ff = j.at("forgetting_factor").get<double>();
  constexpr double kPhiTol = 1e-9;
  constexpr double kSoftmaxEps = 1e-8;
  constexpr double kErrTol = 1e-8;
  constexpr double kWTol = 1e-7;
  constexpr double kPTol = 1e-6;
  constexpr double kLossTol = 1e-8;
  constexpr double kEncTol = 1e-7;

  for (std::size_t si = 0; si < steps.size(); ++si) {
    const auto& st = steps[si];
    std::vector<double> x;
    for (const auto& v : st.at("x")) {
      x.push_back(v.get<double>());
    }
    if (static_cast<int>(x.size()) != d_in) {
      throw std::runtime_error("step x dim mismatch");
    }

    std::vector<double> phi(static_cast<std::size_t>(d_rff));
    cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, d_in, W_rff.data(), b_rff.data(), d_rff, phi.data());

    const auto& exp_phi = st.at("expected_phi");
    for (int t = 0; t < d_rff; ++t) {
      double ev = exp_phi[t].get<double>();
      if (!near_eq(phi[static_cast<std::size_t>(t)], ev, kPhiTol)) {
        std::cerr << "step " << si << " phi mismatch at " << t << "\n";
        return 1;
      }
    }

    const int K = static_cast<int>(infer.labels.size());
    const auto& rl = st.at("routing_labs");
    if (static_cast<int>(rl.size()) != K) {
      std::cerr << "step " << si << " routing_labs length mismatch\n";
      return 1;
    }
    for (int k = 0; k < K; ++k) {
      if (infer.labels[static_cast<std::size_t>(k)] != rl[k].get<std::string>()) {
        std::cerr << "step " << si << " label order mismatch at " << k << "\n";
        return 1;
      }
    }

    auto w_work = load_vec_map(st.at("w_before"));
    auto p_work = load_p_map(st.at("P_before"), d_rff);
    double y = st.at("y").get<double>();
    const auto& ghj = st.at("gh_scales");
    std::vector<double> gh_k(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
      gh_k[static_cast<std::size_t>(i)] = ghj[i].get<double>();
    }

    std::string router_label = st.at("router_train_label").get<std::string>();
    cypha::regression::MkeScalarTrainStepOutputs step_out{};
    constexpr double kPiFloor = 0.02;
    cypha::regression::mke_scalar_train_step_from_phi(infer, mem, replay, phi.data(), d_rff, y, w_work, p_work,
                                                      gh_k.data(), temperature, ff, kPiFloor, tsp, world_lr,
                                                      delta_lr, ood_sigma, rng, enc_updates, &extras, &router_label,
                                                      kSoftmaxEps, &step_out);

    double exp_err_sq = st.at("expected_err_sq").get<double>();
    if (!near_eq(step_out.err_sq, exp_err_sq, kErrTol)) {
      std::cerr << "step " << si << " err_sq mismatch: got " << step_out.err_sq << " exp " << exp_err_sq << "\n";
      return 1;
    }

    const auto& w_after_j = st.at("w_after");
    const auto& p_after_j = st.at("P_after");
    for (const auto& [lbl, exp_w] : w_after_j.items()) {
      auto it = w_work.find(lbl);
      if (it == w_work.end()) {
        std::cerr << "step " << si << " w_after key missing in work: " << lbl << "\n";
        return 1;
      }
      auto ew = exp_w.get<std::vector<double>>();
      if (ew.size() != it->second.size()) {
        std::cerr << "step " << si << " w length mismatch " << lbl << "\n";
        return 1;
      }
      for (std::size_t t = 0; t < ew.size(); ++t) {
        if (!near_eq(it->second[t], ew[t], kWTol)) {
          std::cerr << "step " << si << " w mismatch " << lbl << " at " << t << "\n";
          return 1;
        }
      }
    }
    for (const auto& [lbl, exp_p] : p_after_j.items()) {
      auto it = p_work.find(lbl);
      if (it == p_work.end()) {
        std::cerr << "step " << si << " P_after key missing in work: " << lbl << "\n";
        return 1;
      }
      auto ep = exp_p.get<std::vector<double>>();
      if (ep.size() != it->second.size()) {
        std::cerr << "step " << si << " P length mismatch " << lbl << "\n";
        return 1;
      }
      for (std::size_t t = 0; t < ep.size(); ++t) {
        if (!near_eq(it->second[t], ep[t], kPTol)) {
          std::cerr << "step " << si << " P mismatch " << lbl << " at " << t << "\n";
          return 1;
        }
      }
    }

    double exp_loss = st.at("expected_router_loss").get<double>();
    if (!near_eq(step_out.router_loss, exp_loss, kLossTol)) {
      std::cerr << "step " << si << " router loss mismatch: got " << step_out.router_loss << " exp " << exp_loss
                << "\n";
      return 1;
    }

    const auto& exp_enc = st.at("enc_w_rowmajor").get<std::vector<double>>();
    const std::size_t enc_sz = static_cast<std::size_t>(d_latent * d_latent);
    if (exp_enc.size() != enc_sz || infer.enc_w.size() != enc_sz) {
      std::cerr << "step " << si << " enc_w size mismatch\n";
      return 1;
    }
    for (std::size_t t = 0; t < enc_sz; ++t) {
      if (!near_eq(infer.enc_w[t], exp_enc[t], kEncTol)) {
        std::cerr << "step " << si << " enc_w mismatch at " << t << "\n";
        return 1;
      }
    }
  }

  if (!replay_u01_storage.empty() && replay_u01_pos != replay_u01_storage.size()) {
    std::cerr << "replay_u01 consumption mismatch: pos=" << replay_u01_pos << " len=" << replay_u01_storage.size()
              << "\n";
    return 1;
  }

  std::cout << "mke_train extended parity OK\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: mke_train_step_golden <fixtures/mke_train_step|mke_train_extended_dir>\n";
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

    if (j.contains("steps") && j.at("steps").is_array() && !j.at("steps").empty()) {
      return run_extended_mke(j, dir);
    }
    return run_legacy_single(j, dir);
  } catch (const std::exception& e) {
    std::cerr << "mke_train_step_golden: " << e.what() << "\n";
    return 1;
  }
}
