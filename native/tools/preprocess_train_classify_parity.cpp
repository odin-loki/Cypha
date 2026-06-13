// Raw feature rows -> native preprocessor.json -> dif_train_classify_sequence -> batch_llr_from_x.
// Parity vs Python (Trainer: fit preprocessor on X_raw, transform, then CyphaDIF.train_step).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/csv_ingest.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace fs = std::filesystem;

namespace {

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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: preprocess_train_classify_parity <fixture_dir>\n"
                   "  e.g. fixtures/studio_trainer_preprocess_classify_hotpath\n"
                   "       fixtures/csv_preprocess_classify_hotpath (CSV ingest + preprocess train)\n"
                   "       fixtures/studio_trainer_preprocess_gh_classify_hotpath (sidecar use_gh)\n";
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

    cypha::PreprocessorState pre =
        cypha::PreprocessorState::from_json_file((dir / "preprocessor.json").string().c_str());

    fs::path cypha_path = dir / "before.cypha";
    fs::path ff_path = dir / "f_field.json";
    cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
    const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
    int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
    int d = static_cast<int>(enc.shape[0]);

    const int d_in = j.at("d_in").get<int>();
    const int d_raw = j.at("d_raw").get<int>();
    const int n = j.at("n").get<int>();
    const int K = j.at("K").get<int>();
    if (d_in != d) {
      std::cerr << "d_in mismatch enc_W\n";
      return 1;
    }

    std::ifstream jf(ff_path);
    if (!jf) {
      throw std::runtime_error("cannot open f_field.json");
    }
    std::stringstream fj;
    fj << jf.rdbuf();
    std::vector<double> fflat = flatten_f_field(nlohmann::json::parse(fj.str()));
    if (static_cast<int>(fflat.size()) != d * fd) {
      throw std::runtime_error("f_field size mismatch");
    }

    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
    cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root_node, fflat.data(), fd);

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
    std::vector<double> replay_u01_storage;
    if (j.contains("replay_u01")) {
      replay_u01_storage = j.at("replay_u01").get<std::vector<double>>();
    }
    std::size_t replay_u01_pos = 0;
    unsigned rseed = static_cast<unsigned>(j.value("rng_seed", 7755));
    std::mt19937 rng{rseed};
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

    const auto* exp_losses = j.contains("expected_step_losses") ? &j.at("expected_step_losses") : nullptr;
    const int n_steps_side = j.at("n_steps").get<int>();

    std::vector<std::vector<double>> xs;
    std::vector<std::string> labels;

    if (j.contains("csv") && !j["csv"].is_null()) {
      const std::string csv_name = j.at("csv").get<std::string>();
      const auto& cs = j.at("csv_spec");
      cypha::CsvDenseSpec spec;
      spec.has_header = cs.value("has_header", true);
      std::string delim = cs.value("delimiter", ",");
      if (delim.size() != 1) {
        throw std::runtime_error("csv_spec.delimiter must be a single character");
      }
      spec.delimiter = delim[0];
      spec.target_col_name = cs.value("target_col_name", std::string{});
      if (spec.target_col_name.empty()) {
        spec.target_col_index = cs.at("target_col_index").get<int>();
      } else {
        spec.target_col_index = -1;
      }
      if (cs.contains("feature_col_names") && cs["feature_col_names"].is_array() && !cs["feature_col_names"].empty()) {
        for (const auto& v : cs.at("feature_col_names")) {
          spec.feature_col_names.push_back(v.get<std::string>());
        }
      } else if (cs.contains("feature_col_indices") && !cs["feature_col_indices"].is_null()) {
        for (const auto& v : cs.at("feature_col_indices")) {
          spec.feature_col_indices.push_back(v.get<int>());
        }
      }
      std::string task = cs.value("task", "classification");
      spec.regression = (task == "regression");
      if (spec.regression) {
        throw std::runtime_error("csv_preprocess_classify_hotpath: classification task required");
      }

      cypha::CsvDenseResult csv = cypha::load_csv_dense(dir / csv_name, spec);
      if (csv.n_features != d_raw) {
        throw std::runtime_error("CSV n_features mismatch d_raw");
      }
      if (csv.n_rows != n_steps_side) {
        throw std::runtime_error("CSV n_rows mismatch n_steps");
      }
      if (exp_losses != nullptr &&
          exp_losses->size() != static_cast<std::size_t>(csv.n_rows)) {
        throw std::runtime_error("expected_step_losses length mismatch CSV rows");
      }

      xs.reserve(static_cast<std::size_t>(csv.n_rows));
      labels.reserve(static_cast<std::size_t>(csv.n_rows));
      for (int r = 0; r < csv.n_rows; ++r) {
        std::vector<double> xraw(static_cast<std::size_t>(d_raw));
        const std::size_t base = static_cast<std::size_t>(r * d_raw);
        for (int c = 0; c < d_raw; ++c) {
          xraw[static_cast<std::size_t>(c)] = csv.x_rowmajor[base + static_cast<std::size_t>(c)];
        }
        std::vector<double> xpp = pre.transform_one(xraw);
        if (static_cast<int>(xpp.size()) != d) {
          throw std::runtime_error("preprocessor output dim mismatch vs enc_W.rows");
        }
        xs.push_back(std::move(xpp));
        labels.push_back(csv.y_class[static_cast<std::size_t>(r)]);
      }
    } else {
      const auto& steps = j.at("steps");
      if (exp_losses != nullptr && exp_losses->size() != steps.size()) {
        throw std::runtime_error("expected_step_losses length mismatch steps");
      }
      xs.reserve(steps.size());
      labels.reserve(steps.size());
      for (const auto& step : steps) {
        std::vector<double> xraw;
        for (const auto& v : step.at("x_raw")) {
          xraw.push_back(v.get<double>());
        }
        if (static_cast<int>(xraw.size()) != d_raw) {
          throw std::runtime_error("x_raw dim mismatch");
        }
        std::vector<double> xpp = pre.transform_one(xraw);
        if (static_cast<int>(xpp.size()) != d) {
          throw std::runtime_error("preprocessor output dim mismatch vs enc_W.rows");
        }
        xs.push_back(std::move(xpp));
        labels.push_back(step.at("label").get<std::string>());
      }
    }

    const bool use_gh = j.value("use_gh", false);
    std::vector<double> got_losses;
    if (use_gh) {
      std::vector<double> gh_inv = j.at("gh_inv_v_clean").get<std::vector<double>>();
      if (static_cast<int>(gh_inv.size()) != d) {
        throw std::runtime_error("gh_inv_v_clean dim mismatch");
      }
      double gh_r_base = j.at("gh_r_base").get<double>();
      double chi = j.value("chi_start", 1.0);
      double psi = j.value("psi_start", 1.0);
      double nig_alpha = j.value("nig_alpha", 0.98);
      got_losses = cypha::dif_gh_train_classify_sequence(infer, mem, replay, xs, labels, gh_inv, gh_r_base, chi, psi,
                                                         nig_alpha, world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates,
                                                         &extras);
      if (j.contains("expected_chi_end")) {
        double ec = j.at("expected_chi_end").get<double>();
        constexpr double kGhTol = 1e-9;
        if (!near_eq(chi, ec, kGhTol)) {
          std::cerr << "chi_end mismatch: got " << chi << " expected " << ec << "\n";
          return 1;
        }
      }
      if (j.contains("expected_psi_end")) {
        double ep = j.at("expected_psi_end").get<double>();
        constexpr double kGhTol = 1e-9;
        if (!near_eq(psi, ep, kGhTol)) {
          std::cerr << "psi_end mismatch: got " << psi << " expected " << ep << "\n";
          return 1;
        }
      }
    } else {
      got_losses = cypha::dif_train_classify_sequence(
          infer, mem, replay, xs, labels, world_lr, delta_lr, world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates,
          &extras);
    }
    if (exp_losses != nullptr) {
      for (std::size_t si = 0; si < got_losses.size(); ++si) {
        double exp_l = (*exp_losses)[si].get<double>();
        constexpr double kLossTol = 1e-9;
        if (!near_eq(got_losses[si], exp_l, kLossTol)) {
          std::cerr << "step " << si << " loss mismatch: got " << got_losses[si] << " expected " << exp_l << "\n";
          return 1;
        }
      }
    }

    if (!replay_u01_storage.empty() && replay_u01_pos != replay_u01_storage.size()) {
      std::cerr << "replay_u01 consumption mismatch: pos=" << replay_u01_pos << " len=" << replay_u01_storage.size()
                << "\n";
      return 1;
    }

    const auto& lo = j.at("label_order");
    if (static_cast<int>(lo.size()) != K) {
      throw std::runtime_error("label_order length mismatch K");
    }
    for (int k = 0; k < K; ++k) {
      if (infer.labels[static_cast<std::size_t>(k)] != lo[k].get<std::string>()) {
        std::cerr << "label_order mismatch at " << k << "\n";
        return 1;
      }
    }

    std::vector<double> x_all = j.at("x_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_llr = j.at("expected_llr_rowmajor").get<std::vector<double>>();
    if (static_cast<int>(x_all.size()) != n * d || static_cast<int>(exp_llr.size()) != n * K) {
      std::cerr << "bad row-major sizes\n";
      return 1;
    }

    std::vector<double> llr;
    cypha::batch_llr_from_x(infer, x_all.data(), n, llr);

    constexpr double kLlrTol = 1e-9;
    for (std::size_t i = 0; i < exp_llr.size(); ++i) {
      if (!near_eq(llr[i], exp_llr[i], kLlrTol)) {
        std::cerr << "LLR mismatch at " << i << " got " << llr[i] << " exp " << exp_llr[i] << "\n";
        return 1;
      }
    }

    std::cout << "preprocess_train_classify parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "preprocess_train_classify_parity (studio / csv_preprocess_classify_hotpath): " << e.what() << "\n";
    return 1;
  }
}
