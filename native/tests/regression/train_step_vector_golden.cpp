// One `dif_train_step_vector` from `reference.cypha` vs Python-recorded loss (fixtures/train_step_vector/).
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

#include "cypha/load_cypha.hpp"
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
      std::cerr << "usage: train_step_vector_golden <fixtures/train_step_vector_dir>\n";
      return 2;
    }
    fs::path dir = fs::path(argv[1]);
    fs::path root = dir.parent_path();
    std::ifstream sf(dir / "sidecar.json");
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    fs::path cypha_path = root / "reference.cypha";
    fs::path ff_path = root / "f_field.json";
    cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
    const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
    int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
    int d = static_cast<int>(enc.shape[0]);

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

    std::vector<double> x;
    for (const auto& v : j.at("x")) {
      x.push_back(v.get<double>());
    }
    if (static_cast<int>(x.size()) != d) {
      throw std::runtime_error("x dim mismatch");
    }
    std::string label = j.at("label").get<std::string>();
    double expected_loss = j.at("expected_loss").get<double>();
    int total_steps_before = j.at("total_steps_before").get<int>();

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
    std::mt19937 rng{424242};
    int enc_updates = 0;
    int step_count = total_steps_before;
    cypha::TrainStepExtras extras{};
    extras.total_steps = &step_count;
    extras.ood_sigma = nullptr;
    extras.llr_ema = nullptr;

    double loss = cypha::dif_train_step_vector(infer, mem, replay, x.data(), d, label, world_lr, delta_lr, world_lr,
                                               delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr, &extras);

    constexpr double kAtol = 1e-9;
    if (!near_eq(loss, expected_loss, kAtol)) {
      std::cerr << "loss mismatch: got " << loss << " expected " << expected_loss << "\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
