// Train kernel LLR, patch into .cypha root, save/reload, verify scores match.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/sync_infer.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

void compare_vec(const std::vector<double>& a, const std::vector<double>& b, const char* name, double atol) {
  if (a.size() != b.size()) {
    throw std::runtime_error(std::string("size mismatch: ") + name);
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!near_eq(a[i], b[i], atol)) {
      std::cerr << name << "[" << i << "] got " << a[i] << " expected " << b[i] << "\n";
      throw std::runtime_error(std::string("mismatch: ") + name);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string out_path = "kernel_cypha_roundtrip_out.cypha";
    if (argc >= 2) {
      out_path = argv[1];
    }

    constexpr int d = 4;
    constexpr int fd = 8;
    cypha::FreshModelParams fp;
    fp.input_dim = d;
    fp.field_dim = fd;
    fp.temperature = 1.15;
    fp.world_lr = 0.008;
    fp.delta_lr = 0.05;
    cypha::CNode root = cypha::create_fresh_model_root(fp);
    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fd);
    cypha::init_encoder_projection_w(d, 4242, infer.enc_w);
    cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fd);
    cypha::ReplayBuffer replay(10000);
    cypha::TrainStepParams tsp;
    tsp.replay_ratio = 0.0;
    cypha::TrainStepExtras extras;
    int total_steps = 0;
    extras.total_steps = &total_steps;
    cypha::KernelMemory km(d, 16, 4242);
    extras.kernel_mem = &km;
    extras.use_kernel_llr = true;
    extras.kernel_blend = 0.75;

    const std::vector<std::vector<double>> train_h = {
        {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.9, 0.1, 0.0, 0.0},
        {0.1, 0.9, 0.0, 0.0}, {0.8, 0.2, 0.1, 0.0}, {0.2, 0.8, 0.0, 0.1},
    };
    const std::vector<std::string> train_y = {"0", "1", "0", "1", "0", "1"};
    std::mt19937 rng(4242);
    int enc_updates = 0;
    for (std::size_t i = 0; i < train_h.size(); ++i) {
      cypha::dif_train_step_vector(infer, mem, replay, train_h[i].data(), d, train_y[i], 0.008, 0.05, 0.008, 0.05,
                                   15.0, tsp, rng, enc_updates, nullptr, &extras);
    }
    cypha::sync_infer_model_from_memory(infer, mem);

    const double h_test[d] = {0.95, 0.05, 0.0, 0.0};
    const std::vector<std::string> labels = {"0", "1"};
    std::vector<double> scores_before;
    km.score_all(h_test, labels, scores_before);

    cypha::CNode merged = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(root, mem);
    cypha::patch_kernel_into_root(merged, km, true, 0.75);
    cypha::save_cypha_file(out_path.c_str(), merged);

    cypha::CNode loaded = cypha::load_cypha_file(out_path.c_str());
    cypha::KernelMemory km2(d, 16, 0);
    bool use_kernel = false;
    double blend = 0.0;
    if (!cypha::try_load_kernel_from_root(loaded, km2, use_kernel, blend)) {
      throw std::runtime_error("kernel not loaded from .cypha");
    }
    if (!use_kernel || !near_eq(blend, 0.75, 1e-12)) {
      throw std::runtime_error("kernel flags mismatch after load");
    }
    std::vector<double> scores_after;
    km2.score_all(h_test, labels, scores_after);
    compare_vec(scores_after, scores_before, "kernel scores", 1e-12);

    std::cout << "kernel_cypha_roundtrip OK -> " << out_path << " (n_basis=" << km2.n_basis() << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "kernel_cypha_roundtrip FAIL: " << e.what() << "\n";
    return 1;
  }
}
