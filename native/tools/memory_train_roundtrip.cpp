// Train one DIFMemory step (parity_fixtures/memory_train), merge state, save .cypha, reload and compare to after.cypha.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/nig_field.hpp"

namespace {

constexpr double kFpTol = 1e-12;

bool near_fp(double x, double y) { return std::abs(x - y) <= kFpTol; }

bool cnode_deep_equal(const cypha::CNode& a, const cypha::CNode& b) {
  if (a.kind != b.kind) {
    return false;
  }
  switch (a.kind) {
    case cypha::CNode::Nil:
      return true;
    case cypha::CNode::Bool:
      return a.b == b.b;
    case cypha::CNode::Int:
      return a.i == b.i;
    case cypha::CNode::Float:
      return near_fp(a.f, b.f);
    case cypha::CNode::Str:
      return a.s == b.s;
    case cypha::CNode::Tensor:
      if (a.shape != b.shape || a.tensor.size() != b.tensor.size()) {
        return false;
      }
      for (std::size_t i = 0; i < a.tensor.size(); ++i) {
        if (!near_fp(a.tensor[i], b.tensor[i])) {
          return false;
        }
      }
      return true;
    case cypha::CNode::Map: {
      if (a.map.size() != b.map.size()) {
        return false;
      }
      std::unordered_map<std::string, const cypha::CNode*> by_key;
      by_key.reserve(b.map.size() * 2);
      for (const auto& kv : b.map) {
        by_key.emplace(kv.first, &kv.second);
      }
      for (const auto& kv : a.map) {
        auto it = by_key.find(kv.first);
        if (it == by_key.end() || !cnode_deep_equal(kv.second, *it->second)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

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
    if (argc != 2 && argc != 3) {
      std::fprintf(stderr,
                   "usage: memory_train_roundtrip <parity_fixtures/memory_train_dir> [out.cypha]\n"
                   "  default out: <dir>/roundtrip_native.cypha\n");
      return 2;
    }
    const std::string dir = argv[1];
    const std::string sidepath = dir + "/sidecar.json";
    std::ifstream sf(sidepath);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    const auto j = nlohmann::json::parse(buf.str());

    std::vector<double> h;
    for (const auto& v : j["h"]) {
      h.push_back(v.get<double>());
    }
    std::vector<double> h_field;
    for (const auto& v : j["h_field"]) {
      h_field.push_back(v.get<double>());
    }
    const std::string label = j["label"].get<std::string>();
    const double temperature = j["temperature"].get<double>();
    const double ood_sigma = j["ood_sigma"].get<double>();
    const double world_lr = j["world_lr"].get<double>();
    const double delta_lr = j["delta_lr"].get<double>();
    const double expected_loss = j["expected_loss"].get<double>();
    const int field_dim = j["field_dim"].get<int>();
    const std::vector<double> f_flat = flatten_f_field(j["f_field"]);

    std::unordered_map<std::string, double> ctx;
    for (auto it = j["context_prior"].begin(); it != j["context_prior"].end(); ++it) {
      ctx[it.key()] = it.value().get<double>();
    }

    cypha::CNode before = cypha::load_cypha_file((dir + "/before.cypha").c_str());
    cypha::CyphaDifMemoryState st = cypha::CyphaDifMemoryState::from_cypha_root(before, f_flat.data(), field_dim);

    const double loss = st.memory_train(h.data(), label, h_field.data(), ctx, temperature, ood_sigma, world_lr,
                                        delta_lr, nullptr);
    constexpr double kLossTol = 1e-9;
    if (std::abs(loss - expected_loss) > kLossTol) {
      std::fprintf(stderr, "loss got %g expected %g\n", loss, expected_loss);
      return 1;
    }

    cypha::CNode merged = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(before, st);
    cypha::patch_field_a_eff_into_root(merged);
    const std::string out_path = (argc == 3) ? std::string(argv[2]) : (dir + "/roundtrip_native.cypha");
    std::vector<std::uint8_t> cypha_bytes = cypha::save_cypha_to_buffer(merged);
    cypha::save_cypha_file(out_path.c_str(), merged);

    {
      std::ifstream rf(out_path, std::ios::binary);
      if (!rf) {
        throw std::runtime_error("cannot reopen output .cypha for byte compare");
      }
      std::vector<std::uint8_t> on_disk((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
      if (on_disk.size() != cypha_bytes.size() ||
          std::memcmp(on_disk.data(), cypha_bytes.data(), cypha_bytes.size()) != 0) {
        std::fprintf(stderr, "on-disk .cypha bytes differ from save_cypha_to_buffer\n");
        return 1;
      }
    }

    cypha::CNode rt = cypha::load_cypha_file(out_path.c_str());
    cypha::CNode after = cypha::load_cypha_file((dir + "/after.cypha").c_str());
    if (!cnode_deep_equal(rt, after)) {
      std::fprintf(stderr, "roundtrip tree mismatch vs after.cypha (see %s)\n", out_path.c_str());
      return 1;
    }

    cypha::CNode from_buf =
        cypha::load_cypha_from_buffer(cypha_bytes.data(), cypha_bytes.size());
    if (!cnode_deep_equal(from_buf, rt)) {
      std::fprintf(stderr, "save_cypha_to_buffer / load_cypha_from_buffer mismatch vs file round-trip\n");
      return 1;
    }

    std::printf("memory_train_roundtrip OK\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "memory_train_roundtrip: %s\n", e.what());
    return 1;
  }
}
