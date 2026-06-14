// cypha_export_gguf — export CyphaDIF inference tensors to GGUF v3.
//
// Default mode embeds float32 tensor blobs (enc_W, world.mu, class D, inv_v, llr_bias).
// Use --header-only for metadata-only smoke (no weight bytes).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;  // "GGUF" little-endian
constexpr std::uint32_t kGgufVersion = 3;
constexpr std::uint32_t kGgufTypeString = 8;
constexpr std::uint32_t kGgufTypeUint32 = 4;
constexpr std::uint32_t kGgufTensorTypeF32 = 0;
constexpr std::uint64_t kGgufAlignment = 32;

struct Args {
  std::string cypha_path;
  std::string out_path;
  std::string manifest_path;
  bool embed_weights{true};
  bool dry_run{false};
  bool help{false};
};

void usage() {
  std::cerr
      << "cypha_export_gguf — export CyphaDIF tensors to GGUF v3\n\n"
      << "usage: cypha_export_gguf --cypha PATH --out PATH [options]\n\n"
      << "options:\n"
      << "  --manifest PATH      tensor manifest JSON (default: <out>.manifest.json)\n"
      << "  --header-only        metadata + tensor info only (no weight blobs)\n"
      << "  --dry-run            load model and print summary; do not write files\n"
      << "  --help               show this message\n\n"
      << "notes:\n"
      << "  - Default embeds enc_W, world.mu (field-conditioned), D [K,d], inv_v, llr_bias.\n"
      << "  - Field-conditioned world.mu and llr_bias are frozen at export time.\n"
      << "  - Sidecar manifest JSON lists tensor shapes and optional inline data.\n";
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (k == "--help" || k == "-h") {
      a.help = true;
    } else if (k == "--cypha") {
      a.cypha_path = need("--cypha");
    } else if (k == "--out") {
      a.out_path = need("--out");
    } else if (k == "--manifest") {
      a.manifest_path = need("--manifest");
    } else if (k == "--header-only") {
      a.embed_weights = false;
    } else if (k == "--full-gguf") {
      a.embed_weights = true;
    } else if (k == "--dry-run") {
      a.dry_run = true;
    } else {
      throw std::runtime_error("unknown argument: " + k);
    }
  }
  if (!a.help && a.cypha_path.empty()) {
    throw std::runtime_error("--cypha is required");
  }
  if (!a.help && !a.dry_run && a.out_path.empty()) {
    throw std::runtime_error("--out is required unless --dry-run");
  }
  return a;
}

int field_dim_from_root(const cypha::CNode& root) {
  const cypha::CNode* fh = cypha::map_get(root, "field_h");
  if (fh == nullptr || fh->kind != cypha::CNode::Tensor || fh->shape.size() != 1) {
    throw std::runtime_error("field_h missing or invalid in .cypha root");
  }
  return static_cast<int>(fh->shape[0]);
}

std::vector<double> compute_mu0_with_field(const cypha::CyphaInferModel& m) {
  const int d = m.d_latent;
  std::vector<double> mu0(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = m.mu_world[static_cast<std::size_t>(j)];
  }
  double h_sq = 0.0;
  for (double v : m.field_h) {
    h_sq += v * v;
  }
  if (std::isfinite(h_sq) && h_sq <= 1e8) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int t = 0; t < m.field_dim; ++t) {
        acc += m.f_field[static_cast<std::size_t>(j * m.field_dim + t)] *
               m.field_h[static_cast<std::size_t>(t)];
      }
      mu0[static_cast<std::size_t>(j)] += acc;
    }
  }
  return mu0;
}

template <typename T>
std::vector<float> to_f32(const std::vector<T>& in) {
  std::vector<float> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<float>(in[i]);
  }
  return out;
}

struct TensorManifestEntry {
  std::string name;
  std::vector<std::int64_t> shape;
  std::string dtype{"float32"};
  std::vector<float> data;
};

struct ExportTensors {
  int d{0};
  int k{0};
  float temperature{1.0f};
  std::vector<TensorManifestEntry> tensors;
};

ExportTensors build_tensors(const cypha::CyphaInferModel& m) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  if (K == 0) {
    throw std::runtime_error("model has no classes; nothing to export");
  }

  ExportTensors out;
  out.d = d;
  out.k = K;
  out.temperature = static_cast<float>(m.temperature);

  const std::vector<float> enc_w = to_f32(m.enc_w);
  out.tensors.push_back({"enc_W", {d, d}, "float32", enc_w});

  const std::vector<float> world_mu = to_f32(compute_mu0_with_field(m));
  out.tensors.push_back({"world.mu", {1, d}, "float32", world_mu});

  const std::vector<float> inv_v = to_f32(m.inv_v);
  out.tensors.push_back({"inv_v", {1, d}, "float32", inv_v});

  std::vector<float> d_mat(static_cast<std::size_t>(K * d));
  for (int kk = 0; kk < K; ++kk) {
    for (int j = 0; j < d; ++j) {
      d_mat[static_cast<std::size_t>(kk * d + j)] =
          static_cast<float>(m.D[static_cast<std::size_t>(kk * d + j)]);
    }
  }
  out.tensors.push_back({"D", {K, d}, "float32", d_mat});

  std::vector<float> d_t(static_cast<std::size_t>(d * K));
  for (int kk = 0; kk < K; ++kk) {
    for (int j = 0; j < d; ++j) {
      d_t[static_cast<std::size_t>(j * K + kk)] =
          static_cast<float>(m.D[static_cast<std::size_t>(kk * d + j)]);
    }
  }
  out.tensors.push_back({"D_T", {d, K}, "float32", d_t});

  std::vector<double> ctx;
  cypha::context_prior_for_labels(m, m.labels, ctx);

  std::vector<float> llr_bias(static_cast<std::size_t>(K));
  for (int kk = 0; kk < K; ++kk) {
    double d_sq = 0.0;
    for (int j = 0; j < d; ++j) {
      const double Dkj = m.D[static_cast<std::size_t>(kk * d + j)];
      d_sq += Dkj * Dkj * m.inv_v[static_cast<std::size_t>(j)];
    }
    const double nk = m.n_obs[static_cast<std::size_t>(kk)];
    const double u_k = m.v_mean / (nk + 1.0);
    llr_bias[static_cast<std::size_t>(kk)] =
        static_cast<float>(-0.5 * d_sq - u_k + ctx[static_cast<std::size_t>(kk)]);
  }
  out.tensors.push_back({"llr_bias", {1, K}, "float32", llr_bias});

  return out;
}

void write_u32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
  buf.push_back(static_cast<std::uint8_t>(v & 0xff));
  buf.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
  buf.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
  buf.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
}

void write_u64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
  }
}

void write_string(std::vector<std::uint8_t>& buf, const std::string& s) {
  write_u64(buf, static_cast<std::uint64_t>(s.size()));
  buf.insert(buf.end(), s.begin(), s.end());
}

void write_metadata_string(std::vector<std::uint8_t>& buf, const std::string& key,
                           const std::string& value) {
  write_string(buf, key);
  write_u32(buf, kGgufTypeString);
  write_string(buf, value);
}

void write_metadata_uint32(std::vector<std::uint8_t>& buf, const std::string& key,
                           std::uint32_t value) {
  write_string(buf, key);
  write_u32(buf, kGgufTypeUint32);
  write_u32(buf, value);
}

void write_tensor_info(std::vector<std::uint8_t>& buf, const TensorManifestEntry& t,
                       std::uint64_t offset) {
  write_string(buf, t.name);
  write_u32(buf, static_cast<std::uint32_t>(t.shape.size()));
  for (const auto dim : t.shape) {
    write_u64(buf, static_cast<std::uint64_t>(dim));
  }
  write_u32(buf, kGgufTensorTypeF32);
  write_u64(buf, offset);
}

std::string default_manifest_path(const std::string& out_path) {
  return out_path + ".manifest.json";
}

nlohmann::json manifest_to_json(const cypha::CyphaInferModel& m, const ExportTensors& exp,
                                bool embed_weights) {
  nlohmann::json j;
  j["format"] = "cypha-gguf-manifest-v1";
  j["gguf_version"] = kGgufVersion;
  j["architecture"] = "cypha-dif";
  j["d_latent"] = exp.d;
  j["num_classes"] = exp.k;
  j["labels"] = m.labels;
  j["temperature"] = m.temperature;
  j["field_dim"] = m.field_dim;
  j["weights_embedded"] = embed_weights;
  j["notes"] =
      "Manifest for CyphaDIF VectorEncoder+LLR inference tensors. "
      "world.mu includes baked field shift; llr_bias includes MDL/context terms.";

  nlohmann::json tensors = nlohmann::json::array();
  for (const auto& t : exp.tensors) {
    nlohmann::json entry;
    entry["name"] = t.name;
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["num_elements"] = t.data.size();
    if (!embed_weights) {
      entry["data"] = "omitted — default export embeds weights in .gguf";
    } else {
      entry["data"] = t.data;
    }
    tensors.push_back(entry);
  }
  j["tensors"] = tensors;
  return j;
}

std::size_t tensor_info_wire_size(const TensorManifestEntry& t) {
  return 8 + t.name.size() + 4 + 8 * t.shape.size() + 4 + 8;
}

std::vector<std::uint8_t> build_gguf(const ExportTensors& exp, bool embed_weights) {
  std::vector<std::uint8_t> buf;
  const std::uint64_t n_tensors = static_cast<std::uint64_t>(exp.tensors.size());
  const std::uint64_t n_kv = embed_weights ? 6 : 5;

  write_u32(buf, kGgufMagic);
  write_u32(buf, kGgufVersion);
  write_u64(buf, n_tensors);
  write_u64(buf, n_kv);

  write_metadata_string(buf, "general.architecture", "cypha-dif");
  write_metadata_string(buf, "general.name", "cypha-dif-export");
  write_metadata_uint32(buf, "cypha.d_latent", static_cast<std::uint32_t>(exp.d));
  write_metadata_uint32(buf, "cypha.num_classes", static_cast<std::uint32_t>(exp.k));
  write_metadata_string(buf, "cypha.export_mode", embed_weights ? "weights" : "header-only");
  if (embed_weights) {
    write_metadata_uint32(buf, "general.alignment", static_cast<std::uint32_t>(kGgufAlignment));
  }

  std::size_t info_bytes = 0;
  for (const auto& t : exp.tensors) {
    info_bytes += tensor_info_wire_size(t);
  }

  const std::uint64_t data_start =
      ((static_cast<std::uint64_t>(buf.size()) + info_bytes + kGgufAlignment - 1) /
       kGgufAlignment) *
      kGgufAlignment;

  std::uint64_t running = data_start;
  for (const auto& t : exp.tensors) {
    const std::uint64_t offset = embed_weights ? running : 0;
    write_tensor_info(buf, t, offset);
    if (embed_weights) {
      running += static_cast<std::uint64_t>(t.data.size() * sizeof(float));
    }
  }

  if (embed_weights) {
    while (buf.size() % kGgufAlignment != 0) {
      buf.push_back(0);
    }
    for (const auto& t : exp.tensors) {
      const auto* p = reinterpret_cast<const std::uint8_t*>(t.data.data());
      buf.insert(buf.end(), p, p + t.data.size() * sizeof(float));
    }
  }

  return buf;
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot write " + path);
  }
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_json_file(const std::string& path, const nlohmann::json& j) {
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("cannot write " + path);
  }
  f << j.dump(2) << '\n';
}

void print_summary(const cypha::CyphaInferModel& m, const ExportTensors& exp, const Args& a) {
  std::cout << "cypha_export_gguf summary\n"
            << "  cypha:       " << a.cypha_path << '\n'
            << "  d_latent:    " << exp.d << '\n'
            << "  classes:     " << exp.k << '\n'
            << "  field_dim:   " << m.field_dim << '\n'
            << "  temperature: " << m.temperature << '\n'
            << "  tensors:     " << exp.tensors.size() << '\n'
            << "  mode:        " << (a.embed_weights ? "weights" : "header-only") << '\n';
  for (const auto& t : exp.tensors) {
    std::cout << "    - " << t.name << " [";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << t.shape[i];
    }
    std::cout << "]\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse_args(argc, argv);
    if (a.help) {
      usage();
      return 0;
    }

    const cypha::CNode root = cypha::load_cypha_file(a.cypha_path.c_str());
    const int field_dim = field_dim_from_root(root);
    const cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, nullptr, field_dim);
    const ExportTensors exp = build_tensors(model);

    print_summary(model, exp, a);

    if (a.dry_run) {
      std::cout << "dry-run: no files written\n";
      return 0;
    }

    const std::string manifest_path =
        a.manifest_path.empty() ? default_manifest_path(a.out_path) : a.manifest_path;
    const nlohmann::json manifest = manifest_to_json(model, exp, a.embed_weights);
    write_json_file(manifest_path, manifest);

    const std::vector<std::uint8_t> gguf = build_gguf(exp, a.embed_weights);
    write_file(a.out_path, gguf);

    std::cout << "wrote GGUF (" << gguf.size() << " bytes): " << a.out_path << '\n'
              << "wrote manifest: " << manifest_path << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cypha_export_gguf: " << e.what() << '\n';
    return 1;
  }
}
