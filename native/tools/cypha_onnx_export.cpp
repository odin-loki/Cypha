// cypha_onnx_export — export VectorEncoder inference (batch_encode + score_matrix) to ONNX.
//
// Default writes a valid .onnx ModelProto (minimal embedded protobuf writer, no deps).
// Use --format json for an ONNX-ready intermediate (.onnx.json) with weights inline.
//
// Graph (field-conditioned μ₀ baked at export time):
//   x [batch,d] -> Gemm(enc_W, transB=1) -> [Tanh?] -> Sub(mu0) -> Mul(inv_v)
//   -> MatMul(D_T) -> Add(llr_bias) -> llr [-> Mul(1/T) -> Softmax -> probs]
#include <cmath>
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
#include "cypha/onnx_min_writer.hpp"

namespace {

struct Args {
  std::string cypha_path;
  std::string out_path;
  std::string format{"onnx"};  // onnx | json
  bool encoder_tanh{false};
  bool with_softmax{false};
  bool dry_run{false};
  bool help{false};
};

void usage() {
  std::cerr
      << "cypha_onnx_export — export CyphaDIF VectorEncoder inference to ONNX\n\n"
      << "usage: cypha_onnx_export --cypha PATH --out PATH [options]\n\n"
      << "options:\n"
      << "  --format onnx|json   output format (default onnx)\n"
      << "  --activation tanh|none post-encode activation (default none; VectorEncoder)\n"
      << "  --with-softmax       append temperature-scaled Softmax output \"probs\" (default: off)\n"
      << "  --dry-run            load model and print graph summary; do not write file\n"
      << "  --help               show this message\n\n"
      << "notes:\n"
      << "  - score_matrix field/context priors are baked into mu0 and llr_bias at export time.\n"
      << "  - kernel LLR blend and deliberation are not exported (inference-only subgraph).\n"
      << "  - JSON format is ONNX-ready; convert with onnx.helper or external tooling if needed.\n";
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
    } else if (k == "--format") {
      a.format = need("--format");
    } else if (k == "--activation") {
      const std::string v = need("--activation");
      if (v == "tanh") {
        a.encoder_tanh = true;
      } else if (v != "none") {
        throw std::runtime_error("--activation must be tanh or none");
      }
    } else if (k == "--with-softmax") {
      a.with_softmax = true;
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
  if (a.format != "onnx" && a.format != "json") {
    throw std::runtime_error("--format must be onnx or json");
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

cypha::onnx::InferGraphSpec build_spec(const cypha::CyphaInferModel& m, bool encoder_tanh,
                                       bool with_softmax) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  if (K == 0) {
    throw std::runtime_error("model has no classes; nothing to export");
  }

  cypha::onnx::InferGraphSpec spec;
  spec.d = d;
  spec.k = K;
  spec.encoder_tanh = encoder_tanh;
  spec.with_softmax = with_softmax;
  spec.temperature = static_cast<float>(m.temperature);

  spec.enc_w = to_f32(m.enc_w);

  spec.mu0 = to_f32(compute_mu0_with_field(m));

  spec.inv_v = to_f32(m.inv_v);

  spec.d_t.assign(static_cast<std::size_t>(d * K), 0.0f);
  for (int kk = 0; kk < K; ++kk) {
    for (int j = 0; j < d; ++j) {
      spec.d_t[static_cast<std::size_t>(j * K + kk)] =
          static_cast<float>(m.D[static_cast<std::size_t>(kk * d + j)]);
    }
  }

  std::vector<double> ctx;
  cypha::context_prior_for_labels(m, m.labels, ctx);

  spec.llr_bias.assign(static_cast<std::size_t>(K), 0.0f);
  for (int kk = 0; kk < K; ++kk) {
    double d_sq = 0.0;
    for (int j = 0; j < d; ++j) {
      const double Dkj = m.D[static_cast<std::size_t>(kk * d + j)];
      d_sq += Dkj * Dkj * m.inv_v[static_cast<std::size_t>(j)];
    }
    const double nk = m.n_obs[static_cast<std::size_t>(kk)];
    const double u_k = m.v_mean / (nk + 1.0);
    spec.llr_bias[static_cast<std::size_t>(kk)] =
        static_cast<float>(-0.5 * d_sq - u_k + ctx[static_cast<std::size_t>(kk)]);
  }

  return spec;
}

void print_summary(const cypha::CyphaInferModel& m, const cypha::onnx::InferGraphSpec& spec,
                   const Args& a) {
  std::cout << "cypha_onnx_export summary\n"
            << "  cypha:       " << a.cypha_path << '\n'
            << "  d_latent:    " << spec.d << '\n'
            << "  classes:     " << spec.k << " [";
  for (std::size_t i = 0; i < m.labels.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << m.labels[i];
  }
  std::cout << "]\n"
            << "  field_dim:   " << m.field_dim << " (mu0 field shift baked)\n"
            << "  temperature: " << m.temperature << '\n'
            << "  format:      " << a.format << '\n'
            << "  activation:  " << (spec.encoder_tanh ? "tanh" : "none") << '\n'
            << "  softmax:     " << (spec.with_softmax ? "yes" : "no") << '\n'
            << "  nodes:       Gemm";
  if (spec.encoder_tanh) {
    std::cout << " -> Tanh";
  }
  std::cout << " -> Sub -> Mul -> MatMul -> Add";
  if (spec.with_softmax) {
    std::cout << " -> Mul(1/T) -> Softmax";
  }
  std::cout << '\n';
}

nlohmann::json spec_to_json(const cypha::CyphaInferModel& m, const cypha::onnx::InferGraphSpec& spec) {
  nlohmann::json j;
  j["format"] = "cypha-onnx-ready-v1";
  j["opset"] = cypha::onnx::kOpsetVersion;
  j["ir_version"] = cypha::onnx::kIrVersion;
  j["producer"] = "cypha_onnx_export";
  j["d_latent"] = spec.d;
  j["num_classes"] = spec.k;
  j["labels"] = m.labels;
  j["temperature"] = m.temperature;
  j["encoder_activation"] = spec.encoder_tanh ? "tanh" : "none";
  j["with_softmax"] = spec.with_softmax;

  j["inputs"] = nlohmann::json::array({{{"name", "x"}, {"shape", {"batch", spec.d}}}});
  nlohmann::json outputs = nlohmann::json::array({{{"name", "llr"}, {"shape", {"batch", spec.k}}}});
  if (spec.with_softmax) {
    outputs.push_back({{"name", "probs"}, {"shape", {"batch", spec.k}}});
  }
  j["outputs"] = outputs;

  nlohmann::json nodes = nlohmann::json::array();
  nodes.push_back({{"name", "encode_gemm"},
                   {"op", "Gemm"},
                   {"inputs", {"x", "enc_W"}},
                   {"outputs", {"h_raw"}},
                   {"attributes", {{"transB", 1}, {"alpha", 1.0}, {"beta", 0.0}}}});
  std::string h_tensor = "h_raw";
  if (spec.encoder_tanh) {
    nodes.push_back(
        {{"name", "encode_tanh"}, {"op", "Tanh"}, {"inputs", {"h_raw"}}, {"outputs", {"h"}}});
    h_tensor = "h";
  }
  nodes.push_back({{"name", "shift_mu0"}, {"op", "Sub"}, {"inputs", {h_tensor, "mu0"}}, {"outputs", {"h0"}}});
  nodes.push_back({{"name", "scale_inv_v"}, {"op", "Mul"}, {"inputs", {"h0", "inv_v"}}, {"outputs", {"R"}}});
  nodes.push_back(
      {{"name", "score_matmul"}, {"op", "MatMul"}, {"inputs", {"R", "D_T"}}, {"outputs", {"cross"}}});
  nodes.push_back(
      {{"name", "score_bias"}, {"op", "Add"}, {"inputs", {"cross", "llr_bias"}}, {"outputs", {"llr"}}});
  if (spec.with_softmax) {
    nodes.push_back({{"name", "scale_temp"},
                     {"op", "Mul"},
                     {"inputs", {"llr", "inv_temp"}},
                     {"outputs", {"llr_scaled"}}});
    nodes.push_back({{"name", "softmax"},
                     {"op", "Softmax"},
                     {"inputs", {"llr_scaled"}},
                     {"outputs", {"probs"}},
                     {"attributes", {{"axis", -1}}}});
  }
  j["nodes"] = nodes;

  j["initializers"] = {
      {"enc_W", {{"shape", {spec.d, spec.d}}, {"dtype", "float32"}, {"data", spec.enc_w}}},
      {"mu0", {{"shape", {1, spec.d}}, {"dtype", "float32"}, {"data", spec.mu0}}},
      {"inv_v", {{"shape", {1, spec.d}}, {"dtype", "float32"}, {"data", spec.inv_v}}},
      {"D_T", {{"shape", {spec.d, spec.k}}, {"dtype", "float32"}, {"data", spec.d_t}}},
      {"llr_bias", {{"shape", {1, spec.k}}, {"dtype", "float32"}, {"data", spec.llr_bias}}}};
  if (spec.with_softmax) {
    const float inv_t = 1.0f / std::max(spec.temperature, 1e-8f);
    j["initializers"]["inv_temp"] = {
        {"shape", {1, 1}}, {"dtype", "float32"}, {"data", std::vector<float>{inv_t}}};
  }

  j["notes"] =
      "Field-conditioned mu0 and context/MDL biases are frozen at export. "
      "Load in ONNX Runtime or convert via onnx.helper.make_model from this JSON.";
  return j;
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
    const cypha::onnx::InferGraphSpec spec = build_spec(model, a.encoder_tanh, a.with_softmax);

    print_summary(model, spec, a);

    if (a.dry_run) {
      std::cout << "dry-run: no file written\n";
      return 0;
    }

    if (a.format == "json") {
      const nlohmann::json j = spec_to_json(model, spec);
      write_json_file(a.out_path, j);
      std::cout << "wrote JSON ONNX-ready graph: " << a.out_path << '\n';
    } else {
      const std::vector<std::uint8_t> onnx = cypha::onnx::build_cypha_infer_model(spec);
      write_file(a.out_path, onnx);
      std::cout << "wrote ONNX ModelProto (" << onnx.size() << " bytes): " << a.out_path << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cypha_onnx_export: " << e.what() << '\n';
    return 1;
  }
}
