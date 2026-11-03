/// CTest smoke: export reference.cypha to ONNX and validate ModelProto structure.
/// Exit 0 on success.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/onnx_min_writer.hpp"

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "onnx_export_smoke: FAIL - %s\n", msg.c_str());
  std::exit(1);
}

std::vector<float> to_f32(const std::vector<double>& in) {
  std::vector<float> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<float>(in[i]);
  }
  return out;
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
  if (h_sq <= 1e8) {
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

cypha::onnx::InferGraphSpec build_spec(const cypha::CyphaInferModel& m, bool with_softmax) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  if (K == 0) {
    throw std::runtime_error("model has no classes");
  }

  cypha::onnx::InferGraphSpec spec;
  spec.d = d;
  spec.k = K;
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

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot write " + path.string());
  }
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      fail("usage: onnx_export_smoke CYPHA_PATH [OUT_ONNX]");
    }
    const std::string cypha_path = argv[1];
    const std::filesystem::path out_path =
        (argc >= 3) ? std::filesystem::path(argv[2])
                    : (std::filesystem::temp_directory_path() / "cypha_onnx_export_smoke.onnx");

    const cypha::CNode root = cypha::load_cypha_file(cypha_path.c_str());
    const cypha::CNode* fh = cypha::map_get(root, "field_h");
    if (fh == nullptr || fh->kind != cypha::CNode::Tensor || fh->shape.size() != 1) {
      fail("field_h missing or invalid");
    }
    const int field_dim = static_cast<int>(fh->shape[0]);
    const cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, nullptr, field_dim);

    const cypha::onnx::InferGraphSpec spec = build_spec(model, true);
    const std::vector<std::uint8_t> onnx = cypha::onnx::build_cypha_infer_model(spec);
    write_bytes(out_path, onnx);

    const cypha::onnx::ParsedOnnxModel parsed = cypha::onnx::parse_model(onnx);
    cypha::onnx::validate_cypha_infer_model(parsed, true);

    std::printf("onnx_export_smoke: PASS (%zu bytes, %zu nodes, d=%d k=%d)\n", onnx.size(),
                parsed.nodes.size(), spec.d, spec.k);
    for (const auto& node : parsed.nodes) {
      std::printf("  node %s op=%s\n", node.name.c_str(), node.op.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    fail(e.what());
  }
}
