#pragma once

/// Minimal header-only ONNX ModelProto writer (protobuf wire format, no external deps).
/// Supports Gemm, MatMul, Sub, Mul, Add, Tanh, Div, Softmax for Cypha VectorEncoder inference.
///
/// JSON intermediate: use cypha_onnx_export --format json for an ONNX-ready graph description
/// that can be converted with onnx.helper / torch.onnx when needed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cypha::onnx {

constexpr int kTensorFloat = 1;
constexpr int kTensorDouble = 7;
constexpr int kIrVersion = 8;
constexpr int kOpsetVersion = 13;

namespace detail {

inline void write_varint(std::vector<std::uint8_t>& out, std::uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<std::uint8_t>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v));
}

inline void write_tag(std::vector<std::uint8_t>& out, int field, int wire) {
  write_varint(out, static_cast<std::uint64_t>((field << 3) | wire));
}

inline void write_bytes_field(std::vector<std::uint8_t>& out, int field, const std::uint8_t* data,
                              std::size_t len) {
  write_tag(out, field, 2);
  write_varint(out, len);
  out.insert(out.end(), data, data + len);
}

inline void write_string_field(std::vector<std::uint8_t>& out, int field, const std::string& s) {
  write_bytes_field(out, field, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
}

inline void write_message_field(std::vector<std::uint8_t>& out, int field,
                                const std::vector<std::uint8_t>& msg) {
  write_bytes_field(out, field, msg.data(), msg.size());
}

inline void write_int64_field(std::vector<std::uint8_t>& out, int field, std::int64_t v) {
  write_tag(out, field, 0);
  write_varint(out, static_cast<std::uint64_t>(v));
}

inline void write_int32_field(std::vector<std::uint8_t>& out, int field, std::int32_t v) {
  write_int64_field(out, field, static_cast<std::int64_t>(v));
}

inline void write_float_field(std::vector<std::uint8_t>& out, int field, float v) {
  write_tag(out, field, 5);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, 4);
  out.push_back(static_cast<std::uint8_t>(bits & 0xFF));
  out.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xFF));
  out.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xFF));
  out.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xFF));
}

inline std::vector<std::uint8_t> encode_tensor_shape(const std::vector<std::int64_t>& dims,
                                                     bool batch_dynamic) {
  std::vector<std::uint8_t> shape_msg;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    std::vector<std::uint8_t> dim_msg;
    if (batch_dynamic && i == 0) {
      write_string_field(dim_msg, 2, "batch");
    } else {
      write_int64_field(dim_msg, 1, dims[i]);
    }
    write_message_field(shape_msg, 1, dim_msg);
  }
  std::vector<std::uint8_t> tensor_type;
  write_int32_field(tensor_type, 1, kTensorFloat);
  write_message_field(tensor_type, 2, shape_msg);
  std::vector<std::uint8_t> type_proto;
  write_message_field(type_proto, 1, tensor_type);
  return type_proto;
}

inline std::vector<std::uint8_t> encode_value_info(const std::string& name,
                                                   const std::vector<std::int64_t>& dims,
                                                   bool batch_dynamic) {
  std::vector<std::uint8_t> msg;
  write_string_field(msg, 1, name);
  write_message_field(msg, 2, encode_tensor_shape(dims, batch_dynamic));
  return msg;
}

inline std::vector<std::uint8_t> encode_initializer_f32(const std::string& name,
                                                        const std::vector<std::int64_t>& dims,
                                                        const std::vector<float>& data) {
  std::vector<std::uint8_t> msg;
  write_string_field(msg, 1, name);
  for (std::int64_t d : dims) {
    write_int64_field(msg, 2, d);
  }
  write_int32_field(msg, 3, kTensorFloat);
  std::vector<std::uint8_t> raw(data.size() * 4);
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &data[i], 4);
    raw[i * 4 + 0] = static_cast<std::uint8_t>(bits & 0xFF);
    raw[i * 4 + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xFF);
    raw[i * 4 + 2] = static_cast<std::uint8_t>((bits >> 16) & 0xFF);
    raw[i * 4 + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xFF);
  }
  write_bytes_field(msg, 9, raw.data(), raw.size());
  return msg;
}

inline std::vector<std::uint8_t> encode_attr_int(const std::string& name, std::int64_t v) {
  std::vector<std::uint8_t> msg;
  write_string_field(msg, 1, name);
  write_int32_field(msg, 2, 2);  // INT
  write_int64_field(msg, 3, v);
  return msg;
}

inline std::vector<std::uint8_t> encode_attr_float(const std::string& name, float v) {
  std::vector<std::uint8_t> msg;
  write_string_field(msg, 1, name);
  write_int32_field(msg, 2, 1);  // FLOAT
  write_float_field(msg, 4, v);
  return msg;
}

inline std::vector<std::uint8_t> encode_node(const std::string& name, const std::string& op,
                                             const std::vector<std::string>& inputs,
                                             const std::vector<std::string>& outputs,
                                             const std::vector<std::vector<std::uint8_t>>& attrs) {
  std::vector<std::uint8_t> msg;
  for (const auto& in : inputs) {
    write_string_field(msg, 1, in);
  }
  for (const auto& out : outputs) {
    write_string_field(msg, 2, out);
  }
  write_string_field(msg, 3, name);
  write_string_field(msg, 4, op);
  for (const auto& a : attrs) {
    write_message_field(msg, 5, a);
  }
  return msg;
}

inline std::vector<std::uint8_t> encode_opset() {
  std::vector<std::uint8_t> msg;
  write_string_field(msg, 1, "");
  write_int64_field(msg, 2, kOpsetVersion);
  return msg;
}

}  // namespace detail

struct InferGraphSpec {
  int d{0};
  int k{0};
  std::vector<float> enc_w;       ///< (d, d) row-major
  std::vector<float> mu0;         ///< (d,) field-baked prior mean
  std::vector<float> inv_v;       ///< (d,)
  std::vector<float> d_t;         ///< (d, k) D transposed: D_T[j,k] = D[k,j]
  std::vector<float> llr_bias;    ///< (k,) = -0.5*d_sq - u_k + ctx
  bool encoder_tanh{false};
  float temperature{1.0f};
  bool with_softmax{false};
};

/// Build ONNX ModelProto bytes for: x -> encode -> llr [-> softmax probs].
inline std::vector<std::uint8_t> build_cypha_infer_model(const InferGraphSpec& spec) {
  if (spec.d <= 0 || spec.k <= 0) {
    throw std::runtime_error("onnx export: d and k must be positive");
  }
  const std::size_t need_w = static_cast<std::size_t>(spec.d * spec.d);
  const std::size_t need_d = static_cast<std::size_t>(spec.d);
  const std::size_t need_dt = static_cast<std::size_t>(spec.d * spec.k);
  const std::size_t need_b = static_cast<std::size_t>(spec.k);
  if (spec.enc_w.size() != need_w || spec.mu0.size() != need_d || spec.inv_v.size() != need_d ||
      spec.d_t.size() != need_dt || spec.llr_bias.size() != need_b) {
    throw std::runtime_error("onnx export: tensor size mismatch in InferGraphSpec");
  }

  std::vector<std::uint8_t> graph;
  detail::write_string_field(graph, 2, "cypha_infer");

  const std::vector<std::int64_t> in_dims{0, spec.d};
  detail::write_message_field(graph, 11, detail::encode_value_info("x", in_dims, true));

  const std::vector<std::int64_t> llr_dims{0, spec.k};
  detail::write_message_field(graph, 12, detail::encode_value_info("llr", llr_dims, true));
  if (spec.with_softmax) {
    detail::write_message_field(graph, 12, detail::encode_value_info("probs", llr_dims, true));
  }

  detail::write_message_field(graph, 5,
                              detail::encode_initializer_f32("enc_W", {spec.d, spec.d}, spec.enc_w));
  detail::write_message_field(graph, 5,
                              detail::encode_initializer_f32("mu0", {1, spec.d}, spec.mu0));
  detail::write_message_field(graph, 5,
                              detail::encode_initializer_f32("inv_v", {1, spec.d}, spec.inv_v));
  detail::write_message_field(graph, 5,
                              detail::encode_initializer_f32("D_T", {spec.d, spec.k}, spec.d_t));
  detail::write_message_field(graph, 5,
                              detail::encode_initializer_f32("llr_bias", {1, spec.k}, spec.llr_bias));

  const auto gemm_attrs = std::vector<std::vector<std::uint8_t>>{
      detail::encode_attr_int("transB", 1),
      detail::encode_attr_float("alpha", 1.0f),
      detail::encode_attr_float("beta", 0.0f),
  };
  detail::write_message_field(
      graph, 1,
      detail::encode_node("encode_gemm", "Gemm", {"x", "enc_W"}, {"h_raw"}, gemm_attrs));

  std::string h_name = "h_raw";
  if (spec.encoder_tanh) {
    detail::write_message_field(graph, 1,
                                detail::encode_node("encode_tanh", "Tanh", {"h_raw"}, {"h"}, {}));
    h_name = "h";
  }

  detail::write_message_field(graph, 1,
                              detail::encode_node("shift_mu0", "Sub", {h_name, "mu0"}, {"h0"}, {}));
  detail::write_message_field(graph, 1,
                              detail::encode_node("scale_inv_v", "Mul", {"h0", "inv_v"}, {"R"}, {}));
  detail::write_message_field(
      graph, 1, detail::encode_node("score_matmul", "MatMul", {"R", "D_T"}, {"cross"}, {}));
  detail::write_message_field(
      graph, 1, detail::encode_node("score_bias", "Add", {"cross", "llr_bias"}, {"llr"}, {}));

  if (spec.with_softmax) {
    const float inv_t = 1.0f / std::max(spec.temperature, 1e-8f);
    detail::write_message_field(
        graph, 5, detail::encode_initializer_f32("inv_temp", {1, 1}, std::vector<float>{inv_t}));
    detail::write_message_field(
        graph, 1,
        detail::encode_node("scale_temp", "Mul", {"llr", "inv_temp"}, {"llr_scaled"}, {}));
    const auto sm_attrs =
        std::vector<std::vector<std::uint8_t>>{detail::encode_attr_int("axis", -1)};
    detail::write_message_field(
        graph, 1,
        detail::encode_node("softmax", "Softmax", {"llr_scaled"}, {"probs"}, sm_attrs));
  }

  std::vector<std::uint8_t> model;
  detail::write_int64_field(model, 1, kIrVersion);
  detail::write_string_field(model, 2, "cypha_onnx_export");
  detail::write_message_field(model, 7, graph);
  detail::write_message_field(model, 8, detail::encode_opset());
  return model;
}

}  // namespace cypha::onnx
