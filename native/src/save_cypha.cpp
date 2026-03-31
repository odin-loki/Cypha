#include "cypha/load_cypha.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cypha {

namespace {

constexpr unsigned char kF64 = 0;
constexpr unsigned char kArr = 1;
constexpr unsigned char kStr = 2;
constexpr unsigned char kNone = 3;
constexpr unsigned char kI64 = 4;
constexpr unsigned char kBool = 5;
constexpr unsigned char kDict = 6;

constexpr std::uint8_t kFileVersion = 3;
constexpr std::uint32_t kEndianSentinel = 0x01020304u;

template <class T>
void append_trivial(std::vector<std::uint8_t>& b, const T& v) {
  static_assert(std::is_trivially_copyable<T>::value, "");
  const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}

void append_bytes(std::vector<std::uint8_t>& b, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  b.insert(b.end(), p, p + n);
}

void write_value(std::vector<std::uint8_t>& b, const CNode& n) {
  switch (n.kind) {
    case CNode::Nil:
      b.push_back(kNone);
      return;
    case CNode::Bool:
      b.push_back(kBool);
      b.push_back(static_cast<std::uint8_t>(n.b ? 1 : 0));
      return;
    case CNode::Int:
      b.push_back(kI64);
      append_trivial(b, n.i);
      return;
    case CNode::Float:
      b.push_back(kF64);
      append_trivial(b, n.f);
      return;
    case CNode::Str: {
      b.push_back(kStr);
      if (n.s.size() > 65535U) {
        throw std::runtime_error("save_cypha: string too long");
      }
      const auto slen = static_cast<std::uint16_t>(n.s.size());
      append_trivial(b, slen);
      append_bytes(b, n.s.data(), n.s.size());
      return;
    }
    case CNode::Tensor: {
      b.push_back(kArr);
      if (n.shape.size() > 255U) {
        throw std::runtime_error("save_cypha: tensor ndim");
      }
      b.push_back(static_cast<std::uint8_t>(n.shape.size()));
      std::uint64_t count = 1;
      for (std::uint32_t s : n.shape) {
        append_trivial(b, s);
        count *= s;
      }
      if (count != n.tensor.size()) {
        throw std::runtime_error("save_cypha: tensor shape/data mismatch");
      }
      append_bytes(b, n.tensor.data(), n.tensor.size() * sizeof(double));
      return;
    }
    case CNode::Map: {
      b.push_back(kDict);
      if (n.map.size() > 0xFFFFFFFFu) {
        throw std::runtime_error("save_cypha: dict too large");
      }
      const auto n_sub = static_cast<std::uint32_t>(n.map.size());
      append_trivial(b, n_sub);
      for (const auto& kv : n.map) {
        if (kv.first.size() > 65535U) {
          throw std::runtime_error("save_cypha: dict key too long");
        }
        const auto klen = static_cast<std::uint16_t>(kv.first.size());
        append_trivial(b, klen);
        append_bytes(b, kv.first.data(), kv.first.size());
        write_value(b, kv.second);
      }
      return;
    }
    default:
      throw std::runtime_error("save_cypha: unsupported CNode kind");
  }
}

}  // namespace

CNode clone_cnode(const CNode& n) {
  CNode o;
  o.kind = n.kind;
  o.b = n.b;
  o.i = n.i;
  o.f = n.f;
  o.s = n.s;
  o.shape = n.shape;
  o.tensor = n.tensor;
  o.map.reserve(n.map.size());
  for (const auto& kv : n.map) {
    o.map.emplace_back(kv.first, clone_cnode(kv.second));
  }
  return o;
}

std::vector<std::uint8_t> save_cypha_to_buffer(const CNode& root) {
  if (root.kind != CNode::Map) {
    throw std::runtime_error("save_cypha: root must be a map");
  }
  std::vector<std::uint8_t> b;
  static const char kMagic[] = "CYPHA\0";
  append_bytes(b, kMagic, 6);
  b.push_back(kFileVersion);
  append_trivial(b, kEndianSentinel);
  if (root.map.size() > 0xFFFFFFFFu) {
    throw std::runtime_error("save_cypha: too many top-level keys");
  }
  const auto n_fields = static_cast<std::uint32_t>(root.map.size());
  append_trivial(b, n_fields);
  for (const auto& kv : root.map) {
    if (kv.first.size() > 65535U) {
      throw std::runtime_error("save_cypha: top key too long");
    }
    const auto klen = static_cast<std::uint16_t>(kv.first.size());
    append_trivial(b, klen);
    append_bytes(b, kv.first.data(), kv.first.size());
    write_value(b, kv.second);
  }
  return b;
}

void save_cypha_file(const char* path, const CNode& root) {
  if (path == nullptr) {
    throw std::runtime_error("save_cypha: null path");
  }
  std::vector<std::uint8_t> b = save_cypha_to_buffer(root);
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error(std::string("save_cypha: cannot open ") + path);
  }
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

}  // namespace cypha
