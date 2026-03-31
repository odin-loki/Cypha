#include "cypha/load_cypha.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cypha {

namespace {

constexpr unsigned char kF64 = 0;
constexpr unsigned char kArr = 1;
constexpr unsigned char kStr = 2;
constexpr unsigned char kNone = 3;
constexpr unsigned char kI64 = 4;
constexpr unsigned char kBool = 5;
constexpr unsigned char kDict = 6;

void need(std::size_t have, std::size_t need, const char* what) {
  if (have < need) {
    throw std::runtime_error(std::string("Truncated Cypha binary while reading ") + what);
  }
}

CNode read_value(const std::uint8_t*& p, std::size_t& rem) {
  need(rem, 1, "dtype");
  unsigned char dtype = *p++;
  rem--;

  CNode out;
  switch (dtype) {
    case kNone:
      out.kind = CNode::Nil;
      return out;
    case kBool:
      need(rem, 1, "bool payload");
      out.kind = CNode::Bool;
      out.b = (*p++ != 0);
      rem--;
      return out;
    case kI64: {
      need(rem, 8, "i64");
      std::int64_t v = 0;
      std::memcpy(&v, p, 8);
      p += 8;
      rem -= 8;
      out.kind = CNode::Int;
      out.i = v;
      return out;
    }
    case kF64: {
      need(rem, 8, "f64");
      double v = 0;
      std::memcpy(&v, p, 8);
      p += 8;
      rem -= 8;
      out.kind = CNode::Float;
      out.f = v;
      return out;
    }
    case kArr: {
      need(rem, 1, "array ndim");
      unsigned char ndim = *p++;
      rem--;
      need(rem, std::size_t{4} * ndim, "array shape");
      std::vector<std::uint32_t> shape(ndim);
      std::uint64_t count = 1;
      for (unsigned j = 0; j < ndim; ++j) {
        std::uint32_t s = 0;
        std::memcpy(&s, p, 4);
        p += 4;
        rem -= 4;
        shape[j] = s;
        count *= s;
      }
      need(rem, count * 8, "array data");
      std::vector<double> data(count);
      std::memcpy(data.data(), p, count * 8);
      p += count * 8;
      rem -= count * 8;
      out.kind = CNode::Tensor;
      out.shape = std::move(shape);
      out.tensor = std::move(data);
      return out;
    }
    case kStr: {
      need(rem, 2, "string length");
      std::uint16_t slen = 0;
      std::memcpy(&slen, p, 2);
      p += 2;
      rem -= 2;
      need(rem, slen, "string bytes");
      out.kind = CNode::Str;
      out.s.assign(reinterpret_cast<const char*>(p), slen);
      p += slen;
      rem -= slen;
      return out;
    }
    case kDict: {
      need(rem, 4, "dict count");
      std::uint32_t n_sub = 0;
      std::memcpy(&n_sub, p, 4);
      p += 4;
      rem -= 4;
      out.kind = CNode::Map;
      out.map.reserve(n_sub);
      for (std::uint32_t i = 0; i < n_sub; ++i) {
        need(rem, 2, "dict key length");
        std::uint16_t klen = 0;
        std::memcpy(&klen, p, 2);
        p += 2;
        rem -= 2;
        need(rem, klen, "dict key");
        std::string key(reinterpret_cast<const char*>(p), klen);
        p += klen;
        rem -= klen;
        out.map.emplace_back(std::move(key), read_value(p, rem));
      }
      return out;
    }
    default:
      throw std::runtime_error("Unknown dtype in Cypha binary: " + std::to_string(int(dtype)));
  }
}

}  // namespace

CNode load_cypha_from_buffer(const std::uint8_t* raw, std::size_t raw_size) {
  if (raw == nullptr) {
    throw std::runtime_error("load_cypha: null buffer");
  }
  if (raw_size < 6 + 9) {
    throw std::runtime_error("Cypha buffer too small");
  }
  static const char kMagic[] = "CYPHA\0";
  if (std::memcmp(raw, kMagic, 6) != 0) {
    throw std::runtime_error("Bad Cypha magic");
  }
  std::size_t off = 6;
  std::uint8_t version = raw[off++];
  std::uint32_t endian = 0;
  std::memcpy(&endian, raw + off, 4);
  off += 4;
  std::uint32_t n_fields = 0;
  std::memcpy(&n_fields, raw + off, 4);
  off += 4;
  if (version > 3) {
    throw std::runtime_error("Unsupported Cypha binary version");
  }
  if (version >= 3 && endian != 0x01020304u) {
    throw std::runtime_error("Endian mismatch in Cypha binary (expected little-endian sentinel)");
  }

  const std::uint8_t* p = raw + off;
  std::size_t rem = raw_size - off;

  CNode root;
  root.kind = CNode::Map;
  root.map.reserve(n_fields);
  for (std::uint32_t i = 0; i < n_fields; ++i) {
    need(rem, 2, "top-level key length");
    std::uint16_t klen = 0;
    std::memcpy(&klen, p, 2);
    p += 2;
    rem -= 2;
    need(rem, klen, "top-level key");
    std::string key(reinterpret_cast<const char*>(p), klen);
    p += klen;
    rem -= klen;
    root.map.emplace_back(std::move(key), read_value(p, rem));
  }
  if (rem != 0) {
    throw std::runtime_error("Trailing garbage in Cypha binary");
  }
  return root;
}

CNode load_cypha_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error(std::string("Cannot open ") + path);
  }
  std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_cypha_from_buffer(raw.data(), raw.size());
}

const CNode* map_get(const CNode& n, std::string_view key) {
  if (n.kind != CNode::Map) {
    return nullptr;
  }
  for (const auto& kv : n.map) {
    if (kv.first.size() == key.size() && kv.first.compare(0, key.size(), key) == 0) {
      return &kv.second;
    }
  }
  return nullptr;
}

const CNode& map_get_required(const CNode& n, std::string_view key) {
  const CNode* q = map_get(n, key);
  if (!q) {
    throw std::runtime_error(std::string("Missing required key: ") + std::string(key));
  }
  return *q;
}

}  // namespace cypha
