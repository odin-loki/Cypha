#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cypha {

struct CNode {
  enum Kind { Nil, Bool, Int, Float, Str, Tensor, Map } kind{Nil};
  bool b{};
  std::int64_t i{};
  double f{};
  std::string s;
  std::vector<std::uint32_t> shape;
  std::vector<double> tensor;
  /// Key order matches Python dict serialization order.
  std::vector<std::pair<std::string, CNode>> map;
};

[[nodiscard]] CNode load_cypha_file(const char* path);

/// Parse v3 Cypha binary from memory (same layout as ``load_cypha_file`` / Python ``cypha_load_binary``).
[[nodiscard]] CNode load_cypha_from_buffer(const std::uint8_t* data, std::size_t len);

/// Deep copy (recursive). Used before patching a loaded root for save.
[[nodiscard]] CNode clone_cnode(const CNode& n);

/// Serialize root map to Cypha binary v3 bytes (same layout as Python ``cypha_save_binary``).
[[nodiscard]] std::vector<std::uint8_t> save_cypha_to_buffer(const CNode& root);

/// Write ``save_cypha_to_buffer(root)`` to ``path``.
void save_cypha_file(const char* path, const CNode& root);

const CNode* map_get(const CNode& n, std::string_view key);

const CNode& map_get_required(const CNode& n, std::string_view key);

}  // namespace cypha
