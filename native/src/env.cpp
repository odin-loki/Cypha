#include "cypha/env.hpp"

#include <cstdlib>

namespace cypha {

std::optional<std::string> env_get(const char* name) {
#if defined(_MSC_VER)
  char* buf = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) != 0 || buf == nullptr) {
    return std::nullopt;
  }
  std::string value(buf);
  free(buf);
  return value;
#else
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return std::nullopt;
  }
  return std::string(v);
#endif
}

std::string env_get_or(const char* name, std::string_view fallback) {
  std::optional<std::string> v = env_get(name);
  if (v.has_value()) {
    return *v;
  }
  return std::string(fallback);
}

}  // namespace cypha
