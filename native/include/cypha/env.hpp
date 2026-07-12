#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace cypha {

/// Reads environment variable `name`. Returns std::nullopt if unset (or empty
/// on platforms where that distinction cannot be made). On MSVC this uses
/// `_dupenv_s` internally (freeing the temporary buffer before returning) to
/// avoid the C4996 deprecation warning on `std::getenv`; elsewhere it wraps
/// plain `std::getenv`. Behavior otherwise matches `std::getenv`.
std::optional<std::string> env_get(const char* name);

/// Convenience wrapper: returns the environment variable's value, or `fallback`
/// (as a std::string) if the variable is unset.
std::string env_get_or(const char* name, std::string_view fallback);

}  // namespace cypha
