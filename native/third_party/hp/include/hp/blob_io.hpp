#pragma once
// hp/blob_io.hpp — binary I/O primitives for Cypha predictor checkpoints.

#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <type_traits>
#include <vector>

namespace hp::blob {

inline void write_pod(std::ostream& os, const auto& v) {
    os.write(reinterpret_cast<const char*>(&v), static_cast<std::streamsize>(sizeof(v)));
}

inline void read_pod(std::istream& is, auto& v) {
    is.read(reinterpret_cast<char*>(&v), static_cast<std::streamsize>(sizeof(v)));
}

template <typename T>
void write_vec(std::ostream& os, const std::vector<T>& v) {
    const std::uint64_t n = v.size();
    write_pod(os, n);
    if (n > 0) {
        os.write(reinterpret_cast<const char*>(v.data()),
                 static_cast<std::streamsize>(n * sizeof(T)));
    }
}

template <typename T>
void read_vec(std::istream& is, std::vector<T>& v) {
    std::uint64_t n = 0;
    read_pod(is, n);
    v.resize(n);
    if (n > 0) {
        is.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(T)));
    }
}

template <typename T, std::size_t N>
void write_array(std::ostream& os, const std::array<T, N>& a) {
    if (N > 0) {
        os.write(reinterpret_cast<const char*>(a.data()),
                 static_cast<std::streamsize>(N * sizeof(T)));
    }
}

template <typename T, std::size_t N>
void read_array(std::istream& is, std::array<T, N>& a) {
    if (N > 0) {
        is.read(reinterpret_cast<char*>(a.data()),
                static_cast<std::streamsize>(N * sizeof(T)));
    }
}

template <typename T>
void write_trivial_object(std::ostream& os, const T& o) {
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char*>(&o), static_cast<std::streamsize>(sizeof(T)));
}

template <typename T>
void read_trivial_object(std::istream& is, T& o) {
    static_assert(std::is_trivially_copyable_v<T>);
    is.read(reinterpret_cast<char*>(&o), static_cast<std::streamsize>(sizeof(T)));
}

}  // namespace hp::blob
