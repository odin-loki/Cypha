#pragma once
//
// hp/hash_table.hpp — eager zero-init hash tables for context/match slots.

#include <cstdint>
#include <vector>

namespace hp {

template <typename T>
class HashTable {
 public:
    explicit HashTable(int table_bits)
        : mask_((1u << table_bits) - 1),
          tab_(static_cast<std::size_t>(1) << table_bits, 0) {}

    T get(std::uint32_t idx) const { return tab_[idx & mask_]; }
    T& ref(std::uint32_t idx) { return tab_[idx & mask_]; }

 private:
    std::uint32_t mask_;
    std::vector<T> tab_;
};

}  // namespace hp
