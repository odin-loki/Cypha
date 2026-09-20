#pragma once
//
// hp/hash_table.hpp — eager zero-init hash tables for context/match slots.

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "hp/blob_io.hpp"

namespace hp {

template <typename T>
class HashTable {
 public:
    explicit HashTable(int table_bits)
        : mask_((1u << table_bits) - 1),
          tab_(static_cast<std::size_t>(1) << table_bits, 0) {}

    T get(std::uint32_t idx) const { return tab_[idx & mask_]; }
    T& ref(std::uint32_t idx) { return tab_[idx & mask_]; }
    T& at(std::size_t i) { return tab_[i]; }
    const T& at(std::size_t i) const { return tab_[i]; }
    std::size_t size() const { return tab_.size(); }
    const T* data() const { return tab_.data(); }
    T* data() { return tab_.data(); }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        blob::write_vec(os, tab_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        blob::read_vec(is, tab_);
    }

 private:
    std::uint32_t mask_;
    std::vector<T> tab_;
};

}  // namespace hp
