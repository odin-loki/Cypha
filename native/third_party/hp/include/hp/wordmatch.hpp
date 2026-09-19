#pragma once
//
// hp/wordmatch.hpp — match models keyed on word context, not byte context.
//
// hp's byte-keyed match bank already occupies four of the six most
// decorrelated slots. A word-keyed variant reads a different axis: "the last
// time we were in this phrase, what byte came next", which fires on repeated
// *word* sequences whose byte suffixes have already diverged (inflected
// endings, different markup around the same tokens).
//
// order = how many completed words are hashed. Smaller tables than the byte
// match bank — this is a specialist.

#include <array>
#include <cstdint>
#include <vector>

#include "hp/models.hpp"

namespace hp {

class WordMatchModel {
 public:
    // ring_bits: history window for match distance; default uses ring->mask().
    // Word-match shares byte_ring_ with a shorter window than the byte models.
    WordMatchModel(ByteRing* ring, int table_bits, int word_order, int ring_bits = -1)
        : ring_(ring), order_(word_order < 1 ? 1 : word_order),
          tab_mask_((1u << table_bits) - 1),
          ring_mask_(ring_bits >= 0 ? ((1u << ring_bits) - 1) : ring->mask()),
          tab_(table_bits) {
        counter_init(st_.data(), st_.size());
    }

    void set_ring(ByteRing* ring) {
        ring_ = ring;
        if (ring != nullptr) {
            ring_mask_ = ring->mask();
        }
    }

    // Called once per byte after the shared ring has been updated.
    // whist is the rolling hash of the last `order_` completed words.
    void push_byte(int byte, std::uint64_t whist, int at_word_boundary) {
        const std::uint32_t pos = ring_->pos();
        if (len_ > 0) {
            if (ptr_ < pos && ring_->at(ptr_) == static_cast<std::uint8_t>(byte)) {
                if (len_ < 65535) ++len_;
                ++ptr_;
            } else {
                len_ = 0;
            }
        }

        if (at_word_boundary && whist != 0) {
            const std::uint32_t h =
                hash2(0x574D0000ull + static_cast<std::uint64_t>(order_), whist) &
                tab_mask_;
            if (len_ == 0) {
                const std::uint32_t cand = tab_.get(h);
                if (cand > 0 && cand < pos) {
                    ptr_ = cand;
                    len_ = 1;
                }
            }
            tab_.ref(h) = pos;
        }
        if (len_ > 0 && (pos - ptr_) > ring_mask_) len_ = 0;
    }

    int predict(int c0, int bitpos) {
        valid_ = false;
        const std::uint32_t pos = ring_->pos();
        if (len_ == 0 || ptr_ >= pos) return 0;
        const int pred_byte = ring_->at(ptr_);
        if (bitpos > 0) {
            if (((pred_byte | 0x100) >> (8 - bitpos)) != c0) {
                len_ = 0;
                return 0;
            }
        }
        expected_ = (pred_byte >> (7 - bitpos)) & 1;
        const int lq = len_ > 31 ? 31 : len_;
        sidx_ = lq * 2 + expected_;
        valid_ = true;
        return counter_predict(st_[sidx_]);
    }

    void update(int y) {
        if (valid_) counter_update(st_[sidx_], y, 255);
    }

    int match_len() const { return len_; }

 private:
    ByteRing* ring_;
    int order_;
    std::uint32_t tab_mask_;
    std::uint32_t ring_mask_;
    HashTable<std::uint32_t> tab_;
    std::array<Counter, 64> st_{};
    std::uint32_t ptr_ = 0;
    int len_ = 0, expected_ = 0, sidx_ = 0;
    bool valid_ = false;
};

}  // namespace hp
