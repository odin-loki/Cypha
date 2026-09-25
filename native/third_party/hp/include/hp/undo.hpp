#pragma once
//
// hp/undo.hpp — checkpoint undo for hp::Predictor speculative bit updates.
//
// Default backtrack: ``UndoRecorderScope`` + ``restore_patches`` (delta undo).
// ``push_predictor`` / ``pop_predictor`` retain optional full snapshots for tests.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace hp {

struct Config;
class Predictor;

class UndoFrame {
 public:
    UndoFrame() = default;
    UndoFrame(UndoFrame&&) = default;
    UndoFrame& operator=(UndoFrame&&) = default;
    UndoFrame(const UndoFrame&) = delete;
    UndoFrame& operator=(const UndoFrame&) = delete;
    ~UndoFrame();

    void clear();

    bool empty() const { return !has_snap_ && patches_.empty() && sizes_.empty(); }

    /// A frame that also records byte boundaries (StreamRewind). Bit-tree
    /// frames leave it off: the predictor then skips end-of-byte work.
    void set_records_byte_end(bool on) { byte_end_ = on; }
    bool records_byte_end() const { return byte_end_; }

    void push_predictor(const Predictor& p, const Config& cfg);
    void pop_predictor(Predictor& p) const;

    template <typename T>
    void note(T& cell) {
        static_assert(sizeof(T) <= 8, "UndoFrame patches hold at most 8 bytes");
        note_bytes(&cell, static_cast<int>(sizeof(T)));
    }

    // Duplicates are kept: restore_patches() replays newest-first, so the
    // oldest saved bytes for an address are written last. Skipping the old
    // linear dedupe scan makes a frame O(n) instead of O(n^2).
    void note_bytes(void* addr, int size) {
        Patch patch{};
        patch.addr = addr;
        patch.size = size;
        std::memset(patch.old_bytes, 0, sizeof(patch.old_bytes));
        std::memcpy(patch.old_bytes, addr, static_cast<std::size_t>(size));
        patches_.push_back(patch);
    }

    // Record a container's size so elements appended under this frame are
    // dropped on restore. The container must not reallocate while the frame is
    // live (reserve its capacity), since cells inside it may also be patched.
    template <typename V>
    void note_size(V& v) {
        sizes_.push_back(SizeRec{&v, v.size(), [](void* c, std::size_t n) {
                                     static_cast<V*>(c)->resize(n);
                                 }});
    }

    void restore_patches() const {
        for (std::size_t i = patches_.size(); i-- > 0;) {
            const Patch& p = patches_[i];
            std::memcpy(p.addr, p.old_bytes, static_cast<std::size_t>(p.size));
        }
        for (std::size_t i = sizes_.size(); i-- > 0;) {
            sizes_[i].truncate(sizes_[i].container, sizes_[i].size);
        }
    }

 private:
    struct Patch {
        void* addr;
        int size;
        alignas(8) unsigned char old_bytes[8];
    };

    struct SizeRec {
        void* container;
        std::size_t size;
        void (*truncate)(void*, std::size_t);
    };

    std::vector<Patch> patches_;
    std::vector<SizeRec> sizes_;
    std::unique_ptr<Predictor> snap_;
    bool has_snap_ = false;
    bool byte_end_ = false;
};

class UndoRecorderScope {
 public:
    explicit UndoRecorderScope(UndoFrame& frame) : prev_(active_) {
        frame.clear();
        active_ = &frame;
    }
    ~UndoRecorderScope() { active_ = prev_; }

    UndoRecorderScope(const UndoRecorderScope&) = delete;
    UndoRecorderScope& operator=(const UndoRecorderScope&) = delete;

    static UndoFrame* active() { return active_; }

 private:
    UndoFrame* prev_;
    // Per thread: ensemble members score on worker threads, each with its own frames.
    static inline thread_local UndoFrame* active_ = nullptr;
};

inline void hp_undo_note(std::uint8_t& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void hp_undo_note(std::uint16_t& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void hp_undo_note(std::uint32_t& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void hp_undo_note(std::uint64_t& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void hp_undo_note(bool& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void hp_undo_note(int& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

template <typename V>
inline void hp_undo_note_size(V& container) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note_size(container);
    }
}

inline void hp_undo_note(std::int16_t& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

class PredictorUndoStack {
 public:
    void clear();

    UndoFrame& push_frame();

    void pop_frame(Predictor& pred);

    std::size_t depth() const { return depth_; }

 private:
    static constexpr std::size_t kReserve = 64;
    // frames_[0..depth_) are live. Popped frames stay allocated and the next
    // push reuses them, so a scorer that keeps its stack across calls does
    // not reallocate patch buffers at every tree node.
    std::vector<UndoFrame> frames_;
    std::size_t depth_ = 0;
};

}  // namespace hp
