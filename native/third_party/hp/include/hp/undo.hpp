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

    bool empty() const { return !has_snap_ && patches_.empty(); }

    void push_predictor(const Predictor& p, const Config& cfg);
    void pop_predictor(Predictor& p) const;

    template <typename T>
    void note(T& cell) {
        note_bytes(&cell, static_cast<int>(sizeof(T)));
    }

    void note_bytes(void* addr, int size) {
        for (const Patch& p : patches_) {
            if (p.addr == addr) {
                return;
            }
        }
        Patch patch{};
        patch.addr = addr;
        patch.size = size;
        std::memset(patch.old_bytes, 0, sizeof(patch.old_bytes));
        std::memcpy(patch.old_bytes, addr, static_cast<std::size_t>(size));
        patches_.push_back(patch);
    }

    void restore_patches() const {
        for (std::size_t i = patches_.size(); i-- > 0;) {
            const Patch& p = patches_[i];
            std::memcpy(p.addr, p.old_bytes, static_cast<std::size_t>(p.size));
        }
    }

 private:
    struct Patch {
        void* addr;
        int size;
        alignas(8) unsigned char old_bytes[8];
    };

    std::vector<Patch> patches_;
    std::unique_ptr<Predictor> snap_;
    bool has_snap_ = false;
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
    static inline UndoFrame* active_ = nullptr;
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

    std::size_t depth() const { return frames_.size(); }

 private:
    std::vector<UndoFrame> frames_;
};

}  // namespace hp
