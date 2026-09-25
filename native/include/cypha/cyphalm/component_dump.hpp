#pragma once

/// Per-byte inputs of a composite model's mixing stages, so the stages can be
/// replayed offline (``cyphalm_lm_quality --dump-components DIR``;
/// bench/lm_compare/mixsim.py reproduces the served NLL from them and screens
/// other mixing settings in seconds). Members read true bytes on their own,
/// index counts depend only on the context and the experts' output-layer
/// adaptation only on the truth, so everything after the models' own
/// distributions is a function of these arrays and the start state.
///
/// A directory of flat little-endian arrays, one row per scored byte, and
/// ``meta.json``:
///
///   truth.bin     uint8                   the byte that followed
///   models.bin    dtype [models][256]     log P before the ensemble mix: this model, then each member
///   neural.bin    dtype [experts][256]    each neural expert's log P, before it read the byte
///   ig_n.bin      int32 [2]               ∞-gram match length: longest, reliable (≥16) part
///   ig_total.bin  uint64 [2]              occurrences followed by a byte
///   ig_count.bin  uint32 [2][256]         next-byte counts
///   served.bin    dtype [256]             the served log P, after every stage
///
/// dtype is float16, float32 (default) or float64 (``meta.json`` "dtype",
/// a numpy name). The ∞-gram files exist only with an index, neural.bin only
/// with experts. ``meta.json`` holds the shapes, each stage's mode and
/// learning rate, whether mixing weights learned, every mixing weight at the
/// first byte (``HpSequenceBackend::mixing_state``) and the caller's keys.
/// A plain model (no mixing stage) dumps its served distribution as its one
/// model. The session cache is not replayed: a model with one is refused.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/hp_backend.hpp"

namespace cypha::cyphalm {

class ComponentDump {
 public:
    enum class Dtype { F16, F32, F64 };
    /// "f16", "f32" or "f64"; anything else throws std::invalid_argument.
    static Dtype parse_dtype(const std::string& name);

    /// Creates ``dir`` and the array files, and records ``hp``'s mixing
    /// settings and weights as they are now: construct it right before the
    /// first byte is scored. ``learn_mix``: whether the mixing weights learn
    /// from the bytes read (learning on and ``hp.mixing_learning()``).
    /// Throws std::runtime_error on a session cache or an unwritable file.
    ComponentDump(const std::string& dir, const HpSequenceBackend& hp, Dtype dtype, bool learn_mix);

    /// One row. Call after ``hp.next_byte_log_probs(256)`` returned
    /// ``served`` and before ``truth`` is read.
    void write(const HpSequenceBackend& hp, std::uint8_t truth, const std::vector<double>& served);

    /// Close the arrays and write meta.json with ``extra``'s keys added (the
    /// harness adds its eval block, whose ``nll_bits_per_byte`` the
    /// simulator checks itself against).
    void finish(const nlohmann::json& extra = nlohmann::json::object());

    std::size_t rows() const { return rows_; }

 private:
    void put_(std::ofstream& f, const double* v, std::size_t n);

    std::string dir_;
    Dtype dtype_;
    std::size_t models_ = 0, experts_ = 0, rows_ = 0;
    bool ig_ = false;
    nlohmann::json meta_;
    std::ofstream truth_, models_f_, neural_, ig_n_, ig_total_, ig_count_, served_;
    std::vector<char> buf_;
};

}  // namespace cypha::cyphalm
