#include "cypha/cyphalm/component_dump.hpp"

#include <bit>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

namespace fs = std::filesystem;

// The arrays are the host's bytes; mixsim.py reads them little-endian.
static_assert(std::endian::native == std::endian::little, "component dumps are little-endian");

/// IEEE half, round to nearest even (subnormals, ±inf, NaN kept).
std::uint16_t to_half(float f) {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7fffffffu;
    if (x >= 0x7f800000u) return static_cast<std::uint16_t>(sign | (x > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (x >= 0x477ff000u) return static_cast<std::uint16_t>(sign | 0x7c00u);  // rounds past 65504
    if (x < 0x38800000u) {  // below 2^-14: a half subnormal (or 0)
        if (x < 0x33000000u) return static_cast<std::uint16_t>(sign);  // below 2^-25
        const std::uint32_t e = x >> 23;
        const std::uint32_t m = (x & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - e;  // 14 .. 24: units of 2^-24
        std::uint32_t h = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = (x >> 13) - (112u << 10);  // exponent bias 127 -> 15
    const std::uint32_t rem = x & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // a carry moves into the exponent
    return static_cast<std::uint16_t>(sign | h);
}

const char* numpy_name(ComponentDump::Dtype d) {
    return d == ComponentDump::Dtype::F16 ? "float16" : d == ComponentDump::Dtype::F64 ? "float64" : "float32";
}

std::ofstream open_array(const fs::path& p) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("--dump-components: cannot write " + p.string());
    return f;
}

}  // namespace

ComponentDump::Dtype ComponentDump::parse_dtype(const std::string& name) {
    if (name == "f16") return Dtype::F16;
    if (name == "f32") return Dtype::F32;
    if (name == "f64") return Dtype::F64;
    throw std::invalid_argument("dump dtype: expected f16, f32 or f64, got \"" + name + "\"");
}

ComponentDump::ComponentDump(const std::string& dir, const HpSequenceBackend& hp, Dtype dtype, bool learn_mix)
    : dir_(dir), dtype_(dtype) {
    if (hp.has_session_cache()) {
        throw std::runtime_error("--dump-components: the session cache stage is not replayed (mixsim.py); "
                                 "dump without it");
    }
    fs::create_directories(dir_);
    const fs::path d(dir_);
    models_ = hp.is_composite() ? 1 + hp.ensemble_size() : 1;
    experts_ = hp.neural_count();
    ig_ = hp.has_infinigram();
    truth_ = open_array(d / "truth.bin");
    models_f_ = open_array(d / "models.bin");
    served_ = open_array(d / "served.bin");
    if (experts_ > 0) neural_ = open_array(d / "neural.bin");
    if (ig_) {
        ig_n_ = open_array(d / "ig_n.bin");
        ig_total_ = open_array(d / "ig_total.bin");
        ig_count_ = open_array(d / "ig_count.bin");
    }
    // Settings and the start of every learned weight: the replay's inputs
    // besides the arrays.
    const MixingOptions mo = hp.mixing_options();
    const HpSequenceBackend::MixingState s = hp.mixing_state();
    meta_ = {{"format", "cyphalm_components"},
             {"version", 1},
             {"dtype", numpy_name(dtype_)},
             {"models", models_},
             {"neural", experts_},
             {"infinigram", ig_},
             {"learn_mix", learn_mix},
             {"ensemble_learning_rate", hp.ensemble_learning_rate()},
             {"infinigram_learning_rate", hp.infinigram_learning_rate()},
             {"neural_learning_rate", hp.neural_learning_rate()},
             {"final_temperature", mo.final_temperature},
             {"final_temperature_lr", mo.final_temperature_lr},
             {"infinigram_mode", infinigram_mode_name(mo.infinigram_mode)},
             {"neural_mix", neural_mix_name(mo.neural_mix)},
             {"ensemble_gate", mo.ensemble_gate},
             {"start",
              {{"ensemble", s.ensemble},
               {"infinigram", s.ig},
               {"neural", s.neural},
               {"neural_log", s.neural_log},
               {"neural_switch", s.neural_switch},
               {"gate", s.gate},
               {"final_temperature", s.final_temp}}}};
}

void ComponentDump::put_(std::ofstream& f, const double* v, std::size_t n) {
    const std::size_t width = dtype_ == Dtype::F16 ? 2 : dtype_ == Dtype::F32 ? 4 : 8;
    buf_.resize(n * width);
    char* o = buf_.data();
    for (std::size_t i = 0; i < n; ++i, o += width) {
        if (dtype_ == Dtype::F64) {
            std::memcpy(o, &v[i], 8);
        } else if (dtype_ == Dtype::F32) {
            const float x = static_cast<float>(v[i]);
            std::memcpy(o, &x, 4);
        } else {
            const std::uint16_t h = to_half(static_cast<float>(v[i]));
            std::memcpy(o, &h, 2);
        }
    }
    f.write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
}

void ComponentDump::write(const HpSequenceBackend& hp, std::uint8_t truth, const std::vector<double>& served) {
    const HpSequenceBackend::ScoredParts parts = hp.scored_parts();
    const bool plain = parts.models.empty();
    if (served.size() != 256 || (plain ? models_ != 1 : parts.models.size() != models_) ||
        parts.neural.size() != experts_ || (ig_ && parts.ig_longest == nullptr)) {
        throw std::runtime_error("--dump-components: the model's stages changed during the dump");
    }
    truth_.put(static_cast<char>(truth));
    if (plain) {
        put_(models_f_, served.data(), 256);
    } else {
        for (const auto* lp : parts.models) {
            if (lp->size() != 256) throw std::runtime_error("--dump-components: needs 256-byte distributions");
            put_(models_f_, lp->data(), 256);
        }
    }
    for (const auto* lp : parts.neural) put_(neural_, lp->data(), 256);
    if (ig_) {
        for (const InfiniGram::Result* r : {parts.ig_longest, parts.ig_reliable}) {
            const std::int32_t n = r->n;
            const std::uint64_t total = r->total;
            ig_n_.write(reinterpret_cast<const char*>(&n), sizeof(n));
            ig_total_.write(reinterpret_cast<const char*>(&total), sizeof(total));
            ig_count_.write(reinterpret_cast<const char*>(r->count.data()),
                            static_cast<std::streamsize>(sizeof(std::uint32_t) * r->count.size()));
        }
    }
    put_(served_, served.data(), 256);
    ++rows_;
}

void ComponentDump::finish(const nlohmann::json& extra) {
    for (std::ofstream* f : {&truth_, &models_f_, &neural_, &ig_n_, &ig_total_, &ig_count_, &served_}) {
        if (!f->is_open()) continue;
        f->close();
        if (!*f) throw std::runtime_error("--dump-components: write failed in " + dir_);
    }
    meta_["bytes"] = rows_;
    for (auto it = extra.begin(); it != extra.end(); ++it) meta_[it.key()] = it.value();
    std::ofstream m(fs::path(dir_) / "meta.json");
    m << meta_.dump(1) << "\n";
    if (!m) throw std::runtime_error("--dump-components: cannot write meta.json in " + dir_);
}

}  // namespace cypha::cyphalm
