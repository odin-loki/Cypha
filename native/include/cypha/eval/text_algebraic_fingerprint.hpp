#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// MS2: algebraic fingerprint vector on text (generated or scored).
struct TextAlgebraicFingerprint {
    double linear_complexity = 0.0;
    double spectral_flatness = 0.0;
    double run_length_entropy = 0.0;
    double ngram_entropy_1 = 0.0;
    double ngram_entropy_2 = 0.0;
    double ngram_entropy_3 = 0.0;

    std::vector<double> as_vector() const;
    /// Optional scalar: centroid of normalized features in [0, 1].
    double spectrum_position() const;
};

TextAlgebraicFingerprint compute_text_algebraic_fingerprint(const std::string& text);

nlohmann::json fingerprint_to_json(const TextAlgebraicFingerprint& fp, bool include_spectrum_position = true);

}  // namespace cypha::cyphalm
