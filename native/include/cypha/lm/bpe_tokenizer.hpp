#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm {

/// Inference-only BPE tokenizer: load merges.txt + vocab.json, encode/decode.
class BpeTokenizer {
public:
    static BpeTokenizer load(const std::string& merges_path, const std::string& vocab_path);

    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& ids) const;

    std::uint32_t vocab_size() const { return static_cast<std::uint32_t>(id_to_token_.size()); }

private:
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;

    static std::vector<std::string> tokenize_chars(const std::string& text);
    std::vector<std::string> apply_bpe(const std::vector<std::string>& tokens) const;
};

}  // namespace cypha::cyphalm
