#include "cypha/cyphalm/bpe_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

namespace {

std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

}  // namespace

BpeTokenizer BpeTokenizer::load(const std::string& merges_path, const std::string& vocab_path) {
    BpeTokenizer tok;
    {
        std::ifstream vf(vocab_path);
        if (!vf) throw std::runtime_error("cannot open vocab: " + vocab_path);
        nlohmann::json j;
        vf >> j;
        if (!j.is_object()) throw std::runtime_error("vocab.json must be an object");
        std::size_t max_id = 0;
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::uint32_t id = it.value().get<std::uint32_t>();
            tok.token_to_id_[it.key()] = id;
            max_id = std::max(max_id, static_cast<std::size_t>(id));
        }
        tok.id_to_token_.assign(max_id + 1, "");
        for (const auto& kv : tok.token_to_id_) {
            if (kv.second < tok.id_to_token_.size()) tok.id_to_token_[kv.second] = kv.first;
        }
    }
    {
        std::ifstream mf(merges_path);
        if (!mf) throw std::runtime_error("cannot open merges: " + merges_path);
        std::string line;
        while (std::getline(mf, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string a, b;
            if (!(iss >> a >> b)) continue;
            tok.merges_.emplace_back(a, b);
        }
    }
    return tok;
}

std::vector<std::string> BpeTokenizer::tokenize_chars(const std::string& text) {
    std::vector<std::string> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        out.push_back(std::string(1, static_cast<char>(c)));
    }
    return out;
}

std::vector<std::string> BpeTokenizer::apply_bpe(const std::vector<std::string>& tokens) const {
    std::vector<std::string> words = tokens;
    for (const auto& merge : merges_) {
        std::vector<std::string> next;
        next.reserve(words.size());
        for (std::size_t i = 0; i < words.size();) {
            if (i + 1 < words.size() && words[i] == merge.first && words[i + 1] == merge.second) {
                next.push_back(words[i] + words[i + 1]);
                i += 2;
            } else {
                next.push_back(words[i]);
                ++i;
            }
        }
        words.swap(next);
    }
    return words;
}

std::vector<std::uint32_t> BpeTokenizer::encode(const std::string& text) const {
    const std::vector<std::string> chars = tokenize_chars(text);
    const std::vector<std::string> merged = apply_bpe(chars);
    std::vector<std::uint32_t> ids;
    ids.reserve(merged.size());
    for (const std::string& tok : merged) {
        auto it = token_to_id_.find(tok);
        if (it == token_to_id_.end()) {
            auto unk = token_to_id_.find("<unk>");
            if (unk == token_to_id_.end()) ids.push_back(0);
            else ids.push_back(unk->second);
        } else {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string out;
    for (std::uint32_t id : ids) {
        if (id < id_to_token_.size() && !id_to_token_[id].empty()) out += id_to_token_[id];
    }
    return out;
}

}  // namespace cypha::cyphalm
