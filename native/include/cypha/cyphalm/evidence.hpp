#pragma once

/// Evidence cards (S1, docs/reports/S1_REASON_CODES_AND_EVIDENCE.md): where in CyphaLM's training text an answer's
/// bytes, or a question's entity, were seen. A card is made after the answer, from an FM-index with locate
/// (fm_infinigram.hpp). Nothing here feeds an answer, a gate or a tag.

#include "cypha/cyphalm/fm_infinigram.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace cypha::cyphalm::evidence {

/// Where each part of a training text comes from: sorted start offsets, each with a label (a wiki page, a chat turn).
struct Sources {
    std::vector<std::size_t> at;
    std::vector<std::string> label;

    void add(std::size_t offset, std::string l) {
        at.push_back(offset);
        label.push_back(std::move(l));
    }
    /// The pages of enwik8 XML text[0..n), placed at base: "wiki:<title>" from each "<title>".
    void add_pages(const std::uint8_t* text, std::size_t n, std::size_t base = 0) {
        const char* s = reinterpret_cast<const char*>(text);
        const std::string_view v(s, n);
        for (std::size_t p = v.find("<title>"); p != std::string_view::npos; p = v.find("<title>", p + 7)) {
            const std::size_t e = v.find("</title>", p);
            if (e == std::string_view::npos) break;
            add(base + p, "wiki:" + std::string(v.substr(p + 7, e - p - 7)));
        }
    }
    /// The turns of a chat text ("User: " at its start or after a blank line), placed at base: "<kind>:<question
    /// line>", kind from kind_of(turn text up to its first blank line).
    template <class F>
    void add_turns(const std::string& chat, std::size_t base, F kind_of) {
        for (std::size_t p = 0; p < chat.size();) {
            if (chat.compare(p, 6, "User: ") == 0) {
                const std::size_t nl = chat.find('\n', p), end = chat.find("\n\n", p);
                const std::string line = chat.substr(p, std::min<std::size_t>(nl == std::string::npos ? chat.size() : nl, p + 90) - p);
                add(base + p, kind_of(chat.substr(p, (end == std::string::npos ? chat.size() : end) - p)) + ":" + line);
            }
            const std::size_t next = chat.find("\n\nUser: ", p);
            if (next == std::string::npos) break;
            p = next + 2;
        }
    }
    /// "label+offset" for text position p.
    std::string where(std::size_t p) const {
        const auto it = std::upper_bound(at.begin(), at.end(), p);
        if (it == at.begin()) return "start+" + std::to_string(p);
        const std::size_t k = static_cast<std::size_t>(it - at.begin()) - 1;
        return label[k] + "+" + std::to_string(p - at[k]);
    }
    std::size_t bytes() const {
        std::size_t b = at.size() * sizeof(std::size_t);
        for (const auto& l : label) b += sizeof(std::string) + l.size();
        return b;
    }
};

/// A match in the training text: its length, how often it occurs, and up to a few of its places.
struct Found {
    std::size_t n = 0, count = 0;
    std::vector<std::string> places;
};

inline std::vector<std::string> places(const FmInfiniGram& ix, const Sources& src, FmInfiniGram::Rows r,
                                       std::size_t most) {
    std::vector<std::size_t> pos;
    for (std::size_t row = r.lo; row < r.hi && pos.size() < most; ++row) pos.push_back(ix.locate(row));
    std::sort(pos.begin(), pos.end());
    std::vector<std::string> out;
    for (std::size_t p : pos) out.push_back(src.where(p));
    return out;
}

/// The longest suffix of s[0..len) (at most cap bytes) found in the index, its count and up to `most` places
/// (where the match starts).
inline Found longest(const FmInfiniGram& ix, const Sources& src, const std::string& s, std::size_t cap,
                     std::size_t most = 3) {
    Found f;
    const auto r = ix.longest_suffix(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), cap, f.n);
    f.count = r.hi - r.lo;
    f.places = places(ix, src, r, most);
    return f;
}

/// Where p occurs in the index: its count and up to `most` places (n = p's length when it occurs).
inline Found occurrences(const FmInfiniGram& ix, const Sources& src, const std::string& p, std::size_t most = 3) {
    Found f;
    if (p.empty()) return f;
    const auto r = ix.rows(reinterpret_cast<const std::uint8_t*>(p.data()), p.size());
    f.count = r.hi - r.lo;
    f.n = f.count ? p.size() : 0;
    f.places = places(ix, src, r, most);
    return f;
}

}  // namespace cypha::cyphalm::evidence
