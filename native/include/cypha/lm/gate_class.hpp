#pragma once

/// O3 (GENERATION_PLAN §13.41; docs/reports/OPT_BOOLEAN.md): small Boolean functions of one byte that run in loops, as
/// tables built at compile time from the code they replace. tools/lm/opt_boolean_smoke.cpp checks each one against
/// that code on all 256 bytes.
///
/// - gate_class: the mixers' class of the previous byte, 0 space, 1 letter, 2 digit, 3 markup ([]{}<>|&=), 4 other, as
///       std::isspace(b) ? 0 : std::isalpha(b) ? 1 : std::isdigit(b) ? 2 : std::strchr("[]{}<>|&=", b) && b ? 3 : 4
///   gives it in the C locale (nothing in Cypha calls setlocale). The chain is up to four library calls; its minimal
///   sum of products is 28 products and 140 literals over the byte's 8 bits (scripts/bool_min.py cypha). The table is
///   one load.
/// - is_gate_markup: std::string_view("[]{}|=<>&*#").find(c) != npos (the sentence gates' markup count), a 256-bit set.
/// - is_sentence_end: t == "." || t == "!" || t == "?" || t == "\n" (the sentence gates' stop), one length test and the
///   same kind of set.
///
/// Author: Odin Loch

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cypha::cyphalm {

inline constexpr std::array<std::uint8_t, 256> kGateClass = [] {
    std::array<std::uint8_t, 256> t{};
    for (int c = 0; c < 256; ++c) {
        const bool sp = c == ' ' || (c >= '\t' && c <= '\r');               // isspace, C locale
        const bool al = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');  // isalpha
        const bool dg = c >= '0' && c <= '9';                              // isdigit
        bool mk = false;                                                   // strchr(..., c) && c
        for (const char x : std::string_view("[]{}<>|&=")) mk = mk || c == static_cast<unsigned char>(x);
        t[static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(sp ? 0 : al ? 1 : dg ? 2 : mk ? 3 : 4);
    }
    return t;
}();

inline int gate_class(unsigned char b) { return kGateClass[b]; }

// A set of bytes as four 64-bit words.
inline constexpr std::array<std::uint64_t, 4> byte_set(std::string_view s) {
    std::array<std::uint64_t, 4> m{};
    for (const char x : s) {
        const auto u = static_cast<unsigned char>(x);
        m[u >> 6] |= std::uint64_t{1} << (u & 63);
    }
    return m;
}
inline constexpr bool in_set(const std::array<std::uint64_t, 4>& m, char c) {
    const auto u = static_cast<unsigned char>(c);
    return (m[u >> 6] >> (u & 63)) & 1;
}

inline constexpr std::array<std::uint64_t, 4> kGateMarkup = byte_set("[]{}|=<>&*#");
inline bool is_gate_markup(char c) { return in_set(kGateMarkup, c); }

inline constexpr std::array<std::uint64_t, 4> kSentenceEnd = byte_set(".!?\n");
inline bool is_sentence_end(std::string_view t) { return t.size() == 1 && in_set(kSentenceEnd, t[0]); }

}  // namespace cypha::cyphalm
