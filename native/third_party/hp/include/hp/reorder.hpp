#pragma once
//
// hp/reorder.hpp — lossless page permutation (starlit / payload_lex analog).
//
// Split MediaWiki dump on <page>…</page>. Prefix and suffix stay put.
// Encoder sorts pages by a key, stores the permutation, compresses the
// concatenated stream. Decoder unpermutes. Nothing else is transmitted.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hp {

struct PageSpan {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::string key;
    std::uint32_t orig = 0;
};

inline bool page_tag_at(const std::uint8_t* p, std::size_t n, std::size_t i,
                        const char* tag) {
    const std::size_t m = std::strlen(tag);
    if (i + m > n) return false;
    return std::memcmp(p + i, tag, m) == 0;
}

inline std::string slice_between(const std::uint8_t* p, std::size_t a, std::size_t b,
                                 const char* open, const char* close) {
    const std::size_t ol = std::strlen(open);
    const std::size_t cl = std::strlen(close);
    std::size_t i = a;
    while (i + ol <= b) {
        if (std::memcmp(p + i, open, ol) == 0) {
            i += ol;
            if (open[ol - 1] != '>') {
                while (i < b && p[i] != '>') ++i;
                if (i < b) ++i;
            }
            const std::size_t s = i;
            while (i + cl <= b && std::memcmp(p + i, close, cl) != 0) ++i;
            std::string out(reinterpret_cast<const char*>(p + s), i - s);
            for (char& c : out) {
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            }
            return out;
        }
        ++i;
    }
    return {};
}

inline void split_pages(const std::uint8_t* p, std::size_t n,
                        std::size_t& prefix_end, std::size_t& suffix_begin,
                        std::vector<PageSpan>& pages) {
    pages.clear();
    prefix_end = 0;
    suffix_begin = n;
    std::size_t i = 0;
    while (i < n) {
        if (page_tag_at(p, n, i, "<page>")) {
            if (pages.empty()) prefix_end = i;
            const std::size_t begin = i;
            i += 6;
            while (i < n && !page_tag_at(p, n, i, "</page>")) ++i;
            if (i + 7 <= n) i += 7;
            PageSpan s;
            s.begin = begin;
            s.end = i;
            s.orig = static_cast<std::uint32_t>(pages.size());
            pages.push_back(s);
            suffix_begin = i;
        } else {
            ++i;
        }
    }
}

inline void concat_perm(const std::uint8_t* p,
                        std::size_t prefix_end, std::size_t suffix_begin,
                        std::size_t n, const std::vector<PageSpan>& ordered,
                        std::vector<std::uint8_t>& out) {
    out.clear();
    out.reserve(n);
    out.insert(out.end(), p, p + prefix_end);
    for (const PageSpan& s : ordered)
        out.insert(out.end(), p + s.begin, p + s.end);
    out.insert(out.end(), p + suffix_begin, p + n);
}

// kind 0 = title (article reorder). kind 1 = <text> payload (payload_lex analog).
inline void page_permute(const std::vector<std::uint8_t>& in,
                         std::vector<std::uint8_t>& out,
                         std::vector<std::uint32_t>& perm, int kind) {
    std::size_t prefix_end = 0, suffix_begin = in.size();
    std::vector<PageSpan> pages;
    split_pages(in.data(), in.size(), prefix_end, suffix_begin, pages);
    if (pages.size() < 2) {
        out = in;
        perm.clear();
        return;
    }
    for (PageSpan& s : pages) {
        if (kind == 1) {
            s.key = slice_between(in.data(), s.begin, s.end, "<text", "</text>");
            if (s.key.size() > 256) s.key.resize(256);
        } else {
            s.key = slice_between(in.data(), s.begin, s.end, "<title>", "</title>");
        }
    }
    std::sort(pages.begin(), pages.end(), [](const PageSpan& a, const PageSpan& b) {
        if (a.key != b.key) return a.key < b.key;
        return a.orig < b.orig;
    });
    perm.resize(pages.size());
    int changed = 0;
    for (std::size_t i = 0; i < pages.size(); ++i) {
        perm[i] = pages[i].orig;
        if (perm[i] != i) changed = 1;
    }
    if (!changed) {
        out = in;
        perm.clear();
        return;
    }
    concat_perm(in.data(), prefix_end, suffix_begin, in.size(), pages, out);
}

inline void page_unpermute(const std::vector<std::uint8_t>& body,
                           const std::vector<std::uint32_t>& perm,
                           std::vector<std::uint8_t>& out) {
    if (perm.empty()) {
        out = body;
        return;
    }
    std::size_t prefix_end = 0, suffix_begin = body.size();
    std::vector<PageSpan> pages;
    split_pages(body.data(), body.size(), prefix_end, suffix_begin, pages);
    if (pages.size() != perm.size()) {
        out = body;
        return;
    }
    std::vector<PageSpan> orig(pages.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        const std::uint32_t o = perm[i];
        if (o >= orig.size()) {
            out = body;
            return;
        }
        orig[o] = pages[i];
        orig[o].orig = o;
    }
    concat_perm(body.data(), prefix_end, suffix_begin, body.size(), orig, out);
}

}  // namespace hp
