// F2: relation patterns learned from text by distant supervision, then used to grow the fact store.
//
// Reads enwik8 (2006 Wikipedia XML: 12,346 pages, 3,664 redirects, 8,587 main-namespace articles) and runs:
//   1 SEEDS     (subject, relation, object) from `| key = value` fields of non-citation templates (infoboxes, persondata,
//               succession boxes) mapped to 32 relations (31 have seeds), from wiki-table rows `| Capital || [[X]]`, and, because 2006 infoboxes
//               are sparse, from page categories (`1905 births`, `Cities in Y`, `Natives of Z`, `American novelists`, `Novels by W`).
//               Values are cleaned (links, refs, templates); dates are parsed to year/month/day.
//   2 PATTERNS  For each seed, every prose sentence of the subject's own article (markup cleaned; templates, tables, lists dropped)
//               that holds the subject (title, lead bold form, surname of a person page, or a sentence-initial he/she/it/his/her/its,
//               written [P]) and the object (a link, capitalised chunk, date or lexicon n-gram equal to the seed object after
//               normalisation, and passing the relation's type check). Pattern = the tokens between the two mentions (1..6, digits
//               -> #, month names -> <M>) with slot order, in four variants (core, +1 token right, +1 left, both):
//               `[S] was born in [O]`, `^ [S] ( born [O] )`, `[O] , the capital of [S]`. Counted by distinct training subjects.
//   3 RELIABILITY  Seed subjects are split 80/20 by hash. Candidate patterns (>= 2 training subjects) are applied to the held-out
//               subjects' articles; an extraction is a check when the subject has a seed for that relation, and is correct when it
//               equals a seed object (a place also counts when a located_in seed puts it inside the seed place). Kept: >= 20 checks and
//               Wilson 95% lower bound >= 0.7 (ecs.hpp pattern_admissible). Reliability = mean of Beta(1+correct, 1+wrong).
//               Type constraints: dates must parse (year <= 2006); capital/largest_city/country objects must be links to place-like
//               pages; person relations need person-like pages or 2-4 capitalised words; lexicon relations need a seed value;
//               birth/death relations need a person page, capital/language/currency a country/state page.
//   4 EXTRACTION  Kept patterns are applied to every article; facts that are not seeds become ECS evidence items: one per
//               (sentence, pattern family), mass = omega * tau * pi with pi = pattern reliability, tau = 20/21 (infobox prior), omega =
//               (1 + ln n)/n for repeats of the same sentence text. Other values of the same (subject, relation) and conflicting seeds
//               oppose it with weight phi (relation functionality); impossible lifespans are hard violations. ecs::fuse gives the
//               opinion and grade. Base rate a per relation = pooled held-out precision of its kept patterns (0.5..0.95); --base-rate
//               0.5 gives the PLAN default. P >= 0.8 is admitted, 0.5..0.8 a candidate; only those are written.
//   Never reads bench/eval/t_qa_wiki.tsv.
//
// Measured 2026-10-27 (enwik8, one core, 16 s, peak RAM 107 MB; 5-fold --cv run):
//   8,148 seeds on 2,177 subject pages (infobox/template 3,286, table 16, category 4,846) over 32 relations;
//   676 candidate patterns, 44 kept (birth_year 14, death_year 16, death_date 4, birth_date 3, birth_place 3, language 4);
//   no kept pattern for capital, author, director, spouse, genre, ... (fewer than 20 checks; seeds too few).
//   920 new facts with P >= 0.5: 832 admitted (birth_date 455, death_date 308, birth_place 37, birth_year 20, death_year 11,
//   language 1) and 88 candidates; 919 have one evidence sentence, so none reaches grade Moderate.
//   Hand check of 50 random admitted facts: 49 right, 1 unverifiable; Wilson 95% [0.927, 1.000] (49/49) or [0.895, 0.996] (49/50).
//   Without --cv (a single 80/20 split) only 7 birth_year and 4 death_year patterns pass, and 17 facts are admitted.
//   Report: docs/reports/F2_PATTERN_LEARNING.md
//
// D4 (2026-10-28, docs/reports/D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md):
//   Seed fix: a template opened mid-line ("* {{Book reference |") is named, so its fields are no longer read under a stale
//   name; this drops 664 citation seeds (all 101 infobox "author" seeds, 408 publisher, 155 location): 7,484 text seeds.
//   --wikidata FILE (scripts/wikidata_seeds.py): Wikidata values are distant supervision only. Four fifths of the subjects (by
//   hash) get them as extra seeds, which choose and check patterns; they never become facts, never vote in ECS, and an
//   extraction equal to one still enters the store (the text states it). The other fifth is gold: it only scores the new facts
//   (facts_wd_check.tsv). --wikidata-gold-only scores a run without Wikidata seeds. --phi FILE overrides the hand-set phi.
//   --cv with Wikidata: 70 kept patterns (death_place is new), 1,216 admitted facts (store +16.2%); gold check 173/182.
//   --shard FILE --dev DEV --test TEST: the fact store (text seeds + admitted facts) as CyphaLM training text: three question
//   phrasings per fact (cyphalm_trie dktext's template first) with a one-sentence answer, and FILE_sentences.txt. Only pages
//   before --slice-end (8,388,608, the T-QA slice); rows from the BPC holdout get source *_holdout. Questions equal to a dev
//   or test question are dropped and the written rows re-checked (exit 1 on a match); test answers are never read. Prints the
//   dev coverage (reads dev answers only).
//
// F2R (2026-10-28, docs/reports/F2R_RAW_BYTE_PATTERNS.md):
//   --raw: the same seeds, mentions, type checks and gate, but a pattern is the dump's own bytes between the two mentions
//   (XML escapes and markup kept, digit runs -> #, month names -> <M>), each mention widened over the link or quote markup
//   that wraps it and kept around its slot: '''[S]''' ([[[O]]], [S] was born in [[[O]]]. A mention that cuts a link is not
//   used. patterns.tsv gets a 12th column, frame (the key without ^ / $, [P] as [S]): fill [S], and the bytes before [O]
//   are the prefix CyphaLM continues. --cv with Wikidata: 81 kept, 1,025 admitted (cleaned 1,216), gold 151/159; 41 facts
//   the cleaned run lacks (hand check 30/30); no kept pattern for capital, author, director, composer or founder.
//   --join FILE (with --shard): another run's admitted facts join the store. In raw mode the shard writes a fact of a
//   relation with a raw frame in that frame's bytes ("'''Al Gore''' ([[March 31]], [[1948]]).").
//   Every mode writes a year fact as a year (2026-10-28; before, the default shard wrote 6 with a full date).
//   --reach PARSE (with --dev; grading only): how often the frames' prefixes, with a dev question's subject filled, occur in
//   the training bytes [0, --holdout-from), and with --shard also in the shard text.
//
// S3 (2026-10-28, docs/reports/S3_S5_GAP_MAP_AND_HYPOTHESES.md):
//   --gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F] [--demand-weight W]: the gap map (DISCOVERY 3.2) of the store after the run:
//   schema gaps with P(exists) from f(T, R), contradiction gaps from the ECS fusion, certainty gaps, and demand from
//   CyphaLM's own <|unknown ...|> tags (dev rows only), each with VOI (gap_map.hpp). Writes FILE, FILE.cells.tsv and
//   FILE.demand.tsv. With --dev it grades whether each tag names its question's slot. Bookkeeping only: it answers nothing.
//
// Build: g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
// Usage: pattern_learn <enwik8> <outdir> [--cv] [--folds K] [--min-train N] [--base-rate A] [--sample N] [--seed S]
//          [--wikidata FILE [--wikidata-gold-only]] [--phi FILE] [--shard FILE --dev DEV --test TEST [--slice-end B] [--holdout-from B]]
//          [--raw] [--join FACTS] [--reach PARSE] [--gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F] [--demand-weight W]]
//        pattern_learn --wilson K N
//   <outdir> gets facts_f2.tsv (subject, relation, object, P, grade, n_evidence, best_pattern, source_page, byte_offset;
//   byte_offset is the start of the line holding the strongest evidence), facts_f2.evidence.tsv (the sentence for each fact),
//   seeds.tsv, patterns.tsv (kept first), pages.tsv (titles and text offsets), sample_admitted.tsv (random admitted facts,
//   fixed seed, to hand-judge) and stats.txt.
//   Default is one 80/20 split (fold 0 of K = 5); --cv repeats it over all K folds and pools the checks.
//
// Author: Odin Loch

#include "cypha/eval/ecs.hpp"
#include "cypha/eval/gap_map.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ecs = cypha::eval;
using std::string;
using std::vector;

namespace {

constexpr std::size_t npos = std::string::npos;

// ============================================================ strings
string lower(string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
string trim(const string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}
bool starts_with(const string& s, const char* p) { return s.compare(0, std::strlen(p), p) == 0; }
void replace_all(string& s, const string& a, const string& b) {
    if (a.empty()) return;
    std::size_t p = 0;
    while ((p = s.find(a, p)) != npos) { s.replace(p, a.size(), b); p += b.size(); }
}
// Latin-1 supplement letters (UTF-8 C3 xx) to ASCII; other bytes pass through.
string fold_ascii(const string& s) {
    static const char* T = "aaaaaaaceeeeiiiidnooooo?ouuuuyty";  // U+00E0..U+00FF
    string o;
    o.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0xC3 && i + 1 < s.size()) {
            unsigned char d = static_cast<unsigned char>(s[i + 1]);
            if (d >= 0x80 && d <= 0xBF) {
                char x = T[d & 0x1F];
                if (d < 0xA0) x = static_cast<char>(std::toupper(static_cast<unsigned char>(x)));
                o += x;
                ++i;
                continue;
            }
        }
        o += static_cast<char>(c);
    }
    return o;
}
// Normal form of a question: folded, lower case, alnum words (punctuation stripped).
string norm_q(const string& in) {
    string o;
    bool sp = true;
    for (unsigned char c : fold_ascii(in)) {
        if (std::isalnum(c) || c >= 0x80) { o += static_cast<char>(std::tolower(c)); sp = false; }
        else if (!sp) { o += ' '; sp = true; }
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}
// Normal form of an entity string for matching: norm_q with no "(qualifier)" and no leading "the".
string norm_ent(const string& in) {
    std::size_t p = in.find(" (");
    string o = norm_q(p != npos && p > 0 ? in.substr(0, p) : in);
    if (o.compare(0, 4, "the ") == 0) o.erase(0, 4);
    return o;
}
vector<string> split(const string& s, char d) {
    vector<string> out;
    for (std::size_t a = 0;;) {
        std::size_t b = s.find(d, a);
        out.push_back(s.substr(a, b == npos ? npos : b - a));
        if (b == npos) return out;
        a = b + 1;
    }
}
std::uint64_t fnv(const string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// ============================================================ wiki markup cleaning
string xml_decode(const string& s) {
    string o;
    o.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '&') {
            if (s.compare(i, 5, "&amp;") == 0) { o += '&'; i += 5; continue; }
            if (s.compare(i, 4, "&lt;") == 0) { o += '<'; i += 4; continue; }
            if (s.compare(i, 4, "&gt;") == 0) { o += '>'; i += 4; continue; }
            if (s.compare(i, 6, "&quot;") == 0) { o += '"'; i += 6; continue; }
        }
        o += s[i++];
    }
    return o;
}
string fix_entities(string s) {
    for (const char* d : {"&ndash;", "&mdash;", "\xe2\x80\x93", "\xe2\x80\x94", "&#8211;", "&#8212;"}) replace_all(s, d, "-");
    for (const char* d : {"&nbsp;", "\xc2\xa0"}) replace_all(s, d, " ");
    replace_all(s, "\xe2\x80\x99", "'");
    replace_all(s, "\xe2\x80\x98", "'");
    replace_all(s, "\xe2\x80\x9c", "\"");
    replace_all(s, "\xe2\x80\x9d", "\"");
    return s;
}
void erase_span(string& s, const string& open, const string& close) {
    std::size_t p = 0;
    while ((p = s.find(open, p)) != npos) {
        std::size_t e = s.find(close, p + open.size());
        if (e == npos) { s.erase(p); break; }
        s.erase(p, e + close.size() - p);
    }
}
void erase_refs(string& s) {
    erase_span(s, "<!--", "-->");
    std::size_t p = 0;
    while ((p = s.find("<ref", p)) != npos) {
        char c = p + 4 < s.size() ? s[p + 4] : '>';
        if (c != ' ' && c != '>' && c != '/') { p += 4; continue; }
        std::size_t gt = s.find('>', p);
        if (gt == npos) { s.erase(p); break; }
        if (s[gt - 1] == '/') { s.erase(p, gt + 1 - p); continue; }
        std::size_t e = s.find("</ref>", gt);
        if (e == npos) { s.erase(p); break; }
        s.erase(p, e + 6 - p);
    }
}
string strip_tags(const string& s, const char* sep = " ") {
    string o;
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '<') {
            std::size_t gt = s.find('>', i);
            if (gt != npos && gt - i < 200) { o += sep; i = gt + 1; continue; }
        }
        o += s[i++];
    }
    return o;
}
void remove_templates(string& s) {
    for (int guard = 0; guard < 80; ++guard) {
        std::size_t e = s.find("}}");
        if (e == npos) break;
        std::size_t b = s.rfind("{{", e);
        if (b == npos) { s.erase(e, 2); continue; }
        s.erase(b, e + 2 - b);
    }
    std::size_t p = s.find("{{");
    if (p != npos) s.erase(p);
}
// Remove [[Image:..]], [[Category:..]], [[xx:interwiki]] with nested links inside captions.
void remove_namespace_links(string& s) {
    std::size_t p = 0;
    while ((p = s.find("[[", p)) != npos) {
        std::size_t q = p + 2;
        std::size_t colon = s.find(':', q), close = s.find("]]", q), nest = s.find("[[", q);
        bool ns = colon != npos && colon - q <= 14 && (close == npos || colon < close) && (nest == npos || colon < nest) &&
                  colon + 1 < s.size() && s[colon + 1] != ' ' && colon > q;
        if (!ns) { p += 2; continue; }
        int depth = 0;
        std::size_t i = p;
        bool done = false;
        for (; i + 1 < s.size(); ++i) {
            if (s[i] == '[' && s[i + 1] == '[') { ++depth; ++i; }
            else if (s[i] == ']' && s[i + 1] == ']') { --depth; ++i; if (depth == 0) { ++i; done = true; break; } }
        }
        if (!done) i = s.size();
        s.erase(p, i - p);
    }
}
void external_links(string& s) {
    for (const char* pre : {"[http", "[ftp"}) {
        std::size_t p = 0;
        while ((p = s.find(pre, p)) != npos) {
            std::size_t e = s.find(']', p);
            if (e == npos) { s.erase(p); break; }
            std::size_t sp = s.find(' ', p);
            string text = (sp != npos && sp < e) ? s.substr(sp + 1, e - sp - 1) : "";
            s.replace(p, e + 1 - p, text);
        }
    }
}
string plain_of_links(const string& in) {  // [[a|b]] -> b, [[a]] -> a
    string o;
    std::size_t i = 0;
    while (i < in.size()) {
        std::size_t b = in.find("[[", i);
        if (b == npos) { o += in.substr(i); break; }
        o += in.substr(i, b - i);
        std::size_t e = in.find("]]", b + 2);
        if (e == npos) { o += in.substr(b + 2); break; }
        string inner = in.substr(b + 2, e - b - 2);
        std::size_t bar = inner.rfind('|');
        o += bar == npos ? inner : inner.substr(bar + 1);
        i = e + 2;
    }
    return o;
}
// One raw (XML-escaped) wiki line to a prose string that keeps [[target|display]] links. Bold phrases go to *bold.
string clean_line(const string& raw, vector<string>* bold = nullptr) {
    string s = xml_decode(raw);
    erase_refs(s);
    remove_templates(s);
    s = strip_tags(s);
    s = fix_entities(s);
    remove_namespace_links(s);
    external_links(s);
    if (bold) {
        std::size_t p = 0;
        while ((p = s.find("'''", p)) != npos) {
            std::size_t e = s.find("'''", p + 3);
            if (e == npos) break;
            string b = trim(plain_of_links(s.substr(p + 3, e - p - 3)));
            if (b.size() >= 3 && b.size() <= 60 && bold->size() < 3) bold->push_back(b);
            p = e + 3;
        }
    }
    string o;
    for (std::size_t k = 0; k < s.size();) {
        if (s[k] == '\'' && k + 1 < s.size() && s[k + 1] == '\'') { while (k < s.size() && s[k] == '\'') ++k; continue; }
        if (s[k] == '&') {
            std::size_t sc = s.find(';', k);
            if (sc != npos && sc - k <= 8) { o += ' '; k = sc + 1; continue; }
        }
        o += s[k++];
    }
    return o;
}
// F2R (--raw): clean_line's steps done in place. The line is XML-decoded with a map back to the raw bytes (rb[i], re[i]:
// the raw span of decoded byte i); after that a removed byte becomes \x01 and a replaced span keeps its length, so the
// tokens of the result carry their raw positions. Links and their markup stay for tokenize_line.
string clean_line_inplace(const string& raw, vector<std::uint32_t>& rb, vector<std::uint32_t>& re) {
    string s;
    rb.clear();
    re.clear();
    for (std::size_t i = 0; i < raw.size();) {
        std::size_t n = 1;
        char c = raw[i];
        for (auto [e, d] : {std::pair<const char*, char>{"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&quot;", '"'}})
            if (raw.compare(i, std::strlen(e), e) == 0) { c = d; n = std::strlen(e); break; }
        s += c;
        rb.push_back(static_cast<std::uint32_t>(i));
        re.push_back(static_cast<std::uint32_t>(i + n));
        i += n;
    }
    auto blank = [&](std::size_t p, std::size_t n) { for (std::size_t k = p; k < s.size() && k - p < n; ++k) s[k] = '\x01'; };
    auto span = [&](const char* open, const char* close) {
        for (std::size_t p = 0; (p = s.find(open, p)) != npos;) {
            std::size_t e = s.find(close, p + std::strlen(open));
            blank(p, e == npos ? npos : e + std::strlen(close) - p);
            if (e == npos) break;
        }
    };
    span("<!--", "-->");
    for (std::size_t p = 0; (p = s.find("<ref", p)) != npos;) {  // erase_refs
        char c = p + 4 < s.size() ? s[p + 4] : '>';
        if (c != ' ' && c != '>' && c != '/') { p += 4; continue; }
        std::size_t gt = s.find('>', p), e = gt == npos || s[gt - 1] == '/' ? gt : s.find("</ref>", gt);
        blank(p, e == npos ? npos : (e == gt ? gt + 1 : e + 6) - p);
        if (e == npos) break;
    }
    for (int guard = 0; guard < 80; ++guard) {  // remove_templates
        std::size_t e = s.find("}}");
        if (e == npos) break;
        std::size_t b = s.rfind("{{", e);
        if (b == npos) blank(e, 2);
        else blank(b, e + 2 - b);
    }
    if (std::size_t p = s.find("{{"); p != npos) blank(p, npos);
    for (std::size_t i = 0; i < s.size(); ++i) {  // strip_tags
        std::size_t gt = s[i] == '<' ? s.find('>', i) : npos;
        if (gt != npos && gt - i < 200) { s[i] = ' '; blank(i + 1, gt - i); i = gt; }
    }
    for (auto [from, to] : {std::pair<const char*, char>{"&ndash;", '-'}, {"&mdash;", '-'}, {"\xe2\x80\x93", '-'}, {"\xe2\x80\x94", '-'},
                            {"&#8211;", '-'}, {"&#8212;", '-'}, {"&nbsp;", ' '}, {"\xc2\xa0", ' '}, {"\xe2\x80\x99", '\''},
                            {"\xe2\x80\x98", '\''}, {"\xe2\x80\x9c", '"'}, {"\xe2\x80\x9d", '"'}})  // fix_entities
        for (std::size_t p = 0; (p = s.find(from, p)) != npos;) {
            s[p] = to;
            re[p] = re[p + std::strlen(from) - 1];  // the token spans the whole entity in the raw bytes
            blank(p + 1, std::strlen(from) - 1);
        }
    for (std::size_t p = 0; (p = s.find("[[", p)) != npos;) {  // remove_namespace_links
        std::size_t q = p + 2, colon = s.find(':', q), close = s.find("]]", q), nest = s.find("[[", q);
        bool ns = colon != npos && colon - q <= 14 && (close == npos || colon < close) && (nest == npos || colon < nest) &&
                  colon + 1 < s.size() && s[colon + 1] != ' ' && colon > q;
        if (!ns) { p += 2; continue; }
        int depth = 0;
        std::size_t i = p;
        bool done = false;
        for (; i + 1 < s.size(); ++i) {
            if (s[i] == '[' && s[i + 1] == '[') { ++depth; ++i; }
            else if (s[i] == ']' && s[i + 1] == ']') { --depth; ++i; if (depth == 0) { ++i; done = true; break; } }
        }
        blank(p, (done ? i : s.size()) - p);
    }
    for (const char* pre : {"[http", "[ftp"})  // external_links: keep the link text
        for (std::size_t p = 0; (p = s.find(pre, p)) != npos;) {
            std::size_t e = s.find(']', p), sp = s.find(' ', p);
            if (e == npos) { blank(p, npos); break; }
            blank(p, (sp != npos && sp < e ? sp + 1 : e) - p);
            s[e] = '\x01';
        }
    for (std::size_t k = 0; k < s.size();) {  // '' runs; other &entities; -> a space
        if (s[k] == '\'' && k + 1 < s.size() && s[k + 1] == '\'') { while (k < s.size() && s[k] == '\'') s[k++] = '\x01'; continue; }
        std::size_t sc = s[k] == '&' ? s.find(';', k) : npos;
        if (sc != npos && sc - k <= 8) { s[k] = ' '; blank(k + 1, sc - k); k = sc + 1; continue; }
        ++k;
    }
    return s;
}

// ============================================================ tokens, links, sentences
struct Tok { string o, w; bool cap = false; int link = -1; std::uint32_t rb = 0, re = 0; };  // rb/re: raw dump bytes (--raw)
struct Lnk { string target; int tb = 0, te = 0; };
struct Sent {
    vector<Tok> t;
    vector<Lnk> l;
    std::uint64_t off = 0;  // byte offset of the line start in the dump
    int idx = 0;            // sentence number within the page
    std::uint64_t hash = 0;
};

bool is_wc(unsigned char c) { return std::isalnum(c) || c >= 0x80; }
// pos (raw mode): pos[i] is the line position of s[i]; tokens get rb/re there. \x01 marks a byte cleaned away in place.
void add_word_tokens(const string& s, int link, vector<Tok>& T, const std::uint32_t* pos = nullptr) {
    std::size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isspace(c) || c == 1) { ++i; continue; }
        const std::size_t i0 = i;
        Tok t;
        if (is_wc(c)) {
            std::size_t j = i + 1;
            while (j < n) {
                unsigned char d = static_cast<unsigned char>(s[j]);
                if (is_wc(d)) ++j;
                else if (d == '\'' && j + 1 < n && s[j + 1] == 's' && (j + 2 >= n || !is_wc(static_cast<unsigned char>(s[j + 2])))) break;  // possessive 's
                else if ((d == '\'' || d == '-') && j + 1 < n && is_wc(static_cast<unsigned char>(s[j + 1]))) j += 2;
                else break;
            }
            t.o = s.substr(i, j - i);
            string f = fold_ascii(t.o);
            t.cap = std::isupper(static_cast<unsigned char>(f[0])) != 0;
            t.w = lower(f);
            i = j;
        } else {
            t.o = string(1, static_cast<char>(c));
            t.w = t.o;
            ++i;
        }
        if (pos) { t.rb = pos[i0]; t.re = pos[i - 1] + 1; }
        t.link = link;
        T.push_back(std::move(t));
    }
}
// track (raw mode): tokens get rb/re = their positions in s.
void tokenize_line(const string& s, vector<Tok>& T, vector<Lnk>& L, bool track = false) {
    vector<std::uint32_t> P;
    auto at = [&](std::size_t from, std::size_t n) -> const std::uint32_t* {  // positions of s.substr(from, n)
        if (!track) return nullptr;
        P.resize(n);
        for (std::size_t k = 0; k < n; ++k) P[k] = static_cast<std::uint32_t>(from + k);
        return P.data();
    };
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t b = s.find("[[", i);
        if (b == npos) { add_word_tokens(s.substr(i), -1, T, at(i, s.size() - i)); break; }
        add_word_tokens(s.substr(i, b - i), -1, T, at(i, b - i));
        std::size_t e = s.find("]]", b + 2);
        if (e == npos) { add_word_tokens(s.substr(b + 2), -1, T, at(b + 2, s.size() - b - 2)); break; }
        string inner = s.substr(b + 2, e - b - 2);
        std::size_t bar = inner.find('|');
        string target = trim(bar == npos ? inner : inner.substr(0, bar));
        string disp = bar == npos ? target : inner.substr(inner.rfind('|') + 1);
        std::size_t dfrom = bar == npos ? b + 2 + inner.find(target) : b + 3 + inner.rfind('|');  // where disp starts in s
        std::size_t h = target.find('#');
        if (h != npos) target = trim(target.substr(0, h));
        if (!target.empty() && target[0] == ':') target.erase(0, 1);
        i = e + 2;
        std::size_t te = i;
        while (te < s.size() && te - i < 4 && s[te] >= 'a' && s[te] <= 'z') ++te;
        const std::uint32_t* dp = nullptr;
        if (track) {  // disp, then the letters glued after "]]"
            vector<std::uint32_t> Q;
            for (std::size_t k = 0; k < disp.size(); ++k) Q.push_back(static_cast<std::uint32_t>(dfrom + k));
            for (std::size_t k = i; k < te; ++k) Q.push_back(static_cast<std::uint32_t>(k));
            P = Q;
            dp = P.data();
        }
        disp += s.substr(i, te - i);
        i = te;
        if (target.empty() || target.size() > 90 || inner.find("[[") != npos) { add_word_tokens(disp, -1, T, dp); continue; }
        int idx = static_cast<int>(L.size());
        Lnk l;
        l.target = target;
        l.tb = static_cast<int>(T.size());
        add_word_tokens(disp, idx, T, dp);
        l.te = static_cast<int>(T.size());
        if (l.te > l.tb) L.push_back(l);
    }
}
bool is_abbrev(const string& w) {
    static const std::set<string> A = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "prof", "gen", "col", "lt", "sgt",
                                       "mt", "inc", "ltd", "co", "corp", "ca", "fl", "ft", "vol", "ed", "eds", "approx", "cf", "rev",
                                       "hon", "gov", "pres", "capt", "dept", "est", "ave", "bros", "pp", "ph", "sen", "rep", "cmdr"};
    return (w.size() == 1 && std::isalpha(static_cast<unsigned char>(w[0]))) || A.count(w) > 0;
}
void make_sentences(const vector<Tok>& T, const vector<Lnk>& L, std::uint64_t off, vector<Sent>& out, int& sidx) {
    std::size_t b = 0;
    auto flush = [&](std::size_t e) {
        if (e > b && e - b >= 4 && e - b <= 120) {
            Sent s;
            s.off = off;
            s.idx = sidx;
            int lastold = -2;
            for (std::size_t k = b; k < e; ++k) {
                Tok t = T[k];
                if (t.link >= 0) {
                    if (t.link != lastold) {
                        Lnk nl = L[static_cast<std::size_t>(t.link)];
                        nl.tb = static_cast<int>(k - b);
                        nl.te = nl.tb;
                        s.l.push_back(nl);
                        lastold = t.link;
                    }
                    s.l.back().te = static_cast<int>(k - b) + 1;
                    t.link = static_cast<int>(s.l.size()) - 1;
                } else {
                    lastold = -2;
                }
                s.t.push_back(std::move(t));
            }
            string j;
            for (auto& t : s.t) { j += t.w; j += ' '; }
            s.hash = fnv(j);
            out.push_back(std::move(s));
        }
        ++sidx;
        b = e;
    };
    for (std::size_t i = 0; i < T.size(); ++i) {
        const string& w = T[i].w;
        if (w != "." && w != "!" && w != "?") continue;
        bool endok = (i + 1 >= T.size()) || T[i + 1].cap || std::isdigit(static_cast<unsigned char>(T[i + 1].w[0])) || T[i + 1].w == "\"" || T[i + 1].w == "(";
        if (w == ".") {
            if (i > 0 && is_abbrev(T[i - 1].w)) endok = false;
            if (T[i].link >= 0 && i + 1 < T.size() && T[i + 1].link == T[i].link) endok = false;
        }
        if (i + 1 < T.size() && T[i + 1].w == ".") endok = false;
        if (endok) flush(i + 1);
    }
    flush(T.size());
}

// ============================================================ pages
struct PageRef { string title; std::size_t tb = 0, te = 0; };  // text byte range in the dump
bool g_raw = false;              // F2R --raw: patterns over the dump's own bytes
const string* g_dump = nullptr;  // the dump, for raw patterns

// Prose sentences of a page: lines outside templates, tables, lists and headings. bold gets the first lead line's bold phrases.
void page_sentences(const string& data, const PageRef& pg, vector<Sent>& out, vector<string>* bold = nullptr,
                    std::size_t max_sent = static_cast<std::size_t>(-1)) {
    const string raw = data.substr(pg.tb, pg.te - pg.tb);
    int depth = 0, table_depth = 0, sidx = 0;
    bool first_prose = true;
    std::size_t pos = 0;
    while (pos <= raw.size()) {
        std::size_t nl = raw.find('\n', pos);
        if (nl == npos) nl = raw.size();
        string line = xml_decode(raw.substr(pos, nl - pos));
        string nc = line;
        erase_refs(nc);
        string t = trim(nc);
        auto count_of = [](const string& s, const char* k) { int n = 0; for (std::size_t p = s.find(k); p != npos; p = s.find(k, p + 2)) ++n; return n; };
        int delta = count_of(nc, "{{") - count_of(nc, "}}");
        bool in_tpl = depth > 0;
        if (t.compare(0, 2, "{|") == 0) ++table_depth;
        bool table_end = t.compare(0, 2, "|}") == 0;
        bool skip = in_tpl || table_depth > 0 || t.empty() || t[0] == '|' || t[0] == '!' || t[0] == '=' || t[0] == '*' || t[0] == '#' ||
                    t[0] == ':' || t[0] == ';' || t[0] == '<' || t[0] == '{' || starts_with(t, "[[Category") || starts_with(t, "[[category") ||
                    starts_with(t, "__") || starts_with(t, "[[Image:") || starts_with(t, "[[image:") || (delta > 0 && t.find("{{") == 0);
        if (!skip && t.size() < 8000) {
            vector<string> bl;
            string c = clean_line(raw.substr(pos, nl - pos), (first_prose && bold) ? &bl : nullptr);
            if (first_prose && bold && !bl.empty()) *bold = bl;
            vector<Tok> T;
            vector<Lnk> L;
            if (g_raw) {  // tokens with their raw byte spans in the dump
                vector<std::uint32_t> mb, me;
                tokenize_line(clean_line_inplace(raw.substr(pos, nl - pos), mb, me), T, L, true);
                const auto base = static_cast<std::uint32_t>(pg.tb + pos);
                for (Tok& t : T) { t.re = base + me[t.re - 1]; t.rb = base + mb[t.rb]; }
            } else {
                tokenize_line(trim(c), T, L);
            }
            if (T.size() >= 4) {
                first_prose = false;
                make_sentences(T, L, pg.tb + pos, out, sidx);
                if (out.size() >= max_sent) return;
            }
        }
        depth = std::max(0, depth + delta);
        if (table_end) table_depth = std::max(0, table_depth - 1);
        if (nl >= raw.size()) break;
        pos = nl + 1;
    }
}

}  // namespace

namespace {

// ============================================================ dates
struct Date {
    int y = INT_MIN, m = 0, d = 0;
    bool ok() const { return y != INT_MIN; }
};
const char* const kMonths[12] = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"};
int month_of(const string& w) {
    for (int i = 0; i < 12; ++i) {
        if (w == kMonths[i]) return i + 1;
        if (w.size() == 3 && w == string(kMonths[i]).substr(0, 3) && w != "may") return i + 1;
    }
    if (w == "sept") return 9;
    return 0;
}
bool all_digits(const string& s, std::size_t lo, std::size_t hi) {
    if (s.size() < lo || s.size() > hi) return false;
    for (unsigned char c : s) if (!std::isdigit(c)) return false;
    return true;
}
string fmt_date(const Date& D) {
    if (!D.ok()) return "";
    char b[48];
    if (D.y < 0) { std::snprintf(b, sizeof b, "%d BC", -D.y); return b; }
    if (D.m && D.d) std::snprintf(b, sizeof b, "%04d-%02d-%02d", D.y, D.m, D.d);
    else if (D.m) std::snprintf(b, sizeof b, "%04d-%02d", D.y, D.m);
    else std::snprintf(b, sizeof b, "%04d", D.y);
    return b;
}
// Two dates agree when the years are equal and every month/day both give is equal.
bool date_compat(const Date& a, const Date& b) {
    if (!a.ok() || !b.ok() || a.y != b.y) return false;
    if (a.m && b.m && a.m != b.m) return false;
    if (a.d && b.d && a.d != b.d) return false;
    return true;
}
// Date from a seed value: "{{birth date|1905|3|4}}", "[[April 1]], [[1947]]", "384 BC", "1905-03-04".
Date parse_date_str(const string& raw0) {
    string raw = raw0;
    erase_refs(raw);
    Date D;
    std::size_t tp = raw.find("{{");
    if (tp != npos) {
        std::size_t e = raw.find("}}", tp);
        string inner = raw.substr(tp + 2, (e == npos ? raw.size() : e) - tp - 2);
        vector<string> parts;
        std::size_t p = 0;
        while (true) {
            std::size_t q = inner.find('|', p);
            parts.push_back(trim(inner.substr(p, q == npos ? npos : q - p)));
            if (q == npos) break;
            p = q + 1;
        }
        if (parts.size() >= 2 && lower(parts[0]).find("date") != npos) {
            int v[3] = {0, 0, 0}, n = 0;
            for (std::size_t k = 1; k < parts.size() && n < 3; ++k)
                if (all_digits(parts[k], 1, 4)) v[n++] = std::atoi(parts[k].c_str());
            if (n >= 1 && v[0] > 0) {
                D.y = v[0];
                if (n >= 2 && v[1] >= 1 && v[1] <= 12) D.m = v[1];
                if (n >= 3 && v[2] >= 1 && v[2] <= 31) D.d = v[2];
                return D;
            }
        }
    }
    string s = plain_of_links(raw);
    remove_templates(s);
    s = fix_entities(strip_tags(s));
    vector<string> W;
    string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c)) cur += static_cast<char>(std::tolower(c));
        else if (!cur.empty()) { W.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) W.push_back(cur);
    if (W.size() >= 3 && all_digits(W[0], 4, 4) && all_digits(W[1], 2, 2) && all_digits(W[2], 2, 2)) {
        D.y = std::atoi(W[0].c_str());
        D.m = std::atoi(W[1].c_str());
        D.d = std::atoi(W[2].c_str());
        if (D.m < 1 || D.m > 12) D.m = 0;
        if (D.d < 1 || D.d > 31) D.d = 0;
        return D;
    }
    int mi = -1, yi = -1;
    for (std::size_t i = 0; i < W.size(); ++i) if (mi < 0 && month_of(W[i])) mi = static_cast<int>(i);
    for (std::size_t i = 0; i < W.size(); ++i) {
        bool bc = i + 1 < W.size() && (W[i + 1] == "bc" || W[i + 1] == "bce");
        if ((all_digits(W[i], 3, 4) && std::atoi(W[i].c_str()) >= 100) || (bc && all_digits(W[i], 1, 4))) { yi = static_cast<int>(i); break; }
    }
    if (yi < 0) return D;
    D.y = std::atoi(W[static_cast<std::size_t>(yi)].c_str());
    if (static_cast<std::size_t>(yi) + 1 < W.size() && (W[static_cast<std::size_t>(yi) + 1] == "bc" || W[static_cast<std::size_t>(yi) + 1] == "bce")) D.y = -D.y;
    if (mi >= 0) {
        D.m = month_of(W[static_cast<std::size_t>(mi)]);
        for (int k : {mi - 1, mi + 1}) {
            if (k < 0 || k >= static_cast<int>(W.size()) || k == yi) continue;
            if (all_digits(W[static_cast<std::size_t>(k)], 1, 2)) {
                int v = std::atoi(W[static_cast<std::size_t>(k)].c_str());
                if (v >= 1 && v <= 31) { D.d = v; break; }
            }
        }
    }
    return D;
}

// ============================================================ relations
enum RT { T_PLACE, T_PLACE_LINK, T_PERSON, T_NAME, T_DATE, T_YEAR, T_LEX };
struct RelDef { const char* name; RT type; double phi; bool multi; };
// phi = functionality (PLAN 7.2): hand-set here, to be re-estimated from Wikidata later.
const RelDef kRel[] = {
    {"capital", T_PLACE_LINK, 0.95, false}, {"largest_city", T_PLACE_LINK, 0.90, false}, {"country", T_PLACE_LINK, 0.80, false},
    {"location", T_PLACE, 0.50, false},     {"located_in", T_PLACE, 0.60, true},         {"birth_place", T_PLACE, 0.95, false},
    {"death_place", T_PLACE, 0.95, false},  {"birth_date", T_DATE, 1.00, false},         {"death_date", T_DATE, 1.00, false},
    {"birth_year", T_YEAR, 1.00, false},    {"death_year", T_YEAR, 1.00, false},         {"founded", T_YEAR, 0.90, false},
    {"author", T_PERSON, 0.30, true},       {"director", T_PERSON, 0.60, true},          {"composer", T_PERSON, 0.50, true},
    {"founder", T_PERSON, 0.30, true},      {"creator", T_PERSON, 0.40, true},           {"spouse", T_PERSON, 0.30, true},
    {"predecessor", T_PERSON, 0.50, true},  {"successor", T_PERSON, 0.50, true},         {"developer", T_NAME, 0.50, true},
    {"publisher", T_NAME, 0.60, true},      {"owner", T_NAME, 0.80, true},               {"genre", T_LEX, 0.10, true},
    {"language", T_LEX, 0.20, true},        {"currency", T_LEX, 0.90, false},            {"occupation", T_LEX, 0.10, true},
    {"nationality", T_LEX, 0.50, true},     {"party", T_LEX, 0.50, true},                {"family", T_LEX, 0.95, false},
    {"order", T_LEX, 0.95, false},          {"class", T_LEX, 0.95, false},           {"inventor", T_PERSON, 0.50, true}};
constexpr int kNRel = static_cast<int>(sizeof(kRel) / sizeof(kRel[0]));
int rel_index(const string& n) {
    for (int i = 0; i < kNRel; ++i) if (n == kRel[i].name) return i;
    return -1;
}
bool is_datey(int r) { return kRel[r].type == T_DATE || kRel[r].type == T_YEAR; }

string flat_key(const string& k) {
    string o;
    for (unsigned char c : k) if (std::isalnum(c)) o += static_cast<char>(std::tolower(c));
    return o;
}
// Infobox / table field name (flattened) to a relation, or -1.
int key_to_rel(const string& nk, const string& tpl) {
    static const std::unordered_map<string, string> M = {
        {"capital", "capital"}, {"capitalcity", "capital"}, {"largestcity", "largest_city"}, {"country", "country"}, {"nation", "country"},
        {"location", "location"}, {"headquarters", "location"}, {"hq", "location"}, {"city", "location"}, {"place", "location"},
        {"origin", "location"}, {"birthplace", "birth_place"}, {"placeofbirth", "birth_place"}, {"placebirth", "birth_place"},
        {"deathplace", "death_place"}, {"placeofdeath", "death_place"}, {"placedeath", "death_place"}, {"birthdate", "birth_date"},
        {"dateofbirth", "birth_date"}, {"datebirth", "birth_date"}, {"born", "birth_date"}, {"deathdate", "death_date"},
        {"dateofdeath", "death_date"}, {"datedeath", "death_date"}, {"died", "death_date"}, {"author", "author"}, {"authors", "author"},
        {"director", "director"}, {"directedby", "director"}, {"composer", "composer"}, {"composers", "composer"}, {"founder", "founder"},
        {"founders", "founder"}, {"foundedby", "founder"}, {"founded", "founded"}, {"foundation", "founded"}, {"established", "founded"},
        {"genre", "genre"}, {"genres", "genre"}, {"musicgenre", "genre"}, {"officiallanguages", "language"},
        {"officiallanguage", "language"}, {"officiallang", "language"}, {"language", "language"}, {"languages", "language"},
        {"currency", "currency"}, {"developer", "developer"}, {"developers", "developer"}, {"creator", "creator"}, {"creators", "creator"},
        {"designer", "creator"}, {"designers", "creator"}, {"publisher", "publisher"}, {"spouse", "spouse"}, {"party", "party"},
        {"before", "predecessor"}, {"predecessor", "predecessor"}, {"after", "successor"}, {"successor", "successor"},
        {"occupation", "occupation"}, {"owner", "owner"}, {"familia", "family"}, {"family", "family"}, {"ordo", "order"},
        {"classis", "class"}, {"nationality", "nationality"}};
    if (nk == "music") return tpl.find("film") != npos ? rel_index("composer") : -1;
    auto it = M.find(nk);
    return it == M.end() ? -1 : rel_index(it->second);
}

// ============================================================ global type lexicons (filled in pass 1)
std::unordered_set<string> g_titles;                 // norm_ent of every page title
std::unordered_set<string> g_place;                  // place-like pages + objects of place seeds
std::unordered_set<string> g_person;                 // person-like pages
std::unordered_set<string> g_country;                // country / state / province pages
std::unordered_map<string, std::uint64_t> g_lex;     // norm string -> mask of LEX relations that take it as an object

const std::unordered_set<string>& stop_words() {
    static const std::unordered_set<string> S = {
        "the", "a", "an", "in", "on", "at", "he", "she", "it", "his", "her", "its", "they", "their", "this", "that", "these", "those", "however",
        "as", "after", "before", "during", "when", "while", "since", "from", "for", "by", "with", "of", "and", "but", "or", "if", "there", "then",
        "some", "many", "most", "all", "one", "two", "three", "in", "also", "such", "other", "another", "each", "both", "although", "though",
        "because", "so", "to", "we", "you", "i", "who", "which", "what", "where", "how", "why", "not", "no", "yes", "see", "according",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "monday",
        "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mr", "mrs", "dr"};
    return S;
}
bool is_datename(const string& w) {
    static const std::unordered_set<string> D = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"};
    return month_of(w) != 0 || D.count(w) > 0 || w == "may";
}

// ============================================================ candidate object spans
enum { K_LINK = 1, K_CHUNK = 2, K_DATE = 3, K_LEX = 4 };
struct Cand {
    int b = 0, e = 0, kind = 0;
    string k1, k2, k3, surf;  // k1 target, k2 display, k3 target before its first comma
    Date d;
    int ntok() const { return e - b; }
};

bool is_year_tok(const string& w) { return all_digits(w, 3, 4) && std::atoi(w.c_str()) >= 100 && std::atoi(w.c_str()) <= 2100; }

void find_dates(const Sent& s, vector<Cand>& out) {
    const auto& t = s.t;
    const int n = static_cast<int>(t.size());
    vector<char> used(static_cast<std::size_t>(n), 0);
    auto push = [&](int b, int e, Date D) {
        // BC / AD marker after the year
        if (e < n && (t[static_cast<std::size_t>(e)].w == "bc" || t[static_cast<std::size_t>(e)].w == "bce")) { D.y = -D.y; ++e; }
        else if (e < n && (t[static_cast<std::size_t>(e)].w == "ad" || t[static_cast<std::size_t>(e)].w == "ce")) ++e;
        Cand c;
        c.b = b; c.e = e; c.kind = K_DATE; c.d = D; c.k1 = fmt_date(D); c.k2 = c.k1; c.surf = c.k1;
        out.push_back(c);
        for (int k = b; k < e; ++k) used[static_cast<std::size_t>(k)] = 1;
    };
    for (int i = 0; i < n; ++i) {
        const Tok& tk = t[static_cast<std::size_t>(i)];
        int mi = tk.cap ? month_of(tk.w) : 0;
        if (tk.w == "may" && tk.cap) mi = 5;
        if (!mi) continue;
        Date D;
        D.m = mi;
        int b = i, e = i + 1;
        bool have_year = false;
        auto W = [&](int k) -> const string& { return t[static_cast<std::size_t>(k)].w; };
        if (e < n && all_digits(W(e), 1, 2) && std::atoi(W(e).c_str()) >= 1 && std::atoi(W(e).c_str()) <= 31) {
            D.d = std::atoi(W(e).c_str());
            ++e;
            if (e + 1 < n && W(e) == "," && is_year_tok(W(e + 1))) { D.y = std::atoi(W(e + 1).c_str()); e += 2; have_year = true; }
            else if (e < n && is_year_tok(W(e))) { D.y = std::atoi(W(e).c_str()); ++e; have_year = true; }
        } else if (e < n && is_year_tok(W(e))) {
            D.y = std::atoi(W(e).c_str());
            ++e;
            have_year = true;
        }
        if (!have_year) continue;
        if (D.d == 0 && b > 0 && all_digits(W(b - 1), 1, 2) && std::atoi(W(b - 1).c_str()) >= 1 && std::atoi(W(b - 1).c_str()) <= 31 && !used[static_cast<std::size_t>(b - 1)]) {
            D.d = std::atoi(W(b - 1).c_str());
            --b;
        }
        push(b, e, D);
        i = e - 1;
    }
    for (int i = 0; i < n; ++i) {
        if (used[static_cast<std::size_t>(i)]) continue;
        const string& w = t[static_cast<std::size_t>(i)].w;
        bool bc = i + 1 < n && (t[static_cast<std::size_t>(i + 1)].w == "bc" || t[static_cast<std::size_t>(i + 1)].w == "bce");
        bool yr = is_year_tok(w) || (bc && all_digits(w, 1, 4) && std::atoi(w.c_str()) > 0);
        if (!yr) continue;
        if (i >= 2 && t[static_cast<std::size_t>(i - 1)].w == "," && all_digits(t[static_cast<std::size_t>(i - 2)].w, 1, 3)) continue;  // 1,500
        if (i + 1 < n && (t[static_cast<std::size_t>(i + 1)].w == "%" || t[static_cast<std::size_t>(i + 1)].w == ",") && i + 2 < n && all_digits(t[static_cast<std::size_t>(i + 2)].w, 3, 3)) continue;
        Date D;
        D.y = std::atoi(w.c_str());
        push(i, i + 1, D);
    }
}

string join_o(const Sent& s, int b, int e) {
    string o;
    for (int k = b; k < e; ++k) { if (!o.empty()) o += ' '; o += s.t[static_cast<std::size_t>(k)].o; }
    return o;
}
string join_w(const Sent& s, int b, int e) {
    string o;
    for (int k = b; k < e; ++k) { if (!o.empty()) o += ' '; o += s.t[static_cast<std::size_t>(k)].w; }
    return o;
}
// Link surface as an answer: target without "(qualifier)", or the display text for a pipe trick.
string link_surface(const Sent& s, const Lnk& L) {
    string t = L.target;
    std::size_t p = t.find(" (");
    if (p != npos && p > 0 && t.back() == ')') t = t.substr(0, p);
    string d = join_o(s, L.tb, L.te);
    if (!d.empty() && lower(d) != lower(t) && lower(t).compare(0, lower(d).size(), lower(d)) == 0 && d.size() >= 3) return d;
    return t;
}

void gen_cands(const Sent& s, vector<Cand>& out) {
    out.clear();
    const int n = static_cast<int>(s.t.size());
    std::set<std::pair<int, int>> link_spans;
    for (const Lnk& L : s.l) {
        Cand c;
        c.b = L.tb; c.e = L.te; c.kind = K_LINK;
        c.k1 = norm_ent(L.target);
        c.k2 = norm_ent(join_o(s, L.tb, L.te));
        c.surf = link_surface(s, L);
        std::size_t cm = L.target.find(',');
        if (cm != npos && cm > 0) c.k3 = norm_ent(L.target.substr(0, cm));
        if (c.k1.empty()) continue;
        link_spans.insert({c.b, c.e});
        out.push_back(c);
    }
    static const std::unordered_set<string> conn = {"of", "de", "la", "le", "von", "van", "der", "du", "del", "di", "da", "al", "bin", "ibn"};
    for (int i = 0; i < n;) {
        const Tok& tk = s.t[static_cast<std::size_t>(i)];
        bool capok = tk.cap && !(i == 0 && stop_words().count(tk.w)) && !(tk.w.size() == 1 && tk.w != "a" && false);
        if (!capok || tk.w.size() < 2) { ++i; continue; }
        int j = i + 1;
        while (j < n && j - i < 7) {
            const Tok& u = s.t[static_cast<std::size_t>(j)];
            if (u.cap && u.w.size() >= 1 && std::isalpha(static_cast<unsigned char>(u.w[0]))) { ++j; continue; }
            if (conn.count(u.w) && j + 1 < n && s.t[static_cast<std::size_t>(j + 1)].cap) { j += 2; continue; }
            break;
        }
        if (!link_spans.count({i, j})) {
            Cand c;
            c.b = i; c.e = j; c.kind = K_CHUNK;
            c.surf = join_o(s, i, j);
            c.k1 = c.k2 = norm_ent(c.surf);
            if (!c.k1.empty()) out.push_back(c);
        }
        i = j;
    }
    find_dates(s, out);
    for (int len = 1; len <= 3; ++len)
        for (int i = 0; i + len <= n; ++i) {
            bool bad = false;
            for (int k = i; k < i + len; ++k) if (!is_wc(static_cast<unsigned char>(s.t[static_cast<std::size_t>(k)].w[0]))) bad = true;
            if (bad) continue;
            string key = join_w(s, i, i + len);
            auto it = g_lex.find(key);
            if (it == g_lex.end()) continue;
            Cand c;
            c.b = i; c.e = i + len; c.kind = K_LEX; c.k1 = c.k2 = key; c.surf = key;
            out.push_back(c);
        }
}

// ============================================================ type constraints (PLAN 3.2: a capital is a place; a date parses)
bool name_like(const Cand& c) {
    int n = c.ntok();
    if (n < 2 || n > 4) return false;
    std::istringstream is(c.k1);
    string w;
    while (is >> w) if (stop_words().count(w) || is_datename(w)) return false;
    return true;
}
bool type_ok(int r, const Cand& c) {
    switch (kRel[r].type) {
        case T_DATE:
        case T_YEAR: return c.kind == K_DATE && c.d.ok() && c.d.y <= 2006 && c.d.y >= -3500;
        case T_PLACE_LINK: return c.kind == K_LINK && (g_place.count(c.k1) || g_place.count(c.k2));
        case T_PLACE:
            if (c.kind == K_LINK) return g_place.count(c.k1) || g_place.count(c.k2);
            if (c.kind == K_CHUNK) return g_place.count(c.k1) > 0;
            return false;
        case T_PERSON:
            if (c.kind == K_LINK) {
                if (g_person.count(c.k1)) return true;
                if (g_titles.count(c.k1)) return false;  // a page that is not a person
                return name_like(c) && !g_place.count(c.k1);
            }
            if (c.kind == K_CHUNK) return name_like(c) && !g_place.count(c.k1) && !g_titles.count(c.k1) ? true : (g_person.count(c.k1) > 0);
            return false;
        case T_NAME:
            if (c.kind != K_LINK && c.kind != K_CHUNK) return false;
            if (c.ntok() > 5 || all_digits(c.k1, 1, 10)) return false;
            { std::istringstream is(c.k1); string w; is >> w; if (stop_words().count(w) || is_datename(w)) return false; }
            return !g_person.count(c.k1) || true;
        case T_LEX: {
            std::uint64_t bit = 1ULL << r;
            auto a = g_lex.find(c.k1);
            if (a != g_lex.end() && (a->second & bit)) return true;
            auto b = g_lex.find(c.k2);
            return b != g_lex.end() && (b->second & bit);
        }
    }
    return false;
}

// ============================================================ subject mentions
struct Form { vector<string> w; string first_o; bool first_cap = false; };
struct SubjForms { vector<Form> forms; string title_key; bool person = false; };
struct Span { int b = 0, e = 0; bool pron = false; };

SubjForms make_forms(const string& title, const vector<string>& bold, bool person) {
    SubjForms sf;
    string base = title;
    std::size_t p = base.find(" (");
    if (p != npos && p > 0 && base.back() == ')') base = base.substr(0, p);
    sf.title_key = norm_ent(title);
    sf.person = person;
    std::set<string> seen;
    auto add = [&](const string& str) {
        vector<Tok> T;
        add_word_tokens(str, -1, T);
        if (T.empty() || T.size() > 8) return;
        string key;
        for (auto& t : T) key += t.w + " ";
        if (!seen.insert(key).second) return;
        Form f;
        for (auto& t : T) f.w.push_back(t.w);
        f.first_o = T[0].o;
        f.first_cap = T[0].cap;
        sf.forms.push_back(f);
    };
    add(base);
    for (const string& b : bold) if (norm_ent(b) != norm_ent(base)) add(b);
    if (person) {  // surname of a plain "Given Family" title (not "X of Y", "X, Y", or a numbered ruler)
        vector<Tok> T;
        add_word_tokens(base, -1, T);
        bool plain = T.size() >= 2 && T.size() <= 4;
        for (auto& t : T) if (t.w == "of" || t.w == "the" || t.w == "," || t.w == "de" || t.w == "von" || t.w == "van") plain = false;
        bool roman = true;
        if (!T.empty()) for (unsigned char c : T.back().o) if (!std::strchr("IVXLC", c)) roman = false;
        if (plain && !roman && T.back().cap && T.back().w.size() >= 4 && std::isalpha(static_cast<unsigned char>(T.back().w[0]))) add(T.back().o);
    }
    std::sort(sf.forms.begin(), sf.forms.end(), [](const Form& a, const Form& b) { return a.w.size() > b.w.size(); });
    return sf;
}
vector<Span> find_subject(const Sent& s, const SubjForms& sf) {
    vector<Span> out;
    const int n = static_cast<int>(s.t.size());
    vector<char> used(static_cast<std::size_t>(n), 0);
    for (const Form& f : sf.forms) {
        const int m = static_cast<int>(f.w.size());
        for (int i = 0; i + m <= n; ++i) {
            bool ok = true;
            for (int k = 0; k < m && ok; ++k) {
                if (used[static_cast<std::size_t>(i + k)] || s.t[static_cast<std::size_t>(i + k)].w != f.w[static_cast<std::size_t>(k)]) ok = false;
            }
            if (!ok) continue;
            if (f.first_cap && s.t[static_cast<std::size_t>(i)].o != f.first_o) continue;
            out.push_back({i, i + m});
            for (int k = 0; k < m; ++k) used[static_cast<std::size_t>(i + k)] = 1;
            i += m - 1;
        }
    }
    // a sentence-initial pronoun stands for the page subject ("Its capital is [[Algiers]]", "He was born in [[Paris]]")
    if (n >= 3 && !used[0]) {
        const string& w = s.t[0].w;
        bool pp = w == "he" || w == "she" || w == "his" || w == "her";
        bool pn = w == "it" || w == "its";
        if ((sf.person && pp) || (!sf.person && pn)) out.push_back({0, 1, true});
    }
    std::sort(out.begin(), out.end(), [](const Span& a, const Span& b) { return a.b < b.b; });
    return out;
}
// Subject type a relation needs: 1 person, 2 place, 3 country-like (country, state, province), 0 any.
int subj_type(int r) {
    static const char* const P[] = {"birth_place", "death_place", "birth_date", "death_date", "birth_year", "death_year", "spouse", "occupation", "nationality"};
    static const char* const C[] = {"capital", "largest_city", "currency", "language"};
    for (const char* n : P) if (std::strcmp(kRel[r].name, n) == 0) return 1;
    for (const char* n : C) if (std::strcmp(kRel[r].name, n) == 0) return 3;
    return std::strcmp(kRel[r].name, "located_in") == 0 ? 2 : 0;
}
bool g_cur_person = false, g_cur_place = false, g_cur_country = false;  // type of the page being read
bool subject_ok(int r) {
    int t = subj_type(r);
    return t == 0 || (t == 1 && g_cur_person) || (t == 2 && g_cur_place) || (t == 3 && g_cur_country);
}

// ============================================================ patterns: the text around two mentions
constexpr int kMaxBetween = 6;
string ctx_tok(const Sent& s, int k) {
    if (k < 0) return "^";
    if (k >= static_cast<int>(s.t.size())) return "$";
    const Tok& t = s.t[static_cast<std::size_t>(k)];
    if (all_digits(t.w, 1, 10)) return "#";
    if (t.cap && month_of(t.w)) return "<M>";
    return t.w;
}
// F2R (--raw): a mention's raw span, widened over the markup that wraps it: the whole [[...]] link (target, pipe and the
// letters glued after it) when its first or last token is a link token, then a run of 2+ quotes present on both sides.
std::pair<std::size_t, std::size_t> raw_mention(const Sent& s, int b, int e) {
    const string& d = *g_dump;
    std::size_t rb = s.t[static_cast<std::size_t>(b)].rb, re = s.t[static_cast<std::size_t>(e - 1)].re;
    if (s.t[static_cast<std::size_t>(b)].link >= 0) rb = std::min(rb, d.rfind("[[", rb));
    if (s.t[static_cast<std::size_t>(e - 1)].link >= 0) {
        std::size_t p = d.find("]]", d.rfind("[[", s.t[static_cast<std::size_t>(e - 1)].rb) + 2);
        if (p != npos) {
            p += 2;
            for (int k = 0; k < 4 && p < d.size() && d[p] >= 'a' && d[p] <= 'z'; ++k) ++p;
            re = std::max(re, p);
        }
    }
    std::size_t q1 = 0, q2 = 0;
    while (q1 < rb && d[rb - 1 - q1] == '\'') ++q1;
    while (re + q2 < d.size() && d[re + q2] == '\'') ++q2;
    if (q1 >= 2 && q2 >= 2) { rb -= std::min(q1, q2); re += std::min(q1, q2); }
    return {rb, re};
}
string raw_norm(const string& x) {  // digit runs -> #, capitalised month names -> <M>; every other byte as the dump has it
    string o;
    for (std::size_t i = 0; i < x.size();) {
        unsigned char c = static_cast<unsigned char>(x[i]);
        std::size_t j = i + 1;
        if (std::isdigit(c)) {
            while (j < x.size() && std::isdigit(static_cast<unsigned char>(x[j]))) ++j;
            o += '#';
        } else if (std::isalpha(c)) {
            while (j < x.size() && std::isalpha(static_cast<unsigned char>(x[j]))) ++j;
            o += std::isupper(c) && month_of(lower(x.substr(i, j - i))) ? string("<M>") : x.substr(i, j - i);
        } else {
            o += static_cast<char>(c);
        }
        i = j;
    }
    return o;
}
constexpr std::size_t kRawMaxGap = 200;  // bytes between the two mentions
// F2R: the raw pattern. Each slot keeps the markup of its mention ('''[S]''', [[[O]]]); between them go the dump's own
// bytes (raw_norm). Context: the bytes out to the end of the next token on each side, ^ / $ at the sentence ends.
bool raw_keys(const Sent& s, bool pron, bool s_first, int fb, int fe, int sb2, int se2, std::array<string, 4>& keys) {
    const string& d = *g_dump;
    auto cuts_link = [&](int b, int e) {  // the mention starts or ends inside a link ("New York" in [[Ballston Spa, New York]])
        auto L = [&](int k) { return k >= 0 && k < static_cast<int>(s.t.size()) ? s.t[static_cast<std::size_t>(k)].link : -2; };
        return (L(b) >= 0 && L(b - 1) == L(b)) || (L(e - 1) >= 0 && L(e) == L(e - 1));
    };
    if (cuts_link(fb, fe) || cuts_link(sb2, se2)) return false;
    auto [b1, e1] = raw_mention(s, fb, fe);
    auto [b2, e2] = raw_mention(s, sb2, se2);
    if (e1 > b2 || b2 - e1 > kRawMaxGap) return false;
    auto slot = [&](std::size_t b, std::size_t e, const char* lab) {
        std::size_t o = b, c = e;
        while (o < e && (d[o] == '\'' || d[o] == '[')) ++o;
        while (c > o && (d[c - 1] == '\'' || d[c - 1] == ']')) --c;
        return d.substr(b, o - b) + lab + d.substr(c, e - c);
    };
    const char* SL = pron ? "[P]" : "[S]";
    const string core = slot(b1, e1, s_first ? SL : "[O]") + raw_norm(d.substr(e1, b2 - e1)) + slot(b2, e2, s_first ? "[O]" : SL);
    string lc = "^", rc = "$";
    for (int k = fb - 1; k >= 0; --k)
        if (s.t[static_cast<std::size_t>(k)].re <= b1) { lc = raw_norm(d.substr(s.t[static_cast<std::size_t>(k)].rb, b1 - s.t[static_cast<std::size_t>(k)].rb)); break; }
    for (std::size_t k = static_cast<std::size_t>(se2); k < s.t.size(); ++k)
        if (s.t[k].rb >= e2) { rc = raw_norm(d.substr(e2, s.t[k].re - e2)); break; }
    if ((core + lc + rc).find('\t') != npos) return false;
    keys = {core, core + rc, lc + core, lc + core + rc};
    return true;
}
// Four lexical variants of the pattern for a subject span and an object span: core, +right token, +left token, both.
// Returns false when the spans overlap or are too far apart or too close.
bool make_keys(const Sent& s, const Span& S, int ob, int oe, std::array<string, 4>& keys, bool& s_first) {
    int fb, fe, sb2, se2;  // first slot, second slot
    if (S.e <= ob) { s_first = true; fb = S.b; fe = S.e; sb2 = ob; se2 = oe; }
    else if (oe <= S.b) { s_first = false; fb = ob; fe = oe; sb2 = S.b; se2 = S.e; }
    else return false;
    int between = sb2 - fe;
    if (between < 1 || between > kMaxBetween) return false;
    if (g_raw) return raw_keys(s, S.pron, s_first, fb, fe, sb2, se2, keys);
    string mid;
    for (int k = fe; k < sb2; ++k) { mid += ' '; mid += ctx_tok(s, k); }
    const char* SL = S.pron ? "[P]" : "[S]";  // [P]: a sentence-initial pronoun standing for the subject (learned separately)
    const char* A = s_first ? SL : "[O]";
    const char* B = s_first ? "[O]" : SL;
    string core = string(A) + mid + " " + B;
    keys[0] = core;
    keys[1] = core + " " + ctx_tok(s, se2);
    keys[2] = ctx_tok(s, fb - 1) + " " + core;
    keys[3] = ctx_tok(s, fb - 1) + " " + core + " " + ctx_tok(s, se2);
    return true;
}
// Pattern family: the core with no outer context, so the four variants of one match count as one kind of evidence.
string family_of(const string& key) {
    std::size_t a = std::min(key.find("[S]"), key.find("[P]")), b = key.find("[O]");
    std::size_t first = std::min(a, b), last = std::max(a, b);
    return key.substr(first, last + 3 - first);
}

}  // namespace

namespace {

// ============================================================ seeds (pass 1)
struct Obj { string surf, k1, k2, k3; Date d; vector<string> alt; };  // alt: more surface keys (Wikidata aliases)
struct Seed {
    string subj;
    int rel = 0;
    Obj o;
    int src = 0;  // 0 infobox/template, 1 wiki table, 2 category, 3 Wikidata (distant supervision only: never a fact)
    string tpl;
    std::uint64_t off = 0;
};
const char* const kSrcName[4] = {"infobox", "table", "category", "wikidata"};

vector<Seed> g_seeds;
std::unordered_map<string, vector<int>> g_by_subj;  // page title -> indices into g_seeds

struct LinkRaw { string target, display; std::size_t pos = 0; };
vector<LinkRaw> raw_links(const string& s) {
    vector<LinkRaw> out;
    for (std::size_t i = s.find("[["); i != npos; i = s.find("[[", i)) {
        std::size_t e = s.find("]]", i + 2);
        if (e == npos) break;
        string inner = s.substr(i + 2, e - i - 2);
        LinkRaw L;
        L.pos = i;
        std::size_t bar = inner.find('|');
        L.target = trim(bar == npos ? inner : inner.substr(0, bar));
        L.display = trim(bar == npos ? L.target : inner.substr(inner.rfind('|') + 1));
        std::size_t colon = L.target.find(':');
        bool ns = colon != npos && colon + 1 < L.target.size() && L.target[colon + 1] != ' ';
        std::size_t h = L.target.find('#');
        if (h != npos) L.target = trim(L.target.substr(0, h));
        if (!L.target.empty() && L.target[0] == ':') L.target.erase(0, 1);
        i = e + 2;
        if (ns || L.target.empty() || L.target.size() > 80 || inner.find("[[") != npos) continue;
        out.push_back(L);
    }
    return out;
}
string pretty_target(const LinkRaw& L) {
    string t = L.target;
    std::size_t p = t.find(" (");
    if (p != npos && p > 0 && t.back() == ')') t = t.substr(0, p);
    if (!L.display.empty() && lower(L.display) != lower(L.target) && L.display.size() >= 3 && lower(t).compare(0, L.display.size(), lower(L.display)) == 0)
        return L.display;
    return t;
}
bool has_alpha(const string& s) { for (unsigned char c : s) if (std::isalpha(c)) return true; return false; }
int alpha_count(const string& s) { int n = 0; for (unsigned char c : s) if (std::isalpha(c)) ++n; return n; }

// The object(s) a field value gives for relation r.
vector<Obj> make_objs(int r, const string& rawv) {
    vector<Obj> out;
    const RelDef& R = kRel[r];
    if (is_datey(r)) {
        Date D = parse_date_str(rawv);
        if (!D.ok() || D.y > 2006 || D.y < -3500) return out;
        if (R.type == T_YEAR) D.m = D.d = 0;
        Obj o;
        o.d = D;
        o.surf = o.k1 = o.k2 = fmt_date(D);
        out.push_back(o);
        return out;
    }
    string v = rawv;
    erase_refs(v);
    v = fix_entities(strip_tags(v, "\n"));
    static const std::set<string> junk = {"", "?", "n/a", "na", "none", "unknown", "-", "unavailable", "tbd", "tba", "yes", "no", "various"};
    auto accept = [&](Obj o) {
        if (o.k1.empty() || junk.count(o.k1) || o.k1.find("list of") == 0 || all_digits(o.k1, 1, 12)) return;
        if (o.surf.size() > 70) return;
        for (const Obj& x : out) if (x.k1 == o.k1) return;
        out.push_back(std::move(o));
    };
    auto links = raw_links(v);
    std::size_t cap = R.multi ? 4 : 1;
    if (!links.empty()) {
        string pre = plain_of_links(v.substr(0, links[0].pos));
        if (alpha_count(pre) < 3) {
            for (std::size_t i = 0; i < links.size() && out.size() < cap; ++i) {
                Obj o;
                o.surf = pretty_target(links[i]);
                o.k1 = norm_ent(links[i].target);
                o.k2 = norm_ent(links[i].display);
                std::size_t cm = links[i].target.find(',');
                if (cm != npos && cm > 0) o.k3 = norm_ent(links[i].target.substr(0, cm));
                accept(o);
            }
            return out;
        }
    }
    string p = plain_of_links(v);
    remove_templates(p);
    string q;
    for (std::size_t k = 0; k < p.size(); ++k) if (p[k] != '\'') q += p[k];
    p = q;
    if (p.find('=') != npos || p.find('|') != npos || p.size() > 160) return out;
    bool person = R.type == T_PERSON;
    if (person && p.find_first_of(",;&") != npos) return out;
    vector<string> pieces;
    {
        string cur;
        for (unsigned char c : p) {
            if (c == '\n' || (R.multi && (c == ',' || c == ';' || c == '/'))) { pieces.push_back(cur); cur.clear(); }
            else cur += static_cast<char>(c);
        }
        pieces.push_back(cur);
    }
    for (string& pc : pieces) {
        pc = trim(pc);
        std::size_t par = pc.find(" (");
        if (par != npos && par > 0) pc = trim(pc.substr(0, par));
        if (!R.multi && (R.type == T_PLACE || R.type == T_PLACE_LINK)) { std::size_t cm = pc.find(','); if (cm != npos) pc = trim(pc.substr(0, cm)); }
        if (pc.empty() || pc.size() > 50 || !has_alpha(pc) || out.size() >= cap) continue;
        Obj o;
        o.surf = pc;
        o.k1 = o.k2 = norm_ent(pc);
        accept(o);
    }
    return out;
}

bool cite_like_tpl(const string& t) {
    for (const char* k : {"cite", "citation", "web reference", "isbn", "harv", "audio", "multi-", "gutenberg", "imdb", "us patent", "note", "ref", "fn", "dablink", "otheruses"})
        if (t.find(k) != npos) return true;
    return false;
}
int count_of(const string& s, const char* k) {
    int n = 0;
    for (std::size_t p = s.find(k); p != npos; p = s.find(k, p + 2)) ++n;
    return n;
}
vector<string> split_top(const string& s) {
    vector<string> out;
    string cur;
    int br = 0, cb = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s.compare(i, 2, "[[") == 0) { ++br; cur += "[["; ++i; continue; }
        if (s.compare(i, 2, "]]") == 0) { br = std::max(0, br - 1); cur += "]]"; ++i; continue; }
        if (s.compare(i, 2, "{{") == 0) { ++cb; cur += "{{"; ++i; continue; }
        if (s.compare(i, 2, "}}") == 0) { cb = std::max(0, cb - 1); cur += "}}"; ++i; continue; }
        if (s[i] == '|' && br == 0 && cb == 0) { out.push_back(cur); cur.clear(); continue; }
        cur += s[i];
    }
    out.push_back(cur);
    return out;
}

struct PageFlags { bool place = false, person = false, country = false; };

void add_seed(const string& subj, int r, const Obj& o, int src, const string& tpl, std::uint64_t off) {
    if (norm_ent(subj) == o.k1) return;
    Seed s;
    s.subj = subj; s.rel = r; s.o = o; s.src = src; s.tpl = tpl; s.off = off;
    g_by_subj[subj].push_back(static_cast<int>(g_seeds.size()));
    g_seeds.push_back(std::move(s));
}

bool starts_any(const string& c, std::initializer_list<const char*> pre, string& rest) {
    for (const char* p : pre) {
        std::size_t n = std::strlen(p);
        if (c.compare(0, n, p) == 0 && c.size() > n) { rest = c.substr(n); return true; }
    }
    return false;
}

void category_seeds(const string& title, const string& raw, std::size_t base, PageFlags& fl) {
    static const std::set<string> occ = {"novelists", "poets", "painters", "composers", "philosophers", "mathematicians", "physicists", "actors",
                                         "actresses", "politicians", "writers", "scientists", "chemists", "singers", "musicians", "economists",
                                         "historians", "astronomers", "architects", "engineers", "inventors", "explorers", "sculptors",
                                         "dramatists", "playwrights", "journalists", "theologians", "biologists"};
    std::size_t p = 0;
    while ((p = raw.find("[[Category:", p)) != npos) {
        std::size_t e = raw.find("]]", p);
        if (e == npos) break;
        string c = raw.substr(p + 11, e - p - 11);
        std::size_t bar = c.find('|');
        if (bar != npos) c = c.substr(0, bar);
        c = trim(xml_decode(c));
        std::uint64_t off = base + p;
        p = e + 2;
        auto ends = [&](const char* suf) { std::size_t n = std::strlen(suf); return c.size() > n && c.compare(c.size() - n, n, suf) == 0; };
        if (ends(" births") || ends(" deaths") || c == "Living people") fl.person = true;
        string rest;
        if (ends(" births") || ends(" deaths")) {
            bool birth = ends(" births");
            string num = c.substr(0, c.size() - (birth ? 7 : 7));
            bool bc = false;
            if (num.size() > 3 && num.compare(num.size() - 3, 3, " BC") == 0) { bc = true; num = num.substr(0, num.size() - 3); }
            if (all_digits(num, 1, 4)) {
                Date D;
                D.y = std::atoi(num.c_str()) * (bc ? -1 : 1);
                Obj o;
                o.d = D;
                o.surf = o.k1 = o.k2 = fmt_date(D);
                add_seed(title, rel_index(birth ? "birth_year" : "death_year"), o, 2, "category", off);
            }
            continue;
        }
        if (starts_any(c, {"Cities in ", "Towns in ", "Villages in ", "Municipalities of ", "Municipalities in ", "Rivers of ", "Rivers in ",
                           "Islands of ", "Islands in ", "Mountains of ", "Lakes of ", "Provinces of ", "Counties of ", "Districts of ",
                           "Regions of ", "Cities and towns in ", "Capitals in ", "States of ", "Cantons of "}, rest)) {
            fl.place = true;
            if (rest.compare(0, 4, "the ") == 0) rest.erase(0, 4);
            if (!rest.empty() && std::isupper(static_cast<unsigned char>(rest[0])) && rest.find(" by ") == npos) {
                Obj o;
                o.surf = rest;
                o.k1 = o.k2 = norm_ent(rest);
                add_seed(title, rel_index("located_in"), o, 2, "category", off);
            }
            continue;
        }
        if (starts_any(c, {"Countries in ", "Countries of ", "Capitals of ", "Geography of ", "Islands", "Rivers", "Mountains"}, rest)) fl.place = true;
        if (starts_any(c, {"Countries in ", "Countries of ", "States of ", "Provinces of "}, rest)) fl.country = true;
        {
            struct CP { const char* pre; const char* rel; };
            static const CP cps[] = {{"Natives of ", "birth_place"}, {"People from ", "birth_place"}, {"Novels by ", "author"}, {"Books by ", "author"},
                                     {"Plays by ", "author"}, {"Poems by ", "author"}, {"Short stories by ", "author"}, {"Operas by ", "composer"},
                                     {"Compositions by ", "composer"}, {"Symphonies by ", "composer"}, {"Films directed by ", "director"},
                                     {"Paintings by ", "creator"}, {"Albums by ", "creator"}, {"Songs by ", "creator"}};
            bool hit = false;
            for (const CP& cp : cps) {
                std::size_t n = std::strlen(cp.pre);
                if (c.compare(0, n, cp.pre) != 0 || c.size() <= n) continue;
                string v = c.substr(n);
                if (v.compare(0, 4, "the ") == 0) v.erase(0, 4);
                if (!std::isupper(static_cast<unsigned char>(v[0]))) break;
                Obj o;
                o.surf = v;
                o.k1 = o.k2 = norm_ent(v);
                add_seed(title, rel_index(cp.rel), o, 2, "category", off);
                hit = true;
                if (std::strcmp(cp.rel, "birth_place") == 0) fl.person = true;
                break;
            }
            if (hit) continue;
        }
        std::size_t sp = c.find(' ');
        if (sp != npos && sp > 3) {
            string adj = c.substr(0, sp), noun = c.substr(sp + 1);
            if (std::isupper(static_cast<unsigned char>(adj[0])) && occ.count(noun) && adj.find('-') == npos) {
                Obj a, b;
                a.surf = adj; a.k1 = a.k2 = norm_ent(adj);
                string sing = noun == "actresses" ? "actress" : noun.substr(0, noun.size() - 1);
                b.surf = sing; b.k1 = b.k2 = norm_ent(sing);
                add_seed(title, rel_index("nationality"), a, 2, "category", off);
                add_seed(title, rel_index("occupation"), b, 2, "category", off);
                fl.person = true;
            }
        }
    }
}

void lead_type(const vector<Sent>& sents, PageFlags& fl) {
    static const std::unordered_set<string> PL = {"city", "town", "village", "country", "state", "province", "region", "island", "river", "lake", "mountain",
                                                  "municipality", "county", "capital", "republic", "kingdom", "territory", "nation", "district", "canton",
                                                  "department", "peninsula", "sea", "ocean", "continent", "archipelago", "borough", "prefecture", "commune",
                                                  "suburb", "locality", "settlement", "metropolis", "desert", "valley", "seaport", "port"};
    static const std::unordered_set<string> PE = {"painter", "writer", "author", "poet", "novelist", "composer", "philosopher", "mathematician", "physicist",
                                                  "actor", "actress", "singer", "musician", "politician", "president", "scientist", "emperor", "king", "queen",
                                                  "general", "director", "playwright", "chemist", "economist", "historian", "astronomer", "architect",
                                                  "engineer", "inventor", "explorer", "sculptor", "dramatist", "journalist", "theologian", "biologist", "pope",
                                                  "saint", "bishop", "prime", "minister", "footballer", "player", "cricketer", "athlete", "comedian"};
    if (sents.empty()) return;
    const Sent& s = sents[0];
    int n = static_cast<int>(s.t.size());
    int k = -1;
    for (int i = 0; i < n && i < 40; ++i) {
        const string& w = s.t[static_cast<std::size_t>(i)].w;
        if (w == "is" || w == "was" || w == "are" || w == "were") { k = i; break; }
    }
    if (k < 0) return;
    bool pl = false, pe = false;
    static const std::unordered_set<string> stop = {".", ",", ";", "(", "that", "which", "who", "whose", "where", "of", "in", "on", "at", "by", "for", "from",
                                                    "to", "with", "during", "after", "before", "as"};
    for (int i = k + 1; i < n && i <= k + 8; ++i) {  // the predicate nominal: "was a German philosopher", up to the first preposition
        const string& w = s.t[static_cast<std::size_t>(i)].w;
        if (stop.count(w)) break;
        if (PL.count(w)) pl = true;
        if (PE.count(w)) pe = true;
    }
    if (pl && !pe) fl.place = true;
    if (pe && !pl) fl.person = true;
}

// Pass 1 for one page: template fields, table rows, categories, lead type.
void scan_page(const string& data, const PageRef& pg, PageFlags& fl) {
    const string raw = data.substr(pg.tb, pg.te - pg.tb);
    int depth = 0, table_depth = 0;
    string tpl;
    bool have_pending = false;
    string pending_key;
    std::size_t pos = 0;
    static const char* const place_tpl[] = {"country", "city", "us state", "settlement", "town", "province", "river", "island", "lake", "mountain", "municipality", "county"};
    static const char* const person_tpl[] = {"biography", "president", "infobox pm", "celebrity", "philosopher", "person", "musician", "actor", "writer", "scientist", "persondata", "politician", "athlete"};
    auto emit = [&](const string& key, const string& val, int src, std::uint64_t off) {
        int r = key_to_rel(flat_key(key), tpl);
        if (r < 0) return;
        for (const Obj& o : make_objs(r, val)) add_seed(pg.title, r, o, src, src == 0 ? tpl : "table", off);
    };
    while (pos <= raw.size()) {
        std::size_t nl = raw.find('\n', pos);
        if (nl == npos) nl = raw.size();
        string nc = xml_decode(raw.substr(pos, nl - pos));
        erase_refs(nc);
        string t = trim(nc);
        int delta = count_of(nc, "{{") - count_of(nc, "}}");
        std::uint64_t off = pg.tb + pos;
        bool starts_tpl = t.compare(0, 2, "{{") == 0;
        std::size_t open_at = starts_tpl ? nc.find("{{") : npos;
        if (!starts_tpl && delta > 0) {  // "* {{Book reference |" left open mid-line: name it, so its fields are not read under a stale name
            vector<std::size_t> st;
            for (std::size_t q = 0; q + 1 < nc.size(); ++q) {
                if (nc.compare(q, 2, "{{") == 0) st.push_back(q++);
                else if (nc.compare(q, 2, "}}") == 0 && !st.empty()) { st.pop_back(); ++q; }
            }
            if (!st.empty()) open_at = st.front();
        }
        if (depth == 0 && open_at != npos) {
            std::size_t e = nc.find('|', open_at + 2);
            string nm = nc.substr(open_at + 2, e == npos ? npos : e - open_at - 2);
            std::size_t cl = nm.find("}}");
            if (cl != npos) nm = nm.substr(0, cl);
            nm = trim(nm);
            for (char& c : nm) if (c == '_') c = ' ';
            tpl = lower(nm);
            for (const char* k : place_tpl) if (tpl.find(k) != npos && tpl.find("infobox") != npos) fl.place = true;
            if ((tpl.find("country") != npos && tpl.find("infobox") != npos) || tpl == "us state") fl.country = fl.place = true;
            for (const char* k : person_tpl) if (tpl.find(k) != npos && (tpl.find("infobox") != npos || tpl == "persondata")) fl.person = true;
        }
        if (t.compare(0, 2, "{|") == 0) ++table_depth;
        bool table_end = t.compare(0, 2, "|}") == 0;
        if (depth > 0 || starts_tpl) {
            if (!cite_like_tpl(tpl)) {
                vector<string> pieces = split_top(nc);
                for (std::size_t k = 0; k < pieces.size(); ++k) {
                    string piece = pieces[k];
                    if (k + 1 == pieces.size() && depth + delta <= 0) {
                        string tp = trim(piece);
                        while (tp.size() >= 2 && tp.compare(tp.size() - 2, 2, "}}") == 0) tp.erase(tp.size() - 2);
                        piece = tp;
                    }
                    std::size_t eq = piece.find('=');
                    if (eq == npos) continue;
                    string key = trim(piece.substr(0, eq));
                    if (key.empty() || key.size() > 32 || key.find_first_of("[{<\"'") != npos) continue;
                    emit(key, trim(piece.substr(eq + 1)), 0, off);
                }
            }
            have_pending = false;
        } else if (t.size() > 1 && t[0] == '|' && t[1] != '-' && t[1] != '+' && t[1] != '}') {
            string body = t.substr(1);
            std::size_t dbl = body.find("||");
            auto key_ok = [](const string& kr) {
                if (kr.empty() || kr.size() > 30 || !std::isalpha(static_cast<unsigned char>(kr[0]))) return false;
                int letters = 0;
                for (unsigned char c : kr) { if (std::isalpha(c)) ++letters; else if (c != ' ' && c != '-' && c != '/') return false; }
                return letters >= 4;
            };
            if (dbl != npos) {
                string k = trim(plain_of_links(strip_tags(body.substr(0, dbl))));
                while (!k.empty() && (k.back() == ':' || k.back() == ' ')) k.pop_back();
                string v = body.substr(dbl + 2);
                std::size_t d2 = v.find("||");
                if (d2 != npos) v = v.substr(0, d2);
                if (key_ok(k) && body.substr(0, dbl).find('=') == npos && (!raw_links(v).empty() || is_datey(std::max(0, key_to_rel(flat_key(k), tpl))))) emit(k, v, 1, off);
                have_pending = false;
            } else if (have_pending) {
                string v = body;
                std::size_t bar = v.find('|');
                if (bar != npos && v.find("[[") > bar) v = v.substr(bar + 1);
                if (!raw_links(v).empty()) emit(pending_key, v, 1, off);
                have_pending = false;
            } else {
                string k = trim(plain_of_links(strip_tags(body)));
                if (!k.empty() && k.back() == ':') {
                    k.pop_back();
                    k = trim(k);
                    if (key_ok(k)) { pending_key = k; have_pending = true; }
                } else {
                    have_pending = false;
                }
            }
        } else {
            have_pending = false;
        }
        depth = std::max(0, depth + delta);
        if (table_end) table_depth = std::max(0, table_depth - 1);
        if (nl >= raw.size()) break;
        pos = nl + 1;
    }
    category_seeds(pg.title, raw, pg.tb, fl);
    vector<Sent> ss;
    page_sentences(data, pg, ss, nullptr, 1);
    lead_type(ss, fl);
}

}  // namespace

namespace {

// ============================================================ learning, reliability, extraction
struct Pat {
    int rel = 0;
    string key, family;
    int train_subj = 0, train_sent = 0, checks = 0, correct = 0, unchecked = 0;
    double prec = 0, lo = 0, reliability = 0;  // reliability = Beta(1+correct, 1+wrong) posterior mean
    bool kept = false;
};
struct TrainCnt { int subj = 0, sent = 0; };
struct PStat { int train_subj = 0, train_sent = 0, checks = 0, correct = 0, unchecked = 0; };

vector<Pat> g_pats;                                   // patterns of the current stage
double g_base[64];                                    // base rate a per relation (PLAN 7.2)
double g_phi[64];                                     // --phi overrides of the hand-set functionality (< 0: kRel)
std::unordered_map<string, vector<int>> g_patmap;     // pattern text -> ids in g_pats

bool heldout(const string& subj, int K, int fold) { return static_cast<int>((fnv(subj) >> 20) % static_cast<std::uint64_t>(K)) == fold; }

bool obj_matches(int r, const Cand& c, const Obj& o) {
    if (kRel[r].type == T_DATE) return c.kind == K_DATE && date_compat(c.d, o.d);
    if (kRel[r].type == T_YEAR) return c.kind == K_DATE && c.d.ok() && c.d.y == o.d.y;
    for (const string* a : {&c.k1, &c.k2, &c.k3}) {
        if (a->empty()) continue;
        for (const string* b : {&o.k1, &o.k2, &o.k3}) if (*a == *b) return true;
        for (const string& b : o.alt) if (*a == b) return true;
    }
    return false;
}
// Place -> parents from the located_in seeds ("Detroit" -> "Michigan"). Used only to score patterns against coarse gold.
std::unordered_map<string, std::unordered_set<string>> g_loc;
bool obj_consistent(int r, const Cand& c, const Obj& o) {
    if (obj_matches(r, c, o)) return true;
    if (kRel[r].type != T_PLACE && kRel[r].type != T_PLACE_LINK) return false;
    for (const string* a : {&c.k1, &c.k2, &c.k3}) {
        auto it = g_loc.find(*a);
        if (it == g_loc.end()) continue;
        for (const string* b : {&o.k1, &o.k2, &o.k3}) if (!b->empty() && it->second.count(*b)) return true;
        for (const string& b : o.alt) if (it->second.count(b)) return true;
    }
    return false;
}
string obj_key(int r, const Cand& c) {
    if (kRel[r].type == T_YEAR) return std::to_string(c.d.y);
    return c.kind == K_DATE ? fmt_date(c.d) : c.k1;
}
string obj_surface(int r, const Cand& c) {
    if (kRel[r].type == T_YEAR) { Date D; D.y = c.d.y; return fmt_date(D); }
    return c.surf;
}

struct PageCtx {
    vector<Sent> sents;
    SubjForms forms;
};
void load_page(const string& data, const PageRef& pg, PageCtx& px) {
    px.sents.clear();
    vector<string> bold;
    page_sentences(data, pg, px.sents, &bold);
    string nk = norm_ent(pg.title);
    g_cur_person = g_person.count(nk) > 0;
    g_cur_place = g_place.count(nk) > 0;
    g_cur_country = g_country.count(nk) > 0;
    px.forms = make_forms(pg.title, bold, g_cur_person);
}

// Apply the patterns in g_patmap to one sentence: callback(pattern id, candidate, subject span, key text).
template <class F>
void apply_patterns(const Sent& s, const vector<Span>& spans, const vector<Cand>& cands, const string& title_key, F&& cb) {
    std::array<string, 4> keys;
    for (const Span& S : spans)
        for (const Cand& c : cands) {
            if (c.b < S.e && S.b < c.e) continue;
            if (c.k1 == title_key) continue;
            bool sf;
            if (!make_keys(s, S, c.b, c.e, keys, sf)) continue;
            for (const string& k : keys) {
                auto it = g_patmap.find(k);
                if (it == g_patmap.end()) continue;
                for (int pid : it->second) {
                    int rr = g_pats[static_cast<std::size_t>(pid)].rel;
                    if (subject_ok(rr) && type_ok(rr, c)) cb(pid, c, S);
                }
            }
        }
}

// Step 2: patterns from the training seeds of one page.
void train_page(const string& data, const PageRef& pg, const vector<int>& sidx, std::unordered_map<string, TrainCnt>& tm) {
    PageCtx px;
    load_page(data, pg, px);
    std::unordered_set<string> page_keys;
    vector<Cand> cands;
    std::array<string, 4> keys;
    const string tk = px.forms.title_key;
    for (const Sent& s : px.sents) {
        auto spans = find_subject(s, px.forms);
        if (spans.empty()) continue;
        gen_cands(s, cands);
        for (int si : sidx) {
            const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
            for (const Cand& c : cands) {
                if (!subject_ok(sd.rel) || !type_ok(sd.rel, c) || !obj_matches(sd.rel, c, sd.o)) continue;
                for (const Span& S : spans) {
                    bool sf;
                    if (c.b < S.e && S.b < c.e) continue;
                    if (!make_keys(s, S, c.b, c.e, keys, sf)) continue;
                    for (const string& k : keys) {
                        string pk = std::to_string(sd.rel) + "\t" + k;
                        ++tm[pk].sent;
                        page_keys.insert(pk);
                    }
                }
            }
        }
    }
    for (const string& pk : page_keys) ++tm[pk].subj;
}

// Steps 3: apply candidate patterns to the held-out subjects' articles and count correct extractions.
void eval_page(const string& data, const PageRef& pg, std::unordered_map<string, PStat>& stat) {
    PageCtx px;
    load_page(data, pg, px);
    auto sit = g_by_subj.find(pg.title);
    if (sit == g_by_subj.end()) return;
    std::set<std::pair<int, string>> seen;
    vector<Cand> cands;
    for (const Sent& s : px.sents) {
        auto spans = find_subject(s, px.forms);
        if (spans.empty()) continue;
        gen_cands(s, cands);
        apply_patterns(s, spans, cands, px.forms.title_key, [&](int pid, const Cand& c, const Span&) {
            const Pat& p = g_pats[static_cast<std::size_t>(pid)];
            if (!seen.insert({pid, obj_key(p.rel, c)}).second) return;
            bool known = false, ok = false;
            for (int si : sit->second) {
                const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                if (sd.rel != p.rel) continue;
                known = true;
                if (obj_consistent(p.rel, c, sd.o)) { ok = true; break; }
            }
            PStat& st = stat[std::to_string(p.rel) + "\t" + p.key];
            if (!known) { ++st.unchecked; return; }
            ++st.checks;
            if (ok) ++st.correct;
        });
    }
}

// ============================================================ extraction and ECS
struct EvRec { int pid; double rel; std::uint64_t off, shash; int sidx; string family; };
struct FactAcc {
    string subj, surf, okey;
    int rel = 0;
    Date d;
    vector<EvRec> ev;
    int total = 0;
    string snippet;
    double snip_rel = -1;
};
std::unordered_map<string, FactAcc> g_new;

string sentence_text(const Sent& s) {
    string o;
    for (std::size_t i = 0; i < s.t.size(); ++i) {
        if (i && s.t[i].w != "," && s.t[i].w != "." && s.t[i].w != ")" && s.t[i - 1].w != "(") o += ' ';
        o += s.t[i].o;
    }
    return o.size() > 260 ? o.substr(0, 260) + "..." : o;
}

void extract_page(const string& data, const PageRef& pg) {
    PageCtx px;
    load_page(data, pg, px);
    auto sit = g_by_subj.find(pg.title);
    vector<Cand> cands;
    for (const Sent& s : px.sents) {
        auto spans = find_subject(s, px.forms);
        if (spans.empty()) continue;
        gen_cands(s, cands);
        apply_patterns(s, spans, cands, px.forms.title_key, [&](int pid, const Cand& c, const Span&) {
            const Pat& p = g_pats[static_cast<std::size_t>(pid)];
            if (sit != g_by_subj.end())
                for (int si : sit->second) {
                    const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                    if (sd.src < 3 && sd.rel == p.rel && obj_matches(p.rel, c, sd.o)) return;  // already a text seed
                }
            string fk = pg.title + "\x1f" + std::to_string(p.rel) + "\x1f" + obj_key(p.rel, c);
            FactAcc& f = g_new[fk];
            if (f.subj.empty()) { f.subj = pg.title; f.rel = p.rel; f.surf = obj_surface(p.rel, c); f.okey = obj_key(p.rel, c); f.d = c.d; }
            ++f.total;
            if (f.ev.size() < 96) f.ev.push_back({pid, p.reliability, s.off, s.hash, s.idx, p.family});
            if (p.reliability > f.snip_rel) { f.snip_rel = p.reliability; f.snippet = sentence_text(s); }
        });
    }
}

constexpr double kTauText = 20.0 / 21.0;  // Wikipedia prose treated as infobox-grade (PLAN 7.1 prior Beta(20,1)); revisit with audits

struct Fused {
    const FactAcc* f = nullptr;
    ecs::Opinion op;
    int n_ev = 0;
    string best_pattern;
    std::uint64_t best_off = 0;
};

// Evidence items of one fact: the best pattern per (sentence, pattern family), repeats of the same sentence text weighted by (1+ln n)/n.
vector<ecs::EvidenceItem> make_items(const FactAcc& f, const vector<Pat>& pats, int* best_i = nullptr, EvRec* best_ev = nullptr) {
    std::map<std::tuple<std::uint64_t, int, string>, EvRec> best;
    for (const EvRec& e : f.ev) {
        auto k = std::make_tuple(e.off, e.sidx, e.family);
        auto it = best.find(k);
        if (it == best.end() || e.rel > it->second.rel) best[k] = e;
    }
    vector<EvRec> items;
    for (auto& kv : best) items.push_back(kv.second);
    std::unordered_map<std::uint64_t, int> cid;
    vector<int> ids;
    for (auto& e : items) ids.push_back(cid.emplace(e.shash, static_cast<int>(cid.size())).first->second);
    auto om = ecs::independence_weights(ids);
    vector<ecs::EvidenceItem> out;
    double bm = -1;
    for (std::size_t i = 0; i < items.size(); ++i) {
        ecs::EvidenceItem e;
        e.stance = ecs::Stance::Support;
        e.omega = om[i];
        e.tau = kTauText;
        e.pi = items[i].rel;
        if (e.mass() > bm) { bm = e.mass(); if (best_i) *best_i = items[i].pid; if (best_ev) *best_ev = items[i]; }
        out.push_back(e);
    }
    (void)pats;
    return out;
}

vector<Fused> fuse_all() {
    // group facts by (subject, relation)
    std::unordered_map<string, vector<const FactAcc*>> groups;
    for (auto& kv : g_new) groups[kv.second.subj + "\x1f" + std::to_string(kv.second.rel)].push_back(&kv.second);
    vector<Fused> out;
    const double seed_tau = ecs::prior_infobox().mean();
    for (auto& g : groups) {
        vector<vector<ecs::EvidenceItem>> items;
        for (const FactAcc* f : g.second) items.push_back(make_items(*f, g_pats));
        const int r = g.second[0]->rel;
        auto sit = g_by_subj.find(g.second[0]->subj);
        for (std::size_t i = 0; i < g.second.size(); ++i) {
            const FactAcc& f = *g.second[i];
            vector<ecs::EvidenceItem> all = items[i];
            // competing values: other extracted values and seed values for the same subject and relation
            for (std::size_t j = 0; j < g.second.size(); ++j) {
                if (j == i) continue;
                const FactAcc& h = *g.second[j];
                bool compat = is_datey(r) ? (kRel[r].type == T_YEAR ? h.d.y == f.d.y : date_compat(h.d, f.d)) : h.okey == f.okey;
                if (compat) continue;
                for (auto e : items[j]) { e.stance = ecs::Stance::CompetingValue; all.push_back(e); }
            }
            int hard = 0;
            if (sit != g_by_subj.end()) {
                for (int si : sit->second) {
                    const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                    if (sd.src == 3) continue;  // Wikidata values never vote on a fact
                    if (sd.rel == r) {
                        bool compat = is_datey(r) ? (kRel[r].type == T_YEAR ? sd.o.d.y == f.d.y : date_compat(sd.o.d, f.d)) : sd.o.k1 == f.okey || sd.o.k2 == f.okey;
                        if (!compat) {
                            ecs::EvidenceItem e;
                            e.stance = ecs::Stance::CompetingValue;
                            e.tau = seed_tau;
                            all.push_back(e);
                        }
                    }
                    // lifespan sanity: death before birth, or a life longer than 125 years, is a hard violation
                    bool is_birth = kRel[r].name == string("birth_year") || kRel[r].name == string("birth_date");
                    bool is_death = kRel[r].name == string("death_year") || kRel[r].name == string("death_date");
                    string sn = kRel[sd.rel].name;
                    if (is_birth && (sn == "death_year" || sn == "death_date") && (sd.o.d.y < f.d.y || sd.o.d.y - f.d.y > 125)) ++hard;
                    if (is_death && (sn == "birth_year" || sn == "birth_date") && (f.d.y < sd.o.d.y || f.d.y - sd.o.d.y > 125)) ++hard;
                }
            }
            ecs::FusionConfig cfg;
            cfg.phi = g_phi[r] >= 0 ? g_phi[r] : kRel[r].phi;
            cfg.base_rate = g_base[r];
            cfg.hard_violations = hard;
            Fused fu;
            fu.f = &f;
            fu.op = ecs::fuse(all, cfg);
            int bi = -1;
            EvRec be{};
            make_items(f, g_pats, &bi, &be);
            fu.n_ev = static_cast<int>(items[i].size());
            if (bi >= 0) { fu.best_pattern = g_pats[static_cast<std::size_t>(bi)].key; fu.best_off = be.off; }
            out.push_back(fu);
        }
    }
    std::sort(out.begin(), out.end(), [](const Fused& a, const Fused& b) {
        if (a.op.P != b.op.P) return a.op.P > b.op.P;
        if (a.f->subj != b.f->subj) return a.f->subj < b.f->subj;
        return a.f->okey < b.f->okey;
    });
    return out;
}

}  // namespace

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
double peak_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
}
bool read_file(const string& path, string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    f.read(&out[0], static_cast<std::streamsize>(out.size()));
    return true;
}
// Main-namespace, non-redirect pages with their text byte range.
vector<PageRef> list_pages(const string& data, long& total, long& redirects) {
    vector<PageRef> out;
    std::size_t pos = 0;
    while (true) {
        std::size_t pb = data.find("<page>", pos);
        if (pb == npos) break;
        std::size_t pe = data.find("</page>", pb);
        if (pe == npos) break;
        pos = pe + 7;
        ++total;
        std::size_t tb = data.find("<title>", pb), te = data.find("</title>", tb);
        if (tb == npos || te == npos || te > pe) continue;
        string title = xml_decode(data.substr(tb + 7, te - tb - 7));
        std::size_t xb = data.find("<text", te);
        if (xb == npos || xb > pe) continue;
        std::size_t xg = data.find('>', xb);
        if (xg == npos || xg > pe || data[xg - 1] == '/') continue;
        std::size_t xe = data.find("</text>", xg);
        if (xe == npos || xe > pe) continue;
        string head = lower(trim(data.substr(xg + 1, std::min<std::size_t>(40, xe - xg - 1))));
        if (head.compare(0, 9, "#redirect") == 0) { ++redirects; continue; }
        std::size_t colon = title.find(':');
        if (colon != npos && colon + 1 < title.size() && title[colon + 1] != ' ') continue;  // namespace page
        out.push_back({title, xg + 1, xe});
    }
    return out;
}

// ============================================================ D4: Wikidata values (distant supervision only)
// Row: subject title \t relation \t surface|surface|... (label, enwiki title, aliases; dates YYYY-MM-DD, "Month YYYY", YYYY or
// "N BC") \t subject item \t value item. Subjects in the gold fifth (by hash) are never seeds: their values only score new facts.
// A Wikidata value is never a fact: it can only choose and check patterns (and widen the type lexicons).
std::unordered_map<string, vector<Obj>> g_gold;  // subject \x1f relation -> Wikidata values used only for scoring
bool wd_gold_subject(const string& subj) { return (fnv(subj) >> 40) % 5 == 0; }
bool load_wikidata(const string& path, bool gold_only, const std::unordered_map<string, int>& page_of, long& n_seed, long& n_gold) {
    std::ifstream f(path);
    string line;
    if (!f) return false;
    while (std::getline(f, line)) {
        vector<string> c = split(line, '\t');
        int r = c.size() >= 3 ? rel_index(c[1]) : -1;
        if (r < 0 || !page_of.count(c[0])) continue;
        vector<string> sf = split(c[2], '|');
        Obj o;
        o.surf = sf[0];
        if (is_datey(r)) {
            o.d = parse_date_str(sf[0]);
            if (!o.d.ok() || o.d.y > 2006) continue;
            if (kRel[r].type == T_YEAR) o.d.m = o.d.d = 0;
            o.surf = o.k1 = o.k2 = fmt_date(o.d);
        } else {
            vector<string> keys;
            for (const string& x : sf) {
                string k = norm_ent(x);
                if (!k.empty() && std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
            }
            if (keys.empty()) continue;
            o.k1 = keys[0];
            if (keys.size() > 1) o.k2 = keys[1];
            if (keys.size() > 2) o.k3 = keys[2];
            if (keys.size() > 3) o.alt.assign(keys.begin() + 3, keys.end());
        }
        if (gold_only || wd_gold_subject(c[0])) { g_gold[c[0] + "\x1f" + c[1]].push_back(o); ++n_gold; }
        else { add_seed(c[0], r, o, 3, "wikidata", 0); ++n_seed; }
    }
    return true;
}
// New facts against held-out Wikidata values: right if one value agrees (a place inside it counts, as for the pattern checks),
// coarse if a Wikidata place lies inside the fact's place (located_in, up to 3 steps), wrong if the subject has values for the
// relation and none agrees, unchecked (none) otherwise. Writes facts_wd_check.tsv.
bool inside(const string& a, const string& b, int depth) {
    auto it = g_loc.find(a);
    if (a.empty() || it == g_loc.end() || depth == 0) return false;
    for (const string& p : it->second) if (p == b || inside(p, b, depth - 1)) return true;
    return false;
}
void score_gold(const vector<Fused>& fused, const string& outdir, const std::function<void(const string&)>& say) {
    std::ofstream f(outdir + "/facts_wd_check.tsv");
    f << "subject\trelation\tobject\tP\twikidata\twikidata_values\tgold_fifth\n";
    std::map<int, std::array<int, 5>> n;  // admitted right, admitted checks, candidate right, candidate checks, admitted coarse
    for (const Fused& fu : fused) {
        if (fu.op.P < 0.5) continue;
        const FactAcc& fa = *fu.f;
        auto it = g_gold.find(fa.subj + "\x1f" + kRel[fa.rel].name);
        string st = "none", vals;
        if (it != g_gold.end()) {
            Cand c;
            c.kind = is_datey(fa.rel) ? K_DATE : K_LINK;
            c.d = fa.d;
            c.k1 = fa.okey;
            c.k2 = norm_ent(fa.surf);
            st = "wrong";
            for (const Obj& o : it->second) {
                vals += (vals.empty() ? "" : "; ") + o.surf;
                if (obj_consistent(fa.rel, c, o)) st = "right";
            }
            for (const Obj& o : it->second)
                for (const string* k : {&o.k1, &o.k2, &o.k3})
                    if (st == "wrong" && (inside(*k, c.k1, 3) || inside(*k, c.k2, 3))) st = "coarse";
            auto& a = n[fa.rel];
            int k = fu.op.P >= 0.8 ? 0 : 2;
            a[static_cast<std::size_t>(k) + 1]++;
            if (st == "right") a[static_cast<std::size_t>(k)]++;
            if (st == "coarse" && k == 0) a[4]++;
        }
        f << fa.subj << '\t' << kRel[fa.rel].name << '\t' << fa.surf << '\t' << fu.op.P << '\t' << st << '\t' << vals << '\t' << wd_gold_subject(fa.subj) << '\n';
    }
    say("\nWIKIDATA CHECK of new facts on gold subjects (admitted P>=0.8: right/checked, Wilson 95%, coarse | candidates 0.5..0.8: right/checked)");
    char b[256];
    std::array<int, 5> t{};
    for (auto& [r, a] : n) {
        std::snprintf(b, sizeof b, "  %-13s %4d/%-4d [%.3f, %.3f] coarse %d | %d/%d", kRel[r].name, a[0], a[1], ecs::wilson_lower(a[0], a[1]),
                      ecs::wilson_upper(a[0], a[1]), a[4], a[2], a[3]);
        say(b);
        for (std::size_t k = 0; k < 5; ++k) t[k] += a[k];
    }
    std::snprintf(b, sizeof b, "  %-13s %4d/%-4d [%.3f, %.3f] coarse %d | %d/%d", "TOTAL", t[0], t[1], ecs::wilson_lower(t[0], t[1]),
                  ecs::wilson_upper(t[0], t[1]), t[4], t[2], t[3]);
    say(b);
}

// ============================================================ D4: the fact-text shard (TOOL_PROFILE P1, CYPHALM_ANSWERING R1)
// The fact store (text seeds + admitted facts, P >= 0.8) written as question-form turns and plain sentences for CyphaLM.
// Questions: the cyphalm_trie dktext template first, then two variants. {E} subject, {O} object, {at} on/in, {a} a/an.
struct QTpl { const char* rel; const char* q[3]; const char* stmt; };
const QTpl kQT[] = {
    {"birth_date", {"When was {E} born?", "What is {E}'s date of birth?", "On what date was {E} born?"}, "{E} was born {at} {O}."},
    {"birth_year", {"When was {E} born?", "In what year was {E} born?", "What year was {E} born?"}, "{E} was born in {O}."},
    {"death_date", {"When did {E} die?", "What is {E}'s date of death?", "On what date did {E} die?"}, "{E} died {at} {O}."},
    {"death_year", {"When did {E} die?", "In what year did {E} die?", "What year did {E} die?"}, "{E} died in {O}."},
    {"birth_place", {"Where was {E} born?", "What is {E}'s place of birth?", "In what place was {E} born?"}, "{E} was born in {O}."},
    {"death_place", {"Where did {E} die?", "What is {E}'s place of death?", "In what place did {E} die?"}, "{E} died in {O}."},
    {"capital", {"What is the capital of {E}?", "What is {E}'s capital?", "Which city is the capital of {E}?"}, "The capital of {E} is {O}."},
    {"largest_city", {"What is the largest city in {E}?", "Which is the largest city of {E}?", "What is {E}'s largest city?"}, "The largest city in {E} is {O}."},
    {"country", {"In which country is {E}?", "What country is {E} in?", "Which country is {E} located in?"}, "{E} is in {O}."},
    {"location", {"Where is {E}?", "Where is {E} located?", "Where is {E} based?"}, "{E} is located in {O}."},
    {"located_in", {"Where is {E}?", "Where is {E} located?", "What region is {E} in?"}, "{E} is in {O}."},
    {"founded", {"When was {E} founded?", "In what year was {E} founded?", "When was {E} established?"}, "{E} was founded in {O}."},
    {"author", {"Who wrote {E}?", "Who is the author of {E}?", "Who was {E} written by?"}, "{E} was written by {O}."},
    {"director", {"Who directed {E}?", "Who was the director of {E}?", "Who was {E} directed by?"}, "{E} was directed by {O}."},
    {"composer", {"Who composed {E}?", "Who was the composer of {E}?", "Who wrote the music for {E}?"}, "{E} was composed by {O}."},
    {"founder", {"Who founded {E}?", "Who was the founder of {E}?", "Who established {E}?"}, "{E} was founded by {O}."},
    {"creator", {"Who created {E}?", "Who was the creator of {E}?", "Who made {E}?"}, "{E} was created by {O}."},
    {"inventor", {"Who invented {E}?", "Who was the inventor of {E}?", "Who is credited with inventing {E}?"}, "{E} was invented by {O}."},
    {"spouse", {"Who was {E} married to?", "Who was {E}'s spouse?", "Whom did {E} marry?"}, "{E} was married to {O}."},
    {"predecessor", {"Who preceded {E}?", "Who was {E}'s predecessor?", "Who came before {E}?"}, "{E} was preceded by {O}."},
    {"successor", {"Who succeeded {E}?", "Who was {E}'s successor?", "Who came after {E}?"}, "{E} was succeeded by {O}."},
    {"developer", {"Who developed {E}?", "Who is the developer of {E}?", "Which company developed {E}?"}, "{E} was developed by {O}."},
    {"publisher", {"Who published {E}?", "Who is the publisher of {E}?", "Which company published {E}?"}, "{E} was published by {O}."},
    {"owner", {"Who owns {E}?", "Who is the owner of {E}?", "Who is {E} owned by?"}, "{E} is owned by {O}."},
    {"genre", {"What genre is {E}?", "What is the genre of {E}?", "To what genre does {E} belong?"}, "{E} belongs to the {O} genre."},
    {"language", {"What is the official language of {E}?", "What language is spoken in {E}?", "Which language is official in {E}?"}, "The official language of {E} is {O}."},
    {"currency", {"What is the currency of {E}?", "What currency is used in {E}?", "What is {E}'s currency?"}, "The currency of {E} is {O}."},
    {"occupation", {"What was {E}'s occupation?", "What did {E} do for a living?", "What was {E}'s profession?"}, "{E} was {a} {O}."},
    {"nationality", {"What nationality was {E}?", "What was {E}'s nationality?", "What was the nationality of {E}?"}, "{E}'s nationality was {O}."},
    {"party", {"Which party did {E} belong to?", "What political party was {E} a member of?", "What was {E}'s party?"}, "{E} was a member of the {O}."},
    {"family", {"What family does {E} belong to?", "In which family is {E}?", "What is the family of {E}?"}, "{E} belongs to the family {O}."},
    {"order", {"What order does {E} belong to?", "In which order is {E}?", "What is the order of {E}?"}, "{E} belongs to the order {O}."},
    {"class", {"What class does {E} belong to?", "In which class is {E}?", "What is the class of {E}?"}, "{E} belongs to the class {O}."}};
const QTpl kLangOther = {"language", {"What language is {E} in?", "In what language is {E}?", "What is the language of {E}?"}, "The language of {E} is {O}."};

string human_date(const Date& D) {  // the dump's own style: "April 4, 1932", "April 1932", "235", "490 BC"
    if (D.y < 0) return fmt_date(D);
    if (!D.m) return std::to_string(D.y);
    string m = kMonths[D.m - 1];
    m[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(m[0])));
    return D.d ? m + " " + std::to_string(D.d) + ", " + std::to_string(D.y) : m + " " + std::to_string(D.y);
}
string fill(string t, const string& E, const string& O, const Date& D) {
    replace_all(t, "{E}", E);
    replace_all(t, "{O}", O);
    replace_all(t, "{at}", D.d ? "on" : "in");
    replace_all(t, "{a}", std::strchr("aeiouAEIOU", O.empty() ? 'x' : O[0]) ? "an" : "a");
    if (!t.empty()) t[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(t[0])));
    if (t.size() > 2 && t.compare(t.size() - 2, 2, "..") == 0) t.pop_back();  // "Washington, D.C.."
    return t;
}
// Seed relations left out of the shard: under 0.8 of their audited seed sentences (10, or all when fewer) were true as stated
// (D4 report). Extracted facts of these relations still enter, since their patterns passed the precision gate.
const std::set<string> kShardSkipSeeds = {"country", "location", "language", "nationality", "party", "predecessor", "spouse", "successor"};
// A sentence from the fact's own pattern when its core is "[S] word word ... [O]" (at least two plain words), else "".
string pattern_sentence(const string& key) {
    string core = family_of(key);
    if (core.compare(0, 4, "[S] ") != 0 || core.size() < 8 || core.compare(core.size() - 4, 4, " [O]") != 0) return "";
    string mid = core.substr(4, core.size() - 8);
    int words = 1;
    for (char c : mid) {
        if (c == ' ') ++words;
        else if (!std::islower(static_cast<unsigned char>(c))) return "";
    }
    return words >= 2 ? "{E} " + mid + " {O}." : "";
}
struct TqaQ { string q, title; vector<string> ans; int in_slice = 0; };
// T-QA rows: question, answers, source_title, in_slice. Answers are kept only when asked for (never for the frozen test).
vector<TqaQ> load_tqa(const string& path, bool answers) {
    vector<TqaQ> out;
    std::ifstream f(path);
    string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        vector<string> c = split(line, '\t');
        if (c.size() < 4) continue;
        out.push_back({c[0], c[2], answers ? split(c[1], '|') : vector<string>{}, std::atoi(c[3].c_str())});
    }
    return out;
}
// Relations a T-QA question asks for, from its words (dev coverage report only).
vector<string> q_rels(const string& nq) {
    auto has = [&](const char* w) { return (" " + nq + " ").find(string(" ") + w + " ") != npos; };
    bool when = has("when") || has("year") || has("date");
    if (has("born")) return when ? vector<string>{"birth_date", "birth_year"} : has("where") ? vector<string>{"birth_place"} : vector<string>{};
    if (has("die") || has("died") || has("death")) return when ? vector<string>{"death_date", "death_year"} : vector<string>{"death_place"};
    for (auto& [w, r] : vector<std::pair<const char*, const char*>>{{"capital", "capital"}, {"wrote", "author"}, {"author", "author"}, {"written", "author"},
             {"directed", "director"}, {"director", "director"}, {"composed", "composer"}, {"composer", "composer"}, {"invented", "inventor"},
             {"inventor", "inventor"}, {"painted", "creator"}, {"created", "creator"}, {"creator", "creator"}, {"designed", "creator"},
             {"married", "spouse"}, {"wife", "spouse"}, {"husband", "spouse"}, {"spouse", "spouse"}, {"language", "language"}, {"currency", "currency"}})
        if (has(w)) return {r};
    if (has("founded") || has("founder") || has("established")) return {has("who") ? "founder" : "founded"};
    if (has("country")) return {"country"};
    if (has("where")) return {"location", "located_in", "country"};
    return {};
}
// ============================================================ F2R: raw frames, and the raw-form shard
// The frame of a raw pattern: the key without its ^ / $ ends, [P] read as [S]. Fill [S] with the subject; the bytes before
// [O] (they end with the object's opening markup, e.g. "[[") are the prefix CyphaLM continues. A linked date fills its
// slot as the dump writes it: [[[O]]] -> [[June 14]], [[1856]].
string raw_frame(string k) {
    if (!k.empty() && k[0] == '^') k.erase(0, 1);
    if (!k.empty() && k.back() == '$') k.pop_back();
    replace_all(k, "[P]", "[S]");
    return k;
}
struct ShardOpt { string path, dev, test, join; std::size_t slice_end = 8388608, holdout_from = 7340032; std::uint32_t seed = 0; };
// F2R: the first frame cyphalm_trie --r3 would load for relation r (kept patterns by training subjects, then frame bytes)
// that is a bare core: the subject slot first, the object slot last, no context, no # or <M>. "" when there is none.
string shard_raw_frame(int r) {
    vector<std::tuple<int, string, string>> v;
    for (const Pat& p : g_pats) {
        string f = raw_frame(p.key), core = f, x = f;
        while (!core.empty() && (core[0] == '\'' || core[0] == '[') && core.compare(0, 3, "[S]") != 0) core.erase(0, 1);
        while (!core.empty() && (core.back() == '\'' || core.back() == ']') && core.size() >= 3 && core.compare(core.size() - 3, 3, "[O]") != 0) core.pop_back();
        if (p.rel != r || core.compare(0, 3, "[S]") != 0 || core.size() < 7 || core.compare(core.size() - 3, 3, "[O]") != 0 ||
            f.find('#') != npos || f.find("<M>") != npos || p.key[0] == '^' || p.key.back() == '$')
            continue;
        replace_all(x, "[S]", "\x01");
        replace_all(x, "[O]", "\x02");
        v.emplace_back(-p.train_subj, x, f);
    }
    std::sort(v.begin(), v.end());
    return v.empty() ? "" : std::get<2>(v[0]);
}
// A fact in a raw frame: the subject (title) in [S], the object in [O] as the dump writes it there (a linked date is
// "March 31]], [[1948"), an open "(" closed, and a full stop.
string raw_fill(const string& frame, const string& E, const string& surf, const Date& D, bool date) {
    string f = frame, O = surf;
    const std::size_t o = f.find("[O]");
    const bool linked = o >= 2 && f.compare(o - 2, 2, "[[") == 0;
    if (date) {
        O = human_date(D);
        if (D.y > 0 && D.m) {
            string m = kMonths[D.m - 1];
            m[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(m[0])));
            O = D.d ? m + " " + std::to_string(D.d) + (linked ? "]], [[" : ", ") + std::to_string(D.y) : m + (linked ? "]] [[" : " ") + std::to_string(D.y);
        }
    }
    f.replace(o, 3, O);
    replace_all(f, "[S]", E);
    if (std::count(f.begin(), f.end(), '(') > std::count(f.begin(), f.end(), ')')) f += ")";
    return f.back() == '.' ? f : f + ".";
}
// Returns 1 when the leakage check fails.
int write_shard(const ShardOpt& O, const vector<Fused>& fused, const vector<PageRef>& pages, const std::unordered_map<string, int>& page_of,
                const std::function<void(const string&)>& say, string* train_text = nullptr) {
    struct SF { string subj, surf, pattern; int rel; Date d; bool seed; };
    vector<SF> facts;
    std::set<string> seen;
    auto add = [&](SF f) {
        if (kRel[f.rel].type == T_YEAR) f.d.m = f.d.d = 0;  // a year fact is a year (before 2026-10-28 the shard wrote 6 with a full date)
        if (seen.insert(f.subj + "\x1f" + kRel[f.rel].name + "\x1f" + (is_datey(f.rel) ? fmt_date(f.d) : norm_ent(f.surf))).second) facts.push_back(f);
    };
    for (const Seed& sd : g_seeds)
        if (sd.src < 3 && !kShardSkipSeeds.count(kRel[sd.rel].name)) add({sd.subj, sd.o.surf, "", sd.rel, sd.o.d, true});
    for (const Fused& fu : fused) if (fu.op.P >= 0.8) add({fu.f->subj, fu.f->surf, fu.best_pattern, fu.f->rel, fu.f->d, false});
    if (!O.join.empty()) {  // F2R --join: another run's admitted facts (its facts_f2.tsv) join the store
        std::ifstream jf(O.join);
        if (!jf) { say("join: cannot read " + O.join); return 1; }
        long n_read = 0;
        const std::size_t before = facts.size();
        string l;
        std::getline(jf, l);
        while (std::getline(jf, l)) {
            auto c = split(l, '\t');
            int r = c.size() >= 7 ? rel_index(c[1]) : -1;
            if (r < 0 || std::atof(c[3].c_str()) < 0.8 || !page_of.count(c[0])) continue;
            ++n_read;
            add({c[0], c[2], c[6], r, is_datey(r) ? parse_date_str(c[2]) : Date{}, false});
        }
        say("join " + O.join + ": admitted facts read " + std::to_string(n_read) + ", new to this run's store " + std::to_string(facts.size() - before));
    }
    std::map<int, string> rawf;  // F2R: in raw mode a fact is written in its relation's first R3 frame, when it has one;
    if (g_raw)                    // not a year: the dump writes "died on [[April 15]], [[1865]]", never "died on [[1865]]"
        for (int r = 0; r < kNRel; ++r)
            if (string f = kRel[r].type == T_YEAR ? "" : shard_raw_frame(r); !f.empty()) { rawf[r] = f; say("  shard_raw_frame " + string(kRel[r].name) + " " + f); }
    std::unordered_set<string> evalq;  // dev and test questions, normalised; the test file's answers are never read
    std::unordered_map<string, int> evalq_src;
    for (int k = 0; k < 2; ++k)
        for (const TqaQ& r : load_tqa(k ? O.test : O.dev, false)) { evalq.insert(norm_q(r.q)); evalq_src[norm_q(r.q)] |= 1 << k; }
    if (evalq.empty()) { say("shard: cannot read the dev/test questions"); return 1; }
    struct Row { string q, ent, rel, src, resp; };
    vector<Row> rows;
    vector<string> sents;
    long region[3] = {0, 0, 0}, from_pattern = 0, dropped_dev = 0, dropped_test = 0, long_resp = 0;
    for (const SF& f : facts) {
        const std::size_t off = pages[static_cast<std::size_t>(page_of.at(f.subj))].tb;
        int reg = off >= O.slice_end ? 2 : off >= O.holdout_from ? 1 : 0;
        ++region[reg];
        if (reg == 2) continue;
        const QTpl* T = nullptr;
        for (const QTpl& t : kQT) if (kRel[f.rel].name == string(t.rel)) T = &t;
        if (!T) continue;
        if (T->rel == string("language") && !g_country.count(norm_ent(f.subj))) T = &kLangOther;
        string E = f.subj;
        std::size_t qp = E.find(" (");
        if (qp != npos && qp > 0 && E.back() == ')') E = E.substr(0, qp);
        const string Ov = is_datey(f.rel) ? human_date(f.d) : f.surf;
        string ps = f.seed || g_raw ? "" : pattern_sentence(f.pattern);
        from_pattern += !ps.empty() || rawf.count(f.rel);
        const string resp = rawf.count(f.rel) ? raw_fill(rawf[f.rel], E, f.surf, f.d, is_datey(f.rel)) : fill(ps.empty() ? T->stmt : ps, E, Ov, f.d);
        long_resp += resp.size() > 80;
        sents.push_back(resp);
        string src = string(f.seed ? "seed" : "extracted") + (reg == 1 ? "_holdout" : "");
        for (const char* q : T->q) {
            string qq = fill(q, E, Ov, f.d);
            auto it = evalq_src.find(norm_q(qq));
            if (it != evalq_src.end()) { dropped_dev += it->second & 1; dropped_test += (it->second >> 1) & 1; continue; }
            rows.push_back({qq, f.subj, kRel[f.rel].name, src, resp});
        }
    }
    std::mt19937_64 prng(O.seed);
    std::shuffle(rows.begin(), rows.end(), prng);
    long leak = 0;  // the final check, on the rows as written
    for (const Row& r : rows) leak += evalq.count(norm_q(r.q)) > 0;
    std::ofstream f(O.path);
    f << "question\tentity\trelation\tsource\tresponse\n";
    std::map<string, int> by_rel, by_src;
    for (const Row& r : rows) {
        f << r.q << '\t' << r.ent << '\t' << r.rel << '\t' << r.src << '\t' << r.resp << '\n';
        ++by_rel[r.rel];
        ++by_src[r.src];
        if (train_text && r.src.find("_holdout") == npos) *train_text += r.resp + "\n";  // what a BPC-clean run trains on
    }
    string sp = O.path.size() > 4 && O.path.compare(O.path.size() - 4, 4, ".tsv") == 0 ? O.path.substr(0, O.path.size() - 4) : O.path;
    std::ofstream fs(sp + "_sentences.txt");
    for (const string& s : sents) fs << s << '\n';
    char b[512];
    std::snprintf(b, sizeof b, "\nSHARD %s: store facts %zu (BPC-train pages %ld, BPC-holdout pages %ld, after the slice %ld, left out); sentences %zu (%ld from %s, %ld longer than 80 bytes); rows %zu",
                  O.path.c_str(), facts.size(), region[0], region[1], region[2], sents.size(), from_pattern, g_raw ? "a raw frame" : "the fact's own pattern", long_resp, rows.size());
    say(b);
    for (auto& [k, v] : by_src) say("  shard_source " + k + " " + std::to_string(v));
    for (auto& [k, v] : by_rel) say("  shard_relation " + k + " " + std::to_string(v));
    say("  leak_check dropped_equal_to_dev " + std::to_string(dropped_dev) + " dropped_equal_to_test " + std::to_string(dropped_test) +
        " final_question_equal " + std::to_string(leak));
    // Dev coverage (report only; reads the dev answers, never the test's): in-slice dev questions whose (entity, relation) is in the shard.
    std::unordered_map<string, vector<const Row*>> by_pair;
    std::unordered_set<string> ents;
    for (const Row& r : rows) { by_pair[norm_ent(r.ent) + "\x1f" + r.rel].push_back(&r); ents.insert(norm_ent(r.ent)); }
    int n = 0, ne = 0, np = 0, ng = 0;
    for (const TqaQ& d : load_tqa(O.dev, true)) {
        if (!d.in_slice) continue;
        ++n;
        ne += ents.count(norm_ent(d.title)) > 0;
        bool pair = false, gold = false;
        for (const string& r : q_rels(norm_q(d.q))) {
            auto it = by_pair.find(norm_ent(d.title) + "\x1f" + r);
            if (it == by_pair.end()) continue;
            pair = true;
            for (const Row* x : it->second)
                for (const string& a : d.ans)
                    if (!norm_q(a).empty() && (" " + norm_q(x->resp) + " ").find(" " + norm_q(a) + " ") != npos) gold = true;
        }
        np += pair;
        ng += gold;
        if (pair) say("  dev_covered " + string(gold ? "gold_in_response " : "other_value      ") + d.q);
    }
    std::snprintf(b, sizeof b, "  dev_coverage in_slice %d entity_in_shard %d entity_and_relation %d response_holds_a_dev_answer %d", n, ne, np, ng);
    say(b);
    return leak ? 1 : 0;
}

// ============================================================ F2R: frames for R3, and how often they meet the training bytes
// A cleaned pattern as cyphalm_trie's render_frame writes it (tokens joined by spaces; , . ; : ) ' glued; ^ dropped).
string clean_frame(const string& key) {
    string out, prev;
    for (string t : split(key, ' ')) {
        if (t.empty() || t == "^") continue;
        const bool glue = (t.size() == 1 && std::strchr(",.;:)'", t[0])) || prev == "(" || (prev == "'" && t == "s");
        prev = t;
        if (t == "[P]") t = "[S]";
        if (!out.empty() && !glue) out += ' ';
        out += t;
    }
    return out;
}
// --reach PARSE (grading only, dev only). For each dev question with an E1 entity (PARSE: cyphalm_trie's parse file) and a
// template relation (the dktext template that, filled with the entity, equals the question), the frames of that relation
// (and its year relation for a date): "kept" = every kept pattern, "r3" = what cyphalm_trie --r3 loads (kept first, then by
// training subjects, 3 at most; no <M>, #, or object before subject). A frame reaches when its prefix, with the subject
// (article dropped) filled, occurs verbatim in the training bytes [0, train_end); it ties when the prefix from the subject's
// last word on occurs; it leads to gold when a dev answer (its year for a date) is in the 48 bytes after a tie. With --shard
// the same is measured again over the training bytes plus the shard's responses on rows not marked _holdout.
void reach_report(const vector<Pat>& pats, const string& data, std::size_t train_end, const string& dev, const string& parse_path,
                  const string& shard_text, const std::function<void(const string&)>& say) {
    std::unordered_map<string, string> ent;
    {
        std::ifstream f(parse_path);
        for (string l; std::getline(f, l);) {
            auto c = split(l, '\t');
            if (c.size() >= 9 && c[0] != "question") ent[c[0]] = c[7] == "-" ? "" : c[7];
        }
    }
    const std::string_view train(data.data(), std::min(train_end, data.size()));
    std::map<string, vector<string>> fr[2];  // [0] kept, [1] r3: relation -> frames
    {
        vector<std::tuple<int, int, string, string, const Pat*>> v;  // cyphalm_trie's order: kept, training subjects, frame bytes
        for (const Pat& p : pats) {
            string f = g_raw ? raw_frame(p.key) : clean_frame(p.key), x = f;
            replace_all(x, "[S]", "\x01");
            replace_all(x, "[O]", "\x02");
            v.emplace_back(-static_cast<int>(p.kept), -p.train_subj, x, f, &p);
        }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return std::tie(std::get<0>(a), std::get<1>(a), std::get<2>(a)) < std::tie(std::get<0>(b), std::get<1>(b), std::get<2>(b));
        });
        for (const auto& [k1, k2, x, f, p] : v) {
            const std::size_t s = f.find("[S]"), o = f.find("[O]");
            if (s == npos || o == npos || o < s || f.find("<M>") != npos || f.find('#') != npos) continue;
            auto& r3 = fr[1][kRel[p->rel].name];
            if (r3.size() < 3 && std::find(r3.begin(), r3.end(), f) == r3.end()) r3.push_back(f);
            auto& kp = fr[0][kRel[p->rel].name];
            if (p->kept && std::find(kp.begin(), kp.end(), f) == kp.end()) kp.push_back(f);
        }
    }
    for (const auto& [rel, fs] : fr[1])
        for (const string& f : fs) say("  REACH_FRAME r3 " + rel + " | " + f);
    int n_rel = 0, n_in = 0;
    int all[2][2][4] = {}, in[2][2][4] = {};  // [frames kept / r3][training bytes / + shard text][with a frame, reach, tie, gold]
    for (const TqaQ& d : load_tqa(dev, true)) {
        auto it = ent.find(d.q);
        if (it == ent.end() || it->second.empty()) continue;
        static const std::set<string> dk = {"birth_date", "birth_place", "death_date", "author", "director", "composer", "capital",
                                            "language", "founder", "location", "country", "inventor"};  // cyphalm_trie kForms with F2 relations
        string rel;
        for (const QTpl& t : kQT) {
            string q = t.q[0];
            replace_all(q, "{E}", it->second);
            if (rel.empty() && dk.count(t.rel) && norm_q(q) == norm_q(d.q)) rel = t.rel;
        }
        if (rel.empty()) continue;
        string subj = it->second;
        for (const char* a : {"the ", "The ", "a ", "an "}) if (subj.rfind(a, 0) == 0) { subj = subj.substr(std::strlen(a)); break; }
        const string last = subj.substr(subj.rfind(' ') == npos ? 0 : subj.rfind(' ') + 1);
        const bool date = rel == "birth_date" || rel == "death_date";
        vector<string> golds;
        for (const string& a : d.ans) {
            if (!date) { if (a.size() >= 3) golds.push_back(a); continue; }
            for (std::size_t i = 0; i < a.size();) {  // a date's years
                std::size_t j = i;
                while (j < a.size() && std::isdigit(static_cast<unsigned char>(a[j]))) ++j;
                if (j - i >= 3 && j - i <= 4) golds.push_back(a.substr(i, j - i));
                i = j > i ? j : i + 1;
            }
        }
        ++n_rel;
        n_in += d.in_slice;
        string line = "  REACH in_slice " + std::to_string(d.in_slice) + " " + rel + " | " + d.q + " | subject \"" + subj + "\"";
        for (int k = 0; k < 2; ++k) {
            vector<string> fs;
            for (const string& r : {rel, string(date ? (rel == "birth_date" ? "birth_year" : "death_year") : "")})
                if (fr[k].count(r)) fs.insert(fs.end(), fr[k][r].begin(), fr[k][r].end());
            for (int h = 0; h < (shard_text.empty() ? 1 : 2); ++h) {
                const vector<std::string_view> hay = h ? vector<std::string_view>{train, shard_text} : vector<std::string_view>{train};
                bool reach = false, tie = false, gold = false;
                string hit;
                for (const string& f : fs) {
                    string pre = f.substr(0, f.find("[O]")), full = pre;
                    replace_all(full, "[S]", subj);
                    string t = last + pre.substr(pre.find("[S]") + 3);
                    replace_all(t, "[S]", subj);
                    for (const std::string_view& H : hay) {
                        if (H.find(full) != npos) { reach = true; if (hit.empty()) hit = full; }
                        std::size_t p = H.find(t);
                        tie = tie || p != npos;
                        for (int occ = 0; p != npos && occ < 400 && !gold; ++occ, p = H.find(t, p + 1)) {
                            const std::string_view w = H.substr(p + t.size(), 48);
                            for (const string& g : golds) gold = gold || w.find(g) != npos;
                        }
                    }
                }
                int v[4] = {!fs.empty(), reach, tie, gold && d.in_slice};
                for (int j = 0; j < 4; ++j) { all[k][h][j] += v[j]; in[k][h][j] += v[j] && d.in_slice; }
                line += string(k ? " || r3" : " || kept") + (h ? "+shard" : "") + " frames " + std::to_string(fs.size()) + " reach " +
                        std::to_string(reach) + " tie " + std::to_string(tie) + " gold " + std::to_string(v[3]) + (hit.empty() ? "" : " [" + hit + "]");
            }
        }
        say(line);
    }
    char b[360];
    for (int k = 0; k < 2; ++k)
        for (int h = 0; h < (shard_text.empty() ? 1 : 2); ++h) {
            std::snprintf(b, sizeof b, "REACH_SUMMARY %s frames=%s text=%s: questions with entity and template relation %d (in-slice %d); "
                          "with a frame %d (%d); prefix reaches %d (%d); ties %d (%d); leads to a dev answer (in-slice) %d", g_raw ? "raw" : "cleaned",
                          k ? "r3" : "kept", h ? "train+shard" : "train", n_rel, n_in, all[k][h][0], in[k][h][0], all[k][h][1], in[k][h][1],
                          all[k][h][2], in[k][h][2], in[k][h][3]);
            say(b);
        }
}

// ============================================================ S3: the gap map (DISCOVERY 3.2, 3.5, 6.3; docs/reports/S3_S5_GAP_MAP_AND_HYPOTHESES.md)
// --gaps FILE, after a normal run: what the store lacks, per entity and relation, with VOI. Bookkeeping only: nothing here
// answers a question, and Wikidata gives only the type-level rate f(T, R), never a value. Writes FILE (gaps by VOI),
// FILE.cells.tsv (f per type and relation) and, with --demand, FILE.demand.tsv (one row per CyphaLM tag).
//   types      F2's own sets on the title: country, else person, else place, else other (lists and disambiguations skipped).
//   filled     a text seed or a stored extracted value (P >= 0.5); a date slot also fills its year slot.
//   P(exists)  the estimator of f(T, R) with the lowest Brier score on the held-out Wikidata fifth's empty store slots: the
//              store's f, or Wikidata's rate (over covered or all entities, over all or only empty store slots), fit on the
//              other fifths, then refit on all. Relations with no Wikidata value keep the store's f. Listed at >= --gap-floor.
//   contradiction  a stored extracted value of a functional relation with competing-value or constraint mass in its ECS
//              fusion; p = its P over its P
//              plus the best rival's (a text seed counts 20/21). certainty: a stored extracted value that is Believed at
//              grade Low, or Known on one sentence; p = P.
//   demand     --demand ROWS[,ROWS...]: cyphalm_trie qa rows, dev rows only; each tag adds demand to its slot (gap_map.hpp).
//   status     parked when no automatic rung can fill the slot now (no kept pattern for R, or the entity has no page), else
//              open, or queued when asked about. VOI = gap_map.hpp score(): EIG of a 0.9 / 0.9 test x (1 + w demand),
//              w = --demand-weight (0: 0 of 750 dev tags name their question's slot; the demand column is kept).
//   grading    with --dev (grading only, dev only): does a tag name its question's slot (subject: the in-slice title, else
//              --parse's E1 entity; relation: q_rels), and is that slot a gap?
// Demand weight 0 (S3, dev): 0 of CyphaLM's 750 dev tags name their question's slot, so tag demand is logged but not ranked.
struct GapOpt { string path, parse; vector<string> demand; double floor = 0.05, demand_weight = 0.0; };
void write_gaps(const GapOpt& G, const string& dev, const vector<PageRef>& pages, const vector<Fused>& fused,
                const std::function<void(const string&)>& say) {
    const char* kT[4] = {"country", "person", "place", "other"};
    auto tname = [&](int t) { return t < 0 ? "unknown" : kT[t]; };
    vector<string> ent;
    vector<int> type;
    vector<std::uint64_t> fill, wd;
    std::unordered_map<string, int> eix;
    for (const PageRef& pg : pages) {
        string lt = lower(pg.title), nk = norm_ent(pg.title);
        if (lt.compare(0, 8, "list of ") == 0 || lt.find("(disambiguation)") != npos || eix.count(nk)) continue;
        eix[nk] = static_cast<int>(ent.size());
        ent.push_back(pg.title);
        type.push_back(g_country.count(nk) ? 0 : g_person.count(nk) ? 1 : g_place.count(nk) ? 2 : 3);
    }
    const std::size_t N = ent.size();
    fill.assign(N, 0);
    wd.assign(N, 0);
    auto bit = [](int r) { return std::uint64_t{1} << r; };
    auto at = [&](const string& t) { auto it = eix.find(norm_ent(t)); return it == eix.end() ? -1 : it->second; };
    const int bd = rel_index("birth_date"), by = rel_index("birth_year"), dd = rel_index("death_date"), dy = rel_index("death_year");
    auto put = [&](int e, int r) { if (e >= 0) fill[e] |= bit(r) | (r == bd ? bit(by) : 0) | (r == dd ? bit(dy) : 0); };
    for (const Seed& sd : g_seeds) {
        if (sd.src < 3) put(at(sd.subj), sd.rel);
        else if (int e = at(sd.subj); e >= 0) wd[e] |= bit(sd.rel);
    }
    for (auto& [k, v] : g_gold) if (int e = at(k.substr(0, k.find('\x1f'))); e >= 0) wd[e] |= bit(rel_index(k.substr(k.find('\x1f') + 1)));
    for (const Fused& fu : fused) if (fu.op.P >= 0.5) put(at(fu.f->subj), fu.f->rel);
    std::uint64_t wd_rel = 0, kept = 0;
    for (std::uint64_t w : wd) wd_rel |= w;
    for (const Pat& p : g_pats) kept |= bit(p.rel);
    // f(T, R) five ways: [0] store; Wikidata [1] on covered entities (>= 1 Wikidata value), [2] on their empty store slots,
    // [3] on all entities (no Wikidata value counts as none), [4] on all empty store slots; [5..8] = [1..4] on four fifths.
    struct C { long n[9] = {}, w[9] = {}; };
    vector<C> cell(4 * kNRel);
    for (std::size_t e = 0; e < N; ++e)
        for (int r = 0; r < kNRel; ++r) {
            C& c = cell[type[e] * kNRel + r];
            const bool f = fill[e] & bit(r), w = wd[e] & bit(r), cov = wd[e] != 0, four = !wd_gold_subject(ent[e]);
            const bool in[9] = {true, cov, cov && !f, true, !f, cov && four, cov && !f && four, four, !f && four};
            for (int k = 0; k < 9; ++k) if (in[k]) { ++c.n[k]; c.w[k] += k ? w : f; }
        }
    auto rate = [](const C& c, int k) { return c.n[k] ? static_cast<double>(c.w[k]) / static_cast<double>(c.n[k]) : 0.0; };
    // Calibration on the held-out fifth's empty store slots, outcome = Wikidata has a value: [0] covered entities only,
    // [1] all entities (where P(exists) is used; it picks the estimator).
    const int est_src[5] = {0, 5, 6, 7, 8};
    double brier[2][5] = {}, ll[2][5] = {}, mp[2][5] = {}, obs[2] = {};
    long ncal[2] = {};
    for (std::size_t e = 0; e < N; ++e) {
        if (!wd_gold_subject(ent[e])) continue;
        for (int r = 0; r < kNRel; ++r) {
            if ((fill[e] & bit(r)) || !(wd_rel & bit(r))) continue;
            const double y = (wd[e] & bit(r)) ? 1.0 : 0.0;
            for (int pop = wd[e] ? 0 : 1; pop < 2; ++pop) {
                ++ncal[pop];
                obs[pop] += y;
                for (int k = 0; k < 5; ++k) {
                    const double q = rate(cell[type[e] * kNRel + r], est_src[k]), qc = std::min(0.999, std::max(0.001, q));
                    brier[pop][k] += (q - y) * (q - y);
                    ll[pop][k] -= y ? std::log2(qc) : std::log2(1 - qc);
                    mp[pop][k] += q;
                }
            }
        }
    }
    const char* kEst[5] = {"store f", "Wikidata f, covered", "Wikidata f, covered empty slots", "Wikidata f, all", "Wikidata f, all empty slots"};
    int best = 0;
    for (int k = 0; k < 5; ++k) if (ncal[1] && brier[1][k] < brier[1][best]) best = k;
    char b[512];
    for (int pop = 0; pop < 2; ++pop)
        for (int k = 0; k < 5; ++k) {
            const double n = ncal[pop] ? static_cast<double>(ncal[pop]) : 1.0;
            std::snprintf(b, sizeof b, "  GAPCAL %s %-32s Brier %.4f  log-loss %.4f bits  mean predicted %.4f  observed %.4f  slots %ld%s", pop ? "all    " : "covered",
                          kEst[k], brier[pop][k] / n, ll[pop][k] / n, mp[pop][k] / n, obs[pop] / n, ncal[pop], pop && k == best ? "  <- P(exists)" : "");
            say(b);
        }
    auto pex = [&](int t, int r) { const C& c = cell[t * kNRel + r]; return best && (wd_rel & bit(r)) && c.n[best] ? rate(c, best) : rate(c, 0); };
    // The map.
    ecs::GapMap gm;
    gm.norm = norm_ent;
    for (std::size_t e = 0; e < N; ++e)
        for (int r = 0; r < kNRel; ++r) {
            const double p = pex(type[e], r);
            if ((fill[e] & bit(r)) || p < G.floor) continue;
            ecs::Gap& g = gm.open(ecs::GapKind::Schema, ent[e], kRel[r].name, kT[type[e]]);
            g.p_exists = g.p = p;
            g.status = (kept & bit(r)) ? ecs::GapStatus::Open : ecs::GapStatus::Parked;
        }
    const double seed_tau = ecs::prior_infobox().mean();
    std::map<string, int> states;
    long contested_d = 0;
    for (const Fused& fu : fused) {
        if (fu.op.P < 0.5) continue;
        const FactAcc& f = *fu.f;
        const ecs::EpistemicState st = ecs::classify(fu.op);
        ++states[string(ecs::state_name(st)) + " " + ecs::grade_name(fu.op.grade)];
        const int e = at(f.subj);
        if (!kRel[f.rel].multi && (fu.op.s_competing > 0 || fu.op.s_constraint > 0)) {  // a functional slot (2.3)
            double rival = 0;
            for (const Fused& o : fused) {
                if (o.f == fu.f || o.f->subj != f.subj || o.f->rel != f.rel) continue;
                const bool compat = is_datey(f.rel) ? (kRel[f.rel].type == T_YEAR ? o.f->d.y == f.d.y : date_compat(o.f->d, f.d)) : o.f->okey == f.okey;
                if (!compat) rival = std::max(rival, o.op.P);
            }
            if (rival == 0 && fu.op.s_competing > 0) rival = seed_tau;
            ecs::Gap* g = gm.find(ecs::GapKind::Contradiction, f.subj, kRel[f.rel].name);
            if (!g) {  // the slot's leading value (fused is sorted by P, highest first)
                g = &gm.open(ecs::GapKind::Contradiction, f.subj, kRel[f.rel].name, e < 0 ? "unknown" : kT[type[e]]);
                g->p = rival > 0 ? fu.op.P / (fu.op.P + rival) : fu.op.P;
                g->p_exists = 1;
                g->note = f.surf + " P " + std::to_string(fu.op.P).substr(0, 5) + " d " + std::to_string(fu.op.d).substr(0, 5) + (rival > 0 ? " rival P " + std::to_string(rival).substr(0, 5) : " lifespan");
            }
            if (st == ecs::EpistemicState::Contested) { ++contested_d; g->note += " Contested"; }
        }
        if ((st == ecs::EpistemicState::Believed && fu.op.grade == ecs::Grade::Low) || (st == ecs::EpistemicState::Known && fu.n_ev == 1)) {
            if (gm.find(ecs::GapKind::Certainty, f.subj, kRel[f.rel].name)) continue;
            ecs::Gap& g = gm.open(ecs::GapKind::Certainty, f.subj, kRel[f.rel].name, e < 0 ? "unknown" : kT[type[e]]);
            g.p = fu.op.P;
            g.p_exists = 1;
            g.note = f.surf + " " + ecs::state_name(st) + " " + ecs::grade_name(fu.op.grade) + " sentences " + std::to_string(fu.n_ev);
        }
    }
    // Seed-only conflicts, which ECS never sees (report only): functional relations with two text-seed values of different
    // years (dates) or keys (others). Places inside each other are counted too, so this is an upper bound.
    std::set<string> seed_conf;
    {
        std::unordered_map<string, std::set<string>> vals;
        for (const Seed& sd : g_seeds)
            if (sd.src < 3 && kRel[sd.rel].phi >= 0.9) vals[sd.subj + "\x1f" + kRel[sd.rel].name].insert(is_datey(sd.rel) ? std::to_string(sd.o.d.y) : sd.o.k1);
        for (auto& [k, v] : vals) if (v.size() > 1) seed_conf.insert(k);
    }
    // Demand: CyphaLM's own tags, dev rows only.
    vector<TqaQ> devq = dev.empty() ? vector<TqaQ>{} : load_tqa(dev, false);
    std::unordered_map<string, string> e1;
    if (!G.parse.empty()) {
        std::ifstream f(G.parse);
        for (string l; std::getline(f, l);) { auto c = split(l, '\t'); if (c.size() >= 9 && c[0] != "question") e1[c[0]] = c[7] == "-" ? "" : c[7]; }
    }
    // A question's slot: 0 no store relation, 1 no page, 2 filled, 3 schema gap, 4 empty below the floor.
    auto qslot = [&](const TqaQ& d, int& e, vector<string>& rels) {
        rels = q_rels(norm_q(d.q));
        string subj = d.in_slice ? d.title : (e1.count(d.q) ? e1[d.q] : "");
        e = subj.empty() ? -1 : at(subj);
        if (rels.empty()) return 0;
        if (e < 0) return 1;
        bool gap = false;
        for (const string& r : rels) { if (fill[e] & bit(rel_index(r))) return 2; gap |= pex(type[e], rel_index(r)) >= G.floor; }
        return gap ? 3 : 4;
    };
    const char* kQs[5] = {"no store relation", "no page", "filled", "schema gap", "empty below floor"};
    std::map<std::pair<string, string>, std::pair<long, std::set<string>>> clusters;
    long lad[8] = {};  // tags, complete, id consistent, question asks a store relation, relation right, entity right, slot right, real gap
    std::ofstream dm;
    if (!G.demand.empty()) {
        dm.open(G.path + ".demand.tsv");
        dm << "file\trow\tin_slice\tform\tlm\tcomplete\tid_consistent\treason\tentity\trelation\tgap\tentity_type\tlanded_on\tquestion_slot\trelation_right\tentity_right\treal_gap\n";
    }
    for (const string& path : G.demand) {
        std::ifstream f(path);
        string line, label = path.substr(path.rfind('/') == npos ? 0 : path.rfind('/') + 1);
        std::getline(f, line);
        vector<string> h = split(line, '\t');
        const std::size_t ia = std::find(h.begin(), h.end(), "answer") - h.begin();
        long rows = 0, tags = 0;
        while (std::getline(f, line)) {
            vector<string> c = split(line, '\t');
            if (c.size() <= ia || c[0] != "dev") continue;  // dev only; other sets are skipped unread
            ++rows;
            const ecs::UnknownTag t = ecs::parse_unknown_tag(c[ia]);
            if (!t.found) continue;
            ++tags;
            const int e = t.entity.empty() ? -1 : at(t.entity);
            const string ty = tname(e < 0 ? -1 : type[e]);
            const ecs::Gap* g = gm.add_demand(t, ty);
            if (!t.relation.empty()) { auto& cl = clusters[{t.relation, ty}]; ++cl.first; cl.second.insert(norm_ent(t.entity)); }
            int qe = -1, qs = 0;
            bool rr = false, er = false, real = false;
            vector<string> rels;
            const std::size_t row = static_cast<std::size_t>(std::atol(c[1].c_str()));
            if (row < devq.size()) {
                const TqaQ& d = devq[row];
                qs = qslot(d, qe, rels);
                rr = std::find(rels.begin(), rels.end(), t.relation) != rels.end();
                const string subj = d.in_slice ? d.title : (e1.count(d.q) ? e1[d.q] : "");
                er = !t.entity.empty() && !subj.empty() && norm_ent(t.entity) == norm_ent(subj);
                real = rr && er && (qs == 1 || qs == 3);
            }
            const long v[8] = {1, t.complete(), t.id_consistent(), qs != 0, rr, er, rr && er, real};
            for (int k = 0; k < 8; ++k) lad[k] += v[k];
            dm << label << '\t' << c[1] << '\t' << c[2] << '\t' << c[3] << '\t' << c[4] << '\t' << t.complete() << '\t' << t.id_consistent() << '\t' << t.reason << '\t'
               << t.entity << '\t' << t.relation << '\t' << t.gap << '\t' << ty << '\t' << (g ? ecs::gap_kind_name(g->kind) : "none") << '\t'
               << (row < devq.size() ? kQs[qs] : "") << '\t' << rr << '\t' << er << '\t' << real << '\n';
        }
        say("  DEMAND " + label + ": dev rows " + std::to_string(rows) + ", tags " + std::to_string(tags));
    }
    for (const ecs::Gap& g : gm.gaps())  // a demand gap no rung can fill (no kept pattern for its relation, or no page) is parked
        if (g.kind == ecs::GapKind::Demand && (rel_index(g.relation) < 0 || !(kept & bit(rel_index(g.relation))) || at(g.entity) < 0))
            gm.find(g.kind, g.entity, g.relation)->status = ecs::GapStatus::Parked;
    ecs::VoiConfig vc;
    vc.demand_weight = G.demand_weight;
    gm.score(vc);
    // Outputs.
    std::ofstream out(G.path);
    out << "gap_id\tkind\tstatus\tentity\ttype\trelation\tp_exists\tf_store\tf_wd\tp\teig_bits\tdemand\tvoi\tnote\n";
    for (const ecs::Gap* g : gm.ranked()) {
        const int r = rel_index(g->relation);
        const int t = std::find(kT, kT + 4, g->type) - kT;
        const bool c = r >= 0 && t < 4;
        std::snprintf(b, sizeof b, "\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t", g->p_exists, c ? rate(cell[t * kNRel + r], 0) : -1.0,
                      c ? rate(cell[t * kNRel + r], 1) : -1.0, g->p, g->eig, g->demand, g->voi);
        out << g->id << '\t' << ecs::gap_kind_name(g->kind) << '\t' << ecs::gap_status_name(g->status) << '\t' << g->entity << '\t' << g->type << '\t' << g->relation << b << g->note << '\n';
    }
    std::ofstream co(G.path + ".cells.tsv");
    co << "type\trelation\tentities\twith_store\tf_store\twd_entities\twith_wd\tf_wd\twd_empty_slots\twith_wd_empty\tf_wd_empty\tp_exists\tkept_pattern\tschema_gaps\n";
    std::map<std::pair<int, int>, long> gaps_tr;
    std::map<string, long> kind_n, status_n, rel_n, ids;
    for (const ecs::Gap& g : gm.gaps()) {
        ++kind_n[ecs::gap_kind_name(g.kind)];
        ++status_n[string(ecs::gap_kind_name(g.kind)) + " " + ecs::gap_status_name(g.status)];
        ++ids[g.id];
        if (g.kind == ecs::GapKind::Schema) { ++rel_n[g.relation]; ++gaps_tr[{static_cast<int>(std::find(kT, kT + 4, g.type) - kT), rel_index(g.relation)}]; }
    }
    for (int t = 0; t < 4; ++t)
        for (int r = 0; r < kNRel; ++r) {
            const C& c = cell[t * kNRel + r];
            std::snprintf(b, sizeof b, "%s\t%s\t%ld\t%ld\t%.4f\t%ld\t%ld\t%.4f\t%ld\t%ld\t%.4f\t%.4f\t%d\t%ld", kT[t], kRel[r].name, c.n[0], c.w[0], rate(c, 0), c.n[1], c.w[1],
                          rate(c, 1), c.n[2], c.w[2], rate(c, 2), pex(t, r), (kept & bit(r)) ? 1 : 0, gaps_tr[{t, r}]);
            co << b << '\n';
        }
    // Summary.
    long tn[4] = {};
    for (int t : type) ++tn[t];
    std::snprintf(b, sizeof b, "\nGAPMAP %s: entities %zu (country %ld, person %ld, place %ld, other %ld); gaps %zu; P(exists) floor %.2f", G.path.c_str(), N, tn[0], tn[1], tn[2], tn[3], gm.size(), G.floor);
    say(b);
    for (auto& [k, v] : kind_n) say("  GAPKIND " + k + " " + std::to_string(v));
    for (auto& [k, v] : status_n) say("  GAPSTATUS " + k + " " + std::to_string(v));
    for (auto& [k, v] : rel_n) say("  GAPREL " + k + " " + std::to_string(v));
    for (double fl : {0.05, 0.10, 0.20, 0.50, 0.80}) {  // listed schema gaps; on the held-out fifth, how many Wikidata confirms
        long n = 0, ng = 0, nc = 0;
        double sp = 0;
        for (const ecs::Gap& g : gm.gaps()) {
            if (g.kind != ecs::GapKind::Schema || g.p_exists < fl) continue;
            ++n;
            sp += g.p_exists;
            const int e = at(g.entity), r = rel_index(g.relation);
            if (e < 0 || !wd_gold_subject(ent[e]) || !(wd_rel & bit(r))) continue;
            ++ng;
            nc += (wd[e] & bit(r)) != 0;
        }
        std::snprintf(b, sizeof b, "  GAPFLOOR %.2f schema %ld, sum P(exists) %.0f; held-out fifth %ld, Wikidata has a value %ld (%.3f)", fl, n, sp, ng, nc, ng ? static_cast<double>(nc) / ng : 0.0);
        say(b);
    }
    long shared = 0;
    for (const ecs::Gap& g : gm.gaps()) shared += ids[g.id] > 1;
    say("  GAPID 16-bit ids: distinct " + std::to_string(ids.size()) + ", gaps sharing an id " + std::to_string(shared));
    say("  CONTRADICTION slots " + std::to_string(kind_n["contradiction"]) + ", values Contested by d >= 0.5: " + std::to_string(contested_d) +
        "; seed-only conflicting slots (functional relations, upper bound) " + std::to_string(seed_conf.size()));
    for (auto& [k, v] : states) say("  STORED_STATE " + k + " " + std::to_string(v));
    for (auto& [k, v] : clusters) say("  DEMAND_CLUSTER " + k.first + " " + k.second + " demand " + std::to_string(v.first) + " entities " + std::to_string(v.second.size()));
    if (!G.demand.empty()) {
        std::snprintf(b, sizeof b, "  GRADE tags %ld complete %ld id_consistent %ld | question asks a store relation %ld relation_right %ld entity_right %ld slot_right %ld real_gap %ld",
                      lad[0], lad[1], lad[2], lad[3], lad[4], lad[5], lad[6], lad[7]);
        say(b);
    }
    long qn[2][5] = {};
    for (const TqaQ& d : devq) { int e; vector<string> rels; ++qn[d.in_slice ? 1 : 0][qslot(d, e, rels)]; }
    for (int k = 0; k < 5 && !devq.empty(); ++k) say(string("  QSLOT ") + kQs[k] + ": in-slice " + std::to_string(qn[1][k]) + ", not in slice " + std::to_string(qn[0][k]));
    std::map<string, int> top[2];  // [0] all gaps, [1] P1's queue: gaps a rung can fill now (not parked)
    int k[2] = {};
    for (const ecs::Gap* g : gm.ranked())
        for (int q = 0; q < 2; ++q) {
            if (q && g->status == ecs::GapStatus::Parked) continue;
            if (k[q] < 100) ++top[q][ecs::gap_kind_name(g->kind)];
            if (q && k[q] < 15) {
                std::snprintf(b, sizeof b, "  TOPQ %2d %s %-13s %-7s %s | %s (%s) p %.3f eig %.3f demand %d voi %.3f %s", k[q] + 1, g->id.c_str(), ecs::gap_kind_name(g->kind),
                              ecs::gap_status_name(g->status), g->relation.c_str(), g->entity.c_str(), g->type.c_str(), g->p, g->eig, g->demand, g->voi, g->note.c_str());
                say(b);
            }
            ++k[q];
        }
    for (int q = 0; q < 2; ++q)
        for (auto& [kk, v] : top[q]) say(string(q ? "  TOP100_QUEUE " : "  TOP100_ALL ") + kk + " " + std::to_string(v));
}

struct Args {
    string enwik, outdir;
    int folds = 5;
    bool cv = false;
    int min_train = 2;
    int sample = 50;
    double base_rate = -1;  // < 0: estimate per relation from the kept patterns' held-out precision
    std::uint32_t seed = 20261027;
    string wikidata;          // D4: Wikidata values (seeds for 4/5 of the subjects, scoring gold for 1/5)
    bool wd_gold_only = false;  // all Wikidata values only score (to check a run without Wikidata seeds)
    ShardOpt shard;
    string reach;               // F2R: the E1 parse file for --reach (with --dev)
    GapOpt gaps;                // S3: --gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F]
};

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && string(argv[1]) == "--wilson") {
        double k = std::atof(argv[2]), n = std::atof(argv[3]);
        std::printf("%.0f/%.0f = %.3f  Wilson95 [%.3f, %.3f]\n", k, n, n > 0 ? k / n : 0.0, ecs::wilson_lower(k, n), ecs::wilson_upper(k, n));
        return 0;
    }
    Args A;
    if (argc < 3) {
        std::fprintf(stderr, "usage: pattern_learn <enwik8> <outdir> [--folds K] [--cv] [--min-train N] [--sample N] [--seed S] [--base-rate A]\n"
                             "         [--wikidata FILE [--wikidata-gold-only]] [--phi FILE] [--shard FILE --dev DEV --test TEST [--slice-end B] [--holdout-from B]]\n"
                             "         [--raw] [--join FACTS] [--reach PARSE] [--gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F] [--demand-weight W]]\n"
                             "       pattern_learn --wilson K N\n");
        return 2;
    }
    std::fill(g_phi, g_phi + 64, -1.0);
    A.enwik = argv[1];
    A.outdir = argv[2];
    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        if (a == "--folds" && i + 1 < argc) A.folds = std::max(2, std::atoi(argv[++i]));
        else if (a == "--cv") A.cv = true;
        else if (a == "--min-train" && i + 1 < argc) A.min_train = std::atoi(argv[++i]);
        else if (a == "--sample" && i + 1 < argc) A.sample = std::atoi(argv[++i]);
        else if (a == "--base-rate" && i + 1 < argc) A.base_rate = std::atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) A.seed = static_cast<std::uint32_t>(std::atoll(argv[++i]));
        else if (a == "--wikidata" && i + 1 < argc) A.wikidata = argv[++i];
        else if (a == "--wikidata-gold-only") A.wd_gold_only = true;
        else if (a == "--phi" && i + 1 < argc) {  // relation \t ... \t phi (last column), e.g. wikidata_seeds.py's phi.tsv
            std::ifstream pf(argv[++i]);
            for (string l; std::getline(pf, l);) { auto c = split(l, '\t'); int r = rel_index(c[0]); if (r >= 0) g_phi[r] = std::atof(c.back().c_str()); }
        }
        else if (a == "--shard" && i + 1 < argc) A.shard.path = argv[++i];
        else if (a == "--dev" && i + 1 < argc) A.shard.dev = argv[++i];
        else if (a == "--test" && i + 1 < argc) A.shard.test = argv[++i];
        else if (a == "--slice-end" && i + 1 < argc) A.shard.slice_end = static_cast<std::size_t>(std::atoll(argv[++i]));
        else if (a == "--holdout-from" && i + 1 < argc) A.shard.holdout_from = static_cast<std::size_t>(std::atoll(argv[++i]));
        else if (a == "--raw") g_raw = true;
        else if (a == "--reach" && i + 1 < argc) A.reach = argv[++i];
        else if (a == "--join" && i + 1 < argc) A.shard.join = argv[++i];
        else if (a == "--gaps" && i + 1 < argc) A.gaps.path = argv[++i];
        else if (a == "--demand" && i + 1 < argc) A.gaps.demand = split(argv[++i], ',');
        else if (a == "--parse" && i + 1 < argc) A.gaps.parse = argv[++i];
        else if (a == "--gap-floor" && i + 1 < argc) A.gaps.floor = std::atof(argv[++i]);
        else if (a == "--demand-weight" && i + 1 < argc) A.gaps.demand_weight = std::atof(argv[++i]);
    }
    if (!A.reach.empty() && A.shard.dev.empty()) { std::fprintf(stderr, "--reach needs --dev\n"); return 2; }
    if (!A.shard.join.empty() && A.shard.path.empty()) { std::fprintf(stderr, "--join needs --shard\n"); return 2; }
    A.shard.seed = A.seed;
    if (!A.shard.path.empty() && (A.shard.dev.empty() || A.shard.test.empty())) { std::fprintf(stderr, "--shard needs --dev and --test (leakage check)\n"); return 2; }
    std::ostringstream rep;
    auto say = [&](const string& s) { std::cout << s << "\n"; rep << s << "\n"; };
    char buf[512];
    const double t0 = now_s();
    string data;
    if (!read_file(A.enwik, data)) { std::fprintf(stderr, "cannot read %s\n", A.enwik.c_str()); return 1; }
    g_dump = &data;
    if (g_raw) say("F2R raw mode: patterns are the dump's own bytes between the two mentions");
    long total = 0, redirects = 0;
    vector<PageRef> pages = list_pages(data, total, redirects);
    std::unordered_map<string, int> page_of;
    for (std::size_t i = 0; i < pages.size(); ++i) page_of[pages[i].title] = static_cast<int>(i);
    std::snprintf(buf, sizeof buf, "pages in dump %ld, redirects %ld, main-namespace articles %zu", total, redirects, pages.size());
    say(buf);

    // ---- pass 1: seeds and type lexicons
    double t1 = now_s();
    for (const PageRef& pg : pages) {
        PageFlags fl;
        scan_page(data, pg, fl);
        string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    {
        std::ofstream pf(A.outdir + "/pages.tsv");
        for (const PageRef& pg : pages) pf << pg.title << '\t' << pg.tb << '\n';
    }
    if (!A.wikidata.empty()) {
        long ns = 0, ng = 0;
        if (!load_wikidata(A.wikidata, A.wd_gold_only, page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", A.wikidata.c_str()); return 1; }
        say("wikidata values: " + std::to_string(ns) + " seeds, " + std::to_string(ng) + " gold (scoring only), " + std::to_string(g_gold.size()) + " gold (subject, relation) pairs");
    }
    for (const Seed& sd : g_seeds) {
        RT ty = kRel[sd.rel].type;
        if (ty == T_PLACE || ty == T_PLACE_LINK) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    say("pass 1 (seeds, types): " + std::to_string(now_s() - t1) + " s; place pages " + std::to_string(g_place.size()) + ", person pages " + std::to_string(g_person.size()) +
        ", lexicon strings " + std::to_string(g_lex.size()));

    // seed table
    {
        std::map<int, std::array<int, 4>> cnt;
        std::map<int, std::unordered_set<string>> subj;
        for (const Seed& sd : g_seeds) { ++cnt[sd.rel][static_cast<std::size_t>(sd.src)]; subj[sd.rel].insert(sd.subj); }
        say("\nSEEDS per relation (infobox/template | wiki table | category | text total | wikidata | subjects)");
        int tot = 0, wd = 0;
        for (int r = 0; r < kNRel; ++r) {
            auto& c = cnt[r];
            int t = c[0] + c[1] + c[2];
            tot += t;
            wd += c[3];
            std::snprintf(buf, sizeof buf, "  %-13s %6d %6d %6d %7d %7d %7zu", kRel[r].name, c[0], c[1], c[2], t, c[3], subj[r].size());
            say(buf);
        }
        say("  total seeds " + std::to_string(tot) + (wd ? " from the text + " + std::to_string(wd) + " from Wikidata" : "") + " over " + std::to_string(g_by_subj.size()) + " subject pages");
        std::ofstream f(A.outdir + "/seeds.tsv");
        f << "subject\trelation\tobject\tsource\ttemplate\tbyte_offset\n";
        for (const Seed& sd : g_seeds) f << sd.subj << '\t' << kRel[sd.rel].name << '\t' << sd.o.surf << '\t' << kSrcName[sd.src] << '\t' << sd.tpl << '\t' << sd.off << '\n';
    }

    // ---- passes 2-3: train on 80% of the subjects, test on 20%
    std::unordered_map<string, PStat> stat;
    int folds_run = A.cv ? A.folds : 1;
    double t2 = now_s();
    long train_seeds_total = 0, held_seeds_total = 0;
    for (int fold = 0; fold < folds_run; ++fold) {
        std::unordered_map<string, TrainCnt> tm;
        for (auto& kv : g_by_subj) {
            if (heldout(kv.first, A.folds, fold)) { held_seeds_total += static_cast<long>(kv.second.size()); continue; }
            train_seeds_total += static_cast<long>(kv.second.size());
            auto pit = page_of.find(kv.first);
            if (pit != page_of.end()) train_page(data, pages[static_cast<std::size_t>(pit->second)], kv.second, tm);
        }
        g_pats.clear();
        g_patmap.clear();
        for (auto& kv : tm) {
            if (kv.second.subj < A.min_train) continue;
            std::size_t tab = kv.first.find('\t');
            Pat p;
            p.rel = std::atoi(kv.first.substr(0, tab).c_str());
            p.key = kv.first.substr(tab + 1);
            p.family = family_of(p.key);
            p.train_subj = kv.second.subj;
            p.train_sent = kv.second.sent;
            g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
            g_pats.push_back(p);
        }
        std::unordered_map<string, PStat> local;
        for (auto& kv : g_by_subj) {
            if (!heldout(kv.first, A.folds, fold)) continue;
            auto pit = page_of.find(kv.first);
            if (pit != page_of.end()) eval_page(data, pages[static_cast<std::size_t>(pit->second)], local);
        }
        for (const Pat& p : g_pats) {
            PStat& st = stat[std::to_string(p.rel) + "\t" + p.key];
            if (fold == 0 || A.cv) { st.train_subj += p.train_subj; st.train_sent += p.train_sent; }
            auto it = local.find(std::to_string(p.rel) + "\t" + p.key);
            if (it != local.end()) { st.checks += it->second.checks; st.correct += it->second.correct; st.unchecked += it->second.unchecked; }
        }
        std::fprintf(stderr, "fold %d: %zu candidate patterns\n", fold, g_pats.size());
    }
    say("pass 2-3 (train 80% / test 20% by subject, " + std::string(A.cv ? "all folds" : "fold 0 only") + "): " + std::to_string(now_s() - t2) + " s");

    // ---- step 3: reliability filter
    vector<Pat> kept_all, cand_all;
    for (auto& kv : stat) {
        std::size_t tab = kv.first.find('\t');
        Pat p;
        p.rel = std::atoi(kv.first.substr(0, tab).c_str());
        p.key = kv.first.substr(tab + 1);
        p.family = family_of(p.key);
        p.train_subj = kv.second.train_subj;
        p.train_sent = kv.second.train_sent;
        p.checks = kv.second.checks;
        p.correct = kv.second.correct;
        p.unchecked = kv.second.unchecked;
        p.prec = p.checks ? static_cast<double>(p.correct) / p.checks : 0;
        p.lo = ecs::wilson_lower(p.correct, p.checks);
        p.reliability = ecs::beta_mean(1.0 + p.correct, 1.0 + (p.checks - p.correct));
        p.kept = ecs::pattern_admissible(static_cast<std::size_t>(p.correct), static_cast<std::size_t>(p.checks));
        cand_all.push_back(p);
        if (p.kept) kept_all.push_back(p);
    }
    std::sort(cand_all.begin(), cand_all.end(), [](const Pat& a, const Pat& b) {
        if (a.kept != b.kept) return a.kept;
        if (a.rel != b.rel) return a.rel < b.rel;
        if (a.lo != b.lo) return a.lo > b.lo;
        return a.key < b.key;
    });
    {
        std::ofstream f(A.outdir + "/patterns.tsv");
        f << "relation\tpattern\ttrain_subjects\ttrain_sentences\tchecks\tcorrect\tunchecked\tprecision\twilson_lower\treliability\tkept" << (g_raw ? "\tframe\n" : "\n");
        for (const Pat& p : cand_all)
            f << kRel[p.rel].name << '\t' << p.key << '\t' << p.train_subj << '\t' << p.train_sent << '\t' << p.checks << '\t' << p.correct << '\t' << p.unchecked << '\t'
              << p.prec << '\t' << p.lo << '\t' << p.reliability << '\t' << (p.kept ? 1 : 0) << (g_raw ? "\t" + raw_frame(p.key) : string()) << '\n';
    }
    say("\nPATTERNS per relation (candidates = >= " + std::to_string(A.min_train) + " training subjects; kept = >= 20 checks and Wilson lower bound >= 0.7)");
    say("  relation      candidates checked>=1 checks>=20 kept   top kept patterns (precision, checks, Wilson lower)");
    for (int r = 0; r < kNRel; ++r) {
        int nc = 0, n1 = 0, n20 = 0, nk = 0;
        vector<const Pat*> top;
        for (const Pat& p : cand_all) {
            if (p.rel != r) continue;
            ++nc;
            if (p.checks >= 1) ++n1;
            if (p.checks >= 20) ++n20;
            if (p.kept) { ++nk; top.push_back(&p); }
        }
        if (!nc) continue;
        std::sort(top.begin(), top.end(), [](const Pat* a, const Pat* b) { if (a->lo != b->lo) return a->lo > b->lo; return a->checks > b->checks; });
        std::snprintf(buf, sizeof buf, "  %-13s %9d %9d %9d %5d", kRel[r].name, nc, n1, n20, nk);
        say(buf);
        for (std::size_t i = 0; i < top.size() && i < 5; ++i) {
            std::snprintf(buf, sizeof buf, "      %-52s  %.3f  n=%d  lo=%.3f  rel=%.3f", top[i]->key.c_str(), top[i]->prec, top[i]->checks, top[i]->lo, top[i]->reliability);
            say(buf);
        }
    }
    say("  seeds in the training split " + std::to_string(train_seeds_total) + ", in the held-out split " + std::to_string(held_seeds_total) + (A.cv ? " (summed over folds)" : ""));

    // base rate a per relation: pooled held-out precision of its kept patterns (clamped 0.5..0.95), or the fixed --base-rate
    for (int r = 0; r < kNRel; ++r) {
        double c = 0, n = 0;
        for (const Pat& p : kept_all) if (p.rel == r) { c += p.correct; n += p.checks; }
        g_base[r] = A.base_rate >= 0 ? A.base_rate : (n > 0 ? std::min(0.95, std::max(0.5, c / n)) : 0.5);
    }
    // ---- step 4: extraction with the kept patterns
    g_pats.clear();
    g_patmap.clear();
    for (const Pat& p : kept_all) {
        g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
        g_pats.push_back(p);
    }
    double t4 = now_s();
    for (const PageRef& pg : pages) {
        string lt = lower(pg.title);
        if (lt.compare(0, 8, "list of ") == 0 || lt.find("(disambiguation)") != npos) continue;
        extract_page(data, pg);
    }
    say("pass 4 (extraction over all " + std::to_string(pages.size()) + " articles): " + std::to_string(now_s() - t4) + " s; raw new facts " + std::to_string(g_new.size()));
    vector<Fused> fused = fuse_all();

    // ---- output
    std::ofstream out(A.outdir + "/facts_f2.tsv");
    std::ofstream ev(A.outdir + "/facts_f2.evidence.tsv");
    out << "subject\trelation\tobject\tP\tgrade\tn_evidence\tbest_pattern\tsource_page\tbyte_offset\n";
    ev << "subject\trelation\tobject\tP\tstrongest_evidence_sentence\n";
    struct RelOut { int raw = 0, cand = 0, adm = 0, adm_mod = 0; };
    std::map<int, RelOut> ro;
    vector<const Fused*> admitted;
    for (const Fused& fu : fused) {
        const FactAcc& f = *fu.f;
        ++ro[f.rel].raw;
        if (fu.op.P < 0.5) continue;
        bool adm = fu.op.P >= 0.8;
        if (adm) { ++ro[f.rel].adm; admitted.push_back(&fu); if (ecs::grade_at_least(fu.op.grade, ecs::Grade::Moderate)) ++ro[f.rel].adm_mod; }
        else ++ro[f.rel].cand;
        char pb[32];
        std::snprintf(pb, sizeof pb, "%.4f", fu.op.P);
        out << f.subj << '\t' << kRel[f.rel].name << '\t' << f.surf << '\t' << pb << '\t' << ecs::grade_name(fu.op.grade) << '\t' << fu.n_ev << '\t' << fu.best_pattern << '\t'
            << f.subj << '\t' << fu.best_off << '\n';
        ev << f.subj << '\t' << kRel[f.rel].name << '\t' << f.surf << '\t' << pb << '\t' << f.snippet << '\n';
    }
    say("\nNEW FACTS per relation (seeds before | raw extractions | candidates 0.5<=P<0.8 | admitted P>=0.8 | admitted also grade>=Moderate | seeds+admitted after)");
    std::map<int, int> seedn;
    for (const Seed& sd : g_seeds) if (sd.src < 3) ++seedn[sd.rel];
    int sb = 0, sa = 0, rc = 0, rr = 0, ra = 0, rm = 0;
    for (int r = 0; r < kNRel; ++r) {
        RelOut o = ro[r];
        if (!seedn[r] && !o.raw) continue;
        std::snprintf(buf, sizeof buf, "  %-13s %7d %7d %7d %7d %7d %9d", kRel[r].name, seedn[r], o.raw, o.cand, o.adm, o.adm_mod, seedn[r] + o.adm);
        say(buf);
        sb += seedn[r]; sa += seedn[r] + o.adm; rr += o.raw; rc += o.cand; ra += o.adm; rm += o.adm_mod;
    }
    std::snprintf(buf, sizeof buf, "  %-13s %7d %7d %7d %7d %7d %9d", "TOTAL", sb, rr, rc, ra, rm, sa);
    say(buf);

    // ---- step 5: the sample to hand-judge (fixed seed)
    {
        vector<int> idx(admitted.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::mt19937 rng(A.seed);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::ofstream sm(A.outdir + "/sample_admitted.tsv");
        sm << "n\tsubject\trelation\tobject\tP\tpattern\tsentence\n";
        for (int k = 0; k < A.sample && k < static_cast<int>(idx.size()); ++k) {
            const Fused& fu = *admitted[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
            sm << (k + 1) << '\t' << fu.f->subj << '\t' << kRel[fu.f->rel].name << '\t' << fu.f->surf << '\t' << fu.op.P << '\t' << fu.best_pattern << '\t' << fu.f->snippet << '\n';
        }
    }
    if (!g_gold.empty()) score_gold(fused, A.outdir, say);
    string shard_text;
    int exit_code = A.shard.path.empty() ? 0 : write_shard(A.shard, fused, pages, page_of, say, &shard_text);
    if (!A.reach.empty()) reach_report(cand_all, data, A.shard.holdout_from, A.shard.dev, A.reach, shard_text, say);
    if (!A.gaps.path.empty()) write_gaps(A.gaps, A.shard.dev, pages, fused, say);
    std::snprintf(buf, sizeof buf, "\nrun time %.1f s, peak RAM %.0f MB, facts written %s/facts_f2.tsv", now_s() - t0, peak_mb(), A.outdir.c_str());
    say(buf);
    std::ofstream(A.outdir + "/stats.txt") << rep.str();
    return exit_code;
}
