// F1: can a fact store extracted from the Wikipedia slice answer the 8 questions the trie failed?
//
// Reads enwik8 (XML-escaped MediaWiki dump), keeps only pages that end inside the train range
// [0, 7340032) (a page that straddles the boundary is dropped), and builds a symbolic fact store:
//   1. Template/table fields: "| key = value" lines inside {{...}} (citation templates skipped) and
//      wiki-table rows "| key || value" or "| key:" / "| value", with <br>-parallel lists split. Keys are
//      lowercased and stripped of spaces/underscores, then mapped to relations (capital, author,
//      composer, language, currency, ...). Value = first [[link]] target, else cleaned text.
//   2. Sentence patterns over article text: "<Title> is a novel/composition/film by [[X]]",
//      "written/composed/directed/founded/invented by", "studied with / student of / taught by / his
//      teacher", "taught [[Y]] / tutor of [[Y]] / [[X]] was Y's tutor" (inverse), "feeds on / eats /
//      diet is", "capital is / [[X]] is the capital of [[Y]]". The page title must head the clause.
// Each fact keeps subject, relation, object, source (infobox key or pattern id), page, byte offset
// of the line in the corpus and a support count. A question classifier (regex templates, with
// inverse forms) maps a question to (relations, subject); subject lookup is case-insensitive title,
// then #REDIRECT alias, then leading "the/a/an" stripped, then singularised. Answer = object with the
// most support (ties: infobox/table before pattern, then earliest byte); no fact -> ABSTAIN.
// No entity string appears in the code; the 8 seed questions are only in bench/eval/.
//
// Measured 2026-10-27 (enwik8 train range; 915 pages, 377 redirects, 141 facts, about 1.3 s):
//   8/8 on bench/eval/t_qa_wiki_seed.tsv; all 8 source articles lie in the train range.
//   Hand-checked precision of extracted facts (30 random facts, seed 20261028): 27/30 = 0.90,
//   Wilson 95% lower bound 0.744. Patterns were written after reading those 8 articles, so 8/8 is
//   an in-sample score; the precision sample and the unseen spot-check are the generality evidence.
//   Report: docs/reports/F1_INFOBOX_QA.md
//
// Build: g++ -O2 -std=c++20 native/tools/qa_infobox.cpp -o qa_infobox
// Usage: qa_infobox <enwik8> [questions.tsv]          questions.tsv: question TAB answer|variant|...
//        qa_infobox <enwik8> --ask "What is the capital of Albania?"
//        qa_infobox <enwik8> --sample 30 --seed 20261028   random facts with source text, for hand checks
//        qa_infobox <enwik8> --show "Aristotle"             all facts about one subject
//        qa_infobox --wilson K N                            Wilson 95% lower bound for K correct of N
//        option --limit BYTES sets the train range (default 7340032)

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kTrainBytes = 7340032;
const std::size_t npos = std::string::npos;

// ---------------------------------------------------------------- strings
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}
bool starts_with(const std::string& s, const char* p) { return s.compare(0, std::strlen(p), p) == 0; }
int count_of(const std::string& s, const char* p) {
    int n = 0;
    for (std::size_t i = s.find(p); i != npos; i = s.find(p, i + std::strlen(p))) ++n;
    return n;
}
std::string truncate(const std::string& s, std::size_t n) { return s.size() <= n ? s : s.substr(0, n) + "..."; }

// The dump is XML-escaped once: &lt; &gt; &quot; &amp;. Decode one level only.
std::string xml_decode(const std::string& s) {
    std::string o;
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

void erase_span(std::string& s, const std::string& open, const std::string& close) {
    std::size_t p = 0;
    while ((p = s.find(open, p)) != npos) {
        std::size_t e = s.find(close, p + open.size());
        if (e == npos) { s.erase(p); break; }
        s.erase(p, e + close.size() - p);
    }
}
// Remove <!-- comments --> and <ref>..</ref> / <ref ... /> from one line.
void erase_refs(std::string& s) {
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
std::string strip_tags(const std::string& s, const std::string& br_sep) {
    std::string o;
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '<') {
            std::size_t gt = s.find('>', i);
            if (gt != npos && gt - i < 200) {
                std::string tag = lower(s.substr(i + 1, gt - i - 1));
                if (tag.compare(0, 2, "br") == 0) o += br_sep;
                i = gt + 1;
                continue;
            }
        }
        o += s[i++];
    }
    return o;
}
// Remove innermost {{...}} repeatedly.
void remove_templates(std::string& s) {
    for (int guard = 0; guard < 60; ++guard) {
        std::size_t e = s.find("}}");
        if (e == npos) break;
        std::size_t b = s.rfind("{{", e);
        if (b == npos) { s.erase(e, 2); continue; }
        s.erase(b, e + 2 - b);
    }
    std::size_t p = s.find("{{");
    if (p != npos) s.erase(p);
}

// ---------------------------------------------------------------- links
struct Link {
    std::string target, display;
    std::size_t pos = 0, end = 0;  // [pos, end) covers the whole [[...]]
};
// Parse a [[...]] starting at s[i]. Returns false for malformed or namespace links
// (Image:, Category:, interlanguage links: a colon not followed by a space).
bool parse_link_at(const std::string& s, std::size_t i, Link& L, bool& skippable) {
    skippable = false;
    std::size_t e = s.find("]]", i + 2);
    if (e == npos) return false;
    std::string inner = s.substr(i + 2, e - i - 2);
    L.pos = i;
    L.end = e + 2;
    if (inner.find("[[") != npos || inner.size() > 200) { skippable = true; return false; }
    std::size_t bar = inner.find('|');
    std::string target = trim(bar == npos ? inner : inner.substr(0, bar));
    std::string display = bar == npos ? target : trim(inner.substr(inner.rfind('|') + 1));
    if (!target.empty() && target[0] == ':') target = target.substr(1);
    std::size_t colon = target.find(':');
    if (colon != npos && (colon + 1 >= target.size() || target[colon + 1] != ' ')) { skippable = true; return false; }
    std::size_t hash = target.find('#');
    if (hash == 0) { skippable = true; return false; }
    if (hash != npos) target = trim(target.substr(0, hash));
    if (target.empty() || target.size() > 80) { skippable = true; return false; }
    L.target = target;
    L.display = display.empty() ? target : display;
    return true;
}
std::vector<Link> find_links(const std::string& s) {
    std::vector<Link> out;
    for (std::size_t i = s.find("[["); i != npos; i = s.find("[[", i)) {
        Link L;
        bool skip = false;
        if (parse_link_at(s, i, L, skip)) { out.push_back(L); i = L.end; }
        else if (skip && L.end > i) i = L.end;
        else i += 2;
    }
    return out;
}
// Rewrite links as [[target]] or [[target|display]]; drop namespace links.
std::string canon_links(const std::string& s) {
    std::string o;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t b = s.find("[[", i);
        if (b == npos) { o += s.substr(i); break; }
        o += s.substr(i, b - i);
        Link L;
        bool skip = false;
        if (parse_link_at(s, b, L, skip)) {
            o += "[[" + L.target;
            if (lower(L.display) != lower(L.target)) o += "|" + L.display;
            o += "]]";
            i = L.end;
        } else if (skip && L.end > b) {
            i = L.end;
        } else {
            o += "[[";
            i = b + 2;
        }
    }
    return o;
}
std::string plain_text(const std::string& in) {
    std::string s = in, o;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t b = s.find("[[", i);
        if (b == npos) { o += s.substr(i); break; }
        o += s.substr(i, b - i);
        Link L;
        bool skip = false;
        if (parse_link_at(s, b, L, skip)) { o += L.display; i = L.end; }
        else if (skip && L.end > b) i = L.end;
        else i = b + 2;
    }
    std::string r;
    for (std::size_t k = 0; k < o.size();) {
        if (o[k] == '\'' && k + 1 < o.size() && o[k + 1] == '\'') { while (k < o.size() && o[k] == '\'') ++k; continue; }
        if (o[k] == '&') {
            std::size_t sc = o.find(';', k);
            if (sc != npos && sc - k <= 8) { r += (o.compare(k, 6, "&nbsp;") == 0 ? " " : ""); k = sc + 1; continue; }
        }
        r += o[k++];
    }
    std::string c;
    for (char ch : r) {
        if (std::isspace(static_cast<unsigned char>(ch))) { if (!c.empty() && c.back() != ' ') c += ' '; }
        else c += ch;
    }
    return trim(c);
}
// Value of an infobox/table field: first link target, else cleaned text.
std::string clean_value(const std::string& raw) {
    std::string v = raw;
    erase_refs(v);
    v = strip_tags(v, "\x1e");
    std::size_t br = v.find('\x1e');
    if (br != npos) v = v.substr(0, br);
    auto links = find_links(v);
    if (!links.empty()) return links[0].target;
    remove_templates(v);
    std::string p = plain_text(v);
    std::size_t paren = p.find(" (");
    if (paren != npos) p = trim(p.substr(0, paren));
    if (p.empty() || p.size() > 80 || p.find('=') != npos || p.find('|') != npos) return "";
    return p;
}
std::vector<std::string> all_link_targets(const std::string& raw, std::size_t cap) {
    std::string v = raw;
    erase_refs(v);
    v = strip_tags(v, " ");
    std::vector<std::string> out;
    for (auto& L : find_links(v)) {
        out.push_back(L.target);
        if (out.size() >= cap) break;
    }
    return out;
}

// ---------------------------------------------------------------- facts
struct Fact {
    std::string subj, rel, obj, source, page, snippet;
    std::uint64_t off = 0;
    int count = 0;
};
std::vector<Fact> g_facts;
std::unordered_map<std::string, std::size_t> g_fact_index;
std::unordered_map<std::string, std::string> g_title_index;  // norm_key(title) -> title
std::unordered_map<std::string, std::string> g_alias;        // norm_key(redirect) -> norm_key(target)
std::map<std::string, long> g_unmapped_keys;

std::string norm_key(const std::string& in) {
    std::string s = lower(trim(in));
    for (const char* a : {"the ", "a ", "an "})
        if (starts_with(s, a)) { s = s.substr(std::strlen(a)); break; }
    return s;
}
std::string norm_field_key(const std::string& in) {
    std::string o;
    for (char c : lower(in))
        if (std::isalnum(static_cast<unsigned char>(c))) o += c;
    return o;
}

const std::unordered_map<std::string, std::string> kKeyRel = {
    {"capital", "capital"}, {"capitalcity", "capital"}, {"largestcity", "largest_city"},
    {"author", "author"}, {"authors", "author"}, {"writer", "writer"}, {"writers", "writer"},
    {"composer", "composer"}, {"composers", "composer"}, {"director", "director"}, {"directedby", "director"},
    {"producer", "producer"}, {"artist", "artist"}, {"founder", "founder"}, {"founders", "founder"},
    {"foundedby", "founder"}, {"inventor", "inventor"}, {"discoverer", "discoverer"},
    {"birthplace", "birthplace"}, {"placeofbirth", "birthplace"}, {"deathplace", "deathplace"},
    {"placeofdeath", "deathplace"}, {"spouse", "spouse"}, {"spouses", "spouse"}, {"currency", "currency"},
    {"officiallanguages", "language"}, {"officiallanguage", "language"}, {"languages", "language"},
    {"language", "language"}, {"genre", "genre"}, {"predecessor", "predecessor"}, {"successor", "successor"},
    {"headquarters", "headquarters"}, {"diet", "diet"}, {"doctoraladvisor", "teacher"},
    {"academicadvisors", "teacher"}, {"academicadvisor", "teacher"}, {"advisor", "teacher"},
    {"teacher", "teacher"}, {"doctoralstudents", "student"}, {"notablestudents", "student"},
    {"students", "student"}, {"nationality", "nationality"}, {"occupation", "occupation"}};
// Relations whose infobox value is a list: keep every link, not just the first.
bool is_list_rel(const std::string& r) {
    static const std::set<std::string> L = {"student", "language", "genre", "diet", "founder", "author", "writer", "composer"};
    return L.count(r) > 0;
}

void add_fact(const std::string& subj, const std::string& rel, const std::string& obj_in, const std::string& source,
              const std::string& page, std::uint64_t off, const std::string& snippet) {
    std::string obj = trim(obj_in);
    if (subj.empty() || obj.empty() || obj.size() > 80 || lower(obj) == lower(subj)) return;
    std::string key = norm_key(subj) + "\x1f" + rel + "\x1f" + lower(obj);
    auto it = g_fact_index.find(key);
    if (it != g_fact_index.end()) { ++g_facts[it->second].count; return; }
    g_fact_index[key] = g_facts.size();
    g_facts.push_back({subj, rel, obj, source, page, truncate(snippet, 220), off, 1});
}

bool is_person_rel(const std::string& r) {
    static const std::set<std::string> P = {"author", "writer", "composer", "director", "producer", "artist", "founder",
                                            "inventor", "discoverer", "spouse", "teacher", "student"};
    return P.count(r) > 0;
}

void emit_field(const std::string& key_raw, const std::string& val_raw, const std::string& page, std::uint64_t off,
                const char* kind) {
    std::string nk = norm_field_key(key_raw);
    auto it = kKeyRel.find(nk);
    if (it == kKeyRel.end()) { ++g_unmapped_keys[nk]; return; }
    std::string snip = trim(key_raw) + " = " + truncate(trim(val_raw), 120);
    std::string src = std::string(kind) + ":" + nk;
    if (is_list_rel(it->second)) {
        for (auto& t : all_link_targets(val_raw, 6)) add_fact(page, it->second, t, src, page, off, snip);
        if (all_link_targets(val_raw, 1).empty()) add_fact(page, it->second, clean_value(val_raw), src, page, off, snip);
    } else {
        std::string v = clean_value(val_raw);
        // unlinked person values that are lists or citation strings are not one name
        bool linked = !all_link_targets(val_raw, 1).empty();
        if (!linked && is_person_rel(it->second) && v.find_first_of(",;&") != npos) return;
        add_fact(page, it->second, v, src, page, off, snip);
    }
}

// Split on '|' outside [[...]] and {{...}}.
std::vector<std::string> split_top(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
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

std::string clean_table_key(const std::string& cell) {
    std::string p = plain_text(cell);
    while (!p.empty() && (p[0] == '-' || p[0] == '*' || p[0] == ' ')) p.erase(0, 1);
    while (!p.empty() && (p.back() == ':' || p.back() == ' ')) p.pop_back();
    return p;
}

}  // namespace

// ================================================================ sentence patterns
namespace {

enum class Mode { Fwd1, FwdMulti, ByObj, Inv, Poss, CapRev };
struct Pat {
    const char* id;
    std::vector<const char*> must;  // cheap prefilter: any of these (lowercase) must occur
    std::regex re;
    const char* rel;
    Mode mode;
};

std::vector<Pat> make_patterns() {
    auto R = [](const char* p) { return std::regex(p, std::regex::ECMAScript | std::regex::icase); };
    const char* kBy = R"((?:\]\])?s?(?:\s+(?:written|composed|created|directed|painted|authored))?\s+by\b)";
    std::vector<Pat> v;
    auto noun = [&](const char* id, const char* nouns, const char* rel) {
        std::string re = std::string(R"(\b(?:is|was|are|were)\s+(?:an?|the)\b[^.;]{0,80}?\b(?:)") + nouns + R"()\b)" + kBy;
        v.push_back({id, {" by "}, R(re.c_str()), rel, Mode::ByObj});
    };
    noun("by_author_noun", "novel|novella|book|play|poem|short story|essay|treatise|trilogy", "author");
    noun("by_composer_noun", "composition|symphony|symphonic poem|opera|ballet|concerto|oratorio|sonata|suite|tone poem|rhapsody|overture", "composer");
    noun("by_director_noun", "film|movie", "director");
    noun("by_artist_noun", "painting|sculpture", "artist");
    v.push_back({"written_by", {"written"}, R(R"(\bwritten\s+by\b)"), "author", Mode::ByObj});
    v.push_back({"composed_by", {"composed"}, R(R"(\bcomposed\s+by\b)"), "composer", Mode::ByObj});
    v.push_back({"directed_by", {"directed"}, R(R"(\bdirected\s+by\b)"), "director", Mode::ByObj});
    v.push_back({"painted_by", {"painted"}, R(R"(\bpainted\s+by\b)"), "artist", Mode::ByObj});
    v.push_back({"founded_by", {"founded", "established"}, R(R"(\b(?:founded|established)\s+by\b)"), "founder", Mode::ByObj});
    v.push_back({"invented_by", {"invented", "devised"}, R(R"(\b(?:invented|devised)\s+by\b)"), "inventor", Mode::ByObj});
    v.push_back({"discovered_by", {"discovered"}, R(R"(\bdiscovered\s+by\b)"), "discoverer", Mode::ByObj});
    // teacher(subject) = object
    v.push_back({"studied_with", {"studied", "trained", "learned", "apprenticed"},
                 R(R"(\b(?:studied|trained|learned|apprenticed|worked)\s+(?:with|under)\b)"), "teacher", Mode::Fwd1});
    v.push_back({"student_of", {"student", "pupil", "disciple", "apprentice", "prot"},
                 R(R"(\b(?:was|were|became|is|as)\s+(?:an?\s+|the\s+)?(?:\w+\s+)?(?:student|pupil|disciple|apprentice|protege|protégé)s?\s+of\b)"),
                 "teacher", Mode::Fwd1});
    v.push_back({"taught_by", {" by"}, R(R"(\b(?:taught|tutored|instructed|educated|mentored|coached|schooled)\s+by\b)"), "teacher", Mode::Fwd1});
    v.push_back({"his_teacher", {"teacher", "tutor", "mentor", "master", "instructor"},
                 R(R"(\b(?:his|her)\s+(?:own\s+)?(?:teacher|tutor|mentor|master|instructor)\s*,?\s*)"), "teacher", Mode::Fwd1});
    // inverse: "<Title> taught [[Y]]" gives teacher(Y) = Title
    v.push_back({"taught_obj", {"taught", "tutored", "mentored", "instructed", "coached"},
                 R(R"(\b(?:taught|tutored|mentored|instructed|coached)\b(?!\s+by\b)\s*)"), "teacher", Mode::Inv});
    v.push_back({"tutor_of", {"tutor", "teacher", "mentor", "instructor"},
                 R(R"(\b(?:tutor|teacher|mentor|instructor)\s+(?:of|to)\s+(?:the\s+)?)"), "teacher", Mode::Inv});
    v.push_back({"poss_tutor", {"'s"},
                 R(R"(\[\[([^\]|]+)(?:\|[^\]]*)?\]\]\s+(?:was|became|served as|is)\s+(?:the\s+)?([^'\[\]]{1,40}?)'s\s+(?:tutor|teacher|mentor|instructor|master)\b)"),
                 "teacher", Mode::Poss});
    // diet
    v.push_back({"feeds_on", {"feed", "fed ", "feast", "prey", "graz", "subsist"},
                 R(R"(\b(?:feed|feeds|feeding|fed|feast|feasts|prey|preys|graze|grazes|subsist|subsists)\s+(?:(?:almost|mainly|mostly|primarily|exclusively|only|largely|chiefly|entirely|principally|solely|usually|often|typically|partly)\s+)*(?:on|upon)\s+(?:the\s+)?)"),
                 "diet", Mode::FwdMulti});
    v.push_back({"eats", {" eat", "consume", "devour"}, R(R"(\b(?:eat|eats|eating|consume|consumes|devour|devours)\s+(?:the\s+)?)"), "diet", Mode::FwdMulti});
    v.push_back({"diet_is", {"diet"},
                 R(R"(\bdiet\s+(?:consists?|is|includes?|comprises?|are)\s+(?:(?:mainly|mostly|primarily|largely|entirely|chiefly)\s+)?(?:of\s+)?)"),
                 "diet", Mode::FwdMulti});
    // capital
    v.push_back({"capital_is", {"capital"},
                 R(R"(\bcapital(?:\s+city)?(?:\s+of\s+[^.;,]{1,40}?)?\s+(?:is|was)\s+(?:the\s+)?(?:city\s+of\s+)?)"), "capital", Mode::Fwd1});
    v.push_back({"is_capital_of", {"capital"},
                 R(R"(\[\[([^\]|]+)(?:\|[^\]]*)?\]\]\s*(?:is|was)\s+the\s+(?:\w+\s+){0,2}capital(?:\s+city)?\s+of\s+(?:the\s+)?\[\[([^\]|]+)(?:\|[^\]]*)?\]\])"),
                 "capital", Mode::CapRev});
    return v;
}

// Forms of the page title that count as "the article subject is mentioned".
std::vector<std::string> title_forms(const std::string& title) {
    std::string t = title;
    std::size_t p = t.find(" (");
    if (p != npos) t = t.substr(0, p);
    t = lower(trim(t));
    std::vector<std::string> f{t};
    for (const char* a : {"the ", "a ", "an "})
        if (starts_with(t, a)) { f.push_back(t.substr(std::strlen(a))); break; }
    return f;
}
bool boundary_at(const std::string& s, std::size_t i) { return i >= s.size() || !std::isalnum(static_cast<unsigned char>(s[i])); }
// The word before a title mention must be absent (sentence start) or a function word, so that
// "Seidokan Aikido, founded by ..." is not read as a statement about "Aikido".
bool function_word_before(const std::string& prefix, std::size_t p) {
    static const std::set<std::string> ok = {"the", "a", "an", "of", "in", "on", "at", "for", "and", "or", "that", "this",
                                             "its", "his", "her", "their", "as", "while", "since", "because", "although",
                                             "when", "if", "then", "but", "by", "with", "from", "to", "also", "modern"};
    std::size_t i = p;
    while (i > 0 && (prefix[i - 1] == '[' || prefix[i - 1] == '"' || prefix[i - 1] == ' ')) --i;
    if (i == 0) return true;
    char c = prefix[i - 1];
    if (!std::isalnum(static_cast<unsigned char>(c))) return true;  // punctuation, ']' etc.: new clause
    std::size_t j = i;
    while (j > 0 && std::isalnum(static_cast<unsigned char>(prefix[j - 1]))) --j;
    return ok.count(prefix.substr(j, i - j)) > 0;
}
// Position just after the first acceptable title mention, or npos.
std::size_t mention_end(const std::string& prefix, const std::vector<std::string>& forms) {
    for (auto& f : forms) {
        if (f.empty()) continue;
        for (std::size_t p = prefix.find(f); p != npos; p = prefix.find(f, p + 1)) {
            if (p > 0 && std::isalnum(static_cast<unsigned char>(prefix[p - 1]))) continue;
            if (!function_word_before(prefix, p)) continue;
            std::size_t e = p + f.size();
            if (boundary_at(prefix, e)) return e;
            if (prefix[e] == 's' && boundary_at(prefix, e + 1)) return e + 1;
            if (prefix.compare(e, 2, "es") == 0 && boundary_at(prefix, e + 2)) return e + 2;
        }
    }
    return npos;
}

// End of the clause that holds the object(s): next sentence/clause stop.
std::size_t clause_end(const std::string& s, std::size_t from) {
    std::string l = lower(s);
    std::size_t e = s.size();
    for (const char* st : {";", " but ", " which ", " while ", " that ", " when ", " where ", " who ", " whom ", " whose ",
                           " and then ", " although ", " because ", " after ", " before "}) {
        std::size_t p = l.find(st, from);
        if (p != npos) e = std::min(e, p);
    }
    // a '.' outside a link ends the clause too
    int br = 0;
    for (std::size_t i = from; i < e; ++i) {
        if (s.compare(i, 2, "[[") == 0) { ++br; ++i; }
        else if (s.compare(i, 2, "]]") == 0) { br = std::max(0, br - 1); ++i; }
        else if (s[i] == '.' && br == 0) { e = i; break; }
    }
    return e;
}
// Links chained like "[[a]]s and [[b]]" or "[[a]], [[b]], and [[c]]" starting at the first link at/after 'from'.
std::vector<Link> chain_links(const std::string& s, std::size_t from, std::size_t maxdist, std::size_t cap, bool multi) {
    std::vector<Link> out;
    std::string w = s.substr(from);
    auto links = find_links(w);
    if (links.empty() || links[0].pos > maxdist) return out;
    out.push_back(links[0]);
    if (!multi) return out;
    std::size_t end = links[0].end;
    for (std::size_t k = 1; k < links.size() && out.size() < cap; ++k) {
        std::string gap = lower(w.substr(end, links[k].pos - end));
        std::size_t i = 0;
        while (i < gap.size() && i < 3 && std::islower(static_cast<unsigned char>(gap[i]))) ++i;  // plural suffix
        std::string rest = trim(gap.substr(i));
        bool ok = rest == "," || rest == ", and" || rest == ", or" || rest == "and" || rest == "or" || rest == "/" || rest == ",and";
        if (!ok) break;
        out.push_back(links[k]);
        end = links[k].end;
    }
    return out;
}
const std::set<std::string> kRoles = {"composer", "writer", "author", "novelist", "playwright", "poet", "philosopher", "director",
                                      "filmmaker", "painter", "artist", "sculptor", "architect", "scientist", "mathematician",
                                      "dramatist", "essayist", "satirist", "musician", "economist", "historian", "physicist",
                                      "engineer", "journalist", "screenwriter", "producer", "chemist", "inventor", "explorer"};
// Object of a "... by <modifiers> <role> [[Person]]" phrase: the link right after a role noun, else the first link.
bool by_object(const std::string& s, std::size_t from, std::string& out) {
    std::size_t e = clause_end(s, from);
    std::size_t comma = s.find(',', from);
    if (comma != npos) e = std::min(e, comma);
    std::string w = s.substr(from, e - from);
    auto links = find_links(w);
    if (links.empty()) return false;
    for (auto& L : links) {
        std::string pre = lower(trim(w.substr(0, L.pos)));
        std::size_t sp = pre.find_last_of(' ');
        std::string word = sp == npos ? pre : pre.substr(sp + 1);
        if (kRoles.count(word)) { out = L.target; return true; }
    }
    out = links[0].target;
    return true;
}

void match_sentence(const std::string& s, const std::string& title, const std::string& page, std::uint64_t off) {
    static const std::vector<Pat> pats = make_patterns();
    std::string ls = lower(s);
    auto forms = title_forms(title);
    std::set<std::string> seen;
    auto emit = [&](const std::string& subj, const char* rel, const std::string& obj, const char* id) {
        std::string k = std::string(rel) + "\x1f" + norm_key(subj) + "\x1f" + lower(obj);
        if (!seen.insert(k).second) return;
        add_fact(subj, rel, obj, std::string("pat:") + id, page, off, s);
    };
    for (const auto& P : pats) {
        bool ok = false;
        for (auto m : P.must) if (ls.find(m) != npos) { ok = true; break; }
        if (!ok) continue;
        int nm = 0;
        for (auto it = std::sregex_iterator(s.begin(), s.end(), P.re); it != std::sregex_iterator() && nm < 3; ++it, ++nm) {
            const std::smatch& m = *it;
            std::size_t mpos = static_cast<std::size_t>(m.position()), mend = mpos + static_cast<std::size_t>(m.length());
            if (P.mode == Mode::Poss) {
                std::string poss = lower(trim(m[2].str())), tl = forms[0];
                if (poss.empty() || tl.compare(0, poss.size(), poss) != 0 || (tl.size() > poss.size() && tl[poss.size()] != ' ')) continue;
                emit(title, P.rel, m[1].str(), P.id);
                continue;
            }
            if (P.mode == Mode::CapRev) {
                emit(m[2].str(), P.rel, m[1].str(), P.id);
                continue;
            }
            std::size_t me = mention_end(ls.substr(0, mend), forms);  // the page subject must be named first
            if (me == npos) continue;
            if (P.mode == Mode::ByObj && mpos > me && ls.substr(me, mpos - me).find(',') != npos) continue;  // other clause
            if (P.mode == Mode::ByObj) {
                std::string obj;
                if (by_object(s, mend, obj)) emit(title, P.rel, obj, P.id);
            } else if (P.mode == Mode::Fwd1) {
                for (auto& L : chain_links(s.substr(0, clause_end(s, mend)), mend, 40, 1, false)) emit(title, P.rel, L.target, P.id);
            } else if (P.mode == Mode::FwdMulti) {
                for (auto& L : chain_links(s.substr(0, clause_end(s, mend)), mend, 60, 6, true)) emit(title, P.rel, L.target, P.id);
            } else if (P.mode == Mode::Inv) {
                for (auto& L : chain_links(s.substr(0, clause_end(s, mend)), mend, 6, 6, true)) {
                    if (!std::isupper(static_cast<unsigned char>(L.target[0]))) continue;  // people/places are capitalised
                    emit(L.target, P.rel, title, P.id);
                }
            }
        }
    }
}

// Normalise a wiki line to plain-ish sentences that keep [[links]].
std::string strip_parens(const std::string& s) {
    std::string o;
    int link = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s.compare(i, 2, "[[") == 0) { ++link; o += "[["; ++i; continue; }
        if (s.compare(i, 2, "]]") == 0) { link = std::max(0, link - 1); o += "]]"; ++i; continue; }
        if (s[i] == '(' && link == 0) {
            std::size_t e = s.find(')', i);
            if (e != npos && e - i < 400) { i = e; continue; }
        }
        o += s[i];
    }
    return o;
}
std::vector<std::string> split_sentences(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    int link = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        cur += s[i];
        if (s.compare(i, 2, "[[") == 0) { ++link; cur += '['; ++i; continue; }
        if (s.compare(i, 2, "]]") == 0) { link = std::max(0, link - 1); cur += ']'; ++i; continue; }
        if (link == 0 && (s[i] == '.' || s[i] == '?' || s[i] == '!') && i + 1 < s.size() && s[i + 1] == ' ') {
            std::size_t j = i + 1;
            while (j < s.size() && s[j] == ' ') ++j;
            if (j < s.size() && (std::isupper(static_cast<unsigned char>(s[j])) || s[j] == '[' || s[j] == '"')) {
                out.push_back(trim(cur));
                cur.clear();
            }
        }
    }
    if (!trim(cur).empty()) out.push_back(trim(cur));
    return out;
}

// ---------------------------------------------------------------- page processing
void process_page(const std::string& title, const std::string& raw, std::uint64_t base) {
    int depth = 0;
    std::string tpl_name;  // name of the outermost template we are inside
    bool have_pending = false;
    std::string pending_key;
    std::size_t pos = 0;
    while (true) {
        std::size_t nl = raw.find('\n', pos);
        if (nl == npos) nl = raw.size();
        std::uint64_t off = base + pos;
        std::string line = xml_decode(raw.substr(pos, nl - pos));
        std::string nc = line;
        erase_refs(nc);
        std::string t = trim(nc);
        int delta = count_of(nc, "{{") - count_of(nc, "}}");
        bool in_tpl = depth > 0;
        if (depth == 0 && delta > 0) {
            std::size_t b = nc.find("{{");
            std::size_t e = nc.find('|', b + 2);
            tpl_name = lower(trim(nc.substr(b + 2, e == npos ? npos : e - b - 2)));
        }
        bool citation = starts_with(tpl_name, "cite") || starts_with(tpl_name, "citation") || starts_with(tpl_name, "ref");
        bool table_row = !in_tpl && t.size() > 1 && t[0] == '|' && t[1] != '-' && t[1] != '+' && t[1] != '}';
        if (in_tpl || (delta > 0 && starts_with(t, "{{"))) {
            for (const auto& piece : split_top(nc)) {
                if (citation) break;
                std::size_t eq = piece.find('=');
                if (eq == npos) continue;
                std::string key = trim(piece.substr(0, eq));
                if (key.empty() || key.size() > 32 || key.find_first_of("[{<\"'") != npos) continue;
                emit_field(key, trim(piece.substr(eq + 1)), title, off, "infobox");
            }
            have_pending = false;
        } else if (table_row) {
            std::string body = t.substr(1);
            std::size_t dbl = body.find("||");
            if (have_pending && dbl == npos) {
                emit_field(pending_key, body, title, off, "table");
                have_pending = false;
            } else if (dbl != npos) {
                std::string kc = body.substr(0, dbl), vc = body.substr(dbl + 2);
                std::size_t d2 = vc.find("||");
                if (d2 != npos) vc = vc.substr(0, d2);
                auto split_br = [](const std::string& c) {
                    std::string x = c;
                    erase_refs(x);
                    x = strip_tags(x, "\x1e");
                    std::vector<std::string> o;
                    std::string cur;
                    for (char ch : x) { if (ch == '\x1e') { o.push_back(cur); cur.clear(); } else cur += ch; }
                    o.push_back(cur);
                    return o;
                };
                auto keys = split_br(kc), vals = split_br(vc);
                if (kc.find('=') == npos)
                    for (std::size_t i = 0; i < keys.size() && i < vals.size(); ++i) {
                        std::string k = clean_table_key(keys[i]);
                        if (!k.empty() && k.size() <= 40) emit_field(k, vals[i], title, off, "table");
                    }
                have_pending = false;
            } else {
                std::string k = clean_table_key(strip_tags(body, " "));
                bool colon = !trim(plain_text(strip_tags(body, " "))).empty() && plain_text(strip_tags(body, " ")).back() == ':';
                if (colon && body.find('=') == npos && k.size() <= 40 && !k.empty()) { pending_key = k; have_pending = true; }
                else have_pending = false;
            }
        } else {
            have_pending = false;
            bool skip = t.empty() || t[0] == '|' || t[0] == '{' || t[0] == '!' || t[0] == '=' || starts_with(t, "[[Image") ||
                        starts_with(t, "[[Category") || starts_with(t, "[[image") || starts_with(t, "[[category") || starts_with(t, "__");
            if (!skip && t.size() < 6000) {
                std::string s = t;
                while (!s.empty() && (s[0] == '*' || s[0] == '#' || s[0] == ':' || s[0] == ' ')) s.erase(0, 1);
                remove_templates(s);
                s = strip_tags(s, " ");
                std::string a;
                for (std::size_t i = 0; i < s.size();) {
                    if (s[i] == '\'' && i + 1 < s.size() && s[i + 1] == '\'') { while (i < s.size() && s[i] == '\'') ++i; continue; }
                    a += s[i++];
                }
                a = strip_parens(canon_links(a));
                for (auto& sent : split_sentences(a))
                    if (sent.size() >= 20 && sent.size() <= 1200) match_sentence(sent, title, title, off);
            }
        }
        depth = std::max(0, depth + delta);
        if (nl >= raw.size()) break;
        pos = nl + 1;
    }
}

long g_pages = 0, g_redirects = 0;

void build_from_corpus(const std::string& data, std::size_t limit) {
    std::size_t pos = 0;
    while (true) {
        std::size_t pb = data.find("<page>", pos);
        if (pb == npos || pb >= limit) break;
        std::size_t pe = data.find("</page>", pb);
        if (pe == npos || pe + 7 > limit) break;  // page straddles the train boundary: drop it
        pos = pe + 7;
        std::size_t tb = data.find("<title>", pb);
        std::size_t te = data.find("</title>", tb);
        if (tb == npos || te == npos || te > pe) continue;
        std::string title = xml_decode(data.substr(tb + 7, te - tb - 7));
        std::size_t xb = data.find("<text", te);
        if (xb == npos || xb > pe) continue;
        std::size_t xg = data.find('>', xb);
        if (xg == npos || xg > pe || data[xg - 1] == '/') continue;
        std::size_t xe = data.find("</text>", xg);
        if (xe == npos || xe > pe) continue;
        ++g_pages;
        std::string raw = data.substr(xg + 1, xe - xg - 1);
        std::string head = lower(trim(raw.substr(0, 40)));
        if (starts_with(head, "#redirect")) {
            auto links = find_links(xml_decode(raw.substr(0, std::min<std::size_t>(raw.size(), 300))));
            if (!links.empty()) { g_alias[norm_key(title)] = norm_key(links[0].target); ++g_redirects; }
            continue;
        }
        g_title_index[norm_key(title)] = title;
        process_page(title, raw, xg + 1);
    }
}

}  // namespace

// ================================================================ question classifier v0
namespace {

struct Parsed {
    std::vector<std::string> rels;
    std::string subj;
    bool inverse = false;  // answer is the subject whose object is 'subj' (e.g. "Who did X teach?")
    std::string tmpl;
};

// noun in "what is the <noun> of X" -> relation
const std::unordered_map<std::string, std::vector<std::string>> kNounRel = {
    {"capital", {"capital"}}, {"capital city", {"capital"}}, {"largest city", {"largest_city"}},
    {"currency", {"currency"}}, {"official language", {"language"}}, {"language", {"language"}},
    {"official languages", {"language"}}, {"author", {"author", "writer"}}, {"writer", {"writer", "author"}},
    {"composer", {"composer"}}, {"director", {"director"}}, {"founder", {"founder"}}, {"producer", {"producer"}},
    {"inventor", {"inventor"}}, {"discoverer", {"discoverer"}}, {"artist", {"artist"}}, {"spouse", {"spouse"}},
    {"wife", {"spouse"}}, {"husband", {"spouse"}}, {"successor", {"successor"}}, {"predecessor", {"predecessor"}},
    {"genre", {"genre"}}, {"headquarters", {"headquarters"}}, {"birthplace", {"birthplace"}},
    {"place of birth", {"birthplace"}}, {"place of death", {"deathplace"}}, {"nationality", {"nationality"}},
    {"teacher", {"teacher"}}, {"tutor", {"teacher"}}, {"mentor", {"teacher"}}, {"diet", {"diet"}}};

struct QTemplate {
    std::regex re;
    std::vector<std::string> rels;
    int subj_group;
    bool inverse;
    const char* name;
};

std::string clean_question(std::string q) {
    for (const char* bad : {"\xe2\x80\x99", "\xe2\x80\x98"}) {
        std::size_t p;
        while ((p = q.find(bad)) != npos) q.replace(p, 3, "'");
    }
    q = trim(q);
    while (!q.empty() && (q.back() == '?' || q.back() == '.' || q.back() == '!' || q.back() == ' ')) q.pop_back();
    std::string o;
    for (char c : q) {
        if (std::isspace(static_cast<unsigned char>(c))) { if (!o.empty() && o.back() != ' ') o += ' '; }
        else o += c;
    }
    return o;
}

bool classify(const std::string& question, Parsed& out) {
    static const std::vector<QTemplate> T = [] {
        auto R = [](const char* p) { return std::regex(std::string("^") + p + "$", std::regex::ECMAScript | std::regex::icase); };
        std::vector<QTemplate> v;
        v.push_back({R(R"(who (?:wrote|authored|is the author of|was the author of) (.+))"), {"author", "writer"}, 1, false, "who-wrote"});
        v.push_back({R(R"(who (?:composed|is the composer of|was the composer of) (.+))"), {"composer"}, 1, false, "who-composed"});
        v.push_back({R(R"(who (?:directed|is the director of|was the director of) (.+))"), {"director"}, 1, false, "who-directed"});
        v.push_back({R(R"(who (?:painted|is the artist of) (.+))"), {"artist"}, 1, false, "who-painted"});
        v.push_back({R(R"(who (?:founded|established) (.+))"), {"founder"}, 1, false, "who-founded"});
        v.push_back({R(R"(who invented (.+))"), {"inventor"}, 1, false, "who-invented"});
        v.push_back({R(R"(who discovered (.+))"), {"discoverer"}, 1, false, "who-discovered"});
        v.push_back({R(R"((?:what|which) (?:city )?(?:is|was) the capital(?: city)? of (.+))"), {"capital"}, 1, false, "capital-of"});
        v.push_back({R(R"((?:what|which) (?:city )?(?:is|was) (.+?)'s? capital(?: city)?)"), {"capital"}, 1, false, "x's-capital"});
        v.push_back({R(R"(capital(?: city)? of (.+))"), {"capital"}, 1, false, "capital-of-bare"});
        v.push_back({R(R"(who (?:was|is) (.+?)'s? (?:teacher|tutor|mentor|instructor|master))"), {"teacher"}, 1, false, "x's-teacher"});
        v.push_back({R(R"(who (?:taught|tutored|mentored|instructed|coached) (.+))"), {"teacher"}, 1, false, "who-taught"});
        v.push_back({R(R"(who (?:was|is) the (?:teacher|tutor|mentor|instructor) of (.+))"), {"teacher"}, 1, false, "teacher-of"});
        v.push_back({R(R"(who did (.+?) (?:study|train|learn) (?:with|under))"), {"teacher"}, 1, false, "studied-with"});
        v.push_back({R(R"(who (?:was|were) (.+?) (?:taught|tutored|mentored|instructed) by)"), {"teacher"}, 1, false, "taught-by"});
        v.push_back({R(R"(who did (.+?) (?:teach|tutor|mentor|instruct|coach))"), {"teacher"}, 1, true, "who-did-x-teach"});
        v.push_back({R(R"(what (?:does|do|did) (?:an? |the )?(.+?) (?:eat|feed on|consume))"), {"diet"}, 1, false, "what-does-eat"});
        v.push_back({R(R"(what (?:is|are) (?:an? |the )?(.+?)'s? diet)"), {"diet"}, 1, false, "x's-diet"});
        v.push_back({R(R"(what (?:is|are) the diet of (?:an? |the )?(.+))"), {"diet"}, 1, false, "diet-of"});
        return v;
    }();
    static const std::regex generic(R"(^(?:what|who|which) (?:is|was|are|were) the (.+?) of (.+)$)", std::regex::ECMAScript | std::regex::icase);
    std::string q = clean_question(question);
    std::smatch m;
    for (const auto& t : T)
        if (std::regex_match(q, m, t.re)) {
            out = {t.rels, trim(m[static_cast<std::size_t>(t.subj_group)].str()), t.inverse, t.name};
            return true;
        }
    if (std::regex_match(q, m, generic)) {
        auto it = kNounRel.find(lower(trim(m[1].str())));
        if (it != kNounRel.end()) { out = {it->second, trim(m[2].str()), false, "generic-the-N-of-X"}; return true; }
    }
    return false;
}

// Subject candidates: raw, article-stripped, then redirect targets; the title index decides which exist.
std::vector<std::string> resolve_subject(const std::string& subj, std::string& how) {
    std::string s = trim(subj);
    while (!s.empty() && (s.front() == '"' || s.front() == '\'')) s.erase(0, 1);
    while (!s.empty() && (s.back() == '"' || s.back() == '\'')) s.pop_back();
    std::vector<std::string> keys;
    auto push = [&](const std::string& k) { if (!k.empty() && std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k); };
    std::string raw = lower(s), stripped = norm_key(s);
    how = "no title match";
    for (const std::string& c : {raw, stripped}) {
        if (g_title_index.count(c)) { push(c); if (how == "no title match") how = "title match"; }
    }
    // plural question subject ("albatrosses") -> singular title
    for (const std::string& c : {raw, stripped}) {
        for (std::size_t cut : {std::size_t(2), std::size_t(1)}) {
            if (c.size() > cut + 2 && c.compare(c.size() - cut, cut, cut == 2 ? "es" : "s") == 0 && g_title_index.count(c.substr(0, c.size() - cut))) {
                push(c.substr(0, c.size() - cut));
                if (how == "no title match") how = "singularised title match";
            }
        }
    }
    for (const std::string& c : {raw, stripped}) {
        auto a = g_alias.find(c);
        if (a != g_alias.end() && a->second != c && !g_title_index.count(c)) { push(a->second); how = "redirect -> " + a->second; }
    }
    push(raw);
    push(stripped);
    return keys;
}

struct Candidate {
    std::string answer;
    int support = 0;
    int best_prio = 9;
    std::uint64_t first_off = ~0ull;
    const Fact* ev = nullptr;
};

int source_priority(const std::string& src) { return starts_with(src, "infobox") ? 0 : starts_with(src, "table") ? 1 : 2; }

std::unordered_map<std::string, std::vector<std::size_t>> g_by_subj, g_by_obj;
void build_indices() {
    for (std::size_t i = 0; i < g_facts.size(); ++i) {
        g_by_subj[norm_key(g_facts[i].subj)].push_back(i);
        g_by_obj[norm_key(g_facts[i].obj)].push_back(i);
    }
}

std::string norm_answer(const std::string& a) {
    std::string o;
    for (char c : lower(a))
        o += (std::isalnum(static_cast<unsigned char>(c)) || static_cast<unsigned char>(c) > 127) ? c : ' ';
    std::string r;
    for (char c : o) { if (c == ' ') { if (!r.empty() && r.back() != ' ') r += ' '; } else r += c; }
    r = trim(r);
    for (const char* a2 : {"the ", "a ", "an "})
        if (starts_with(r, a2)) { r = r.substr(std::strlen(a2)); break; }
    return r;
}

// Returns the answer (empty = abstain); prints the evidence.
std::string answer_question(const std::string& question) {
    Parsed p;
    std::printf("Q: %s\n", question.c_str());
    if (!classify(question, p)) {
        std::printf("   parse: no template matched\n   ABSTAIN\n");
        return "";
    }
    std::string how;
    auto keys = resolve_subject(p.subj, how);
    std::string rl;
    for (auto& r : p.rels) rl += (rl.empty() ? "" : "|") + r;
    std::printf("   parse: template=%s rel=%s%s subject=\"%s\" (%s)\n", p.tmpl.c_str(), rl.c_str(), p.inverse ? " (inverse)" : "",
                p.subj.c_str(), how.c_str());
    std::map<std::string, Candidate> cand;
    std::set<std::size_t> used;
    for (auto& k : keys) {
        auto& idx = p.inverse ? g_by_obj : g_by_subj;
        auto it = idx.find(k);
        if (it == idx.end()) continue;
        for (std::size_t fi : it->second) {
            const Fact& f = g_facts[fi];
            if (std::find(p.rels.begin(), p.rels.end(), f.rel) == p.rels.end()) continue;
            if (!used.insert(fi).second) continue;
            std::string ans = p.inverse ? f.subj : f.obj;
            Candidate& c = cand[lower(ans)];
            c.answer = ans;
            c.support += f.count;
            int pr = source_priority(f.source);
            if (pr < c.best_prio || (pr == c.best_prio && f.off < c.first_off)) { c.best_prio = pr; c.first_off = f.off; c.ev = &f; }
        }
    }
    if (cand.empty()) {
        std::printf("   no fact for this subject and relation\n   ABSTAIN\n");
        return "";
    }
    std::vector<Candidate> v;
    for (auto& kv : cand) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const Candidate& a, const Candidate& b) {
        if (a.support != b.support) return a.support > b.support;
        if (a.best_prio != b.best_prio) return a.best_prio < b.best_prio;
        return a.first_off < b.first_off;
    });
    const Candidate& top = v[0];
    std::printf("   ANSWER: %s\n", top.answer.c_str());
    std::printf("   evidence: %s(%s) = %s  [src=%s page=\"%s\" off=%llu support=%d]\n", top.ev->rel.c_str(), top.ev->subj.c_str(),
                top.ev->obj.c_str(), top.ev->source.c_str(), top.ev->page.c_str(), static_cast<unsigned long long>(top.ev->off),
                top.support);
    std::printf("   text: %s\n", top.ev->snippet.c_str());
    if (v.size() > 1) {
        std::printf("   others:");
        for (std::size_t i = 1; i < v.size() && i < 5; ++i) std::printf(" %s (%d)", v[i].answer.c_str(), v[i].support);
        std::printf("\n");
    }
    return top.answer;
}

double wilson_lower(double k, double n, double z = 1.96) {
    if (n <= 0) return 0;
    double p = k / n, z2 = z * z;
    double centre = (p + z2 / (2 * n)) / (1 + z2 / n);
    double margin = z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
    return centre - margin;
}

void print_stats() {
    std::map<std::string, std::array<long, 4>> per;  // distinct facts, occurrences, from infobox/table, from patterns
    for (auto& f : g_facts) {
        auto& a = per[f.rel];
        a[0] += 1;
        a[1] += f.count;
        (source_priority(f.source) < 2 ? a[2] : a[3]) += 1;
    }
    std::printf("pages=%ld redirects=%ld facts=%zu\n", g_pages, g_redirects, g_facts.size());
    std::printf("%-14s %8s %8s %10s %9s\n", "relation", "facts", "support", "infobox/tbl", "pattern");
    for (auto& kv : per)
        std::printf("%-14s %8ld %8ld %10ld %9ld\n", kv.first.c_str(), kv.second[0], kv.second[1], kv.second[2], kv.second[3]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus, qfile, ask, show;
    std::size_t limit = kTrainBytes;
    int sample = 0;
    unsigned seed = 20261027;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--limit") limit = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--ask") ask = next();
        else if (a == "--sample") sample = std::atoi(next().c_str());
        else if (a == "--seed") seed = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--show") show = next();
        else if (a == "--wilson") {
            double k = std::atof(next().c_str()), n = std::atof(next().c_str());
            std::printf("wilson95 lower bound for %g/%g: %.3f (point %.3f)\n", k, n, wilson_lower(k, n), n > 0 ? k / n : 0);
            return 0;
        } else if (corpus.empty()) corpus = a;
        else if (qfile.empty()) qfile = a;
    }
    if (corpus.empty()) {
        std::fprintf(stderr,
                     "usage: qa_infobox <enwik8> [questions.tsv] [--ask \"question\"] [--sample N [--seed S]] [--show Subject]\n"
                     "                  [--limit BYTES (default 7340032)]   |   qa_infobox --wilson K N\n");
        return 2;
    }
    std::ifstream in(corpus, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", corpus.c_str()); return 1; }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < limit) limit = data.size();
    build_from_corpus(data, limit);
    build_indices();
    print_stats();

    if (!show.empty()) {
        std::printf("--- facts for subject \"%s\"\n", show.c_str());
        std::string how;
        for (auto& k : resolve_subject(show, how))
            for (std::size_t fi : g_by_subj[k]) {
                const Fact& f = g_facts[fi];
                std::printf("  %s(%s) = %s  [%s n=%d off=%llu]\n", f.rel.c_str(), f.subj.c_str(), f.obj.c_str(), f.source.c_str(),
                            f.count, static_cast<unsigned long long>(f.off));
            }
    }
    if (sample > 0) {
        std::mt19937 rng(seed);
        std::vector<std::size_t> idx(g_facts.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        std::printf("--- random sample of %d facts (seed %u)\n", sample, seed);
        for (int i = 0; i < sample && i < static_cast<int>(idx.size()); ++i) {
            const Fact& f = g_facts[idx[static_cast<std::size_t>(i)]];
            std::printf("[%02d] %s(%s) = %s  [%s page=\"%s\" off=%llu n=%d]\n     text: %s\n", i + 1, f.rel.c_str(), f.subj.c_str(),
                        f.obj.c_str(), f.source.c_str(), f.page.c_str(), static_cast<unsigned long long>(f.off), f.count,
                        f.snippet.c_str());
        }
    }
    if (!ask.empty()) answer_question(ask);
    if (!qfile.empty()) {
        std::ifstream qf(qfile);
        std::string line;
        int total = 0, hit = 0, abstain = 0;
        while (std::getline(qf, line)) {
            if (trim(line).empty()) continue;
            std::size_t tab = line.find('\t');
            std::string q = tab == npos ? line : line.substr(0, tab);
            std::string acc = tab == npos ? "" : line.substr(tab + 1);
            std::string a = answer_question(q);
            ++total;
            if (a.empty()) { ++abstain; std::printf("   RESULT: ABSTAIN (accepted: %s)\n\n", acc.c_str()); continue; }
            bool ok = false;
            std::stringstream ss(acc);
            std::string v;
            while (std::getline(ss, v, '|')) if (!v.empty() && norm_answer(v) == norm_answer(a)) ok = true;
            hit += ok;
            std::printf("   RESULT: %s (accepted: %s)\n\n", ok ? "HIT" : "MISS", acc.c_str());
        }
        std::printf("SCORE: %d/%d correct, %d abstained, %d wrong\n", hit, total, abstain, total - hit - abstain);
    }
    return 0;
}
