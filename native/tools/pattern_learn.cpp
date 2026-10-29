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
// S4 (2026-10-28, docs/reports/S4_UNSEEN_AND_FRONTIER.md):
//   --frontier FILE [--frontier-test] [--frontier-split B] (with --gaps): estimates of the unseen (DISCOVERY 3.3) on
//   words, link entities, facts and their values, checked on held-out pages (UNSEEN, UNSEEN_REL, HEAPS lines; the test
//   half only with --frontier-test), Chapman per relation against Wikidata (CHAPMAN), and the frontier report: FILE, one
//   row per (type, relation) cell, with the two rankings (FRONTIER_TOP_READ, FRONTIER_TOP_SLICE). Self-assessment only.
//
// S7 (2026-10-28, docs/reports/S7_HYPOTHESIS_TESTS.md):
//   --test-kinds ENWIK8 SEEDS WD_VALUES PATTERNS OUT [--read-test]: s and c of the tests T1-T6 (DISCOVERY 7.1) on known-true
//   and known-false hypotheses, and 7.2's loop with pre-registration (I5, I6) and BH batch control. Self-assessment only.
//
// S8b (2026-10-29, docs/reports/S8B_LINK_PREDICTION.md):
//   --gen-links ENWIK8 SEEDS FACTS_F2 WD_VALUES PATTERNS S15_DIR OUT [--read-test]: link prediction, G6 (neighbourhoods) and
//   G7 (relation paths): rediscovery of hidden facts, their top candidates through S7's loop, the S15 temporal holdout, and
//   S6's plausibility with path evidence on S15's pairs. Cypha's own hypotheses only.
//
// S8a (2026-10-29, docs/reports/S8A_RULES_AND_CONTRADICTIONS.md):
//   --gen-rules ENWIK8 SEEDS FACTS_F2 WD_VALUES PATTERNS S15_DIR OUT [--read-test]: G2 rule induction (gen_rules.hpp) with
//   rediscovery on the seed store (10% hidden at random and least connected: Hits@1/@10, MRR) and the temporal holdout
//   (S15) on the 2006-text store, S7's tests (T2 back) and loop deciding the rules' predictions, and G14 explaining
//   contested slots. Cypha's own hypotheses only.
//
// S8c (2026-10-29, docs/reports/S8C_EVIDENCE_SOURCES.md):
//   --sources S15_DEV TABLES WD_VALUES SEEDS FACTS_F2 OUT S8A_R1 S8A_R2 S8B_PROMO: evidence sources other than Wikidata
//   (sources.hpp; DBpedia, the current Wikipedia lead, GeoNames; tables from scripts/evidence_sources.py), their
//   reliability on S15's choose half and by Dawid-Skene, and S8's dev promotions with the accepted ones, graded on S15's
//   confirm half. Cypha's own evidence only.
//
// S8d (2026-10-29, docs/reports/S8D_PROMOTION.md): `decide` needs a positive independent result (hypothesis.hpp).
//   --sources ... [--calibrate | --fresh FRESH_G2 FRESH_LINKS]: each source re-measured on the generators' choose-half dev
//   hypotheses (a source carries a promotion only if LR+ >= 10 there), and the fresh confirm-half set read once.
//   --gen-rules ... --fresh-confirm and --gen-links ... --fresh-confirm write that set: G2 and G6/G7 on S15 dev's
//   confirm-half slots with their facts hidden. No S15 label is read to make it.
//
// S9 (2026-10-29, docs/reports/S9_WORLDS_AND_FRAGILITY.md):
//   --worlds ENWIK8 SEEDS WD_VALUES FACTS_F2 S15_DEV OUT [--read-confirm]: possible worlds (worlds.hpp). T2 through worlds
//   on S7's dev set and, once, on S15 dev's choose half; fragility of the v2 store (single points of failure, blast radius
//   per source); hypothetical cards as data; I9 checked on every world. Bookkeeping only: it answers nothing.
//
// S10 (2026-10-29, docs/reports/S10_CURIOSITY_SCHEDULER.md):
//   --schedule OUT --dev DIR S15_FILE [--gate DIR S15_FILE] [--gaps GAP_MAP]: the curiosity scheduler (curiosity.hpp:
//   Thompson sampling, learning progress, floors, retirement) against the uniform scheduler, replayed on recorded P2
//   pools (a fresh G2 set, a fresh G6/G7 set and their source stage). --gen-rules / --gen-links --fresh-test-half write
//   the gate pool (S8d's construction on S15 test's bit-17 half); --sources --fresh writes each hypothesis's source
//   tests run (*.s8d_fresh_ran.tsv). Scheduling only: it answers nothing.
//
// S11a (2026-10-29, docs/reports/S11A_CONCEPTS_AND_OPEN_RELATIONS.md):
//   --gen-concepts ENWIK8 SEEDS WD_VALUES PATTERNS S15_DIR OUT [--attrs A1|A2] [--part ...] [--read-test]: G9 concept
//   formation (gen_concepts.hpp: closed concepts, candidate classes, class rules with P5's information-gain test), the
//   rediscovery of held-out classes against a random-class null, its predictions on hidden S15 slots through S8a's loop
//   (a pool for --sources --fresh and S10), and G11's unparsed phrase clusters graded against S15's properties.
//   Cypha's own hypotheses only.
//
// S11c (2026-10-29, docs/reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md; gen_analogy.hpp):
//   --gen-anomaly ENWIK8 SEEDS FACTS_F2 WD_VALUES PATTERNS S15_DIR OUT [--read-test] [--fresh-dev | --fresh-test]: G13 on
//   planted corruptions of store K and on store T's own anomalies (graded by Wikidata); G3 on planted deletions; pools.
//   --gen-analogy ENWIK8 SEEDS FACTS_F2 WD_VALUES PATTERNS S15_DIR VEC_PREFIX SIMVEC OUT [--read-test] [--fresh-dev |
//   --fresh-test]: G4 and G5 on S8a's R1 random hide and S15's temporal read; pools. --pool-grade s15|wd FILE HALF HYPS
//   SRC_FRESH grades a pool after S8d's source stage. Hypotheses for Cypha's own store only: it answers nothing.
//
// F2b (2026-10-29, docs/reports/F2_PATTERN_LEARNING.md, "F2b"; every flag off by default, outputs unchanged without them):
//   --f2b-xa reads every article, with the fact's subject a link ([L]) and the object any candidate or the page's own subject;
//   --f2b-wild 1|2 adds wildcard gaps (first token, <*>, last two | <*>, last two); --f2b-classes FILE adds word-class gaps;
//   --f2b-gold-split scores D4's gold fifth by halves (dev; test only with --f2b-read-test). --f2b-selftest checks the keys.
//
// S9b and S7d (2026-10-29; the S9 report, "S9b", and the S7 report, "S7d"): --worlds … --s9b writes each dev item's
//   Wikidata-subject flag, the store's phi table, Wikidata counted per claim and whether fragility points at errors;
//   --gaps … --gaps-spof SPOF adds S9's single points of failure as fragile gaps; --s7d S7_ITEMS S9_ITEMS S9B_ITEMS S15_ITEMS
//   OUT reads T2 in the loop, Wikidata as one source and K3 against K4. Cypha's own bookkeeping: it answers nothing.
//
// S12a (2026-10-29, docs/reports/S12A_DEDUCTION_LAWS_REFINEMENT.md; gen_laws.hpp, gen_deep.hpp):
//   --gen-laws ENWIK8 SEEDS FACTS_F2 OUT [--read-test]: G15's planted-law gate and its real read (the 2006 country and
//   place infoboxes; persons' years). --gen-deep ENWIK8 SEEDS FACTS_F2 WD_VALUES PATTERNS S15_DIR OUT [--read-test]
//   [--fresh-dev | --fresh-test]: G1 by depth, and the pools of G1 and G20 with the prospective log. Hypotheses and laws
//   for Cypha's own store only: it answers nothing. S12c (the same report, "S12c"): --gen-laws … [--power-min-exp]
//   [--order-relaxed MARGIN] [--s12c-dev] [--s12c-fresh write|read]; --gen-deep … [--s12c] [--g1-prior chain|cap3|depth].
//   G19 (refinement) was removed by S12c's registered rule; its code is in git at main's 0ce072f.
//
// Build: g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
// Usage: pattern_learn <enwik8> <outdir> [--cv] [--folds K] [--min-train N] [--base-rate A] [--sample N] [--seed S]
//          [--wikidata FILE [--wikidata-gold-only]] [--phi FILE] [--shard FILE --dev DEV --test TEST [--slice-end B] [--holdout-from B]]
//          [--raw] [--join FACTS] [--reach PARSE] [--gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F] [--demand-weight W] [--gaps-spof SPOF]]
//          [--frontier FILE [--frontier-test] [--frontier-split B]]
//        pattern_learn --wilson K N
//   <outdir> gets facts_f2.tsv (subject, relation, object, P, grade, n_evidence, best_pattern, source_page, byte_offset;
//   byte_offset is the start of the line holding the strongest evidence), facts_f2.evidence.tsv (the sentence for each fact),
//   seeds.tsv, patterns.tsv (kept first), pages.tsv (titles and text offsets), sample_admitted.tsv (random admitted facts,
//   fixed seed, to hand-judge) and stats.txt.
//   Default is one 80/20 split (fold 0 of K = 5); --cv repeats it over all K folds and pools the checks.
//
// Author: Odin Loch

#include "cypha/eval/abc_closed.hpp"
#include "cypha/eval/curiosity.hpp"
#include "cypha/eval/ecs.hpp"
#include "cypha/eval/epistemic.hpp"
#include "cypha/eval/gap_map.hpp"
#include "cypha/eval/gen_concepts.hpp"
#include "cypha/eval/gen_deep.hpp"
#include "cypha/eval/gen_analogy.hpp"
#include "cypha/eval/gen_links.hpp"
#include "cypha/eval/gen_laws.hpp"
#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/sources.hpp"
#include "cypha/eval/unknown_signals.hpp"
#include "cypha/eval/worlds.hpp"

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
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
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

// F2c (docs/reports/F2_PATTERN_LEARNING.md, "F2c"): --lexicon-type-check FILE [--lexicon-type-arm P|PL] (PL by default; the table is
// data/f2c/wd_seed_types.tsv). A Wikidata seed value widens
// the person lexicon only if Wikidata types it as human (P31 Q5); with arm PL it widens the place lexicon only if it has a coordinate
// location (P625). Text seeds widen as before. Off: every output unchanged.
namespace lexfix {
int g_arm = 0;                                                              // 0 off, 1 P, 2 PL
std::unordered_map<string, std::pair<std::vector<string>, bool>> g_types;  // item -> P31 classes, has P625
std::set<string> g_blocked_person, g_blocked_place;                         // values kept out (pass 1)
bool load(const string& path) {
    std::ifstream f(path);
    if (!f) return false;
    string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::size_t a = line.find('\t'), b = a == string::npos ? a : line.find('\t', a + 1);
        if (b == string::npos) continue;
        auto& t = g_types[line.substr(0, a)];
        std::size_t p = a + 1;
        while (p < b) {
            std::size_t q = line.find(',', p);
            if (q == string::npos || q > b) q = b;
            if (q > p) t.first.push_back(line.substr(p, q - p));
            p = q + 1;
        }
        t.second = line.compare(b + 1, 1, "1") == 0;
    }
    return true;
}
bool is_human(const string& q) {
    auto it = g_types.find(q);
    return it != g_types.end() && std::find(it->second.first.begin(), it->second.first.end(), "Q5") != it->second.first.end();
}
bool has_geo(const string& q) {
    auto it = g_types.find(q);
    return it != g_types.end() && it->second.second;
}
}  // namespace lexfix

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

// S4b-b (decided by measurement, PLAN ledger 55; docs/reports/S4_UNSEEN_AND_FRONTIER.md): the token spans of the date
// candidates among cands (find_dates' K_DATE). A link inside one is a linked year or day; see ecs::span_within. U leaves
// such links out of its mentions. G11 keeps them (S11a-b: the skip moved G11's dev choice, so it was not adopted there).
vector<std::pair<int, int>> date_spans(const vector<Cand>& cands) {
    vector<std::pair<int, int>> d;
    for (const Cand& c : cands)
        if (c.kind == K_DATE) d.push_back({c.b, c.e});
    return d;
}
constexpr bool kUSkipDateLinks = true;  // S4b-b: U's mentions leave out links inside date candidates

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
    std::size_t a = std::min({key.find("[S]"), key.find("[P]"), key.find("[L]")}), b = key.find("[O]");  // [L]: F2b only
    std::size_t first = std::min(a, b), last = std::max(a, b);
    return key.substr(first, last + 3 - first);
}

// F2b (docs/reports/F2_PATTERN_LEARNING.md, "F2b"; off by default, so every output is unchanged without the flags):
// generalised gaps beside each exact pattern, and a subject slot label for cross-article reading ([L]: the subject is a
// link in another article). Cleaned tokens only (not --raw).
int g_f2b_wild = 0;                              // --f2b-wild 1: first gap token, <*>, last two; 2: <*>, last two
std::unordered_map<string, string> g_f2b_class;  // --f2b-classes: gap word -> C<k> (F2's stop words, digits and months stay)
bool f2b_gen_on() { return g_f2b_wild > 0 || !g_f2b_class.empty(); }
// The keys of one pair of mentions, make_keys' rule with the subject slot labelled SL: the exact core (gen false), or the
// generalised cores (gen true), each in the four context variants. False when the spans overlap or the gap is 0 or > 6.
bool f2b_keys(const Sent& s, int sb, int se, const char* SL, int ob, int oe, bool gen, vector<string>& out) {
    out.clear();
    int fb, fe, sb2, se2;
    bool s_first;
    if (se <= ob) { s_first = true; fb = sb; fe = se; sb2 = ob; se2 = oe; }
    else if (oe <= sb) { s_first = false; fb = ob; fe = oe; sb2 = sb; se2 = se; }
    else return false;
    const int L = sb2 - fe;
    if (L < 1 || L > kMaxBetween) return false;
    vector<string> gap;
    for (int k = fe; k < sb2; ++k) gap.push_back(ctx_tok(s, k));
    auto join = [](const vector<string>& v) { string m; for (const string& x : v) { m += ' '; m += x; } return m; };
    vector<string> mids;
    const auto u = static_cast<std::size_t>(L);
    if (!gen) {
        mids.push_back(join(gap));
    } else {
        if (g_f2b_wild == 1 && L >= 4) mids.push_back(join({gap[0], "<*>", gap[u - 2], gap[u - 1]}));
        if (g_f2b_wild == 2 && L >= 3) mids.push_back(join({"<*>", gap[u - 2], gap[u - 1]}));
        if (!g_f2b_class.empty()) {
            vector<string> cg = gap;
            bool any = false;
            for (string& w : cg) {
                if (stop_words().count(w)) continue;
                auto it = g_f2b_class.find(w);
                if (it != g_f2b_class.end()) { w = it->second; any = true; }
            }
            if (any) mids.push_back(join(cg));
        }
    }
    const char* A = s_first ? SL : "[O]";
    const char* B = s_first ? "[O]" : SL;
    const string lc = ctx_tok(s, fb - 1), rc = ctx_tok(s, se2);
    for (const string& mid : mids) {
        const string core = string(A) + mid + " " + B;
        out.push_back(core);
        out.push_back(core + " " + rc);
        out.push_back(lc + " " + core);
        out.push_back(lc + " " + core + " " + rc);
    }
    return !out.empty();
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
    string vqid;  // F2c: the Wikidata value item (source 3 only)
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
namespace lexfix {
// F2c: may this seed's value widen the lexicon of type ty? Off, or a text seed: yes, as before.
bool widen_ok(const Seed& sd, RT ty) {
    if (g_arm == 0 || sd.src != 3) return true;
    if (ty == T_PERSON) {
        if (is_human(sd.vqid)) return true;
        g_blocked_person.insert(sd.o.k1);
        return false;
    }
    if ((ty == T_PLACE || ty == T_PLACE_LINK) && g_arm == 2) {
        if (has_geo(sd.vqid)) return true;
        g_blocked_place.insert(sd.o.k1);
        return false;
    }
    return true;
}
}  // namespace lexfix

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
    vector<string> gk;  // F2b
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
            if (f2b_gen_on() && f2b_keys(s, S.b, S.e, S.pron ? "[P]" : "[S]", c.b, c.e, true, gk))  // F2b
                for (const string& k : gk) {
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
    vector<string> gk;  // F2b
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
                    if (f2b_gen_on() && f2b_keys(s, S.b, S.e, S.pron ? "[P]" : "[S]", c.b, c.e, true, gk))  // F2b
                        for (const string& k : gk) {
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
        apply_patterns(s, spans, cands, px.forms.title_key, [&](int pid, const Cand& c, const Span& S) {
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
            // F2b: with generalised keys one pair of mentions can match several patterns; it is still one evidence item
            const string fam = f2b_gen_on() ? "@" + std::to_string(S.b) + "," + std::to_string(S.e) + "," + std::to_string(c.b) + "," + std::to_string(c.e) : p.family;
            if (f.ev.size() < 96) f.ev.push_back({pid, p.reliability, s.off, s.hash, s.idx, fam});
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
        else {
            const std::size_t before = g_seeds.size();
            add_seed(c[0], r, o, 3, "wikidata", 0);
            if (g_seeds.size() > before && c.size() > 4) g_seeds.back().vqid = c[4];  // F2c
            ++n_seed;
        }
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
//              fusion; p = its P over its P plus the best rival's (a text seed counts 20/21).
//   certainty  a stored extracted value that is Believed at grade Low, or Known on one sentence; p = P.
//   demand     --demand ROWS[,ROWS...]: cyphalm_trie qa rows, dev rows only; each tag adds demand to its slot (gap_map.hpp).
//   status     parked when no automatic rung can fill the slot now (no kept pattern for R, or the entity has no page), else
//              open, or queued when asked about. VOI = gap_map.hpp score(): EIG of a 0.9 / 0.9 test x (1 + w demand),
//              w = --demand-weight (0: 0 of 750 dev tags name their question's slot; the demand column is kept).
//   grading    with --dev (grading only, dev only): does a tag name its question's slot (subject: the in-slice title, else
//              --parse's E1 entity; relation: q_rels), and is that slot a gap?
struct GapOpt { string path, parse, spof; vector<string> demand; double floor = 0.05, demand_weight = 0.0; };  // spof: S9b's --gaps-spof
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
    if (!G.spof.empty()) {  // S9b: S9's single points of failure (--worlds OUT.spof.tsv) as fragile gaps
        std::ifstream f(G.spof);
        string l;
        std::getline(f, l);
        long rows = 0, added = 0, skipped = 0;
        while (std::getline(f, l)) {
            const vector<string> c = split(l, '\t');  // gap_id kind status entity relation object p top_source note
            if (c.size() < 8) continue;
            ++rows;
            const int e = at(c[3]);
            const std::size_t before = gm.size();
            const ecs::Gap* g = gm.add_fragile(c[3], c[4], std::atof(c[6].c_str()), e < 0 ? "unknown" : kT[type[e]], c[5] + " single point of failure, top source " + c[7]);
            if (!g) ++skipped;
            else added += gm.size() > before;
        }
        say("  GAPSPOF " + G.spof + ": rows " + std::to_string(rows) + ", fragile gaps " + std::to_string(added) + ", slots already asked about " + std::to_string(skipped));
    }
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

// S4 (docs/reports/S4_UNSEEN_AND_FRONTIER.md; DISCOVERY 3.3, 3.4): estimates of the unseen, checked on held-out pages,
// and the frontier report. Self-assessment only: nothing here answers a question or feeds CyphaLM.
//   items      words (runs of ASCII letters in a page's raw text, lower-cased), entities (link targets, norm_ent, no
//              #section, namespace and language links skipped), facts (subject, relation, object: text seeds and stored
//              extracted facts with P >= 0.5; an observation is a seed or an evidence sentence, on the page that holds
//              it) and values (relation, object of the same facts). A page is the sampling unit.
//   views      store: E = pages starting before --frontier-split (50,000,000), H = the rest; cyphalm: E = pages starting
//              before 7,340,032 (CyphaLM's wiki train), H = pages from 8,388,608; random: E and H by a title-hash bit.
//              H is split into dev and test by another hash bit; test is printed only with --frontier-test.
//   estimates  on E: GT0 = n1 / N; GTu = observations of items on exactly one page / N; Q1/U; Chao extrapolation of the
//              distinct items to E + H (pages as units, and observations); Heaps (words). Targets on H: the share of H's
//              observations (incidences) whose item is not in E; the distinct items in E + H. Intervals: a bootstrap over
//              H's pages (E fixed) for the target, a 20-group jackknife over E's pages for the estimate.
//   Chapman    per relation with a kept pattern: A text-seed slots, B slots where a kept pattern fires on the subject's
//              page, m both; N_ref = |A u B u W|, W the slots with a Wikidata value.
//   --frontier FILE (with --gaps): one row per (type, relation) cell: known, unseen in the slice ((entities - known) x
//              P(exists): S3's P(exists) is calibrated on empty slots), expected (known + unseen), coverage, certainty (mean P of the cell's slots; a text seed 20/21), demand (CyphaLM's tags),
//              the missing mass of the cell's fact observations over all pages, new facts per 10 MiB of comparable text,
//              and the relation's values (seen, Chao1 with pages as units).
struct FrontierOpt { string path; bool test = false; std::size_t split = 50000000; };
using S4Items = vector<vector<std::pair<std::uint32_t, std::uint32_t>>>;  // per page: (item, count), sorted by item

struct S4Kind {
    string name;
    S4Items pages;
    vector<int> group;  // a relation index for facts and values, else -1
    vector<int> type;   // facts: the subject's type (write_gaps' rule; -1 not a page)
};

void s4_finish_page(vector<std::uint32_t>& ids, vector<std::pair<std::uint32_t, std::uint32_t>>& out) {
    std::sort(ids.begin(), ids.end());
    for (std::size_t i = 0; i < ids.size();) {
        std::size_t j = i;
        while (j < ids.size() && ids[j] == ids[i]) ++j;
        out.push_back({ids[i], static_cast<std::uint32_t>(j - i)});
        i = j;
    }
    ids.clear();
}

// Words and entities of every page.
void s4_text_items(const string& data, const vector<PageRef>& pages, S4Kind& words, S4Kind& ents) {
    std::unordered_map<string, std::uint32_t> wid, eid;
    words.pages.assign(pages.size(), {});
    ents.pages.assign(pages.size(), {});
    vector<std::uint32_t> ids;
    string w;
    auto letter = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    for (std::size_t p = 0; p < pages.size(); ++p) {
        const std::size_t tb = pages[p].tb, te = pages[p].te;
        for (std::size_t i = tb; i < te;) {
            if (!letter(static_cast<unsigned char>(data[i]))) { ++i; continue; }
            w.clear();
            while (i < te && letter(static_cast<unsigned char>(data[i]))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(data[i++])));
            auto it = wid.try_emplace(w, static_cast<std::uint32_t>(wid.size())).first;
            ids.push_back(it->second);
        }
        s4_finish_page(ids, words.pages[p]);
        for (std::size_t i = data.find("[[", tb); i != npos && i < te; i = data.find("[[", i + 2)) {
            std::size_t j = i + 2;
            while (j < te && j < i + 300 && data[j] != '|' && data[j] != ']' && data[j] != '\n' && data[j] != '[') ++j;
            if (j >= te || (data[j] != '|' && data[j] != ']')) continue;
            string t = data.substr(i + 2, j - i - 2);
            if (std::size_t h = t.find('#'); h != npos) t.erase(h);
            if (std::size_t c = t.find(':'); c != npos && c > 0) {  // Image:, Category:, fr:, ... (a one-word prefix)
                bool ns = true;
                for (std::size_t k = 0; k < c; ++k) ns &= letter(static_cast<unsigned char>(t[k])) || t[k] == '-';
                if (ns) continue;
            }
            std::replace(t.begin(), t.end(), '_', ' ');
            const string k = norm_ent(xml_decode(t));
            if (k.empty()) continue;
            ids.push_back(eid.try_emplace(k, static_cast<std::uint32_t>(eid.size())).first->second);
        }
        s4_finish_page(ids, ents.pages[p]);
    }
    words.group.assign(wid.size(), -1);
    ents.group.assign(eid.size(), -1);
}

// Facts and values: text seeds and stored extracted facts (P >= 0.5), on the page holding each observation.
void s4_fact_items(const vector<PageRef>& pages, const vector<Fused>& fused, const std::unordered_map<string, int>& etype, S4Kind& facts,
                   S4Kind& vals) {
    vector<vector<std::uint32_t>> fp(pages.size()), vp(pages.size());
    std::unordered_map<string, std::uint32_t> fid, vid;
    auto page_at = [&](std::uint64_t off) -> long {
        auto it = std::upper_bound(pages.begin(), pages.end(), off, [](std::uint64_t o, const PageRef& p) { return o < p.tb; });
        if (it == pages.begin()) return -1;
        --it;
        return off < it->te ? static_cast<long>(it - pages.begin()) : -1;
    };
    auto add = [&](std::uint64_t off, const string& subj, int r, const string& okey) {
        const long p = page_at(off);
        if (p < 0 || okey.empty()) return;
        const string rs = std::to_string(r);
        const string sk = norm_ent(subj);
        auto f = fid.try_emplace(sk + "\x1f" + rs + "\x1f" + okey, static_cast<std::uint32_t>(fid.size()));
        if (f.second) {
            facts.group.push_back(r);
            const auto t = etype.find(sk);
            facts.type.push_back(t == etype.end() ? -1 : t->second);
        }
        auto v = vid.try_emplace(rs + "\x1f" + okey, static_cast<std::uint32_t>(vid.size()));
        if (v.second) vals.group.push_back(r);
        fp[static_cast<std::size_t>(p)].push_back(f.first->second);
        vp[static_cast<std::size_t>(p)].push_back(v.first->second);
    };
    for (const Seed& sd : g_seeds) if (sd.src < 3) add(sd.off, sd.subj, sd.rel, sd.o.k1);
    for (const Fused& fu : fused)
        if (fu.op.P >= 0.5)
            for (const EvRec& e : fu.f->ev) add(e.off, fu.f->subj, fu.f->rel, fu.f->okey);
    facts.pages.assign(pages.size(), {});
    vals.pages.assign(pages.size(), {});
    for (std::size_t p = 0; p < pages.size(); ++p) {
        s4_finish_page(fp[p], facts.pages[p]);
        s4_finish_page(vp[p], vals.pages[p]);
    }
}

// One kind in one view: E's statistics against one held-out set. role: 1 E, 2 H-dev, 3 H-test (0: not used).
struct S4Res {
    ecs::ClumpedStats st;
    std::size_t f1 = 0, f2 = 0;  // abundance: items seen once / twice
    double gt0 = 0, gtu = 0, q1u = 0, t_obs = 0, t_inc = 0, h_obs = 0, h_pages = 0, distinct = 0, chao_inc = 0, chao_ab = 0;
    double lo_obs = 0, hi_obs = 0, lo_inc = 0, hi_inc = 0, jk_gt0 = 0, jk_gtu = 0;
};

S4Res s4_eval(const S4Kind& K, const vector<char>& role, char h, int group, bool intervals) {
    S4Res R;
    const std::size_t n = K.group.size();
    vector<std::uint32_t> cnt(n, 0), unit(n, 0);
    vector<std::size_t> E;
    for (std::size_t p = 0; p < role.size(); ++p) if (role[p] == 1) E.push_back(p);
    auto in = [&](std::uint32_t i) { return group < 0 || K.group[i] == group; };
    auto stats = [&](const vector<std::uint32_t>& c, const vector<std::uint32_t>& u, std::size_t T, std::size_t* f1, std::size_t* f2) {
        vector<std::size_t> cc, uu;
        cc.reserve(n);
        uu.reserve(n);
        std::size_t a1 = 0, a2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!c[i] || !in(static_cast<std::uint32_t>(i))) continue;
            cc.push_back(c[i]);
            uu.push_back(u[i]);
            a1 += c[i] == 1;
            a2 += c[i] == 2;
        }
        if (f1) *f1 = a1, *f2 = a2;
        return ecs::clumped_stats(cc, uu, T);
    };
    for (std::size_t p : E)
        for (auto [i, c] : K.pages[p]) cnt[i] += c, ++unit[i];
    std::size_t T = 0;
    for (std::size_t p : E) {
        bool any = false;
        for (auto [i, c] : K.pages[p]) any |= in(i);
        T += any;
    }
    R.st = stats(cnt, unit, T, &R.f1, &R.f2);
    R.gt0 = ecs::good_turing_missing_mass(R.f1, R.st.total);
    R.gtu = ecs::unit_missing_mass(R.st);
    R.q1u = ecs::incidence_missing_mass(R.st);
    // Held-out: per page (observations new, observations, incidences new, incidences).
    vector<std::array<double, 4>> hp;
    vector<char> mark(n, 0);
    double newd = 0;
    for (std::size_t p = 0; p < role.size(); ++p) {
        if (role[p] != h) continue;
        std::array<double, 4> a{0, 0, 0, 0};
        for (auto [i, c] : K.pages[p]) {
            if (!in(i)) continue;
            const bool nw = cnt[i] == 0;
            a[0] += nw ? c : 0;
            a[1] += c;
            a[2] += nw;
            a[3] += 1;
            if (nw && !mark[i]) mark[i] = 1, ++newd;
        }
        if (a[3] > 0) hp.push_back(a);
    }
    double s[4] = {0, 0, 0, 0};
    for (const auto& a : hp) for (int k = 0; k < 4; ++k) s[k] += a[k];
    R.t_obs = s[1] > 0 ? s[0] / s[1] : 0;
    R.t_inc = s[3] > 0 ? s[2] / s[3] : 0;
    R.h_obs = s[1];
    R.h_pages = static_cast<double>(hp.size());
    R.distinct = static_cast<double>(R.st.s_obs) + newd;
    R.chao_inc = ecs::chao_extrapolated_richness(R.st.s_obs, R.st.q1, R.st.q2, R.st.units, R.h_pages);
    R.chao_ab = ecs::chao_extrapolated_richness(R.st.s_obs, R.f1, R.f2, R.st.total, R.h_obs);
    if (!intervals || hp.empty()) return R;
    std::mt19937_64 rng(20261028);
    vector<double> bo, bi;
    for (int b = 0; b < 2000; ++b) {
        double x[4] = {0, 0, 0, 0};
        for (std::size_t k = 0; k < hp.size(); ++k) {
            const auto& a = hp[rng() % hp.size()];
            for (int q = 0; q < 4; ++q) x[q] += a[q];
        }
        bo.push_back(x[1] > 0 ? x[0] / x[1] : 0);
        bi.push_back(x[3] > 0 ? x[2] / x[3] : 0);
    }
    std::sort(bo.begin(), bo.end());
    std::sort(bi.begin(), bi.end());
    R.lo_obs = bo[49], R.hi_obs = bo[1949], R.lo_inc = bi[49], R.hi_inc = bi[1949];
    // Delete-a-group jackknife over 20 groups of E's pages (by position).
    const int G = 20;
    vector<double> j0, ju;
    for (int g = 0; g < G; ++g) {
        vector<std::uint32_t> c2 = cnt, u2 = unit;
        std::size_t T2 = 0;
        for (std::size_t k = 0; k < E.size(); ++k) {
            if (static_cast<int>(k % G) != g) { bool any = false; for (auto [i, c] : K.pages[E[k]]) any |= in(i); T2 += any; continue; }
            for (auto [i, c] : K.pages[E[k]]) c2[i] -= c, --u2[i];
        }
        std::size_t a1 = 0, a2 = 0;
        const ecs::ClumpedStats st = stats(c2, u2, T2, &a1, &a2);
        j0.push_back(ecs::good_turing_missing_mass(a1, st.total));
        ju.push_back(ecs::unit_missing_mass(st));
    }
    auto se = [&](const vector<double>& v) {
        double m = 0, q = 0;
        for (double x : v) m += x;
        m /= G;
        for (double x : v) q += (x - m) * (x - m);
        return std::sqrt((G - 1.0) / G * q);
    };
    R.jk_gt0 = se(j0), R.jk_gtu = se(ju);
    return R;
}

// Heaps on E's growth curve (every 25th page, in page order), checked on E + H.
void s4_heaps(const S4Kind& W, const vector<char>& role, const S4Res& R, const string& label, const std::function<void(const string&)>& say) {
    vector<char> seen(W.group.size(), 0);
    vector<double> N, V;
    double tok = 0, typ = 0;
    std::size_t k = 0;
    for (std::size_t p = 0; p < role.size(); ++p) {
        if (role[p] != 1) continue;
        for (auto [i, c] : W.pages[p]) { tok += c; if (!seen[i]) seen[i] = 1, ++typ; }
        if (++k % 25 == 0) N.push_back(tok), V.push_back(typ);
    }
    N.push_back(tok), V.push_back(typ);
    const ecs::HeapsFit f = ecs::fit_heaps(N, V);
    const double vp = ecs::heaps_vocab(f, tok + R.h_obs), rate = f.beta * ecs::heaps_vocab(f, tok) / tok;
    char b[400];
    std::snprintf(b, sizeof b, "  HEAPS %s K %.3f beta %.4f r2 %.5f points %zu | vocab E+H predicted %.0f observed %.0f ratio %.4f | new-word rate predicted %.5f target %.5f ratio %.4f",
                  label.c_str(), f.K, f.beta, f.r2, f.n, vp, R.distinct, R.distinct > 0 ? vp / R.distinct : 0.0, rate, R.t_obs, R.t_obs > 0 ? rate / R.t_obs : 0.0);
    say(b);
}

void s4_line(const char* tag, const string& label, const S4Res& R, const std::function<void(const string&)>& say) {
    char b[900];
    auto rt = [](double e, double t) { return t > 0 ? e / t : 0.0; };
    std::snprintf(b, sizeof b,
                  "  %s %s | E pages %zu obs %zu items %zu n1 %zu q1 %zu | GT0 %.5f (jk se %.5f) GTu %.5f (jk se %.5f) Q1/U %.5f | target obs %.5f [%.5f, %.5f] inc %.5f [%.5f, %.5f] "
                  "H pages %.0f obs %.0f | GT0/t %.4f [%.4f, %.4f] GTu/t %.4f [%.4f, %.4f] Q1U/t %.4f | distinct E+H %.0f chao_pages %.0f (%.4f) chao_obs %.0f (%.4f)",
                  tag, label.c_str(), R.st.units, R.st.total, R.st.s_obs, R.f1, R.st.q1, R.gt0, R.jk_gt0, R.gtu, R.jk_gtu, R.q1u, R.t_obs, R.lo_obs, R.hi_obs, R.t_inc, R.lo_inc,
                  R.hi_inc, R.h_pages, R.h_obs, rt(R.gt0, R.t_obs), rt(R.gt0, R.hi_obs), rt(R.gt0, R.lo_obs), rt(R.gtu, R.t_obs), rt(R.gtu, R.hi_obs), rt(R.gtu, R.lo_obs),
                  rt(R.q1u, R.t_inc), R.distinct, R.chao_inc, rt(R.chao_inc, R.distinct), R.chao_ab, rt(R.chao_ab, R.distinct));
    say(b);
}

void write_frontier(const FrontierOpt& F, const GapOpt& G, const string& data, const vector<PageRef>& pages, const vector<Fused>& fused,
                    const std::function<void(const string&)>& say) {
    // Types (write_gaps' rule).
    std::unordered_map<string, int> etype;
    for (const PageRef& pg : pages) {
        const string lt = lower(pg.title), nk = norm_ent(pg.title);
        if (lt.compare(0, 8, "list of ") == 0 || lt.find("(disambiguation)") != npos || etype.count(nk)) continue;
        etype[nk] = g_country.count(nk) ? 0 : g_person.count(nk) ? 1 : g_place.count(nk) ? 2 : 3;
    }
    S4Kind words{"words", {}, {}, {}}, ents{"entities", {}, {}, {}}, facts{"facts", {}, {}, {}}, vals{"values", {}, {}, {}};
    s4_text_items(data, pages, words, ents);
    s4_fact_items(pages, fused, etype, facts, vals);
    // Views.
    struct View { string name; vector<char> role; };
    vector<View> views(3);
    views[0].name = "store", views[1].name = "cyphalm", views[2].name = "random";
    for (View& v : views) v.role.assign(pages.size(), 0);
    for (std::size_t p = 0; p < pages.size(); ++p) {
        const std::uint64_t hsh = fnv(pages[p].title);
        const char half = ((hsh >> 21) & 1) ? 3 : 2;
        views[0].role[p] = pages[p].tb < F.split ? 1 : half;
        views[1].role[p] = pages[p].tb < 7340032 ? 1 : pages[p].tb >= 8388608 ? half : 0;
        views[2].role[p] = ((hsh >> 22) & 1) ? half : 1;
    }
    say("\nS4 UNSEEN (items: words " + std::to_string(words.group.size()) + ", entities " + std::to_string(ents.group.size()) + ", facts " +
        std::to_string(facts.group.size()) + ", values " + std::to_string(vals.group.size()) + ")");
    for (const View& v : views)
        for (char h : {char(2), char(3)}) {
            if (h == 3 && (!F.test || v.name == "random")) continue;
            const string set = h == 2 ? "dev" : "test";
            for (const S4Kind* K : {&words, &ents, &facts, &vals}) {
                const S4Res R = s4_eval(*K, v.role, h, -1, true);
                s4_line("UNSEEN", v.name + " " + set + " " + K->name, R, say);
                if (K == &words) s4_heaps(words, v.role, R, v.name + " " + set, say);
            }
            for (const S4Kind* K : {&facts, &vals})
                for (int r = 0; r < kNRel; ++r) {
                    const S4Res R = s4_eval(*K, v.role, h, r, false);
                    if (R.h_obs < 1) continue;
                    char b[400];
                    std::snprintf(b, sizeof b, "  UNSEEN_REL %s %s %s %-13s E obs %zu items %zu | GT0 %.4f GTu %.4f | target %.4f H obs %.0f | GT0/t %.4f GTu/t %.4f",
                                  v.name.c_str(), set.c_str(), K->name.c_str(), kRel[r].name, R.st.total, R.st.s_obs, R.gt0, R.gtu, R.t_obs, R.h_obs,
                                  R.t_obs > 0 ? R.gt0 / R.t_obs : 0.0, R.t_obs > 0 ? R.gtu / R.t_obs : 0.0);
                    say(b);
                }
        }
    // Chapman per relation with a kept pattern: A text seeds, B kept patterns firing on the subject's page, W Wikidata.
    std::uint64_t kept = 0;
    for (const Pat& p : g_pats) kept |= std::uint64_t{1} << p.rel;
    std::set<std::pair<string, int>> A, B, W;
    for (const Seed& sd : g_seeds) {
        const string k = norm_ent(sd.subj);
        if (!etype.count(k) || !(kept >> sd.rel & 1)) continue;
        (sd.src < 3 ? A : W).insert({k, sd.rel});
    }
    for (auto& [key, v] : g_gold) {
        const string k = norm_ent(key.substr(0, key.find('\x1f')));
        const int r = rel_index(key.substr(key.find('\x1f') + 1));
        if (r >= 0 && etype.count(k) && (kept >> r & 1)) W.insert({k, r});
    }
    for (const PageRef& pg : pages) {
        const string lt = lower(pg.title);
        if (lt.compare(0, 8, "list of ") == 0 || lt.find("(disambiguation)") != npos) continue;
        PageCtx px;
        load_page(data, pg, px);
        vector<Cand> cands;
        const string k = norm_ent(pg.title);
        for (const Sent& s : px.sents) {
            auto spans = find_subject(s, px.forms);
            if (spans.empty()) continue;
            gen_cands(s, cands);
            apply_patterns(s, spans, cands, px.forms.title_key, [&](int pid, const Cand&, const Span&) { B.insert({k, g_pats[static_cast<std::size_t>(pid)].rel}); });
        }
    }
    say("\nS4 CHAPMAN per relation with a kept pattern (A text seeds, B kept patterns firing, W Wikidata; N_ref = |A u B u W|)");
    for (int r = 0; r < kNRel; ++r) {
        if (!(kept >> r & 1)) continue;
        double a = 0, bb = 0, m = 0, ab = 0, w = 0, mw = 0;
        std::set<string> all;
        for (auto& [k, rr] : A) if (rr == r) { ++a; all.insert(k); m += B.count({k, r}); }
        for (auto& [k, rr] : B) if (rr == r) { ++bb; all.insert(k); }
        ab = static_cast<double>(all.size());
        for (auto& [k, rr] : W) if (rr == r) { ++w; mw += all.count(k); all.insert(k); }
        const ecs::ChapmanResult c = ecs::chapman(a, bb, m), ct = ecs::chapman(ab, w, mw);
        const double nref = static_cast<double>(all.size());
        char b[400];
        std::snprintf(b, sizeof b, "  CHAPMAN %-13s A %.0f B %.0f m %.0f N_hat %.0f (sd %.0f) | N_ref %.0f ratio %.4f lower_bound %s | text %.0f vs Wikidata %.0f m %.0f: N_hat %.0f",
                      kRel[r].name, a, bb, m, c.estimate, std::sqrt(c.variance), nref, nref > 0 ? c.estimate / nref : 0.0, c.estimate <= nref ? "holds" : "fails", ab, w, mw, ct.estimate);
        say(b);
    }
    // The frontier: S3's cells, demand and the store's slots.
    std::map<std::pair<string, string>, vector<string>> cell;
    {
        std::ifstream f(G.path + ".cells.tsv");
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) { auto c = split(l, '\t'); if (c.size() >= 14) cell[{c[0], c[1]}] = c; }
    }
    std::map<std::pair<string, string>, long> demand;
    {
        std::ifstream f(G.path + ".demand.tsv");
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) { auto c = split(l, '\t'); if (c.size() >= 12) ++demand[{c[11], c[9]}]; }
    }
    const char* kT[4] = {"country", "person", "place", "other"};
    std::map<std::pair<string, int>, double> slotP;  // (subject, relation) -> best P
    for (const Seed& sd : g_seeds) if (sd.src < 3) { double& p = slotP[{norm_ent(sd.subj), sd.rel}]; p = std::max(p, 20.0 / 21.0); }
    for (const Fused& fu : fused) if (fu.op.P >= 0.5) { double& p = slotP[{norm_ent(fu.f->subj), fu.f->rel}]; p = std::max(p, fu.op.P); }
    double cP[4][64] = {}, cN[4][64] = {};
    for (auto& [k, p] : slotP) if (auto it = etype.find(k.first); it != etype.end()) cP[it->second][k.second] += p, ++cN[it->second][k.second];
    // Missing mass per cell over all pages (facts whose subject has the cell's type), and per relation the values.
    double mib = 0;
    for (const PageRef& pg : pages) mib += static_cast<double>(pg.te - pg.tb);
    mib /= 1048576.0;
    std::map<std::pair<int, int>, std::array<double, 5>> cellmm;  // (type, rel) -> obs, n1, q1_mass, incidences, q1
    {
        vector<std::uint32_t> cnt(facts.group.size(), 0), unit(facts.group.size(), 0);
        for (const auto& pg : facts.pages) for (auto [i, c] : pg) cnt[i] += c, ++unit[i];
        for (std::size_t i = 0; i < facts.group.size(); ++i) {
            if (facts.type[i] < 0 || !cnt[i]) continue;
            auto& a = cellmm[{facts.type[i], facts.group[i]}];
            a[0] += cnt[i];
            a[1] += cnt[i] == 1;
            a[2] += unit[i] == 1 ? cnt[i] : 0;
            a[3] += unit[i];
            a[4] += unit[i] == 1;
        }
    }
    vector<double> vseen(kNRel, 0), vchao(kNRel, 0);
    for (int r = 0; r < kNRel; ++r) {
        vector<std::uint32_t> cnt(vals.group.size(), 0), unit(vals.group.size(), 0);
        std::size_t T = 0;
        for (const auto& pg : vals.pages) {
            bool any = false;
            for (auto [i, c] : pg) if (vals.group[i] == r) cnt[i] += c, ++unit[i], any = true;
            T += any;
        }
        vector<std::size_t> cc, uu;
        for (std::size_t i = 0; i < cnt.size(); ++i) if (cnt[i]) cc.push_back(cnt[i]), uu.push_back(unit[i]);
        const ecs::ClumpedStats st = ecs::clumped_stats(cc, uu, T);
        vseen[r] = static_cast<double>(st.s_obs);
        vchao[r] = ecs::chao_extrapolated_richness(st.s_obs, st.q1, st.q2, st.units, 1e18);
    }
    struct Row { string type, rel; double ent, known, expected, cov, unseen, cert, dem, obs, per_mib, gt0, gtu, new10, newd10, vs, vc; int kept; };
    vector<Row> rows;
    for (int t = 0; t < 4; ++t)
        for (int r = 0; r < kNRel; ++r) {
            auto it = cell.find({kT[t], kRel[r].name});
            if (it == cell.end()) continue;
            const auto& c = it->second;
            Row w;
            w.type = kT[t], w.rel = kRel[r].name;
            w.ent = std::atof(c[2].c_str()), w.known = std::atof(c[3].c_str());
            // S3's P(exists) is the chance that an empty store slot has a value (calibrated on empty slots), so the slots
            // still unseen are the empty ones times it.
            w.unseen = std::max(0.0, w.ent - w.known) * std::atof(c[11].c_str());
            w.expected = w.known + w.unseen;
            w.cov = w.expected > 0 ? w.known / w.expected : 1.0;
            w.cert = cN[t][r] ? cP[t][r] / cN[t][r] : 0.0;
            w.dem = static_cast<double>(demand[{kT[t], kRel[r].name}]);
            const auto mm = cellmm.count({t, r}) ? cellmm[{t, r}] : std::array<double, 5>{0, 0, 0, 0, 0};
            w.obs = mm[0], w.per_mib = mm[0] / mib;
            w.gt0 = mm[0] > 0 ? mm[1] / mm[0] : 1.0, w.gtu = mm[0] > 0 ? mm[2] / mm[0] : 1.0;
            w.new10 = w.gtu * w.per_mib * 10.0;
            w.newd10 = mm[4] / mib * 10.0;  // Q1/U x incidences per MiB: distinct new facts, not observations
            w.vs = vseen[r], w.vc = vchao[r];
            w.kept = std::atoi(c[12].c_str());
            rows.push_back(w);
        }
    std::ofstream out(F.path);
    out << "type\trelation\tentities\tknown\texpected\tcoverage\tunseen_in_slice\tcertainty\tdemand\tfact_obs\tobs_per_mib\tgt0\tgtu\tnew_facts_per_10mib\t"
           "new_distinct_per_10mib\tvalues_seen\tvalues_chao1\tkept_pattern\n";
    char b[600];
    for (const Row& w : rows) {
        std::snprintf(b, sizeof b, "%s\t%s\t%.0f\t%.0f\t%.1f\t%.4f\t%.1f\t%.4f\t%.0f\t%.0f\t%.3f\t%.4f\t%.4f\t%.2f\t%.2f\t%.0f\t%.0f\t%d\n", w.type.c_str(), w.rel.c_str(), w.ent,
                      w.known, w.expected, w.cov, w.unseen, w.cert, w.dem, w.obs, w.per_mib, w.gt0, w.gtu, w.new10, w.newd10, w.vs, w.vc, w.kept);
        out << b;
    }
    std::snprintf(b, sizeof b, "\nS4 FRONTIER %s: %zu cells, article text %.1f MiB", F.path.c_str(), rows.size(), mib);
    say(b);
    for (int pass = 0; pass < 2; ++pass) {
        vector<const Row*> v;
        for (const Row& w : rows) v.push_back(&w);
        std::sort(v.begin(), v.end(), [&](const Row* x, const Row* y) {
            const double a = pass ? x->unseen : x->new10, c = pass ? y->unseen : y->new10;
            return a != c ? a > c : x->type + x->rel < y->type + y->rel;
        });
        for (std::size_t k = 0; k < 10 && k < v.size(); ++k) {
            const Row& w = *v[k];
            std::snprintf(b, sizeof b, "  %s %2zu %-7s %-13s new/10MiB %.1f (distinct %.1f) | known %.0f expected %.1f coverage %.3f unseen %.1f | certainty %.3f demand %.0f | values %.0f chao1 %.0f",
                          pass ? "FRONTIER_TOP_SLICE" : "FRONTIER_TOP_READ", k + 1, w.type.c_str(), w.rel.c_str(), w.new10, w.newd10, w.known, w.expected, w.cov, w.unseen, w.cert, w.dem,
                          w.vs, w.vc);
            say(b);
        }
    }
}

// ============================================================ S4b: unknown-unknown signals (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b")
//
// --signals DIR [--signals-dolly PAGES.xml] [--surprise SCORES.tsv,...] [--signals-gaps FILE], after the normal passes
// (it needs the kept patterns). DISCOVERY §3.4's signals under unknown_signals.hpp's alarm rule, on the registered
// streams (C, S) and planted pools (D: Dolly pages from scripts/s4b_dolly_pages.py; L: the "List of" and disambiguation
// pages the store's reader skips):
//   U (unparsed, a coverage gap): per page, n sentences with >= 2 entity mentions (links or the page's subject), x of them
//     parsed by a kept pattern; the score -ln P(X <= x | n, q_type), q_type the parse rate of the page's type over C;
//   V (vocabulary, a lexical gap): CyphaLM's view (S4); ln((v + 0.5) / (e + 0.5)), v the page's words new to the reader,
//     e Heaps' expectation for its tokens;
//   Σ (surprise, a prediction gap): cyphalm_trie surprise's bits per byte (--surprise).
// Writes DIR/units.tsv and the SIGNAL lines; each alarm of the unshifted stream (running reference) opens a gap in S3's
// form in --signals-gaps FILE, and the first detection of each planted pool in FILE.planted.tsv. Rule 10: Cypha's
// self-assessment only; nothing here answers a question or changes a fact, and S3's files are not touched.
// S4c adds --signals-scope dev|test [--signals-plain PAGES.xml] [--signals-vec FILE]: the out-of-scope signal (O1 register,
// O2 rare content words, O3 topic distance) on the first 50 prose words of a page, with pool P (wiki pages with their
// markup removed, scripts/s4c_plain_pages.py) as the control. Without --signals-scope the output is S4b's.
struct SignalOpt {
    string dir, dolly, gaps;
    vector<string> surprise;
    string scope, plain, vec = "data/cyphalm/simvec_ppmi32.bin";  // S4c: --signals-scope dev|test, --signals-plain, --signals-vec
};

namespace s4b {

struct Page {
    const string* src = nullptr;
    PageRef ref;
    int type = 3;  // write_gaps' rule: country 0, person 1, place 2, other 3
};
enum { kC = 0, kS = 1, kD = 2, kL = 3, kP = 4 };  // kP: S4c's control, wiki pages with their markup removed
const char* const kSetName[5] = {"C", "S", "D", "L", "P"};

struct Signal {
    string name, gap_kind;
    vector<string> strata;
    vector<ecs::SignalUnit> u[5];
    vector<const Page*> pg[5];
    vector<string> detail[5];
    vector<int> pools{kD, kL};  // the planted pools analysed
    string tag = "S4b";         // in the gap notes
    string header{};            // S4b-b: a line printed before the signal's analysis (empty: none)
};

// U: (eligible sentences, parsed ones) of a page. skip_dates (S4b-b): a link inside a date candidate is not a mention.
std::pair<int, int> unparsed_counts(const Page& p, bool skip_dates) {
    PageCtx px;
    load_page(*p.src, p.ref, px);
    int n = 0, x = 0;
    vector<Cand> cands, dates;
    for (const Sent& s : px.sents) {
        const auto spans = find_subject(s, px.forms);
        vector<std::pair<int, int>> ds;
        if (skip_dates) dates.clear(), find_dates(s, dates), ds = date_spans(dates);
        std::set<string> ents;
        for (const Lnk& L : s.l)
            if (string k = norm_ent(L.target); !k.empty() && k != px.forms.title_key && !(skip_dates && ecs::span_within(L.tb, L.te, ds))) ents.insert(k);
        if (ents.size() + (spans.empty() ? 0 : 1) < 2) continue;
        ++n;
        if (spans.empty()) continue;
        gen_cands(s, cands);
        bool fired = false;
        apply_patterns(s, spans, cands, px.forms.title_key, [&](int, const Cand&, const Span&) { fired = true; });
        x += fired;
    }
    return {n, x};
}

// U's signal: q_τ from the calibration set (sets[kC]), units from every set. own: each set's name in a unit's identity.
// skip_dates as unparsed_counts. Prints the rates over the calibration set and each set's totals (S4b's lines).
Signal unparsed_signal(const vector<const Page*> (&sets)[4], const char* const (&own)[4], bool skip_dates, const std::function<void(const string&)>& say) {
    Signal g{"unparsed", "coverage", {"country", "person", "place", "other"}, {}, {}, {}};
    char b[600];
    vector<std::pair<int, int>> cnt[4];
    double sn[4] = {0, 0, 0, 0}, sx[4] = {0, 0, 0, 0};
    std::size_t pages_t[4] = {0, 0, 0, 0}, with[4] = {0, 0, 0, 0};
    for (int s = 0; s < 4; ++s)
        for (const Page* p : sets[s]) {
            cnt[s].push_back(unparsed_counts(*p, skip_dates));
            if (s == kC) sn[p->type] += cnt[s].back().first, sx[p->type] += cnt[s].back().second, ++pages_t[p->type], with[p->type] += cnt[s].back().first >= 3;
        }
    double q[4];
    for (int t = 0; t < 4; ++t) {
        q[t] = sn[t] > 0 ? sx[t] / sn[t] : 0.0;
        std::snprintf(b, sizeof b, "  SIGNAL_U_RATE C %-7s pages %zu units (n >= 3) %zu | eligible sentences %.0f parsed %.0f | unparsed rate %.4f", g.strata[static_cast<std::size_t>(t)].c_str(),
                      pages_t[t], with[t], sn[t], sx[t], sn[t] > 0 ? 1.0 - q[t] : 0.0);
        say(b);
    }
    for (int s = 0; s < 4; ++s) {
        double n_all = 0, x_all = 0;
        for (std::size_t i = 0; i < sets[s].size(); ++i) {
            const auto [n, x] = cnt[s][i];
            n_all += n, x_all += x;
            if (n < 3) continue;
            const Page* p = sets[s][i];
            g.u[s].push_back({ecs::binom_lower_tail_score(static_cast<std::size_t>(n), static_cast<std::size_t>(x), q[p->type]), p->type,
                              ecs::signal_unit_id(own[s], p->ref.tb)});
            g.pg[s].push_back(p);
            g.detail[s].push_back("n " + std::to_string(n) + " parsed " + std::to_string(x));
        }
        std::snprintf(b, sizeof b, "  SIGNAL_U_SET %s pages %zu units %zu | eligible sentences %.0f parsed %.0f unparsed rate %.4f", kSetName[s], sets[s].size(), g.u[s].size(), n_all,
                      x_all, n_all > 0 ? 1.0 - x_all / n_all : 0.0);
        say(b);
    }
    return g;
}

// V: reads a page's words (S4's: maximal runs of ASCII letters, lower-cased) into the vocabulary; (tokens, new words).
std::pair<double, double> read_words(const Page& p, std::unordered_set<string>& vocab) {
    const string& d = *p.src;
    double tok = 0, fresh = 0;
    string w;
    auto letter = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    for (std::size_t i = p.ref.tb; i < p.ref.te;) {
        if (!letter(static_cast<unsigned char>(d[i]))) { ++i; continue; }
        w.clear();
        while (i < p.ref.te && letter(static_cast<unsigned char>(d[i]))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(d[i++])));
        ++tok;
        fresh += vocab.insert(w).second ? 1 : 0;
    }
    return {tok, fresh};
}

void gap_row(std::ostream& o, const Signal& g, const Page& p, int stratum, double share, const string& note) {
    char b[160];
    std::snprintf(b, sizeof b, "\t-\t-1.0000\t-1.0000\t-1.0000\t%.4f\t0.0000\t0\t0.0000\t", share);
    o << ecs::gap_id(p.ref.title, "-") << '\t' << g.gap_kind << "\topen\t" << p.ref.title << '\t' << g.strata[static_cast<std::size_t>(stratum)] << b << note << '\n';
}

void analyse(const Signal& g, std::ofstream& gaps, std::ofstream& planted, const std::function<void(const string&)>& say) {
    const ecs::SignalRule rule;
    const int K = static_cast<int>(g.strata.size());
    char b[700];
    string per;
    for (int k = 0; k < K; ++k) {
        std::size_t c = 0, s = 0;
        for (const auto& u : g.u[kC]) c += u.stratum == k;
        for (const auto& u : g.u[kS]) s += u.stratum == k;
        per += " " + g.strata[static_cast<std::size_t>(k)] + " " + std::to_string(c) + "/" + std::to_string(s);
    }
    std::snprintf(b, sizeof b, "\nSIGNAL %s units C %zu S %zu D %zu L %zu%s | strata (C/S):%s", g.name.c_str(), g.u[kC].size(), g.u[kS].size(), g.u[kD].size(),
                  g.u[kL].size(), std::find(g.pools.begin(), g.pools.end(), kP) != g.pools.end() ? (" P " + std::to_string(g.u[kP].size())).c_str() : "", per.c_str());
    say(b);
    if (g.u[kS].size() < 300 || g.u[kC].size() < 100) { say("  too few units for the rule"); return; }
    const ecs::SignalRun run[2] = {ecs::signal_run(g.u[kC], g.u[kS], K, true, rule), ecs::signal_run(g.u[kC], g.u[kS], K, false, rule)};
    const char* ref_name[2] = {"running", "fixed"};
    for (int r = 0; r < 2; ++r) {
        const ecs::SignalRun& R = run[r];
        std::size_t ca = 0, sa = 0;
        for (const auto& a : R.alarms) (a.in_test ? sa : ca) += 1;
        const double perm = r == 1 ? ecs::signal_perm_alarms_per_k(R, rule, 200, 20261029) : -1.0;
        std::snprintf(b, sizeof b, "  SIGNAL_FA %s %-7s | C scored %zu events %.4f alarms %zu | S scored %zu events %zu rate %.4f alarms %zu (per 1,000 %.3f)%s",
                      g.name.c_str(), ref_name[r], R.c_scored, R.c_scored ? static_cast<double>(R.c_events) / static_cast<double>(R.c_scored) : 0.0, ca, R.s_scored,
                      R.s_events, R.s_scored ? static_cast<double>(R.s_events) / static_cast<double>(R.s_scored) : 0.0, sa,
                      R.s_scored ? 1000.0 * static_cast<double>(sa) / static_cast<double>(R.s_scored) : 0.0,
                      perm >= 0 ? (" | 200 random orders: " + std::to_string(perm).substr(0, 6) + " alarms per 1,000").c_str() : "");
        say(b);
        for (const auto& a : R.alarms) {
            const Page& p = *g.pg[a.in_test ? kS : kC][a.pos];
            std::snprintf(b, sizeof b, "    ALARM %s %s %s unit %zu %s | R %.0f events %zu/%zu since the last restart", g.name.c_str(), ref_name[r], a.in_test ? "S" : "C", a.pos,
                          p.ref.title.c_str(), a.R, a.events, a.units);
            say(b);
            if (r == 0)
                gap_row(gaps, g, p, g.u[a.in_test ? kS : kC][a.pos].stratum, static_cast<double>(a.events) / static_cast<double>(a.units),
                        g.tag + " " + g.name + " (running reference): alarm at " + (a.in_test ? "S" : "C") + " unit " + std::to_string(a.pos) + ", R " +
                            std::to_string(static_cast<long long>(a.R)) + ", events " + std::to_string(a.events) + " of the last " + std::to_string(a.units) + " units");
        }
    }
    // The planted pools' events against the fixed reference (C): the event rate a shift brings.
    for (int set : g.pools) {
        ecs::SignalRef ref(K, false, rule);
        ref.fill(g.u[kC]);
        std::size_t n = 0, e = 0;
        for (const auto& u : g.u[set])
            if (ref.ready(u.stratum)) ++n, e += ref.p(u) < rule.alpha;
        std::snprintf(b, sizeof b, "  SIGNAL_POOL %s %s units %zu scored %zu events %zu rate %.4f (fixed reference)", g.name.c_str(), kSetName[set], g.u[set].size(), n, e,
                      n ? static_cast<double>(e) / static_cast<double>(n) : 0.0);
        say(b);
    }
    // Planted shifts: 200 replicates per (pool, pi, reference); t0 uniform over S's units [100, |S| - 200].
    const std::size_t W = 200, reps = 200, nS = g.u[kS].size();
    for (int set : g.pools) {
        if (g.u[set].empty()) continue;
        bool demo = false;
        for (double pi : {1.0, 0.5, 0.2})
            for (int r = 0; r < 2; ++r) {
                std::mt19937_64 rng(20261029ull ^ ecs::signal_unit_id(g.name + kSetName[set] + ref_name[r], static_cast<std::size_t>(pi * 10)));
                std::uniform_int_distribution<std::size_t> T(100, nS - W);
                vector<int> delays;
                std::size_t plant_alarms = 0;
                for (std::size_t rep = 0; rep < reps; ++rep) {
                    const std::size_t t0 = T(rng);
                    const ecs::SignalDetection d = ecs::signal_planted(g.u[kC], g.u[kS], g.u[set], K, r == 0, rule, run[r], t0, pi, W, rng);
                    delays.push_back(d.delay);
                    plant_alarms += d.delay > 0 && d.plant;
                    if (d.delay > 0 && !demo && r == 0 && pi == 1.0) {
                        demo = true;
                        const Page& p = *(d.plant ? g.pg[set] : g.pg[kS])[d.index];
                        gap_row(planted, g, p, (d.plant ? g.u[set] : g.u[kS])[d.index].stratum, static_cast<double>(d.events) / static_cast<double>(d.units),
                                g.tag + " " + g.name + " (running reference), planted pool " + kSetName[set] + " at S unit " + std::to_string(t0) + ": alarm after " +
                                    std::to_string(d.delay) + " units, R " + std::to_string(static_cast<long long>(d.R)) + ", events " + std::to_string(d.events) +
                                    " of the last " + std::to_string(d.units) + " units");
                    }
                }
                const ecs::SignalDelayStats st = ecs::signal_delay_stats(delays, W);
                std::snprintf(b, sizeof b, "  SIGNAL_DETECT %s pool %s pi %.1f %-7s | within 50 %3zu/%zu within 200 %3zu/%zu | delay median %s p90 %s | alarm on a plant %zu",
                              g.name.c_str(), kSetName[set], pi, ref_name[r], st.within50, st.reps, st.within_window, st.reps,
                              st.median > W ? ">200" : std::to_string(static_cast<int>(st.median)).c_str(), st.p90 > W ? ">200" : std::to_string(static_cast<int>(st.p90)).c_str(),
                              plant_alarms);
                say(b);
            }
    }
}

// S4c: the similar-context expert's word vectors (data/cyphalm/README.md: "CYSIMV1\0", rows and dims (uint32), each row's
// word (a length byte, then its bytes), the rows' scales (float32), the rows (int8), FNV-1a 64 of every byte before it).
struct SimVec {
    std::unordered_map<string, std::size_t> row;
    std::size_t dim = 0;
    vector<float> v;  // rows x dim, dequantised
    const float* of(const string& w) const {
        auto it = row.find(w);
        return it == row.end() ? nullptr : v.data() + it->second * dim;
    }
};
bool load_simvec(const string& path, SimVec& sv, string& err) {
    string f;
    if (!read_file(path, f)) { err = "cannot read " + path; return false; }
    if (f.size() < 24 || f.compare(0, 8, string("CYSIMV1\0", 8)) != 0) { err = "not a CYSIMV1 file"; return false; }
    std::uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a 64's offset basis
    for (std::size_t i = 0; i + 8 < f.size(); ++i) h = (h ^ static_cast<unsigned char>(f[i])) * 1099511628211ull;
    std::uint64_t stored = 0;
    std::memcpy(&stored, f.data() + f.size() - 8, 8);
    if (h != stored) { err = "hash mismatch"; return false; }
    std::size_t pos = 8;
    std::uint32_t rows = 0, dims = 0;
    std::memcpy(&rows, f.data() + pos, 4), std::memcpy(&dims, f.data() + pos + 4, 4);
    pos += 8;
    vector<string> words(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t n = static_cast<unsigned char>(f[pos++]);
        words[r] = f.substr(pos, n), pos += n;
    }
    vector<float> scale(rows);
    std::memcpy(scale.data(), f.data() + pos, 4ull * rows), pos += 4ull * rows;
    sv.dim = dims;
    sv.v.resize(static_cast<std::size_t>(rows) * dims);
    for (std::size_t i = 0; i < sv.v.size(); ++i) sv.v[i] = scale[i / dims] * static_cast<float>(static_cast<signed char>(f[pos + i]));
    for (std::uint32_t r = 0; r < rows; ++r) sv.row.emplace(words[r], r);
    return true;
}

}  // namespace s4b

void write_signals(const SignalOpt& O, const string& data, const vector<PageRef>& pages, const std::function<void(const string&)>& say) {
    using namespace s4b;
    string dolly;
    if (!O.dolly.empty() && !read_file(O.dolly, dolly)) { say("S4b: cannot read " + O.dolly); return; }
    string plain;  // S4c's pool P
    if (!O.scope.empty() && !O.plain.empty() && !read_file(O.plain, plain)) { say("S4c: cannot read " + O.plain); return; }
    auto type_of = [](const string& title) {
        const string nk = norm_ent(title);
        return g_country.count(nk) ? 0 : g_person.count(nk) ? 1 : g_place.count(nk) ? 2 : 3;
    };
    std::deque<Page> store;  // stable addresses
    vector<const Page*> set[4], reader;
    for (const PageRef& pg : pages) {
        store.push_back({&data, pg, type_of(pg.title)});
        const Page* p = &store.back();
        const bool skip = ecs::signal_store_skips(pg.title);
        if (pg.tb < ecs::kSignalReaderTo) reader.push_back(p);
        if (skip && pg.tb >= ecs::kSignalCalibFrom) set[kL].push_back(p);
        else if (!skip && pg.tb >= ecs::kSignalCalibFrom && pg.tb < ecs::kSignalSplit) set[kC].push_back(p);
        else if (!skip && pg.tb >= ecs::kSignalSplit && pg.tb < ecs::kSignalTestTo) set[kS].push_back(p);
    }
    if (!dolly.empty()) {
        long tot = 0, red = 0;
        for (const PageRef& pg : list_pages(dolly, tot, red)) {
            store.push_back({&dolly, pg, type_of(pg.title)});
            set[kD].push_back(&store.back());
        }
    }
    char b[600];
    std::snprintf(b, sizeof b, "\nS4b SIGNALS pages: reader (before %zu) %zu, C %zu, S %zu, D %zu, L %zu", ecs::kSignalReaderTo, reader.size(), set[kC].size(), set[kS].size(),
                  set[kD].size(), set[kL].size());
    say(b);
    std::ofstream units(O.dir + "/units.tsv");
    units << "signal\tset\tbody_offset\ttitle\tstratum\tscore\tdetail\n";
    vector<Signal> sig;

    // S4b-b (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b-b"): --signals-scope u-dev builds U twice, keep (S4b's reader)
    // and skip (links inside date candidates are not mentions), on S4c's dev split: C's pages before kSignalDevSplit
    // calibrate, the rest of C is the stream, D's even records and L are the pools. S is never scored; V, Σ and S4c do not run.
    const bool udev = O.scope == "u-dev";
    if (udev) {
        vector<const Page*> ds[4];
        for (const Page* p : set[kC]) ds[p->ref.tb >= ecs::kSignalDevSplit ? kS : kC].push_back(p);
        for (const Page* p : set[kD]) {  // "Dolly NNNNN": the record's line number
            const long line = std::atol(p->ref.title.c_str() + std::min<std::size_t>(6, p->ref.title.size()));
            if (line % 2 == 0) ds[kD].push_back(p);
        }
        ds[kL] = set[kL];
        static const char* const own[4] = {"C", "C", "D", "L"};
        for (const bool skip : {false, true}) {
            const char* vn = skip ? "skip" : "keep";
            // The unparsed rate over all of C (S4b's calibration stream), by page type.
            double sn[4] = {0, 0, 0, 0}, sx[4] = {0, 0, 0, 0};
            for (const Page* p : set[kC]) {
                const auto [n, x] = unparsed_counts(*p, skip);
                sn[p->type] += n, sx[p->type] += x;
            }
            const char* tn[4] = {"country", "person", "place", "other"};
            for (int t = 0; t < 4; ++t) {
                std::snprintf(b, sizeof b, "S4b-b U_RATE_ALL_C %s %-7s | eligible sentences %.0f parsed %.0f | unparsed rate %.4f", vn, tn[t], sn[t], sx[t],
                              sn[t] > 0 ? 1.0 - sx[t] / sn[t] : 0.0);
                say(b);
            }
            say(string("S4b-b dev split, U ") + vn + ":");
            Signal g = unparsed_signal(ds, own, skip, say);
            for (auto& d : g.detail)
                for (string& x : d) x = string(vn) + " " + x;
            g.tag = string("S4b-b dev ") + vn;
            g.header = string("\nS4b-b U ") + vn + " (dev split: stream = C from " + std::to_string(ecs::kSignalDevSplit) + ")";
            sig.push_back(std::move(g));
        }
    } else {
        // U: the unparsed rate per page type.
        static const char* const own[4] = {"C", "S", "D", "L"};
        sig.push_back(unparsed_signal(set, own, kUSkipDateLinks, say));
    }
    // V: new words against Heaps, CyphaLM's view.
    if (!udev) {
        Signal g{"vocabulary", "lexical", {"tok50-299", "tok300-1499", "tok1500+"}, {}, {}, {}};
        std::unordered_set<string> vocab;
        vector<double> Np, Vp;
        double N = 0;
        for (std::size_t k = 0; k < reader.size(); ++k) {
            N += read_words(*reader[k], vocab).first;
            if ((k + 1) % 25 == 0) Np.push_back(N), Vp.push_back(static_cast<double>(vocab.size()));
        }
        Np.push_back(N), Vp.push_back(static_cast<double>(vocab.size()));
        const ecs::HeapsFit f = ecs::fit_heaps(Np, Vp);
        std::snprintf(b, sizeof b, "  SIGNAL_V_HEAPS reader pages %zu tokens %.0f vocabulary %zu | K %.3f beta %.4f r2 %.5f points %zu", reader.size(), N, vocab.size(), f.K, f.beta, f.r2, f.n);
        say(b);
        auto score = [&](int s, std::unordered_set<string>& V, double& n_read) {
            double sv = 0, se = 0;
            for (const Page* p : set[s]) {
                const auto [n, v] = read_words(*p, V);
                const double e = ecs::heaps_vocab(f, n_read + n) - ecs::heaps_vocab(f, n_read);
                n_read += n;
                sv += v, se += e;
                if (n < 50) continue;
                g.u[s].push_back({std::log((v + 0.5) / (e + 0.5)), n < 300 ? 0 : n < 1500 ? 1 : 2, ecs::signal_unit_id(kSetName[s], p->ref.tb)});
                g.pg[s].push_back(p);
                char d[96];
                std::snprintf(d, sizeof d, "tokens %.0f new %.0f expected %.2f", n, v, e);
                g.detail[s].push_back(d);
            }
            std::snprintf(b, sizeof b, "  SIGNAL_V_SET %s pages %zu units %zu | new words %.0f expected %.1f ratio %.4f | tokens read after %.0f", kSetName[s], set[s].size(), g.u[s].size(),
                          sv, se, se > 0 ? sv / se : 0.0, n_read);
            say(b);
        };
        score(kC, vocab, N);
        const std::unordered_set<string> at_s = vocab;
        const double n_at_s = N;
        score(kS, vocab, N);
        for (int s : {kD, kL}) {
            std::unordered_set<string> V = at_s;
            double n = n_at_s;
            score(s, V, n);
        }
        sig.push_back(std::move(g));
    }
    // Σ: CyphaLM's surprise, from cyphalm_trie surprise.
    if (!O.surprise.empty() && !udev) {
        Signal g{"surprise", "prediction", {"bytes256-1023", "bytes1024-2047", "bytes2048"}, {}, {}, {}};
        std::map<std::pair<string, std::size_t>, std::pair<std::size_t, double>> sc;
        for (const string& path : O.surprise) {
            std::ifstream f(path);
            string line;
            std::getline(f, line);
            while (std::getline(f, line)) {
                const auto c = split(line, '\t');
                if (c.size() >= 7) sc[{c[0], std::strtoull(c[1].c_str(), nullptr, 10)}] = {std::strtoull(c[4].c_str(), nullptr, 10), std::atof(c[6].c_str())};
            }
        }
        for (int s = 0; s < 4; ++s) {
            std::size_t want = 0;
            for (const Page* p : set[s]) {
                want += p->ref.te - p->ref.tb >= 256;
                auto it = sc.find({kSetName[s], p->ref.tb});
                if (it == sc.end()) continue;
                const auto [n, bpb] = it->second;
                g.u[s].push_back({bpb, n < 1024 ? 0 : n < 2048 ? 1 : 2, ecs::signal_unit_id(kSetName[s], p->ref.tb)});
                g.pg[s].push_back(p);
                g.detail[s].push_back("scored_bytes " + std::to_string(n));
            }
            double m = 0;
            for (const auto& u : g.u[s]) m += u.score;
            std::snprintf(b, sizeof b, "  SIGNAL_S_SET %s pages with >= 256 body bytes %zu, scored %zu | mean bits per byte %.4f", kSetName[s], want, g.u[s].size(),
                          g.u[s].empty() ? 0.0 : m / static_cast<double>(g.u[s].size()));
            say(b);
        }
        sig.push_back(std::move(g));
    }
    // S4c: the out-of-scope signal, read from the first ecs::kOodWindow prose words of a page (docs/reports/S4_UNSEEN_AND_FRONTIER.md,
    // "S4c"). --signals-scope dev: C's first half calibrates, its second half is the stream (S is never scored); test: C
    // then S, as S4b. Pools: D's even (dev) or odd (test) Dolly records, P's even or odd pages, and L.
    if (O.scope == "dev" || O.scope == "test") {
        const bool dev = O.scope == "dev";
        // The prose words of a page: S3's prose reader, then S4c's list rule; tokens starting with an ASCII letter.
        auto words_of = [&](const Page& p, std::size_t limit) {
            vector<Sent> ss;
            page_sentences(*p.src, p.ref, ss);
            vector<string> w;
            for (const Sent& st : ss) {
                if (ecs::ood_list_line(*p.src, st.off)) continue;
                for (const Tok& t : st.t)
                    if (!t.w.empty() && t.w[0] >= 'a' && t.w[0] <= 'z') {
                        w.push_back(t.w);
                        if (w.size() >= limit) return w;
                    }
            }
            return w;
        };
        ecs::OodRef ref;
        vector<const Page*> rd;
        for (const Page* p : reader)
            if (!ecs::signal_store_skips(p->ref.title)) rd.push_back(p);
        vector<vector<string>> rd_words;
        for (const Page* p : rd) {
            rd_words.push_back(words_of(*p, static_cast<std::size_t>(-1)));
            for (const string& w : rd_words.back()) ref.add(w);
        }
        ref.finish();
        ecs::OodProfile prof;  // O4 (round 2)
        prof.build(ref, rd_words);
        ecs::OodTfidf tfidf;  // O5 (round 3): the reader pages' TF-IDF vectors
        tfidf.build(ref, rd_words);
        SimVec sv;
        string err;
        if (!load_simvec(O.vec, sv, err)) { say("S4c: vectors: " + err); return; }
        auto vec_of = [&](const string& w) { return sv.of(w); };
        vector<vector<double>> bank;
        for (const Page* p : rd)
            if (const auto w = words_of(*p, ecs::kOodWindow); w.size() >= ecs::kOodWindow)
                if (auto v = ecs::ood_page_vector(w, ref, sv.dim, vec_of); !v.empty()) bank.push_back(std::move(v));
        string fw;
        for (std::size_t i = 0; i < ref.top.size() && i < 12; ++i) fw += " " + ref.top[i];
        std::snprintf(b, sizeof b, "\nS4c SCOPE %s | reference: reader pages %zu, words %llu, distinct %zu, F %zu (share %.4f):%s ... | vectors %zu x %zu, bank %zu | O4 profile %zu words, %zu windows",
                      O.scope.c_str(), rd.size(), static_cast<unsigned long long>(ref.N), ref.count.size(), ref.top.size(), 1.0 - ref.share.back(), fw.c_str(),
                      sv.row.size(), sv.dim, bank.size(), prof.bin.size(), prof.windows);
        say(b);
        // The sets: calibration, stream, D, L and P.
        vector<const Page*> P;
        if (!plain.empty()) {
            long tot = 0, red = 0;
            for (const PageRef& pg : list_pages(plain, tot, red)) {
                store.push_back({&plain, pg, type_of(pg.title)});
                P.push_back(&store.back());
            }
        }
        vector<const Page*> os[5];
        for (const Page* p : set[kC]) os[dev && p->ref.tb >= ecs::kSignalDevSplit ? kS : kC].push_back(p);
        if (!dev) os[kS] = set[kS];
        for (const Page* p : set[kD]) {  // "Dolly NNNNN": the record's line number
            const long line = std::atol(p->ref.title.c_str() + std::min<std::size_t>(6, p->ref.title.size()));
            if ((line % 2 == 0) == dev) os[kD].push_back(p);
        }
        os[kL] = set[kL];
        for (std::size_t i = 0; i < P.size(); ++i)
            if ((i % 2 == 0) == dev) os[kP].push_back(P[i]);
        const char* own[5] = {"C", dev ? "C" : "S", "D", "L", "P"};  // a unit's identity: the set it comes from
        constexpr int NO = 5;
        Signal G[NO] = {{"O1-register", "scope", {"country", "person", "place", "other"}, {}, {}, {}},
                        {"O2-rare", "scope", {"country", "person", "place", "other"}, {}, {}, {}},
                        {"O3-topic", "scope", {"country", "person", "place", "other"}, {}, {}, {}},
                        {"O4-delta", "scope", {"country", "person", "place", "other"}, {}, {}, {}},
                        {"O5-novelty", "scope", {"country", "person", "place", "other"}, {}, {}, {}}};
        auto scores = [&](const vector<string>& w) {
            return std::array<double, NO>{ecs::ood_register(w, ref), ecs::ood_rare(w, ref), ecs::ood_topic_distance(ecs::ood_page_vector(w, ref, sv.dim, vec_of), bank),
                                          ecs::ood_delta(w, prof), tfidf.novelty(w, ref)};
        };
        for (int s = 0; s < 5; ++s) {
            double sum[NO] = {0, 0, 0, 0, 0};
            std::size_t units_n = 0;
            for (const Page* p : os[s]) {
                const auto w = words_of(*p, ecs::kOodWindow);
                if (w.size() < ecs::kOodWindow) continue;
                const auto sc = scores(w);
                ++units_n;
                for (int k = 0; k < NO; ++k) {
                    G[k].u[s].push_back({sc[static_cast<std::size_t>(k)], p->type, ecs::signal_unit_id(own[s], p->ref.tb)});
                    G[k].pg[s].push_back(p);
                    G[k].detail[s].push_back(dev ? "dev" : "test");
                    sum[k] += sc[static_cast<std::size_t>(k)];
                }
            }
            std::snprintf(b, sizeof b, "  SIGNAL_O_SET %s %s pages %zu units %zu (share %.3f) | mean O1 %.4f O2 %.4f O3 %.4f O4 %.4f O5 %.4f", O.scope.c_str(), kSetName[s], os[s].size(),
                          units_n, os[s].empty() ? 0.0 : static_cast<double>(units_n) / static_cast<double>(os[s].size()), units_n ? sum[0] / static_cast<double>(units_n) : 0.0,
                          units_n ? sum[1] / static_cast<double>(units_n) : 0.0, units_n ? sum[2] / static_cast<double>(units_n) : 0.0,
                          units_n ? sum[3] / static_cast<double>(units_n) : 0.0, units_n ? sum[4] / static_cast<double>(units_n) : 0.0);
            say(b);
        }
        // The control's first check: each P page against its source page in the dump.
        {
            std::unordered_map<string, const PageRef*> src;
            for (const PageRef& pg : pages)
                if (pg.tb >= ecs::kSignalPlainFrom) src.emplace(pg.title, &pg);
            std::size_t n = 0, same_w = 0, same[NO] = {0, 0, 0, 0, 0}, units_both = 0;
            double dabs[NO] = {0, 0, 0, 0, 0};
            for (const Page* p : os[kP]) {
                auto it = src.find(p->ref.title);
                if (it == src.end()) continue;
                ++n;
                const Page sp{&data, *it->second, p->type};
                const auto wp = words_of(*p, ecs::kOodWindow), ws = words_of(sp, ecs::kOodWindow);
                if (wp.size() < ecs::kOodWindow || ws.size() < ecs::kOodWindow) continue;
                ++units_both;
                same_w += wp == ws;
                const auto a = scores(wp), c = scores(ws);
                for (int k = 0; k < NO; ++k) {
                    const double d = std::fabs(a[static_cast<std::size_t>(k)] - c[static_cast<std::size_t>(k)]);
                    same[k] += d <= 1e-12, dabs[k] += d;
                }
            }
            std::snprintf(b, sizeof b, "  SIGNAL_O_SAME %s P pages with a source %zu, both units %zu | same window %zu | same score O1 %zu O2 %zu O3 %zu O4 %zu O5 %zu | mean |diff| O1 %.4f O2 %.4f O3 %.4f O4 %.4f O5 %.4f",
                          O.scope.c_str(), n, units_both, same_w, same[0], same[1], same[2], same[3], same[4], units_both ? dabs[0] / static_cast<double>(units_both) : 0.0,
                          units_both ? dabs[1] / static_cast<double>(units_both) : 0.0, units_both ? dabs[2] / static_cast<double>(units_both) : 0.0,
                          units_both ? dabs[3] / static_cast<double>(units_both) : 0.0, units_both ? dabs[4] / static_cast<double>(units_both) : 0.0);
            say(b);
        }
        for (Signal& g : G) {
            g.pools = {kD, kP, kL};
            g.tag = dev ? "S4c dev" : "S4c test";
            sig.push_back(std::move(g));
        }
    }
    for (const Signal& g : sig)
        for (int s = 0; s < 5; ++s)
            for (std::size_t i = 0; i < g.u[s].size(); ++i) {
                char v[32];
                std::snprintf(v, sizeof v, "%.6f", g.u[s][i].score);
                units << g.name << '\t' << kSetName[s] << '\t' << g.pg[s][i]->ref.tb << '\t' << g.pg[s][i]->ref.title << '\t'
                      << g.strata[static_cast<std::size_t>(g.u[s][i].stratum)] << '\t' << v << '\t' << g.detail[s][i] << '\n';
            }
    const string gap_path = O.gaps.empty() ? O.dir + "/signal_gaps.tsv" : O.gaps;
    std::ofstream gaps(gap_path), planted(gap_path + ".planted.tsv");
    for (std::ofstream* o : {&gaps, &planted}) *o << "gap_id\tkind\tstatus\tentity\ttype\trelation\tp_exists\tf_store\tf_wd\tp\teig_bits\tdemand\tvoi\tnote\n";
    for (const Signal& g : sig) {
        if (!g.header.empty()) say(g.header);
        analyse(g, gaps, planted, say);
    }
    say("S4b: units " + O.dir + "/units.tsv, gaps " + gap_path + " (and .planted.tsv); Rule 10: self-assessment only");
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
    FrontierOpt frontier;       // S4: --frontier FILE [--frontier-test] [--frontier-split B] (with --gaps)
    SignalOpt signals;          // S4b: --signals DIR [--signals-dolly PAGES.xml] [--surprise SCORES.tsv,...] [--signals-gaps FILE]
};

}  // namespace

// ============================================================ F2b: cross-article reading, gold halves (docs/reports/F2_PATTERN_LEARNING.md, "F2b")
// --f2b-xa: every sentence of every article is read; the fact's subject can be a link to an article with seeds (its slot
// [L]), and the object any candidate F2 reads or a mention of the page's own subject ("[O] is the capital of [L]" in the
// city's article). F2's folds, gate and type checks apply, with the subject's type from the linked page. A check counts
// once per (pattern, subject, object) in a fold. At extraction each article's sentence is its own evidence item.
// --f2b-gold-split scores D4's gold fifth in two halves by (fnv >> 50) & 1 (dev 0, test 1); the test half is printed
// only with --f2b-read-test. Off by default: without the flags every output is unchanged.
namespace f2b {

bool g_xa = false, g_gold_split = false, g_read_test = false;
std::unordered_map<string, string> g_redir;  // redirect page title -> its target title

bool load_classes(const string& path) {  // word \t class
    std::ifstream f(path);
    if (!f) return false;
    for (string l; std::getline(f, l);) {
        auto c = split(l, '\t');
        if (c.size() >= 2 && !c[0].empty()) g_f2b_class[c[0]] = "C" + c[1];
    }
    return !g_f2b_class.empty();
}
string norm_title(string t) {
    for (char& ch : t) if (ch == '_') ch = ' ';
    t = trim(t);
    string o;
    for (char ch : t) { if (ch == ' ' && !o.empty() && o.back() == ' ') continue; o += ch; }
    if (!o.empty() && o[0] >= 'a' && o[0] <= 'z') o[0] = static_cast<char>(o[0] - 'a' + 'A');
    return o;
}
void load_redirects(const string& data) {
    std::size_t pos = 0;
    while (true) {
        std::size_t pb = data.find("<page>", pos);
        if (pb == npos) break;
        std::size_t pe = data.find("</page>", pb);
        if (pe == npos) break;
        pos = pe + 7;
        std::size_t tb = data.find("<title>", pb), te = data.find("</title>", tb);
        if (tb == npos || te == npos || te > pe) continue;
        std::size_t xb = data.find("<text", te);
        if (xb == npos || xb > pe) continue;
        std::size_t xg = data.find('>', xb);
        if (xg == npos || xg > pe) continue;
        string head = data.substr(xg + 1, std::min<std::size_t>(300, pe - xg - 1));
        if (lower(trim(head)).compare(0, 9, "#redirect") != 0) continue;
        std::size_t a = head.find("[["), b = head.find("]]", a == npos ? 0 : a);
        if (a == npos || b == npos) continue;
        string tgt = head.substr(a + 2, b - a - 2);
        std::size_t h = tgt.find('#');
        if (h != npos) tgt = tgt.substr(0, h);
        g_redir[xml_decode(data.substr(tb + 7, te - tb - 7))] = norm_title(xml_decode(tgt));
    }
}
// A link target to an article index, through up to three redirects; -1 when it is not a main-namespace article.
int resolve(const string& target, const std::unordered_map<string, int>& page_of) {
    string t = norm_title(target);
    for (int hop = 0; hop < 4; ++hop) {
        auto it = page_of.find(t);
        if (it != page_of.end()) return it->second;
        auto r = g_redir.find(t);
        if (r == g_redir.end()) return -1;
        t = r->second;
    }
    return -1;
}
bool subject_ok_ent(int r, const string& key) {
    const int t = subj_type(r);
    return t == 0 || (t == 1 && g_person.count(key)) || (t == 2 && g_place.count(key)) || (t == 3 && g_country.count(key));
}
bool listish(const string& title) {
    const string lt = lower(title);
    return lt.compare(0, 8, "list of ") == 0 || lt.find("(disambiguation)") != npos;
}
// The cross-article pairs of one sentence: subject link (its article index and span) and object candidates (F2's
// candidates other than the link, and the page subject's own mentions).
struct XPair { int subj_page; int sb, se; Cand obj; };
void pairs_of(const Sent& s, const vector<Cand>& cands, const vector<Span>& spans, const PageRef& pg, const string& title_key,
              const vector<PageRef>& pages, const std::unordered_map<string, int>& page_of, vector<XPair>& out) {
    out.clear();
    vector<Cand> own;  // the page subject as an object
    string base = pg.title;
    std::size_t q = base.find(" (");
    if (q != npos && q > 0 && base.back() == ')') base = base.substr(0, q);
    for (const Span& S : spans) {
        if (S.pron) continue;
        Cand c;
        c.b = S.b; c.e = S.e; c.kind = K_LINK;
        c.k1 = title_key;
        c.k2 = norm_ent(join_o(s, S.b, S.e));
        c.surf = base;
        std::size_t cm = pg.title.find(',');
        if (cm != npos && cm > 0) c.k3 = norm_ent(pg.title.substr(0, cm));
        own.push_back(c);
    }
    for (const Cand& L : cands) {
        if (L.kind != K_LINK) continue;
        const Lnk* lk = nullptr;
        for (const Lnk& l : s.l) if (l.tb == L.b && l.te == L.e) { lk = &l; break; }
        if (!lk) continue;
        const int pi = resolve(lk->target, page_of);
        if (pi < 0 || pages[static_cast<std::size_t>(pi)].title == pg.title || listish(pages[static_cast<std::size_t>(pi)].title)) continue;
        const string ek = norm_ent(pages[static_cast<std::size_t>(pi)].title);
        auto add = [&](const Cand& o) {
            if (o.b < L.e && L.b < o.e) return;
            if (o.k1 == ek) return;
            out.push_back({pi, L.b, L.e, o});
        };
        for (const Cand& o : cands) if (&o != &L) add(o);
        for (const Cand& o : own) add(o);
    }
}
// One pass of cross-article reading over every article. mode 0: train (fold's training subjects, into tm); 1: eval (its
// held-out subjects, into stat; seen dedupes per fold); 2: extract (any subject, into g_new).
void xa_pass(int mode, const string& data, const vector<PageRef>& pages, const std::unordered_map<string, int>& page_of, int K, int fold,
             std::unordered_map<string, TrainCnt>* tm, std::unordered_map<string, PStat>* stat) {
    std::unordered_set<string> subj_keys, seen;
    vector<Cand> cands;
    vector<XPair> xp;
    vector<string> ks, gks;
    for (const PageRef& pg : pages) {
        if (mode == 2 && listish(pg.title)) continue;
        PageCtx px;
        load_page(data, pg, px);
        for (const Sent& s : px.sents) {
            if (s.l.empty()) continue;
            auto spans = find_subject(s, px.forms);
            gen_cands(s, cands);
            pairs_of(s, cands, spans, pg, px.forms.title_key, pages, page_of, xp);
            for (const XPair& x : xp) {
                const string& E = pages[static_cast<std::size_t>(x.subj_page)].title;
                const string ek = norm_ent(E);
                auto sit = g_by_subj.find(E);
                if (mode != 2) {
                    if (sit == g_by_subj.end()) continue;
                    const bool held = heldout(E, K, fold);
                    if ((mode == 0) == held) continue;
                }
                const Cand& c = x.obj;
                vector<int> match;  // mode 0: the subject's seeds this object states
                if (mode == 0) {
                    for (int si : sit->second) {
                        const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                        if (subject_ok_ent(sd.rel, ek) && type_ok(sd.rel, c) && obj_matches(sd.rel, c, sd.o)) match.push_back(sd.rel);
                    }
                    if (match.empty()) continue;
                }
                if (!f2b_keys(s, x.sb, x.se, "[L]", x.obj.b, x.obj.e, false, ks)) continue;
                if (f2b_gen_on() && f2b_keys(s, x.sb, x.se, "[L]", x.obj.b, x.obj.e, true, gks)) ks.insert(ks.end(), gks.begin(), gks.end());
                if (mode == 0) {
                    std::sort(match.begin(), match.end());
                    match.erase(std::unique(match.begin(), match.end()), match.end());
                    for (int rel : match) {
                        for (const string& k : ks) {
                            const string pk = std::to_string(rel) + "\t" + k;
                            ++(*tm)[pk].sent;
                            if (subj_keys.insert(pk + "\x1f" + E).second) ++(*tm)[pk].subj;
                        }
                    }
                    continue;
                }
                for (const string& k : ks) {
                    auto it = g_patmap.find(k);
                    if (it == g_patmap.end()) continue;
                    for (int pid : it->second) {
                        const Pat& p = g_pats[static_cast<std::size_t>(pid)];
                        if (!subject_ok_ent(p.rel, ek) || !type_ok(p.rel, c)) continue;
                        if (mode == 1) {
                            if (!seen.insert(std::to_string(pid) + "\x1f" + E + "\x1f" + obj_key(p.rel, c)).second) continue;
                            bool known = false, ok = false;
                            for (int si : sit->second) {
                                const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                                if (sd.rel != p.rel) continue;
                                known = true;
                                if (obj_consistent(p.rel, c, sd.o)) { ok = true; break; }
                            }
                            PStat& st = (*stat)[std::to_string(p.rel) + "\t" + p.key];
                            if (!known) { ++st.unchecked; continue; }
                            ++st.checks;
                            if (ok) ++st.correct;
                            continue;
                        }
                        bool seeded = false;  // mode 2: already a text seed of E
                        if (sit != g_by_subj.end())
                            for (int si : sit->second) {
                                const Seed& sd = g_seeds[static_cast<std::size_t>(si)];
                                if (sd.src < 3 && sd.rel == p.rel && obj_matches(p.rel, c, sd.o)) { seeded = true; break; }
                            }
                        if (seeded) continue;
                        const string fk = E + "\x1f" + std::to_string(p.rel) + "\x1f" + obj_key(p.rel, c);
                        FactAcc& f = g_new[fk];
                        if (f.subj.empty()) { f.subj = E; f.rel = p.rel; f.surf = obj_surface(p.rel, c); f.okey = obj_key(p.rel, c); f.d = c.d; }
                        ++f.total;
                        const string fam = "x@" + std::to_string(x.sb) + "," + std::to_string(x.se) + "," + std::to_string(c.b) + "," + std::to_string(c.e);
                        if (f.ev.size() < 96) f.ev.push_back({pid, p.reliability, s.off, s.hash, s.idx, fam});
                        if (p.reliability > f.snip_rel) { f.snip_rel = p.reliability; f.snippet = sentence_text(s); }
                    }
                }
            }
        }
    }
}

bool gold_test_half(const string& subj) { return ((fnv(subj) >> 50) & 1) != 0; }
// D4's score_gold, in two halves; the test half only with --f2b-read-test. Also how many admitted facts rest on two or
// more evidence items and reach grade Moderate. Writes facts_wd_check_f2b.tsv (test rows only with --f2b-read-test).
void score_halves(const vector<Fused>& fused, const string& outdir, const std::function<void(const string&)>& say) {
    std::ofstream f(outdir + "/facts_wd_check_f2b.tsv");
    f << "subject\trelation\tobject\tP\twikidata\twikidata_values\thalf\n";
    std::map<int, std::array<int, 3>> n[2];  // admitted right, admitted checked, coarse
    int adm = 0, adm2 = 0, admmod = 0;
    std::map<int, std::array<int, 3>> evr;  // per relation: admitted, >= 2 evidence, >= Moderate
    for (const Fused& fu : fused) {
        if (fu.op.P >= 0.8) {
            ++adm;
            auto& e = evr[fu.f->rel];
            ++e[0];
            if (fu.n_ev >= 2) { ++adm2; ++e[1]; }
            if (ecs::grade_at_least(fu.op.grade, ecs::Grade::Moderate)) { ++admmod; ++e[2]; }
        }
        if (fu.op.P < 0.8) continue;
        const FactAcc& fa = *fu.f;
        auto it = g_gold.find(fa.subj + "\x1f" + kRel[fa.rel].name);
        if (it == g_gold.end()) continue;
        const int h = gold_test_half(fa.subj) ? 1 : 0;
        if (h == 1 && !g_read_test) continue;
        Cand c;
        c.kind = is_datey(fa.rel) ? K_DATE : K_LINK;
        c.d = fa.d;
        c.k1 = fa.okey;
        c.k2 = norm_ent(fa.surf);
        string st = "wrong", vals;
        for (const Obj& o : it->second) {
            vals += (vals.empty() ? "" : "; ") + o.surf;
            if (obj_consistent(fa.rel, c, o)) st = "right";
        }
        for (const Obj& o : it->second)
            for (const string* k : {&o.k1, &o.k2, &o.k3})
                if (st == "wrong" && (inside(*k, c.k1, 3) || inside(*k, c.k2, 3))) st = "coarse";
        auto& a = n[h][fa.rel];
        ++a[1];
        if (st == "right") ++a[0];
        if (st == "coarse") ++a[2];
        f << fa.subj << '\t' << kRel[fa.rel].name << '\t' << fa.surf << '\t' << fu.op.P << '\t' << st << '\t' << vals << '\t' << (h ? "test" : "dev") << '\n';
    }
    char b[256];
    for (int h = 0; h < 2; ++h) {
        if (h == 1 && !g_read_test) continue;
        say(string("\nF2B GOLD ") + (h ? "test" : "dev") + " half: admitted facts (P >= 0.8) against Wikidata, right/checked, Wilson 95%, coarse");
        std::array<int, 3> t{};
        for (auto& [r, a] : n[h]) {
            std::snprintf(b, sizeof b, "  F2B GOLD %-4s %-13s %4d/%-4d %.3f [%.3f, %.3f] coarse %d", h ? "test" : "dev", kRel[r].name, a[0], a[1],
                          a[1] ? static_cast<double>(a[0]) / a[1] : 0.0, ecs::wilson_lower(a[0], a[1]), ecs::wilson_upper(a[0], a[1]), a[2]);
            say(b);
            for (std::size_t k = 0; k < 3; ++k) t[k] += a[k];
        }
        std::snprintf(b, sizeof b, "  F2B GOLD %-4s %-13s %4d/%-4d %.3f [%.3f, %.3f] coarse %d", h ? "test" : "dev", "TOTAL", t[0], t[1],
                      t[1] ? static_cast<double>(t[0]) / t[1] : 0.0, ecs::wilson_lower(t[0], t[1]), ecs::wilson_upper(t[0], t[1]), t[2]);
        say(b);
    }
    std::snprintf(b, sizeof b, "\nF2B EVIDENCE admitted %d | two or more evidence items %d (%.3f) | grade Moderate or better %d", adm, adm2,
                  adm ? static_cast<double>(adm2) / adm : 0.0, admmod);
    say(b);
    for (auto& [r, e] : evr) {
        std::snprintf(b, sizeof b, "  F2B EVIDENCE %-13s admitted %5d | >= 2 items %5d | >= Moderate %5d", kRel[r].name, e[0], e[1], e[2]);
        say(b);
    }
}


// pattern_learn --f2b-selftest: the F2b key rules on built sentences (pattern_learn is not a CMake target, so this is not a CTest).
int selftest() {
    int fails = 0, checks = 0;
    auto check = [&](bool ok, const string& what) { ++checks; if (!ok) { ++fails; std::fprintf(stderr, "F2B SELFTEST FAIL: %s\n", what.c_str()); } };
    auto sent = [](const string& text) {
        vector<Tok> T;
        vector<Lnk> L;
        tokenize_line(text, T, L);
        vector<Sent> out;
        int si = 0;
        make_sentences(T, L, 0, out, si);
        return out.empty() ? Sent{} : out[0];
    };
    auto tok_at = [](const Sent& s, const string& w) { for (std::size_t i = 0; i < s.t.size(); ++i) if (s.t[i].w == w) return static_cast<int>(i); return -1; };
    vector<string> k;
    // exact keys equal make_keys' for [S]
    {
        const Sent s = sent("Foo was born in Paris in 1950 .");
        Span S{0, 1, false};
        std::array<string, 4> mk;
        bool sf;
        const int ob = tok_at(s, "paris");
        check(make_keys(s, S, ob, ob + 1, mk, sf) && f2b_keys(s, 0, 1, "[S]", ob, ob + 1, false, k) && k.size() == 4 &&
              k[0] == mk[0] && k[1] == mk[1] && k[2] == mk[2] && k[3] == mk[3], "exact keys equal make_keys");
        check(k[0] == "[S] was born in [O]", "exact core");
    }
    // wildcard mode 1: the three novel wordings share one core; mode 2 keeps only the last two
    g_f2b_wild = 1;
    {
        const char* txt[3] = {"Foo is a novel by Bar Baz .", "Foo is a 1957 novel by Bar Baz .", "Foo is a science fiction novel by Bar Baz ."};
        for (const char* t : txt) {
            const Sent s = sent(t);
            const int ob = tok_at(s, "bar");
            check(f2b_keys(s, 0, 1, "[S]", ob, ob + 2, true, k) && k[0] == "[S] is <*> novel by [O]", string("wild 1: ") + t);
        }
        const Sent s = sent("Foo is by Bar Baz .");  // gap of 2: no wildcard key
        check(!f2b_keys(s, 0, 1, "[S]", 3, 5, true, k), "wild 1 needs a gap of 4");
    }
    g_f2b_wild = 2;
    {
        const Sent s = sent("Foo was a novel by Bar Baz .");
        const int ob = tok_at(s, "bar");
        check(f2b_keys(s, 0, 1, "[S]", ob, ob + 2, true, k) && k[0] == "[S] <*> novel by [O]", "wild 2 core");
        const Sent s2 = sent("Foo by Bar Baz now .");
        check(!f2b_keys(s2, 0, 1, "[S]", 2, 4, true, k), "wild 2 needs a gap of 3");
    }
    g_f2b_wild = 0;
    // object first, link subject second: "[O] is the capital of [L]"
    {
        const Sent s = sent("Luanda is the capital of [[Angola]] .");
        const int lb = tok_at(s, "angola");
        check(f2b_keys(s, lb, lb + 1, "[L]", 0, 1, false, k) && k[0] == "[O] is the capital of [L]" && k[2] == "^ [O] is the capital of [L]",
              "orientation and context");
        check(family_of(k[3]) == "[O] is the capital of [L]", "family with [L]");
    }
    // word classes: content words become classes, stop words and digits stay
    g_f2b_class = {{"novel", "C7"}, {"science", "C3"}, {"fiction", "C3"}, {"is", "C1"}, {"by", "C9"}};
    {
        const Sent s = sent("Foo is a science fiction novel by Bar Baz .");
        const int ob = tok_at(s, "bar");
        check(f2b_keys(s, 0, 1, "[S]", ob, ob + 2, true, k) && k[0] == "[S] C1 a C3 C3 C7 by [O]", "classes in the gap");
        const Sent s2 = sent("Foo is a 1957 novel by Bar Baz .");
        check(f2b_keys(s2, 0, 1, "[S]", tok_at(s2, "bar"), tok_at(s2, "bar") + 2, true, k) && k[0] == "[S] C1 a # C7 by [O]", "digits stay");
    }
    g_f2b_class.clear();
    check(!f2b_gen_on(), "generalised keys off again");
    // titles and redirects
    check(f2b::norm_title("foo_bar  baz") == "Foo bar baz", "norm_title");
    {
        std::unordered_map<string, int> po = {{"Angola", 3}};
        f2b::g_redir = {{"Republic of Angola", "Angola"}, {"Angolan republic", "Republic of Angola"}, {"Loop", "Loop"}};
        check(f2b::resolve("angola", po) == 3 && f2b::resolve("Republic_of_Angola", po) == 3 && f2b::resolve("Angolan republic", po) == 3,
              "resolve through redirects");
        check(f2b::resolve("Loop", po) == -1 && f2b::resolve("Nowhere", po) == -1, "unresolved targets");
        f2b::g_redir.clear();
    }
    {  // F2c: the lexicon type check (a Wikidata value widens a lexicon only when Wikidata's type agrees)
        const string fn = "/tmp/pl_f2c_selftest_types.tsv";
        { std::ofstream f(fn); f << "# test\nQ31\tQ3624078,Q6256\t1\nQ90\tQ515\t1\nQ1001\tQ5\t0\nQ2\t\t0\n"; }
        lexfix::g_types.clear();
        check(lexfix::load(fn) && lexfix::g_types.size() == 4, "F2c load");
        check(lexfix::is_human("Q1001") && !lexfix::is_human("Q31") && !lexfix::is_human("Q2") && !lexfix::is_human("Q777"), "F2c human");
        check(lexfix::has_geo("Q90") && lexfix::has_geo("Q31") && !lexfix::has_geo("Q1001") && !lexfix::has_geo("Q777"), "F2c coordinates");
        Seed wd, tx;
        wd.src = 3; wd.o.k1 = "belgium"; wd.vqid = "Q31";
        tx.src = 0; tx.o.k1 = "belgium"; tx.vqid = "Q31";
        lexfix::g_arm = 0;
        check(lexfix::widen_ok(wd, T_PERSON) && lexfix::widen_ok(wd, T_PLACE), "F2c off: everything widens");
        lexfix::g_arm = 1;
        check(!lexfix::widen_ok(wd, T_PERSON) && lexfix::widen_ok(tx, T_PERSON), "F2c P: a country stays out of the person lexicon; text seeds widen");
        wd.vqid = "Q1001";
        check(lexfix::widen_ok(wd, T_PERSON), "F2c P: a human widens");
        wd.vqid = "Q2";
        check(lexfix::widen_ok(wd, T_PLACE) && !lexfix::widen_ok(wd, T_PERSON), "F2c P: places unchanged; an untyped value stays out of persons");
        lexfix::g_arm = 2;
        check(!lexfix::widen_ok(wd, T_PLACE_LINK) && (wd.vqid = "Q90", lexfix::widen_ok(wd, T_PLACE)), "F2c PL: places need coordinates");
        check(lexfix::g_blocked_person.count("belgium") == 1, "F2c blocked values recorded");
        lexfix::g_arm = 0;
        lexfix::g_types.clear();
        lexfix::g_blocked_person.clear();
        lexfix::g_blocked_place.clear();
        std::remove(fn.c_str());
    }
    std::printf("f2b selftest: %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

}  // namespace f2b

// ============================================================ S6c: G12's closed form for S6's P0 and S7's T7
// (docs/reports/S6_HYPOTHESIS_SCORING.md, "S6c"; abc_closed.hpp). The 2006 articles as S11b's G12 read them: each
// article's title and its link targets (redirects resolved), dates and lists kept out of the B role. score(s, o) is the
// leave-out closed form (PPMI, hub cap 200, S11b's dev choice): it never reads a document holding both. Scoring Cypha's
// own hypotheses only: it answers nothing (PLAN rule 10).
namespace s6c {
namespace abc = cypha::eval::abc;

struct Text {
    vector<string> names;                                 // entity id -> title
    std::unordered_map<string, std::uint32_t> id;         // title -> entity id
    std::unordered_map<string, vector<std::uint32_t>> by_form;  // norm_ent of a title or a redirect title -> entities
    vector<char> article, usable;
    abc::Cooc cooc;
    std::unique_ptr<abc::Closed> closed;
    std::uint32_t intern(const string& t) {
        const auto it = id.find(t);
        if (it != id.end()) return it->second;
        const auto k = static_cast<std::uint32_t>(names.size());
        id.emplace(t, k), names.push_back(t);
        return k;
    }
};

// Redirect pages of the dump: title -> target (the first link target of the text).
std::unordered_map<string, string> load_redirects(const string& data) {
    std::unordered_map<string, string> out;
    for (std::size_t pos = 0;;) {
        const std::size_t pb = data.find("<page>", pos);
        if (pb == npos) break;
        const std::size_t pe = data.find("</page>", pb);
        if (pe == npos) break;
        pos = pe + 7;
        const std::size_t tb = data.find("<title>", pb), te = data.find("</title>", tb);
        if (tb == npos || te == npos || te > pe) continue;
        const std::size_t xb = data.find("<text", te);
        if (xb == npos || xb > pe) continue;
        const std::size_t xg = data.find('>', xb);
        const std::size_t xe = data.find("</text>", xg);
        if (xg == npos || xe == npos || xe > pe) continue;
        const string head = lower(trim(data.substr(xg + 1, std::min<std::size_t>(40, xe - xg - 1))));
        if (head.compare(0, 9, "#redirect") != 0) continue;
        const auto t = abc::link_targets(xml_decode(data.substr(xg + 1, std::min<std::size_t>(400, xe - xg - 1))));
        if (!t.empty()) out[xml_decode(data.substr(tb + 7, te - tb - 7))] = t[0];
    }
    return out;
}

void build(const string& data, const vector<PageRef>& pages, Text& T) {
    const auto red = load_redirects(data);
    vector<vector<std::uint32_t>> docs;
    vector<std::uint32_t> arts;
    for (const PageRef& pg : pages) {
        const string raw = xml_decode(data.substr(pg.tb, pg.te - pg.tb));
        vector<std::uint32_t> d;
        const std::uint32_t a = T.intern(pg.title);
        arts.push_back(a);
        d.push_back(a);
        for (const string& t : abc::link_targets(raw)) d.push_back(T.intern(abc::resolve(t, red)));
        docs.push_back(std::move(d));
    }
    T.cooc = abc::Cooc::build(std::move(docs), T.names.size());
    vector<vector<string>> forms(T.names.size());
    for (std::uint32_t e = 0; e < T.names.size(); ++e) {
        const string k = norm_ent(T.names[e]);
        if (!k.empty()) forms[e].push_back(k);
    }
    for (const auto& [from, to] : red) {
        const auto it = T.id.find(abc::resolve(to, red));
        if (it == T.id.end()) continue;
        const string k = norm_ent(from);
        auto& f = forms[it->second];
        if (k.size() >= 3 && std::find(f.begin(), f.end(), k) == f.end()) f.push_back(k);
    }
    for (std::uint32_t e = 0; e < T.names.size(); ++e)
        for (const string& f : forms[e]) T.by_form[f].push_back(e);
    T.article.assign(T.names.size(), 0);
    for (std::uint32_t a : arts) T.article[a] = 1;
    T.usable.assign(T.names.size(), 0);
    for (std::uint32_t e = 0; e < T.names.size(); ++e) T.usable[e] = !abc::date_or_list_title(T.names[e]);
    T.closed = std::make_unique<abc::Closed>(T.cooc, T.usable, 200, 2);
}

// An object's entity: among the usable entities whose surface form matches norm_ent(v), with at least 2 documents,
// the one with the most documents (the lower id on a tie); -1 if none (S11b's rule).
long entity_of(const Text& T, const string& v) {
    const auto it = T.by_form.find(norm_ent(v));
    if (it == T.by_form.end()) return -1;
    long best = -1;
    for (std::uint32_t e : it->second)
        if (T.usable[e] && T.cooc.df(e) >= 2 && (best < 0 || T.cooc.df(e) > T.cooc.df(static_cast<std::uint32_t>(best)))) best = e;
    return best;
}

// The closed form of (s, o): s the article with that title, o its entity. -1 when not applicable.
double score(const Text& T, const string& s, const string& o) {
    const auto ait = T.id.find(s);
    if (ait == T.id.end() || !T.article[ait->second]) return -1.0;
    const long c = entity_of(T, o);
    if (c < 0 || static_cast<std::uint32_t>(c) == ait->second) return -1.0;
    return T.closed->score(ait->second, static_cast<std::uint32_t>(c)).score;
}

// Pooled AUROC with pair weights: v[k] = (true score, false score) of pair k, both members weighted w[k]. Ties count
// one half.
double weighted_auroc(const vector<std::pair<double, double>>& v, const vector<double>& w) {
    vector<std::pair<double, std::pair<int, std::size_t>>> it;  // (score, (label, pair))
    for (std::size_t k = 0; k < v.size(); ++k) it.push_back({v[k].first, {1, k}}), it.push_back({v[k].second, {0, k}});
    std::sort(it.begin(), it.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    double below = 0, num = 0, pw = 0, nw = 0;
    for (std::size_t i = 0; i < it.size();) {
        std::size_t j = i;
        double gp = 0, gn = 0;
        while (j < it.size() && it[j].first == it[i].first) (it[j].second.first ? gp : gn) += w[it[j].second.second], ++j;
        num += gp * (below + 0.5 * gn), below += gn, pw += gp, nw += gn;
        i = j;
    }
    return pw > 0 && nw > 0 ? num / (pw * nw) : 0.5;
}

// LCI's P_gain (PLAN 8.2): Bayesian bootstrap over pairs (Dirichlet(1) weights, `draws` draws), the share of draws in
// which the pooled AUROC of v1 exceeds that of v0 by more than eps.
double p_gain(const vector<std::pair<double, double>>& v0, const vector<std::pair<double, double>>& v1, double eps, int draws, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::exponential_distribution<double> ex(1.0);
    int hit = 0;
    vector<double> w(v0.size());
    for (int d = 0; d < draws; ++d) {
        for (double& x : w) x = ex(g);
        if (weighted_auroc(v1, w) - weighted_auroc(v0, w) > eps) ++hit;
    }
    return draws > 0 ? static_cast<double>(hit) / draws : 0.0;
}

}  // namespace s6c

// ============================================================ S6: scoring a planted set (docs/reports/S6_HYPOTHESIS_SCORING.md)
//
// pattern_learn --score-planted SEEDS.tsv WD_VALUES.tsv OUT_PREFIX [--subjects N] [--read-test] [--read-fresh]
// Holds one fact out of each of N subjects of the seed store (a known-true hypothesis), pairs it with a known-false one
// (the same subject and relation, an object drawn by triple from the relation in the reduced store), and scores both
// against the reduced store with hypothesis.hpp's terms. S6b's P0: an isotonic fit (dev) of z(logit P_assoc) +
// z(ln(1 + Adamic-Adar)), times the hard constraint, kept by one joint test on dev. The dev variants are at commit
// 9a8c6a7. S6's test half prints only with --read-test (read once, for S6's change); the fresh set (the qualifying
// subjects S6 did not use) only with --read-fresh. Nothing here answers a question: these are Cypha's own hypotheses
// (PLAN rule 10).
namespace s6 {

struct Item {
    string s, r, o;
    int label = 0, pair = 0;
    bool dev = true;
    double base = 0, p = 0, p_nocopy = 0, aa = 0, hard = 1, bridging = 0, impact = 0;
    double p0 = 0, nov = 0, dl = 0, inter = 0, prio = 0;
    double abc = -1, p0abc = 0;  // S6c: G12's closed form (abc_closed.hpp; -1 when not applicable) and P0 with it
};
using Score = std::function<double(const Item&)>;

const char* group_of(const string& r) {
    static const char* D[] = {"birth_year", "death_year", "birth_date", "death_date", "founded"};
    static const char* P[] = {"birth_place", "death_place", "located_in", "country", "capital", "location", "largest_city"};
    static const char* L[] = {"occupation", "genre", "language", "nationality", "party", "currency", "class", "order", "family"};
    for (const char* x : D) if (r == x) return "dates";
    for (const char* x : P) if (r == x) return "places";
    for (const char* x : L) if (r == x) return "lexical";
    return "people";
}

// The hard constraint (DISCOVERY 6.1's constraint term, hard part): 0 when a date slot of s already holds another year,
// or death comes before birth, or a life is longer than 125 years (as pass 3's fusion); 1 otherwise.
double hard_constraint(const ecs::TripleStore& st, const string& s, int r, const string& o) {
    if (!is_datey(r)) return 1.0;
    auto kind = [](const string& n) { return n == "birth_year" || n == "birth_date" ? 1 : n == "death_year" || n == "death_date" ? 2 : n == "founded" ? 3 : 0; };
    const int k = kind(kRel[r].name);
    const Date D = parse_date_str(o);
    if (!D.ok()) return 1.0;
    for (const auto& f : st.facts_of(s)) {
        const int kf = kind(f.r);
        if (!kf) continue;
        const Date F = parse_date_str(f.o);
        if (!F.ok()) continue;
        if (kf == k && F.y != D.y) return 0.0;
        if (k == 1 && kf == 2 && (F.y < D.y || F.y - D.y > 125)) return 0.0;
        if (k == 2 && kf == 1 && (D.y < F.y || D.y - F.y > 125)) return 0.0;
    }
    return 1.0;
}

// AUROC (label 1 above label 0) over items idx, by mid-ranks.
double auroc(const vector<Item>& it, const vector<std::size_t>& idx, const Score& f) {
    vector<std::pair<double, int>> v;
    v.reserve(idx.size());
    for (std::size_t i : idx) v.push_back({f(it[i]), it[i].label});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    double rank_pos = 0, npos = 0, nneg = 0;
    for (std::size_t i = 0; i < v.size();) {
        std::size_t j = i;
        while (j < v.size() && v[j].first == v[i].first) ++j;
        const double mid = 0.5 * static_cast<double>(i + 1 + j);
        for (std::size_t k = i; k < j; ++k)
            if (v[k].second) rank_pos += mid, ++npos; else ++nneg;
        i = j;
    }
    return npos && nneg ? (rank_pos - npos * (npos + 1) / 2) / (npos * nneg) : 0.5;
}

// AUROC of f, and of f - g when g is given, with 95% bootstrap intervals over pairs (one pair per subject); and the
// share of pairs whose true member scores higher.
struct Read { double auc, lo, hi, pair_acc; };
Read read(const vector<Item>& it, const vector<std::size_t>& pt, const vector<std::size_t>& pf, const Score& f, const Score* g = nullptr) {
    vector<std::size_t> all;
    double acc = 0;
    for (std::size_t k = 0; k < pt.size(); ++k) {
        all.push_back(pt[k]), all.push_back(pf[k]);
        const double a = f(it[pt[k]]), b = f(it[pf[k]]);
        acc += a > b ? 1.0 : a == b ? 0.5 : 0.0;
    }
    Read r{auroc(it, all, f) - (g ? auroc(it, all, *g) : 0.0), 0, 0, pt.empty() ? 0.0 : acc / static_cast<double>(pt.size())};
    std::mt19937_64 rng(20261028);
    vector<double> bs;
    for (int rep = 0; rep < 2000 && !pt.empty(); ++rep) {
        vector<std::size_t> idx;
        for (std::size_t k = 0; k < pt.size(); ++k) {
            const std::size_t j = rng() % pt.size();
            idx.push_back(pt[j]), idx.push_back(pf[j]);
        }
        bs.push_back(auroc(it, idx, f) - (g ? auroc(it, idx, *g) : 0.0));
    }
    std::sort(bs.begin(), bs.end());
    if (!bs.empty()) r.lo = bs[49], r.hi = bs[1949];
    return r;
}

// One planted set: its reduced store, its pairs, and the counts of what was skipped or dropped.
struct PlantedSet {
    ecs::TripleStore st;
    vector<Item> it;
    std::size_t subjects = 0, held = 0, skipped_unseen = 0, dropped_gone = 0, dropped_nofalse = 0;
};

// The planted-set rule on the shuffled subjects from `from` on: each subject's triples are shuffled and the first whose
// object occurs for another subject is held out, until held + skipped reaches `limit`; then the reduced store (the full
// store without these hold-outs) and the false partners. Returns where the next set starts. Every draw is from `rng`.
// S6c: a triple in `exclude` (an earlier set's hold-out) is never held out again.
std::size_t build_set(PlantedSet& ps, const vector<string>& subj, std::size_t from, std::size_t limit,
                      std::map<string, vector<std::pair<string, string>>>& by_subj, const ecs::TripleStore& full,
                      std::unordered_map<string, std::set<string>>& wd, std::mt19937_64& rng, const std::set<string>* exclude = nullptr) {
    auto shuffle = [&](auto& v) { for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng() % i]); };
    struct Held { string s, r, o; };
    vector<Held> held;
    std::set<string> held_keys;
    std::size_t i = from;
    for (; i < subj.size() && held.size() + ps.skipped_unseen < limit; ++i) {
        auto fs = by_subj[subj[i]];
        shuffle(fs);
        bool got = false;
        for (const auto& [r, o] : fs)
            if (full.count(r, o) >= 2 && !(exclude && exclude->count(subj[i] + '\x1f' + r + '\x1f' + o))) {
                held.push_back({subj[i], r, o});
                held_keys.insert(subj[i] + '\x1f' + r + '\x1f' + o);
                got = true;
                break;
            }
        if (!got) ++ps.skipped_unseen;
    }
    ps.subjects = i - from, ps.held = held.size();
    std::unordered_map<string, vector<string>> objects_of_rel;  // one entry per triple, in subject order
    for (const auto& [s, fs] : by_subj)
        for (const auto& [r, o] : fs)
            if (!held_keys.count(s + '\x1f' + r + '\x1f' + o)) ps.st.add(s, r, o), objects_of_rel[r].push_back(o);
    for (const auto& h : held) {
        if (ps.st.count(h.r, h.o) == 0) { ++ps.dropped_gone; continue; }
        const auto& pool = objects_of_rel[h.r];
        const auto& wds = wd[h.s + '\x1f' + h.r];
        const int ri = rel_index(h.r);
        const Date hd = is_datey(ri) ? parse_date_str(h.o) : Date{};
        string neg;
        for (int t = 0; t < 100 && neg.empty(); ++t) {
            const string& c = pool[rng() % pool.size()];
            if (c == h.o || c == h.s || full.has(h.s, h.r, c) || wds.count(lower(c))) continue;
            if (is_datey(ri)) {
                const Date cd = parse_date_str(c);
                if (cd.ok() && hd.ok() && cd.y == hd.y) continue;
            }
            neg = c;
        }
        if (neg.empty()) { ++ps.dropped_nofalse; continue; }
        const bool dev = ((fnv(h.s) >> 7) & 1) == 0;
        const int pr = static_cast<int>(ps.it.size() / 2);
        Item a, b;
        a.s = b.s = h.s, a.r = b.r = h.r, a.o = h.o, b.o = neg, a.label = 1, b.label = 0, a.pair = b.pair = pr, a.dev = b.dev = dev;
        ps.it.push_back(a), ps.it.push_back(b);
    }
    // The raw terms, against this set's reduced store.
    const ecs::LinkPredictor lp(ps.st, {}), lp_nocopy(ps.st, {1.0, false});
    for (auto& x : ps.it) {
        x.base = ecs::base_rate(ps.st, x.r, x.o);
        x.p = lp.prob(x.s, x.r, x.o);
        x.p_nocopy = lp_nocopy.prob(x.s, x.r, x.o);
        x.aa = ecs::adamic_adar(ps.st, x.s, x.o);
        x.hard = hard_constraint(ps.st, x.s, rel_index(x.r), x.o);
        x.bridging = ecs::graph_distance(ps.st, x.s, x.o, 6) - 1;
        x.impact = ecs::hypothesis_impact(ps.st.has_relation(x.s, x.r) ? 0.0 : 1.0, 0.0);
        x.nov = ecs::novelty_bits(x.p);
        x.dl = ecs::compression_gain_bits(lp, ps.st, x.s, x.r, x.o);
    }
    return i;
}

// S6c's fresh set (the report's "S6c"), built the same way by --score-planted and --test-kinds. S6's planted and fresh
// sets and S7's set C are rebuilt with their seeds. The qualifying subjects outside S6's dev pairs are shuffled with
// mt19937_64(20261031); each gives its first triple (shuffled, the same generator) whose object occurs for another
// subject and that none of those sets held out. Then S6's rule: the reduced store without these hold-outs, and false
// partners with S6's Wikidata filter.
void s6c_fresh(PlantedSet& G, std::map<string, vector<std::pair<string, string>>>& by_subj, const ecs::TripleStore& full,
               std::unordered_map<string, std::set<string>>& wd) {
    std::mt19937_64 rng6(20261028);
    vector<string> subj;
    for (const auto& [s, fs] : by_subj)
        if (fs.size() >= 2) subj.push_back(s);
    for (std::size_t i = subj.size(); i > 1; --i) std::swap(subj[i - 1], subj[rng6() % i]);
    PlantedSet A, F;
    const std::size_t next = build_set(A, subj, 0, 2000, by_subj, full, wd, rng6);
    build_set(F, subj, next, std::numeric_limits<std::size_t>::max(), by_subj, full, wd, rng6);
    std::map<string, std::pair<string, string>> h6;
    std::set<string> earlier, dev_subj;
    for (const auto* ps : {&A, &F})
        for (const auto& x : ps->it)
            if (x.label) h6[x.s] = {x.r, x.o}, earlier.insert(x.s + '\x1f' + x.r + '\x1f' + x.o);
    for (const auto& x : A.it)
        if (x.dev) dev_subj.insert(x.s);
    std::mt19937_64 rng7(20261029);  // S7's set C, as S7 builds it
    for (const string& s : subj) {
        const auto it6 = h6.find(s);
        if (it6 == h6.end()) continue;
        auto fs = by_subj[s];
        for (std::size_t i = fs.size(); i > 1; --i) std::swap(fs[i - 1], fs[rng7() % i]);
        for (const auto& [r, o] : fs)
            if (!(r == it6->second.first && o == it6->second.second) && full.count(r, o) >= 2) {
                earlier.insert(s + '\x1f' + r + '\x1f' + o);
                break;
            }
    }
    vector<string> gs;
    for (const auto& [s, fs] : by_subj)
        if (fs.size() >= 2 && !dev_subj.count(s)) gs.push_back(s);
    std::mt19937_64 rng(20261031);
    for (std::size_t i = gs.size(); i > 1; --i) std::swap(gs[i - 1], gs[rng() % i]);
    build_set(G, gs, 0, std::numeric_limits<std::size_t>::max(), by_subj, full, wd, rng, &earlier);
}

int run(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: pattern_learn --score-planted SEEDS.tsv WD_VALUES.tsv OUT_PREFIX [--subjects N] [--read-test] [--read-fresh]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[4];
    std::size_t nsub = 2000;
    bool read_test = false, read_fresh = false;
    string abc_path;                          // S6c: --abc ENWIK8 (the closed form as a P0 term, w chosen on dev)
    bool s6c_dump = false, read_s6c = false;  // S6c: --s6c-dump (the fresh set's pairs, no scores); --read-s6c-fresh (read once)
    for (int i = 5; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--subjects" && i + 1 < argc) nsub = static_cast<std::size_t>(std::atoll(argv[++i]));
        else if (a == "--read-test") read_test = true;
        else if (a == "--read-fresh") read_fresh = true;
        else if (a == "--abc" && i + 1 < argc) abc_path = argv[++i];
        else if (a == "--s6c-dump") s6c_dump = true;
        else if (a == "--read-s6c-fresh") read_s6c = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    // The store: distinct seed triples of a known relation.
    std::map<string, vector<std::pair<string, string>>> by_subj;  // sorted, so the order does not depend on the file
    std::set<string> seen;
    {
        std::ifstream f(argv[2]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            if (seen.insert(c[0] + '\x1f' + c[1] + '\x1f' + c[2]).second) by_subj[c[0]].push_back({c[1], c[2]});
        }
    }
    ecs::TripleStore full;
    for (auto& [s, fs] : by_subj) {
        std::sort(fs.begin(), fs.end());
        for (const auto& [r, o] : fs) full.add(s, r, o);
    }
    std::unordered_map<string, std::set<string>> wd;  // (s, r) -> lowered Wikidata surfaces
    {
        std::ifstream f(argv[3]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 3) continue;
            for (const auto& x : split(c[2], '|')) wd[c[0] + '\x1f' + c[1]].insert(lower(x));
        }
    }
    // Qualifying subjects (two or more triples), shuffled (Fisher-Yates with mt19937_64, the same on every platform).
    std::mt19937_64 rng(20261028);
    vector<string> subj;
    for (const auto& [s, fs] : by_subj)
        if (fs.size() >= 2) subj.push_back(s);
    for (std::size_t i = subj.size(); i > 1; --i) std::swap(subj[i - 1], subj[rng() % i]);
    PlantedSet A, F;  // S6's planted set (dev and test halves), then the fresh set (S6b) from the subjects A did not use
    const std::size_t next = build_set(A, subj, 0, nsub, by_subj, full, wd, rng);
    build_set(F, subj, next, std::numeric_limits<std::size_t>::max(), by_subj, full, wd, rng);
    char b[512];
    for (const auto* ps : {&A, &F}) {
        std::snprintf(b, sizeof b, "S6 %s: subjects %zu, held out %zu (skipped: no seen object %zu) | reduced store %zu of %zu triples | pairs %zu (dropped: object gone %zu, no false partner %zu)",
                      ps == &A ? "planted" : "fresh", ps->subjects, ps->held, ps->skipped_unseen, ps->st.size(), full.size(), ps->it.size() / 2, ps->dropped_gone, ps->dropped_nofalse);
        std::cout << b << "\n";
    }
    vector<std::size_t> dt, df, tt, tf, ft, ff;  // dev / test / fresh pairs: the true and the false member
    for (std::size_t i = 0; i + 1 < A.it.size(); i += 2) (A.it[i].dev ? dt : tt).push_back(i), (A.it[i].dev ? df : tf).push_back(i + 1);
    for (std::size_t i = 0; i + 1 < F.it.size(); i += 2) ft.push_back(i), ff.push_back(i + 1);
    std::cout << "S6 dev pairs " << dt.size() << " test pairs " << tt.size() << " fresh pairs " << ft.size() << "\n";
    // Everything fitted is fitted on A's dev half: the z-score pools and the isotonic fits.
    auto logit = [](double p) { p = std::min(1 - 1e-12, std::max(1e-12, p)); return std::log(p / (1 - p)); };
    auto dev_pool = [&](const Score& f) {
        vector<double> v;
        for (std::size_t i : dt) v.push_back(f(A.it[i])), v.push_back(f(A.it[i + 1]));
        return ecs::pool_stats(v);
    };
    const Score lp_raw = [=](const Item& x) { return logit(x.p); }, lpn_raw = [=](const Item& x) { return logit(x.p_nocopy); };
    const Score aa_raw = [](const Item& x) { return std::log1p(x.aa); };
    const auto zp = dev_pool(lp_raw), zpn = dev_pool(lpn_raw), za = dev_pool(aa_raw);
    auto fit = [&](const Score& raw, bool hard) {
        vector<double> p;
        vector<int> y;
        for (std::size_t i : dt) p.push_back(raw(A.it[i])), y.push_back(1), p.push_back(raw(A.it[i + 1])), y.push_back(0);
        auto cal = std::make_shared<ecs::IsotonicCalibrator>();
        cal->fit(p, y);
        return Score([=](const Item& x) { return ecs::plausibility_prior(1.0, cal->predict(raw(x)), hard ? x.hard : 1.0); });
    };
    const Score s_full = [=](const Item& x) { return ecs::zscore(lp_raw(x), zp) + ecs::zscore(aa_raw(x), za); };
    const Score s_nocopy = [=](const Item& x) { return ecs::zscore(lpn_raw(x), zpn) + ecs::zscore(aa_raw(x), za); };
    const Score p0_full = fit(s_full, true);
    const vector<std::pair<string, Score>> abl = {
        {"joint_removed", fit([](const Item& x) { return x.p_nocopy; }, false)},  // no path, no constraint, no copy rules
        {"without_path", fit([](const Item& x) { return x.p; }, true)},
        {"without_constraint", fit(s_full, false)},
        {"without_copy_rules", fit(s_nocopy, true)},
        {"without_rules", fit([](const Item& x) { return x.base; }, false)}};
    // The joint test (S6b): keep {path, constraint, copy rules} when removing them together costs >= 0.01 on dev, with
    // the interval of the drop above 0.
    const Read joint = read(A.it, dt, df, p0_full, &abl[0].second);
    const bool keep = joint.auc >= 0.01 && joint.lo > 0.0;
    std::snprintf(b, sizeof b, "S6b JOINT dev drop %.4f ci95 %.4f %.4f -> %s", joint.auc, joint.lo, joint.hi, keep ? "keep path, constraint and copy rules" : "cut all three");
    std::cout << b << "\n";
    const Score p0 = keep ? p0_full : abl[0].second;
    for (auto* ps : {&A, &F})
        for (auto& x : ps->it) x.p0 = p0(x);
    const auto zn = dev_pool([](const Item& x) { return x.nov; }), zi = dev_pool([](const Item& x) { return x.impact; });
    const auto zd = dev_pool([](const Item& x) { return x.dl; }), zb = dev_pool([](const Item& x) { return x.bridging; });
    for (auto* ps : {&A, &F})
        for (auto& x : ps->it) {
            x.inter = ecs::interestingness(ecs::zscore(x.nov, zn), ecs::zscore(x.impact, zi), ecs::zscore(x.dl, zd), ecs::zscore(x.bridging, zb));
            x.prio = ecs::voi_priority(ecs::expected_information_gain(x.p0, 0.9, 0.9).eig, x.impact, x.inter, 1.0);
        }
    const vector<std::pair<string, Score>> terms = {
        {"P0", [](const Item& x) { return x.p0; }},
        {"base_rate", [](const Item& x) { return x.base; }},
        {"hard_constraint", [](const Item& x) { return x.hard; }},
        {"novelty", [](const Item& x) { return x.nov; }},
        {"compression_dL", [](const Item& x) { return x.dl; }},
        {"bridging", [](const Item& x) { return x.bridging; }},
        {"impact", [](const Item& x) { return x.impact; }},
        {"interestingness", [](const Item& x) { return x.inter; }},
        {"priority", [](const Item& x) { return x.prio; }}};
    auto pr = [&](const char* tag, const string& name, const Read& r, const char* what = "auroc") {
        std::snprintf(b, sizeof b, "%s %-22s %s %.4f ci95 %.4f %.4f pair_acc %.4f", tag, name.c_str(), what, r.auc, r.lo, r.hi, r.pair_acc);
        std::cout << b << "\n";
    };
    auto table = [&](const char* tag, const vector<Item>& it, const vector<std::size_t>& pt, const vector<std::size_t>& pf) {
        for (const auto& [name, f] : terms) pr(tag, name, read(it, pt, pf, f));
        if (tag == string("FRESH") || tag == string("TEST") || tag == string("DEV")) {
            const Read g = read(it, pt, pf, terms[0].second);
            std::snprintf(b, sizeof b, "%s GATE P0 auroc %.4f ci95 %.4f %.4f -> %s (>= 0.75)", tag, g.auc, g.lo, g.hi, g.auc >= 0.75 ? "met" : "missed");
            std::cout << b << "\n";
        }
        for (const auto& [name, f] : abl) {
            pr(tag, name, read(it, pt, pf, f));
            pr(tag, "P0_minus_" + name, read(it, pt, pf, terms[0].second, &f), "delta");
        }
        for (const char* g : {"dates", "places", "lexical", "people"}) {
            vector<std::size_t> gt, gf;
            for (std::size_t k = 0; k < pt.size(); ++k)
                if (string(group_of(it[pt[k]].r)) == g) gt.push_back(pt[k]), gf.push_back(pf[k]);
            pr(tag, string("P0_") + g + "_" + std::to_string(gt.size()), read(it, gt, gf, terms[0].second));
        }
    };
    table("DEV", A.it, dt, df);
    if (read_test) table("TEST", A.it, tt, tf);
    if (read_fresh) table("FRESH", F.it, ft, ff);
    std::ofstream f(out + ".planted.tsv");
    f << "set\tpair\tsplit\tlabel\tsubject\trelation\tobject\tbase\tp_link\taa\thard\tP0\tnovelty\tdL\tbridging\timpact\tI\tpriority\n";
    for (const auto* ps : {&A, &F})
        for (const auto& x : ps->it) {
            if (ps == &A ? !x.dev && !read_test : !read_fresh) continue;
            f << (ps == &A ? "planted" : "fresh") << '\t' << x.pair << '\t' << (ps == &F ? "fresh" : x.dev ? "dev" : "test") << '\t' << x.label << '\t' << x.s << '\t' << x.r << '\t'
              << x.o << '\t' << x.base << '\t' << x.p << '\t' << x.aa << '\t' << x.hard << '\t' << x.p0 << '\t' << x.nov << '\t' << x.dl << '\t' << x.bridging << '\t'
              << x.impact << '\t' << x.inter << '\t' << x.prio << "\n";
        }
    // ---- S6c (the report's "S6c"): G12's closed form as a P0 term, w chosen on S6's dev pairs; S6c's fresh set read
    // once with --read-s6c-fresh. Without --abc nothing below runs.
    if (!abc_path.empty()) {
        const double t6c = now_s();
        string data;
        if (!read_file(abc_path, data)) { std::fprintf(stderr, "cannot read %s\n", abc_path.c_str()); return 1; }
        long total = 0, redirects = 0;
        const vector<PageRef> pages = list_pages(data, total, redirects);
        s6c::Text T;
        s6c::build(data, pages, T);
        PlantedSet G;
        s6c_fresh(G, by_subj, full, wd);
        for (auto& x : G.it) x.p0 = p0(x);
        for (auto* ps : {&A, &G})
            for (auto& x : ps->it) x.abc = s6c::score(T, x.s, x.o);
        std::snprintf(b, sizeof b, "S6C TEXT articles %zu entities %zu | built and scored in %.1f s", pages.size(), T.names.size(), now_s() - t6c);
        std::cout << b << "\n";
        std::snprintf(b, sizeof b, "S6C FRESH SET subjects %zu, held out %zu (skipped: no unused seen object %zu) | reduced store %zu of %zu | pairs %zu (dropped: object gone %zu, no false partner %zu)",
                      G.subjects, G.held, G.skipped_unseen, G.st.size(), full.size(), G.it.size() / 2, G.dropped_gone, G.dropped_nofalse);
        std::cout << b << "\n";
        {
            std::uint64_t chk = 1469598103934665603ULL;
            for (const auto& x : G.it) chk = (chk ^ fnv(std::to_string(x.label) + '\x1f' + x.s + '\x1f' + x.r + '\x1f' + x.o)) * 1099511628211ULL;
            std::snprintf(b, sizeof b, "S6C FRESH SET pairs %zu | reduced store %zu | checksum %016llx", G.it.size() / 2, G.st.size(), static_cast<unsigned long long>(chk));
            std::cout << b << "\n";
        }
        vector<std::size_t> gt, gf;
        for (std::size_t i = 0; i + 1 < G.it.size(); i += 2) gt.push_back(i), gf.push_back(i + 1);
        if (s6c_dump) {
            std::ofstream d(out + ".s6c_fresh_pairs.tsv");
            d << "pair\tlabel\tsubject\trelation\tobject\n";
            for (const auto& x : G.it) d << x.pair << '\t' << x.label << '\t' << x.s << '\t' << x.r << '\t' << x.o << '\n';
        }
        // dev: coverage, the closed form alone, and P0 with the term for each w
        std::size_t app = 0, both = 0;
        vector<std::size_t> bt, bf;
        for (std::size_t k = 0; k < dt.size(); ++k) {
            const auto &xt = A.it[dt[k]], &xf = A.it[df[k]];
            app += (xt.abc >= 0) + (xf.abc >= 0);
            if (xt.abc >= 0 && xf.abc >= 0) ++both, bt.push_back(dt[k]), bf.push_back(df[k]);
        }
        const Score abc_raw = [](const Item& x) { return x.abc < 0 ? -1.0 : std::log1p(x.abc); };
        {
            const Read r = read(A.it, bt, bf, abc_raw);
            std::snprintf(b, sizeof b, "S6C DEV applicable %zu of %zu items (%.3f) | pairs with both members applicable %zu of %zu | the closed form alone on them: auroc %.4f ci95 %.4f %.4f pair_acc %.4f",
                          app, 2 * dt.size(), dt.empty() ? 0.0 : static_cast<double>(app) / (2.0 * dt.size()), both, dt.size(), r.auc, r.lo, r.hi, r.pair_acc);
            std::cout << b << "\n";
        }
        vector<double> va;
        for (std::size_t i : dt)
            for (std::size_t j : {i, i + 1})
                if (A.it[j].abc >= 0) va.push_back(std::log1p(A.it[j].abc));
        const auto zab = ecs::pool_stats(va);
        auto zabc = [zab](const Item& x) { return x.abc < 0 ? 0.0 : ecs::zscore(std::log1p(x.abc), zab); };
        const double kW[7] = {0, 0.25, 0.5, 0.75, 1, 1.5, 2};
        double bw = 0, bauc = -1;
        vector<double> dev_auc(7);
        for (int k = 0; k < 7; ++k) {
            const double w = kW[k];
            const Score pw = fit([=](const Item& x) { return s_full(x) + w * zabc(x); }, true);
            const Read r = read(A.it, dt, df, pw);
            dev_auc[static_cast<std::size_t>(k)] = r.auc;
            std::snprintf(b, sizeof b, "S6C DEV w %.2f P0_abc auroc %.4f ci95 %.4f %.4f pair_acc %.4f", w, r.auc, r.lo, r.hi, r.pair_acc);
            std::cout << b << "\n";
            if (r.auc > bauc + 0.001) bauc = r.auc, bw = w;
        }
        const Score p0abc = fit([=](const Item& x) { return s_full(x) + bw * zabc(x); }, true);
        const Read dgain = read(A.it, dt, df, p0abc, &p0);
        std::snprintf(b, sizeof b, "S6C CHOICE w %.2f (dev auroc %.4f) | P0_abc minus P0 on dev %+.4f ci95 %+.4f %+.4f", bw, bauc, dgain.auc, dgain.lo, dgain.hi);
        std::cout << b << "\n";
        for (const char* g : {"dates", "places", "lexical", "people"}) {
            vector<std::size_t> pt, pf;
            for (std::size_t k = 0; k < dt.size(); ++k)
                if (string(group_of(A.it[dt[k]].r)) == g) pt.push_back(dt[k]), pf.push_back(df[k]);
            const Read r0 = read(A.it, pt, pf, p0), r1 = read(A.it, pt, pf, p0abc);
            std::snprintf(b, sizeof b, "S6C DEV group %-8s pairs %4zu P0 %.4f P0_abc %.4f", g, pt.size(), r0.auc, r1.auc);
            std::cout << b << "\n";
        }
        if (read_s6c && bw > 0) {
            for (auto& x : G.it) x.p0abc = p0abc(x);
            const Score q1 = [](const Item& x) { return x.p0abc; }, q0 = [](const Item& x) { return x.p0; };
            const Read r0 = read(G.it, gt, gf, q0), r1 = read(G.it, gt, gf, q1), d = read(G.it, gt, gf, q1, &q0);
            std::snprintf(b, sizeof b, "S6C FRESH P0 auroc %.4f ci95 %.4f %.4f pair_acc %.4f | P0_abc auroc %.4f ci95 %.4f %.4f pair_acc %.4f", r0.auc, r0.lo, r0.hi,
                          r0.pair_acc, r1.auc, r1.lo, r1.hi, r1.pair_acc);
            std::cout << b << "\n";
            std::snprintf(b, sizeof b, "S6C FRESH P0_abc minus P0 %+.4f ci95 %+.4f %+.4f | S6's gate (P0_abc >= 0.75): %s", d.auc, d.lo, d.hi, r1.auc >= 0.75 ? "met" : "missed");
            std::cout << b << "\n";
            {
                std::size_t ga = 0;
                for (const auto& x : G.it) ga += x.abc >= 0;
                vector<std::size_t> ct, cf;
                for (std::size_t k = 0; k < gt.size(); ++k)
                    if (G.it[gt[k]].abc >= 0 && G.it[gf[k]].abc >= 0) ct.push_back(gt[k]), cf.push_back(gf[k]);
                const Read ra = read(G.it, ct, cf, abc_raw);
                std::snprintf(b, sizeof b, "S6C FRESH applicable %zu of %zu items | the closed form alone on %zu pairs with both applicable: auroc %.4f ci95 %.4f %.4f",
                              ga, G.it.size(), ct.size(), ra.auc, ra.lo, ra.hi);
                std::cout << b << "\n";
            }
            auto pairs_of = [&](const vector<std::size_t>& pt, const vector<std::size_t>& pf, const Score& f) {
                vector<std::pair<double, double>> v;
                for (std::size_t k = 0; k < pt.size(); ++k) v.push_back({f(G.it[pt[k]]), f(G.it[pf[k]])});
                return v;
            };
            // LCI (PLAN 8.2) on the fresh set; S15's canary is read by --gen-links (S8b's read 4) and joined in the report.
            const double pg = s6c::p_gain(pairs_of(gt, gf, q0), pairs_of(gt, gf, q1), 0.005, 4000, 20261031);
            double r_ok = 1.0;
            for (const char* g : {"dates", "places", "lexical", "people"}) {
                vector<std::size_t> pt, pf;
                for (std::size_t k = 0; k < gt.size(); ++k)
                    if (string(group_of(G.it[gt[k]].r)) == g) pt.push_back(gt[k]), pf.push_back(gf[k]);
                const Read r0g = read(G.it, pt, pf, q0), r1g = read(G.it, pt, pf, q1), dg = read(G.it, pt, pf, q1, &q0);
                const double pnf = s6c::p_gain(pairs_of(pt, pf, q0), pairs_of(pt, pf, q1), -0.01, 4000, 20261031);
                if (pnf < 0.95) r_ok = 0.0;
                std::snprintf(b, sizeof b, "S6C FRESH group %-8s pairs %4zu P0 %.4f P0_abc %.4f minus %+.4f ci95 %+.4f %+.4f | canary P(delta > -0.01) %.4f", g, pt.size(),
                              r0g.auc, r1g.auc, dg.auc, dg.lo, dg.hi, pnf);
                std::cout << b << "\n";
            }
            const double gain_dev = dgain.auc, gain_fresh = d.auc;
            const double gam = gain_dev > 0 ? std::min(1.0, std::max(0.0, gain_fresh / gain_dev) / 0.5) : (gain_fresh > 0 ? 1.0 : 0.0);
            double br0 = 0, br1 = 0, ece0 = 0, ece1 = 0;
            {
                double n0[10] = {}, s0[10] = {}, y0[10] = {}, n1[10] = {}, s1[10] = {}, y1[10] = {};
                for (const auto& x : G.it) {
                    br0 += (x.p0 - x.label) * (x.p0 - x.label), br1 += (x.p0abc - x.label) * (x.p0abc - x.label);
                    const int k0 = std::min(9, static_cast<int>(x.p0 * 10)), k1 = std::min(9, static_cast<int>(x.p0abc * 10));
                    n0[k0] += 1, s0[k0] += x.p0, y0[k0] += x.label, n1[k1] += 1, s1[k1] += x.p0abc, y1[k1] += x.label;
                }
                const double n = std::max<double>(1, G.it.size());
                br0 /= n, br1 /= n;
                for (int k = 0; k < 10; ++k) ece0 += std::fabs(s0[k] - y0[k]) / n, ece1 += std::fabs(s1[k] - y1[k]) / n;
            }
            const double cc = br1 <= br0 && ece1 <= ece0 + 0.01 ? 1.0 : 0.5;
            int sg[3] = {0, 0, 0};
            for (int blk = 0; blk < 3; ++blk) {
                vector<std::size_t> pt, pf;
                for (std::size_t k = 0; k < gt.size(); ++k)
                    if (fnv(G.it[gt[k]].s) % 3 == static_cast<std::uint64_t>(blk)) pt.push_back(gt[k]), pf.push_back(gf[k]);
                const double dd = auroc(G.it, [&] { vector<std::size_t> a; for (std::size_t k = 0; k < pt.size(); ++k) a.push_back(pt[k]), a.push_back(pf[k]); return a; }(), q1) -
                                  auroc(G.it, [&] { vector<std::size_t> a; for (std::size_t k = 0; k < pt.size(); ++k) a.push_back(pt[k]), a.push_back(pf[k]); return a; }(), q0);
                sg[blk] = dd > 0 ? 1 : dd < 0 ? -1 : 0;
            }
            const double sig = (sg[0] == sg[1] && sg[1] == sg[2] && sg[0] != 0) ? 1.0 : 0.5;
            struct rusage ru6;
            getrusage(RUSAGE_SELF, &ru6);
            const double kap = (now_s() - t0) < 900 && ru6.ru_maxrss < 3 * 1024 * 1024 ? 1.0 : 0.0;
            double lci = pg * r_ok * gam * cc * sig * kap;
            if (gt.size() < 200) lci = std::min(lci, 0.5);
            std::snprintf(b, sizeof b,
                          "S6C FRESH LCI P_gain %.4f (eps 0.005, 4,000 draws) | n_eff %zu | r(fresh groups) %.0f | gamma %.3f (gain dev %+.4f fresh %+.4f) | c %.1f (Brier P0 %.4f P0_abc %.4f, ECE %.4f %.4f) | "
                          "sigma %.1f (blocks %+d %+d %+d) | kappa %.0f | LCI %.4f (before S15's canary)",
                          pg, gt.size(), r_ok, gam, gain_dev, gain_fresh, cc, br0, br1, ece0, ece1, sig, sg[0], sg[1], sg[2], kap, lci);
            std::cout << b << "\n";
            std::ofstream fo(out + ".s6c_fresh.tsv");
            fo << "pair\tlabel\tsubject\trelation\tobject\tabc\tP0\tP0_abc\n";
            for (const auto& x : G.it) fo << x.pair << '\t' << x.label << '\t' << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.abc << '\t' << x.p0 << '\t' << x.p0abc << '\n';
        } else if (read_s6c) {
            std::cout << "S6C FRESH not read: w = 0 on dev, so the term is not adopted (rule 1)\n";
        }
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S6 seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s6

// ============================================================ S7: testing hypotheses (docs/reports/S7_HYPOTHESIS_TESTS.md)
//
// pattern_learn --test-kinds ENWIK8 SEEDS.tsv WD_VALUES.tsv PATTERNS.tsv OUT_PREFIX [--read-test]
// Measures the sensitivity s and specificity c of the tests of DISCOVERY 7.1 (T1 consistency, T3 held-out prediction for
// rules, T4 corpus, T5 Wikidata, T6 compression; T2 consequence was cut: no rule fired) on known-true and known-false hypotheses, and
// runs 7.2's loop (pre-registration, cheapest first, Bayes factors, BH batch control) on them. The set: U, the one-fact
// subjects S6 never used (their fact held out), and C, a second fact of each subject S6 held one out of (S6's own pairs
// are never read); false partners by S6's rule without its Wikidata filter (wd_true marks those Wikidata lists). Dev and
// test by bit 11 of FNV-1a(subject); test prints only with --read-test. PATTERNS.tsv is the run's patterns.tsv (its
// kept rows are T4's patterns); T4 reads only the prose of the subject's own article. Nothing here answers a question:
// these are Cypha's own hypotheses (PLAN rule 10).
namespace s7 {

using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;
constexpr double kEps[4] = {0.0, 0.5, 1.0, 2.0};  // T6's dead band in bits, chosen on dev
const char* const kT4Var[2] = {"pattern", "pattern_or_comention"};

struct Item {
    string s, r, o;
    int label = 0, pair = 0;
    char set = 'U';
    bool dev = true, wd_true = false, t5_ok = false;
    double p = 0, aa = 0, dl_others = 0, p0 = 0, post = 0;
    TestOutcome t1 = kInc, t5 = kInc, t6[4] = {kInc, kInc, kInc, kInc}, t4[2] = {kInc, kInc};
    string state;
    TestOutcome t1v[3] = {kInc, kInc, kInc};  // S7b: T1 under phi_hand, phi_wd, phi_store
    char cat = ' ';                           // S7b, S15 items: the category (a, b, c, u)
    bool reach = false;                       // S7b, S15 items: both members' objects occur for r in the store
    TestOutcome t1_reg = kInc;                // S7b, S15 items: T1 with phi re-estimated on the S15-reduced store (registered)
    double abc = -1;                          // S6c: G12's closed form (-1 not applicable)
    TestOutcome t7 = kInc;                    // S6c: T7, run only where T4 is inconclusive
};

bool functional_rel(const string& r) {
    const int i = rel_index(r);
    return i >= 0 && kRel[i].phi >= 0.9;
}

// Wikidata's values of (s, r) as objects (load_wikidata's keys: every surface, normalised).
using WdObjs = std::unordered_map<string, vector<Obj>>;
WdObjs load_wd_objs(const string& path) {
    WdObjs out;
    std::ifstream f(path);
    string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        const auto c = split(line, '\t');
        const int r = c.size() >= 3 ? rel_index(c[1]) : -1;
        if (r < 0) continue;
        const auto sf = split(c[2], '|');
        Obj o;
        o.surf = sf[0];
        if (is_datey(r)) {
            o.d = parse_date_str(sf[0]);
            if (!o.d.ok()) continue;
            if (kRel[r].type == T_YEAR) o.d.m = o.d.d = 0;
            o.k1 = o.k2 = fmt_date(o.d);
        } else {
            vector<string> keys;
            for (const string& x : sf) {
                const string k = norm_ent(x);
                if (!k.empty() && std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
            }
            if (keys.empty()) continue;
            o.k1 = keys[0];
            if (keys.size() > 1) o.k2 = keys[1];
            if (keys.size() > 2) o.k3 = keys[2];
            if (keys.size() > 3) o.alt.assign(keys.begin() + 3, keys.end());
        }
        out[c[0] + '\x1f' + c[1]].push_back(o);
    }
    return out;
}

// Does a stored object (a seed surface) equal one of Wikidata's values? Dates by year (year relations) or by
// compatible date; other values by any shared normalised key.
bool wd_match(int r, const string& o, const vector<Obj>& wv) {
    if (is_datey(r)) {
        Date D = parse_date_str(o);
        if (!D.ok()) return false;
        for (const Obj& w : wv)
            if (kRel[r].type == T_YEAR ? w.d.y == D.y : date_compat(w.d, D)) return true;
        return false;
    }
    const string k = norm_ent(o);
    for (const Obj& w : wv) {
        if (k == w.k1 || k == w.k2 || k == w.k3) return true;
        for (const string& a : w.alt)
            if (k == a) return true;
    }
    return false;
}

// T1: a date conflict (S6's hard constraint), a functional (phi >= 0.9, not a date) relation of s already holding
// another value, or an object of the wrong type (a place relation's object known as a person page, a person relation's
// as a place) is negative. Positive when a date cross-check was possible (s has another date fact) and passed.
TestOutcome t1(const ecs::TripleStore& st, const string& s, const string& r, const string& o) {
    const int ri = rel_index(r);
    if (s6::hard_constraint(st, s, ri, o) == 0.0) return kNeg;
    if (!is_datey(ri) && functional_rel(r) && ecs::functional_conflict(st, s, r, o)) return kNeg;
    const string k = norm_ent(o);
    const RT ty = kRel[ri].type;
    if ((ty == T_PLACE || ty == T_PLACE_LINK) && g_person.count(k) && !g_place.count(k)) return kNeg;
    if (ty == T_PERSON && (g_place.count(k) || g_country.count(k)) && !g_person.count(k)) return kNeg;
    if (is_datey(ri)) {
        for (const auto& f : st.facts_of(s)) {
            const int fi = rel_index(f.r);
            if (fi >= 0 && is_datey(fi) && parse_date_str(f.o).ok()) return kPos;
        }
    }
    return kInc;
}

// T4's reading of one article: the pattern extractions of relation r, and the candidates of r's type in sentences
// that mention the subject (prose only: templates, tables, lists and categories are not sentences).
struct PageScan {
    vector<Cand> pat, co;
};
PageScan scan_rel(const string& data, const PageRef& pg, int r) {
    PageScan out;
    PageCtx px;
    load_page(data, pg, px);
    vector<Cand> cands;
    const string tk = px.forms.title_key;
    for (const Sent& s : px.sents) {
        const auto spans = find_subject(s, px.forms);
        if (spans.empty()) continue;
        gen_cands(s, cands);
        apply_patterns(s, spans, cands, tk, [&](int pid, const Cand& c, const Span&) {
            if (g_pats[static_cast<std::size_t>(pid)].rel == r) out.pat.push_back(c);
        });
        for (const Cand& c : cands) {
            bool overlap = false;
            for (const Span& S : spans) overlap = overlap || (c.b < S.e && S.b < c.e);
            if (!overlap && c.k1 != tk && subject_ok(r) && type_ok(r, c)) out.co.push_back(c);
        }
    }
    return out;
}

// T4 for one object: variant 0 positive on a pattern extraction of o, variant 1 also on a co-mention; negative (both)
// when a pattern extracts another value of a functional r and nothing supports o.
void t4(const PageScan& ps, int r, const string& o, TestOutcome out[2]) {
    out[0] = out[1] = kInc;
    const auto objs = make_objs(r, o);
    if (objs.empty()) return;
    bool pat_for = false, co_for = false, against = false;
    for (const Cand& c : ps.pat) {
        bool m = false;
        for (const Obj& O : objs) m = m || obj_matches(r, c, O);
        (m ? pat_for : against) = true;
    }
    for (const Cand& c : ps.co)
        for (const Obj& O : objs) co_for = co_for || obj_matches(r, c, O);
    against = against && functional_rel(kRel[r].name);
    out[0] = pat_for ? kPos : against ? kNeg : kInc;
    out[1] = pat_for || co_for ? kPos : against ? kNeg : kInc;
}

// Tallies of one test over items, with S6's labels (false members Wikidata lists are left out) unless `all_false`.
template <class F>
ecs::TestTally tally(const vector<Item>& it, bool dev, char set, bool all_false, F&& outcome) {
    ecs::TestTally t;
    for (const Item& x : it) {
        if (x.dev != dev || (set != '*' && x.set != set) || (!x.label && x.wd_true && !all_false)) continue;
        t.add(x.label == 1, outcome(x));
    }
    return t;
}

string line_of(const char* tag, const string& name, const ecs::TestTally& t) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "%s %-26s n %5.0f applicable %.3f s %.3f [%.3f, %.3f] c %.3f [%.3f, %.3f] J %+.3f [%+.3f, %+.3f] MI %.4f bits"
                  " | true pos %.0f neg %.0f inc %.0f | false pos %.0f neg %.0f inc %.0f",
                  tag, name.c_str(), t.total(), t.applicable(), t.sensitivity(), t.s_lo(), t.s_hi(), t.specificity(), t.c_lo(), t.c_hi(), t.youden(),
                  t.youden_lo(), t.youden_hi(), t.mutual_information_bits(), t.t_pos, t.t_neg, t.t_inc, t.f_pos, t.f_neg, t.f_inc);
    return b;
}

// T3 on planted rules inside one split: its subjects are cut in three by FNV-1a bits 13+ (0 generator, 1 hidden, 2 truth).
// The generator third proposes pairwise rules (r2, o2) => (r, o) with support >= 5 and confidence >= 0.5; T3 predicts the
// hidden third (rule_holdout_test); the truth third says whether the rule holds (precision >= 0.5 on >= 3 cases).
ecs::TestTally t3_planted(const ecs::TripleStore& st, bool dev, std::size_t* proposed) {
    struct Cnt {
        double body = 0, both = 0;
    };
    std::array<std::map<string, Cnt>, 3> cnt;  // key r2 \x1f o2 \x1f r \x1f o, per third
    std::array<std::map<string, double>, 3> body;  // key r2 \x1f o2 \x1f r: subjects with the body and slot r filled
    for (const auto& [s, fs] : st.subjects()) {
        if ((((fnv(s) >> 11) & 1) == 0) != dev) continue;
        const int third = static_cast<int>((fnv(s) >> 13) % 3);
        std::set<string> rels;
        for (const auto& f : fs) rels.insert(f.r);
        std::set<string> seen_body;
        for (const auto& b : fs)
            for (const string& r : rels) {
                const string bk = b.r + '\x1f' + b.o + '\x1f' + r;
                if (!seen_body.insert(bk).second) continue;
                body[static_cast<std::size_t>(third)][bk] += 1;
                for (const auto& h : fs)
                    if (h.r == r && !(h.r == b.r && h.o == b.o)) cnt[static_cast<std::size_t>(third)][bk + '\x1f' + h.o].both += 1;
            }
    }
    ecs::TestTally t;
    std::size_t n = 0;
    for (const auto& [k, c] : cnt[0]) {
        const string bk = k.substr(0, k.rfind('\x1f'));
        const double nb = body[0][bk];
        if (nb < 5 || c.both / nb < 0.5) continue;
        ++n;
        auto get = [&](int third) {
            const auto it = cnt[static_cast<std::size_t>(third)].find(k);
            const auto ib = body[static_cast<std::size_t>(third)].find(bk);
            return std::pair<double, double>{it == cnt[static_cast<std::size_t>(third)].end() ? 0.0 : it->second.both,
                                             ib == body[static_cast<std::size_t>(third)].end() ? 0.0 : ib->second};
        };
        const auto [hh, hn] = get(1);
        const auto [th, tn] = get(2);
        if (tn < 3) continue;
        t.add(th / tn >= 0.5, ecs::rule_holdout_test(hh, hn));
    }
    if (proposed) *proposed = n;
    return t;
}


// ------------------------------------------------------------ S7b (docs/reports/S7_HYPOTHESIS_TESTS.md, "S7b")
// phi for T1, three-way likelihoods, batch control; the confirm read on S15's dev half.
enum PhiSrc { PHI_HAND, PHI_WD, PHI_STORE, N_PHI };
const char* const kPhiName[N_PHI] = {"hand", "wikidata", "store"};
using PhiTable = std::array<double, 64>;

PhiTable phi_hand() {
    PhiTable p{};
    for (int r = 0; r < kNRel; ++r) p[static_cast<std::size_t>(r)] = kRel[r].phi;
    return p;
}
// D4's estimate (relation \t ... \t phi); kRel's value where the file has none. Empty path: kRel's.
PhiTable phi_from_file(const string& path) {
    PhiTable p = phi_hand();
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        const int r = c.size() >= 2 ? rel_index(c[0]) : -1;
        if (r >= 0) p[static_cast<std::size_t>(r)] = std::atof(c.back().c_str());
    }
    return p;
}
// The store's own: among subjects with a value of r, the share with exactly one distinct value (distinct years for
// date relations, distinct normalised keys otherwise); kRel's value for a relation no subject has. `n` gets the subjects.
PhiTable phi_from_store(const ecs::TripleStore& st, std::array<double, 64>* n = nullptr) {
    PhiTable p = phi_hand();
    std::array<double, 64> one{}, any{};
    for (const auto& [s, fs] : st.subjects()) {
        std::map<int, std::set<string>> vals;
        for (const auto& f : fs) {
            const int r = rel_index(f.r);
            if (r < 0) continue;
            string k = norm_ent(f.o);
            if (is_datey(r)) {
                const Date D = parse_date_str(f.o);
                if (D.ok()) k = std::to_string(D.y);
            }
            vals[r].insert(k);
        }
        for (const auto& [r, v] : vals) any[static_cast<std::size_t>(r)] += 1, one[static_cast<std::size_t>(r)] += v.size() == 1;
    }
    for (int r = 0; r < kNRel; ++r)
        if (any[static_cast<std::size_t>(r)] > 0) p[static_cast<std::size_t>(r)] = one[static_cast<std::size_t>(r)] / any[static_cast<std::size_t>(r)];
    if (n) *n = any;
    return p;
}

// T1 with phi: a relation is functional when phi >= 0.9. phi gates the functional conflict (non-date relations) and S6's
// same-kind date conflict (a date slot of s already holding another year); the lifespan checks and the type checks always
// apply. With kRel's phi this is t1 above, outcome for outcome.
TestOutcome t1_phi(const ecs::TripleStore& st, const string& s, const string& r, const string& o, const PhiTable& phi) {
    const int ri = rel_index(r);
    const bool functional = phi[static_cast<std::size_t>(ri)] >= 0.9;
    if (is_datey(ri)) {
        auto kind = [](const string& n) { return n == "birth_year" || n == "birth_date" ? 1 : n == "death_year" || n == "death_date" ? 2 : n == "founded" ? 3 : 0; };
        const int k = kind(kRel[ri].name);
        const Date D = parse_date_str(o);
        if (D.ok())
            for (const auto& f : st.facts_of(s)) {
                const int kf = kind(f.r);
                if (!kf) continue;
                const Date F = parse_date_str(f.o);
                if (!F.ok()) continue;
                if (kf == k && F.y != D.y && functional) return kNeg;
                if (k == 1 && kf == 2 && (F.y < D.y || F.y - D.y > 125)) return kNeg;
                if (k == 2 && kf == 1 && (D.y < F.y || D.y - F.y > 125)) return kNeg;
            }
    }
    if (!is_datey(ri) && functional && ecs::functional_conflict(st, s, r, o)) return kNeg;
    const string k = norm_ent(o);
    const RT ty = kRel[ri].type;
    if ((ty == T_PLACE || ty == T_PLACE_LINK) && g_person.count(k) && !g_place.count(k)) return kNeg;
    if (ty == T_PERSON && (g_place.count(k) || g_country.count(k)) && !g_person.count(k)) return kNeg;
    if (is_datey(ri)) {
        for (const auto& f : st.facts_of(s)) {
            const int fi = rel_index(f.r);
            if (fi >= 0 && is_datey(fi) && parse_date_str(f.o).ok()) return kPos;
        }
    }
    return kInc;
}

// One test kind in a loop: its outcome on an item, two-way s and c (add-one, S7's amended form) and three-way likelihoods.
struct LoopKind {
    const char* name;
    const char* source;
    bool disconfirmation, independent;
    std::function<TestOutcome(const Item&)> f;
    double s, c;
    ecs::OutcomeLikelihoods lik;
    std::function<bool(const Item&)> run_if = {};  // S6c: when set, the test runs only where it holds (else it is not run)
};
enum Batch { B0_ALL, B1_SUPPORTED, B2_RELATION, B_NONE };
const char* const kBatchName[4] = {"B0_all", "B1_among_supported", "B2_per_relation", "none"};
struct LoopStats {
    double sup_t = 0, sup_f = 0, ref_t = 0, ref_f = 0, n = 0, brier_post = 0, brier_prior = 0;
    std::map<char, double> sup_cat;
    std::size_t by_decide = 0, kept = 0, entries = 0;
    bool log_ok = true;
    double sup_n() const { return sup_t + sup_f; }
    double ref_n() const { return ref_t + ref_f; }
    double sup_prec() const { return sup_n() > 0 ? sup_t / sup_n() : 0.0; }
    double ref_prec() const { return ref_n() > 0 ? ref_f / ref_n() : 0.0; }
    double decided() const { return n > 0 ? (sup_n() + ref_n()) / n : 0.0; }
};

// 7.2's loop on items idx, one batch: pre-registration, cheapest first, BFs (two-way, or three-way when `three`), then
// batch control; a hypothesis `decide` supports but the batch step does not keep goes back to the queue. Items keep
// their fields (S7's loop writes post and state; this one does not). False members Wikidata lists are left out of the
// counts (S6's labels), as in S7.
LoopStats loop2(const vector<Item>& it, const vector<std::size_t>& idx, const vector<LoopKind>& kinds, bool three, Batch batch,
                const string& log_path = "") {
    ecs::HypothesisStore hs;
    ecs::PreregLog log;
    auto hid = [](std::size_t i) { return "S7B-" + std::to_string(i); };
    for (std::size_t i : idx) {
        const Item& x = it[i];
        ecs::Hypothesis h;
        h.id = hid(i);
        h.key = x.r + '|' + x.s + '|' + x.o;
        h.statement = x.s + " " + x.r + " " + x.o;
        h.generator = "planted-store";
        for (const LoopKind& K : kinds) {
            ecs::HypTest t;
            t.kind = K.name, t.source = K.source;
            t.predicted = "positive", t.refuted_if = "negative";
            t.disconfirmation = K.disconfirmation, t.independent = K.independent;
            t.sensitivity = K.s, t.specificity = K.c;
            if (three) t.likelihoods = K.lik;
            h.tests.push_back(t);
        }
        hs.propose(h);
        hs.score(h.id, x.p0);
        hs.transition(h.id, ecs::HypState::Queued, "scored");
        if (!log.register_plan(*hs.find(h.id))) continue;
        for (std::size_t k = 0; k < kinds.size(); ++k) {
            if (hs.find(h.id)->state != ecs::HypState::Queued) break;
            if (kinds[k].run_if && !kinds[k].run_if(x)) continue;
            hs.transition(h.id, ecs::HypState::UnderTest, "next test");
            hs.record_outcome(h.id, k, kinds[k].f(x), &log);
        }
    }
    vector<bool> keep(idx.size(), false);
    auto bh_on = [&](const vector<std::size_t>& js) {
        vector<double> e;
        for (std::size_t j : js) e.push_back(ecs::e_value(*hs.find(hid(idx[j]))));
        const auto rej = ecs::bh_reject_e(e, 0.1);
        for (std::size_t q = 0; q < js.size(); ++q) keep[js[q]] = rej[q];
    };
    vector<std::size_t> all(idx.size());
    std::iota(all.begin(), all.end(), 0);
    if (batch == B0_ALL) bh_on(all);
    else if (batch == B1_SUPPORTED) {  // the library's form (bh_reject_e_among)
        vector<double> e;
        vector<bool> sel;
        for (std::size_t j : all) {
            const auto* h = hs.find(hid(idx[j]));
            e.push_back(ecs::e_value(*h)), sel.push_back(h->state == ecs::HypState::Supported);
        }
        keep = ecs::bh_reject_e_among(e, sel, 0.1);
    } else if (batch == B2_RELATION) {
        std::map<string, vector<std::size_t>> by_rel;
        for (std::size_t j : all) by_rel[it[idx[j]].r].push_back(j);
        for (const auto& [r, js] : by_rel) bh_on(js);
    } else
        std::fill(keep.begin(), keep.end(), true);
    LoopStats L;
    for (std::size_t j = 0; j < idx.size(); ++j) {
        const Item& x = it[idx[j]];
        const string id = hid(idx[j]);
        if (hs.find(id)->state == ecs::HypState::Supported) {
            ++L.by_decide;
            if (keep[j]) ++L.kept;
            else hs.transition(id, ecs::HypState::Queued, "batch control: not kept");
        }
        if (x.label == 0 && x.wd_true) continue;
        const auto* h = hs.find(id);
        if (h->state == ecs::HypState::Supported) (x.label ? L.sup_t : L.sup_f) += 1, L.sup_cat[x.cat] += 1;
        if (h->state == ecs::HypState::Refuted) (x.label ? L.ref_t : L.ref_f) += 1;
        L.brier_post += (h->posterior - x.label) * (h->posterior - x.label), L.brier_prior += (x.p0 - x.label) * (x.p0 - x.label), L.n += 1;
    }
    L.entries = log.entries().size(), L.log_ok = log.verify();
    if (!log_path.empty()) {
        std::ofstream lf(log_path);
        lf << log.serialize();
    }
    return L;
}

string loop_line(const string& tag, const string& cfg, const LoopStats& L) {
    char b[768];
    std::snprintf(b, sizeof b,
                  "%s %-30s supported %.0f precision %.3f [%.3f, %.3f] | refuted %.0f precision %.3f [%.3f, %.3f] | decided %.3f of %.0f | "
                  "by decide %zu kept %zu | Brier posterior %.4f prior %.4f | log %zu %s",
                  tag.c_str(), cfg.c_str(), L.sup_n(), L.sup_prec(), ecs::wilson_lower(L.sup_t, L.sup_n()), ecs::wilson_upper(L.sup_t, L.sup_n()), L.ref_n(),
                  L.ref_prec(), ecs::wilson_lower(L.ref_f, L.ref_n()), ecs::wilson_upper(L.ref_f, L.ref_n()), L.decided(), L.n, L.by_decide, L.kept,
                  L.brier_post / std::max(1.0, L.n), L.brier_prior / std::max(1.0, L.n), L.entries, L.log_ok ? "ok" : "BROKEN");
    return b;
}

// ---- S7c (the S7 report's "S7c"): batch rules that state their guarantee, read on each hypothesis's loop result.
struct LoopRes {
    double post = 0, e = 1, pnull = 1, p0 = 0;
    bool sup = false, ref = false, counted = true;
    int label = 0, fold = 0;
    string subj;
};
enum S7cRule { K1, K2, K3, K4, KREF, N_K };
const char* const kKName[N_K] = {"K1_e-BH", "K2_prior-weighted_e-BH", "K3_conformal_selection", "K4_null-model_BH", "ref_S7b_among_supported"};

vector<ecs::HypTest> s7c_plan(const vector<LoopKind>& kinds) {
    vector<ecs::HypTest> plan;
    for (const LoopKind& K : kinds) {
        ecs::HypTest t;
        t.kind = K.name, t.source = K.source;
        t.predicted = "positive", t.refuted_if = "negative";
        t.disconfirmation = K.disconfirmation, t.independent = K.independent;
        t.sensitivity = K.s, t.specificity = K.c;
        t.likelihoods = K.lik;
        plan.push_back(t);
    }
    return plan;
}

using PnullMemo = std::map<std::pair<double, double>, double>;
double pnull_of(PnullMemo& memo, const vector<ecs::HypTest>& plan, double p0, double e) {
    const auto key = std::make_pair(p0, e);
    auto it = memo.find(key);
    if (it == memo.end()) it = memo.emplace(key, ecs::null_model_pvalue(plan, p0, e)).first;
    return it->second;
}

// loop2's loop with three-way likelihoods and no batch step, keeping each hypothesis's posterior, e-value, decide's state
// and the null-model p-value of its e-value. Folds: bit 13 of FNV-1a(subject).
// sup_src (S7d): when given, each hypothesis's supporting independent sources, in idx order.
vector<LoopRes> loop_res(const vector<Item>& it, const vector<std::size_t>& idx, const vector<LoopKind>& kinds, const string& log_path = "",
                         vector<vector<string>>* sup_src = nullptr) {
    ecs::HypothesisStore hs;
    ecs::PreregLog log;
    const vector<ecs::HypTest> plan = s7c_plan(kinds);
    PnullMemo memo;
    vector<LoopRes> out;
    out.reserve(idx.size());
    for (std::size_t i : idx) {
        const Item& x = it[i];
        ecs::Hypothesis h;
        h.id = "S7C-" + std::to_string(i);
        h.key = x.r + '|' + x.s + '|' + x.o;
        h.statement = x.s + " " + x.r + " " + x.o;
        h.generator = "planted-store";
        h.tests = plan;
        hs.propose(h);
        hs.score(h.id, x.p0);
        hs.transition(h.id, ecs::HypState::Queued, "scored");
        if (log.register_plan(*hs.find(h.id)))
            for (std::size_t k = 0; k < kinds.size(); ++k) {
                if (hs.find(h.id)->state != ecs::HypState::Queued) break;
                hs.transition(h.id, ecs::HypState::UnderTest, "next test");
                hs.record_outcome(h.id, k, kinds[k].f(x), &log);
            }
        const ecs::Hypothesis* hh = hs.find(h.id);
        LoopRes r;
        r.post = hh->posterior, r.e = ecs::e_value(*hh), r.p0 = x.p0;
        r.sup = hh->state == ecs::HypState::Supported, r.ref = hh->state == ecs::HypState::Refuted;
        r.label = x.label, r.counted = !(x.label == 0 && x.wd_true), r.fold = static_cast<int>((fnv(x.s) >> 13) & 1), r.subj = x.s;
        r.pnull = pnull_of(memo, plan, x.p0, r.e);
        out.push_back(r);
        if (sup_src) sup_src->push_back(ecs::supporting_independent_sources(*hh));
    }
    if (!log_path.empty()) {
        std::ofstream lf(log_path);
        lf << log.serialize();
    }
    return out;
}

// A rule's promotions over batch b (indices into res). K3 calibrates on cal (indices into cres). gated (the amendment):
// K1, K2 and K4 see e = 0 or p = 1 where decide does not support; ungated is the registered R-and-S form.
vector<bool> s7c_apply(int rule, const vector<LoopRes>& res, const vector<std::size_t>& b, const vector<LoopRes>& cres,
                       const vector<std::size_t>& cal, bool gated = true) {
    const std::size_t m = b.size();
    vector<bool> sup(m), out(m, false), rej;
    for (std::size_t j = 0; j < m; ++j) sup[j] = res[b[j]].sup;
    if (rule == K1 || rule == K2) {
        vector<double> e(m), w(m);
        for (std::size_t j = 0; j < m; ++j) {
            const LoopRes& r = res[b[j]];
            e[j] = gated && !r.sup ? 0.0 : r.e;
            w[j] = r.p0 / std::max(1e-9, 1.0 - r.p0);
        }
        rej = rule == K1 ? ecs::bh_reject_e(e, 0.1) : ecs::weighted_e_bh(e, w, 0.1);
    } else if (rule == K4) {
        vector<double> p(m);
        for (std::size_t j = 0; j < m; ++j) p[j] = gated && !res[b[j]].sup ? 1.0 : res[b[j]].pnull;
        rej = ecs::benjamini_hochberg(p, 0.1).reject;
    } else if (rule == K3) {
        vector<double> cs, bs;
        vector<bool> ct;
        for (std::size_t i : cal) cs.push_back(cres[i].sup ? cres[i].post : -1.0), ct.push_back(cres[i].label == 1);
        for (std::size_t j = 0; j < m; ++j) bs.push_back(sup[j] ? res[b[j]].post : -1.0);
        rej = ecs::conformal_select(cs, ct, bs, 0.1);
    } else {
        vector<double> e(m);
        for (std::size_t j = 0; j < m; ++j) e[j] = res[b[j]].e;
        rej = ecs::bh_reject_e_among(e, sup, 0.1);
    }
    for (std::size_t j = 0; j < m; ++j) out[j] = rej[j] && sup[j];
    return out;
}

// A rule over a set S: K1, K2, K4 and the reference on S as one batch; K3 on S's two bit-13 folds, each calibrated on the other.
vector<bool> s7c_run(int rule, const vector<LoopRes>& res, const vector<std::size_t>& S, bool gated = true) {
    if (rule != K3) return s7c_apply(rule, res, S, res, {}, gated);
    vector<bool> out(S.size(), false);
    for (int f = 0; f < 2; ++f) {
        vector<std::size_t> b, c, pos;
        for (std::size_t j = 0; j < S.size(); ++j)
            if (res[S[j]].fold == f) b.push_back(S[j]), pos.push_back(j);
            else c.push_back(S[j]);
        const auto sel = s7c_apply(K3, res, b, res, c);
        for (std::size_t q = 0; q < b.size(); ++q) out[pos[q]] = sel[q];
    }
    return out;
}

struct KStats {
    double pt = 0, pf = 0, rt = 0, rf = 0, bt = 0, bf = 0, n = 0;  // promoted (Known), refuted, Believed (supported, not promoted)
    double prom() const { return pt + pf; }
    double prec() const { return prom() > 0 ? pt / prom() : 0.0; }
    double fdp() const { return prom() > 0 ? pf / prom() : 0.0; }
    double decided() const { return n > 0 ? (prom() + rt + rf) / n : 0.0; }
};
KStats s7c_stats(const vector<LoopRes>& res, const vector<std::size_t>& S, const vector<bool>& prom) {
    KStats k;
    for (std::size_t j = 0; j < S.size(); ++j) {
        const LoopRes& r = res[S[j]];
        if (!r.counted) continue;
        k.n += 1;
        if (prom[j]) (r.label ? k.pt : k.pf) += 1;
        else if (r.sup) (r.label ? k.bt : k.bf) += 1;
        if (r.ref) (r.label ? k.rt : k.rf) += 1;
    }
    return k;
}
string s7c_line(const string& tag, const string& name, const KStats& k) {
    char b[768];
    const double bn = k.bt + k.bf;
    std::snprintf(b, sizeof b,
                  "S7C %s %-30s promoted %.0f precision %.3f [%.3f, %.3f] | refuted %.0f precision %.3f | Known-decided %.3f of %.0f | "
                  "Believed %.0f precision %.3f",
                  tag.c_str(), name.c_str(), k.prom(), k.prec(), ecs::wilson_lower(k.pt, k.prom()), ecs::wilson_upper(k.pt, k.prom()), k.rt + k.rf,
                  k.rt + k.rf > 0 ? k.rf / (k.rt + k.rf) : 0.0, k.decided(), k.n, bn, bn > 0 ? k.bt / bn : 0.0);
    return b;
}

// Realised FDR (mean false share among promotions, 0 when none) with its standard error over reps.
struct FdrAcc {
    double s[N_K] = {}, s2[N_K] = {}, prom[N_K] = {};
    int reps = 0;
    void add(int k, double fdp, double np) { s[k] += fdp, s2[k] += fdp * fdp, prom[k] += np; }
    double fdr(int k) const { return reps ? s[k] / reps : 0.0; }
    double se(int k) const {
        if (reps < 2) return 0.0;
        const double m = fdr(k);
        return std::sqrt(std::max(0.0, (s2[k] / reps - m * m) / (reps - 1)));
    }
};

// Model simulation: priors drawn from pool, truths drawn with the prior (mixed) or all false (planted nulls), outcomes
// from the three-way likelihoods; the loop as decide runs it. Batch and calibration sets drawn the same way.
LoopRes s7c_sim_one(std::mt19937_64& g, double p0, bool truth, const vector<ecs::HypTest>& plan, PnullMemo& memo) {
    const ecs::HypothesisConfig cfg;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    LoopRes r;
    r.p0 = p0, r.label = truth;
    double post = p0, e = 1.0, moved = 0.0, last = p0;
    bool indep = false;
    int done = 0;
    for (const ecs::HypTest& t : plan) {
        const double* pr = truth ? t.likelihoods.t : t.likelihoods.f;
        const double u = U(g) * (pr[0] + pr[1] + pr[2]);
        const int o = u < pr[0] ? 0 : u < pr[0] + pr[1] ? 1 : 2;
        const ecs::TestOutcome out = o == 0 ? kPos : o == 1 ? kNeg : kInc;
        const double bf = ecs::bayes_factor(out, t);
        post = ecs::update_posterior(post, bf), e *= bf, indep = indep || (t.independent && o == 0), ++done;  // a positive one (S8d)
        moved += std::fabs(post - last), last = post;
        if (post > cfg.supported_p && indep) { r.sup = true; break; }
        if (post < cfg.refuted_p) { r.ref = true; break; }
        if (done >= cfg.irreducible_k && moved < cfg.irreducible_move) break;
    }
    r.post = post, r.e = e, r.pnull = pnull_of(memo, plan, p0, e);
    return r;
}
FdrAcc s7c_simulate(const vector<double>& pool, const vector<LoopKind>& kinds, std::size_t m, int reps, bool planted_nulls, std::uint64_t seed) {
    const vector<ecs::HypTest> plan = s7c_plan(kinds);
    PnullMemo memo;
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    FdrAcc acc;
    vector<std::size_t> b(m), c(m);
    std::iota(b.begin(), b.end(), 0);
    std::iota(c.begin(), c.end(), m);
    for (int rep = 0; rep < reps; ++rep) {
        vector<LoopRes> res(2 * m);
        for (std::size_t i = 0; i < 2 * m; ++i) {
            const double p0 = pool[g() % pool.size()];
            const bool truth = planted_nulls ? false : U(g) < p0;
            res[i] = s7c_sim_one(g, p0, truth, plan, memo);
        }
        for (int k = 0; k < N_K; ++k) {
            const auto pr = s7c_apply(k, res, b, res, c);
            const KStats st = s7c_stats(res, b, pr);
            acc.add(k, st.fdp(), st.prom());
        }
        ++acc.reps;
    }
    return acc;
}

// Real-data resampling: the set's subjects split at random into a calibration half and a batch half, reps times.
FdrAcc s7c_resample(const vector<LoopRes>& res, int reps, std::uint64_t seed) {
    std::map<string, vector<std::size_t>> by_s;
    for (std::size_t i = 0; i < res.size(); ++i) by_s[res[i].subj].push_back(i);
    vector<const vector<std::size_t>*> groups;
    for (const auto& kv : by_s) groups.push_back(&kv.second);
    std::mt19937_64 g(seed);
    FdrAcc acc;
    for (int rep = 0; rep < reps; ++rep) {
        for (std::size_t i = groups.size(); i > 1; --i) std::swap(groups[i - 1], groups[g() % i]);
        vector<std::size_t> c, b;
        for (std::size_t q = 0; q < groups.size(); ++q)
            for (std::size_t i : *groups[q]) (q < groups.size() / 2 ? c : b).push_back(i);
        for (int k = 0; k < N_K; ++k) {
            const auto pr = s7c_apply(k, res, b, res, c);
            const KStats st = s7c_stats(res, b, pr);
            acc.add(k, st.fdp(), st.prom());
        }
        ++acc.reps;
    }
    return acc;
}
string fdr_line(const string& tag, const FdrAcc& a) {
    string s = "S7C " + tag + " realised FDR (reps " + std::to_string(a.reps) + "):";
    char b[200];
    for (int k = 0; k < N_K; ++k) {
        std::snprintf(b, sizeof b, " %s %.4f (se %.4f, promoted %.1f)", kKName[k], a.fdr(k), a.se(k), a.reps ? a.prom[k] / a.reps : 0.0);
        s += b;
    }
    return s;
}
struct S7cS15 {
    bool have = false;
    KStats st[N_K];
    FdrAcc rs;
};

// Does a store object equal an S15 value? Years for year relations, compatible dates for date relations, the normalised
// key otherwise.
bool same_value(int r, const string& store_o, const string& v) {
    if (is_datey(r)) {
        const Date a = parse_date_str(store_o), d = parse_date_str(v);
        return kRel[r].type == T_YEAR ? a.ok() && d.ok() && a.y == d.y : date_compat(a, d);
    }
    const string k = norm_ent(v);
    return !k.empty() && norm_ent(store_o) == k;
}

// S15's dev half as pairs (true row, false row) for pattern_learn's relations and enwik8 subjects.
struct S15Pair {
    string s, r, vt, vf;
    char cat = ' ';
};
vector<S15Pair> load_s15(const string& path, const std::unordered_map<string, int>& page_of) {
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    std::map<string, vector<string>> rows;  // id -> columns
    vector<string> order;
    while (std::getline(f, l)) {
        auto c = split(l, '\t');
        if (c.size() < 17) continue;
        order.push_back(c[0]);
        rows[c[0]] = std::move(c);
    }
    vector<S15Pair> out;
    for (const string& id : order) {
        const auto& t = rows[id];
        if (t[15] != "1" || rel_index(t[5]) < 0 || !page_of.count(t[1])) continue;
        const auto fit = rows.find(t[16]);
        if (fit == rows.end() || fit->second[15] != "0") continue;
        S15Pair p;
        p.s = t[1], p.r = t[5], p.vt = t[7], p.vf = fit->second[7], p.cat = t[11].empty() ? ' ' : t[11][0];
        out.push_back(p);
    }
    return out;
}

int run(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: pattern_learn --test-kinds ENWIK8 SEEDS.tsv WD_VALUES.tsv PATTERNS.tsv OUT_PREFIX [--read-test]\n"
                     "       [--phi-wd PHI.tsv] [--s15 TEMPORAL_HOLDOUT_DEV.tsv [--read-confirm]]   (S7b)\n"
                     "       [--s7c [--read-c2]]   (S7c: batch rules with a guarantee; C2 read once)\n"
                     "       [--s6c [--read-s6c-fresh]]   (S6c: T7, G12's closed form as a test; the loop on S6c's fresh set read once)\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[6];
    bool read_test = false, read_confirm = false;
    string phi_wd_path, s15_path;  // S7b
    bool s7c = false, read_c2 = false;  // S7c
    bool s6c_on = false, read_s6c = false;  // S6c
    for (int i = 7; i < argc; ++i) {
        if (string(argv[i]) == "--read-test") read_test = true;
        else if (string(argv[i]) == "--read-confirm") read_confirm = true;
        else if (string(argv[i]) == "--phi-wd" && i + 1 < argc) phi_wd_path = argv[++i];
        else if (string(argv[i]) == "--s15" && i + 1 < argc) s15_path = argv[++i];
        else if (string(argv[i]) == "--s7c") s7c = true;
        else if (string(argv[i]) == "--read-c2") read_c2 = true;
        else if (string(argv[i]) == "--s6c") s6c_on = true;
        else if (string(argv[i]) == "--read-s6c-fresh") read_s6c = true;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    char b[1024];
    // ---- the article text and its type lexicons (main's pass 1, with Wikidata as in the run that kept the patterns)
    string data;
    if (!read_file(argv[2], data)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    g_dump = &data;
    long total = 0, redirects = 0;
    const vector<PageRef> pages = list_pages(data, total, redirects);
    std::unordered_map<string, int> page_of;
    for (std::size_t i = 0; i < pages.size(); ++i) page_of[pages[i].title] = static_cast<int>(i);
    for (const PageRef& pg : pages) {
        PageFlags fl;
        scan_page(data, pg, fl);
        const string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    {
        long ns = 0, ng = 0;
        if (!load_wikidata(argv[4], false, page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    }
    for (const Seed& sd : g_seeds) {
        const RT ty = kRel[sd.rel].type;
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    // ---- T4's patterns: the kept rows of the run's patterns.tsv
    {
        std::ifstream f(argv[5]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 11 || c[10] != "1" || rel_index(c[0]) < 0) continue;
            Pat p;
            p.rel = rel_index(c[0]);
            p.key = c[1];
            p.family = family_of(p.key);
            p.kept = true;
            g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
            g_pats.push_back(p);
        }
    }
    // ---- the store (S6's), the Wikidata values, and which (s, r) slots the store took from Wikidata
    std::map<string, vector<std::pair<string, string>>> by_subj;
    std::set<string> seen, wd_src;
    {
        std::ifstream f(argv[3]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            if (c[3] == "wikidata") wd_src.insert(c[0] + '\x1f' + c[1]);
            if (seen.insert(c[0] + '\x1f' + c[1] + '\x1f' + c[2]).second) by_subj[c[0]].push_back({c[1], c[2]});
        }
    }
    ecs::TripleStore full;
    for (auto& [s, fs] : by_subj) {
        std::sort(fs.begin(), fs.end());
        for (const auto& [r, o] : fs) full.add(s, r, o);
    }
    std::unordered_map<string, std::set<string>> wd6;  // S6's form: (s, r) -> lowered surfaces (S6's sets need it)
    {
        std::ifstream f(argv[4]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 3) continue;
            for (const auto& x : split(c[2], '|')) wd6[c[0] + '\x1f' + c[1]].insert(lower(x));
        }
    }
    const WdObjs wd = load_wd_objs(argv[4]);
    // ---- S6's planted and fresh sets, rebuilt with S6's code and seed: S6's hold-out of each subject
    std::mt19937_64 rng6(20261028);
    vector<string> subj;
    for (const auto& [s, fs] : by_subj)
        if (fs.size() >= 2) subj.push_back(s);
    for (std::size_t i = subj.size(); i > 1; --i) std::swap(subj[i - 1], subj[rng6() % i]);
    std::map<string, std::pair<string, string>> h6;
    {
        s6::PlantedSet A, F;
        const std::size_t next = s6::build_set(A, subj, 0, 2000, by_subj, full, wd6, rng6);
        s6::build_set(F, subj, next, std::numeric_limits<std::size_t>::max(), by_subj, full, wd6, rng6);
        for (const auto* ps : {&A, &F})
            for (const auto& x : ps->it)
                if (x.label) h6[x.s] = {x.r, x.o};
        std::snprintf(b, sizeof b, "S7 S6 rebuilt: planted pairs %zu, fresh pairs %zu, subjects with an S6 hold-out %zu", A.it.size() / 2,
                      F.it.size() / 2, h6.size());
        std::cout << b << "\n";
    }
    // ---- S7's hold-outs: U (the one-fact subjects), C (a second fact of S6's subjects, in S6's order)
    struct Held {
        string s, r, o;
        char set;
    };
    vector<Held> held;
    std::set<string> held_keys;
    std::size_t u_subjects = 0, u_unseen = 0, c_none = 0;
    for (const auto& [s, fs] : by_subj) {
        if (fs.size() != 1) continue;
        ++u_subjects;
        if (full.count(fs[0].first, fs[0].second) < 2) { ++u_unseen; continue; }
        held.push_back({s, fs[0].first, fs[0].second, 'U'});
    }
    std::mt19937_64 rng7(20261029);
    for (const string& s : subj) {
        const auto it6 = h6.find(s);
        if (it6 == h6.end()) continue;
        auto fs = by_subj[s];
        for (std::size_t i = fs.size(); i > 1; --i) std::swap(fs[i - 1], fs[rng7() % i]);
        bool got = false;
        for (const auto& [r, o] : fs)
            if (!(r == it6->second.first && o == it6->second.second) && full.count(r, o) >= 2) {
                held.push_back({s, r, o, 'C'});
                got = true;
                break;
            }
        if (!got) ++c_none;
    }
    for (const auto& h : held) held_keys.insert(h.s + '\x1f' + h.r + '\x1f' + h.o);
    ecs::TripleStore st;  // the reduced store: every test reads it
    std::unordered_map<string, vector<string>> objects_of_rel;
    for (const auto& [s, fs] : by_subj)
        for (const auto& [r, o] : fs)
            if (!held_keys.count(s + '\x1f' + r + '\x1f' + o)) st.add(s, r, o), objects_of_rel[r].push_back(o);
    // ---- false partners (S6's rule, no Wikidata filter)
    std::mt19937_64 rngf(20261030);
    vector<Item> it;
    std::size_t gone = 0, nofalse = 0;
    for (const auto& h : held) {
        if (st.count(h.r, h.o) == 0) { ++gone; continue; }
        const auto& pool = objects_of_rel[h.r];
        const int ri = rel_index(h.r);
        const Date hd = is_datey(ri) ? parse_date_str(h.o) : Date{};
        string neg;
        for (int t = 0; t < 100 && neg.empty(); ++t) {
            const string& c = pool[rngf() % pool.size()];
            if (c == h.o || c == h.s || full.has(h.s, h.r, c)) continue;
            if (is_datey(ri)) {
                const Date cd = parse_date_str(c);
                if (cd.ok() && hd.ok() && cd.y == hd.y) continue;
            }
            neg = c;
        }
        if (neg.empty()) { ++nofalse; continue; }
        const auto wit = wd.find(h.s + '\x1f' + h.r);
        Item a, x;
        a.s = x.s = h.s, a.r = x.r = h.r, a.o = h.o, x.o = neg, a.label = 1, x.label = 0;
        a.pair = x.pair = static_cast<int>(it.size() / 2);
        a.set = x.set = h.set;
        a.dev = x.dev = ((fnv(h.s) >> 11) & 1) == 0;
        a.t5_ok = x.t5_ok = !wd_src.count(h.s + '\x1f' + h.r);
        x.wd_true = wit != wd.end() && wd_match(ri, neg, wit->second);
        it.push_back(a), it.push_back(x);
    }
    std::size_t nu = 0, nc = 0, nd = 0, nwd = 0;
    for (std::size_t i = 0; i < it.size(); i += 2) (it[i].set == 'U' ? nu : nc)++, nd += it[i].dev, nwd += it[i + 1].wd_true;
    std::snprintf(b, sizeof b,
                  "S7 SET U: one-fact subjects %zu (no seen object %zu) | C: subjects with an S6 hold-out %zu (no second fact %zu) | pairs U %zu C %zu "
                  "(dropped: object gone %zu, no false partner %zu) | dev pairs %zu test %zu | false members Wikidata lists %zu | reduced store %zu of %zu",
                  u_subjects, u_unseen, h6.size(), c_none, nu, nc, gone, nofalse, nd, it.size() / 2 - nd, nwd, st.size(), full.size());
    std::cout << b << "\n";
    // ---- the tests
    const ecs::LinkPredictor lp(st, {});
    const double t_tests = now_s();
    for (std::size_t i = 0; i < it.size(); i += 2) {
        const int ri = rel_index(it[i].r);
        PageScan ps;
        const auto pit = page_of.find(it[i].s);
        const bool page = pit != page_of.end();
        if (page) ps = scan_rel(data, pages[static_cast<std::size_t>(pit->second)], ri);
        for (std::size_t j = i; j < i + 2; ++j) {
            Item& x = it[j];
            x.t1 = t1(st, x.s, x.r, x.o);
            x.p = lp.prob(x.s, x.r, x.o);
            x.aa = ecs::adamic_adar(st, x.s, x.o);
            x.dl_others = ecs::compression_gain_bits(lp, st, x.s, x.r, x.o) - std::log2(std::max(1e-12, x.p));
            for (int k = 0; k < 4; ++k) x.t6[k] = x.dl_others > kEps[k] ? kPos : x.dl_others < -kEps[k] ? kNeg : kInc;
            const auto wit = wd.find(x.s + '\x1f' + x.r);
            x.t5 = !x.t5_ok || wit == wd.end() ? kInc : wd_match(ri, x.o, wit->second) ? kPos : kNeg;
            if (page) t4(ps, ri, x.o, x.t4);
        }
    }
    std::snprintf(b, sizeof b, "S7 tests ran on %zu hypotheses in %.1f s", it.size(), now_s() - t_tests);
    std::cout << b << "\n";
    // ---- dev: every variant, then the choices (mutual information on dev, both sets)
    using OF = std::function<TestOutcome(const Item&)>;
    auto report = [&](const char* tag, bool dev) {
        for (const char set : {'U', 'C', '*'}) {
            const string sn = set == '*' ? "all" : string(1, set);
            std::cout << line_of(tag, "T1_consistency_" + sn, tally(it, dev, set, false, [](const Item& x) { return x.t1; })) << "\n";
            for (int k = 0; k < 4; ++k)
                std::cout << line_of(tag, "T6_eps" + std::to_string(kEps[k]).substr(0, 3) + "_" + sn,
                                     tally(it, dev, set, false, [k](const Item& x) { return x.t6[k]; })) << "\n";
            std::cout << line_of(tag, "T5_wikidata_" + sn, tally(it, dev, set, true, [](const Item& x) { return x.t5; })) << "\n";
            std::cout << line_of(tag, "T5_wikidata_s6labels_" + sn, tally(it, dev, set, false, [](const Item& x) { return x.t5; })) << "\n";
            for (int k = 0; k < 2; ++k)
                std::cout << line_of(tag, string("T4_") + kT4Var[k] + "_" + sn, tally(it, dev, set, false, [k](const Item& x) { return x.t4[k]; })) << "\n";
        }
    };
    report("DEV", true);
    auto best = [&](int n, const std::function<TestOutcome(const Item&, int)>& f) {
        int bk = 0;
        double bmi = -1;
        for (int k = 0; k < n; ++k) {
            const double mi = tally(it, true, '*', false, [&](const Item& x) { return f(x, k); }).mutual_information_bits();
            if (mi > bmi + 1e-12) bmi = mi, bk = k;
        }
        return bk;
    };
    const int k6 = best(4, [](const Item& x, int k) { return x.t6[k]; });
    const int k4 = best(2, [](const Item& x, int k) { return x.t4[k]; });
    std::snprintf(b, sizeof b, "S7 CHOICE (dev, mutual information): T6 eps %.1f bits, T4 %s", kEps[k6], kT4Var[k4]);
    std::cout << b << "\n";
    struct Kind {
        const char* name;
        const char* source;
        bool disconfirmation, independent, all_false;
        double cost;
        OF f;
    };
    const vector<Kind> kinds = {
        {"T1 consistency", "store", true, false, false, 1, [](const Item& x) { return x.t1; }},
        {"T6 compression", "store", false, false, false, 3, [k6](const Item& x) { return x.t6[k6]; }},
        {"T5 Wikidata", "wikidata", true, true, true, 5, [](const Item& x) { return x.t5; }},
        {"T4 corpus", "corpus", true, true, false, 10, [k4](const Item& x) { return x.t4[k4]; }}};
    // The earn-its-place rule (dev, both sets): applicable to >= 5% and J's interval above 0. T3 by the same rule.
    vector<double> s_dev, c_dev, s_add1, c_add1;  // the loop's s and c: dev point estimates (registered) and add-one (amended)
    vector<bool> kept;
    for (const Kind& K : kinds) {
        const auto t = tally(it, true, '*', K.all_false, K.f);
        const bool keep = t.applicable() >= 0.05 && t.youden_lo() > 0.0;
        s_dev.push_back(t.sensitivity()), c_dev.push_back(t.specificity()), kept.push_back(keep);
        s_add1.push_back((t.t_pos + 1) / (t.t_pos + t.t_neg + 2)), c_add1.push_back((t.f_neg + 1) / (t.f_pos + t.f_neg + 2));
        std::snprintf(b, sizeof b, "S7 EARN %-15s applicable %.3f J %+.3f [%+.3f, %+.3f] -> %s", K.name, t.applicable(), t.youden(), t.youden_lo(),
                      t.youden_hi(), keep ? "kept" : "cut");
        std::cout << b << "\n";
    }
    {
        std::size_t np = 0;
        const auto t = t3_planted(st, true, &np);
        const bool keep = t.applicable() >= 0.05 && t.youden_lo() > 0.0;
        std::cout << line_of("DEV", "T3_planted_rules", t) << " | rules proposed " << np << "\n";
        std::snprintf(b, sizeof b, "S7 EARN %-15s applicable %.3f J %+.3f [%+.3f, %+.3f] -> %s", "T3 held-out", t.applicable(), t.youden(), t.youden_lo(),
                      t.youden_hi(), keep ? "kept" : "cut");
        std::cout << b << "\n";
    }
    // ---- the prior: S6's P0 without the constraint, fitted on dev (z pools and isotonic)
    auto logit = [](double p) { p = std::min(1 - 1e-12, std::max(1e-12, p)); return std::log(p / (1 - p)); };
    vector<double> vl, va;
    for (const Item& x : it)
        if (x.dev) vl.push_back(logit(x.p)), va.push_back(std::log1p(x.aa));
    const auto zl = ecs::pool_stats(vl), za = ecs::pool_stats(va);
    auto raw = [&](const Item& x) { return ecs::zscore(logit(x.p), zl) + ecs::zscore(std::log1p(x.aa), za); };
    ecs::IsotonicCalibrator cal;
    {
        vector<double> p;
        vector<int> y;
        for (const Item& x : it)
            if (x.dev && !(x.label == 0 && x.wd_true)) p.push_back(raw(x)), y.push_back(x.label);
        cal.fit(p, y);
    }
    for (Item& x : it) x.p0 = std::min(0.98, std::max(0.02, cal.predict(raw(x))));
    // ---- the loop (7.2) on one split, one batch: pre-registration, cheapest first, Bayes factors, BH at q 0.1
    // Amended after the dev read, before the test read: a test with no error on dev has a point estimate of 1, so its BF
    // reaches bayes_factor's clamp (about 10^6). The add-one estimates are the loop's primary; the registered point
    // estimates are printed beside them.
    auto loop = [&](const string& tag, bool dev, bool add1, const string& log_path) {
        ecs::HypothesisStore hs;
        ecs::PreregLog log;
        vector<std::size_t> idx;
        for (std::size_t i = 0; i < it.size(); ++i)
            if (it[i].dev == dev) idx.push_back(i);
        std::size_t refused = 0;
        for (std::size_t i : idx) {
            Item& x = it[i];
            ecs::Hypothesis h;
            h.id = "S7-" + std::to_string(i);
            h.key = x.r + '|' + x.s + '|' + x.o;
            h.statement = x.s + " " + x.r + " " + x.o;
            h.generator = "planted-store";
            for (std::size_t k = 0; k < kinds.size(); ++k) {
                if (!kept[k]) continue;
                ecs::HypTest t;
                t.kind = kinds[k].name, t.source = kinds[k].source;
                t.predicted = "positive", t.refuted_if = "negative";
                t.disconfirmation = kinds[k].disconfirmation, t.independent = kinds[k].independent;
                t.sensitivity = add1 ? s_add1[k] : s_dev[k], t.specificity = add1 ? c_add1[k] : c_dev[k];
                h.tests.push_back(t);
            }
            hs.propose(h);
            hs.score(h.id, x.p0);
            hs.transition(h.id, ecs::HypState::Queued, "scored");
            if (!log.register_plan(*hs.find(h.id))) { ++refused; continue; }
            std::size_t ti = 0;
            for (std::size_t k = 0; k < kinds.size(); ++k) {
                if (!kept[k]) continue;
                if (hs.find(h.id)->state != ecs::HypState::Queued) break;
                hs.transition(h.id, ecs::HypState::UnderTest, "next test");
                if (!hs.record_outcome(h.id, ti++, kinds[k].f(x), &log)) ++refused;
            }
        }
        // BH over the batch's e-values; a supported hypothesis whose null BH does not reject goes back to the queue.
        vector<double> e;
        for (std::size_t i : idx) e.push_back(ecs::e_value(*hs.find("S7-" + std::to_string(i))));
        const auto rej = ecs::bh_reject_e(e, 0.1);
        std::size_t n_sup = 0, n_rej = 0, n_demoted = 0;
        for (std::size_t j = 0; j < idx.size(); ++j) n_rej += rej[j];
        std::map<string, std::array<double, 2>> by_state;  // state -> {true, false}
        double brier_post = 0, brier_prior = 0, n = 0;
        for (std::size_t j = 0; j < idx.size(); ++j) {
            Item& x = it[idx[j]];
            const string id = "S7-" + std::to_string(idx[j]);
            if (hs.find(id)->state == ecs::HypState::Supported) {
                ++n_sup;
                if (!rej[j]) ++n_demoted, hs.transition(id, ecs::HypState::Queued, "BH batch control: not rejected at q 0.1");
            }
            const auto* h = hs.find(id);
            x.post = h->posterior, x.state = ecs::hyp_state_name(h->state);
            if (x.label == 0 && x.wd_true) continue;  // S6's labels
            by_state[x.state][x.label ? 0 : 1] += 1;
            brier_post += (x.post - x.label) * (x.post - x.label), brier_prior += (x.p0 - x.label) * (x.p0 - x.label), n += 1;
        }
        std::snprintf(b, sizeof b, "%s LOOP BH q 0.1: nulls rejected %zu of %zu | supported by decide %zu, sent back to the queue %zu", tag.c_str(), n_rej,
                      idx.size(), n_sup, n_demoted);
        std::cout << b << "\n";
        for (const auto& [sname, c] : by_state) {
            std::snprintf(b, sizeof b, "%s LOOP %-12s true %.0f false %.0f", tag.c_str(), sname.c_str(), c[0], c[1]);
            std::cout << b << "\n";
        }
        const auto sup = by_state["supported"], ref = by_state["refuted"];
        std::snprintf(b, sizeof b,
                      "%s LOOP supported precision %.3f [%.3f, %.3f] (n %.0f) | refuted precision %.3f [%.3f, %.3f] (n %.0f) | decided %.3f of %.0f | "
                      "Brier posterior %.4f prior %.4f | plans refused %zu | log entries %zu verify %s",
                      tag.c_str(), sup[0] + sup[1] > 0 ? sup[0] / (sup[0] + sup[1]) : 0.0, ecs::wilson_lower(sup[0], sup[0] + sup[1]),
                      ecs::wilson_upper(sup[0], sup[0] + sup[1]), sup[0] + sup[1], ref[0] + ref[1] > 0 ? ref[1] / (ref[0] + ref[1]) : 0.0,
                      ecs::wilson_lower(ref[1], ref[0] + ref[1]), ecs::wilson_upper(ref[1], ref[0] + ref[1]), ref[0] + ref[1],
                      n > 0 ? (sup[0] + sup[1] + ref[0] + ref[1]) / n : 0.0, n, brier_post / std::max(1.0, n), brier_prior / std::max(1.0, n), refused,
                      log.entries().size(), log.verify() ? "ok" : "BROKEN");
        std::cout << b << "\n";
        std::ofstream lf(log_path);
        lf << log.serialize();
    };
    for (int k = 0; k < static_cast<int>(kinds.size()); ++k) {
        std::snprintf(b, sizeof b, "S7 LOOP_INPUT %-15s %s s %.4f c %.4f (add-one s %.4f c %.4f)", kinds[static_cast<std::size_t>(k)].name,
                      kept[static_cast<std::size_t>(k)] ? "in the plan" : "cut", s_dev[static_cast<std::size_t>(k)], c_dev[static_cast<std::size_t>(k)],
                      s_add1[static_cast<std::size_t>(k)], c_add1[static_cast<std::size_t>(k)]);
        std::cout << b << "\n";
    }
    loop("DEV[add1]", true, true, out + ".dev.prereg.tsv");
    loop("DEV[point]", true, false, out + ".dev_point.prereg.tsv");
    if (read_test) {
        report("TEST", false);
        std::size_t np = 0;
        std::cout << line_of("TEST", "T3_planted_rules", t3_planted(st, false, &np)) << " | rules proposed " << np << "\n";
        loop("TEST[point]", false, false, out + ".test_point.prereg.tsv");
        loop("TEST[add1]", false, true, out + ".test.prereg.tsv");
    }

    // ---- S7b (the S7 report's "S7b"): phi for T1, three-way likelihoods and batch control, chosen on dev; then S15's dev
    // half: the choose half always, the confirm half with --read-confirm (read once). S7's lines above are unchanged.
    {
        std::array<double, 64> nphi{};
        const PhiTable ph[N_PHI] = {phi_hand(), phi_from_file(phi_wd_path), phi_from_store(st, &nphi)};
        for (int r = 0; r < kNRel; ++r) {
            const std::size_t u = static_cast<std::size_t>(r);
            std::snprintf(b, sizeof b, "S7B PHI %-13s hand %.3f wikidata %.3f store %.3f (subjects %.0f) | functional: hand %d wikidata %d store %d",
                          kRel[r].name, ph[0][u], ph[1][u], ph[2][u], nphi[u], ph[0][u] >= 0.9, ph[1][u] >= 0.9, ph[2][u] >= 0.9);
            std::cout << b << "\n";
        }
        std::size_t mism = 0;
        for (Item& x : it) {
            for (int k = 0; k < N_PHI; ++k) x.t1v[k] = t1_phi(st, x.s, x.r, x.o, ph[k]);
            mism += x.t1v[PHI_HAND] != x.t1;
        }
        std::snprintf(b, sizeof b, "S7B T1 under phi_hand equals S7's T1: mismatches %zu of %zu", mism, it.size());
        std::cout << b << "\n";
        int kphi = PHI_HAND;
        double bmi = -1;
        for (int k = 0; k < N_PHI; ++k) {
            for (const char set : {'U', 'C', '*'})
                std::cout << line_of("S7B DEV", string("T1_phi_") + kPhiName[k] + "_" + (set == '*' ? string("all") : string(1, set)),
                                     tally(it, true, set, false, [k](const Item& x) { return x.t1v[k]; })) << "\n";
            const double mi = tally(it, true, '*', false, [k](const Item& x) { return x.t1v[k]; }).mutual_information_bits();
            if (mi > bmi + 1e-12) bmi = mi, kphi = k;
        }
        std::snprintf(b, sizeof b, "S7B CHOICE phi %s (dev, T1's mutual information %.4f bits)", kPhiName[kphi], bmi);
        std::cout << b << "\n";
        // The loop's kinds with the chosen T1: two-way add-one s and c, three-way add-one likelihoods, all from dev.
        vector<LoopKind> lk;
        for (const Kind& K : kinds) {
            const OF f = string(K.name) == "T1 consistency" ? OF([kphi](const Item& x) { return x.t1v[kphi]; }) : K.f;
            const auto t = tally(it, true, '*', K.all_false, f);
            const ecs::OutcomeLikelihoods L = ecs::likelihoods_from(t);
            lk.push_back({K.name, K.source, K.disconfirmation, K.independent, f, (t.t_pos + 1) / (t.t_pos + t.t_neg + 2),
                          (t.f_neg + 1) / (t.f_pos + t.f_neg + 2), L});
            std::snprintf(b, sizeof b,
                          "S7B LIK %-15s two-way s %.4f c %.4f | three-way true pos %.4f neg %.4f inc %.4f, false pos %.4f neg %.4f inc %.4f | "
                          "BF pos %.3f neg %.4f inc %.4f",
                          K.name, lk.back().s, lk.back().c, L.t[0], L.t[1], L.t[2], L.f[0], L.f[1], L.f[2], L.t[0] / L.f[0], L.t[1] / L.f[1], L.t[2] / L.f[2]);
            std::cout << b << "\n";
        }
        vector<std::size_t> dev_idx;
        for (std::size_t i = 0; i < it.size(); ++i)
            if (it[i].dev) dev_idx.push_back(i);
        struct Cfg {
            bool three;
            Batch batch;
        };
        auto cfg_name = [](const Cfg& c) { return string(c.three ? "three-way " : "two-way ") + kBatchName[c.batch]; };
        // Ties (within 0.001 of decided share) go to the earlier: two-way before three-way, then B0, B2, B1.
        const Cfg order[6] = {{false, B0_ALL}, {false, B2_RELATION}, {false, B1_SUPPORTED}, {true, B0_ALL}, {true, B2_RELATION}, {true, B1_SUPPORTED}};
        int best = -1;
        double bshare = -1;
        for (int c = 0; c < 6; ++c) {
            const LoopStats L = loop2(it, dev_idx, lk, order[c].three, order[c].batch);
            const bool q = L.sup_n() >= 10 && L.sup_prec() >= 0.90 && (L.ref_n() == 0 || L.ref_prec() >= 0.80);
            std::cout << loop_line("S7B DEV", cfg_name(order[c]), L) << " | " << (q ? "qualifies" : "does not qualify") << "\n";
            if (q && L.decided() > bshare + 0.001) bshare = L.decided(), best = c;
        }
        for (const bool three : {false, true})
            std::cout << loop_line("S7B DEV", cfg_name({three, B_NONE}), loop2(it, dev_idx, lk, three, B_NONE)) << " | reference only\n";
        const Cfg chosen = best >= 0 ? order[best] : order[0];
        std::snprintf(b, sizeof b, "S7B CHOICE loop %s%s", cfg_name(chosen).c_str(), best >= 0 ? "" : " (none qualifies: S7's configuration stays)");
        std::cout << b << "\n";
        if (best >= 0) loop2(it, dev_idx, lk, chosen.three, chosen.batch, out + ".s7b_dev.prereg.tsv");
        // ---- S6c (the S6 report's "S6c"): T7, G12's closed form as a test. It runs only where T4 is inconclusive, and its
        // three-way likelihoods are measured on dev given T4 inconclusive, so P(T4, T7 | h) = P(T4 | h) P(T7 | T4 silent, h)
        // with no independence assumed between them. It is never independent evidence (I4). With --read-s6c-fresh, the
        // loop on S6c's fresh set (T1, T6, T4; T5 left out) with and without T7, read once. S7b's lines are unchanged.
        if (s6c_on) {
            const double t6c = now_s();
            s6c::Text T6c;
            s6c::build(data, pages, T6c);
            for (Item& x : it) x.abc = s6c::score(T6c, x.s, x.o);
            vector<double> sc;
            std::size_t ndev = 0, ninc = 0;
            for (const Item& x : it) {
                if (!x.dev || (!x.label && x.wd_true)) continue;
                ++ndev, ninc += x.t4[k4] == kInc;
                if (x.abc >= 0 && x.t4[k4] == kInc) sc.push_back(x.abc);
            }
            std::sort(sc.begin(), sc.end());
            auto t7_of = [](double abc, double tau) { return abc < 0 ? kInc : abc == 0 ? kNeg : abc >= tau ? kPos : kInc; };
            auto tally7 = [&](double tau) {
                ecs::TestTally t;
                for (const Item& x : it)
                    if (x.dev && !(!x.label && x.wd_true) && x.t4[k4] == kInc) t.add(x.label == 1, t7_of(x.abc, tau));
                return t;
            };
            std::snprintf(b, sizeof b, "S6C T7 text built and scored in %.1f s | dev hypotheses %zu, T4 inconclusive %zu (%.3f; T7 consulted) | closed form applicable there %zu",
                          now_s() - t6c, ndev, ninc, ndev ? static_cast<double>(ninc) / ndev : 0.0, sc.size());
            std::cout << b << "\n";
            const double kQ[5] = {0.5, 0.6, 0.7, 0.8, 0.9};
            double tau = 0, bmi7 = -1;
            for (double q : kQ) {
                const double tq = sc.empty() ? 1.0 : std::max(1e-9, sc[std::min(sc.size() - 1, static_cast<std::size_t>(q * static_cast<double>(sc.size())))]);
                const auto t = tally7(tq);
                std::snprintf(b, sizeof b, "T7_q%.1f_tau%.1f", q, tq);
                std::cout << line_of("S6C DEV", b, t) << "\n";
                if (t.mutual_information_bits() > bmi7 + 1e-12) bmi7 = t.mutual_information_bits(), tau = tq;
            }
            for (Item& x : it) x.t7 = t7_of(x.abc, tau);
            const ecs::TestTally t7d = tally7(tau);
            const ecs::OutcomeLikelihoods L7 = ecs::likelihoods_from(t7d);
            std::snprintf(b, sizeof b, "S6C CHOICE T7 tau %.2f (dev MI %.4f bits) | three-way given T4 inconclusive: true pos %.4f neg %.4f inc %.4f, false pos %.4f neg %.4f inc %.4f | "
                          "LR+ %.3f LR- %.3f LR_inc %.3f -> %s",
                          tau, bmi7, L7.t[0], L7.t[1], L7.t[2], L7.f[0], L7.f[1], L7.f[2], L7.t[0] / L7.f[0], L7.t[1] / L7.f[1], L7.t[2] / L7.f[2],
                          L7.t[0] / L7.f[0] >= 3.0 ? "LR+ >= 3 (rule 2's dev part holds)" : "LR+ < 3 (rule 2's dev part fails)");
            std::cout << b << "\n";
            // dependence on T1 and T6 within each label (dev, T4 inconclusive): odds ratio of T7 positive against each positive
            for (const int lab : {1, 0})
                for (int which = 0; which < 2; ++which) {
                    double c[2][2] = {{0, 0}, {0, 0}};
                    for (const Item& x : it) {
                        if (!x.dev || (!x.label && x.wd_true) || x.t4[k4] != kInc || x.label != lab) continue;
                        const bool o = (which == 0 ? x.t1v[kphi] : x.t6[k6]) == kPos;
                        c[o][x.t7 == kPos] += 1;
                    }
                    const double orr = ((c[1][1] + 0.5) * (c[0][0] + 0.5)) / ((c[1][0] + 0.5) * (c[0][1] + 0.5));
                    std::snprintf(b, sizeof b, "S6C DEV dependence %s hypotheses: T7 positive by %s positive: %.0f of %.0f, otherwise %.0f of %.0f | odds ratio %.3f",
                                  lab ? "true" : "false", which == 0 ? "T1" : "T6", c[1][1], c[1][0] + c[1][1], c[0][1], c[0][0] + c[0][1], orr);
                    std::cout << b << "\n";
                }
            const LoopKind k7{"T7 indirect links", "corpus links", false, false, [](const Item& x) { return x.t7; },
                              (t7d.t_pos + 1) / (t7d.t_pos + t7d.t_neg + 2), (t7d.f_neg + 1) / (t7d.f_pos + t7d.f_neg + 2), L7,
                              [k4](const Item& x) { return x.t4[k4] == kInc; }};
            vector<LoopKind> lk_n5, lk_n57;  // T5 left out (as on S15), then T7 last
            for (const LoopKind& K : lk)
                if (string(K.name) != "T5 Wikidata") lk_n5.push_back(K);
            lk_n57 = lk_n5;
            lk_n57.push_back(k7);
            vector<LoopKind> lk7 = lk;
            lk7.push_back(k7);
            for (const Batch bt : {B_NONE, chosen.batch}) {
                std::cout << loop_line("S6C DEV", string("S7b's kinds, ") + kBatchName[bt], loop2(it, dev_idx, lk, true, bt)) << "\n";
                std::cout << loop_line("S6C DEV", string("S7b's kinds + T7, ") + kBatchName[bt], loop2(it, dev_idx, lk7, true, bt)) << "\n";
                std::cout << loop_line("S6C DEV", string("no T5, ") + kBatchName[bt], loop2(it, dev_idx, lk_n5, true, bt)) << "\n";
                std::cout << loop_line("S6C DEV", string("no T5 + T7, ") + kBatchName[bt], loop2(it, dev_idx, lk_n57, true, bt)) << "\n";
            }
            // S6c's fresh set as loop items: every test reads its reduced store; the prior is S7's (dev z pools and fit).
            s6::PlantedSet G;
            s6::s6c_fresh(G, by_subj, full, wd6);
            std::uint64_t chk = 1469598103934665603ULL;
            for (const auto& x : G.it) chk = (chk ^ fnv(std::to_string(x.label) + '\x1f' + x.s + '\x1f' + x.r + '\x1f' + x.o)) * 1099511628211ULL;
            std::snprintf(b, sizeof b, "S6C FRESH SET pairs %zu | reduced store %zu | checksum %016llx", G.it.size() / 2, G.st.size(), static_cast<unsigned long long>(chk));
            std::cout << b << "\n";
            if (read_s6c) {
                vector<Item> itf;
                const ecs::LinkPredictor lpf(G.st, {});
                std::map<string, PageScan> scans;
                for (const auto& g : G.it) {
                    Item y;
                    y.s = g.s, y.r = g.r, y.o = g.o, y.label = g.label, y.pair = g.pair, y.set = 'F', y.dev = false;
                    const int ri = rel_index(y.r);
                    y.t1v[kphi] = t1_phi(G.st, y.s, y.r, y.o, ph[kphi]);
                    y.p = lpf.prob(y.s, y.r, y.o);
                    y.aa = ecs::adamic_adar(G.st, y.s, y.o);
                    y.dl_others = ecs::compression_gain_bits(lpf, G.st, y.s, y.r, y.o) - std::log2(std::max(1e-12, y.p));
                    for (int k = 0; k < 4; ++k) y.t6[k] = y.dl_others > kEps[k] ? kPos : y.dl_others < -kEps[k] ? kNeg : kInc;
                    y.t5 = kInc;
                    const auto pit = page_of.find(y.s);
                    if (pit != page_of.end()) {
                        const string sk = y.s + '\x1f' + y.r;
                        auto sit = scans.find(sk);
                        if (sit == scans.end()) sit = scans.emplace(sk, scan_rel(data, pages[static_cast<std::size_t>(pit->second)], ri)).first;
                        t4(sit->second, ri, y.o, y.t4);
                    }
                    y.p0 = std::min(0.98, std::max(0.02, cal.predict(raw(y))));
                    y.abc = s6c::score(T6c, y.s, y.o);
                    y.t7 = t7_of(y.abc, tau);
                    itf.push_back(y);
                }
                vector<std::size_t> fidx(itf.size());
                std::iota(fidx.begin(), fidx.end(), 0);
                std::size_t finc = 0;
                for (const Item& y : itf) finc += y.t4[k4] == kInc;
                std::snprintf(b, sizeof b, "S6C FRESH LOOP hypotheses %zu | T4 inconclusive (T7 consulted) %zu", itf.size(), finc);
                std::cout << b << "\n";
                std::cout << line_of("S6C FRESH", "T7_given_T4_inconclusive", [&] {
                    ecs::TestTally t;
                    for (const Item& y : itf)
                        if (y.t4[k4] == kInc) t.add(y.label == 1, y.t7);
                    return t;
                }()) << "\n";
                for (const Batch bt : {B_NONE, chosen.batch}) {
                    std::cout << loop_line("S6C FRESH", string("no T5, ") + kBatchName[bt], loop2(itf, fidx, lk_n5, true, bt, out + ".s6c_fresh_noT7_" + kBatchName[bt] + ".prereg.tsv")) << "\n";
                    std::cout << loop_line("S6C FRESH", string("no T5 + T7, ") + kBatchName[bt], loop2(itf, fidx, lk_n57, true, bt, out + ".s6c_fresh_T7_" + kBatchName[bt] + ".prereg.tsv")) << "\n";
                }
            }
        }
        S7cS15 s7c15;  // S7c: the S15 choose half's reads, for the choice below
        // ---- S15's dev half (T5 left out: S15's labels are current Wikidata)
        if (!s15_path.empty()) {
            const vector<S15Pair> sp = load_s15(s15_path, page_of);
            std::set<string> drop;  // the store's facts that are S15 dev true facts (both halves)
            for (const S15Pair& p : sp) {
                const int ri = rel_index(p.r);
                for (const auto& f : full.facts_of(p.s))
                    if (f.r == p.r && same_value(ri, f.o, p.vt)) drop.insert(p.s + '\x1f' + p.r + '\x1f' + f.o);
            }
            ecs::TripleStore st15;
            std::unordered_map<string, std::map<string, std::map<string, int>>> surf;  // r -> key -> store surface -> count
            for (const auto& [s, fs] : by_subj)
                for (const auto& [r, o] : fs)
                    if (!drop.count(s + '\x1f' + r + '\x1f' + o)) st15.add(s, r, o), surf[r][norm_ent(o)][o] += 1;
            auto form = [&](const string& r, const string& v) -> string {
                if (is_datey(rel_index(r))) return v;
                const auto ir = surf.find(r);
                if (ir == surf.end()) return v;
                const auto ik = ir->second.find(norm_ent(v));
                if (ik == ir->second.end()) return v;
                string bs;
                int bc = -1;
                for (const auto& [o, c] : ik->second)
                    if (c > bc) bc = c, bs = o;
                return bs;
            };
            const ecs::LinkPredictor lp15(st15, {});
            const PhiTable ph15 = kphi == PHI_STORE ? phi_from_store(st15) : ph[kphi];
            for (int r = 0; r < kNRel; ++r)
                if ((ph15[static_cast<std::size_t>(r)] >= 0.9) != (ph[kphi][static_cast<std::size_t>(r)] >= 0.9)) {
                    std::snprintf(b, sizeof b, "S7B S15 PHI %-13s dev table %.3f, re-estimated on the S15-reduced store %.3f", kRel[r].name,
                                  ph[kphi][static_cast<std::size_t>(r)], ph15[static_cast<std::size_t>(r)]);
                    std::cout << b << "\n";
                }
            vector<Item> it15;
            std::size_t same_surface = 0;
            for (const S15Pair& p : sp) {
                Item a, x;
                a.s = x.s = p.s, a.r = x.r = p.r;
                a.o = form(p.r, p.vt), x.o = form(p.r, p.vf);
                if (a.o == x.o) { ++same_surface; continue; }
                a.label = 1, x.label = 0;
                a.pair = x.pair = static_cast<int>(it15.size() / 2);
                a.set = x.set = 'S';
                a.dev = x.dev = ((fnv(p.s) >> 17) & 1) == 0;  // the choose half; the confirm half otherwise
                a.cat = x.cat = p.cat;
                a.reach = x.reach = st15.count(p.r, a.o) >= 1 && st15.count(p.r, x.o) >= 1;
                it15.push_back(a), it15.push_back(x);
            }
            std::map<string, PageScan> scans;
            for (Item& y : it15) {
                const int ri = rel_index(y.r);
                const string sk = y.s + '\x1f' + y.r;
                auto sit = scans.find(sk);
                if (sit == scans.end()) sit = scans.emplace(sk, scan_rel(data, pages[static_cast<std::size_t>(page_of.at(y.s))], ri)).first;
                y.t1v[kphi] = t1_phi(st15, y.s, y.r, y.o, ph[kphi]);  // amended: the dev table, as is
                y.t1_reg = t1_phi(st15, y.s, y.r, y.o, ph15);         // registered: re-estimated on the S15-reduced store
                y.p = lp15.prob(y.s, y.r, y.o);
                y.aa = ecs::adamic_adar(st15, y.s, y.o);
                y.dl_others = ecs::compression_gain_bits(lp15, st15, y.s, y.r, y.o) - std::log2(std::max(1e-12, y.p));
                for (int k = 0; k < 4; ++k) y.t6[k] = y.dl_others > kEps[k] ? kPos : y.dl_others < -kEps[k] ? kNeg : kInc;
                y.t5 = kInc;
                t4(sit->second, ri, y.o, y.t4);
                y.p0 = std::min(0.98, std::max(0.02, cal.predict(raw(y))));
            }
            std::size_t ch = 0, cf = 0, chr = 0, cfr = 0;
            for (std::size_t i = 0; i < it15.size(); i += 2) (it15[i].dev ? ch : cf)++, (it15[i].dev ? chr : cfr) += it15[i].reach;
            std::snprintf(b, sizeof b,
                          "S7B S15 pairs %zu (same surface dropped %zu) | store facts held out %zu, S15-reduced store %zu of %zu | phi %s | "
                          "choose half %zu pairs (reachable %zu) | confirm half %zu pairs (reachable %zu)",
                          it15.size() / 2, same_surface, drop.size(), st15.size(), full.size(), kPhiName[kphi], ch, chr, cf, cfr);
            std::cout << b << "\n";
            {  // the items of the halves read (the confirm half only with --read-confirm)
                std::ofstream f(out + ".s15_items.tsv");
                f << "pair\thalf\tcategory\treachable\tlabel\tsubject\trelation\tobject\tT1\tT6\tT4\tP0\n";
                for (const Item& y : it15) {
                    if (!y.dev && !read_confirm) continue;
                    f << y.pair << '\t' << (y.dev ? "choose" : "confirm") << '\t' << y.cat << '\t' << y.reach << '\t' << y.label << '\t' << y.s << '\t'
                      << y.r << '\t' << y.o << '\t' << ecs::test_outcome_name(y.t1v[kphi]) << '\t' << ecs::test_outcome_name(y.t6[k6]) << '\t'
                      << ecs::test_outcome_name(y.t4[k4]) << '\t' << y.p0 << '\n';
                }
            }
            vector<LoopKind> lk15, lk15r;  // amended (the dev phi table) and registered (phi re-estimated on S15)
            for (const LoopKind& K : lk)
                if (string(K.name) != "T5 Wikidata") {
                    lk15.push_back(K), lk15r.push_back(K);
                    if (string(K.name) == "T1 consistency") lk15r.back().f = [](const Item& x) { return x.t1_reg; };
                }
            for (const bool confirm : {false, true}) {
                if (confirm && !read_confirm) break;
                const string tag = confirm ? "S7B S15_CONFIRM" : "S7B S15_CHOOSE";
                for (const bool all : {false, true}) {
                    vector<std::size_t> idx;
                    for (std::size_t i = 0; i < it15.size(); ++i)
                        if (it15[i].dev != confirm && (all || it15[i].reach)) idx.push_back(i);
                    const string sn = all ? "all" : "reachable";
                    for (const LoopKind& K : lk15) {
                        ecs::TestTally t;
                        for (std::size_t i : idx) t.add(it15[i].label == 1, K.f(it15[i]));
                        std::cout << line_of(tag.c_str(), string(K.name).substr(0, 2) + "_" + sn, t) << "\n";
                    }
                    {
                        ecs::TestTally t;
                        for (std::size_t i : idx) t.add(it15[i].label == 1, it15[i].t1_reg);
                        std::cout << line_of(tag.c_str(), "T1_" + sn + "_registered_phi", t) << "\n";
                    }
                    struct Run {
                        Cfg c;
                        const vector<LoopKind>* k;
                        const char* what;
                    };
                    for (const Run& R : {Run{chosen, &lk15, "amended phi"}, Run{chosen, &lk15r, "registered phi"}, Run{Cfg{false, B0_ALL}, &lk15, "amended phi"}}) {
                        const bool primary = confirm && !all && R.k == &lk15 && R.c.three == chosen.three && R.c.batch == chosen.batch;
                        const LoopStats L = loop2(it15, idx, *R.k, R.c.three, R.c.batch, primary ? out + ".s15_confirm.prereg.tsv" : "");
                        string cats;
                        for (const auto& [k, v] : L.sup_cat) cats += string(" ") + k + " " + std::to_string(static_cast<int>(v));
                        std::cout << loop_line(tag, sn + " " + cfg_name(R.c) + ", " + R.what, L) << " | supported by category:" << cats << "\n";
                    }
                }
            }
            // ---- S7c on S15's choose half (reachable primary, all beside it): every rule, and the resampling check
            if (s7c) {
                for (const bool all : {false, true}) {
                    vector<std::size_t> idx;
                    for (std::size_t i = 0; i < it15.size(); ++i)
                        if (it15[i].dev && (all || it15[i].reach)) idx.push_back(i);
                    const vector<LoopRes> res = loop_res(it15, idx, lk15);
                    vector<std::size_t> S(res.size());
                    std::iota(S.begin(), S.end(), 0);
                    const string tag = string("S15_CHOOSE ") + (all ? "all" : "reachable");
                    for (int k = 0; k < N_K; ++k) {
                        const KStats st = s7c_stats(res, S, s7c_run(k, res, S));
                        std::cout << s7c_line(tag, kKName[k], st) << "\n";
                        if (k == K1 || k == K2 || k == K4)
                            std::cout << s7c_line(tag, string(kKName[k]) + "_R-and-S", s7c_stats(res, S, s7c_run(k, res, S, false))) << "\n";
                        if (!all) s7c15.st[k] = st;
                    }
                    if (!all) {
                        s7c15.rs = s7c_resample(res, 400, 20261103);
                        s7c15.have = true;
                        std::cout << fdr_line("S15_CHOOSE resampling", s7c15.rs) << "\n";
                    }
                }
            }
        }
        // ---- S7c: dev reads, the guarantee checks and the choice; then C2, read once with --read-c2
        if (s7c) {
            const vector<LoopRes> rd = loop_res(it, dev_idx, lk);
            vector<std::size_t> S(rd.size());
            std::iota(S.begin(), S.end(), 0);
            KStats dst[N_K];
            for (int k = 0; k < N_K; ++k) {
                dst[k] = s7c_stats(rd, S, s7c_run(k, rd, S));
                std::cout << s7c_line("DEV", kKName[k], dst[k]) << "\n";
                if (k == K1 || k == K2 || k == K4)
                    std::cout << s7c_line("DEV", string(kKName[k]) + "_R-and-S", s7c_stats(rd, S, s7c_run(k, rd, S, false))) << "\n";
            }
            {  // K2's condition, roughly: the mean e-value of false dev hypotheses in each prior quartile
                vector<std::pair<double, double>> fe;
                for (const LoopRes& r : rd)
                    if (r.counted && r.label == 0) fe.push_back({r.p0, r.e});
                std::sort(fe.begin(), fe.end());
                string l = "S7C DEV mean e of false hypotheses by prior quartile:";
                for (int q = 0; q < 4; ++q) {
                    const std::size_t a = fe.size() * static_cast<std::size_t>(q) / 4, z = fe.size() * static_cast<std::size_t>(q + 1) / 4;
                    double se = 0;
                    for (std::size_t i = a; i < z; ++i) se += fe[i].second;
                    char bb[128];
                    std::snprintf(bb, sizeof bb, " Q%d (p0 %.3f-%.3f) %.3f (n %zu)", q + 1, z > a ? fe[a].first : 0.0, z > a ? fe[z - 1].first : 0.0,
                                  z > a ? se / static_cast<double>(z - a) : 0.0, z - a);
                    l += bb;
                }
                std::cout << l << "\n";
            }
            vector<double> pool;
            for (const LoopRes& r : rd) pool.push_back(r.p0);
            const FdrAcc sim_mixed = s7c_simulate(pool, lk, rd.size(), 400, false, 20261104);
            const FdrAcc sim_null = s7c_simulate(pool, lk, rd.size(), 400, true, 20261105);
            const FdrAcc rs_dev = s7c_resample(rd, 400, 20261106);
            std::cout << fdr_line("SIM mixed", sim_mixed) << "\n" << fdr_line("SIM planted-nulls", sim_null) << "\n"
                      << fdr_line("DEV resampling", rs_dev) << "\n";
            // The decision rule: qualifiers, then the largest Known-decided share on dev; ties within 0.5 points to the simpler.
            int win = -1;
            double wshare = -1;
            for (const int k : {static_cast<int>(K1), static_cast<int>(K2), static_cast<int>(K4), static_cast<int>(K3)}) {
                const KStats& d = dst[k];
                const bool q_dev = d.prom() >= 10 && ecs::wilson_lower(d.pt, d.prom()) >= 0.90;
                const bool q_s15 = s7c15.have && (s7c15.st[k].prom() < 10 || s7c15.st[k].prec() >= 0.90);
                auto ok = [k](const FdrAcc& a) { return a.fdr(k) <= 0.10 + 2 * a.se(k); };
                const bool q_fdr = ok(sim_mixed) && ok(sim_null) && ok(rs_dev) && s7c15.have && ok(s7c15.rs);
                const bool qual = q_dev && q_s15 && q_fdr;
                std::snprintf(b, sizeof b, "S7C QUALIFY %-26s dev %d s15 %d fdr-checks %d -> %s | Known-decided on dev %.4f", kKName[k], q_dev, q_s15,
                              q_fdr, qual ? "qualifies" : "does not qualify", d.decided());
                std::cout << b << "\n";
                if (qual && d.decided() > wshare + 0.005) wshare = d.decided(), win = k;
            }
            std::cout << "S7C CHOICE "
                      << (win >= 0 ? string("hybrid: Known = ") + kKName[win] + ", Believed = decide's other supported"
                                   : string("none qualifies: S7b's step stays, relabelled Believed-only"))
                      << "\n";
            // ---- C2: a third fact of S6's subjects (never judged), built as S7's C was
            std::mt19937_64 rngc(20261029);
            vector<Held> held2;
            std::size_t c2_short = 0;
            for (const string& s2 : subj) {
                const auto it6 = h6.find(s2);
                if (it6 == h6.end()) continue;
                auto fs = by_subj[s2];
                for (std::size_t i = fs.size(); i > 1; --i) std::swap(fs[i - 1], fs[rngc() % i]);
                int got = 0;
                for (const auto& [r, o] : fs)
                    if (!(r == it6->second.first && o == it6->second.second) && full.count(r, o) >= 2 && ++got == 2) {
                        held2.push_back({s2, r, o, 'D'});
                        break;
                    }
                if (got < 2) ++c2_short;
            }
            std::set<string> keys2 = held_keys;
            for (const auto& h : held2) keys2.insert(h.s + '\x1f' + h.r + '\x1f' + h.o);
            ecs::TripleStore st2;
            std::unordered_map<string, vector<string>> obj2;
            for (const auto& [s2, fs] : by_subj)
                for (const auto& [r, o] : fs)
                    if (!keys2.count(s2 + '\x1f' + r + '\x1f' + o)) st2.add(s2, r, o), obj2[r].push_back(o);
            std::mt19937_64 rngf2(20261101);
            vector<Item> it2;
            std::size_t gone2 = 0, nof2 = 0;
            for (const auto& h : held2) {
                if (st2.count(h.r, h.o) == 0) { ++gone2; continue; }
                const auto& pool2 = obj2[h.r];
                const int ri = rel_index(h.r);
                const Date hd = is_datey(ri) ? parse_date_str(h.o) : Date{};
                string neg;
                for (int t = 0; t < 100 && neg.empty(); ++t) {
                    const string& c = pool2[rngf2() % pool2.size()];
                    if (c == h.o || c == h.s || full.has(h.s, h.r, c)) continue;
                    if (is_datey(ri)) {
                        const Date cd = parse_date_str(c);
                        if (cd.ok() && hd.ok() && cd.y == hd.y) continue;
                    }
                    neg = c;
                }
                if (neg.empty()) { ++nof2; continue; }
                const auto wit = wd.find(h.s + '\x1f' + h.r);
                Item a, x;
                a.s = x.s = h.s, a.r = x.r = h.r, a.o = h.o, x.o = neg, a.label = 1, x.label = 0;
                a.pair = x.pair = static_cast<int>(it2.size() / 2);
                a.set = x.set = 'D';
                a.t5_ok = x.t5_ok = !wd_src.count(h.s + '\x1f' + h.r);
                x.wd_true = wit != wd.end() && wd_match(ri, neg, wit->second);
                it2.push_back(a), it2.push_back(x);
            }
            std::size_t f0 = 0;
            for (std::size_t i = 0; i < it2.size(); i += 2) f0 += ((fnv(it2[i].s) >> 13) & 1) == 0;
            std::snprintf(b, sizeof b,
                          "S7C C2 subjects with an S6 hold-out %zu (no second qualifying fact after C's %zu) | pairs %zu (dropped: object gone %zu, "
                          "no false partner %zu) | fold 0 pairs %zu, fold 1 %zu | reduced store %zu of %zu",
                          h6.size(), c2_short, it2.size() / 2, gone2, nof2, f0, it2.size() / 2 - f0, st2.size(), full.size());
            std::cout << b << "\n";
            if (read_c2) {
                const ecs::LinkPredictor lp2(st2, {});
                for (std::size_t i = 0; i < it2.size(); i += 2) {
                    const int ri = rel_index(it2[i].r);
                    PageScan ps;
                    const auto pit = page_of.find(it2[i].s);
                    const bool page = pit != page_of.end();
                    if (page) ps = scan_rel(data, pages[static_cast<std::size_t>(pit->second)], ri);
                    for (std::size_t j = i; j < i + 2; ++j) {
                        Item& x = it2[j];
                        x.t1v[kphi] = t1_phi(st2, x.s, x.r, x.o, ph[kphi]);
                        x.p = lp2.prob(x.s, x.r, x.o);
                        x.aa = ecs::adamic_adar(st2, x.s, x.o);
                        x.dl_others = ecs::compression_gain_bits(lp2, st2, x.s, x.r, x.o) - std::log2(std::max(1e-12, x.p));
                        for (int k = 0; k < 4; ++k) x.t6[k] = x.dl_others > kEps[k] ? kPos : x.dl_others < -kEps[k] ? kNeg : kInc;
                        const auto wit = wd.find(x.s + '\x1f' + x.r);
                        x.t5 = !x.t5_ok || wit == wd.end() ? kInc : wd_match(ri, x.o, wit->second) ? kPos : kNeg;
                        if (page) t4(ps, ri, x.o, x.t4);
                        x.p0 = std::min(0.98, std::max(0.02, cal.predict(raw(x))));
                    }
                }
                vector<std::size_t> idx2(it2.size());
                std::iota(idx2.begin(), idx2.end(), 0);
                const vector<LoopRes> rc = loop_res(it2, idx2, lk, out + ".c2.prereg.tsv");
                vector<std::size_t> S2(rc.size());
                std::iota(S2.begin(), S2.end(), 0);
                vector<bool> prw;
                for (int k = 0; k < N_K; ++k) {
                    const auto pr = s7c_run(k, rc, S2);
                    if (k == win) prw = pr;
                    std::cout << s7c_line("C2", kKName[k], s7c_stats(rc, S2, pr)) << "\n";
                    if (k == K1 || k == K2 || k == K4)
                        std::cout << s7c_line("C2", string(kKName[k]) + "_R-and-S", s7c_stats(rc, S2, s7c_run(k, rc, S2, false))) << "\n";
                }
                for (int f = 0; f < 2 && win == K3; ++f) {  // K3's two folds, each with its own guarantee
                    vector<std::size_t> Sf;
                    vector<bool> pf;
                    for (std::size_t j = 0; j < S2.size(); ++j)
                        if (rc[j].fold == f) Sf.push_back(j), pf.push_back(prw[j]);
                    std::cout << s7c_line("C2 fold " + std::to_string(f), kKName[K3], s7c_stats(rc, Sf, pf)) << "\n";
                }
                if (win >= 0) {
                    const KStats w = s7c_stats(rc, S2, prw);
                    std::snprintf(b, sizeof b,
                                  "S7C C2 CONFIRM %s: Known precision %.3f [%.3f, %.3f] on %.0f -> %s | Known-decided %.3f (S7b's whole-batch e-BH on dev: "
                                  "0.111) | Believed %.0f precision %.3f",
                                  kKName[win], w.prec(), ecs::wilson_lower(w.pt, w.prom()), ecs::wilson_upper(w.pt, w.prom()), w.prom(),
                                  ecs::wilson_lower(w.pt, w.prom()) >= 0.90 ? "confirmed" : "not confirmed", w.decided(), w.bt + w.bf,
                                  w.bt + w.bf > 0 ? w.bt / (w.bt + w.bf) : 0.0);
                    std::cout << b << "\n";
                }
                std::ofstream f(out + ".c2_items.tsv");
                f << "pair\tfold\tlabel\twd_true\tt5_eligible\tsubject\trelation\tobject\tT1\tT6\tT5\tT4\tP0\tposterior\te\tp_null\tsupported\trefuted\ttier\n";
                for (std::size_t j = 0; j < it2.size(); ++j) {
                    const Item& x = it2[j];
                    const LoopRes& r = rc[j];
                    const int tier = r.sup ? (win >= 0 && prw[j] ? 2 : 1) : 0;
                    f << x.pair << '\t' << r.fold << '\t' << x.label << '\t' << x.wd_true << '\t' << x.t5_ok << '\t' << x.s << '\t' << x.r << '\t' << x.o
                      << '\t' << ecs::test_outcome_name(x.t1v[kphi]) << '\t' << ecs::test_outcome_name(x.t6[k6]) << '\t' << ecs::test_outcome_name(x.t5)
                      << '\t' << ecs::test_outcome_name(x.t4[k4]) << '\t' << x.p0 << '\t' << r.post << '\t' << r.e << '\t' << r.pnull << '\t' << r.sup
                      << '\t' << r.ref << '\t' << (tier == 2 ? "Known" : tier == 1 ? "Believed" : "") << '\n';
                }
            }
        }
    }
    {
        std::ofstream f(out + ".items.tsv");
        f << "pair\tset\tsplit\tlabel\twd_true\tt5_eligible\tsubject\trelation\tobject\tT1\tT6\tT5\tT4\tdl_others\tP0\tposterior\tstate\n";
        for (const Item& x : it) {
            if (!x.dev && !read_test) continue;
            f << x.pair << '\t' << x.set << '\t' << (x.dev ? "dev" : "test") << '\t' << x.label << '\t' << x.wd_true << '\t' << x.t5_ok << '\t' << x.s << '\t'
              << x.r << '\t' << x.o << '\t' << ecs::test_outcome_name(x.t1) << '\t'
              << ecs::test_outcome_name(x.t6[k6]) << '\t' << ecs::test_outcome_name(x.t5) << '\t' << ecs::test_outcome_name(x.t4[k4]) << '\t'
              << x.dl_others << '\t' << x.p0 << '\t' << x.post << '\t' << x.state << '\n';
        }
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S7 seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s7

// ============================================================ S8b: link prediction, G6 and G7 (docs/reports/S8B_LINK_PREDICTION.md)
//
// pattern_learn --gen-links ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test]
// Four reads; the test halves print only with --read-test (read once, for the final setting):
//   ROD   rediscovery on the full store (S6's and S7's seed store): 10% of the triples hidden at random, and separately
//         the 10% least connected (lowest deg s + deg o); each hidden slot's objects ranked among r's objects by the base
//         rate, S6's link predictor (LP), G6 (CN, Jaccard, Adamic-Adar, resource allocation), G7 (path ranking) and their
//         combination; Hits@1, Hits@10, MRR. Dev and test by bit 17 of FNV-1a(subject).
//   PROMO the chosen generator's top candidate for each hidden slot of the random hide, through S7's loop (T1, T6, T4,
//         S7's dev s and c; T5 left out, Wikidata being the truth), BH at q 0.1; judged by the hidden facts and Wikidata.
//   S15   the temporal holdout (T9): the 2006-text store (text seeds and F2's facts, no Wikidata row) ranks values for
//         S15's slots it lacks; Hits by S15 category, and promotion judged by S15. S15's own dev and test halves.
//   GATE  S6's next change: P0 with G7's path evidence on S15's true and false pairs (one per subject, its slot removed
//         from the full store); z pools and isotonic fits on S15 dev, read on S15 test.
// Nothing here answers a question: these are Cypha's own hypotheses (PLAN rule 10).
namespace s8b {

using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;

enum Gen { kBase, kLP, kCN, kJac, kAA, kRA, kG7, kCombo, kNGen };
const char* const kGenName[kNGen] = {"base_rate", "LP", "G6_CN", "G6_Jaccard", "G6_AA", "G6_RA", "G7_paths", "combination"};
// S7's dev point estimates of s and c (S7 report; its loop's registered inputs), in cost order.
struct TestKind {
    const char* name;
    const char* source;
    bool disconfirmation, independent;
    double s, c;
};
const TestKind kKinds[3] = {{"T1 consistency", "store", true, false, 0.7551, 0.9184},
                            {"T6 compression", "store", false, false, 0.2553, 0.9493},
                            {"T4 corpus", "corpus", true, true, 0.9933, 0.8137}};

// The article text, its pages and the type lexicons T1 and T4 read (S7's pass, the same files).
struct Corpus {
    string data;
    vector<PageRef> pages;
    std::unordered_map<string, int> page_of;
};
bool load_corpus(const char* enwik8, const char* wd_path, const char* patterns_path, Corpus& C) {
    if (!read_file(enwik8, C.data)) { std::fprintf(stderr, "cannot read %s\n", enwik8); return false; }
    g_dump = &C.data;
    long total = 0, redirects = 0;
    C.pages = list_pages(C.data, total, redirects);
    for (std::size_t i = 0; i < C.pages.size(); ++i) C.page_of[C.pages[i].title] = static_cast<int>(i);
    for (const PageRef& pg : C.pages) {
        PageFlags fl;
        scan_page(C.data, pg, fl);
        const string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    long ns = 0, ng = 0;
    if (!load_wikidata(wd_path, false, C.page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", wd_path); return false; }
    for (const Seed& sd : g_seeds) {
        const RT ty = kRel[sd.rel].type;
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    std::ifstream f(patterns_path);
    string l;
    std::getline(f, l);
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < 11 || c[10] != "1" || rel_index(c[0]) < 0) continue;
        Pat p;
        p.rel = rel_index(c[0]);
        p.key = c[1];
        p.family = family_of(p.key);
        p.kept = true;
        g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
        g_pats.push_back(p);
    }
    return true;
}

// Does value a (S15 or Wikidata form) name object b? Dates by year (year relations) or compatible date; others by
// normalised name (S7's rule).
bool same_value(int ri, const string& a, const string& b) {
    if (is_datey(ri)) {
        const Date A = parse_date_str(a), B = parse_date_str(b);
        if (!A.ok() || !B.ok()) return false;
        return kRel[ri].type == T_YEAR ? A.y == B.y : date_compat(A, B);
    }
    const string ka = norm_ent(a);
    return !ka.empty() && ka == norm_ent(b);
}

// A relation's objects indexed by normalised name, or by year for a date relation; find() keeps the ones same_value
// accepts.
struct ValueIndex {
    int ri = -1;
    const vector<string>* objs = nullptr;
    std::unordered_map<string, vector<int>> by_key;
    string key(const string& v) const {
        if (ri >= 0 && is_datey(ri)) {
            const Date D = parse_date_str(v);
            return D.ok() ? std::to_string(D.y) : string();
        }
        return norm_ent(v);
    }
    void build(const string& r, const vector<string>& o) {
        ri = rel_index(r), objs = &o, by_key.clear();
        for (std::size_t i = 0; i < o.size(); ++i) {
            const string k = key(o[i]);
            if (!k.empty()) by_key[k].push_back(static_cast<int>(i));
        }
    }
    vector<int> find(const string& v) const {
        vector<int> out;
        const string k = key(v);
        const auto it = k.empty() ? by_key.end() : by_key.find(k);
        if (it == by_key.end()) return out;
        for (int i : it->second)
            if (same_value(ri, v, (*objs)[static_cast<std::size_t>(i)])) out.push_back(i);
        return out;
    }
};

// Each relation's objects in a store, distinct and sorted, with how many triples hold each.
std::map<string, vector<std::pair<string, std::size_t>>> objects_by_rel(const ecs::TripleStore& st) {
    std::map<string, std::map<string, std::size_t>> m;
    for (const auto& [s, fs] : st.subjects())
        for (const auto& f : fs) ++m[f.r][f.o];
    std::map<string, vector<std::pair<string, std::size_t>>> out;
    for (auto& [r, os] : m) out[r].assign(os.begin(), os.end());
    return out;
}

// One query: a slot (s, r), its candidates (r's objects not already held by s for r) and their scores.
struct Target {
    string value, cat;  // the true value (a store object, or S15's form) and S15's category
    vector<int> cands;  // the candidates it names (empty: unreachable)
};
struct Query {
    string s, r;
    bool dev = true;
    vector<string> cands;
    vector<double> base, lp, cn, jac, aa, ra, g7;
    vector<int> g7_best;
    vector<Target> targets;
    vector<char> truth;  // per candidate: names a target (left out of the competition, the filtered setting)
};

// The scores of every candidate of q (store st; PathRanker pr on st's graph).
void score_query(Query& q, const ecs::TripleStore& st, const ecs::LinkScorer& ls, ecs::PathRanker& pr) {
    const std::size_t n = q.cands.size();
    q.base.resize(n), q.cn.assign(n, 0), q.jac.assign(n, 0), q.aa.assign(n, 0), q.ra.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) q.base[i] = ecs::base_rate(st, q.r, q.cands[i]);
    q.lp = ls.probs(q.s, q.r, q.cands);
    const auto ns = ecs::neighbour_scores(st, q.s);
    for (std::size_t i = 0; i < n; ++i) {
        const auto it = ns.find(q.cands[i]);
        if (it == ns.end()) continue;
        q.cn[i] = it->second.cn, q.jac[i] = it->second.jaccard, q.aa[i] = it->second.aa, q.ra[i] = it->second.ra;
    }
    q.g7 = pr.score(q.s, q.r, q.cands, &q.g7_best);
}

double logit_c(double p) {
    p = std::min(1 - 1e-12, std::max(1e-12, p));
    return std::log(p / (1 - p));
}
vector<double> combo_features(const Query& q, std::size_t i) {
    return {logit_c(q.lp[i]), std::log1p(1000.0 * q.ra[i]), q.g7[i], std::log(std::max(1e-12, q.base[i]))};
}

// A generator's primary score per candidate (ties go to the base rate, then mid-rank).
vector<double> primary(const Query& q, int g, const ecs::Logistic* combo) {
    switch (g) {
        case kBase: return q.base;
        case kLP: return q.lp;
        case kCN: return q.cn;
        case kJac: return q.jac;
        case kAA: return q.aa;
        case kRA: return q.ra;
        case kG7: return q.g7;
        default: {
            vector<double> v(q.cands.size(), 0.0);
            for (std::size_t i = 0; combo && i < v.size(); ++i) v[i] = combo->logit(combo_features(q, i));
            return v;
        }
    }
}

double rank_of(const Query& q, const vector<double>& p, int t) {
    double higher = 0, ties = 0;
    for (std::size_t j = 0; j < q.cands.size(); ++j) {
        if (static_cast<int>(j) == t || q.truth[j]) continue;
        if (p[j] > p[static_cast<std::size_t>(t)] || (p[j] == p[static_cast<std::size_t>(t)] && q.base[j] > q.base[static_cast<std::size_t>(t)])) higher += 1;
        else if (p[j] == p[static_cast<std::size_t>(t)] && q.base[j] == q.base[static_cast<std::size_t>(t)]) ties += 1;
    }
    return 1 + higher + 0.5 * ties;
}
// The best rank among the candidates a target names; 0 when it names none.
double target_rank(const Query& q, const Target& t, const vector<double>& p) {
    double best = 0;
    for (int c : t.cands) {
        const double r = rank_of(q, p, c);
        if (best == 0 || r < best) best = r;
    }
    return best;
}
// The top candidate by (primary, base rate), the lower index on a tie; -1 when there is none.
int top_of(const Query& q, const vector<double>& p) {
    int b = -1;
    for (std::size_t j = 0; j < q.cands.size(); ++j)
        if (b < 0 || p[j] > p[static_cast<std::size_t>(b)] || (p[j] == p[static_cast<std::size_t>(b)] && q.base[j] > q.base[static_cast<std::size_t>(b)]))
            b = static_cast<int>(j);
    return b;
}

struct CI {
    double mean = 0, lo = 0, hi = 0;
};
CI boot(const vector<double>& v, std::uint64_t seed) {
    CI c;
    if (v.empty()) return c;
    c.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    std::mt19937_64 rng(seed);
    vector<double> bs;
    for (int rep = 0; rep < 2000; ++rep) {
        double t = 0;
        for (std::size_t k = 0; k < v.size(); ++k) t += v[rng() % v.size()];
        bs.push_back(t / static_cast<double>(v.size()));
    }
    std::sort(bs.begin(), bs.end());
    c.lo = bs[49], c.hi = bs[1949];
    return c;
}

// Hits@1, Hits@10 and MRR of every generator over a set of queries; per-target rows go to `rows`.
struct GenRead {
    vector<double> rr, h1, h10;
};
std::array<GenRead, kNGen> evaluate(const vector<Query>& qs, bool dev, const ecs::Logistic* combo, const vector<const ecs::Logistic*>* combo_per_query = nullptr) {
    std::array<GenRead, kNGen> out;
    for (std::size_t qi = 0; qi < qs.size(); ++qi) {
        const Query& q = qs[qi];
        if (q.dev != dev) continue;
        for (int g = 0; g < kNGen; ++g) {
            const vector<double> p = primary(q, g, g == kCombo && combo_per_query ? (*combo_per_query)[qi] : combo);
            for (const Target& t : q.targets) {
                const double r = target_rank(q, t, p);
                out[static_cast<std::size_t>(g)].rr.push_back(r > 0 ? 1.0 / r : 0.0);
                out[static_cast<std::size_t>(g)].h1.push_back(r > 0 && r <= 1.0 ? 1.0 : 0.0);
                out[static_cast<std::size_t>(g)].h10.push_back(r > 0 && r <= 10.0 ? 1.0 : 0.0);
            }
        }
    }
    return out;
}

// Logistic rows for the combination: each reachable target's candidate against 20 drawn non-truth candidates.
ecs::Logistic fit_combo(const vector<Query>& qs, const std::function<bool(const Query&)>& use, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    vector<vector<double>> X;
    vector<int> y;
    for (const Query& q : qs) {
        if (!use(q)) continue;
        vector<std::size_t> others;
        for (std::size_t j = 0; j < q.cands.size(); ++j)
            if (!q.truth[j]) others.push_back(j);
        for (const Target& t : q.targets) {
            if (t.cands.empty()) continue;
            X.push_back(combo_features(q, static_cast<std::size_t>(t.cands[0]))), y.push_back(1);
            for (int k = 0; k < 20 && !others.empty(); ++k) X.push_back(combo_features(q, others[rng() % others.size()])), y.push_back(0);
        }
    }
    ecs::Logistic lr;
    lr.fit(X, y, 1.0);
    return lr;
}

// The candidates of slot (s, r) in store st: r's objects, less those s already holds for r.
vector<string> candidates(const ecs::TripleStore& st, const std::map<string, vector<std::pair<string, std::size_t>>>& objs, const string& s, const string& r) {
    vector<string> out;
    const auto it = objs.find(r);
    if (it == objs.end()) return out;
    for (const auto& [o, n] : it->second)
        if (!st.has(s, r, o)) out.push_back(o);
    return out;
}

// One hypothesis through S7's loop: pre-registered, its tests cheapest first, then decide. Returns the outcomes.
struct LoopItem {
    string s, r, o;
    bool dev = true;
    double score = 0, prior = 0.5, post = 0;
    int truth = -1;  // 1 true, 0 false, -1 unjudged
    TestOutcome out[3] = {kInc, kInc, kInc};
    string state;
    bool promoted = false;
    string explain;
};
void run_tests(LoopItem& h, const ecs::TripleStore& st, const ecs::LinkPredictor& lp, const Corpus& C) {
    const int ri = rel_index(h.r);
    h.out[0] = s7::t1(st, h.s, h.r, h.o);
    const double p = lp.prob(h.s, h.r, h.o);
    const double dl_others = ecs::compression_gain_bits(lp, st, h.s, h.r, h.o) - std::log2(std::max(1e-12, p));
    h.out[1] = dl_others > 0 ? kPos : dl_others < 0 ? kNeg : kInc;  // S7's eps 0
    h.out[2] = kInc;
    const auto pit = C.page_of.find(h.s);
    if (ri >= 0 && pit != C.page_of.end()) {
        TestOutcome o4[2];
        s7::t4(s7::scan_rel(C.data, C.pages[static_cast<std::size_t>(pit->second)], ri), ri, h.o, o4);
        h.out[2] = o4[1];  // S7's choice: pattern or co-mention
    }
}
// S7's loop over one batch (one split): pre-registration, the tests in cost order, decide, BH at q 0.1.
struct LoopRead {
    double judged_sup = 0, true_sup = 0, judged_pro = 0, true_pro = 0, judged_ref = 0, false_ref = 0, n = 0, sup = 0, pro = 0, ref = 0, unjudged_pro = 0;
    bool log_ok = true;
    std::size_t log_entries = 0;
};
LoopRead run_loop(vector<LoopItem*>& items, const string& log_path) {
    ecs::HypothesisStore hs;
    ecs::PreregLog log;
    LoopRead R;
    for (std::size_t i = 0; i < items.size(); ++i) {
        LoopItem& x = *items[i];
        ecs::Hypothesis h;
        h.id = "S8b-" + std::to_string(i);
        h.key = x.r + '|' + x.s + '|' + x.o;
        h.statement = x.s + " " + x.r + " " + x.o;
        h.generator = "G6/G7 link prediction";
        for (const TestKind& K : kKinds) {
            ecs::HypTest t;
            t.kind = K.name, t.source = K.source, t.predicted = "positive", t.refuted_if = "negative";
            t.disconfirmation = K.disconfirmation, t.independent = K.independent, t.sensitivity = K.s, t.specificity = K.c;
            h.tests.push_back(t);
        }
        hs.propose(h);
        hs.score(h.id, x.prior);
        hs.transition(h.id, ecs::HypState::Queued, "scored");
        if (!log.register_plan(*hs.find(h.id))) continue;
        for (std::size_t k = 0; k < 3; ++k) {
            if (hs.find(h.id)->state != ecs::HypState::Queued) break;
            hs.transition(h.id, ecs::HypState::UnderTest, "next test");
            hs.record_outcome(h.id, k, x.out[k], &log);
        }
    }
    vector<double> e;
    for (std::size_t i = 0; i < items.size(); ++i) e.push_back(ecs::e_value(*hs.find("S8b-" + std::to_string(i))));
    const auto rej = ecs::bh_reject_e(e, 0.1);
    for (std::size_t i = 0; i < items.size(); ++i) {
        LoopItem& x = *items[i];
        const auto* h = hs.find("S8b-" + std::to_string(i));
        x.post = h->posterior, x.state = ecs::hyp_state_name(h->state);
        R.n += 1;
        if (h->state == ecs::HypState::Supported) {
            R.sup += 1;
            if (x.truth >= 0) R.judged_sup += 1, R.true_sup += x.truth;
            x.promoted = rej[i];
            if (x.promoted) {
                R.pro += 1;
                if (x.truth >= 0) R.judged_pro += 1, R.true_pro += x.truth;
                else R.unjudged_pro += 1;
            }
        } else if (h->state == ecs::HypState::Refuted) {
            R.ref += 1;
            if (x.truth >= 0) R.judged_ref += 1, R.false_ref += 1 - x.truth;
        }
    }
    R.log_ok = log.verify();
    R.log_entries = log.entries().size();
    std::ofstream lf(log_path);
    lf << log.serialize();
    return R;
}
string loop_line(const string& tag, const LoopRead& R) {
    char b[768];
    std::snprintf(b, sizeof b,
                  "%s hypotheses %.0f | supported %.0f (judged %.0f, true %.0f, precision %.3f [%.3f, %.3f]) | promoted after BH %.0f (judged %.0f, "
                  "true %.0f, precision %.3f [%.3f, %.3f], unjudged %.0f) | refuted %.0f (judged %.0f, false %.0f, precision %.3f) | log %zu entries verify %s",
                  tag.c_str(), R.n, R.sup, R.judged_sup, R.true_sup, R.judged_sup > 0 ? R.true_sup / R.judged_sup : 0.0, ecs::wilson_lower(R.true_sup, R.judged_sup),
                  ecs::wilson_upper(R.true_sup, R.judged_sup), R.pro, R.judged_pro, R.true_pro, R.judged_pro > 0 ? R.true_pro / R.judged_pro : 0.0,
                  ecs::wilson_lower(R.true_pro, R.judged_pro), ecs::wilson_upper(R.true_pro, R.judged_pro), R.unjudged_pro, R.ref, R.judged_ref, R.false_ref,
                  R.judged_ref > 0 ? R.false_ref / R.judged_ref : 0.0, R.log_entries, R.log_ok ? "ok" : "BROKEN");
    return b;
}

// Isotonic prior from dev items' scores and truths (judged only), clamped to [0.02, 0.98].
void set_priors(vector<LoopItem>& items) {
    vector<double> p;
    vector<int> y;
    for (const LoopItem& x : items)
        if (x.dev && x.truth >= 0) p.push_back(x.score), y.push_back(x.truth);
    ecs::IsotonicCalibrator cal;
    cal.fit(p, y);
    for (LoopItem& x : items) x.prior = std::min(0.98, std::max(0.02, cal.predict(x.score)));
}

// AUROC of scores over (true, false) pairs, with a 95% bootstrap interval over pairs (S6's seed and resamples).
struct PairRead {
    double auc = 0, lo = 0, hi = 0;
};
double auroc_of(const vector<std::pair<double, int>>& v0) {
    auto v = v0;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    double rp = 0, np = 0, nn = 0;
    for (std::size_t i = 0; i < v.size();) {
        std::size_t j = i;
        while (j < v.size() && v[j].first == v[i].first) ++j;
        const double mid = 0.5 * static_cast<double>(i + 1 + j);
        for (std::size_t k = i; k < j; ++k)
            if (v[k].second) rp += mid, ++np; else ++nn;
        i = j;
    }
    return np && nn ? (rp - np * (np + 1) / 2) / (np * nn) : 0.5;
}
PairRead pair_auroc(const vector<std::pair<double, double>>& pairs) {  // (true score, false score)
    auto flat = [&](const vector<std::size_t>& idx) {
        vector<std::pair<double, int>> v;
        for (std::size_t k : idx) v.push_back({pairs[k].first, 1}), v.push_back({pairs[k].second, 0});
        return auroc_of(v);
    };
    vector<std::size_t> all(pairs.size());
    std::iota(all.begin(), all.end(), 0);
    PairRead r;
    r.auc = flat(all);
    std::mt19937_64 rng(20261028);
    vector<double> bs;
    for (int rep = 0; rep < 2000 && !pairs.empty(); ++rep) {
        vector<std::size_t> idx;
        for (std::size_t k = 0; k < pairs.size(); ++k) idx.push_back(rng() % pairs.size());
        bs.push_back(flat(idx));
    }
    std::sort(bs.begin(), bs.end());
    if (!bs.empty()) r.lo = bs[49], r.hi = bs[1949];
    return r;
}

struct S15Row {
    string id, s, r, value, cat, pair;
    int label = 0;
};
vector<S15Row> load_s15(const string& path) {
    vector<S15Row> out;
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < 17) continue;
        out.push_back({c[0], c[1], c[5], c[7], c[11], c[16], c[15] == "1" ? 1 : 0});
    }
    return out;
}

int run(int argc, char** argv) {
    if (argc < 9) {
        std::fprintf(stderr, "usage: pattern_learn --gen-links ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test] [--fresh-confirm | --fresh-test-half]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[8];
    bool read_test = false, fresh_confirm = false;  // --fresh-confirm: S8d's fresh G6/G7 set (docs/reports/S8D_PROMOTION.md)
    bool fresh_test = false;  // --fresh-test-half: S10's gate set, the same on S15 test's bit-17 half (docs/reports/S10_CURIOSITY_SCHEDULER.md)
    double abc_w = 0;         // --abc-w W: S6c, the GATE read with G12's closed form in P0 (the S6 report's "S6c")
    for (int i = 9; i < argc; ++i) {
        if (string(argv[i]) == "--abc-w" && i + 1 < argc) { abc_w = std::atof(argv[++i]); continue; }
        if (string(argv[i]) == "--read-test") read_test = true;
        else if (string(argv[i]) == "--fresh-confirm") fresh_confirm = true;
        else if (string(argv[i]) == "--fresh-test-half") fresh_test = true;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    char b[1024];
    Corpus C;
    if (!load_corpus(argv[2], argv[5], argv[6], C)) return 1;
    // ---- the full store (S6's and S7's) and the 2006-text store
    ecs::TripleStore full, text;
    std::set<string> seen;
    vector<std::array<string, 3>> triples;  // the full store's, sorted
    {
        std::map<string, vector<std::pair<string, string>>> by_subj;
        std::ifstream f(argv[3]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            if (seen.insert(c[0] + '\x1f' + c[1] + '\x1f' + c[2]).second) by_subj[c[0]].push_back({c[1], c[2]});
            if (c[3] == "category" || c[3] == "infobox" || c[3] == "table") text.add(c[0], c[1], c[2]);
        }
        for (auto& [s, fs] : by_subj) {
            std::sort(fs.begin(), fs.end());
            for (const auto& [r, o] : fs) full.add(s, r, o), triples.push_back({s, r, o});
        }
        std::ifstream g(argv[4]);
        std::getline(g, l);
        while (std::getline(g, l)) {
            const auto c = split(l, '\t');
            if (c.size() >= 3 && rel_index(c[1]) >= 0 && !c[0].empty() && !c[2].empty()) text.add(c[0], c[1], c[2]);
        }
    }
    const s7::WdObjs wd = s7::load_wd_objs(argv[5]);
    std::snprintf(b, sizeof b, "S8b STORES full %zu triples (%zu subjects) | 2006-text %zu triples (%zu subjects) | Wikidata slots %zu | pages %zu",
                  full.size(), full.subjects().size(), text.size(), text.subjects().size(), wd.size(), C.pages.size());
    std::cout << b << "\n";

    // ================================================================ ROD: rediscovery on the full store
    auto deg = [&](const string& x) { return static_cast<double>(full.neighbours(x).size()); };
    const std::size_t nh = triples.size() / 10;
    vector<std::size_t> order(triples.size());
    std::iota(order.begin(), order.end(), 0);
    vector<std::size_t> hide_random = order, hide_least = order;
    {
        std::mt19937_64 rng(20261029);
        for (std::size_t i = hide_random.size(); i > 1; --i) std::swap(hide_random[i - 1], hide_random[rng() % i]);
        hide_random.resize(nh);
        std::stable_sort(hide_least.begin(), hide_least.end(), [&](std::size_t a, std::size_t c) {
            const double da = deg(triples[a][0]) + deg(triples[a][2]), dc = deg(triples[c][0]) + deg(triples[c][2]);
            if (da != dc) return da < dc;
            return fnv(triples[a][0] + '\x1f' + triples[a][1] + '\x1f' + triples[a][2]) < fnv(triples[c][0] + '\x1f' + triples[c][1] + '\x1f' + triples[c][2]);
        });
        hide_least.resize(nh);
    }
    struct RodSet {
        string name;
        ecs::TripleStore st;
        vector<Query> qs;
        std::unique_ptr<ecs::RelGraph> g;
        std::unique_ptr<ecs::PathRanker> pr;
        std::unique_ptr<ecs::LinkPredictor> lp;
    };
    RodSet sets[2];
    sets[0].name = "random", sets[1].name = "least";
    for (int k = 0; k < 2; ++k) {
        RodSet& S = sets[k];
        std::set<std::size_t> hid((k == 0 ? hide_random : hide_least).begin(), (k == 0 ? hide_random : hide_least).end());
        std::map<std::pair<string, string>, vector<string>> slots;  // (s, r) -> hidden objects
        for (std::size_t i = 0; i < triples.size(); ++i) {
            if (hid.count(i)) slots[{triples[i][0], triples[i][1]}].push_back(triples[i][2]);
            else S.st.add(triples[i][0], triples[i][1], triples[i][2]);
        }
        const auto objs = objects_by_rel(S.st);
        S.g = std::make_unique<ecs::RelGraph>(S.st);
        S.pr = std::make_unique<ecs::PathRanker>(S.st, *S.g, ecs::PathRanker::Config{});
        S.lp = std::make_unique<ecs::LinkPredictor>(S.st, ecs::LinkPredictor::Config{});
        const ecs::LinkScorer ls(S.st, {});
        const double tq = now_s();
        std::size_t reach = 0, tg = 0;
        for (const auto& [slot, os] : slots) {
            Query q;
            q.s = slot.first, q.r = slot.second;
            q.dev = ((fnv(q.s) >> 17) & 1) == 0;
            if (!q.dev && !read_test) continue;
            q.cands = candidates(S.st, objs, q.s, q.r);
            q.truth.assign(q.cands.size(), 0);
            for (const string& o : os) {
                Target t;
                t.value = o;
                const auto it = std::lower_bound(q.cands.begin(), q.cands.end(), o);
                if (it != q.cands.end() && *it == o) t.cands.push_back(static_cast<int>(it - q.cands.begin())), q.truth[static_cast<std::size_t>(t.cands[0])] = 1;
                reach += !t.cands.empty(), ++tg;
                q.targets.push_back(t);
            }
            score_query(q, S.st, ls, *S.pr);
            S.qs.push_back(std::move(q));
        }
        std::snprintf(b, sizeof b, "ROD %s: hidden %zu triples in %zu slots | reduced store %zu | scored %zu slots (%s) in %.1f s | reachable %zu of %zu targets",
                      S.name.c_str(), nh, slots.size(), S.st.size(), S.qs.size(), read_test ? "dev and test" : "dev only", now_s() - tq, reach, tg);
        std::cout << b << "\n";
    }
    // The combination: cross-fitted on dev (two folds by bit 18), then fitted on all dev for test and the other reads.
    auto fold = [](const Query& q) { return ((fnv(q.s) >> 18) & 1) != 0; };
    const ecs::Logistic combo_a = fit_combo(sets[0].qs, [&](const Query& q) { return q.dev && !fold(q); }, 11);
    const ecs::Logistic combo_b = fit_combo(sets[0].qs, [&](const Query& q) { return q.dev && fold(q); }, 12);
    const ecs::Logistic combo = fit_combo(sets[0].qs, [](const Query& q) { return q.dev; }, 13);
    if (combo.ok) {
        std::snprintf(b, sizeof b, "S8b COMBINATION weights (logit LP, log RA, G7, log base) %.4f %.4f %.4f %.4f intercept %.4f", combo.w[0], combo.w[1], combo.w[2],
                      combo.w[3], combo.b);
        std::cout << b << "\n";
    } else std::cout << "S8b COMBINATION not fitted\n";
    vector<const ecs::Logistic*> cross;
    for (const Query& q : sets[0].qs) cross.push_back(fold(q) ? &combo_a : &combo_b);
    auto rod_lines = [&](const string& tag, const std::array<GenRead, kNGen>& R) {
        for (int g = 0; g < kNGen; ++g) {
            const GenRead& G = R[static_cast<std::size_t>(g)];
            const CI h1 = boot(G.h1, 1), h10 = boot(G.h10, 2), mrr = boot(G.rr, 3);
            std::snprintf(b, sizeof b, "%s %-12s n %zu hits1 %.4f [%.4f, %.4f] hits10 %.4f [%.4f, %.4f] mrr %.4f [%.4f, %.4f]", tag.c_str(), kGenName[g], G.rr.size(),
                          h1.mean, h1.lo, h1.hi, h10.mean, h10.lo, h10.hi, mrr.mean, mrr.lo, mrr.hi);
            std::cout << b << "\n";
        }
    };
    // Per relation (dev, random hide; or the given split), every generator's Hits@10 and MRR.
    auto per_relation = [&](const string& tag, const vector<Query>& qs, bool dev, const ecs::Logistic* cmb, const vector<const ecs::Logistic*>* cq) {
        std::map<string, std::array<std::array<double, 3>, kNGen>> m;  // r -> gen -> {n, hits10, rr}
        for (std::size_t qi = 0; qi < qs.size(); ++qi) {
            const Query& q = qs[qi];
            if (q.dev != dev) continue;
            for (int g = 0; g < kNGen; ++g) {
                const auto p = primary(q, g, g == kCombo && cq ? (*cq)[qi] : cmb);
                for (const Target& t : q.targets) {
                    const double r = target_rank(q, t, p);
                    auto& a = m[q.r][static_cast<std::size_t>(g)];
                    a[0] += 1, a[1] += r > 0 && r <= 10 ? 1 : 0, a[2] += r > 0 ? 1 / r : 0;
                }
            }
        }
        for (const auto& [r, gs] : m) {
            string line = tag + " " + r + " n " + std::to_string(static_cast<int>(gs[0][0]));
            for (int g = 0; g < kNGen; ++g) {
                std::snprintf(b, sizeof b, " | %s h10 %.3f mrr %.3f", kGenName[g], gs[static_cast<std::size_t>(g)][1] / gs[static_cast<std::size_t>(g)][0],
                              gs[static_cast<std::size_t>(g)][2] / gs[static_cast<std::size_t>(g)][0]);
                line += b;
            }
            std::cout << line << "\n";
        }
    };
    const auto dev_random = evaluate(sets[0].qs, true, &combo, &cross);
    rod_lines("ROD random DEV", dev_random);
    per_relation("ROD_REL random DEV", sets[0].qs, true, &combo, &cross);
    // Choices on dev (random hide): G6's measure, then the generator for PROMO and S15; and the earn-its-place rule.
    auto mean = [](const vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    int g6 = kCN;
    for (int g : {kCN, kJac, kAA, kRA})
        if (mean(dev_random[static_cast<std::size_t>(g)].rr) > mean(dev_random[static_cast<std::size_t>(g6)].rr) + 1e-12) g6 = g;
    int chosen = kBase;
    for (int g : std::array<int, 5>{kBase, kLP, g6, kG7, kCombo})
        if (mean(dev_random[static_cast<std::size_t>(g)].rr) > mean(dev_random[static_cast<std::size_t>(chosen)].rr) + 1e-12) chosen = g;
    std::snprintf(b, sizeof b, "S8b CHOICE (dev MRR, random hide): G6 measure %s, generator %s", kGenName[g6], kGenName[chosen]);
    std::cout << b << "\n";
    for (int g : std::array<int, 4>{kLP, g6, kG7, kCombo}) {
        vector<double> d;
        const auto& A = dev_random[static_cast<std::size_t>(g)].rr;
        const auto& B = dev_random[kBase].rr;
        for (std::size_t i = 0; i < A.size(); ++i) d.push_back(A[i] - B[i]);
        const CI c = boot(d, 4);
        std::snprintf(b, sizeof b, "S8b EARN %-12s dev MRR minus base %.4f [%.4f, %.4f] -> %s", kGenName[g], c.mean, c.lo, c.hi, c.lo > 0 ? "earns its place" : "does not beat the base rate");
        std::cout << b << "\n";
    }
    const auto dev_least = evaluate(sets[1].qs, true, &combo);
    rod_lines("ROD least DEV", dev_least);
    // Dates: every generator's Hits@10 on the five date relations (L4).
    auto dates_line = [&](const string& tag, const vector<Query>& qs, bool dev, const ecs::Logistic* cmb, const vector<const ecs::Logistic*>* cq) {
        static const std::set<string> D = {"birth_year", "death_year", "birth_date", "death_date", "founded"};
        vector<Query> sub;
        vector<const ecs::Logistic*> csub;
        for (std::size_t qi = 0; qi < qs.size(); ++qi)
            if (D.count(qs[qi].r) && qs[qi].dev == dev) sub.push_back(qs[qi]), csub.push_back(cq ? (*cq)[qi] : cmb);
        const auto R = evaluate(sub, dev, cmb, &csub);
        string line = tag;
        for (int g = 0; g < kNGen; ++g) {
            std::snprintf(b, sizeof b, " | %s h10 %.4f", kGenName[g], mean(R[static_cast<std::size_t>(g)].h10));
            line += b;
        }
        std::cout << line << " (n " << R[0].h10.size() << ")\n";
    };
    dates_line("ROD_DATES random DEV", sets[0].qs, true, &combo, &cross);
    if (read_test) {
        rod_lines("ROD random TEST", evaluate(sets[0].qs, false, &combo));
        per_relation("ROD_REL random TEST", sets[0].qs, false, &combo, nullptr);
        rod_lines("ROD least TEST", evaluate(sets[1].qs, false, &combo));
        dates_line("ROD_DATES random TEST", sets[0].qs, false, &combo, nullptr);
    }
    {
        std::ofstream f(out + ".rod.tsv");
        f << "hide\tsplit\tsubject\trelation\tvalue\treachable\tcandidates";
        for (int g = 0; g < kNGen; ++g) f << '\t' << kGenName[g];
        f << "\n";
        for (int k = 0; k < 2; ++k)
            for (std::size_t qi = 0; qi < sets[k].qs.size(); ++qi) {
                const Query& q = sets[k].qs[qi];
                vector<vector<double>> ps;
                for (int g = 0; g < kNGen; ++g) ps.push_back(primary(q, g, g == kCombo && k == 0 && q.dev ? cross[qi] : &combo));
                for (const Target& t : q.targets) {
                    f << sets[k].name << '\t' << (q.dev ? "dev" : "test") << '\t' << q.s << '\t' << q.r << '\t' << t.value << '\t' << !t.cands.empty() << '\t' << q.cands.size();
                    for (int g = 0; g < kNGen; ++g) f << '\t' << target_rank(q, t, ps[static_cast<std::size_t>(g)]);
                    f << "\n";
                }
            }
    }

    // ================================================================ PROMO: the chosen generator's top candidates through S7's loop
    auto explain = [](const Query& q, int c, ecs::PathRanker& pr) {
        const auto* M = pr.model(q.r);
        const int k = q.g7_best.empty() ? -1 : q.g7_best[static_cast<std::size_t>(c)];
        return M && k >= 0 ? pr.path_name(M->paths[static_cast<std::size_t>(k)]) : string("-");
    };
    {
        RodSet& S = sets[0];
        vector<LoopItem> items;
        for (std::size_t qi = 0; qi < S.qs.size(); ++qi) {
            const Query& q = S.qs[qi];
            const auto p = primary(q, chosen, chosen == kCombo && q.dev ? cross[qi] : &combo);
            const int top = top_of(q, p);
            if (top < 0) continue;
            LoopItem h;
            h.s = q.s, h.r = q.r, h.o = q.cands[static_cast<std::size_t>(top)], h.dev = q.dev, h.score = p[static_cast<std::size_t>(top)];
            h.explain = explain(q, top, *S.pr);
            const int ri = rel_index(h.r);
            if (q.truth[static_cast<std::size_t>(top)]) h.truth = 1;
            else {
                const auto wit = wd.find(h.s + '\x1f' + h.r);
                if (wit != wd.end()) h.truth = s7::wd_match(ri, h.o, wit->second) ? 1 : 0;
            }
            items.push_back(h);
        }
        set_priors(items);
        const double tt = now_s();
        for (LoopItem& h : items)
            if (h.dev || read_test) run_tests(h, S.st, *S.lp, C);
        std::snprintf(b, sizeof b, "PROMO tests ran on %zu hypotheses in %.1f s (generator %s)", items.size(), now_s() - tt, kGenName[chosen]);
        std::cout << b << "\n";
        for (const bool dev : {true, false}) {
            if (!dev && !read_test) continue;
            vector<LoopItem*> batch;
            double judged = 0, top_true = 0;
            for (LoopItem& h : items)
                if (h.dev == dev) batch.push_back(&h), judged += h.truth >= 0, top_true += h.truth == 1;
            const LoopRead R = run_loop(batch, out + (dev ? ".promo_dev.prereg.tsv" : ".promo_test.prereg.tsv"));
            std::snprintf(b, sizeof b, "PROMO %s top candidates judged %.0f of %zu, true %.0f (precision of the top candidate %.3f)", dev ? "DEV" : "TEST", judged,
                          batch.size(), top_true, judged > 0 ? top_true / judged : 0.0);
            std::cout << b << "\n" << loop_line(string("PROMO ") + (dev ? "DEV" : "TEST"), R) << "\n";
            const bool measurable = R.judged_pro >= 10;
            std::snprintf(b, sizeof b, "PROMO %s GATE promotion precision %s", dev ? "DEV" : "TEST",
                          !measurable ? "not measurable (fewer than 10 judged promotions)" : R.true_pro / R.judged_pro >= 0.9 ? "met (>= 0.9)" : "missed (< 0.9)");
            std::cout << b << "\n";
        }
        std::ofstream f(out + ".promo.tsv");
        f << "split\tsubject\trelation\tobject\tscore\tprior\tT1\tT6\tT4\tposterior\tstate\tpromoted\ttruth\tpath\n";
        for (const LoopItem& h : items) {
            if (!h.dev && !read_test) continue;
            f << (h.dev ? "dev" : "test") << '\t' << h.s << '\t' << h.r << '\t' << h.o << '\t' << h.score << '\t' << h.prior << '\t' << ecs::test_outcome_name(h.out[0])
              << '\t' << ecs::test_outcome_name(h.out[1]) << '\t' << ecs::test_outcome_name(h.out[2]) << '\t' << h.post << '\t' << h.state << '\t' << h.promoted << '\t'
              << h.truth << '\t' << h.explain << "\n";
        }
        // ---- S8d: the chosen generator on S15 dev's confirm-half slots with their facts hidden (docs/reports/S8D_PROMOTION.md).
        // S15's labels are not read. Priors by PROMO's isotonic map (fitted on its dev items only); tests T1, T6, T4.
        // S10: --fresh-test-half takes the slots of S15 test's subjects with bit 17 = 1 instead.
        if (fresh_confirm || fresh_test) {
            const char* ftag = fresh_test ? "S10 FRESH" : "S8D FRESH";
            const char* fslots = fresh_test ? "test-half (bit 17 = 1)" : "confirm-half";
            std::set<std::pair<string, string>> cslots;
            for (const S15Row& x : load_s15(string(argv[7]) + (fresh_test ? "/temporal_holdout_test.tsv" : "/temporal_holdout_dev.tsv")))
                if (((fnv(x.s) >> 17) & 1) == 1 && rel_index(x.r) >= 0) cslots.insert({x.s, x.r});
            ecs::TripleStore fst;
            std::size_t held = 0;
            for (const auto& t : triples) {
                if (cslots.count({t[0], t[1]})) ++held;
                else fst.add(t[0], t[1], t[2]);
            }
            const auto fobjs = objects_by_rel(fst);
            const ecs::RelGraph fg(fst);
            ecs::PathRanker fpr(fst, fg, ecs::PathRanker::Config{});
            const ecs::LinkScorer fls(fst, {});
            const ecs::LinkPredictor flp(fst, ecs::LinkPredictor::Config{});
            vector<LoopItem> fit = items;  // set_priors fits on the dev items with a truth, so on PROMO's alone
            const std::size_t n0 = fit.size();
            std::size_t in_store = 0;
            for (const auto& [s, r] : cslots) {
                if (fst.facts_of(s).empty()) continue;
                ++in_store;
                Query q;
                q.s = s, q.r = r, q.dev = false;
                q.cands = candidates(fst, fobjs, s, r);
                if (q.cands.empty()) continue;
                q.truth.assign(q.cands.size(), 0);
                score_query(q, fst, fls, fpr);
                const auto p = primary(q, chosen, &combo);
                const int top = top_of(q, p);
                if (top < 0) continue;
                LoopItem h;
                h.s = s, h.r = r, h.o = q.cands[static_cast<std::size_t>(top)], h.dev = false, h.score = p[static_cast<std::size_t>(top)];
                h.explain = explain(q, top, fpr);
                fit.push_back(h);
            }
            set_priors(fit);
            vector<LoopItem> fh(fit.begin() + static_cast<std::ptrdiff_t>(n0), fit.end());
            for (LoopItem& h : fh) run_tests(h, fst, flp, C);
            vector<LoopItem*> batch;
            for (LoopItem& h : fh) batch.push_back(&h);
            const LoopRead R = run_loop(batch, out + ".fresh_links.prereg.tsv");
            std::snprintf(b, sizeof b, "%s G6/G7 (generator %s) %s slots %zu | store facts hidden %zu, reduced store %zu | slots whose subject is in "
                          "the store %zu | hypotheses %zu | supported %.0f, promoted after BH %.0f, refuted %.0f (no grade read) | log %zu entries verify %s",
                          ftag, kGenName[chosen], fslots, cslots.size(), held, fst.size(), in_store, fh.size(), R.sup, R.pro, R.ref, R.log_entries, R.log_ok ? "ok" : "BROKEN");
            std::cout << b << "\n";
            std::ofstream ff(out + ".fresh_links.hyps.tsv");
            ff << "split\tsubject\trelation\tobject\tscore\tprior\tT1\tT6\tT4\tposterior\tstate\tpromoted\tpath\n";
            for (const LoopItem& h : fh)
                ff << "fresh\t" << h.s << '\t' << h.r << '\t' << h.o << '\t' << h.score << '\t' << h.prior << '\t' << ecs::test_outcome_name(h.out[0]) << '\t'
                   << ecs::test_outcome_name(h.out[1]) << '\t' << ecs::test_outcome_name(h.out[2]) << '\t' << h.post << '\t' << h.state << '\t' << h.promoted << '\t'
                   << h.explain << "\n";
        }
    }

    // ================================================================ S15: the temporal holdout on the 2006-text store
    const string s15dir = argv[7];
    const vector<S15Row> s15dev = load_s15(s15dir + "/temporal_holdout_dev.tsv");
    const vector<S15Row> s15test = read_test ? load_s15(s15dir + "/temporal_holdout_test.tsv") : vector<S15Row>{};
    {
        const auto objs = objects_by_rel(text);
        std::map<string, vector<string>> names;  // r -> its objects (the candidates of an empty slot)
        std::map<string, ValueIndex> vidx;
        for (const auto& [r, os] : objs) {
            for (const auto& [o, n] : os) names[r].push_back(o);
            vidx[r].build(r, names[r]);
        }
        const ecs::RelGraph g(text);
        ecs::PathRanker pr(text, g, {});
        const ecs::LinkScorer ls(text, {});
        const ecs::LinkPredictor lp(text, {});
        vector<Query> qs;
        vector<vector<string>> falses;
        std::map<string, std::array<double, 2>> reach_cat;  // category -> {targets, reachable}
        for (const auto* rows : {&s15dev, &s15test}) {
            std::map<std::pair<string, string>, vector<const S15Row*>> slots;
            for (const S15Row& x : *rows) slots[{x.s, x.r}].push_back(&x);
            for (const auto& [slot, xs] : slots) {
                const auto oit = objs.find(slot.second);
                if (oit == objs.end() || text.facts_of(slot.first).empty() || text.has_relation(slot.first, slot.second)) continue;
                Query q;
                q.s = slot.first, q.r = slot.second, q.dev = rows == &s15dev;
                q.cands = candidates(text, objs, q.s, q.r);
                q.truth.assign(q.cands.size(), 0);
                const int ri = rel_index(q.r);
                vector<string> fv;
                for (const S15Row* x : xs) {
                    if (!x->label) { fv.push_back(x->value); continue; }
                    Target t;
                    t.value = x->value, t.cat = x->cat;
                    if (q.cands == names[q.r]) t.cands = vidx[q.r].find(x->value);  // the slot is empty: all of r's objects
                    else
                        for (std::size_t j = 0; j < q.cands.size(); ++j)
                            if (same_value(ri, x->value, q.cands[j])) t.cands.push_back(static_cast<int>(j));
                    for (int j : t.cands) q.truth[static_cast<std::size_t>(j)] = 1;
                    if (q.dev) reach_cat[t.cat][0] += 1, reach_cat[t.cat][1] += !t.cands.empty();
                    q.targets.push_back(t);
                }
                if (q.targets.empty()) continue;
                score_query(q, text, ls, pr);
                qs.push_back(std::move(q));
                falses.push_back(fv);
            }
        }
        for (const auto& [c, v] : reach_cat) {
            std::snprintf(b, sizeof b, "S15 DEV category %s targets %.0f reachable %.0f (%.3f)", c.c_str(), v[0], v[1], v[0] > 0 ? v[1] / v[0] : 0.0);
            std::cout << b << "\n";
        }
        auto cat_lines = [&](const string& tag, bool dev) {
            for (const string cat : {"a", "b", "c", "u", "*"}) {
                vector<Query> sub;
                for (const Query& q : qs) {
                    if (q.dev != dev) continue;
                    Query c = q;
                    c.targets.clear();
                    for (const Target& t : q.targets)
                        if (cat == "*" || t.cat == cat) c.targets.push_back(t);
                    if (!c.targets.empty()) sub.push_back(std::move(c));
                }
                const auto R = evaluate(sub, dev, &combo);
                for (int g : std::array<int, 5>{kBase, kLP, g6, kG7, kCombo}) {
                    const GenRead& G = R[static_cast<std::size_t>(g)];
                    vector<double> rr_r, h10_r;  // among reachable targets
                    std::size_t k = 0;
                    for (const Query& q : sub)
                        for (const Target& t : q.targets) {
                            if (!t.cands.empty()) rr_r.push_back(G.rr[k]), h10_r.push_back(G.h10[k]);
                            ++k;
                        }
                    const CI h10 = boot(G.h10, 5), h10r = boot(h10_r, 6);
                    std::snprintf(b, sizeof b, "%s category %s %-12s n %zu hits1 %.4f hits10 %.4f [%.4f, %.4f] mrr %.4f | reachable n %zu hits10 %.4f [%.4f, %.4f] mrr %.4f",
                                  tag.c_str(), cat.c_str(), kGenName[g], G.rr.size(), mean(G.h1), h10.mean, h10.lo, h10.hi, mean(G.rr), h10_r.size(), h10r.mean,
                                  h10r.lo, h10r.hi, mean(rr_r));
                    std::cout << b << "\n";
                }
            }
        };
        cat_lines("S15 DEV", true);
        if (read_test) cat_lines("S15 TEST", false);
        // Promotion on S15's slots: the chosen generator's top candidate, judged by S15.
        vector<LoopItem> items;
        for (std::size_t qi = 0; qi < qs.size(); ++qi) {
            const Query& q = qs[qi];
            const auto p = primary(q, chosen, &combo);
            const int top = top_of(q, p);
            if (top < 0) continue;
            LoopItem h;
            h.s = q.s, h.r = q.r, h.o = q.cands[static_cast<std::size_t>(top)], h.dev = q.dev, h.score = p[static_cast<std::size_t>(top)];
            h.truth = q.truth[static_cast<std::size_t>(top)] ? 1 : 0;
            h.explain = explain(q, top, pr);
            items.push_back(h);
        }
        set_priors(items);
        for (LoopItem& h : items) run_tests(h, text, lp, C);
        for (const bool dev : {true, false}) {
            if (!dev && !read_test) continue;
            vector<LoopItem*> batch;
            double top_true = 0;
            for (LoopItem& h : items)
                if (h.dev == dev) batch.push_back(&h), top_true += h.truth == 1;
            const LoopRead R = run_loop(batch, out + (dev ? ".s15_dev.prereg.tsv" : ".s15_test.prereg.tsv"));
            std::snprintf(b, sizeof b, "S15 %s PROMO top candidates %zu, true %.0f (%.3f)", dev ? "DEV" : "TEST", batch.size(), top_true,
                          batch.empty() ? 0.0 : top_true / static_cast<double>(batch.size()));
            std::cout << b << "\n" << loop_line(string("S15 ") + (dev ? "DEV" : "TEST") + " PROMO", R) << "\n";
        }
        std::ofstream f(out + ".s15.tsv");
        f << "split\tsubject\trelation\tvalue\tcategory\treachable\tcandidates";
        for (int g : std::array<int, 5>{kBase, kLP, g6, kG7, kCombo}) f << '\t' << kGenName[g];
        f << "\ttop\ttop_path\n";
        for (const Query& q : qs) {
            const auto pc = primary(q, chosen, &combo);
            const int top = top_of(q, pc);
            for (const Target& t : q.targets) {
                f << (q.dev ? "dev" : "test") << '\t' << q.s << '\t' << q.r << '\t' << t.value << '\t' << t.cat << '\t' << !t.cands.empty() << '\t' << q.cands.size();
                for (int g : std::array<int, 5>{kBase, kLP, g6, kG7, kCombo}) f << '\t' << target_rank(q, t, primary(q, g, &combo));
                f << '\t' << (top >= 0 ? q.cands[static_cast<std::size_t>(top)] : "") << '\t' << (top >= 0 ? explain(q, top, pr) : "-") << "\n";
            }
        }
        std::snprintf(b, sizeof b, "S15 queries %zu (dev %zu)", qs.size(), static_cast<std::size_t>(std::count_if(qs.begin(), qs.end(), [](const Query& q) { return q.dev; })));
        std::cout << b << "\n";
    }

    // ================================================================ GATE: S6's next change on S15's pairs
    {
        const auto fobjs = objects_by_rel(full);
        std::map<string, vector<string>> fnames;
        std::map<string, ValueIndex> fidx;
        for (const auto& [r, os] : fobjs) {
            for (const auto& [o, n] : os) fnames[r].push_back(o);
            fidx[r].build(r, fnames[r]);
        }
        // A value's store object for r: the most held object it names, s's own triple not counted (the first on a tie).
        auto map_value = [&](const string& r, const string& v, const string& s) -> string {
            const auto it = fobjs.find(r);
            if (it == fobjs.end()) return "";
            string best;
            std::size_t bn = 0;
            for (int i : fidx[r].find(v)) {
                const auto& [o, n] = it->second[static_cast<std::size_t>(i)];
                const std::size_t left = n - (full.has(s, r, o) ? 1 : 0);
                if (left >= 1 && left > bn) best = o, bn = left;
            }
            return best;
        };
        struct GPair {
            string s, r, ot, of;
            bool dev = true;
            double lp_t = 0, lp_f = 0, aa_t = 0, aa_f = 0, hard_t = 1, hard_f = 1, g7_t = 0, g7_f = 0, p0_t = 0, p0_f = 0, p1_t = 0, p1_f = 0;
        };
        vector<GPair> pairs;
        std::mt19937_64 rng(20261029);
        for (const auto* rows : {&s15dev, &s15test}) {
            std::map<string, const S15Row*> by_id;
            for (const S15Row& x : *rows) by_id[x.id] = &x;
            std::map<string, vector<GPair>> by_subj;
            for (const S15Row& x : *rows) {
                if (!x.label || rel_index(x.r) < 0 || !fobjs.count(x.r)) continue;
                const auto fit = by_id.find(x.pair);
                if (fit == by_id.end()) continue;
                bool other = false;
                for (const auto& f : full.facts_of(x.s)) other = other || f.r != x.r;
                if (!other) continue;
                GPair p;
                p.s = x.s, p.r = x.r, p.dev = rows == &s15dev;
                p.ot = map_value(x.r, x.value, x.s), p.of = map_value(x.r, fit->second->value, x.s);
                if (p.ot.empty() || p.of.empty() || p.ot == p.of) continue;
                by_subj[x.s].push_back(p);
            }
            for (auto& [s, ps] : by_subj) pairs.push_back(ps[rng() % ps.size()]);
        }
        std::set<std::pair<string, string>> removed;
        for (const GPair& p : pairs) removed.insert({p.s, p.r});
        ecs::TripleStore gs;
        for (const auto& t : triples)
            if (!removed.count({t[0], t[1]})) gs.add(t[0], t[1], t[2]);
        vector<GPair> kept;
        for (const GPair& p : pairs)
            if (gs.count(p.r, p.ot) >= 1 && gs.count(p.r, p.of) >= 1 && !gs.facts_of(p.s).empty()) kept.push_back(p);
        const ecs::RelGraph g(gs);
        ecs::PathRanker pr(gs, g, {});
        const ecs::LinkPredictor lp(gs, {});
        for (GPair& p : kept) {
            if (!p.dev && !read_test) continue;
            const int ri = rel_index(p.r);
            p.lp_t = lp.prob(p.s, p.r, p.ot), p.lp_f = lp.prob(p.s, p.r, p.of);
            p.aa_t = ecs::adamic_adar(gs, p.s, p.ot), p.aa_f = ecs::adamic_adar(gs, p.s, p.of);
            p.hard_t = s6::hard_constraint(gs, p.s, ri, p.ot), p.hard_f = s6::hard_constraint(gs, p.s, ri, p.of);
            const auto sc = pr.score(p.s, p.r, {p.ot, p.of});
            p.g7_t = sc[0], p.g7_f = sc[1];
        }
        std::size_t nd = 0, nt = 0;
        for (const GPair& p : kept) (p.dev ? nd : nt)++;
        std::snprintf(b, sizeof b, "GATE pairs drawn %zu (one per subject), kept %zu (dev %zu, test %zu) | reduced store %zu of %zu", pairs.size(), kept.size(), nd, nt,
                      gs.size(), full.size());
        std::cout << b << "\n";
        // z pools and isotonic fits on dev pairs.
        vector<double> vl, va, vg;
        for (const GPair& p : kept)
            if (p.dev) vl.push_back(logit_c(p.lp_t)), vl.push_back(logit_c(p.lp_f)), va.push_back(std::log1p(p.aa_t)), va.push_back(std::log1p(p.aa_f)), vg.push_back(p.g7_t),
                vg.push_back(p.g7_f);
        const auto zl = ecs::pool_stats(vl), za = ecs::pool_stats(va), zg = ecs::pool_stats(vg);
        auto raw0 = [&](double lpv, double aav) { return ecs::zscore(logit_c(lpv), zl) + ecs::zscore(std::log1p(aav), za); };
        auto raw1 = [&](double lpv, double aav, double g7v) { return raw0(lpv, aav) + ecs::zscore(g7v, zg); };
        ecs::IsotonicCalibrator c0, c1;
        {
            vector<double> x0, x1;
            vector<int> y;
            for (const GPair& p : kept)
                if (p.dev)
                    x0.push_back(raw0(p.lp_t, p.aa_t)), x1.push_back(raw1(p.lp_t, p.aa_t, p.g7_t)), y.push_back(1), x0.push_back(raw0(p.lp_f, p.aa_f)),
                        x1.push_back(raw1(p.lp_f, p.aa_f, p.g7_f)), y.push_back(0);
            c0.fit(x0, y), c1.fit(x1, y);
        }
        for (GPair& p : kept) {
            p.p0_t = ecs::plausibility_prior(1.0, c0.predict(raw0(p.lp_t, p.aa_t)), p.hard_t);
            p.p0_f = ecs::plausibility_prior(1.0, c0.predict(raw0(p.lp_f, p.aa_f)), p.hard_f);
            p.p1_t = ecs::plausibility_prior(1.0, c1.predict(raw1(p.lp_t, p.aa_t, p.g7_t)), p.hard_t);
            p.p1_f = ecs::plausibility_prior(1.0, c1.predict(raw1(p.lp_f, p.aa_f, p.g7_f)), p.hard_f);
        }
        for (const bool dev : {true, false}) {
            if (!dev && !read_test) continue;
            vector<std::pair<double, double>> v0, v1, vlp, vaa, vg7, vh;
            vector<double> dpair;
            for (const GPair& p : kept) {
                if (p.dev != dev) continue;
                v0.push_back({p.p0_t, p.p0_f}), v1.push_back({p.p1_t, p.p1_f}), vlp.push_back({p.lp_t, p.lp_f}), vaa.push_back({p.aa_t, p.aa_f});
                vg7.push_back({p.g7_t, p.g7_f}), vh.push_back({p.hard_t, p.hard_f});
            }
            const char* tag = dev ? "DEV" : "TEST";
            for (const auto& [name, v] : vector<std::pair<string, vector<std::pair<double, double>>>>{
                     {"P0_S6b", v0}, {"P0_prime", v1}, {"LP", vlp}, {"AA", vaa}, {"G7", vg7}, {"hard", vh}}) {
                const PairRead r = pair_auroc(v);
                std::snprintf(b, sizeof b, "GATE %s %-9s pairs %zu auroc %.4f [%.4f, %.4f]", tag, name.c_str(), v.size(), r.auc, r.lo, r.hi);
                std::cout << b << "\n";
            }
            // P0' minus P0: the same pairs, paired bootstrap.
            std::mt19937_64 rng2(20261028);
            vector<double> bs;
            auto flat = [](const vector<std::pair<double, double>>& v, const vector<std::size_t>& idx) {
                vector<std::pair<double, int>> f;
                for (std::size_t k : idx) f.push_back({v[k].first, 1}), f.push_back({v[k].second, 0});
                return auroc_of(f);
            };
            for (int rep = 0; rep < 2000 && !v0.empty(); ++rep) {
                vector<std::size_t> idx;
                for (std::size_t k = 0; k < v0.size(); ++k) idx.push_back(rng2() % v0.size());
                bs.push_back(flat(v1, idx) - flat(v0, idx));
            }
            std::sort(bs.begin(), bs.end());
            const double d = pair_auroc(v1).auc - pair_auroc(v0).auc;
            const PairRead g1 = pair_auroc(v1);
            std::snprintf(b, sizeof b, "GATE %s P0_prime_minus_P0 %.4f [%.4f, %.4f] | S6's gate (P0_prime >= 0.75): %s", tag, d, bs.empty() ? 0.0 : bs[49],
                          bs.empty() ? 0.0 : bs[1949], g1.auc >= 0.75 ? "met" : "missed");
            std::cout << b << "\n";
        }
        // S6c (the S6 report's "S6c"): S6b's P0 with G12's closed form added at weight w (chosen on S6's dev pairs); z pool
        // and isotonic fit on dev pairs, as above. Without --abc-w nothing here runs.
        if (abc_w > 0) {
            s6c::Text T;
            s6c::build(C.data, C.pages, T);
            vector<double> at(kept.size(), -1.0), af(kept.size(), -1.0), za_v;
            for (std::size_t k = 0; k < kept.size(); ++k) {
                if (!kept[k].dev && !read_test) continue;
                at[k] = s6c::score(T, kept[k].s, kept[k].ot), af[k] = s6c::score(T, kept[k].s, kept[k].of);
                if (kept[k].dev)
                    for (double v : {at[k], af[k]})
                        if (v >= 0) za_v.push_back(std::log1p(v));
            }
            const auto zab = ecs::pool_stats(za_v);
            auto zabc = [&](double v) { return v < 0 ? 0.0 : ecs::zscore(std::log1p(v), zab); };
            auto raw2 = [&](std::size_t k, bool t) { return t ? raw0(kept[k].lp_t, kept[k].aa_t) + abc_w * zabc(at[k]) : raw0(kept[k].lp_f, kept[k].aa_f) + abc_w * zabc(af[k]); };
            ecs::IsotonicCalibrator c2;
            {
                vector<double> x2;
                vector<int> y;
                for (std::size_t k = 0; k < kept.size(); ++k)
                    if (kept[k].dev) x2.push_back(raw2(k, true)), y.push_back(1), x2.push_back(raw2(k, false)), y.push_back(0);
                c2.fit(x2, y);
            }
            for (const bool dev : {true, false}) {
                if (!dev && !read_test) continue;
                vector<std::pair<double, double>> v0, v2, va;
                std::size_t app = 0;
                for (std::size_t k = 0; k < kept.size(); ++k) {
                    if (kept[k].dev != dev) continue;
                    const GPair& p = kept[k];
                    v0.push_back({p.p0_t, p.p0_f});
                    v2.push_back({ecs::plausibility_prior(1.0, c2.predict(raw2(k, true)), p.hard_t), ecs::plausibility_prior(1.0, c2.predict(raw2(k, false)), p.hard_f)});
                    app += (at[k] >= 0) + (af[k] >= 0);
                    if (at[k] >= 0 && af[k] >= 0) va.push_back({at[k], af[k]});
                }
                const char* tag = dev ? "DEV" : "TEST";
                const PairRead r0 = pair_auroc(v0), r2 = pair_auroc(v2), ra = pair_auroc(va);
                std::mt19937_64 rng3(20261028);
                vector<double> bs;
                auto flat = [](const vector<std::pair<double, double>>& v, const vector<std::size_t>& idx) {
                    vector<std::pair<double, int>> f;
                    for (std::size_t k : idx) f.push_back({v[k].first, 1}), f.push_back({v[k].second, 0});
                    return auroc_of(f);
                };
                for (int rep = 0; rep < 2000 && !v0.empty(); ++rep) {
                    vector<std::size_t> idx;
                    for (std::size_t k = 0; k < v0.size(); ++k) idx.push_back(rng3() % v0.size());
                    bs.push_back(flat(v2, idx) - flat(v0, idx));
                }
                std::sort(bs.begin(), bs.end());
                const double canary = s6c::p_gain(v0, v2, -0.01, 4000, 20261031);
                std::snprintf(b, sizeof b,
                              "S6C GATE %s pairs %zu | applicable %zu of %zu items | closed form alone on %zu pairs with both applicable %.4f [%.4f, %.4f] | P0 %.4f [%.4f, %.4f] | "
                              "P0_abc (w %.2f) %.4f [%.4f, %.4f] | minus %+.4f [%+.4f, %+.4f] | canary P(delta > -0.01) %.4f | S6's gate (P0_abc >= 0.75): %s",
                              tag, v0.size(), app, 2 * v0.size(), va.size(), ra.auc, ra.lo, ra.hi, r0.auc, r0.lo, r0.hi, abc_w, r2.auc, r2.lo, r2.hi, r2.auc - r0.auc,
                              bs.empty() ? 0.0 : bs[49], bs.empty() ? 0.0 : bs[1949], canary, r2.auc >= 0.75 ? "met" : "missed");
                std::cout << b << "\n";
            }
        }
        std::ofstream f(out + ".gate.tsv");
        f << "split\tsubject\trelation\ttrue\tfalse\tLP_t\tLP_f\tAA_t\tAA_f\thard_t\thard_f\tG7_t\tG7_f\tP0_t\tP0_f\tP0prime_t\tP0prime_f\n";
        for (const GPair& p : kept) {
            if (!p.dev && !read_test) continue;
            f << (p.dev ? "dev" : "test") << '\t' << p.s << '\t' << p.r << '\t' << p.ot << '\t' << p.of << '\t' << p.lp_t << '\t' << p.lp_f << '\t' << p.aa_t << '\t'
              << p.aa_f << '\t' << p.hard_t << '\t' << p.hard_f << '\t' << p.g7_t << '\t' << p.g7_f << '\t' << p.p0_t << '\t' << p.p0_f << '\t' << p.p1_t << '\t'
              << p.p1_f << "\n";
        }
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S8b seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s8b

// ============================================================ S8a: G2 rule induction and G14 contradiction resolution
// pattern_learn --gen-rules ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test | --t4-pattern]
// (DISCOVERY 5 G2 and G14, 7, 12.2; docs/reports/S8A_RULES_AND_CONTRADICTIONS.md). R1, rediscovery on store K (every seed
// row): 10% of its facts hidden at random, and separately the 10% least connected; rules (gen_rules.hpp) mined on the
// rest rank each hidden fact's slot, and their predictions go through S7's tests and loop. R2, the temporal holdout
// (S15) on store T (2006 text: the non-Wikidata seeds and the admitted facts): ranks of S15's unstated (b) and future (c)
// facts, and the loop's promotions graded by S15. G14 explains store T's contested slots, graded by S15. Dev by bit 11
// of FNV-1a(subject) (R1) or S15's dev file (R2); test only with --read-test. Noisy-or or maximum is chosen on R1's dev
// MRR. Nothing here answers a question: these are Cypha's own hypotheses (PLAN rule 10).
namespace s8a {

using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;
using Facts = std::map<string, vector<std::pair<string, string>>>;

// Two values of relation ri are the same: dates by year (year relations) or by compatible date; else D4's normalisation.
bool same_value(int ri, const string& a, const string& b) {
    if (a == b) return true;
    if (ri < 0) return false;
    if (is_datey(ri)) {
        const Date A = parse_date_str(a), B = parse_date_str(b);
        if (!A.ok() || !B.ok()) return false;
        return kRel[ri].type == T_YEAR ? A.y == B.y : date_compat(A, B);
    }
    const string ka = norm_ent(a);
    return !ka.empty() && ka == norm_ent(b);
}
bool holds_value(const ecs::RuleIndex& ix, const string& s, const string& r, const string& o) {
    const int ri = rel_index(r);
    for (const auto& v : ix.values(s, r))
        if (same_value(ri, v, o)) return true;
    return false;
}

ecs::TripleStore make_store(const Facts& by) {
    ecs::TripleStore st;
    for (const auto& [s, fs] : by)
        for (const auto& [r, o] : fs) st.add(s, r, o);
    return st;
}

// The tests of one hypothesis, in cost order: S7's T1, T6, T5, T4; T2 consequences and T3 held-out rules from the kept rules.
constexpr int kT = 6;
const char* const kTestName[kT] = {"T1 consistency", "T2 consequence", "T3 held-out", "T6 compression", "T5 Wikidata", "T4 corpus"};
const char* const kTestSource[kT] = {"store", "store", "rules", "store", "wikidata", "corpus"};
const bool kTestDisconf[kT] = {true, true, true, false, true, true};
const bool kTestIndep[kT] = {false, false, false, false, true, true};

struct Tests {
    TestOutcome t[kT] = {kInc, kInc, kInc, kInc, kInc, kInc};
    double score = 0, dl = 0;
    int rule = -1;
};

struct Ctx {
    int t4_variant = 1;  // 1: S7's pattern_or_comention (registered); 0: pattern only (--t4-pattern, exploratory, dev)
    const string* data = nullptr;
    const vector<PageRef>* pages = nullptr;
    const std::unordered_map<string, int>* page_of = nullptr;
    const s7::WdObjs* wd = nullptr;
    std::map<string, s7::PageScan> scans;  // (s, r) -> T4's reading of s's article
};

// T2 (DISCOVERY 7.1): positive when a consequence of h is in the store and none is contradicted; negative when a
// consequence from a rule with PCA >= 0.8 lands in a functional slot (phi >= 0.9) that holds only other values.
TestOutcome t2(const ecs::RuleIndex& ix, const ecs::RuleSet& rs, const string& s, const string& r, const string& o) {
    bool confirmed = false, contradicted = false;
    for (const auto& c : rs.consequences(s, r, o)) {
        const int ci = rel_index(c.r);
        if (ci < 0) continue;
        if (holds_value(ix, c.s, c.r, c.o)) confirmed = true;
        else if (c.pca >= 0.8 && kRel[ci].phi >= 0.9 && !ix.values(c.s, c.r).empty()) contradicted = true;
    }
    return contradicted ? kNeg : confirmed ? kPos : kInc;
}

Tests run_tests(Ctx& cx, const ecs::TripleStore& st, const ecs::RuleIndex& ix, const ecs::RuleSet& rs, const ecs::LinkPredictor& lp,
                const vector<ecs::RuleSet::Cand>& rc, const string& s, const string& r, const string& o, bool t5_eligible) {
    Tests T;
    const int ri = rel_index(r);
    T.t[0] = s7::t1(st, s, r, o);
    T.t[1] = t2(ix, rs, s, r, o);
    for (const auto& c : rc)
        if (same_value(ri, c.o, o)) {
            T.rule = c.best, T.score = c.score;
            break;
        }
    if (T.rule >= 0) T.t[2] = ecs::rule_t3(rs.rules()[static_cast<std::size_t>(T.rule)]);
    T.dl = ecs::compression_gain_bits(lp, st, s, r, o) - std::log2(std::max(1e-12, lp.prob(s, r, o)));
    T.t[3] = T.dl > 0 ? kPos : T.dl < 0 ? kNeg : kInc;  // S7's choice: epsilon 0
    if (t5_eligible) {
        const auto wit = cx.wd->find(s + '\x1f' + r);
        if (wit != cx.wd->end()) T.t[4] = s7::wd_match(ri, o, wit->second) ? kPos : kNeg;
    }
    const auto pit = cx.page_of->find(s);
    if (pit != cx.page_of->end()) {
        const string k = s + '\x1f' + r;
        auto sit = cx.scans.find(k);
        if (sit == cx.scans.end()) sit = cx.scans.emplace(k, s7::scan_rel(*cx.data, (*cx.pages)[static_cast<std::size_t>(pit->second)], ri)).first;
        TestOutcome out[2];
        s7::t4(sit->second, ri, o, out);
        T.t[5] = out[cx.t4_variant];  // S7's choice: pattern_or_comention
    }
    return T;
}

// A labelled pair member (s and c) or a generated hypothesis (label: 1 true, 0 false, -1 ungraded).
struct Item {
    string s, r, o;
    int label = -1, label2 = -1;  // the registered grader; the amended S15 grader (R1)
    Tests T;
    double p0 = 0, post = 0, e = 1;
    string state;
    bool supported = false, promoted73 = false, promoted_s7 = false;
};

ecs::TestTally tally(const vector<Item>& it, int k) {
    ecs::TestTally t;
    for (const Item& x : it)
        if (x.label >= 0) t.add(x.label == 1, x.T.t[k]);
    return t;
}

struct Plan {
    bool in[kT] = {true, true, true, true, true, true};
    double s[kT] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, c[kT] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
};

// s and c on labelled pairs (add-one), and S7's earn-its-place rule (applicable >= 5%, Youden's J interval above 0).
Plan plan_from(const vector<Item>& pairs, const string& tag, const bool allowed[kT], bool print_earn) {
    Plan P;
    char b[512];
    for (int k = 0; k < kT; ++k) {
        const auto t = tally(pairs, k);
        P.s[k] = (t.t_pos + 1) / (t.t_pos + t.t_neg + 2), P.c[k] = (t.f_neg + 1) / (t.f_pos + t.f_neg + 2);
        P.in[k] = allowed[k] && t.applicable() >= 0.05 && t.youden_lo() > 0.0;
        std::cout << s7::line_of(tag.c_str(), kTestName[k], t) << "\n";
        if (!print_earn) continue;
        std::snprintf(b, sizeof b, "%s EARN %-15s applicable %.3f J %+.3f [%+.3f, %+.3f] -> %s (add-one s %.4f c %.4f)", tag.c_str(), kTestName[k],
                      t.applicable(), t.youden(), t.youden_lo(), t.youden_hi(), !allowed[k] ? "not in this read" : P.in[k] ? "kept" : "cut", P.s[k], P.c[k]);
        std::cout << b << "\n";
    }
    return P;
}

// S7's loop over generated hypotheses: proposed, scored (P0), queued, pre-registered, tested cheapest first, decided. A
// HypothesisStore per chunk of 256 (its find is a linear scan); a decision does not depend on the chunk. Promotion:
// supported by decide, then kept by BH at q 0.1 over the generator's batch of decisions (DISCOVERY 7.3). S7's form, BH
// over every tested hypothesis, is computed beside it.
void run_loop(vector<Item>& hyps, const Plan& P, const string& tag, const string& log_path) {
    ecs::PreregLog log;
    std::size_t refused = 0;
    for (std::size_t c0 = 0; c0 < hyps.size(); c0 += 256) {
        ecs::HypothesisStore hs;
        for (std::size_t i = c0; i < std::min(hyps.size(), c0 + 256); ++i) {
            Item& x = hyps[i];
            ecs::Hypothesis h;
            h.id = tag + "-" + std::to_string(i);
            h.key = x.r + '|' + x.s + '|' + x.o;
            h.statement = x.s + " " + x.r + " " + x.o;
            h.generator = "G2";
            for (int k = 0; k < kT; ++k) {
                if (!P.in[k]) continue;
                ecs::HypTest t;
                t.kind = kTestName[k], t.source = kTestSource[k], t.predicted = "positive", t.refuted_if = "negative";
                t.disconfirmation = kTestDisconf[k], t.independent = kTestIndep[k];
                t.sensitivity = P.s[k], t.specificity = P.c[k];
                h.tests.push_back(t);
            }
            hs.propose(h);
            hs.score(h.id, x.p0);
            hs.transition(h.id, ecs::HypState::Queued, "scored");
            if (!log.register_plan(*hs.find(h.id))) {
                ++refused;
                continue;
            }
            std::size_t ti = 0;
            for (int k = 0; k < kT; ++k) {
                if (!P.in[k]) continue;
                if (hs.find(h.id)->state != ecs::HypState::Queued) break;
                hs.transition(h.id, ecs::HypState::UnderTest, "next test");
                if (!hs.record_outcome(h.id, ti++, x.T.t[k], &log)) ++refused;
            }
            const ecs::Hypothesis* hh = hs.find(h.id);
            x.post = hh->posterior, x.state = ecs::hyp_state_name(hh->state), x.e = ecs::e_value(*hh);
            x.supported = hh->state == ecs::HypState::Supported;
        }
    }
    vector<double> e_all, e_sup;
    vector<std::size_t> sup;
    for (std::size_t i = 0; i < hyps.size(); ++i) {
        e_all.push_back(hyps[i].e);
        if (hyps[i].supported) e_sup.push_back(hyps[i].e), sup.push_back(i);
    }
    const auto r_all = ecs::bh_reject_e(e_all, 0.1), r_sup = ecs::bh_reject_e(e_sup, 0.1);
    for (std::size_t i = 0; i < hyps.size(); ++i) hyps[i].promoted_s7 = hyps[i].supported && r_all[i];
    for (std::size_t j = 0; j < sup.size(); ++j) hyps[sup[j]].promoted73 = r_sup[j];
    std::ofstream(log_path) << log.serialize();
    std::cout << tag << " LOOP log entries " << log.entries().size() << " verify " << (log.verify() ? "ok" : "BROKEN") << " plans or results refused "
              << refused << "\n";
}

// Precision of a selection, graded items only, with Wilson 95%: by the registered grader (label) or the amended one
// (label2), over every relation or the non-lexical ones (amendment 2: exact values do not grade a taxonomy).
void print_precision(const string& tag, const char* what, const vector<Item>& h, const std::function<bool(const Item&)>& sel, int which = 0,
                     bool nonlex = false) {
    double n = 0, t = 0, f = 0, u = 0;
    for (const Item& x : h)
        if (sel(x) && !(nonlex && kRel[rel_index(x.r)].type == T_LEX)) {
            ++n;
            const int l = which ? x.label2 : x.label;
            if (l == 1) ++t;
            else if (l == 0) ++f;
            else ++u;
        }
    char b[512];
    std::snprintf(b, sizeof b, "%s PRECISION %-22s %s%s selected %6.0f graded %6.0f true %6.0f false %6.0f ungraded %6.0f precision %.3f [%.3f, %.3f]",
                  tag.c_str(), what, which ? "[S15 grader]" : "[registered]", nonlex ? "[non-lexical]" : "[all]", n, t + f, t, f, u,
                  t + f > 0 ? t / (t + f) : 0.0, ecs::wilson_lower(t, t + f), ecs::wilson_upper(t, t + f));
    std::cout << b << "\n";
}
void print_precisions(const string& tag, const vector<Item>& h, bool amended) {
    for (int which = 0; which < (amended ? 2 : 1); ++which)
        for (int nl = 0; nl < 2; ++nl) {
            print_precision(tag, "all_hypotheses", h, [](const Item&) { return true; }, which, nl);
            print_precision(tag, "supported(decide)", h, [](const Item& x) { return x.supported; }, which, nl);
            print_precision(tag, "promoted(7.3,gate)", h, [](const Item& x) { return x.promoted73; }, which, nl);
            print_precision(tag, "promoted(S7_form)", h, [](const Item& x) { return x.promoted_s7; }, which, nl);
        }
}
// The S15 grade of (s, r, o): 1 when S15 lists it, 0 when S15's false row has it or S15 lists other values, else -1.
int s15_grade(const std::map<string, vector<std::pair<string, int>>>& slot, int ri, const string& s, const string& r, const string& o) {
    const auto it = slot.find(s + '\x1f' + r);
    if (it == slot.end()) return -1;
    bool tr = false, fa = false, any = false;
    for (const auto& [v, label] : it->second) {
        if (label == 1) any = true;
        if (same_value(ri, v, o)) (label == 1 ? tr : fa) = true;
    }
    return tr ? 1 : (fa || any) ? 0 : -1;
}

void print_states(const string& tag, const vector<Item>& h) {
    std::map<string, std::array<double, 3>> m;  // state -> true, false, ungraded
    for (const Item& x : h) m[x.state][x.label == 1 ? 0 : x.label == 0 ? 1 : 2] += 1;
    for (const auto& [s, c] : m) {
        char b[256];
        std::snprintf(b, sizeof b, "%s STATE %-12s true %6.0f false %6.0f ungraded %6.0f", tag.c_str(), s.c_str(), c[0], c[1], c[2]);
        std::cout << b << "\n";
    }
}

// ---- ranking a hidden or held-out value in its slot (filtered setting)
enum Mode { M_RULES, M_BASE, M_LP };
const char* const kModeName[3] = {"G2_rules", "base_rate", "S6_link_predictor"};

// 1-based rank of the first candidate equal to target, skipping filtered values (they are other true values of the slot);
// 0 when no candidate equals it. Candidates: every object of r in the store and every rule candidate. Order: the mode's
// score, then the base rate (count), then the string.
std::size_t rank_value(const ecs::TripleStore& st, int ri, const string& s, const string& target, const vector<string>& objs,
                       const vector<ecs::RuleSet::Cand>& rc, const ecs::LinkPredictor* lp, Mode m, const std::function<bool(const string&)>& filtered) {
    const string r = kRel[ri].name;
    std::unordered_map<string, double> rsc;
    for (const auto& c : rc) rsc[c.o] = c.score;
    struct C {
        double a, b;
        const string* o;
    };
    vector<C> v;
    std::unordered_set<string> seen;
    auto add = [&](const string& o) {
        if (!seen.insert(o).second) return;
        double a = 0;
        if (m == M_RULES) {
            const auto it = rsc.find(o);
            a = it == rsc.end() ? 0.0 : it->second;
        } else if (m == M_LP) a = lp->prob(s, r, o);
        v.push_back({a, static_cast<double>(st.count(r, o)), &o});
    };
    for (const auto& c : rc) add(c.o);
    for (const auto& o : objs) add(o);
    std::sort(v.begin(), v.end(), [](const C& x, const C& y) { return x.a != y.a ? x.a > y.a : x.b != y.b ? x.b > y.b : *x.o < *y.o; });
    std::size_t pos = 0;
    for (const C& c : v) {
        const bool hit = same_value(ri, *c.o, target);
        if (!hit && filtered(*c.o)) continue;
        ++pos;
        if (hit) return pos;
    }
    return 0;
}

struct RankAcc {
    vector<double> h1, h10, rr, cov;
};
void rank_add(RankAcc& a, std::size_t rank, bool covered) {
    a.h1.push_back(rank == 1), a.h10.push_back(rank >= 1 && rank <= 10), a.rr.push_back(rank ? 1.0 / static_cast<double>(rank) : 0.0);
    a.cov.push_back(covered);
}
std::pair<double, double> boot_mean(const vector<double>& x) {
    if (x.empty()) return {0, 0};
    std::mt19937_64 rng(20261029);
    vector<double> m;
    for (int b = 0; b < 2000; ++b) {
        double s = 0;
        for (std::size_t i = 0; i < x.size(); ++i) s += x[rng() % x.size()];
        m.push_back(s / static_cast<double>(x.size()));
    }
    std::sort(m.begin(), m.end());
    return {m[49], m[1949]};
}
double mean(const vector<double>& x) {
    double s = 0;
    for (double v : x) s += v;
    return x.empty() ? 0.0 : s / static_cast<double>(x.size());
}
void print_rank(const string& tag, const RankAcc& a) {
    const auto c1 = boot_mean(a.h1), c10 = boot_mean(a.h10), cr = boot_mean(a.rr);
    char b[512];
    std::snprintf(b, sizeof b, "%s n %5zu hits1 %.3f [%.3f, %.3f] hits10 %.3f [%.3f, %.3f] mrr %.3f [%.3f, %.3f] rule_covered %.3f", tag.c_str(), a.h1.size(),
                  mean(a.h1), c1.first, c1.second, mean(a.h10), c10.first, c10.second, mean(a.rr), cr.first, cr.second, mean(a.cov));
    std::cout << b << "\n";
}

// Objects of each relation in a store, by base rate (count, descending), then string.
std::map<string, vector<string>> objects_by_rel(const ecs::TripleStore& st, const Facts& by) {
    std::map<string, std::map<string, std::size_t>> c;
    for (const auto& [s, fs] : by)
        for (const auto& [r, o] : fs) c[r][o] = st.count(r, o);
    std::map<string, vector<string>> out;
    for (auto& [r, m] : c) {
        vector<std::pair<std::size_t, string>> v;
        for (auto& [o, n] : m) v.push_back({n, o});
        std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
        for (auto& [n, o] : v) out[r].push_back(o);
    }
    return out;
}

void write_rules(const string& path, const vector<ecs::Rule>& rules) {
    std::ofstream f(path);
    f << "kept\tkind\trule\tsupport\tbody\tpca_body\tstd_conf\tpca_conf\thead_coverage\tp0\tp_value\thidden_support\thidden_pca\n";
    for (const auto& R : rules)
        f << R.kept << '\t' << ecs::rule_kind_name(R.kind) << '\t' << R.text() << '\t' << R.support << '\t' << R.body << '\t' << R.pca_body << '\t' << R.std_conf()
          << '\t' << R.pca_conf() << '\t' << R.head_coverage() << '\t' << R.p0 << '\t' << R.p_value << '\t' << R.hid_support << '\t' << R.hid_pca << '\n';
}
void print_rule_counts(const string& tag, const vector<ecs::Rule>& rules) {
    std::map<string, std::array<std::size_t, 2>> c;
    for (const auto& R : rules) c[ecs::rule_kind_name(R.kind)][R.kept]++;
    for (const auto& [k, v] : c) std::cout << tag << " RULES " << k << " candidates " << v[0] + v[1] << " kept " << v[1] << "\n";
    std::vector<const ecs::Rule*> top;
    for (const auto& R : rules)
        if (R.kept) top.push_back(&R);
    std::sort(top.begin(), top.end(), [](const ecs::Rule* a, const ecs::Rule* b) {
        return a->support * a->pca_conf() != b->support * b->pca_conf() ? a->support * a->pca_conf() > b->support * b->pca_conf() : a->text() < b->text();
    });
    for (std::size_t i = 0; i < top.size() && i < 12; ++i) {
        char b[600];
        std::snprintf(b, sizeof b, "%s TOP %2zu %-8s support %5.0f pca %.3f std %.3f hc %.3f | %s", tag.c_str(), i + 1, ecs::rule_kind_name(top[i]->kind), top[i]->support,
                      top[i]->pca_conf(), top[i]->std_conf(), top[i]->head_coverage(), top[i]->text().c_str());
        std::cout << b << "\n";
    }
    for (const auto& R : rules)
        if (R.kept && R.kind != ecs::RuleKind::Const) {
            char b[600];
            std::snprintf(b, sizeof b, "%s VARRULE %-8s support %5.0f pca %.3f std %.3f | %s", tag.c_str(), ecs::rule_kind_name(R.kind), R.support, R.pca_conf(),
                          R.std_conf(), R.text().c_str());
            std::cout << b << "\n";
        }
}

// ---- S15
struct S15Row {
    string s, r, v, cat;
    int label = 0;
};
vector<S15Row> load_s15(const string& path) {
    vector<S15Row> out;
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    const auto h = split(l, '\t');
    auto col = [&](const char* n) { return static_cast<std::size_t>(std::find(h.begin(), h.end(), string(n)) - h.begin()); };
    const std::size_t cs = col("subject"), cr = col("relation"), cv = col("value"), cc = col("category"), cl = col("label");
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < h.size()) continue;
        out.push_back({c[cs], c[cr], c[cv], c[cc], std::atoi(c[cl].c_str())});
    }
    return out;
}

// ---- G14: a place contains another through store K's located_in, country or location chain (<= 3 steps), or holds it
// as its capital or largest city.
bool contains(const ecs::TripleStore& K, const string& outer, const string& inner) {
    const string ko = norm_ent(outer);
    if (ko.empty()) return false;
    for (const auto& f : K.facts_of(outer))
        if ((f.r == "capital" || f.r == "largest_city") && norm_ent(f.o) == norm_ent(inner)) return true;
    std::set<string> seen{inner};
    vector<string> frontier{inner};
    for (int d = 0; d < 3 && !frontier.empty(); ++d) {
        vector<string> next;
        for (const auto& n : frontier)
            for (const auto& f : K.facts_of(n)) {
                if (f.r != "located_in" && f.r != "country" && f.r != "location") continue;
                if (norm_ent(f.o) == ko) return true;
                if (seen.insert(f.o).second) next.push_back(f.o);
            }
        frontier.swap(next);
    }
    return false;
}
long days_of(const Date& D) {  // days from civil (proleptic Gregorian), for the gap of two day-precision dates
    long y = D.y, m = D.m, d = D.d;
    y -= m <= 2;
    const long era = (y >= 0 ? y : y - 399) / 400;
    const long yoe = y - era * 400, doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1, doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

int run(int argc, char** argv) {
    if (argc < 9) {
        std::fprintf(stderr, "usage: pattern_learn --gen-rules ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test] [--fresh-confirm | --fresh-test-half]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[8], s15dir = argv[7];
    bool read_test = false, fresh_confirm = false;  // --fresh-confirm: S8d's fresh G2 set (docs/reports/S8D_PROMOTION.md)
    bool fresh_test = false;  // --fresh-test-half: S10's gate set, the same on S15 test's bit-17 half (docs/reports/S10_CURIOSITY_SCHEDULER.md)
    int t4_variant = 1;
    for (int i = 9; i < argc; ++i) {
        if (string(argv[i]) == "--read-test") read_test = true;
        else if (string(argv[i]) == "--fresh-confirm") fresh_confirm = true;
        else if (string(argv[i]) == "--fresh-test-half") fresh_test = true;
        else if (string(argv[i]) == "--t4-pattern") t4_variant = 0;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    char b[1024];
    // ---- the article text, its type lexicons and T4's patterns (as S7's --test-kinds)
    string data;
    if (!read_file(argv[2], data)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    g_dump = &data;
    long total = 0, redirects = 0;
    const vector<PageRef> pages = list_pages(data, total, redirects);
    std::unordered_map<string, int> page_of;
    for (std::size_t i = 0; i < pages.size(); ++i) page_of[pages[i].title] = static_cast<int>(i);
    for (const PageRef& pg : pages) {
        PageFlags fl;
        scan_page(data, pg, fl);
        const string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    {
        long ns = 0, ng = 0;
        if (!load_wikidata(argv[5], false, page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", argv[5]); return 1; }
    }
    for (const Seed& sd : g_seeds) {
        const RT ty = kRel[sd.rel].type;
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    {
        std::ifstream f(argv[6]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 11 || c[10] != "1" || rel_index(c[0]) < 0) continue;
            Pat p;
            p.rel = rel_index(c[0]);
            p.key = c[1];
            p.family = family_of(p.key);
            p.kept = true;
            g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
            g_pats.push_back(p);
        }
    }
    // ---- store K (every seed row) and store T (non-Wikidata seed rows and the admitted facts), with T's source counts
    Facts kby, tby;
    std::set<string> wd_src, kseen, tseen;
    std::map<string, double> t_sources;  // s r o -> rows supporting it in T
    {
        std::ifstream f(argv[3]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            const string k3 = c[0] + '\x1f' + c[1] + '\x1f' + c[2];
            if (c[3] == "wikidata") wd_src.insert(c[0] + '\x1f' + c[1]);
            if (kseen.insert(k3).second) kby[c[0]].push_back({c[1], c[2]});
            if (c[3] == "wikidata") continue;
            t_sources[k3] += 1;
            if (tseen.insert(k3).second) tby[c[0]].push_back({c[1], c[2]});
        }
    }
    {
        std::ifstream f(argv[4]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 3 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            const string k3 = c[0] + '\x1f' + c[1] + '\x1f' + c[2];
            t_sources[k3] += 1;
            if (tseen.insert(k3).second) tby[c[0]].push_back({c[1], c[2]});
        }
    }
    for (auto* by : {&kby, &tby})
        for (auto& [s, fs] : *by) std::sort(fs.begin(), fs.end());
    const ecs::TripleStore K = make_store(kby), Tst = make_store(tby);
    const s7::WdObjs wd = s7::load_wd_objs(argv[5]);
    std::snprintf(b, sizeof b, "S8A STORES K %zu triples %zu subjects | T %zu triples %zu subjects", K.size(), kby.size(), Tst.size(), tby.size());
    std::cout << b << "\n";
    auto dev_subject = [](const string& s) { return ((fnv(s) >> 11) & 1) == 0; };
    // S15's rows by slot, for amendment 1's grader: dev always, test only with --read-test
    std::map<string, vector<std::pair<string, int>>> s15_slot[2];
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
        for (const auto& R : load_s15(s15dir + (sp == 0 ? "/temporal_holdout_dev.tsv" : "/temporal_holdout_test.tsv")))
            s15_slot[sp][R.s + '\x1f' + R.r].push_back({R.v, R.label});

    // ============================================================ R1: rediscovery on store K
    vector<std::array<string, 3>> trip;
    for (const auto& [s, fs] : kby)
        for (const auto& [r, o] : fs) trip.push_back({s, r, o});
    const std::size_t n_hide = trip.size() / 10;
    vector<std::size_t> h_rand(trip.size()), h_low(trip.size());
    for (std::size_t i = 0; i < trip.size(); ++i) h_rand[i] = h_low[i] = i;
    {
        std::mt19937_64 rng(20261029);
        for (std::size_t i = h_rand.size(); i > 1; --i) std::swap(h_rand[i - 1], h_rand[rng() % i]);
        vector<std::pair<std::size_t, std::uint64_t>> key(trip.size());
        for (std::size_t i = 0; i < trip.size(); ++i)
            key[i] = {K.neighbours(trip[i][0]).size() + K.neighbours(trip[i][2]).size(), fnv(trip[i][0] + '\x1f' + trip[i][1] + '\x1f' + trip[i][2])};
        std::sort(h_low.begin(), h_low.end(), [&](std::size_t a, std::size_t c) { return key[a] < key[c]; });
    }
    bool noisy_or = true;  // set by the random hide's dev MRR
    for (int proto = 0; proto < 2; ++proto) {
        const string P = proto == 0 ? "random" : "least_connected";
        const auto& order = proto == 0 ? h_rand : h_low;
        std::set<std::size_t> hidden(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_hide));
        Facts red;
        for (std::size_t i = 0; i < trip.size(); ++i)
            if (!hidden.count(i)) red[trip[i][0]].push_back({trip[i][1], trip[i][2]});
        const ecs::TripleStore st = make_store(red);
        const ecs::RuleIndex ix(st);
        const double tm = now_s();
        const vector<ecs::Rule> rules = ecs::mine_rules(ix);
        std::snprintf(b, sizeof b, "R1 %s hidden %zu reduced store %zu mined in %.1f s", P.c_str(), hidden.size(), st.size(), now_s() - tm);
        std::cout << b << "\n";
        print_rule_counts("R1 " + P, rules);
        write_rules(out + ".r1_" + P + ".rules.tsv", rules);
        const ecs::LinkPredictor lp(st, {});
        const auto objs = objects_by_rel(st, red);
        static const vector<string> none;
        // the ranking reads, both combination rules on dev for the choice
        for (int pass = 0; pass < (proto == 0 ? 2 : 1); ++pass) {
            const bool nor = proto == 0 ? pass == 0 : noisy_or;
            const ecs::RuleSet rs(ix, rules, nor);
            RankAcc acc[2][3];  // split x mode
            for (std::size_t i : hidden) {
                const auto& [s, r, o] = trip[i];
                const bool dev = dev_subject(s);
                if (!dev && !read_test) continue;
                const int ri = rel_index(r);
                const auto rc = rs.candidates(s, r);
                bool covered = false;
                for (const auto& c : rc) covered = covered || same_value(ri, c.o, o);
                auto filt = [&](const string& c) {
                    for (const auto& f : K.facts_of(s))
                        if (f.r == r && same_value(ri, f.o, c)) return true;
                    return false;
                };
                const auto it = objs.find(r);
                for (int m = 0; m < 3; ++m) {
                    if (pass == 1 && m != M_RULES) continue;
                    rank_add(acc[dev ? 0 : 1][m], rank_value(st, ri, s, o, it == objs.end() ? none : it->second, rc, &lp, static_cast<Mode>(m), filt), covered);
                }
            }
            for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
                for (int m = 0; m < 3; ++m) {
                    if (acc[sp][m].h1.empty()) continue;
                    print_rank(string(sp == 0 ? "DEV" : "TEST") + " R1 RANK " + P + " " + kModeName[m] + (m == M_RULES ? (nor ? "[noisy_or]" : "[max]") : ""), acc[sp][m]);
                }
            if (proto == 0) {
                static double mrr_dev[2];
                mrr_dev[pass] = mean(acc[0][M_RULES].rr);
                if (pass == 1) {
                    noisy_or = mrr_dev[0] >= mrr_dev[1];
                    std::snprintf(b, sizeof b, "S8A CHOICE (R1 random, dev MRR): noisy-or %.4f max %.4f -> %s", mrr_dev[0], mrr_dev[1], noisy_or ? "noisy-or" : "max");
                    std::cout << b << "\n";
                }
            }
        }
        if (proto != 0) continue;
        // ---- R1 promotions (random hide): s and c pairs, then G2's hypotheses through the loop
        const ecs::RuleSet rs(ix, rules, noisy_or);
        Ctx cx;
        cx.data = &data, cx.pages = &pages, cx.page_of = &page_of, cx.wd = &wd, cx.t4_variant = t4_variant;
        std::unordered_map<string, vector<string>> pool;
        for (const auto& [s, fs] : red)
            for (const auto& [r, o] : fs) pool[r].push_back(o);
        std::mt19937_64 rngf(20261030);
        vector<Item> pairs[2];
        for (std::size_t i : hidden) {
            const auto& [s, r, o] = trip[i];
            const int ri = rel_index(r);
            const bool dev = dev_subject(s);
            const auto& pl = pool[r];
            if (pl.empty()) continue;
            const Date hd = is_datey(ri) ? parse_date_str(o) : Date{};
            string neg;
            for (int t = 0; t < 100 && neg.empty(); ++t) {
                const string& c = pl[rngf() % pl.size()];
                if (c == o || c == s || K.has(s, r, c)) continue;
                if (is_datey(ri)) {
                    const Date cd = parse_date_str(c);
                    if (cd.ok() && hd.ok() && cd.y == hd.y) continue;
                }
                neg = c;
            }
            if (neg.empty() || (!dev && !read_test)) continue;
            const bool t5 = !wd_src.count(s + '\x1f' + r);
            const auto rc = rs.candidates(s, r);
            Item a, x;
            a.s = x.s = s, a.r = x.r = r, a.o = o, x.o = neg, a.label = 1, x.label = 0;
            const auto wit = wd.find(s + '\x1f' + r);
            if (wit != wd.end() && s7::wd_match(ri, neg, wit->second)) x.label = -1;  // Wikidata lists it: not known false
            a.T = run_tests(cx, st, ix, rs, lp, rc, s, r, o, t5);
            x.T = run_tests(cx, st, ix, rs, lp, rc, s, r, neg, t5);
            pairs[dev ? 0 : 1].push_back(a), pairs[dev ? 0 : 1].push_back(x);
        }
        const bool all6[kT] = {true, true, true, true, true, true};
        const Plan PL = plan_from(pairs[0], "DEV R1 PAIRS", all6, true);
        if (read_test) plan_from(pairs[1], "TEST R1 PAIRS", all6, false);
        for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
            const string tag = string(sp == 0 ? "DEV" : "TEST") + " R1";
            vector<Item> hyps;
            const double th = now_s();
            for (const auto& s : ix.subjects()) {
                if (dev_subject(s) != (sp == 0)) continue;
                for (const auto& r : ix.relations()) {
                    const auto rc = rs.candidates(s, r);
                    if (rc.empty()) continue;
                    const int ri = rel_index(r);
                    for (const auto& c : rc) {
                        if (holds_value(ix, s, r, c.o)) continue;
                        Item x;
                        x.s = s, x.r = r, x.o = c.o;
                        bool tr = false, any = false;
                        for (const auto& f : K.facts_of(s))
                            if (f.r == r) any = true, tr = tr || same_value(ri, f.o, c.o);
                        x.label = tr ? 1 : any ? 0 : -1;
                        x.T = run_tests(cx, st, ix, rs, lp, rc, s, r, c.o, !wd_src.count(s + '\x1f' + r));
                        x.p0 = std::min(0.98, std::max(0.02, c.score * s6::hard_constraint(st, s, ri, c.o)));
                        hyps.push_back(std::move(x));
                    }
                }
            }
            std::snprintf(b, sizeof b, "%s HYPOTHESES %zu tested in %.1f s", tag.c_str(), hyps.size(), now_s() - th);
            std::cout << b << "\n";
            run_loop(hyps, PL, tag, out + (sp == 0 ? ".r1_dev" : ".r1_test") + ".prereg.tsv");
            // amendment 1: the S15 grader (K or S15 lists it; S15's false row or other S15 values)
            for (Item& x : hyps) {
                const int ri = rel_index(x.r);
                bool k_true = false;
                for (const auto& f : K.facts_of(x.s))
                    if (f.r == x.r && same_value(ri, f.o, x.o)) k_true = true;
                int g = s15_grade(s15_slot[0], ri, x.s, x.r, x.o);
                if (g < 0 && sp == 1) g = s15_grade(s15_slot[1], ri, x.s, x.r, x.o);
                x.label2 = k_true ? 1 : g;
            }
            print_states(tag, hyps);
            print_precisions(tag, hyps, true);
            std::ofstream f(out + (sp == 0 ? ".r1_dev" : ".r1_test") + ".hyps.tsv");
            f << "subject\trelation\tobject\tlabel\tlabel_s15\tP0\tposterior\tstate\tpromoted\tT1\tT2\tT3\tT6\tT5\tT4\n";
            for (const Item& x : hyps)
                f << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.label << '\t' << x.label2 << '\t' << x.p0 << '\t' << x.post << '\t' << x.state << '\t' << x.promoted73 << '\t'
                  << ecs::test_outcome_name(x.T.t[0]) << '\t' << ecs::test_outcome_name(x.T.t[1]) << '\t' << ecs::test_outcome_name(x.T.t[2]) << '\t'
                  << ecs::test_outcome_name(x.T.t[3]) << '\t' << ecs::test_outcome_name(x.T.t[4]) << '\t' << ecs::test_outcome_name(x.T.t[5]) << '\n';
        }
        // ---- S8d: G2 on S15 dev's confirm-half slots with their facts hidden (docs/reports/S8D_PROMOTION.md). S15's labels
        // are not read; R1's plan and combination rule; T5 left out (Wikidata is what S15 grades against).
        // S10: --fresh-test-half takes the slots of S15 test's subjects with bit 17 = 1 instead.
        if (fresh_confirm || fresh_test) {
            const string ftag = fresh_test ? "S10 FRESH" : "S8D FRESH";
            std::set<std::pair<string, string>> cslots;
            for (const auto& R : load_s15(s15dir + (fresh_test ? "/temporal_holdout_test.tsv" : "/temporal_holdout_dev.tsv")))
                if (((fnv(R.s) >> 17) & 1) == 1 && rel_index(R.r) >= 0) cslots.insert({R.s, R.r});
            Facts fred;
            std::size_t held = 0;
            for (const auto& [s, fs] : kby)
                for (const auto& [r, o] : fs) {
                    if (cslots.count({s, r})) ++held;
                    else fred[s].push_back({r, o});
                }
            const ecs::TripleStore fst = make_store(fred);
            const ecs::RuleIndex fix(fst);
            const vector<ecs::Rule> frules = ecs::mine_rules(fix);
            const ecs::RuleSet frs(fix, frules, noisy_or);
            const ecs::LinkPredictor flp(fst, {});
            Ctx fcx;
            fcx.data = &data, fcx.pages = &pages, fcx.page_of = &page_of, fcx.wd = &wd, fcx.t4_variant = t4_variant;
            vector<Item> fh;
            std::size_t in_store = 0, with_cand = 0;
            for (const auto& [s, r] : cslots) {
                if (fst.facts_of(s).empty()) continue;
                ++in_store;
                const auto rc = frs.candidates(s, r);
                if (rc.empty()) continue;
                ++with_cand;
                const int ri = rel_index(r);
                for (const auto& c : rc) {
                    if (holds_value(fix, s, r, c.o)) continue;
                    Item x;
                    x.s = s, x.r = r, x.o = c.o;
                    x.T = run_tests(fcx, fst, fix, frs, flp, rc, s, r, c.o, false);
                    x.p0 = std::min(0.98, std::max(0.02, c.score * s6::hard_constraint(fst, s, ri, c.o)));
                    fh.push_back(std::move(x));
                }
            }
            std::snprintf(b, sizeof b, "%s G2 %s slots %zu | store facts hidden %zu, reduced store %zu | slots whose subject is in the store %zu, "
                          "with a rule candidate %zu | hypotheses %zu", ftag.c_str(), fresh_test ? "test-half (bit 17 = 1)" : "confirm-half", cslots.size(), held, fst.size(), in_store, with_cand, fh.size());
            std::cout << b << "\n";
            run_loop(fh, PL, ftag + " G2", out + ".fresh_g2.prereg.tsv");
            std::map<string, std::size_t> states;
            std::size_t promoted = 0;
            for (const Item& x : fh) ++states[x.state], promoted += x.promoted73;
            string line = ftag + " G2 states (no grade read):";
            for (const auto& [s, n] : states) line += " " + s + " " + std::to_string(n);
            std::cout << line << " | promoted (BH among supported) " << promoted << "\n";
            std::ofstream f(out + ".fresh_g2.hyps.tsv");
            f << "split\tsubject\trelation\tobject\tP0\tposterior\tstate\tpromoted\tT1\tT2\tT3\tT6\tT5\tT4\n";
            for (const Item& x : fh)
                f << "fresh\t" << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.p0 << '\t' << x.post << '\t' << x.state << '\t' << x.promoted73 << '\t'
                  << ecs::test_outcome_name(x.T.t[0]) << '\t' << ecs::test_outcome_name(x.T.t[1]) << '\t' << ecs::test_outcome_name(x.T.t[2]) << '\t'
                  << ecs::test_outcome_name(x.T.t[3]) << '\t' << ecs::test_outcome_name(x.T.t[4]) << '\t' << ecs::test_outcome_name(x.T.t[5]) << '\n';
        }
    }

    // ============================================================ R2: the temporal holdout (S15) on store T
    const ecs::RuleIndex ixT(Tst);
    const vector<ecs::Rule> rulesT = ecs::mine_rules(ixT);
    print_rule_counts("R2 T", rulesT);
    write_rules(out + ".r2_T.rules.tsv", rulesT);
    const ecs::RuleSet rsT(ixT, rulesT, noisy_or);
    const ecs::LinkPredictor lpT(Tst, {});
    const auto objsT = objects_by_rel(Tst, tby);
    Ctx cxT;
    cxT.data = &data, cxT.pages = &pages, cxT.page_of = &page_of, cxT.wd = &wd, cxT.t4_variant = t4_variant;
    if (read_test && t4_variant == 0) {
        std::fprintf(stderr, "--t4-pattern is exploratory: dev only\n");
        return 2;
    }
    const bool no5[kT] = {true, true, true, true, false, true};
    Plan PT;
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
        const string tag = string(sp == 0 ? "DEV" : "TEST") + " R2";
        const vector<S15Row> rows = load_s15(s15dir + (sp == 0 ? "/temporal_holdout_dev.tsv" : "/temporal_holdout_test.tsv"));
        std::map<string, vector<const S15Row*>> slot;  // s r -> rows
        std::set<string> subj;
        for (const auto& R : rows) slot[R.s + '\x1f' + R.r].push_back(&R), subj.insert(R.s);
        // ranking of b and c facts
        RankAcc acc[2];  // b, c
        static const vector<string> none;
        for (const auto& R : rows) {
            if (R.label != 1 || (R.cat != "b" && R.cat != "c")) continue;
            const int ri = rel_index(R.r);
            if (ri < 0 || Tst.facts_of(R.s).empty() || holds_value(ixT, R.s, R.r, R.v)) continue;
            const auto rc = rsT.candidates(R.s, R.r);
            bool covered = false;
            for (const auto& c : rc) covered = covered || same_value(ri, c.o, R.v);
            const auto& sl = slot[R.s + '\x1f' + R.r];
            auto filt = [&](const string& c) {
                if (holds_value(ixT, R.s, R.r, c)) return true;
                for (const S15Row* o2 : sl)
                    if (o2->label == 1 && same_value(ri, o2->v, c)) return true;
                return false;
            };
            const auto it = objsT.find(R.r);
            rank_add(acc[R.cat == "b" ? 0 : 1], rank_value(Tst, ri, R.s, R.v, it == objsT.end() ? none : it->second, rc, &lpT, M_RULES, filt), covered);
        }
        print_rank(tag + " RANK b(unstated)", acc[0]);
        print_rank(tag + " RANK c(future)", acc[1]);
        // s and c on S15's own pairs (dev decides the plan)
        vector<Item> pairs;
        for (const auto& R : rows) {
            const int ri = rel_index(R.r);
            if (ri < 0 || Tst.facts_of(R.s).empty() || holds_value(ixT, R.s, R.r, R.v)) continue;
            Item x;
            x.s = R.s, x.r = R.r, x.o = R.v, x.label = R.label;
            x.T = run_tests(cxT, Tst, ixT, rsT, lpT, rsT.candidates(R.s, R.r), R.s, R.r, R.v, false);
            pairs.push_back(std::move(x));
        }
        if (sp == 0) PT = plan_from(pairs, tag + " PAIRS", no5, true);
        else plan_from(pairs, tag + " PAIRS", no5, false);
        // G2's hypotheses for S15 subjects, graded by S15
        vector<Item> hyps;
        for (const auto& s : subj) {
            if (Tst.facts_of(s).empty()) continue;
            for (const auto& r : ixT.relations()) {
                const auto rc = rsT.candidates(s, r);
                if (rc.empty()) continue;
                const int ri = rel_index(r);
                const auto sit = slot.find(s + '\x1f' + r);
                for (const auto& c : rc) {
                    if (holds_value(ixT, s, r, c.o)) continue;
                    Item x;
                    x.s = s, x.r = r, x.o = c.o;
                    if (sit != slot.end()) {
                        bool tr = false, fa = false, any = false;
                        for (const S15Row* R : sit->second) {
                            if (R->label == 1) any = true;
                            if (same_value(ri, R->v, c.o)) (R->label == 1 ? tr : fa) = true;
                        }
                        x.label = tr ? 1 : (fa || any) ? 0 : -1;
                    }
                    x.T = run_tests(cxT, Tst, ixT, rsT, lpT, rc, s, r, c.o, false);
                    x.p0 = std::min(0.98, std::max(0.02, c.score * s6::hard_constraint(Tst, s, ri, c.o)));
                    hyps.push_back(std::move(x));
                }
            }
        }
        std::cout << tag << " HYPOTHESES " << hyps.size() << "\n";
        run_loop(hyps, PT, tag, out + (sp == 0 ? ".r2_dev" : ".r2_test") + ".prereg.tsv");
        print_states(tag, hyps);
        print_precisions(tag, hyps, false);
        {
            std::ofstream f(out + (sp == 0 ? ".r2_dev" : ".r2_test") + ".hyps.tsv");
            f << "subject\trelation\tobject\tlabel\tP0\tposterior\tstate\tpromoted\tT1\tT2\tT3\tT6\tT4\n";
            for (const Item& x : hyps)
                f << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.label << '\t' << x.p0 << '\t' << x.post << '\t' << x.state << '\t' << x.promoted73 << '\t'
                  << ecs::test_outcome_name(x.T.t[0]) << '\t' << ecs::test_outcome_name(x.T.t[1]) << '\t' << ecs::test_outcome_name(x.T.t[2]) << '\t'
                  << ecs::test_outcome_name(x.T.t[3]) << '\t' << ecs::test_outcome_name(x.T.t[5]) << '\n';
        }

        // ============================================================ G14 on store T's contested slots of this split's subjects
        static const std::set<string> kTimeVarying = {"capital", "largest_city", "currency", "location", "country"};
        std::map<string, std::array<double, 2>> acc14;  // explanation -> right, graded
        double base_right = 0, base_n = 0, g14_err_right = 0, g14_err_n = 0;
        std::size_t contested = 0, multi = 0, graded = 0, slots_all = 0;
        std::map<string, std::size_t> truth_n;
        std::ofstream gf(out + (sp == 0 ? ".g14_dev" : ".g14_test") + ".tsv");
        gf << "subject\trelation\tvalue_a\tvalue_b\texplanation\tbits\twrong\twhy\ttruth\tright\n";
        for (const auto& [s, fs] : tby) {
            std::map<string, vector<string>> vals;
            for (const auto& [r, o] : fs) vals[r].push_back(o);
            for (const auto& [r, vs] : vals) {
                const int ri = rel_index(r);
                if (ri < 0 || vs.size() < 2 || !(is_datey(ri) || kRel[ri].phi >= 0.9)) continue;
                // clusters of mutually compatible values
                vector<vector<string>> cl;
                for (const auto& v : vs) {
                    bool put = false;
                    for (auto& c : cl) {
                        bool all = true;
                        for (const auto& w : c) all = all && same_value(ri, v, w);
                        if (all) { c.push_back(v); put = true; break; }
                    }
                    if (!put) cl.push_back({v});
                }
                if (cl.size() < 2) continue;
                ++slots_all;
                if (!subj.count(s)) continue;
                ++contested;
                if (cl.size() > 2) { ++multi; continue; }
                auto rep = [&](const vector<string>& c) {  // the most precise value; sources summed
                    string best = c[0];
                    for (const auto& v : c) {
                        const Date D = parse_date_str(v), B = parse_date_str(best);
                        if (is_datey(ri) && D.ok() && B.ok() && (D.m > 0) + (D.d > 0) > (B.m > 0) + (B.d > 0)) best = v;
                    }
                    double n = 0;
                    for (const auto& v : c) n += t_sources[s + '\x1f' + r + '\x1f' + v];
                    return std::pair<string, double>{best, n};
                };
                const auto [a, na] = rep(cl[0]);
                const auto [bb, nb] = rep(cl[1]);
                ecs::ContestedSlot slotc;
                slotc.s = s, slotc.r = r, slotc.a = a, slotc.b = bb, slotc.sources_a = na, slotc.sources_b = nb;
                ecs::ContestFacts F;
                F.is_date = is_datey(ri);
                F.time_varying = kTimeVarying.count(r) > 0;
                if (!F.is_date) F.a_contains_b = contains(K, a, bb), F.b_contains_a = contains(K, bb, a);
                if (F.is_date) {
                    const Date A = parse_date_str(a), B = parse_date_str(bb);
                    if (A.ok() && B.ok()) {
                        if (A.m && A.d && B.m && B.d) F.day_gap = static_cast<int>(std::labs(days_of(A) - days_of(B)));
                        F.same_year = A.y == B.y && (!A.d || !B.d);
                    }
                }
                const auto pit = page_of.find(s);
                if (pit == page_of.end()) F.t4_a = F.t4_b = -1;
                else {
                    const auto ps = s7::scan_rel(data, pages[static_cast<std::size_t>(pit->second)], ri);
                    TestOutcome o1[2], o2[2];
                    s7::t4(ps, ri, a, o1);
                    s7::t4(ps, ri, bb, o2);
                    F.t4_a = o1[1] == kPos, F.t4_b = o2[1] == kPos;
                }
                const auto E = ecs::explain_contest(slotc, F);
                // truth from S15
                string truth = "ungraded";
                int wrong_true = -1;
                const auto sit = slot.find(s + '\x1f' + r);
                if (sit != slot.end()) {
                    bool ma = false, mb = false;
                    for (const S15Row* R : sit->second)
                        if (R->label == 1) ma = ma || same_value(ri, R->v, a), mb = mb || same_value(ri, R->v, bb);
                    if (ma && mb) truth = "both";
                    else if (ma != mb) {
                        const bool gran = !F.is_date && (ma ? contains(K, bb, a) : contains(K, a, bb));
                        if (gran) truth = "granularity";
                        else truth = "error", wrong_true = ma ? 1 : 0;
                    }
                }
                int right = -1;
                if (truth != "ungraded") {
                    ++graded, ++truth_n[truth];
                    if (E.kind == ecs::HiddenVar::SourceError) right = truth == "error" && E.wrong == wrong_true;
                    else right = truth == "both" || truth == "granularity";
                    auto& A = acc14[ecs::hidden_var_name(E.kind)];
                    A[0] += right, A[1] += 1;
                    if (truth == "error") {
                        const int base_wrong = na != nb ? (na > nb ? 1 : 0) : 1;
                        base_right += base_wrong == wrong_true, base_n += 1;
                        g14_err_right += right, g14_err_n += 1;
                    }
                }
                gf << s << '\t' << r << '\t' << a << '\t' << bb << '\t' << ecs::hidden_var_name(E.kind) << '\t' << E.bits << '\t' << E.wrong << '\t' << E.why << '\t'
                   << truth << '\t' << right << '\n';
            }
        }
        double rt = 0, nt = 0;
        for (const auto& [k, v] : acc14) {
            rt += v[0], nt += v[1];
            std::snprintf(b, sizeof b, "%s G14 %-12s right %4.0f of %4.0f (%.3f)", tag.c_str(), k.c_str(), v[0], v[1], v[1] > 0 ? v[0] / v[1] : 0.0);
            std::cout << b << "\n";
        }
        string tn;
        for (const auto& [k, v] : truth_n) tn += " " + k + " " + std::to_string(v);
        std::snprintf(b, sizeof b,
                      "%s G14 contested slots (all subjects) %zu, of this split's S15 subjects %zu (3+ values %zu), graded %zu (truth:%s) | accuracy %.3f [%.3f, %.3f] | "
                      "on error slots: G14 %.0f/%.0f (%.3f), more-sources baseline %.0f/%.0f (%.3f)",
                      tag.c_str(), slots_all, contested, multi, graded, tn.c_str(), nt > 0 ? rt / nt : 0.0, ecs::wilson_lower(rt, nt), ecs::wilson_upper(rt, nt), g14_err_right,
                      g14_err_n, g14_err_n > 0 ? g14_err_right / g14_err_n : 0.0, base_right, base_n, base_n > 0 ? base_right / base_n : 0.0);
        std::cout << b << "\n";
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S8A seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s8a

// pattern_learn --worlds ENWIK8 SEEDS.tsv WD_VALUES.tsv FACTS_F2.tsv S15_DEV.tsv OUT_PREFIX [--read-confirm]
// (DISCOVERY 4.4, 7.2 T2, 7.6, 12 I9; docs/reports/S9_WORLDS_AND_FRAGILITY.md). Possible worlds (worlds.hpp). T2 through
// worlds on S7's dev set (S7's reduced store, rules mined on it): six variants (depth 1-3, type check on or off), chosen
// by mutual information, three-way likelihoods with Wilson and pair-bootstrap intervals, S8a's T2 beside it. The fresh
// confirm read on S15 dev's choose half only with --read-confirm. Fragility of the v2 store (seeds and admitted facts):
// states, each held fact's top source and weakest premise retracted, single points of failure as certainty gaps, blast
// radius per source instance, and hypothetical cards as data. Every base store's fingerprint is checked after every
// world (I9). Nothing here answers a question: worlds are Cypha's bookkeeping about its own store (PLAN rule 10).
namespace s9 {

using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;
using s7::PhiTable;

int date_kind(const string& n) {
    return n == "birth_year" || n == "birth_date" ? 1 : n == "death_year" || n == "death_date" ? 2 : n == "founded" ? 3 : 0;
}

// S7's T1 checks (t1_phi) applied to a fact against the world's held facts (stored or assumed, the fact itself aside):
// a functional slot (phi >= 0.9) holding only other values (same_value), a date slot's other year when functional, the
// lifespan checks, then (with type_check) the type checks. The type check is tried last, so a "type" clash is one no
// other check found.
ecs::ConflictChecks world_checks(const PhiTable& phi, bool type_check) {
    ecs::ConflictChecks cc;
    cc.held = [phi, type_check](const ecs::World& w, const ecs::Triple& t, ecs::Triple& culprit, string& kind) {
        const int ri = rel_index(t.r);
        if (ri < 0) return false;
        const bool functional = phi[static_cast<std::size_t>(ri)] >= 0.9;
        const auto facts = w.held_facts_of(t.s);
        if (is_datey(ri)) {
            const int k = date_kind(kRel[ri].name);
            const Date D = parse_date_str(t.o);
            if (k && D.ok())
                for (const auto& f : facts) {
                    if (f.r == t.r && f.o == t.o) continue;
                    const int kf = date_kind(f.r);
                    if (!kf) continue;
                    const Date F = parse_date_str(f.o);
                    if (!F.ok()) continue;
                    if (kf == k && F.y != D.y && functional) {
                        culprit = {t.s, f.r, f.o}, kind = "date slot";
                        return true;
                    }
                    if ((k == 1 && kf == 2 && (F.y < D.y || F.y - D.y > 125)) || (k == 2 && kf == 1 && (D.y < F.y || D.y - F.y > 125))) {
                        culprit = {t.s, f.r, f.o}, kind = "lifespan";
                        return true;
                    }
                }
        } else if (functional) {
            bool other = false, same = false;
            ecs::Triple c;
            for (const auto& f : facts) {
                if (f.r != t.r || f.o == t.o) continue;
                if (s8a::same_value(ri, f.o, t.o)) same = true;
                else if (!other) other = true, c = {t.s, f.r, f.o};
            }
            if (other && !same) {
                culprit = c, kind = "functional";
                return true;
            }
        }
        if (type_check) {
            const string k = norm_ent(t.o);
            const RT ty = kRel[ri].type;
            if ((ty == T_PLACE || ty == T_PLACE_LINK) && g_person.count(k) && !g_place.count(k)) {
                culprit = {t.o, "page_type", "person"}, kind = "type";
                return true;
            }
            if (ty == T_PERSON && (g_place.count(k) || g_country.count(k)) && !g_person.count(k)) {
                culprit = {t.o, "page_type", "place"}, kind = "type";
                return true;
            }
        }
        return false;
    };
    cc.pair = [phi](const ecs::Triple& a, const ecs::Triple& b) {
        const int ri = rel_index(a.r);
        return a.r == b.r && ri >= 0 && phi[static_cast<std::size_t>(ri)] >= 0.9 && !s8a::same_value(ri, a.o, b.o);
    };
    cc.confirmed = [](const ecs::World& w, const ecs::Triple& t) {  // the store holds a same value (an assumption confirms nothing)
        const int ri = rel_index(t.r);
        for (const auto& f : w.store().facts_of(t.s))
            if (f.r == t.r && w.in_store({t.s, f.r, f.o}) && s8a::same_value(ri, f.o, t.o)) return true;
        return false;
    };
    return cc;
}

// The six variants: depth 1-3 x the type check off (0) or on (1). The registered default is depth 3 with the type check.
constexpr int kDefD = 2, kDefT = 1;
string variant_name(int d, int ty) { return "T2 world d" + std::to_string(d + 1) + (ty ? "+type" : " notype"); }

struct Item {
    string s, r, o;
    int label = 0, pair = 0;
    char set = 'U', cat = ' ';
    bool dev = true, wd_true = false, reach = false;
    TestOutcome t2[3][2];  // [depth - 1][type check]
    TestOutcome t2_s8a = kInc, t1 = kInc;
    std::size_t derived3 = 0, confirmed3 = 0;
    string clash3;  // the default variant's first contradiction kind
    std::set<string> soft3;  // exploratory: relations where a derived fact (default variant, conf >= 0.8) meets other held values, any phi
};

// T2 through worlds for every variant of one item. The contradictions with kind "type" count only with the type check.
// Each world's base store is fingerprinted after it (I9).
struct I9Count {
    std::size_t worlds = 0, changed = 0;
};
void t2_item(Item& x, const ecs::TripleStore& st, std::uint64_t fp, const ecs::RuleBook& book, const ecs::ConflictChecks& cc, I9Count& i9) {
    const ecs::Triple h{x.s, x.r, x.o};
    for (int d = 0; d < 3; ++d) {
        ecs::ChainResult res;
        ecs::t2_world(st, book, h, ecs::ChainConfig{d + 1, 0.8}, cc, &res);
        ++i9.worlds;
        i9.changed += ecs::store_fingerprint(st) != fp;
        bool neg_any = false, neg_notype = false;
        for (const auto& c : res.contradictions) neg_any = true, neg_notype = neg_notype || c.kind != "type";
        bool pos = false;
        for (const auto& c : res.confirmed) pos = pos || c.by_store;
        x.t2[d][1] = neg_any ? kNeg : pos ? kPos : kInc;
        x.t2[d][0] = neg_notype ? kNeg : pos ? kPos : kInc;
        if (d == kDefD) {
            x.derived3 = res.derived.size();
            for (const auto& c : res.confirmed) x.confirmed3 += c.by_store;
            if (!res.contradictions.empty()) x.clash3 = res.contradictions[0].kind;
            for (const auto& c : res.derived) {  // exploratory: the slot holds other values, whatever its phi
                if (c.conf < 0.8) continue;
                const int ri = rel_index(c.t.r);
                bool other = false, same = false;
                auto look = [&](const string& v) { (s8a::same_value(ri, v, c.t.o) ? same : other) = true; };
                for (const auto& f : st.facts_of(c.t.s))
                    if (f.r == c.t.r) look(f.o);
                if (h.s == c.t.s && h.r == c.t.r) look(h.o);
                if (other && !same) x.soft3.insert(c.t.r);
            }
        }
    }
}

// Items counted: S6's labels (a false member Wikidata lists is left out), as S7's tallies.
bool counted(const Item& x) { return x.label == 1 || !x.wd_true; }

template <class F>
ecs::TestTally tally(const vector<Item>& it, const vector<std::size_t>& idx, F&& outcome) {
    ecs::TestTally t;
    for (std::size_t i : idx)
        if (counted(it[i])) t.add(it[i].label == 1, outcome(it[i]));
    return t;
}

// Three-way likelihoods (add-one) with Wilson intervals on the raw proportions, and pair-bootstrap intervals of the
// likelihood ratios (2,000 resamples of pairs, seed 20261029).
template <class F>
string lik_lines(const string& tag, const string& name, const vector<Item>& it, const vector<std::size_t>& idx, F&& outcome, double lr_out[3]) {
    const ecs::TestTally t = tally(it, idx, outcome);
    const ecs::OutcomeLikelihoods L = ecs::likelihoods_from(t);
    std::map<int, vector<std::size_t>> pairs;
    for (std::size_t i : idx)
        if (counted(it[i])) pairs[it[i].pair].push_back(i);
    vector<const vector<std::size_t>*> pv;
    for (const auto& [k, v] : pairs) pv.push_back(&v);
    std::mt19937_64 rng(20261029);
    vector<double> lr[3];
    for (int b = 0; b < 2000 && !pv.empty(); ++b) {
        ecs::TestTally tb;
        for (std::size_t j = 0; j < pv.size(); ++j)
            for (std::size_t i : *pv[rng() % pv.size()]) tb.add(it[i].label == 1, outcome(it[i]));
        const ecs::OutcomeLikelihoods Lb = ecs::likelihoods_from(tb);
        for (int o = 0; o < 3; ++o) lr[o].push_back(Lb.t[o] / Lb.f[o]);
    }
    const double nt = t.t_pos + t.t_neg + t.t_inc, nf = t.f_pos + t.f_neg + t.f_inc;
    const double ct[3] = {t.t_pos, t.t_neg, t.t_inc}, cf[3] = {t.f_pos, t.f_neg, t.f_inc};
    const char* on[3] = {"pos", "neg", "inc"};
    string s;
    char b[512];
    for (int o = 0; o < 3; ++o) {
        std::sort(lr[o].begin(), lr[o].end());
        const double lo = lr[o].empty() ? 0 : lr[o][static_cast<std::size_t>(0.025 * (lr[o].size() - 1))];
        const double hi = lr[o].empty() ? 0 : lr[o][static_cast<std::size_t>(0.975 * (lr[o].size() - 1))];
        lr_out[o] = L.t[o] / L.f[o];
        std::snprintf(b, sizeof b, "%s LIK %-18s %s seen %4.0f | P(.|true) %.4f raw %.4f [%.4f, %.4f] | P(.|false) %.4f raw %.4f [%.4f, %.4f] | LR %.3f [%.3f, %.3f]\n",
                      tag.c_str(), name.c_str(), on[o], ct[o] + cf[o], L.t[o], nt > 0 ? ct[o] / nt : 0.0, ecs::wilson_lower(ct[o], nt), ecs::wilson_upper(ct[o], nt), L.f[o],
                      nf > 0 ? cf[o] / nf : 0.0, ecs::wilson_lower(cf[o], nf), ecs::wilson_upper(cf[o], nf), lr_out[o], lo, hi);
        s += b;
    }
    return s;
}

const char* oname(TestOutcome o) { return ecs::test_outcome_name(o); }

int run(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: pattern_learn --worlds ENWIK8 SEEDS.tsv WD_VALUES.tsv FACTS_F2.tsv S15_DEV.tsv OUT_PREFIX [--read-confirm] [--s9b]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[7];
    bool read_confirm = false, s9b = false;  // s9b: S9b's reads (the S9 report, "S9b"); without it the output is S9's
    for (int i = 8; i < argc; ++i) {
        if (string(argv[i]) == "--read-confirm") read_confirm = true;
        else if (string(argv[i]) == "--s9b") s9b = true;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    char b[2048];
    // ---- the article text and its type lexicons (S7's setup)
    string data;
    if (!read_file(argv[2], data)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    g_dump = &data;
    long total = 0, redirects = 0;
    const vector<PageRef> pages = list_pages(data, total, redirects);
    std::unordered_map<string, int> page_of;
    for (std::size_t i = 0; i < pages.size(); ++i) page_of[pages[i].title] = static_cast<int>(i);
    for (const PageRef& pg : pages) {
        PageFlags fl;
        scan_page(data, pg, fl);
        const string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    {
        long ns = 0, ng = 0;
        if (!load_wikidata(argv[4], false, page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    }
    for (const Seed& sd : g_seeds) {
        const RT ty = kRel[sd.rel].type;
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    // ---- the store (S7's), its seed rows with their sources, and the Wikidata values
    struct Row {
        string s, r, o, source;
        std::size_t off = 0;
    };
    vector<Row> seed_rows;
    std::map<string, vector<std::pair<string, string>>> by_subj;
    std::set<string> seen, wd_src;
    {
        std::ifstream f(argv[3]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            seed_rows.push_back({c[0], c[1], c[2], c[3], static_cast<std::size_t>(std::strtoull(c[5].c_str(), nullptr, 10))});
            if (c[3] == "wikidata") wd_src.insert(c[0] + '\x1f' + c[1]);
            if (seen.insert(c[0] + '\x1f' + c[1] + '\x1f' + c[2]).second) by_subj[c[0]].push_back({c[1], c[2]});
        }
    }
    ecs::TripleStore full;
    for (auto& [s, fs] : by_subj) {
        std::sort(fs.begin(), fs.end());
        for (const auto& [r, o] : fs) full.add(s, r, o);
    }
    std::unordered_map<string, std::set<string>> wd6;
    {
        std::ifstream f(argv[4]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 3) continue;
            for (const auto& x : split(c[2], '|')) wd6[c[0] + '\x1f' + c[1]].insert(lower(x));
        }
    }
    const s7::WdObjs wd = s7::load_wd_objs(argv[4]);
    // ---- S7's dev set, rebuilt with S7's code and seeds (S6's sets, then U and C hold-outs, the reduced store, false partners)
    std::mt19937_64 rng6(20261028);
    vector<string> subj;
    for (const auto& [s, fs] : by_subj)
        if (fs.size() >= 2) subj.push_back(s);
    for (std::size_t i = subj.size(); i > 1; --i) std::swap(subj[i - 1], subj[rng6() % i]);
    std::map<string, std::pair<string, string>> h6;
    {
        s6::PlantedSet A, F;
        const std::size_t next = s6::build_set(A, subj, 0, 2000, by_subj, full, wd6, rng6);
        s6::build_set(F, subj, next, std::numeric_limits<std::size_t>::max(), by_subj, full, wd6, rng6);
        for (const auto* ps : {&A, &F})
            for (const auto& x : ps->it)
                if (x.label) h6[x.s] = {x.r, x.o};
    }
    struct Held {
        string s, r, o;
        char set;
    };
    vector<Held> held;
    std::set<string> held_keys;
    for (const auto& [s, fs] : by_subj) {
        if (fs.size() != 1) continue;
        if (full.count(fs[0].first, fs[0].second) < 2) continue;
        held.push_back({s, fs[0].first, fs[0].second, 'U'});
    }
    std::mt19937_64 rng7(20261029);
    for (const string& s : subj) {
        const auto it6 = h6.find(s);
        if (it6 == h6.end()) continue;
        auto fs = by_subj[s];
        for (std::size_t i = fs.size(); i > 1; --i) std::swap(fs[i - 1], fs[rng7() % i]);
        for (const auto& [r, o] : fs)
            if (!(r == it6->second.first && o == it6->second.second) && full.count(r, o) >= 2) {
                held.push_back({s, r, o, 'C'});
                break;
            }
    }
    for (const auto& h : held) held_keys.insert(h.s + '\x1f' + h.r + '\x1f' + h.o);
    ecs::TripleStore st;  // S7's reduced store
    std::unordered_map<string, vector<string>> objects_of_rel;
    for (const auto& [s, fs] : by_subj)
        for (const auto& [r, o] : fs)
            if (!held_keys.count(s + '\x1f' + r + '\x1f' + o)) st.add(s, r, o), objects_of_rel[r].push_back(o);
    std::mt19937_64 rngf(20261030);
    vector<Item> it;
    std::size_t n_pairs = 0;
    for (const auto& h : held) {
        if (st.count(h.r, h.o) == 0) continue;
        const auto& pool = objects_of_rel[h.r];
        const int ri = rel_index(h.r);
        const Date hd = is_datey(ri) ? parse_date_str(h.o) : Date{};
        string neg;
        for (int t = 0; t < 100 && neg.empty(); ++t) {
            const string& c = pool[rngf() % pool.size()];
            if (c == h.o || c == h.s || full.has(h.s, h.r, c)) continue;
            if (is_datey(ri)) {
                const Date cd = parse_date_str(c);
                if (cd.ok() && hd.ok() && cd.y == hd.y) continue;
            }
            neg = c;
        }
        if (neg.empty()) continue;
        const int pr = static_cast<int>(n_pairs++);
        const bool dev = ((fnv(h.s) >> 11) & 1) == 0;
        if (!dev) continue;  // S7's test half is not read
        const auto wit = wd.find(h.s + '\x1f' + h.r);
        Item a, x;
        a.s = x.s = h.s, a.r = x.r = h.r, a.o = h.o, x.o = neg, a.label = 1, x.label = 0;
        a.pair = x.pair = pr;
        a.set = x.set = h.set;
        x.wd_true = wit != wd.end() && s7::wd_match(ri, neg, wit->second);
        it.push_back(a), it.push_back(x);
    }
    std::size_t nu = 0, nc = 0, nwd = 0;
    for (std::size_t i = 0; i < it.size(); i += 2) (it[i].set == 'U' ? nu : nc)++, nwd += it[i + 1].wd_true;
    std::snprintf(b, sizeof b, "S9 SET S7's dev pairs %zu (U %zu, C %zu) of %zu pairs | false members Wikidata lists %zu | reduced store %zu of %zu", it.size() / 2, nu, nc,
                  n_pairs, nwd, st.size(), full.size());
    std::cout << b << "\n";
    // ---- rules mined on the reduced store (S8a's defaults), the store's phi (S7b's choice), the checks
    const ecs::RuleIndex ix(st);
    const vector<ecs::Rule> rules = ecs::mine_rules(ix);
    const ecs::RuleBook book(rules);
    const ecs::RuleSet rs(ix, rules);
    const PhiTable phi = s7::phi_from_store(st);
    const ecs::ConflictChecks cc = world_checks(phi, true);
    {
        std::map<string, int> kinds;
        for (const auto& R : book.rules()) kinds[ecs::rule_kind_name(R.kind)]++;
        string ks;
        for (const auto& [k, v] : kinds) ks += " " + k + " " + std::to_string(v);
        std::snprintf(b, sizeof b, "S9 DEV RULES mined %zu kept %zu:%s", rules.size(), book.rules().size(), ks.c_str());
        std::cout << b << "\n";
    }
    // ---- T2 through worlds on every dev item, S8a's T2 and T1 (phi_store) beside it
    I9Count i9;
    const std::uint64_t fp_st = ecs::store_fingerprint(st);
    const double t_dev = now_s();
    for (Item& x : it) {
        t2_item(x, st, fp_st, book, cc, i9);
        x.t2_s8a = s8a::t2(ix, rs, x.s, x.r, x.o);
        x.t1 = s7::t1_phi(st, x.s, x.r, x.o, phi);
    }
    std::snprintf(b, sizeof b, "S9 DEV worlds %zu in %.1f s", i9.worlds, now_s() - t_dev);
    std::cout << b << "\n";
    if (s9b) {  // S9b for S7d: does the reduced store hold a Wikidata claim about each dev item's subject?
        std::set<string> wd_fact;
        for (const Row& w : seed_rows)
            if (w.source == "wikidata") wd_fact.insert(w.s + '\x1f' + w.r + '\x1f' + w.o);
        std::ofstream f(out + ".s9b_dev_items.tsv");
        f << "pair\tlabel\tsubject\trelation\tobject\twd_subject\n";
        std::size_t nw = 0;
        for (const Item& x : it) {
            bool w = false;
            const auto bs = by_subj.find(x.s);
            if (bs != by_subj.end())
                for (const auto& [r, o] : bs->second) {
                    const string k = x.s + '\x1f' + r + '\x1f' + o;
                    if (!held_keys.count(k) && wd_fact.count(k)) {
                        w = true;
                        break;
                    }
                }
            nw += w;
            f << x.pair << '\t' << x.label << '\t' << x.s << '\t' << x.r << '\t' << x.o << '\t' << w << '\n';
        }
        std::snprintf(b, sizeof b, "S9B DEV items whose subject has a Wikidata claim in the reduced store %zu of %zu", nw, it.size());
        std::cout << b << "\n";
    }
    vector<std::size_t> all_idx, u_idx, c_idx;
    for (std::size_t i = 0; i < it.size(); ++i) all_idx.push_back(i), (it[i].set == 'U' ? u_idx : c_idx).push_back(i);
    double mi[3][2];
    for (int d = 0; d < 3; ++d)
        for (int ty = 0; ty < 2; ++ty) {
            auto f = [d, ty](const Item& x) { return x.t2[d][ty]; };
            const auto t = tally(it, all_idx, f);
            mi[d][ty] = t.mutual_information_bits();
            std::cout << s7::line_of("S9 DEV", variant_name(d, ty), t) << "\n";
            std::cout << s7::line_of("S9 DEV U", variant_name(d, ty), tally(it, u_idx, f)) << "\n";
            std::cout << s7::line_of("S9 DEV C", variant_name(d, ty), tally(it, c_idx, f)) << "\n";
        }
    std::cout << s7::line_of("S9 DEV", "T2 S8a (d1, hand phi)", tally(it, all_idx, [](const Item& x) { return x.t2_s8a; })) << "\n";
    std::cout << s7::line_of("S9 DEV", "T1 phi_store", tally(it, all_idx, [](const Item& x) { return x.t1; })) << "\n";
    // The highest MI wins. A tie keeps the registered default, or else the tied variant closest to it: the default's type
    // setting first, then the deeper.
    int cd = kDefD, ct = kDefT;
    double best_mi = -1;
    for (int d = 0; d < 3; ++d)
        for (int ty = 0; ty < 2; ++ty) best_mi = std::max(best_mi, mi[d][ty]);
    if (mi[kDefD][kDefT] < best_mi - 1e-12) {
        int bd = -1, bt = -1;
        for (const int ty : {kDefT, 1 - kDefT})
            for (int d = 2; d >= 0; --d)
                if (bd < 0 && mi[d][ty] >= best_mi - 1e-12) bd = d, bt = ty;
        cd = bd, ct = bt;
    }
    auto chosen = [cd, ct](const Item& x) { return x.t2[cd][ct]; };
    const ecs::TestTally tc = tally(it, all_idx, chosen);
    const bool earns = tc.applicable() >= 0.05 && tc.youden_lo() > 0.0;
    std::snprintf(b, sizeof b, "S9 CHOICE %s (MI %.5f bits; registered default %s MI %.5f) | EARN applicable %.3f J %+.3f [%+.3f, %+.3f] -> %s", variant_name(cd, ct).c_str(),
                  mi[cd][ct], variant_name(kDefD, kDefT).c_str(), mi[kDefD][kDefT], tc.applicable(), tc.youden(), tc.youden_lo(), tc.youden_hi(),
                  earns ? "earns its place" : "cut");
    std::cout << b << "\n";
    double lr_dev[3], lr_tmp[3];
    std::cout << lik_lines("S9 DEV", variant_name(cd, ct), it, all_idx, chosen, lr_dev);
    if (cd != kDefD || ct != kDefT) std::cout << lik_lines("S9 DEV", variant_name(kDefD, kDefT), it, all_idx, [](const Item& x) { return x.t2[kDefD][kDefT]; }, lr_tmp);
    std::cout << lik_lines("S9 DEV", "T2 S8a", it, all_idx, [](const Item& x) { return x.t2_s8a; }, lr_tmp);
    {  // the default variant's contradiction kinds, and T1 x T2 (chosen) on dev
        std::map<string, std::array<int, 2>> kinds;
        for (const Item& x : it)
            if (counted(x) && !x.clash3.empty()) kinds[x.clash3][x.label == 1]++;
        for (const auto& [k, v] : kinds) {
            std::snprintf(b, sizeof b, "S9 DEV CLASH %-11s true %d false %d", k.c_str(), v[1], v[0]);
            std::cout << b << "\n";
        }
        int joint[3][3][2] = {};
        for (const Item& x : it)
            if (counted(x)) joint[static_cast<int>(x.t1)][static_cast<int>(chosen(x))][x.label == 1]++;
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c) {
                std::snprintf(b, sizeof b, "S9 DEV JOINT T1 %-12s T2 %-12s true %4d false %4d", oname(static_cast<TestOutcome>(a)), oname(static_cast<TestOutcome>(c)), joint[a][c][1],
                              joint[a][c][0]);
                std::cout << b << "\n";
            }
    }
    {  // exploratory (dev only, no choice made on it): why T2 cannot say no. The kept rules, the derived facts' slots, and
       // what a soft contradiction (a derived fact meeting other held values in a slot with phi >= theta) would do.
        s8a::write_rules(out + ".dev_rules.tsv", book.rules());
        std::map<string, std::array<int, 2>> rel_items;
        for (const Item& x : it)
            if (counted(x))
                for (const string& r : x.soft3) rel_items[r][x.label == 1]++;
        for (const auto& [r, v] : rel_items) {
            std::snprintf(b, sizeof b, "S9 DEV EXPLORE SLOT %-13s phi_store %.3f | items whose derived fact meets other held values: true %d false %d", r.c_str(),
                          phi[static_cast<std::size_t>(rel_index(r))], v[1], v[0]);
            std::cout << b << "\n";
        }
        for (const double th : {0.5, 0.7, 0.8, 0.85}) {
            auto soft = [&](const Item& x) {
                for (const string& r : x.soft3)
                    if (phi[static_cast<std::size_t>(rel_index(r))] >= th) return kNeg;
                return x.t2[kDefD][kDefT];
            };
            std::snprintf(b, sizeof b, "S9 DEV EXPLORE soft phi>=%.2f", th);
            std::cout << s7::line_of(b, "T2 d3 soft", tally(it, all_idx, soft)) << "\n";
        }
    }
    {  // the dev items
        std::ofstream f(out + ".dev_items.tsv");
        f << "pair\tset\tlabel\twd_true\tsubject\trelation\tobject\tT1\tT2_d1_notype\tT2_d1_type\tT2_d2_notype\tT2_d2_type\tT2_d3_notype\tT2_d3_type\tT2_S8a\tderived_d3\tconfirmed_d3\tclash_d3\n";
        for (const Item& x : it) {
            f << x.pair << '\t' << x.set << '\t' << x.label << '\t' << x.wd_true << '\t' << x.s << '\t' << x.r << '\t' << x.o << '\t' << oname(x.t1);
            for (int d = 0; d < 3; ++d)
                for (int ty = 0; ty < 2; ++ty) f << '\t' << oname(x.t2[d][ty]);
            f << '\t' << oname(x.t2_s8a) << '\t' << x.derived3 << '\t' << x.confirmed3 << '\t' << x.clash3 << '\n';
        }
    }
    std::ofstream cards(out + ".cards.txt");
    {  // cards: the first 20 dev worlds (item order) whose chosen T2 reached a contradiction
        const ecs::ConflictChecks ccv = world_checks(phi, ct == 1);
        int n = 0;
        for (const Item& x : it) {
            if (n >= 20) break;
            if (chosen(x) != kNeg) continue;
            const ecs::Triple h{x.s, x.r, x.o};
            ecs::World w(st);
            w.add(h.s, h.r, h.o);
            const ecs::ChainResult r = ecs::chain(w, book, ecs::ChainConfig{cd + 1, 0.8}, ccv);
            ++i9.worlds, i9.changed += ecs::store_fingerprint(st) != fp_st;
            ecs::WorldCard c;
            c.added = w.assumed_added();
            c.consequences = r.derived;
            c.contradictions = r.contradictions;
            c.depth = cd + 1;
            const vector<ecs::Triple> as{h};
            for (const auto& v : ecs::minimal_conflicting_subsets(st, book, as, ecs::ChainConfig{cd + 1, 0.8}, ccv)) {
                c.minimal_conflicts.emplace_back();
                for (std::size_t k : v) c.minimal_conflicts.back().push_back(as[k]);
            }
            cards << "# dev pair " << x.pair << " (" << (x.label ? "true" : "false") << " member)\n" << ecs::format_world_card(c) << "\n";
            ++n;
        }
        std::snprintf(b, sizeof b, "S9 CARDS dev worlds with a contradiction %d", n);
        std::cout << b << "\n";
        int np = 0;  // illustrations: the first five dev worlds where T2 (chosen) is positive
        for (const Item& x : it) {
            if (np >= 5) break;
            if (chosen(x) != kPos) continue;
            ecs::World w(st);
            w.add(x.s, x.r, x.o);
            const ecs::ChainResult r = ecs::chain(w, book, ecs::ChainConfig{cd + 1, 0.8}, ccv);
            ++i9.worlds, i9.changed += ecs::store_fingerprint(st) != fp_st;
            ecs::WorldCard c;
            c.added = w.assumed_added();
            c.consequences = r.derived;
            c.depth = cd + 1;
            string conf;
            for (const auto& q : r.confirmed)
                if (q.by_store) conf += (conf.empty() ? "" : "; ") + q.t.text();
            cards << "# dev pair " << x.pair << " (" << (x.label ? "true" : "false") << " member), T2 positive: confirms " << conf << "\n" << ecs::format_world_card(c) << "\n";
            ++np;
        }
    }
    // ---- the fresh confirm read: S15 dev's choose half (bit 17 = 0) on S7b's S15-reduced store, the dev phi table as is
    std::uint64_t fp15 = 0;
    ecs::TripleStore st15;
    if (read_confirm) {
        const vector<s7::S15Pair> sp = s7::load_s15(argv[6], page_of);
        std::set<string> drop;
        for (const s7::S15Pair& p : sp) {
            const int ri = rel_index(p.r);
            for (const auto& f : full.facts_of(p.s))
                if (f.r == p.r && s7::same_value(ri, f.o, p.vt)) drop.insert(p.s + '\x1f' + p.r + '\x1f' + f.o);
        }
        std::unordered_map<string, std::map<string, std::map<string, int>>> surf;
        for (const auto& [s, fs] : by_subj)
            for (const auto& [r, o] : fs)
                if (!drop.count(s + '\x1f' + r + '\x1f' + o)) st15.add(s, r, o), surf[r][norm_ent(o)][o] += 1;
        auto form = [&](const string& r, const string& v) -> string {
            if (is_datey(rel_index(r))) return v;
            const auto ir = surf.find(r);
            if (ir == surf.end()) return v;
            const auto ik = ir->second.find(norm_ent(v));
            if (ik == ir->second.end()) return v;
            string bs;
            int bc = -1;
            for (const auto& [o, c] : ik->second)
                if (c > bc) bc = c, bs = o;
            return bs;
        };
        vector<Item> it15;
        int pr = 0;
        for (const s7::S15Pair& p : sp) {
            Item a, x;
            a.s = x.s = p.s, a.r = x.r = p.r;
            a.o = form(p.r, p.vt), x.o = form(p.r, p.vf);
            if (a.o == x.o) continue;
            const int pair = pr++;
            if (((fnv(p.s) >> 17) & 1) != 0) continue;  // the confirm half (S7b's) is not read
            a.label = 1, x.label = 0;
            a.pair = x.pair = pair;
            a.set = x.set = 'S';
            a.cat = x.cat = p.cat;
            a.reach = x.reach = st15.count(p.r, a.o) >= 1 && st15.count(p.r, x.o) >= 1;
            it15.push_back(a), it15.push_back(x);
        }
        const ecs::RuleIndex ix15(st15);
        const vector<ecs::Rule> rules15 = ecs::mine_rules(ix15);
        const ecs::RuleBook book15(rules15);
        const ecs::RuleSet rs15(ix15, rules15);
        fp15 = ecs::store_fingerprint(st15);
        for (Item& x : it15) {
            t2_item(x, st15, fp15, book15, cc, i9);  // every variant computed; only the chosen one is judged
            x.t2_s8a = s8a::t2(ix15, rs15, x.s, x.r, x.o);
            x.t1 = s7::t1_phi(st15, x.s, x.r, x.o, phi);
        }
        vector<std::size_t> idx_all, idx_reach;
        for (std::size_t i = 0; i < it15.size(); ++i) {
            idx_all.push_back(i);
            if (it15[i].reach) idx_reach.push_back(i);
        }
        std::snprintf(b, sizeof b, "S9 CONFIRM S15 choose half pairs %zu (reachable %zu) | S15-reduced store %zu of %zu | rules kept %zu", it15.size() / 2,
                      idx_reach.size() / 2, st15.size(), full.size(), book15.rules().size());
        std::cout << b << "\n";
        bool holds = true;
        for (const bool reach : {true, false}) {
            const auto& idx = reach ? idx_reach : idx_all;
            const string tag = reach ? "S9 CONFIRM REACH" : "S9 CONFIRM ALL";
            std::cout << s7::line_of(tag.c_str(), variant_name(cd, ct), tally(it15, idx, chosen)) << "\n";
            std::cout << s7::line_of(tag.c_str(), "T2 S8a (d1, hand phi)", tally(it15, idx, [](const Item& x) { return x.t2_s8a; })) << "\n";
            double lr[3];
            std::cout << lik_lines(tag, variant_name(cd, ct), it15, idx, chosen, lr);
            if (!reach) continue;
            const ecs::TestTally t = tally(it15, idx, chosen);
            const double seen_pos = t.t_pos + t.f_pos, seen_neg = t.t_neg + t.f_neg;
            const bool pos_ok = seen_pos < 5 || lr[0] > 1.0, neg_ok = seen_neg < 5 || lr[1] < 1.0;
            holds = pos_ok && neg_ok;
            std::snprintf(b, sizeof b, "S9 CONFIRM RULE positive seen %.0f LR %.3f (%s) | negative seen %.0f LR %.3f (%s) | dev LR pos %.3f neg %.3f -> %s", seen_pos, lr[0],
                          seen_pos < 5 ? "not judged" : pos_ok ? "holds" : "fails", seen_neg, lr[1], seen_neg < 5 ? "not judged" : neg_ok ? "holds" : "fails", lr_dev[0], lr_dev[1],
                          holds ? "directions hold" : "directions fail");
            std::cout << b << "\n";
        }
        std::ofstream f(out + ".confirm_items.tsv");
        f << "pair\tcategory\treachable\tlabel\tsubject\trelation\tobject\tT1\tT2_chosen\tT2_S8a\tderived_d3\tclash_d3\n";
        for (const Item& x : it15)
            f << x.pair << '\t' << x.cat << '\t' << x.reach << '\t' << x.label << '\t' << x.s << '\t' << x.r << '\t' << x.o << '\t' << oname(x.t1) << '\t' << oname(chosen(x)) << '\t'
              << oname(x.t2_s8a) << '\t' << x.derived3 << '\t' << x.clash3 << '\n';
    }
    // ---- fragility (7.6) on the v2 store: every seed row and every admitted text fact
    vector<string> src_name;
    std::unordered_map<string, std::size_t> src_id;
    auto sid = [&](const string& n) {
        auto [p, ins] = src_id.emplace(n, src_name.size());
        if (ins) src_name.push_back(n);
        return p->second;
    };
    vector<ecs::FragFact> F;
    std::unordered_map<string, std::size_t> fidx;
    std::set<std::pair<std::size_t, std::size_t>> fact_src;
    const double tau_text = ecs::prior_infobox().mean(), tau_wd = ecs::prior_wikidata().mean();
    auto add_row = [&](const string& s, const string& r, const string& o, const string& source, double tau) {
        const string k = s + '\x1f' + r + '\x1f' + o;
        auto [p, ins] = fidx.emplace(k, F.size());
        if (ins) {
            F.emplace_back();
            F.back().s = s, F.back().r = r, F.back().o = o;
        }
        const std::size_t i = p->second, so = sid(source);
        if (!fact_src.insert({i, so}).second) return;
        ecs::SourceItem si;
        si.source = so;
        si.item.tau = tau;
        F[i].sources.push_back(si);
    };
    auto page_at = [&](std::size_t off) -> string {
        auto pit = std::upper_bound(pages.begin(), pages.end(), off, [](std::size_t v, const PageRef& p) { return v < p.tb; });
        return pit == pages.begin() ? string("?") : std::prev(pit)->title;
    };
    std::size_t n_text_rows = 0;
    for (const Row& w : seed_rows) {
        if (w.source == "wikidata") add_row(w.s, w.r, w.o, "wikidata", tau_wd);
        else add_row(w.s, w.r, w.o, w.source + "|" + page_at(w.off), tau_text);
    }
    {
        std::ifstream f(argv[5]);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 9 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            add_row(c[0], c[1], c[2], "text|" + c[7], tau_text);
            ++n_text_rows;
        }
    }
    ecs::TripleStore v2;
    for (const auto& x : F) v2.add(x.s, x.r, x.o);
    const std::uint64_t fp_v2 = ecs::store_fingerprint(v2);
    const PhiTable phi2 = s7::phi_from_store(v2);
    const ecs::RuleIndex ix2(v2);
    const vector<ecs::Rule> rules2 = ecs::mine_rules(ix2);
    const ecs::RuleBook book2(rules2);
    std::size_t n_deriv = 0, n_rival_pairs = 0, n_hard = 0;
    auto idx_of = [&](const string& s, const string& r, const string& o) -> long {
        const auto p = fidx.find(s + '\x1f' + r + '\x1f' + o);
        return p == fidx.end() ? -1 : static_cast<long>(p->second);
    };
    for (std::size_t i = 0; i < F.size(); ++i) {  // depth-1 derivations from other stored facts
        const ecs::FragFact& x = F[i];
        for (std::size_t k : book2.with_head(x.r)) {
            const ecs::Rule& R = book2.rules()[k];
            vector<vector<std::size_t>> inst;
            auto one = [&](long p) {
                if (p >= 0 && static_cast<std::size_t>(p) != i) inst.push_back({static_cast<std::size_t>(p)});
            };
            if (R.kind == ecs::RuleKind::Const) {
                if (x.o == R.head_o) one(idx_of(x.s, R.r1, R.c1));
            } else if (R.kind == ecs::RuleKind::Same) one(idx_of(x.s, R.r1, x.o));
            else if (R.kind == ecs::RuleKind::Inverse) one(idx_of(x.o, R.r1, x.s));
            else {
                vector<string> zs;
                if (R.inv1) zs = v2.subjects_with(R.r1, x.s);
                else
                    for (const auto& f : v2.facts_of(x.s))
                        if (f.r == R.r1) zs.push_back(f.o);
                for (const string& Z : zs) {
                    if (Z == x.o) continue;
                    const long p1 = R.inv1 ? idx_of(Z, R.r1, x.s) : idx_of(x.s, R.r1, Z);
                    const long p2 = R.inv2 ? idx_of(x.o, R.r2, Z) : idx_of(Z, R.r2, x.o);
                    if (p1 >= 0 && p2 >= 0 && static_cast<std::size_t>(p1) != i && static_cast<std::size_t>(p2) != i)
                        inst.push_back({static_cast<std::size_t>(p1), static_cast<std::size_t>(p2)});
                }
            }
            for (auto& v : inst) F[i].derivations.push_back(ecs::FactDerivation{k, std::move(v), R.pca_conf()}), ++n_deriv;
        }
    }
    for (const auto& [s, fs] : v2.subjects()) {  // rivals in functional slots, and lifespan violations
        std::map<string, vector<std::size_t>> slot;
        for (const auto& f : fs) slot[f.r].push_back(static_cast<std::size_t>(idx_of(s, f.r, f.o)));
        for (const auto& [r, v] : slot) {
            const int ri = rel_index(r);
            const double ph = phi2[static_cast<std::size_t>(ri)];
            if (ph < 0.9) continue;
            for (std::size_t a : v) {
                F[a].phi = ph;
                for (std::size_t c : v)
                    if (a != c && !s8a::same_value(ri, F[a].o, F[c].o)) F[a].rivals.push_back(c), ++n_rival_pairs;
            }
        }
        for (const auto& f : fs) {
            const int k = date_kind(f.r);
            if (k != 1) continue;
            const Date B = parse_date_str(f.o);
            if (!B.ok()) continue;
            for (const auto& g : fs) {
                if (date_kind(g.r) != 2) continue;
                const Date D = parse_date_str(g.o);
                if (!D.ok() || !(D.y < B.y || D.y - B.y > 125)) continue;
                F[static_cast<std::size_t>(idx_of(s, f.r, f.o))].hard++, F[static_cast<std::size_t>(idx_of(s, g.r, g.o))].hard++, ++n_hard;
            }
        }
    }
    vector<ecs::FragFact> F_claim;  // S9b: the same model with Wikidata counted per claim
    vector<string> claim_name;
    if (s9b) {
        F_claim = F, claim_name = src_name;
        std::unordered_map<string, std::size_t> claim_id;
        for (auto& x : F_claim)
            for (auto& si : x.sources)
                if (src_name[si.source] == "wikidata") {
                    const string n = "wikidata|" + x.s + "|" + x.r + "|" + x.o;
                    auto [p, ins] = claim_id.emplace(n, claim_name.size());
                    if (ins) claim_name.push_back(n);
                    si.source = p->second;
                }
    }
    const ecs::FragilityModel M(std::move(F));
    const ecs::Retraction none;
    std::map<string, std::size_t> by_state;
    std::size_t n_held = 0, n_known = 0, n_bel_low = 0, n_contested = 0;
    vector<ecs::EpistemicState> s0(M.size());
    for (std::size_t i = 0; i < M.size(); ++i) {
        s0[i] = M.state(i, none);
        by_state[ecs::state_name(s0[i])]++;
        if (ecs::held_state(s0[i])) ++n_held;
        if (s0[i] == ecs::EpistemicState::Known) ++n_known;
        if (s0[i] == ecs::EpistemicState::Contested) ++n_contested;
        if (s0[i] == ecs::EpistemicState::Believed && M.full(i, none).grade == ecs::Grade::Low) ++n_bel_low;
    }
    {
        std::map<string, int> kinds;
        for (const auto& R : book2.rules()) kinds[ecs::rule_kind_name(R.kind)]++;
        string ks, ss;
        for (const auto& [k, v] : kinds) ks += " " + k + " " + std::to_string(v);
        for (const auto& [k, v] : by_state) ss += " " + k + " " + std::to_string(v);
        std::snprintf(b, sizeof b,
                      "S9 FRAG store v2 facts %zu (seed rows %zu, text facts %zu) | source instances %zu | rules kept %zu:%s | derivations %zu | rival pairs %zu | "
                      "lifespan violations %zu | states:%s | held %zu (Believed Low %zu = %.4f; Known %zu) | Contested %.4f of stored",
                      M.size(), seed_rows.size(), n_text_rows, src_name.size(), book2.rules().size(), ks.c_str(), n_deriv, n_rival_pairs, n_hard, ss.c_str(), n_held, n_bel_low,
                      n_held ? static_cast<double>(n_bel_low) / n_held : 0.0, n_known, M.size() ? static_cast<double>(n_contested) / M.size() : 0.0);
        std::cout << b << "\n";
    }
    std::size_t n_spof = 0, n_prem = 0, n_prem_drop = 0, n_top_drop = 0;
    std::map<string, std::array<std::size_t, 2>> spof_by_kind;  // the top source's kind -> held, single points of failure
    {
        std::ofstream ff(out + ".fragility.tsv"), fs(out + ".spof.tsv");
        ff << "subject\trelation\tobject\tstate\tgrade\tP\tsources\tderivations\trivals\tworlds\tdrops\tfragility\ttop_source\ttop_dropped\tweakest_premise\tpremise_dropped\n";
        fs << "gap_id\tkind\tstatus\tentity\trelation\tobject\tp\ttop_source\tnote\n";
        for (std::size_t i = 0; i < M.size(); ++i) {
            if (!ecs::held_state(s0[i])) continue;
            const auto& x = M.facts()[i];
            const auto fr = M.fragility(i);
            const auto top = M.top_source(i);
            const auto wp = M.weakest_premise(i);
            const ecs::Opinion o = M.full(i, none);
            const string tn = top ? src_name[*top] : "";
            const string tk = tn.substr(0, tn.find('|'));
            spof_by_kind[tk][0]++;
            n_top_drop += fr.top_dropped;
            if (fr.premise_world) ++n_prem, n_prem_drop += fr.premise_dropped;
            const bool spof = fr.worlds > 0 && fr.drops == fr.worlds;
            if (spof) {
                ++n_spof, spof_by_kind[tk][1]++;
                fs << ecs::gap_id(x.s, x.r) << "\tcertainty\topen\t" << x.s << '\t' << x.r << '\t' << x.o << '\t' << o.P << '\t' << tn << "\tsingle point of failure: "
                   << (fr.top_dropped ? "its top source" : "") << (fr.top_dropped && fr.premise_dropped ? " and " : "") << (fr.premise_dropped ? "its weakest premise" : "")
                   << " retracted, it is no longer held\n";
            }
            ff << x.s << '\t' << x.r << '\t' << x.o << '\t' << ecs::state_name(s0[i]) << '\t' << ecs::grade_name(o.grade) << '\t' << o.P << '\t' << x.sources.size() << '\t'
               << x.derivations.size() << '\t' << x.rivals.size() << '\t' << fr.worlds << '\t' << fr.drops << '\t' << fr.value() << '\t' << tn << '\t' << fr.top_dropped << '\t'
               << (wp ? M.facts()[*wp].s + " " + M.facts()[*wp].r + " " + M.facts()[*wp].o : "") << '\t' << fr.premise_dropped << '\n';
        }
    }
    std::snprintf(b, sizeof b,
                  "S9 FRAG held %zu | single points of failure %zu (%.4f) | top-source worlds drop %zu | premise worlds %zu (%.4f of held), drop %zu (%.4f of those)", n_held,
                  n_spof, n_held ? static_cast<double>(n_spof) / n_held : 0.0, n_top_drop, n_prem, n_held ? static_cast<double>(n_prem) / n_held : 0.0, n_prem_drop,
                  n_prem ? static_cast<double>(n_prem_drop) / n_prem : 0.0);
    std::cout << b << "\n";
    for (const auto& [k, v] : spof_by_kind) {
        std::snprintf(b, sizeof b, "S9 FRAG top source %-9s held %5zu single points of failure %5zu (%.4f)", k.c_str(), v[0], v[1], v[0] ? static_cast<double>(v[1]) / v[0] : 0.0);
        std::cout << b << "\n";
    }
    // ---- blast radius per source instance (one level of cascade)
    vector<std::pair<std::size_t, std::size_t>> blast;  // (radius, source)
    std::map<string, std::array<std::size_t, 3>> blast_kind;  // kind -> instances, total radius, max radius
    for (std::size_t s = 0; s < src_name.size(); ++s) {
        ecs::Retraction R;
        R.sources.insert(s);
        const std::size_t n = M.dropped_by(R).size();
        blast.push_back({n, s});
        const string k = src_name[s].substr(0, src_name[s].find('|'));
        auto& bk = blast_kind[k];
        bk[0]++, bk[1] += n, bk[2] = std::max(bk[2], n);
    }
    std::sort(blast.begin(), blast.end(), [](const auto& a, const auto& c) { return a.first != c.first ? a.first > c.first : a.second < c.second; });
    {
        std::ofstream fb(out + ".blast.tsv");
        fb << "source\tfacts\tradius\tshare_of_held\n";
        for (const auto& [n, s] : blast)
            fb << src_name[s] << '\t' << M.facts_of_source(s).size() << '\t' << n << '\t' << (n_held ? static_cast<double>(n) / n_held : 0.0) << '\n';
    }
    std::size_t max_other = 0;
    string max_other_name;
    for (const auto& [n, s] : blast)
        if (src_name[s] != "wikidata" && n > max_other) max_other = n, max_other_name = src_name[s];
    for (std::size_t i = 0; i < blast.size() && i < 10; ++i) {
        std::snprintf(b, sizeof b, "S9 BLAST %2zu %-50s facts %5zu radius %5zu (%.4f of held)", i + 1, src_name[blast[i].second].c_str(), M.facts_of_source(blast[i].second).size(),
                      blast[i].first, n_held ? static_cast<double>(blast[i].first) / n_held : 0.0);
        std::cout << b << "\n";
    }
    for (const auto& [k, v] : blast_kind) {
        std::snprintf(b, sizeof b, "S9 BLAST kind %-9s instances %5zu mean radius %.2f max %zu", k.c_str(), v[0], v[0] ? static_cast<double>(v[1]) / v[0] : 0.0, v[2]);
        std::cout << b << "\n";
    }
    std::snprintf(b, sizeof b, "S9 BLAST largest instance other than Wikidata: %s radius %zu", max_other_name.c_str(), max_other);
    std::cout << b << "\n";
    {  // cards: the five source instances with the largest blast radius, as worlds over the v2 store
        const ecs::ConflictChecks cc2 = world_checks(phi2, true);
        for (std::size_t q = 0; q < blast.size() && q < 5; ++q) {
            const std::size_t s = blast[q].second;
            ecs::World w(v2);
            ecs::Retraction R;
            R.sources.insert(s);
            for (std::size_t i : M.facts_of_source(s))
                if (!M.alive(i, R)) w.retract(M.facts()[i].s, M.facts()[i].r, M.facts()[i].o);
            ecs::WorldCard c;
            c.retracted = w.assumed_retracted();
            c.depth = 3;
            c.disappeared = ecs::vanished(w, book2, ecs::ChainConfig{3, 0.8});
            for (std::size_t i : M.dropped_by(R)) c.held_drops.push_back({M.facts()[i].s, M.facts()[i].r, M.facts()[i].o});
            ++i9.worlds, i9.changed += ecs::store_fingerprint(v2) != fp_v2;
            cards << "# source instance retracted: " << src_name[s] << " (" << c.retracted.size() << " facts lose their only source)\n" << ecs::format_world_card(c) << "\n";
        }
    }
    if (s9b) {  // ---- S9b: the store's phi for fusion (S7d item 6), Wikidata per claim, and whether fragility points at errors
        {
            std::ofstream fp(out + ".phi_store.tsv");
            fp << "relation\tphi\n";
            for (int r = 0; r < kNRel; ++r) fp << kRel[r].name << '\t' << phi2[static_cast<std::size_t>(r)] << '\n';
        }
        const ecs::FragilityModel Mc(std::move(F_claim));
        std::size_t c_held = 0, c_spof = 0, c_top = 0, same_state = 0;
        for (std::size_t i = 0; i < Mc.size(); ++i) {
            const ecs::EpistemicState sc = Mc.state(i, none);
            same_state += sc == s0[i];
            if (!ecs::held_state(sc)) continue;
            ++c_held;
            const auto fr = Mc.fragility(i);
            c_top += fr.top_dropped;
            c_spof += fr.worlds > 0 && fr.drops == fr.worlds;
        }
        std::snprintf(b, sizeof b, "S9B CLAIM held %zu | single points of failure %zu (%.4f) | top-source worlds drop %zu | states equal to one-source %zu of %zu", c_held, c_spof,
                      c_held ? static_cast<double>(c_spof) / c_held : 0.0, c_top, same_state, Mc.size());
        std::cout << b << "\n";
        vector<std::pair<std::size_t, std::size_t>> cb;
        std::map<string, std::array<std::size_t, 3>> ck;
        for (std::size_t s = 0; s < claim_name.size(); ++s) {
            if (Mc.facts_of_source(s).empty()) continue;
            ecs::Retraction R;
            R.sources.insert(s);
            const std::size_t n = Mc.dropped_by(R).size();
            cb.push_back({n, s});
            const string k = claim_name[s].substr(0, claim_name[s].find('|'));
            auto& v = ck[k];
            v[0]++, v[1] += n, v[2] = std::max(v[2], n);
        }
        std::sort(cb.begin(), cb.end(), [](const auto& a, const auto& c) { return a.first != c.first ? a.first > c.first : a.second < c.second; });
        for (std::size_t i = 0; i < cb.size() && i < 5; ++i) {
            std::snprintf(b, sizeof b, "S9B CLAIM BLAST %zu %-60s radius %zu (%.4f of held)", i + 1, claim_name[cb[i].second].c_str(), cb[i].first,
                          c_held ? static_cast<double>(cb[i].first) / c_held : 0.0);
            std::cout << b << "\n";
        }
        for (const auto& [k, v] : ck) {
            std::snprintf(b, sizeof b, "S9B CLAIM BLAST kind %-9s instances %5zu mean radius %.2f max %zu", k.c_str(), v[0], v[0] ? static_cast<double>(v[1]) / v[0] : 0.0, v[2]);
            std::cout << b << "\n";
        }
        // Held facts with no Wikidata source whose slot Wikidata covers: disagreement with Wikidata, single points of failure
        // against facts with two or more sources (not single points of failure); bootstrap over subjects.
        struct Acc {
            double n[2] = {0, 0}, d[2] = {0, 0};
        };
        std::map<string, Acc> by_s;
        std::size_t excl_wd = 0, uncovered = 0, other = 0;
        for (std::size_t i = 0; i < M.size(); ++i) {
            if (!ecs::held_state(s0[i])) continue;
            const auto& x = M.facts()[i];
            bool has_wd = false;
            for (const auto& si : x.sources) has_wd = has_wd || src_name[si.source] == "wikidata";
            if (has_wd) {
                ++excl_wd;
                continue;
            }
            const auto wit = wd.find(x.s + '\x1f' + x.r);
            if (wit == wd.end()) {
                ++uncovered;
                continue;
            }
            const auto fr = M.fragility(i);
            const bool spof = fr.worlds > 0 && fr.drops == fr.worlds;
            const int g = spof ? 0 : x.sources.size() >= 2 ? 1 : -1;
            if (g < 0) {
                ++other;
                continue;
            }
            Acc& a = by_s[x.s];
            a.n[g] += 1, a.d[g] += !s7::wd_match(rel_index(x.r), x.o, wit->second);
        }
        double N[2] = {0, 0}, D[2] = {0, 0};
        vector<const Acc*> subs;
        for (const auto& [k, a] : by_s) {
            subs.push_back(&a);
            for (int g = 0; g < 2; ++g) N[g] += a.n[g], D[g] += a.d[g];
        }
        auto rate = [](double d, double n) { return n > 0 ? d / n : 0.0; };
        const double diff = rate(D[0], N[0]) - rate(D[1], N[1]);
        std::mt19937_64 rb(20261029);
        vector<double> bd;
        for (int rr = 0; rr < 2000 && !subs.empty(); ++rr) {
            double n[2] = {0, 0}, d[2] = {0, 0};
            for (std::size_t q = 0; q < subs.size(); ++q) {
                const Acc* a = subs[rb() % subs.size()];
                for (int g = 0; g < 2; ++g) n[g] += a->n[g], d[g] += a->d[g];
            }
            bd.push_back(rate(d[0], n[0]) - rate(d[1], n[1]));
        }
        std::sort(bd.begin(), bd.end());
        const double lo = bd.empty() ? 0.0 : bd[static_cast<std::size_t>(0.025 * static_cast<double>(bd.size()))];
        const double hi = bd.empty() ? 0.0 : bd[std::min(bd.size() - 1, static_cast<std::size_t>(0.975 * static_cast<double>(bd.size())))];
        std::snprintf(b, sizeof b,
                      "S9B GRADE held facts with no Wikidata source, slot covered by Wikidata: single points of failure %.0f disagree %.0f (%.4f [%.4f, %.4f]) | "
                      "two or more sources %.0f disagree %.0f (%.4f [%.4f, %.4f]) | difference %+.4f [%+.4f, %+.4f] (bootstrap over %zu subjects) -> %s | "
                      "left out: with a Wikidata source %zu, slot not covered %zu, one source not a single point of failure %zu",
                      N[0], D[0], rate(D[0], N[0]), ecs::wilson_lower(D[0], N[0]), ecs::wilson_upper(D[0], N[0]), N[1], D[1], rate(D[1], N[1]),
                      ecs::wilson_lower(D[1], N[1]), ecs::wilson_upper(D[1], N[1]), diff, lo, hi, subs.size(),
                      lo > 0.0 ? "fragility points at errors: the fragile kind joins S3's documented command" : "not shown: the fragile kind stays a flag", excl_wd,
                      uncovered, other);
        std::cout << b << "\n";
    }
    // ---- I9: every base store unchanged; a write-back attempt on purpose
    {
        ecs::World w(st);
        w.add("Cypha", "country", "Nowhere");
        const bool wrote = w.write_back(st);
        ++i9.worlds;
        const bool same = ecs::store_fingerprint(st) == fp_st && ecs::store_fingerprint(v2) == fp_v2 && (!read_confirm || ecs::store_fingerprint(st15) == fp15);
        std::snprintf(b, sizeof b, "S9 I9 worlds %zu, base store changed after %zu | write-back attempt %s | fingerprints at the end %s (S7 reduced %016llx, v2 %016llx%s)", i9.worlds,
                      i9.changed, wrote ? "ACCEPTED" : "refused", same ? "unchanged" : "CHANGED", static_cast<unsigned long long>(fp_st), static_cast<unsigned long long>(fp_v2),
                      read_confirm ? ", S15 reduced checked" : "");
        std::cout << b << "\n";
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S9 seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s9

// ============================================================ S7d: T2 in the loop, Wikidata as one source, K3 against K4
// (docs/reports/S7_HYPOTHESIS_TESTS.md, "S7d"). pattern_learn --s7d S7_ITEMS S9_DEV_ITEMS S9B_DEV_ITEMS S15_ITEMS OUT
//   S7_ITEMS: --test-kinds … --s7c's OUT.items.tsv (dev); S9_DEV_ITEMS: --worlds' OUT.dev_items.tsv; S9B_DEV_ITEMS: --worlds
//   --s9b's OUT.s9b_dev_items.tsv; S15_ITEMS: --test-kinds' OUT.s15_items.tsv (S15's choose half). It first reproduces S7c's
//   dev lines from the joined items. Cypha's own hypothesis machinery only: nothing here answers a question.
namespace s7d {

using s7::Item;
using s7::LoopKind;
using s7::LoopRes;
using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;

TestOutcome outcome_of(const string& n) { return n == "positive" ? kPos : n == "negative" ? kNeg : kInc; }

struct Table {
    vector<string> head;
    vector<vector<string>> rows;
    std::size_t col(const string& n) const {
        const auto it = std::find(head.begin(), head.end(), n);
        if (it == head.end()) throw std::runtime_error("no column " + n);
        return static_cast<std::size_t>(it - head.begin());
    }
};
bool read_table(const string& path, Table& t) {
    std::ifstream f(path);
    if (!f) return false;
    string l;
    if (!std::getline(f, l)) return false;
    t.head = split(l, '\t');
    while (std::getline(f, l))
        if (!l.empty()) t.rows.push_back(split(l, '\t'));
    return true;
}

// A loop kind with three-way add-one likelihoods (and S7's add-one s and c) tallied over the items `over`, with S6's labels
// (false members Wikidata lists are left out) unless all_false: S7b's construction.
LoopKind make_kind(const char* name, const char* src, bool disc, bool indep, bool all_false, std::function<TestOutcome(const Item&)> f,
                   const vector<Item>& it, const vector<std::size_t>& over) {
    ecs::TestTally t;
    for (std::size_t i : over) {
        const Item& x = it[i];
        if (!x.label && x.wd_true && !all_false) continue;
        t.add(x.label == 1, f(x));
    }
    return {name, src, disc, indep, f, (t.t_pos + 1) / (t.t_pos + t.t_neg + 2), (t.f_neg + 1) / (t.f_pos + t.f_neg + 2), ecs::likelihoods_from(t)};
}

// The loop over idx in parts: part[j] picks the kinds for item idx[j]; results in idx order.
vector<LoopRes> loop_parts(const vector<Item>& it, const vector<std::size_t>& idx, const vector<int>& part, const vector<vector<LoopKind>>& kinds,
                           vector<vector<string>>* sup = nullptr) {
    vector<LoopRes> out(idx.size());
    vector<vector<string>> ss(idx.size());
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        vector<std::size_t> sub, pos;
        for (std::size_t j = 0; j < idx.size(); ++j)
            if (part[j] == static_cast<int>(k)) sub.push_back(idx[j]), pos.push_back(j);
        vector<vector<string>> s2;
        const vector<LoopRes> r = s7::loop_res(it, sub, kinds[k], "", &s2);
        for (std::size_t q = 0; q < sub.size(); ++q) out[pos[q]] = r[q], ss[pos[q]] = s2[q];
    }
    if (sup) *sup = std::move(ss);
    return out;
}

string kname(int k) { return s7::kKName[k]; }

int run(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: pattern_learn --s7d S7_ITEMS.tsv S9_DEV_ITEMS.tsv S9B_DEV_ITEMS.tsv S15_ITEMS.tsv OUT_PREFIX\n");
        return 2;
    }
    const double t0 = now_s();
    char b[2048];
    Table a, w, v, q;
    if (!read_table(argv[2], a) || !read_table(argv[3], w) || !read_table(argv[4], v) || !read_table(argv[5], q)) {
        std::fprintf(stderr, "cannot read an input\n");
        return 1;
    }
    // ---- dev items: S7's run (T6, T5, T4, P0) joined with S9's (T1 under the store's phi, T2) and S9b's (Wikidata claims)
    std::map<std::pair<int, int>, std::size_t> wrow, vrow;
    for (std::size_t i = 0; i < w.rows.size(); ++i) wrow[{std::atoi(w.rows[i][w.col("pair")].c_str()), std::atoi(w.rows[i][w.col("label")].c_str())}] = i;
    for (std::size_t i = 0; i < v.rows.size(); ++i) vrow[{std::atoi(v.rows[i][v.col("pair")].c_str()), std::atoi(v.rows[i][v.col("label")].c_str())}] = i;
    vector<Item> it;
    vector<TestOutcome> t2;
    vector<bool> wds;
    std::size_t mism = 0;
    for (const auto& r : a.rows) {
        if (r[a.col("split")] != "dev") continue;
        Item x;
        x.pair = std::atoi(r[a.col("pair")].c_str()), x.label = std::atoi(r[a.col("label")].c_str());
        x.set = r[a.col("set")][0], x.dev = true, x.wd_true = r[a.col("wd_true")] == "1";
        x.s = r[a.col("subject")], x.r = r[a.col("relation")], x.o = r[a.col("object")];
        x.t6[0] = outcome_of(r[a.col("T6")]), x.t5 = outcome_of(r[a.col("T5")]), x.t4[0] = outcome_of(r[a.col("T4")]);
        x.p0 = std::atof(r[a.col("P0")].c_str());
        const auto wi = wrow.find({x.pair, x.label});
        const auto vi = vrow.find({x.pair, x.label});
        if (wi == wrow.end() || vi == vrow.end()) {
            ++mism;
            continue;
        }
        const auto& wr = w.rows[wi->second];
        const auto& vr = v.rows[vi->second];
        if (wr[w.col("subject")] != x.s || wr[w.col("relation")] != x.r || wr[w.col("object")] != x.o || vr[v.col("subject")] != x.s) {
            ++mism;
            continue;
        }
        x.t1 = outcome_of(wr[w.col("T1")]);
        it.push_back(x), t2.push_back(outcome_of(wr[w.col("T2_d1_type")])), wds.push_back(vr[v.col("wd_subject")] == "1");
    }
    std::snprintf(b, sizeof b, "S7D DEV items joined %zu (S7 %zu, S9 %zu, S9b %zu) | not joined %zu", it.size(), a.rows.size(), w.rows.size(), v.rows.size(), mism);
    std::cout << b << "\n";
    if (mism) return 1;
    vector<std::size_t> all(it.size());
    std::iota(all.begin(), all.end(), 0);
    vector<std::size_t> fold[2];
    for (std::size_t i : all) fold[(fnv(it[i].s) >> 13) & 1].push_back(i);
    const Item* base = it.data();
    auto F1 = [](const Item& x) { return x.t1; };
    auto F6 = [](const Item& x) { return x.t6[0]; };
    auto F5 = [](const Item& x) { return x.t5; };
    auto F4 = [](const Item& x) { return x.t4[0]; };
    auto F2 = [base, &t2](const Item& x) { return t2[static_cast<std::size_t>(&x - base)]; };
    const LoopKind K1k = make_kind("T1 consistency", "store", true, false, false, F1, it, all), K6k = make_kind("T6 compression", "store", false, false, false, F6, it, all),
                   K5k = make_kind("T5 Wikidata", "wikidata", true, true, true, F5, it, all), K4k = make_kind("T4 corpus", "corpus", true, true, false, F4, it, all);
    const vector<LoopKind> lk = {K1k, K6k, K5k, K4k};
    // ---- the reproduction of S7c's dev lines (S8d's decide)
    vector<vector<string>> src0;
    const vector<LoopRes> rd = loop_parts(it, all, vector<int>(all.size(), 0), {lk}, &src0);
    vector<std::size_t> S(rd.size());
    std::iota(S.begin(), S.end(), 0);
    s7::KStats d0[s7::N_K];
    for (int k = 0; k < s7::N_K; ++k) {
        d0[k] = s7::s7c_stats(rd, S, s7::s7c_run(k, rd, S));
        std::cout << "S7D REPRODUCE " << s7::s7c_line("DEV", kname(k), d0[k]) << "\n";
    }
    // ---- item 1: T2 in the loop (cross-fitted by bit 13; the full-dev fit for the simulations)
    {
        const LoopKind K2full = make_kind("T2 worlds", "store", true, false, false, F2, it, all);
        vector<vector<LoopKind>> kf(2);
        for (int f = 0; f < 2; ++f) kf[static_cast<std::size_t>(f)] = {K1k, make_kind("T2 worlds", "store", true, false, false, F2, it, fold[1 - f]), K6k, K5k, K4k};
        vector<int> part(all.size());
        for (std::size_t j = 0; j < all.size(); ++j) part[j] = static_cast<int>((fnv(it[all[j]].s) >> 13) & 1);
        const vector<LoopRes> r2 = loop_parts(it, all, part, kf);
        for (int f = 0; f < 2; ++f) {
            const auto& L = kf[static_cast<std::size_t>(f)][1].lik;
            std::snprintf(b, sizeof b, "S7D T2 likelihoods for fold %d (fitted on fold %d): true pos %.4f neg %.4f inc %.4f | false pos %.4f neg %.4f inc %.4f | BF pos %.3f inc %.4f",
                          f, 1 - f, L.t[0], L.t[1], L.t[2], L.f[0], L.f[1], L.f[2], L.t[0] / L.f[0], L.t[2] / L.f[2]);
            std::cout << b << "\n";
        }
        {
            const auto& L = K2full.lik;
            std::snprintf(b, sizeof b, "S7D T2 likelihoods, all dev: true pos %.4f neg %.4f inc %.4f | false pos %.4f neg %.4f inc %.4f | BF pos %.3f inc %.4f", L.t[0], L.t[1],
                          L.t[2], L.f[0], L.f[1], L.f[2], L.t[0] / L.f[0], L.t[2] / L.f[2]);
            std::cout << b << "\n";
        }
        s7::KStats d2[s7::N_K];
        for (int k = 0; k < s7::N_K; ++k) {
            d2[k] = s7::s7c_stats(r2, S, s7::s7c_run(k, r2, S));
            std::cout << "S7D T2 " << s7::s7c_line("DEV", kname(k), d2[k]) << "\n";
        }
        std::size_t moved_sup = 0, moved_ref = 0;
        for (std::size_t j = 0; j < S.size(); ++j) moved_sup += rd[j].sup != r2[j].sup, moved_ref += rd[j].ref != r2[j].ref;
        std::snprintf(b, sizeof b, "S7D T2 items whose decide state changed: supported %zu, refuted %zu", moved_sup, moved_ref);
        std::cout << b << "\n";
        vector<double> pool;
        for (const LoopRes& r : rd) pool.push_back(r.p0);
        const vector<LoopKind> lk2 = {K1k, K2full, K6k, K5k, K4k};
        const s7::FdrAcc sm = s7::s7c_simulate(pool, lk2, rd.size(), 400, false, 20261104), sn = s7::s7c_simulate(pool, lk2, rd.size(), 400, true, 20261105),
                         rs = s7::s7c_resample(r2, 400, 20261106);
        std::cout << "S7D T2 " << s7::fdr_line("SIM mixed", sm) << "\nS7D T2 " << s7::fdr_line("SIM planted-nulls", sn) << "\nS7D T2 " << s7::fdr_line("DEV resampling", rs)
                  << "\n";
        const s7::KStats& k0 = d0[s7::K4];
        const s7::KStats& k2 = d2[s7::K4];
        auto ok = [](const s7::FdrAcc& x) { return x.fdr(s7::K4) <= 0.10 + 2 * x.se(s7::K4); };
        const bool ra = k2.decided() - k0.decided() >= 0.005, rb = ecs::wilson_lower(k2.pt, k2.prom()) >= 0.90 && k2.prec() >= k0.prec() - 0.01,
                   rc = ok(sm) && ok(sn) && ok(rs);
        std::snprintf(b, sizeof b,
                      "S7D T2 RULE (a) Known-decided %.4f -> %.4f (%+.4f; needs +0.0050) %s | (b) Known precision %.3f [%.3f] (without T2 %.3f) %s | (c) K4 FDR checks %s | "
                      "(d) S8e's fresh set: not read here -> %s",
                      k0.decided(), k2.decided(), k2.decided() - k0.decided(), ra ? "met" : "not met", k2.prec(), ecs::wilson_lower(k2.pt, k2.prom()), k0.prec(),
                      rb ? "met" : "not met", rc ? "met" : "not met", ra && rb && rc ? "T2 enters the loop (on dev)" : "T2 stays out of the loop");
        std::cout << b << "\n";
    }
    // ---- item 2: Wikidata as one source (T5 not independent where the reduced store holds a Wikidata claim about the subject)
    {
        std::size_t t5only = 0, t5only_t = 0, kn = 0;
        const vector<bool> k4 = s7::s7c_run(s7::K4, rd, S);
        for (std::size_t j = 0; j < S.size(); ++j) {
            if (!k4[j] || !rd[j].counted) continue;
            ++kn;
            if (src0[j].size() == 1 && src0[j][0] == "wikidata") ++t5only, t5only_t += rd[j].label == 1;
        }
        LoopKind K5one = K5k;
        K5one.independent = false;
        vector<int> part(all.size());
        std::size_t nwd = 0;
        for (std::size_t j = 0; j < all.size(); ++j) part[j] = wds[all[j]] ? 1 : 0, nwd += wds[all[j]];
        vector<vector<string>> src1;
        const vector<LoopRes> r1 = loop_parts(it, all, part, {lk, {K1k, K6k, K5one, K4k}}, &src1);
        s7::KStats d1[s7::N_K];
        for (int k = 0; k < s7::N_K; ++k) {
            d1[k] = s7::s7c_stats(r1, S, s7::s7c_run(k, r1, S));
            std::cout << "S7D ONE-SOURCE " << s7::s7c_line("DEV", kname(k), d1[k]) << "\n";
        }
        const s7::KStats& p = d0[s7::K4];
        const s7::KStats& o = d1[s7::K4];
        std::snprintf(b, sizeof b,
                      "S7D WIKIDATA items whose subject has a Wikidata claim in the store %zu of %zu | per claim: Known %.0f, resting on Wikidata alone %zu (true %zu, "
                      "precision %.3f [%.3f, %.3f]) | one source: Known %.0f at %.3f [%.3f] (%+.0f, %.1f%% of per claim's), Known-decided %.4f -> %.4f",
                      nwd, all.size(), p.prom(), t5only, t5only_t, t5only ? static_cast<double>(t5only_t) / static_cast<double>(t5only) : 0.0,
                      ecs::wilson_lower(static_cast<double>(t5only_t), static_cast<double>(t5only)), ecs::wilson_upper(static_cast<double>(t5only_t), static_cast<double>(t5only)),
                      o.prom(), o.prec(), ecs::wilson_lower(o.pt, o.prom()), o.prom() - p.prom(), p.prom() > 0 ? 100.0 * (o.prom() - p.prom()) / p.prom() : 0.0, p.decided(),
                      o.decided());
        std::cout << b << "\n";
        {  // where the promotions that rest on Wikidata alone (per claim) go when Wikidata is one source
            const vector<bool> k4o = s7::s7c_run(s7::K4, r1, S);
            std::size_t in_wd = 0, t4pos = 0, still = 0, alone_one = 0, diff = 0;
            for (std::size_t j = 0; j < S.size(); ++j) {
                diff += k4[j] != k4o[j];
                if (k4o[j] && r1[j].counted && src1[j].size() == 1 && src1[j][0] == "wikidata") ++alone_one;
                if (!k4[j] || !rd[j].counted || !(src0[j].size() == 1 && src0[j][0] == "wikidata")) continue;
                in_wd += wds[all[j]], t4pos += it[all[j]].t4[0] == kPos, still += k4o[j];
            }
            std::snprintf(b, sizeof b,
                          "S7D WIKIDATA the %zu resting on Wikidata alone (per claim): subject with a Wikidata claim in the store %zu | T4 positive when run %zu | Known "
                          "under one source %zu | Known promotions that differ between the two %zu | resting on Wikidata alone under one source %zu (subjects with no "
                          "Wikidata claim in the store)",
                          t5only, in_wd, t4pos, still, diff, alone_one);
            std::cout << b << "\n";
        }
        std::cout << "S7D WIKIDATA RULE " << (t5only == 0 ? "no Known promotion rests on Wikidata alone: per claim" : "Known promotions rest on Wikidata alone: one source")
                  << " (Known checked " << kn << ")\n";
    }
    // ---- item 5a: plausible nulls (false members from the top prior quartile), exchangeable calibration
    {
        vector<std::size_t> tr, fa;
        vector<double> fp0;
        for (std::size_t j = 0; j < rd.size(); ++j)
            if (rd[j].counted && rd[j].label == 0) fp0.push_back(rd[j].p0);
        std::sort(fp0.begin(), fp0.end());
        const double q3 = fp0.empty() ? 1.0 : fp0[fp0.size() * 3 / 4];
        for (std::size_t j = 0; j < rd.size(); ++j) {
            if (!rd[j].counted) continue;
            if (rd[j].label == 1) tr.push_back(j);
            else if (rd[j].p0 >= q3) fa.push_back(j);
        }
        double fe = 0;
        for (std::size_t j : fa) fe += rd[j].e;
        std::snprintf(b, sizeof b, "S7D 5a pools: true %zu | false in the top prior quartile (p0 >= %.4f) %zu, mean e %.3f, supported by decide %zu", tr.size(), q3, fa.size(),
                      fa.empty() ? 0.0 : fe / static_cast<double>(fa.size()), static_cast<std::size_t>(std::count_if(fa.begin(), fa.end(), [&](std::size_t j) { return rd[j].sup; })));
        std::cout << b << "\n";
        const int rules[3] = {s7::K3, s7::K4, s7::KREF};
        bool k4_over = false, k3_ok = true;
        for (const double pi : {0.5, 0.8}) {
            std::mt19937_64 g(20261107 + static_cast<std::uint64_t>(pi * 10));
            double sfd[3] = {}, s2[3] = {}, np[3] = {};
            const int reps = 400;
            const std::size_t m = 600, nf = static_cast<std::size_t>(pi * static_cast<double>(m));
            for (int rep = 0; rep < reps; ++rep) {
                vector<LoopRes> B, C;
                auto draw = [&](vector<LoopRes>& out) {
                    for (std::size_t i = 0; i < m; ++i) out.push_back(rd[i < nf ? fa[g() % fa.size()] : tr[g() % tr.size()]]);
                };
                draw(B), draw(C);
                vector<std::size_t> ib(m), ic(m);
                std::iota(ib.begin(), ib.end(), 0), std::iota(ic.begin(), ic.end(), 0);
                for (int u = 0; u < 3; ++u) {
                    const vector<bool> pr = s7::s7c_apply(rules[u], B, ib, C, ic);
                    double n = 0, f = 0;
                    for (std::size_t i = 0; i < m; ++i)
                        if (pr[i]) n += 1, f += B[i].label == 0;
                    const double fdp = n > 0 ? f / n : 0.0;
                    sfd[u] += fdp, s2[u] += fdp * fdp, np[u] += n;
                }
            }
            for (int u = 0; u < 3; ++u) {
                const double mean = sfd[u] / reps, se = std::sqrt(std::max(0.0, (s2[u] / reps - mean * mean) / (reps - 1)));
                const bool over = mean > 0.10 + 2 * se;
                if (rules[u] == s7::K4 && over) k4_over = true;
                if (rules[u] == s7::K3 && over) k3_ok = false;
                std::snprintf(b, sizeof b, "S7D 5a false fraction %.1f %-26s realised FDR %.4f (SE %.4f) promoted per batch %.1f%s", pi, kname(rules[u]).c_str(), mean, se,
                              np[u] / reps, over ? "  > 0.10 + 2 SE" : "");
                std::cout << b << "\n";
            }
        }
        // ---- item 5b: K3 calibrated on S7's dev, applied to S15's choose half (reachable)
        vector<Item> it15;
        for (const auto& r : q.rows) {
            if (r[q.col("half")] != "choose" || r[q.col("reachable")] != "1") continue;
            Item x;
            x.pair = std::atoi(r[q.col("pair")].c_str()), x.label = std::atoi(r[q.col("label")].c_str());
            x.s = r[q.col("subject")], x.r = r[q.col("relation")], x.o = r[q.col("object")];
            x.t1 = outcome_of(r[q.col("T1")]), x.t6[0] = outcome_of(r[q.col("T6")]), x.t4[0] = outcome_of(r[q.col("T4")]);
            x.p0 = std::atof(r[q.col("P0")].c_str());
            it15.push_back(x);
        }
        vector<std::size_t> i15(it15.size());
        std::iota(i15.begin(), i15.end(), 0);
        const vector<LoopRes> r15 = s7::loop_res(it15, i15, {K1k, K6k, K4k});
        vector<std::size_t> b15(r15.size());
        std::iota(b15.begin(), b15.end(), 0);
        auto fdp_of = [&](int rule, const vector<std::size_t>& bb, double* n_out) {
            const vector<bool> pr = rule == s7::K3 ? s7::s7c_apply(rule, r15, bb, rd, S) : s7::s7c_apply(rule, r15, bb, r15, {});
            double n = 0, f = 0;
            for (std::size_t i = 0; i < bb.size(); ++i)
                if (pr[i]) n += 1, f += r15[bb[i]].label == 0;
            if (n_out) *n_out = n;
            return n > 0 ? f / n : 0.0;
        };
        std::map<string, vector<std::size_t>> by_subj;
        for (std::size_t i = 0; i < it15.size(); ++i) by_subj[it15[i].s].push_back(i);
        vector<const vector<std::size_t>*> subs;
        for (const auto& [k, vv] : by_subj) subs.push_back(&vv);
        double pt[2], n0[2];
        vector<double> bs[2];
        const int r5b[2] = {s7::K3, s7::K4};
        for (int u = 0; u < 2; ++u) pt[u] = fdp_of(r5b[u], b15, &n0[u]);
        std::mt19937_64 gb(20261108);
        for (int rep = 0; rep < 2000; ++rep) {
            vector<std::size_t> bb;
            for (std::size_t z = 0; z < subs.size(); ++z)
                for (std::size_t i : *subs[gb() % subs.size()]) bb.push_back(i);
            for (int u = 0; u < 2; ++u) bs[u].push_back(fdp_of(r5b[u], bb, nullptr));
        }
        double hi[2], lo[2];
        for (int u = 0; u < 2; ++u) {
            std::sort(bs[u].begin(), bs[u].end());
            lo[u] = bs[u][static_cast<std::size_t>(0.025 * 2000)], hi[u] = bs[u][static_cast<std::size_t>(0.975 * 2000)];
            std::snprintf(b, sizeof b, "S7D 5b S15 choose half (reachable, %zu hypotheses, %zu subjects) %-26s promoted %.0f false share %.4f [%.4f, %.4f]%s", it15.size(),
                          subs.size(), kname(r5b[u]).c_str(), n0[u], pt[u], lo[u], hi[u], u == 0 ? " (calibrated on S7's dev)" : "");
            std::cout << b << "\n";
        }
        const bool ra = k4_over && k3_ok, rb = pt[0] <= 0.10 && hi[0] <= 0.15,
                   rc = std::fabs(d0[s7::K3].decided() - d0[s7::K4].decided()) <= 0.01;
        std::snprintf(b, sizeof b, "S7D K3 RULE (a) K4 over and K3 within in 5a: %s | (b) K3 across distributions %.4f, upper %.4f: %s | (c) Known-decided K3 %.4f K4 %.4f: %s -> %s",
                      ra ? "met" : "not met", pt[0], hi[0], rb ? "met" : "not met", d0[s7::K3].decided(), d0[s7::K4].decided(), rc ? "met" : "not met",
                      ra && rb && rc ? "the Known tier switches to K3" : "K4 stays");
        std::cout << b << "\n";
    }
    std::cout << "S7D FINAL S8e's fresh set: not read by this run (pass it once it is ready)\n";
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::snprintf(b, sizeof b, "S7D seconds %.1f max_rss_mb %.1f", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s7d

// ============================================================ S8c: other evidence sources and their reliability
// pattern_learn --sources S15_DEV TABLES WD_VALUES SEEDS FACTS_F2 OUT_PREFIX S8A_R1 S8A_R2 S8B_PROMO
// (DISCOVERY 7.1 T4/T5, 7.4, I4; docs/reports/S8C_EVIDENCE_SOURCES.md; Odin, 2026-10-29: "Get another evidence source").
// Sources (sources.hpp): WD (S15 dev's true rows and D4's wd_values), DB-IB (DBpedia's mapping-based properties, TABLES/
// dbpedia_ib.tsv), WP-LEAD (the current Wikipedia lead paragraph, TABLES/wp_lead.tsv, in place of DBpedia's abstracts),
// GN (GeoNames, TABLES/geonames.tsv and geonames_alias.tsv) and TXT (the 2006-text store: non-Wikidata seeds and admitted
// facts). Read 1: each source's three-way likelihoods per relation group on S15's choose half (bit 17 of FNV-1a(subject) =
// 0), against S15's labels (agreement with Wikidata), and its coverage by S15 category (the copy test). Read 2:
// Dawid-Skene over S8's dev hypotheses (no reference; an upper bound for correlated sources). Read 3: S8's dev hypotheses, their
// recorded posteriors updated by the accepted sources, decided and batch-controlled as in S7b, graded on S15's confirm half.
// Evidence for Cypha's own store only, never an answer for CyphaLM (PLAN rule 10). S15's test half is never read.
namespace s8c {
namespace src = cypha::eval::sources;
using ecs::TestOutcome;

constexpr int kNSrc = 5;
const char* const kSrcName[kNSrc] = {"WD", "DB-IB", "WP-LEAD", "GN", "TXT"};
enum { S_WD = 0, S_DB = 1, S_LEAD = 2, S_GN = 3, S_TXT = 4 };

struct Row {
    string id, s, sitem, stype, r, v, vitem, pair;
    char cat = ' ';
    int label = -1;
};

vector<Row> load_s15(const string& path) {
    vector<Row> out;
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < 17) continue;
        Row r;
        r.id = c[0], r.s = c[1], r.sitem = c[2], r.stype = c[3], r.r = c[5], r.v = c[7], r.vitem = c[8];
        r.cat = c[11].empty() ? ' ' : c[11][0];
        r.label = c[15] == "1" ? 1 : c[15] == "0" ? 0 : -1;
        r.pair = c[16];
        out.push_back(std::move(r));
    }
    return out;
}

bool choose_half(const string& s) { return ((fnv(s) >> 17) & 1) == 0; }

string group_of(const string& r) {
    static const std::unordered_map<string, string> g = {
        {"birth_date", "dates"}, {"death_date", "dates"}, {"birth_year", "dates"}, {"death_year", "dates"}, {"founded", "dates"},
        {"dissolved", "dates"}, {"birth_place", "places"}, {"death_place", "places"}, {"located_in", "places"}, {"country", "places"},
        {"capital", "places"}, {"largest_city", "places"}, {"headquarters", "places"}, {"location", "places"}};
    const auto it = g.find(r);
    if (it != g.end()) return it->second;
    static const std::unordered_set<string> lex = {"occupation", "genre", "language", "currency", "nationality", "award_received",
                                                   "educated_at", "member_of", "member_of_political_party", "party",
                                                   "position_held", "class", "family", "order"};
    return lex.count(r) ? "lexical" : "links";
}

struct Sources {
    src::Table wd, db, gn, txt;
    src::LeadText lead;
    TestOutcome answer(int k, const string& s, const string& r, const src::Value& v) const {
        switch (k) {
            case S_WD: return src::answer(wd, s, r, v);
            case S_DB: return src::answer(db, s, r, v);
            case S_LEAD: return lead.answer(s, r, v);
            case S_GN: return src::answer(gn, s, r, v);
            default: return src::answer(txt, s, r, v);
        }
    }
};

// One S8 dev hypothesis with what its generator recorded.
struct Hyp {
    string gen, s, r, o;
    double p0 = 0, post = 0;
    bool promoted = false;
};

vector<Hyp> load_hyps(const string& path, const string& gen, const string& want_split = "dev") {
    vector<Hyp> out;
    std::ifstream f(path);
    string l;
    std::getline(f, l);
    const auto h = split(l, '\t');
    auto col = [&](const char* n) { return static_cast<std::size_t>(std::find(h.begin(), h.end(), string(n)) - h.begin()); };
    const std::size_t cs = col("subject"), cr = col("relation"), co = col("object"), cpr = col("posterior"), cpm = col("promoted"),
                      csp = col("split");
    std::size_t cp0 = col("P0");
    if (cp0 >= h.size()) cp0 = col("prior");
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < h.size()) continue;
        if (csp < h.size() && c[csp] != want_split) continue;
        Hyp x;
        x.gen = gen, x.s = c[cs], x.r = c[cr], x.o = c[co];
        x.p0 = std::atof(c[cp0].c_str()), x.post = std::atof(c[cpr].c_str());
        x.promoted = cpm < h.size() && c[cpm] == "1";
        out.push_back(std::move(x));
    }
    return out;
}

// The grader on S15 (one half): +1 true, 0 false, -1 ungraded (registered rule).
struct Grader {
    std::unordered_map<string, vector<std::pair<src::Value, int>>> slot;  // norm_subject(s) + '\x1f' + r -> (value, label)
    void add(const Row& x) { slot[src::norm_subject(x.s) + '\x1f' + x.r].push_back({src::make_value(x.v, x.vitem), x.label}); }
    int grade(const string& s, const string& r, const src::Value& v) const {
        const auto it = slot.find(src::norm_subject(s) + '\x1f' + r);
        if (it == slot.end()) return -1;
        bool any_true = false, match_false = false;
        for (const auto& [val, lab] : it->second) {
            if (lab == 1) {
                any_true = true;
                if (src::same_value(r, val, v)) return 1;
            } else if (lab == 0 && src::same_value(r, val, v)) {
                match_false = true;
            }
        }
        if (match_false) return 0;
        return any_true && src::single_valued(r) ? 0 : -1;
    }
};

double odds(double p) {
    p = std::min(1.0 - 1e-12, std::max(1e-12, p));
    return p / (1.0 - p);
}

// ============================================================ S8d (docs/reports/S8D_PROMOTION.md)
// Calibration: each accepted (source, group) re-measured on the generators' own dev hypotheses graded by S15's choose half
// (each distinct one once); a source carries a promotion only where its calibrated LR+ >= 10. The source stage: a
// hypothesis starts from its generator's posterior, the accepted sources follow as tests (DB-IB, WP-LEAD, GN) with the
// calibrated likelihoods, and HypothesisStore's decide (a positive independent result, S8d) stops it; Known = row 70's K4
// on the source tests. Read on S8c's dev set (the confirm half, already read) and, with --fresh, once on the fresh set.
using Accepted = std::map<std::pair<int, string>, ecs::OutcomeLikelihoods>;

string item_key(const Hyp& h) { return src::norm_subject(h.s) + '\x1f' + h.r + '\x1f' + src::norm_name(h.o); }

struct Stage {
    double p0 = 0, post = 0, e = 1;
    bool sup = false, ref = false;
    int known = 0, graded = -1, ran = 0;  // ran: source tests run before it left the queue (S10's cost)
    string by;
    TestOutcome o[kNSrc] = {TestOutcome::Inconclusive, TestOutcome::Inconclusive, TestOutcome::Inconclusive, TestOutcome::Inconclusive,
                            TestOutcome::Inconclusive};
    vector<ecs::HypTest> plan;
};

vector<Stage> source_stage(const Sources& S, const vector<Hyp>& H, const Accepted& cal, const std::map<std::pair<int, string>, bool>& carry,
                           const Grader& G) {
    vector<Stage> st(H.size());
    for (std::size_t i = 0; i < H.size(); ++i) {
        const Hyp& h = H[i];
        Stage& z = st[i];
        const string grp = group_of(h.r);
        const src::Value v = src::make_value(h.o);
        for (int k = 0; k < kNSrc; ++k) z.o[k] = S.answer(k, h.s, h.r, v);
        ecs::Hypothesis x;
        x.id = "S8d-" + std::to_string(i);
        x.key = h.r + '|' + h.s + '|' + h.o;
        x.statement = h.s + " " + h.r + " " + h.o;
        x.generator = h.gen;
        vector<TestOutcome> outs;
        for (int k = 1; k < kNSrc; ++k) {
            const auto it = cal.find({k, grp});
            if (it == cal.end()) continue;
            ecs::HypTest t;
            t.kind = string("S8d ") + kSrcName[k], t.source = kSrcName[k], t.predicted = "positive", t.refuted_if = "negative";
            t.disconfirmation = true;
            t.independent = carry.at({k, grp});  // a source under the bar never carries a promotion
            t.likelihoods = it->second;
            x.tests.push_back(t), outs.push_back(z.o[k]);
        }
        z.plan = x.tests;
        z.p0 = std::min(1.0 - 1e-6, std::max(1e-6, h.post));
        ecs::HypothesisStore hs;
        hs.propose(x);
        hs.score(x.id, z.p0);
        hs.transition(x.id, ecs::HypState::Queued, "scored");
        for (std::size_t k = 0; k < outs.size(); ++k) {
            if (hs.find(x.id)->state != ecs::HypState::Queued) break;
            hs.transition(x.id, ecs::HypState::UnderTest, "next source");
            hs.record_outcome(x.id, k, outs[k]);
            ++z.ran;
        }
        const ecs::Hypothesis* hh = hs.find(x.id);
        z.post = hh->posterior, z.e = ecs::e_value(*hh);
        z.sup = hh->state == ecs::HypState::Supported, z.ref = hh->state == ecs::HypState::Refuted;
        for (const ecs::HypTest& t : hh->tests)
            if (ecs::independent_evidence(*hh, t) && t.positive) z.by += string(z.by.empty() ? "" : "+") + t.source;
        z.graded = G.grade(h.s, h.r, v);
    }
    vector<vector<ecs::HypTest>> plans;
    vector<double> pri, ev;
    vector<bool> sp;
    for (const Stage& z : st) plans.push_back(z.plan), pri.push_back(z.p0), ev.push_back(z.e), sp.push_back(z.sup);
    const vector<int> tiers = ecs::hybrid_null_model_tiers(plans, pri, ev, sp, 0.1);
    for (std::size_t i = 0; i < st.size(); ++i) st[i].known = tiers[i];
    return st;
}

// Precision lines of one read: per generator (rows), then each distinct hypothesis once (supported or Known when any copy
// is), by the source that gave the positive, by relation group and by relation. Returns {graded, true} of the supported.
std::pair<double, double> stage_report(const string& tag, const vector<Hyp>& H, const vector<Stage>& st, const string& tsv) {
    char b[1024];
    struct Acc {
        double n = 0, g = 0, sup = 0, sg = 0, stt = 0, kn = 0, kg = 0, kt = 0, ref = 0, rg = 0, rf = 0;
    };
    auto line = [&](const string& what, const Acc& a) {
        std::snprintf(b, sizeof b,
                      "S8D %s %-26s hyps %5.0f graded %5.0f | supported %4.0f, graded %4.0f, true %4.0f, precision %.3f [%.3f, %.3f] | Known %4.0f, "
                      "graded %4.0f, precision %.3f [%.3f, %.3f] | refuted %4.0f, graded %4.0f, false %.3f",
                      tag.c_str(), what.c_str(), a.n, a.g, a.sup, a.sg, a.stt, a.sg ? a.stt / a.sg : 0.0, ecs::wilson_lower(a.stt, a.sg),
                      ecs::wilson_upper(a.stt, a.sg), a.kn, a.kg, a.kg ? a.kt / a.kg : 0.0, ecs::wilson_lower(a.kt, a.kg), ecs::wilson_upper(a.kt, a.kg),
                      a.ref, a.rg, a.rg ? a.rf / a.rg : 0.0);
        std::cout << b << "\n";
    };
    auto add = [](Acc& a, int g, bool sup, bool known, bool ref) {
        a.n += 1;
        if (g >= 0) a.g += 1;
        if (sup) a.sup += 1, a.sg += g >= 0, a.stt += g == 1;
        if (known) a.kn += 1, a.kg += g >= 0, a.kt += g == 1;
        if (ref) a.ref += 1, a.rg += g >= 0, a.rf += g == 0;
    };
    std::map<string, Acc> gen;
    for (std::size_t i = 0; i < H.size(); ++i) add(gen[H[i].gen], st[i].graded, st[i].sup, st[i].known == 2, st[i].ref);
    for (const auto& [g, a] : gen) line("generator " + g, a);
    // each distinct hypothesis once
    struct D {
        std::size_t first = 0;
        int graded = -1;
        bool sup = false, known = false, ref = false;
        string by;
    };
    std::map<string, D> dist;
    vector<string> order;
    for (std::size_t i = 0; i < H.size(); ++i) {
        const string k = item_key(H[i]);
        auto it = dist.find(k);
        if (it == dist.end()) {
            D d;
            d.first = i, d.graded = st[i].graded;
            it = dist.emplace(k, d).first, order.push_back(k);
        }
        D& d = it->second;
        if (st[i].sup && !d.sup) d.by = st[i].by;
        d.sup = d.sup || st[i].sup, d.known = d.known || st[i].known == 2, d.ref = d.ref || st[i].ref;
    }
    Acc all;
    std::map<string, Acc> by_src, by_grp, by_rel;
    for (const string& k : order) {
        const D& d = dist[k];
        const bool ref = d.ref && !d.sup;
        const Hyp& h = H[d.first];
        add(all, d.graded, d.sup, d.known, ref);
        add(by_grp[group_of(h.r)], d.graded, d.sup, d.known, ref);
        add(by_rel[h.r], d.graded, d.sup, d.known, ref);
        if (d.sup) add(by_src[d.by], d.graded, true, d.known, false);
    }
    line("all (each distinct once)", all);
    for (const auto& [s, a] : by_src) line("positive from " + s, a);
    for (const auto& [g, a] : by_grp) line("group " + g, a);
    for (const auto& [r, a] : by_rel)
        if (a.sup > 0) line("relation " + r, a);
    std::ofstream f(tsv);
    f << "generator\tsubject\trelation\tobject\tposterior_gen\tDB-IB\tWP-LEAD\tGN\tposterior_src\te_src\tsupported\tknown\trefuted\tby\tgraded\n";
    for (std::size_t i = 0; i < H.size(); ++i) {
        const Stage& z = st[i];
        std::snprintf(b, sizeof b, "%s\t%s\t%s\t%s\t%.6f\t%s\t%s\t%s\t%.6f\t%.6g\t%d\t%d\t%d\t%s\t%d\n", H[i].gen.c_str(), H[i].s.c_str(), H[i].r.c_str(),
                      H[i].o.c_str(), z.p0, ecs::test_outcome_name(z.o[1]), ecs::test_outcome_name(z.o[2]), ecs::test_outcome_name(z.o[3]), z.post, z.e,
                      z.sup ? 1 : 0, z.known == 2 ? 1 : 0, z.ref ? 1 : 0, z.by.c_str(), z.graded);
        f << b;
    }
    return {all.sg, all.stt};
}

void s8d(const Sources& S, const vector<Hyp>& hyps, const Accepted& accepted, const Grader& conf, const Grader& choose, const string& fresh_g2,
         const string& fresh_links, const string& out) {
    char b[1024];
    // ---- calibration: the generators' dev hypotheses on S15's choose half, each distinct one once
    std::map<std::pair<int, string>, ecs::TestTally> cell;
    std::map<int, ecs::TestTally> pooled;
    {
        std::set<string> seen;
        double n = 0, nt = 0;
        for (const Hyp& h : hyps) {
            if (!choose_half(h.s) || !seen.insert(item_key(h)).second) continue;
            const src::Value v = src::make_value(h.o);
            const int g = choose.grade(h.s, h.r, v);
            if (g < 0) continue;
            n += 1, nt += g;
            for (const auto& [kg, L] : accepted) {
                if (kg.second != group_of(h.r)) continue;
                const TestOutcome o = S.answer(kg.first, h.s, h.r, v);
                cell[kg].add(g == 1, o), pooled[kg.first].add(g == 1, o);
            }
        }
        std::snprintf(b, sizeof b, "S8D CALIBRATION set: %.0f distinct dev hypotheses of S8a-R1, S8a-R2 and S8b graded on S15's choose half, %.0f true (%.3f)",
                      n, nt, n > 0 ? nt / n : 0.0);
        std::cout << b << "\n";
    }
    Accepted cal;
    std::map<std::pair<int, string>, bool> carry;
    std::cout << "S8D CALIBRATION (add-one; a cell with < 10 graded true or false uses the source's pooled cell; carries a promotion only if LR+ >= 10):\n";
    auto tl = [&](const ecs::TestTally& t) {
        std::snprintf(b, sizeof b, "true %3.0f (pos %3.0f neg %3.0f inc %3.0f) false %3.0f (pos %3.0f neg %3.0f inc %3.0f) | pos then true %3.0f/%-3.0f",
                      t.t_pos + t.t_neg + t.t_inc, t.t_pos, t.t_neg, t.t_inc, t.f_pos + t.f_neg + t.f_inc, t.f_pos, t.f_neg, t.f_inc, t.t_pos,
                      t.t_pos + t.f_pos);
        return string(b);
    };
    for (const auto& [k, t] : pooled) {
        const ecs::OutcomeLikelihoods L = ecs::likelihoods_from(t);
        std::snprintf(b, sizeof b, "  %-7s pooled   %s | LR+ %7.2f", kSrcName[k], tl(t).c_str(), src::positive_lr(L));
        std::cout << b << "\n";
    }
    for (const auto& [kg, L1] : accepted) {
        bool own = false;
        const ecs::OutcomeLikelihoods L = src::calibrated_likelihoods(cell[kg], pooled[kg.first], 10.0, &own);
        cal[kg] = L, carry[kg] = src::may_carry_promotion(L);
        std::snprintf(b, sizeof b, "  %-7s %-8s %s | uses %-6s LR+ %7.2f BF(neg) %.3f BF(inc) %.3f | read 1 LR+ %7.2f -> %s", kSrcName[kg.first],
                      kg.second.c_str(), tl(cell[kg]).c_str(), own ? "own" : "pooled", src::positive_lr(L), L.t[1] / L.f[1], L.t[2] / L.f[2],
                      src::positive_lr(L1), carry[kg] ? "MAY CARRY" : "may not carry");
        std::cout << b << "\n";
    }
    // ---- S8c's dev set again, with the fix and the calibrated sources (the confirm half was read before; nothing is chosen here)
    {
        const vector<Stage> st = source_stage(S, hyps, cal, carry, conf);
        stage_report("DEV-CONFIRM(already read)", hyps, st, out + ".s8d_dev_confirm.tsv");
        const vector<Stage> sc = source_stage(S, hyps, cal, carry, choose);
        stage_report("DEV-CHOOSE(calibration set, in-sample)", hyps, sc, out + ".s8d_dev_choose.tsv");
    }
    if (fresh_g2.empty()) return;
    // ---- the fresh set: read once
    vector<Hyp> fresh;
    for (const auto& [path, gen] : {std::pair<string, string>{fresh_g2, "G2-fresh"}, {fresh_links, "G6G7-fresh"}}) {
        auto v = load_hyps(path, gen, "fresh");
        fresh.insert(fresh.end(), v.begin(), v.end());
    }
    std::size_t off_half = 0;
    for (const Hyp& h : fresh) off_half += choose_half(h.s);
    std::snprintf(b, sizeof b, "S8D FRESH set: %zu hypotheses (%s, %s), subjects outside the confirm half %zu", fresh.size(), fresh_g2.c_str(),
                  fresh_links.c_str(), off_half);
    std::cout << b << "\n";
    const vector<Stage> st = source_stage(S, fresh, cal, carry, conf);
    const auto [g, t] = stage_report("FRESH", fresh, st, out + ".s8d_fresh.tsv");
    {  // S10: the source tests each hypothesis ran (its cost), in the rows' order (docs/reports/S10_CURIOSITY_SCHEDULER.md)
        std::ofstream rf(out + ".s8d_fresh_ran.tsv");
        rf << "generator\tsubject\trelation\tobject\tsources_run\n";
        for (std::size_t i = 0; i < fresh.size(); ++i)
            rf << fresh[i].gen << '\t' << fresh[i].s << '\t' << fresh[i].r << '\t' << fresh[i].o << '\t' << st[i].ran << '\n';
    }
    std::snprintf(b, sizeof b, "S8D GATE (fresh, supported, each distinct once): graded %.0f, precision %.3f [%.3f, %.3f] -> %s", g, g > 0 ? t / g : 0.0,
                  ecs::wilson_lower(t, g), ecs::wilson_upper(t, g), g >= 20 && t / g >= 0.9 ? "MET" : "NOT MET");
    std::cout << b << "\n";
}

int run(int argc, char** argv) {
    if (argc < 11) {
        std::fprintf(stderr, "usage: pattern_learn --sources S15_DEV TABLES WD_VALUES SEEDS FACTS_F2 OUT_PREFIX S8A_R1 S8A_R2 S8B_PROMO "
                             "[--calibrate | --fresh FRESH_G2 FRESH_LINKS]\n");
        return 2;
    }
    const double t0 = now_s();
    const string s15p = argv[2], tables = argv[3], out = argv[7];
    bool calibrate = false;  // S8d: calibration on the generators' choose-half hypotheses, and S8c's dev set again
    string fresh_g2, fresh_links;  // S8d: the fresh confirm-half set, read once
    for (int i = 11; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--calibrate") calibrate = true;
        else if (a == "--fresh" && i + 2 < argc) fresh_g2 = argv[i + 1], fresh_links = argv[i + 2], i += 2, calibrate = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    char b[1024];
    // ---- the sources
    Sources S;
    S.wd.name = "WD", S.db.name = "DB-IB", S.gn.name = "GN", S.txt.name = "TXT";
    const vector<Row> rows = load_s15(s15p);
    for (const Row& x : rows)
        if (x.label == 1) S.wd.add(x.s, x.r, x.v, x.vitem);
    {
        std::ifstream f(argv[4]);  // D4's wd_values: subject, relation, surfaces, subject_item, value
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 5) continue;
            for (const string& sf : split(c[2], '|'))
                if (!sf.empty()) S.wd.add(c[0], c[1], sf, c[4]);
        }
    }
    auto load_table = [&](src::Table& t, const string& path, int vcol, int qcol) {
        std::ifstream f(path);
        string l;
        std::getline(f, l);
        std::size_t n = 0;
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (static_cast<int>(c.size()) <= std::max(vcol, qcol)) continue;
            t.add(c[0], c[1], c[static_cast<std::size_t>(vcol)], qcol >= 0 ? c[static_cast<std::size_t>(qcol)] : string());
            ++n;
        }
        return n;
    };
    const std::size_t n_db = load_table(S.db, tables + "/dbpedia_ib.tsv", 2, 3);
    const std::size_t n_gn = load_table(S.gn, tables + "/geonames.tsv", 2, -1);
    {
        std::ifstream f(tables + "/geonames_alias.tsv");
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() >= 2) S.gn.add_alias(c[0], c[1]);
        }
    }
    std::size_t n_lead = 0;
    {
        std::ifstream f(tables + "/wp_lead.tsv");
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() >= 2) S.lead.add(c[0], c[1]), ++n_lead;
        }
    }
    std::size_t n_txt = 0;
    {
        std::ifstream f(argv[5]);  // seeds: subject, relation, object, source, ...
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() >= 4 && c[3] != "wikidata" && c[3] != "source") S.txt.add(c[0], c[1], c[2]), ++n_txt;
        }
        std::ifstream g(argv[6]);  // admitted facts: subject, relation, object, ...
        std::getline(g, l);
        while (std::getline(g, l)) {
            const auto c = split(l, '\t');
            if (c.size() >= 3) S.txt.add(c[0], c[1], c[2]), ++n_txt;
        }
    }
    std::snprintf(b, sizeof b, "S8C sources: S15 dev rows %zu | DB-IB rows %zu subjects %zu | WP-LEAD extracts %zu | GN facts %zu aliases %zu | "
                  "TXT facts %zu | WD slots %zu", rows.size(), n_db, S.db.subjects.size(), n_lead, n_gn, S.gn.alias.size(), n_txt,
                  S.wd.slots.size());
    std::cout << b << "\n";

    // ---- read 1: S15's choose half, agreement with Wikidata (S15's labels), per source and relation group
    std::unordered_map<string, const Row*> by_id;
    for (const Row& x : rows) by_id[x.id] = &x;
    struct Cell {
        ecs::TestTally t;
        double cov[4] = {0, 0, 0, 0}, n[4] = {0, 0, 0, 0};  // true items covered / all, by category a, b, c, u
    };
    std::map<std::pair<int, string>, Cell> rel1;   // (source, relation)
    std::map<std::pair<int, string>, Cell> grp1;   // (source, group or "all")
    std::size_t pairs_choose = 0;
    for (const Row& x : rows) {
        if (x.label != 1 || !choose_half(x.s)) continue;
        const auto fi = by_id.find(x.pair);
        if (fi == by_id.end() || fi->second->label != 0) continue;
        ++pairs_choose;
        const Row& y = *fi->second;
        const int ci = x.cat == 'a' ? 0 : x.cat == 'b' ? 1 : x.cat == 'c' ? 2 : 3;
        for (int k = 1; k < kNSrc; ++k) {
            const TestOutcome ot = S.answer(k, x.s, x.r, src::make_value(x.v, x.vitem));
            const TestOutcome of = S.answer(k, y.s, y.r, src::make_value(y.v, y.vitem));
            for (auto* m : {&rel1, &grp1}) {
                const string key = m == &rel1 ? x.r : group_of(x.r);
                for (const string& kk : {key, string(m == &grp1 ? "all" : "")}) {
                    if (kk.empty()) continue;
                    Cell& c = (*m)[{k, kk}];
                    c.t.add(true, ot), c.t.add(false, of);
                    c.n[ci] += 1, c.cov[ci] += ot != TestOutcome::Inconclusive;
                }
            }
        }
    }
    std::ofstream rf(out + ".reliability.tsv");
    rf << "source\tscope\tn_true\tn_false\tcoverage\tt_pos\tt_neg\tt_inc\tf_pos\tf_neg\tf_inc\tsens_covered\tfpr\tfpr_hi95\tlr_pos\tmi_bits"
          "\tcov_a\tcov_b\tcov_c\tcov_u\n";
    auto lr_pos = [](const ecs::TestTally& t) {
        const ecs::OutcomeLikelihoods L = ecs::likelihoods_from(t);
        return L.t[0] / L.f[0];
    };
    auto emit = [&](int k, const string& scope, const Cell& c) {
        const auto& t = c.t;
        const double nt = t.t_pos + t.t_neg + t.t_inc, nf = t.f_pos + t.f_neg + t.f_inc;
        double cv[4];
        for (int i = 0; i < 4; ++i) cv[i] = c.n[i] > 0 ? c.cov[i] / c.n[i] : -1;
        std::snprintf(b, sizeof b, "%s\t%s\t%.0f\t%.0f\t%.4f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.4f\t%.4f\t%.4f\t%.2f\t%.5f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                      kSrcName[k], scope.c_str(), nt, nf, nt > 0 ? (t.t_pos + t.t_neg) / nt : 0.0, t.t_pos, t.t_neg, t.t_inc, t.f_pos,
                      t.f_neg, t.f_inc, t.t_pos + t.t_neg > 0 ? t.t_pos / (t.t_pos + t.t_neg) : 0.0, nf > 0 ? t.f_pos / nf : 0.0,
                      ecs::wilson_upper(t.f_pos, nf), lr_pos(t), t.mutual_information_bits(), cv[0], cv[1], cv[2], cv[3]);
        rf << b;
    };
    for (const auto& [k, c] : grp1) emit(k.first, "group:" + k.second, c);
    for (const auto& [k, c] : rel1) emit(k.first, "rel:" + k.second, c);
    std::snprintf(b, sizeof b, "S8C read 1 (S15 choose half, %zu pairs; reference = S15's labels, i.e. agreement with Wikidata):", pairs_choose);
    std::cout << b << "\n";
    for (int k = 1; k < kNSrc; ++k)
        for (const char* g : {"all", "dates", "places", "links", "lexical"}) {
            const auto it = grp1.find({k, g});
            if (it == grp1.end()) continue;
            const auto& t = it->second.t;
            const double nt = t.t_pos + t.t_neg + t.t_inc, nf = t.f_pos + t.f_neg + t.f_inc;
            const auto& c = it->second;
            std::snprintf(b, sizeof b,
                          "  %-7s %-8s true %5.0f covered %.3f pos|true %.3f [%.3f, %.3f] | pos|false %.4f (hi95 %.4f) neg|false %.3f | "
                          "LR+ %7.2f | cov a %.3f b %.3f c %.3f u %.3f",
                          kSrcName[k], g, nt, nt > 0 ? (t.t_pos + t.t_neg) / nt : 0.0, nt > 0 ? t.t_pos / nt : 0.0,
                          ecs::wilson_lower(t.t_pos, nt), ecs::wilson_upper(t.t_pos, nt), nf > 0 ? t.f_pos / nf : 0.0,
                          ecs::wilson_upper(t.f_pos, nf), nf > 0 ? t.f_neg / nf : 0.0, lr_pos(t), c.n[0] ? c.cov[0] / c.n[0] : -1.0,
                          c.n[1] ? c.cov[1] / c.n[1] : -1.0, c.n[2] ? c.cov[2] / c.n[2] : -1.0, c.n[3] ? c.cov[3] / c.n[3] : -1.0);
            std::cout << b << "\n";
        }

    // ---- read 2: Dawid-Skene over S8's dev hypotheses (no reference)
    vector<Hyp> hyps;
    for (const auto& [path, gen] : {std::pair<string, string>{argv[8], "S8a-R1"}, {argv[9], "S8a-R2"}, {argv[10], "S8b"}}) {
        auto v = load_hyps(path, gen);
        hyps.insert(hyps.end(), v.begin(), v.end());
    }
    std::map<string, std::size_t> item_of;
    vector<vector<int>> Y;
    vector<std::array<string, 3>> items;
    for (const Hyp& h : hyps) {
        const string key = src::norm_subject(h.s) + '\x1f' + h.r + '\x1f' + src::norm_name(h.o);
        if (item_of.count(key)) continue;
        item_of[key] = Y.size();
        items.push_back({h.s, h.r, h.o});
        vector<int> y(kNSrc);
        const src::Value v = src::make_value(h.o);
        for (int k = 0; k < kNSrc; ++k) y[static_cast<std::size_t>(k)] = src::outcome_index(S.answer(k, h.s, h.r, v));
        Y.push_back(std::move(y));
    }
    const src::DSFit ds = src::dawid_skene(Y, kNSrc);
    const vector<double> post = src::ds_posterior(ds, Y);
    std::snprintf(b, sizeof b, "S8C read 2 (Dawid-Skene over %zu distinct S8 dev hypotheses, %d iterations): share true %.3f", Y.size(),
                  ds.iterations, ds.prior);
    std::cout << b << "\n";
    std::ofstream df(out + ".ds.tsv");
    df << "source\tt_pos\tt_neg\tt_inc\tf_pos\tf_neg\tf_inc\taccuracy_conclusive\n";
    for (int k = 0; k < kNSrc; ++k) {
        const auto& T = ds.t[static_cast<std::size_t>(k)];
        const auto& F = ds.f[static_cast<std::size_t>(k)];
        const double num = ds.prior * T[0] + (1 - ds.prior) * F[1], den = ds.prior * (T[0] + T[1]) + (1 - ds.prior) * (F[0] + F[1]);
        std::snprintf(b, sizeof b, "  %-7s P(pos|T) %.3f P(neg|T) %.3f P(inc|T) %.3f | P(pos|F) %.4f P(neg|F) %.3f P(inc|F) %.3f | accuracy when conclusive %.3f",
                      kSrcName[k], T[0], T[1], T[2], F[0], F[1], F[2], den > 0 ? num / den : 0.0);
        std::cout << b << "\n";
        std::snprintf(b, sizeof b, "%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n", kSrcName[k], T[0], T[1], T[2], F[0], F[1], F[2],
                      den > 0 ? num / den : 0.0);
        df << b;
    }
    // ---- the registered decision: which (source, group) become test kinds
    std::map<std::pair<int, string>, ecs::OutcomeLikelihoods> accepted;
    std::cout << "S8C decision (choose half; LR+ >= 10, false-positive Wilson hi < 0.10, coverage >= 0.05, copy test b, c < 0.9 of a; "
                 "the Dawid-Skene dependence ratio is dropped, see the report's amendment):\n";
    for (int k = 1; k < kNSrc; ++k) {
        if (k == S_TXT) continue;  // the 2006-text store is the generators' own source: never independent (I4)
        const Cell& all = grp1[{k, "all"}];
        const double ca = all.n[0] ? all.cov[0] / all.n[0] : 0.0, cb = all.n[1] ? all.cov[1] / all.n[1] : 0.0,
                     cc = all.n[2] ? all.cov[2] / all.n[2] : 0.0;
        const bool copy_ok = ca > 0 && cb < 0.9 * ca && cc < 0.9 * ca;
        for (const char* g : {"dates", "places", "links", "lexical"}) {
            const auto it = grp1.find({k, g});
            if (it == grp1.end()) continue;
            const auto& t = it->second.t;
            const double nt = t.t_pos + t.t_neg + t.t_inc, nf = t.f_pos + t.f_neg + t.f_inc;
            const double cov = nt > 0 ? (t.t_pos + t.t_neg) / nt : 0.0, lr = lr_pos(t), fhi = ecs::wilson_upper(t.f_pos, nf);
            const bool ok = lr >= 10 && fhi < 0.10 && cov >= 0.05 && copy_ok;
            std::snprintf(b, sizeof b, "  %-7s %-8s LR+ %7.2f fpr_hi %.4f coverage %.3f copy-test (b %.3f, c %.3f of a %.3f) %s -> %s", kSrcName[k],
                          g, lr, fhi, cov, cb, cc, ca, copy_ok ? "pass" : "fail", ok ? "ACCEPTED" : "no");
            std::cout << b << "\n";
            if (ok) accepted[{k, g}] = ecs::likelihoods_from(t);
        }
    }

    // ---- read 3: promotion with the accepted sources, graded on S15's confirm half
    Grader conf, choose;
    for (const Row& x : rows) (choose_half(x.s) ? choose : conf).add(x);
    struct Res {
        double post2 = 0, e = 1;
        bool indep = false, supported = false, refuted = false;
        int graded = -1;
        string by;
        TestOutcome o[kNSrc] = {TestOutcome::Inconclusive, TestOutcome::Inconclusive, TestOutcome::Inconclusive,
                                TestOutcome::Inconclusive, TestOutcome::Inconclusive};
    };
    vector<Res> res(hyps.size());
    vector<double> evals(hyps.size());
    vector<bool> sel(hyps.size());
    for (std::size_t i = 0; i < hyps.size(); ++i) {
        const Hyp& h = hyps[i];
        Res& R = res[i];
        const src::Value v = src::make_value(h.o);
        double o = odds(h.post);
        R.e = odds(h.post) / odds(h.p0);
        for (int k = 0; k < kNSrc; ++k) R.o[k] = S.answer(k, h.s, h.r, v);
        for (int k = 1; k < kNSrc; ++k) {
            const auto it = accepted.find({k, group_of(h.r)});
            if (it == accepted.end()) continue;
            const int oi = src::outcome_index(R.o[k]);
            const double bf = std::max(1e-6, it->second.t[oi]) / std::max(1e-6, it->second.f[oi]);
            o *= bf, R.e *= bf;
            if (R.o[k] == TestOutcome::Positive) R.indep = true, R.by += string(R.by.empty() ? "" : "+") + kSrcName[k];  // S8d: a positive one
        }
        R.post2 = o / (1.0 + o);
        R.supported = R.post2 > 0.9 && R.indep;
        R.refuted = R.post2 < 0.05;
        R.graded = conf.grade(h.s, h.r, v);
        evals[i] = R.e, sel[i] = R.supported;
    }
    const vector<bool> keep_among = ecs::bh_reject_e_among(evals, sel, 0.1), keep_all = ecs::bh_reject_e(evals, 0.1);
    std::ofstream pf(out + ".promo.tsv");
    pf << "generator\tsubject\trelation\tobject\tP0\tposterior_gen\tWD\tDB-IB\tWP-LEAD\tGN\tTXT\tposterior_src\tsupported\tpromoted_among"
          "\tpromoted_ebh\trefuted\tby\tgraded_confirm\tgraded_choose\tgen_promoted\n";
    std::cout << "S8C read 3 (promotion; graded on S15's confirm half: precision = true / graded, Wilson 95%):\n";
    std::map<string, std::array<double, 12>> tally;  // gen -> sup, sup_g, sup_t, among_g, among_t, ebh_g, ebh_t, ref_g, ref_f, gen_g, gen_t, n
    for (std::size_t i = 0; i < hyps.size(); ++i) {
        const Hyp& h = hyps[i];
        const Res& R = res[i];
        const int gch = choose.grade(h.s, h.r, src::make_value(h.o));
        std::snprintf(b, sizeof b, "%s\t%s\t%s\t%s\t%.6f\t%.6f\t%s\t%s\t%s\t%s\t%s\t%.6f\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\n", h.gen.c_str(),
                      h.s.c_str(), h.r.c_str(), h.o.c_str(), h.p0, h.post, ecs::test_outcome_name(R.o[0]), ecs::test_outcome_name(R.o[1]),
                      ecs::test_outcome_name(R.o[2]), ecs::test_outcome_name(R.o[3]), ecs::test_outcome_name(R.o[4]), R.post2,
                      R.supported ? 1 : 0, keep_among[i] ? 1 : 0, keep_all[i] ? 1 : 0, R.refuted ? 1 : 0, R.by.c_str(), R.graded, gch,
                      h.promoted ? 1 : 0);
        pf << b;
        for (const string& g : {h.gen, string("all")}) {
            auto& T = tally[g];
            T[11] += 1;
            if (R.supported) T[0] += 1;
            if (R.graded >= 0) {
                if (R.supported) T[1] += 1, T[2] += R.graded;
                if (keep_among[i]) T[3] += 1, T[4] += R.graded;
                if (keep_all[i]) T[5] += 1, T[6] += R.graded;
                if (R.refuted) T[7] += 1, T[8] += 1 - R.graded;
                if (h.promoted) T[9] += 1, T[10] += R.graded;
            }
        }
    }
    for (const auto& [g, T] : tally) {
        std::snprintf(b, sizeof b,
                      "  %-7s hyps %5.0f | supported %4.0f, graded %3.0f, precision %.3f [%.3f, %.3f] | BH among supported: graded %3.0f "
                      "precision %.3f [%.3f, %.3f] | whole-batch e-BH: graded %3.0f precision %.3f [%.3f, %.3f] | refuted graded %3.0f, "
                      "false %.3f | generator's own promotions graded %2.0f precision %.3f",
                      g.c_str(), T[11], T[0], T[1], T[1] ? T[2] / T[1] : 0.0, ecs::wilson_lower(T[2], T[1]), ecs::wilson_upper(T[2], T[1]),
                      T[3], T[3] ? T[4] / T[3] : 0.0, ecs::wilson_lower(T[4], T[3]), ecs::wilson_upper(T[4], T[3]), T[5],
                      T[5] ? T[6] / T[5] : 0.0, ecs::wilson_lower(T[6], T[5]), ecs::wilson_upper(T[6], T[5]), T[7], T[7] ? T[8] / T[7] : 0.0,
                      T[9], T[9] ? T[10] / T[9] : 0.0);
        std::cout << b << "\n";
    }
    // P7: the hypothesis that started this.
    {
        const src::Value v = src::make_value("Mumbai");
        std::cout << "  India capital Mumbai:";
        for (int k = 0; k < kNSrc; ++k) std::cout << " " << kSrcName[k] << " " << ecs::test_outcome_name(S.answer(k, "India", "capital", v));
        std::cout << "\n";
    }
    if (calibrate) s8d(S, hyps, accepted, conf, choose, fresh_g2, fresh_links, out);  // S8d (docs/reports/S8D_PROMOTION.md)
    std::snprintf(b, sizeof b, "S8C seconds %.1f", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s8c

// ============================================================ S10 (docs/reports/S10_CURIOSITY_SCHEDULER.md)
// --schedule OUT --dev DIR S15_FILE [--gate DIR S15_FILE] [--gaps GAP_MAP] [--draws N] [--draws-dev N] [--seeds K]:
// DISCOVERY 9's curiosity scheduler (curiosity.hpp) against the uniform scheduler, replayed on recorded P2 pools. A
// pool DIR holds a fresh G2 set and a fresh G6/G7 set (s8a.fresh_g2.*, s8b.fresh_links.*: hypotheses and their
// pre-registration logs) and their source stage (src.s8d_fresh.tsv, src.s8d_fresh_ran.tsv from --sources --fresh).
// Graded by S15_FILE's rows for subjects with bit 17 = 1; the gate pool is graded only with --gate (read once).
// Scheduling only: it answers nothing.
namespace s10 {
namespace cur = cypha::eval::curiosity;

struct Meta {
    string gen, group, s, r, o;
};

struct Pool {
    vector<cur::PoolItem> items;
    vector<Meta> meta;
    double total_cost = 0, impact_max = 1;
    std::size_t slots = 0, min_cost_guard = 0, demand_items = 0;
};

struct ArmSet {
    vector<cur::Arm> arms;
    std::map<string, int> id;
    int get(const string& gen, const string& group, ecs::Tier t) {
        const string name = gen + " " + group + " " + ecs::tier_name(t);
        const auto it = id.find(name);
        if (it != id.end()) return it->second;
        cur::Arm a;
        a.name = name, a.generator = gen, a.tier = t;
        arms.push_back(a);
        return id[name] = static_cast<int>(arms.size() - 1);
    }
};

// Results recorded per hypothesis in a pre-registration log (the number after the id's last '-' is its row).
vector<int> prereg_results(const string& path) {
    vector<int> n;
    std::ifstream f(path);
    string l;
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() < 3 || c[1] != "result") continue;
        const auto pos = c[2].rfind('-');
        if (pos == string::npos) continue;
        const std::size_t i = std::strtoul(c[2].c_str() + pos + 1, nullptr, 10);
        if (n.size() <= i) n.resize(i + 1, 0);
        ++n[i];
    }
    return n;
}

struct TsvTable {
    vector<string> head;
    vector<vector<string>> rows;
    std::size_t col(const char* name) const {
        const auto it = std::find(head.begin(), head.end(), string(name));
        if (it == head.end()) throw std::runtime_error(string("missing column ") + name);
        return static_cast<std::size_t>(it - head.begin());
    }
};
TsvTable read_tsv(const string& path) {
    TsvTable t;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    string l;
    std::getline(f, l);
    t.head = split(l, '\t');
    while (std::getline(f, l))
        if (!l.empty()) t.rows.push_back(split(l, '\t'));
    return t;
}

// A recorded pool. `grade` false: every item ungraded (the gate pool before its read).
Pool load_pool(const string& dir, const string& s15_file, bool grade, const std::map<std::pair<string, string>, double>& demand, ArmSet& A) {
    Pool P;
    const TsvTable st = read_tsv(dir + "/src.s8d_fresh.tsv"), ran = read_tsv(dir + "/src.s8d_fresh_ran.tsv");
    const TsvTable g2 = read_tsv(dir + "/s8a.fresh_g2.hyps.tsv"), lk = read_tsv(dir + "/s8b.fresh_links.hyps.tsv");
    const vector<int> g2n = prereg_results(dir + "/s8a.fresh_g2.prereg.tsv"), lkn = prereg_results(dir + "/s8b.fresh_links.prereg.tsv");
    if (st.rows.size() != ran.rows.size() || st.rows.size() != g2.rows.size() + lk.rows.size())
        throw std::runtime_error("pool " + dir + ": the stage, cost and hypothesis files do not line up");
    s8c::Grader G;
    if (grade)
        for (const s8c::Row& x : s8c::load_s15(s15_file))
            if (!s8c::choose_half(x.s)) G.add(x);  // bit 17 = 1
    const std::size_t cg = st.col("generator"), cs = st.col("subject"), cr = st.col("relation"), co = st.col("object"),
                      cps = st.col("posterior_src"), csup = st.col("supported"), crun = ran.col("sources_run");
    std::set<std::pair<string, string>> slots;
    std::size_t i2 = 0, il = 0;
    for (std::size_t i = 0; i < st.rows.size(); ++i) {
        const auto& row = st.rows[i];
        const bool is_g2 = row[cg] == "G2-fresh";
        const TsvTable& H = is_g2 ? g2 : lk;
        const std::size_t k = is_g2 ? i2++ : il++;
        if (k >= H.rows.size()) throw std::runtime_error("pool " + dir + ": more stage rows than hypotheses");
        const auto& h = H.rows[k];
        if (h[H.col("subject")] != row[cs] || h[H.col("relation")] != row[cr] || h[H.col("object")] != row[co] ||
            ran.rows[i][ran.col("subject")] != row[cs] || ran.rows[i][ran.col("object")] != row[co])
            throw std::runtime_error("pool " + dir + ": row " + std::to_string(i) + " does not match");
        Meta m;
        m.gen = is_g2 ? "G2" : "G6/G7", m.group = s8c::group_of(row[cr]), m.s = row[cs], m.r = row[cr], m.o = row[co];
        cur::PoolItem it;
        it.prior = std::atof(h[is_g2 ? H.col("P0") : H.col("prior")].c_str());
        it.posterior = std::atof(row[cps].c_str());
        it.supported = row[csup] == "1";
        const vector<int>& gn = is_g2 ? g2n : lkn;
        const double cost = (k < gn.size() ? gn[k] : 0) + std::atof(ran.rows[i][crun].c_str());
        if (cost < 1) ++P.min_cost_guard;
        it.cost = std::max(1.0, cost);
        const auto dit = demand.find({m.s, m.r});
        it.demand = dit == demand.end() ? 0.0 : dit->second;
        P.demand_items += it.demand > 0;
        it.impact = ecs::hypothesis_impact(1.0, 0.0, it.demand);
        it.reward = cur::pull_reward(it.supported, it.impact, it.prior, it.posterior);
        it.bits = cur::bits_removed(it.prior, it.posterior);
        it.arm = A.get(m.gen, m.group, ecs::tier_of(it.prior));
        it.graded = grade ? G.grade(m.s, m.r, cypha::eval::sources::make_value(m.o)) : -1;
        P.total_cost += it.cost, P.impact_max = std::max(P.impact_max, it.impact);
        slots.insert({m.s, m.r});
        P.items.push_back(it), P.meta.push_back(m);
    }
    P.slots = slots.size();
    return P;
}

// S11e item 5: S11c-format pools from dir added to P (a.fresh_g3, g.fresh_g4, g.fresh_g5 graded on the S15 file's
// bit-17 = 1 half, as load_pool; a.fresh_g13 graded by Wikidata), each with its source stage, its cost file and its
// pre-registration log. A generator whose files are missing is skipped and named.
int wd_grade10(const s7::WdObjs& wd, const string& s, const string& r, const string& o) {
    const auto it = wd.find(s + '\x1f' + r);
    const int ri = rel_index(r);
    if (it == wd.end() || ri < 0 || it->second.empty()) return -1;
    return s7::wd_match(ri, o, it->second) ? 1 : 0;
}
void add_extra(Pool& P, const string& dir, const string& s15_file, const s7::WdObjs& wd, const std::map<std::pair<string, string>, double>& demand, ArmSet& A) {
    s8c::Grader G;
    for (const s8c::Row& x : s8c::load_s15(s15_file))
        if (!s8c::choose_half(x.s)) G.add(x);  // bit 17 = 1
    const std::pair<const char*, const char*> gens[4] = {{"a", "g3"}, {"g", "g4"}, {"g", "g5"}, {"a", "g13"}};
    for (const auto& [pre, g] : gens) {
        const string hp = dir + "/" + pre + ".fresh_" + g + ".hyps.tsv", sp = dir + "/src_" + g + ".s8d_fresh.tsv", rp = dir + "/src_" + g + ".s8d_fresh_ran.tsv",
                     lp = dir + "/" + pre + ".fresh_" + g + ".prereg.tsv";
        if (!std::ifstream(hp) || !std::ifstream(sp) || !std::ifstream(rp)) {
            std::cout << "S10 EXTRA " << dir << ": " << g << " skipped (files missing)\n";
            continue;
        }
        const TsvTable H = read_tsv(hp), st = read_tsv(sp), ran = read_tsv(rp);
        const vector<int> n = prereg_results(lp);
        if (st.rows.size() != ran.rows.size() || st.rows.size() != H.rows.size())
            throw std::runtime_error("extra pool " + dir + " " + g + ": the stage, cost and hypothesis files do not line up");
        const std::size_t cs = st.col("subject"), cr = st.col("relation"), co = st.col("object"), cps = st.col("posterior_src"), csup = st.col("supported"),
                          crun = ran.col("sources_run"), hgen = H.col("generator"), hp0 = H.col("P0");
        std::size_t added = 0;
        for (std::size_t i = 0; i < st.rows.size(); ++i) {
            const auto& row = st.rows[i];
            const auto& h = H.rows[i];
            if (h[H.col("subject")] != row[cs] || h[H.col("relation")] != row[cr] || h[H.col("object")] != row[co])
                throw std::runtime_error("extra pool " + dir + " " + g + ": row " + std::to_string(i) + " does not match");
            Meta m;
            m.gen = h[hgen], m.group = s8c::group_of(row[cr]), m.s = row[cs], m.r = row[cr], m.o = row[co];
            cur::PoolItem it;
            it.prior = std::atof(h[hp0].c_str());
            it.posterior = std::atof(row[cps].c_str());
            it.supported = row[csup] == "1";
            const double cost = (i < n.size() ? n[i] : 0) + std::atof(ran.rows[i][crun].c_str());
            if (cost < 1) ++P.min_cost_guard;
            it.cost = std::max(1.0, cost);
            const auto dit = demand.find({m.s, m.r});
            it.demand = dit == demand.end() ? 0.0 : dit->second;
            P.demand_items += it.demand > 0;
            it.impact = ecs::hypothesis_impact(1.0, 0.0, it.demand);
            it.reward = cur::pull_reward(it.supported, it.impact, it.prior, it.posterior);
            it.bits = cur::bits_removed(it.prior, it.posterior);
            it.arm = A.get(m.gen, m.group, ecs::tier_of(it.prior));
            it.graded = m.gen == "G13" ? wd_grade10(wd, m.s, m.r, m.o) : G.grade(m.s, m.r, cypha::eval::sources::make_value(m.o));
            P.total_cost += it.cost, P.impact_max = std::max(P.impact_max, it.impact);
            P.items.push_back(it), P.meta.push_back(m);
            ++added;
        }
        std::cout << "S10 EXTRA " << dir << ": " << g << " " << added << " hypotheses\n";
    }
    std::set<std::pair<string, string>> slots;
    for (const Meta& m : P.meta) slots.insert({m.s, m.r});
    P.slots = slots.size();
}

struct Config {
    string name;
    cur::WithinArm order = cur::WithinArm::Random;
    double lp = 0;
};

cur::SchedulerConfig uniform_cfg() {
    cur::SchedulerConfig c;
    c.policy = cur::Policy::Uniform, c.floors = false, c.retire = false;
    return c;
}
cur::SchedulerConfig s10_cfg(const Config& k, bool retire = true) {
    cur::SchedulerConfig c;
    c.policy = cur::Policy::Thompson, c.lp_weight = k.lp, c.retire = retire;
    return c;
}

// One draw's pooled reads (over its seeds) for one policy.
struct Read {
    double value = 0, cost = 0, graded = 0, graded_false = 0, brier = 0, promoted = 0, pulls = 0;
    void add(const cur::ReplayResult& r) {
        value += r.value, cost += r.cost, graded += r.graded, graded_false += r.graded_false, brier += r.brier, promoted += r.promoted, pulls += r.pulls;
    }
    double V() const { return cost > 0 ? 1000.0 * value / cost : 0.0; }
    double fdr() const { return graded > 0 ? graded_false / graded : -1.0; }
    double brier_mean() const { return graded > 0 ? brier / graded : -1.0; }
};

struct DrawSet {
    vector<Read> u, s;
    vector<double> arm_pulls_u, arm_pulls_s, arm_prom_u, arm_prom_s;
    double retired_runs = 0, s_seconds = 0, s_pulls = 0;
    std::map<string, double> retired_arms;
};

// draws x seeds replays of U and one S10 configuration on stratified bootstraps of the pool, at budget f x total cost.
// draws = 0: the pool itself, no bootstrap (one "draw").
DrawSet run_draws(const Pool& P, const ArmSet& A, const Config& k, double f, std::size_t draws, std::size_t seeds, std::uint64_t base,
                  bool retire = true) {
    DrawSet D;
    const std::size_t na = A.arms.size();
    D.arm_pulls_u.assign(na, 0), D.arm_pulls_s.assign(na, 0), D.arm_prom_u.assign(na, 0), D.arm_prom_s.assign(na, 0);
    const cur::SchedulerConfig cu = uniform_cfg(), cs = s10_cfg(k, retire);
    const std::size_t nd = std::max<std::size_t>(1, draws);
    for (std::size_t d = 0; d < nd; ++d) {
        const vector<cur::PoolItem> pool = draws == 0 ? P.items : cur::stratified_bootstrap(P.items, na, cur::mix_seed(base, d, 1));
        double tot = 0;
        for (const auto& it : pool) tot += it.cost;
        const double budget = f * tot;
        Read ru, rs;
        for (std::size_t q = 0; q < seeds; ++q) {
            const std::uint64_t os = cur::mix_seed(base, d * 1000 + q, 2);
            const auto ou = cur::arm_orders(pool, na, cur::WithinArm::Random, os), osd = cur::arm_orders(pool, na, k.order, os);
            const cur::ReplayResult a = cur::replay(pool, A.arms, ou, cu, budget, cur::mix_seed(base, d * 1000 + q, 3));
            const double t0 = now_s();
            const cur::ReplayResult b = cur::replay(pool, A.arms, osd, cs, budget, cur::mix_seed(base, d * 1000 + q, 4));
            D.s_seconds += now_s() - t0, D.s_pulls += b.pulls;
            ru.add(a), rs.add(b);
            bool any = false;
            for (std::size_t x = 0; x < na; ++x) {
                D.arm_pulls_u[x] += a.arm_pulls[x], D.arm_pulls_s[x] += b.arm_pulls[x];
                D.arm_prom_u[x] += a.arm_promoted[x], D.arm_prom_s[x] += b.arm_promoted[x];
                if (b.retired[x]) any = true, D.retired_arms[A.arms[x].name] += 1;
            }
            D.retired_runs += any;
        }
        D.u.push_back(ru), D.s.push_back(rs);
    }
    return D;
}

double mean_V(const vector<Read>& v) {
    double s = 0;
    for (const Read& r : v) s += r.V();
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}
// FDR pooled over every run of the set.
double pooled_fdr(const vector<Read>& v) {
    double g = 0, f = 0;
    for (const Read& r : v) g += r.graded, f += r.graded_false;
    return g > 0 ? f / g : -1.0;
}

void print_pool(const string& tag, const Pool& P, const ArmSet& A) {
    char b[1024];
    std::map<int, std::array<double, 6>> by;  // arm -> n, cost, supported, graded supported, true supported, graded
    double g2 = 0;
    for (std::size_t i = 0; i < P.items.size(); ++i) {
        const auto& it = P.items[i];
        auto& x = by[it.arm];
        x[0] += 1, x[1] += it.cost, x[2] += it.supported, x[5] += it.graded >= 0;
        if (it.supported && it.graded >= 0) x[3] += 1, x[4] += it.graded == 1;
        g2 += P.meta[i].gen == "G2";
    }
    std::snprintf(b, sizeof b, "S10 POOL %s: hypotheses %zu (G2 %.0f, G6/G7 %.0f; G6/G7 share %.3f) | distinct slots %zu | total cost %.0f (mean %.2f tests) | "
                  "cost guard used %zu | items with demand %zu | largest impact %.2f", tag.c_str(), P.items.size(), g2, static_cast<double>(P.items.size()) - g2,
                  P.items.empty() ? 0.0 : 1.0 - g2 / static_cast<double>(P.items.size()), P.slots, P.total_cost,
                  P.items.empty() ? 0.0 : P.total_cost / static_cast<double>(P.items.size()), P.min_cost_guard, P.demand_items, P.impact_max);
    std::cout << b << "\n";
    for (const auto& [a, x] : by) {
        std::snprintf(b, sizeof b, "S10 POOL %s arm %-30s hyps %4.0f cost %5.0f supported %3.0f (graded %3.0f, true %3.0f) graded %4.0f", tag.c_str(),
                      A.arms[static_cast<std::size_t>(a)].name.c_str(), x[0], x[1], x[2], x[3], x[4], x[5]);
        std::cout << b << "\n";
    }
}

void print_compare(const string& tag, const DrawSet& D) {
    char b[1024];
    const double vu = mean_V(D.u), vs = mean_V(D.s);
    double win = 0;
    for (std::size_t d = 0; d < D.u.size(); ++d) win += D.s[d].V() > 1.10 * D.u[d].V();
    double pu = 0, ps = 0, gu = 0, gs = 0;
    for (std::size_t d = 0; d < D.u.size(); ++d) pu += D.u[d].promoted, ps += D.s[d].promoted, gu += D.u[d].graded, gs += D.s[d].graded;
    std::snprintf(b, sizeof b, "S10 %s: V (x1000 per cost) U %.3f S10 %.3f | gain %+.3f | V_S10 > 1.1 V_U in %.3f of %zu draws | promotions per draw U %.2f S10 %.2f | "
                  "graded U %.2f S10 %.2f | FDR U %.3f S10 %.3f", tag.c_str(), vu, vs, vu != 0 ? vs / vu - 1.0 : 0.0,
                  D.u.empty() ? 0.0 : win / static_cast<double>(D.u.size()), D.u.size(), pu / static_cast<double>(D.u.size()),
                  ps / static_cast<double>(D.u.size()), gu / static_cast<double>(D.u.size()), gs / static_cast<double>(D.u.size()), pooled_fdr(D.u),
                  pooled_fdr(D.s));
    std::cout << b << "\n";
}

void print_arms(const string& tag, const DrawSet& D, const ArmSet& A, double runs) {
    char b[512];
    for (std::size_t x = 0; x < A.arms.size(); ++x) {
        if (D.arm_pulls_u[x] + D.arm_pulls_s[x] == 0) continue;
        std::snprintf(b, sizeof b, "S10 %s arm %-30s pulls per run U %7.2f S10 %7.2f | promotions per run U %5.2f S10 %5.2f", tag.c_str(),
                      A.arms[x].name.c_str(), D.arm_pulls_u[x] / runs, D.arm_pulls_s[x] / runs, D.arm_prom_u[x] / runs, D.arm_prom_s[x] / runs);
        std::cout << b << "\n";
    }
    for (const auto& [n, c] : D.retired_arms) {
        std::snprintf(b, sizeof b, "S10 %s retired %-30s in %.0f of %.0f runs", tag.c_str(), n.c_str(), c, runs);
        std::cout << b << "\n";
    }
}

int run(int argc, char** argv) {
    if (argc < 6 || string(argv[3]) != "--dev") {
        std::fprintf(stderr, "usage: pattern_learn --schedule OUT_PREFIX --dev DIR S15_FILE [--gate DIR S15_FILE] [--gaps GAP_MAP] [--draws N] [--draws-dev N] "
                             "[--seeds K] [--extra-dev DIR [--extra-only] --wd WD_VALUES]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[2], dev_dir = argv[4], dev_s15 = argv[5];
    string gate_dir, gate_s15, gaps, wd_file;
    vector<string> extra_dev;
    bool extra_only = false;
    std::size_t draws = 4000, draws_dev = 1000, seeds = 5;
    for (int i = 6; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--gate" && i + 2 < argc) gate_dir = argv[i + 1], gate_s15 = argv[i + 2], i += 2;
        else if (a == "--extra-dev" && i + 1 < argc) extra_dev.push_back(argv[++i]);  // S11e item 5
        else if (a == "--extra-only") extra_only = true;
        else if (a == "--wd" && i + 1 < argc) wd_file = argv[++i];
        else if (a == "--gaps" && i + 1 < argc) gaps = argv[++i];
        else if (a == "--draws" && i + 1 < argc) draws = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--draws-dev" && i + 1 < argc) draws_dev = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--seeds" && i + 1 < argc) seeds = std::strtoul(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    char b[1024];
    std::map<std::pair<string, string>, double> demand;
    if (!gaps.empty()) {
        const TsvTable g = read_tsv(gaps);
        const std::size_t ce = g.col("entity"), cr = g.col("relation"), cd = g.col("demand");
        for (const auto& row : g.rows) {
            const double d = std::atof(row[cd].c_str());
            if (d > 0) demand[{row[ce], row[cr]}] += d;
        }
    }
    ArmSet A;
    Pool dev = extra_only ? Pool{} : load_pool(dev_dir, dev_s15, true, demand, A);
    if (!extra_dev.empty()) {  // S11e item 5 (report only)
        const s7::WdObjs wd = wd_file.empty() ? s7::WdObjs{} : s7::load_wd_objs(wd_file);
        for (const auto& d : extra_dev) add_extra(dev, d, dev_s15, wd, demand, A);
        std::map<string, std::array<double, 4>> by;  // generator -> hypotheses, cost, supported, graded true among supported
        for (std::size_t i = 0; i < dev.items.size(); ++i) {
            auto& x = by[dev.meta[i].gen];
            x[0] += 1, x[1] += dev.items[i].cost, x[2] += dev.items[i].supported, x[3] += dev.items[i].supported && dev.items[i].graded == 1;
        }
        string l = "S10 POOL DEV by generator (S11e; the G6/G7 count in the line below includes every non-G2 generator):";
        for (const auto& [g, x] : by) {
            char c[160];
            std::snprintf(c, sizeof c, " %s %.0f (cost %.0f, supported %.0f, true %.0f)", g.c_str(), x[0], x[1], x[2], x[3]);
            l += c;
        }
        std::cout << l << "\n";
    }
    print_pool("DEV", dev, A);
    // ---- the four configurations on dev at f = 0.25; the registered choice
    const vector<Config> configs = {{"random order, lambda 0", cur::WithinArm::Random, 0.0},
                                    {"random order, lambda 1", cur::WithinArm::Random, 1.0},
                                    {"P0 first, lambda 0", cur::WithinArm::PriorFirst, 0.0},
                                    {"P0 first, lambda 1", cur::WithinArm::PriorFirst, 1.0}};
    vector<double> vdev;
    double vu_dev = 0;
    for (const Config& k : configs) {
        const DrawSet D = run_draws(dev, A, k, 0.25, draws_dev, seeds, 1000001);
        print_compare("DEV f 0.25 " + k.name, D);
        vdev.push_back(mean_V(D.s)), vu_dev = mean_V(D.u);
    }
    // Registered: the simpler configuration stays unless another beats it by more than 2%, order first, then lambda.
    auto better = [](double v, double base) { return v > base * 1.02 + 1e-12 * (base == 0); };
    std::size_t pick = 0;
    const bool order_wins = better(std::max(vdev[2], vdev[3]), std::max(vdev[0], vdev[1]));
    const std::size_t o = order_wins ? 2 : 0;
    pick = better(vdev[o + 1], vdev[o]) ? o + 1 : o;
    const Config chosen = configs[pick];
    const double gain_dev = vu_dev != 0 ? vdev[pick] / vu_dev - 1.0 : 0.0;
    std::snprintf(b, sizeof b, "S10 DEV CHOICE: %s (V %.3f; U %.3f; dev gain %+.3f)", chosen.name.c_str(), vdev[pick], vu_dev, gain_dev);
    std::cout << b << "\n";
    {
        const DrawSet D = run_draws(dev, A, chosen, 0.25, draws_dev, seeds, 1000001);
        print_arms("DEV f 0.25", D, A, static_cast<double>(D.u.size() * seeds));
    }
    for (const double f : {0.1, 0.5, 1.0}) {
        const DrawSet D = run_draws(dev, A, chosen, f, draws_dev, seeds, 1000001);
        std::snprintf(b, sizeof b, "DEV f %.2f", f);
        print_compare(b, D);
        print_arms(b, D, A, static_cast<double>(D.u.size() * seeds));
    }
    {  // Q9: at f = 1 with retirement off both process everything and must tie exactly, draw by draw.
        const DrawSet D = run_draws(dev, A, chosen, 1.0, 200, seeds, 1000001, false);
        std::size_t tie = 0;
        for (std::size_t d = 0; d < D.u.size(); ++d)
            tie += std::fabs(D.u[d].value - D.s[d].value) <= 1e-9 * (1.0 + std::fabs(D.u[d].value)) && D.u[d].cost == D.s[d].cost &&
                   D.u[d].pulls == D.s[d].pulls;
        std::snprintf(b, sizeof b, "S10 DEV CHECK f 1.0, retirement off: exact ties in %zu of %zu draws", tie, D.u.size());
        std::cout << b << "\n";
    }
    {  // U0: no arms, a random order over all hypotheses (for reference)
        ArmSet A0;
        A0.arms.push_back({"all", "all", ecs::Tier::Conjecture});
        Pool P0 = dev;
        for (auto& it : P0.items) it.arm = 0;
        const DrawSet D = run_draws(P0, A0, configs[0], 0.25, draws_dev, seeds, 1000001);
        std::snprintf(b, sizeof b, "S10 DEV REFERENCE U0 (one arm, random order over all hypotheses) f 0.25: V %.3f", mean_V(D.u));
        std::cout << b << "\n";
    }
    if (gate_dir.empty()) {
        std::snprintf(b, sizeof b, "S10 seconds %.1f (no gate read)", now_s() - t0);
        std::cout << b << "\n";
        return 0;
    }
    // ================================================================ the gate: read once
    const Pool gate = load_pool(gate_dir, gate_s15, true, demand, A);
    print_pool("GATE", gate, A);
    const DrawSet D = run_draws(gate, A, chosen, 0.25, draws, seeds, 2000003);
    print_compare("GATE f 0.25 " + chosen.name, D);
    print_arms("GATE f 0.25", D, A, static_cast<double>(D.u.size() * seeds));
    // LCI (registered)
    const std::size_t n = D.u.size();
    double pg = 0, r_n = 0, r_ok = 0, dbr = 0, dbr_n = 0;
    vector<double> block(3, 0.0), block_n(3, 0.0);
    std::ofstream tf(out + ".gate_draws.tsv");
    tf << "draw\tV_U\tV_S10\tFDR_U\tFDR_S10\tBrier_U\tBrier_S10\tpromoted_U\tpromoted_S10\tgraded_U\tgraded_S10\n";
    for (std::size_t d = 0; d < n; ++d) {
        const Read &u = D.u[d], &s = D.s[d];
        const double vu = u.V(), vs = s.V();
        pg += vu > 0 ? vs > 1.10 * vu : vs - vu > 0.10 * std::fabs(vu);
        if (u.graded > 0 && s.graded > 0) {
            r_n += 1, r_ok += s.fdr() <= u.fdr() + 0.05 + 1e-12;
            dbr += s.brier_mean() - u.brier_mean(), dbr_n += 1;
        }
        const std::size_t bl = std::min<std::size_t>(2, d * 3 / std::max<std::size_t>(1, n));
        block[bl] += vs - vu, block_n[bl] += 1;
        std::snprintf(b, sizeof b, "%zu\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\n", d, vu, vs, u.fdr(), s.fdr(), u.brier_mean(), s.brier_mean(),
                      u.promoted, s.promoted, u.graded, s.graded);
        tf << b;
    }
    const double p_gain = n ? pg / static_cast<double>(n) : 0.0;
    const double r_share = r_n > 0 ? r_ok / r_n : 0.0;
    const double r = r_share >= 0.95 ? 1.0 : 0.0;
    const double vu = mean_V(D.u), vs = mean_V(D.s);
    const double gain_gate = vu != 0 ? vs / vu - 1.0 : 0.0;
    const double G = gain_dev > 0 ? gain_gate / gain_dev : 0.0;
    const double gamma = gain_dev > 0 ? std::min(1.0, std::max(0.0, G) / 0.5) : (gain_gate > 0 ? 1.0 : 0.0);
    const double brier_rise = dbr_n > 0 ? dbr / dbr_n : 0.0;
    const double c = brier_rise <= 0.01 ? 1.0 : 0.5;
    vector<double> bmeans;
    for (int k = 0; k < 3; ++k) bmeans.push_back(block_n[k] > 0 ? block[k] / block_n[k] : 0.0);
    const double sigma = ecs::stability_sigma(bmeans);
    const double us_per_pull = D.s_pulls > 0 ? 1e6 * D.s_seconds / D.s_pulls : 0.0;
    const double state_bytes = static_cast<double>(A.arms.size()) * (sizeof(cur::ArmState) + sizeof(cur::Arm) + 10 * sizeof(double) + 64);
    const double kappa = ecs::cost_kappa(state_bytes, 1e6, us_per_pull, 1000.0);
    const double product = p_gain * r * gamma * c * sigma * kappa;
    const double lci = gate.slots < ecs::kMinEffectiveN ? std::min(product, ecs::kLowNEffCap) : product;
    std::snprintf(b, sizeof b,
                  "S10 GATE LCI: P_gain %.4f | n_eff %zu | r %.0f (FDR_S10 <= FDR_U + 0.05 in %.4f of %.0f draws; pooled FDR U %.3f S10 %.3f) | gamma %.3f "
                  "(gain gate %+.3f, dev %+.3f) | c %.1f (Brier rise %+.4f) | sigma %.1f (blocks %+.3f %+.3f %+.3f) | kappa %.0f (%.2f us per pull, state %.0f B) | "
                  "LCI %.4f -> %s",
                  p_gain, gate.slots, r, r_share, r_n, pooled_fdr(D.u), pooled_fdr(D.s), gamma, gain_gate, gain_dev, c, brier_rise, sigma, bmeans[0], bmeans[1],
                  bmeans[2], kappa, us_per_pull, state_bytes, lci, lci >= 0.95 ? "MET" : "NOT MET");
    std::cout << b << "\n";
    {  // Q8: S10's promotions on the gate by relation group (the pool itself, every seed)
        std::map<string, double> by;
        for (std::size_t x = 0; x < A.arms.size(); ++x) {
            const string& nm = A.arms[x].name;
            const auto p1 = nm.find(' '), p2 = nm.find(' ', p1 + 1);
            by[nm.substr(p1 + 1, p2 - p1 - 1)] += D.arm_prom_s[x];
        }
        double tot = 0;
        for (const auto& [g, v] : by) tot += v;
        string line = "S10 GATE S10's promotions by group:";
        for (const auto& [g, v] : by) {
            std::snprintf(b, sizeof b, " %s %.3f", g.c_str(), tot > 0 ? v / tot : 0.0);
            line += b;
        }
        std::cout << line << "\n";
    }
    for (const double f : {0.1, 0.5, 1.0}) {
        const DrawSet E = run_draws(gate, A, chosen, f, 1000, seeds, 2000003);
        std::snprintf(b, sizeof b, "GATE f %.2f", f);
        print_compare(b, E);
        print_arms(b, E, A, static_cast<double>(E.u.size() * seeds));
    }
    std::snprintf(b, sizeof b, "S10 seconds %.1f", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s10

// ============================================================ S11a: G9 concept formation and G11 open relations
// pattern_learn --gen-concepts ENWIK8 SEEDS.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--attrs A1|A2]
//                [--part all|rediscover|promote|open] [--read-test]
// (DISCOVERY 5 G9 and G11, P5, P10; docs/reports/S11A_CONCEPTS_AND_OPEN_RELATIONS.md). G9: concepts of store K
// (gen_concepts.hpp), the rediscovery of held-out classes against a random-class null, and its class rules' predictions
// on hidden S15 slots through S8a's loop. G11: unparsed (subject, link) pairs clustered by phrase, graded against S15's
// properties. Dev by the class hash (bit 19) or S15's dev file; test only with --read-test. Cypha's own hypotheses only:
// nothing here answers a question (PLAN rule 10).
namespace s11a {

namespace cc = cypha::eval::concepts;
using ecs::TestOutcome;
using s8a::Facts;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;

// Store K's facts as G9's context reads them: non-date relations, values by norm_ent; A2 adds one hop from a place
// (r>country, r>located_in). rels: every relation each subject holds; surf: each attribute's surface strings.
using SurfMap = std::map<std::pair<string, string>, std::map<string, int>>;
vector<cc::Fact3> context_facts(const Facts& by, bool a2, std::map<string, std::set<string>>& rels, SurfMap* surf, const Facts* extra = nullptr) {
    vector<cc::Fact3> out;
    static const std::set<string> hop = {"birth_place", "death_place", "location", "located_in"};
    for (const auto& [s, fs] : by)
        for (const auto& [r, o] : fs) {
            rels[s].insert(r);
            const int ri = rel_index(r);
            if (ri < 0 || is_datey(ri)) continue;
            const string v = norm_ent(o);
            if (v.empty()) continue;
            out.push_back({s, r, v});
            if (surf) ++(*surf)[{r, v}][o];
            if (!a2 || !hop.count(r)) continue;
            const auto zt = by.find(o);
            if (zt == by.end()) continue;
            for (const auto& [r2, c] : zt->second)
                if ((r2 == "country" || r2 == "located_in") && !norm_ent(c).empty()) out.push_back({s, r + ">" + r2, norm_ent(c)});
        }
    if (extra)  // S11d (A3): S15's added facts, each an attribute (r, norm_ent(value))
        for (const auto& [s, fs] : *extra)
            for (const auto& [r, o] : fs) {
                rels[s].insert(r);
                const string v = norm_ent(o);
                if (v.empty()) continue;
                out.push_back({s, r, v});
                if (surf) ++(*surf)[{r, v}][o];
            }
    return out;
}

struct G9 {
    cc::FormalContext C;
    vector<cc::Concept> cs;
    vector<std::size_t> cand;
    vector<cc::ClassRule> rules;
    vector<char> kept_class;  // per concept
    bool truncated = false;
};
G9 run_g9(const Facts& by, bool a2, SurfMap* surf = nullptr, const Facts* extra = nullptr) {
    std::map<string, std::set<string>> rels;
    const auto facts = context_facts(by, a2, rels, surf, extra);
    G9 g;
    g.C = cc::build_context(facts, 10.0, &rels);
    g.cs = cc::closed_concepts(g.C, 10.0, 500000, &g.truncated);
    g.cand = cc::candidate_classes(g.C, g.cs, 10.0, 0.9);
    g.rules = cc::class_rules(g.C, g.cs, g.cand);
    g.kept_class.assign(g.cs.size(), 0);
    for (const auto& R : g.rules)
        if (R.kept) g.kept_class[R.cls] = 1;
    return g;
}
string intent_text(const G9& g, std::size_t k) {
    string t;
    for (int a : g.cs[k].intent) t += (t.empty() ? "" : " & ") + g.C.attr_text(a);
    return t;
}

// The setup S8a's run makes: the dump, its pages and type lexicons, Wikidata's seeds, T4's patterns, and store K.
struct Env {
    string data;
    vector<PageRef> pages;
    std::unordered_map<string, int> page_of;
    Facts kby;
    s7::WdObjs wd;
};
bool load_env(Env& E, const string& enwik8, const string& seeds, const string& wdv, const string& patterns) {
    if (!read_file(enwik8, E.data)) { std::fprintf(stderr, "cannot read %s\n", enwik8.c_str()); return false; }
    g_dump = &E.data;
    long total = 0, redirects = 0;
    E.pages = list_pages(E.data, total, redirects);
    for (std::size_t i = 0; i < E.pages.size(); ++i) E.page_of[E.pages[i].title] = static_cast<int>(i);
    for (const PageRef& pg : E.pages) {
        PageFlags fl;
        scan_page(E.data, pg, fl);
        const string nk = norm_ent(pg.title);
        g_titles.insert(nk);
        if (fl.place) g_place.insert(nk);
        if (fl.person) g_person.insert(nk);
        if (fl.country) g_country.insert(nk);
    }
    {
        long ns = 0, ng = 0;
        if (!load_wikidata(wdv, false, E.page_of, ns, ng)) { std::fprintf(stderr, "cannot read %s\n", wdv.c_str()); return false; }
    }
    for (const Seed& sd : g_seeds) {
        const RT ty = kRel[sd.rel].type;
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    {
        std::ifstream f(patterns);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 11 || c[10] != "1" || rel_index(c[0]) < 0) continue;
            Pat p;
            p.rel = rel_index(c[0]);
            p.key = c[1];
            p.family = family_of(p.key);
            p.kept = true;
            g_patmap[p.key].push_back(static_cast<int>(g_pats.size()));
            g_pats.push_back(p);
        }
    }
    {
        std::ifstream f(seeds);
        string l;
        std::getline(f, l);
        std::set<string> seen;
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
            if (seen.insert(c[0] + '\x1f' + c[1] + '\x1f' + c[2]).second) E.kby[c[0]].push_back({c[1], c[2]});
        }
        for (auto& [s, fs] : E.kby) std::sort(fs.begin(), fs.end());
    }
    E.wd = s7::load_wd_objs(wdv);
    return true;
}

// ============================================================ G9, read 1: rediscovery of held-out classes
void rediscover(const Env& E, bool a2, bool read_test) {
    char b[1024];
    const string A = a2 ? "A2" : "A1";
    const double t0 = now_s();
    const G9 full = run_g9(E.kby, a2);
    std::size_t kept_rules = 0, kept_classes = 0;
    for (const auto& R : full.rules) kept_rules += R.kept;
    for (char c : full.kept_class) kept_classes += c;
    std::snprintf(b, sizeof b, "G9 %s FULL context: attributes %zu entities %zu | closed concepts %zu%s | candidate classes %zu | class rules %zu kept %zu | "
                  "kept classes %zu | %.1f s", A.c_str(), full.C.attr_r.size(), full.C.entities.size(), full.cs.size(), full.truncated ? " (STOPPED)" : "",
                  full.cand.size(), full.rules.size(), kept_rules, kept_classes, now_s() - t0);
    std::cout << b << "\n";
    // the kept rules with the largest support, as examples
    {
        vector<const cc::ClassRule*> kr;
        for (const auto& R : full.rules)
            if (R.kept) kr.push_back(&R);
        std::sort(kr.begin(), kr.end(), [](const cc::ClassRule* x, const cc::ClassRule* y) { return x->support > y->support || (x->support == y->support && x->p_value < y->p_value); });
        for (std::size_t i = 0; i < kr.size() && i < 15; ++i) {
            std::snprintf(b, sizeof b, "G9 %s KEPT RULE %-70s => %-34s support %3.0f PCA %.3f baseline %.3f p %.2g", A.c_str(), intent_text(full, kr[i]->cls).c_str(),
                          full.C.attr_text(kr[i]->head).c_str(), kr[i]->support, kr[i]->pca_conf(), kr[i]->baseline, kr[i]->p_value);
            std::cout << b << "\n";
        }
    }
    // the held-out classes: direct attributes (r, v) of the full A1 context with 15..500 subjects
    const G9 full1 = a2 ? run_g9(E.kby, false) : full;
    struct H {
        string r, v;
        vector<string> names;
        bool dev = true;
    };
    vector<H> hs;
    for (std::size_t a = 0; a < full1.C.attr_r.size(); ++a) {
        const double n = static_cast<double>(full1.C.ext[a].size());
        if (n < 15 || n > 500) continue;
        H h;
        h.r = full1.C.attr_r[a], h.v = full1.C.attr_v[a];
        for (int e : full1.C.ext[a]) h.names.push_back(full1.C.entities[static_cast<std::size_t>(e)]);
        h.dev = ((fnv(h.r + '\x1f' + h.v) >> 19) & 1) == 0;
        hs.push_back(std::move(h));
    }
    std::size_t ndev = 0;
    for (const H& h : hs) ndev += h.dev;
    std::snprintf(b, sizeof b, "G9 %s HELD-OUT classes %zu (dev %zu, test %zu)", A.c_str(), hs.size(), ndev, hs.size() - ndev);
    std::cout << b << "\n";
    struct Acc {
        double n = 0, red = 0, redun = 0, kept = 0, null = 0, both = 0;
    };
    std::map<string, Acc> acc[2];  // [dev, test] by relation, "" = all
    vector<std::pair<double, double>> jj[2];  // amendment 1 (secondary): (J_new, J_null) per held-out class
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const H& h = hs[i];
        if (!h.dev && !read_test) continue;
        Facts red;
        for (const auto& [s, fs] : E.kby)
            for (const auto& [r, o] : fs)
                if (!(r == h.r && norm_ent(o) == h.v)) red[s].push_back({r, o});
        const G9 g = run_g9(red, a2);
        vector<int> hp;
        for (const string& nm : h.names)
            if (const int e = g.C.entity(nm); e >= 0) hp.push_back(e);
        std::sort(hp.begin(), hp.end());
        const double hn = static_cast<double>(h.names.size());
        double jn = 0, jk = 0, jm = 0, jnull = 0;
        std::size_t best = 0, bestm = 0;
        for (std::size_t k : g.cand) {
            const double j = cc::jaccard(g.cs[k].extent, hp, hn);
            if (j > jn) jn = j, best = k;
            if (g.kept_class[k] && j > jk) jk = j;
        }
        for (std::size_t a = 0; a < g.C.attr_r.size(); ++a)
            if (const double j = cc::jaccard(g.C.ext[a], hp, hn); j > jm) jm = j, bestm = a;
        // the null: a random set of |H| context entities holding a value of r in store K
        {
            vector<int> pool;
            for (std::size_t e = 0; e < g.C.entities.size(); ++e) {
                const auto it = E.kby.find(g.C.entities[e]);
                bool has = false;
                if (it != E.kby.end())
                    for (const auto& f : it->second) has = has || f.first == h.r;
                if (has) pool.push_back(static_cast<int>(e));
            }
            if (pool.size() < h.names.size()) {
                pool.resize(g.C.entities.size());
                for (std::size_t e = 0; e < pool.size(); ++e) pool[e] = static_cast<int>(e);
            }
            std::mt19937_64 rng(20261101 + i);
            const std::size_t m = std::min(pool.size(), h.names.size());
            for (std::size_t k = 0; k < m; ++k) std::swap(pool[k], pool[k + rng() % (pool.size() - k)]);
            vector<int> rn(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(m));
            std::sort(rn.begin(), rn.end());
            for (std::size_t k : g.cand) jnull = std::max(jnull, cc::jaccard(g.cs[k].extent, rn, hn));
        }
        const int sp = h.dev ? 0 : 1;
        jj[sp].push_back({jn, jnull});
        for (const string& key : {string(), h.r}) {
            Acc& a = acc[sp][key];
            a.n += 1, a.red += jn >= 0.5, a.redun += jm >= 0.5, a.kept += jk >= 0.5, a.null += jnull >= 0.5, a.both += jn >= 0.5 && jm >= 0.5;
        }
        std::snprintf(b, sizeof b, "G9 %s %s CLASS %-11s %-28s n %3.0f in context %3zu | J_new %.3f%s J_named %.3f J_kept %.3f | null %.3f | best class: %s (%zu) | best named: %s",
                      A.c_str(), h.dev ? "DEV" : "TEST", h.r.c_str(), h.v.c_str(), hn, hp.size(), jn, jn >= 0.5 ? " REDISCOVERED" : "", jm, jk, jnull,
                      g.cand.empty() ? "-" : intent_text(g, best).c_str(), g.cand.empty() ? std::size_t{0} : g.cs[best].extent.size(),
                      g.C.attr_r.empty() ? "-" : g.C.attr_text(static_cast<int>(bestm)).c_str());
        std::cout << b << "\n";
    }
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
        for (const auto& [key, a] : acc[sp]) {
            std::snprintf(b, sizeof b,
                          "G9 %s %s REDISCOVERY %-12s classes %3.0f | rediscovered %3.0f rate %.3f [%.3f, %.3f] | null %3.0f rate %.3f [%.3f, %.3f] | redundant %3.0f (%.3f) | "
                          "kept classes %3.0f (%.3f) | rediscovered and redundant %3.0f",
                          A.c_str(), sp == 0 ? "DEV" : "TEST", key.empty() ? "all" : key.c_str(), a.n, a.red, a.n ? a.red / a.n : 0.0, ecs::wilson_lower(a.red, a.n),
                          ecs::wilson_upper(a.red, a.n), a.null, a.n ? a.null / a.n : 0.0, ecs::wilson_lower(a.null, a.n), ecs::wilson_upper(a.null, a.n), a.redun,
                          a.n ? a.redun / a.n : 0.0, a.kept, a.n ? a.kept / a.n : 0.0, a.both);
            std::cout << b << "\n";
            if (key.empty() && sp == 1) {
                const bool met = ecs::wilson_lower(a.red, a.n) > ecs::wilson_upper(a.null, a.n);
                std::snprintf(b, sizeof b, "G9 %s GATE (rediscovery of a class, test): Wilson lower %.3f vs null Wilson upper %.3f -> %s", A.c_str(),
                              ecs::wilson_lower(a.red, a.n), ecs::wilson_upper(a.null, a.n), met ? "MET" : "NOT MET");
                std::cout << b << "\n";
            }
        }
    // amendment 1 (secondary, registered after the dev read and before the test read): J_new against the null, paired
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
        const auto& v = jj[sp];
        double wins = 0, losses = 0, d = 0, p30 = 0, n30 = 0;
        for (const auto& [a, z] : v) wins += a > z, losses += a < z, d += a - z, p30 += a >= 0.3, n30 += z >= 0.3;
        const double n = static_cast<double>(v.size()), m = wins + losses;
        const double p_sign = m > 0 ? ecs::binomial_upper_tail(wins, m, 0.5) : 1.0;  // one-sided sign test
        vector<double> boot;
        std::mt19937_64 rng(20261102);
        for (int r = 0; r < 2000 && !v.empty(); ++r) {
            double t = 0;
            for (std::size_t k = 0; k < v.size(); ++k) {
                const auto& [a, z] = v[rng() % v.size()];
                t += a - z;
            }
            boot.push_back(t / n);
        }
        std::sort(boot.begin(), boot.end());
        std::snprintf(b, sizeof b, "G9 %s %s SECONDARY classes %.0f | J_new > null %.0f, < %.0f, sign test p %.2g | mean J_new - null %+.3f [%+.3f, %+.3f] | "
                      "J_new >= 0.3 %.0f (%.3f) vs null %.0f (%.3f)", A.c_str(), sp == 0 ? "DEV" : "TEST", n, wins, losses, p_sign, n ? d / n : 0.0,
                      boot.empty() ? 0.0 : boot[50], boot.empty() ? 0.0 : boot[1949], p30, n ? p30 / n : 0.0, n30, n ? n30 / n : 0.0);
        std::cout << b << "\n";
    }
}

// ============================================================ S11d: G9 on a richer store (A3), against fresh held-out classes
// The added facts: S15 rows with label 1, an entity value and a relation pattern_learn does not know (14 relations).
Facts load_added(const vector<string>& files) {
    Facts X;
    std::set<string> seen;
    for (const string& path : files) {
        std::ifstream f(path);
        string l;
        std::getline(f, l);
        while (std::getline(f, l)) {
            const auto c = split(l, '\t');
            if (c.size() < 17 || c[15] != "1" || c[8].empty() || rel_index(c[5]) >= 0) continue;
            if (seen.insert(c[1] + '\x1f' + c[5] + '\x1f' + c[7]).second) X[c[1]].push_back({c[5], c[7]});
        }
    }
    for (auto& [s, fs] : X) std::sort(fs.begin(), fs.end());
    return X;
}

// Fresh classes: the A3 attributes with 15..500 subjects that are not S11a's held-out classes (A1 of store K in that range),
// dev when bit 21 of FNV-1a(r + "\x1f" + v) is 0. Secondary: S11a's own classes (dev on every run, test on the test run).
void rediscover_a3(const Env& E, const Facts& X, bool read_test) {
    char b[1024];
    const double t0 = now_s();
    const G9 full = run_g9(E.kby, false, nullptr, &X);
    std::size_t kept_rules = 0, kept_classes = 0;
    for (const auto& R : full.rules) kept_rules += R.kept;
    for (char c : full.kept_class) kept_classes += c;
    std::snprintf(b, sizeof b, "G9 A3 FULL context: attributes %zu entities %zu | closed concepts %zu%s | candidate classes %zu | class rules %zu kept %zu | "
                  "kept classes %zu | %.1f s", full.C.attr_r.size(), full.C.entities.size(), full.cs.size(), full.truncated ? " (STOPPED)" : "",
                  full.cand.size(), full.rules.size(), kept_rules, kept_classes, now_s() - t0);
    std::cout << b << "\n";
    {
        vector<const cc::ClassRule*> kr;
        for (const auto& R : full.rules)
            if (R.kept) kr.push_back(&R);
        std::sort(kr.begin(), kr.end(), [](const cc::ClassRule* x, const cc::ClassRule* y) { return x->support > y->support || (x->support == y->support && x->p_value < y->p_value); });
        for (std::size_t i = 0; i < kr.size() && i < 20; ++i) {
            std::snprintf(b, sizeof b, "G9 A3 KEPT RULE %-70s => %-40s support %3.0f PCA %.3f baseline %.3f p %.2g", intent_text(full, kr[i]->cls).c_str(),
                          full.C.attr_text(kr[i]->head).c_str(), kr[i]->support, kr[i]->pca_conf(), kr[i]->baseline, kr[i]->p_value);
            std::cout << b << "\n";
        }
    }
    const G9 k1 = run_g9(E.kby, false);
    std::set<std::pair<string, string>> s11a_set;
    for (std::size_t a = 0; a < k1.C.attr_r.size(); ++a)
        if (const double n = static_cast<double>(k1.C.ext[a].size()); n >= 15 && n <= 500) s11a_set.insert({k1.C.attr_r[a], k1.C.attr_v[a]});
    struct H {
        string r, v;
        vector<string> names;
        int group = 0;  // 0 fresh dev, 1 fresh test, 2 S11a dev, 3 S11a test
    };
    vector<H> hs;
    auto add_from = [&](const G9& g, bool fresh) {
        for (std::size_t a = 0; a < g.C.attr_r.size(); ++a) {
            const double n = static_cast<double>(g.C.ext[a].size());
            if (n < 15 || n > 500) continue;
            const bool in11a = s11a_set.count({g.C.attr_r[a], g.C.attr_v[a]}) > 0;
            if (fresh == in11a) continue;
            H h;
            h.r = g.C.attr_r[a], h.v = g.C.attr_v[a];
            for (int e : g.C.ext[a]) h.names.push_back(g.C.entities[static_cast<std::size_t>(e)]);
            const std::uint64_t f = fnv(h.r + '\x1f' + h.v);
            h.group = fresh ? (cc::fresh_class_is_dev(f) ? 0 : 1) : (((f >> 19) & 1) == 0 ? 2 : 3);
            if (h.group == 1 && !read_test) continue;
            if (h.group == 3 && !read_test) continue;
            hs.push_back(std::move(h));
        }
    };
    add_from(full, true);
    add_from(k1, false);  // S11a's classes, their members in store K
    std::size_t cnt[4] = {0, 0, 0, 0};
    for (const H& h : hs) ++cnt[h.group];
    std::snprintf(b, sizeof b, "G9 A3 HELD-OUT classes: fresh dev %zu, fresh test %zu | S11a's dev %zu, S11a's test %zu%s", cnt[0], cnt[1], cnt[2], cnt[3],
                  read_test ? "" : " (test classes not read)");
    std::cout << b << "\n";
    static const char* gname[4] = {"FRESH DEV", "FRESH TEST", "S11A-DEV", "S11A-TEST"};
    cc::RediscoveryTally tal[4];
    std::map<string, cc::RediscoveryTally> by_rel[4];
    vector<std::pair<double, double>> jj[4];
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const H& h = hs[i];
        Facts redK, redX;
        for (const auto& [s, fs] : E.kby)
            for (const auto& [r, o] : fs)
                if (!(r == h.r && norm_ent(o) == h.v)) redK[s].push_back({r, o});
        for (const auto& [s, fs] : X)
            for (const auto& [r, o] : fs)
                if (!(r == h.r && norm_ent(o) == h.v)) redX[s].push_back({r, o});
        const G9 g = run_g9(redK, false, nullptr, &redX);
        vector<int> hp;
        for (const string& nm : h.names)
            if (const int e = g.C.entity(nm); e >= 0) hp.push_back(e);
        std::sort(hp.begin(), hp.end());
        const double hn = static_cast<double>(h.names.size());
        cc::ClassRead cr;
        std::size_t best = 0, bestm = 0;
        for (std::size_t k : g.cand) {
            const double j = cc::jaccard(g.cs[k].extent, hp, hn);
            if (j > cr.j_new) cr.j_new = j, best = k;
            if (g.kept_class[k] && j > cr.j_kept) cr.j_kept = j;
        }
        for (std::size_t a = 0; a < g.C.attr_r.size(); ++a)
            if (const double j = cc::jaccard(g.C.ext[a], hp, hn); j > cr.j_named) cr.j_named = j, bestm = a;
        {  // the null: |H| random context entities holding a value of r in store K or the added facts
            vector<int> pool;
            for (std::size_t e = 0; e < g.C.entities.size(); ++e) {
                bool has = false;
                for (const Facts* F : {&E.kby, &X})
                    if (const auto it = F->find(g.C.entities[e]); it != F->end())
                        for (const auto& f : it->second) has = has || f.first == h.r;
                if (has) pool.push_back(static_cast<int>(e));
            }
            if (pool.size() < h.names.size()) {
                pool.resize(g.C.entities.size());
                for (std::size_t e = 0; e < pool.size(); ++e) pool[e] = static_cast<int>(e);
            }
            std::mt19937_64 rng(20261103 + i);
            const std::size_t m = std::min(pool.size(), h.names.size());
            for (std::size_t k = 0; k < m; ++k) std::swap(pool[k], pool[k + rng() % (pool.size() - k)]);
            vector<int> rn(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(m));
            std::sort(rn.begin(), rn.end());
            for (std::size_t k : g.cand) cr.j_null = std::max(cr.j_null, cc::jaccard(g.cs[k].extent, rn, hn));
        }
        tal[h.group].add(cr);
        by_rel[h.group][h.r].add(cr);
        jj[h.group].push_back({cr.j_new, cr.j_null});
        std::snprintf(b, sizeof b, "G9 A3 %s CLASS %-18s %-40s n %3.0f in context %3zu | J_new %.3f%s J_named %.3f J_kept %.3f | null %.3f | best class: %s (%zu) | best named: %s",
                      gname[h.group], h.r.c_str(), h.v.c_str(), hn, hp.size(), cr.j_new, cr.j_new >= 0.5 ? (cr.j_named < 0.5 ? " REDISCOVERED (strict)" : " REDISCOVERED") : "",
                      cr.j_named, cr.j_kept, cr.j_null, g.cand.empty() ? "-" : intent_text(g, best).c_str(), g.cand.empty() ? std::size_t{0} : g.cs[best].extent.size(),
                      g.C.attr_r.empty() ? "-" : g.C.attr_text(static_cast<int>(bestm)).c_str());
        std::cout << b << "\n";
    }
    for (int gi = 0; gi < 4; ++gi) {
        if (tal[gi].n == 0) continue;
        auto line = [&](const string& key, const cc::RediscoveryTally& a) {
            std::snprintf(b, sizeof b,
                          "G9 A3 %s REDISCOVERY %-26s classes %3.0f | rediscovered %3.0f rate %.3f [%.3f, %.3f] | strict %3.0f rate %.3f [%.3f, %.3f] | null %3.0f rate %.3f [%.3f, %.3f] | "
                          "redundant %3.0f | kept classes %3.0f",
                          gname[gi], key.c_str(), a.n, a.red, a.red / a.n, ecs::wilson_lower(a.red, a.n), ecs::wilson_upper(a.red, a.n), a.strict, a.strict / a.n,
                          ecs::wilson_lower(a.strict, a.n), ecs::wilson_upper(a.strict, a.n), a.null, a.null / a.n, ecs::wilson_lower(a.null, a.n),
                          ecs::wilson_upper(a.null, a.n), a.redun, a.kept);
            std::cout << b << "\n";
        };
        line("all", tal[gi]);
        for (const auto& [r, a] : by_rel[gi]) line(r, a);
        if (gi == 1) {
            const cc::RediscoveryTally& a = tal[1];
            std::snprintf(b, sizeof b, "G9 A3 GATE (fresh test classes): rediscovery Wilson lower %.3f vs null Wilson upper %.3f -> %s | strict Wilson lower %.3f -> %s | "
                          "decision: %s", ecs::wilson_lower(a.red, a.n), ecs::wilson_upper(a.null, a.n), a.gate_met() ? "MET" : "NOT MET",
                          ecs::wilson_lower(a.strict, a.n), a.strict_met() ? "MET" : "NOT MET",
                          a.gate_met() && a.strict_met() ? "G9 A3 ADOPTED (open to Odin's veto)" : "not adopted");
            std::cout << b << "\n";
        }
        // S11a's amendment 1 measures (secondary)
        const auto& v = jj[gi];
        double wins = 0, losses = 0, d = 0;
        for (const auto& [a, z] : v) wins += a > z, losses += a < z, d += a - z;
        const double n = static_cast<double>(v.size()), m = wins + losses;
        const double p_sign = m > 0 ? ecs::binomial_upper_tail(wins, m, 0.5) : 1.0;
        vector<double> boot;
        std::mt19937_64 rng(20261104);
        for (int r = 0; r < 2000; ++r) {
            double t = 0;
            for (std::size_t k = 0; k < v.size(); ++k) {
                const auto& [a, z] = v[rng() % v.size()];
                t += a - z;
            }
            boot.push_back(t / n);
        }
        std::sort(boot.begin(), boot.end());
        std::snprintf(b, sizeof b, "G9 A3 %s SECONDARY classes %.0f | J_new > null %.0f, < %.0f, sign test p %.2g | mean J_new - null %+.3f [%+.3f, %+.3f]", gname[gi], n,
                      wins, losses, p_sign, d / n, boot[50], boot[1949]);
        std::cout << b << "\n";
    }
}

// ============================================================ G9, read 2: predictions on hidden S15 slots through S8a's loop
s8a::Tests g9_tests(s8a::Ctx& cx, const ecs::TripleStore& st, const ecs::LinkPredictor& lp, const string& s, const string& r, const string& o,
                    const cc::ClassRule& best) {
    s8a::Tests T;
    const int ri = rel_index(r);
    T.t[0] = s7::t1(st, s, r, o);
    // T2 stays inconclusive: class rules have no consequence rules
    T.t[2] = cc::class_rule_t3(best), T.score = best.pca_conf();
    T.dl = ecs::compression_gain_bits(lp, st, s, r, o) - std::log2(std::max(1e-12, lp.prob(s, r, o)));
    T.t[3] = T.dl > 0 ? kPos : T.dl < 0 ? kNeg : kInc;  // S7's choice: epsilon 0
    // T5 left out: Wikidata is what S15 grades against
    const auto pit = cx.page_of->find(s);
    if (pit != cx.page_of->end()) {
        const string k = s + '\x1f' + r;
        auto sit = cx.scans.find(k);
        if (sit == cx.scans.end()) sit = cx.scans.emplace(k, s7::scan_rel(*cx.data, (*cx.pages)[static_cast<std::size_t>(pit->second)], ri)).first;
        TestOutcome out[2];
        s7::t4(sit->second, ri, o, out);
        T.t[5] = out[cx.t4_variant];
    }
    return T;
}

struct SlotSet {
    std::set<std::pair<string, string>> slots;
    std::map<string, vector<std::pair<string, int>>> rows;  // s r -> (value, label), only when graded
};
SlotSet load_slots(const string& path, bool read_labels) {
    SlotSet S;
    for (const auto& R : s8a::load_s15(path)) {
        const int ri = rel_index(R.r);
        if (((fnv(R.s) >> 17) & 1) != 0 || ri < 0 || is_datey(ri)) continue;
        S.slots.insert({R.s, R.r});
        if (read_labels) S.rows[R.s + '\x1f' + R.r].push_back({R.v, R.label});
    }
    return S;
}

vector<s8a::Item> g9_hypotheses(const Env& E, const std::set<std::pair<string, string>>& slots, bool a2, const string& tag, ecs::TripleStore& st_out,
                                const Facts* extra = nullptr) {
    char b[512];
    Facts red;
    std::size_t held = 0;
    for (const auto& [s, fs] : E.kby)
        for (const auto& [r, o] : fs) {
            if (slots.count({s, r})) ++held;
            else red[s].push_back({r, o});
        }
    SurfMap surf;
    {
        std::map<string, std::set<string>> rels;
        context_facts(E.kby, false, rels, &surf);  // surface strings from the full store K (display only)
    }
    const G9 g = run_g9(red, a2, nullptr, extra);
    st_out = s8a::make_store(red);
    const ecs::LinkPredictor lp(st_out, {});
    s8a::Ctx cx;
    cx.data = &E.data, cx.pages = &E.pages, cx.page_of = &E.page_of, cx.wd = &E.wd, cx.t4_variant = 1;
    std::map<string, vector<const cc::ClassRule*>> by_head;  // r -> kept rules with a head of r
    for (const auto& R : g.rules)
        if (R.kept) by_head[g.C.attr_r[static_cast<std::size_t>(R.head)]].push_back(&R);
    vector<s8a::Item> out;
    std::size_t in_ctx = 0, with = 0;
    for (const auto& [s, r] : slots) {
        const int e = g.C.entity(s);
        if (e < 0) continue;
        ++in_ctx;
        std::map<string, std::pair<double, const cc::ClassRule*>> got;  // v -> (product of 1 - conf, best rule)
        for (const cc::ClassRule* R : by_head[r]) {
            const auto& ext = g.cs[R->cls].extent;
            if (!std::binary_search(ext.begin(), ext.end(), e)) continue;
            const string& v = g.C.attr_v[static_cast<std::size_t>(R->head)];
            auto it = got.find(v);
            if (it == got.end()) it = got.emplace(v, std::make_pair(1.0, R)).first;
            it->second.first *= 1.0 - R->pca_conf();
            if (R->pca_conf() > it->second.second->pca_conf()) it->second.second = R;
        }
        if (!got.empty()) ++with;
        const int ri = rel_index(r);
        for (const auto& [v, pr] : got) {
            string o = v;
            if (const auto sit = surf.find({r, v}); sit != surf.end()) {
                int bestn = -1;
                for (const auto& [str, n] : sit->second)
                    if (n > bestn) bestn = n, o = str;
            }
            s8a::Item x;
            x.s = s, x.r = r, x.o = o;
            x.T = g9_tests(cx, st_out, lp, s, r, o, *pr.second);
            x.p0 = std::min(0.98, std::max(0.02, (1.0 - pr.first) * s6::hard_constraint(st_out, s, ri, o)));
            out.push_back(std::move(x));
        }
    }
    std::size_t kept = 0;
    for (const auto& R : g.rules) kept += R.kept;
    std::snprintf(b, sizeof b, "%s G9 slots %zu | store facts hidden %zu, reduced store %zu | context attributes %zu entities %zu | candidate classes %zu kept rules %zu | "
                  "slots whose subject is in the context %zu, with a prediction %zu | hypotheses %zu", tag.c_str(), slots.size(), held, st_out.size(),
                  g.C.attr_r.size(), g.C.entities.size(), g.cand.size(), kept, in_ctx, with, out.size());
    std::cout << b << "\n";
    return out;
}

void write_pool(const string& path, const vector<s8a::Item>& h) {
    std::ofstream f(path);
    f << "split\tsubject\trelation\tobject\tP0\tposterior\tstate\tpromoted\tT1\tT2\tT3\tT6\tT5\tT4\n";
    for (const s8a::Item& x : h)
        f << "fresh\t" << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.p0 << '\t' << x.post << '\t' << x.state << '\t' << x.promoted73 << '\t'
          << ecs::test_outcome_name(x.T.t[0]) << '\t' << ecs::test_outcome_name(x.T.t[1]) << '\t' << ecs::test_outcome_name(x.T.t[2]) << '\t'
          << ecs::test_outcome_name(x.T.t[3]) << '\t' << ecs::test_outcome_name(x.T.t[4]) << '\t' << ecs::test_outcome_name(x.T.t[5]) << '\n';
}

void promote(const Env& E, const string& s15dir, const string& out, bool a2, bool read_test, const Facts* extra = nullptr) {
    const string D_ = extra ? "S11D A3 DEV" : "S11A DEV", T_ = extra ? "S11D A3 TEST" : "S11A TEST";  // S11d: G9 on A3
    // dev: S15 dev's choose half; its labels set the plan (s and c) and grade the dev read
    const SlotSet D = load_slots(s15dir + "/temporal_holdout_dev.tsv", true);
    ecs::TripleStore st;
    vector<s8a::Item> dh = g9_hypotheses(E, D.slots, a2, D_, st, extra);
    for (s8a::Item& x : dh) x.label = s8a::s15_grade(D.rows, rel_index(x.r), x.s, x.r, x.o);
    const bool allowed[s8a::kT] = {true, true, true, true, false, true};
    const s8a::Plan P = s8a::plan_from(dh, D_ + " G9", allowed, true);
    s8a::run_loop(dh, P, D_ + " G9", out + ".fresh_g9_dev.prereg.tsv");
    s8a::print_states(D_ + " G9", dh);
    s8a::print_precisions(D_ + " G9", dh, false);
    write_pool(out + ".fresh_g9_dev.hyps.tsv", dh);
    if (!read_test) return;
    // test (once): S15 test's bit 17 = 0 half; slots first, labels only after the loop
    const SlotSet T = load_slots(s15dir + "/temporal_holdout_test.tsv", false);
    vector<s8a::Item> th = g9_hypotheses(E, T.slots, a2, T_, st, extra);
    s8a::run_loop(th, P, T_ + " G9", out + ".fresh_g9.prereg.tsv");
    write_pool(out + ".fresh_g9.hyps.tsv", th);
    const SlotSet TL = load_slots(s15dir + "/temporal_holdout_test.tsv", true);
    for (s8a::Item& x : th) x.label = s8a::s15_grade(TL.rows, rel_index(x.r), x.s, x.r, x.o);
    s8a::print_states(T_ + " G9", th);
    s8a::print_precisions(T_ + " G9", th, false);
}

// ============================================================ G11: open relations
struct S15Props {
    std::unordered_map<string, vector<std::pair<string, string>>> by_s;  // subject -> (property, norm_ent(value)) of true rows
    std::unordered_set<string> subjects;
};
S15Props load_props(const string& path) {
    S15Props P;
    for (const auto& R : s8a::load_s15(path)) {
        P.subjects.insert(R.s);
        if (R.label == 1) P.by_s[R.s].push_back({R.r, norm_ent(R.v)});
    }
    return P;
}

void open_relations(const Env& E, const string& s15dir, bool read_test, bool skip_dates = false, const string& out = "") {
    char b[1024];
    const double t0 = now_s();
    if (skip_dates) std::cout << "G11 READER: links inside date candidates skipped (S11d, --g11-skip-dates)\n";
    // ---- unparsed (subject, link) pairs and their core keys
    vector<std::array<string, 3>> rec;  // (core, subject, object key)
    std::size_t pairs_all = 0, parsed = 0;
    vector<Cand> cands;
    std::array<string, 4> keys;
    for (const PageRef& pg : E.pages) {
        PageCtx px;
        load_page(E.data, pg, px);
        const string tk = px.forms.title_key;
        for (const Sent& s : px.sents) {
            const auto spans = find_subject(s, px.forms);
            if (spans.empty()) continue;
            gen_cands(s, cands);
            std::set<std::pair<int, int>> fired;
            apply_patterns(s, spans, cands, tk, [&](int, const Cand& c, const Span&) { fired.insert({c.b, c.e}); });
            const vector<std::pair<int, int>> ds = skip_dates ? date_spans(cands) : vector<std::pair<int, int>>{};
            for (const Cand& c : cands) {
                if (c.kind != K_LINK || c.k1 == tk) continue;
                if (skip_dates && ecs::span_within(c.b, c.e, ds)) continue;  // S11d: a linked year or day inside a date
                for (const Span& S : spans) {
                    if (c.b < S.e && S.b < c.e) continue;
                    bool sf = false;
                    if (!make_keys(s, S, c.b, c.e, keys, sf)) continue;
                    ++pairs_all;
                    if (fired.count({c.b, c.e})) { ++parsed; continue; }
                    string core = keys[0];
                    replace_all(core, "[P]", "[S]");
                    if (!cc::has_word(core)) continue;
                    rec.push_back({core, pg.title, c.k1});
                }
            }
        }
    }
    std::snprintf(b, sizeof b, "G11 PAIRS (subject mention, link) with a key %zu | parsed by a kept pattern %zu | unparsed with a word between %zu | %.1f s", pairs_all, parsed,
                  rec.size(), now_s() - t0);
    std::cout << b << "\n";
    vector<std::array<string, 3>> rec2 = rec;
    for (auto& r : rec2) r[0] = cc::connector_of(r[0]);
    const vector<cc::OpenCluster> cl[2] = {cc::open_clusters(rec, 5), cc::open_clusters(rec2, 5)};
    const S15Props dev = load_props(s15dir + "/temporal_holdout_dev.tsv");
    auto props_of = [](const S15Props& P) {
        return [&P](const string& s, const string& o) {
            vector<string> out;
            const auto it = P.by_s.find(s);
            if (it != P.by_s.end())
                for (const auto& [p, v] : it->second)
                    if (v == o) out.push_back(p);
            return out;
        };
    };
    auto gradable_in = [](const S15Props& P) { return [&P](const string& s) { return P.subjects.count(s) > 0; }; };
    struct Prop {
        std::size_t k;
        cc::ClusterGrade d;
    };
    vector<Prop> props[2];
    double obj[2] = {0, 0};
    for (int m = 0; m < 2; ++m) {
        std::set<std::array<string, 3>> hit;
        for (std::size_t k = 0; k < cl[m].size(); ++k) {
            const cc::ClusterGrade g = cc::grade_cluster(cl[m][k], props_of(dev), gradable_in(dev));
            if (g.gradable >= 3 && !g.prop.empty() && g.precision() >= 0.5) {
                props[m].push_back({k, g});
                for (const auto& [s, o] : cl[m][k].pairs)
                    for (const string& p : props_of(dev)(s, o))
                        if (p == g.prop) hit.insert({s, o, p});
            }
        }
        obj[m] = static_cast<double>(hit.size());
        std::snprintf(b, sizeof b, "G11 DEV %s clusters (>= 5 pairs) %zu | proposed relations %zu | distinct dev pairs holding their cluster's property %.0f",
                      m == 0 ? "C1 (core)" : "C2 (connector)", cl[m].size(), props[m].size(), obj[m]);
        std::cout << b << "\n";
    }
    const int ch = obj[1] >= 1.10 * obj[0] && obj[1] > 0 ? 1 : 0;
    std::snprintf(b, sizeof b, "G11 DEV CHOICE: %s (C2 needs >= 10%% more: %.0f vs %.0f)", ch ? "C2 (connector)" : "C1 (core)", obj[1], obj[0]);
    std::cout << b << "\n";
    for (std::size_t i = 0; i < props[ch].size(); ++i) {
        const auto& p = props[ch][i];
        const auto& c = cl[ch][p.k];
        string ex;
        for (std::size_t j = 0; j < c.pairs.size() && j < 3; ++j) ex += (j ? "; " : "") + c.pairs[j].first + " -> " + c.pairs[j].second;
        std::snprintf(b, sizeof b, "G11 DEV PROPOSAL %-44s pairs %4zu | %-26s dev %3.0f/%-3.0f %.3f | e.g. %s", c.key.c_str(), c.pairs.size(), p.d.prop.c_str(), p.d.hits,
                      p.d.gradable, p.d.precision(), ex.c_str());
        std::cout << b << "\n";
    }
    std::ofstream prop_tsv;  // S11d: the proposals with their proposed names (each one's dev property), as data for P10
    if (skip_dates && !out.empty()) {
        prop_tsv.open(out + ".g11_proposals.tsv");
        prop_tsv << "clustering\tkey\tproposed_name\tkind\tpairs\tdev_hits\tdev_gradable\ttest_hits\ttest_gradable\tholds_on_test\n";
        if (!read_test)
            for (const auto& p : props[ch])
                prop_tsv << (ch ? "C2" : "C1") << '\t' << cl[ch][p.k].key << '\t' << p.d.prop << '\t' << (rel_index(p.d.prop) >= 0 ? "pattern_learn" : "s15_added") << '\t'
                         << cl[ch][p.k].pairs.size() << '\t' << p.d.hits << '\t' << p.d.gradable << "\t\t\t\n";
    }
    if (!read_test) return;
    // ---- the test read (once): each proposal's dev property on its test-subject pairs
    const S15Props tst = load_props(s15dir + "/temporal_holdout_test.tsv");
    double holds[2] = {0, 0}, nprop[2] = {0, 0};
    std::set<std::array<string, 3>> g_all[2], h_all[2];  // [pattern_learn relation, S15's added relation]
    for (const auto& p : props[ch]) {
        const auto& c = cl[ch][p.k];
        const cc::ClusterGrade t = cc::grade_cluster(c, props_of(tst), gradable_in(tst), p.d.prop);
        const int kind = rel_index(p.d.prop) >= 0 ? 0 : 1;
        nprop[kind] += 1;
        const bool ok = t.gradable >= 3 && t.precision() >= 0.5;
        holds[kind] += ok;
        if (prop_tsv.is_open())
            prop_tsv << (ch ? "C2" : "C1") << '\t' << c.key << '\t' << p.d.prop << '\t' << (kind == 0 ? "pattern_learn" : "s15_added") << '\t' << c.pairs.size() << '\t'
                     << p.d.hits << '\t' << p.d.gradable << '\t' << t.hits << '\t' << t.gradable << '\t' << (ok ? 1 : 0) << '\n';
        for (const auto& [s, o] : c.pairs) {
            if (!tst.subjects.count(s)) continue;
            g_all[kind].insert({s, o, p.d.prop});
            for (const string& q : props_of(tst)(s, o))
                if (q == p.d.prop) h_all[kind].insert({s, o, q});
        }
        std::snprintf(b, sizeof b, "G11 TEST PROPOSAL %-44s %-26s test %3.0f/%-3.0f %.3f%s", c.key.c_str(), p.d.prop.c_str(), t.hits, t.gradable, t.precision(),
                      ok ? " HOLDS" : t.gradable < 3 ? " (too few)" : "");
        std::cout << b << "\n";
    }
    for (int kind = 0; kind < 3; ++kind) {
        const double n = kind == 2 ? nprop[0] + nprop[1] : nprop[kind], ho = kind == 2 ? holds[0] + holds[1] : holds[kind];
        const double g = kind == 2 ? static_cast<double>(g_all[0].size() + g_all[1].size()) : static_cast<double>(g_all[kind].size());
        const double h = kind == 2 ? static_cast<double>(h_all[0].size() + h_all[1].size()) : static_cast<double>(h_all[kind].size());
        std::snprintf(b, sizeof b, "G11 TEST %-30s proposals %3.0f hold %3.0f | pooled test pairs %4.0f holding %4.0f precision %.3f [%.3f, %.3f]",
                      kind == 0 ? "pattern_learn relations" : kind == 1 ? "S15's added relations" : "all", n, ho, g, h, g > 0 ? h / g : 0.0, ecs::wilson_lower(h, g),
                      ecs::wilson_upper(h, g));
        std::cout << b << "\n";
        if (kind == 2 && skip_dates) {
            const bool adopt = cc::adopt_g11_skip(ho, h, g);
            std::snprintf(b, sizeof b, "G11 S11D RULE (>= 8 hold and pooled >= 58/107 = 0.542): hold %.0f, pooled %.3f -> %s", ho, g > 0 ? h / g : 0.0,
                          adopt ? "the skip becomes G11's default reader" : "the skip stays opt-in");
            std::cout << b << "\n";
        }
    }
}

int run(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: pattern_learn --gen-concepts ENWIK8 SEEDS.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--attrs A1|A2|A3] "
                             "[--part all|rediscover|promote|open] [--read-test] [--g11-skip-dates]\n");
        return 2;
    }
    const double t0 = now_s();
    bool read_test = false, a2 = false, a3 = false, skip_dates = false;
    string part = "all";
    for (int i = 8; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--read-test") read_test = true;
        else if (a == "--attrs" && i + 1 < argc) {
            const string v = argv[++i];
            if (v != "A1" && v != "A2" && v != "A3") { std::fprintf(stderr, "--attrs A1, A2 or A3\n"); return 2; }
            a2 = v == "A2", a3 = v == "A3";
        } else if (a == "--g11-skip-dates") skip_dates = true;
        else if (a == "--part" && i + 1 < argc) part = argv[++i];
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    Env E;
    if (!load_env(E, argv[2], argv[3], argv[4], argv[5])) return 1;
    std::size_t nk = 0;
    for (const auto& [s, fs] : E.kby) nk += fs.size();
    std::cout << "S11A store K " << nk << " triples " << E.kby.size() << " subjects | attributes " << (a3 ? "A3" : a2 ? "A2" : "A1") << (read_test ? " | TEST READ" : "") << "\n";
    Facts X;  // S11d: S15's added facts (A3), from S15 dev, and from S15 test too on the test run
    if (a3) {
        const string s15 = argv[6];
        vector<string> files = {s15 + "/temporal_holdout_dev.tsv"};
        if (read_test) files.push_back(s15 + "/temporal_holdout_test.tsv");
        X = load_added(files);
        std::size_t nx = 0;
        for (const auto& [s, fs] : X) nx += fs.size();
        std::cout << "S11D added facts " << nx << " over " << X.size() << " subjects (from S15 " << (read_test ? "dev and test" : "dev") << ")\n";
    }
    if (part == "all" || part == "rediscover") {
        if (a3) rediscover_a3(E, X, read_test);
        else rediscover(E, a2, read_test);
    }
    if (part == "all" || part == "promote") promote(E, argv[6], argv[7], a2, read_test, a3 ? &X : nullptr);
    if (part == "all" || part == "open") open_relations(E, argv[6], read_test, skip_dates, argv[7]);
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    std::printf("S11A seconds %.1f peak RAM %.0f MB\n", now_s() - t0, static_cast<double>(ru.ru_maxrss) / 1024.0);
    return 0;
}

}  // namespace s11a

// ============================================================ S11c: G13 anomalies, G3 abduction, G4 and G5 analogy
// (docs/reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md; native/include/cypha/eval/gen_analogy.hpp)
//
// pattern_learn --gen-anomaly ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test] [--fresh-dev | --fresh-test]
//   G13 on planted corruptions of store K and on store T's own anomalies (graded by Wikidata); G3 on planted deletions
//   from store K; with --fresh-*, G3's and G13's hypothesis pools.
// pattern_learn --gen-analogy ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR VEC_PREFIX SIMVEC OUT_PREFIX [--read-test] [--fresh-dev | --fresh-test]
//   G4 and G5 on S8a's R1 random hide of store K (base rate and G2 beside them), the S15 temporal read on store T, and
//   with --fresh-* their hypothesis pools.
// pattern_learn --pool-grade s15|wd FILE HALF HYPS SRC_FRESH
//   grades a pool after S8d's source stage (`--sources … --fresh HYPS EMPTY`): S8c's grader on one half of an S15 file
//   (HALF 0 or 1 by bit 17; -1 all), or Wikidata values.
// Pools (--fresh-dev: S15 dev's bit-17 = 1 slots, S8d's; --fresh-test: S15 test's bit-17 = 0 slots) are written in
// S8d's format so S10's replay can read them. Nothing here answers a question (rule 10).
namespace s11c {
namespace an = cypha::eval::analogy;
using ecs::TestOutcome;
constexpr TestOutcome kPos = TestOutcome::Positive, kNeg = TestOutcome::Negative, kInc = TestOutcome::Inconclusive;
using Facts = s8a::Facts;

bool year_of(const string& v, int& y) {
    const Date D = parse_date_str(v);
    if (!D.ok()) return false;
    y = D.y;
    return true;
}

// Stores K and T as S8a reads them, with each triple's source rows (K: every seed row; T: S8a's t_sources).
struct Stores {
    Facts kby, tby;
    std::set<string> wd_src;
    std::map<string, double> k_sources, t_sources;  // s r o -> rows
};
string key3(const string& s, const string& r, const string& o) { return s + '\x1f' + r + '\x1f' + o; }
bool load_stores(const char* seeds, const char* facts, Stores& S) {
    std::set<string> kseen, tseen;
    std::ifstream f(seeds);
    if (!f) return false;
    string l;
    std::getline(f, l);
    while (std::getline(f, l)) {
        const auto c = split(l, '\t');
        if (c.size() != 6 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
        const string k = key3(c[0], c[1], c[2]);
        if (c[3] == "wikidata") S.wd_src.insert(c[0] + '\x1f' + c[1]);
        S.k_sources[k] += 1;
        if (kseen.insert(k).second) S.kby[c[0]].push_back({c[1], c[2]});
        if (c[3] == "wikidata") continue;
        S.t_sources[k] += 1;
        if (tseen.insert(k).second) S.tby[c[0]].push_back({c[1], c[2]});
    }
    std::ifstream g(facts);
    if (!g) return false;
    std::getline(g, l);
    while (std::getline(g, l)) {
        const auto c = split(l, '\t');
        if (c.size() < 3 || rel_index(c[1]) < 0 || c[0].empty() || c[2].empty()) continue;
        const string k = key3(c[0], c[1], c[2]);
        S.t_sources[k] += 1;
        if (tseen.insert(k).second) S.tby[c[0]].push_back({c[1], c[2]});
    }
    for (auto* by : {&S.kby, &S.tby})
        for (auto& [s, fs] : *by) std::sort(fs.begin(), fs.end());
    return true;
}

bool dev_subject(const string& s) { return ((fnv(s) >> 11) & 1) == 0; }  // S8a's R1 split
int half17(const string& s) { return static_cast<int>((fnv(s) >> 17) & 1); }

// "outer contains inner": a located_in or country chain of length <= 3 from inner reaches outer (registered for G13).
bool contains_chain(const ecs::TripleStore& st, const string& outer, const string& inner) {
    std::set<string> seen{inner};
    vector<string> fr{inner};
    for (int d = 0; d < 3 && !fr.empty(); ++d) {
        vector<string> nx;
        for (const auto& n : fr)
            for (const auto& f : st.facts_of(n)) {
                if (f.r != "located_in" && f.r != "country") continue;
                if (f.o == outer) return true;
                if (seen.insert(f.o).second) nx.push_back(f.o);
            }
        fr.swap(nx);
    }
    return false;
}

// The names that have a page "name (...)" besides them.
std::unordered_set<string> ambiguous_names(const vector<PageRef>& pages) {
    std::unordered_set<string> out;
    for (const PageRef& pg : pages) {
        const auto p = pg.title.find(" (");
        if (p != string::npos && p > 0 && pg.title.back() == ')') out.insert(pg.title.substr(0, p));
    }
    return out;
}

// T4 on the subject's page for one fact: +1 positive, -1 negative, 0 inconclusive (S7's pattern-or-co-mention).
struct PageReader {
    const s8b::Corpus* C = nullptr;
    std::map<string, s7::PageScan> scans;
    int says(const an::FactRef& f) {
        const int ri = rel_index(f.r);
        const auto pit = C->page_of.find(f.s);
        if (ri < 0 || pit == C->page_of.end()) return 0;
        const string k = f.s + '\x1f' + f.r;
        auto it = scans.find(k);
        if (it == scans.end()) it = scans.emplace(k, s7::scan_rel(C->data, C->pages[static_cast<std::size_t>(pit->second)], ri)).first;
        TestOutcome o[2];
        s7::t4(it->second, ri, f.o, o);
        return o[1] == kPos ? 1 : o[1] == kNeg ? -1 : 0;
    }
};

// Wikidata's grade of a fact: 1 it lists the value, 0 it lists other values only, -1 nothing for (s, r).
int wd_grade(const s7::WdObjs& wd, const an::FactRef& f) {
    const auto it = wd.find(f.s + '\x1f' + f.r);
    const int ri = rel_index(f.r);
    if (it == wd.end() || ri < 0 || it->second.empty()) return -1;
    return s7::wd_match(ri, f.o, it->second) ? 1 : 0;
}

std::pair<double, double> boot_diff(const vector<double>& a, const vector<double>& b, std::uint64_t seed = 20261029) {
    // 95% bootstrap interval of mean(a - b), paired by index
    if (a.empty() || a.size() != b.size()) return {0, 0};
    std::mt19937_64 rng(seed);
    vector<double> m;
    for (int it = 0; it < 2000; ++it) {
        double s = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const std::size_t j = rng() % a.size();
            s += a[j] - b[j];
        }
        m.push_back(s / static_cast<double>(a.size()));
    }
    std::sort(m.begin(), m.end());
    return {m[49], m[1949]};
}
string wilson(double k, double n) {
    char b[96];
    std::snprintf(b, sizeof b, "%.0f/%.0f = %.3f [%.3f, %.3f]", k, n, n > 0 ? k / n : 0.0, ecs::wilson_lower(k, n), ecs::wilson_upper(k, n));
    return b;
}

// ---- the generator loop for pools (S8b's: T1, T6, T4 with S7's dev point estimates), with the generator's own name
struct PoolItem {
    string s, r, o, gen, explain;
    double score = 0, prior = 0.5, post = 0;
    TestOutcome out[3] = {kInc, kInc, kInc};
    string state;
    bool promoted = false, skip_loop = false;  // skip_loop: G13's hypotheses go straight to the source stage
};
void pool_loop(vector<PoolItem>& items, const ecs::TripleStore& st, const s8b::Corpus& C, const string& log_path) {
    const ecs::LinkPredictor lp(st, ecs::LinkPredictor::Config{});
    ecs::HypothesisStore hs;
    ecs::PreregLog log;
    vector<std::size_t> idx;
    for (std::size_t i = 0; i < items.size(); ++i) {
        PoolItem& x = items[i];
        if (x.skip_loop) {
            x.post = x.prior, x.state = "queued";
            continue;
        }
        s8b::LoopItem li;
        li.s = x.s, li.r = x.r, li.o = x.o;
        s8b::run_tests(li, st, lp, C);
        for (int k = 0; k < 3; ++k) x.out[k] = li.out[k];
        ecs::Hypothesis h;
        h.id = "S11c-" + std::to_string(i);
        h.key = x.r + '|' + x.s + '|' + x.o;
        h.statement = x.s + " " + x.r + " " + x.o;
        h.generator = x.gen;
        for (const s8b::TestKind& K : s8b::kKinds) {
            ecs::HypTest t;
            t.kind = K.name, t.source = K.source, t.predicted = "positive", t.refuted_if = "negative";
            t.disconfirmation = K.disconfirmation, t.independent = K.independent, t.sensitivity = K.s, t.specificity = K.c;
            h.tests.push_back(t);
        }
        hs.propose(h);
        hs.score(h.id, x.prior);
        hs.transition(h.id, ecs::HypState::Queued, "scored");
        if (!log.register_plan(*hs.find(h.id))) continue;
        for (std::size_t k = 0; k < 3; ++k) {
            if (hs.find(h.id)->state != ecs::HypState::Queued) break;
            hs.transition(h.id, ecs::HypState::UnderTest, "next test");
            hs.record_outcome(h.id, k, x.out[k], &log);
        }
        const auto* hh = hs.find(h.id);
        x.post = hh->posterior, x.state = ecs::hyp_state_name(hh->state);
        idx.push_back(i);
    }
    std::ofstream(log_path) << log.serialize();
}
void write_pool(const string& path, const vector<PoolItem>& items) {
    std::ofstream f(path);
    f << "split\tsubject\trelation\tobject\tgenerator\tscore\tP0\tT1\tT6\tT4\tposterior\tstate\tpromoted\texplain\n";
    for (const PoolItem& x : items)
        f << "fresh\t" << x.s << '\t' << x.r << '\t' << x.o << '\t' << x.gen << '\t' << x.score << '\t' << x.prior << '\t' << ecs::test_outcome_name(x.out[0])
          << '\t' << ecs::test_outcome_name(x.out[1]) << '\t' << ecs::test_outcome_name(x.out[2]) << '\t' << x.post << '\t' << x.state << '\t'
          << (x.promoted ? 1 : 0) << '\t' << x.explain << '\n';
}
void pool_states(const string& tag, const vector<PoolItem>& items) {
    std::map<string, std::size_t> st;
    for (const auto& x : items) ++st[x.state];
    string l = tag + " states (no grade read):";
    for (const auto& [s, n] : st) l += " " + s + " " + std::to_string(n);
    std::cout << l << " | hypotheses " << items.size() << "\n";
}

// The slots of a pool: S15 dev's bit-17 = 1 half (--fresh-dev, S8d's slots) or S15 test's bit-17 = 0 half (--fresh-test).
std::set<std::pair<string, string>> pool_slots(const string& s15dir, bool test) {
    std::set<std::pair<string, string>> out;
    for (const auto& R : s8a::load_s15(s15dir + (test ? "/temporal_holdout_test.tsv" : "/temporal_holdout_dev.tsv")))
        if (half17(R.s) == (test ? 0 : 1) && rel_index(R.r) >= 0) out.insert({R.s, R.r});
    return out;
}

// ============================================================ G13 and G3: --gen-anomaly
struct Planted {
    int type = 0;  // 1..7: C1 century, C2 transposition, C3 BC sign, C4 swap, C5 capital, C6 spouse, C7 founder
    string subject;
    vector<an::FactRef> before, after;  // C4 changes two facts
};
const char* const kCorrName[8] = {"", "C1_century", "C2_transposition", "C3_bc_sign", "C4_swap", "C5_capital", "C6_spouse", "C7_founder"};

// A year written back into a value of the same form: "YYYY-MM-DD", plain digits, or "N BC". "" when it cannot be.
string with_year(const string& v, int y) {
    if (y == 0) return "";
    auto dig = [](char c) { return c >= '0' && c <= '9'; };
    const bool iso = v.size() == 10 && v[4] == '-' && v[7] == '-' && std::all_of(v.begin(), v.begin() + 4, dig);
    if (iso) return y >= 1000 && y <= 9999 ? std::to_string(y) + v.substr(4) : "";
    const bool digits = !v.empty() && v.size() <= 4 && std::all_of(v.begin(), v.end(), dig);
    const bool bc = v.size() > 3 && v.compare(v.size() - 3, 3, " BC") == 0 && std::all_of(v.begin(), v.end() - 3, dig);
    if (!digits && !bc) return "";
    return y > 0 ? std::to_string(y) : std::to_string(-y) + " BC";
}
int transpose_year(int y) {
    const int a = std::abs(y);
    if (a < 10) return 0;
    const int d1 = a % 10, d2 = (a / 10) % 10;
    if (d1 == d2) return 0;
    const int t = a - d1 - 10 * d2 + 10 * d1 + d2;
    return y < 0 ? -t : t;
}

vector<Planted> plant(const Facts& kby, const ecs::TripleStore& K, bool want_dev, std::mt19937_64& rng) {
    vector<Planted> out;
    std::set<string> used;
    std::set<string> taken;  // triples already changed
    vector<string> persons, capitals;
    {
        std::set<string> cap;
        for (const auto& [s, fs] : kby)
            for (const auto& [r, o] : fs) {
                if (r == "capital") cap.insert(o);
            }
        capitals.assign(cap.begin(), cap.end());
        for (const auto& [s, fs] : kby) {
            bool p = false;
            for (const auto& [r, o] : fs) p = p || r == "birth_year" || r == "birth_date";
            if (p) persons.push_back(s);
        }
    }
    auto is_date = [](const string& r) { return r == "birth_year" || r == "death_year" || r == "birth_date" || r == "death_date"; };
    for (int type = 1; type <= 7; ++type) {
        vector<Planted> cand;
        for (const auto& [s, fs] : kby) {
            if (dev_subject(s) != want_dev) continue;
            if (type == 4) {
                string by, dy;
                for (const auto& [r, o] : fs) {
                    if (r == "birth_year" && by.empty()) by = o;
                    if (r == "death_year" && dy.empty()) dy = o;
                }
                int yb = 0, yd = 0;
                if (by.empty() || dy.empty() || by == dy || !year_of(by, yb) || !year_of(dy, yd) || yb == yd) continue;
                Planted P;
                P.type = 4, P.subject = s;
                P.before = {{s, "birth_year", by}, {s, "death_year", dy}};
                P.after = {{s, "birth_year", dy}, {s, "death_year", by}};
                cand.push_back(P);
                continue;
            }
            for (const auto& [r, o] : fs) {
                Planted P;
                P.type = type, P.subject = s;
                P.before = {{s, r, o}};
                string nv;
                int y = 0;
                if (type <= 3) {
                    if (!is_date(r) || !year_of(o, y)) continue;
                    if (type == 1) nv = with_year(o, y + ((fnv(key3(s, r, o)) & 1) ? 100 : -100));
                    else if (type == 2) nv = with_year(o, transpose_year(y));
                    else nv = y < 0 ? with_year(o, -y) : "";
                } else if (type == 5) {
                    if (r != "capital" || capitals.size() < 2) continue;
                    for (int t = 0; t < 20 && nv.empty(); ++t) {
                        const string& c = capitals[rng() % capitals.size()];
                        if (c != o && c != s) nv = c;
                    }
                } else {
                    if (r != (type == 6 ? "spouse" : "founder") || persons.size() < 2) continue;
                    for (int t = 0; t < 20 && nv.empty(); ++t) {
                        const string& c = persons[rng() % persons.size()];
                        if (c != o && c != s) nv = c;
                    }
                }
                if (nv.empty() || nv == o || K.has(s, r, nv)) continue;
                P.after = {{s, r, nv}};
                cand.push_back(P);
            }
        }
        for (std::size_t i = cand.size(); i > 1; --i) std::swap(cand[i - 1], cand[rng() % i]);
        int n = 0;
        for (const Planted& P : cand) {
            if (n >= 100) break;
            if (used.count(P.subject)) continue;
            bool clash = false;
            for (const auto& f : P.before) clash = clash || taken.count(f.key());
            if (clash) continue;
            used.insert(P.subject);
            for (const auto& f : P.before) taken.insert(f.key());
            out.push_back(P), ++n;
        }
    }
    return out;
}

// Whether a planted change could be seen at all: the subject holds the fact its anomaly kind compares it with.
bool detectable(const Planted& P, const ecs::TripleStore& Kp) {
    const string& s = P.subject;
    auto has_rel = [&](const string& x, std::initializer_list<const char*> rs) {
        for (const auto& f : Kp.facts_of(x))
            for (const char* r : rs)
                if (f.r == r) return true;
        return false;
    };
    switch (P.type) {
        case 1:
        case 2:
        case 3: {
            const string& r = P.before[0].r;
            const bool birth = r == "birth_year" || r == "birth_date";
            const bool other_kind = birth ? has_rel(s, {"death_year", "death_date"}) : has_rel(s, {"birth_year", "birth_date"});
            const bool sibling = has_rel(s, {r == "birth_year" ? "birth_date" : r == "birth_date" ? "birth_year" : r == "death_year" ? "death_date" : "death_year"});
            return other_kind || sibling;
        }
        case 4: return true;
        case 5: return has_rel(P.after[0].o, {"country"});
        case 6: return has_rel(s, {"birth_year", "birth_date"}) && has_rel(P.after[0].o, {"birth_year", "birth_date"});
        case 7: return has_rel(s, {"founded"}) && has_rel(P.after[0].o, {"birth_year", "birth_date", "death_year", "death_date"});
    }
    return false;
}

an::AnomalyEvidence evidence_for(const ecs::TripleStore& st, const std::map<string, double>& sources, PageReader& page,
                                 const std::unordered_set<string>& amb) {
    an::AnomalyEvidence ev;
    ev.year_of = year_of;
    ev.sources = [&sources](const an::FactRef& f) {
        const auto it = sources.find(f.key());
        return it == sources.end() ? 0.0 : it->second;
    };
    ev.page = [&page](const an::FactRef& f) { return page.says(f); };
    ev.contains = [&st](const string& a, const string& b) { return contains_chain(st, a, b); };
    ev.ambiguous = [&amb](const string& n) { return amb.count(n) > 0; };
    return ev;
}

// G13-P: planted corruptions of store K (one split).
void g13_planted(const Stores& S, const s8b::Corpus& C, const std::unordered_set<string>& amb, bool dev, const string& out) {
    const string tag = dev ? "DEV G13-P" : "TEST G13-P";
    const ecs::TripleStore K = s8a::make_store(S.kby);
    std::mt19937_64 rng(20261101);
    const vector<Planted> P = plant(S.kby, K, dev, rng);
    // the corrupted store and its source counts (a changed fact keeps the original's)
    std::set<string> drop;
    std::map<string, double> src = S.k_sources;
    std::map<string, std::pair<int, std::size_t>> changed;  // new triple -> (type, event)
    for (std::size_t e = 0; e < P.size(); ++e)
        for (std::size_t j = 0; j < P[e].before.size(); ++j) {
            drop.insert(P[e].before[j].key());
            src[P[e].after[j].key()] = S.k_sources.count(P[e].before[j].key()) ? S.k_sources.at(P[e].before[j].key()) : 1.0;
            changed[P[e].after[j].key()] = {P[e].type, e};
        }
    Facts kp;
    for (const auto& [s, fs] : S.kby)
        for (const auto& [r, o] : fs)
            if (!drop.count(key3(s, r, o))) kp[s].push_back({r, o});
    for (const auto& Pl : P)
        for (const auto& f : Pl.after) kp[f.s].push_back({f.r, f.o});
    const ecs::TripleStore Kp = s8a::make_store(kp);
    PageReader page;
    page.C = &C;
    const an::AnomalyEvidence ev = evidence_for(Kp, src, page, amb);
    const auto A = an::find_anomalies(Kp, year_of);
    // per event: the first anomaly that involves one of its changed facts
    vector<int> first(P.size(), -1);
    std::size_t clean_flags = 0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        bool any = false;
        for (const auto& f : A[i].facts) {
            const auto it = changed.find(f.key());
            if (it == changed.end()) continue;
            any = true;
            if (first[it->second.second] < 0) first[it->second.second] = static_cast<int>(i);
        }
        if (!any) ++clean_flags;
    }
    std::mt19937_64 brng(20261103);
    std::array<double, 8> n{}, det{}, nd{}, dd{}, named{}, base{};
    vector<double> acc_g13, acc_base;
    std::ofstream f(out + (dev ? ".g13p_dev.tsv" : ".g13p_test.tsv"));
    f << "type\tsubject\tbefore\tafter\tdetectable\tdetected\tkind\tverdict\tnamed\tnamed_ok\tbaseline_ok\n";
    for (std::size_t e = 0; e < P.size(); ++e) {
        const Planted& Pl = P[e];
        const bool can = detectable(Pl, Kp);
        const bool got = first[e] >= 0;
        n[Pl.type] += 1, nd[Pl.type] += can;
        det[Pl.type] += got, dd[Pl.type] += got && can;
        string kind = "-", verdict = "-", nm = "-";
        int ok = -1, bok = -1;
        if (got) {
            const an::Anomaly& a = A[static_cast<std::size_t>(first[e])];
            const auto x = an::explain_anomaly(a, ev);
            kind = an::anomaly_kind_name(a.kind), verdict = an::verdict_name(x.verdict);
            auto is_changed = [&](int i) {
                if (i < 0) return false;
                for (const auto& g : Pl.after)
                    if (g == a.facts[static_cast<std::size_t>(i)]) return true;
                return false;
            };
            ok = is_changed(x.wrong);
            if (x.wrong >= 0) nm = a.facts[static_cast<std::size_t>(x.wrong)].r + "=" + a.facts[static_cast<std::size_t>(x.wrong)].o;
            // the baseline: fewer sources, then at random
            vector<double> sv;
            for (const auto& g : a.facts) sv.push_back(ev.sources(g));
            const double mn = *std::min_element(sv.begin(), sv.end());
            vector<int> low;
            for (std::size_t i = 0; i < sv.size(); ++i)
                if (sv[i] == mn) low.push_back(static_cast<int>(i));
            bok = is_changed(low[brng() % low.size()]);
            named[Pl.type] += ok, base[Pl.type] += bok;
            acc_g13.push_back(ok), acc_base.push_back(bok);
        }
        f << kCorrName[Pl.type] << '\t' << Pl.subject << '\t' << Pl.before[0].r << "=" << Pl.before[0].o << '\t' << Pl.after[0].r << "=" << Pl.after[0].o << '\t'
          << can << '\t' << got << '\t' << kind << '\t' << verdict << '\t' << nm << '\t' << ok << '\t' << bok << '\n';
    }
    char b[512];
    double tn = 0, td = 0, tnd = 0, tdd = 0;
    for (int t = 1; t <= 7; ++t) {
        tn += n[t], td += det[t], tnd += nd[t], tdd += dd[t];
        std::snprintf(b, sizeof b, "%s %-17s planted %3.0f | recall %s | detectable %3.0f, recall among them %s | named right %s | baseline %s", tag.c_str(),
                      kCorrName[t], n[t], wilson(det[t], n[t]).c_str(), nd[t], wilson(dd[t], nd[t]).c_str(), wilson(named[t], det[t]).c_str(),
                      wilson(base[t], det[t]).c_str());
        std::cout << b << "\n";
    }
    double sn = 0, sb = 0;
    for (std::size_t i = 0; i < acc_g13.size(); ++i) sn += acc_g13[i], sb += acc_base[i];
    const auto di = boot_diff(acc_g13, acc_base);
    std::snprintf(b, sizeof b, "%s ALL planted %.0f | recall %s | detectable %.0f, recall among them %s | named right %s | baseline %s | difference %+.3f [%+.3f, %+.3f]",
                  tag.c_str(), tn, wilson(td, tn).c_str(), tnd, wilson(tdd, tnd).c_str(), wilson(sn, td).c_str(), wilson(sb, td).c_str(),
                  td > 0 ? (sn - sb) / td : 0.0, di.first, di.second);
    std::cout << b << "\n";
    std::snprintf(b, sizeof b, "%s anomalies on the corrupted store %zu, of which involve no planted fact %zu", tag.c_str(), A.size(), clean_flags);
    std::cout << b << "\n";
    std::map<string, std::size_t> kinds;
    for (const auto& a : A) ++kinds[an::anomaly_kind_name(a.kind)];
    string kl;
    for (const auto& [k, c] : kinds) kl += " " + k + " " + std::to_string(c);
    std::cout << tag << " anomalies by kind:" << kl << "\n";
}

// G13-R: store T's own anomalies, graded by Wikidata. Returns the G13 pool hypotheses of the split when asked.
void g13_real(const Stores& S, const s8b::Corpus& C, const std::unordered_set<string>& amb, const s7::WdObjs& wd, bool read_test, const string& out,
              int pool_split, vector<PoolItem>* pool, vector<PoolItem>* qpool = nullptr) {
    const ecs::TripleStore T = s8a::make_store(S.tby);
    PageReader page;
    page.C = &C;
    const an::AnomalyEvidence ev = evidence_for(T, S.t_sources, page, amb);
    const auto A = an::find_anomalies(T, year_of);
    struct Acc {
        double n = 0, err = 0, err_g = 0, err_ok = 0, base_g = 0, base_ok = 0, other = 0, other_g = 0, other_ok = 0;
    };
    std::map<string, Acc> by_kind[2];
    Acc all[2];
    std::map<string, std::size_t> verdicts[2];
    std::ofstream f(out + ".g13r.tsv");
    f << "split\tkind\tfacts\tverdict\twrong\tbits\twhy\tgrades\tverdict_ok\tbaseline_ok\n";
    std::mt19937_64 brng(20261103);
    std::map<string, double> pool_p0;  // fact -> P0 (G13's pool)
    std::map<string, an::FactRef> pool_fact;
    for (const auto& a : A) {
        const int sp = dev_subject(a.facts[0].s) ? 0 : 1;
        const auto x = an::explain_anomaly(a, ev);
        if (pool && sp == pool_split)
            for (std::size_t i = 0; i < a.facts.size(); ++i) {
                const double p0 = static_cast<int>(i) == x.wrong ? 0.2 : 0.8;
                const string k = a.facts[i].key();
                auto it = pool_p0.find(k);
                if (it == pool_p0.end()) pool_p0[k] = p0, pool_fact[k] = a.facts[i];
                else it->second = std::min(it->second, p0);
            }
        if (sp == 1 && !read_test) continue;
        vector<int> g;
        string gs, fs;
        for (const auto& fr : a.facts) g.push_back(wd_grade(wd, fr)), gs += (gs.empty() ? "" : ",") + std::to_string(g.back()), fs += (fs.empty() ? "" : " | ") + fr.r + "=" + fr.o;
        int ok = -1, bok = -1;
        Acc& k = by_kind[sp][an::anomaly_kind_name(a.kind)];
        k.n += 1, all[sp].n += 1;
        ++verdicts[sp][an::verdict_name(x.verdict)];
        if (x.verdict == an::Verdict::Error || x.verdict == an::Verdict::DifferentEntity || (x.verdict == an::Verdict::Qualifier && x.wrong >= 0)) {
            k.err += 1, all[sp].err += 1;
            const int gw = g[static_cast<std::size_t>(x.wrong)];
            if (gw >= 0) ok = gw == 0, k.err_g += 1, k.err_ok += ok, all[sp].err_g += 1, all[sp].err_ok += ok;
            // the baseline: fewer sources, then at random
            vector<double> sv;
            for (const auto& fr : a.facts) sv.push_back(ev.sources(fr));
            const double mn = *std::min_element(sv.begin(), sv.end());
            vector<int> low;
            for (std::size_t i = 0; i < sv.size(); ++i)
                if (sv[i] == mn) low.push_back(static_cast<int>(i));
            const int bw = low[brng() % low.size()];
            const int gb = g[static_cast<std::size_t>(bw)];
            if (gb >= 0) bok = gb == 0, k.base_g += 1, k.base_ok += bok, all[sp].base_g += 1, all[sp].base_ok += bok;
        } else {
            k.other += 1, all[sp].other += 1;
            bool any = false, right = true;
            for (int v : g)
                if (v >= 0) any = true, right = right && v == 1;
            if (any) ok = right, k.other_g += 1, k.other_ok += ok, all[sp].other_g += 1, all[sp].other_ok += ok;
        }
        f << (sp ? "test" : "dev") << '\t' << an::anomaly_kind_name(a.kind) << '\t' << fs << '\t' << an::verdict_name(x.verdict) << '\t'
          << (x.wrong >= 0 ? a.facts[static_cast<std::size_t>(x.wrong)].r : "-") << '\t' << x.bits << '\t' << x.why << '\t' << gs << '\t' << ok << '\t' << bok << '\n';
    }
    char b[512];
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
        const string tag = sp ? "TEST G13-R" : "DEV G13-R";
        for (const auto& [kname, k] : by_kind[sp]) {
            std::snprintf(b, sizeof b, "%s %-22s anomalies %4.0f | error verdicts %4.0f, graded %3.0f, right %s | baseline %s | other verdicts %3.0f, graded %3.0f, right %s",
                          tag.c_str(), kname.c_str(), k.n, k.err, k.err_g, wilson(k.err_ok, k.err_g).c_str(), wilson(k.base_ok, k.base_g).c_str(), k.other, k.other_g,
                          wilson(k.other_ok, k.other_g).c_str());
            std::cout << b << "\n";
        }
        string vl;
        for (const auto& [v, c] : verdicts[sp]) vl += " " + v + " " + std::to_string(c);
        const Acc& a = all[sp];
        std::snprintf(b, sizeof b, "%s ALL anomalies %.0f (verdicts:%s) | error verdicts graded %.0f, right %s | baseline %s | other verdicts graded %.0f, right %s",
                      tag.c_str(), a.n, vl.c_str(), a.err_g, wilson(a.err_ok, a.err_g).c_str(), wilson(a.base_ok, a.base_g).c_str(), a.other_g,
                      wilson(a.other_ok, a.other_g).c_str());
        std::cout << b << "\n";
        if (sp == 1) {
            const bool g2 = a.err_g >= 10 && a.err_ok / a.err_g >= 0.9;
            std::snprintf(b, sizeof b, "TEST G13-R RULE part 2 (error-verdict precision >= 0.9 with >= 10 graded): %s", g2 ? "MET" : "NOT MET");
            std::cout << b << "\n";
        }
    }
    if (pool)
        for (const auto& [k, p0] : pool_p0) {
            PoolItem x;
            const auto& fr = pool_fact[k];
            x.s = fr.s, x.r = fr.r, x.o = fr.o, x.gen = "G13", x.prior = p0, x.score = 1 - p0, x.skip_loop = true;
            x.explain = p0 < 0.5 ? "named wrong" : "questioned";
            pool->push_back(x);
            if (qpool) {  // S11e item 1: the same facts as questions only, P0 0.5 whatever the verdict
                PoolItem q = x;
                q.prior = 0.5, q.score = 0.5, q.explain = "questioned";
                qpool->push_back(q);
            }
        }
}

// ---- G3: abduction
struct G3Case {
    an::FactRef O, B;
    int rule = -1;
};
// A derivation of O = (x, r, y) in the store by a kept rule through one body fact: the highest-PCA rule (then rule
// order); for a path rule atom 2 when fnv(O) is even, else atom 1 (the first join node in sorted order).
bool derivation(const ecs::RuleIndex& ix, const ecs::RuleSet& rs, const an::FactRef& O, an::FactRef& B, int& rule) {
    const auto& R = rs.rules();
    double best = -1;
    bool found = false;
    const ecs::TripleStore& st = ix.store();
    for (std::size_t i = 0; i < R.size(); ++i) {
        const ecs::Rule& Q = R[i];
        if (Q.head != O.r || Q.pca_conf() <= best) continue;
        an::FactRef b;
        bool ok = false;
        if (Q.kind == ecs::RuleKind::Const) ok = Q.head_o == O.o && st.has(O.s, Q.r1, Q.c1), b = {O.s, Q.r1, Q.c1};
        else if (Q.kind == ecs::RuleKind::Same) ok = st.has(O.s, Q.r1, O.o), b = {O.s, Q.r1, O.o};
        else if (Q.kind == ecs::RuleKind::Inverse) ok = st.has(O.o, Q.r1, O.s), b = {O.o, Q.r1, O.s};
        else {
            auto z1 = an::detail::vals_or_in(ix, O.s, Q.r1, !Q.inv1);
            std::sort(z1.begin(), z1.end());
            for (const auto& z : z1) {
                if (z == O.s || z == O.o) continue;
                const an::FactRef a2 = Q.inv2 ? an::FactRef{O.o, Q.r2, z} : an::FactRef{z, Q.r2, O.o};
                if (!st.has(a2.s, a2.r, a2.o)) continue;
                const an::FactRef a1 = Q.inv1 ? an::FactRef{z, Q.r1, O.s} : an::FactRef{O.s, Q.r1, z};
                b = (fnv(O.key()) & 1) == 0 ? a2 : a1, ok = true;
                break;
            }
        }
        if (ok && !(b == O)) best = Q.pca_conf(), B = b, rule = static_cast<int>(i), found = true;
    }
    return found;
}

double fact_bits(const ecs::LinkPredictor& lp, const an::FactRef& f) { return -std::log2(std::max(1e-12, lp.prob(f.s, f.r, f.o))); }

const double kRuleW[4] = {0.0, 0.25, 0.5, 1.0};  // S11e item 2: the rule term's weight in G3's order
struct G3Read {
    vector<double> r1, r5, r10, rnd1, rnd5, rnd10, lp1, lp5, lp10, pca1, pca5, pca10;
    vector<std::pair<double, int>> calib;  // (bits, is the deleted fact) for every ranked candidate fact, for the prior
    std::size_t drawn = 0, explained = 0, no_cand = 0, lost_rule = 0;
    vector<double> w1[4], w5[4], w10[4];          // S11e: recall at each kRuleW
    vector<std::pair<double, int>> wcal[4];       // S11e: (score at that weight, hit) for the prior
    std::size_t w_check = 0, w_mismatch = 0;      // S11e: ranks at w = 1 against S11c's
};
G3Read g3_planted(const Facts& kby, bool dev) {
    G3Read out;
    const ecs::TripleStore K = s8a::make_store(kby);
    const ecs::RuleIndex ix(K);
    const ecs::RuleSet rs(ix, ecs::mine_rules(ix), true);
    vector<G3Case> cand;
    for (const auto& [s, fs] : kby) {
        if (dev_subject(s) != dev) continue;
        for (const auto& [r, o] : fs) {
            G3Case c;
            c.O = {s, r, o};
            if (derivation(ix, rs, c.O, c.B, c.rule)) cand.push_back(c);
        }
    }
    std::mt19937_64 rng(20261102);
    for (std::size_t i = cand.size(); i > 1; --i) std::swap(cand[i - 1], cand[rng() % i]);
    vector<G3Case> cases;
    std::set<string> Bs, Os;
    for (const auto& c : cand) {
        if (cases.size() >= 500) break;
        if (Bs.count(c.B.key()) || Os.count(c.B.key()) || Bs.count(c.O.key())) continue;
        cases.push_back(c), Bs.insert(c.B.key()), Os.insert(c.O.key());
    }
    out.drawn = cases.size();
    // amendment 2 (dev): each case on its own, its B alone deleted and the rules mined again (3 threads; cases are
    // independent, so the order of the results does not depend on the threads)
    struct One {
        bool explained = false, none = false;
        std::size_t rank = 0, n = 0, rl = 0, rp = 0;
        vector<std::pair<double, int>> calib;
        string no_rule;
        std::size_t rw[4] = {0, 0, 0, 0};          // S11e: ranks at each kRuleW
        vector<std::pair<double, int>> wcal[4];
    };
    vector<One> res(cases.size());
    auto work = [&](std::size_t from, std::size_t step) {
        for (std::size_t ci = from; ci < cases.size(); ci += step) {
            const G3Case& c = cases[ci];
            One& R = res[ci];
            Facts red;
            for (const auto& [s, fs] : kby)
                for (const auto& [r, o] : fs)
                    if (!(s == c.B.s && r == c.B.r && o == c.B.o)) red[s].push_back({r, o});
            const ecs::TripleStore st = s8a::make_store(red);
            const ecs::RuleIndex ix2(st);
            const ecs::RuleSet rs2(ix2, ecs::mine_rules(ix2), true);
            if (an::derivable(rs2, c.O.s, c.O.r, c.O.o)) {
                R.explained = true;
                continue;
            }
            const ecs::LinkPredictor lp(st, ecs::LinkPredictor::Config{});
            auto bits = [&](const an::FactRef& f) { return fact_bits(lp, f); };
            const auto ex = an::abduce(ix2, rs2, c.O.s, c.O.r, c.O.o, bits);
            const auto rf = an::ranked_facts(ex);
            if (rf.empty()) {
                R.none = true;
                const string rt = rs.rules()[static_cast<std::size_t>(c.rule)].text();
                bool kept = false;
                for (const auto& q : rs2.rules()) kept = kept || q.text() == rt;
                if (!kept) R.no_rule = rt;
                continue;
            }
            for (std::size_t i = 0; i < rf.size(); ++i) {
                const bool hit = rf[i].first == c.B;
                R.calib.push_back({rf[i].second, hit});
                if (hit && !R.rank) R.rank = i + 1;
            }
            R.n = rf.size();
            // the link predictor alone, and the rule's PCA alone, on the same candidate facts
            struct C2 {
                an::FactRef f;
                double lpb, pcab, sup;
            };
            std::map<an::FactRef, C2> m;
            for (const auto& e : ex) {
                const auto& f = e.add.front();
                const double pb = -std::log2(std::max(1e-12, rs2.rules()[static_cast<std::size_t>(e.rule)].pca_conf()));
                auto it = m.find(f);
                if (it == m.end()) m[f] = {f, bits(f), pb, e.support};
                else if (pb < it->second.pcab || (pb == it->second.pcab && e.support > it->second.sup)) it->second.pcab = pb, it->second.sup = e.support;
            }
            vector<C2> v;
            for (auto& [k, x] : m) v.push_back(x);
            auto rank_by = [&](auto key) {
                std::sort(v.begin(), v.end(), [&](const C2& a, const C2& b) {
                    const double ka = key(a), kb = key(b);
                    if (ka != kb) return ka < kb;
                    if (a.sup != b.sup) return a.sup > b.sup;
                    return a.f < b.f;
                });
                for (std::size_t i = 0; i < v.size(); ++i)
                    if (v[i].f == c.B) return i + 1;
                return std::size_t{0};
            };
            R.rl = rank_by([](const C2& a) { return a.lpb; }), R.rp = rank_by([](const C2& a) { return a.pcab; });
            for (int k = 0; k < 4; ++k) {  // S11e item 2
                const auto rw = an::ranked_facts_weighted(ex, kRuleW[k]);
                for (std::size_t i = 0; i < rw.size(); ++i) {
                    const bool hit = rw[i].first == c.B;
                    R.wcal[k].push_back({rw[i].second, hit});
                    if (hit && !R.rw[k]) R.rw[k] = i + 1;
                }
            }
        }
    };
    {
        vector<std::thread> th;
        for (std::size_t t = 0; t < 3; ++t) th.emplace_back(work, t, 3);
        for (auto& t : th) t.join();
    }
    std::map<string, std::size_t> lost;
    for (const One& R : res) {
        if (R.explained) {
            ++out.explained;
            continue;
        }
        if (R.none) {
            ++out.no_cand;
            if (!R.no_rule.empty()) ++lost[R.no_rule];
            for (auto* v : {&out.r1, &out.r5, &out.r10, &out.rnd1, &out.rnd5, &out.rnd10, &out.lp1, &out.lp5, &out.lp10, &out.pca1, &out.pca5, &out.pca10}) v->push_back(0);
            for (int k = 0; k < 4; ++k) out.w1[k].push_back(0), out.w5[k].push_back(0), out.w10[k].push_back(0);
            continue;
        }
        for (int k = 0; k < 4; ++k) {
            const std::size_t q = R.rw[k];
            out.w1[k].push_back(q && q <= 1), out.w5[k].push_back(q && q <= 5), out.w10[k].push_back(q && q <= 10);
            out.wcal[k].insert(out.wcal[k].end(), R.wcal[k].begin(), R.wcal[k].end());
        }
        ++out.w_check, out.w_mismatch += R.rw[3] != R.rank;
        out.calib.insert(out.calib.end(), R.calib.begin(), R.calib.end());
        const double n = static_cast<double>(R.n);
        const std::size_t rank = R.rank, rl = R.rl, rp = R.rp;
        out.r1.push_back(rank && rank <= 1), out.r5.push_back(rank && rank <= 5), out.r10.push_back(rank && rank <= 10);
        out.rnd1.push_back(rank ? std::min(1.0, 1.0 / n) : 0.0), out.rnd5.push_back(rank ? std::min(1.0, 5.0 / n) : 0.0), out.rnd10.push_back(rank ? std::min(1.0, 10.0 / n) : 0.0);
        out.lp1.push_back(rl && rl <= 1), out.lp5.push_back(rl && rl <= 5), out.lp10.push_back(rl && rl <= 10);
        out.pca1.push_back(rp && rp <= 1), out.pca5.push_back(rp && rp <= 5), out.pca10.push_back(rp && rp <= 10);
    }
    out.lost_rule = 0;
    for (const auto& [k, c] : lost) out.lost_rule += c;
    return out;
}
void print_g3(const string& tag, const G3Read& R) {
    char b[768];
    std::snprintf(b, sizeof b, "%s cases drawn %zu | still explained after the deletion %zu | read %zu (no candidate %zu, of which the rule was no longer kept %zu)",
                  tag.c_str(), R.drawn, R.explained, R.r1.size(), R.no_cand, R.lost_rule);
    std::cout << b << "\n";
    auto row = [&](const char* what, const vector<double>& a1, const vector<double>& a5, const vector<double>& a10) {
        const auto c1 = s8a::boot_mean(a1), c5 = s8a::boot_mean(a5), c10 = s8a::boot_mean(a10);
        std::snprintf(b, sizeof b, "%s RECALL %-22s @1 %.3f [%.3f, %.3f] @5 %.3f [%.3f, %.3f] @10 %.3f [%.3f, %.3f]", tag.c_str(), what, s8a::mean(a1), c1.first,
                      c1.second, s8a::mean(a5), c5.first, c5.second, s8a::mean(a10), c10.first, c10.second);
        std::cout << b << "\n";
    };
    row("G3 (bits)", R.r1, R.r5, R.r10);
    row("random order", R.rnd1, R.rnd5, R.rnd10);
    row("link predictor alone", R.lp1, R.lp5, R.lp10);
    row("rule PCA alone", R.pca1, R.pca5, R.pca10);
    const auto d10 = boot_diff(R.r10, R.rnd10), d1l = boot_diff(R.r1, R.lp1), d1p = boot_diff(R.r1, R.pca1);
    std::snprintf(b, sizeof b, "%s DIFF recall@10 G3 - random %+.3f [%+.3f, %+.3f] | recall@1 G3 - LP alone %+.3f [%+.3f, %+.3f] | G3 - PCA alone %+.3f [%+.3f, %+.3f]",
                  tag.c_str(), s8a::mean(R.r10) - s8a::mean(R.rnd10), d10.first, d10.second, s8a::mean(R.r1) - s8a::mean(R.lp1), d1l.first, d1l.second,
                  s8a::mean(R.r1) - s8a::mean(R.pca1), d1p.first, d1p.second);
    std::cout << b << "\n";
}

// S11e item 2: G3's recall at each rule weight on the cases read, and rule 2's choice (the index into kRuleW).
int print_g3_weights(const string& tag, const G3Read& R) {
    char b[768];
    std::snprintf(b, sizeof b, "%s S11E WEIGHT CHECK ranks at w = 1 equal S11c's G3 ranks on %zu of %zu cases", tag.c_str(), R.w_check - R.w_mismatch, R.w_check);
    std::cout << b << "\n";
    for (int k = 0; k < 4; ++k) {
        const auto c1 = s8a::boot_mean(R.w1[k]), c10 = s8a::boot_mean(R.w10[k]);
        const auto d1 = boot_diff(R.w1[k], R.w1[3]), d10 = boot_diff(R.w10[k], R.w10[3]);
        std::snprintf(b, sizeof b,
                      "%s S11E RECALL w %.2f @1 %.3f [%.3f, %.3f] @5 %.3f @10 %.3f [%.3f, %.3f] | @1 - w1 %+.3f [%+.3f, %+.3f] | @10 - w1 %+.3f [%+.3f, %+.3f]",
                      tag.c_str(), kRuleW[k], s8a::mean(R.w1[k]), c1.first, c1.second, s8a::mean(R.w5[k]), s8a::mean(R.w10[k]), c10.first, c10.second,
                      s8a::mean(R.w1[k]) - s8a::mean(R.w1[3]), d1.first, d1.second, s8a::mean(R.w10[k]) - s8a::mean(R.w10[3]), d10.first, d10.second);
        std::cout << b << "\n";
    }
    // rule 2: the highest recall@1 (ties to the larger w) replaces w = 1 only if its @1 difference interval is above 0
    // and its @10 is not lower by more than 0.02
    int best = 3;
    for (int k = 3; k >= 0; --k)
        if (s8a::mean(R.w1[k]) > s8a::mean(R.w1[best])) best = k;
    int pick = 3;
    if (best != 3) {
        const auto d1 = boot_diff(R.w1[best], R.w1[3]);
        if (d1.first > 0 && s8a::mean(R.w10[best]) >= s8a::mean(R.w10[3]) - 0.02) pick = best;
    }
    std::snprintf(b, sizeof b, "%s S11E RULE 2: highest recall@1 at w %.2f -> G3's order uses w %.2f", tag.c_str(), kRuleW[best], kRuleW[pick]);
    std::cout << b << "\n";
    return pick;
}

// G3's pool: the unexplained observations of the store with the pool's slots hidden; each distinct added fact that
// falls in a hidden slot, at its best explanation. wi: the rule weight's index into kRuleW (S11c: 3, w = 1).
void g3_pool(const Facts& kby, const std::set<std::pair<string, string>>& slots, const G3Read& dev_read, vector<PoolItem>& pool, std::size_t& held,
             std::size_t& observations, int wi = 3) {
    Facts red;
    held = 0;
    for (const auto& [s, fs] : kby)
        for (const auto& [r, o] : fs) {
            if (slots.count({s, r})) ++held;
            else red[s].push_back({r, o});
        }
    const ecs::TripleStore st = s8a::make_store(red);
    const ecs::RuleIndex ix(st);
    const ecs::RuleSet rs(ix, ecs::mine_rules(ix), true);
    const ecs::LinkPredictor lp(st, ecs::LinkPredictor::Config{});
    // only facts in hidden slots are kept, so others need no bits
    auto bits = [&](const an::FactRef& f) { return slots.count({f.s, f.r}) ? fact_bits(lp, f) : 1e9; };
    std::map<an::FactRef, std::pair<double, string>> best;
    observations = 0;
    for (const auto& [s, fs] : red)
        for (const auto& [r, o] : fs) {
            if (an::derivable(rs, s, r, o)) continue;
            ++observations;
            for (const auto& e : an::abduce(ix, rs, s, r, o, bits, 5000, kRuleW[wi])) {
                const auto& f = e.add.front();
                if (!slots.count({f.s, f.r})) continue;
                auto it = best.find(f);
                const string why = "explains " + r + "(" + s + ", " + o + ") by " + rs.rules()[static_cast<std::size_t>(e.rule)].text();
                if (it == best.end() || e.bits < it->second.first) best[f] = {e.bits, why};
            }
        }
    // the prior: an isotonic map from bits to P(the added fact is the deleted one), fitted on G3-P's dev cases
    vector<double> p;
    vector<int> y;
    for (const auto& [bt, hit] : (wi == 3 ? dev_read.calib : dev_read.wcal[wi])) p.push_back(-bt), y.push_back(hit);
    ecs::IsotonicCalibrator cal;
    cal.fit(p, y);
    for (const auto& [f, bw] : best) {
        PoolItem x;
        x.s = f.s, x.r = f.r, x.o = f.o, x.gen = "G3", x.score = bw.first, x.explain = bw.second;
        x.prior = std::min(0.98, std::max(0.02, cal.predict(-bw.first)));
        pool.push_back(x);
    }
}

int run_anomaly(int argc, char** argv) {
    if (argc < 9) {
        std::fprintf(stderr, "usage: pattern_learn --gen-anomaly ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test] [--fresh-dev | --fresh-test] "
                             "[--g3-weights] [--g13-questions]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[8], s15dir = argv[7];
    bool read_test = false, fresh_dev = false, fresh_test = false, g3_weights = false, g13_questions = false;
    for (int i = 9; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--read-test") read_test = true;
        else if (a == "--fresh-dev") fresh_dev = true;
        else if (a == "--fresh-test") fresh_test = true;
        else if (a == "--g3-weights") g3_weights = true;      // S11e item 2
        else if (a == "--g13-questions") g13_questions = true;  // S11e item 1
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    s8b::Corpus C;
    if (!s8b::load_corpus(argv[2], argv[5], argv[6], C)) return 1;
    Stores S;
    if (!load_stores(argv[3], argv[4], S)) {
        std::fprintf(stderr, "cannot read %s or %s\n", argv[3], argv[4]);
        return 1;
    }
    const s7::WdObjs wd = s7::load_wd_objs(argv[5]);
    const auto amb = ambiguous_names(C.pages);
    char b[512];
    std::snprintf(b, sizeof b, "S11C STORES K %zu subjects | T %zu subjects | ambiguous names %zu", S.kby.size(), S.tby.size(), amb.size());
    std::cout << b << "\n";
    // ---- G13
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) g13_planted(S, C, amb, sp == 0, out);
    vector<PoolItem> g13pool, g13qpool;
    g13_real(S, C, amb, wd, read_test, out, fresh_test ? 1 : 0, (fresh_dev || fresh_test) ? &g13pool : nullptr,
             (fresh_dev || fresh_test) && g13_questions ? &g13qpool : nullptr);
    // ---- G3
    const G3Read dev3 = g3_planted(S.kby, true);
    print_g3("DEV G3-P", dev3);
    const int g3w = g3_weights ? print_g3_weights("DEV G3-P", dev3) : 3;
    if (read_test) {
        const G3Read t3 = g3_planted(S.kby, false);
        print_g3("TEST G3-P", t3);
        const double r10 = s8a::mean(t3.r10);
        const auto d = boot_diff(t3.r10, t3.rnd10);
        std::snprintf(b, sizeof b, "TEST G3 RULE (recall@10 >= 0.6 and above random order): recall@10 %.3f, difference [%+.3f, %+.3f] -> %s", r10, d.first, d.second,
                      r10 >= 0.6 && d.first > 0 ? "KEPT" : "NOT KEPT");
        std::cout << b << "\n";
    }
    // ---- pools
    if (fresh_dev || fresh_test) {
        const auto slots = pool_slots(s15dir, fresh_test);
        const string ptag = fresh_test ? "S11C POOL test (S15 test, bit 17 = 0)" : "S11C POOL dev (S15 dev, bit 17 = 1)";
        vector<PoolItem> g3pool;
        std::size_t held = 0, obs = 0;
        g3_pool(S.kby, slots, dev3, g3pool, held, obs);
        Facts red;
        for (const auto& [s, fs] : S.kby)
            for (const auto& [r, o] : fs)
                if (!slots.count({s, r})) red[s].push_back({r, o});
        const ecs::TripleStore st = s8a::make_store(red);
        pool_loop(g3pool, st, C, out + ".fresh_g3.prereg.tsv");
        std::snprintf(b, sizeof b, "%s G3 slots %zu | store facts hidden %zu | unexplained observations %zu | hypotheses %zu", ptag.c_str(), slots.size(), held, obs,
                      g3pool.size());
        std::cout << b << "\n";
        pool_states(ptag + " G3", g3pool);
        write_pool(out + ".fresh_g3.hyps.tsv", g3pool);
        const ecs::TripleStore T = s8a::make_store(S.tby);
        pool_loop(g13pool, T, C, out + ".fresh_g13.prereg.tsv");
        pool_states(ptag + " G13 (store T, " + (fresh_test ? "test" : "dev") + " subjects by bit 11)", g13pool);
        write_pool(out + ".fresh_g13.hyps.tsv", g13pool);
        if (g3_weights) {  // S11e item 2: G3's pool with the chosen rule weight
            vector<PoolItem> g3w_pool;
            std::size_t held2 = 0, obs2 = 0;
            g3_pool(S.kby, slots, dev3, g3w_pool, held2, obs2, g3w);
            pool_loop(g3w_pool, st, C, out + ".fresh_g3w.prereg.tsv");
            std::snprintf(b, sizeof b, "%s G3 (S11e, w %.2f) hypotheses %zu", ptag.c_str(), kRuleW[g3w], g3w_pool.size());
            std::cout << b << "\n";
            pool_states(ptag + " G3 (S11e)", g3w_pool);
            write_pool(out + ".fresh_g3w.hyps.tsv", g3w_pool);
        }
        if (g13_questions) {  // S11e item 1: G13's facts as questions only
            pool_loop(g13qpool, T, C, out + ".fresh_g13q.prereg.tsv");
            pool_states(ptag + " G13 questions only (S11e)", g13qpool);
            write_pool(out + ".fresh_g13q.hyps.tsv", g13qpool);
        }
    }
    std::snprintf(b, sizeof b, "S11C --gen-anomaly done in %.1f s", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

// ============================================================ G4 and G5: --gen-analogy
struct SlotRank {
    vector<ecs::RuleSet::Cand> rc;  // the generator's candidates with scores (rank_value's M_RULES order)
};
vector<ecs::RuleSet::Cand> as_cands(const vector<std::pair<string, double>>& v) {
    vector<ecs::RuleSet::Cand> out;
    for (const auto& [o, s] : v) {
        ecs::RuleSet::Cand c;
        c.o = o, c.score = s;
        out.push_back(c);
    }
    return out;
}

// G5's candidates for (s, r): every object of r with a vector, scored cosine + 2 (so they rank above the unscored).
vector<ecs::RuleSet::Cand> g5_cands(const an::VectorAnalogy& va, const string& s, const string& r, const vector<string>& objs) {
    vector<ecs::RuleSet::Cand> out;
    const auto q = va.query(s, r);
    if (q.empty()) return out;
    for (const auto& o : objs) {
        double c = 0;
        if (o == s || !va.score(q, o, c)) continue;
        ecs::RuleSet::Cand x;
        x.o = o, x.score = c + 2.0;
        out.push_back(x);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.score != b.score ? a.score > b.score : a.o < b.o; });
    return out;
}

struct QueryRow {
    string s, r, o;
    bool dev = true;
    std::size_t rank_base = 0, rank_g2 = 0, rank_g4[3] = {0, 0, 0}, rank_g5[2] = {0, 0};
    double top_g4[3] = {0, 0, 0}, top_g5[2] = {0, 0};
    string top_g4o[3], top_g5o[2];
    bool scored_g4[3] = {false, false, false}, scored_g5[2] = {false, false};
    std::size_t rank_g4d = 0;  // S11e item 3: lambda 0.25, 0 for date relations
    double top_g4d = 0;
    string top_g4do;
    bool scored_g4d = false;
};
const double kLambda[3] = {0.0, 0.25, 1.0};
// S11e item 3: the constant-role weight for (r): lambda, or lambda_date for date relations
double g4_lambda(const string& r, double lambda, double lambda_date) {
    const int ri = rel_index(r);
    return ri >= 0 && is_datey(ri) ? lambda_date : lambda;
}

double rr(std::size_t k) { return k ? 1.0 / static_cast<double>(k) : 0.0; }

int run_analogy(int argc, char** argv) {
    if (argc < 11) {
        std::fprintf(stderr, "usage: pattern_learn --gen-analogy ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR VEC_PREFIX SIMVEC OUT_PREFIX "
                             "[--read-test] [--fresh-dev | --fresh-test] [--g4-date-lambda]\n");
        return 2;
    }
    const double t0 = now_s();
    const string s15dir = argv[7], vecp = argv[8], simvec = argv[9], out = argv[10];
    bool read_test = false, fresh_dev = false, fresh_test = false, g4_date = false;
    for (int i = 11; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--read-test") read_test = true;
        else if (a == "--fresh-dev") fresh_dev = true;
        else if (a == "--fresh-test") fresh_test = true;
        else if (a == "--g4-date-lambda") g4_date = true;  // S11e item 3
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    s8b::Corpus C;
    if (!s8b::load_corpus(argv[2], argv[5], argv[6], C)) return 1;
    Stores S;
    if (!load_stores(argv[3], argv[4], S)) return 1;
    an::WordVectors wv[2];
    {
        string err;
        if (!wv[0].load_f32(vecp + ".vectors.f32", vecp + ".vocab.tsv", err) || !wv[1].load_projected(simvec, err)) {
            std::fprintf(stderr, "vectors: %s\n", err.c_str());
            return 1;
        }
    }
    char b[768];
    std::snprintf(b, sizeof b, "S11C VECTORS %zu words x %d (PPMI-SVD) | %zu words x %d (simvec)", wv[0].size(), wv[0].dim(), wv[1].size(), wv[1].dim());
    std::cout << b << "\n";
    // ---- read 1: S8a's R1 random hide of store K
    vector<std::array<string, 3>> trip;
    for (const auto& [s, fs] : S.kby)
        for (const auto& [r, o] : fs) trip.push_back({s, r, o});
    const ecs::TripleStore K = s8a::make_store(S.kby);
    vector<std::size_t> h(trip.size());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = i;
    {
        std::mt19937_64 rng(20261029);
        for (std::size_t i = h.size(); i > 1; --i) std::swap(h[i - 1], h[rng() % i]);
    }
    const std::set<std::size_t> hidden(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(trip.size() / 10));
    Facts red;
    for (std::size_t i = 0; i < trip.size(); ++i)
        if (!hidden.count(i)) red[trip[i][0]].push_back({trip[i][1], trip[i][2]});
    const ecs::TripleStore st = s8a::make_store(red);
    const ecs::RuleIndex ix(st);
    const ecs::RuleSet rs(ix, ecs::mine_rules(ix), true);
    const auto objs = s8a::objects_by_rel(st, red);
    const an::VectorAnalogy va0(st, wv[0]), va1(st, wv[1]);
    const an::VectorAnalogy* va[2] = {&va0, &va1};
    static const vector<string> none;
    vector<QueryRow> Q;
    std::map<std::pair<string, string>, vector<an::Analog>> analog_cache;
    const double tq = now_s();
    for (std::size_t i : hidden) {
        const auto& [s, r, o] = trip[i];
        QueryRow q;
        q.s = s, q.r = r, q.o = o, q.dev = dev_subject(s);
        if (!q.dev && !read_test) continue;
        const int ri = rel_index(r);
        auto filt = [&](const string& c) {
            for (const auto& f : K.facts_of(s))
                if (f.r == r && s8a::same_value(ri, f.o, c)) return true;
            return false;
        };
        const auto it = objs.find(r);
        const vector<string>& ob = it == objs.end() ? none : it->second;
        q.rank_base = s8a::rank_value(st, ri, s, o, ob, {}, nullptr, s8a::M_BASE, filt);
        q.rank_g2 = s8a::rank_value(st, ri, s, o, ob, rs.candidates(s, r), nullptr, s8a::M_RULES, filt);
        auto ait = analog_cache.find({s, r});
        if (ait == analog_cache.end()) ait = analog_cache.emplace(std::make_pair(s, r), an::find_analogs(st, s, r)).first;
        for (int l = 0; l < 3; ++l) {
            const auto tr = an::transfer(st, ait->second, s, r, kLambda[l]);
            q.scored_g4[l] = !tr.empty();
            if (!tr.empty()) q.top_g4[l] = tr.front().second, q.top_g4o[l] = tr.front().first;
            q.rank_g4[l] = s8a::rank_value(st, ri, s, o, ob, as_cands(tr), nullptr, s8a::M_RULES, filt);
        }
        if (g4_date) {  // S11e item 3
            const auto tr = an::transfer(st, ait->second, s, r, g4_lambda(r, kLambda[1], 0.0));
            q.scored_g4d = !tr.empty();
            if (!tr.empty()) q.top_g4d = tr.front().second, q.top_g4do = tr.front().first;
            q.rank_g4d = s8a::rank_value(st, ri, s, o, ob, as_cands(tr), nullptr, s8a::M_RULES, filt);
        }
        for (int v = 0; v < 2; ++v) {
            const auto gc = g5_cands(*va[v], s, r, ob);
            q.scored_g5[v] = !gc.empty();
            if (!gc.empty()) q.top_g5[v] = gc.front().score - 2.0, q.top_g5o[v] = gc.front().o;
            q.rank_g5[v] = s8a::rank_value(st, ri, s, o, ob, gc, nullptr, s8a::M_RULES, filt);
        }
        Q.push_back(std::move(q));
    }
    std::snprintf(b, sizeof b, "S11C R1 hidden %zu, reduced store %zu, queries read %zu in %.1f s", hidden.size(), st.size(), Q.size(), now_s() - tq);
    std::cout << b << "\n";
    // G5's per-relation calibration on dev: on only where its dev MRR beats the base rate's (interval above 0, >= 20 queries)
    std::set<string> g5_on[2];
    for (int v = 0; v < 2; ++v) {
        std::map<string, std::pair<vector<double>, vector<double>>> per;
        for (const auto& q : Q)
            if (q.dev) per[q.r].first.push_back(rr(q.rank_g5[v])), per[q.r].second.push_back(rr(q.rank_base));
        for (const auto& [r, p] : per) {
            if (p.first.size() < 20) continue;
            const auto d = boot_diff(p.first, p.second);
            const bool on = d.first > 0;
            if (on) g5_on[v].insert(r);
            std::snprintf(b, sizeof b, "DEV G5 CALIBRATION %s %-13s queries %3zu MRR G5 %.3f base %.3f difference [%+.3f, %+.3f] -> %s", v == 0 ? "ppmi300" : "simvec32",
                          r.c_str(), p.first.size(), s8a::mean(p.first), s8a::mean(p.second), d.first, d.second, on ? "ON" : "off");
            std::cout << b << "\n";
        }
    }
    auto g5_rank = [&](const QueryRow& q, int v) { return g5_on[v].count(q.r) ? q.rank_g5[v] : q.rank_base; };
    // the reads
    auto read_line = [&](const string& tag, const string& what, const vector<double>& h1, const vector<double>& h10, const vector<double>& mrr,
                         const vector<double>& base_mrr, double scored) {
        const auto c1 = s8a::boot_mean(h1), c10 = s8a::boot_mean(h10), cm = s8a::boot_mean(mrr);
        const auto d = boot_diff(mrr, base_mrr);
        std::snprintf(b, sizeof b, "%s R1 RANK %-16s n %4zu Hits@1 %.3f [%.3f, %.3f] Hits@10 %.3f [%.3f, %.3f] MRR %.3f [%.3f, %.3f] | MRR - base %+.3f [%+.3f, %+.3f] | scored %.3f",
                      tag.c_str(), what.c_str(), mrr.size(), s8a::mean(h1), c1.first, c1.second, s8a::mean(h10), c10.first, c10.second, s8a::mean(mrr), cm.first,
                      cm.second, s8a::mean(mrr) - s8a::mean(base_mrr), d.first, d.second, scored);
        std::cout << b << "\n";
        return d;
    };
    int lam = 0, vec = 0;
    std::pair<double, double> test_d4, test_d5;
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
        const string tag = sp ? "TEST" : "DEV";
        auto collect = [&](const std::function<std::size_t(const QueryRow&)>& rk, const std::function<bool(const QueryRow&)>& sc, vector<double>& h1, vector<double>& h10,
                           vector<double>& m, vector<double>& bm, double& scored) {
            double n = 0, k = 0;
            for (const auto& q : Q) {
                if (q.dev != (sp == 0)) continue;
                const std::size_t r = rk(q);
                h1.push_back(r == 1), h10.push_back(r >= 1 && r <= 10), m.push_back(rr(r)), bm.push_back(rr(q.rank_base));
                n += 1, k += sc(q);
            }
            scored = n > 0 ? k / n : 0;
        };
        vector<double> h1, h10, m, bm;
        double scd = 0;
        collect([](const QueryRow& q) { return q.rank_base; }, [](const QueryRow&) { return true; }, h1, h10, m, bm, scd);
        read_line(tag, "base_rate", h1, h10, m, bm, scd);
        h1.clear(), h10.clear(), m.clear(), bm.clear();
        collect([](const QueryRow& q) { return q.rank_g2; }, [](const QueryRow&) { return true; }, h1, h10, m, bm, scd);
        read_line(tag, "G2_rules", h1, h10, m, bm, scd);
        double best = -1;
        for (int l = 0; l < 3; ++l) {
            h1.clear(), h10.clear(), m.clear(), bm.clear();
            collect([l](const QueryRow& q) { return q.rank_g4[l]; }, [l](const QueryRow& q) { return q.scored_g4[l]; }, h1, h10, m, bm, scd);
            char w[48];
            std::snprintf(w, sizeof w, "G4_lambda_%.2f", kLambda[l]);
            const auto d = read_line(tag, w, h1, h10, m, bm, scd);
            if (sp == 0 && s8a::mean(m) > best) best = s8a::mean(m), lam = l;
            if (sp == 1 && l == lam) test_d4 = d;
        }
        best = -1;
        for (int v = 0; v < 2; ++v) {
            h1.clear(), h10.clear(), m.clear(), bm.clear();
            collect([&, v](const QueryRow& q) { return g5_rank(q, v); }, [&, v](const QueryRow& q) { return q.scored_g5[v] && g5_on[v].count(q.r) > 0; }, h1, h10, m, bm, scd);
            const auto d = read_line(tag, v == 0 ? "G5_ppmi300" : "G5_simvec32", h1, h10, m, bm, scd);
            if (sp == 0 && s8a::mean(m) > best) best = s8a::mean(m), vec = v;
            if (sp == 1 && v == vec) test_d5 = d;
        }
        if (sp == 0) {
            std::snprintf(b, sizeof b, "S11C CHOICE (dev MRR): G4 lambda %.2f | G5 vectors %s (on for %zu relations)", kLambda[lam], vec == 0 ? "ppmi300" : "simvec32",
                          g5_on[vec].size());
            std::cout << b << "\n";
        }
    }
    if (g4_date) {  // S11e item 3, dev only: lambda 0.25 everywhere against lambda_date = 0, overall, on dates, on the rest
        for (int part = 0; part < 3; ++part) {
            vector<double> m25, md, mb;
            double n25 = 0, nd = 0, nn = 0;
            for (const auto& q : Q) {
                if (!q.dev) continue;
                const int ri = rel_index(q.r);
                const bool date = ri >= 0 && is_datey(ri);
                if ((part == 1 && !date) || (part == 2 && date)) continue;
                m25.push_back(rr(q.rank_g4[1])), md.push_back(rr(q.rank_g4d)), mb.push_back(rr(q.rank_base));
                n25 += q.scored_g4[1], nd += q.scored_g4d, nn += 1;
            }
            const auto d = boot_diff(md, m25), d25 = boot_diff(m25, mb), dd = boot_diff(md, mb);
            const char* const pn[3] = {"all", "date relations", "other relations"};
            std::snprintf(b, sizeof b,
                          "DEV S11E G4 DATES %-15s n %4zu | MRR base %.3f, lambda 0.25 %.3f (scored %.3f), lambda_date 0 %.3f (scored %.3f) | lambda_date 0 - lambda 0.25 %+.3f "
                          "[%+.3f, %+.3f] | lambda 0.25 - base %+.3f [%+.3f, %+.3f] | lambda_date 0 - base %+.3f [%+.3f, %+.3f]",
                          pn[part], m25.size(), s8a::mean(mb), s8a::mean(m25), nn > 0 ? n25 / nn : 0.0, s8a::mean(md), nn > 0 ? nd / nn : 0.0,
                          s8a::mean(md) - s8a::mean(m25), d.first, d.second, s8a::mean(m25) - s8a::mean(mb), d25.first, d25.second, s8a::mean(md) - s8a::mean(mb),
                          dd.first, dd.second);
            std::cout << b << "\n";
            if (part == 0) {
                std::snprintf(b, sizeof b, "DEV S11E RULE 3 (overall MRR difference interval lower end >= -0.005): [%+.3f, %+.3f] -> %s", d.first, d.second,
                              d.first >= -0.005 ? "ADOPT lambda_date = 0" : "KEEP lambda 0.25 for dates");
                std::cout << b << "\n";
            }
        }
    }
    if (read_test) {
        std::snprintf(b, sizeof b, "TEST RULE (MRR beats the base rate, interval above 0): G4 [%+.3f, %+.3f] -> %s | G5 [%+.3f, %+.3f] -> %s", test_d4.first, test_d4.second,
                      test_d4.first > 0 ? "EARNS ITS PLACE" : "DOES NOT", test_d5.first, test_d5.second, test_d5.first > 0 ? "EARNS ITS PLACE" : "DOES NOT");
        std::cout << b << "\n";
    }
    {
        std::ofstream f(out + ".r1_queries.tsv");
        f << "split\tsubject\trelation\tobject\trank_base\trank_g2\trank_g4\trank_g5\ttop_g4\ttop_g5\n";
        for (const auto& q : Q)
            f << (q.dev ? "dev" : "test") << '\t' << q.s << '\t' << q.r << '\t' << q.o << '\t' << q.rank_base << '\t' << q.rank_g2 << '\t' << q.rank_g4[lam] << '\t'
              << g5_rank(q, vec) << '\t' << q.top_g4o[lam] << '\t' << q.top_g5o[vec] << '\n';
    }
    // ---- read 2: S15's temporal holdout on store T (b and c facts), as S8a's R2
    {
        const ecs::TripleStore T = s8a::make_store(S.tby);
        const ecs::RuleIndex ixT(T);
        const ecs::RuleSet rsT(ixT, ecs::mine_rules(ixT), true);
        const auto objsT = s8a::objects_by_rel(T, S.tby);
        const an::VectorAnalogy vaT(T, wv[vec]);
        for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
            const string tag = string(sp ? "TEST" : "DEV") + " R2";
            const auto rows = s8a::load_s15(s15dir + (sp ? "/temporal_holdout_test.tsv" : "/temporal_holdout_dev.tsv"));
            std::map<string, vector<const s8a::S15Row*>> slot;
            for (const auto& R : rows) slot[R.s + '\x1f' + R.r].push_back(&R);
            s8a::RankAcc acc[4][2];  // base, G2, G4, G5 x b, c
            s8a::RankAcc accd[2];    // S11e item 3: G4 with lambda_date = 0
            for (const auto& R : rows) {
                if (R.label != 1 || (R.cat != "b" && R.cat != "c")) continue;
                const int ri = rel_index(R.r);
                if (ri < 0 || T.facts_of(R.s).empty() || s8a::holds_value(ixT, R.s, R.r, R.v)) continue;
                const auto& sl = slot[R.s + '\x1f' + R.r];
                auto filt = [&](const string& c) {
                    if (s8a::holds_value(ixT, R.s, R.r, c)) return true;
                    for (const auto* o2 : sl)
                        if (o2->label == 1 && s8a::same_value(ri, o2->v, c)) return true;
                    return false;
                };
                const auto it = objsT.find(R.r);
                const vector<string>& ob = it == objsT.end() ? none : it->second;
                const int c = R.cat == "b" ? 0 : 1;
                s8a::rank_add(acc[0][c], s8a::rank_value(T, ri, R.s, R.v, ob, {}, nullptr, s8a::M_BASE, filt), false);
                s8a::rank_add(acc[1][c], s8a::rank_value(T, ri, R.s, R.v, ob, rsT.candidates(R.s, R.r), nullptr, s8a::M_RULES, filt), false);
                const auto tr = an::transfer(T, an::find_analogs(T, R.s, R.r), R.s, R.r, kLambda[lam]);
                s8a::rank_add(acc[2][c], s8a::rank_value(T, ri, R.s, R.v, ob, as_cands(tr), nullptr, s8a::M_RULES, filt), !tr.empty());
                const auto gc = g5_on[vec].count(R.r) ? g5_cands(vaT, R.s, R.r, ob) : vector<ecs::RuleSet::Cand>{};
                s8a::rank_add(acc[3][c], s8a::rank_value(T, ri, R.s, R.v, ob, gc, nullptr, s8a::M_RULES, filt), !gc.empty());
                if (g4_date) {
                    const auto trd = an::transfer(T, an::find_analogs(T, R.s, R.r), R.s, R.r, g4_lambda(R.r, kLambda[lam], 0.0));
                    s8a::rank_add(accd[c], s8a::rank_value(T, ri, R.s, R.v, ob, as_cands(trd), nullptr, s8a::M_RULES, filt), !trd.empty());
                }
            }
            const char* const nm[4] = {"base_rate", "G2_rules", "G4", "G5"};
            for (int g = 0; g < 4; ++g) {
                s8a::print_rank(tag + " RANK b(unstated) " + nm[g], acc[g][0]);
                s8a::print_rank(tag + " RANK c(future) " + nm[g], acc[g][1]);
            }
            if (g4_date) {
                s8a::print_rank(tag + " RANK b(unstated) G4_lambda_date_0 (S11e)", accd[0]);
                s8a::print_rank(tag + " RANK c(future) G4_lambda_date_0 (S11e)", accd[1]);
            }
        }
    }
    // ---- pools: the top candidate per slot with the pool's slots hidden
    if (fresh_dev || fresh_test) {
        // priors: isotonic from score to P(true) on read 1's dev hidden slots (truth: store K holds the top candidate)
        ecs::IsotonicCalibrator cal[2], cal_d;
        vector<double> pd;  // S11e item 3: the variant's top scores and truth on read 1's dev hidden slots
        vector<int> yd;
        {
            vector<double> p[2];
            vector<int> y[2];
            std::set<std::pair<string, string>> seen;
            for (const auto& q : Q) {
                if (!q.dev || !seen.insert({q.s, q.r}).second) continue;
                const int ri = rel_index(q.r);
                auto holds = [&](const string& o) {
                    for (const auto& f : K.facts_of(q.s))
                        if (f.r == q.r && s8a::same_value(ri, f.o, o)) return true;
                    return false;
                };
                if (q.scored_g4[lam]) p[0].push_back(q.top_g4[lam]), y[0].push_back(holds(q.top_g4o[lam]));
                if (q.scored_g5[vec] && g5_on[vec].count(q.r)) p[1].push_back(q.top_g5[vec]), y[1].push_back(holds(q.top_g5o[vec]));
                if (g4_date && q.scored_g4d) pd.push_back(q.top_g4d), yd.push_back(holds(q.top_g4do));
            }
            for (int g = 0; g < 2; ++g) cal[g].fit(p[g], y[g]);
            if (g4_date) cal_d.fit(pd, yd);
        }
        const auto slots = pool_slots(s15dir, fresh_test);
        Facts fred;
        std::size_t held = 0;
        for (const auto& [s, fs] : S.kby)
            for (const auto& [r, o] : fs) {
                if (slots.count({s, r})) ++held;
                else fred[s].push_back({r, o});
            }
        const ecs::TripleStore fst = s8a::make_store(fred);
        const auto fobjs = s8a::objects_by_rel(fst, fred);
        const an::VectorAnalogy fva(fst, wv[vec]);
        vector<PoolItem> pool[2], poold;
        std::size_t in_store = 0;
        for (const auto& [s, r] : slots) {
            if (fst.facts_of(s).empty()) continue;
            ++in_store;
            if (g4_date) {  // S11e item 3: G4 with lambda_date = 0
                const auto trd = an::transfer(fst, an::find_analogs(fst, s, r), s, r, g4_lambda(r, kLambda[lam], 0.0));
                if (!trd.empty()) {
                    PoolItem x;
                    x.s = s, x.r = r, x.o = trd.front().first, x.gen = "G4", x.score = trd.front().second;
                    x.prior = std::min(0.98, std::max(0.02, cal_d.predict(x.score)));
                    x.explain = "analog transfer (no carried dates)";
                    poold.push_back(x);
                }
            }
            const auto tr = an::transfer(fst, an::find_analogs(fst, s, r), s, r, kLambda[lam]);
            if (!tr.empty()) {
                PoolItem x;
                x.s = s, x.r = r, x.o = tr.front().first, x.gen = "G4", x.score = tr.front().second;
                x.prior = std::min(0.98, std::max(0.02, cal[0].predict(x.score)));
                x.explain = "analog transfer";
                pool[0].push_back(x);
            }
            if (g5_on[vec].count(r)) {
                const auto it = fobjs.find(r);
                const auto gc = g5_cands(fva, s, r, it == fobjs.end() ? none : it->second);
                if (!gc.empty()) {
                    PoolItem x;
                    x.s = s, x.r = r, x.o = gc.front().o, x.gen = "G5", x.score = gc.front().score - 2.0;
                    x.prior = std::min(0.98, std::max(0.02, cal[1].predict(x.score)));
                    x.explain = "vector offset";
                    pool[1].push_back(x);
                }
            }
        }
        const string ptag = fresh_test ? "S11C POOL test (S15 test, bit 17 = 0)" : "S11C POOL dev (S15 dev, bit 17 = 1)";
        std::snprintf(b, sizeof b, "%s slots %zu | store facts hidden %zu | slots whose subject is in the store %zu | G4 hypotheses %zu | G5 hypotheses %zu", ptag.c_str(),
                      slots.size(), held, in_store, pool[0].size(), pool[1].size());
        std::cout << b << "\n";
        for (int g = 0; g < 2; ++g) {
            const string gn = g == 0 ? "g4" : "g5";
            pool_loop(pool[g], fst, C, out + ".fresh_" + gn + ".prereg.tsv");
            pool_states(ptag + (g == 0 ? " G4" : " G5"), pool[g]);
            write_pool(out + ".fresh_" + gn + ".hyps.tsv", pool[g]);
        }
        if (g4_date) {
            std::map<string, std::size_t> by;
            for (const auto& x : poold) ++by[x.r];
            string l;
            for (const auto& [r, n] : by) l += " " + r + " " + std::to_string(n);
            std::snprintf(b, sizeof b, "%s G4 (S11e, lambda_date 0) hypotheses %zu by relation:%s", ptag.c_str(), poold.size(), l.c_str());
            std::cout << b << "\n";
            pool_loop(poold, fst, C, out + ".fresh_g4d.prereg.tsv");
            pool_states(ptag + " G4 (S11e, lambda_date 0)", poold);
            write_pool(out + ".fresh_g4d.hyps.tsv", poold);
        }
    }
    std::snprintf(b, sizeof b, "S11C --gen-analogy done in %.1f s", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

// ============================================================ --pool-grade: a pool after S8d's source stage
int run_grade(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: pattern_learn --pool-grade s15|wd FILE HALF HYPS SRC_FRESH\n");
        return 2;
    }
    const string kind = argv[2], file = argv[3];
    const int half = std::atoi(argv[4]);
    s8c::Grader G;
    s7::WdObjs wd;
    if (kind == "s15") {
        for (const auto& R : s8c::load_s15(file))
            if (half < 0 || half17(R.s) == half) G.add(R);
    } else if (kind == "wd") {
        wd = s7::load_wd_objs(file);
    } else {
        std::fprintf(stderr, "grader must be s15 or wd\n");
        return 2;
    }
    // the pool's rows and the source stage's rows, in the same order
    auto rows_of = [](const string& path) {
        vector<vector<string>> out;
        std::ifstream f(path);
        string l;
        std::getline(f, l);
        out.push_back(split(l, '\t'));
        while (std::getline(f, l)) out.push_back(split(l, '\t'));
        return out;
    };
    const auto H = rows_of(argv[5]), Z = rows_of(argv[6]);
    if (H.empty() || Z.empty()) return 1;
    auto col = [](const vector<string>& h, const char* n) { return static_cast<std::size_t>(std::find(h.begin(), h.end(), string(n)) - h.begin()); };
    const std::size_t hs = col(H[0], "subject"), hr = col(H[0], "relation"), ho = col(H[0], "object"), hg = col(H[0], "generator"), hst = col(H[0], "state");
    const std::size_t zs = col(Z[0], "subject"), zr = col(Z[0], "relation"), zo = col(Z[0], "object"), zsup = col(Z[0], "supported"), zref = col(Z[0], "refuted"),
                      zby = col(Z[0], "by");
    struct Acc {
        double n = 0, g = 0, t = 0, sup = 0, sg = 0, st = 0, ref = 0, rg = 0, rf = 0, lsup = 0, lsg = 0, lst = 0;
    };
    std::map<string, Acc> by_gen, by_grp, by_rel, by_src;
    Acc all;
    std::set<string> seen;
    std::size_t mismatched = 0, dup = 0;
    for (std::size_t i = 1, j = 1; i < H.size() && j < Z.size(); ++i, ++j) {
        const auto& h = H[i];
        const auto& z = Z[j];
        if (h.size() <= std::max({hs, hr, ho}) || z.size() <= std::max({zs, zr, zo, zsup})) continue;
        if (h[hs] != z[zs] || h[hr] != z[zr] || h[ho] != z[zo]) {
            ++mismatched;
            continue;
        }
        const string k = cypha::eval::sources::norm_subject(h[hs]) + '\x1f' + h[hr] + '\x1f' + cypha::eval::sources::norm_name(h[ho]);
        if (!seen.insert(k).second) {
            ++dup;
            continue;
        }
        int g = -1;
        if (kind == "s15") g = G.grade(h[hs], h[hr], cypha::eval::sources::make_value(h[ho]));
        else g = wd_grade(wd, {h[hs], h[hr], h[ho]});
        const bool sup = z[zsup] == "1", ref = z[zref] == "1";
        const bool lsup = hst < h.size() && (h[hst] == "supported" || h[hst] == "Supported");
        const string gen = hg < h.size() ? h[hg] : "?";
        auto add = [&](Acc& a) {
            a.n += 1;
            if (g >= 0) a.g += 1, a.t += g == 1;
            if (sup) a.sup += 1, a.sg += g >= 0, a.st += g == 1;
            if (ref) a.ref += 1, a.rg += g >= 0, a.rf += g == 0;
            if (lsup) a.lsup += 1, a.lsg += g >= 0, a.lst += g == 1;
        };
        add(all), add(by_gen[gen]), add(by_grp[s8c::group_of(h[hr])]), add(by_rel[h[hr]]);
        if (sup) add(by_src[zby < z.size() ? z[zby] : "?"]);
    }
    char b[768];
    auto line = [&](const string& what, const Acc& a) {
        std::snprintf(b, sizeof b, "POOL-GRADE %-28s hyps %5.0f graded %5.0f (true %.3f) | promoted (supported after sources) %4.0f, graded %3.0f, precision %s | "
                                   "refuted %4.0f, graded %4.0f, false %s | supported by the generator loop %4.0f, graded %3.0f, precision %s",
                      what.c_str(), a.n, a.g, a.g > 0 ? a.t / a.g : 0.0, a.sup, a.sg, wilson(a.st, a.sg).c_str(), a.ref, a.rg, wilson(a.rf, a.rg).c_str(), a.lsup,
                      a.lsg, wilson(a.lst, a.lsg).c_str());
        std::cout << b << "\n";
    };
    line("all (each distinct once)", all);
    for (const auto& [k, a] : by_gen) line("generator " + k, a);
    for (const auto& [k, a] : by_grp) line("group " + k, a);
    for (const auto& [k, a] : by_src) line("positive from " + k, a);
    for (const auto& [k, a] : by_rel)
        if (a.sup > 0) line("relation " + k, a);
    std::snprintf(b, sizeof b, "POOL-GRADE grader %s %s half %d | rows that did not line up %zu | duplicates %zu", kind.c_str(), file.c_str(), half, mismatched, dup);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s11c

// ============================================================ S12a: G1 deep deduction, G15 numeric laws, G20 the wild tier
// (docs/reports/S12A_DEDUCTION_LAWS_REFINEMENT.md; native/include/cypha/eval/gen_laws.hpp, gen_deep.hpp)
// pattern_learn --gen-laws ENWIK8 SEEDS.tsv FACTS_F2.tsv OUT_PREFIX [--read-test] [--power-min-exp] [--order-relaxed M]
//                [--s12c-dev] [--s12c-fresh write|read]
//   the planted-law gate (dev replicates; the test replicates once, with --read-test) and G15's real read (the country
//   and place infoboxes of the 2006 text; persons' birth and death years in stores K and T). S12c: --power-min-exp and
//   --order-relaxed M change G15's checks (items 1, 2); --s12c-dev runs the dev grid and its registered choice;
//   --s12c-fresh write writes the fresh planted tables (seeds S12C|test|…) unsearched, read searches them once.
// pattern_learn --gen-deep ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test]
//                [--fresh-dev | --fresh-test] [--s12c] [--g1-prior chain|cap3|depth]
//   G1 by depth (S8a's R1 random hide of store K; store T on S15), and the pools of G1 and G20 with the prospective log.
//   S12c: --s12c compares G1's priors by depth on R1's dev facts (cross-fitted); --g1-prior sets the pool's prior.
//   G19 (refinement) was removed by S12c's registered rule (code in git at main's 0ce072f). Hypotheses for Cypha's own
//   store only: it answers nothing.
namespace s12a {
namespace lw = cypha::eval::laws;
namespace dp = cypha::eval::deep;
using Facts = s8a::Facts;
using s11c::wilson;

// ---- numbers in infobox values: "$21.5 billion", "$316.2&amp;nbsp;[[1 E9|billion]]", "647,500", "0.780"
bool parse_number(string v, double& out) {
    for (;;) {
        const auto a = v.find("&lt;!--");
        if (a == string::npos) break;
        const auto b = v.find("--&gt;", a);
        v.erase(a, b == string::npos ? string::npos : b + 6 - a);
    }
    for (const string& e : {string("&amp;nbsp;"), string("&nbsp;")})
        for (std::size_t p; (p = v.find(e)) != string::npos;) v.replace(p, e.size(), " ");
    string w;
    for (std::size_t i = 0; i < v.size();) {
        if (v.compare(i, 2, "[[") == 0) {
            const auto e = v.find("]]", i);
            if (e == string::npos) break;
            const string in = v.substr(i + 2, e - i - 2);
            const auto bar = in.rfind('|');
            w += bar == string::npos ? in : in.substr(bar + 1);
            i = e + 2;
        } else {
            w += v[i++];
        }
    }
    std::size_t i = 0;
    while (i < w.size() && !std::isdigit(static_cast<unsigned char>(w[i]))) ++i;
    if (i == w.size()) return false;
    const bool neg = i > 0 && w[i - 1] == '-';
    string num;
    bool dot = false;
    std::size_t j = i;
    for (; j < w.size(); ++j) {
        const char c = w[j];
        if (std::isdigit(static_cast<unsigned char>(c))) num += c;
        else if (c == ',' && j + 1 < w.size() && std::isdigit(static_cast<unsigned char>(w[j + 1]))) continue;
        else if (c == '.' && !dot && j + 1 < w.size() && std::isdigit(static_cast<unsigned char>(w[j + 1]))) dot = true, num += c;
        else break;
    }
    double x = std::atof(num.c_str());
    if (neg) x = -x;
    const string rest = lower(w.substr(j, 24));
    if (rest.find("trillion") != string::npos) x *= 1e12;
    else if (rest.find("billion") != string::npos) x *= 1e9;
    else if (rest.find("million") != string::npos) x *= 1e6;
    out = x;
    return true;
}

// ---- the template fields of a page: "| key = value" or "key = value |", the first occurrence of each key (lower case)
std::map<string, string> page_fields(const string& data, const PageRef& pg) {
    std::map<string, string> f;
    std::size_t p = pg.tb;
    while (p < pg.te) {
        std::size_t e = data.find('\n', p);
        if (e == string::npos || e > pg.te) e = pg.te;
        const string l = data.substr(p, e - p);
        p = e + 1;
        std::size_t a = 0;
        while (a < l.size() && (l[a] == ' ' || l[a] == '|')) ++a;
        const std::size_t eq = l.find('=', a);
        if (eq == string::npos) continue;
        string k = trim(l.substr(a, eq - a));
        if (k.empty() || k.size() > 40) continue;
        bool ok = true;
        for (unsigned char c : k) ok = ok && (std::isalnum(c) || c == '_' || c >= 0x80);
        if (!ok) continue;
        string val = l.substr(eq + 1);
        while (!val.empty() && (val.back() == ' ' || val.back() == '|' || val.back() == '\r')) val.pop_back();
        k = lower(k);
        if (!f.count(k)) f[k] = trim(val);
    }
    return f;
}

// ---- G15's real tables
struct Col {
    const char* field;
    const char* name;
    lw::Scale scale;
};
lw::NumTable infobox_table(const s8b::Corpus& C, const string& type, const vector<Col>& cols, const std::function<bool(const std::map<string, string>&)>& member) {
    lw::NumTable t;
    t.type = type;
    for (const Col& c : cols) t.add_attr(c.name, c.scale);
    for (const PageRef& pg : C.pages) {
        const auto f = page_fields(C.data, pg);
        if (!member(f)) continue;
        const std::size_t i = t.add_entity(pg.title);
        for (std::size_t a = 0; a < cols.size(); ++a) {
            const auto it = f.find(cols[a].field);
            double x = 0;
            if (it != f.end() && parse_number(it->second, x)) t.v[a][i] = x;
        }
    }
    return t;
}
// birth and death years of a store's subjects (the first parsable birth_* and death_* fact in the subject's sorted facts)
vector<lw::PersonYears> person_years(const Facts& by, int split) {  // split: -1 all, 0 dev (bit 11 = 0), 1 test
    vector<lw::PersonYears> out;
    for (const auto& [s, fs] : by) {
        if (split >= 0 && s11c::dev_subject(s) != (split == 0)) continue;
        bool hb = false, hd = false;
        lw::PersonYears p;
        p.name = s;
        for (const auto& [r, o] : fs) {
            int y = 0;
            if (!hb && (r == "birth_year" || r == "birth_date") && s11c::year_of(o, y)) hb = true, p.birth = y;
            if (!hd && (r == "death_year" || r == "death_date") && s11c::year_of(o, y)) hd = true, p.death = y;
        }
        if (hb && hd) out.push_back(p);
    }
    return out;
}
lw::NumTable person_table(const Facts& by, const string& type) {
    lw::NumTable t;
    t.type = type;
    const std::size_t b = t.add_attr("birth", lw::Scale::Interval), d = t.add_attr("death", lw::Scale::Interval);
    for (const auto& [s, fs] : by) {
        bool hb = false, hd = false;
        double yb = 0, yd = 0;
        for (const auto& [r, o] : fs) {
            int y = 0;
            if (!hb && (r == "birth_year" || r == "birth_date") && s11c::year_of(o, y)) hb = true, yb = y;
            if (!hd && (r == "death_year" || r == "death_date") && s11c::year_of(o, y)) hd = true, yd = y;
        }
        if (!hb && !hd) continue;
        const std::size_t i = t.add_entity(s);
        if (hb) t.v[b][i] = yb;
        if (hd) t.v[d][i] = yd;
    }
    return t;
}

void print_laws(const lw::NumTable& t, const vector<lw::Law>& laws, const std::function<string(const lw::Law&)>& tag) {
    char b[640];
    std::size_t searched = laws.size(), screened = 0, kept = 0;
    std::map<std::size_t, std::size_t> with;
    for (std::size_t a = 0; a < t.attrs.size(); ++a)
        for (std::size_t i = 0; i < t.entities.size(); ++i) with[a] += t.has(a, i);
    string attrs;
    for (std::size_t a = 0; a < t.attrs.size(); ++a) attrs += " " + t.attrs[a] + "(" + std::to_string(with[a]) + ")";
    for (const auto& L : laws) screened += L.screen, kept += L.kept;
    std::snprintf(b, sizeof b, "S12A LAWS %s | entities %zu | attributes:%s | formulas searched %zu, nearly constant %zu, kept %zu", t.type.c_str(),
                  t.entities.size(), attrs.c_str(), searched, screened, kept);
    std::cout << b << "\n";
    for (const auto& L : laws) {
        if (!L.kept && !(L.screen && L.kind != lw::LawKind::Order)) continue;
        std::snprintf(b, sizeof b, "  %s %-11s %-60s n %4zu | constant %.6g | %s %.4f (classic CV %.3f) inliers %.3f | p %.2e BH %d | held out %zu within %.3f | %s%s",
                      L.kept ? "KEPT" : "    ", lw::law_kind_name(L.kind), L.text(t).c_str(), L.n, L.constant,
                      L.kind == lw::LawKind::Power ? "robust CV" : L.kind == lw::LawKind::Diff ? "rel spread" : "share", L.spread, L.classic_cv, L.inlier,
                      L.p_value, L.bh ? 1 : 0, L.heldout_n, L.heldout_share, L.redundant ? "redundant " : "", tag(L).c_str());
        std::cout << b << "\n";
    }
}

// The expected laws (registered): attribute set -> exponent vector, up to sign.
struct Expected {
    lw::LawKind kind;
    vector<std::pair<string, int>> terms;  // Power: name -> exponent; Diff/Order: {X, -1}, {Y, +1} meaning Y - X or X <= Y
    string what;
};
bool matches(const lw::NumTable& t, const lw::Law& L, const Expected& e) {
    if (L.kind != e.kind || L.attr.size() != e.terms.size()) return false;
    if (L.kind != lw::LawKind::Power) return t.attrs[L.attr[0]] == e.terms[0].first && t.attrs[L.attr[1]] == e.terms[1].first;
    std::map<string, int> got;
    for (std::size_t k = 0; k < L.attr.size(); ++k) got[t.attrs[L.attr[k]]] = L.exp[k];
    bool same = true, neg = true;
    for (const auto& [n, x] : e.terms) {
        const auto it = got.find(n);
        if (it == got.end()) return false;
        same = same && it->second == x, neg = neg && it->second == -x;
    }
    return same || neg;
}

// ============================================================ S12c: the dev grid (items 1 and 2) and the fresh gate read
const lw::PlantKind kAllKinds[] = {lw::PlantKind::Ratio, lw::PlantKind::Product, lw::PlantKind::Power, lw::PlantKind::Scale, lw::PlantKind::Offset, lw::PlantKind::Order};
const double kLevels[] = {0.0, 0.02, 0.05, 0.10, 0.20};

struct GridRes {
    std::map<string, std::array<double, 3>> found;  // kind -> found at sigma <= 0.05 (of 60), at 0.10 (of 20), at 0.20 (of 20)
    std::set<string> false_tables;                  // "kind|sigma|rep|planted" or "|null" with a false law
    double gf = 0, gn = 0, nf = 0, nn = 0;          // sigma <= 0.05: found / tables; null tables with a false law / null tables
};

GridRes s12c_run(const vector<lw::PersonYears>& persons, const string& tag, const lw::LawConfig& cfg) {
    GridRes R;
    for (const auto kind : kAllKinds)
        for (const double sig : kLevels)
            for (int rep = 0; rep < 20; ++rep) {
                const std::uint64_t seed = fnv(tag + "|" + lw::plant_kind_name(kind) + "|" + std::to_string(sig) + "|" + std::to_string(rep));
                const lw::Planted P = lw::plant(kind, sig, seed, persons);
                const auto L = lw::search_laws(P.table, cfg), Ln = lw::search_laws(P.null_table, cfg);
                const bool found = lw::planted_found(P, L);
                const string key = string(lw::plant_kind_name(kind)) + "|" + std::to_string(sig) + "|" + std::to_string(rep);
                if (lw::false_laws(P, L, false) > 0) R.false_tables.insert(key + "|planted");
                const bool nfl = lw::false_laws(P, Ln, true) > 0;
                if (nfl) R.false_tables.insert(key + "|null");
                auto& f = R.found[lw::plant_kind_name(kind)];
                if (sig <= 0.05 + 1e-12) f[0] += found, R.gf += found, R.gn += 1, R.nf += nfl, R.nn += 1;
                else if (sig < 0.15) f[1] += found;
                else f[2] += found;
            }
    return R;
}

int s12c_dev_grid(const s11c::Stores& S, const string& out) {
    char b[768];
    const auto persons = person_years(S.kby, 0);
    struct C {
        string name;
        lw::LawConfig cfg;
    };
    auto mk = [](bool power, double margin) {
        lw::LawConfig c;
        c.power_min_exp = power;
        if (margin >= 0) c.order_share = 0.90, c.order_heldout_viol = 0.20, c.order_margin = margin;
        return c;
    };
    const vector<C> grid = {{"A", mk(false, -1)}, {"P", mk(true, -1)}, {"O1", mk(false, 0.20)}, {"O2", mk(false, 0.30)}, {"P+O1", mk(true, 0.20)}, {"P+O2", mk(true, 0.30)}};
    std::map<string, GridRes> res;
    std::ofstream tsv(out + ".s12c_dev.tsv");
    tsv << "config\tkind\tfound_le_005\tfound_010\tfound_020\n";
    for (const auto& c : grid) {
        res[c.name] = s12c_run(persons, "S12A|dev", c.cfg);
        const auto& R = res[c.name];
        for (const auto kind : kAllKinds) {
            const auto& f = R.found.at(lw::plant_kind_name(kind));
            std::size_t fp = 0, fn = 0;
            for (const auto& k : R.false_tables)
                if (k.rfind(string(lw::plant_kind_name(kind)) + "|", 0) == 0) (k.size() >= 5 && k.compare(k.size() - 5, 5, "|null") == 0 ? fn : fp) += 1;
            std::snprintf(b, sizeof b, "S12C DEV %-5s %-7s: sigma <= 0.05 found %s | sigma 0.10 %2.0f/20 | sigma 0.20 %2.0f/20 | tables with a false law: planted %zu, null %zu", c.name.c_str(),
                          lw::plant_kind_name(kind), wilson(f[0], 60).c_str(), f[1], f[2], fp, fn);
            std::cout << b << "\n";
            tsv << c.name << '\t' << lw::plant_kind_name(kind) << '\t' << f[0] << '\t' << f[1] << '\t' << f[2] << '\n';
        }
        std::snprintf(b, sizeof b, "S12C DEV %-5s pooled (sigma <= 0.05): recall %s | null tables with a false law %s | tables with a false law (all levels, planted and null) %zu", c.name.c_str(),
                      wilson(R.gf, R.gn).c_str(), wilson(R.nf, R.nn).c_str(), R.false_tables.size());
        std::cout << b << "\n";
    }
    // the registered choice
    const auto& A = res["A"];
    const auto& P = res["P"];
    bool subset = true;
    for (const auto& k : P.false_tables) subset = subset && A.false_tables.count(k);
    const bool item1 = P.found.at("POWER")[0] > A.found.at("POWER")[0] && subset;
    std::snprintf(b, sizeof b, "S12C DEV CHOICE item 1 (power spread on the predicted attribute): POWER sigma <= 0.05 %.0f/60 -> %.0f/60, no new false law %s -> %s",
                  A.found.at("POWER")[0], P.found.at("POWER")[0], subset ? "yes" : "no", item1 ? "ADOPTED" : "not adopted");
    std::cout << b << "\n";
    string pick = "none";
    double best = -1;
    for (const string o : {"O1", "O2"}) {
        const auto& R = res[item1 ? "P+" + o : o];
        if (!R.false_tables.empty()) continue;
        const double rec = R.found.at("ORDER")[0];
        if (rec > best || (rec == best && o == "O2")) best = rec, pick = o;
    }
    std::snprintf(b, sizeof b, "S12C DEV CHOICE item 2 (orders at 90%% with a margin over independence): O0 ORDER sigma <= 0.05 %.0f/60, O1 %.0f/60 (false tables %zu), O2 %.0f/60 (false tables %zu) -> %s",
                  res[item1 ? "P" : "A"].found.at("ORDER")[0], res[item1 ? "P+O1" : "O1"].found.at("ORDER")[0], res[item1 ? "P+O1" : "O1"].false_tables.size(),
                  res[item1 ? "P+O2" : "O2"].found.at("ORDER")[0], res[item1 ? "P+O2" : "O2"].false_tables.size(), pick == "none" ? "orders stay as now" : pick.c_str());
    std::cout << b << "\n";
    return 0;
}

int s12c_fresh(const s11c::Stores& S, const string& out, const lw::LawConfig& cfg, bool do_read) {
    char b[768];
    const auto persons = person_years(S.kby, 1);
    std::ofstream tab(out + ".s12c_fresh_tables.tsv");
    tab << "kind\tsigma\trep\tseed\tc\ttable\tentity\tvalues\n";
    std::ofstream res;
    if (do_read) res.open(out + ".s12c_fresh.tsv"), res << "kind\tsigma\trep\tseed\tfound\tfalse_laws\tnull_false_laws\n";
    std::map<string, std::array<double, 5>> found;  // kind -> found per level
    double gf = 0, gn = 0, nf = 0, nn = 0, pf = 0;
    std::size_t tables = 0;
    for (const auto kind : kAllKinds)
        for (std::size_t li = 0; li < 5; ++li)
            for (int rep = 0; rep < 20; ++rep) {
                const double sig = kLevels[li];
                const std::uint64_t seed = fnv(string("S12C|test|") + lw::plant_kind_name(kind) + "|" + std::to_string(sig) + "|" + std::to_string(rep));
                const lw::Planted P = lw::plant(kind, sig, seed, persons);
                for (int w = 0; w < 2; ++w) {
                    const lw::NumTable& T = w == 0 ? P.table : P.null_table;
                    ++tables;
                    for (std::size_t e = 0; e < T.entities.size(); ++e) {
                        tab << lw::plant_kind_name(kind) << '\t' << sig << '\t' << rep << '\t' << seed << '\t';
                        char cb[40];
                        std::snprintf(cb, sizeof cb, "%.17g", P.c);
                        tab << cb << '\t' << (w == 0 ? "planted" : "null") << '\t' << T.entities[e] << '\t';
                        for (std::size_t a = 0; a < T.attrs.size(); ++a) {
                            std::snprintf(cb, sizeof cb, "%.17g", T.v[a][e]);
                            tab << (a ? "," : "") << T.attrs[a] << '=' << cb;
                        }
                        tab << '\n';
                    }
                }
                if (!do_read) continue;
                const auto L = lw::search_laws(P.table, cfg), Ln = lw::search_laws(P.null_table, cfg);
                const bool f = lw::planted_found(P, L);
                const std::size_t fl = lw::false_laws(P, L, false), nfl = lw::false_laws(P, Ln, true);
                found[lw::plant_kind_name(kind)][li] += f;
                if (sig <= 0.05 + 1e-12) gf += f, gn += 1, nf += nfl > 0, nn += 1, pf += fl > 0;
                res << lw::plant_kind_name(kind) << '\t' << sig << '\t' << rep << '\t' << seed << '\t' << f << '\t' << fl << '\t' << nfl << '\n';
            }
    tab.close();
    std::snprintf(b, sizeof b, "S12C FRESH tables written to %s: %zu tables (planted and null), seeds S12C|test|kind|sigma|rep, store K's test persons %zu", (out + ".s12c_fresh_tables.tsv").c_str(),
                  tables, persons.size());
    std::cout << b << "\n";
    if (!do_read) return 0;
    std::snprintf(b, sizeof b, "S12C FRESH configuration: power_min_exp %d, order share %.2f, held-out violations %.2f, margin %.2f", cfg.power_min_exp ? 1 : 0, cfg.order_share,
                  cfg.order_heldout_viol, cfg.order_margin);
    std::cout << b << "\n";
    for (const auto kind : kAllKinds) {
        const auto& f = found[lw::plant_kind_name(kind)];
        std::snprintf(b, sizeof b, "S12C FRESH %-7s: sigma 0 %2.0f/20 | 0.02 %2.0f/20 | 0.05 %2.0f/20 | 0.10 %2.0f/20 | 0.20 %2.0f/20 | sigma <= 0.05 %s", lw::plant_kind_name(kind), f[0], f[1], f[2],
                      f[3], f[4], wilson(f[0] + f[1] + f[2], 60).c_str());
        std::cout << b << "\n";
    }
    std::snprintf(b, sizeof b, "S12C FRESH pooled (sigma <= 0.05): recall %s | null tables with a false law %s | planted tables with a false law %s", wilson(gf, gn).c_str(),
                  wilson(nf, nn).c_str(), wilson(pf, gn).c_str());
    std::cout << b << "\n";
    const bool pass = gn > 0 && gf / gn >= 0.90 && nn > 0 && nf / nn <= 0.05;
    std::snprintf(b, sizeof b, "S12C GATE (fresh test tables, read once: recall >= 0.90 and false-law rate <= 0.05, point estimates): recall %.3f, false-law rate %.3f -> %s", gf / gn, nf / nn,
                  pass ? "MET" : "NOT MET");
    std::cout << b << "\n";
    return 0;
}

int run_laws(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: pattern_learn --gen-laws ENWIK8 SEEDS.tsv FACTS_F2.tsv OUT_PREFIX [--read-test] [--power-min-exp] [--order-relaxed M] [--s12c-dev] [--s12c-fresh write|read]\n");
        return 2;
    }
    const double t0 = now_s();
    const string out = argv[5];
    bool read_test = false, s12c_dev = false;
    string fresh;
    lw::LawConfig cfg;
    for (int i = 6; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--read-test") read_test = true;
        else if (a == "--power-min-exp") cfg.power_min_exp = true;
        else if (a == "--order-relaxed" && i + 1 < argc) cfg.order_share = 0.90, cfg.order_heldout_viol = 0.20, cfg.order_margin = std::atof(argv[++i]);
        else if (a == "--s12c-dev") s12c_dev = true;
        else if (a == "--s12c-fresh" && i + 1 < argc) fresh = argv[++i];
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (!fresh.empty() && fresh != "write" && fresh != "read") {
        std::fprintf(stderr, "--s12c-fresh takes write or read\n");
        return 2;
    }
    s11c::Stores S;
    if (!s11c::load_stores(argv[3], argv[4], S)) return 1;
    if (s12c_dev) return s12c_dev_grid(S, out);
    if (!fresh.empty()) return s12c_fresh(S, out, cfg, fresh == "read");
    char b[768];
    // ============================================================ the planted-law gate
    const lw::PlantKind kinds[] = {lw::PlantKind::Ratio, lw::PlantKind::Product, lw::PlantKind::Power, lw::PlantKind::Scale, lw::PlantKind::Offset, lw::PlantKind::Order};
    const double levels[] = {0.0, 0.02, 0.05, 0.10, 0.20};
    const int reps = 20;
    std::ofstream tsv(out + ".planted.tsv");
    tsv << "split\tkind\tsigma\trep\tseed\tc\tentities\tfound\tfalse_laws\tnull_false_laws\tkept\tnull_kept\n";
    for (int split = 0; split < (read_test ? 2 : 1); ++split) {
        const string sp = split == 0 ? "dev" : "test";
        const auto persons = person_years(S.kby, split);
        std::snprintf(b, sizeof b, "S12A PLANTED %s: persons with birth and death years (store K, %s subjects) %zu", sp.c_str(), sp.c_str(), persons.size());
        std::cout << b << "\n";
        double gf = 0, gn = 0, nf = 0, nn = 0, inf_ = 0, inn = 0;  // gate: found / tables; null tables with a false law / null tables; independent false
        for (const auto kind : kinds)
            for (const double sig : levels) {
                double f = 0, fl = 0, nfl = 0;
                for (int rep = 0; rep < reps; ++rep) {
                    const std::uint64_t seed = fnv("S12A|" + sp + "|" + lw::plant_kind_name(kind) + "|" + std::to_string(sig) + "|" + std::to_string(rep));
                    const lw::Planted P = lw::plant(kind, sig, seed, persons);
                    const auto L = lw::search_laws(P.table, cfg), Ln = lw::search_laws(P.null_table, cfg);
                    const bool found = lw::planted_found(P, L);
                    const std::size_t fls = lw::false_laws(P, L, false), nfls = lw::false_laws(P, Ln, true);
                    std::size_t k1 = 0, k2 = 0;
                    for (const auto& x : L) k1 += x.kept;
                    for (const auto& x : Ln) k2 += x.kept;
                    f += found, fl += fls > 0, nfl += nfls > 0;
                    tsv << sp << '\t' << lw::plant_kind_name(kind) << '\t' << sig << '\t' << rep << '\t' << seed << '\t' << P.c << '\t' << P.table.entities.size() << '\t'
                        << found << '\t' << fls << '\t' << nfls << '\t' << k1 << '\t' << k2 << '\n';
                    if (sig <= 0.05 + 1e-12) gf += found, gn += 1, nf += nfls > 0, nn += 1, inf_ += fls > 0, inn += 1;
                }
                std::snprintf(b, sizeof b, "S12A PLANTED %s %-7s sigma %.2f: found %s | tables with a false law %s | null tables with a false law %s", sp.c_str(),
                              lw::plant_kind_name(kind), sig, wilson(f, reps).c_str(), wilson(fl, reps).c_str(), wilson(nfl, reps).c_str());
                std::cout << b << "\n";
            }
        std::snprintf(b, sizeof b, "S12A PLANTED %s pooled (sigma <= 0.05): recall %s | null tables with a false law %s | planted tables with a false law %s", sp.c_str(),
                      wilson(gf, gn).c_str(), wilson(nf, nn).c_str(), wilson(inf_, inn).c_str());
        std::cout << b << "\n";
        if (split == 1) {
            const bool pass = gn > 0 && gf / gn >= 0.90 && nn > 0 && nf / nn <= 0.05;
            std::snprintf(b, sizeof b, "S12A GATE (test replicates, read once: recall >= 0.90 and false-law rate <= 0.05, point estimates): recall %.3f, false-law rate %.3f -> %s",
                          gf / gn, nf / nn, pass ? "MET" : "NOT MET");
            std::cout << b << "\n";
        }
    }
    // ============================================================ G15's real read (one run, nothing chosen)
    s8b::Corpus C;
    if (!read_file(argv[2], C.data)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    {
        long tot = 0, red = 0;
        C.pages = list_pages(C.data, tot, red);
    }
    const string a2 = "areami\xc2\xb2", d2 = "population_densitymi\xc2\xb2";
    const vector<Col> country_cols = {{"area", "area_km2", lw::Scale::Ratio},
                                      {a2.c_str(), "area_mi2", lw::Scale::Ratio},
                                      {"population_estimate", "population", lw::Scale::Ratio},
                                      {"population_density", "density_km2", lw::Scale::Ratio},
                                      {d2.c_str(), "density_mi2", lw::Scale::Ratio},
                                      {"gdp_ppp", "gdp_ppp", lw::Scale::Ratio},
                                      {"gdp_ppp_per_capita", "gdp_ppp_per_capita", lw::Scale::Ratio},
                                      {"hdi", "hdi", lw::Scale::Ratio}};
    const vector<Col> place_cols = {{"population_total", "population", lw::Scale::Ratio},
                                    {"area_total", "area_total", lw::Scale::Ratio},
                                    {"area_land", "area_land", lw::Scale::Ratio},
                                    {"area_water", "area_water", lw::Scale::Ratio},
                                    {"population_density", "density", lw::Scale::Ratio},
                                    {"elevation", "elevation", lw::Scale::Interval}};
    const auto country = infobox_table(C, "countries (2006 infoboxes)", country_cols,
                                       [](const std::map<string, string>& f) { return f.count("population_estimate") || f.count("gdp_ppp"); });
    const auto place = infobox_table(C, "places (2006 infoboxes with population_total)", place_cols, [](const std::map<string, string>& f) {
        return f.count("population_total") && !f.count("population_estimate") && !f.count("gdp_ppp");
    });
    const vector<Expected> exp_country = {
        {lw::LawKind::Power, {{"density_km2", 1}, {"area_km2", 1}, {"population", -1}}, "density = population / area"},
        {lw::LawKind::Power, {{"area_km2", 1}, {"area_mi2", -1}}, "area in km2 = 2.59 x area in mi2"},
        {lw::LawKind::Power, {{"density_km2", 1}, {"density_mi2", -1}}, "density per mi2 = 2.59 x density per km2"},
        {lw::LawKind::Power, {{"gdp_ppp", 1}, {"gdp_ppp_per_capita", -1}, {"population", -1}}, "GDP per capita = GDP / population"},
        // the same laws through the other unit (implied by the first three; kept only if the span allows)
        {lw::LawKind::Power, {{"density_mi2", 1}, {"area_mi2", 1}, {"population", -1}}, "density per mi2 = population / area in mi2"},
    };
    const vector<Expected> exp_person = {{lw::LawKind::Order, {{"birth", -1}, {"death", 1}}, "birth <= death"},
                                         {lw::LawKind::Diff, {{"birth", -1}, {"death", 1}}, "death - birth bounded (lifespan)"}};
    auto read_table = [&](const lw::NumTable& t, const vector<Expected>& ex) {
        const auto laws = lw::search_laws(t, cfg);
        std::size_t outside = 0, kept = 0;
        vector<bool> got(ex.size(), false);
        auto tag = [&](const lw::Law& L) {
            for (std::size_t i = 0; i < ex.size(); ++i)
                if (matches(t, L, ex[i])) {
                    if (L.kept) got[i] = true;
                    return "EXPECTED: " + ex[i].what;
                }
            return string(L.kept ? "outside the expected list" : "");
        };
        for (const auto& L : laws)
            if (L.kept) {
                ++kept;
                bool e = false;
                for (const auto& x : ex) e = e || matches(t, L, x);
                outside += !e;
            }
        print_laws(t, laws, tag);
        string g;
        for (std::size_t i = 0; i < ex.size(); ++i) g += string(" | ") + (got[i] ? "kept: " : "not kept: ") + ex[i].what;
        std::snprintf(b, sizeof b, "S12A REAL %s: kept %zu, outside the expected list %zu%s", t.type.c_str(), kept, outside, g.c_str());
        std::cout << b << "\n";
    };
    read_table(country, exp_country);
    read_table(place, {});
    read_table(person_table(S.kby, "persons (store K)"), exp_person);
    read_table(person_table(S.tby, "persons (store T)"), exp_person);
    std::snprintf(b, sizeof b, "S12A --gen-laws done in %.1f s", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

// ============================================================ G1, G20 (G19 removed by S12c's registered rule)
int run_deep(int argc, char** argv) {
    if (argc < 9) {
        std::fprintf(stderr, "usage: pattern_learn --gen-deep ENWIK8 SEEDS.tsv FACTS_F2.tsv WD_VALUES.tsv PATTERNS.tsv S15_DIR OUT_PREFIX [--read-test] [--fresh-dev | --fresh-test] [--s12c] [--g1-prior chain|cap3|depth]\n");
        return 2;
    }
    const double t0 = now_s();
    const string s15dir = argv[7], out = argv[8];
    bool read_test = false, fresh_dev = false, fresh_test = false, s12c = false;
    dp::G1Prior g1mode = dp::G1Prior::Chain;
    for (int i = 9; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--read-test") read_test = true;
        else if (a == "--fresh-dev") fresh_dev = true;
        else if (a == "--fresh-test") fresh_test = true;
        else if (a == "--s12c") s12c = true;
        else if (a == "--g1-prior" && i + 1 < argc) {
            const string m = argv[++i];
            if (m == "chain") g1mode = dp::G1Prior::Chain;
            else if (m == "cap3") g1mode = dp::G1Prior::Cap3;
            else if (m == "depth") g1mode = dp::G1Prior::Depth;
            else {
                std::fprintf(stderr, "--g1-prior takes chain, cap3 or depth\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    s11c::Stores S;
    if (!s11c::load_stores(argv[3], argv[4], S)) return 1;
    char b[768];
    const ecs::TripleStore K = s8a::make_store(S.kby);
    // S15 rows by slot: dev always; test with --read-test (S8a's amendment 1 grader)
    std::map<string, vector<std::pair<string, int>>> s15_slot[2];
    std::map<string, string> s15_cat[2];  // s r value(norm) -> category of the true row
    for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
        for (const auto& R : s8a::load_s15(s15dir + (sp == 0 ? "/temporal_holdout_dev.tsv" : "/temporal_holdout_test.tsv"))) {
            s15_slot[sp][R.s + '\x1f' + R.r].push_back({R.v, R.label});
            if (R.label == 1) s15_cat[sp][R.s + '\x1f' + R.r + '\x1f' + R.v] = R.cat;
        }
    auto k_holds = [&](const string& s, const string& r, const string& o) {
        const int ri = rel_index(r);
        for (const auto& f : K.facts_of(s))
            if (f.r == r && s8a::same_value(ri, f.o, o)) return true;
        return false;
    };
    // the amended grader for R1: K (the full store) or S15 lists it -> 1; S15 says otherwise -> 0; else -1
    auto grade_r1 = [&](const string& s, const string& r, const string& o, bool dev) {
        if (k_holds(s, r, o)) return 1;
        const int ri = rel_index(r);
        int g = s8a::s15_grade(s15_slot[0], ri, s, r, o);
        if (g < 0 && !dev) g = s8a::s15_grade(s15_slot[1], ri, s, r, o);
        return g;
    };
    // ============================================================ read 1: S8a's R1 random hide of store K
    vector<std::array<string, 3>> trip;
    for (const auto& [s, fs] : S.kby)
        for (const auto& [r, o] : fs) trip.push_back({s, r, o});
    vector<std::size_t> h(trip.size());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = i;
    {
        std::mt19937_64 rng(20261029);
        for (std::size_t i = h.size(); i > 1; --i) std::swap(h[i - 1], h[rng() % i]);
    }
    const std::set<std::size_t> hidden(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(trip.size() / 10));
    Facts red;
    for (std::size_t i = 0; i < trip.size(); ++i)
        if (!hidden.count(i)) red[trip[i][0]].push_back({trip[i][1], trip[i][2]});
    const ecs::TripleStore st = s8a::make_store(red);
    const ecs::RuleIndex ix(st);
    const vector<ecs::Rule> rules = ecs::mine_rules(ix);
    const ecs::RuleBook book(rules);
    std::size_t nkept = 0;
    for (const auto& R : rules) nkept += R.kept;
    std::snprintf(b, sizeof b, "S12A R1 random hide: hidden %zu, reduced store %zu, candidate rules %zu, kept %zu", hidden.size(), st.size(), rules.size(), nkept);
    std::cout << b << "\n";
    // ---- G1 by depth
    dp::DepthCalib g1cal;  // S12c item 3: graded counts by depth on R1's dev facts
    struct G1Fact {
        int depth;
        double conf;
        int grade;
        string s;
    };
    vector<G1Fact> g1dev;
    {
        const double tc = now_s();
        const auto derived = dp::deep_chain(st, book, 6);
        struct Acc {
            double n = 0, g = 0, t = 0, nl_g = 0, nl_t = 0, conf = 0;
        };
        std::map<int, Acc> by[2];
        std::ofstream df(out + ".g1_r1.tsv");
        df << "split\tdepth\tsubject\trelation\tobject\tconf\tgrade\tgroup\tpremises\n";
        for (const auto& d : derived) {
            const bool dev = s11c::dev_subject(d.t.s);
            if (!dev && !read_test) continue;
            const int g = grade_r1(d.t.s, d.t.r, d.t.o, dev);
            Acc& a = by[dev ? 0 : 1][d.depth];
            a.n += 1, a.conf += d.conf;
            if (dev && g >= 0) {
                g1cal.add(d.depth, g == 1);
                if (d.depth >= 2) g1dev.push_back({d.depth, d.conf, g, d.t.s});
            }
            if (g >= 0) {
                a.g += 1, a.t += g == 1;
                if (s8c::group_of(d.t.r) != "lexical") a.nl_g += 1, a.nl_t += g == 1;
            }
            string prem;
            for (const auto& p : d.premises) prem += (prem.empty() ? "" : " ; ") + p.text();
            df << (dev ? "dev" : "test") << '\t' << d.depth << '\t' << d.t.s << '\t' << d.t.r << '\t' << d.t.o << '\t' << d.conf << '\t' << g << '\t'
               << s8c::group_of(d.t.r) << '\t' << prem << '\n';
        }
        for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
            for (const auto& [dep, a] : by[sp]) {
                std::snprintf(b, sizeof b, "S12A G1 R1 %s depth %d: derived %5.0f (mean chain confidence %.3f) | graded %4.0f precision %s | non-lexical %s",
                              sp == 0 ? "dev" : "TEST", dep, a.n, a.n > 0 ? a.conf / a.n : 0.0, a.g, wilson(a.t, a.g).c_str(), wilson(a.nl_t, a.nl_g).c_str());
                std::cout << b << "\n";
            }
            Acc deep;
            for (const auto& [dep, a] : by[sp])
                if (dep >= 2) deep.n += a.n, deep.g += a.g, deep.t += a.t;
            std::snprintf(b, sizeof b, "S12A G1 R1 %s depth >= 2 (G1's hypotheses): derived %.0f, graded %.0f, precision %s", sp == 0 ? "dev" : "TEST", deep.n, deep.g,
                          wilson(deep.t, deep.g).c_str());
            std::cout << b << "\n";
        }
        std::snprintf(b, sizeof b, "S12A G1 R1 chained in %.1f s", now_s() - tc);
        std::cout << b << "\n";
    }
    // ---- S12c item 3: G1's prior by depth, compared on R1's dev facts (cross-fitted)
    if (s12c) {
        auto half = [&](const string& subj) { return static_cast<int>((fnv(subj) >> 3) & 1); };
        dp::DepthCalib cal[2];
        for (const auto& f : g1dev) cal[half(f.s)].add(f.depth, f.grade == 1);
        const char* names[3] = {"C0 chain confidence", "C1 cap at depth 3", "C2 calibrated by depth (cross-fitted)"};
        double brier[3] = {0, 0, 0}, logl[3] = {0, 0, 0};
        vector<std::pair<double, int>> pg[3];
        for (const auto& f : g1dev) {
            const double p[3] = {dp::clamp_prior(f.conf), f.depth <= 3 ? dp::clamp_prior(f.conf) : 0.02, dp::clamp_prior(cal[1 - half(f.s)].prior(f.depth))};
            for (int k = 0; k < 3; ++k) {
                brier[k] += (p[k] - f.grade) * (p[k] - f.grade);
                logl[k] += -std::log2(f.grade == 1 ? p[k] : 1.0 - p[k]);
                pg[k].push_back({p[k], f.grade});
            }
        }
        const double n = static_cast<double>(g1dev.size());
        int best = 0;
        for (int k = 0; k < 3; ++k) {
            double ece = 0;  // 10 equal-width bins
            for (int bin = 0; bin < 10; ++bin) {
                double c = 0, sp = 0, sy = 0;
                for (const auto& [p, y] : pg[k])
                    if (std::min(9, static_cast<int>(p * 10)) == bin) c += 1, sp += p, sy += y;
                if (c > 0) ece += c / n * std::fabs(sp / c - sy / c);
            }
            std::snprintf(b, sizeof b, "S12C G1 PRIOR %s: R1 dev graded facts at depth >= 2 %.0f | Brier %.4f | log loss %.4f bits | ECE %.4f", names[k], n, brier[k] / n, logl[k] / n, ece);
            std::cout << b << "\n";
        }
        // the registered rule: the lowest Brier; within 0.002 of the best, the simpler (C0, C1, C2)
        double bmin = std::min({brier[0], brier[1], brier[2]}) / n;
        for (int k = 0; k < 3; ++k)
            if (brier[k] / n <= bmin + 0.002) {
                best = k;
                break;
            }
        for (const auto& [d, tg] : g1cal.tg) {
            std::snprintf(b, sizeof b, "S12C G1 PRIOR depth %d (all R1 dev graded): true %.0f of %.0f -> calibrated prior %.4f", d, tg.first, tg.second, g1cal.prior(d));
            std::cout << b << "\n";
        }
        // paired Brier differences, 95% interval clustered by subject (facts of one subject are not independent)
        for (const auto& [k1, k0] : {std::pair<int, int>{2, 0}, {2, 1}, {1, 0}}) {
            std::map<string, double> by_s;
            double sum = 0;
            for (std::size_t i = 0; i < g1dev.size(); ++i) {
                const double d = (pg[k1][i].first - pg[k1][i].second) * (pg[k1][i].first - pg[k1][i].second) - (pg[k0][i].first - pg[k0][i].second) * (pg[k0][i].first - pg[k0][i].second);
                by_s[g1dev[i].s] += d, sum += d;
            }
            const double mean = sum / n;
            double v = 0;
            std::map<string, double> cnt;
            for (const auto& f : g1dev) cnt[f.s] += 1;
            for (const auto& [s, d] : by_s) v += (d - cnt[s] * mean) * (d - cnt[s] * mean);
            const double se = std::sqrt(v) / n;
            std::snprintf(b, sizeof b, "S12C G1 PRIOR Brier %s minus %s: %+.4f [%+.4f, %+.4f] (paired, 95%%, clustered by subject: %zu subjects)", names[k1], names[k0], mean, mean - 1.96 * se,
                          mean + 1.96 * se, by_s.size());
            std::cout << b << "\n";
        }
        std::snprintf(b, sizeof b, "S12C G1 PRIOR CHOICE (lowest Brier, ties within 0.002 to the simpler): %s -> --g1-prior %s", names[best], best == 0 ? "chain" : best == 1 ? "cap3" : "depth");
        std::cout << b << "\n";
    }
    // ============================================================ read 2: store T on S15 (G1 by depth)
    {
        const ecs::TripleStore T = s8a::make_store(S.tby);
        const ecs::RuleIndex tix(T);
        const vector<ecs::Rule> trules = ecs::mine_rules(tix);
        const ecs::RuleBook tbook(trules);
        std::size_t tk = 0;
        for (const auto& R : trules) tk += R.kept;
        const auto derived = dp::deep_chain(T, tbook, 6);
        std::map<int, std::array<double, 4>> by[2];  // depth -> derived on S15 subjects, graded, true, strict (b or c)
        std::set<string> s15_subj[2];
        for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
            for (const auto& [k, v] : s15_slot[sp]) s15_subj[sp].insert(k.substr(0, k.find('\x1f')));
        for (const auto& d : derived)
            for (int sp = 0; sp < (read_test ? 2 : 1); ++sp) {
                if (!s15_subj[sp].count(d.t.s)) continue;
                auto& a = by[sp][d.depth];
                a[0] += 1;
                const int g = s8a::s15_grade(s15_slot[sp], rel_index(d.t.r), d.t.s, d.t.r, d.t.o);
                if (g >= 0) {
                    a[1] += 1, a[2] += g == 1;
                    if (g == 1) {
                        const auto it = s15_slot[sp].find(d.t.s + '\x1f' + d.t.r);
                        for (const auto& [v, lab] : it->second) {
                            const auto ci = s15_cat[sp].find(d.t.s + '\x1f' + d.t.r + '\x1f' + v);
                            if (lab == 1 && s8a::same_value(rel_index(d.t.r), v, d.t.o) && ci != s15_cat[sp].end() && (ci->second == "b" || ci->second == "c")) {
                                a[3] += 1;
                                break;
                            }
                        }
                    }
                }
            }
        std::snprintf(b, sizeof b, "S12A G1 R2 store T: candidate rules %zu, kept %zu, derived facts %zu", trules.size(), tk, derived.size());
        std::cout << b << "\n";
        for (int sp = 0; sp < (read_test ? 2 : 1); ++sp)
            for (const auto& [dep, a] : by[sp]) {
                std::snprintf(b, sizeof b, "S12A G1 R2 %s depth %d: derived for S15 subjects %.0f | graded %.0f precision %s | true and unstated or later (b, c) %.0f",
                              sp == 0 ? "dev" : "TEST", dep, a[0], a[1], wilson(a[2], a[1]).c_str(), a[3]);
                std::cout << b << "\n";
            }
    }
    // ============================================================ the pools (S11c's construction) and the prospective log
    if (fresh_dev || fresh_test) {
        s8b::Corpus C;
        if (!s8b::load_corpus(argv[2], argv[5], argv[6], C)) return 1;
        const auto slots = s11c::pool_slots(s15dir, fresh_test);
        Facts fred;
        std::size_t held = 0;
        for (const auto& [s, fs] : S.kby)
            for (const auto& [r, o] : fs) {
                if (slots.count({s, r})) ++held;
                else fred[s].push_back({r, o});
            }
        const ecs::TripleStore fst = s8a::make_store(fred);
        const ecs::RuleIndex fix(fst);
        const vector<ecs::Rule> frules = ecs::mine_rules(fix);
        const ecs::RuleBook fbook(frules);
        const string ptag = fresh_test ? "S12A POOL test (S15 test, bit 17 = 0)" : "S12A POOL dev (S15 dev, bit 17 = 1)";
        vector<s11c::PoolItem> pool[2];  // G1, G20 (G19 removed by S12c)
        // G1: facts first derived at depth >= 2 in a pool slot; the best chain per slot (S12c: by the chosen prior)
        {
            std::map<std::pair<string, string>, const cypha::eval::Derived*> best;
            const auto derived = dp::deep_chain(fst, fbook, 6);
            for (const auto& d : derived) {
                if (d.depth < 2 || !slots.count({d.t.s, d.t.r})) continue;
                if (g1mode == dp::G1Prior::Cap3 && d.depth > 3) continue;
                auto& bp = best[{d.t.s, d.t.r}];
                if (g1mode == dp::G1Prior::Depth) {
                    const double pd = dp::g1_prior(g1mode, d.depth, d.conf, g1cal), pb = bp ? dp::g1_prior(g1mode, bp->depth, bp->conf, g1cal) : -1.0;
                    if (!bp || pd > pb || (pd == pb && (d.conf > bp->conf || (d.conf == bp->conf && d.t.o < bp->t.o)))) bp = &d;
                } else if (!bp || d.conf > bp->conf || (d.conf == bp->conf && d.t.o < bp->t.o)) {
                    bp = &d;
                }
            }
            for (const auto& [k, d] : best) {
                s11c::PoolItem x;
                x.s = d->t.s, x.r = d->t.r, x.o = d->t.o, x.gen = "G1", x.score = d->conf;
                x.prior = dp::g1_prior(g1mode, d->depth, d->conf, g1cal);
                string prem;
                for (const auto& p : d->premises) prem += (prem.empty() ? "" : " ; ") + p.text();
                x.explain = "depth " + std::to_string(d->depth) + ": " + prem;
                pool[0].push_back(x);
            }
        }
        // G20: the wild tier, 5% of the main pool, at least 20
        {
            const ecs::LinkPredictor lp(fst, ecs::LinkPredictor::Config{});
            vector<std::pair<string, string>> sv(slots.begin(), slots.end());
            const std::size_t budget = std::max<std::size_t>(20, pool[0].size() / 20);
            for (const auto& w : dp::wild_tier(fst, lp, sv, budget, 20261029)) {
                s11c::PoolItem x;
                x.s = w.t.s, x.r = w.t.r, x.o = w.t.o, x.gen = "G20", x.score = w.lp;
                x.prior = std::min(0.98, std::max(0.02, w.lp));
                char e[96];
                std::snprintf(e, sizeof e, "wild: P %.4f, base rate %.4f", w.lp, w.base);
                x.explain = e;
                pool[1].push_back(x);
            }
        }
        std::snprintf(b, sizeof b, "%s slots %zu | store facts hidden %zu | kept rules %zu | G1 %zu, G20 %zu hypotheses", ptag.c_str(), slots.size(), held,
                      static_cast<std::size_t>(std::count_if(frules.begin(), frules.end(), [](const ecs::Rule& R) { return R.kept; })), pool[0].size(),
                      pool[1].size());
        std::cout << b << "\n";
        const char* gn[2] = {"g1", "g20"};
        for (int g = 0; g < 2; ++g) {
            s11c::pool_loop(pool[g], fst, C, out + ".fresh_" + gn[g] + ".prereg.tsv");
            s11c::pool_states(ptag + " " + gn[g], pool[g]);
            s11c::write_pool(out + ".fresh_" + gn[g] + ".hyps.tsv", pool[g]);
        }
        // the prospective log: every pool hypothesis, timestamped, with the check registered for it
        {
            const std::time_t now = std::time(nullptr);
            char ts[32];
            std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
            std::ofstream pf(out + ".prospective.tsv");
            pf << "logged_utc\tid\tgenerator\ttier\tsubject\trelation\tobject\tprior\tstate_after_loop\tcheck\n";
            std::size_t id = 0;
            for (int g = 0; g < 2; ++g)
                for (const auto& x : pool[g])
                    pf << ts << '\t' << "S12A-" << (fresh_test ? "T" : "D") << '-' << id++ << '\t' << x.gen << '\t' << (g == 1 ? "wild" : "main") << '\t' << x.s << '\t' << x.r
                       << '\t' << x.o << '\t' << x.prior << '\t' << x.state << '\t'
                       << "graded when a later source (a later Wikidata, S8e's slots or Odin) lists the slot; not before" << '\n';
            std::snprintf(b, sizeof b, "S12A PROSPECTIVE log %s: %zu hypotheses logged at %s", (out + ".prospective.tsv").c_str(), id, ts);
            std::cout << b << "\n";
        }
    }
    std::snprintf(b, sizeof b, "S12A --gen-deep done in %.1f s", now_s() - t0);
    std::cout << b << "\n";
    return 0;
}

}  // namespace s12a

int main(int argc, char** argv) {
    {  // F2c: --lexicon-type-check FILE [--lexicon-type-arm P|PL] apply to every mode's pass 1; they are taken out of argv here
        int w = 1, arm = 2;  // PL, the arm F2c's rule chose
        bool on = false;
        for (int i = 1; i < argc; ++i) {
            const string a = argv[i];
            if (a == "--lexicon-type-check" && i + 1 < argc) {
                if (!lexfix::load(argv[++i])) { std::fprintf(stderr, "cannot read %s\n", argv[i]); return 1; }
                on = true;
                continue;
            }
            if (a == "--lexicon-type-arm" && i + 1 < argc) { arm = string(argv[++i]) == "P" ? 1 : 2; continue; }
            argv[w++] = argv[i];
        }
        argc = w;
        argv[argc] = nullptr;
        lexfix::g_arm = on && !lexfix::g_types.empty() ? arm : 0;
    }
    if (argc >= 2 && string(argv[1]) == "--score-planted") return s6::run(argc, argv);  // S6 (hypothesis scoring)
    if (argc >= 2 && string(argv[1]) == "--test-kinds") return s7::run(argc, argv);  // S7 (testing hypotheses)
    if (argc >= 2 && string(argv[1]) == "--gen-links") return s8b::run(argc, argv);  // S8b (link prediction, G6 and G7)
    if (argc >= 2 && string(argv[1]) == "--gen-rules") return s8a::run(argc, argv);  // S8a (G2 rules, G14 contradictions)
    if (argc >= 2 && string(argv[1]) == "--sources") return s8c::run(argc, argv);  // S8c (evidence sources and their reliability)
    if (argc >= 2 && string(argv[1]) == "--worlds") return s9::run(argc, argv);  // S9 (possible worlds, T2, fragility)
    if (argc >= 2 && string(argv[1]) == "--schedule") return s10::run(argc, argv);  // S10 (the curiosity scheduler)
    if (argc >= 2 && string(argv[1]) == "--s7d") return s7d::run(argc, argv);  // S7d (T2 in the loop, Wikidata as one source, K3 against K4)
    if (argc >= 2 && string(argv[1]) == "--gen-concepts") return s11a::run(argc, argv);  // S11a (G9 concepts, G11 open relations)
    if (argc >= 2 && string(argv[1]) == "--gen-anomaly") return s11c::run_anomaly(argc, argv);  // S11c (G13 anomalies, G3 abduction)
    if (argc >= 2 && string(argv[1]) == "--gen-analogy") return s11c::run_analogy(argc, argv);  // S11c (G4 and G5 analogy)
    if (argc >= 2 && string(argv[1]) == "--pool-grade") return s11c::run_grade(argc, argv);  // S11c (a pool after the source stage)
    if (argc >= 2 && string(argv[1]) == "--f2b-selftest") return f2b::selftest();  // F2b (key rules)
    if (argc >= 2 && string(argv[1]) == "--gen-laws") return s12a::run_laws(argc, argv);  // S12a (G15 numeric laws, the planted-law gate)
    if (argc >= 2 && string(argv[1]) == "--gen-deep") return s12a::run_deep(argc, argv);  // S12a (G1 deep deduction, G20 wild tier; G19 removed by S12c)
    if (argc >= 4 && string(argv[1]) == "--wilson") {
        double k = std::atof(argv[2]), n = std::atof(argv[3]);
        std::printf("%.0f/%.0f = %.3f  Wilson95 [%.3f, %.3f]\n", k, n, n > 0 ? k / n : 0.0, ecs::wilson_lower(k, n), ecs::wilson_upper(k, n));
        return 0;
    }
    Args A;
    if (argc < 3) {
        std::fprintf(stderr, "usage: pattern_learn <enwik8> <outdir> [--folds K] [--cv] [--min-train N] [--sample N] [--seed S] [--base-rate A]\n"
                             "         [--wikidata FILE [--wikidata-gold-only]] [--phi FILE] [--shard FILE --dev DEV --test TEST [--slice-end B] [--holdout-from B]]\n"
                             "         [--raw] [--join FACTS] [--reach PARSE] [--gaps FILE [--demand ROWS,...] [--parse PARSE] [--gap-floor F] [--demand-weight W] [--gaps-spof SPOF]]\n"
                             "         [--frontier FILE [--frontier-test] [--frontier-split B]]\n"
                             "         [--signals DIR [--signals-dolly PAGES.xml] [--surprise SCORES.tsv,...] [--signals-gaps FILE]\n"
                             "          [--signals-scope dev|test|u-dev [--signals-plain PAGES.xml] [--signals-vec FILE]]]\n"
                             "         [--f2b-xa] [--f2b-wild 1|2] [--f2b-classes FILE] [--f2b-gold-split] [--f2b-read-test]\n"
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
        else if (a == "--gaps-spof" && i + 1 < argc) A.gaps.spof = argv[++i];  // S9b: fragile gaps from --worlds OUT.spof.tsv
        else if (a == "--frontier" && i + 1 < argc) A.frontier.path = argv[++i];
        else if (a == "--frontier-test") A.frontier.test = true;
        else if (a == "--frontier-split" && i + 1 < argc) A.frontier.split = static_cast<std::size_t>(std::atoll(argv[++i]));
        else if (a == "--signals" && i + 1 < argc) A.signals.dir = argv[++i];
        else if (a == "--signals-dolly" && i + 1 < argc) A.signals.dolly = argv[++i];
        else if (a == "--surprise" && i + 1 < argc) A.signals.surprise = split(argv[++i], ',');
        else if (a == "--signals-gaps" && i + 1 < argc) A.signals.gaps = argv[++i];
        else if (a == "--signals-scope" && i + 1 < argc) A.signals.scope = argv[++i];  // S4c: dev | test; S4b-b: u-dev
        else if (a == "--signals-plain" && i + 1 < argc) A.signals.plain = argv[++i];
        else if (a == "--signals-vec" && i + 1 < argc) A.signals.vec = argv[++i];
        else if (a == "--f2b-xa") f2b::g_xa = true;  // F2b: cross-article reading
        else if (a == "--f2b-wild" && i + 1 < argc) g_f2b_wild = std::atoi(argv[++i]);
        else if (a == "--f2b-classes" && i + 1 < argc) {
            if (!f2b::load_classes(argv[++i])) { std::fprintf(stderr, "cannot read %s\n", argv[i]); return 1; }
        }
        else if (a == "--f2b-gold-split") f2b::g_gold_split = true;
        else if (a == "--f2b-read-test") f2b::g_gold_split = f2b::g_read_test = true;
    }
    if (g_raw && (f2b::g_xa || f2b_gen_on())) { std::fprintf(stderr, "--f2b-xa, --f2b-wild and --f2b-classes read cleaned tokens, not --raw\n"); return 2; }
    if (!A.reach.empty() && A.shard.dev.empty()) { std::fprintf(stderr, "--reach needs --dev\n"); return 2; }
    if (!A.frontier.path.empty() && A.gaps.path.empty()) { std::fprintf(stderr, "--frontier needs --gaps\n"); return 2; }
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
        if ((ty == T_PLACE || ty == T_PLACE_LINK) && lexfix::widen_ok(sd, ty)) { g_place.insert(sd.o.k1); if (!sd.o.k2.empty()) g_place.insert(sd.o.k2); }
        if (ty == T_PERSON && lexfix::widen_ok(sd, ty)) g_person.insert(sd.o.k1);
        if (ty == T_LEX && sd.o.k1.size() >= 3) g_lex[sd.o.k1] |= 1ULL << sd.rel;
        if (string(kRel[sd.rel].name) == "located_in") g_loc[norm_ent(sd.subj)].insert(sd.o.k1);
    }
    say("pass 1 (seeds, types): " + std::to_string(now_s() - t1) + " s; place pages " + std::to_string(g_place.size()) + ", person pages " + std::to_string(g_person.size()) +
        ", lexicon strings " + std::to_string(g_lex.size()));
    if (lexfix::g_arm) {  // F2c
        long np = 0, nl = 0;
        std::ofstream bf(A.outdir + "/lexfix_blocked.tsv");
        bf << "lexicon\tvalue\tstill_in_lexicon\n";
        for (const string& v : lexfix::g_blocked_person) { bool in = g_person.count(v) > 0; np += !in; bf << "person\t" << v << '\t' << in << '\n'; }
        for (const string& v : lexfix::g_blocked_place) { bool in = g_place.count(v) > 0; nl += !in; bf << "place\t" << v << '\t' << in << '\n'; }
        say("F2C lexicon type check (arm " + string(lexfix::g_arm == 2 ? "PL" : "P") + ", " + std::to_string(lexfix::g_types.size()) +
            " typed items): person values kept out " + std::to_string(lexfix::g_blocked_person.size()) + " (" + std::to_string(np) +
            " not in the person lexicon from another source), place values kept out " + std::to_string(lexfix::g_blocked_place.size()) + " (" +
            std::to_string(nl) + " not in the place lexicon)");
    }
    if (f2b::g_xa) {  // F2b
        f2b::load_redirects(data);
        say("F2B cross-article reading: redirects " + std::to_string(f2b::g_redir.size()));
    }
    if (f2b_gen_on()) say("F2B generalised gaps: wildcard mode " + std::to_string(g_f2b_wild) + ", word classes " + std::to_string(g_f2b_class.size()));

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
        if (f2b::g_xa) f2b::xa_pass(0, data, pages, page_of, A.folds, fold, &tm, nullptr);  // F2b
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
        if (f2b::g_xa) f2b::xa_pass(1, data, pages, page_of, A.folds, fold, nullptr, &local);  // F2b
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
    if (f2b::g_xa) f2b::xa_pass(2, data, pages, page_of, A.folds, 0, nullptr, nullptr);  // F2b
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
    if (!g_gold.empty()) {
        if (f2b::g_gold_split) f2b::score_halves(fused, A.outdir, say);  // F2b: gold-dev, and gold-test only with --f2b-read-test
        else score_gold(fused, A.outdir, say);
    }
    string shard_text;
    int exit_code = A.shard.path.empty() ? 0 : write_shard(A.shard, fused, pages, page_of, say, &shard_text);
    if (!A.reach.empty()) reach_report(cand_all, data, A.shard.holdout_from, A.shard.dev, A.reach, shard_text, say);
    if (!A.gaps.path.empty()) write_gaps(A.gaps, A.shard.dev, pages, fused, say);
    if (!A.frontier.path.empty()) write_frontier(A.frontier, A.gaps, data, pages, fused, say);
    if (!A.signals.dir.empty()) write_signals(A.signals, data, pages, say);  // S4b
    std::snprintf(buf, sizeof buf, "\nrun time %.1f s, peak RAM %.0f MB, facts written %s/facts_f2.tsv", now_s() - t0, peak_mb(), A.outdir.c_str());
    say(buf);
    std::ofstream(A.outdir + "/stats.txt") << rep.str();
    return exit_code;
}
