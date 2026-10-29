// cypha_report_card: CyphaLM's evaluation contract on one card (bill of work M1-M4; docs/reports/REPORT_CARD.md).
//
// The runs are made by scripts/cypha_report_card.sh (cyphalm_trie qa / context8k / sample, scribe ladder, peak RSS).
// This tool reads what they wrote and writes the card: one dated Markdown table and one TSV with the same numbers.
// Proportions get Wilson 95% intervals (ecs.hpp), SCRIBE the interval scribe prints, BPC the experts' paired gain
// intervals cyphalm_trie prints; a baseline comparison is paired per question and per (prompt, temperature) and goes
// through lci.hpp's Bayesian bootstrap and compute_lci.
//
// Usage:
//   cypha_report_card summarize --dir DIR --label L [--baseline B] [--budget-mb 200]
//       reads DIR/L.meta.tsv, L.qa.out, L.qa.rows.tsv, L.ctx.out, L.scribe.md, L.scribe_export/scores.jsonl,
//       L.gen_dev.txt, L.fresh.scribe.md, L.*.peak (each optional), and the same for B; writes DIR/L.card.md and
//       DIR/L.card.tsv.
//   cypha_report_card selftest
//
// Author: Odin Loch

#include "cypha/eval/ecs.hpp"
#include "cypha/eval/lci.hpp"
#include "cypha/eval/scribe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ev = cypha::eval;

namespace rc {

// ---------------------------------------------------------------------------- small helpers

std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> v;
    std::ifstream in(path);
    std::string s;
    while (std::getline(in, s)) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
        v.push_back(s);
    }
    return v;
}
bool exists(const std::string& p) { return fs::exists(p); }
std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> v;
    std::istringstream is(s);
    std::string t;
    while (is >> t) v.push_back(t);
    return v;
}
std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> v;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find('\t', a);
        v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return v;
}
bool starts(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool parse_num(const std::string& t, double& out) {
    if (t.empty()) return false;
    char* e = nullptr;
    out = std::strtod(t.c_str(), &e);
    return e && *e == '\0' && std::isfinite(out);
}
std::string fmt(double v, int prec) {
    if (!std::isfinite(v)) return "n/a";
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}

// Wilson 95% interval of k / n (ecs.hpp).
struct Prop {
    double k = 0, n = 0;
    double p() const { return n > 0 ? k / n : NAN; }
    double lo() const { return n > 0 ? ev::wilson_lower(k, n) : NAN; }
    double hi() const { return n > 0 ? ev::wilson_upper(k, n) : NAN; }
};

// ---------------------------------------------------------------------------- what the tools print

// A tally line: PREFIX TAG SUBSET n N right R abstained A wrong W ...
struct Tally {
    bool found = false;
    double n = 0, right = 0, abst = 0, wrong = 0;
};
Tally find_tally(const std::vector<std::string>& lines, const std::string& prefix, const std::string& tag,
                 const std::string& subset) {
    Tally t;
    for (const std::string& l : lines) {
        const std::vector<std::string> w = split_ws(l);
        if (w.size() < 11 || w[0] != prefix || w[1] != tag || w[2] != subset || w[3] != "n") continue;
        std::map<std::string, double> kv;
        for (std::size_t i = 3; i + 1 < w.size(); i += 2) {
            double v;
            if (parse_num(w[i + 1], v)) kv[w[i]] = v;
        }
        t.found = true;
        t.n = kv["n"], t.right = kv["right"], t.abst = kv["abstained"], t.wrong = kv["wrong"];
        return t;
    }
    return t;
}

// ABSTAIN TAG abstained A not_in_slice x/y ... | in_slice_base_wrong u/v ... | tags T complete C reason_ok R ...
struct Abstain {
    bool found = false;
    double abstained = 0, nis_hit = 0, nis_total = 0, should_hit = 0, should_total = 0, tags = 0, complete = 0,
           reason_ok = 0;
};
Abstain find_abstain(const std::vector<std::string>& lines, const std::string& prefix, const std::string& tag) {
    Abstain a;
    for (const std::string& l : lines) {
        const std::vector<std::string> w = split_ws(l);
        if (w.size() < 4 || w[0] != prefix || w[1] != tag) continue;
        auto frac = [](const std::string& s, double& x, double& y) {
            const std::size_t k = s.find('/');
            if (k == std::string::npos) return false;
            return parse_num(s.substr(0, k), x) && parse_num(s.substr(k + 1), y);
        };
        for (std::size_t i = 2; i + 1 < w.size(); ++i) {
            double v;
            if (w[i] == "abstained" && parse_num(w[i + 1], v)) a.abstained = v;
            else if (w[i] == "not_in_slice") frac(w[i + 1], a.nis_hit, a.nis_total);
            else if (w[i] == "in_slice_base_wrong") frac(w[i + 1], a.should_hit, a.should_total);
            else if (w[i] == "tags" && parse_num(w[i + 1], v)) a.tags = v;
            else if (w[i] == "complete" && parse_num(w[i + 1], v)) a.complete = v;
            else if (w[i] == "reason_ok" && parse_num(w[i + 1], v)) a.reason_ok = v;
        }
        a.found = true;
        return a;
    }
    return a;
}

// Every "<key>_bpc <number>" pair in the output (first occurrence wins), outside the example lines.
std::map<std::string, double> bpc_pairs(const std::vector<std::string>& lines) {
    std::map<std::string, double> m;
    for (const std::string& l : lines) {
        if (starts(l, "EX ") || starts(l, "RIGHT ") || starts(l, "SIM_NB") || starts(l, "TEST_RIGHT")) continue;
        const std::vector<std::string> w = split_ws(l);
        for (std::size_t i = 0; i + 1 < w.size(); ++i) {
            const std::string& k = w[i];
            if (k.size() > 4 && k.compare(k.size() - 4, 4, "_bpc") == 0) {
                double v;
                if (parse_num(w[i + 1], v) && !m.count(k)) m[k] = v;
            }
        }
    }
    return m;
}

// "<name>_lci <stack> gain_bpc G lo90 L hi90 H ...": an expert's paired gain on the holdout (cyphalm_trie).
struct Gain {
    std::string name;
    double gain = NAN, lo = NAN, hi = NAN;
};
std::vector<Gain> lci_gains(const std::vector<std::string>& lines) {
    std::vector<Gain> v;
    for (const std::string& l : lines) {
        const std::vector<std::string> w = split_ws(l);
        if (w.size() < 8 || w[0].size() < 5 || w[0].compare(w[0].size() - 4, 4, "_lci") != 0) continue;
        Gain g;
        g.name = w[0] + " " + w[1];
        for (std::size_t i = 2; i + 1 < w.size(); ++i) {
            if (w[i] == "gain_bpc") parse_num(w[i + 1], g.gain);
            else if (w[i] == "lo90") parse_num(w[i + 1], g.lo);
            else if (w[i] == "hi90") parse_num(w[i + 1], g.hi);
        }
        if (std::isfinite(g.gain)) v.push_back(g);
    }
    return v;
}

// The fullest line of a BPC stack: the candidate keys in order of preference, the first present.
std::pair<std::string, double> headline(const std::map<std::string, double>& m, const std::vector<std::string>& pref) {
    for (const std::string& k : pref) {
        auto it = m.find(k);
        if (it != m.end()) return {k, it->second};
    }
    return {"", NAN};
}
const std::vector<std::string> kQaPref = {"holdout2_trie_ig_cls_sim_copy_bpc", "holdout2_trie_ig_cls_sim_bpc",
                                          "holdout2_trie_ig_cls_copy_bpc", "holdout2_trie_ig_sim_copy_bpc",
                                          "holdout2_trie_ig_copy_bpc", "holdout2_trie_ig_cls_bpc",
                                          "holdout2_trie_ig_sim_bpc", "holdout2_trie_ig_bpc"};
const std::vector<std::string> kCtxPref = {"trie_plus_context_8192_cls_sim_copy_bpc", "trie_plus_context_8192_cls_sim_bpc",
                                           "trie_plus_context_8192_cls_copy_bpc", "trie_plus_context_8192_sim_copy_bpc",
                                           "trie_plus_context_8192_copy_bpc", "trie_plus_context_8192_cls_bpc",
                                           "trie_plus_context_8192_sim_bpc", "trie_plus_context_8192_bpc"};

// A row of scribe ladder's rung-mean table: | name | n | SCRIBE m [lo, hi] | parity | S m [lo, hi] | bytes |
struct Rung {
    bool found = false;
    double n = 0, scribe = NAN, slo = NAN, shi = NAN, surf = NAN, flo = NAN, fhi = NAN, bytes = NAN;
};
bool parse_ci(const std::string& cell, double& m, double& lo, double& hi) {
    // "0.063 [0.051, 0.078]"
    std::string c = cell;
    for (char& ch : c)
        if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
    const std::vector<std::string> w = split_ws(c);
    return w.size() == 3 && parse_num(w[0], m) && parse_num(w[1], lo) && parse_num(w[2], hi);
}
Rung find_rung(const std::vector<std::string>& lines, const std::string& name) {
    Rung r;
    for (const std::string& l : lines) {
        if (!starts(l, "| ")) continue;
        std::vector<std::string> cells;
        std::size_t a = 1;
        for (;;) {
            const std::size_t b = l.find('|', a);
            if (b == std::string::npos) break;
            cells.push_back(trim(l.substr(a, b - a)));
            a = b + 1;
        }
        if (cells.size() < 6 || cells[0] != name) continue;
        if (!parse_num(cells[1], r.n) || !parse_ci(cells[2], r.scribe, r.slo, r.shi) ||
            !parse_ci(cells[4], r.surf, r.flo, r.fhi))
            continue;
        parse_num(cells[5], r.bytes);
        r.found = true;
        return r;
    }
    return r;
}

// peak file: "peak_rss_kb N seconds S rc R"
struct Peak {
    bool found = false;
    double mb = NAN, seconds = NAN, rc = NAN;
};
Peak read_peak(const std::string& path) {
    Peak p;
    if (!exists(path)) return p;
    for (const std::string& l : read_lines(path)) {
        const std::vector<std::string> w = split_ws(l);
        for (std::size_t i = 0; i + 1 < w.size(); ++i) {
            double v;
            if (!parse_num(w[i + 1], v)) continue;
            if (w[i] == "peak_rss_kb") p.mb = v / 1000.0, p.found = true;
            else if (w[i] == "seconds") p.seconds = v;
            else if (w[i] == "rc") p.rc = v;
        }
    }
    return p;
}

// meta: key \t value
std::map<std::string, std::string> read_meta(const std::string& path) {
    std::map<std::string, std::string> m;
    for (const std::string& l : read_lines(path)) {
        const std::size_t k = l.find('\t');
        if (k != std::string::npos) m[l.substr(0, k)] = l.substr(k + 1);
    }
    return m;
}

// ---------------------------------------------------------------------------- pairing for comparisons

// scribe.cpp's unescape_gen (the GEN lines' escapes), so the kept samples are the ones scribe scored.
std::string unescape_gen(const std::string& in) {
    std::string o;
    auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : 0); };
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') {
            o.push_back('\n');
            ++i;
        } else if (in[i] == '\\' && i + 3 < in.size() && in[i + 1] == 'x') {
            o.push_back(static_cast<char>((hex(in[i + 2]) << 4) | hex(in[i + 3])));
            i += 3;
        } else {
            o.push_back(in[i]);
        }
    }
    return o;
}

// The (prompt, kind) keys of the samples scribe kept, in the order it scored them (load_gen_trie's rule: cleaned
// text under 100 bytes dropped).
std::vector<std::string> kept_sample_keys(const std::vector<std::string>& gen_lines) {
    std::vector<std::string> keys;
    std::string prompt = "?";
    for (const std::string& l : gen_lines) {
        if (starts(l, "PROMPT\t")) {
            const std::vector<std::string> c = split_tab(l);
            if (c.size() >= 2) prompt = c[1];
            continue;
        }
        if (!starts(l, "GEN\t")) continue;
        const std::size_t t2 = l.find('\t', 4);
        if (t2 == std::string::npos) continue;
        const std::string text = ev::clean_generated_text(unescape_gen(l.substr(t2 + 1)));
        if (text.size() < 100) continue;
        keys.push_back(prompt + "|" + l.substr(4, t2 - 4));
    }
    return keys;
}

// The "scribe" and "surface_s" fields of the "system":"trie" lines of scribe's export, in order.
bool json_num(const std::string& line, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\":";
    const std::size_t k = line.find(pat);
    if (k == std::string::npos) return false;
    std::size_t a = k + pat.size(), b = a;
    while (b < line.size() && line[b] != ',' && line[b] != '}') ++b;
    return parse_num(line.substr(a, b - a), out);
}
struct SampleScores {
    std::map<std::string, std::pair<double, double>> by_key;  // key -> (scribe, surface S)
    std::size_t kept = 0, scored = 0;
    bool present = false, aligned = false;
};
SampleScores sample_scores(const std::string& gen_path, const std::string& export_scores) {
    SampleScores s;
    if (!exists(gen_path) || !exists(export_scores)) return s;
    s.present = true;
    const std::vector<std::string> keys = kept_sample_keys(read_lines(gen_path));
    std::vector<std::pair<double, double>> sc;
    for (const std::string& l : read_lines(export_scores)) {
        if (l.find("\"system\":\"trie\"") == std::string::npos) continue;
        double a = NAN, b = NAN;
        json_num(l, "scribe", a);
        json_num(l, "surface_s", b);
        sc.push_back({a, b});
    }
    s.kept = keys.size();
    s.scored = sc.size();
    s.aligned = keys.size() == sc.size();
    if (!s.aligned) return s;
    for (std::size_t i = 0; i < keys.size(); ++i) s.by_key[keys[i]] = sc[i];
    return s;
}

// Raw grades per dev question (chat form, trie_ig) from qa's rows file: 0 right, 1 abstained, 2 wrong.
std::map<std::string, int> dev_grades(const std::string& rows_path) {
    std::map<std::string, int> g;
    for (const std::string& l : read_lines(rows_path)) {
        const std::vector<std::string> c = split_tab(l);
        if (c.size() < 6 || c[0] != "dev" || c[3] != "chat" || c[4] != "trie_ig") continue;
        double v;
        if (parse_num(c[5], v)) g[c[1]] = static_cast<int>(v);
    }
    return g;
}

// ---------------------------------------------------------------------------- one card

struct Run {
    std::string dir, label;
    std::map<std::string, std::string> meta;
    std::vector<std::string> qa, ctx, scribe, fresh_scribe;
    std::map<std::string, double> qa_bpc, ctx_bpc;
    std::vector<Gain> gains;
    Peak pk_qa, pk_ctx, pk_gen, pk_scribe, pk_fresh;
    bool test_read = false;
};
Run load_run(const std::string& dir, const std::string& label) {
    Run r;
    r.dir = dir, r.label = label;
    const std::string b = dir + "/" + label;
    r.meta = read_meta(b + ".meta.tsv");
    r.qa = read_lines(b + ".qa.out");
    r.ctx = read_lines(b + ".ctx.out");
    r.scribe = read_lines(b + ".scribe.md");
    r.fresh_scribe = read_lines(b + ".fresh.scribe.md");
    r.qa_bpc = bpc_pairs(r.qa);
    r.ctx_bpc = bpc_pairs(r.ctx);
    r.gains = lci_gains(r.qa);
    r.pk_qa = read_peak(b + ".qa.peak");
    r.pk_ctx = read_peak(b + ".ctx.peak");
    r.pk_gen = read_peak(b + ".gen.peak");
    r.pk_scribe = read_peak(b + ".scribe.peak");
    r.pk_fresh = read_peak(b + ".fresh.peak");
    // A part that exited non-zero is not read (its output may be cut short); the Cost rows say it failed.
    auto failed = [](const Peak& p) { return p.found && std::isfinite(p.rc) && p.rc != 0; };
    if (failed(r.pk_qa)) r.qa.clear(), r.qa_bpc.clear(), r.gains.clear();
    if (failed(r.pk_ctx)) r.ctx.clear(), r.ctx_bpc.clear();
    if (failed(r.pk_gen) || failed(r.pk_scribe)) r.scribe.clear();
    if (failed(r.pk_gen) || failed(r.pk_fresh)) r.fresh_scribe.clear();
    // Rule 1: a test tally counts only when the run was made with --read-test (the meta says so).
    r.test_read = r.meta.count("split") && r.meta.at("split") == "test";
    return r;
}

struct Row {
    std::string section, metric, value, lo, hi, n, split, note;
};

struct Card {
    std::vector<Row> rows;
    void add(const std::string& sec, const std::string& m, double v, double lo, double hi, double n, const std::string& split,
             const std::string& note, int prec = 4) {
        rows.push_back({sec, m, fmt(v, prec), fmt(lo, prec), fmt(hi, prec), std::isfinite(n) ? fmt(n, 0) : "", split, note});
    }
    void add_prop(const std::string& sec, const std::string& m, const Prop& p, const std::string& split, const std::string& note) {
        add(sec, m, p.p(), p.lo(), p.hi(), p.n, split, note.empty() ? fmt(p.k, 0) + " of " + fmt(p.n, 0) : note, 3);
    }
};

void answers_rows(Card& c, const Run& r, const std::string& prefix, const std::string& abst_prefix, const std::string& split) {
    const Tally t = find_tally(r.qa, prefix, "chat/trie_ig", "all");
    if (!t.found) return;
    const std::string sec = split == "test" ? "Answers (frozen test)" : "Answers (dev)";
    c.add_prop(sec, "right, kept by the gate (" + prefix + " chat/trie_ig all)", {t.right, t.n}, split, "");
    c.add_prop(sec, "wrong, kept by the gate", {t.wrong, t.n}, split,
               fmt(t.wrong, 0) + " of " + fmt(t.n, 0) + "; the gate's target is 2% (chosen out of fold, 5-fold inside dev)");
    c.add_prop(sec, "abstained (the gate or CyphaLM's own don't-know)", {t.abst, t.n}, split, "");
    c.add_prop(sec, "coverage (answered)", {t.right + t.wrong, t.n}, split, "");
    // Before the gate: everything CyphaLM wrote counts; only its own don't-know message abstains.
    const Tally raw = find_tally(r.qa, split == "test" ? "TEST_RAW" : "RAW", "chat/trie_ig", "all");
    if (raw.found) {
        c.add_prop(sec, "right, before the gate", {raw.right, raw.n}, split, "");
        c.add_prop(sec, "wrong, before the gate", {raw.wrong, raw.n}, split, "");
        c.add_prop(sec, "abstained, before the gate (CyphaLM's own don't-know)", {raw.abst, raw.n}, split, "");
    }
    const Abstain a = find_abstain(r.qa, abst_prefix, "chat/trie_ig");
    if (!a.found) return;
    const std::string es = split == "test" ? "ECS: abstaining (frozen test)" : "ECS: abstaining (dev)";
    c.add_prop(es, "don't-know precision (abstained and outside the slice)", {a.nis_hit, a.abstained}, split, "");
    c.add_prop(es, "don't-know recall (questions outside the slice refused)", {a.nis_hit, a.nis_total}, split, "");
    c.add_prop(es, "refused, of the in-slice questions the base rows (--base) get wrong", {a.should_hit, a.should_total}, split, "");
    c.add_prop(es, "<|unknown|> tags complete", {a.complete, a.tags}, split, "");
    c.add_prop(es, "<|unknown|> tags with the right reason", {a.reason_ok, a.tags}, split, "");
}

void bpc_rows(Card& c, const Run& r) {
    const std::string note_fixed = "fixed holdout; no absolute interval (cyphalm_trie writes no per-block bits)";
    const auto q = headline(r.qa_bpc, kQaPref);
    if (std::isfinite(q.second)) {
        std::string note = note_fixed + "; line " + q.first;
        if (q.first.find("copy") != std::string::npos) note += " (qa scores copy on the first 512 KiB of the holdout's second half)";
        c.add("BPC", "qa holdout, the configuration's fullest stack", q.second, NAN, NAN, NAN, "dev", note);
    }
    const auto base = r.qa_bpc.find("holdout2_trie_ig_bpc");
    if (base != r.qa_bpc.end()) c.add("BPC", "qa holdout, trie + InfiniGram", base->second, NAN, NAN, NAN, "dev", note_fixed);
    for (const Gain& g : r.gains)
        c.add("BPC", "gain " + g.name + " (paired, 90%)", g.gain, g.lo, g.hi, NAN, "dev", "cyphalm_trie's block bootstrap");
    const auto k = headline(r.ctx_bpc, kCtxPref);
    if (std::isfinite(k.second))
        c.add("BPC", "context8k, the configuration's fullest stack", k.second, NAN, NAN, NAN, "dev", note_fixed + "; line " + k.first);
}

void scribe_rows(Card& c, const std::vector<std::string>& md, const std::string& split, const std::string& tag) {
    const std::string sec = "Text (SCRIBE v2, surface S" + tag + ")";
    struct N {
        const char* name;
        const char* label;
    };
    const N names[] = {{"trie samples (pooled)", "CyphaLM samples, pooled"},
                       {"trie samples T=0.6", "CyphaLM samples, T 0.6"},
                       {"trie samples T=0.8", "CyphaLM samples, T 0.8"},
                       {"trie samples T=1.0", "CyphaLM samples, T 1.0"},
                       {"held-out human", "human text (reference rung)"},
                       {"sentences shuffled across docs", "sentences shuffled across documents (control rung)"}};
    for (const N& nm : names) {
        const Rung r = find_rung(md, nm.name);
        if (!r.found) continue;
        c.add(sec, std::string(nm.label) + ": SCRIBE v2", r.scribe, r.slo, r.shi, r.n, split, "scribe's 95% bootstrap", 3);
        c.add(sec, std::string(nm.label) + ": surface S", r.surf, r.flo, r.fhi, r.n, split, "scribe's 95% bootstrap", 3);
    }
}

void cost_rows(Card& c, const Run& r) {
    struct P {
        const char* name;
        const Peak* p;
    };
    const P ps[] = {{"qa", &r.pk_qa},
                    {"context8k", &r.pk_ctx},
                    {"sample", &r.pk_gen},
                    {"scribe ladder", &r.pk_scribe},
                    {"scribe ladder (fresh)", &r.pk_fresh}};
    for (const P& p : ps) {
        if (!p.p->found) continue;
        const bool failed = std::isfinite(p.p->rc) && p.p->rc != 0;
        c.add("Cost", std::string(p.name) + " peak RSS (MB)", p.p->mb, NAN, NAN, NAN, "",
              failed ? "one run; no interval; the part FAILED (exit code " + fmt(p.p->rc, 0) + "), its numbers are not read"
                     : std::string("one run; no interval"),
              1);
        c.add("Cost", std::string(p.name) + " wall time (s)", p.p->seconds, NAN, NAN, NAN, "", "one run on a shared machine", 0);
    }
}

// ---------------------------------------------------------------------------- comparison

struct Compare {
    bool answers = false, scribe = false;
    ev::BootstrapResult d_right, d_wrong, d_scribe, d_surface;
    std::size_t n_q = 0, n_pairs = 0, only_a = 0, only_b = 0;
    std::vector<double> temp_gains;
    double fresh_gain = NAN, dev_gain = NAN;
    ev::LciDecision lci;
    bool lci_read = false;
    double budget_mb = 200, cand_peak = NAN;
    std::string why_no_scribe;
};

Compare compare(const Run& cand, const Run& base, double budget_mb) {
    Compare cm;
    cm.budget_mb = budget_mb;
    cm.cand_peak = cand.pk_qa.mb;
    ev::BootstrapOptions bo;
    // answers, paired per dev question
    const std::map<std::string, int> ga = dev_grades(cand.dir + "/" + cand.label + ".qa.rows.tsv");
    const std::map<std::string, int> gb = dev_grades(base.dir + "/" + base.label + ".qa.rows.tsv");
    std::vector<double> dr, dw, nw;
    for (const auto& [q, g] : ga) {
        auto it = gb.find(q);
        if (it == gb.end()) continue;
        dr.push_back((g == 0) - (it->second == 0));
        dw.push_back((g == 2) - (it->second == 2));
        nw.push_back((g != 2) - (it->second != 2));
    }
    cm.n_q = dr.size();
    if (!dr.empty()) {
        cm.answers = true;
        cm.d_right = ev::bayesian_bootstrap(dr, 0.0, {}, bo);
        cm.d_wrong = ev::bayesian_bootstrap(dw, 0.0, {}, bo);
    }
    // SCRIBE, paired per (prompt, temperature) kept in both
    const SampleScores sa = sample_scores(cand.dir + "/" + cand.label + ".gen_dev.txt",
                                          cand.dir + "/" + cand.label + ".scribe_export/scores.jsonl");
    const SampleScores sb = sample_scores(base.dir + "/" + base.label + ".gen_dev.txt",
                                          base.dir + "/" + base.label + ".scribe_export/scores.jsonl");
    std::vector<double> ds, df;
    std::map<std::string, std::vector<double>> by_temp;
    if (!sa.present || !sb.present) {
        cm.why_no_scribe = std::string("no per-sample export (L.gen_dev.txt and L.scribe_export/scores.jsonl) for ") +
                           (!sa.present ? "this run" : "the baseline");
    } else if (!sa.aligned || !sb.aligned) {
        cm.why_no_scribe = "the export and the kept samples do not line up (" + std::to_string(sa.kept) + "/" +
                           std::to_string(sa.scored) + ", " + std::to_string(sb.kept) + "/" + std::to_string(sb.scored) + ")";
    } else {
        for (const auto& [k, v] : sa.by_key) {
            auto it = sb.by_key.find(k);
            if (it == sb.by_key.end()) {
                ++cm.only_a;
                continue;
            }
            ds.push_back(v.first - it->second.first);
            df.push_back(v.second - it->second.second);
            by_temp[k.substr(k.find('|') + 1)].push_back(ds.back());
        }
        for (const auto& [k, v] : sb.by_key)
            if (!sa.by_key.count(k)) ++cm.only_b;
    }
    cm.n_pairs = ds.size();
    if (!ds.empty()) {
        cm.scribe = true;
        cm.d_scribe = ev::bayesian_bootstrap(ds, 0.0, {}, bo);
        cm.d_surface = ev::bayesian_bootstrap(df, 0.0, {}, bo);
        for (const auto& [k, v] : by_temp) {
            double s = 0;
            for (double x : v) s += x;
            cm.temp_gains.push_back(s / static_cast<double>(v.size()));
        }
        double s = 0;
        for (double x : ds) s += x;
        cm.dev_gain = s / static_cast<double>(ds.size());
        // LCI (rule 3): gamma only with a fresh read of both runs.
        const Rung fa = find_rung(cand.fresh_scribe, "trie samples (pooled)");
        const Rung fb = find_rung(base.fresh_scribe, "trie samples (pooled)");
        ev::LciInputs in;
        in.test_diffs = ds;
        in.eps = 0.0;
        if (cm.answers) {
            in.canaries.push_back({"answers_not_wrong", nw, {}, 0.01});
            in.canaries.push_back({"answers_right", dr, {}, 0.01});
        }
        in.gain_dev = cm.dev_gain;
        in.seed_gains = cm.temp_gains;
        in.ram = std::isfinite(cm.cand_peak) ? cm.cand_peak : 0.0;
        in.ram_budget = budget_mb;
        if (fa.found && fb.found) {
            cm.fresh_gain = fa.scribe - fb.scribe;
            in.gain_ext = cm.fresh_gain;
            cm.lci_read = true;
        }
        cm.lci = ev::compute_lci(in);
    }
    return cm;
}

void compare_rows(Card& c, const Compare& cm, const Run& cand, const Run& base) {
    const std::string sec = "Against " + base.label + " (paired, dev)";
    if (cm.answers) {
        c.add(sec, "raw right answers, difference (90%)", cm.d_right.mean, cm.d_right.lo90, cm.d_right.hi90, double(cm.n_q), "dev",
              "per T-QA dev question; lci.hpp Bayesian bootstrap", 3);
        c.add(sec, "raw wrong answers, difference (90%)", cm.d_wrong.mean, cm.d_wrong.lo90, cm.d_wrong.hi90, double(cm.n_q), "dev",
              "per T-QA dev question", 3);
    }
    if (cm.scribe) {
        c.add(sec, "SCRIBE v2, difference (90%)", cm.d_scribe.mean, cm.d_scribe.lo90, cm.d_scribe.hi90, double(cm.n_pairs), "dev",
              "per (prompt, temperature) kept in both; " + std::to_string(cm.only_a) + " only in this run, " +
                  std::to_string(cm.only_b) + " only in the baseline",
              3);
        c.add(sec, "surface S, difference (90%)", cm.d_surface.mean, cm.d_surface.lo90, cm.d_surface.hi90, double(cm.n_pairs), "dev",
              "paired as SCRIBE", 3);
    } else {
        c.rows.push_back({sec, "SCRIBE v2, difference", "n/a", "", "", "", "dev",
                          cm.why_no_scribe.empty() ? "no exported per-sample scores" : cm.why_no_scribe});
    }
    const auto qa_c = headline(cand.qa_bpc, kQaPref), qa_b = headline(base.qa_bpc, kQaPref);
    if (std::isfinite(qa_c.second) && std::isfinite(qa_b.second))
        c.add(sec, "qa holdout BPC, difference (lower is better)", qa_c.second - qa_b.second, NAN, NAN, NAN, "dev",
              "points only; no paired interval (no per-block bits)");
    const auto k_c = headline(cand.ctx_bpc, kCtxPref), k_b = headline(base.ctx_bpc, kCtxPref);
    if (std::isfinite(k_c.second) && std::isfinite(k_b.second))
        c.add(sec, "context8k BPC, difference (lower is better)", k_c.second - k_b.second, NAN, NAN, NAN, "dev", "points only");
    if (!cm.scribe) return;
    const ev::LciDecision& d = cm.lci;
    c.add("LCI against " + base.label, "P_gain (SCRIBE v2, ε 0)", d.p_gain, d.gain_lo90, d.gain_hi90, double(d.n_eff), "dev",
          "interval: the gain's 90%", 3);
    for (const auto& cr : d.canaries)
        c.add("LCI against " + base.label, "canary " + cr.name + ": P(non-inferior within 1 point)", cr.p_noninferior, NAN, NAN,
              double(cr.n_eff), "dev", cr.pass ? "passes" : "fails", 3);
    c.add("LCI against " + base.label, "r (every canary passes)", d.r, NAN, NAN, NAN, "dev", "", 0);
    c.add("LCI against " + base.label, "σ (same sign at the three temperatures, standing in for seeds)", d.sigma, NAN, NAN,
          double(cm.temp_gains.size()), "dev", "", 1);
    c.add("LCI against " + base.label, "κ (built qa peak within " + fmt(cm.budget_mb, 0) + " MB)", d.kappa, NAN, NAN, NAN, "",
          "candidate peak " + fmt(cm.cand_peak, 1) + " MB", 0);
    if (cm.lci_read) {
        c.add("LCI against " + base.label, "γ (fresh gain against dev gain)", d.gamma, NAN, NAN, NAN, "fresh",
              "fresh " + fmt(cm.fresh_gain, 3) + ", dev " + fmt(cm.dev_gain, 3), 2);
        c.add("LCI against " + base.label, "LCI" + std::string(d.n_eff_capped ? " (capped at 0.5: n_eff < 200)" : ""), d.lci, NAN,
              NAN, double(d.n_eff), "dev+fresh", d.meets_threshold ? "meets 0.95" : "below 0.95", 3);
    } else {
        c.rows.push_back({"LCI against " + base.label, "γ and LCI", "not read", "", "", "", "", "needs --fresh on both runs (rule 3)"});
    }
}

// ---------------------------------------------------------------------------- writing

std::string render_md(const Card& c, const Run& r, const std::string& baseline) {
    std::ostringstream o;
    auto m = [&](const char* k) { return r.meta.count(k) ? r.meta.at(k) : std::string("?"); };
    o << "# CyphaLM report card: " << r.label << "\n\n";
    o << "**Odin Loch** · " << m("date") << " · commit `" << m("commit") << "` · protocol `" << m("protocol") << "` · text source `"
      << m("gen_source") << "` · split **" << m("split") << "**\n\n";
    o << "Flags: `" << (m("flags").empty() ? std::string("(defaults)") : m("flags")) << "`";
    if (!baseline.empty()) o << " · baseline: `" << baseline << "`";
    o << "\n\n";
    o << "Rules: docs/reports/REPORT_CARD.md. Proportions: Wilson 95%. SCRIBE v2 and surface S: scribe's 95% bootstrap. "
         "BPC: points on the fixed holdout with the experts' paired 90% gain intervals. Comparisons: lci.hpp, 90%.\n\n";
    std::string sec;
    for (const Row& row : c.rows) {
        if (row.section != sec) {
            sec = row.section;
            o << "\n### " << sec << "\n\n| Metric | Value | Interval | n | Split | Note |\n|---|---:|---|---:|---|---|\n";
        }
        const std::string iv = (row.lo.empty() || row.lo == "n/a") ? "—" : "[" + row.lo + ", " + row.hi + "]";
        o << "| " << row.metric << " | " << row.value << " | " << iv << " | " << row.n << " | " << row.split << " | " << row.note
          << " |\n";
    }
    return o.str();
}
std::string render_tsv(const Card& c, const Run& r) {
    std::ostringstream o;
    o << "label\tcommit\tdate\tsection\tmetric\tvalue\tlo\thi\tn\tsplit\tnote\n";
    auto m = [&](const char* k) { return r.meta.count(k) ? r.meta.at(k) : std::string(""); };
    for (const Row& row : c.rows)
        o << r.label << "\t" << m("commit") << "\t" << m("date") << "\t" << row.section << "\t" << row.metric << "\t" << row.value
          << "\t" << row.lo << "\t" << row.hi << "\t" << row.n << "\t" << row.split << "\t" << row.note << "\n";
    return o.str();
}

Card build_card(const Run& r, const Run* base, double budget_mb) {
    Card c;
    bpc_rows(c, r);
    answers_rows(c, r, "CV", "ABSTAIN", "dev");
    if (r.test_read) answers_rows(c, r, "TEST_GATED", "TEST_ABSTAIN", "test");
    scribe_rows(c, r.scribe, "dev", "");
    if (!r.fresh_scribe.empty()) scribe_rows(c, r.fresh_scribe, "fresh", ", fresh prompts");
    cost_rows(c, r);
    if (base) compare_rows(c, compare(r, *base, budget_mb), r, *base);
    return c;
}

int summarize(const std::string& dir, const std::string& label, const std::string& baseline, double budget_mb) {
    const Run r = load_run(dir, label);
    if (r.qa.empty() && r.ctx.empty() && r.scribe.empty() && !r.pk_qa.found && !r.pk_ctx.found && !r.pk_gen.found) {
        std::cerr << "nothing to summarise for " << dir << "/" << label << "\n";
        return 1;
    }
    Run b;
    if (!baseline.empty()) b = load_run(dir, baseline);
    const Card c = build_card(r, baseline.empty() ? nullptr : &b, budget_mb);
    std::ofstream(dir + "/" + label + ".card.md") << render_md(c, r, baseline);
    std::ofstream(dir + "/" + label + ".card.tsv") << render_tsv(c, r);
    std::cout << render_md(c, r, baseline);
    return 0;
}

// ---------------------------------------------------------------------------- selftest (a tiny fixture)

int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(cond)) {                                                             \
            ++g_failed;                                                            \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
        }                                                                          \
    } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

void put(const std::string& path, const std::string& s) { std::ofstream(path) << s; }

std::string gen_line(const std::string& kind, const std::string& text) { return "GEN\t" + kind + "\t" + text + "\n"; }
std::string long_text(char c) { return std::string("The ") + std::string(140, c) + " ends here."; }

int selftest() {
    const fs::path d = fs::temp_directory_path() / ("cypha_report_card_selftest_" + std::to_string(std::rand()));
    fs::create_directories(d);
    const std::string D = d.string();
    // Wilson against a hand value: 2 of 130 -> [0.00423, 0.05434] (z 1.96).
    const Prop p{2, 130};
    CHECK_NEAR(p.lo(), 0.004230, 2e-5);
    CHECK_NEAR(p.hi(), 0.054344, 2e-4);
    CHECK_NEAR(Prop({0, 130}).hi(), 0.028702, 2e-5);  // qa prints wrong_rate_wilson_hi 0.0287 for 0 of 130
    // qa output: tallies, the abstain line, the BPC stack, the gains, and a test tally.
    const std::string qa =
        "holdout_mix_bpc 2.348543\n"
        "holdout2_trie_bpc 2.322445 holdout2_trie_ig_bpc 2.072881\n"
        "cls shape contexts 1 holdout2_trie_cls_bpc 2.139407 holdout2_trie_ig_cls_bpc 1.994441\n"
        "sim default keys 1 holdout2_trie_ig_cls_sim_bpc 1.839792\n"
        "sim_lci trie_ig_cls gain_bpc 0.154571 lo90 0.145523 hi90 0.164681 p_gain_gt_0.005 1.0 blocks 256\n"
        "holdout2_trie_cls_sim_copy_bpc 1.842738 holdout2_trie_ig_cls_sim_copy_bpc 1.789464 (first 524288 bytes)\n"
        "copy_lci trie_ig_cls_sim gain_bpc 0.050261 lo90 0.044289 hi90 0.057133 p 1 blocks 256\n"
        "RAW chat/trie_ig all n 130 right 7 abstained 66 wrong 57 coverage 0.4923\n"
        "CV chat/trie_ig all n 130 right 2 abstained 128 wrong 0 coverage 0.0154 sel_acc 1.0\n"
        "CV chat/trie_ig in_slice n 100 right 2 abstained 98 wrong 0 coverage 0.02\n"
        "ABSTAIN chat/trie_ig abstained 66 not_in_slice 30/30 precision_nis 0.4545 recall_nis 1.0 | in_slice_base_wrong 36/98 "
        "precision_should 1.0 | tags 49 complete 21 reason_ok 30 entity_in_question 0\n"
        "TEST_GATED chat/trie_ig all n 200 right 0 abstained 197 wrong 3 coverage 0.015\n"
        "EX trie_ig grade 2 | q => fake_bpc 9.99\n";
    put(D + "/A.qa.out", qa);
    put(D + "/A.ctx.out", "trie_only_bpc 2.3199\ntrie_plus_context_8192_cls_sim_bpc 1.8404\ntrie_plus_context_8192_cls_sim_copy_bpc 1.7897\n");
    put(D + "/A.meta.tsv", "date\t2026-10-29\ncommit\tabc1234\nprotocol\tc1l\ngen_source\ta4d\nsplit\tdev\nflags\t\n");
    put(D + "/A.qa.peak", "peak_rss_kb 184600 seconds 700.5 rc 0\n");
    put(D + "/A.scribe.md",
        "| rung | n | SCRIBE | parity | surface S | mean bytes |\n|---|---|---|---|---|---|\n"
        "| held-out human | 500 | 0.923 [0.910, 0.935] | 1.00 | 0.912 [0.906, 0.918] | 531 |\n"
        "| trie samples (pooled) | 608 | 0.063 [0.051, 0.078] | 0.07 | 0.860 [0.854, 0.867] | 325 |\n"
        "| sentences shuffled across docs | 500 | 0.344 [0.333, 0.356] | 0.37 | 0.905 [0.901, 0.908] | 532 |\n");
    const Run a = load_run(D, "A");
    const auto qh = headline(a.qa_bpc, kQaPref);
    CHECK(qh.first == "holdout2_trie_ig_cls_sim_copy_bpc");
    CHECK_NEAR(qh.second, 1.789464, 1e-9);
    CHECK(!a.qa_bpc.count("fake_bpc"));  // example lines are not read
    CHECK_NEAR(headline(a.ctx_bpc, kCtxPref).second, 1.7897, 1e-9);
    CHECK(a.gains.size() == 2 && std::fabs(a.gains[0].lo - 0.145523) < 1e-9);
    const Tally t = find_tally(a.qa, "CV", "chat/trie_ig", "all");
    CHECK(t.found && t.n == 130 && t.right == 2 && t.abst == 128 && t.wrong == 0);
    const Abstain ab = find_abstain(a.qa, "ABSTAIN", "chat/trie_ig");
    CHECK(ab.found && ab.abstained == 66 && ab.nis_hit == 30 && ab.nis_total == 30 && ab.should_hit == 36 && ab.tags == 49 &&
          ab.reason_ok == 30);
    const Rung rh = find_rung(a.scribe, "trie samples (pooled)");
    CHECK(rh.found && rh.n == 608 && std::fabs(rh.scribe - 0.063) < 1e-12 && std::fabs(rh.fhi - 0.867) < 1e-12);
    // Rule 1: a dev card never shows the test tally, even when the output holds one.
    const Card ca = build_card(a, nullptr, 200);
    bool has_test = false, has_dev = false;
    for (const Row& row : ca.rows) {
        has_test = has_test || row.split == "test";
        has_dev = has_dev || (row.section == "Answers (dev)" && row.metric.rfind("right", 0) == 0 && row.value == "0.015");
    }
    CHECK(!has_test);
    CHECK(has_dev);
    bool has_raw = false;
    for (const Row& row : ca.rows) has_raw = has_raw || (row.metric == "wrong, before the gate" && row.value == "0.438");
    CHECK(has_raw);  // 57 of 130
    Run at = a;
    at.test_read = true;
    bool shows_test = false;
    for (const Row& row : build_card(at, nullptr, 200).rows) shows_test = shows_test || (row.split == "test" && row.value == "0.015");
    CHECK(shows_test);  // wrong 3 of 200 = 0.015 on the test card
    // A part that exited non-zero is not read, and the card says it failed.
    put(D + "/F.qa.out", qa);
    put(D + "/F.qa.peak", "peak_rss_kb 90000 seconds 12.0 rc 137\n");
    bool f_read = false, f_said = false;
    for (const Row& row : build_card(load_run(D, "F"), nullptr, 200).rows) {
        f_read = f_read || row.value == "1.7895";
        f_said = f_said || row.note.find("FAILED (exit code 137)") != std::string::npos;
    }
    CHECK(!f_read && f_said);
    const std::string md = render_md(ca, a, "");
    CHECK(md.find("abc1234") != std::string::npos && md.find("1.7895") != std::string::npos);
    const std::string tsv = render_tsv(ca, a);
    CHECK(std::count(tsv.begin(), tsv.end(), '\n') == static_cast<long>(ca.rows.size()) + 1);
    // Pairing: B drops prompt 0's T0.8 sample (under 100 bytes), in the middle, so pairing by position would be wrong.
    std::string gA = "MODEL\tx\nPROMPT\t0\tp\n" + gen_line("trie_T0.6", long_text('a')) + gen_line("trie_T0.8", long_text('b')) +
                     "PROMPT\t1\tq\n" + gen_line("trie_T0.6", long_text('c')) + gen_line("trie_T0.8", long_text('d')) +
                     gen_line("trie_T1.0", "short");
    std::string gB = "MODEL\ty\nPROMPT\t0\tp\n" + gen_line("trie_T0.6", long_text('a')) + gen_line("trie_T0.8", "tiny") +
                     "PROMPT\t1\tq\n" + gen_line("trie_T0.6", long_text('c')) + gen_line("trie_T0.8", long_text('d'));
    put(D + "/A.gen_dev.txt", gA);
    put(D + "/B.gen_dev.txt", gB);
    fs::create_directories(D + "/A.scribe_export");
    fs::create_directories(D + "/B.scribe_export");
    auto sline = [](double s, double f) {
        return "{\"id\":\"x\",\"system\":\"trie\",\"scribe\":" + fmt(s, 3) + ",\"surface_s\":" + fmt(f, 3) + ",\"v1\":{}}\n";
    };
    put(D + "/A.scribe_export/scores.jsonl", "{\"id\":\"h\",\"system\":\"human\",\"scribe\":0.9}\n" + sline(0.30, 0.9) +
                                                 sline(0.20, 0.9) + sline(0.40, 0.9) + sline(0.50, 0.9));
    put(D + "/B.scribe_export/scores.jsonl", sline(0.10, 0.8) + sline(0.15, 0.8) + sline(0.05, 0.8));
    const std::vector<std::string> kb = kept_sample_keys(read_lines(D + "/B.gen_dev.txt"));
    CHECK(kb.size() == 3 && kb[1] == "1|trie_T0.6" && kb[2] == "1|trie_T0.8");
    put(D + "/A.qa.rows.tsv", "set\trow\tin_slice\tform\tlm\tgrade\n" + std::string("dev\t0\t1\tchat\ttrie_ig\t0\n") +
                                  "dev\t1\t1\tchat\ttrie_ig\t1\ndev\t1\t1\tchat\ttrie\t0\n");
    put(D + "/B.qa.rows.tsv", "set\trow\tin_slice\tform\tlm\tgrade\n" + std::string("dev\t0\t1\tchat\ttrie_ig\t2\n") +
                                  "dev\t1\t1\tchat\ttrie_ig\t1\n");
    put(D + "/B.meta.tsv", "split\tdev\n");
    const Run b = load_run(D, "B");
    const Compare cm = compare(a, b, 200);
    CHECK(cm.scribe && cm.n_pairs == 3 && cm.only_a == 1 && cm.only_b == 0);
    // Pairs (0,T0.6) 0.3-0.10, (1,T0.6) 0.4-0.15, (1,T0.8) 0.5-0.05. Pairing by position would give 0.2 instead.
    CHECK_NEAR(cm.dev_gain, (0.20 + 0.25 + 0.45) / 3.0, 1e-9);
    CHECK(cm.answers && cm.n_q == 2);
    CHECK_NEAR(cm.d_right.mean, 0.5, 0.2);    // one question goes wrong -> right
    CHECK(cm.lci.p_gain > 0.99);              // every pair gains
    CHECK(!cm.lci_read);                      // no fresh read: gamma and LCI not read
    CHECK(cm.lci.kappa == 1.0);               // 184.6 MB within 200
    fs::remove_all(d);
    std::printf("cypha_report_card selftest: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

}  // namespace rc

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "selftest") return rc::selftest();
    if (argc >= 2 && std::string(argv[1]) == "summarize") {
        std::string dir = ".", label, baseline;
        double budget = 200;
        for (int i = 2; i < argc; ++i) {
            const std::string s = argv[i];
            auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
            if (s == "--dir") dir = next();
            else if (s == "--label") label = next();
            else if (s == "--baseline") baseline = next();
            else if (s == "--budget-mb") budget = std::atof(next().c_str());
        }
        if (label.empty()) {
            std::cerr << "summarize needs --label\n";
            return 2;
        }
        return rc::summarize(dir, label, baseline, budget);
    }
    std::cerr << "usage: cypha_report_card summarize --dir DIR --label L [--baseline B] [--budget-mb 200] | selftest\n";
    return 2;
}
