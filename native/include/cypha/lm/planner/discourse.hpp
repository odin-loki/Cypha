#pragma once

/// The planner's discourse: the whole paragraph planned on fact and entity IDs (GENERATION_PLAN §13.42 Q3,
/// docs/reports/PLAN_DISCOURSE.md). Odin: "Keep it focussed but let it drift a little and long context."
///   sample --k5 TEXT [--k5-plan STOREDIR --k5-plan-model MODEL ...] --k5-plan-discourse on|SPEC
///
/// P1 (writer.hpp) writes the main entity's facts, at most four, then hands the paragraph to K5, which drifts off the
/// subject. Here:
/// 1. Linked entities. From the main entity, the store's links: the values of its facts (they are on its page), the
///    subjects whose facts have it as a value (it is on their pages), and the subjects that share one of its facts'
///    values under the same relation; then values and subjects from those, up to `hops` links. Every store row lies on its
///    subject's page, so these are the entities that co-occur with it on its pages. Each candidate gets a static fit
///    (the link's kind, K5's prompt association of its name, the topic fit of its name to the prompt, the structure
///    statistics' document PPMI with the main entity); the best `ents` are kept at hop 1, half as many at hop 2.
///    A shared value links only from the main entity and only if at most `hub` subjects share it under that relation
///    (not "country: United States" or "occupation: writer"); past hop 1, a subject that has a linked entity as a
///    value links under the same limit (not everyone "born in Cleveland"). Overview pages ("List of ...", "Politics of
///    India") are not entities.
/// 2. The paragraph. P1's plan first (make_plan, in its order), then at every step the unstated fact with the highest
///    focus weight: the entity grid's change when the fact's subject (S) and value (O) join the grid of the whole
///    paragraph so far, the LDA topic fit of its words to the whole paragraph, the prompt association, the document
///    PPMI with the main entity, the bank's relation transition, minus `w_hop` per link from the main entity. The
///    paragraph keeps to the subject, then moves a little through linked entities.
/// 3. Long context. The fill and the gate read the whole paragraph (every sentence, fact and free, joins the experts'
///    context: the grid over all sentences, the topic over the paragraph). A fact is not stated twice: not the same
///    (subject, relation), not the same (subject, value), and not one a sentence of the prompt or the paragraph already
///    holds (subject and value both in it); past P1's plan, not the same (relation, value) as a fact already written
///    either (a third subject that "is Israeli" repeats the paragraph), and not a second birth or death year of one
///    subject (the fill writes either as "Name (date - date)" and makes up the other date). A skeleton is not used
///    twice in a paragraph while the relation has another among its best. Names are written without the page title's
///    "( ... )" or ", Region" ("Alcaeus (poet)" -> "Alcaeus"), and without K5's space inside a word at a non-ASCII
///    letter ("Luleå", not "Lule å"). When the repeat ban leaves no fill (the paragraph ends as an earlier sentence
///    ended), it is lifted at the sentence edge, then wholly, as K5 lifts it. Every `gap` fact sentences (0: none, the
///    default, PLAN_DISCOURSE.md), K5 writes one free sentence; its 8 drafts are chosen by the paragraph's topic (LDA
///    over the whole paragraph), its entities (the grid's change) and the prompt association, each z-scored over the
///    drafts, minus the markup share as in K5's gate.
/// 4. Past the facts. When P1's plan runs out, the linked entities' facts go on; only when none is left does K5 write
///    the rest, with the same paragraph gate.
/// Prompts with no plan (path k5) are plain K5, as P1. The search runs on IDs; only the winners become text.
///
/// Author: Odin Loch

#include "cypha/lm/planner/writer.hpp"
#include "cypha/lm/stopwords.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cypha::cyphalm::planner {

struct DiscourseParams {
    int hops = 2;        // linked entities up to this many links from the main entity
    int ents = 12;       // linked entities kept at hop 1 (hop 2 and further: half as many)
    int per_ent = 2;     // facts per linked entity
    int main_max = 8;    // facts of the main entity in all (P1's plan first)
    int gap = 0;         // one free K5 sentence after every `gap` fact sentences (0: none until the facts run out)
    int scan = 4000;     // store rows read per entity when linking (bounds the inverse and shared links)
    int hub = 25;        // a shared value links only if at most this many subjects share it (not a country, an occupation)
    // the focus weight of a fact (step 2)
    double w_grid = 0.25, w_topic = 1, w_assoc = 1, w_cooc = 0.25, w_trans = 0.25, w_hop = 2;
    // the static fit of a linked entity by its link's kind: a value of a fact, has it as a value, shares a value
    double k_obj = 1, k_inv = 0.5, k_share = 0;
    // the free sentences' gate (z-scored over K5's drafts)
    double g_pmi = 1, g_topic = 1, g_grid = 1;

    std::string describe() const {
        char b[512];
        std::snprintf(b, sizeof b, "hops=%d,ents=%d,per_ent=%d,main_max=%d,gap=%d,scan=%d,hub=%d,w_grid=%g,w_topic=%g,w_assoc=%g,w_cooc=%g,"
                      "w_trans=%g,w_hop=%g,k_obj=%g,k_inv=%g,k_share=%g,g_pmi=%g,g_topic=%g,g_grid=%g",
                      hops, ents, per_ent, main_max, gap, scan, hub, w_grid, w_topic, w_assoc, w_cooc, w_trans, w_hop, k_obj, k_inv, k_share,
                      g_pmi, g_topic, g_grid);
        return b;
    }
};

// "on" (the defaults) or KEY=VALUE,... over DiscourseParams' names.
inline bool parse_discourse(const std::string& spec, DiscourseParams& p, std::string& err) {
    if (spec == "on" || spec == "1" || spec.empty()) return true;
    std::size_t a = 0;
    while (a <= spec.size()) {
        std::size_t e = spec.find(',', a);
        if (e == std::string::npos) e = spec.size();
        const std::string f = spec.substr(a, e - a);
        a = e + 1;
        if (f.empty() || f == "on") {
            if (e == spec.size()) break;
            continue;
        }
        const std::size_t q = f.find('=');
        if (q == std::string::npos) return err = "--k5-plan-discourse: " + f + " is not KEY=VALUE", false;
        const std::string k = f.substr(0, q), v = f.substr(q + 1);
        char* end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (v.empty() || *end) return err = "--k5-plan-discourse: " + k + " wants a number", false;
        int* ip = k == "hops" ? &p.hops : k == "ents" ? &p.ents : k == "per_ent" ? &p.per_ent : k == "main_max" ? &p.main_max
                : k == "gap" ? &p.gap : k == "scan" ? &p.scan : k == "hub" ? &p.hub : nullptr;
        double* dp = k == "w_grid" ? &p.w_grid : k == "w_topic" ? &p.w_topic : k == "w_assoc" ? &p.w_assoc : k == "w_cooc" ? &p.w_cooc
                   : k == "w_trans" ? &p.w_trans : k == "w_hop" ? &p.w_hop : k == "k_obj" ? &p.k_obj : k == "k_inv" ? &p.k_inv
                   : k == "k_share" ? &p.k_share : k == "g_pmi" ? &p.g_pmi : k == "g_topic" ? &p.g_topic : k == "g_grid" ? &p.g_grid : nullptr;
        if (ip) {
            if (x < 0 || x > 1e6 || x != std::floor(x)) return err = "--k5-plan-discourse: " + k + " wants a whole number >= 0", false;
            *ip = static_cast<int>(x);
        } else if (dp) {
            *dp = x;
        } else {
            return err = "--k5-plan-discourse: unknown key " + k, false;
        }
        if (e == spec.size()) break;
    }
    return true;
}

// z-scores of v (0 when all are equal).
inline std::vector<double> zscores(const std::vector<double>& v) {
    std::vector<double> z(v.size(), 0.0);
    if (v.size() < 2) return z;
    double mu = 0, sd = 0;
    for (const double x : v) mu += x;
    mu /= static_cast<double>(v.size());
    for (const double x : v) sd += (x - mu) * (x - mu);
    sd = std::sqrt(sd / static_cast<double>(v.size()));
    if (sd < 1e-12) return z;
    for (std::size_t i = 0; i < v.size(); ++i) z[i] = (v[i] - mu) / sd;
    return z;
}

// Whether tokens t hold the sequence n (case-insensitive).
inline bool holds_seq(const std::vector<std::string>& t, const std::vector<std::string>& n) {
    if (n.empty() || n.size() > t.size()) return false;
    for (std::size_t a = 0; a + n.size() <= t.size(); ++a) {
        bool ok = true;
        for (std::size_t k = 0; k < n.size() && ok; ++k) ok = t[a + k].size() == n[k].size() && lower_s(t[a + k]) == lower_s(n[k]);
        if (ok) return true;
    }
    return false;
}

// The name as a sentence writes it: the store's page title without a trailing "( ... )" and without a ", Region" after
// a word ("Alcaeus ( poet )" -> "Alcaeus", "Balclutha , New Zealand" -> "Balclutha"). The full title is kept if
// nothing would be left.
inline std::vector<std::string> write_name(const std::vector<std::string>& t) {
    std::vector<std::string> o = t;
    if (o.size() > 2 && o.back() == ")") {
        const auto it = std::find(o.begin() + 1, o.end(), std::string("("));
        if (it != o.end()) o.erase(it, o.end());
    }
    const auto letter = [](const std::string& w) {  // starts with a letter (ASCII or a UTF-8 lead byte)
        const auto c = w.empty() ? 0 : static_cast<unsigned char>(w[0]);
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0xC0;
    };
    const auto upper = [](const std::string& w) {
        const auto c = w.empty() ? 0 : static_cast<unsigned char>(w[0]);
        return (c >= 'A' && c <= 'Z') || c >= 0xC0;
    };
    for (std::size_t i = 1; i + 1 < o.size(); ++i)
        if (o[i] == "," && letter(o[i - 1]) && upper(o[i + 1])) {
            o.resize(i);
            break;
        }
    return o.empty() ? t : o;
}

// Which of a name's K5 tokens are written without a space before them: K5 splits a word at every non-ASCII letter
// ("Luleå" -> "Lule", "å"), and the renderer would print "Lule å". A token is glued to the one before it when the raw
// name has no space between them and one of the two is non-ASCII ("Luleå", "Östergötland", "Anaïs"); hyphens and
// initials keep K5's spacing, as everywhere else in its text. One flag per token of wkn::tokens(raw).
inline std::vector<char> name_glue(std::string_view raw) {
    std::vector<char> g;
    std::size_t end = std::string_view::npos;
    bool prev_hi = false;
    wkn::tokenize(raw, [&](std::string_view t) {
        const auto at = static_cast<std::size_t>(t.data() - raw.data());
        const bool hi = std::any_of(t.begin(), t.end(), [](char ch) { return static_cast<unsigned char>(ch) >= 0x80; });
        g.push_back(end != std::string_view::npos && at == end && (hi || prev_hi) ? 1 : 0);
        end = at + t.size(), prev_hi = hi;
    });
    return g;
}

// Birth and death years: past P1's plan, one of them per subject. The fill writes either one in a "Name (date - date)"
// shape and makes up the other date, so the second sentence repeats the first with other dates.
inline bool life_date(const std::string& rel) { return rel == "birth_year" || rel == "death_year"; }

// A Wikipedia overview page ("List of ...", "Politics of India", "Religion in India"), not an entity to write about.
inline bool overview(const std::string& n) {
    static const char* const heads[] = {
        "List of ", "Lists of ", "Outline of ", "Index of ", "Timeline of ", "History of ", "Geography of ", "Politics of ",
        "Demographics of ", "Economy of ", "Military of ", "Culture of ", "Foreign relations of ", "Communications in ",
        "Transport in ", "Transportation in ", "Religion in ", "Education in ", "Sport in ", "Tourism in ", "Music of ",
        "Cinema of ", "Cuisine of ", "Government of ", "Regions of ", "Districts of ", "Provinces of ", "Counties of ",
        "Municipalities of ", "States of ", "Islands of ", "Subdivisions of ", "Administrative divisions of ",
        "Reservoirs and dams in ", "Telecommunications in ", "Languages of ", "Flag of ", "Location of ", "Deaths in ",
        "Place of "};
    for (const char* h : heads)
        if (n.rfind(h, 0) == 0) return true;
    return false;
}

// One entity of the paragraph's plan: the main entity (hop 0) or a linked one.
struct Linked {
    std::uint32_t ent = ~0u;
    int hop = 0;
    int kind = 0;  // 0 main, 1 a value of the parent's facts, 2 has the parent as a value, 3 shares a value with the parent
    std::uint32_t parent = ~0u;  // index in the entity list
    double fit = 0;
    std::uint32_t via = ~0u;     // kinds 2 and 3: the value that links it (the parent, or the shared value); its fact first
};

// The paragraph's plan: P1's plan (spine), the entities and the fact pool.
struct DPlan {
    Plan base;
    std::vector<Linked> ents;          // [0]: the main entity
    std::vector<std::uint32_t> facts;  // the pool (fact IDs): the spine first, in its order
    std::vector<std::uint32_t> of;     // each pool fact's entity (index in ents)
    std::size_t spine = 0;             // the pool's first `spine` facts are P1's plan
};

class DiscourseWriter final : public wkn::StructureWriter {
 public:
    DiscourseWriter(std::shared_ptr<const Model> md, DiscourseParams p) : md_(std::move(md)), p_(p) {}
    ~DiscourseWriter() override {
        const Model& md = *md_;
        if (md.planned == 0) return;
        std::fprintf(stderr, "k5-plan-discourse: %llu prompts planned; %llu fact sentences (%llu from linked entities), %llu free "
                     "sentences between facts, %llu after them; repeat ban lifted for %llu facts at the sentence edge, %llu "
                     "wholly; fact steps %.2f thread-s\n",
                     static_cast<unsigned long long>(md.planned.load()), static_cast<unsigned long long>(n_fact_.load()),
                     static_cast<unsigned long long>(n_linked_.load()), static_cast<unsigned long long>(n_free_mid_.load()),
                     static_cast<unsigned long long>(n_free_end_.load()), static_cast<unsigned long long>(n_lift_edge_.load()),
                     static_cast<unsigned long long>(n_lift_all_.load()), static_cast<double>(md.steps_ns.load()) / 1e9);
    }

    // Step 1: the linked entities and the fact pool.
    DPlan plan(const wkn::K5& m, std::string_view prompt, const std::vector<double>& assoc, const std::vector<double>& theta) const {
        const Model& md = *md_;
        const auto& fs = md.fs;
        DPlan dp;
        dp.base = make_plan(md, m, prompt, assoc);
        if (dp.base.path == "k5") return dp;
        const std::uint32_t main = dp.base.ent;
        std::unordered_map<std::uint32_t, std::size_t> at;  // entity -> index in ents
        const auto add_ent = [&](const Linked& l) {
            at[l.ent] = dp.ents.size();
            dp.ents.push_back(l);
        };
        add_ent({main, 0, 0, ~0u, 0, ~0u});
        // P1's plan is the spine. In the co-occurring path its facts' subjects are linked entities (kind 2: they have the
        // main entity as a value); their own links start at hop 2.
        std::vector<std::size_t> cooc_subj;
        for (const auto fi : dp.base.facts) {
            const std::uint32_t sj = fs.fact(fi).subj;
            if (!at.count(sj)) cooc_subj.push_back(dp.ents.size()), add_ent({sj, 1, 2, 0, 0, main});
            dp.facts.push_back(fi), dp.of.push_back(static_cast<std::uint32_t>(at[sj]));
        }
        dp.spine = dp.facts.size();
        // Linked entities, hop by hop (integer lookups on the store's subject and value indexes).
        std::vector<std::size_t> frontier = {0};
        for (int hop = 1; hop <= p_.hops && !frontier.empty(); ++hop) {
            std::map<std::uint32_t, Linked> cand;  // entity -> its best link
            const auto offer = [&](std::uint32_t e, int kind, std::size_t parent, std::uint32_t via) {
                if (at.count(e) || numeric(e) || overview(fs.ent(e)) || !plannable(e)) return;
                const double k = kind == 1 ? p_.k_obj : kind == 2 ? p_.k_inv : p_.k_share;
                const double fit = k + p_.w_assoc * name_assoc(m, e, assoc) + p_.w_topic * name_topic(e, theta) + p_.w_cooc * cooc(e, main);
                auto it = cand.find(e);
                if (it == cand.end() || fit > it->second.fit) cand[e] = {e, hop, kind, static_cast<std::uint32_t>(parent), fit, via};
            };
            for (const std::size_t pi : frontier) {
                const std::uint32_t pe = dp.ents[pi].ent;
                for (const auto fi : fs.of_subj(pe)) {  // its facts' values, and (from the main entity) the subjects sharing a value
                    const auto& f = fs.fact(fi);
                    offer(f.val, 1, pi, ~0u);
                    // A shared value links only at hop 1 (two weak links in a row drift off the subject: Gustav
                    // Radbruch -> "occupation: Jurist" -> Clarence Thomas), and only if few subjects share it under the
                    // relation: sharing "country: United States" or "occupation: writer" with hundreds of subjects says
                    // nothing about the subject. The value's whole row list is counted (a cut-off count let 35 states
                    // that are "located in the United States" through as fewer than `hub`).
                    if (hop > 1 || numeric(f.val)) continue;
                    std::set<std::uint32_t> share;
                    for (const auto fj : fs.of_val(f.val)) {
                        if (fs.fact(fj).rel == f.rel && fs.fact(fj).subj != pe) share.insert(fs.fact(fj).subj);
                        if (share.size() > static_cast<std::size_t>(p_.hub)) break;
                    }
                    if (share.size() > static_cast<std::size_t>(p_.hub)) continue;
                    for (const auto sj : share) offer(sj, 3, pi, f.val);
                }
                // The subjects that have it as a value. Past hop 1, under a relation where at most `hub` subjects have
                // it: Avery Hopwood -> Cleveland -> "born in Cleveland" -> Halle Berry is the shared-value link again,
                // two links long. (From the main entity all count: "nationality: Israeli" is the prompt's subject.)
                std::map<std::uint32_t, std::size_t> per_rel;
                if (hop > 1)
                    for (const auto fi : fs.of_val(pe)) ++per_rel[fs.fact(fi).rel];
                int n = 0;
                for (const auto fi : fs.of_val(pe)) {
                    if (++n > p_.scan) break;
                    if (hop > 1 && per_rel[fs.fact(fi).rel] > static_cast<std::size_t>(p_.hub)) continue;
                    offer(fs.fact(fi).subj, 2, pi, pe);
                }
            }
            std::vector<Linked> v;
            for (const auto& [e, l] : cand) v.push_back(l);
            std::stable_sort(v.begin(), v.end(), [](const Linked& a, const Linked& b) { return a.fit > b.fit; });
            const std::size_t keep = static_cast<std::size_t>(hop == 1 ? p_.ents : std::max(1, p_.ents / 2));
            frontier.clear();
            if (hop == 1) frontier = cooc_subj;
            for (std::size_t i = 0; i < v.size() && i < keep; ++i) {
                frontier.push_back(dp.ents.size());
                add_ent(v[i]);
            }
        }
        // The pool: the main entity's other facts, then each linked entity's (the fact that links it first).
        std::set<std::uint32_t> in(dp.facts.begin(), dp.facts.end());
        for (std::size_t i = 0; i < dp.ents.size(); ++i) {
            std::size_t have = 0;
            for (std::size_t j = 0; j < dp.facts.size(); ++j) have += dp.of[j] == i;
            const std::size_t cap = static_cast<std::size_t>(i == 0 ? p_.main_max : p_.per_ent);
            for (const auto fi : entity_facts(m, dp.ents[i].ent, assoc, dp.ents[i].via)) {
                if (have >= cap) break;
                if (in.count(fi)) continue;
                in.insert(fi);
                dp.facts.push_back(fi), dp.of.push_back(static_cast<std::uint32_t>(i));
                ++have;
            }
        }
        return dp;
    }

    std::string generate(const wkn::K5& m, std::string_view prompt, std::uint64_t seed, const wkn::Options& o, wkn::GenStats* gst) const override {
        if (m.vocab().size() <= 2) return {};
        const Model& md = *md_;
        wkn::K5::State s = m.start(prompt, o);
        const std::vector<double> assoc = m.prompt_assoc(s.ctx);
        st::Context c;
        c.sents = st::split(wkn::tokens(prompt));
        c.n_prompt = c.sents.size();
        c.prompt_chars = prompt.size();
        const st::Experts ex(md.stats, m, md.runs(m), &md.topics, &md.grid);
        ex.prepare(c);  // the prompt's topic mixture, for the linked entities' fit
        const DPlan dp = plan(m, prompt, assoc, c.theta);
        if (dp.base.path == "k5") {  // as P1: plain K5
            wkn::Options o2 = o;
            o2.structure = nullptr;
            std::string out = m.generate(prompt, seed, o2, gst);
            log(prompt, dp, {}, {}, {}, 0, 0);
            return out;
        }
        wkn::GenStats g;
        const auto t_steps = std::chrono::steady_clock::now();
        FillOpts hfo;  // GENERATION_PLAN §13.48 H1: only --k5-plan-honest reaches the discourse writer's fill
        hfo.honest = md.fo.honest;
        const Filler fl(m, o, md.beam, 3, md.fill_dba, hfo.honest ? &hfo : nullptr);
        const auto& fs = md.fs;
        std::vector<char> used(dp.facts.size(), 0);
        // (subject, relation), (subject, value), and past P1's plan (relation, value): a third subject that "is Israeli"
        // or "is in New Zealand" repeats the paragraph rather than saying something new
        std::set<std::pair<std::uint32_t, std::uint32_t>> said_rel, said_val, said_rv;
        std::set<std::uint32_t> said_life;  // subjects with a birth or death year written
        const auto said = [&](std::size_t j) {
            const auto& f = fs.fact(dp.facts[j]);
            if (said_rel.count({f.subj, f.rel}) || said_val.count({f.subj, f.val})) return true;
            if (j >= dp.spine && said_rv.count({f.rel, f.val})) return true;
            if (j >= dp.spine && said_life.count(f.subj) && life_date(fs.rel(f.rel))) return true;
            const auto S = write_tok(f.subj);
            const auto O = write_tok(f.val);
            for (const auto& z : c.sents)
                if (holds_seq(z.tok, S) && holds_seq(z.tok, O)) return true;
            return false;
        };
        std::set<std::vector<std::uint16_t>> used_shape;  // the skeletons written in this paragraph (their class codes)
        std::vector<std::size_t> end_at, written;  // each fact sentence's end offset; its pool index
        std::vector<std::string> sent_tok;
        std::vector<Cand> cands;
        std::size_t next_spine = 0;
        std::uint32_t prev_subj = ~0u;
        std::string prev_rel = "<s>";
        int since_free = 0, n_mid = 0, n_end = 0;
        std::uint64_t free_no = 1000;  // the free sentences' streams (K5's draft numbers, as P1's tail)
        const int K = std::max(o.gate_k, 1);
        std::vector<wkn::K5::State> drafts(static_cast<std::size_t>(K));
        const pp::ParaPlan* plan = o.para_plan.plan.get();  // PARA_PLAN.md: the paragraph-plan term (null: off)
        const pp::ParaPlan::Track ptr = plan ? plan->begin(prompt) : pp::ParaPlan::Track{};
        while (s.out.size() < o.bytes && s.step < o.max_words) {
            ex.prepare(c);
            // the next fact: P1's plan in its order, then the highest focus weight
            std::size_t pick = dp.facts.size();
            while (next_spine < dp.spine && pick == dp.facts.size()) {
                if (!used[next_spine] && !said(next_spine)) pick = next_spine;
                else used[next_spine] = 1;
                ++next_spine;
            }
            if (pick == dp.facts.size()) {
                double best = -1e300;
                for (std::size_t j = dp.spine; j < dp.facts.size(); ++j) {
                    if (used[j]) continue;
                    if (said(j)) {
                        used[j] = 1;
                        continue;
                    }
                    const double w = focus(m, c, assoc, dp, j, prev_subj, prev_rel);
                    if (w > best) best = w, pick = j;
                }
            }
            if (pick == dp.facts.size()) break;  // no fact left
            used[pick] = 1;
            const std::uint32_t fi = dp.facts[pick];
            const auto& f = fs.fact(fi);
            const auto* sk = md.bank.of(fs.rel(f.rel));
            if (!sk) continue;
            const auto S = write_tok(f.subj);
            const auto O = write_tok(f.val);
            // a skeleton is not used twice in one paragraph while the relation has another among its best
            auto pk = md.best_skeletons(m, c, *sk, S, O, static_cast<std::size_t>(kSkel) + used_shape.size());
            std::vector<std::size_t> fresh;
            for (const auto i : pk)
                if (!used_shape.count((*sk)[i].codes) && fresh.size() < static_cast<std::size_t>(kSkel)) fresh.push_back(i);
            if (!fresh.empty()) pk.swap(fresh);
            else if (pk.size() > static_cast<std::size_t>(kSkel)) pk.resize(static_cast<std::size_t>(kSkel));
            md.candidates(m, ex, fl, c, s, assoc, *sk, pk, S, O, cands);
            if (cands.empty()) {
                // The repeat ban reads the text's last 11 bytes and the fill's first byte together. When the paragraph
                // ends as an earlier sentence ended ("..., California." twice), every word's leading space is banned
                // and no skeleton can be filled. K5's own sampler lifts the ban when it leaves nothing; here the ban
                // is first lifted across the sentence boundary only (a marker after the text), then wholly.
                wkn::K5::State s2 = s;
                s2.rep = wkn::RepeatBan(std::string(prompt) + s.out + '\x01');  // the ban's text: the prompt, then the output
                md.candidates(m, ex, fl, c, s2, assoc, *sk, pk, S, O, cands);
                if (cands.empty()) {
                    s2.rep = wkn::RepeatBan();
                    md.candidates(m, ex, fl, c, s2, assoc, *sk, pk, S, O, cands);
                    if (!cands.empty()) ++n_lift_all_;
                } else {
                    ++n_lift_edge_;
                }
            }
            if (cands.empty()) continue;
            const std::size_t b = md.choose(c, cands);
            const auto& win = cands[b].fill;
            // the names' tokens that are written without a space (name_glue), placed by the skeleton's slots
            std::vector<char> glue(win.tok.size(), 0);
            {
                const auto gS = name_glue(fs.ent(f.subj)), gO = name_glue(fs.ent(f.val));
                std::size_t at = 0;
                bool fits = true;
                for (const auto code : (*sk)[cands[b].skel].codes) {
                    const bool slot = code == kSlotS || code == kSlotO;
                    const auto& nm = code == kSlotS ? S : O;
                    const auto& gl = code == kSlotS ? gS : gO;
                    const std::size_t n = slot ? nm.size() : 1;
                    if (at + n > glue.size()) {
                        fits = false;
                        break;
                    }
                    // (the dba fill may write a name before its slot: then the tokens there are not the name's, no glue)
                    if (slot && std::equal(nm.begin(), nm.end(), win.tok.begin() + static_cast<std::ptrdiff_t>(at)))
                        for (std::size_t k = 1; k < n && k < gl.size(); ++k) glue[at + k] = gl[k];
                    at += n;
                }
                if (!fits || at != glue.size()) std::fill(glue.begin(), glue.end(), 0);
            }
            for (std::size_t i = 0; i < win.tok.size(); ++i) {  // only the winner becomes text
                const std::string pc = glue[i] ? win.tok[i] : wkn::piece(win.tok[i], s.prev, s.has_prev);
                s.out += pc;
                s.rep.add(pc);
                wkn::K5::learn(s, win.ids[i], o);
                s.prev = win.tok[i], s.has_prev = true;
                ++s.step, ++g.steps;
            }
            g.gate_cands += cands.size(), ++g.gate_sents;
            st::Sent z;
            z.tok = win.tok, z.ended = true;
            c.sents.push_back(std::move(z));
            said_rel.insert({f.subj, f.rel}), said_val.insert({f.subj, f.val}), said_rv.insert({f.rel, f.val});
            if (life_date(fs.rel(f.rel))) said_life.insert(f.subj);
            used_shape.insert((*sk)[cands[b].skel].codes);
            prev_subj = f.subj, prev_rel = fs.rel(f.rel);
            end_at.push_back(s.out.size());
            written.push_back(pick);
            std::string t;
            for (const auto& x : win.tok) t += (t.empty() ? "" : " ") + x;
            sent_tok.push_back(t);
            ++n_fact_;
            if (dp.ents[dp.of[pick]].hop > 0) ++n_linked_;
            if (p_.gap > 0 && ++since_free >= p_.gap && s.out.size() < o.bytes && s.step < o.max_words) {
                ex.prepare(c);
                free_sentence(m, c, s, seed, free_no++, o, assoc, drafts, g, plan ? &ptr : nullptr);
                since_free = 0, ++n_mid;
            }
        }
        md.steps_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_steps).count());
        md.dists += fl.dists();
        ++md.planned;
        // past the facts: K5's free sentences, chosen with the whole paragraph
        while (s.step < o.max_words && s.out.size() < o.bytes) {
            ex.prepare(c);
            free_sentence(m, c, s, seed, free_no++, o, assoc, drafts, g, plan ? &ptr : nullptr);
            ++n_end;
        }
        n_free_mid_ += static_cast<std::uint64_t>(n_mid), n_free_end_ += static_cast<std::uint64_t>(n_end);
        std::string out = std::move(s.out);
        wkn::K5::cut_bytes(out, o.bytes);
        std::vector<int> stated;
        for (const auto e : end_at) stated.push_back(e <= out.size() ? 1 : 0);
        log(prompt, dp, written, stated, sent_tok, n_mid, n_end);
        if (gst) gst->add(g);
        return out;
    }

    const DiscourseParams& params() const { return p_; }

 private:
    // A year or a number (not an entity to link through).
    bool numeric(std::uint32_t e) const {
        const auto& t = md_->fs.ent_tok(e);
        return t.empty() || (!t[0].empty() && t[0][0] >= '0' && t[0][0] <= '9');
    }
    // The name a sentence writes (write_name on the store's tokens).
    std::vector<std::string> write_tok(std::uint32_t e) const { return numeric(e) ? md_->fs.ent_tok(e) : write_name(md_->fs.ent_tok(e)); }
    // An entity with at least one fact under a relation of the skeleton bank.
    bool plannable(std::uint32_t e) const {
        const Model& md = *md_;
        for (const auto fi : md.fs.of_subj(e))
            if (md.bank.of(md.fs.rel(md.fs.fact(fi).rel))) return true;
        return false;
    }
    // Mean K5 prompt association of a token sequence.
    static double tok_assoc(const wkn::K5& m, const std::vector<std::string>& t, const std::vector<double>& assoc) {
        if (t.empty() || assoc.empty()) return 0;
        double s = 0;
        for (const auto& w : t) s += assoc[m.id(w)];
        return s / static_cast<double>(t.size());
    }
    double name_assoc(const wkn::K5& m, std::uint32_t e, const std::vector<double>& assoc) const { return tok_assoc(m, md_->fs.ent_tok(e), assoc); }
    // Mean LDA log ratio of the tokens' content words under theta (0 without LDA or content words).
    double tok_topic(const std::vector<std::string>& t, const std::vector<double>& theta) const {
        const auto& L = md_->topics;
        if (!L.trained() || theta.empty()) return 0;
        const std::size_t n = L.content_count(t);
        return n ? L.log_ratio(theta, t) / static_cast<double>(n) : 0.0;
    }
    double name_topic(std::uint32_t e, const std::vector<double>& theta) const { return tok_topic(md_->fs.ent_tok(e), theta); }
    // Document PPMI of two entities' names in the structure statistics (0 if either is not a name with an id).
    double cooc(std::uint32_t a, std::uint32_t b) const {
        if (a == b) return 0;
        const auto& stt = md_->stats;
        const auto join = [&](std::uint32_t e) {
            std::string o;
            for (const auto& t : md_->fs.ent_tok(e)) o += (o.empty() ? "" : " ") + t;
            return o;
        };
        const st::NameStat* x = stt.name(join(a));
        const st::NameStat* y = stt.name(join(b));
        if (!x || !y || x->id < 0 || y->id < 0) return 0;
        return stt.name_ppmi(x->id, y->id, x->docs, y->docs);
    }
    // An entity's facts as make_plan takes them: one per bank relation (relations with more than two values left out),
    // the Wikidata-checked or text one first; ordered by the bank's transition from a document's start, then the
    // prompt association of the value. A fact whose value is `via` (the link to the paragraph) comes first.
    std::vector<std::uint32_t> entity_facts(const wkn::K5& m, std::uint32_t e, const std::vector<double>& assoc, std::uint32_t via = ~0u) const {
        const Model& md = *md_;
        const auto& fs = md.fs;
        std::map<std::uint32_t, std::vector<std::uint32_t>> by;
        for (const auto fi : fs.of_subj(e))
            if (md.bank.of(fs.rel(fs.fact(fi).rel))) by[fs.fact(fi).rel].push_back(fi);
        std::vector<std::pair<double, std::uint32_t>> pick;
        for (const auto& [r, v] : by) {
            std::set<std::uint32_t> vals;
            for (const auto fi : v) vals.insert(fs.fact(fi).val);
            if (vals.size() > 2) continue;
            std::uint32_t best = v[0];
            for (const auto fi : v) {
                const auto& f = fs.fact(fi);
                const auto& bb = fs.fact(best);
                if (via != ~0u && (f.val == via) != (bb.val == via)) {
                    if (f.val == via) best = fi;
                    continue;
                }
                if ((f.wd == 1 && bb.wd != 1) || (f.wd == bb.wd && f.text && !bb.text)) best = fi;
            }
            const double first = via != ~0u && fs.fact(best).val == via ? 1e6 : 0.0;
            pick.push_back({first + md.bank.trans_lp("<s>", fs.rel(r)) + tok_assoc(m, fs.ent_tok(fs.fact(best).val), assoc), best});
        }
        std::stable_sort(pick.begin(), pick.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::uint32_t> out;
        for (const auto& [sc, fi] : pick) out.push_back(fi);
        return out;
    }
    // Step 2's focus weight of pool fact j after context c (prepared).
    double focus(const wkn::K5& m, const st::Context& c, const std::vector<double>& assoc, const DPlan& dp, std::size_t j,
                 std::uint32_t prev_subj, const std::string& prev_rel) const {
        const Model& md = *md_;
        const auto& fs = md.fs;
        const auto& f = fs.fact(dp.facts[j]);
        const auto S = write_tok(f.subj);
        const auto O = write_tok(f.val);
        double grid = 0;
        if (md.grid.trained()) {  // the fact's subject as S and its value (a name) as O
            eg::SentRoles r;
            if (!S.empty() && words::alpha(S.back())) r.emplace_back(words::stem(S.back()), eg::kS);
            if (!O.empty() && words::alpha(O.back()) && words::capital(O[0])) {
                const std::string k = words::stem(O.back());
                if (r.empty() || r[0].first != k) r.emplace_back(k, eg::kO);
            }
            std::sort(r.begin(), r.end());
            grid = md.grid.append_logp(c.grid, r);
        }
        std::vector<std::string> both = S;
        both.insert(both.end(), O.begin(), O.end());
        const double topic = tok_topic(both, c.theta);
        const double as = tok_assoc(m, both, assoc);
        const double co = cooc(f.subj, dp.ents[0].ent);
        const double tr = md.bank.trans_lp(f.subj == prev_subj ? prev_rel : std::string("<s>"), fs.rel(f.rel));
        return p_.w_grid * grid + p_.w_topic * topic + p_.w_assoc * as + p_.w_cooc * co + p_.w_trans * tr -
               p_.w_hop * static_cast<double>(dp.ents[dp.of[j]].hop);
    }
    // One free K5 sentence: K drafts from s, the paragraph gate's choice (c prepared); s and c take it.
    void free_sentence(const wkn::K5& m, st::Context& c, wkn::K5::State& s, std::uint64_t seed, std::uint64_t no, const wkn::Options& o,
                       const std::vector<double>& assoc, std::vector<wkn::K5::State>& drafts, wkn::GenStats& g,
                       const pp::ParaPlan::Track* ptr = nullptr) const {
        const Model& md = *md_;
        const int K = static_cast<int>(drafts.size());
        std::vector<double> pmi(static_cast<std::size_t>(K)), topic(pmi.size()), grid(pmi.size()), mk(pmi.size()), rec(pmi.size());
        std::vector<double> ar;  // K5_COHERENCE.md (--k5-gate K,wr=W): the association with the previous sentence
        if (o.gate_wr != 0 && !c.sents.empty()) {
            std::vector<std::uint32_t> prev;
            for (const auto& t : c.sents.back().tok) prev.push_back(m.id(t));
            ar = m.prompt_assoc(prev);
        }
        const pp::ParaPlan* plan = ptr ? o.para_plan.plan.get() : nullptr;  // PARA_PLAN.md
        std::vector<double> para(plan ? pmi.size() : 0);
        const pp::ParaPlan::Prev pprev = plan ? plan->prev(*ptr, s.out) : pp::ParaPlan::Prev{};
        std::vector<std::vector<std::string>> toks(pmi.size());
        std::vector<wkn::GenStats> gs(pmi.size());
        std::vector<std::uint32_t> words;
        std::vector<double> lp;
        for (int k = 0; k < K; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            m.draft(s, seed, no, k, o, drafts[ku], words, lp, gs[ku]);
            const auto& out = drafts[ku].out;
            std::size_t nm = 0;
            for (std::size_t i = s.out.size(); i < out.size(); ++i) nm += std::string_view("[]{}|=<>&*#").find(out[i]) != std::string_view::npos;
            mk[ku] = static_cast<double>(nm) / static_cast<double>(std::max<std::size_t>(out.size() - s.out.size(), 1));
            pmi[ku] = st::mean_assoc(assoc, words);
            if (!ar.empty()) rec[ku] = st::mean_assoc(ar, words);
            for (const auto w : words) toks[ku].push_back(m.vocab()[w]);
            topic[ku] = tok_topic(toks[ku], c.theta);
            grid[ku] = md.grid.trained() ? md.grid.append_logp(c.grid, md.grid.roles(toks[ku])) : 0.0;
            if (plan) para[ku] = plan->term(*ptr, pprev, std::string_view(out).substr(s.out.size()), o.para_plan.orph, o.para_plan.lead);
        }
        const auto zp = zscores(pmi), zt = zscores(topic), zg = zscores(grid);
        const auto zr = ar.empty() ? std::vector<double>{} : zscores(rec);
        const auto zq = plan ? zscores(para) : std::vector<double>{};
        int best = 0;
        double bsc = -1e300;
        for (int k = 0; k < K; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            double sc = p_.g_pmi * zp[ku] + p_.g_topic * zt[ku] + p_.g_grid * zg[ku] - o.gate_wm * mk[ku];
            if (!ar.empty()) sc += o.gate_wr * zr[ku];  // K5_COHERENCE.md
            if (plan) sc += o.para_plan.w * zq[ku];  // PARA_PLAN.md
            if (k == 0 || sc > bsc) bsc = sc, best = k;
        }
        const auto bu = static_cast<std::size_t>(best);
        g.add(gs[bu]);
        ++g.gate_sents;
        st::Sent z;
        z.tok = toks[bu];
        z.ended = !z.tok.empty() && (z.tok.back() == "." || z.tok.back() == "!" || z.tok.back() == "?");
        if (!z.tok.empty()) c.sents.push_back(std::move(z));
        s = std::move(drafts[bu]);
    }
    // The PLAN line (P1's columns, so k5_planner_measures.py reads it), then one field "D|hops|entities|free|links" with
    // each written fact's hop, the linked entities (;; between), the free sentences between and after the facts, and
    // each linked entity's hop:kind:parent:fit (kind 1 a value of the parent's facts, 2 has the parent as a value,
    // 3 shares a value with it; parent an index into the entity list, 0 the main entity).
    void log(std::string_view prompt, const DPlan& dp, const std::vector<std::size_t>& written, const std::vector<int>& stated,
             const std::vector<std::string>& sents, int n_mid, int n_end) const {
        const Model& md = *md_;
        const auto& pl = dp.base;
        std::string line = "PLAN\t" + log_clean(std::string(prompt.substr(0, 60))) + "\t" + pl.path + "\t" +
                           (pl.ent == ~0u ? std::string("-") : log_clean(md.fs.ent(pl.ent))) + "\t" + std::to_string(dp.facts.size()) + "\t" +
                           std::to_string(written.size()) + "\t" + std::to_string(std::accumulate(stated.begin(), stated.end(), 0)) + "\t" +
                           std::to_string(pl.orders);
        std::string hops;
        for (std::size_t i = 0; i < written.size(); ++i) {
            const auto& f = md.fs.fact(dp.facts[written[i]]);
            line += "\t" + log_clean(md.fs.ent(f.subj)) + "|" + md.fs.rel(f.rel) + "|" + log_clean(md.fs.ent(f.val)) + "|" + std::to_string(stated[i]) +
                    "|" + std::to_string(f.wd) + "|" + (f.text ? "text" : "wd") + "|" + log_clean(i < sents.size() ? sents[i] : "");
            hops += (i ? "," : "") + std::to_string(dp.ents[dp.of[written[i]]].hop);
        }
        if (pl.path != "k5") {
            std::string ents, why;
            for (std::size_t i = 1; i < dp.ents.size(); ++i) {
                const auto& l = dp.ents[i];
                char b[64];
                std::snprintf(b, sizeof b, "%d:%d:%u:%.2f", l.hop, l.kind, l.parent, l.fit);
                ents += (i > 1 ? ";;" : "") + log_clean(md.fs.ent(l.ent));
                why += (i > 1 ? ";;" : "") + std::string(b);
            }
            line += "\tD|" + hops + "|" + ents + "|" + std::to_string(n_mid) + "," + std::to_string(n_end) + "|" + why;
        }
        std::lock_guard<std::mutex> lk(log_mutex());
        static std::FILE* fp = [] {
            const char* p = std::getenv("K5_PLAN_LOG");
            std::FILE* f = p ? std::fopen(p, "w") : nullptr;
            return f ? f : stderr;
        }();
        std::fprintf(fp, "%s\n", line.c_str());
        std::fflush(fp);
    }

    std::shared_ptr<const Model> md_;
    DiscourseParams p_;
    mutable std::atomic<std::uint64_t> n_fact_{0}, n_linked_{0}, n_free_mid_{0}, n_free_end_{0}, n_lift_edge_{0}, n_lift_all_{0};
};

// cyphalm_trie --k5-plan-discourse on|SPEC, after apply_plan_flags: the planner's writer becomes the discourse writer
// over the same model. False (err) if the planner is not on or SPEC is bad.
inline bool apply_discourse_flag(const std::string& spec, wkn::Options& o, std::string& err) {
    DiscourseParams p;
    if (!parse_discourse(spec, p, err)) return false;
    const auto* w = dynamic_cast<const Writer*>(o.structure.get());
    if (!w) return err = "--k5-plan-discourse needs the planner (--k5-plan, or its default files)", false;
    std::shared_ptr<const Model> md = w->model();
    o.structure = std::make_shared<DiscourseWriter>(md, p);
    o.structure_desc += ",discourse=" + p.describe();
    std::fprintf(stderr, "k5-plan-discourse: %s\n", p.describe().c_str());
    return true;
}

// Selftest: the spec parser, z-scores and the sequence match.
inline bool discourse_self_test(std::string& err) {
    DiscourseParams p;
    if (!parse_discourse("on", p, err) || p.hops != 2) return err = "discourse: on", false;
    if (!parse_discourse("hops=3,gap=2,w_hop=0.5", p, err) || p.hops != 3 || p.gap != 2 || p.w_hop != 0.5) return err = "discourse: spec", false;
    DiscourseParams q;
    std::string e2;
    if (parse_discourse("hops=1.5", q, e2) || parse_discourse("nope=1", q, e2) || parse_discourse("gap", q, e2)) return err = "discourse: bad spec accepted", false;
    const auto z = zscores({1, 2, 3});
    if (std::abs(z[0] + z[2]) > 1e-12 || std::abs(z[1]) > 1e-12 || z[2] <= 0 || zscores({2, 2})[0] != 0) return err = "discourse: zscores", false;
    if (!holds_seq({"In", "new", "york", "City"}, {"New", "York"}) || holds_seq({"New"}, {"New", "York"})) return err = "discourse: holds_seq", false;
    using V = std::vector<std::string>;
    if (write_name({"Alcaeus", "(", "poet", ")"}) != V{"Alcaeus"} || write_name({"Balclutha", ",", "New", "Zealand"}) != V{"Balclutha"} ||
        write_name({"John", "Stuart", ",", "3rd", "Earl"}) != V{"John", "Stuart", ",", "3rd", "Earl"} || write_name({"(", "x", ")"}) != V{"(", "x", ")"} ||
        write_name({"New", "York"}) != V{"New", "York"})
        return err = "discourse: write_name", false;
    if (!overview("List of Israeli musical artists") || !overview("Religion in India") || overview("University of Auckland") || overview("India"))
        return err = "discourse: overview", false;
    using G = std::vector<char>;
    if (name_glue("Lule\xc3\xa5 University") != G{0, 1, 0} || name_glue("Ana\xc3\xafs Nin") != G{0, 1, 1, 0} ||
        name_glue("Rhein-Neckar-Kreis") != G{0, 0, 0, 0, 0} || name_glue("L. L. Zamenhof") != G{0, 0, 0, 0, 0})
        return err = "discourse: name_glue", false;
    if (!life_date("birth_year") || !life_date("death_year") || life_date("birth_place")) return err = "discourse: life_date", false;
    return true;
}

}  // namespace cypha::cyphalm::planner
