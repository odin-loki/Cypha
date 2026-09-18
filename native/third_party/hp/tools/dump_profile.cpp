// dump_profile.cpp — diagnostic dump of experts + layer-1 mixer dots + cheap context.
// Not the codec. Same flags as the champ binary.
//
// Output int16 LE records:
//   [n_exp stretched expert_p] [n_gates layer1 dots] [c0] [wiki] [mlen] [entropy] [bit]
// Sidecar JSON names next to the .i16.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hp/predictor.hpp"

static void push_cm(std::vector<std::string>* n, const char* name) {
    n->push_back(std::string(name) + ":ind");
    if (HP_PY_EXPERT) n->push_back(std::string(name) + ":py");
}

static std::vector<std::string> expert_names() {
    std::vector<std::string> n;
    push_cm(&n, "o1");
    push_cm(&n, "o2");
    push_cm(&n, "o3");
    push_cm(&n, "o4");
    push_cm(&n, "o6");
    push_cm(&n, "word");
    push_cm(&n, "sp13");
    push_cm(&n, "sp24");
    push_cm(&n, "col");
    push_cm(&n, "tag");
    push_cm(&n, "wbi");
#if HP_WORD_STREAMS
    push_cm(&n, "wstr");
#endif
#if HP_BRACKET
    push_cm(&n, "brk");
#endif
#if HP_LINKWORD
    push_cm(&n, "link");
#endif
#if HP_NUMERIC
    push_cm(&n, "num");
#endif
#if HP_PAT_MODEL
    push_cm(&n, "pat");
#endif
#if HP_PPMD
    push_cm(&n, "ppm");
#endif
#if HP_STEMMER
    push_cm(&n, "stem0");
#if HP_STEMMER_N >= 2
    push_cm(&n, "stem1");
#endif
#endif
#if HP_SENWORD
    push_cm(&n, "sen");
#endif
#if HP_SENT_STREAM
    push_cm(&n, "sentst");
#endif
#if HP_SENT_MEM
    push_cm(&n, "sentmem");
#endif
#if HP_SENGRP_MOD
    push_cm(&n, "sengrp");
#endif
#if HP_NEST_MOD
    push_cm(&n, "nest");
#endif
#if HP_PARA_MOD
    push_cm(&n, "para");
#endif
#if HP_LINE_MOD
    push_cm(&n, "line");
#endif
#if HP_STATE_MOD
    push_cm(&n, "state");
#endif
#if HP_DOM_MOD
    push_cm(&n, "dom");
#endif
#if HP_HDR_MOD
    push_cm(&n, "hdr");
#endif
#if HP_DEPTH_MOD
    push_cm(&n, "depth");
#endif
#if HP_FCCXT_MOD
    push_cm(&n, "fccxt");
#endif
#if HP_TPLNAME_MOD
    push_cm(&n, "tpl");
#endif
#if HP_INFOKEY_MOD
    push_cm(&n, "infokey");
#endif
#if HP_BARIDX_MOD
    push_cm(&n, "baridx");
#endif
#if HP_PERIOD_MOD
    push_cm(&n, "period");
#endif
#if HP_PRONOUN_MOD
    push_cm(&n, "pronoun");
#endif
#if HP_HASH2_O6
    push_cm(&n, "o6b");
#endif
#if HP_LINKPIPE_MOD
    push_cm(&n, "linkpipe");
#endif
#if HP_CITE_MOD
    push_cm(&n, "cite");
#endif
#if HP_CAT_MOD
    push_cm(&n, "cat");
#endif
#if HP_REDIR_MOD
    push_cm(&n, "redir");
#endif
#if HP_HEADING_MOD
    push_cm(&n, "heading");
#endif
#if HP_EXTLINK_MOD
    push_cm(&n, "extlink");
#endif
#if HP_REFNAME_MOD
    push_cm(&n, "refname");
#endif
#if HP_QOCXT_MOD
    push_cm(&n, "qocxt");
#endif
#if HP_ENTITY_MOD
    push_cm(&n, "entity");
#endif
#if HP_INDENT_MOD
    push_cm(&n, "indent");
#endif
#if HP_LISTLEVEL_MOD
    push_cm(&n, "listlevel");
#endif
#if HP_ISSE_MOD
    push_cm(&n, "isse");
#endif
#if HP_MAGIC_MOD
    push_cm(&n, "magic");
#endif
#if HP_NOWIKI_MOD
    push_cm(&n, "nowiki");
#endif
#if HP_TITLE_MOD
    push_cm(&n, "title");
#endif
#if HP_PAGEID_MOD
    push_cm(&n, "pageid");
#endif
#if HP_USER_MOD
    push_cm(&n, "user");
#endif
#if HP_TEXT_MOD
    push_cm(&n, "text");
#endif
#if HP_NS_MOD
    push_cm(&n, "ns");
#endif
#if HP_DUMPREDIR_MOD
    push_cm(&n, "dumpredir");
#endif
#if HP_IP_MOD
    push_cm(&n, "ip");
#endif
#if HP_REVCOMMENT_MOD
    push_cm(&n, "revcomment");
#endif
#if HP_MINOR_MOD
    push_cm(&n, "minor");
#endif
#if HP_WIKIMODEL_MOD
    push_cm(&n, "wikimodel");
#endif
#if HP_SECTITLE_MOD
    push_cm(&n, "sectitle");
#endif
#if HP_PARSERFN_MOD
    push_cm(&n, "parserfn");
#endif
#if HP_TABLECLASS_MOD
    push_cm(&n, "tableclass");
#endif
#if HP_ANCHOR_MOD
    push_cm(&n, "anchor");
#endif
#if HP_PUBID_MOD
    push_cm(&n, "pubid");
#endif
#if HP_TEMPPOS_MOD
    push_cm(&n, "temppos");
#endif
#if HP_WIKISTACK_MOD
    push_cm(&n, "wikistack");
#endif
#if HP_LANG_MOD
    push_cm(&n, "lang");
#endif
#if HP_CATSORT_MOD
    push_cm(&n, "catsort");
#endif
#if HP_TBLROW_MOD
    push_cm(&n, "tblrow");
#endif
#if HP_FILEOPT_MOD
    push_cm(&n, "fileopt");
#endif
#if HP_DEFAULTSORT_MOD
    push_cm(&n, "defaultsort");
#endif
#if HP_REDIRTARGET_MOD
    push_cm(&n, "redirtarget");
#endif
#if HP_DAB_MOD
    push_cm(&n, "dab");
#endif
#if HP_HATNOTE_MOD
    push_cm(&n, "hatnote");
#endif
#if HP_LASTLINK_MOD
    push_cm(&n, "lastlink");
#endif
#if HP_FWORD_MOD
    push_cm(&n, "fword");
#endif
#if HP_YEAR_MOD
    push_cm(&n, "year");
#endif
#if HP_CAPMASK_MOD
    push_cm(&n, "capmask");
#endif
#if HP_CELLTXT_MOD
    push_cm(&n, "celltxt");
#endif
#if HP_HTTPHOST_MOD
    push_cm(&n, "httphost");
#endif
#if HP_PAREN_MOD
    push_cm(&n, "paren");
#endif
#if HP_LISTPOS_MOD
    push_cm(&n, "listpos");
#endif
#if HP_SHAPE_MOD
    push_cm(&n, "shape");
#endif
#if HP_SUFFIX_MOD
    push_cm(&n, "suffix");
#endif
#if HP_PREFIX_MOD
    push_cm(&n, "prefix");
#endif
#if HP_CHARCLS_MOD
    push_cm(&n, "charcls");
#endif
#if HP_VOWEL_MOD
    push_cm(&n, "vowel");
#endif
#if HP_CONTR_MOD
    push_cm(&n, "contr");
#endif
#if HP_HYPHEN_MOD
    push_cm(&n, "hyphen");
#endif
#if HP_TOKENCLS_MOD
    push_cm(&n, "tokencls");
#endif
#if HP_RUNLEN_MOD
    push_cm(&n, "runlen");
#endif
#if HP_WPOS_MOD
    push_cm(&n, "wpos");
#endif
#if HP_BLANK_MOD
    push_cm(&n, "blank");
#endif
#if HP_SPRUN_MOD
    push_cm(&n, "sprun");
#endif
#if HP_LINELEN_MOD
    push_cm(&n, "linelen");
#endif
#if HP_TAGDIST_MOD
    push_cm(&n, "tagdist");
#endif
#if HP_MARKDIST_MOD
    push_cm(&n, "markdist");
#endif
#if HP_UPPERGAP_MOD
    push_cm(&n, "uppergap");
#endif
#if HP_MONTH_MOD
    push_cm(&n, "month");
#endif
#if HP_GALLERY_MOD
    push_cm(&n, "gallery");
#endif
#if HP_SECKIND_MOD
    push_cm(&n, "seckind");
#endif
#if HP_CITEKIND_MOD
    push_cm(&n, "citekind");
#endif
#if HP_TAGNAME_MOD
    push_cm(&n, "tagname");
#endif
#if HP_COLSPAN_MOD
    push_cm(&n, "colspan");
#endif
#if HP_STYLE_MOD
    push_cm(&n, "style");
#endif
#if HP_COORD_MOD
    push_cm(&n, "coord");
#endif
#if HP_DIGITGAP_MOD
    push_cm(&n, "digitgap");
#endif
#if HP_DOTGAP_MOD
    push_cm(&n, "dotgap");
#endif
#if HP_COMMAGAP_MOD
    push_cm(&n, "commagap");
#endif
#if HP_WORDLEN_MOD
    push_cm(&n, "wordlen");
#endif
#if HP_SENTLEN_MOD
    push_cm(&n, "sentlen");
#endif
#if HP_LOWERGAP_MOD
    push_cm(&n, "lowergap");
#endif
#if HP_DIGITPOS_MOD
    push_cm(&n, "digitpos");
#endif
#if HP_SLASHGAP_MOD
    push_cm(&n, "slashgap");
#endif
#if HP_DIGLEN_MOD
    push_cm(&n, "diglen");
#endif
#if HP_PREVLINE_MOD
    push_cm(&n, "prevline");
#endif
#if HP_PREVSENT_MOD
    push_cm(&n, "prevsent");
#endif
#if HP_LINKLEN_MOD
    push_cm(&n, "linklen");
#endif
#if HP_TPLLEN_MOD
    push_cm(&n, "tpllen");
#endif
#if HP_PARALEN_MOD
    push_cm(&n, "paralen");
#endif
#if HP_ALNUMLEN_MOD
    push_cm(&n, "alnumlen");
#endif
#if HP_SPLEN_MOD
    push_cm(&n, "splen");
#endif
#if HP_TITLEWORD_MOD
    push_cm(&n, "titleword");
#endif
#if HP_HEADWORD_MOD
    push_cm(&n, "headword");
#endif
#if HP_INIT_MOD
    push_cm(&n, "init");
#endif
#if HP_ORDINAL_MOD
    push_cm(&n, "ordinal");
#endif
#if HP_UNIT_MOD
    push_cm(&n, "unit");
#endif
#if HP_DECIMAL_MOD
    push_cm(&n, "decimal");
#endif
#if HP_REPEAT_MOD
    push_cm(&n, "repeat");
#endif
#if HP_CASEFLIP_MOD
    push_cm(&n, "caseflip");
#endif
#if HP_LEAD_MOD
    push_cm(&n, "lead");
#endif
#if HP_INFOVAL_MOD
    push_cm(&n, "infoval");
#endif
#if HP_LINKTRAIL_MOD
    push_cm(&n, "linktrail");
#endif
#if HP_CELLKIND_MOD
    push_cm(&n, "cellkind");
#endif
#if HP_TBLCOL_MOD
    push_cm(&n, "tblcol");
#endif
#if HP_HEADIDX_MOD
    push_cm(&n, "headidx");
#endif
#if HP_HTMLFMT_MOD
    push_cm(&n, "htmlfmt");
#endif
#if HP_INFOBOX_MOD
    push_cm(&n, "infobox");
#endif
#if HP_SECLEVEL_MOD
    push_cm(&n, "seclevel");
#endif
#if HP_BRACE3_MOD
    push_cm(&n, "brace3");
#endif
#if HP_NAMEDARG_MOD
    push_cm(&n, "namedarg");
#endif
#if HP_INCLUDE_MOD
    push_cm(&n, "include");
#endif
#if HP_SIG_MOD
    push_cm(&n, "sig");
#endif
#if HP_WIKIBOLD_MOD
    push_cm(&n, "wikibold");
#endif
#if HP_URLPART_MOD
    push_cm(&n, "urlpart");
#endif
#if HP_REFIDX_MOD
    push_cm(&n, "refidx");
#endif
    n.push_back("m3");
    n.push_back("m4");
    n.push_back("m6");
    n.push_back("m10");
    n.push_back("m16");
#if HP_MATCH_18
    n.push_back("m8");
#endif
#if HP_MATCH_13
    n.push_back("m13");
#endif
#if HP_MATCH_01
    n.push_back("m1");
#endif
#if HP_MATCH_02
    n.push_back("m2");
#endif
#if HP_MATCH_05
    n.push_back("m5");
#endif
#if HP_MATCH_07
    n.push_back("m7");
#endif
#if HP_MATCH_09
    n.push_back("m9");
#endif
#if HP_MATCH_12
    n.push_back("m12");
#endif
#if HP_MATCH_20
    n.push_back("m20");
#endif
#if HP_SPARSE_UTF8
    n.push_back("utf8gap");
#endif
#if HP_SKIPK_MOD
    n.push_back("skipk");
#endif
#if HP_SKIP3_MOD
    n.push_back("skip3");
#endif
#if HP_SKIP4_MOD
    n.push_back("skip4");
#endif
#if HP_SKIP5_MOD
    n.push_back("skip5");
#endif
#if HP_LZP_MOD
    n.push_back("lzp");
#endif
#if HP_SR_MOD
    n.push_back("sr");
#endif
#if HP_DMC_MOD
    n.push_back("dmc");
#endif
#if HP_WORD_MATCH
    n.push_back("wm1");
    n.push_back("wm2");
    n.push_back("wm3");
#if HP_WMATCH_4
    n.push_back("wm4");
#endif
#if HP_WMATCH_5
    n.push_back("wm5");
#endif
#endif
    n.push_back("hebb");
#if HP_CTW
    n.push_back("ctw");
#endif
    for (int i = 0; i < hp::DiscoveryPool::kSlots; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "disc%d", i);
        push_cm(&n, buf);
    }
    return n;
}

static std::vector<std::string> gate_names() {
    std::vector<std::string> n = {"c0", "alpha", "prev", "match", "entropy", "hebb"};
#if HP_EXTRA_GATES
    n.push_back("wiki");
    n.push_back("pattern");
#endif
#if HP_POS_GATE
    n.push_back("pos");
#endif
#if HP_GATE_SHAPE
    n.push_back("shape");
#endif
#if HP_GATE_BRANCH
    n.push_back("branch");
#endif
#if HP_GATE_DISP
    n.push_back("disp");
#endif
#if HP_GATE_MLEN2
    n.push_back("mlen2");
#endif
#if HP_GATE_ARGMAX
    n.push_back("argmax");
#endif
#if HP_SEN_GROUP
    n.push_back("sengroup");
#endif
#if HP_GATE_BREAK
    n.push_back("break");
#endif
#if HP_GATE_WORDPOS
    n.push_back("wordpos");
#endif
#if HP_GATE_HEDGE
    n.push_back("hedge");
#endif
#if HP_GATE_FWORD
    n.push_back("fword");
#endif
#if HP_GATE_UTF8
    n.push_back("utf8");
#endif
#if HP_GATE_NEST
    n.push_back("nestg");
#endif
#if HP_GATE_AGREE
    n.push_back("agree");
#endif
#if HP_GATE_FCLASS
    n.push_back("fclass");
#endif
#if HP_GATE_WMLEN
    n.push_back("wmlen");
#endif
    return n;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: dump_profile <in> <out.i16> [stride] [maxrec] [mem]\n");
        return 2;
    }
    const int stride = argc > 3 ? std::atoi(argv[3]) : 8;
    const long maxrec = argc > 4 ? std::atol(argv[4]) : 4000000;
    hp::Config cfg;
    if (argc > 5) {
        cfg.table_bits = std::atoi(argv[5]);
        cfg.normalize();
    }
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) {
        std::perror(argv[1]);
        return 1;
    }
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) {
        std::perror(argv[2]);
        return 1;
    }

    hp::Predictor pred(cfg);
    auto enames = expert_names();
    auto gnames = gate_names();

    long kept = 0, seen = 0;
    std::vector<std::int16_t> rec;
    int c;
    int n_exp = 0, n_gates = 0;
    while ((c = std::fgetc(in)) != EOF && kept < maxrec) {
        for (int i = 7; i >= 0; --i) {
            (void)pred.predict();
            const int bit = (c >> i) & 1;
            if ((seen % stride) == 0) {
                n_exp = pred.expert_count();
                n_gates = pred.mixer_n();
                rec.clear();
                rec.reserve(static_cast<std::size_t>(n_exp + n_gates + 5));
                for (int e = 0; e < n_exp; ++e)
                    rec.push_back(static_cast<std::int16_t>(hp::stretch(
                        hp::clamp_int(pred.expert_p(e), 1, 4094))));
                for (int g = 0; g < n_gates; ++g)
                    rec.push_back(static_cast<std::int16_t>(pred.mixer_dot(g)));
                rec.push_back(static_cast<std::int16_t>(pred.c0()));
                rec.push_back(static_cast<std::int16_t>(pred.wiki_state()));
                rec.push_back(static_cast<std::int16_t>(pred.last_match_len()));
                rec.push_back(static_cast<std::int16_t>(pred.entropy_bucket()));
                rec.push_back(static_cast<std::int16_t>(bit));
                std::fwrite(rec.data(), 2, rec.size(), out);
                ++kept;
                if ((kept % 200000) == 0)
                    std::fprintf(stderr, "kept %ld\n", kept);
            }
            pred.update(bit);
            ++seen;
        }
    }
    std::fclose(in);
    std::fclose(out);

    std::string meta = std::string(argv[2]) + ".json";
    std::FILE* jo = std::fopen(meta.c_str(), "wb");
    if (jo) {
        std::fprintf(jo,
                     "{\n  \"n_records\": %ld,\n  \"n_exp\": %d,\n  \"n_gates\": %d,\n"
                     "  \"n_ctx\": 4,\n  \"stride\": %d,\n  \"wrec\": %d,\n"
                     "  \"experts\": [",
                     kept, n_exp, n_gates, stride, n_exp + n_gates + 5);
        for (int i = 0; i < n_exp; ++i) {
            const char* nm = i < (int)enames.size() ? enames[i].c_str() : "?";
            std::fprintf(jo, "%s\"%s\"", i ? "," : "", nm);
        }
        std::fprintf(jo, "],\n  \"gates\": [");
        for (int i = 0; i < n_gates; ++i) {
            const char* nm = i < (int)gnames.size() ? gnames[i].c_str() : "?";
            std::fprintf(jo, "%s\"%s\"", i ? "," : "", nm);
        }
        std::fprintf(jo, "]\n}\n");
        std::fclose(jo);
    }
    if ((int)enames.size() != n_exp)
        std::fprintf(stderr, "WARN name count %zu != n_exp %d\n", enames.size(),
                     n_exp);
    if ((int)gnames.size() != n_gates)
        std::fprintf(stderr, "WARN gate name count %zu != n_gates %d\n",
                     gnames.size(), n_gates);
    std::fprintf(stderr, "dumped %ld rec, %d experts, %d gates -> %s\n", kept, n_exp,
                 n_gates, argv[2]);
    return 0;
}
