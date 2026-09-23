// dump_profile.cpp — diagnostic dump of experts + layer-1 mixer dots + cheap context.
// Not the codec. Built against the gate24-only hp headers.
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
    n->push_back(std::string(name) + ":py");
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
    push_cm(&n, "wstr");
    push_cm(&n, "brk");
    push_cm(&n, "link");
    push_cm(&n, "num");
    push_cm(&n, "sen");
    push_cm(&n, "sentst");
    push_cm(&n, "sentmem");
    push_cm(&n, "sengrp");
    push_cm(&n, "nest");
    push_cm(&n, "para");
    push_cm(&n, "line");
    push_cm(&n, "state");
    push_cm(&n, "tpl");
    push_cm(&n, "infokey");
    push_cm(&n, "o6b");
    push_cm(&n, "linkpipe");
    push_cm(&n, "cat");
    push_cm(&n, "heading");
    push_cm(&n, "title");
    push_cm(&n, "sectitle");
    push_cm(&n, "wikistack");
    push_cm(&n, "capmask");
    push_cm(&n, "uppergap");
    push_cm(&n, "wordlen");
    n.push_back("m3");
    n.push_back("m4");
    n.push_back("m6");
    n.push_back("m10");
    n.push_back("m16");
    n.push_back("m8");
    n.push_back("m1");
    n.push_back("m2");
    n.push_back("m5");
    n.push_back("utf8gap");
    n.push_back("skipk");
    n.push_back("skip3");
    n.push_back("skip4");
    n.push_back("lzp");
    n.push_back("dmc");
    n.push_back("wm1");
    n.push_back("wm2");
    n.push_back("wm3");
    n.push_back("wm4");
    n.push_back("hebb");
    for (int i = 0; i < hp::DiscoveryPool::kSlots; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "disc%d", i);
        push_cm(&n, buf);
    }
    return n;
}

static std::vector<std::string> gate_names() {
    std::vector<std::string> n = {"c0", "alpha", "prev", "match", "entropy", "hebb"};
    n.push_back("wiki");
    n.push_back("pattern");
    n.push_back("argmax");
    n.push_back("wordpos");
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
