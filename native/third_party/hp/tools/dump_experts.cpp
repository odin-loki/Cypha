// tools/dump_experts.cpp
//
// Runs the predictor over a file and dumps each expert's stretched opinion,
// plus the true bit, for spectral analysis.
//
// NOTE ON PURITY: this is a DIAGNOSTIC, not part of the codec. It links the
// predictor read-only and writes a side file. The coding path is untouched
// and test/no_float.sh still applies to include/ and src/ only. Eigenvalue
// work happens in Python where floating point is harmless.
//
// Output format: int16 little-endian, (n_experts + 1) values per record.
// Last value is the true bit (0 or 1).


#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hp/predictor.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: dump_experts <in> <out.i16> [stride] [maxrec]\n");
        return 2;
    }
    const int stride = argc > 3 ? std::atoi(argv[3]) : 37;   // prime, avoids
    const long maxrec = argc > 4 ? std::atol(argv[4]) : 400000;  // bit-position aliasing

    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { std::perror(argv[1]); return 1; }
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::perror(argv[2]); return 1; }

    hp::Config cfg;
    if (argc > 5) {
        cfg.table_bits = std::atoi(argv[5]);
        cfg.normalize();
    }
    hp::Predictor pred(cfg);

    long kept = 0, seen = 0;
    std::vector<std::int16_t> rec;
    int c;
    while ((c = std::fgetc(in)) != EOF && kept < maxrec) {
        for (int i = 7; i >= 0; --i) {
            const int p = pred.predict();
            const int bit = (c >> i) & 1;
            if ((seen % stride) == 0) {
                rec.clear();
                const int n = pred.expert_count();
                for (int e = 0; e < n; ++e)
                    rec.push_back(static_cast<std::int16_t>(hp::stretch(
                        hp::clamp_int(pred.expert_p(e), 1, 4094))));
                rec.push_back(static_cast<std::int16_t>(bit));
                std::fwrite(rec.data(), sizeof(std::int16_t), rec.size(), out);
                ++kept;
            }
            ++seen;
            pred.update(bit);
            (void)p;
        }
    }
    std::fprintf(stderr, "dumped %ld records, %d experts\n", kept,
                 pred.expert_count());
    std::fclose(in);
    std::fclose(out);
    return 0;
}
