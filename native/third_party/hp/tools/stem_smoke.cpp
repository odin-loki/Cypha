#include <cstdio>
#include "hp/stemmer.hpp"

static void check(const char* in, const char* expect) {
    hp::StemWord w;
    for (int i = 0; in[i]; ++i) w.add(in[i]);
    hp::stem_english(w);
    const int ok = w.eq(expect);
    std::printf("%s  %s -> %s  type=%u  %s\n",
                ok ? "OK " : "BAD", in, w.c_str(), w.type(),
                ok ? "" : expect);
}

int main() {
    check("compressing", "compress");
    check("compressed", "compress");
    check("skies", "sky");
    check("the", "the");
    check("running", "run");
    check("national", "nation");
    check("happily", "happili");
    return 0;
}
