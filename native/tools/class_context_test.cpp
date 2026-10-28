// class_context_test: the stored class table (docs/reports/RAM_PROFILE.md, M10).
//
// For 16 settings (shape classes or a small word table, 2 or 3 orders, full or compact storage, min_total 1 or 2)
// a table is built on a fixed text, saved and loaded. The loaded table must equal the built one in every array and
// give the same prediction (every double, the order and the total) at every third position of the text and of an
// unseen text. Then damaged copies of each file must be refused: one bit flipped at spread positions, the file cut
// short, a byte appended, another file. An unbuilt table round-trips too.

#include "cypha/cyphalm/class_context.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using cypha::cyphalm::ClassContext;

namespace {

int checks = 0, failures = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 20) std::cerr << "FAIL " << what << "\n";
    }
}

// Words picked by a fixed LCG, with capitals, numbers, apostrophes and punctuation: the same bytes everywhere.
std::vector<std::uint8_t> text(std::size_t n, std::uint32_t seed) {
    static const char* const words[] = {"the", "of", "and", "Paris", "river", "1789", "it's", "was", "born",
                                        "in",  "a",  "city", "London", "wrote", "42",   "who", "is",  "de"};
    std::uint32_t s = seed;
    std::string t;
    while (t.size() < n) {
        s = s * 1664525u + 1013904223u;
        t += words[(s >> 16) % (sizeof words / sizeof *words)];
        const unsigned p = (s >> 8) & 15u;
        t += p == 0 ? ".\n" : p == 1 ? ", " : p == 2 ? " (" : " ";
    }
    t.resize(n);
    return std::vector<std::uint8_t>(t.begin(), t.end());
}

std::vector<char> read_all(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(is), {});
}

void write_all(const std::string& path, const std::vector<char>& b) {
    std::ofstream os(path, std::ios::binary);
    os.write(b.data(), static_cast<std::streamsize>(b.size()));
}

bool refused(const std::string& path) {
    try {
        (void)ClassContext::load(path);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void same_predictions(const ClassContext& a, const ClassContext& b, const std::vector<std::uint8_t>& t,
                      const std::string& what) {
    std::size_t differ = 0;
    for (std::size_t i = 0; i <= t.size(); i += 3) {
        double pa[256], pb[256];
        const auto ha = a.predict(t.data(), i, pa), hb = b.predict(t.data(), i, pb);
        differ += ha.order != hb.order || ha.total != hb.total || std::memcmp(pa, pb, sizeof pa) != 0;
    }
    expect(differ == 0, what + ": " + std::to_string(differ) + " predictions differ");
}

}  // namespace

int main() {
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string path = dir + "/class_context_test_" + std::to_string(std::random_device{}()) + ".cc";
    const std::string bad = path + ".bad";
    const auto train = text(60000, 7u), unseen = text(6000, 99u);
    const std::unordered_map<std::string, std::uint16_t> table = {
        {"the", 4}, {"of", 5}, {"and", 5}, {"paris", 6}, {"london", 6}, {"river", 7}, {"city", 7}, {"wrote", 8}, {"born", 8}};

    for (int t = 0; t < 2; ++t)
        for (int orders = 2; orders <= 3; ++orders)
            for (int compact = 0; compact < 2; ++compact)
                for (std::uint32_t mt = 1; mt <= 2; ++mt) {
                    const std::string what = std::string(t ? "table" : "shape") + " orders " + std::to_string(orders) +
                                             (compact ? " compact" : " full") + " min_total " + std::to_string(mt);
                    ClassContext::Options o;
                    o.orders = orders, o.compact = compact != 0, o.min_total = mt, o.chunk = 4096;
                    ClassContext built(t ? table : std::unordered_map<std::string, std::uint16_t>{}, o);
                    built.build(train.data(), train.size());
                    expect(built.contexts() > 100, what + ": contexts built");
                    built.save(path);
                    const ClassContext back = ClassContext::load(path);
                    expect(back == built, what + ": loaded table equals the built one");
                    expect(back.contexts() == built.contexts() && back.followers() == built.followers() &&
                               back.bytes() == built.bytes() && back.classes() == built.classes(),
                           what + ": sizes");
                    same_predictions(built, back, train, what + " (training text)");
                    same_predictions(built, back, unseen, what + " (unseen text)");

                    const std::vector<char> good = read_all(path);
                    expect(good.size() >= built.bytes() && good.size() <= built.bytes() + 1024 + 16 * table.size(),
                           what + ": file size " + std::to_string(good.size()) + " near bytes() " +
                               std::to_string(built.bytes()));
                    for (int k = 0; k < 48; ++k) {  // one bit flipped: header, word map, arrays, checksum
                        std::vector<char> b = good;
                        const std::size_t at = k < 16 ? static_cast<std::size_t>(k) * 2 : (b.size() - 1) * static_cast<std::size_t>(k - 16) / 31;
                        b[at] = static_cast<char>(b[at] ^ (1 << (k % 8)));
                        write_all(bad, b);
                        expect(refused(bad), what + ": bit flipped at " + std::to_string(at) + " refused");
                    }
                    for (const std::size_t cut : {std::size_t{0}, std::size_t{7}, std::size_t{12}, good.size() / 2, good.size() - 8,
                                                  good.size() - 1}) {
                        write_all(bad, std::vector<char>(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(cut)));
                        expect(refused(bad), what + ": cut to " + std::to_string(cut) + " bytes refused");
                    }
                    std::vector<char> longer = good;
                    longer.push_back('\0');
                    write_all(bad, longer);
                    expect(refused(bad), what + ": a byte appended refused");
                }
    {
        ClassContext empty({}, ClassContext::Options{});
        empty.save(path);
        expect(ClassContext::load(path) == empty, "unbuilt table round-trips");
        write_all(bad, std::vector<char>{'C', 'Y', 'P', 'H', 'A', 'P', 'K', '\0', 1, 0, 0, 0});
        expect(refused(bad), "another file's magic refused");
        expect(refused(dir + "/class_context_test_no_such_file.cc"), "a missing file refused");
    }
    std::remove(path.c_str());
    std::remove(bad.c_str());
    std::cout << "class_context_test: " << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
