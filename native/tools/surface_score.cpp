// Surface score for open-ended byte samples. No model probabilities.
//
// MAUVE (Pillutla et al., 2021) compares the distribution of generated text
// with human text. Full MAUVE needs a neural embedding. The three surface
// statistics it is usually compared against are distinct-n, repetition, and
// a property of the tokens themselves. This score uses those, matched to a
// reference computed on the same Wikipedia file:
//
//   H  fraction of alphabetic words (length >= 3) that occur in the train slice
//   D  1 - |distinct4(sample) - distinct4(reference)|
//   L  1 - max(0, loop8(sample) - loop8(reference))
//   S  = H * D * L
//
// distinct4 = unique 4-byte grams / (n - 3)
// loop8     = fraction of 8-byte windows that already occurred earlier
// A copy of the file scores near 1. A loop scores near 0. Random letters
// score near 0 because the words are not in the file.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTrain = 7'340'032;

std::uint32_t gram4(const std::string& s, std::size_t i) {
    return (std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
           (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint32_t(std::uint8_t(s[i + 3]));
}

double distinct4(const std::string& s) {
    if (s.size() < 4) return 0;
    std::vector<std::uint32_t> g;
    g.reserve(s.size());
    for (std::size_t i = 0; i + 3 < s.size(); ++i) g.push_back(gram4(s, i));
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

double loop8(const std::string& s) {
    if (s.size() < 16) return 0;
    std::unordered_set<std::uint64_t> seen;
    int loops = 0, n = 0;
    for (std::size_t i = 0; i + 8 <= s.size(); ++i) {
        std::uint64_t h = 14695981039346656037ull;
        for (int k = 0; k < 8; ++k) {
            h ^= static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
            h *= 1099511628211ull;
        }
        if (!seen.insert(h).second) ++loops;
        ++n;
    }
    return static_cast<double>(loops) / static_cast<double>(n);
}

bool is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

double lexicon_hit(const std::string& s, const std::unordered_set<std::string>& vocab) {
    int words = 0, hit = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        if (!is_letter(static_cast<unsigned char>(s[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < s.size() && is_letter(static_cast<unsigned char>(s[j]))) ++j;
        if (j - i >= 3) {
            std::string w = s.substr(i, j - i);
            for (char& c : w) {
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            }
            ++words;
            if (vocab.count(w)) ++hit;
        }
        i = j;
    }
    if (words == 0) return 0;
    return static_cast<double>(hit) / static_cast<double>(words);
}

struct Parts {
    double H, d4, loop, D, L, S;
};

Parts score(const std::string& s, const std::unordered_set<std::string>& vocab, double ref_d4, double ref_loop) {
    Parts p;
    p.H = lexicon_hit(s, vocab);
    p.d4 = distinct4(s);
    p.loop = loop8(s);
    p.D = 1.0 - std::fabs(p.d4 - ref_d4);
    if (p.D < 0) p.D = 0;
    p.L = 1.0 - std::max(0.0, p.loop - ref_loop);
    if (p.L < 0) p.L = 0;
    p.S = p.H * p.D * p.L;
    return p;
}

std::string unescape(const std::string& in) {
    std::string o;
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') {
            o.push_back('\n');
            ++i;
        } else if (in[i] == '\\' && i + 3 < in.size() && in[i + 1] == 'x') {
            auto hex = [](char c) {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return 0;
            };
            o.push_back(static_cast<char>((hex(in[i + 2]) << 4) | hex(in[i + 3])));
            i += 3;
        } else {
            o.push_back(in[i]);
        }
    }
    return o;
}

}  // namespace

int main() {
    std::ifstream in("/mnt/d/cypha_llm/enwik9", std::ios::binary);
    std::vector<char> raw(kTrain);
    in.read(raw.data(), static_cast<std::streamsize>(kTrain));
    if (!in) return 1;
    std::unordered_set<std::string> vocab;
    vocab.reserve(200000);
    for (std::size_t i = 0; i < raw.size();) {
        if (!is_letter(static_cast<unsigned char>(raw[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < raw.size() && is_letter(static_cast<unsigned char>(raw[j]))) ++j;
        if (j - i >= 3 && j - i <= 32) {
            std::string w(raw.begin() + static_cast<std::ptrdiff_t>(i), raw.begin() + static_cast<std::ptrdiff_t>(j));
            for (char& c : w)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            vocab.insert(std::move(w));
        }
        i = j;
    }

    double sum_d4 = 0, sum_loop = 0;
    int windows = 0;
    for (std::size_t off = 100000; off + 200 < kTrain && windows < 40; off += 150000) {
        std::string w(raw.begin() + static_cast<std::ptrdiff_t>(off),
                      raw.begin() + static_cast<std::ptrdiff_t>(off + 200));
        sum_d4 += distinct4(w);
        sum_loop += loop8(w);
        ++windows;
    }
    const double ref_d4 = sum_d4 / windows;
    const double ref_loop = sum_loop / windows;
    std::cout << "ref_distinct4 " << ref_d4 << " ref_loop8 " << ref_loop << " vocab " << vocab.size() << "\n";

    const std::pair<const char*, const char*> samples[] = {
        {"wiki_window", nullptr},
        {"greedy", "e the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the th"},
        {"temperature", "onserer as huswhan sthl terf <tiny on a w aleal copofferar, Auoes ant th roctugherompral.a ath tincha asthuomburererce oves orentor ange a e Yo]]]] pofothe d"},
        {"nucleus", "inarint te fancare is Thend>\\x0a|aleend anens onal inst ne itssiteve and th enotrsthesen ispte ane onitionghem bice [[[atheces ariore ar thesinceis calite a"},
        {"typical", "intis and omestin ang tise it tof thericoren tint astheroteriof an athe o om iche of al stin tind is ond tof ithe t in as on and on the thering s ther t ar"},
        {"match_A", " and the decision not to produce a second stational lunar landing [[NASA]] budget and the decisionthetic immedia''.\\n{{ent|2|births|De Mount]], [[1981]].\\n\\n==End of these would not to prod tis Journal lunar landind]]. Although ''Apollo 20's t"},
        {"routed_A", "isis f ond cirlll  styn]]] \\n\\n||reme by r me  com  d ide errsse ]  and a Soetililst  rtis oactives ofenae]]] gess  (19 m| oleaned inexplondingte Crd  tothe state '' wa not nge us p wera econciniti d ir"},
        {"kn_A", " the entican of the al the [[Bentiond thatersomis pose for the [[cluen Rome>\\n <idern berna]], plexter sic and an berm the the [[sv:Amere moderstan the [[Mard bookand Som the dic supopere a was of Albely norits the a wity reangs, [[Sp"},
        {"backoff_A", "|align=&quot; personalism as &quot; | [[Time species for attackgroups of the Catholicy of General [[barles of algebration (mytholicy of a seriest recognized by the Arabic names from the valinguists, from [[pl:Algority to studiogalaxy's succ"},
        {"tiny_frozen", "e tombs, or it was the largest church north of the [[Alps]] for 400 years. The tombs of Charles States|Second in the as the chief town in the composer and which and of [[Alpako]] l"},
    };

    std::string wiki(raw.begin() + 100000, raw.begin() + 100200);
    for (const auto& item : samples) {
        std::string text = item.second ? unescape(item.second) : wiki;
        const Parts p = score(text, vocab, ref_d4, ref_loop);
        std::cout << item.first << " S=" << p.S << " H=" << p.H << " D=" << p.D << " L=" << p.L << " d4=" << p.d4
                  << " loop8=" << p.loop << "\n";
    }
    return 0;
}
