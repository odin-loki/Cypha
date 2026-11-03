// Match + order-1 backoff. The long context is a pointer into the training
// bytes, not a copied count table. A 4-byte key stores only the index of the
// last time that context was seen. Held-out bytes are never written into the
// map, so a match cannot see the future.
//
// Bits/byte use the distribution below, which sums to 1. Sampling is a
// separate step (temperature, nucleus, repetition penalty) and does not
// change the reported bit rate.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSample = 8'388'608;
constexpr std::size_t kTrain = 7'340'032;
constexpr std::uint32_t kEmpty = 0xffffffffu;

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct PosMap {
    std::vector<std::uint32_t> keys;
    std::vector<std::uint32_t> vals;
    std::vector<std::uint16_t> seen;
    std::uint32_t mask = 0;
    std::uint32_t filled = 0;

    void reset(std::uint32_t cap_bits) {
        keys.assign(1u << cap_bits, kEmpty);
        vals.assign(1u << cap_bits, 0);
        seen.assign(1u << cap_bits, 0);
        mask = (1u << cap_bits) - 1;
        filled = 0;
    }

    void grow() {
        std::vector<std::uint32_t> ok = std::move(keys);
        std::vector<std::uint32_t> ov = std::move(vals);
        std::vector<std::uint16_t> os = std::move(seen);
        reset(static_cast<std::uint32_t>(std::bit_width(ok.size())));
        for (std::size_t i = 0; i < ok.size(); ++i) {
            if (ok[i] != kEmpty) put_new(ok[i], ov[i], os[i]);
        }
    }

    void put_new(std::uint32_t key, std::uint32_t val, std::uint16_t nseen) {
        std::uint32_t i = mix32(key) & mask;
        while (keys[i] != kEmpty) i = (i + 1) & mask;
        keys[i] = key;
        vals[i] = val;
        seen[i] = nseen;
        ++filled;
    }

    void put(std::uint32_t key, std::uint32_t val) {
        if (filled * 10 > mask * 7) grow();
        std::uint32_t i = mix32(key) & mask;
        for (;;) {
            if (keys[i] == kEmpty) {
                keys[i] = key;
                vals[i] = val;
                seen[i] = 1;
                ++filled;
                return;
            }
            if (keys[i] == key) {
                vals[i] = val;
                if (seen[i] < 65535) ++seen[i];
                return;
            }
            i = (i + 1) & mask;
        }
    }

    // Returns false if absent. Index is the follower byte's position.
    bool find(std::uint32_t key, std::uint32_t& follower, int& times) const {
        if (keys.empty()) return false;
        std::uint32_t i = mix32(key) & mask;
        for (int n = 0; n < 64; ++n) {
            if (keys[i] == kEmpty) return false;
            if (keys[i] == key) {
                follower = vals[i];
                times = seen[i];
                return true;
            }
            i = (i + 1) & mask;
        }
        return false;
    }
};

struct Model {
    std::uint32_t uni[256]{};
    std::uint32_t uni_n = 0;
    std::uint32_t o1[256][256]{};
    std::uint32_t o1_n[256]{};
    PosMap pos;
    const std::uint8_t* data = nullptr;

    void init() { pos.reset(20); }

    static std::uint32_t key4(const std::uint8_t* d, std::size_t i) {
        return (std::uint32_t(d[i - 4]) << 24) | (std::uint32_t(d[i - 3]) << 16) |
               (std::uint32_t(d[i - 2]) << 8) | std::uint32_t(d[i - 1]);
    }

    double kt1(unsigned prev, unsigned y) const {
        return (static_cast<double>(o1[prev][y]) + 0.5) / (static_cast<double>(o1_n[prev]) + 128.0);
    }
    double kt0(unsigned y) const {
        return (static_cast<double>(uni[y]) + 0.5) / (static_cast<double>(uni_n) + 128.0);
    }

    // Match length walking backward from a previous follower. 0 if none.
    int match(std::size_t i, unsigned& pred_byte) const {
        if (i < 4) return 0;
        std::uint32_t follower = 0;
        int times = 0;
        if (!pos.find(key4(data, i), follower, times)) return 0;
        if (follower >= i || times < 2) return 0;
        pred_byte = data[follower];
        int len = 4;
        int back = 5;
        while (len < 24 && i >= static_cast<std::size_t>(back) &&
               follower >= static_cast<std::size_t>(back) &&
               data[i - static_cast<std::size_t>(back)] == data[follower - static_cast<std::size_t>(back)]) {
            ++len;
            ++back;
        }
        return len;
    }

    double prob(std::size_t i, unsigned y) const {
        unsigned pred = 0;
        const int len = match(i, pred);
        const unsigned prev = i > 0 ? data[i - 1] : 0;
        const double back = 0.85 * kt1(prev, y) + 0.15 * kt0(y);
        if (len < 4) return back;
        const double esc = 1.0 / (1.0 + std::log2(static_cast<double>(len)));
        if (y == pred) return (1.0 - esc) + esc * back;
        return esc * back;
    }

    void observe(std::size_t i) {
        const unsigned y = data[i];
        const unsigned prev = i > 0 ? data[i - 1] : 0;
        ++uni[y];
        ++uni_n;
        ++o1[prev][y];
        ++o1_n[prev];
        if (i >= 4) pos.put(key4(data, i), static_cast<std::uint32_t>(i));
    }

    std::uint64_t ram() const {
        return pos.filled * 8ull + sizeof(o1) + sizeof(uni);
    }
};

std::string printable(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c >= 32 && c < 127) o += static_cast<char>(c);
        else {
            const char* hex = "0123456789abcdef";
            o += "\\x";
            o += hex[c >> 4];
            o += hex[c & 15];
        }
    }
    return o;
}

double distinct4(const std::string& s) {
    if (s.size() < 4) return 0;
    std::vector<std::uint32_t> g;
    g.reserve(s.size());
    for (std::size_t i = 0; i + 3 < s.size(); ++i) {
        std::uint32_t x = (std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
                          (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint32_t(std::uint8_t(s[i + 3]));
        g.push_back(x);
    }
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

std::string generate(const Model& model, const std::vector<std::uint8_t>& buf, std::size_t prompt_end,
                     int ngen, std::mt19937& rng) {
    std::vector<std::uint8_t> out(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(prompt_end));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < ngen; ++t) {
        const std::size_t i = out.size();
        double raw[256];
        double sum = 0;
        const std::size_t hist = out.size();
        // Score as if `out` is the data. Match indices must lie in the trained
        // prefix, which they do because the map was frozen at kTrain.
        Model view = {};
        // Don't copy the model. Call prob against model with a temporary pointer.
        // prob() reads model.data. Point it at `out` only for the history bytes
        // the match walker needs, but follower indices index the TRAINING buffer.
        // So match() must read followers from the training buffer and history
        // from `out`. Handle that inline here.
        unsigned pred = 0;
        int len = 0;
        if (i >= 4) {
            std::uint32_t key = (std::uint32_t(out[i - 4]) << 24) | (std::uint32_t(out[i - 3]) << 16) |
                                (std::uint32_t(out[i - 2]) << 8) | std::uint32_t(out[i - 1]);
            std::uint32_t follower = 0;
            int times = 0;
            if (model.pos.find(key, follower, times) && follower < kTrain && times >= 2) {
                bool repeat = false;
                if (follower + 8 < kTrain && i >= 8) {
                    const std::size_t from = i > 80 ? i - 80 : 0;
                    for (std::size_t s = from; s + 8 <= i; ++s) {
                        if (std::memcmp(out.data() + s, model.data + follower, 8) == 0) repeat = true;
                    }
                }
                if (!repeat && i >= 16) {
                    for (std::size_t s = 0; s + 16 < i; ++s) {
                        if (std::memcmp(out.data() + s, out.data() + (i - 16), 16) == 0) repeat = true;
                    }
                }
                if (!repeat) {
                    pred = model.data[follower];
                    len = 4;
                    int back = 5;
                    while (len < 24 && i >= static_cast<std::size_t>(back) &&
                           follower >= static_cast<std::size_t>(back) &&
                           out[i - static_cast<std::size_t>(back)] ==
                               model.data[follower - static_cast<std::size_t>(back)]) {
                        ++len;
                        ++back;
                    }
                }
            }
        }
        const unsigned prev = i > 0 ? out[i - 1] : 0;
        for (int y = 0; y < 256; ++y) {
            const double back = 0.85 * model.kt1(prev, static_cast<unsigned>(y)) + 0.15 * model.kt0(static_cast<unsigned>(y));
            double p = back;
            if (len >= 4) {
                const double esc = 1.0 / (1.0 + std::log2(static_cast<double>(len)));
                p = (static_cast<unsigned>(y) == pred) ? (1.0 - esc) + esc * back : esc * back;
            }
            // Repetition penalty. Not part of the bit rate.
            for (std::size_t k = 1; k <= 16 && k <= i; ++k) {
                if (out[i - k] == static_cast<unsigned>(y)) {
                    p /= 1.15;
                    break;
                }
            }
            raw[y] = std::pow(std::max(p, 1e-12), 1.0 / 0.8);
            sum += raw[y];
        }
        struct Item {
            int y;
            double p;
        };
        Item items[256];
        for (int y = 0; y < 256; ++y) items[y] = {y, raw[y] / sum};
        std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
        double acc = 0;
        int keep = 1;
        for (int n = 0; n < 256; ++n) {
            acc += items[n].p;
            keep = n + 1;
            if (acc >= 0.9) break;
        }
        double mass = 0;
        for (int n = 0; n < keep; ++n) mass += items[n].p;
        double r = U(rng) * mass;
        int pick = items[keep - 1].y;
        for (int n = 0; n < keep; ++n) {
            r -= items[n].p;
            if (r <= 0) {
                pick = items[n].y;
                break;
            }
        }
        out.push_back(static_cast<std::uint8_t>(pick));
        (void)hist;
        (void)view;
    }
    return std::string(reinterpret_cast<const char*>(out.data() + prompt_end), static_cast<std::size_t>(ngen));
}

}  // namespace

int main() {
    std::ifstream in("/mnt/d/cypha_llm/enwik9", std::ios::binary);
    if (!in) {
        std::cerr << "cannot open corpus\n";
        return 1;
    }
    std::vector<std::uint8_t> data(kSample);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kSample));
    if (static_cast<std::size_t>(in.gcount()) < kSample) return 1;

    Model model;
    model.data = data.data();
    model.init();
    double train_bits = 0;
    for (std::size_t i = 0; i < kTrain; ++i) {
        train_bits += -std::log2(model.prob(i, data[i]));
        model.observe(i);
        if (((i + 1) & 0x1fffff) == 0) std::cerr << "train " << (i + 1) << "\n";
    }
    double held_bits = 0;
    for (std::size_t i = kTrain; i < kSample; ++i) {
        held_bits += -std::log2(model.prob(i, data[i]));
    }

    const std::string needle = " is a ";
    std::size_t at = std::string::npos;
    for (std::size_t i = 48; i + needle.size() < kTrain; ++i) {
        if (std::memcmp(data.data() + i, needle.data(), needle.size()) == 0) {
            at = i;
            break;
        }
    }
    const std::size_t prompt_end = at == std::string::npos ? 48 : at;
    std::mt19937 rng(42);
    const std::string s1 = generate(model, data, prompt_end, 240, rng);
    // Second prompt: "The " with the three bytes of history that order-1/match need.
    std::vector<std::uint8_t> scratch = data;
    const char* p2 = "The ";
    const std::size_t p2n = 4;
    for (std::size_t i = 0; i < p2n; ++i) scratch[i] = static_cast<std::uint8_t>(p2[i]);
    // generate() reads model.data for followers and `out` copied from buf for history.
    // Pass a buffer whose prefix is "The " by using a small dedicated buffer.
    std::vector<std::uint8_t> buf2(p2, p2 + p2n);
    // Reuse generate by appending buf2 in front of a dummy. Easiest: call with data
    // replaced... generate copies buf[0, prompt_end). Build a one-off.
    std::string s2;
    {
        std::vector<std::uint8_t> seq(buf2);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int t = 0; t < 240; ++t) {
            const std::size_t i = seq.size();
            unsigned pred = 0;
            int len = 0;
            if (i >= 4) {
                std::uint32_t key = (std::uint32_t(seq[i - 4]) << 24) | (std::uint32_t(seq[i - 3]) << 16) |
                                    (std::uint32_t(seq[i - 2]) << 8) | std::uint32_t(seq[i - 1]);
                std::uint32_t follower = 0;
                int times = 0;
                if (model.pos.find(key, follower, times) && follower < kTrain && times >= 2) {
                    bool repeat = false;
                    if (follower + 8 < kTrain && i >= 8) {
                        const std::size_t from = i > 80 ? i - 80 : 0;
                        for (std::size_t s = from; s + 8 <= i; ++s) {
                            if (std::memcmp(seq.data() + s, model.data + follower, 8) == 0) repeat = true;
                        }
                    }
                    if (!repeat && i >= 16) {
                        for (std::size_t s = 0; s + 16 < i; ++s) {
                            if (std::memcmp(seq.data() + s, seq.data() + (i - 16), 16) == 0) repeat = true;
                        }
                    }
                    if (!repeat) {
                        pred = model.data[follower];
                        len = 4;
                        int back = 5;
                        while (len < 24 && i >= static_cast<std::size_t>(back) &&
                               follower >= static_cast<std::size_t>(back) &&
                               seq[i - static_cast<std::size_t>(back)] ==
                                   model.data[follower - static_cast<std::size_t>(back)]) {
                            ++len;
                            ++back;
                        }
                    }
                }
            }
            const unsigned prev = seq[i - 1];
            double raw[256], sum = 0;
            for (int y = 0; y < 256; ++y) {
                const double back = 0.85 * model.kt1(prev, static_cast<unsigned>(y)) + 0.15 * model.kt0(static_cast<unsigned>(y));
                double p = back;
                if (len >= 4) {
                    const double esc = 1.0 / (1.0 + std::log2(static_cast<double>(len)));
                    p = (static_cast<unsigned>(y) == pred) ? (1.0 - esc) + esc * back : esc * back;
                }
                for (std::size_t k = 1; k <= 16 && k <= i; ++k) {
                    if (seq[i - k] == static_cast<unsigned>(y)) {
                        p /= 1.15;
                        break;
                    }
                }
                raw[y] = std::pow(std::max(p, 1e-12), 1.0 / 0.8);
                sum += raw[y];
            }
            struct Item {
                int y;
                double p;
            } items[256];
            for (int y = 0; y < 256; ++y) items[y] = {y, raw[y] / sum};
            std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
            double acc = 0;
            int keep = 1;
            for (int n = 0; n < 256; ++n) {
                acc += items[n].p;
                keep = n + 1;
                if (acc >= 0.9) break;
            }
            double mass = 0;
            for (int n = 0; n < keep; ++n) mass += items[n].p;
            double r = U(rng) * mass;
            int pick = items[0].y;
            for (int n = 0; n < keep; ++n) {
                r -= items[n].p;
                if (r <= 0) {
                    pick = items[n].y;
                    break;
                }
            }
            seq.push_back(static_cast<std::uint8_t>(pick));
        }
        s2.assign(reinterpret_cast<const char*>(seq.data() + p2n), 240);
    }

    const double held_n = static_cast<double>(kSample - kTrain);
    std::cout << "name match_order1\n";
    std::cout << "heldout_bpc " << (held_bits / held_n) << "\n";
    std::cout << "train_online_bpc " << (train_bits / static_cast<double>(kTrain)) << "\n";
    std::cout << "ram_bytes " << model.ram() << "\n";
    std::cout << "pos_filled " << model.pos.filled << "\n";
    std::cout << "distinct4_A " << distinct4(s1) << "\n";
    std::cout << "distinct4_B " << distinct4(s2) << "\n";
    std::cout << "SAMPLE_A " << printable(s1) << "\n";
    std::cout << "SAMPLE_B " << printable(s2) << "\n";
    return 0;
}
