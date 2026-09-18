#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hp/simd_dot.hpp"

static std::int32_t rng_u32(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<std::int32_t>(s);
}

static std::int32_t rand_range(std::uint32_t& s, std::int32_t lim) {
    const std::uint32_t u = static_cast<std::uint32_t>(rng_u32(s));
    const std::int32_t span = lim * 2 + 1;
    return static_cast<std::int32_t>(u % static_cast<std::uint32_t>(span)) - lim;
}

#if defined(__GNUC__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#endif
static std::int64_t oracle_dot(const std::int32_t* a, const std::int32_t* b, int n) {
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<std::int64_t>(a[i]) * b[i];
    return sum;
}

static void oracle_axpy(std::int32_t* w, const std::int32_t* st, int n,
                        std::int32_t err, std::int32_t l1) {
    for (int i = 0; i < n; ++i) {
        const std::int32_t dw = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(st[i]) * err * l1) >> 14);
        w[i] = hp::clamp_int(w[i] + dw, -(1 << 22), (1 << 22));
    }
}

int main() {
    std::uint32_t seed = 0xC0FFEEu;
    int fails = 0;
    int trials = 0;
    std::int32_t a[80];
    std::int32_t b[80];
    std::int32_t w[80];
    std::int32_t w2[80];
    std::int32_t st[80];

    for (int n = 0; n <= 80; ++n) {
        for (int t = 0; t < 64; ++t) {
            ++trials;
            for (int i = 0; i < n; ++i) {
                a[i] = rand_range(seed, 1 << 20);
                b[i] = rand_range(seed, 1 << 20);
                st[i] = rand_range(seed, 2047);
                w[i] = rand_range(seed, 1 << 20);
                w2[i] = w[i];
            }
            const std::int64_t got = hp::dot_i32(a, b, n);
            const std::int64_t want = oracle_dot(a, b, n);
            if (got != want) {
                std::printf("FAIL dot n=%d trial=%d got=%lld want=%lld\n",
                            n, t, static_cast<long long>(got),
                            static_cast<long long>(want));
                ++fails;
            }
            const std::int32_t err = rand_range(seed, 4095);
            const std::int32_t l1 = rand_range(seed, 255) + 1;
            hp::axpy_shift_clamp(w, st, n, err, l1);
            oracle_axpy(w2, st, n, err, l1);
            for (int i = 0; i < n; ++i) {
                if (w[i] != w2[i]) {
                    std::printf("FAIL axpy n=%d i=%d got=%d want=%d err=%d l1=%d\n",
                                n, i, w[i], w2[i], err, l1);
                    ++fails;
                    break;
                }
            }
        }
    }

    if (fails) {
        std::printf("FAIL (%d mismatches, %d trials)\n", fails, trials);
        return 1;
    }
    std::printf("PASS (%d trials, n=0..80)\n", trials);

    for (int i = 0; i < 80; ++i) {
        a[i] = rand_range(seed, 1 << 20);
        b[i] = rand_range(seed, 1 << 20);
    }
    const int reps = 20000000;
    volatile std::int64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) sink += oracle_dot(a, b, 80);
    const auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) sink += hp::dot_i32(a, b, 80);
    const auto t2 = std::chrono::steady_clock::now();
    const double ms_s = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_v = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("bench n=80 reps=%d scalar=%.1f ms simd=%.1f ms sink=%lld\n",
                reps, ms_s, ms_v, static_cast<long long>(sink));
    return 0;
}
