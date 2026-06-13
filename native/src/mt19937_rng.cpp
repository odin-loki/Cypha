#include "cypha/mt19937_rng.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../third_party/numpy_rng/ziggurat_constants.h"

namespace cypha {
namespace {

constexpr std::uint32_t kDefaultPoolSize = 4;
constexpr std::uint32_t kInitA = 0x43b0d7e5u;
constexpr std::uint32_t kMultA = 0x931e8875u;
constexpr std::uint32_t kInitB = 0x8b51f9ddu;
constexpr std::uint32_t kMultB = 0x58f38dedu;
constexpr std::uint32_t kMixMultL = 0xca01f9ddu;
constexpr std::uint32_t kMixMultR = 0x4973f715u;
constexpr std::uint32_t kXShift = 16;

struct U128 {
  std::uint64_t high{0};
  std::uint64_t low{0};
};

inline U128 u128_from_u64(std::uint64_t hi, std::uint64_t lo) {
  return {hi, lo};
}

inline U128 u128_add(U128 a, U128 b) {
  U128 r;
  r.low = a.low + b.low;
  r.high = a.high + b.high + (r.low < b.low ? 1u : 0u);
  return r;
}

inline void u128_mul64(std::uint64_t x, std::uint64_t y, std::uint64_t* hi, std::uint64_t* lo) {
  const std::uint64_t x0 = x & 0xffffffffu;
  const std::uint64_t x1 = x >> 32;
  const std::uint64_t y0 = y & 0xffffffffu;
  const std::uint64_t y1 = y >> 32;
  const std::uint64_t w0 = x0 * y0;
  const std::uint64_t t = x1 * y0 + (w0 >> 32);
  const std::uint64_t w1 = t & 0xffffffffu;
  const std::uint64_t w2 = t >> 32;
  const std::uint64_t w1b = w1 + x0 * y1;
  *lo = x * y;
  *hi = x1 * y1 + w2 + (w1b >> 32);
}

inline U128 u128_mul(U128 a, U128 b) {
  std::uint64_t h1 = a.high * b.low + a.low * b.high;
  std::uint64_t lo_hi = 0;
  std::uint64_t lo_lo = 0;
  u128_mul64(a.low, b.low, &lo_hi, &lo_lo);
  return {lo_hi + h1, lo_lo};
}

inline U128 u128_mul_u64(U128 a, std::uint64_t b) {
  std::uint64_t h1 = a.high * b;
  std::uint64_t lo_hi = 0;
  std::uint64_t lo_lo = 0;
  u128_mul64(a.low, b, &lo_hi, &lo_lo);
  return {lo_hi + h1, lo_lo};
}

inline std::uint64_t pcg_rotr64(std::uint64_t value, unsigned rot) {
  return (value >> rot) | (value << ((-static_cast<int>(rot)) & 63));
}

struct Pcg64State {
  U128 state{};
  U128 inc{};
};

inline U128 pcg_multiplier() {
  return u128_from_u64(2549297995355413924ULL, 4865540595714422341ULL);
}

inline void pcg_step(Pcg64State* rng) {
  const U128 mul = pcg_multiplier();
  rng->state = u128_add(u128_mul(rng->state, mul), rng->inc);
}

inline std::uint64_t pcg_output(const U128& state) {
  return pcg_rotr64(state.high ^ state.low, static_cast<unsigned>(state.high >> 58));
}

inline void pcg_seed(Pcg64State* rng, U128 initstate, U128 initseq) {
  rng->state = {0, 0};
  rng->inc.high = (initseq.high << 1u) | (initseq.low >> 63u);
  rng->inc.low = (initseq.low << 1u) | 1u;
  pcg_step(rng);
  rng->state = u128_add(rng->state, initstate);
  pcg_step(rng);
}

inline std::uint32_t hashmix(std::uint32_t value, std::uint32_t* hash_const) {
  value ^= *hash_const;
  *hash_const *= kMultA;
  value *= *hash_const;
  value ^= value >> kXShift;
  return value;
}

inline std::uint32_t mix32(std::uint32_t x, std::uint32_t y) {
  std::uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

std::vector<std::uint32_t> int_to_uint32_array(std::uint64_t n) {
  std::vector<std::uint32_t> out;
  if (n == 0) {
    out.push_back(0);
    return out;
  }
  while (n > 0) {
    out.push_back(static_cast<std::uint32_t>(n & 0xffffffffu));
    n >>= 32;
  }
  return out;
}

void mix_entropy(std::vector<std::uint32_t>& mixer, const std::vector<std::uint32_t>& entropy) {
  std::uint32_t hash_const = kInitA;
  for (std::size_t i = 0; i < mixer.size(); ++i) {
    if (i < entropy.size()) {
      mixer[i] = hashmix(entropy[i], &hash_const);
    } else {
      mixer[i] = hashmix(0, &hash_const);
    }
  }
  for (std::size_t i_src = 0; i_src < mixer.size(); ++i_src) {
    for (std::size_t i_dst = 0; i_dst < mixer.size(); ++i_dst) {
      if (i_src != i_dst) {
        mixer[i_dst] = mix32(mixer[i_dst], hashmix(mixer[i_src], &hash_const));
      }
    }
  }
  for (std::size_t i_src = mixer.size(); i_src < entropy.size(); ++i_src) {
    for (std::size_t i_dst = 0; i_dst < mixer.size(); ++i_dst) {
      mixer[i_dst] = mix32(mixer[i_dst], hashmix(entropy[i_src], &hash_const));
    }
  }
}

std::vector<std::uint64_t> seed_sequence_generate_u64(const std::vector<std::uint32_t>& pool, int n_words) {
  std::uint32_t hash_const = kInitB;
  std::vector<std::uint32_t> state(static_cast<std::size_t>(n_words * 2), 0);
  std::size_t cycle = 0;
  for (int i_dst = 0; i_dst < n_words * 2; ++i_dst) {
    std::uint32_t data_val = pool[cycle % pool.size()];
    ++cycle;
    data_val ^= hash_const;
    hash_const *= kMultB;
    data_val *= hash_const;
    data_val ^= data_val >> kXShift;
    state[static_cast<std::size_t>(i_dst)] = data_val;
  }
  std::vector<std::uint64_t> out(static_cast<std::size_t>(n_words));
  for (int i = 0; i < n_words; ++i) {
    const std::size_t j = static_cast<std::size_t>(i * 2);
    out[static_cast<std::size_t>(i)] =
        (static_cast<std::uint64_t>(state[j + 1]) << 32) | static_cast<std::uint64_t>(state[j]);
  }
  return out;
}

inline double uint64_to_double(std::uint64_t rnd) {
  return static_cast<double>(rnd >> 11) * (1.0 / 9007199254740992.0);
}

struct BitGen {
  Pcg64State pcg{};
  int has_uint32{0};
  std::uint32_t uinteger{0};
};

std::uint64_t next_uint64(BitGen* st) {
  st->has_uint32 = 0;
  pcg_step(&st->pcg);
  return pcg_output(st->pcg.state);
}

std::uint32_t next_uint32(BitGen* st) {
  if (st->has_uint32) {
    st->has_uint32 = 0;
    return st->uinteger;
  }
  const std::uint64_t next = next_uint64(st);
  st->has_uint32 = 1;
  st->uinteger = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xffffffffu);
}

std::uint32_t bounded_lemire_uint32(BitGen* st, std::uint32_t rng) {
  const std::uint32_t rng_excl = rng + 1;
  std::uint64_t m = static_cast<std::uint64_t>(next_uint32(st)) * static_cast<std::uint64_t>(rng_excl);
  std::uint32_t leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
  if (leftover < rng_excl) {
    const std::uint32_t threshold = (0xffffffffu - rng) % rng_excl;
    while (leftover < threshold) {
      m = static_cast<std::uint64_t>(next_uint32(st)) * static_cast<std::uint64_t>(rng_excl);
      leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
    }
  }
  return static_cast<std::uint32_t>(m >> 32);
}

std::uint64_t random_bounded_inclusive(BitGen* st, std::uint64_t off, std::uint64_t rng) {
  if (rng == 0) {
    return off;
  }
  if (rng <= 0xffffffffu) {
    return off + bounded_lemire_uint32(st, static_cast<std::uint32_t>(rng));
  }
  // Fallback for large ranges (not needed for Izaac embed).
  for (;;) {
    const std::uint64_t val = next_uint64(st) % (rng + 1);
    if (val <= rng) {
      return off + val;
    }
  }
}

double next_double(BitGen* st) { return uint64_to_double(next_uint64(st)); }

double random_standard_normal(BitGen* st) {
  for (;;) {
    std::uint64_t r = next_uint64(st);
    int idx = static_cast<int>(r & 0xff);
    r >>= 8;
    int sign = static_cast<int>(r & 0x1);
    std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
    double x = static_cast<double>(rabs) * wi_double[idx];
    if (sign & 0x1) {
      x = -x;
    }
    if (rabs < ki_double[idx]) {
      return x;
    }
    if (idx == 0) {
      for (;;) {
        double xx = -ziggurat_nor_inv_r * std::log1p(-next_double(st));
        double yy = -std::log1p(-next_double(st));
        if (yy + yy > xx * xx) {
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
        }
      }
    }
    if (((fi_double[idx - 1] - fi_double[idx]) * next_double(st) + fi_double[idx]) < std::exp(-0.5 * x * x)) {
      return x;
    }
  }
}

BitGen make_bitgen(int seed) {
  auto entropy = int_to_uint32_array(static_cast<std::uint64_t>(seed));
  std::vector<std::uint32_t> pool(kDefaultPoolSize, 0);
  mix_entropy(pool, entropy);
  auto words = seed_sequence_generate_u64(pool, 4);
  BitGen bg;
  pcg_seed(&bg.pcg, u128_from_u64(words[0], words[1]), u128_from_u64(words[2], words[3]));
  return bg;
}

}  // namespace

struct NumpyDefaultRng::Impl {
  BitGen bitgen;
  explicit Impl(int seed) : bitgen(make_bitgen(seed)) {}
};

NumpyDefaultRng::NumpyDefaultRng(int seed) : impl_(new Impl(seed)) {}

NumpyDefaultRng::~NumpyDefaultRng() { delete impl_; }

double NumpyDefaultRng::normal(double loc, double scale) {
  return loc + scale * random_standard_normal(&impl_->bitgen);
}

double NumpyDefaultRng::uniform(double low, double high) {
  return low + (high - low) * next_double(&impl_->bitgen);
}

int NumpyDefaultRng::integers(int low, int high) {
  if (high <= low) {
    throw std::invalid_argument("integers: high must be greater than low");
  }
  const std::uint64_t range = static_cast<std::uint64_t>(high - low - 1);
  const std::uint64_t v = random_bounded_inclusive(&impl_->bitgen, static_cast<std::uint64_t>(low), range);
  return static_cast<int>(v);
}

std::vector<int> NumpyDefaultRng::permutation(int n) {
  if (n < 0) {
    throw std::invalid_argument("permutation: n must be non-negative");
  }
  std::vector<int> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = i;
  }
  for (int i = n - 1; i > 0; --i) {
    const int j = integers(0, i + 1);
    std::swap(out[static_cast<std::size_t>(i)], out[static_cast<std::size_t>(j)]);
  }
  return out;
}

}  // namespace cypha
