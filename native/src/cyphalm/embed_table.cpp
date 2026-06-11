#include "cypha/cyphalm/embed_table.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cypha/numpy_default_rng.hpp"

namespace cypha::cyphalm {

namespace {

// Default irreducible polynomials for GF(2^n) — match ``galois.GF(2**n)``.
std::uint32_t galois_irreducible_poly(int n) {
  static const std::uint32_t kTable[] = {0, 3, 7, 11, 19, 37, 91, 131, 285, 529, 1135, 2053, 4331};
  if (n < 1 || n > 12) {
    throw std::invalid_argument("GF(2^n) degree out of supported range [1, 12]");
  }
  return kTable[n];
}

int gcd_int(int a, int b) {
  while (b != 0) {
    const int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

bool valid_permutation_exponent(int k, int n) {
  const int modulus = (1 << n) - 1;
  const int exp = (1 << k) + 1;
  return gcd_int(exp, modulus) == 1;
}

int find_valid_k(int poly_degree_exp, int n) {
  int k = poly_degree_exp;
  while (k < n + 10) {
    if (valid_permutation_exponent(k, n)) {
      return k;
    }
    ++k;
  }
  throw std::runtime_error("No valid permutation exponent found for GF(2^n)");
}

std::uint32_t gf_mul(std::uint32_t a, std::uint32_t b, int n, std::uint32_t irr) {
  const std::uint32_t mask = (1u << n) - 1u;
  std::uint32_t r = 0;
  a &= mask;
  b &= mask;
  for (int i = 0; i < n; ++i) {
    if (b & 1u) {
      r ^= a;
    }
    b >>= 1u;
    const bool high = (a & (1u << (n - 1))) != 0;
    a = (a << 1u) & mask;
    if (high) {
      a ^= irr;
    }
  }
  return r & mask;
}

std::uint32_t gf_pow(std::uint32_t base, std::uint32_t exp, int n, std::uint32_t irr) {
  const std::uint32_t mask = (1u << n) - 1u;
  std::uint32_t r = 1u;
  std::uint32_t b = base & mask;
  std::uint32_t e = exp;
  while (e > 0) {
    if (e & 1u) {
      r = gf_mul(r, b, n, irr);
    }
    b = gf_mul(b, b, n, irr);
    e >>= 1u;
  }
  return r & mask;
}

void field_to_bits(std::uint32_t element, int n, double* out) {
  for (int i = 0; i < n; ++i) {
    const int bit = static_cast<int>((element >> static_cast<unsigned>(n - 1 - i)) & 1u);
    out[static_cast<std::size_t>(i)] = static_cast<double>(bit);
  }
}

std::uint32_t poly_eval(std::uint32_t token_id, std::uint32_t a, std::uint32_t b, int n, int k,
                        std::uint32_t irr) {
  const std::uint32_t exp = static_cast<std::uint32_t>((1 << k) + 1);
  const std::uint32_t xp = gf_pow(token_id, exp, n, irr);
  return gf_mul(a, xp, n, irr) ^ b;
}

}  // namespace

EmbedTable::EmbedTable(std::uint32_t vocab_size, std::uint32_t d_embed, std::uint32_t seed)
    : vocab_size_(vocab_size), d_embed_(d_embed), table_(vocab_size * d_embed) {
  if (vocab_size < 1) {
    throw std::invalid_argument("vocab_size must be >= 1");
  }
  if (d_embed < 1) {
    throw std::invalid_argument("d_embed must be >= 1");
  }

  int n = static_cast<int>(std::ceil(std::log2(static_cast<double>(vocab_size))));
  if (n < 1) {
    n = 1;
  }
  const int n_blocks = static_cast<int>(std::ceil(static_cast<double>(d_embed) / static_cast<double>(n)));
  const int k = find_valid_k(1, n);
  const std::uint32_t irr = galois_irreducible_poly(n);
  const std::uint32_t field_size = 1u << n;

  cypha::NumpyDefaultRng rng(static_cast<int>(seed));
  std::vector<int> nonzero;
  nonzero.reserve(static_cast<std::size_t>(field_size - 1));
  for (std::uint32_t v = 1; v < field_size; ++v) {
    nonzero.push_back(static_cast<int>(v));
  }
  const int a_idx = rng.integers(0, static_cast<int>(nonzero.size()));
  const std::uint32_t a = static_cast<std::uint32_t>(nonzero[static_cast<std::size_t>(a_idx)]);
  const std::uint32_t b = static_cast<std::uint32_t>(rng.integers(0, static_cast<int>(field_size)));

  std::vector<double> block(static_cast<std::size_t>(n));
  for (std::uint32_t t = 0; t < vocab_size; ++t) {
    const std::uint32_t val = poly_eval(t, a, b, n, k, irr);
    field_to_bits(val, n, block.data());
    for (int bi = 0; bi < n_blocks; ++bi) {
      const int start = bi * n;
      const int end = std::min(start + n, static_cast<int>(d_embed));
      double* row = table_.data() + static_cast<std::size_t>(t) * d_embed_;
      for (int j = start; j < end; ++j) {
        row[static_cast<std::size_t>(j)] = block[static_cast<std::size_t>(j - start)];
      }
    }
  }
}

const double* EmbedTable::embed(std::uint32_t token_id) const {
  if (token_id >= vocab_size_) {
    throw std::out_of_range("token_id out of range");
  }
  return table_.data() + static_cast<std::size_t>(token_id) * d_embed_;
}

std::vector<double> EmbedTable::embed_vec(std::uint32_t token_id) const {
  const double* p = embed(token_id);
  return std::vector<double>(p, p + d_embed_);
}

}  // namespace cypha::cyphalm
