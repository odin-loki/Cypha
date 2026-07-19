#pragma once

/// Portable ``std::shuffle`` matching libstdc++ (GCC 13+) bit-for-bit on ``std::mt19937``.
///
/// MSVC's ``std::shuffle`` uses a different Fisher–Yates schedule than libstdc++'s
/// paired-int fast path, so the same seed yields different permutations and diverges
/// online DIF train order (D01 Cypha −4/80 on MSVC). This helper freezes the libstdc++
/// algorithm so MinGW stays unchanged and MSVC matches it.

#include <cstddef>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

namespace cypha {

namespace detail {

template <class IntType, class URBG>
std::pair<IntType, IntType> gen_two_uniform_ints(IntType b0, IntType b1, URBG& g) {
  IntType x = std::uniform_int_distribution<IntType>{0, (b0 * b1) - 1}(g);
  return {static_cast<IntType>(x / b1), static_cast<IntType>(x % b1)};
}

}  // namespace detail

template <class RandomIt, class URBG>
void portable_shuffle(RandomIt first, RandomIt last, URBG&& g) {
  if (first == last) {
    return;
  }

  using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
  using ud_t = typename std::make_unsigned<diff_t>::type;
  using distr_t = std::uniform_int_distribution<ud_t>;
  using param_t = typename distr_t::param_type;
  using gen_t = typename std::remove_reference<URBG>::type;
  using uc_t = typename std::common_type<typename gen_t::result_type, ud_t>::type;

  const uc_t urngrange = static_cast<uc_t>(g.max() - g.min());
  const uc_t urange = static_cast<uc_t>(last - first);

  // libstdc++ fast path: one UID draw yields two swap indices when the generator
  // range is large enough relative to the sequence length (true for mt19937 @ n≲320).
  if (urange > 0 && urngrange / urange >= urange) {
    RandomIt i = first + 1;
    if ((urange % 2) == 0) {
      distr_t d{0, 1};
      std::iter_swap(i++, first + static_cast<diff_t>(d(g)));
    }
    while (i != last) {
      const uc_t swap_range = static_cast<uc_t>(i - first) + 1;
      const auto pospos = detail::gen_two_uniform_ints(swap_range, swap_range + 1, g);
      std::iter_swap(i++, first + static_cast<diff_t>(pospos.first));
      std::iter_swap(i++, first + static_cast<diff_t>(pospos.second));
    }
    return;
  }

  distr_t d;
  for (RandomIt i = first + 1; i != last; ++i) {
    std::iter_swap(i, first + static_cast<diff_t>(d(g, param_t(0, static_cast<ud_t>(i - first)))));
  }
}

}  // namespace cypha
