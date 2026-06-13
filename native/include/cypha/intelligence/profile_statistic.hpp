#pragma once

#include <cstddef>

namespace cypha::intelligence {

enum class ProfileStatistic : std::size_t {
  Alpha = 0,
  DEff = 1,
  SigmaBranch = 2,
  Tau = 3,
  REu = 4,
  Lipschitz = 5,
  Calibration = 6,
};

inline constexpr std::size_t kProfileStatisticCount = 7;

}  // namespace cypha::intelligence
