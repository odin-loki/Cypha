#pragma once

#include <vector>

namespace cypha::cyphalm {

/// H18: one-step elementary CA rule 110 on binarized SSM hidden state.
class CAStateCell {
 public:
  /// Binarize ``h`` (sign), apply rule 110 on a ring, map back to ±scale continuous values.
  static std::vector<double> step_rule110(const std::vector<double>& h, double scale = 1.0);

 private:
  static bool rule110_bit(bool left, bool center, bool right);
};

}  // namespace cypha::cyphalm
