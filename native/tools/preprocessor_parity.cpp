#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/preprocessor.hpp"

namespace {
bool near(double a, double b) { return std::abs(a - b) <= 1e-10; }
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: preprocessor_parity <fixtures/preprocessor_dir>\n";
      return 2;
    }
    std::string dir = argv[1];
    auto pre = cypha::PreprocessorState::from_json_file((dir + "/preprocessor.json").c_str());
    std::ifstream sf(dir + "/sidecar.json");
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());
    std::vector<double> x;
    for (const auto& v : j["x"]) {
      x.push_back(v.get<double>());
    }
    std::vector<double> exp;
    for (const auto& v : j["expected"]) {
      exp.push_back(v.get<double>());
    }
    std::vector<double> y = pre.transform_one(x);
    if (y.size() != exp.size()) {
      throw std::runtime_error("output size mismatch");
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (!near(y[i], exp[i])) {
        std::cerr << "mismatch at " << i << " got " << y[i] << " expected " << exp[i] << "\n";
        return 1;
      }
    }
    std::cout << "preprocessor parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "preprocessor_parity: " << e.what() << "\n";
    return 1;
  }
}
