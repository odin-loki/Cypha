#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/multilabel_dif.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: multilabel_dif_parity <fixtures/multilabel_dif>\n";
      return 2;
    }
    fs::path dir = fs::path(argv[1]);
    std::ifstream sf(dir / "sidecar.json");
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    cypha::MultiLabelDifParams p;
    p.input_dim = j.at("d_in").get<int>();
    p.field_dim = j.at("field_dim").get<int>();
    p.world_lr = j.at("world_lr").get<double>();
    p.delta_lr = j.at("delta_lr").get<double>();
    p.ood_sigma = j.at("ood_sigma").get<double>();
    p.train.enc_lr = j.value("enc_lr", 0.0);
    p.train.replay_ratio = j.value("replay_ratio", 0.0);
    p.train.replay_cap = j.value("replay_cap", 10000);
    p.train.align_every = j.value("align_every", 500);

    if (j.contains("label_rng_seeds")) {
      for (auto it = j.at("label_rng_seeds").begin(); it != j.at("label_rng_seeds").end(); ++it) {
        p.label_rng_seeds[it.key()] = static_cast<std::uint32_t>(it.value().get<std::uint64_t>());
      }
    }
    auto load_matrix_map = [&](const char* key, std::unordered_map<std::string, std::vector<double>>& out,
                               int rows, int cols) {
      if (!j.contains(key)) {
        return;
      }
      for (auto it = j.at(key).begin(); it != j.at(key).end(); ++it) {
        std::vector<double> w;
        for (const auto& row : it.value()) {
          for (const auto& v : row) {
            w.push_back(v.get<double>());
          }
        }
        if (static_cast<int>(w.size()) != rows * cols) {
          throw std::runtime_error(std::string(key) + " size mismatch");
        }
        out[it.key()] = std::move(w);
      }
    };
    auto load_vec_map = [&](const char* key, std::unordered_map<std::string, std::vector<double>>& out, int n) {
      if (!j.contains(key)) {
        return;
      }
      for (auto it = j.at(key).begin(); it != j.at(key).end(); ++it) {
        std::vector<double> w = it.value().get<std::vector<double>>();
        if (static_cast<int>(w.size()) != n) {
          throw std::runtime_error(std::string(key) + " size mismatch");
        }
        out[it.key()] = std::move(w);
      }
    };
    const int d = p.input_dim;
    const int fd = p.field_dim;
    load_matrix_map("initial_w_inject", p.initial_w_inject, fd, d);
    load_matrix_map("initial_field_w_t", p.initial_field_w_t, fd, fd);
    load_vec_map("initial_field_sr_vec", p.initial_field_sr_vec, fd);
    if (j.contains("initial_enc_w")) {
      for (auto it = j.at("initial_enc_w").begin(); it != j.at("initial_enc_w").end(); ++it) {
        std::vector<double> w;
        for (const auto& row : it.value()) {
          for (const auto& v : row) {
            w.push_back(v.get<double>());
          }
        }
        if (static_cast<int>(w.size()) != d * d) {
          throw std::runtime_error("initial_enc_w size mismatch");
        }
        p.initial_enc_w[it.key()] = std::move(w);
      }
    }

    cypha::MultiLabelDif mlf(p);
    constexpr double kLossTol = 1e-9;
    /// Probability parity (infer / batch gate path): allow sub-percent drift vs Python float order.
    constexpr double kProbTol = 1e-2;

    for (const auto& step : j.at("train_steps")) {
      std::vector<double> x = step.at("x").get<std::vector<double>>();
      std::unordered_map<std::string, bool> labels;
      for (auto it = step.at("labels").begin(); it != step.at("labels").end(); ++it) {
        labels[it.key()] = it.value().get<bool>();
      }
      auto losses = mlf.train_step(x.data(), p.input_dim, labels);
      for (auto it = step.at("expected_losses").begin(); it != step.at("expected_losses").end(); ++it) {
        const std::string lbl = it.key();
        const double exp = it.value().get<double>();
        auto lit = losses.find(lbl);
        const double got_loss = (lit == losses.end()) ? -1.0 : lit->second;
        if (lit == losses.end() || !near_eq(got_loss, exp, kLossTol)) {
          std::cerr << "train loss mismatch " << lbl << " got " << got_loss << " expected " << exp << "\n";
          return 1;
        }
      }
    }

    {
      const int n = j.at("batch_n").get<int>();
      std::vector<double> xs = j.at("batch_x_rowmajor").get<std::vector<double>>();
      if (static_cast<int>(xs.size()) != n * p.input_dim) {
        throw std::runtime_error("batch_x_rowmajor size mismatch");
      }
      auto got = mlf.predict_batch(xs.data(), n, p.input_dim);
      for (auto it = j.at("expected_batch").begin(); it != j.at("expected_batch").end(); ++it) {
        const std::string lbl = it.key();
        std::vector<double> exp = it.value().get<std::vector<double>>();
        auto git = got.find(lbl);
        if (git == got.end() || git->second.size() != exp.size()) {
          std::cerr << "predict_batch missing " << lbl << "\n";
          return 1;
        }
        for (std::size_t i = 0; i < exp.size(); ++i) {
          if (!near_eq(git->second[i], exp[i], kProbTol)) {
            std::cerr << "predict_batch mismatch " << lbl << "[" << i << "] got " << git->second[i] << " expected "
                      << exp[i] << "\n";
            return 1;
          }
        }
      }
    }

    {
      std::vector<double> xq = j.at("predict_x").get<std::vector<double>>();
      auto got = mlf.predict(xq.data(), p.input_dim);
      for (auto it = j.at("expected_predict").begin(); it != j.at("expected_predict").end(); ++it) {
        const std::string lbl = it.key();
        const double exp = it.value().get<double>();
        auto git = got.find(lbl);
        if (git == got.end() || !near_eq(git->second, exp, kProbTol)) {
          std::cerr << "predict mismatch " << lbl << " got " << (git == got.end() ? -1.0 : git->second)
                    << " expected " << exp << "\n";
          return 1;
        }
      }
    }

    std::cout << "multilabel_dif parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "multilabel_dif_parity: " << e.what() << "\n";
    return 1;
  }
}
