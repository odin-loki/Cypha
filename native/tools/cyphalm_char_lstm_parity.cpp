// Char-LSTM + low-rank GRIA + hybrid blend vs parity_fixtures/cyphalm_char_lstm/sidecar.json
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::vector<double> read_vec(const nlohmann::json& j) {
  std::vector<double> o;
  for (const auto& v : j) {
    o.push_back(v.get<double>());
  }
  return o;
}

void load_char_lstm(cypha::cyphalm::CharLSTMHead& m, const nlohmann::json& j) {
  m.vocab_size = j.at("vocab_size").get<int>();
  m.hidden = j.at("hidden").get<int>();
  const auto& w = j.at("char_lstm");
  m.E = read_vec(w.at("E"));
  m.Wx = read_vec(w.at("Wx"));
  m.Wh = read_vec(w.at("Wh"));
  m.b = read_vec(w.at("b"));
  m.Wy = read_vec(w.at("Wy"));
  m.by = read_vec(w.at("by"));
}

void load_gria(cypha::cyphalm::GRIALowRank& g, const nlohmann::json& j) {
  g.field_dim = j.at("field_dim").get<int>();
  g.vocab_size = j.at("vocab_size").get<int>();
  g.rank = j.at("rank").get<int>();
  const auto& w = j.at("gria");
  g.U = read_vec(w.at("U"));
  g.V = read_vec(w.at("V"));
  g.alpha = read_vec(w.at("alpha"));
  g.bias = read_vec(w.at("bias"));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: cyphalm_char_lstm_parity <parity_fixtures/cyphalm_char_lstm/sidecar.json>\n";
      return 2;
    }
    std::ifstream sf(argv[1]);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    const auto j = nlohmann::json::parse(buf.str());
    constexpr double kTol = 1e-9;

    cypha::cyphalm::CharLSTMHead lstm;
    load_char_lstm(lstm, j);
    cypha::cyphalm::GRIALowRank gria;
    load_gria(gria, j);

    const int token_id = j.at("token_id").get<int>();
    const int target_id = j.at("target_id").get<int>();
    const int gria_target = j.at("gria_target_id").get<int>();
    const double blend_logit = j.at("blend_logit").get<double>();
    const int vocab = j.at("vocab_size").get<int>();

    std::vector<double> h = read_vec(j.at("h_init"));
    std::vector<double> c = read_vec(j.at("c_init"));
    std::vector<double> v = read_vec(j.at("v_field"));

    std::vector<double> log_probs(static_cast<std::size_t>(vocab));
    std::vector<double> h_new;
    std::vector<double> c_new;
    cypha::cyphalm::CharLSTMCache cache;
    lstm.forward_step(token_id, h.data(), c.data(), log_probs.data(), h_new, c_new, &cache);

    const auto& exp = j.at("expected");
    std::vector<double> exp_lp = read_vec(exp.at("log_probs"));
    for (int k = 0; k < vocab; ++k) {
      if (!near_eq(log_probs[static_cast<std::size_t>(k)], exp_lp[static_cast<std::size_t>(k)], kTol)) {
        std::cerr << "log_probs[" << k << "] got " << log_probs[static_cast<std::size_t>(k)] << " exp "
                  << exp_lp[static_cast<std::size_t>(k)] << "\n";
        return 1;
      }
    }

    const double loss = -log_probs[static_cast<std::size_t>(target_id)];
    if (!near_eq(loss, exp.at("loss").get<double>(), kTol)) {
      std::cerr << "loss mismatch\n";
      return 1;
    }

    std::vector<double> exp_h = read_vec(exp.at("h_new"));
    std::vector<double> exp_c = read_vec(exp.at("c_new"));
    for (std::size_t i = 0; i < exp_h.size(); ++i) {
      if (!near_eq(h_new[i], exp_h[i], kTol) || !near_eq(c_new[i], exp_c[i], kTol)) {
        std::cerr << "state mismatch at " << i << "\n";
        return 1;
      }
    }

    cypha::cyphalm::CharLSTMGrad grads = lstm.backward_step(cache, target_id);
    if (!near_eq(exp.at("dWy_norm").get<double>(),
                 std::sqrt(std::inner_product(grads.dWy.begin(), grads.dWy.end(), grads.dWy.begin(), 0.0)), kTol)) {
      std::cerr << "dWy norm mismatch\n";
      return 1;
    }
    double dWx_sum = 0.0;
    for (double x : grads.dWx) dWx_sum += x;
    if (!near_eq(dWx_sum, exp.at("dWx_sum").get<double>(), kTol)) {
      std::cerr << "dWx sum mismatch\n";
      return 1;
    }
    std::vector<double> exp_dh = read_vec(exp.at("dh_prev"));
    for (std::size_t i = 0; i < exp_dh.size(); ++i) {
      if (!near_eq(grads.dh_prev[i], exp_dh[i], kTol)) {
        std::cerr << "dh_prev mismatch at " << i << "\n";
        return 1;
      }
    }

    std::vector<double> log_g(static_cast<std::size_t>(vocab));
    gria.forward(v.data(), log_g.data());
    std::vector<double> exp_lg = read_vec(exp.at("log_probs_gria"));
    for (int k = 0; k < vocab; ++k) {
      if (!near_eq(log_g[static_cast<std::size_t>(k)], exp_lg[static_cast<std::size_t>(k)], kTol)) {
        std::cerr << "gria log_probs[" << k << "] mismatch\n";
        return 1;
      }
    }

    cypha::cyphalm::GRIALowRankGrad gg = gria.cross_entropy_gradients(v.data(), gria_target);
    if (!near_eq(exp.at("dU_sum").get<double>(), std::accumulate(gg.dU.begin(), gg.dU.end(), 0.0), kTol)) {
      std::cerr << "dU sum mismatch\n";
      return 1;
    }
    if (!near_eq(exp.at("dV_sum").get<double>(), std::accumulate(gg.dV.begin(), gg.dV.end(), 0.0), kTol)) {
      std::cerr << "dV sum mismatch\n";
      return 1;
    }

    std::vector<double> log_blend(static_cast<std::size_t>(vocab));
    cypha::cyphalm::blend_log_probs(log_g.data(), log_probs.data(), vocab, blend_logit, log_blend.data());
    std::vector<double> exp_blend = read_vec(exp.at("log_blend"));
    for (int k = 0; k < vocab; ++k) {
      if (!near_eq(log_blend[static_cast<std::size_t>(k)], exp_blend[static_cast<std::size_t>(k)], kTol)) {
        std::cerr << "blend log_probs[" << k << "] mismatch\n";
        return 1;
      }
    }

    const double bg = cypha::cyphalm::blend_logit_grad(log_g.data(), log_probs.data(), vocab, blend_logit, gria_target);
    if (!near_eq(bg, exp.at("blend_logit_grad").get<double>(), kTol)) {
      std::cerr << "blend_logit_grad mismatch got " << bg << "\n";
      return 1;
    }

    std::cout << "cyphalm_char_lstm parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
