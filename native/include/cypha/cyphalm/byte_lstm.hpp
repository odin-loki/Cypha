#pragma once

/// Small float32 byte-level LSTM expert for CyphaLM.
///
/// hp predicts from hashed contexts it has counted; in a context it has never
/// seen its distribution goes flat. An LSTM generalises across contexts, so a
/// mixture of the two (``ByteMixGate``) keeps hp's sharpness where hp has
/// evidence and falls back on the LSTM where it has none.
///
/// One layer: embed(byte) -> LSTM(H) -> softmax(256). Online training with
/// truncated BPTT over non-overlapping windows of ``bptt`` bytes and Adam.

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace cypha::cyphalm {

struct ByteLstmOptions {
    int hidden = 128;
    int bptt = 20;
    float lr = 2e-3f;
    float clip = 5.0f;
    std::uint64_t seed = 7;
};

class ByteLstm {
 public:
    explicit ByteLstm(ByteLstmOptions opt = {});

    /// Log-probabilities (natural log) of the next byte given everything consumed.
    const std::vector<float>& log_probs() const { return logp_; }

    /// Consume the next byte. With ``learn``, its loss under the current
    /// prediction joins the BPTT window (weights update when the window fills).
    /// Returns -ln P(byte) under the prediction made before seeing it.
    float observe(int byte, bool learn = true);

    /// Clear the recurrent state (h, c) and any pending window; weights stay.
    void reset_state();

    int hidden() const { return H_; }

    void write(std::ostream& os) const;
    void read(std::istream& is);

 private:
    struct Step {
        int x = 0;                 // input byte
        std::vector<float> h_prev, c_prev, i, f, g, o, c, h, tanh_c;
        std::vector<float> dlogits;  // softmax - onehot(next byte)
    };

    void forward_(int x);
    void backprop_window_();
    void adam_(std::vector<float>& w, std::vector<float>& m, std::vector<float>& v,
               const std::vector<float>& g);

    ByteLstmOptions opt_;
    int H_;
    // Parameters. Wx: 4H x H (input = embedding), Wh: 4H x H, b: 4H, E: 256 x H,
    // Wy: 256 x H, by: 256. Gate order i, f, g, o.
    std::vector<float> E_, Wx_, Wh_, b_, Wy_, by_;
    std::vector<float> mE_, vE_, mWx_, vWx_, mWh_, vWh_, mb_, vb_, mWy_, vWy_, mby_, vby_;
    long adam_t_ = 0;
    // State.
    std::vector<float> h_, c_, logits_, logp_;
    std::vector<Step> window_;
    int pending_ = -1;  // index in window_ of the step whose output predicts the next byte
};

/// Per-byte mixture of two next-byte distributions with an online-learned gate:
/// w = sigmoid(theta . features), p = w * p_a + (1 - w) * p_b.
/// Features: 1, entropies (bits/8), and max log-probs of both inputs.
class ByteMixGate {
 public:
    static constexpr int kF = 5;
    ByteMixGate();

    /// Mixed log-probs (natural log) for ``a`` (hp) and ``b`` (LSTM). Remembers the
    /// inputs so ``learn`` can update the gate once the true byte is known.
    const std::vector<double>& mix(const std::vector<double>& log_a,
                                   const std::vector<float>& log_b);
    void learn(int byte, double lr = 0.02);
    double last_weight() const { return w_; }
    const double* theta() const { return theta_; }

    void write(std::ostream& os) const;
    void read(std::istream& is);

 private:
    double theta_[kF];
    double feat_[kF] = {};
    double w_ = 0.5;
    std::vector<double> pa_, pb_, out_;
};

}  // namespace cypha::cyphalm
