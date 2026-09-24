#pragma once

/// Pretrained byte-level LSTM as a CyphaLM expert (inference only).
///
/// hp predicts from contexts it has counted; a neural byte model generalises
/// across contexts and is better at markup and punctuation, so the two mix well
/// (docs/reports/CYPHALM_VS_NEURAL_LM.md). The weights come from
/// ``bench/lm_compare/byte_lm.py export`` (PyTorch ``nn.LSTM`` layout, gate
/// order i, f, g, o) and are shared read-only; each stream keeps its own
/// ``State``.
///
/// File "BLM1": u32 magic, u32 layers, u32 d, u32 emb, then float32 arrays:
/// embedding [256][emb]; per layer W_ih [4d][in], W_hh [4d][d], b_ih [4d],
/// b_hh [4d] (in = emb for layer 0, else d); output W [256][d], b [256].
/// Matrices are held as bf16 in memory (the LSTM is trained under bf16
/// autocast); embeddings and biases stay float32.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cypha::cyphalm {

class ByteLstmExpert {
 public:
    struct State {
        std::vector<float> h, c;         // [layers][d]
        std::array<double, 256> log_p{};  // natural-log P(next byte)
    };

    static std::shared_ptr<const ByteLstmExpert> load(const std::string& path);

    int layers() const { return layers_; }
    int hidden() const { return d_; }
    std::size_t parameter_count() const;

    /// Zero state after reading a zero byte (the models are trained to predict
    /// the first byte of a text from that).
    State initial_state() const;
    /// Read one byte: advance the recurrent state and refresh ``log_p``.
    void step(State& s, std::uint8_t byte) const;

 private:
    int layers_ = 0, d_ = 0, emb_ = 0;
    std::vector<float> emb_w_;              // [256][emb]
    std::vector<std::vector<std::uint16_t>> w_;  // bf16, per layer [4d][in + d], rows = [W_ih | W_hh]
    std::vector<std::vector<float>> b_;     // per layer [4d] = b_ih + b_hh
    std::vector<std::uint16_t> out_w_;      // bf16 [256][d]
    std::vector<float> out_b_;              // [256]
};

}  // namespace cypha::cyphalm
