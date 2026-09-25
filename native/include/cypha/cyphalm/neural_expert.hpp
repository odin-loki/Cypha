#pragma once

/// Pretrained byte-level neural models as CyphaLM experts (inference only).
///
/// hp predicts from contexts it has counted; a neural byte model generalises
/// across contexts and is better at markup and punctuation, so the two mix well
/// (docs/reports/CYPHALM_VS_NEURAL_LM.md). Weights come from
/// ``bench/lm_compare/byte_lm.py export`` and are shared read-only; each stream
/// keeps its own ``State`` (copyable, so generation can snapshot and restore it).
///
/// "BLM1" (``ByteLstmExpert``): u32 magic, u32 layers, u32 d, u32 emb, then
/// float32: embedding [256][emb]; per layer W_ih [4d][in], W_hh [4d][d],
/// b_ih [4d], b_hh [4d] (PyTorch ``nn.LSTM`` layout, gates i, f, g, o;
/// in = emb for layer 0, else d); output W [256][d], b [256].
///
/// "BGT1" (``ByteGptExpert``): u32 magic, u32 layers, u32 d, u32 heads,
/// u32 ctx, then float32: token embedding [256][d] (tied output), position
/// embedding [ctx][d]; per layer ln1 w, b [d], qkv [3d][d], proj [d][d],
/// ln2 w, b [d], fc [4d][d], fc2 [d][4d]; final ln w, b [d]. Pre-LN blocks,
/// exact GELU, causal attention; learned positions, so a full window re-primes
/// on its last ctx/2 bytes (as ``byte_lm.py``'s Stepper).
///
/// Matrices are held as bf16 in memory (the models train under bf16
/// autocast; one stream is bound by reading the weights); vectors stay float32.
/// Products and attention run on one kernel set per process
/// (``neural_kernel``): AVX-512 or AVX2 with FMA where the CPU has them, else
/// the portable kernels.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cypha::cyphalm {

class ByteNeuralExpert {
 public:
    struct State {
        std::vector<float> a, b;          // LSTM: h, c. GPT: K [layers][d][ctx], V [layers][ctx][d] caches
        std::vector<std::uint8_t> hist;   // GPT: bytes in the window
        int pos = 0;                      // GPT: next position
        std::array<double, 256> log_p{};  // natural-log P(next byte)
        // Output-layer adaptation (dynamic evaluation of the last layer):
        // this stream's float copy of the output weights and bias, the
        // hidden vector the current log_p came from, and the SGD rate for
        // the next step (0 = frozen).
        std::vector<float> ow, ob, hlast;
        float adapt_lr = 0.0f;
        std::vector<float> work;  // step scratch (no allocation per byte); not state
    };
    virtual ~ByteNeuralExpert() = default;
    /// State after reading a zero byte (the models are trained to predict the
    /// first byte of a text from that).
    virtual State initial_state() const = 0;
    /// Read one byte: advance the state and refresh ``log_p``.
    virtual void step(State& s, std::uint8_t byte) const = 0;
    virtual std::size_t parameter_count() const = 0;
    virtual std::string kind() const = 0;
    /// Give ``s`` its own copy of the output layer, so ``step`` can adapt it
    /// (SGD on each read byte's log loss at ``s.adapt_lr``).
    virtual void init_adaptation(State& s) const = 0;

 protected:
    /// Shared tail of ``step``: SGD on the output layer for ``byte`` (if
    /// adapting), then logits from hidden ``h`` (the adapted copy if any).
    static void adapt_output(State& s, std::uint8_t byte, int d);
    static void output_logits(State& s, const float* h, int d, const std::uint16_t* w_bf16, const float* bias);
};

/// Load a BLM1 or BGT1 file (by magic).
std::shared_ptr<const ByteNeuralExpert> load_neural_expert(const std::string& path);

/// The experts' kernel set: "avx512" or "avx2" (FMA; the two give identical
/// results), else "portable" (the original kernels, no FMA: the results of
/// earlier builds bit for bit; the FMA sets differ from them in float rounding
/// only). Chosen once per process as the best the CPU runs, capped by the
/// environment variable ``CYPHA_NN_KERNEL`` = auto | avx512 | avx2 | portable
/// (an ISA the CPU lacks falls back to the next one down).
std::string neural_kernel();
/// Switch kernel sets (tests, benchmarks, ``--neural-kernel``); false if the
/// name is unknown or the CPU cannot run it. States stay valid either way.
bool set_neural_kernel(const std::string& name);

class ByteLstmExpert final : public ByteNeuralExpert {
 public:
    static std::shared_ptr<const ByteLstmExpert> load(const std::string& path);
    State initial_state() const override;
    void step(State& s, std::uint8_t byte) const override;
    std::size_t parameter_count() const override;
    std::string kind() const override { return "lstm"; }
    void init_adaptation(State& s) const override;
    int layers() const { return layers_; }
    int hidden() const { return d_; }

 private:
    std::size_t work_size_() const;
    int layers_ = 0, d_ = 0, emb_ = 0;
    std::vector<float> emb_w_;                   // [256][emb]
    std::vector<std::vector<std::uint16_t>> w_;  // bf16, per layer [4d][in + d], rows = [W_ih | W_hh]
    std::vector<std::vector<float>> b_;          // per layer [4d] = b_ih + b_hh
    std::vector<std::uint16_t> out_w_;           // bf16 [256][d]
    std::vector<float> out_b_;                   // [256]
};

class ByteGptExpert final : public ByteNeuralExpert {
 public:
    static std::shared_ptr<const ByteGptExpert> load(const std::string& path);
    State initial_state() const override;
    void step(State& s, std::uint8_t byte) const override;
    std::size_t parameter_count() const override;
    std::string kind() const override { return "gpt"; }
    void init_adaptation(State& s) const override;

 private:
    struct Layer {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
        std::vector<std::uint16_t> qkv, proj, fc, fc2;  // bf16
    };
    void forward_(State& s, std::uint8_t byte, bool want_logits) const;  // the byte at s.pos (already in s.hist)
    std::size_t work_size_() const;
    int layers_ = 0, d_ = 0, heads_ = 0, ctx_ = 0;
    std::vector<float> emb_, pos_;       // [256][d], [ctx][d]
    std::vector<std::uint16_t> emb_bf_;  // bf16 [256][d], tied output
    std::vector<Layer> L_;
    std::vector<float> lnf_w_, lnf_b_;
};

}  // namespace cypha::cyphalm
