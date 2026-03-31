#include "cypha/create_model.hpp"

#include <cmath>
#include <stdexcept>

#include "cypha/nig_field.hpp"

namespace cypha {

namespace {

/// Helper: build a scalar CNode.
static CNode make_float(double v) {
    CNode n; n.kind = CNode::Float; n.f = v; return n;
}
static CNode make_int(std::int64_t v) {
    CNode n; n.kind = CNode::Int; n.i = v; return n;
}
static CNode make_empty_list() {
    CNode n; n.kind = CNode::Map; return n;   // empty map = empty list in Cypha serial
}
/// 1-D tensor of zeros.
static CNode zeros_1d(int len) {
    CNode n;
    n.kind = CNode::Tensor;
    n.shape.push_back(static_cast<std::uint32_t>(len));
    n.tensor.assign(static_cast<std::size_t>(len), 0.0);
    return n;
}
/// 1-D tensor filled with value.
static CNode fill_1d(int len, double val) {
    CNode n;
    n.kind = CNode::Tensor;
    n.shape.push_back(static_cast<std::uint32_t>(len));
    n.tensor.assign(static_cast<std::size_t>(len), val);
    return n;
}
/// 2-D tensor of zeros (rows × cols, row-major).
static CNode zeros_2d(int rows, int cols) {
    CNode n;
    n.kind = CNode::Tensor;
    n.shape.push_back(static_cast<std::uint32_t>(rows));
    n.shape.push_back(static_cast<std::uint32_t>(cols));
    n.tensor.assign(static_cast<std::size_t>(rows * cols), 0.0);
    return n;
}
/// Identity matrix (d × d).
static CNode identity_2d(int d) {
    CNode n;
    n.kind = CNode::Tensor;
    n.shape.push_back(static_cast<std::uint32_t>(d));
    n.shape.push_back(static_cast<std::uint32_t>(d));
    n.tensor.assign(static_cast<std::size_t>(d * d), 0.0);
    for (int i = 0; i < d; i++) {
        n.tensor[static_cast<std::size_t>(i * d + i)] = 1.0;
    }
    return n;
}
/// Emit a float32 a_eff tensor (fd×fd) from a zero W_T.
static CNode make_a_eff_from_zero_w_t(int fd) {
    // W_T = zeros(fd,fd)  →  a_eff = diag(a) + W_T  = diag(a)
    std::vector<double> wt(static_cast<std::size_t>(fd * fd), 0.0);
    std::vector<float> aeff;
    recompute_field_a_eff(fd, wt, aeff);
    CNode n;
    n.kind = CNode::Tensor;
    n.shape.push_back(static_cast<std::uint32_t>(fd));
    n.shape.push_back(static_cast<std::uint32_t>(fd));
    // Store as float64 doubles (recompute_field_a_eff gives float32 but CNode stores double)
    n.tensor.resize(static_cast<std::size_t>(fd * fd));
    for (int i = 0; i < fd * fd; i++) {
        n.tensor[static_cast<std::size_t>(i)] = static_cast<double>(aeff[static_cast<std::size_t>(i)]);
    }
    return n;
}

}  // namespace

CNode create_fresh_model_root(const FreshModelParams& p) {
    if (p.input_dim <= 0 || p.input_dim > 4096)
        throw std::invalid_argument("input_dim out of range");
    if (p.field_dim <= 0 || p.field_dim > 4096)
        throw std::invalid_argument("field_dim out of range");

    const int d  = p.input_dim;
    const int fd = p.field_dim;

    // ── Build world sub-map ──────────────────────────────────────────────────
    // Matches Python WorldPrior.__init__:
    //   mu       = zeros(d)
    //   v        = ones(d)          (world variance, default 1)
    //   inv_v    = ones(d)          (computed from v)
    //   n        = 0                (observation count)
    //   drift_ema = zeros(d)
    //   F_field  = zeros(d, fd)     (field→world coupling, initially 0)
    CNode world;
    world.kind = CNode::Map;
    world.map.push_back({"mu",        zeros_1d(d)});
    world.map.push_back({"v",         fill_1d(d, 1.0)});
    world.map.push_back({"n",         make_int(0)});
    world.map.push_back({"drift_ema", make_float(0.0)});  // scalar (memory_train as_double)
    world.map.push_back({"F_field",   zeros_2d(d, fd)});

    // ── Build field_W_T + field_a_eff ────────────────────────────────────────
    // Python NIGField.__init__: W_T = zeros(fd,fd); a_eff = diag(a) + W_T
    CNode field_w_t   = zeros_2d(fd, fd);
    CNode field_a_eff = make_a_eff_from_zero_w_t(fd);

    // field_sr_vec: initial power-iteration vector = uniform 1/sqrt(fd)
    CNode field_sr_vec;
    field_sr_vec.kind = CNode::Tensor;
    field_sr_vec.shape.push_back(static_cast<std::uint32_t>(fd));
    field_sr_vec.tensor.assign(static_cast<std::size_t>(fd),
                                1.0 / std::sqrt(static_cast<double>(fd)));

    // ── Build w_inject ───────────────────────────────────────────────────────
    // Python _W_inject (fd × d): random small values after 1 call, but init = None.
    // We emit a small random-looking but reproducible matrix (scaled identity + zeros).
    // In practice, _W_inject is generated via CyphaDIF.__init__ from a seeded RNG.
    // For a fresh model, emit zeros (d == fd → identity is used automatically;
    // d != fd → explicit zeros means to_field_dim returns false and falls back to first-fd components).
    // The better choice: emit an actual (fd × d) zero matrix so the loader sees it.
    CNode w_inject = zeros_2d(fd, d);
    // A zero w_inject is fine: latent_to_field_signal normalises h and maps via W_inject @ h_norm.
    // Zeros will give zero signal → field stays at rest until training updates W_T.

    // ── Root map ─────────────────────────────────────────────────────────────
    // Order matches Python _save_state key order for maximum compatibility.
    CNode root;
    root.kind = CNode::Map;

    // Encoder: identity (VectorEncoder with no learning at start)
    root.map.push_back({"enc_W",        identity_2d(d)});

    // World prior
    root.map.push_back({"world",        std::move(world)});

    // Field state
    root.map.push_back({"field_h",      zeros_1d(fd)});
    root.map.push_back({"field_W_T",    std::move(field_w_t)});
    root.map.push_back({"field_a_eff",  std::move(field_a_eff)});
    root.map.push_back({"field_sr_vec", std::move(field_sr_vec)});
    root.map.push_back({"field_step",   make_int(0)});
    root.map.push_back({"w_inject",     std::move(w_inject)});

    // Classifier temperature & calibration
    root.map.push_back({"temperature",        make_float(p.temperature)});
    root.map.push_back({"base_temp",          make_float(p.temperature)});
    root.map.push_back({"llr_scale_ema",      make_float(0.0)});
    root.map.push_back({"llr_scale_n",        make_int(0)});
    root.map.push_back({"llr_scale_baseline", make_float(0.0)});
    root.map.push_back({"mahal_ema",          make_float(0.0)});
    root.map.push_back({"mahal_std_ema",      make_float(0.5)});
    root.map.push_back({"llr_ema",            make_float(0.0)});

    // Classes list: empty initially (populated by training)
    root.map.push_back({"classes", make_empty_list()});

    // Training bookkeeping
    root.map.push_back({"total_steps",   make_int(0)});
    root.map.push_back({"total_correct", make_int(0)});

    // Context buffer: empty
    root.map.push_back({"mid_n",    make_float(0.0)});
    root.map.push_back({"mid_freq", make_empty_list()});

    // Hyperparameters embedded for reference (not loaded by from_root, but
    // useful for the Qt shell to restore defaults on reload)
    root.map.push_back({"hparam_world_lr", make_float(p.world_lr)});
    root.map.push_back({"hparam_delta_lr", make_float(p.delta_lr)});

    return root;
}

void create_and_save_fresh_model(const char* path, const FreshModelParams& p) {
    CNode root = create_fresh_model_root(p);
    save_cypha_file(path, root);
}

}  // namespace cypha
