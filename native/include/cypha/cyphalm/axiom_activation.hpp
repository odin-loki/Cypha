#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "cypha/cyphalm/eml_activation.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"

namespace cypha::cyphalm {

enum class AxiomGateFn : std::uint8_t { Sigmoid = 0, Tanh = 1, Eml = 2 };

/// H15: seed-evolved gate grammar — per-dimension eml/sigmoid/tanh mix.
struct AxiomGateGrammar {
    std::vector<AxiomGateFn> i_gate;
    std::vector<AxiomGateFn> f_gate;
    std::vector<AxiomGateFn> g_gate;
    std::vector<AxiomGateFn> o_gate;
};

/// Apply evolved gate fn. Control gates (i/f/o) must stay in ``[0,1]`` — raw ``tanh`` on those
/// roles explodes cell state (H15 300k ``bpc:null``). Candidate gate ``g`` maps to ``[-1,1]``.
inline double apply_axiom_gate(AxiomGateFn fn, double gate_pre, double state_ref, bool candidate = false) {
    if (candidate) {
        switch (fn) {
            case AxiomGateFn::Sigmoid:
                return 2.0 * sigmoid(gate_pre) - 1.0;
            case AxiomGateFn::Tanh:
                return std::tanh(gate_pre);
            case AxiomGateFn::Eml:
                return 2.0 * eml_nand(gate_pre, state_ref) - 1.0;
        }
    } else {
        switch (fn) {
            case AxiomGateFn::Sigmoid:
                return sigmoid(gate_pre);
            case AxiomGateFn::Tanh:
                // Affine map keeps LSTM forget/input/output semantics in [0, 1].
                return 0.5 * (std::tanh(gate_pre) + 1.0);
            case AxiomGateFn::Eml:
                return eml_nand(gate_pre, state_ref);
        }
    }
    return candidate ? std::tanh(gate_pre) : sigmoid(gate_pre);
}

inline AxiomGateGrammar axiom_grammar_from_seed(std::uint64_t seed, int hidden) {
    AxiomGateGrammar g;
    g.i_gate.resize(static_cast<std::size_t>(hidden));
    g.f_gate.resize(static_cast<std::size_t>(hidden));
    g.g_gate.resize(static_cast<std::size_t>(hidden));
    g.o_gate.resize(static_cast<std::size_t>(hidden));
    std::uint64_t s = seed ? seed : 1u;
    auto next = [&]() -> std::uint8_t {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint8_t>((s >> 33) % 3u);
    };
    for (int j = 0; j < hidden; ++j) {
        g.i_gate[static_cast<std::size_t>(j)] = static_cast<AxiomGateFn>(next());
        g.f_gate[static_cast<std::size_t>(j)] = static_cast<AxiomGateFn>(next());
        g.g_gate[static_cast<std::size_t>(j)] = static_cast<AxiomGateFn>(next());
        g.o_gate[static_cast<std::size_t>(j)] = static_cast<AxiomGateFn>(next());
    }
    return g;
}

}  // namespace cypha::cyphalm
