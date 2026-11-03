/// Smoke: cyphalm_generate decode paths (greedy, top-p, beam) on bit-tree log probs.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

std::vector<int> bytes_from_text(const std::string& text) {
    std::vector<int> out;
    for (unsigned char c : text) {
        out.push_back(static_cast<int>(c));
    }
    return out;
}

bool non_empty(const cypha::cyphalm::GenerateOutput& out) {
    return !out.generated_ids.empty();
}

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 16;

    cypha::cyphalm::CyphaLMModel model(cfg);
    const std::vector<int> prompt = bytes_from_text("Hello ");

    cypha::cyphalm::DecodeParams greedy;
    greedy.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
    greedy.temperature = 0.0;
    const auto gout = cypha::cyphalm::generate_decode(model, prompt, 8, greedy);
    if (!non_empty(gout)) {
        std::puts("cyphalm_generate_smoke FAIL greedy empty");
        return 1;
    }

    model.reset_context();
    cypha::cyphalm::DecodeParams topp;
    topp.strategy = cypha::cyphalm::DecodeStrategy::TopP;
    topp.temperature = 0.8;
    topp.top_p = 0.9;
    topp.seed = 7;
    const auto pout = cypha::cyphalm::generate_decode(model, prompt, 8, topp);
    if (!non_empty(pout)) {
        std::puts("cyphalm_generate_smoke FAIL top_p empty");
        return 1;
    }

    model.reset_context();
    cypha::cyphalm::DecodeParams beam;
    beam.strategy = cypha::cyphalm::DecodeStrategy::Beam;
    beam.beam_width = 2;
    const auto bout = cypha::cyphalm::generate_decode(model, prompt, 4, beam);
    if (!non_empty(bout)) {
        std::puts("cyphalm_generate_smoke FAIL beam empty");
        return 1;
    }
    if (bout.strategy != cypha::cyphalm::DecodeStrategy::Beam) {
        std::puts("cyphalm_generate_smoke FAIL beam strategy tag");
        return 1;
    }

    model.reset_context();
    cypha::cyphalm::DecodeParams beam4p;
    beam4p.beam_width = 4;
    beam4p.strategy = cypha::cyphalm::DecodeStrategy::Beam;
    const auto beam4 = cypha::cyphalm::generate_beam(model, prompt, 4, beam4p);
    if (!non_empty(beam4)) {
        std::puts("cyphalm_generate_smoke FAIL generate_beam empty");
        return 1;
    }

    model.reset_context();
    cypha::cyphalm::DecodeParams via_width;
    via_width.beam_width = 3;
    via_width.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
    const auto wout = cypha::cyphalm::generate_decode(model, prompt, 4, via_width);
    if (!non_empty(wout)) {
        std::puts("cyphalm_generate_smoke FAIL beam_width>1 empty");
        return 1;
    }

    std::printf("cyphalm_generate_smoke OK greedy=%zu top_p=%zu beam2=%zu beam4=%zu\n",
                gout.generated_ids.size(), pout.generated_ids.size(), bout.generated_ids.size(),
                beam4.generated_ids.size());
    return 0;
}
