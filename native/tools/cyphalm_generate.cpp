/// CLI: byte-level CyphaLM generation via serve path (greedy / beam / top-p / temperature).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--prompt TEXT] [--max-bytes N] [--strategy greedy|beam|temperature|top_k|top_p] "
                 "[--beam W] [--temperature T] [--top-p P] [--top-k K] [--seed S] [--table-bits M] "
                 "[--latency]\n",
                 argv0);
}

std::vector<int> bytes_from_text(const std::string& text, int vocab_size) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (static_cast<int>(c) < vocab_size) {
            out.push_back(static_cast<int>(c));
        }
    }
    return out;
}

std::string strategy_label(const cypha::cyphalm::DecodeParams& params) {
    if (params.strategy == cypha::cyphalm::DecodeStrategy::Beam || params.beam_width > 1) {
        return "beam";
    }
    if (params.strategy == cypha::cyphalm::DecodeStrategy::Greedy) {
        return "greedy";
    }
    if (params.strategy == cypha::cyphalm::DecodeStrategy::TopP) {
        return "top_p";
    }
    if (params.strategy == cypha::cyphalm::DecodeStrategy::TopK) {
        return "top_k";
    }
    return "temperature";
}

}  // namespace

int main(int argc, char** argv) {
    std::string prompt = "The quick brown fox ";
    int max_bytes = 32;
    std::string strategy = "temperature";
    double temperature = 0.9;
    double top_p = 0.9;
    int top_k = 40;
    int beam = 1;
    std::uint64_t seed = 42;
    int table_bits = 16;
    bool print_latency = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "--max-bytes" || arg == "--max-tokens") && i + 1 < argc) {
            max_bytes = std::atoi(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if ((arg == "--top-p" || arg == "--top_p") && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if ((arg == "--top-k" || arg == "--top_k") && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--beam" && i + 1 < argc) {
            beam = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::atoi(argv[++i]);
        } else if (arg == "--latency") {
            print_latency = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;

    cypha::cyphalm::CyphaLMModel model(cfg);
    const std::vector<int> prompt_ids = bytes_from_text(prompt, cfg.vocab_size);

    cypha::cyphalm::DecodeParams params;
    params.strategy = cypha::cyphalm::decode_strategy_from_string(strategy);
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.beam_width = beam;
    params.seed = seed;

    if (params.strategy == cypha::cyphalm::DecodeStrategy::Beam && params.beam_width < 2) {
        params.beam_width = 4;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto out = cypha::cyphalm::generate_decode(model, prompt_ids, max_bytes, params);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string completion;
    completion.reserve(out.generated_ids.size());
    for (int id : out.generated_ids) {
        if (id >= 0 && id < 256) {
            completion.push_back(static_cast<char>(id));
        }
    }

    const std::string label = strategy_label(params);
    std::printf("strategy=%s beam=%d top_p=%.3f temperature=%.3f prompt_bytes=%zu generated_bytes=%zu\n",
                label.c_str(), params.beam_width, top_p, temperature, prompt.size(),
                completion.size());
    if (print_latency) {
        std::printf("latency_ms=%.3f\n", elapsed_ms);
    }
    std::printf("--- completion ---\n%s\n", completion.c_str());
    return out.generated_ids.empty() ? 1 : 0;
}
