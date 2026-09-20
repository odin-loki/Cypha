/// Generation quality harness: greedy + temperature samples via generate_decode (serve path).
/// Saves JSON artifacts only — no invented quality scores.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct PromptCase {
    const char* id;
    const char* text;
    const char* source;
};

const PromptCase kDefaultPrompts[] = {
    {"enwik_opening",
     "<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi="
     "\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation="
     "\"http://www.mediawiki.org/xml/export-0.3/ http://www.mediawiki.org/xml/"
     "export-0.3.xsd\" version=\"0.3\" xml:lang=\"en\">\n  <page>\n    <title>",
     "enwik8 XML header"},
    {"alice_opening",
     "Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
     "nothing to do: once or twice she had peeped into the book her sister was reading, but it "
     "had no pictures or conversations in it, ",
     "gutenberg/alice"},
    {"code_c",
     "#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world\\n\");\n    return 0;\n"
     "}\n\n// ",
     "synthetic C"},
    {"ascii_prose",
     "The quick brown fox jumps over the lazy dog. In 2026, byte-level language models predict "
     "the next character from context. ",
     "synthetic prose"},
    {"xml_tag",
     "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item id=\"1\">",
     "synthetic XML"},
};

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

std::string ids_to_text(const std::vector<int>& ids) {
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id >= 0 && id < 256) {
            out.push_back(static_cast<char>(id));
        }
    }
    return out;
}

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(static_cast<char>(c));
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
            out += buf;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

nlohmann::json step_array(const std::vector<cypha::cyphalm::GenerateStep>& steps) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& s : steps) {
        nlohmann::json row;
        row["token_id"] = s.token_id;
        row["loss"] = s.loss;
        row["epistemic_var"] = s.epistemic_var;
        row["aleatoric_var"] = s.aleatoric_var;
        row["halted"] = s.halted;
        arr.push_back(row);
    }
    return arr;
}

nlohmann::json run_case(cypha::cyphalm::CyphaLMModel& model, const PromptCase& prompt,
                        const cypha::cyphalm::DecodeParams& params, int max_tokens) {
    nlohmann::json j;
    j["prompt_id"] = prompt.id;
    j["prompt_source"] = prompt.source;
    j["prompt_text"] = prompt.text;
    j["prompt_bytes"] = std::string(prompt.text).size();
    j["strategy"] = params.strategy == cypha::cyphalm::DecodeStrategy::Greedy ? "greedy" : "temperature";
    j["temperature"] = params.temperature;
    j["top_k"] = params.top_k;
    j["seed"] = params.seed;
    j["max_tokens"] = max_tokens;

    const std::vector<int> prompt_ids =
        bytes_from_text(prompt.text, model.config().vocab_size);
    const auto t0 = Clock::now();
    const auto out = cypha::cyphalm::generate_decode(model, prompt_ids, max_tokens, params);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    j["generated_bytes"] = out.generated_ids.size();
    j["completion_text"] = ids_to_text(out.generated_ids);
    j["decode_ms"] = ms;
    j["halted_on_uncertainty"] = out.halted_on_uncertainty;
    j["halted_on_epistemic"] = out.halted_on_epistemic;
    j["per_step"] = step_array(out.per_step);
    return j;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--out <path.json>] [--max-tokens N] [--table-bits M]\n"
                 "  Runs built-in prompt battery with greedy + temperature decoding.\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path;
    int max_tokens = 48;
    int table_bits = 22;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::atoi(argv[++i]);
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
    cypha::cyphalm::normalize_hp_table_bits(cfg);

    cypha::cyphalm::CyphaLMModel model(cfg);

    nlohmann::json root;
    root["harness"] = "cyphalm_generation_harness";
    root["hp_profile"] = "gate24";
    root["hp_table_bits"] = cfg.hp_table_bits;
    root["max_tokens"] = max_tokens;
    root["note"] =
        "Qualitative sampling only. No BLEU/perplexity/quality scores. Completions are from "
        "untrained gate24 hp (cold start) via generate_decode serve path.";

    nlohmann::json runs = nlohmann::json::array();
    for (const auto& prompt : kDefaultPrompts) {
        cypha::cyphalm::DecodeParams greedy;
        greedy.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
        greedy.temperature = 0.0;
        greedy.seed = 42;
        runs.push_back(run_case(model, prompt, greedy, max_tokens));

        cypha::cyphalm::DecodeParams temp;
        temp.strategy = cypha::cyphalm::DecodeStrategy::Temperature;
        temp.temperature = 0.9;
        temp.top_k = 40;
        temp.seed = 42;
        runs.push_back(run_case(model, prompt, temp, max_tokens));

        cypha::cyphalm::DecodeParams temp2;
        temp2.strategy = cypha::cyphalm::DecodeStrategy::Temperature;
        temp2.temperature = 0.9;
        temp2.top_k = 40;
        temp2.seed = 137;
        runs.push_back(run_case(model, prompt, temp2, max_tokens));
    }
    root["runs"] = runs;
    root["run_count"] = runs.size();

    const std::string json_text = root.dump(2);
    if (!out_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(out_path).parent_path(), ec);
        std::ofstream out(out_path);
        if (!out) {
            std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
            return 1;
        }
        out << json_text << '\n';
        std::fprintf(stderr, "wrote %s (%zu runs)\n", out_path.c_str(), runs.size());
    }
    std::cout << json_text << '\n';
    return 0;
}
