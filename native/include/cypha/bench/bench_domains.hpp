#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::bench {

using DomainJson = nlohmann::json;

struct DomainSpec {
    std::string tag;
    std::string module_path;
    std::function<DomainJson()> run;
};

void set_tool_dir(const std::filesystem::path& dir);
void set_ssm_diagnose(bool enabled);
int domain_number(const std::string& tag);
std::vector<DomainSpec> all_domains();

}  // namespace cypha::bench
