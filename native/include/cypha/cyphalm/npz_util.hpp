#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm {

/// Minimal NPZ reader for CyphaLM checkpoint projection matrices (.npy inside zip).
class NpzReader {
public:
    static NpzReader open(const std::string& path);

    bool has(const std::string& name) const;
    std::vector<double> read_f64(const std::string& name) const;
    std::vector<std::uint32_t> shape(const std::string& name) const;

private:
    std::unordered_map<std::string, std::vector<std::uint8_t>> arrays_;
    std::unordered_map<std::string, std::vector<std::uint32_t>> shapes_;

    static std::vector<double> parse_npy_f64(const std::vector<std::uint8_t>& raw,
                                             std::vector<std::uint32_t>& shape_out);
};

/// Minimal NPZ writer (stored/uncompressed .npy entries).
class NpzWriter {
public:
    void add_f64(const std::string& name, const std::vector<double>& data,
                 const std::vector<std::uint32_t>& shape);
    void write(const std::string& path) const;

private:
    struct Entry {
        std::string name;
        std::vector<std::uint8_t> payload;
    };
    std::vector<Entry> entries_;
};

}  // namespace cypha::cyphalm
