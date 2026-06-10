#include "cypha/cyphalm/npz_util.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

constexpr std::uint32_t kLocalFileHeaderSig = 0x04034b50u;

std::uint16_t read_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
}

std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0] | (static_cast<std::uint32_t>(p[1]) << 8) |
                                      (static_cast<std::uint32_t>(p[2]) << 16) |
                                      (static_cast<std::uint32_t>(p[3]) << 24));
}

std::string basename_no_ext(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::size_t dot = name.rfind('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

}  // namespace

std::vector<double> NpzReader::parse_npy_f64(const std::vector<std::uint8_t>& raw,
                                             std::vector<std::uint32_t>& shape_out) {
    if (raw.size() < 10) throw std::runtime_error("npy too short");
    if (raw[0] != 0x93 || raw[1] != 'N' || raw[2] != 'U' || raw[3] != 'M' || raw[4] != 'P' ||
        raw[5] != 'Y') {
        throw std::runtime_error("invalid npy magic");
    }
    const std::uint8_t major = raw[6];
    const std::uint8_t minor = raw[7];
    std::size_t header_len = 0;
    std::size_t header_off = 0;
    if (major == 1 && minor == 0) {
        header_len = read_u16(raw.data() + 8);
        header_off = 10;
    } else {
        header_len = read_u32(raw.data() + 8);
        header_off = 12;
    }
    if (header_off + header_len > raw.size()) throw std::runtime_error("npy header overflow");
    const std::string header(reinterpret_cast<const char*>(raw.data() + header_off), header_len);
    if (header.find("<f8") == std::string::npos && header.find("|f8") == std::string::npos)
        throw std::runtime_error("npy must be float64");

    shape_out.clear();
    const auto shape_pos = header.find("'shape'");
    if (shape_pos != std::string::npos) {
        const auto lparen = header.find('(', shape_pos);
        const auto rparen = header.find(')', lparen);
        if (lparen != std::string::npos && rparen != std::string::npos) {
            std::string inner = header.substr(lparen + 1, rparen - lparen - 1);
            std::replace(inner.begin(), inner.end(), ',', ' ');
            std::istringstream iss(inner);
            std::uint32_t dim = 0;
            while (iss >> dim) shape_out.push_back(dim);
        }
    }
    if (shape_out.empty()) shape_out.push_back(1);

    const std::size_t data_off = header_off + header_len;
    std::uint64_t n = 1;
    for (std::uint32_t d : shape_out) n *= d;
    const std::size_t n_elems = static_cast<std::size_t>(n);
    if (data_off + n_elems * sizeof(double) > raw.size()) throw std::runtime_error("npy data truncated");

    std::vector<double> out(n_elems);
    std::memcpy(out.data(), raw.data() + data_off, n_elems * sizeof(double));
    return out;
}

NpzReader NpzReader::open(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open npz: " + path);
    std::vector<std::uint8_t> zip((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (zip.size() < 22) throw std::runtime_error("npz too small");

    NpzReader reader;
    std::size_t off = 0;
    while (off + 30 <= zip.size()) {
        const std::uint32_t sig = read_u32(zip.data() + off);
        if (sig != kLocalFileHeaderSig) break;
        const std::uint16_t name_len = read_u16(zip.data() + off + 26);
        const std::uint16_t extra_len = read_u16(zip.data() + off + 28);
        const std::uint32_t comp_size = read_u32(zip.data() + off + 18);
        const std::uint16_t method = read_u16(zip.data() + off + 8);
        const std::size_t name_off = off + 30;
        if (name_off + name_len > zip.size()) break;
        const std::string entry_name(reinterpret_cast<const char*>(zip.data() + name_off), name_len);
        const std::size_t data_off = name_off + name_len + extra_len;
        if (data_off + comp_size > zip.size()) break;

        if (method != 0) throw std::runtime_error("compressed npz not supported: " + entry_name);
        if (entry_name.size() > 4 && entry_name.substr(entry_name.size() - 4) == ".npy") {
            const std::string key = basename_no_ext(entry_name);
            std::vector<std::uint8_t> payload(zip.begin() + static_cast<std::ptrdiff_t>(data_off),
                                              zip.begin() + static_cast<std::ptrdiff_t>(data_off + comp_size));
            std::vector<std::uint32_t> shape;
            auto parsed = parse_npy_f64(payload, shape);
            reader.arrays_[key].resize(parsed.size() * sizeof(double));
            std::memcpy(reader.arrays_[key].data(), parsed.data(), parsed.size() * sizeof(double));
            reader.shapes_[key] = shape;
        }
        off = data_off + comp_size;
    }
    return reader;
}

bool NpzReader::has(const std::string& name) const { return arrays_.count(name) > 0; }

std::vector<std::uint32_t> NpzReader::shape(const std::string& name) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) return {};
    return it->second;
}

std::vector<double> NpzReader::read_f64(const std::string& name) const {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) throw std::runtime_error("npz missing array: " + name);
    const std::size_t n = it->second.size() / sizeof(double);
    std::vector<double> out(n);
    std::memcpy(out.data(), it->second.data(), it->second.size());
    return out;
}

void NpzWriter::add_f64(const std::string& name, const std::vector<double>& data,
                        const std::vector<std::uint32_t>& shape) {
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) header += ", ";
        header += std::to_string(shape[i]);
    }
    if (shape.size() == 1) header += ",";
    header += "), }";
    const std::size_t pad = (16 - (10 + header.size()) % 16) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    std::vector<std::uint8_t> npy;
    npy.push_back(0x93);
    npy.insert(npy.end(), {'N', 'U', 'M', 'P', 'Y', 1, 0});
    const std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
    npy.push_back(static_cast<std::uint8_t>(hlen & 0xff));
    npy.push_back(static_cast<std::uint8_t>((hlen >> 8) & 0xff));
    npy.insert(npy.end(), header.begin(), header.end());
    const std::size_t off = npy.size();
    npy.resize(off + data.size() * sizeof(double));
    std::memcpy(npy.data() + off, data.data(), data.size() * sizeof(double));

    entries_.push_back({name + ".npy", std::move(npy)});
}

void NpzWriter::write(const std::string& path) const {
    std::vector<std::uint8_t> zip;
    auto write_u16 = [&](std::uint16_t v) {
        zip.push_back(static_cast<std::uint8_t>(v & 0xff));
        zip.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    };
    auto write_u32 = [&](std::uint32_t v) {
        for (int s = 0; s < 4; ++s) zip.push_back(static_cast<std::uint8_t>((v >> (8 * s)) & 0xff));
    };
    for (const auto& e : entries_) {
        const std::uint32_t crc = 0;  // stored entries; readers ignore CRC for method 0
        const std::uint32_t comp_size = static_cast<std::uint32_t>(e.payload.size());
        const std::uint32_t uncomp_size = comp_size;
        const std::uint16_t name_len = static_cast<std::uint16_t>(e.name.size());
        write_u32(kLocalFileHeaderSig);
        write_u16(20);  // version
        write_u16(0);   // flags
        write_u16(0);   // method stored
        write_u16(0);
        write_u16(0);
        write_u32(crc);
        write_u32(comp_size);
        write_u32(uncomp_size);
        write_u16(name_len);
        write_u16(0);
        zip.insert(zip.end(), e.name.begin(), e.name.end());
        zip.insert(zip.end(), e.payload.begin(), e.payload.end());
    }
    const std::uint32_t cd_start = static_cast<std::uint32_t>(zip.size());
    for (const auto& e : entries_) {
        const std::uint32_t comp_size = static_cast<std::uint32_t>(e.payload.size());
        const std::uint16_t name_len = static_cast<std::uint16_t>(e.name.size());
        write_u32(0x02014b50u);
        write_u16(20);
        write_u16(20);
        write_u16(0);
        write_u16(0);
        write_u16(0);
        write_u16(0);
        write_u32(0);
        write_u32(comp_size);
        write_u32(comp_size);
        write_u16(name_len);
        write_u16(0);
        write_u16(0);
        write_u16(0);
        write_u16(0);
        write_u32(0);
        write_u32(0);
        zip.insert(zip.end(), e.name.begin(), e.name.end());
    }
    const std::uint32_t cd_end = static_cast<std::uint32_t>(zip.size());
    write_u32(0x06054b50u);
    write_u16(0);
    write_u16(0);
    write_u16(static_cast<std::uint16_t>(entries_.size()));
    write_u16(static_cast<std::uint16_t>(entries_.size()));
    write_u32(cd_end - cd_start);
    write_u32(cd_start);
    write_u16(0);

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write npz: " + path);
    out.write(reinterpret_cast<const char*>(zip.data()), static_cast<std::streamsize>(zip.size()));
}

}  // namespace cypha::cyphalm
