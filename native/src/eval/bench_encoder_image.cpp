#include "cypha/bench/bench_encoder_image.hpp"

#include "cypha/bench/bench_paths.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::uint32_t read_be_u32(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("unexpected EOF reading IDX header");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

GrayImage make_image(int rows, int cols, const std::uint8_t* src) {
    GrayImage img;
    img.rows = rows;
    img.cols = cols;
    img.pixels.assign(src, src + static_cast<std::size_t>(rows * cols));
    return img;
}

GrayImage upscale_nearest(const GrayImage& src, int factor) {
    GrayImage out;
    out.rows = src.rows * factor;
    out.cols = src.cols * factor;
    out.pixels.assign(static_cast<std::size_t>(out.rows * out.cols), 0);
    for (int y = 0; y < out.rows; ++y) {
        for (int x = 0; x < out.cols; ++x) {
            const int sy = y / factor;
            const int sx = x / factor;
            out.pixels[static_cast<std::size_t>(y * out.cols + x)] =
                src.pixels[static_cast<std::size_t>(sy * src.cols + sx)];
        }
    }
    return out;
}

VisionDataset synthetic_digits_dataset() {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 8.0);

    VisionDataset ds;
    ds.source = "synthetic_digits";
    const int n = 1200;
    const int n_classes = 10;
    ds.train_images.reserve(static_cast<std::size_t>(n));
    ds.train_labels.reserve(static_cast<std::size_t>(n));
    ds.test_images.clear();
    ds.test_labels.clear();

    for (int i = 0; i < n; ++i) {
        const int label = i % n_classes;
        GrayImage small;
        small.rows = 8;
        small.cols = 8;
        small.pixels.assign(64, 0);
        const int cx = 3 + (label % 3);
        const int cy = 3 + (label / 3);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const double dx = static_cast<double>(x - cx);
                const double dy = static_cast<double>(y - cy);
                const double dist = std::sqrt(dx * dx + dy * dy);
                double v = std::max(0.0, 220.0 - dist * 55.0 - std::abs(label - 4) * 6.0);
                v += noise(rng);
                v = std::clamp(v, 0.0, 255.0);
                small.pixels[static_cast<std::size_t>(y * 8 + x)] = static_cast<std::uint8_t>(v);
            }
        }
        GrayImage img = upscale_nearest(small, 4);
        const std::string lbl = std::to_string(label);
        if (i < static_cast<int>(static_cast<double>(n) * 0.8)) {
            ds.train_images.push_back(std::move(img));
            ds.train_labels.push_back(lbl);
        } else {
            ds.test_images.push_back(std::move(img));
            ds.test_labels.push_back(lbl);
        }
    }
    return ds;
}

std::optional<VisionDataset> try_load_mnist() {
    const fs::path root = data_dir() / "mnist";
    const fs::path train_x = root / "train-images-idx3-ubyte";
    const fs::path train_y = root / "train-labels-idx1-ubyte";
    const fs::path test_x = root / "t10k-images-idx3-ubyte";
    const fs::path test_y = root / "t10k-labels-idx1-ubyte";
    if (!fs::is_regular_file(train_x) || !fs::is_regular_file(train_y) || !fs::is_regular_file(test_x) ||
        !fs::is_regular_file(test_y)) {
        return std::nullopt;
    }
    VisionDataset ds;
    ds.source = "mnist";
    ds.train_images = read_idx3_images(train_x);
    ds.train_labels = read_idx1_labels(train_y);
    ds.test_images = read_idx3_images(test_x);
    ds.test_labels = read_idx1_labels(test_y);
    if (ds.train_images.size() != ds.train_labels.size() || ds.test_images.size() != ds.test_labels.size()) {
        throw std::runtime_error("MNIST image/label count mismatch");
    }
    return ds;
}

}  // namespace

std::vector<GrayImage> read_idx3_images(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open IDX images: " + path.string());
    const std::uint32_t magic = read_be_u32(in);
    if (magic != 2051u) throw std::runtime_error("bad IDX3 magic in " + path.string());
    const std::uint32_t n = read_be_u32(in);
    const std::uint32_t rows = read_be_u32(in);
    const std::uint32_t cols = read_be_u32(in);
    std::vector<GrayImage> images;
    images.reserve(n);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(rows * cols));
    for (std::uint32_t i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        if (!in) throw std::runtime_error("truncated IDX3 payload in " + path.string());
        images.push_back(make_image(static_cast<int>(rows), static_cast<int>(cols), buf.data()));
    }
    return images;
}

std::vector<std::string> read_idx1_labels(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open IDX labels: " + path.string());
    const std::uint32_t magic = read_be_u32(in);
    if (magic != 2049u) throw std::runtime_error("bad IDX1 magic in " + path.string());
    const std::uint32_t n = read_be_u32(in);
    std::vector<std::string> labels;
    labels.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        unsigned char b = 0;
        in.read(reinterpret_cast<char*>(&b), 1);
        if (!in) throw std::runtime_error("truncated IDX1 payload in " + path.string());
        labels.push_back(std::to_string(static_cast<int>(b)));
    }
    return labels;
}

VisionDataset load_vision_dataset() {
    if (auto mnist = try_load_mnist()) return *mnist;
    return synthetic_digits_dataset();
}

std::vector<float> ImageEncoder::raw_pixels(const GrayImage& img) const {
    std::vector<float> out(img.pixels.size());
    for (std::size_t i = 0; i < img.pixels.size(); ++i) {
        out[i] = static_cast<float>(img.pixels[i]) / 255.0f;
    }
    return out;
}

std::vector<float> ImageEncoder::hog_features(const GrayImage& img, int cell_size, int n_bins) const {
    const int rows = img.rows;
    const int cols = img.cols;
    std::vector<double> norm(static_cast<std::size_t>(rows * cols));
    for (int i = 0; i < rows * cols; ++i) {
        norm[static_cast<std::size_t>(i)] = static_cast<double>(img.pixels[static_cast<std::size_t>(i)]) / 255.0;
    }

    auto at = [&](int r, int c) -> double {
        r = std::clamp(r, 0, rows - 1);
        c = std::clamp(c, 0, cols - 1);
        return norm[static_cast<std::size_t>(r * cols + c)];
    };

    std::vector<double> gx(static_cast<std::size_t>(rows * cols));
    std::vector<double> gy(static_cast<std::size_t>(rows * cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            gx[static_cast<std::size_t>(idx)] = at(r, c + 1) - at(r, c - 1);
            gy[static_cast<std::size_t>(idx)] = at(r + 1, c) - at(r - 1, c);
        }
    }

    const int n_cells_x = cols / cell_size;
    const int n_cells_y = rows / cell_size;
    std::vector<float> hog(static_cast<std::size_t>(n_cells_y * n_cells_x * n_bins), 0.0f);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            const double gxv = gx[static_cast<std::size_t>(idx)];
            const double gyv = gy[static_cast<std::size_t>(idx)];
            const double magnitude = std::sqrt(gxv * gxv + gyv * gyv);
            double angle = std::atan2(gyv, gxv);
            if (angle < 0.0) angle += kPi;
            for (int b = 0; b < n_bins; ++b) {
                const double lo = b * kPi / n_bins;
                const double hi = (b + 1) * kPi / n_bins;
                if (angle >= lo && angle < hi) {
                    const int cy = r / cell_size;
                    const int cx = c / cell_size;
                    if (cy < n_cells_y && cx < n_cells_x) {
                        hog[static_cast<std::size_t>((cy * n_cells_x + cx) * n_bins + b)] +=
                            static_cast<float>(magnitude);
                    }
                    break;
                }
            }
        }
    }
    return hog;
}

std::vector<std::vector<float>> ImageEncoder::encode_batch(const std::vector<GrayImage>& images,
                                                           const std::string& mode) const {
    std::vector<std::vector<float>> rows;
    rows.reserve(images.size());
    for (const auto& img : images) {
        rows.push_back(mode == "raw" ? raw_pixels(img) : hog_features(img));
    }
    return rows;
}

}  // namespace cypha::bench
