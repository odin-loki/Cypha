#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cypha::bench {

struct GrayImage {
    int rows{0};
    int cols{0};
    std::vector<std::uint8_t> pixels;  // row-major uint8 greyscale
};

struct VisionDataset {
    std::string source;
    std::vector<GrayImage> train_images;
    std::vector<std::string> train_labels;
    std::vector<GrayImage> test_images;
    std::vector<std::string> test_labels;
};

/// Image feature encoders for greyscale inputs (mirrors ``ImageEncoder``).
class ImageEncoder {
  public:
    std::vector<float> raw_pixels(const GrayImage& img) const;
    std::vector<float> hog_features(const GrayImage& img, int cell_size = 4, int n_bins = 9) const;
    std::vector<std::vector<float>> encode_batch(const std::vector<GrayImage>& images,
                                                 const std::string& mode) const;
};

/// Load MNIST IDX files from ``bench/data/mnist`` or synthesize digits-like data.
VisionDataset load_vision_dataset();

/// Read IDX3 image file (big-endian header).
std::vector<GrayImage> read_idx3_images(const std::filesystem::path& path);

/// Read IDX1 label file.
std::vector<std::string> read_idx1_labels(const std::filesystem::path& path);

}  // namespace cypha::bench
