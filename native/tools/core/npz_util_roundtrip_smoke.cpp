/// Smoke test: NpzWriter -> NpzReader round trip for a small 3x4 matrix.
///
/// NpzWriter/NpzReader only handle float64 (add_f64/read_f64), so the fixture matrix is
/// built as doubles; small exactly-representable values (integers plus a couple of simple
/// fractions) round-trip bit-for-bit through the stored (uncompressed) .npy payload, so
/// this asserts exact equality rather than an epsilon comparison.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "cypha/cyphalm/npz_util.hpp"

int main() {
    namespace fs = std::filesystem;
    using cypha::cyphalm::NpzReader;
    using cypha::cyphalm::NpzWriter;

    constexpr std::uint32_t kRows = 3;
    constexpr std::uint32_t kCols = 4;
    const std::vector<double> matrix = {
        0.0,  1.5,  -2.0, 3.25,
        4.0,  -5.5, 6.0,  7.75,
        -8.0, 9.0,  10.5, -11.25,
    };
    const std::vector<std::uint32_t> shape = {kRows, kCols};

    const fs::path npz_path = fs::temp_directory_path() / "cypha_npz_util_roundtrip_smoke.npz";
    std::error_code ec;
    fs::remove(npz_path, ec);

    int rc = 0;
    try {
        NpzWriter writer;
        writer.add_f64("matrix", matrix, shape);
        writer.write(npz_path.string());

        const NpzReader reader = NpzReader::open(npz_path.string());
        if (!reader.has("matrix")) {
            std::fprintf(stderr, "npz_util_roundtrip_smoke: reader missing 'matrix' array\n");
            rc = 1;
        } else {
            const std::vector<std::uint32_t> read_shape = reader.shape("matrix");
            if (read_shape != shape) {
                std::fprintf(stderr,
                             "npz_util_roundtrip_smoke: shape mismatch (expected %ux%u)\n", kRows,
                             kCols);
                rc = 1;
            } else {
                const std::vector<double> read_back = reader.read_f64("matrix");
                if (read_back.size() != matrix.size()) {
                    std::fprintf(stderr,
                                 "npz_util_roundtrip_smoke: element count mismatch (expected %zu, "
                                 "got %zu)\n",
                                 matrix.size(), read_back.size());
                    rc = 1;
                } else {
                    for (std::size_t i = 0; i < matrix.size(); ++i) {
                        if (read_back[i] != matrix[i]) {
                            std::fprintf(stderr,
                                         "npz_util_roundtrip_smoke: element %zu mismatch (expected "
                                         "%.17g, got %.17g)\n",
                                         i, matrix[i], read_back[i]);
                            rc = 1;
                            break;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "npz_util_roundtrip_smoke: exception: %s\n", ex.what());
        rc = 1;
    }

    fs::remove(npz_path, ec);

    if (rc == 0) {
        std::printf("npz_util_roundtrip_smoke: %ux%u matrix round-trip exact PASS\n", kRows, kCols);
    }
    return rc;
}
