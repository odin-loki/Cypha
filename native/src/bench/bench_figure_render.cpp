#include "cypha/bench/bench_figure_render.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace cypha::bench {

namespace {

struct Color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

constexpr Color kWhite{255, 255, 255};
constexpr Color kBlack{30, 30, 30};
constexpr Color kGrid{220, 220, 220};
constexpr Color kAxis{100, 100, 100};

constexpr Color kSeriesColors[] = {
    {31, 119, 180},
    {255, 127, 14},
    {44, 160, 44},
    {214, 39, 40},
    {148, 103, 189},
    {140, 86, 75},
};

constexpr int kCanvasW = 960;
constexpr int kCanvasH = 540;
constexpr int kMarginL = 72;
constexpr int kMarginR = 32;
constexpr int kMarginT = 56;
constexpr int kMarginB = 120;

class RasterCanvas {
public:
    RasterCanvas() : px_(static_cast<std::size_t>(kCanvasW * kCanvasH * 3), 255) {}

    void fill_rect(int x, int y, int w, int h, Color c) {
        const int x0 = std::max(0, x);
        const int y0 = std::max(0, y);
        const int x1 = std::min(kCanvasW, x + w);
        const int y1 = std::min(kCanvasH, y + h);
        for (int py = y0; py < y1; ++py) {
            for (int px = x0; px < x1; ++px) set(px, py, c);
        }
    }

    void hline(int x, int y, int len, Color c) { fill_rect(x, y, len, 1, c); }

    void vline(int x, int y, int len, Color c) { fill_rect(x, y, 1, len, c); }

    void text(int x, int y, const std::string& s, Color c, int scale = 2) const {
        int cx = x;
        for (char ch : s) {
            draw_glyph(cx, y, ch, c, scale);
            cx += (glyph_width(ch) + 1) * scale;
        }
    }

    int text_width(const std::string& s, int scale = 2) const {
        int w = 0;
        for (char ch : s) w += (glyph_width(ch) + 1) * scale;
        return std::max(0, w - scale);
    }

    bool write_png(const std::filesystem::path& path) const {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        return stbi_write_png(path.string().c_str(), kCanvasW, kCanvasH, 3, px_.data(), kCanvasW * 3) != 0;
    }

private:
    mutable std::vector<uint8_t> px_;

    void set(int x, int y, Color c) const {
        if (x < 0 || y < 0 || x >= kCanvasW || y >= kCanvasH) return;
        const std::size_t i = static_cast<std::size_t>((y * kCanvasW + x) * 3);
        px_[i] = c.r;
        px_[i + 1] = c.g;
        px_[i + 2] = c.b;
    }

    static int glyph_width(char ch) {
        if (ch == ' ') return 3;
        if (ch == '.' || ch == ',' || ch == ':' || ch == '-' || ch == '(' || ch == ')') return 3;
        return 5;
    }

    struct GlyphEntry {
        char ch;
        const char* rows[5];
    };

    static const GlyphEntry* glyph_for(char ch) {
        static const GlyphEntry kDefault = {'?', {"01110", "10001", "00100", "10001", "01110"}};
        static const GlyphEntry kGlyphs[] = {
            {' ', {"00000", "00000", "00000", "00000", "00000"}},
            {'0', {"01110", "10001", "10011", "10101", "01110"}},
            {'1', {"00100", "01100", "00100", "00100", "01110"}},
            {'2', {"01110", "10001", "00010", "00100", "11111"}},
            {'3', {"11110", "00001", "00110", "00001", "11110"}},
            {'4', {"10010", "10010", "11111", "00010", "00010"}},
            {'5', {"11111", "10000", "11110", "00001", "11110"}},
            {'6', {"01110", "10000", "11110", "10001", "01110"}},
            {'7', {"11111", "00001", "00010", "00100", "01000"}},
            {'8', {"01110", "10001", "01110", "10001", "01110"}},
            {'9', {"01110", "10001", "01111", "00001", "01110"}},
            {'.', {"00000", "00000", "00000", "00000", "00100"}},
            {'-', {"00000", "00000", "11111", "00000", "00000"}},
            {':', {"00000", "00100", "00000", "00100", "00000"}},
            {'(', {"00010", "00100", "00100", "00100", "00010"}},
            {')', {"01000", "00100", "00100", "00100", "01000"}},
            {'_', {"00000", "00000", "00000", "00000", "11111"}},
            {'A', {"01110", "10001", "11111", "10001", "10001"}},
            {'B', {"11110", "10001", "11110", "10001", "11110"}},
            {'C', {"01110", "10000", "10000", "10000", "01110"}},
            {'D', {"11110", "10001", "10001", "10001", "11110"}},
            {'E', {"11111", "10000", "11110", "10000", "11111"}},
            {'F', {"11111", "10000", "11110", "10000", "10000"}},
            {'G', {"01110", "10000", "10011", "10001", "01110"}},
            {'H', {"10001", "10001", "11111", "10001", "10001"}},
            {'I', {"01110", "00100", "00100", "00100", "01110"}},
            {'L', {"10000", "10000", "10000", "10000", "11111"}},
            {'M', {"10001", "11011", "10101", "10001", "10001"}},
            {'N', {"10001", "11001", "10101", "10011", "10001"}},
            {'O', {"01110", "10001", "10001", "10001", "01110"}},
            {'P', {"11110", "10001", "11110", "10000", "10000"}},
            {'R', {"11110", "10001", "11110", "10100", "10001"}},
            {'S', {"01111", "10000", "01110", "00001", "11110"}},
            {'T', {"11111", "00100", "00100", "00100", "00100"}},
            {'U', {"10001", "10001", "10001", "10001", "01110"}},
            {'Y', {"10001", "10001", "01110", "00100", "00100"}},
            {'a', {"00000", "01110", "00001", "01111", "01110"}},
            {'c', {"00000", "01110", "10000", "10000", "01110"}},
            {'e', {"00000", "01110", "11111", "10000", "01110"}},
            {'g', {"00000", "01111", "10001", "01111", "00011"}},
            {'h', {"00000", "10110", "10001", "10001", "10001"}},
            {'i', {"00100", "00000", "01100", "00100", "01110"}},
            {'l', {"01100", "00100", "00100", "00100", "01110"}},
            {'m', {"00000", "11010", "10101", "10001", "10001"}},
            {'n', {"00000", "10110", "10001", "10001", "10001"}},
            {'o', {"00000", "01110", "10001", "10001", "01110"}},
            {'r', {"00000", "10110", "11001", "10000", "10000"}},
            {'s', {"00000", "01111", "10000", "00111", "11110"}},
            {'t', {"00100", "01110", "00100", "00100", "00011"}},
            {'u', {"00000", "10001", "10001", "10001", "01111"}},
            {'y', {"00000", "10001", "10001", "01111", "00011"}},
        };
        for (const auto& entry : kGlyphs) {
            if (entry.ch == ch) return &entry;
        }
        return &kDefault;
    }

    void draw_glyph(int x, int y, char ch, Color c, int scale) const {
        const GlyphEntry* glyph = glyph_for(ch);
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                if (glyph->rows[row][col] == '1') {
                    fill_rect_scaled(x + col * scale, y + row * scale, scale, scale, c);
                }
            }
        }
    }

    void fill_rect_scaled(int x, int y, int w, int h, Color c) const {
        for (int py = y; py < y + h; ++py) {
            for (int px = x; px < x + w; ++px) set(px, py, c);
        }
    }
};

std::string truncate_label(const std::string& s, std::size_t max_len = 14) {
    if (s.size() <= max_len) return s;
    return s.substr(0, max_len - 1) + ".";
}

std::vector<double> series_values(const ProfileJson& series_entry) {
    std::vector<double> out;
    if (!series_entry.contains("values") || !series_entry["values"].is_array()) return out;
    for (const auto& v : series_entry["values"]) {
        out.push_back(v.is_number() ? v.get<double>() : 0.0);
    }
    return out;
}

double y_max(const std::vector<std::vector<double>>& all_series) {
    double m = 0.0;
    for (const auto& s : all_series) {
        for (double v : s) m = std::max(m, v);
    }
    if (m <= 0.0) return 1.0;
    return m * 1.12;
}

void draw_y_ticks(RasterCanvas& canvas, int plot_x, int plot_y, int plot_h, double ymax) {
    constexpr int kTicks = 5;
    for (int t = 0; t <= kTicks; ++t) {
        const double frac = static_cast<double>(t) / kTicks;
        const int y = plot_y + plot_h - static_cast<int>(std::lround(frac * plot_h));
        canvas.hline(plot_x, y, kCanvasW - kMarginR - plot_x, kGrid);
        const double val = frac * ymax;
        char buf[16];
        std::snprintf(buf, sizeof(buf), t == kTicks ? "%.2f" : "%.1f", val);
        canvas.text(plot_x - 44, y - 5, buf, kAxis, 2);
    }
    canvas.vline(plot_x, plot_y, plot_h, kAxis);
    canvas.hline(plot_x, plot_y + plot_h, kCanvasW - kMarginL - kMarginR, kAxis);
}

void draw_legend(RasterCanvas& canvas, int x, int y, const std::vector<std::string>& names) {
    int cx = x;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const Color c = kSeriesColors[i % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        canvas.fill_rect(cx, y, 14, 14, c);
        canvas.text(cx + 18, y, names[i], kBlack, 2);
        cx += 18 + canvas.text_width(names[i], 2) + 24;
        if (cx > kCanvasW - 120 && i + 1 < names.size()) {
            cx = x;
            y += 22;
        }
    }
}

bool render_grouped_bar(const ProfileJson& figure, const std::filesystem::path& out_path) {
    if (!figure.contains("categories") || !figure["categories"].is_array()) return false;
    if (!figure.contains("series") || !figure["series"].is_array()) return false;

    std::vector<std::string> categories;
    for (const auto& c : figure["categories"]) categories.push_back(c.get<std::string>());

    std::vector<std::string> series_names;
    std::vector<std::vector<double>> series_data;
    for (const auto& s : figure["series"]) {
        if (!s.is_object()) continue;
        series_names.push_back(s.value("name", "series"));
        series_data.push_back(series_values(s));
    }
    if (categories.empty() || series_names.empty()) return false;

    const int plot_x = kMarginL;
    const int plot_y = kMarginT;
    const int plot_w = kCanvasW - kMarginL - kMarginR;
    const int plot_h = kCanvasH - kMarginT - kMarginB;
    const double ymax = y_max(series_data);

    RasterCanvas canvas;
    canvas.fill_rect(0, 0, kCanvasW, kCanvasH, kWhite);

    const std::string title = figure.value("title", figure.value("figure", "chart"));
    canvas.text((kCanvasW - canvas.text_width(title, 2)) / 2, 12, title, kBlack, 2);

    const std::string y_label = figure.value("y_label", "");
    if (!y_label.empty()) {
        canvas.text(8, plot_y + plot_h / 2 - canvas.text_width(y_label, 2) / 2, y_label, kAxis, 2);
    }

    draw_y_ticks(canvas, plot_x, plot_y, plot_h, ymax);
    draw_legend(canvas, plot_x, kCanvasH - 28, series_names);

    const int n_cat = static_cast<int>(categories.size());
    const int n_series = static_cast<int>(series_names.size());
    const double group_w = static_cast<double>(plot_w) / std::max(1, n_cat);
    const double bar_w = group_w * 0.75 / std::max(1, n_series);

    for (int ci = 0; ci < n_cat; ++ci) {
        const double group_x = plot_x + ci * group_w;
        for (int si = 0; si < n_series; ++si) {
            double val = 0.0;
            if (si < static_cast<int>(series_data.size()) && ci < static_cast<int>(series_data[si].size())) {
                val = series_data[si][static_cast<std::size_t>(ci)];
            }
            const int bh = static_cast<int>(std::lround((val / ymax) * plot_h));
            const int bx = static_cast<int>(std::lround(group_x + (group_w - n_series * bar_w) / 2.0 + si * bar_w));
            const int by = plot_y + plot_h - bh;
            const Color c = kSeriesColors[static_cast<std::size_t>(si) % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
            canvas.fill_rect(bx, by, static_cast<int>(std::lround(bar_w)) - 1, bh, c);
        }
        const std::string label = truncate_label(categories[static_cast<std::size_t>(ci)]);
        const int lx = static_cast<int>(std::lround(group_x + group_w / 2.0 - canvas.text_width(label, 2) / 2));
        canvas.text(std::max(plot_x, lx), plot_y + plot_h + 10, label, kBlack, 2);
    }

    return canvas.write_png(out_path);
}

bool render_simple_bar(const ProfileJson& figure, const std::filesystem::path& out_path) {
    if (!figure.contains("categories") || !figure["categories"].is_array()) return false;
    if (!figure.contains("series") || !figure["series"].is_array() || figure["series"].empty()) return false;

    std::vector<std::string> categories;
    for (const auto& c : figure["categories"]) categories.push_back(c.get<std::string>());

    const auto values = series_values(figure["series"][0]);
    if (categories.empty() || values.empty()) return false;

    const int plot_x = kMarginL;
    const int plot_y = kMarginT;
    const int plot_w = kCanvasW - kMarginL - kMarginR;
    const int plot_h = kCanvasH - kMarginT - kMarginB;
    const double ymax = y_max({values});

    RasterCanvas canvas;
    canvas.fill_rect(0, 0, kCanvasW, kCanvasH, kWhite);

    const std::string title = figure.value("title", figure.value("figure", "chart"));
    canvas.text((kCanvasW - canvas.text_width(title, 2)) / 2, 12, title, kBlack, 2);

    const std::string y_label = figure.value("y_label", "");
    if (!y_label.empty()) {
        canvas.text(8, plot_y + plot_h / 2 - canvas.text_width(y_label, 2) / 2, y_label, kAxis, 2);
    }

    draw_y_ticks(canvas, plot_x, plot_y, plot_h, ymax);

    const int n = static_cast<int>(std::min(categories.size(), values.size()));
    const double bar_slot = static_cast<double>(plot_w) / std::max(1, n);
    const double bar_w = bar_slot * 0.6;

    for (int i = 0; i < n; ++i) {
        const double val = values[static_cast<std::size_t>(i)];
        const int bh = static_cast<int>(std::lround((val / ymax) * plot_h));
        const int bx = static_cast<int>(std::lround(plot_x + i * bar_slot + (bar_slot - bar_w) / 2.0));
        const int by = plot_y + plot_h - bh;
        const Color c = kSeriesColors[static_cast<std::size_t>(i) % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        canvas.fill_rect(bx, by, static_cast<int>(std::lround(bar_w)) - 1, bh, c);

        const std::string label = truncate_label(categories[static_cast<std::size_t>(i)]);
        const int lx = bx + static_cast<int>(std::lround(bar_w / 2.0)) - canvas.text_width(label, 2) / 2;
        canvas.text(std::max(plot_x, lx), plot_y + plot_h + 10, label, kBlack, 2);
    }

    return canvas.write_png(out_path);
}

}  // namespace

std::optional<std::filesystem::path> render_figure_png(const ProfileJson& figure,
                                                       const std::filesystem::path& out_path) {
    const std::string chart_type = figure.value("chart_type", "grouped_bar");
    bool ok = false;
    if (chart_type == "bar") {
        ok = render_simple_bar(figure, out_path);
    } else {
        ok = render_grouped_bar(figure, out_path);
    }
    if (!ok) return std::nullopt;
    return out_path;
}

}  // namespace cypha::bench
