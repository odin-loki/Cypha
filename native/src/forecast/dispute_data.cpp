#include "cypha/forecast/dispute_data.hpp"

#include "cypha/csv_ingest.hpp"
#include "cypha/forecast/event_vocabulary.hpp"
#include "cypha/forecast/views_scoring.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>

namespace cypha::forecast {
namespace {

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

}  // namespace

// The CSVs under bench/data/forecast were saved as UTF-16 by a Windows editor (little-endian, and gml_mid.csv
// big-endian, without a byte-order mark). Read as bytes, every field held NULs, so every GDELT event loaded as
// cameo 0 in the global theater (one token) and every MID record as zeros (FORECAST_EVENT_VOCAB.md). UTF-16 in
// either order, with or without a mark, is now decoded to UTF-8; a UTF-8 mark is dropped.
std::string decode_text_utf8(const std::string& raw) {
  const auto* b = reinterpret_cast<const unsigned char*>(raw.data());
  const std::size_t n = raw.size();
  if (n >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF) {
    return raw.substr(3);
  }
  int order = 0;  // 1 little-endian, 2 big-endian
  std::size_t start = 0;
  if (n >= 2 && b[0] == 0xFF && b[1] == 0xFE) {
    order = 1, start = 2;
  } else if (n >= 2 && b[0] == 0xFE && b[1] == 0xFF) {
    order = 2, start = 2;
  } else if (n >= 4) {
    std::size_t zero_even = 0, zero_odd = 0;
    const std::size_t m = std::min<std::size_t>(n & ~std::size_t{1}, 4096);
    for (std::size_t i = 0; i < m; i += 2) {
      zero_even += b[i] == 0;
      zero_odd += b[i + 1] == 0;
    }
    const std::size_t pairs = m / 2;
    if (zero_odd * 10 >= pairs * 4 && zero_even * 10 < pairs) order = 1;
    if (zero_even * 10 >= pairs * 4 && zero_odd * 10 < pairs) order = 2;
  }
  if (order == 0) {
    return raw;
  }
  std::string out;
  out.reserve(n / 2);
  auto unit = [&](std::size_t i) -> std::uint32_t {
    return order == 1 ? (static_cast<std::uint32_t>(b[i]) | (static_cast<std::uint32_t>(b[i + 1]) << 8))
                      : ((static_cast<std::uint32_t>(b[i]) << 8) | static_cast<std::uint32_t>(b[i + 1]));
  };
  for (std::size_t i = start; i + 1 < n; i += 2) {
    std::uint32_t cp = unit(i);
    if (cp >= 0xD800 && cp < 0xDC00 && i + 3 < n) {
      const std::uint32_t lo = unit(i + 2);
      if (lo >= 0xDC00 && lo < 0xE000) {
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        i += 2;
      }
    }
    append_utf8(out, cp);
  }
  return out;
}

namespace {

std::string read_file_or_empty(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return {};
  }
  return decode_text_utf8(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
}

int parse_int(const std::string& s) {
  try {
    return std::stoi(s);
  } catch (...) {
    return 0;
  }
}

double parse_double(const std::string& s) {
  try {
    return std::stod(s);
  } catch (...) {
    return 0.0;
  }
}

std::string lower_trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.pop_back();
  }
  std::size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
    ++start;
  }
  s = s.substr(start);
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

int col_index(const std::vector<std::string>& header, const std::string& name) {
  const std::string want = lower_trim(name);
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (lower_trim(header[i]) == want) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

Theater theater_from_dyad(const std::string& dyad) {
  const std::string d = lower_trim(dyad);
  if (d.find("chn") != std::string::npos || d.find("twn") != std::string::npos) {
    return Theater::Taiwan;
  }
  if (d.find("ukr") != std::string::npos || d.find("rus") != std::string::npos) {
    return Theater::Ukraine;
  }
  if (d.find("irn") != std::string::npos || d.find("isr") != std::string::npos ||
      d.find("pse") != std::string::npos) {
    return Theater::MiddleEast;
  }
  if (d.find("prk") != std::string::npos || d.find("kor") != std::string::npos) {
    return Theater::Korea;
  }
  if (d.find("ind") != std::string::npos || d.find("pak") != std::string::npos) {
    return Theater::SouthAsia;
  }
  if (d.find("eth") != std::string::npos || d.find("eri") != std::string::npos ||
      d.find("sud") != std::string::npos) {
    return Theater::Africa;
  }
  return Theater::Global;
}

}  // namespace

std::vector<DisputeRecord> load_gml_mid_csv(const std::filesystem::path& path) {
  std::vector<DisputeRecord> out;
  const auto text = read_file_or_empty(path);
  if (text.empty()) {
    return out;
  }
  const auto rows = parse_csv_utf8(text, ',');
  if (rows.size() < 2) {
    return out;
  }
  const auto& header = rows[0];
  const int c_year = col_index(header, "year");
  const int c_dyad = col_index(header, "dyad");
  const int c_theater = col_index(header, "theater");
  const int c_host = col_index(header, "hostility");
  const int c_prev = col_index(header, "prev_hostility");
  const int c_gp = col_index(header, "great_power");
  const int c_dur = col_index(header, "duration");
  const int c_gdp = col_index(header, "gdp_ratio");
  const int c_esc = col_index(header, "escalated");

  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (r.empty()) continue;
    DisputeRecord rec;
    auto at = [&](int c) -> std::string { return c >= 0 && c < static_cast<int>(r.size()) ? r[static_cast<std::size_t>(c)] : ""; };
    rec.year = parse_int(at(c_year));
    rec.dyad = at(c_dyad);
    rec.theater = c_theater >= 0 ? at(c_theater) : theater_name(theater_from_dyad(rec.dyad));
    rec.hostility = parse_int(at(c_host));
    if (rec.hostility <= 0) rec.hostility = 1;
    rec.prev_hostility = c_prev >= 0 ? parse_int(at(c_prev)) : rec.hostility;
    rec.great_power = c_gp >= 0 ? parse_int(at(c_gp)) : 0;
    rec.duration = c_dur >= 0 ? parse_int(at(c_dur)) : 0;
    rec.gdp_ratio = c_gdp >= 0 ? parse_double(at(c_gdp)) : 1.0;
    if (c_esc >= 0) {
      rec.escalated = parse_int(at(c_esc));
    } else {
      rec.escalated = rec.hostility > rec.prev_hostility ? 1 : 0;
    }
    out.push_back(rec);
  }
  return out;
}

std::vector<ViewsObservation> load_views_csv(const std::filesystem::path& path,
                                           const std::string& split_filter) {
  std::vector<ViewsObservation> out;
  const auto text = read_file_or_empty(path);
  if (text.empty()) {
    return out;
  }
  const auto rows = parse_csv_utf8(text, ',');
  if (rows.size() < 2) {
    return out;
  }
  const auto& header = rows[0];
  const int c_country = col_index(header, "country");
  const int c_year = col_index(header, "year");
  const int c_month = col_index(header, "month");
  const int c_fatal = col_index(header, "fatalities");
  const int c_split = col_index(header, "split");

  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (r.size() < 4) continue;
    auto at = [&](int c) -> std::string { return c >= 0 && c < static_cast<int>(r.size()) ? r[static_cast<std::size_t>(c)] : ""; };
    if (c_split >= 0 && !split_filter.empty() && lower_trim(at(c_split)) != lower_trim(split_filter)) {
      continue;
    }
    ViewsObservation obs;
    obs.country = at(c_country);
    obs.year = parse_int(at(c_year));
    obs.month = parse_int(at(c_month));
    obs.fatalities = parse_double(at(c_fatal));
    out.push_back(obs);
  }
  return out;
}

std::filesystem::path resolve_views_csv_path(const std::filesystem::path& data_dir) {
  const auto bulk = data_dir / "views_bulk.csv";
  if (std::filesystem::exists(bulk)) {
    return bulk;
  }
  return data_dir / "sample_views.csv";
}

std::vector<GdeltEvent> load_gdelt_csv(const std::filesystem::path& path) {
  std::vector<GdeltEvent> out;
  const auto text = read_file_or_empty(path);
  if (text.empty()) {
    return out;
  }
  const auto rows = parse_csv_utf8(text, ',');
  if (rows.size() < 2) {
    return out;
  }
  const auto& header = rows[0];
  const int c_year = col_index(header, "year");
  const int c_month = col_index(header, "month");
  const int c_day = col_index(header, "day");
  const int c_a1 = col_index(header, "actor1");
  const int c_a2 = col_index(header, "actor2");
  const int c_cameo = col_index(header, "cameo_code");
  const int c_gold = col_index(header, "goldstein");
  const int c_theater = col_index(header, "theater");

  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (r.size() < 5) continue;
    auto at = [&](int c) -> std::string { return c >= 0 && c < static_cast<int>(r.size()) ? r[static_cast<std::size_t>(c)] : ""; };
    GdeltEvent ev;
    ev.year = parse_int(at(c_year));
    ev.month = parse_int(at(c_month));
    ev.day = parse_int(at(c_day));
    ev.actor1 = at(c_a1);
    ev.actor2 = at(c_a2);
    ev.cameo_code = parse_int(at(c_cameo));
    ev.goldstein = parse_int(at(c_gold));
    ev.theater = c_theater >= 0 ? at(c_theater) : "GLB";
    out.push_back(ev);
  }
  return out;
}

DisputeDataset load_dispute_data(const std::filesystem::path& data_dir) {
  DisputeDataset out;

  const std::filesystem::path mid_candidates[] = {
      data_dir / "gml_mid.csv",
      data_dir / "sample_mid.csv",
      data_dir / "mid_v5.csv",
  };
  for (const auto& p : mid_candidates) {
    if (std::filesystem::exists(p)) {
      auto loaded = load_gml_mid_csv(p);
      if (!loaded.empty()) {
        out.mid = std::move(loaded);
        break;
      }
    }
  }

  const std::filesystem::path ucdp_path = data_dir / "sample_ucdp.csv";
  if (std::filesystem::exists(ucdp_path)) {
    const auto text = read_file_or_empty(ucdp_path);
    const auto rows = parse_csv_utf8(text, ',');
    for (std::size_t i = 1; i < rows.size(); ++i) {
      const auto& r = rows[i];
      if (r.size() < 6) continue;
      ConflictRecord rec;
      rec.year = parse_int(r[0]);
      rec.country = r[1];
      rec.theater = r[2];
      rec.intensity = parse_int(r[3]);
      rec.onset = parse_int(r[4]);
      out.ucdp.push_back(rec);
    }
  }

  const auto icb_path = data_dir / "sample_icb.csv";
  if (std::filesystem::exists(icb_path)) {
    const auto text = read_file_or_empty(icb_path);
    const auto rows = parse_csv_utf8(text, ',');
    for (std::size_t i = 1; i < rows.size(); ++i) {
      const auto& r = rows[i];
      if (r.size() < 6) continue;
      CrisisRecord rec;
      rec.year = parse_int(r[0]);
      rec.crisis_id = r[1];
      rec.theater = r[2];
      rec.gp_involved = parse_int(r[3]);
      rec.escalated_to_war = parse_int(r[4]);
      out.icb.push_back(rec);
    }
  }

  const auto gdelt_path = data_dir / "gdelt_events.csv";
  if (std::filesystem::exists(gdelt_path)) {
    out.gdelt = load_gdelt_csv(gdelt_path);
  } else if (std::filesystem::exists(data_dir / "sample_gdelt.csv")) {
    out.gdelt = load_gdelt_csv(data_dir / "sample_gdelt.csv");
  }

  return out;
}

void mid_to_features(const std::vector<DisputeRecord>& records, std::vector<double>& x_rowmajor,
                     std::vector<std::string>& y_class, int& n_features) {
  constexpr int kF = 6;
  n_features = kF;
  x_rowmajor.clear();
  y_class.clear();
  x_rowmajor.reserve(records.size() * static_cast<std::size_t>(kF));
  y_class.reserve(records.size());
  for (const auto& rec : records) {
    x_rowmajor.push_back(static_cast<double>(rec.hostility));
    x_rowmajor.push_back(static_cast<double>(rec.prev_hostility));
    x_rowmajor.push_back(static_cast<double>(rec.great_power));
    x_rowmajor.push_back(static_cast<double>(rec.duration));
    x_rowmajor.push_back(rec.gdp_ratio);
    x_rowmajor.push_back(static_cast<double>(theater_from_string(rec.theater)));
    y_class.push_back(rec.escalated ? "escalate" : "stable");
  }
}

std::vector<std::uint32_t> gdelt_to_token_sequence(const std::vector<GdeltEvent>& events,
                                                   const EventVocabulary& vocab) {
  std::vector<std::uint32_t> seq;
  seq.reserve(events.size());
  for (const auto& ev : events) {
    const Theater t = theater_from_string(ev.theater);
  const auto level = escalation_from_cameo_root(std::max(1, ev.cameo_code / 10));
    seq.push_back(vocab.from_cameo(ev.cameo_code, t, level));
  }
  return seq;
}

}  // namespace cypha::forecast
