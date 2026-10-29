#pragma once

#include "cypha/forecast/views_scoring.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cypha::forecast {

/// One MID/GML-style dispute-year observation for node-level escalation training.
struct DisputeRecord {
  int year{0};
  std::string dyad;
  std::string theater;
  int hostility{1};       // 1=no dispute .. 5=war
  int prev_hostility{1};
  int great_power{0};     // 0/1 dyad involves great power
  int duration{0};
  double gdp_ratio{1.0};
  int escalated{0};        // label: 1 if hostility increased vs prev year
};

/// UCDP-style armed conflict onset record.
struct ConflictRecord {
  int year{0};
  std::string country;
  std::string theater;
  int intensity{0};       // battle deaths bucket
  int onset{0};           // label: new conflict this year
};

/// ICB-style crisis with great-power involvement.
struct CrisisRecord {
  int year{0};
  std::string crisis_id;
  std::string theater;
  int gp_involved{0};
  int escalated_to_war{0};  // label
};

/// GDELT CAMEO event row for sequence training / live monitoring.
struct GdeltEvent {
  int year{0};
  int month{0};
  int day{0};
  std::string actor1;
  std::string actor2;
  int cameo_code{0};
  int goldstein{0};
  std::string theater;
};

struct DisputeDataset {
  std::vector<DisputeRecord> mid;
  std::vector<ConflictRecord> ucdp;
  std::vector<CrisisRecord> icb;
  std::vector<GdeltEvent> gdelt;
};

/// Text as UTF-8: UTF-16 (either byte order, with or without a mark) is decoded, a UTF-8 mark dropped.
std::string decode_text_utf8(const std::string& raw);

/// Load dispute datasets from CSV under ``data_dir`` (sample_*.csv or bulk gml_mid.csv / gdelt_events.csv / views.csv).
DisputeDataset load_dispute_data(const std::filesystem::path& data_dir);

/// Load GML MID v2-style dyad-year CSV (header: year,dyad,hostility,duration,...).
std::vector<DisputeRecord> load_gml_mid_csv(const std::filesystem::path& path);

/// Load VIEWS country-month fatality CSV (header: country,year,month,fatalities[,split]).
std::vector<ViewsObservation> load_views_csv(const std::filesystem::path& path, const std::string& split_filter = "");

/// Prefer ``views_bulk.csv`` when present (from ``fetch_forecast_data.ps1 -Bulk``), else ``sample_views.csv``.
std::filesystem::path resolve_views_csv_path(const std::filesystem::path& data_dir);

/// Load GDELT event export CSV (flexible column names).
std::vector<GdeltEvent> load_gdelt_csv(const std::filesystem::path& path);

/// Convert MID records to feature matrix + escalation labels for CyphaDIF.
void mid_to_features(const std::vector<DisputeRecord>& records, std::vector<double>& x_rowmajor,
                     std::vector<std::string>& y_class, int& n_features);

/// Convert GDELT events to token sequence ids using ``vocab``.
std::vector<std::uint32_t> gdelt_to_token_sequence(const std::vector<GdeltEvent>& events,
                                                   const class EventVocabulary& vocab);

}  // namespace cypha::forecast
