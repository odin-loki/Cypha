#include "cypha/forecast/event_vocabulary.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace cypha::forecast {
namespace {

constexpr int kCameoRoots[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

}  // namespace

const char* theater_name(Theater t) {
  switch (t) {
    case Theater::Taiwan:
      return "TWN";
    case Theater::Ukraine:
      return "UKR";
    case Theater::MiddleEast:
      return "MID";
    case Theater::Korea:
      return "PRK";
    case Theater::SouthAsia:
      return "SAS";
    case Theater::Africa:
      return "AFR";
    case Theater::Global:
    default:
      return "GLB";
  }
}

Theater theater_from_string(const std::string& s) {
  if (s == "TWN" || s == "taiwan") return Theater::Taiwan;
  if (s == "UKR" || s == "ukraine") return Theater::Ukraine;
  if (s == "MID" || s == "middle_east") return Theater::MiddleEast;
  if (s == "PRK" || s == "korea") return Theater::Korea;
  if (s == "SAS" || s == "south_asia") return Theater::SouthAsia;
  if (s == "AFR" || s == "africa") return Theater::Africa;
  return Theater::Global;
}

const char* escalation_name(EscalationLevel e) {
  switch (e) {
    case EscalationLevel::Verbal:
      return "VERBAL";
    case EscalationLevel::Material:
      return "MATERIAL";
    case EscalationLevel::ForceThreat:
      return "FORCE_THREAT";
    case EscalationLevel::ForceUse:
      return "FORCE_USE";
    case EscalationLevel::War:
      return "WAR";
    case EscalationLevel::Calm:
    default:
      return "CALM";
  }
}

EscalationLevel escalation_from_mid_hostility(int hostility_1_to_5) {
  if (hostility_1_to_5 <= 1) return EscalationLevel::Calm;
  if (hostility_1_to_5 == 2) return EscalationLevel::Verbal;
  if (hostility_1_to_5 == 3) return EscalationLevel::Material;
  if (hostility_1_to_5 == 4) return EscalationLevel::ForceUse;
  return EscalationLevel::War;
}

EscalationLevel escalation_from_cameo_root(int cameo_root) {
  if (cameo_root <= 3) return EscalationLevel::Verbal;
  if (cameo_root <= 6) return EscalationLevel::Material;
  if (cameo_root <= 8) return EscalationLevel::ForceThreat;
  if (cameo_root <= 14) return EscalationLevel::ForceUse;
  if (cameo_root >= 15) return EscalationLevel::War;
  return EscalationLevel::Calm;
}

std::string EventToken::label() const {
  std::ostringstream os;
  os << theater_name(theater) << '_' << escalation_name(level) << "_C" << cameo_root;
  return os.str();
}

EventVocabulary::EventVocabulary() {
  names_.resize(2);
  names_[kPad] = "<PAD>";
  names_[kUnknown] = "<UNK>";
  token_by_id_.resize(2);
  by_theater_.resize(static_cast<std::size_t>(Theater::Count));

  for (int ti = 0; ti < static_cast<int>(Theater::Count); ++ti) {
    const auto theater = static_cast<Theater>(ti);
    for (int ei = 0; ei < static_cast<int>(EscalationLevel::Count); ++ei) {
      const auto level = static_cast<EscalationLevel>(ei);
      for (int root : kCameoRoots) {
        EventToken tok;
        tok.theater = theater;
        tok.level = level;
        tok.cameo_root = root;
        register_token(tok, tok.label());
      }
    }
  }
}

void EventVocabulary::register_token(const EventToken& tok, const std::string& name) {
  if (name_to_id_.contains(name)) {
    return;
  }
  const std::uint32_t id = static_cast<std::uint32_t>(token_by_id_.size());
  EventToken stored = tok;
  stored.id = id;
  token_by_id_.push_back(stored);
  names_.push_back(name);
  name_to_id_[name] = id;
  by_theater_[static_cast<std::size_t>(tok.theater)].push_back(id);
}

std::uint32_t EventVocabulary::encode(const EventToken& tok) const {
  const auto it = name_to_id_.find(tok.label());
  if (it == name_to_id_.end()) {
    return kUnknown;
  }
  return it->second;
}

EventToken EventVocabulary::decode(std::uint32_t id) const {
  if (id >= token_by_id_.size()) {
    return {};
  }
  return token_by_id_[id];
}

const std::string& EventVocabulary::token_string(std::uint32_t id) const {
  if (id >= names_.size()) {
    static const std::string kUnk = "<UNK>";
    return kUnk;
  }
  return names_[id];
}

std::uint32_t EventVocabulary::from_cameo(int cameo_code, Theater theater, EscalationLevel level) const {
  EventToken tok;
  tok.theater = theater;
  tok.level = level;
  tok.cameo_root = std::max(1, std::min(20, cameo_code / 10));
  if (level == EscalationLevel::Calm) {
    tok.level = escalation_from_cameo_root(tok.cameo_root);
  }
  return encode(tok);
}

const std::vector<std::uint32_t>& EventVocabulary::theater_token_ids(Theater t) const {
  return by_theater_[static_cast<std::size_t>(t)];
}

}  // namespace cypha::forecast
