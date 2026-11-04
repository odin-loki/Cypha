// English function words for K5's discourse helpers (entity_grid.hpp, lda.hpp; docs/reports/K5_DISCOURSE.md).
// Lower case. A fixed list, not learned, so the helpers read nothing of SCRIBE.
//
// Author: Odin Loch
#pragma once

#include <string>
#include <string_view>
#include <unordered_set>

namespace cypha::cyphalm::words {

inline const std::unordered_set<std::string_view>& stop() {
    static const std::unordered_set<std::string_view> s = {
        "a", "about", "above", "after", "again", "against", "all", "also", "although", "am", "among", "an", "and", "another",
        "any", "are", "around", "as", "at", "be", "became", "because", "become", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "either",
        "even", "ever", "every", "few", "first", "for", "from", "further", "had", "has", "have", "having", "he", "her",
        "here", "hers", "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its",
        "itself", "just", "last", "later", "least", "less", "like", "made", "make", "many", "may", "me", "might", "more",
        "most", "much", "must", "my", "myself", "near", "neither", "never", "new", "no", "nor", "not", "now", "of", "off",
        "often", "on", "once", "one", "only", "or", "other", "others", "our", "ours", "out", "over", "own", "per", "rather",
        "same", "second", "several", "she", "should", "since", "so", "some", "still", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "though", "three",
        "through", "thus", "to", "too", "two", "under", "until", "up", "upon", "us", "used", "very", "was", "we", "well",
        "were", "what", "when", "where", "whether", "which", "while", "who", "whom", "whose", "why", "will", "with",
        "within", "without", "would", "yet", "you", "your", "yours", "s", "'s", "use", "known", "called", "including",
        "within", "throughout", "towards", "toward", "across", "along", "beyond", "via", "whereas", "unlike", "despite",
        "according", "due", "onto", "able", "another", "certain", "given", "less", "number", "part", "since", "various"};
    return s;
}

inline const std::unordered_set<std::string_view>& determiners() {
    static const std::unordered_set<std::string_view> s = {"the", "a", "an", "this", "that", "these", "those", "its",
                                                           "his", "her", "their", "our", "my", "your", "each", "every",
                                                           "some", "many", "several", "such", "no", "any", "both",
                                                           "another", "whose", "most", "all"};
    return s;
}

inline const std::unordered_set<std::string_view>& prepositions() {
    static const std::unordered_set<std::string_view> s = {
        "of", "in", "on", "at", "by", "for", "with", "from", "to", "into", "onto", "upon", "about", "after", "before",
        "during", "between", "among", "under", "over", "through", "throughout", "against", "within", "without", "across",
        "along", "around", "towards", "toward", "near", "via", "since", "until", "behind", "beyond", "despite", "unlike",
        "like", "than", "as"};
    return s;
}

inline std::string lower(std::string_view t) {
    std::string s(t);
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

inline bool alpha(std::string_view t) {
    if (t.empty()) return false;
    for (const char c : t)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '\'')) return false;
    return (t[0] >= 'A' && t[0] <= 'Z') || (t[0] >= 'a' && t[0] <= 'z');
}

inline bool capital(std::string_view t) { return !t.empty() && t[0] >= 'A' && t[0] <= 'Z'; }

// A crude stem: lower case, a plural -s removed (not -ss, not short words). "genes" -> "gene", "class" -> "class".
inline std::string stem(std::string_view t) {
    std::string s = lower(t);
    if (s.size() > 3 && s.back() == 's' && s[s.size() - 2] != 's' && s[s.size() - 2] != 'u' && s[s.size() - 2] != 'i') s.pop_back();
    return s;
}

// Sentence ends: K5's tokens ".", "!", "?" and the line break.
inline bool sentence_end(std::string_view t) { return t == "." || t == "!" || t == "?" || t == "\n"; }

}  // namespace cypha::cyphalm::words
