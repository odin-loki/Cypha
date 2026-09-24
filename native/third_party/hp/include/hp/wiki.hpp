#pragma once
//
// hp/wiki.hpp — fx2-cmix / paq8 wiki-XML state machine.
//
// THE AXIS
// --------
// The existing tag model is a generic depth counter. `tag:py` is already the
// most decorrelated expert in the ensemble (mean |corr| 0.621), which means
// the *axis* is right and the *features* are weak. This file gives that
// expert the states fx2-cmix actually uses on enwik:
//
//   WIKITABLE     {| ... |}
//   SQUAREOPEN    [[ ... ]]          wiki links
//   HTLINK        http(s):// ...
//   VERTICALBAR   |  inside table or template
//   CURLY         {{ ... }}          templates / infoboxes
//   COMMENT       <!-- ... -->
//   AMP           &entity;
//   plus linkword = linkword * 2104 + j   (fx2-cmix literal)
//
// All of it is derived from bytes already coded. Nothing is transmitted.
//
// Port of the control flow in fx2-cmix src/models/fxcmv1.cpp (GPL), not a
// copy of its tables. States are a 4-bit enum so they fit a mixer gate.

#include <cstdint>

#include "hp/features.hpp"
#include "hp/models.hpp"

namespace hp {

enum WikiState : int {
    kWkText = 0,
    kWkTag,
    kWkTagEnd,
    kWkAmp,
    kWkComment,
    kWkSquareOpen,
    kWkCurly,
    kWkWikiTable,
    kWkVerticalBar,
    kWkHtLink,
    kWkQuote,
    kWkEntity,
    kWkFirstUpper,   // fx2 FIRSTUPPER — line/word starts with A-Z
    kWkHeader,       // fx2 WIKIHEADER — line starts with '>'
    kWkN = 16
};

// Tracker state for the optional upstream context models (Config::extra_cms):
// wiki bold/italic, word-in-sentence, wiki-state transition, <ref> class.
// Kept outside WikiMachine so the machine's bytes (written raw into v1-v4
// checkpoints) are unchanged; Predictor owns it inline (StreamRewind's memcpy
// covers it) and every write is undo-recorded.
struct WikiExtra {
    int wikibold = 0;   // bit 0 italic '', bit 1 bold '''
    int wb_run = 0;
    int sentpos = 0;    // nth word in the current sentence, 0..31
    int sp_inword = 0;
    int prev_state = 0; // wiki state before the last byte
    int refgroup = 0;   // 0 none/closed, 1 <ref>, 2 <ref name=, 3 <ref group=
    int rg_slash = 0;
    int rg_n = 0;
    int rg_name = 0;
    int rg_group = 0;
    std::uint64_t rg_win = 0;
    std::uint8_t rg_buf[4] = {};

    template <typename T>
    static void set(T& f, T v) {
        if (f != v) {
            hp_undo_note(f);
            f = v;
        }
    }
    int state_trans(int state) const { return (prev_state & 15) | ((state & 15) << 4); }
    void tag_open() {
        set(rg_slash, 0);
        set(rg_n, 0);
        set(rg_win, std::uint64_t{0});
        set(rg_name, 0);
        set(rg_group, 0);
    }
    void page_reset() {
        set(wikibold, 0);
        set(wb_run, 0);
        set(sentpos, 0);
        set(sp_inword, 0);
        set(prev_state, 0);
        set(refgroup, 0);
        tag_open();
    }
    // Per byte, before any state change (upstream HP_STATETRANS_MOD,
    // HP_WIKIBOLD_MOD, HP_SENTPOS_MOD).
    void byte_pre(int c, int state) {
        set(prev_state, state);
        if (c == '\n') {
            set(wikibold, 0);
            set(wb_run, 0);
        } else if (c == '\'') {
            if (wb_run < 5) set(wb_run, wb_run + 1);
        } else {
            if (wb_run == 2)
                set(wikibold, wikibold ^ 1);
            else if (wb_run == 3)
                set(wikibold, wikibold ^ 2);
            else if (wb_run == 4 || wb_run == 5)
                set(wikibold, wikibold ^ 3);
            set(wb_run, 0);
        }
        const int letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        if (c == '.' || c == '?' || c == '!' || c == '\n') {
            set(sentpos, 0);
            set(sp_inword, 0);
        } else if (letter) {
            if (!sp_inword) {
                if (sentpos < 31) set(sentpos, sentpos + 1);
                set(sp_inword, 1);
            }
        } else {
            set(sp_inword, 0);
        }
    }
    // A byte inside a tag (after '<', not '/' or '>'); upstream HP_REFGROUP_MOD.
    void tag_byte(int c) {
        const unsigned ch = static_cast<unsigned>(c & 255);
        const unsigned lc = ch | 32u;
        const unsigned packed = (ch == '=') ? '=' : lc;
        if (lc >= 'a' && lc <= 'z' && rg_n < 4) {
            hp_undo_note(rg_buf[rg_n]);
            rg_buf[rg_n] = static_cast<std::uint8_t>(lc);
            set(rg_n, rg_n + 1);
        }
        set(rg_win, (rg_win << 8) | packed);
        if ((rg_win & 0xffffffffffull) == 0x6e616d653dull) set(rg_name, 1);       // name=
        if ((rg_win & 0xffffffffffffull) == 0x67726f75703dull) set(rg_group, 1);  // group=
    }
    void tag_slash() {
        if (rg_n == 0) set(rg_slash, 1);
    }
    void tag_close() {
        if (rg_n == 3 && rg_buf[0] == 'r' && rg_buf[1] == 'e' && rg_buf[2] == 'f') {
            if (rg_slash)
                set(refgroup, 0);
            else if (rg_group)
                set(refgroup, 3);
            else if (rg_name)
                set(refgroup, 2);
            else
                set(refgroup, 1);
        }
    }
};

class WikiMachine {
 public:
    /// ``ex``: trackers for the optional upstream context models (Predictor
    /// passes them only when one is enabled); nullptr is gate24 exactly.
    void push(int byte, WikiExtra* ex = nullptr) {
        const int c = byte;
        prev2_ = prev1_;
        prev1_ = last_;
        last_ = c;
        if (ex) ex->byte_pre(c, state_);
        stack_feed(c);
        if (c >= 'A' && c <= 'Z') uppergap_ = 0;
        else if (uppergap_ < 255) ++uppergap_;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            if (wl_run_ < 31) ++wl_run_;
        } else {
            if (wl_run_) wordlen_ = wl_run_;
            wl_run_ = 0;
        }

        if (in_title_) {
            if (c == '<') in_title_ = 0;
            else
                title_hash_ = mix64(title_hash_ * 31ull +
                                    static_cast<std::uint64_t>(c));
        }
        if (is_paragraph_ || in_table_) {
            const int up = (c >= 'A' && c <= 'Z');
            const int low = (c >= 'a' && c <= 'z');
            if (up || low) {
                if (cap_n_ == 0) cap_bits_ = up ? 1 : 2;
                else {
                    if (up) cap_bits_ |= 4;
                    if (low) cap_bits_ |= 8;
                }
                if (cap_n_ < 15) ++cap_n_;
                cap_mask_ = cap_bits_ | (cap_n_ << 4);
            } else if (cap_n_) {
                cap_mask_ = cap_bits_ | (cap_n_ << 4);
                cap_n_ = 0;
                cap_bits_ = 0;
            }
        }

        // Close states that have a terminator.
        if (state_ == kWkAmp || state_ == kWkEntity) {
            if (c == ';' || c == ' ' || c == '\n' || c == '<') state_ = kWkText;
        }
        if (state_ == kWkHtLink) {
            if (c == ' ' || c == '\n' || c == '<' || c == ']' || c == '"')
                state_ = kWkText;
        }
        if (state_ == kWkQuote && c != '\'') state_ = kWkText;

        if (state_ == kWkComment) {
            if (prev2_ == '-' && prev1_ == '-' && c == '>') {
                state_ = kWkText;
                if (depth_ > 0) --depth_;
            }
            return;
        }

        if (c == '<') {
            in_tag_ = 1;
            tag_name_ = 0;
            if (depth_ < 15) ++depth_;
            state_ = kWkTag;
            xml_slash_ = 0;
            xml_n_ = 0;
            xml_name_ = 0;
            xml_done_ = 0;
            if (ex) ex->tag_open();
            return;
        }
        if (c == '>' && in_tag_) {
            dump_apply_(ex);
            if (ex) ex->tag_close();
            in_tag_ = 0;
            state_ = kWkText;
            return;
        }
        if (c == '/' && in_tag_) {
            if (xml_n_ == 0) xml_slash_ = 1;
            if (ex) ex->tag_slash();
            if (depth_ > 0) --depth_;
            state_ = kWkTagEnd;
            return;
        }
        if (in_tag_) {
            tag_name_ = mix64(tag_name_ * 31 + static_cast<std::uint64_t>(c));
            if (!xml_done_) {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && xml_n_ < 12) {
                    xml_name_ = xml_name_ * 31ull + static_cast<std::uint64_t>(lc);
                    ++xml_n_;
                } else {
                    xml_done_ = 1;
                }
            }
            if (ex) ex->tag_byte(c);
            if (prev1_ == '!' && c == '-') state_ = kWkComment;
            return;
        }

        // Wiki link [[
        if (prev1_ == '[' && c == '[') {
            state_ = kWkSquareOpen;
            sq_ = 1;
            linkword_ = 0;
            ns_collect_ = 1;
            ns_hash_ = 0;
            return;
        }
        if (state_ == kWkSquareOpen) {
            if (c == ']') {
                if (sq_ > 0) --sq_;
                if (sq_ == 0) {
                    state_ = kWkText;
                    linkword_ = 0;
                    link_pipe_ = 0;
                    disp_ = 0;
                    ns_collect_ = 0;
                    ns_hash_ = 0;
                }
            } else if (c == ':') {
                linkword_ = 0;   // fx2: [category:...] drops the hash
                if (ns_hash_ != 0) ns_collect_ = 0;
            } else {
                // fx2-cmix: linkword = linkword * 2104 + j
                linkword_ = linkword_ * 2104ull + static_cast<std::uint64_t>(c);
                if (c == '|') {
                    link_pipe_ = 1;
                    disp_ = 0;
                } else if (link_pipe_) {
                    disp_ = disp_ * 2104ull + static_cast<std::uint64_t>(c);
                }
                if (ns_collect_) {
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z')
                        ns_hash_ = mix64(ns_hash_ * 31ull +
                                         static_cast<std::uint64_t>(lc));
                    else if (c != ' ' && c != '_')
                        ns_collect_ = 0;
                }
            }
            return;
        }

        // Template / infobox {{
        if (prev1_ == '{' && c == '{') {
            state_ = kWkCurly;
            if (depth_ < 15) ++depth_;
            tpl_collect_ = 1;
            tpl_name_ = 0;
            bar_idx_ = 0;
            key_ = 0;
            key_collect_ = 0;
            return;
        }
        if (prev1_ == '{' && c == '|') {
            state_ = kWkWikiTable;
            in_table_ = 1;
            table_reset();
            return;
        }
        if (prev1_ == '|' && c == '}') {
            state_ = kWkText;
            in_table_ = 0;
            table_reset();
            return;
        }
        if (prev1_ == '}' && c == '}') {
            if (depth_ > 0) --depth_;
            state_ = kWkText;
            tpl_collect_ = 0;
            key_collect_ = 0;
            return;
        }

        if (c == '|' && (in_table_ || state_ == kWkCurly || state_ == kWkWikiTable)) {
            state_ = kWkVerticalBar;
            tpl_collect_ = 0;
            if (bar_idx_ < 31) ++bar_idx_;
            key_ = 0;
            key_collect_ = 1;
            return;
        }
        if (state_ == kWkVerticalBar && c != '|') {
            // stay in table, leave the bar-cell marker after first payload byte
            if (c == '\n') state_ = kWkWikiTable;
            else if (in_table_) state_ = kWkWikiTable;
            else state_ = kWkCurly;
        }

        // http(s)://
        if ((prev2_ == 't' || prev2_ == 'T') &&
            (prev1_ == 't' || prev1_ == 'T') &&
            (c == 'p' || c == 'P')) {
            http_run_ = 3;
        } else if (http_run_ > 0) {
            ++http_run_;
            if (http_run_ >= 4 && prev1_ == '/' && c == '/') {
                state_ = kWkHtLink;
                linkword_ = 0;
                http_run_ = 0;
            }
            if (c == ' ' || c == '\n') http_run_ = 0;
        }
        if (state_ == kWkHtLink && c != ' ' && c != '\n') {
            linkword_ = linkword_ * 2104ull + static_cast<std::uint64_t>(c);
        }

        if (c == '&') state_ = kWkAmp;
        if (c == '\'' && prev1_ == '\'') state_ = kWkQuote;

        // senword: body-prose 2104-hash, only in paragraph, not in link/template.
        const int letter = ((c | 32) >= 'a' && (c | 32) <= 'z');
        if (letter && is_paragraph_ && state_ != kWkSquareOpen &&
            state_ != kWkHtLink && state_ != kWkCurly && !in_table_) {
            senword_ = senword_ * 2104ull + static_cast<std::uint64_t>(c);
        }
        if (c == '.' || c == ',' || c == ':' || c == '(' || c == ')' || c == '\n')
            senword_ = 0;

        if (in_table_) table_push(c);
        if (tpl_collect_) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9'))
                tpl_name_ = mix64(tpl_name_ * 31ull + static_cast<std::uint64_t>(c));
            else if (c != '{')
                tpl_collect_ = 0;
        }
        if (key_collect_) {
            if (c == '=') key_collect_ = 0;
            else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                     (c >= '0' && c <= '9') || c == '_')
                key_ = mix64(key_ * 31ull + static_cast<std::uint64_t>(c));
        }

        if (c == '\n') {
            ++line_;
            first_of_line_ = 1;
            line_kind_ = 0;
            if (state_ == kWkText) ++para_;
            heading_ = 0;
            heading_run_ = 0;
            st_collect_ = 0;
            st_skip_ = 0;
        } else if (first_of_line_ && c != ' ' && c != '\t') {
            first_of_line_ = 0;
            line_kind_ = c;
            // fx2: isParagraph = (fc == FIRSTUPPER); WIKIHEADER = line-start '>'
            is_paragraph_ = (c >= 'A' && c <= 'Z') ? 1 : 0;
            wiki_header_ = (c == '>') ? 1 : 0;
            if (is_paragraph_ && state_ == kWkText) state_ = kWkFirstUpper;
            if (wiki_header_ && state_ == kWkText) state_ = kWkHeader;
            if (c == '=') {
                heading_ = 1;
                heading_run_ = 1;
            }
            if (c == '=') {
                st_skip_ = 1;
                st_collect_ = 0;
                st_hash_ = 0;
            }
        } else {
            if (heading_run_) {
                if (c == '=') {
                    if (heading_ < 6) ++heading_;
                } else {
                    heading_run_ = 0;
                }
            }
            if (st_skip_) {
                if (c != '=') {
                    st_skip_ = 0;
                    st_collect_ = 1;
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z')
                        st_hash_ = mix64(st_hash_ * 31ull +
                                         static_cast<std::uint64_t>(lc));
                }
            } else if (st_collect_) {
                if (c == '=') {
                    st_collect_ = 0;
                } else {
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z')
                        st_hash_ = mix64(st_hash_ * 31ull +
                                         static_cast<std::uint64_t>(lc));
                }
            }
        }
    }

    int state() const { return state_; }
    int depth() const { return depth_; }
    int in_tag() const { return in_tag_; }
    int in_table() const { return in_table_; }
    int is_paragraph() const { return is_paragraph_; }
    int wiki_header() const { return wiki_header_; }
    int line() const { return line_; }
    std::uint64_t tag_name() const { return tag_name_; }
    std::uint64_t linkword() const { return linkword_; }
    std::uint64_t senword() const { return senword_; }
    int line_kind() const { return line_kind_; }
    int sen_group() const {
        if (in_table_ || state_ == kWkWikiTable || state_ == kWkVerticalBar)
            return 2;
        if (state_ == kWkSquareOpen) return 3;
        if (line_kind_ == '*') return 1;
        return 0;
    }
    int nest_markup() const {
        return in_table_ || state_ == kWkSquareOpen || state_ == kWkCurly ||
               state_ == kWkWikiTable || state_ == kWkVerticalBar ||
               state_ == kWkHtLink;
    }
    int mute_words() const {
        return 0;
    }

    // Context packed for the existing tag_ ContextModel.
    std::uint64_t context_key() const {
        return (static_cast<std::uint64_t>(state_ & 15) << 48) |
               (static_cast<std::uint64_t>(depth_ & 15) << 44) |
               (static_cast<std::uint64_t>(in_tag_ & 1) << 43) |
               (static_cast<std::uint64_t>(in_table_ & 1) << 42) |
               (static_cast<std::uint64_t>(is_paragraph_ & 1) << 41) |
               (static_cast<std::uint64_t>(wiki_header_ & 1) << 40) |
               ((tag_name_ & 0x1fffffull) << 21) |
               (linkword_ & 0x1fffffull)
               ;
    }
    int sent_domain() const {
        if (in_table_ || state_ == kWkWikiTable || state_ == kWkVerticalBar)
            return 1;
        if (state_ == kWkSquareOpen || state_ == kWkHtLink) return 2;
        if (state_ == kWkCurly) return 3;
        return 0;
    }
    int nl_mode() const {
        if (in_table_ || state_ == kWkWikiTable) return 1;
        if (wiki_header_) return 2;
        return 0;
    }
    int above_cell() const {
        const int r = (tbl_row_ + 3) & 3;
        const int c = tbl_cell_ > 31 ? 31 : tbl_cell_;
        return tbl_cells_[r][c];
    }
    int cell_first() const {
        const int c = tbl_cell_ > 31 ? 31 : tbl_cell_;
        return tbl_cells_[tbl_row_][c];
    }
    int tbl_cell() const {
        return tbl_cell_;
    }
    std::uint64_t tpl_name() const {
        return tpl_name_;
    }
    std::uint64_t infokey() const {
        return key_;
    }
    int bar_idx() const {
        return bar_idx_;
    }
    int in_ref() const {
        return 0;
    }
    int after_pipe() const {
        return link_pipe_;
    }
    std::uint64_t link_disp() const {
        return disp_;
    }
    std::uint64_t cat_ns() const {
        return ns_hash_;
    }
    int in_redir() const {
        return 0;
    }
    int heading_level() const {
        return heading_;
    }
    int in_ext() const {
        return 0;
    }
    std::uint64_t refname() const {
        return 0;
    }
    std::uint64_t entity() const {
        return 0;
    }
    int indent_level() const {
        return 0;
    }
    int list_level() const {
        return 0;
    }
    std::uint64_t magic() const {
        return 0;
    }
    int in_nowiki() const {
        return 0;
    }
    std::uint64_t page_title() const {
        return title_hash_;
    }
    std::uint64_t page_id() const {
        return 0;
    }
    std::uint64_t username() const {
        return 0;
    }
    int in_text() const {
        return 0;
    }
    int ns_id() const {
        return 0;
    }
    int dump_redir() const {
        return 0;
    }
    std::uint64_t ip_hash() const {
        return 0;
    }
    std::uint64_t rev_comment() const {
        return 0;
    }
    int minor_edit() const {
        return 0;
    }
    std::uint64_t wiki_model() const {
        return 0;
    }
    std::uint64_t sectitle() const {
        return st_hash_;
    }
    std::uint64_t parser_fn() const {
        return 0;
    }
    std::uint64_t table_class() const {
        return 0;
    }
    std::uint64_t anchor() const {
        return 0;
    }
    std::uint64_t pub_id() const {
        return 0;
    }
    std::uint64_t temp_pos() const {
        return 0;
    }
    std::uint64_t wiki_stack() const {
        return (static_cast<std::uint64_t>(fc_stk_[0])) |
               (static_cast<std::uint64_t>(fc_stk_[1]) << 8) |
               (static_cast<std::uint64_t>(fc_stk_[2]) << 16) |
               (static_cast<std::uint64_t>(fc_stk_[3]) << 24) |
               (static_cast<std::uint64_t>(br_stk_[0] & 15) << 32) |
               (static_cast<std::uint64_t>(br_stk_[1] & 15) << 36) |
               (static_cast<std::uint64_t>(br_stk_[2] & 15) << 40) |
               (static_cast<std::uint64_t>(above_cell() & 255) << 44) |
               (static_cast<std::uint64_t>(tbl_cell() & 31) << 52) |
               (static_cast<std::uint64_t>(nl_mode() & 3) << 57) |
               (static_cast<std::uint64_t>(is_paragraph_ & 1) << 59);
    }
    std::uint64_t lang_prefix() const {
        return 0;
    }
    std::uint64_t cat_sort() const {
        return 0;
    }
    int tbl_row() const {
        return 0;
    }
    std::uint64_t file_opt() const {
        return 0;
    }
    std::uint64_t default_sort() const {
        return 0;
    }
    std::uint64_t redir_target() const {
        return 0;
    }
    std::uint64_t dab() const {
        return 0;
    }
    std::uint64_t hatnote() const {
        return 0;
    }
    std::uint64_t last_link() const {
        return 0;
    }
    std::uint64_t first_word() const {
        return 0;
    }
    std::uint64_t year() const {
        return 0;
    }
    int cap_mask() const {
        return cap_mask_;
    }
    std::uint64_t cell_text() const {
        return 0;
    }
    std::uint64_t http_host() const {
        return 0;
    }
    std::uint64_t last_paren() const {
        return 0;
    }
    int list_pos() const {
        return 0;
    }
    std::uint64_t word_shape() const {
        return 0;
    }
    std::uint64_t word_suffix() const {
        return 0;
    }
    std::uint64_t word_prefix() const {
        return 0;
    }
    int char_cls() const {
        return 0;
    }
    int vowel_mask() const {
        return 0;
    }
    std::uint64_t contraction() const {
        return 0;
    }
    std::uint64_t hyphen_word() const {
        return 0;
    }
    int token_cls() const {
        return 0;
    }
    int run_len() const {
        return 0;
    }
    int word_pos() const {
        return 0;
    }
    int blank_n() const {
        return 0;
    }
    int space_run() const {
        return 0;
    }
    int line_len() const {
        return 0;
    }
    int tag_dist() const {
        return 0;
    }
    int mark_dist() const {
        return 0;
    }
    int upper_gap() const {
        return uppergap_;
    }
    int month() const {
        return 0;
    }
    int in_gallery() const {
        return 0;
    }
    int sec_kind() const {
        return 0;
    }
    int cite_kind() const {
        return 0;
    }
    std::uint64_t tag_name_cm() const {
        return 0;
    }
    int col_span() const {
        return 0;
    }
    std::uint64_t style_hash() const {
        return 0;
    }
    int in_coord() const {
        return 0;
    }
    int digit_gap() const {
        return 0;
    }
    int dot_gap() const {
        return 0;
    }
    int comma_gap() const {
        return 0;
    }
    int word_len() const {
        return wordlen_;
    }
    int sent_len() const {
        return 0;
    }
    int lower_gap() const {
        return 0;
    }
    int digit_pos() const {
        return 0;
    }
    int slash_gap() const {
        return 0;
    }
    int dig_len() const {
        return 0;
    }
    int prev_line() const {
        return 0;
    }
    int prev_sent() const {
        return 0;
    }
    int link_len() const {
        return 0;
    }
    int tpl_len() const {
        return 0;
    }
    int para_len() const {
        return 0;
    }
    int alnum_len() const {
        return 0;
    }
    int sp_len() const {
        return 0;
    }
    int title_word() const {
        return 0;
    }
    int head_word() const {
        return 0;
    }
    int in_init() const {
        return 0;
    }
    int ordinal() const {
        return 0;
    }
    int unit() const {
        return 0;
    }
    int in_decimal() const {
        return 0;
    }
    int word_repeat() const {
        return 0;
    }
    int case_flip() const {
        return 0;
    }
    int in_lead() const {
        return 0;
    }
    int info_val() const {
        return 0;
    }
    int link_trail() const {
        return 0;
    }
    int cell_kind() const {
        return 0;
    }
    int tbl_col() const {
        return 0;
    }
    int head_idx() const {
        return 0;
    }
    int html_fmt() const {
        return 0;
    }
    int in_infobox() const {
        return 0;
    }
    int sec_level() const {
        return 0;
    }
    int brace3() const {
        return 0;
    }
    int named_arg() const {
        return 0;
    }
    int include_bits() const {
        return 0;
    }
    int sig_run() const {
        return 0;
    }
    int wiki_bold() const {
        return 0;
    }
    int url_part() const {
        return 0;
    }
    int ref_idx() const {
        return 0;
    }
    int is_temp() const {
        return 0;
    }

 private:
    int state_ = kWkText;
    int depth_ = 0;
    int in_tag_ = 0;
    int in_table_ = 0;
    int sq_ = 0;
    int http_run_ = 0;
    int last_ = 0, prev1_ = 0, prev2_ = 0;
    int line_ = 0;
    int para_ = 0;
    int first_of_line_ = 1;
    int line_kind_ = 0;
    int is_paragraph_ = 0;
    int wiki_header_ = 0;
    std::uint64_t tag_name_ = 0;
    std::uint64_t linkword_ = 0;
    std::uint64_t senword_ = 0;
    std::uint64_t tpl_name_ = 0;
    std::uint64_t key_ = 0;
    int bar_idx_ = 0;
    int tpl_collect_ = 0;
    int key_collect_ = 0;
    int link_pipe_ = 0;
    std::uint64_t disp_ = 0;
    int ns_collect_ = 0;
    std::uint64_t ns_hash_ = 0;
    int cap_mask_ = 0;
    int cap_bits_ = 0;
    int cap_n_ = 0;
    int uppergap_ = 0;
    int wordlen_ = 0;
    int wl_run_ = 0;
    int heading_ = 0;
    int heading_run_ = 0;
    void dump_apply_(WikiExtra* ex) {
        auto eq = [&](const char* s) {
            std::uint64_t w = 0;
            int m = 0;
            while (s[m]) {
                w = w * 31ull + static_cast<std::uint64_t>(
                                    static_cast<unsigned char>(s[m]));
                ++m;
            }
            return m == xml_n_ && w == xml_name_;
        };
        if (eq("title")) {
            in_title_ = xml_slash_ ? 0 : 1;
            if (in_title_) title_hash_ = 0;
        }
        if (eq("page") && !xml_slash_) {
            cap_mask_ = 0;
            cap_bits_ = 0;
            cap_n_ = 0;
            uppergap_ = 0;
            wordlen_ = 0;
            wl_run_ = 0;
            if (ex) ex->page_reset();
        }
    }
    int xml_slash_ = 0;
    int xml_n_ = 0;
    int xml_done_ = 0;
    std::uint64_t xml_name_ = 0;
    int in_title_ = 0;
    std::uint64_t title_hash_ = 0;
    int st_skip_ = 0;
    int st_collect_ = 0;
    std::uint64_t st_hash_ = 0;
    void table_reset() {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 32; ++c) tbl_cells_[r][c] = 0;
        tbl_row_ = 0;
        tbl_cell_ = 0;
        tbl_started_ = 0;
    }
    void table_push(int c) {
        if (c == '|' && prev1_ != '{') {
            if (tbl_cell_ < 31) ++tbl_cell_;
            tbl_started_ = 0;
        } else if (c == '\n') {
            tbl_row_ = (tbl_row_ + 1) & 3;
            tbl_cell_ = 0;
            tbl_started_ = 0;
        } else if (!tbl_started_ && c != '-' && c != ' ' && c != '\t') {
            tbl_cells_[tbl_row_][tbl_cell_ > 31 ? 31 : tbl_cell_] =
                static_cast<std::uint8_t>(c);
            tbl_started_ = 1;
        }
    }
    std::uint8_t tbl_cells_[4][32] = {};
    int tbl_row_ = 0;
    int tbl_cell_ = 0;
    int tbl_started_ = 0;
    static int fc_token(int c) {
        if (c >= 'A' && c <= 'Z') return 1;
        if (c == ':') return 2;
        if (c == '<') return 3;
        if (c == '=') return 4;
        if (c == '*') return 5;
        if (c == '#') return 6;
        if (c == '|') return 7;
        if (c == '[') return 8;
        if (c == '{') return 9;
        if (c == '>') return 10;
        if (c == ';') return 11;
        if (c == '\'') return 12;
        return 13;
    }
    static int br_open(int c) {
        if (c == '(') return 1;
        if (c == '{') return 2;
        if (c == '[') return 3;
        if (c == '<') return 4;
        return 0;
    }
    static int br_close(int c) {
        if (c == ')') return 1;
        if (c == '}') return 2;
        if (c == ']') return 3;
        if (c == '>') return 4;
        return 0;
    }
    void stack_feed(int c) {
        if (c == '\n') {
            if (stk_col_ <= 1) {
                if (blank_run_ < 3) ++blank_run_;
            } else {
                blank_run_ = 0;
            }
            if (blank_run_ >= 2) {
                fc_stk_[0] = fc_stk_[1] = fc_stk_[2] = fc_stk_[3] = 0;
                br_stk_[0] = br_stk_[1] = br_stk_[2] = 0;
                br_n_ = 0;
            } else {
                fc_stk_[3] = fc_stk_[2];
                fc_stk_[2] = fc_stk_[1];
                fc_stk_[1] = fc_stk_[0];
                fc_stk_[0] = static_cast<std::uint8_t>(stk_fc_);
            }
            stk_col_ = 0;
            stk_fc_ = 0;
            saw_fc_ = 0;
        } else {
            if (!saw_fc_ && c != ' ' && c != '\t') {
                stk_fc_ = fc_token(c);
                fc_stk_[0] = static_cast<std::uint8_t>(stk_fc_);
                saw_fc_ = 1;
            }
            if (stk_col_ < 255) ++stk_col_;
            const int op = br_open(c);
            if (op) {
                if (br_n_ < 3) {
                    br_stk_[br_n_] = static_cast<std::uint8_t>(op);
                    ++br_n_;
                } else {
                    br_stk_[0] = br_stk_[1];
                    br_stk_[1] = br_stk_[2];
                    br_stk_[2] = static_cast<std::uint8_t>(op);
                }
            }
            const int cl = br_close(c);
            if (cl && br_n_ > 0 && br_stk_[br_n_ - 1] == cl) {
                --br_n_;
                br_stk_[br_n_] = 0;
            }
        }
    }
    std::uint8_t fc_stk_[4] = {};
    std::uint8_t br_stk_[3] = {};
    int br_n_ = 0;
    int stk_col_ = 0;
    int stk_fc_ = 0;
    int saw_fc_ = 0;
    int blank_run_ = 0;

};

}  // namespace hp
