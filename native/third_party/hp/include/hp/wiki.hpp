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

class WikiMachine {
 public:
    void push(int byte) {
        const int c = byte;
        prev2_ = prev1_;
        prev1_ = last_;
        last_ = c;
#if HP_WIKISTACK_MOD
        stack_feed(c);
#endif
#if HP_RUNLEN_MOD
        if (c == prev1_ && prev1_ != 0) {
            if (runlen_ < 15) ++runlen_;
        } else {
            runlen_ = 1;
        }
#endif
#if HP_WPOS_MOD
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            if (wpos_ < 31) ++wpos_;
        } else {
            wpos_ = 0;
        }
#endif
#if HP_BLANK_MOD
        if (c == '\n') {
            if (blank_n_ < 3) ++blank_n_;
        } else if (c != ' ' && c != '\t') {
            blank_n_ = 0;
        }
#endif
#if HP_SPRUN_MOD
        if (c == ' ') {
            if (sprun_ < 31) ++sprun_;
        } else {
            sprun_ = 0;
        }
#endif
#if HP_LINELEN_MOD
        if (c == '\n') linelen_ = 0;
        else if (linelen_ < 255) ++linelen_;
#endif
#if HP_TAGDIST_MOD
        if (c == '<') tagdist_ = 0;
        else if (tagdist_ < 255) ++tagdist_;
#endif
#if HP_MARKDIST_MOD
        if (c == '[' || c == ']' || c == '{' || c == '}' || c == '|' ||
            c == '=' || c == '*' || c == '#' || c == '<' || c == '>')
            markdist_ = 0;
        else if (markdist_ < 255) ++markdist_;
#endif
#if HP_UPPERGAP_MOD
        if (c >= 'A' && c <= 'Z') uppergap_ = 0;
        else if (uppergap_ < 255) ++uppergap_;
#endif
#if HP_DIGITGAP_MOD
        if (c >= '0' && c <= '9') digitgap_ = 0;
        else if (digitgap_ < 255) ++digitgap_;
#endif
#if HP_DOTGAP_MOD
        if (c == '.') dotgap_ = 0;
        else if (dotgap_ < 255) ++dotgap_;
#endif
#if HP_COMMAGAP_MOD
        if (c == ',') commagap_ = 0;
        else if (commagap_ < 255) ++commagap_;
#endif
#if HP_WORDLEN_MOD
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            if (wl_run_ < 31) ++wl_run_;
        } else {
            if (wl_run_) wordlen_ = wl_run_;
            wl_run_ = 0;
        }
#endif
#if HP_SENTLEN_MOD
        if (c == '.' || c == '?' || c == '!') sentlen_ = 0;
        else if (sentlen_ < 255) ++sentlen_;
#endif
#if HP_LOWERGAP_MOD
        if (c >= 'a' && c <= 'z') lowergap_ = 0;
        else if (lowergap_ < 255) ++lowergap_;
#endif
#if HP_DIGITPOS_MOD
        if (c >= '0' && c <= '9') {
            if (digitpos_ < 15) ++digitpos_;
        } else {
            digitpos_ = 0;
        }
#endif
#if HP_SLASHGAP_MOD
        if (c == '/') slashgap_ = 0;
        else if (slashgap_ < 255) ++slashgap_;
#endif
#if HP_DIGLEN_MOD
        if (c >= '0' && c <= '9') {
            if (dl_run_ < 15) ++dl_run_;
        } else {
            if (dl_run_) diglen_ = dl_run_;
            dl_run_ = 0;
        }
#endif
#if HP_PREVLINE_MOD
        if (c == '\n') {
            if (ll_run_) prevline_ = ll_run_;
            ll_run_ = 0;
        } else if (ll_run_ < 255) {
            ++ll_run_;
        }
#endif
#if HP_PREVSENT_MOD
        if (c == '.' || c == '?' || c == '!') {
            if (ss_run_) prevsent_ = ss_run_ > 255 ? 255 : ss_run_;
            ss_run_ = 0;
        } else if (ss_run_ < 255) {
            ++ss_run_;
        }
#endif
#if HP_LINKLEN_MOD
        if (state_ == kWkSquareOpen) {
            if (lk_run_ < 255) ++lk_run_;
        } else if (lk_run_) {
            linklen_ = lk_run_;
            lk_run_ = 0;
        }
#endif
#if HP_TPLLEN_MOD
        if (state_ == kWkCurly) {
            if (tp_run_ < 255) ++tp_run_;
        } else if (tp_run_) {
            tpllen_ = tp_run_;
            tp_run_ = 0;
        }
#endif
#if HP_PARALEN_MOD
        if (c == '\n' && prev1_ == '\n') {
            if (pr_run_) paralen_ = pr_run_ > 255 ? 255 : pr_run_;
            pr_run_ = 0;
        } else if (pr_run_ < 255) {
            ++pr_run_;
        }
#endif
#if HP_ALNUMLEN_MOD
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z')) {
            if (al_run_ < 31) ++al_run_;
        } else {
            if (al_run_) alnumlen_ = al_run_;
            al_run_ = 0;
        }
#endif
#if HP_SPLEN_MOD
        if (c == ' ') {
            if (sp_run_ < 31) ++sp_run_;
        } else {
            if (sp_run_) splen_ = sp_run_;
            sp_run_ = 0;
        }
#endif
#if HP_CASEFLIP_MOD
        if (prev1_ >= 'a' && prev1_ <= 'z' && c >= 'A' && c <= 'Z')
            caseflip_ = 0;
        else if (caseflip_ < 255)
            ++caseflip_;
#endif
#if HP_LEAD_MOD
        if (first_of_line_ && c == '=') lead_ = 0;
#endif
#if HP_HEADIDX_MOD
        if (first_of_line_ && c == '=') {
            if (!hi_on_ && headidx_ < 15) ++headidx_;
            hi_on_ = 1;
        } else if (c == '\n') {
            hi_on_ = 0;
        }
#endif
#if HP_LINKTRAIL_MOD
        if (lt_on_) {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z')
                linktrail_ = 1;
            else {
                linktrail_ = 0;
                lt_on_ = 0;
            }
        }
#endif
#if HP_CELLKIND_MOD
        if (in_table_) {
            if (c == '\n') {
                ck_sol_ = 1;
                ck_bar_ = 0;
            } else if (ck_sol_) {
                if (c != ' ' && c != '\t') {
                    ck_sol_ = 0;
                    if (c == '!')
                        cellkind_ = 2;
                    else if (c == '|')
                        ck_bar_ = 1;
                    else
                        ck_bar_ = 0;
                }
            } else if (ck_bar_) {
                ck_bar_ = 0;
                if (c == '+')
                    cellkind_ = 1;
                else if (c == '-')
                    cellkind_ = 4;
                else
                    cellkind_ = 3;
            }
        } else {
            cellkind_ = 0;
            ck_sol_ = 0;
            ck_bar_ = 0;
        }
#endif
#if HP_TBLCOL_MOD
        if (in_table_) {
            if (c == '\n' || (prev1_ == '|' && c == '-'))
                tblcol_ = 0;
            else if (c == '|' || c == '!') {
                if (tblcol_ < 15) ++tblcol_;
            }
        } else {
            tblcol_ = 0;
        }
#endif
#if HP_INFOVAL_MOD
        if (prev1_ == '{' && c == '{') {
            if (iv_tpl_ < 7) ++iv_tpl_;
            infoval_ = 0;
            iv_set_ = 0;
        } else if (prev1_ == '}' && c == '}') {
            if (iv_tpl_ > 0) --iv_tpl_;
            if (iv_tpl_ == 0) {
                infoval_ = 0;
                iv_set_ = 0;
            }
        } else if (iv_tpl_ > 0) {
            if (c == '|' || c == '=') {
                infoval_ = 0;
                iv_set_ = 0;
            } else if (!iv_set_ && c != ' ' && c != '\n' && c != '\t') {
                iv_set_ = 1;
                if (c >= '0' && c <= '9')
                    infoval_ = 1;
                else if (c == '[')
                    infoval_ = 2;
                else if (c == 'h' || c == 'H')
                    infoval_ = 3;
                else if ((c | 32) >= 'a' && (c | 32) <= 'z')
                    infoval_ = 4;
                else
                    infoval_ = 5;
            }
        }
#endif
#if HP_INFOBOX_MOD
        if (prev1_ == '{' && c == '{') {
            if (ib_depth_ < 7) ++ib_depth_;
            ib_n_ = 0;
            ib_col_ = 1;
        } else if (prev1_ == '}' && c == '}') {
            if (ib_depth_ > 0) --ib_depth_;
            if (ib_depth_ == 0) infobox_ = 0;
            ib_col_ = 0;
        } else if (ib_col_) {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z' && ib_n_ < 8) {
                ib_buf_[ib_n_++] = static_cast<char>(lc);
            } else {
                ib_col_ = 0;
                if (ib_n_ >= 7 && ib_buf_[0] == 'i' && ib_buf_[1] == 'n' &&
                    ib_buf_[2] == 'f' && ib_buf_[3] == 'o' &&
                    ib_buf_[4] == 'b' && ib_buf_[5] == 'o' &&
                    ib_buf_[6] == 'x')
                    infobox_ = 1;
            }
        }
#endif
#if HP_SECLEVEL_MOD
        if (c == '\n') {
            sl_run_ = 0;
            sl_at_ = 0;
        } else if (first_of_line_ && c == '=') {
            sl_run_ = 1;
            sl_at_ = 1;
        } else if (sl_at_ && c == '=') {
            if (sl_run_ < 6) ++sl_run_;
        } else if (sl_at_) {
            sl_at_ = 0;
            seclevel_ = sl_run_;
        }
#endif
#if HP_BRACE3_MOD
        if (c == '{') {
            b3_cls_ = 0;
            if (b3_run_ < 3) ++b3_run_;
            if (b3_run_ == 3) {
                if (brace3_ < 3) ++brace3_;
                b3_run_ = 0;
            }
        } else if (c == '}') {
            b3_run_ = 0;
            if (b3_cls_ < 3) ++b3_cls_;
            if (b3_cls_ == 3) {
                if (brace3_ > 0) --brace3_;
                b3_cls_ = 0;
            }
        } else {
            b3_run_ = 0;
            b3_cls_ = 0;
        }
#endif
#if HP_NAMEDARG_MOD
        if (state_ == kWkCurly || state_ == kWkVerticalBar) {
            if (c == '|')
                namedarg_ = 0;
            else if (c == '=')
                namedarg_ = 1;
        } else {
            namedarg_ = 0;
        }
#endif
#if HP_SIG_MOD
        if (c == '~') {
            if (sig_ < 4) ++sig_;
        } else {
            sig_ = 0;
        }
#endif
#if HP_WIKIBOLD_MOD
        if (c == '\n') {
            wikibold_ = 0;
            wb_run_ = 0;
        } else if (c == '\'') {
            if (wb_run_ < 5) ++wb_run_;
        } else {
            if (wb_run_ == 2)
                wikibold_ ^= 1;
            else if (wb_run_ == 3)
                wikibold_ ^= 2;
            else if (wb_run_ == 4 || wb_run_ == 5)
                wikibold_ ^= 3;
            wb_run_ = 0;
        }
#endif
#if HP_URLPART_MOD
        if (state_ == kWkHtLink) {
            if (c == ' ' || c == '\n' || c == ']' || c == '"' || c == '<')
                urlpart_ = 0;
            else if (c == '#')
                urlpart_ = 4;
            else if (c == '?')
                urlpart_ = 3;
            else if (c == '/' && urlpart_ <= 1)
                urlpart_ = 2;
            else if (urlpart_ == 0)
                urlpart_ = 1;
        } else {
            urlpart_ = 0;
        }
#endif
#if HP_INIT_MOD
        {
            const int lc = c | 32;
            if (init_st_ == 0) {
                init_ = 0;
                if (lc >= 'a' && lc <= 'z') init_st_ = 1;
            } else if (init_st_ == 1) {
                if (c == '.') init_st_ = 2;
                else if (lc < 'a' || lc > 'z') {
                    init_ = 0;
                    init_st_ = 0;
                }
            } else {
                if (lc >= 'a' && lc <= 'z') {
                    init_ = 1;
                    init_st_ = 1;
                } else if (c == ' ') {
                    init_ = 1;
                } else {
                    init_ = 0;
                    init_st_ = 0;
                }
            }
        }
#endif
#if HP_DECIMAL_MOD
        if (c >= '0' && c <= '9') {
            if (dec_pend_) decimal_ = 1;
            dec_saw_ = 1;
            dec_pend_ = 0;
        } else if (c == '.' && dec_saw_) {
            dec_pend_ = 1;
            dec_saw_ = 0;
        } else {
            decimal_ = 0;
            dec_saw_ = 0;
            dec_pend_ = 0;
        }
#endif
#if HP_ORDINAL_MOD
        {
            const int lc = c | 32;
            if (c >= '0' && c <= '9') {
                ord_pend_ = 1;
                ord_n_ = 0;
            } else if (ord_pend_ && lc >= 'a' && lc <= 'z' && ord_n_ < 2) {
                ord_buf_[ord_n_++] = static_cast<char>(lc);
                if (ord_n_ == 2) {
                    int hit = 0;
                    if (ord_buf_[0] == 's' && ord_buf_[1] == 't') hit = 1;
                    else if (ord_buf_[0] == 'n' && ord_buf_[1] == 'd') hit = 2;
                    else if (ord_buf_[0] == 'r' && ord_buf_[1] == 'd') hit = 3;
                    else if (ord_buf_[0] == 't' && ord_buf_[1] == 'h') hit = 4;
                    if (hit) ordinal_ = hit;
                    ord_pend_ = 0;
                }
            } else if (ord_pend_ && (c == ' ' || c == '\t')) {
            } else {
                ord_pend_ = 0;
                ord_n_ = 0;
            }
        }
#endif
#if HP_UNIT_MOD
        {
            const int lc = c | 32;
            if (c >= '0' && c <= '9') {
                un_pend_ = 1;
                un_n_ = 0;
            } else if (un_pend_ && (c == ' ' || c == '\t')) {
            } else if (un_pend_ && lc >= 'a' && lc <= 'z' && un_n_ < 4) {
                un_buf_[un_n_++] = static_cast<char>(lc);
            } else {
                if (un_pend_ && un_n_ > 0) {
                    auto ueq = [&](const char* w, int n) {
                        if (un_n_ != n) return 0;
                        for (int i = 0; i < n; ++i)
                            if (un_buf_[i] != w[i]) return 0;
                        return 1;
                    };
                    int u = 0;
                    if (ueq("km", 2) || ueq("m", 1) || ueq("cm", 2) ||
                        ueq("mm", 2))
                        u = 1;
                    else if (ueq("mi", 2) || ueq("ft", 2) || ueq("in", 2) ||
                             ueq("yd", 2))
                        u = 2;
                    else if (ueq("kg", 2) || ueq("g", 1) || ueq("lb", 2) ||
                             ueq("oz", 2))
                        u = 3;
                    else if (ueq("hz", 2) || ueq("mph", 3) || ueq("kph", 3))
                        u = 4;
                    if (u) unit_ = u;
                }
                un_pend_ = 0;
                un_n_ = 0;
            }
        }
#endif
#if HP_TITLEWORD_MOD || HP_HEADWORD_MOD || HP_REPEAT_MOD
        {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z') {
                cw_acc_ = mix64(cw_acc_ * 31ull + static_cast<std::uint64_t>(lc));
                cw_on_ = 1;
            } else if (cw_on_) {
#if HP_TITLEWORD_MOD
#if HP_TITLE_MOD
                if (in_title_) {
#else
                if (0) {
#endif
                    if (tw_n_ < 8) tw_ring_[tw_n_++] = cw_acc_;
                    else {
                        for (int i = 0; i < 7; ++i) tw_ring_[i] = tw_ring_[i + 1];
                        tw_ring_[7] = cw_acc_;
                    }
                } else {
                    int hit = 0;
                    for (int i = 0; i < tw_n_; ++i)
                        if (tw_ring_[i] == cw_acc_) hit = 1;
                    titleword_ = hit;
                }
#endif
#if HP_HEADWORD_MOD
                if (first_of_line_ && c == '=') {
                    hw_n_ = 0;
                    headword_ = 0;
                }
                if (heading_level() > 0) {
                    if (hw_n_ < 8) hw_ring_[hw_n_++] = cw_acc_;
                    else {
                        for (int i = 0; i < 7; ++i) hw_ring_[i] = hw_ring_[i + 1];
                        hw_ring_[7] = cw_acc_;
                    }
                } else {
                    int hit = 0;
                    for (int i = 0; i < hw_n_; ++i)
                        if (hw_ring_[i] == cw_acc_) hit = 1;
                    headword_ = hit;
                }
#endif
#if HP_REPEAT_MOD
                repeat_ = (cw_acc_ != 0 && cw_acc_ == cw_prev_) ? 1 : 0;
                cw_prev_ = cw_acc_;
#endif
                cw_acc_ = 0;
                cw_on_ = 0;
            }
#if HP_HEADWORD_MOD
            else if (first_of_line_ && c == '=') {
                hw_n_ = 0;
            }
#endif
        }
#endif
#if HP_MONTH_MOD
        {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z') {
                if (mo_n_ < 9) mo_buf_[mo_n_++] = static_cast<char>(lc);
                else {
                    for (int i = 0; i < 8; ++i) mo_buf_[i] = mo_buf_[i + 1];
                    mo_buf_[8] = static_cast<char>(lc);
                    mo_n_ = 9;
                }
            } else {
                if (mo_n_ >= 3) {
                    auto meq = [&](const char* w, int n) {
                        if (mo_n_ != n) return 0;
                        for (int i = 0; i < n; ++i)
                            if (mo_buf_[i] != w[i]) return 0;
                        return 1;
                    };
                    int m = 0;
                    if (meq("january", 7)) m = 1;
                    else if (meq("february", 8)) m = 2;
                    else if (meq("march", 5)) m = 3;
                    else if (meq("april", 5)) m = 4;
                    else if (meq("may", 3)) m = 5;
                    else if (meq("june", 4)) m = 6;
                    else if (meq("july", 4)) m = 7;
                    else if (meq("august", 6)) m = 8;
                    else if (meq("september", 9)) m = 9;
                    else if (meq("october", 7)) m = 10;
                    else if (meq("november", 8)) m = 11;
                    else if (meq("december", 8)) m = 12;
                    if (m) month_ = m;
                }
                mo_n_ = 0;
            }
        }
#endif
#if HP_COLSPAN_MOD
        {
            int lc = c;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) lc = c | 32;
            cs_win_ = (cs_win_ << 8) | static_cast<std::uint64_t>(lc & 255);
            if (cs_eat_) {
                if (c >= '0' && c <= '9') {
                    int v = cs_val_ * 10 + (c - '0');
                    cs_val_ = v > 32 ? 32 : v;
                } else {
                    cs_eat_ = 0;
                    if (cs_val_) colspan_ = cs_val_;
                }
            } else if (cs_win_ == 0x636f6c7370616e3dull ||
                       cs_win_ == 0x726f777370616e3dull) {
                cs_eat_ = 1;
                cs_val_ = 0;
            }
        }
#endif
#if HP_STYLE_MOD
        {
            int lc = c;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) lc = c | 32;
            st_win_ = (st_win_ << 8) | static_cast<std::uint64_t>(lc & 255);
            if (style_eat_) {
                if (c == '"' || c == '\'' || c == '|' || c == '\n' || c == '>')
                    style_eat_ = 0;
                else
                    style_hash_ = mix64(style_hash_ * 31ull +
                                        static_cast<std::uint64_t>(lc & 255));
            } else if ((st_win_ & 0xffffffffffffull) == 0x7374796c653dull) {
                style_eat_ = 1;
                style_hash_ = 0;
            }
        }
#endif

#if HP_EXTLINK_MOD
        if (in_ext_ && (c == ']' || c == '\n')) in_ext_ = 0;
        if (ext_pend_) {
            ext_pend_ = 0;
            if (c != '[') in_ext_ = 1;
        }
#endif
#if HP_ENTITY_MOD
        if (c == '&') {
            ent_collect_ = 1;
            entity_ = 0;
        } else if (ent_collect_) {
            if (c == ';' || c == ' ' || c == '\n' || c == '<')
                ent_collect_ = 0;
            else
                entity_ = mix64(entity_ * 31ull + static_cast<std::uint64_t>(c));
        }
#endif
#if HP_MAGIC_MOD
        if (c == '_' && prev1_ == '_') {
            if (magic_collect_) magic_collect_ = 0;
            else {
                magic_collect_ = 1;
                magic_hash_ = 0;
            }
        } else if (magic_collect_) {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z')
                magic_hash_ = mix64(magic_hash_ * 31ull +
                                    static_cast<std::uint64_t>(lc));
            else if (c != '_')
                magic_collect_ = 0;
        }
#endif
#if HP_TITLE_MOD
        if (in_title_) {
            if (c == '<') in_title_ = 0;
            else
                title_hash_ = mix64(title_hash_ * 31ull +
                                    static_cast<std::uint64_t>(c));
        }
#endif
#if HP_FWORD_MOD
        {
            const int lc = c | 32;
            if (c == '.' || c == '!' || c == '?' || c == '\n') {
                fw_pending_ = 1;
                fw_acc_ = 0;
                fw_n_ = 0;
            } else if (fw_pending_ && is_paragraph_) {
                if (lc >= 'a' && lc <= 'z') {
                    if (fw_n_ < 16) {
                        fw_acc_ = mix64(fw_acc_ * 31ull +
                                        static_cast<std::uint64_t>(lc));
                        ++fw_n_;
                    }
                } else if (fw_n_ > 0) {
                    fword_ = fw_acc_;
                    fw_pending_ = 0;
                }
            }
        }
#endif
#if HP_YEAR_MOD
        if (is_paragraph_ || in_table_ || state_ == kWkSquareOpen) {
            if (c >= '0' && c <= '9') {
                if (yr_n_ < 8) {
                    yr_val_ = yr_val_ * 10 + (c - '0');
                    ++yr_n_;
                }
            } else {
                if (yr_n_ == 4 && yr_val_ >= 1000 && yr_val_ <= 2099)
                    year_ = static_cast<std::uint64_t>(yr_val_);
                yr_n_ = 0;
                yr_val_ = 0;
            }
        }
#endif
#if HP_CAPMASK_MOD
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
#endif
#if HP_CHARCLS_MOD
        {
            int k = 0;
            if (c >= 'A' && c <= 'Z') k = 1;
            else if (c >= 'a' && c <= 'z') k = 2;
            else if (c >= '0' && c <= '9') k = 3;
            else if (c == ' ' || c == '\t') k = 4;
            else if (c == '\n') k = 5;
            else if (c == '<' || c == '>' || c == '{' || c == '}' ||
                     c == '[' || c == ']' || c == '|')
                k = 6;
            else if (c == '\'' || c == '"') k = 7;
            charcls_ = ((charcls_ << 3) | k) & 0xfff;
        }
#endif
#if HP_SHAPE_MOD || HP_SUFFIX_MOD || HP_PREFIX_MOD || HP_VOWEL_MOD || \
    HP_CONTR_MOD || HP_HYPHEN_MOD || HP_TOKENCLS_MOD
        if (is_paragraph_ || in_table_) {
            const int up = (c >= 'A' && c <= 'Z');
            const int low = (c >= 'a' && c <= 'z');
            const int dig = (c >= '0' && c <= '9');
            const int let = up || low;
            const int lc = c | 32;
            const int vow = let && (lc == 'a' || lc == 'e' || lc == 'i' ||
                                    lc == 'o' || lc == 'u' || lc == 'y');
            if (let || dig) {
#if HP_SHAPE_MOD
                {
                    const int t = up ? 1 : (low ? 2 : 3);
                    if (!shape_on_) {
                        shape_ = static_cast<std::uint64_t>(t);
                        shape_on_ = 1;
                    } else {
                        shape_ = ((shape_ << 2) |
                                  static_cast<std::uint64_t>(t)) &
                                 0xffffull;
                    }
                }
#endif
#if HP_PREFIX_MOD
                if (let && pre_n_ < 3) {
                    prefix_ = (prefix_ << 5) |
                              static_cast<std::uint64_t>(lc - 'a' + 1);
                    ++pre_n_;
                }
#endif
#if HP_SUFFIX_MOD
                if (let) {
                    suffix_acc_ = ((suffix_acc_ << 5) |
                                   static_cast<std::uint64_t>(lc - 'a' + 1)) &
                                  0x7fffull;
                    if (suf_n_ < 3) ++suf_n_;
                }
#endif
#if HP_VOWEL_MOD
                if (let)
                    vowel_ = ((vowel_ << 1) | (vow ? 1 : 0)) & 255;
#endif
#if HP_CONTR_MOD
                if (let) {
                    contr_acc_ = mix64(contr_acc_ * 31ull +
                                       static_cast<std::uint64_t>(lc));
                }
#endif
#if HP_HYPHEN_MOD
                if (let)
                    hy_acc_ = mix64(hy_acc_ * 31ull +
                                    static_cast<std::uint64_t>(lc));
#endif
#if HP_TOKENCLS_MOD
                if (tok_cls_ <= 0) tok_cls_ = let ? 1 : 2;
                else if ((tok_cls_ == 1 && dig) || (tok_cls_ == 2 && let))
                    tok_cls_ = 3;
#endif
            } else {
#if HP_SHAPE_MOD
                shape_on_ = 0;
#endif
#if HP_PREFIX_MOD
                pre_n_ = 0;
#endif
#if HP_SUFFIX_MOD
                if (suf_n_) {
                    suffix_ = suffix_acc_;
                    suffix_acc_ = 0;
                    suf_n_ = 0;
                }
#endif
#if HP_CONTR_MOD
                if (c == '\'' && contr_acc_) {
                    contr_apos_ = 1;
                } else if (c != '\'') {
                    if (contr_apos_ && contr_acc_) contr_ = contr_acc_;
                    contr_acc_ = 0;
                    contr_apos_ = 0;
                }
#endif
#if HP_HYPHEN_MOD
                if (c == '-' && hy_acc_) {
                    hy_seen_ = 1;
                } else if (c != '-') {
                    if (hy_seen_ && hy_acc_) hyphen_ = hy_acc_;
                    hy_acc_ = 0;
                    hy_seen_ = 0;
                }
#endif
#if HP_TOKENCLS_MOD
                if (in_tag_) tok_cls_ = 5;
                else if (state_ == kWkSquareOpen || state_ == kWkCurly ||
                         in_table_)
                    tok_cls_ = 6;
                else if (c != ' ' && c != '\n' && c != '\t')
                    tok_cls_ = 4;
                else
                    tok_cls_ = 0;
#endif
            }
        }
#endif
#if HP_PAREN_MOD
        if (is_paragraph_) {
            if (c == '(') {
                ++paren_d_;
                if (paren_d_ == 1) paren_acc_ = 0;
            } else if (c == ')' && paren_d_ > 0) {
                if (paren_d_ == 1 && paren_acc_) lastparen_ = paren_acc_;
                --paren_d_;
            } else if (paren_d_ > 0) {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z')
                    paren_acc_ = mix64(paren_acc_ * 31ull +
                                       static_cast<std::uint64_t>(lc));
            }
        }
#endif
#if HP_USER_MOD
        if (in_user_) {
            if (c == '<') in_user_ = 0;
            else
                user_hash_ = mix64(user_hash_ * 31ull +
                                   static_cast<std::uint64_t>(c));
        }
#endif
#if HP_PAGEID_MOD
        if (in_id_) {
            if (c >= '0' && c <= '9') {
                if (page_id_ < 100000000ull)
                    page_id_ = page_id_ * 10ull + static_cast<std::uint64_t>(c - '0');
            } else {
                in_id_ = 0;
                pageid_seen_ = 1;
            }
        }
#endif
#if HP_NS_MOD
        if (in_ns_) {
            if (c >= '0' && c <= '9') {
                if (ns_id_ < 1000)
                    ns_id_ = ns_id_ * 10 + (c - '0');
            } else {
                in_ns_ = 0;
            }
        }
#endif
#if HP_IP_MOD
        if (in_ip_) {
            if (c == '<') in_ip_ = 0;
            else
                ip_hash_ = mix64(ip_hash_ * 31ull +
                                 static_cast<std::uint64_t>(c));
        }
#endif
#if HP_REVCOMMENT_MOD
        if (in_comment_) {
            if (c == '<') in_comment_ = 0;
            else
                comment_hash_ = mix64(comment_hash_ * 31ull +
                                      static_cast<std::uint64_t>(c));
        }
#endif
#if HP_WIKIMODEL_MOD
        if (in_model_) {
            if (c == '<') in_model_ = 0;
            else
                model_hash_ = mix64(model_hash_ * 31ull +
                                    static_cast<std::uint64_t>(c));
        }
#endif
#if HP_PARSERFN_MOD
        if (pfn_wait_) {
            pfn_wait_ = 0;
            if (c == '#') pfn_collect_ = 1;
        } else if (pfn_collect_) {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z')
                pfn_hash_ = mix64(pfn_hash_ * 31ull +
                                  static_cast<std::uint64_t>(lc));
            else
                pfn_collect_ = 0;
        }
#endif
#if HP_TABLECLASS_MOD
        if (tc_collect_) {
            if (c == '\n') tc_collect_ = 0;
            else {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z')
                    tc_hash_ = mix64(tc_hash_ * 31ull +
                                     static_cast<std::uint64_t>(lc));
            }
        }
#endif
#if HP_PUBID_MOD
        {
            const unsigned lc = static_cast<unsigned>((c | 32) & 255);
            pub_win_ = (pub_win_ << 8) | lc;
            if ((pub_win_ & 0xffffffffu) == 0x6973626eu /*isbn*/ ||
                (pub_win_ & 0xffffffffu) == 0x706d6964u /*pmid*/) {
                pub_collect_ = 1;
                pub_hash_ = 0;
            } else if (pub_collect_) {
                if (c >= '0' && c <= '9')
                    pub_hash_ = mix64(pub_hash_ * 31ull +
                                      static_cast<std::uint64_t>(c - '0'));
                else if (c != '-' && c != ' ')
                    pub_collect_ = 0;
            }
        }
#endif
#if HP_TEMPPOS_MOD
        if (tp_collect_) {
            if (c == '=' || c == '|' || c == '}') {
                if (c == '=') tp_hash_ = 0;
                tp_collect_ = 0;
                tp_done_ = 1;
            } else {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z')
                    tp_hash_ = mix64(tp_hash_ * 31ull +
                                     static_cast<std::uint64_t>(lc));
            }
        }
#endif

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
#if HP_CITE_MOD
            cite_slash_ = 0;
            cite_buf_ = 0;
#endif
#if HP_GALLERY_MOD
            gal_slash_ = 0;
            gal_n_ = 0;
#endif
#if HP_REFNAME_MOD
            refn_collect_ = 0;
            refn_win_ = 0;
            refn_skipq_ = 0;
#endif
#if HP_NOWIKI_MOD
            nw_slash_ = 0;
            nw_n_ = 0;
            nw_name_ = 0;
#endif
#if HP_DUMP_XML
            xml_slash_ = 0;
            xml_n_ = 0;
            xml_name_ = 0;
            xml_done_ = 0;
#endif
#if HP_HTMLFMT_MOD
            hf_slash_ = 0;
            hf_n_ = 0;
#endif
#if HP_INCLUDE_MOD
            ic_slash_ = 0;
            ic_n_ = 0;
#endif
#if HP_REFIDX_MOD
            ri_slash_ = 0;
            ri_n_ = 0;
#endif
            return;
        }
        if (c == '>' && in_tag_) {
#if HP_CITE_MOD
            if ((cite_buf_ & 0xffffffu) == 0x726566u)
                in_ref_ = cite_slash_ ? 0 : 1;
#endif
#if HP_GALLERY_MOD
            if (gal_n_ == 7) {
                static const char kGal[] = "gallery";
                int ok = 1;
                for (int i = 0; i < 7; ++i)
                    if (gal_buf_[i] != kGal[i]) ok = 0;
                if (ok) in_gallery_ = gal_slash_ ? 0 : 1;
            }
#endif
#if HP_REFNAME_MOD
            refn_collect_ = 0;
#endif
#if HP_NOWIKI_MOD
            nowiki_apply_();
#endif
#if HP_DUMP_XML
            dump_apply_();
#endif
#if HP_HTMLFMT_MOD
            if (hf_n_ > 0) {
                auto heq = [&](const char* w, int n) {
                    if (hf_n_ != n) return 0;
                    for (int i = 0; i < n; ++i)
                        if (hf_buf_[i] != w[i]) return 0;
                    return 1;
                };
                int bit = 0;
                if (heq("small", 5)) bit = 1;
                else if (heq("sup", 3)) bit = 2;
                else if (heq("sub", 3)) bit = 4;
                else if (heq("b", 1) || heq("strong", 6)) bit = 8;
                else if (heq("i", 1) || heq("em", 2)) bit = 16;
                else if (heq("big", 3)) bit = 32;
                else if (heq("s", 1) || heq("u", 1) || heq("strike", 6))
                    bit = 64;
                if (bit) {
                    if (hf_slash_) htmlfmt_ &= ~bit;
                    else htmlfmt_ |= bit;
                }
            }
#endif
#if HP_INCLUDE_MOD
            if (ic_n_ > 0) {
                auto ieq = [&](const char* w, int n) {
                    if (ic_n_ != n) return 0;
                    for (int i = 0; i < n; ++i)
                        if (ic_buf_[i] != w[i]) return 0;
                    return 1;
                };
                int bit = 0;
                if (ieq("includeonly", 11)) bit = 1;
                else if (ieq("noinclude", 9)) bit = 2;
                else if (ieq("onlyinclude", 11)) bit = 4;
                if (bit) {
                    if (ic_slash_) include_ &= ~bit;
                    else include_ |= bit;
                }
            }
#endif
#if HP_REFIDX_MOD
            if (!ri_slash_ && ri_n_ == 3 && ri_buf_[0] == 'r' &&
                ri_buf_[1] == 'e' && ri_buf_[2] == 'f') {
                if (refidx_ < 15) ++refidx_;
            }
#endif
            in_tag_ = 0;
            state_ = kWkText;
            return;
        }
        if (c == '/' && in_tag_) {
#if HP_CITE_MOD
            if (cite_buf_ == 0) cite_slash_ = 1;
#endif
#if HP_GALLERY_MOD
            if (gal_n_ == 0) gal_slash_ = 1;
#endif
#if HP_NOWIKI_MOD
            if (nw_n_ == 0) nw_slash_ = 1;
#endif
#if HP_DUMP_XML
            if (xml_n_ == 0) xml_slash_ = 1;
#endif
#if HP_HTMLFMT_MOD
            if (hf_n_ == 0) hf_slash_ = 1;
#endif
#if HP_INCLUDE_MOD
            if (ic_n_ == 0) ic_slash_ = 1;
#endif
#if HP_REFIDX_MOD
            if (ri_n_ == 0) ri_slash_ = 1;
#endif
            if (depth_ > 0) --depth_;
            state_ = kWkTagEnd;
            return;
        }
        if (in_tag_) {
            tag_name_ = mix64(tag_name_ * 31 + static_cast<std::uint64_t>(c));
#if HP_CITE_MOD
            {
                const unsigned ch = static_cast<unsigned>((c | 32) & 255);
                if (ch >= 'a' && ch <= 'z' && (cite_buf_ & 0xff0000u) == 0)
                    cite_buf_ = (cite_buf_ << 8) | ch;
            }
#endif
#if HP_GALLERY_MOD
            {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && gal_n_ < 8)
                    gal_buf_[gal_n_++] = static_cast<char>(lc);
            }
#endif
#if HP_REFNAME_MOD
            {
                const unsigned ch = static_cast<unsigned>(c & 255);
                const unsigned lc = ch | 32u;
                const unsigned packed = (ch == '=') ? '=' : lc;
                refn_win_ = (refn_win_ << 8) | packed;
                if ((refn_win_ & 0xffffffffffull) == 0x6e616d653dull) {
                    refn_collect_ = 1;
                    refname_ = 0;
                    refn_skipq_ = 1;
                } else if (refn_collect_) {
                    if (ch == '"' || ch == '\'') {
                        if (refn_skipq_) refn_skipq_ = 0;
                        else refn_collect_ = 0;
                    } else if (ch == ' ' && refn_skipq_) {
                    } else if (ch == ' ' || ch == '>' || ch == '/') {
                        refn_collect_ = 0;
                    } else {
                        refn_skipq_ = 0;
                        refname_ = mix64(refname_ * 31ull + ch);
                    }
                }
            }
#endif
#if HP_NOWIKI_MOD
            {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && nw_n_ < 8) {
                    nw_name_ = nw_name_ * 31ull + static_cast<std::uint64_t>(lc);
                    ++nw_n_;
                }
            }
#endif
#if HP_DUMP_XML
            if (!xml_done_) {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && xml_n_ < 12) {
                    xml_name_ = xml_name_ * 31ull + static_cast<std::uint64_t>(lc);
                    ++xml_n_;
                } else {
                    xml_done_ = 1;
                }
            }
#endif
#if HP_HTMLFMT_MOD
            {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && hf_n_ < 8)
                    hf_buf_[hf_n_++] = static_cast<char>(lc);
            }
#endif
#if HP_INCLUDE_MOD
            {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && ic_n_ < 12)
                    ic_buf_[ic_n_++] = static_cast<char>(lc);
            }
#endif
#if HP_REFIDX_MOD
            {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z' && ri_n_ < 4)
                    ri_buf_[ri_n_++] = static_cast<char>(lc);
            }
#endif
            if (prev1_ == '!' && c == '-') state_ = kWkComment;
            return;
        }

        // Wiki link [[
        if (prev1_ == '[' && c == '[') {
            state_ = kWkSquareOpen;
            sq_ = 1;
            linkword_ = 0;
#if HP_CAT_MOD
            ns_collect_ = 1;
            ns_hash_ = 0;
#endif
#if HP_ANCHOR_MOD
            an_collect_ = 0;
            an_hash_ = 0;
#endif
#if HP_LANG_MOD
            lang_n_ = 0;
            lang_acc_ = 0;
            lang_hash_ = 0;
#endif
#if HP_CATSORT_MOD || HP_FILEOPT_MOD
            pre_i_ = 0;
            pre_kind_ = 0;
            pipe_col_ = 0;
            extra_hash_ = 0;
#endif
#if HP_LASTLINK_MOD
            ll_acc_ = 0;
            ll_stop_ = 0;
#endif
            return;
        }
        if (state_ == kWkSquareOpen) {
            if (c == ']') {
                if (sq_ > 0) --sq_;
                if (sq_ == 0) {
                    state_ = kWkText;
                    linkword_ = 0;
#if HP_LINKPIPE_MOD
                    link_pipe_ = 0;
                    disp_ = 0;
#endif
#if HP_CAT_MOD
                    ns_collect_ = 0;
                    ns_hash_ = 0;
#endif
#if HP_ANCHOR_MOD
                    an_collect_ = 0;
#endif
#if HP_LANG_MOD
                    lang_n_ = 0;
#endif
#if HP_CATSORT_MOD || HP_FILEOPT_MOD
                    pipe_col_ = 0;
#endif
#if HP_LASTLINK_MOD
                    if (ll_acc_) lastlink_ = ll_acc_;
#endif
#if HP_LINKTRAIL_MOD
                    lt_on_ = 1;
#endif
                }
            } else if (c == ':') {
                linkword_ = 0;   // fx2: [category:...] drops the hash
#if HP_LASTLINK_MOD
                ll_acc_ = 0;
#endif
#if HP_CAT_MOD
                if (ns_hash_ != 0) ns_collect_ = 0;
#endif
#if HP_LANG_MOD
                if (lang_n_ >= 2 && lang_n_ <= 3) lang_hash_ = lang_acc_;
                lang_n_ = 8;
#endif
#if HP_CATSORT_MOD || HP_FILEOPT_MOD
                pre_kind_ = 0;
                if (pre_i_ == 8) {
                    static const char kCat[] = "category";
                    int ok = 1;
                    for (int i = 0; i < 8; ++i)
                        if (pre_buf_[i] != kCat[i]) ok = 0;
                    if (ok) pre_kind_ = 1;
                } else if (pre_i_ == 4) {
                    static const char kFile[] = "file";
                    int ok = 1;
                    for (int i = 0; i < 4; ++i)
                        if (pre_buf_[i] != kFile[i]) ok = 0;
                    if (ok) pre_kind_ = 2;
                } else if (pre_i_ == 5) {
                    static const char kImg[] = "image";
                    int ok = 1;
                    for (int i = 0; i < 5; ++i)
                        if (pre_buf_[i] != kImg[i]) ok = 0;
                    if (ok) pre_kind_ = 2;
                }
#endif
            } else {
                // fx2-cmix: linkword = linkword * 2104 + j
                linkword_ = linkword_ * 2104ull + static_cast<std::uint64_t>(c);
#if HP_LINKPIPE_MOD
                if (c == '|') {
                    link_pipe_ = 1;
                    disp_ = 0;
                } else if (link_pipe_) {
                    disp_ = disp_ * 2104ull + static_cast<std::uint64_t>(c);
                }
#endif
#if HP_CAT_MOD
                if (ns_collect_) {
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z')
                        ns_hash_ = mix64(ns_hash_ * 31ull +
                                         static_cast<std::uint64_t>(lc));
                    else if (c != ' ' && c != '_')
                        ns_collect_ = 0;
                }
#endif
#if HP_ANCHOR_MOD
                if (c == '#') {
                    an_collect_ = 1;
                    an_hash_ = 0;
                } else if (an_collect_) {
                    if (c == '|') {
                        an_collect_ = 0;
                    } else {
                        const int lc = c | 32;
                        if (lc >= 'a' && lc <= 'z')
                            an_hash_ = mix64(an_hash_ * 31ull +
                                             static_cast<std::uint64_t>(lc));
                    }
                }
#endif
#if HP_LANG_MOD
                {
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z' && lang_n_ < 4) {
                        lang_acc_ = mix64(lang_acc_ * 31ull +
                                          static_cast<std::uint64_t>(lc));
                        ++lang_n_;
                    } else if (lang_n_ < 4 && c != '_' && c != ' ') {
                        lang_n_ = 0;
                    }
                }
#endif
#if HP_CATSORT_MOD || HP_FILEOPT_MOD
                {
                    const int lc = c | 32;
                    if (pre_kind_ == 0 && pre_i_ < 12 &&
                        lc >= 'a' && lc <= 'z') {
                        pre_buf_[pre_i_++] = static_cast<char>(lc);
                    }
                    if (c == '|') {
                        pipe_col_ = 1;
                        extra_hash_ = 0;
                    } else if (pipe_col_) {
#if HP_CATSORT_MOD
                        if (pre_kind_ == 1) {
                            extra_hash_ = mix64(extra_hash_ * 31ull +
                                                static_cast<std::uint64_t>(lc));
                        }
#endif
#if HP_FILEOPT_MOD
                        if (pre_kind_ == 2) {
                            extra_hash_ = mix64(extra_hash_ * 31ull +
                                                static_cast<std::uint64_t>(lc));
                        }
#endif
                    }
                }
#endif
#if HP_LASTLINK_MOD
                if (!ll_stop_) {
                    if (c == '|' || c == '#') {
                        ll_stop_ = 1;
                    } else {
                        const int lc = c | 32;
                        if (lc >= 'a' && lc <= 'z')
                            ll_acc_ = mix64(ll_acc_ * 31ull +
                                            static_cast<std::uint64_t>(lc));
                    }
                }
#endif
            }
            return;
        }

        // Template / infobox {{
        if (prev1_ == '{' && c == '{') {
            state_ = kWkCurly;
            if (depth_ < 15) ++depth_;
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
            tpl_collect_ = 1;
            tpl_name_ = 0;
            bar_idx_ = 0;
            key_ = 0;
            key_collect_ = 0;
#endif
#if HP_PARSERFN_MOD
            pfn_wait_ = 1;
            pfn_collect_ = 0;
            pfn_hash_ = 0;
#endif
#if HP_TEMPPOS_MOD
            tp_collect_ = 0;
            tp_done_ = 0;
            tp_hash_ = 0;
#endif
#if HP_DEFAULTSORT_MOD || HP_DAB_MOD || HP_HATNOTE_MOD
            tnm_i_ = 0;
            tnm_done_ = 0;
            tnm_mode_ = 0;
#endif
#if HP_CITEKIND_MOD || HP_COORD_MOD
            ck_i_ = 0;
            ck_done_ = 0;
#endif
            return;
        }
        if (prev1_ == '{' && c == '|') {
            state_ = kWkWikiTable;
            in_table_ = 1;
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
            table_reset();
#endif
#if HP_TABLECLASS_MOD
            tc_collect_ = 1;
            tc_hash_ = 0;
#endif
#if HP_TBLROW_MOD
            tbl_kind_ = 0;
            tbl_rowpend_ = 1;
            tbl_bar_ = 0;
#endif
            return;
        }
#if HP_WIKI_TEMP
        if (prev1_ == '{' && c != '{' && c != '|') is_temp_ = 1;
        if (c == '}') is_temp_ = 0;
#endif
        if (prev1_ == '|' && c == '}') {
            state_ = kWkText;
            in_table_ = 0;
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
            table_reset();
#endif
#if HP_TABLECLASS_MOD
            tc_collect_ = 0;
            tc_hash_ = 0;
#endif
#if HP_TBLROW_MOD
            tbl_kind_ = 0;
            tbl_rowpend_ = 0;
            tbl_bar_ = 0;
#endif
            return;
        }
        if (prev1_ == '}' && c == '}') {
            if (depth_ > 0) --depth_;
            state_ = kWkText;
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
            tpl_collect_ = 0;
            key_collect_ = 0;
#endif
#if HP_DEFAULTSORT_MOD || HP_DAB_MOD || HP_HATNOTE_MOD
            tnm_done_ = 1;
#endif
#if HP_CITEKIND_MOD || HP_COORD_MOD
            ck_done_ = 1;
#endif
            return;
        }

        if (c == '|' && (in_table_ || state_ == kWkCurly || state_ == kWkWikiTable)) {
            state_ = kWkVerticalBar;
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
            tpl_collect_ = 0;
            if (bar_idx_ < 31) ++bar_idx_;
            key_ = 0;
            key_collect_ = 1;
#endif
#if HP_TEMPPOS_MOD
            if (state_ == kWkCurly && !tp_done_) {
                if (tp_collect_) {
                    tp_collect_ = 0;
                    tp_done_ = 1;
                } else {
                    tp_collect_ = 1;
                }
            }
#endif
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
#if HP_HTTPHOST_MOD
                ht_acc_ = 0;
                ht_done_ = 0;
#endif
            }
            if (c == ' ' || c == '\n') http_run_ = 0;
        }
        if (state_ == kWkHtLink && c != ' ' && c != '\n') {
            linkword_ = linkword_ * 2104ull + static_cast<std::uint64_t>(c);
        }
#if HP_HTTPHOST_MOD
        if (state_ == kWkHtLink && !ht_done_) {
            if (c == '/' || c == ':' || c == ']' || c == ' ' || c == '\n' ||
                c == '"') {
                if (ht_acc_) {
                    httphost_ = ht_acc_;
                    ht_done_ = 1;
                }
            } else {
                const int lc = c | 32;
                ht_acc_ = mix64(ht_acc_ * 31ull +
                                static_cast<std::uint64_t>(lc));
            }
        }
#endif

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

#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
        if (in_table_) table_push(c);
#endif
#if HP_TBLROW_MOD
        if (in_table_) {
            if (c == '\n') {
                tbl_rowpend_ = 1;
                tbl_bar_ = 0;
            } else if (tbl_rowpend_) {
                if (c == '|') {
                    tbl_bar_ = 1;
                } else if (c == '!') {
                    tbl_kind_ = 3;
                    tbl_rowpend_ = 0;
                    tbl_bar_ = 0;
                } else if (tbl_bar_ && c == '-') {
                    tbl_kind_ = 1;
                    tbl_rowpend_ = 0;
                    tbl_bar_ = 0;
                } else if (tbl_bar_ && c == '+') {
                    tbl_kind_ = 2;
                    tbl_rowpend_ = 0;
                    tbl_bar_ = 0;
                } else if (tbl_bar_) {
                    tbl_kind_ = 4;
                    tbl_rowpend_ = 0;
                    tbl_bar_ = 0;
                } else if (c != ' ' && c != '\t') {
                    tbl_rowpend_ = 0;
                }
            }
        }
#endif
#if HP_CELLTXT_MOD
        if (in_table_) {
            if (c == '|' || c == '!' || c == '\n') {
                if (cell_acc_) celltxt_ = cell_acc_;
                cell_acc_ = 0;
            } else {
                const int lc = c | 32;
                if (lc >= 'a' && lc <= 'z')
                    cell_acc_ = mix64(cell_acc_ * 31ull +
                                      static_cast<std::uint64_t>(lc));
            }
        }
#endif
#if HP_DEFAULTSORT_MOD || HP_DAB_MOD || HP_HATNOTE_MOD
        if (state_ == kWkCurly) {
            const int lc = c | 32;
            if (!tnm_done_) {
                if (lc >= 'a' && lc <= 'z' && tnm_i_ < 16) {
                    tnm_buf_[tnm_i_++] = static_cast<char>(lc);
                } else if (c == ':' || c == '|' || c == '}' || c == ' ' ||
                           c == '\n') {
                    tnm_done_ = 1;
                    tnm_mode_ = 0;
                    auto eq = [&](const char* w, int n) {
                        if (tnm_i_ != n) return 0;
                        for (int i = 0; i < n; ++i)
                            if (tnm_buf_[i] != w[i]) return 0;
                        return 1;
                    };
                    std::uint64_t nh = 0;
                    for (int i = 0; i < tnm_i_; ++i)
                        nh = mix64(nh * 31ull +
                                   static_cast<std::uint64_t>(tnm_buf_[i]));
                    if (eq("defaultsort", 11)) {
                        tnm_mode_ = 1;
#if HP_DEFAULTSORT_MOD
                        ds_hash_ = nh;
#endif
                    } else if (eq("disambig", 8) || eq("hndis", 5) ||
                               eq("dab", 3) || eq("disambiguation", 14)) {
                        tnm_mode_ = 2;
#if HP_DAB_MOD
                        dab_hash_ = nh;
#endif
                    } else if (eq("for", 3) || eq("about", 5) ||
                               eq("main", 4) || eq("further", 7)) {
                        tnm_mode_ = 3;
#if HP_HATNOTE_MOD
                        hat_hash_ = nh;
#endif
                    }
                }
            } else if (tnm_mode_ == 1 && c != '}' && c != '\n') {
#if HP_DEFAULTSORT_MOD
                const int lc2 = c | 32;
                if (lc2 >= 'a' && lc2 <= 'z')
                    ds_hash_ = mix64(ds_hash_ * 31ull +
                                     static_cast<std::uint64_t>(lc2));
#endif
            }
        }
#endif
#if HP_CITEKIND_MOD || HP_COORD_MOD
        if (state_ == kWkCurly && !ck_done_) {
            const int lc = c | 32;
            if (lc >= 'a' && lc <= 'z' && ck_i_ < 16) {
                ck_buf_[ck_i_++] = static_cast<char>(lc);
            } else if (c == ':' || c == '|' || c == '}' || c == ' ' ||
                       c == '\n') {
                ck_done_ = 1;
                auto eq = [&](const char* w, int n) {
                    if (ck_i_ != n) return 0;
                    for (int i = 0; i < n; ++i)
                        if (ck_buf_[i] != w[i]) return 0;
                    return 1;
                };
                auto haspre = [&](const char* w, int n) {
                    if (ck_i_ < n) return 0;
                    for (int i = 0; i < n; ++i)
                        if (ck_buf_[i] != w[i]) return 0;
                    return 1;
                };
#if HP_CITEKIND_MOD
                int ck = 0;
                if (eq("citation", 8)) ck = 5;
                else if (haspre("cite", 4)) {
                    if (eq("citeweb", 7)) ck = 1;
                    else if (eq("citejournal", 11)) ck = 2;
                    else if (eq("citebook", 8)) ck = 3;
                    else if (eq("citenews", 8)) ck = 4;
                    else ck = 6;
                }
                if (ck) citekind_ = ck;
#endif
#if HP_COORD_MOD
                if (eq("coord", 5) || eq("coordinates", 11) ||
                    eq("coordlink", 9))
                    coord_ = 1;
#endif
            }
        }
#endif
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
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
#endif
#if HP_SECTION_MUTE
        push_mute(c);
#endif

        if (c == '\n') {
            ++line_;
            first_of_line_ = 1;
            line_kind_ = 0;
            if (state_ == kWkText) ++para_;
#if HP_REDIR_MOD
            in_redir_ = 0;
#endif
#if HP_REDIR_MOD || HP_REDIRTARGET_MOD
            redir_i_ = 0;
#endif
#if HP_HEADING_MOD
            heading_ = 0;
            heading_run_ = 0;
#endif
#if HP_SECTITLE_MOD
            st_collect_ = 0;
            st_skip_ = 0;
#endif
#if HP_SECKIND_MOD
            if (sk_collect_ || sk_n_ > 0) seckind_apply_();
            sk_skip_ = 0;
            sk_collect_ = 0;
            sk_n_ = 0;
#endif
#if HP_INDENT_MOD
            indent_ = 0;
            indent_run_ = 0;
#endif
#if HP_LISTLEVEL_MOD
            list_level_ = 0;
            list_run_ = 0;
            list_kind_ = 0;
#endif
        } else if (first_of_line_ && c != ' ' && c != '\t') {
            first_of_line_ = 0;
            line_kind_ = c;
            // fx2: isParagraph = (fc == FIRSTUPPER); WIKIHEADER = line-start '>'
            is_paragraph_ = (c >= 'A' && c <= 'Z') ? 1 : 0;
            wiki_header_ = (c == '>') ? 1 : 0;
            if (is_paragraph_ && state_ == kWkText) state_ = kWkFirstUpper;
            if (wiki_header_ && state_ == kWkText) state_ = kWkHeader;
#if HP_REDIR_MOD || HP_REDIRTARGET_MOD
            if (c == '#') redir_i_ = 1;
#endif
#if HP_HEADING_MOD
            if (c == '=') {
                heading_ = 1;
                heading_run_ = 1;
            }
#endif
#if HP_SECTITLE_MOD
            if (c == '=') {
                st_skip_ = 1;
                st_collect_ = 0;
                st_hash_ = 0;
            }
#endif
#if HP_SECKIND_MOD
            if (c == '=') {
                sk_skip_ = 1;
                sk_collect_ = 0;
                sk_n_ = 0;
            }
#endif
#if HP_INDENT_MOD
            if (c == ':') {
                indent_ = 1;
                indent_run_ = 1;
            }
#endif
#if HP_LISTLEVEL_MOD
            if (c == '*' || c == '#') {
                list_level_ = 1;
                list_run_ = 1;
                list_kind_ = c;
            }
#endif
#if HP_LISTPOS_MOD
            if (c == '*' || c == '#') {
                if (prev_line_list_ && list_kind_line_ == c) ++list_pos_;
                else list_pos_ = 1;
                prev_line_list_ = 1;
                list_kind_line_ = c;
            } else {
                prev_line_list_ = 0;
                list_pos_ = 0;
            }
#endif
        } else {
#if HP_REDIR_MOD || HP_REDIRTARGET_MOD
            if (redir_i_ > 0 && redir_i_ < 9
#if HP_REDIR_MOD
                && !in_redir_
#endif
            ) {
                static const char rd[] = "redirect";
                if ((c | 32) == rd[redir_i_ - 1]) {
                    ++redir_i_;
                    if (redir_i_ == 9) {
#if HP_REDIR_MOD
                        in_redir_ = 1;
#endif
#if HP_REDIRTARGET_MOD
                        rtgt_phase_ = 1;
                        rtgt_hash_ = 0;
#endif
                    }
                } else {
                    redir_i_ = 0;
                }
            }
#endif
#if HP_REDIRTARGET_MOD
            if (rtgt_phase_ == 1) {
                if (c == '[') rtgt_phase_ = 2;
            } else if (rtgt_phase_ == 2) {
                if (c == '[') {
                    rtgt_phase_ = 3;
                    rtgt_hash_ = 0;
                } else if (c != ' ' && c != '\t') {
                    rtgt_phase_ = 0;
                }
            } else if (rtgt_phase_ == 3) {
                if (c == ']' || c == '|' || c == '\n') {
                    rtgt_phase_ = 0;
                } else {
                    const int lc = c | 32;
                    rtgt_hash_ = mix64(rtgt_hash_ * 31ull +
                                       static_cast<std::uint64_t>(lc));
                }
            }
#endif
#if HP_HEADING_MOD
            if (heading_run_) {
                if (c == '=') {
                    if (heading_ < 6) ++heading_;
                } else {
                    heading_run_ = 0;
                }
            }
#endif
#if HP_SECTITLE_MOD
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
#endif
#if HP_SECKIND_MOD
            if (sk_skip_) {
                if (c != '=') {
                    sk_skip_ = 0;
                    sk_collect_ = 1;
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z' && sk_n_ < 16)
                        sk_buf_[sk_n_++] = static_cast<char>(lc);
                }
            } else if (sk_collect_) {
                if (c == '=') {
                    seckind_apply_();
                    sk_collect_ = 0;
                    sk_n_ = 0;
                } else {
                    const int lc = c | 32;
                    if (lc >= 'a' && lc <= 'z' && sk_n_ < 16)
                        sk_buf_[sk_n_++] = static_cast<char>(lc);
                }
            }
#endif
#if HP_INDENT_MOD
            if (indent_run_) {
                if (c == ':') {
                    if (indent_ < 8) ++indent_;
                } else {
                    indent_run_ = 0;
                }
            }
#endif
#if HP_LISTLEVEL_MOD
            if (list_run_) {
                if (c == list_kind_) {
                    if (list_level_ < 8) ++list_level_;
                } else {
                    list_run_ = 0;
                }
            }
#endif
        }
#if HP_EXTLINK_MOD
        if (c == '[' && prev1_ != '[') ext_pend_ = 1;
#endif
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
#if HP_SECTION_MUTE
        return mute_;
#else
        return 0;
#endif
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
#if HP_WIKI_TEMP
               | (static_cast<std::uint64_t>(is_temp_ & 1) << 39)
#endif
#if HP_NLCHAR
               | (static_cast<std::uint64_t>(nl_mode() & 3) << 37)
#endif
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
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
        const int r = (tbl_row_ + 3) & 3;
        const int c = tbl_cell_ > 31 ? 31 : tbl_cell_;
        return tbl_cells_[r][c];
#else
        return 0;
#endif
    }
    int cell_first() const {
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
        const int c = tbl_cell_ > 31 ? 31 : tbl_cell_;
        return tbl_cells_[tbl_row_][c];
#else
        return 0;
#endif
    }
    int tbl_cell() const {
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
        return tbl_cell_;
#else
        return 0;
#endif
    }
    std::uint64_t tpl_name() const {
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
        return tpl_name_;
#else
        return 0;
#endif
    }
    std::uint64_t infokey() const {
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
        return key_;
#else
        return 0;
#endif
    }
    int bar_idx() const {
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
        return bar_idx_;
#else
        return 0;
#endif
    }
    int in_ref() const {
#if HP_CITE_MOD
        return in_ref_;
#else
        return 0;
#endif
    }
    int after_pipe() const {
#if HP_LINKPIPE_MOD
        return link_pipe_;
#else
        return 0;
#endif
    }
    std::uint64_t link_disp() const {
#if HP_LINKPIPE_MOD
        return disp_;
#else
        return 0;
#endif
    }
    std::uint64_t cat_ns() const {
#if HP_CAT_MOD
        return ns_hash_;
#else
        return 0;
#endif
    }
    int in_redir() const {
#if HP_REDIR_MOD
        return in_redir_;
#else
        return 0;
#endif
    }
    int heading_level() const {
#if HP_HEADING_MOD
        return heading_;
#else
        return 0;
#endif
    }
    int in_ext() const {
#if HP_EXTLINK_MOD
        return in_ext_;
#else
        return 0;
#endif
    }
    std::uint64_t refname() const {
#if HP_REFNAME_MOD
        return refname_;
#else
        return 0;
#endif
    }
    std::uint64_t entity() const {
#if HP_ENTITY_MOD
        return entity_;
#else
        return 0;
#endif
    }
    int indent_level() const {
#if HP_INDENT_MOD
        return indent_;
#else
        return 0;
#endif
    }
    int list_level() const {
#if HP_LISTLEVEL_MOD
        return list_level_;
#else
        return 0;
#endif
    }
    std::uint64_t magic() const {
#if HP_MAGIC_MOD
        return magic_hash_;
#else
        return 0;
#endif
    }
    int in_nowiki() const {
#if HP_NOWIKI_MOD
        return nw_bits_;
#else
        return 0;
#endif
    }
    std::uint64_t page_title() const {
#if HP_TITLE_MOD
        return title_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t page_id() const {
#if HP_PAGEID_MOD
        return page_id_;
#else
        return 0;
#endif
    }
    std::uint64_t username() const {
#if HP_USER_MOD
        return user_hash_;
#else
        return 0;
#endif
    }
    int in_text() const {
#if HP_TEXT_MOD
        return in_text_;
#else
        return 0;
#endif
    }
    int ns_id() const {
#if HP_NS_MOD
        return ns_id_;
#else
        return 0;
#endif
    }
    int dump_redir() const {
#if HP_DUMPREDIR_MOD
        return dump_redir_;
#else
        return 0;
#endif
    }
    std::uint64_t ip_hash() const {
#if HP_IP_MOD
        return ip_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t rev_comment() const {
#if HP_REVCOMMENT_MOD
        return comment_hash_;
#else
        return 0;
#endif
    }
    int minor_edit() const {
#if HP_MINOR_MOD
        return minor_;
#else
        return 0;
#endif
    }
    std::uint64_t wiki_model() const {
#if HP_WIKIMODEL_MOD
        return model_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t sectitle() const {
#if HP_SECTITLE_MOD
        return st_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t parser_fn() const {
#if HP_PARSERFN_MOD
        return pfn_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t table_class() const {
#if HP_TABLECLASS_MOD
        return tc_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t anchor() const {
#if HP_ANCHOR_MOD
        return an_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t pub_id() const {
#if HP_PUBID_MOD
        return pub_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t temp_pos() const {
#if HP_TEMPPOS_MOD
        return tp_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t wiki_stack() const {
#if HP_WIKISTACK_MOD
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
#else
        return 0;
#endif
    }
    std::uint64_t lang_prefix() const {
#if HP_LANG_MOD
        return lang_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t cat_sort() const {
#if HP_CATSORT_MOD
        return extra_hash_;
#else
        return 0;
#endif
    }
    int tbl_row() const {
#if HP_TBLROW_MOD
        return tbl_kind_;
#else
        return 0;
#endif
    }
    std::uint64_t file_opt() const {
#if HP_FILEOPT_MOD
        return extra_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t default_sort() const {
#if HP_DEFAULTSORT_MOD
        return ds_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t redir_target() const {
#if HP_REDIRTARGET_MOD
        return rtgt_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t dab() const {
#if HP_DAB_MOD
        return dab_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t hatnote() const {
#if HP_HATNOTE_MOD
        return hat_hash_;
#else
        return 0;
#endif
    }
    std::uint64_t last_link() const {
#if HP_LASTLINK_MOD
        return lastlink_;
#else
        return 0;
#endif
    }
    std::uint64_t first_word() const {
#if HP_FWORD_MOD
        return fword_;
#else
        return 0;
#endif
    }
    std::uint64_t year() const {
#if HP_YEAR_MOD
        return year_;
#else
        return 0;
#endif
    }
    int cap_mask() const {
#if HP_CAPMASK_MOD
        return cap_mask_;
#else
        return 0;
#endif
    }
    std::uint64_t cell_text() const {
#if HP_CELLTXT_MOD
        return celltxt_;
#else
        return 0;
#endif
    }
    std::uint64_t http_host() const {
#if HP_HTTPHOST_MOD
        return httphost_;
#else
        return 0;
#endif
    }
    std::uint64_t last_paren() const {
#if HP_PAREN_MOD
        return lastparen_;
#else
        return 0;
#endif
    }
    int list_pos() const {
#if HP_LISTPOS_MOD
        return list_pos_;
#else
        return 0;
#endif
    }
    std::uint64_t word_shape() const {
#if HP_SHAPE_MOD
        return shape_;
#else
        return 0;
#endif
    }
    std::uint64_t word_suffix() const {
#if HP_SUFFIX_MOD
        return suffix_;
#else
        return 0;
#endif
    }
    std::uint64_t word_prefix() const {
#if HP_PREFIX_MOD
        return prefix_;
#else
        return 0;
#endif
    }
    int char_cls() const {
#if HP_CHARCLS_MOD
        return charcls_;
#else
        return 0;
#endif
    }
    int vowel_mask() const {
#if HP_VOWEL_MOD
        return vowel_;
#else
        return 0;
#endif
    }
    std::uint64_t contraction() const {
#if HP_CONTR_MOD
        return contr_;
#else
        return 0;
#endif
    }
    std::uint64_t hyphen_word() const {
#if HP_HYPHEN_MOD
        return hyphen_;
#else
        return 0;
#endif
    }
    int token_cls() const {
#if HP_TOKENCLS_MOD
        return tok_cls_;
#else
        return 0;
#endif
    }
    int run_len() const {
#if HP_RUNLEN_MOD
        return runlen_;
#else
        return 0;
#endif
    }
    int word_pos() const {
#if HP_WPOS_MOD
        return wpos_;
#else
        return 0;
#endif
    }
    int blank_n() const {
#if HP_BLANK_MOD
        return blank_n_;
#else
        return 0;
#endif
    }
    int space_run() const {
#if HP_SPRUN_MOD
        return sprun_;
#else
        return 0;
#endif
    }
    int line_len() const {
#if HP_LINELEN_MOD
        return linelen_;
#else
        return 0;
#endif
    }
    int tag_dist() const {
#if HP_TAGDIST_MOD
        return tagdist_;
#else
        return 0;
#endif
    }
    int mark_dist() const {
#if HP_MARKDIST_MOD
        return markdist_;
#else
        return 0;
#endif
    }
    int upper_gap() const {
#if HP_UPPERGAP_MOD
        return uppergap_;
#else
        return 0;
#endif
    }
    int month() const {
#if HP_MONTH_MOD
        return month_;
#else
        return 0;
#endif
    }
    int in_gallery() const {
#if HP_GALLERY_MOD
        return in_gallery_;
#else
        return 0;
#endif
    }
    int sec_kind() const {
#if HP_SECKIND_MOD
        return seckind_;
#else
        return 0;
#endif
    }
    int cite_kind() const {
#if HP_CITEKIND_MOD
        return citekind_;
#else
        return 0;
#endif
    }
    std::uint64_t tag_name_cm() const {
#if HP_TAGNAME_MOD
        return tag_name_;
#else
        return 0;
#endif
    }
    int col_span() const {
#if HP_COLSPAN_MOD
        return colspan_;
#else
        return 0;
#endif
    }
    std::uint64_t style_hash() const {
#if HP_STYLE_MOD
        return style_hash_;
#else
        return 0;
#endif
    }
    int in_coord() const {
#if HP_COORD_MOD
        return coord_;
#else
        return 0;
#endif
    }
    int digit_gap() const {
#if HP_DIGITGAP_MOD
        return digitgap_;
#else
        return 0;
#endif
    }
    int dot_gap() const {
#if HP_DOTGAP_MOD
        return dotgap_;
#else
        return 0;
#endif
    }
    int comma_gap() const {
#if HP_COMMAGAP_MOD
        return commagap_;
#else
        return 0;
#endif
    }
    int word_len() const {
#if HP_WORDLEN_MOD
        return wordlen_;
#else
        return 0;
#endif
    }
    int sent_len() const {
#if HP_SENTLEN_MOD
        return sentlen_;
#else
        return 0;
#endif
    }
    int lower_gap() const {
#if HP_LOWERGAP_MOD
        return lowergap_;
#else
        return 0;
#endif
    }
    int digit_pos() const {
#if HP_DIGITPOS_MOD
        return digitpos_;
#else
        return 0;
#endif
    }
    int slash_gap() const {
#if HP_SLASHGAP_MOD
        return slashgap_;
#else
        return 0;
#endif
    }
    int dig_len() const {
#if HP_DIGLEN_MOD
        return diglen_;
#else
        return 0;
#endif
    }
    int prev_line() const {
#if HP_PREVLINE_MOD
        return prevline_;
#else
        return 0;
#endif
    }
    int prev_sent() const {
#if HP_PREVSENT_MOD
        return prevsent_;
#else
        return 0;
#endif
    }
    int link_len() const {
#if HP_LINKLEN_MOD
        return linklen_;
#else
        return 0;
#endif
    }
    int tpl_len() const {
#if HP_TPLLEN_MOD
        return tpllen_;
#else
        return 0;
#endif
    }
    int para_len() const {
#if HP_PARALEN_MOD
        return paralen_;
#else
        return 0;
#endif
    }
    int alnum_len() const {
#if HP_ALNUMLEN_MOD
        return alnumlen_;
#else
        return 0;
#endif
    }
    int sp_len() const {
#if HP_SPLEN_MOD
        return splen_;
#else
        return 0;
#endif
    }
    int title_word() const {
#if HP_TITLEWORD_MOD
        return titleword_;
#else
        return 0;
#endif
    }
    int head_word() const {
#if HP_HEADWORD_MOD
        return headword_;
#else
        return 0;
#endif
    }
    int in_init() const {
#if HP_INIT_MOD
        return init_;
#else
        return 0;
#endif
    }
    int ordinal() const {
#if HP_ORDINAL_MOD
        return ordinal_;
#else
        return 0;
#endif
    }
    int unit() const {
#if HP_UNIT_MOD
        return unit_;
#else
        return 0;
#endif
    }
    int in_decimal() const {
#if HP_DECIMAL_MOD
        return decimal_;
#else
        return 0;
#endif
    }
    int word_repeat() const {
#if HP_REPEAT_MOD
        return repeat_;
#else
        return 0;
#endif
    }
    int case_flip() const {
#if HP_CASEFLIP_MOD
        return caseflip_;
#else
        return 0;
#endif
    }
    int in_lead() const {
#if HP_LEAD_MOD
        return lead_;
#else
        return 0;
#endif
    }
    int info_val() const {
#if HP_INFOVAL_MOD
        return infoval_;
#else
        return 0;
#endif
    }
    int link_trail() const {
#if HP_LINKTRAIL_MOD
        return linktrail_;
#else
        return 0;
#endif
    }
    int cell_kind() const {
#if HP_CELLKIND_MOD
        return cellkind_;
#else
        return 0;
#endif
    }
    int tbl_col() const {
#if HP_TBLCOL_MOD
        return tblcol_;
#else
        return 0;
#endif
    }
    int head_idx() const {
#if HP_HEADIDX_MOD
        return headidx_;
#else
        return 0;
#endif
    }
    int html_fmt() const {
#if HP_HTMLFMT_MOD
        return htmlfmt_;
#else
        return 0;
#endif
    }
    int in_infobox() const {
#if HP_INFOBOX_MOD
        return infobox_;
#else
        return 0;
#endif
    }
    int sec_level() const {
#if HP_SECLEVEL_MOD
        return seclevel_;
#else
        return 0;
#endif
    }
    int brace3() const {
#if HP_BRACE3_MOD
        return brace3_;
#else
        return 0;
#endif
    }
    int named_arg() const {
#if HP_NAMEDARG_MOD
        return namedarg_;
#else
        return 0;
#endif
    }
    int include_bits() const {
#if HP_INCLUDE_MOD
        return include_;
#else
        return 0;
#endif
    }
    int sig_run() const {
#if HP_SIG_MOD
        return sig_;
#else
        return 0;
#endif
    }
    int wiki_bold() const {
#if HP_WIKIBOLD_MOD
        return wikibold_;
#else
        return 0;
#endif
    }
    int url_part() const {
#if HP_URLPART_MOD
        return urlpart_;
#else
        return 0;
#endif
    }
    int ref_idx() const {
#if HP_REFIDX_MOD
        return refidx_;
#else
        return 0;
#endif
    }
    int is_temp() const {
#if HP_WIKI_TEMP
        return is_temp_;
#else
        return 0;
#endif
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
#if HP_WIKI_TEMP
    int is_temp_ = 0;
#endif
#if HP_TPLNAME_MOD || HP_INFOKEY_MOD || HP_BARIDX_MOD
    std::uint64_t tpl_name_ = 0;
    std::uint64_t key_ = 0;
    int bar_idx_ = 0;
    int tpl_collect_ = 0;
    int key_collect_ = 0;
#endif
#if HP_CITE_MOD
    int in_ref_ = 0;
    int cite_slash_ = 0;
    std::uint32_t cite_buf_ = 0;
#endif
#if HP_LINKPIPE_MOD
    int link_pipe_ = 0;
    std::uint64_t disp_ = 0;
#endif
#if HP_CAT_MOD
    int ns_collect_ = 0;
    std::uint64_t ns_hash_ = 0;
#endif
#if HP_REDIR_MOD
    int in_redir_ = 0;
#endif
#if HP_REDIR_MOD || HP_REDIRTARGET_MOD
    int redir_i_ = 0;
#endif
#if HP_REDIRTARGET_MOD
    int rtgt_phase_ = 0;
    std::uint64_t rtgt_hash_ = 0;
#endif
#if HP_LANG_MOD
    int lang_n_ = 0;
    std::uint64_t lang_acc_ = 0;
    std::uint64_t lang_hash_ = 0;
#endif
#if HP_CATSORT_MOD || HP_FILEOPT_MOD
    char pre_buf_[12] = {};
    int pre_i_ = 0;
    int pre_kind_ = 0;
    int pipe_col_ = 0;
    std::uint64_t extra_hash_ = 0;
#endif
#if HP_TBLROW_MOD
    int tbl_kind_ = 0;
    int tbl_rowpend_ = 0;
    int tbl_bar_ = 0;
#endif
#if HP_DEFAULTSORT_MOD || HP_DAB_MOD || HP_HATNOTE_MOD
    char tnm_buf_[16] = {};
    int tnm_i_ = 0;
    int tnm_done_ = 1;
    int tnm_mode_ = 0;
#endif
#if HP_DEFAULTSORT_MOD
    std::uint64_t ds_hash_ = 0;
#endif
#if HP_DAB_MOD
    std::uint64_t dab_hash_ = 0;
#endif
#if HP_HATNOTE_MOD
    std::uint64_t hat_hash_ = 0;
#endif
#if HP_LASTLINK_MOD
    std::uint64_t lastlink_ = 0;
    std::uint64_t ll_acc_ = 0;
    int ll_stop_ = 0;
#endif
#if HP_FWORD_MOD
    std::uint64_t fword_ = 0;
    std::uint64_t fw_acc_ = 0;
    int fw_pending_ = 1;
    int fw_n_ = 0;
#endif
#if HP_YEAR_MOD
    std::uint64_t year_ = 0;
    int yr_n_ = 0;
    int yr_val_ = 0;
#endif
#if HP_CAPMASK_MOD
    int cap_mask_ = 0;
    int cap_bits_ = 0;
    int cap_n_ = 0;
#endif
#if HP_CELLTXT_MOD
    std::uint64_t celltxt_ = 0;
    std::uint64_t cell_acc_ = 0;
#endif
#if HP_HTTPHOST_MOD
    std::uint64_t httphost_ = 0;
    std::uint64_t ht_acc_ = 0;
    int ht_done_ = 1;
#endif
#if HP_PAREN_MOD
    std::uint64_t lastparen_ = 0;
    std::uint64_t paren_acc_ = 0;
    int paren_d_ = 0;
#endif
#if HP_LISTPOS_MOD
    int list_pos_ = 0;
    int prev_line_list_ = 0;
    int list_kind_line_ = 0;
#endif
#if HP_SHAPE_MOD
    std::uint64_t shape_ = 0;
    int shape_on_ = 0;
#endif
#if HP_SUFFIX_MOD
    std::uint64_t suffix_ = 0;
    std::uint64_t suffix_acc_ = 0;
    int suf_n_ = 0;
#endif
#if HP_PREFIX_MOD
    std::uint64_t prefix_ = 0;
    int pre_n_ = 0;
#endif
#if HP_CHARCLS_MOD
    int charcls_ = 0;
#endif
#if HP_VOWEL_MOD
    int vowel_ = 0;
#endif
#if HP_CONTR_MOD
    std::uint64_t contr_ = 0;
    std::uint64_t contr_acc_ = 0;
    int contr_apos_ = 0;
#endif
#if HP_HYPHEN_MOD
    std::uint64_t hyphen_ = 0;
    std::uint64_t hy_acc_ = 0;
    int hy_seen_ = 0;
#endif
#if HP_TOKENCLS_MOD
    int tok_cls_ = 0;
#endif
#if HP_RUNLEN_MOD
    int runlen_ = 0;
#endif
#if HP_WPOS_MOD
    int wpos_ = 0;
#endif
#if HP_BLANK_MOD
    int blank_n_ = 0;
#endif
#if HP_SPRUN_MOD
    int sprun_ = 0;
#endif
#if HP_LINELEN_MOD
    int linelen_ = 0;
#endif
#if HP_TAGDIST_MOD
    int tagdist_ = 0;
#endif
#if HP_MARKDIST_MOD
    int markdist_ = 0;
#endif
#if HP_UPPERGAP_MOD
    int uppergap_ = 0;
#endif
#if HP_MONTH_MOD
    int month_ = 0;
    int mo_n_ = 0;
    char mo_buf_[9] = {};
#endif
#if HP_GALLERY_MOD
    int in_gallery_ = 0;
    int gal_slash_ = 0;
    int gal_n_ = 0;
    char gal_buf_[8] = {};
#endif
#if HP_SECKIND_MOD
    void seckind_apply_() {
        auto eq = [&](const char* w, int n) {
            if (sk_n_ != n) return 0;
            for (int i = 0; i < n; ++i)
                if (sk_buf_[i] != w[i]) return 0;
            return 1;
        };
        int k = 7;
        if (sk_n_ == 0) k = 0;
        else if (eq("references", 10) || eq("refs", 4) || eq("notes", 5) ||
                 eq("footnotes", 9))
            k = 1;
        else if (eq("seealso", 7) || eq("see", 3))
            k = 2;
        else if (eq("externallinks", 13) || eq("external", 8))
            k = 3;
        else if (eq("bibliography", 12) || eq("sources", 7) ||
                 eq("furtherreading", 15))
            k = 4;
        else if (eq("history", 7) || eq("career", 6) || eq("biography", 9) ||
                 eq("life", 4))
            k = 5;
        seckind_ = k;
    }
    int seckind_ = 0;
    int sk_skip_ = 0;
    int sk_collect_ = 0;
    int sk_n_ = 0;
    char sk_buf_[16] = {};
#endif
#if HP_CITEKIND_MOD
    int citekind_ = 0;
#endif
#if HP_CITEKIND_MOD || HP_COORD_MOD
    int ck_i_ = 0;
    int ck_done_ = 0;
    char ck_buf_[16] = {};
#endif
#if HP_COLSPAN_MOD
    int colspan_ = 0;
    int cs_eat_ = 0;
    int cs_val_ = 0;
    std::uint64_t cs_win_ = 0;
#endif
#if HP_STYLE_MOD
    std::uint64_t style_hash_ = 0;
    int style_eat_ = 0;
    std::uint64_t st_win_ = 0;
#endif
#if HP_COORD_MOD
    int coord_ = 0;
#endif
#if HP_DIGITGAP_MOD
    int digitgap_ = 0;
#endif
#if HP_DOTGAP_MOD
    int dotgap_ = 0;
#endif
#if HP_COMMAGAP_MOD
    int commagap_ = 0;
#endif
#if HP_WORDLEN_MOD
    int wordlen_ = 0;
    int wl_run_ = 0;
#endif
#if HP_SENTLEN_MOD
    int sentlen_ = 0;
#endif
#if HP_LOWERGAP_MOD
    int lowergap_ = 0;
#endif
#if HP_DIGITPOS_MOD
    int digitpos_ = 0;
#endif
#if HP_SLASHGAP_MOD
    int slashgap_ = 0;
#endif
#if HP_DIGLEN_MOD
    int diglen_ = 0;
    int dl_run_ = 0;
#endif
#if HP_PREVLINE_MOD
    int prevline_ = 0;
    int ll_run_ = 0;
#endif
#if HP_PREVSENT_MOD
    int prevsent_ = 0;
    int ss_run_ = 0;
#endif
#if HP_LINKLEN_MOD
    int linklen_ = 0;
    int lk_run_ = 0;
#endif
#if HP_TPLLEN_MOD
    int tpllen_ = 0;
    int tp_run_ = 0;
#endif
#if HP_PARALEN_MOD
    int paralen_ = 0;
    int pr_run_ = 0;
#endif
#if HP_ALNUMLEN_MOD
    int alnumlen_ = 0;
    int al_run_ = 0;
#endif
#if HP_SPLEN_MOD
    int splen_ = 0;
    int sp_run_ = 0;
#endif
#if HP_TITLEWORD_MOD || HP_HEADWORD_MOD || HP_REPEAT_MOD
    std::uint64_t cw_acc_ = 0;
    int cw_on_ = 0;
#endif
#if HP_TITLEWORD_MOD
    std::uint64_t tw_ring_[8] = {};
    int tw_n_ = 0;
    int titleword_ = 0;
#endif
#if HP_HEADWORD_MOD
    std::uint64_t hw_ring_[8] = {};
    int hw_n_ = 0;
    int headword_ = 0;
#endif
#if HP_INIT_MOD
    int init_ = 0;
    int init_st_ = 0;
#endif
#if HP_ORDINAL_MOD
    int ordinal_ = 0;
    int ord_pend_ = 0;
    int ord_n_ = 0;
    char ord_buf_[2] = {};
#endif
#if HP_UNIT_MOD
    int unit_ = 0;
    int un_pend_ = 0;
    int un_n_ = 0;
    char un_buf_[4] = {};
#endif
#if HP_DECIMAL_MOD
    int decimal_ = 0;
    int dec_saw_ = 0;
    int dec_pend_ = 0;
#endif
#if HP_REPEAT_MOD
    std::uint64_t cw_prev_ = 0;
    int repeat_ = 0;
#endif
#if HP_CASEFLIP_MOD
    int caseflip_ = 0;
#endif
#if HP_LEAD_MOD
    int lead_ = 1;
#endif
#if HP_INFOVAL_MOD
    int infoval_ = 0;
    int iv_tpl_ = 0;
    int iv_set_ = 0;
#endif
#if HP_LINKTRAIL_MOD
    int linktrail_ = 0;
    int lt_on_ = 0;
#endif
#if HP_CELLKIND_MOD
    int cellkind_ = 0;
    int ck_sol_ = 0;
    int ck_bar_ = 0;
#endif
#if HP_TBLCOL_MOD
    int tblcol_ = 0;
#endif
#if HP_HEADIDX_MOD
    int headidx_ = 0;
    int hi_on_ = 0;
#endif
#if HP_HTMLFMT_MOD
    int htmlfmt_ = 0;
    int hf_n_ = 0;
    int hf_slash_ = 0;
    char hf_buf_[8] = {};
#endif
#if HP_INFOBOX_MOD
    int infobox_ = 0;
    int ib_depth_ = 0;
    int ib_n_ = 0;
    int ib_col_ = 0;
    char ib_buf_[8] = {};
#endif
#if HP_SECLEVEL_MOD
    int seclevel_ = 0;
    int sl_run_ = 0;
    int sl_at_ = 0;
#endif
#if HP_BRACE3_MOD
    int brace3_ = 0;
    int b3_run_ = 0;
    int b3_cls_ = 0;
#endif
#if HP_NAMEDARG_MOD
    int namedarg_ = 0;
#endif
#if HP_INCLUDE_MOD
    int include_ = 0;
    int ic_n_ = 0;
    int ic_slash_ = 0;
    char ic_buf_[12] = {};
#endif
#if HP_SIG_MOD
    int sig_ = 0;
#endif
#if HP_WIKIBOLD_MOD
    int wikibold_ = 0;
    int wb_run_ = 0;
#endif
#if HP_URLPART_MOD
    int urlpart_ = 0;
#endif
#if HP_REFIDX_MOD
    int refidx_ = 0;
    int ri_n_ = 0;
    int ri_slash_ = 0;
    char ri_buf_[4] = {};
#endif
#if HP_HEADING_MOD
    int heading_ = 0;
    int heading_run_ = 0;
#endif
#if HP_EXTLINK_MOD
    int in_ext_ = 0;
    int ext_pend_ = 0;
#endif
#if HP_REFNAME_MOD
    std::uint64_t refname_ = 0;
    std::uint64_t refn_win_ = 0;
    int refn_collect_ = 0;
    int refn_skipq_ = 0;
#endif
#if HP_ENTITY_MOD
    std::uint64_t entity_ = 0;
    int ent_collect_ = 0;
#endif
#if HP_INDENT_MOD
    int indent_ = 0;
    int indent_run_ = 0;
#endif
#if HP_LISTLEVEL_MOD
    int list_level_ = 0;
    int list_run_ = 0;
    int list_kind_ = 0;
#endif
#if HP_MAGIC_MOD
    int magic_collect_ = 0;
    std::uint64_t magic_hash_ = 0;
#endif
#if HP_NOWIKI_MOD
    void nowiki_apply_() {
        int bit = 0;
        auto eq = [&](const char* s) {
            std::uint64_t w = 0;
            int m = 0;
            while (s[m]) {
                w = w * 31ull + static_cast<std::uint64_t>(
                                    static_cast<unsigned char>(s[m]));
                ++m;
            }
            return m == nw_n_ && w == nw_name_;
        };
        if (eq("nowiki")) bit = 1;
        else if (eq("math")) bit = 2;
        else if (eq("pre")) bit = 4;
        else if (eq("code")) bit = 8;
        if (bit) {
            if (nw_slash_) nw_bits_ &= ~bit;
            else nw_bits_ |= bit;
        }
    }
    int nw_slash_ = 0;
    int nw_n_ = 0;
    int nw_bits_ = 0;
    std::uint64_t nw_name_ = 0;
#endif
#if HP_DUMP_XML
    void dump_apply_() {
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
#if HP_TITLE_MOD
        if (eq("title")) {
            in_title_ = xml_slash_ ? 0 : 1;
            if (in_title_) title_hash_ = 0;
        }
#endif
#if HP_TITLEWORD_MOD
        if (eq("title") && !xml_slash_) {
            tw_n_ = 0;
            titleword_ = 0;
            cw_acc_ = 0;
            cw_on_ = 0;
        }
#endif
#if HP_USER_MOD
        if (eq("username")) {
            in_user_ = xml_slash_ ? 0 : 1;
            if (in_user_) user_hash_ = 0;
        }
#endif
#if HP_TEXT_MOD
        if (eq("text")) in_text_ = xml_slash_ ? 0 : 1;
#endif
#if HP_PAGEID_MOD
        if (eq("page") && !xml_slash_) {
            pageid_seen_ = 0;
            page_id_ = 0;
        }
        if (eq("id") && !xml_slash_ && !pageid_seen_) in_id_ = 1;
#endif
#if HP_NS_MOD
        if (eq("ns") && !xml_slash_) {
            in_ns_ = 1;
            ns_id_ = 0;
        }
#endif
#if HP_DUMPREDIR_MOD
        if (eq("page") && !xml_slash_) dump_redir_ = 0;
        if (eq("redirect")) dump_redir_ = 1;
#endif
#if HP_IP_MOD
        if (eq("ip")) {
            in_ip_ = xml_slash_ ? 0 : 1;
            if (in_ip_) ip_hash_ = 0;
        }
#endif
#if HP_REVCOMMENT_MOD
        if (eq("comment")) {
            in_comment_ = xml_slash_ ? 0 : 1;
            if (in_comment_) comment_hash_ = 0;
        }
#endif
#if HP_MINOR_MOD
        if (eq("page") && !xml_slash_) minor_ = 0;
        if (eq("minor")) minor_ = 1;
#endif
#if HP_WIKIMODEL_MOD
        if (eq("model")) {
            in_model_ = xml_slash_ ? 0 : 1;
            if (in_model_) model_hash_ = 0;
        }
#endif
#if HP_LASTLINK_MOD || HP_FWORD_MOD || HP_YEAR_MOD || HP_PAREN_MOD || \
    HP_LISTPOS_MOD || HP_HTTPHOST_MOD || HP_CELLTXT_MOD || HP_CAPMASK_MOD || \
    HP_SHAPE_MOD || HP_SUFFIX_MOD || HP_PREFIX_MOD || HP_CHARCLS_MOD || \
    HP_VOWEL_MOD || HP_CONTR_MOD || HP_HYPHEN_MOD || HP_TOKENCLS_MOD || \
    HP_RUNLEN_MOD || HP_WPOS_MOD || HP_BLANK_MOD || HP_SPRUN_MOD || \
    HP_LINELEN_MOD || HP_TAGDIST_MOD || HP_MARKDIST_MOD || HP_UPPERGAP_MOD || \
    HP_MONTH_MOD || HP_GALLERY_MOD || HP_SECKIND_MOD || HP_CITEKIND_MOD || \
    HP_TAGNAME_MOD || HP_COLSPAN_MOD || HP_STYLE_MOD || HP_COORD_MOD || \
    HP_DIGITGAP_MOD || HP_DOTGAP_MOD || HP_COMMAGAP_MOD || HP_WORDLEN_MOD || \
    HP_SENTLEN_MOD || HP_LOWERGAP_MOD || HP_DIGITPOS_MOD || HP_SLASHGAP_MOD || \
    HP_DIGLEN_MOD || HP_PREVLINE_MOD || HP_PREVSENT_MOD || HP_LINKLEN_MOD || \
    HP_TPLLEN_MOD || HP_PARALEN_MOD || HP_ALNUMLEN_MOD || HP_SPLEN_MOD || \
    HP_TITLEWORD_MOD || HP_HEADWORD_MOD || HP_INIT_MOD || HP_ORDINAL_MOD || \
    HP_UNIT_MOD || HP_DECIMAL_MOD || HP_REPEAT_MOD || HP_CASEFLIP_MOD || \
    HP_LEAD_MOD || HP_INFOVAL_MOD || HP_LINKTRAIL_MOD || HP_CELLKIND_MOD || \
    HP_TBLCOL_MOD || HP_HEADIDX_MOD || HP_HTMLFMT_MOD || HP_INFOBOX_MOD || \
    HP_SECLEVEL_MOD || HP_BRACE3_MOD || HP_NAMEDARG_MOD || HP_INCLUDE_MOD || \
    HP_SIG_MOD || HP_WIKIBOLD_MOD || HP_URLPART_MOD || HP_REFIDX_MOD
        if (eq("page") && !xml_slash_) {
#if HP_LASTLINK_MOD
            lastlink_ = 0;
            ll_acc_ = 0;
#endif
#if HP_FWORD_MOD
            fword_ = 0;
            fw_acc_ = 0;
            fw_pending_ = 1;
            fw_n_ = 0;
#endif
#if HP_YEAR_MOD
            year_ = 0;
            yr_n_ = 0;
            yr_val_ = 0;
#endif
#if HP_CAPMASK_MOD
            cap_mask_ = 0;
            cap_bits_ = 0;
            cap_n_ = 0;
#endif
#if HP_CELLTXT_MOD
            celltxt_ = 0;
            cell_acc_ = 0;
#endif
#if HP_HTTPHOST_MOD
            httphost_ = 0;
            ht_acc_ = 0;
            ht_done_ = 1;
#endif
#if HP_PAREN_MOD
            lastparen_ = 0;
            paren_acc_ = 0;
            paren_d_ = 0;
#endif
#if HP_LISTPOS_MOD
            list_pos_ = 0;
            prev_line_list_ = 0;
            list_kind_line_ = 0;
#endif
#if HP_SHAPE_MOD
            shape_ = 0;
            shape_on_ = 0;
#endif
#if HP_SUFFIX_MOD
            suffix_ = 0;
            suffix_acc_ = 0;
            suf_n_ = 0;
#endif
#if HP_PREFIX_MOD
            prefix_ = 0;
            pre_n_ = 0;
#endif
#if HP_CHARCLS_MOD
            charcls_ = 0;
#endif
#if HP_VOWEL_MOD
            vowel_ = 0;
#endif
#if HP_CONTR_MOD
            contr_ = 0;
            contr_acc_ = 0;
            contr_apos_ = 0;
#endif
#if HP_HYPHEN_MOD
            hyphen_ = 0;
            hy_acc_ = 0;
            hy_seen_ = 0;
#endif
#if HP_TOKENCLS_MOD
            tok_cls_ = 0;
#endif
#if HP_RUNLEN_MOD
            runlen_ = 0;
#endif
#if HP_WPOS_MOD
            wpos_ = 0;
#endif
#if HP_BLANK_MOD
            blank_n_ = 0;
#endif
#if HP_SPRUN_MOD
            sprun_ = 0;
#endif
#if HP_LINELEN_MOD
            linelen_ = 0;
#endif
#if HP_TAGDIST_MOD
            tagdist_ = 0;
#endif
#if HP_MARKDIST_MOD
            markdist_ = 0;
#endif
#if HP_UPPERGAP_MOD
            uppergap_ = 0;
#endif
#if HP_MONTH_MOD
            month_ = 0;
            mo_n_ = 0;
#endif
#if HP_GALLERY_MOD
            in_gallery_ = 0;
            gal_slash_ = 0;
            gal_n_ = 0;
#endif
#if HP_SECKIND_MOD
            seckind_ = 0;
            sk_skip_ = 0;
            sk_collect_ = 0;
            sk_n_ = 0;
#endif
#if HP_CITEKIND_MOD
            citekind_ = 0;
#endif
#if HP_CITEKIND_MOD || HP_COORD_MOD
            ck_i_ = 0;
            ck_done_ = 0;
#endif
#if HP_COLSPAN_MOD
            colspan_ = 0;
            cs_eat_ = 0;
            cs_val_ = 0;
            cs_win_ = 0;
#endif
#if HP_STYLE_MOD
            style_hash_ = 0;
            style_eat_ = 0;
            st_win_ = 0;
#endif
#if HP_COORD_MOD
            coord_ = 0;
#endif
#if HP_DIGITGAP_MOD
            digitgap_ = 0;
#endif
#if HP_DOTGAP_MOD
            dotgap_ = 0;
#endif
#if HP_COMMAGAP_MOD
            commagap_ = 0;
#endif
#if HP_WORDLEN_MOD
            wordlen_ = 0;
            wl_run_ = 0;
#endif
#if HP_SENTLEN_MOD
            sentlen_ = 0;
#endif
#if HP_LOWERGAP_MOD
            lowergap_ = 0;
#endif
#if HP_DIGITPOS_MOD
            digitpos_ = 0;
#endif
#if HP_SLASHGAP_MOD
            slashgap_ = 0;
#endif
#if HP_DIGLEN_MOD
            diglen_ = 0;
            dl_run_ = 0;
#endif
#if HP_PREVLINE_MOD
            prevline_ = 0;
            ll_run_ = 0;
#endif
#if HP_PREVSENT_MOD
            prevsent_ = 0;
            ss_run_ = 0;
#endif
#if HP_LINKLEN_MOD
            linklen_ = 0;
            lk_run_ = 0;
#endif
#if HP_TPLLEN_MOD
            tpllen_ = 0;
            tp_run_ = 0;
#endif
#if HP_PARALEN_MOD
            paralen_ = 0;
            pr_run_ = 0;
#endif
#if HP_ALNUMLEN_MOD
            alnumlen_ = 0;
            al_run_ = 0;
#endif
#if HP_SPLEN_MOD
            splen_ = 0;
            sp_run_ = 0;
#endif
#if HP_TITLEWORD_MOD || HP_HEADWORD_MOD || HP_REPEAT_MOD
            cw_acc_ = 0;
            cw_on_ = 0;
#endif
#if HP_TITLEWORD_MOD
            tw_n_ = 0;
            titleword_ = 0;
#endif
#if HP_HEADWORD_MOD
            hw_n_ = 0;
            headword_ = 0;
#endif
#if HP_INIT_MOD
            init_ = 0;
            init_st_ = 0;
#endif
#if HP_ORDINAL_MOD
            ordinal_ = 0;
            ord_pend_ = 0;
            ord_n_ = 0;
#endif
#if HP_UNIT_MOD
            unit_ = 0;
            un_pend_ = 0;
            un_n_ = 0;
#endif
#if HP_DECIMAL_MOD
            decimal_ = 0;
            dec_saw_ = 0;
            dec_pend_ = 0;
#endif
#if HP_REPEAT_MOD
            cw_prev_ = 0;
            repeat_ = 0;
#endif
#if HP_CASEFLIP_MOD
            caseflip_ = 0;
#endif
#if HP_LEAD_MOD
            lead_ = 1;
#endif
#if HP_INFOVAL_MOD
            infoval_ = 0;
            iv_tpl_ = 0;
            iv_set_ = 0;
#endif
#if HP_LINKTRAIL_MOD
            linktrail_ = 0;
            lt_on_ = 0;
#endif
#if HP_CELLKIND_MOD
            cellkind_ = 0;
            ck_sol_ = 0;
            ck_bar_ = 0;
#endif
#if HP_TBLCOL_MOD
            tblcol_ = 0;
#endif
#if HP_HEADIDX_MOD
            headidx_ = 0;
            hi_on_ = 0;
#endif
#if HP_HTMLFMT_MOD
            htmlfmt_ = 0;
            hf_n_ = 0;
            hf_slash_ = 0;
#endif
#if HP_INFOBOX_MOD
            infobox_ = 0;
            ib_depth_ = 0;
            ib_n_ = 0;
            ib_col_ = 0;
#endif
#if HP_SECLEVEL_MOD
            seclevel_ = 0;
            sl_run_ = 0;
            sl_at_ = 0;
#endif
#if HP_BRACE3_MOD
            brace3_ = 0;
            b3_run_ = 0;
            b3_cls_ = 0;
#endif
#if HP_NAMEDARG_MOD
            namedarg_ = 0;
#endif
#if HP_INCLUDE_MOD
            include_ = 0;
            ic_n_ = 0;
            ic_slash_ = 0;
#endif
#if HP_SIG_MOD
            sig_ = 0;
#endif
#if HP_WIKIBOLD_MOD
            wikibold_ = 0;
            wb_run_ = 0;
#endif
#if HP_URLPART_MOD
            urlpart_ = 0;
#endif
#if HP_REFIDX_MOD
            refidx_ = 0;
            ri_n_ = 0;
            ri_slash_ = 0;
#endif
        }
#endif
    }
    int xml_slash_ = 0;
    int xml_n_ = 0;
    int xml_done_ = 0;
    std::uint64_t xml_name_ = 0;
#endif
#if HP_TITLE_MOD
    int in_title_ = 0;
    std::uint64_t title_hash_ = 0;
#endif
#if HP_PAGEID_MOD
    int in_id_ = 0;
    int pageid_seen_ = 0;
    std::uint64_t page_id_ = 0;
#endif
#if HP_USER_MOD
    int in_user_ = 0;
    std::uint64_t user_hash_ = 0;
#endif
#if HP_TEXT_MOD
    int in_text_ = 0;
#endif
#if HP_NS_MOD
    int in_ns_ = 0;
    int ns_id_ = 0;
#endif
#if HP_DUMPREDIR_MOD
    int dump_redir_ = 0;
#endif
#if HP_IP_MOD
    int in_ip_ = 0;
    std::uint64_t ip_hash_ = 0;
#endif
#if HP_REVCOMMENT_MOD
    int in_comment_ = 0;
    std::uint64_t comment_hash_ = 0;
#endif
#if HP_MINOR_MOD
    int minor_ = 0;
#endif
#if HP_WIKIMODEL_MOD
    int in_model_ = 0;
    std::uint64_t model_hash_ = 0;
#endif
#if HP_SECTITLE_MOD
    int st_skip_ = 0;
    int st_collect_ = 0;
    std::uint64_t st_hash_ = 0;
#endif
#if HP_PARSERFN_MOD
    int pfn_wait_ = 0;
    int pfn_collect_ = 0;
    std::uint64_t pfn_hash_ = 0;
#endif
#if HP_TABLECLASS_MOD
    int tc_collect_ = 0;
    std::uint64_t tc_hash_ = 0;
#endif
#if HP_ANCHOR_MOD
    int an_collect_ = 0;
    std::uint64_t an_hash_ = 0;
#endif
#if HP_PUBID_MOD
    int pub_collect_ = 0;
    std::uint32_t pub_win_ = 0;
    std::uint64_t pub_hash_ = 0;
#endif
#if HP_TEMPPOS_MOD
    int tp_collect_ = 0;
    int tp_done_ = 0;
    std::uint64_t tp_hash_ = 0;
#endif
#if HP_TABLE_ABOVE || HP_FCCXT_MOD || HP_WIKISTACK_MOD
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
#endif
#if HP_WIKISTACK_MOD
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
#endif

#if HP_SECTION_MUTE
    // fx2 skipSeeExternal / isMath: derived, both sides see the same bytes.
    void push_mute(int c) {
        ring_[ring_n_ & 31] = static_cast<std::uint8_t>(c);
        ++ring_n_;
        if (line_len_ < 63) linebuf_[line_len_++] = static_cast<std::uint8_t>(c);

        if (ends_with_ci("&lt;math") || ends_with_ci("<math")) mute_ = 1;
        if (ends_with_ci("&lt;pre") || ends_with_ci("<pre")) mute_ = 1;
        if (ends_with_ci("&lt;nowiki") || ends_with_ci("<nowiki")) mute_ = 1;
        if (ends_with_ci("&lt;/math") || ends_with_ci("</math")) mute_ = 0;
        if (ends_with_ci("&lt;/pre") || ends_with_ci("</pre")) mute_ = 0;
        if (ends_with_ci("&lt;/nowiki") || ends_with_ci("</nowiki")) mute_ = 0;
        if (ends_with_ci("[[category:")) mute_ = 1;
        if (ends_with_ci("<page>") || ends_with_ci("</page>")) mute_ = 0;

        if (c == '\n') {
            if (is_skip_header(linebuf_, line_len_)) mute_ = 1;
            else if (is_any_header(linebuf_, line_len_)) mute_ = 0;
            line_len_ = 0;
        }
    }

    static int ci_eq(int b, int t) {
        if (b >= 'A' && b <= 'Z') b += 32;
        if (t >= 'A' && t <= 'Z') t += 32;
        return b == t;
    }

    bool ends_with_ci(const char* s) const {
        int n = 0;
        while (s[n]) ++n;
        if (ring_n_ < n) return false;
        for (int i = 0; i < n; ++i) {
            if (!ci_eq(ring_[(ring_n_ - n + i) & 31], static_cast<unsigned char>(s[i])))
                return false;
        }
        return true;
    }

    static bool is_any_header(const std::uint8_t* s, int n) {
        int i = 0;
        while (i < n && (s[i] == ' ' || s[i] == '\t')) ++i;
        return i + 1 < n && s[i] == '=' && s[i + 1] == '=';
    }

    static bool contains_ci(const std::uint8_t* s, int n, const char* pat) {
        int m = 0;
        while (pat[m]) ++m;
        for (int i = 0; i + m <= n; ++i) {
            int ok = 1;
            for (int j = 0; j < m; ++j) {
                if (!ci_eq(s[i + j], static_cast<unsigned char>(pat[j]))) {
                    ok = 0;
                    break;
                }
            }
            if (ok) return true;
        }
        return false;
    }

    static bool is_skip_header(const std::uint8_t* s, int n) {
        if (!is_any_header(s, n)) return false;
        return contains_ci(s, n, "references") || contains_ci(s, n, "see also") ||
               contains_ci(s, n, "bibliography") || contains_ci(s, n, "external link");
    }

    std::uint8_t ring_[32] = {};
    std::uint8_t linebuf_[64] = {};
    int ring_n_ = 0;
    int line_len_ = 0;
    int mute_ = 0;
#endif
};

}  // namespace hp
