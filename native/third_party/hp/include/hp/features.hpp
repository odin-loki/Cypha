#pragma once
//
// hp/features.hpp — compile-time ablation flags.
//
// Each ported model / mixer change is a switch so A.3 can accept or reject
// one axis at a time. Defaults are the high-confidence set from PLAN.md.
// Override with -DHP_WIKI_STATES=0 etc.
//
// NOTHING here is transmitted. Encoder and decoder are the same binary, so
// they compile the same set. Changing a flag produces a different archive
// family (kVersion bumped when the default set changes).

#ifndef HP_WIKI_STATES
#define HP_WIKI_STATES 1
#endif
#ifndef HP_WORD_MATCH
#define HP_WORD_MATCH 1
#endif
#ifndef HP_WORD_MATCH_N
#define HP_WORD_MATCH_N 3          // 1-, 2-, 3-word keys
#endif
#ifndef HP_WMATCH_4
#define HP_WMATCH_4 0              // extra word-match {current, last two}
#endif
#ifndef HP_WMATCH_5
#define HP_WMATCH_5 0              // extra word-match case-preserving current
#endif
#ifndef HP_WORD_STREAMS
#define HP_WORD_STREAMS 1          // first-char + prev only; case stream dropped
#endif
#ifndef HP_LINKWORD
#define HP_LINKWORD 1              // dedicated linkword/senword model
#endif
#ifndef HP_NUMERIC
#define HP_NUMERIC 1
#endif
#ifndef HP_PAT_MODEL
#define HP_PAT_MODEL 0             // A.3 reject: r=0.971 vs bracket
#endif
#ifndef HP_BRACKET
#define HP_BRACKET 1
#endif
#ifndef HP_PER_MIXER_LR
#define HP_PER_MIXER_LR 1
#endif
#ifndef HP_PATTERN_CACHE
#define HP_PATTERN_CACHE 1
#endif
#ifndef HP_META_PATTERNS
#define HP_META_PATTERNS 1
#endif
#ifndef HP_NCL
#define HP_NCL 1
#endif
#ifndef HP_NCL_LAMBDA
#define HP_NCL_LAMBDA 4            // Q8; 4/256 ≈ 0.016
#endif
#ifndef HP_HEDGE_L1
#define HP_HEDGE_L1 1
#endif
#ifndef HP_CTW
#define HP_CTW 0                  // A.3 reject: mean|r|=0.796, twin o2:py 0.930
#endif
#ifndef HP_ENGLISH_PRIOR
#define HP_ENGLISH_PRIOR 1
#endif
#ifndef HP_PPMD
#define HP_PPMD 0                  // D3: default off, likely saturated
#endif
#ifndef HP_NIBBLE_BUCKETS
#define HP_NIBBLE_BUCKETS 0
#endif
#ifndef HP_STATE_CAP
#define HP_STATE_CAP 20
#endif
#ifndef HP_DISC_SLOTS
#define HP_DISC_SLOTS 12
#endif
#ifndef HP_DISC_EVAL
#define HP_DISC_EVAL 1024
#endif
#ifndef HP_EXTRA_GATES
#define HP_EXTRA_GATES 1           // wiki + pattern-class gates
#endif
#ifndef HP_SECTION_MUTE
#define HP_SECTION_MUTE 0          // H2.2: mute word models in refs/math
#endif
#ifndef HP_MIXER_NLMS
#define HP_MIXER_NLMS 0            // 1 = normalise layer-1 LMS step by input energy (NLMS)
#endif

#ifndef HP_NLMS_ETYP
#define HP_NLMS_ETYP 20000000      // reference input energy; step scales ETYP/||st||^2
#endif

#ifndef HP_NLMS_EPS
#define HP_NLMS_EPS 65536          // regulariser, keeps the divisor away from 0
#endif

#ifndef HP_LR1_SCALE
#define HP_LR1_SCALE 100           // percent scale on hardcoded layer-1 per-mixer rates (100 = identity)
#endif

#ifndef HP_MIXER_BACKPROP
#define HP_MIXER_BACKPROP 0        // 1 = train layer-1 on backpropagated final error, not local error
#endif

#ifndef HP_MIXER_SKIP
#define HP_MIXER_SKIP 0            // H3.1: skip mixer update if |err| < this
#endif
#ifndef HP_SPARSE_UTF8
#define HP_SPARSE_UTF8 0           // H2.3: gap-1 match for escaped UTF-8
#endif
#ifndef HP_STEMMER
#define HP_STEMMER 0               // H2.1: Porter2 + POS as extra experts
#endif
#ifndef HP_STEMMER_N
#define HP_STEMMER_N 2
#endif
#ifndef HP_STEM_FOLD
#define HP_STEM_FOLD 0             // H2.1b: fold stem hashes into word_/wstr_
#endif
#ifndef HP_SENWORD
#define HP_SENWORD 0               // H2.5: senword own expert (linkword stays)
#endif
#ifndef HP_PRED_GATE
#define HP_PRED_GATE 0             // H2.7: idle word models in table/link/{}
#endif
#ifndef HP_POS_GATE
#define HP_POS_GATE 0              // H2.1c: mixer gate from stem POS type
#endif
#ifndef HP_GATE_SHAPE
#define HP_GATE_SHAPE 0            // st_shape6 mixer gate
#endif
#ifndef HP_GATE_BRANCH
#define HP_GATE_BRANCH 0           // rec_branch3 mixer gate
#endif
#ifndef HP_GATE_DISP
#define HP_GATE_DISP 0             // disp_var mixer gate
#endif
#ifndef HP_GATE_MLEN2
#define HP_GATE_MLEN2 0            // second-longest match mixer gate
#endif
#ifndef HP_GATE_ARGMAX
#define HP_GATE_ARGMAX 0           // loudest-expert mixer gate
#endif
#ifndef HP_SEN_GROUP
#define HP_SEN_GROUP 0             // prose/list/table/link mixer gate
#endif
#ifndef HP_FIRST_WORD
#define HP_FIRST_WORD 0            // fold sentence first-word into wstr
#endif
#ifndef HP_LINK_NUM
#define HP_LINK_NUM 0              // fold number0 into linkword context
#endif
#ifndef HP_GATE_BREAK
#define HP_GATE_BREAK 0            // match-tenure mixer gate
#endif
#ifndef HP_SLOT_SIZES
#define HP_SLOT_SIZES 0            // H2.9: per-model table bit offsets
#endif
#ifndef HP_SLOT_GROW
#define HP_SLOT_GROW 0             // grow tag/word only; do not shrink o1/o2
#endif
#ifndef HP_STATE_TABLE2
#define HP_STATE_TABLE2 0          // H2.10: PY uses capped n0/n1
#endif
#ifndef HP_GATE_WORDPOS
#define HP_GATE_WORDPOS 0          // in-word position mixer gate
#endif
#ifndef HP_GATE_HEDGE
#define HP_GATE_HEDGE 0            // max hedge weight mixer gate
#endif
#ifndef HP_SENT_RECENCY
#define HP_SENT_RECENCY 0          // prev-sentence word at same offset into word_
#endif
#ifndef HP_WT3_CTX
#define HP_WT3_CTX 0               // rolling POS-bucket trail into wstr
#endif
#ifndef HP_GATE_FWORD
#define HP_GATE_FWORD 0            // sentence first-word mixer gate (not fold)
#endif
#ifndef HP_SLOT_GROW_EXTRA
#define HP_SLOT_GROW_EXTRA 0       // tag/link/sen +3 instead of +2
#endif
#ifndef HP_WIKI_TEMP
#define HP_WIKI_TEMP 0             // {name|...} template bit in tag_
#endif
#ifndef HP_SENT_STREAM
#define HP_SENT_STREAM 0           // sentence-boundary word as own expert
#endif
#ifndef HP_SENT_MEM
#define HP_SENT_MEM 0              // similar-sentence memory expert
#endif
#ifndef HP_SENT_MEM_BIG
#define HP_SENT_MEM_BIG 0          // 128-slot / 32-word sentence ring
#endif
#ifndef HP_SENT_ALIGN
#define HP_SENT_ALIGN 0            // ctx = matched sentence word at index
#endif
#ifndef HP_SLOT_SMEM
#define HP_SLOT_SMEM 0             // grow sentmem table +2
#endif
#ifndef HP_SENT_DOM
#define HP_SENT_DOM 0              // prose/table/link/template sentence rings
#endif
#ifndef HP_UTF8_IDLE
#define HP_UTF8_IDLE 0             // idle word models on UTF-8 continuation
#endif
#ifndef HP_MATCH_18
#define HP_MATCH_18 0              // extra byte-match order 8
#endif
#ifndef HP_TABLE_ABOVE
#define HP_TABLE_ABOVE 0           // cell-aligned byte from row above
#endif
#ifndef HP_NLCHAR
#define HP_NLCHAR 0                // LF / table / header line-mode nibble
#endif
#ifndef HP_QUOTE_STACK
#define HP_QUOTE_STACK 0           // '' / "" toggle into bracket ctx
#endif
#ifndef HP_BRK_CLOSE
#define HP_BRK_CLOSE 0             // expected closer byte into bracket ctx
#endif
#ifndef HP_GATE_UTF8
#define HP_GATE_UTF8 0             // UTF-8 leftover mixer gate (no idle)
#endif
#ifndef HP_GATE_NEST
#define HP_GATE_NEST 0             // markup-nest mixer gate
#endif
#ifndef HP_GATE_AGREE
#define HP_GATE_AGREE 0            // expert-agreement mixer gate
#endif
#ifndef HP_GATE_FCLASS
#define HP_GATE_FCLASS 0           // first-char-class mixer gate
#endif
#ifndef HP_WBI_SENTPOS
#define HP_WBI_SENTPOS 0           // sentence word index into wbi_
#endif
#ifndef HP_GATE_WMLEN
#define HP_GATE_WMLEN 0            // longest word-match mixer gate
#endif
#ifndef HP_SLOT_WORD2
#define HP_SLOT_WORD2 0            // word/wbi +2 instead of +1
#endif
#ifndef HP_SLOT_COL2
#define HP_SLOT_COL2 0             // grow col_ +2
#endif
#ifndef HP_SLOT_COL3
#define HP_SLOT_COL3 0             // grow col_ +1 extra bit via add_bits
#endif
#ifndef HP_MATCH_13
#define HP_MATCH_13 0              // extra byte-match order 13
#endif
#ifndef HP_MATCH_01
#define HP_MATCH_01 0              // extra byte-match order 1
#endif
#ifndef HP_MATCH_02
#define HP_MATCH_02 0              // extra byte-match order 2
#endif
#ifndef HP_MATCH_05
#define HP_MATCH_05 0              // extra byte-match order 5
#endif
#ifndef HP_MATCH_07
#define HP_MATCH_07 0              // extra byte-match order 7
#endif
#ifndef HP_MATCH_09
#define HP_MATCH_09 0              // extra byte-match order 9
#endif
#ifndef HP_MATCH_12
#define HP_MATCH_12 0              // extra byte-match order 12
#endif
#ifndef HP_MATCH_20
#define HP_MATCH_20 0              // extra byte-match order 20
#endif
#ifndef HP_SLOT_NUM2
#define HP_SLOT_NUM2 0             // grow numeric +2
#endif
#ifndef HP_SLOT_WORD3
#define HP_SLOT_WORD3 0            // word/wbi +3 (on top of WORD2)
#endif
#ifndef HP_SLOT_WORD4
#define HP_SLOT_WORD4 0            // word/wbi +4
#endif
#ifndef HP_SLOT_WORD5
#define HP_SLOT_WORD5 0            // word/wbi +5
#endif
#ifndef HP_SLOT_WORD6
#define HP_SLOT_WORD6 0            // word/wbi +6
#endif
#ifndef HP_SLOT_WORD7
#define HP_SLOT_WORD7 0            // word/wbi +7 (needs HP_SLOT_MAX>=29)
#endif
#ifndef HP_SLOT_WORD8
#define HP_SLOT_WORD8 0            // word/wbi +8 (needs HP_SLOT_MAX>=30)
#endif
#ifndef HP_SLOT_WORD9
#define HP_SLOT_WORD9 0            // word/wbi +9 (needs HP_SLOT_MAX>=31)
#endif
#ifndef HP_SLOT_WORD10
#define HP_SLOT_WORD10 0           // word/wbi +10 (needs HP_SLOT_MAX>=32)
#endif
#ifndef HP_SLOT_MAX
#define HP_SLOT_MAX 28             // per-model table-bit cap
#endif
#ifndef HP_SLOT_S3
#define HP_SLOT_S3 0               // grow sent-stream +2
#endif
#ifndef HP_SLOT_S4
#define HP_SLOT_S4 0               // grow sent-stream +3 (on top of S3)
#endif
#ifndef HP_SLOT_WSTR2
#define HP_SLOT_WSTR2 0            // grow wstr +2
#endif
#ifndef HP_SLOT_BRK2
#define HP_SLOT_BRK2 0             // grow bracket +2
#endif
#ifndef HP_SLOT_O34
#define HP_SLOT_O34 0              // grow o3/o4 +1 (not via delta==1 word path)
#endif
#ifndef HP_SLOT_O34B
#define HP_SLOT_O34B 0             // grow o3/o4 one more bit on top of O34
#endif
#ifndef HP_SLOT_O6
#define HP_SLOT_O6 0               // grow o6 +1
#endif
#ifndef HP_SLOT_O6B
#define HP_SLOT_O6B 0              // grow o6 one more bit on top of O6
#endif
#ifndef HP_SLOT_O6C
#define HP_SLOT_O6C 0              // grow o6 a third extra bit
#endif
#ifndef HP_SLOT_O6D
#define HP_SLOT_O6D 0              // grow o6 a fourth extra bit
#endif
#ifndef HP_SLOT_O6E
#define HP_SLOT_O6E 0              // grow o6 a fifth extra bit
#endif
#ifndef HP_SLOT_O6F
#define HP_SLOT_O6F 0              // grow o6 a sixth extra bit (needs HP_SLOT_MAX>=32 at mem 26)
#endif
#ifndef HP_SLOT_O6G
#define HP_SLOT_O6G 0              // grow o6 a seventh extra bit (needs HP_SLOT_MAX>=33 at mem 26)
#endif
#ifndef HP_SLOT_O6H
#define HP_SLOT_O6H 0              // grow o6 an eighth extra bit (needs HP_SLOT_MAX>=34 at mem 26)
#endif
#ifndef HP_SLOT_O6I
#define HP_SLOT_O6I 0              // grow o6 a ninth extra bit (needs HP_SLOT_MAX>=35 at mem 26)
#endif
#ifndef HP_SLOT_O6J
#define HP_SLOT_O6J 0              // grow o6 a tenth extra bit (needs HP_SLOT_MAX>=36 at mem 26)
#endif
#ifndef HP_SLOT_O34C
#define HP_SLOT_O34C 0             // grow o3/o4 a third bit
#endif
#ifndef HP_SLOT_O34D
#define HP_SLOT_O34D 0             // grow o3/o4 a fourth bit
#endif
#ifndef HP_SLOT_O34E
#define HP_SLOT_O34E 0             // grow o3/o4 a fifth bit
#endif
#ifndef HP_SLOT_O34F
#define HP_SLOT_O34F 0             // grow o3/o4 a sixth bit (needs HP_SLOT_MAX>=32 at mem 26)
#endif
#ifndef HP_SLOT_O34G
#define HP_SLOT_O34G 0             // grow o3/o4 a seventh bit
#endif
#ifndef HP_SLOT_O12
#define HP_SLOT_O12 0              // grow o1/o2 +1 (they sit at delta 0 under SLOT_GROW)
#endif
#ifndef HP_SLOT_SP
#define HP_SLOT_SP 0               // grow sparse 1-3 / 2-4 +1
#endif
#ifndef HP_MATCH_GROW
#define HP_MATCH_GROW 0            // match hash tables +1 bit
#endif
#ifndef HP_MATCH_GROW2
#define HP_MATCH_GROW2 0           // match hash tables +2 bits
#endif
#ifndef HP_SENT_CUR
#define HP_SENT_CUR 0              // fold current-sentence hash into sentmem ctx
#endif
#ifndef HP_SENT_GRP_CTX
#define HP_SENT_GRP_CTX 0          // fold wiki sen_group into sent-stream ctx
#endif
#ifndef HP_WSTR_GRP
#define HP_WSTR_GRP 0              // fold wiki sen_group into wstr ctx
#endif
#ifndef HP_WORD_GRP
#define HP_WORD_GRP 0              // fold wiki sen_group into word_ ctx
#endif
#ifndef HP_WBI_GRP
#define HP_WBI_GRP 0               // fold wiki sen_group into wbi_ ctx
#endif
#ifndef HP_SMEM_GRP
#define HP_SMEM_GRP 0              // fold wiki sen_group into sentmem ctx
#endif
#ifndef HP_SENGRP_MOD
#define HP_SENGRP_MOD 0            // dedicated sen_group context model
#endif
#ifndef HP_SLOT_SGRP
#define HP_SLOT_SGRP 0             // grow sengrp table +2
#endif
#ifndef HP_SENGRP_WORD
#define HP_SENGRP_WORD 0           // fold word hash into sengrp ctx
#endif
#ifndef HP_SENGRP_POS
#define HP_SENGRP_POS 0            // fold sentence position into sengrp ctx
#endif
#ifndef HP_COL_GRP
#define HP_COL_GRP 0               // fold wiki sen_group into col_
#endif
#ifndef HP_TAG_GRP
#define HP_TAG_GRP 0               // fold wiki sen_group into tag_
#endif
#ifndef HP_SENWORD_GRP
#define HP_SENWORD_GRP 0           // fold wiki sen_group into sen_ (not mixer HP_SEN_GROUP)
#endif
#ifndef HP_LINK_GRP
#define HP_LINK_GRP 0              // fold wiki sen_group into link_
#endif
#ifndef HP_BRK_GRP
#define HP_BRK_GRP 0               // fold wiki sen_group into bracket ctx
#endif
#ifndef HP_NUM_GRP
#define HP_NUM_GRP 0               // fold wiki sen_group into num_ ctx
#endif
#ifndef HP_SP_GRP
#define HP_SP_GRP 0                // fold wiki sen_group into sp13_ and sp24_
#endif
#ifndef HP_SENGRP_C0
#define HP_SENGRP_C0 0             // fold c0_ into sengrp ctx (elif after WORD/POS)
#endif
#ifndef HP_HEBB_GRP
#define HP_HEBB_GRP 0              // fold wiki sen_group into hebb context
#endif
#ifndef HP_NEST_MOD
#define HP_NEST_MOD 0              // dedicated nest-markup context model
#endif
#ifndef HP_PARA_MOD
#define HP_PARA_MOD 0              // dedicated paragraph/FIRSTUPPER context model
#endif
#ifndef HP_LINE_MOD
#define HP_LINE_MOD 0              // dedicated first-of-line char context model
#endif
#ifndef HP_XSIMD
#define HP_XSIMD 1                 // integer mixer dots via xsimd/SSE4.1; bit-identical to scalar
#endif
#ifndef HP_TRACK_EXP_P
#define HP_TRACK_EXP_P (!HP_HEDGE_L1 || HP_CTW || HP_GATE_DISP || HP_GATE_ARGMAX || HP_GATE_AGREE)
#endif
#ifndef HP_PATTERN_CACHE_STATS
#define HP_PATTERN_CACHE_STATS 0   // lookups/hits/class histogram (diagnostics only)
#endif
#ifndef HP_WIKI_AXES
#define HP_WIKI_AXES 0             // bundle: state + sent_domain + header + depth CMs
#endif
#ifndef HP_STATE_MOD
#define HP_STATE_MOD HP_WIKI_AXES  // dedicated wiki.state() CM
#endif
#ifndef HP_DOM_MOD
#define HP_DOM_MOD HP_WIKI_AXES    // dedicated sent_domain CM (body/table/link/curly)
#endif
#ifndef HP_HDR_MOD
#define HP_HDR_MOD HP_WIKI_AXES    // dedicated wiki_header() CM
#endif
#ifndef HP_DEPTH_MOD
#define HP_DEPTH_MOD HP_WIKI_AXES  // dedicated wiki.depth() CM
#endif
#ifndef HP_MIXER_RANK
#define HP_MIXER_RANK 0            // 0 = full W; >0 = U(ctx)×V shared rank for layer-1
#endif
#ifndef HP_MIXER_CLAMP_BITS
#define HP_MIXER_CLAMP_BITS 16     // ±(1<<N); 16 is byte-identical on proxy corpora
#endif
#ifndef HP_MIXER_W16
#define HP_MIXER_W16 0             // int16 packed weights: ratio cost on proxies
#endif
#ifndef HP_MIXER_ST16
#define HP_MIXER_ST16 1            // expert stretches as int16 in mixer buffer
#endif
#ifndef HP_HEDGE_W16
#define HP_HEDGE_W16 1             // hedge gate weights as uint16 Q16
#endif
#ifndef HP_BUF_DELTA
#define HP_BUF_DELTA 3             // buf_bits = table_bits + delta; 3 is byte-identical on proxies
#endif
#ifndef HP_PY_EXPERT
#define HP_PY_EXPERT 1             // 0 = drop Pitman-Yor mixer input (kOutputs=1)
#endif
#ifndef HP_FCCXT_MOD
#define HP_FCCXT_MOD 0             // first-char-of-cell as own CM (not col_ fold)
#endif
#ifndef HP_TPLNAME_MOD
#define HP_TPLNAME_MOD 0           // {{TemplateName hash CM
#endif
#ifndef HP_INFOKEY_MOD
#define HP_INFOKEY_MOD 0           // infobox |key= value CM
#endif
#ifndef HP_BARIDX_MOD
#define HP_BARIDX_MOD 0            // nth | field index CM
#endif
#ifndef HP_PERIOD_MOD
#define HP_PERIOD_MOD 0            // line/table stride period CM
#endif
#ifndef HP_PRONOUN_MOD
#define HP_PRONOUN_MOD 0           // pronoun-ring CM
#endif
#ifndef HP_DMC_MOD
#define HP_DMC_MOD 0               // small DMC graph expert
#endif
#ifndef HP_LZP_MOD
#define HP_LZP_MOD 0               // last-occurrence byte predictor
#endif
#ifndef HP_SR_MOD
#define HP_SR_MOD 0                // recency-rank (MTF) CM
#endif
#ifndef HP_SKIPK_MOD
#define HP_SKIPK_MOD 0             // skip-2 sparse match
#endif
#ifndef HP_HASH_CHK
#define HP_HASH_CHK 0              // dual-stage hash: index + 8-bit checksum, 3-probe
#endif
#ifndef HP_HASH2_O6
#define HP_HASH2_O6 0              // second salt of o6 as extra CM (twin risk)
#endif
#ifndef HP_HASH_P5
#define HP_HASH_P5 0               // HASH_CHK 5-probe instead of 3
#endif
#ifndef HP_SKIP3_MOD
#define HP_SKIP3_MOD 0             // skip-3 sparse MatchModel
#endif
#ifndef HP_SKIP4_MOD
#define HP_SKIP4_MOD 0             // skip-4 sparse MatchModel
#endif
#ifndef HP_SKIP5_MOD
#define HP_SKIP5_MOD 0             // skip-5 sparse MatchModel
#endif
#ifndef HP_LINKPIPE_MOD
#define HP_LINKPIPE_MOD 0          // [[target|display]] after-pipe CM
#endif
#ifndef HP_CITE_MOD
#define HP_CITE_MOD 0              // <ref>…</ref> domain CM
#endif
#ifndef HP_DMC_GROW
#define HP_DMC_GROW 0              // DMC cap 20 bits instead of 18
#endif
#ifndef HP_CAT_MOD
#define HP_CAT_MOD 0               // [[Category: / File: / Image: namespace CM
#endif
#ifndef HP_REDIR_MOD
#define HP_REDIR_MOD 0             // #REDIRECT domain CM
#endif
#ifndef HP_HEADING_MOD
#define HP_HEADING_MOD 0           // leading '=' count (0–6) CM
#endif
#ifndef HP_EXTLINK_MOD
#define HP_EXTLINK_MOD 0           // [http…] single-bracket external-link CM
#endif
#ifndef HP_REFNAME_MOD
#define HP_REFNAME_MOD 0           // <ref name="…"> name hash CM
#endif
#ifndef HP_QOCXT_MOD
#define HP_QOCXT_MOD 0             // dedicated '' / ''' / " quote-stack CM
#endif
#ifndef HP_ENTITY_MOD
#define HP_ENTITY_MOD 0            // &nbsp; / &lt; HTML-entity name CM
#endif
#ifndef HP_INDENT_MOD
#define HP_INDENT_MOD 0            // leading ':' count (talk / dl) CM
#endif
#ifndef HP_LISTLEVEL_MOD
#define HP_LISTLEVEL_MOD 0         // leading '*' / '#' count CM
#endif
#ifndef HP_ISSE_MOD
#define HP_ISSE_MOD 0              // extra CM of hist + o6 p-bucket
#endif
#ifndef HP_MAGIC_MOD
#define HP_MAGIC_MOD 0             // __TOC__ / __NOTOC__ / __NOEDITSECTION__ CM
#endif
#ifndef HP_NOWIKI_MOD
#define HP_NOWIKI_MOD 0            // <nowiki> / <math> / <pre> / <code> domain CM
#endif
#ifndef HP_TITLE_MOD
#define HP_TITLE_MOD 0             // <title>…</title> page-title hash CM
#endif
#ifndef HP_PAGEID_MOD
#define HP_PAGEID_MOD 0            // first <id> after <page> CM
#endif
#ifndef HP_USER_MOD
#define HP_USER_MOD 0              // <username> contributor hash CM
#endif
#ifndef HP_TEXT_MOD
#define HP_TEXT_MOD 0              // inside <text> dump payload domain CM
#endif
#ifndef HP_NS_MOD
#define HP_NS_MOD 0                // dump <ns> namespace id CM
#endif
#ifndef HP_DUMPREDIR_MOD
#define HP_DUMPREDIR_MOD 0         // dump <redirect …/> tag CM (not #REDIRECT)
#endif
#ifndef HP_IP_MOD
#define HP_IP_MOD 0                // dump <ip> anonymous-editor hash CM
#endif
#ifndef HP_REVCOMMENT_MOD
#define HP_REVCOMMENT_MOD 0        // dump <comment> revision-summary hash CM
#endif
#ifndef HP_MINOR_MOD
#define HP_MINOR_MOD 0             // dump <minor/> edit bit CM
#endif
#ifndef HP_WIKIMODEL_MOD
#define HP_WIKIMODEL_MOD 0         // dump <model> wikitext/javascript CM
#endif
#ifndef HP_SECTITLE_MOD
#define HP_SECTITLE_MOD 0          // heading-body text hash (not '=' count)
#endif
#ifndef HP_PARSERFN_MOD
#define HP_PARSERFN_MOD 0          // {{#if / #switch / #expr parser-function CM
#endif
#ifndef HP_TABLECLASS_MOD
#define HP_TABLECLASS_MOD 0        // {| class=… first-line token CM
#endif
#ifndef HP_ANCHOR_MOD
#define HP_ANCHOR_MOD 0            // [[target#section fragment CM
#endif
#ifndef HP_PUBID_MOD
#define HP_PUBID_MOD 0             // ISBN / PMID digit-run CM
#endif
#ifndef HP_TEMPPOS_MOD
#define HP_TEMPPOS_MOD 0           // first positional {{template|arg CM
#endif
#ifndef HP_WIKISTACK_MOD
#define HP_WIKISTACK_MOD 0         // fccxt+brcxt+cell-above packed CM (richer than FCCXT)
#endif
#ifndef HP_REORDER
#define HP_REORDER 0               // sort <page> blocks by <title>
#endif
#ifndef HP_PAYLOAD_LEX
#define HP_PAYLOAD_LEX 0           // sort <page> blocks by <text> payload
#endif
#ifndef HP_LANG_MOD
#define HP_LANG_MOD 0              // [[xx: interwiki / lang prefix CM
#endif
#ifndef HP_CATSORT_MOD
#define HP_CATSORT_MOD 0           // [[Category:Name|sortkey CM
#endif
#ifndef HP_TBLROW_MOD
#define HP_TBLROW_MOD 0            // {| row/caption/header/cell kind CM
#endif
#ifndef HP_FILEOPT_MOD
#define HP_FILEOPT_MOD 0           // [[File: thumb/px/right option CM
#endif
#ifndef HP_DEFAULTSORT_MOD
#define HP_DEFAULTSORT_MOD 0       // {{DEFAULTSORT: key CM
#endif
#ifndef HP_REDIRTARGET_MOD
#define HP_REDIRTARGET_MOD 0       // #REDIRECT [[target]] title hash CM
#endif
#ifndef HP_DAB_MOD
#define HP_DAB_MOD 0               // {{disambig / hndis / dab CM
#endif
#ifndef HP_HATNOTE_MOD
#define HP_HATNOTE_MOD 0           // {{for| / {{about| / {{main| CM
#endif
#ifndef HP_LASTLINK_MOD
#define HP_LASTLINK_MOD 0          // sticky last [[target]] entity CM
#endif
#ifndef HP_FWORD_MOD
#define HP_FWORD_MOD 0             // sentence first-word CM (fxcm fword)
#endif
#ifndef HP_YEAR_MOD
#define HP_YEAR_MOD 0              // last 4-digit year 1000-2099 CM
#endif
#ifndef HP_CAPMASK_MOD
#define HP_CAPMASK_MOD 0           // current-word capitalisation mask CM
#endif
#ifndef HP_CELLTXT_MOD
#define HP_CELLTXT_MOD 0           // current table-cell text hash CM
#endif
#ifndef HP_HTTPHOST_MOD
#define HP_HTTPHOST_MOD 0          // http(s) URL hostname hash CM
#endif
#ifndef HP_PAREN_MOD
#define HP_PAREN_MOD 0             // sticky last (...) group hash CM
#endif
#ifndef HP_LISTPOS_MOD
#define HP_LISTPOS_MOD 0           // nth * / # list item CM
#endif
#ifndef HP_SHAPE_MOD
#define HP_SHAPE_MOD 0             // 2-bit-per-char word shape sequence CM
#endif
#ifndef HP_SUFFIX_MOD
#define HP_SUFFIX_MOD 0            // last-3-letters of word CM
#endif
#ifndef HP_PREFIX_MOD
#define HP_PREFIX_MOD 0            // first-3-letters of word CM
#endif
#ifndef HP_CHARCLS_MOD
#define HP_CHARCLS_MOD 0           // rolling byte-class stream CM
#endif
#ifndef HP_VOWEL_MOD
#define HP_VOWEL_MOD 0             // vowel/consonant bit-ring of word CM
#endif
#ifndef HP_CONTR_MOD
#define HP_CONTR_MOD 0             // internal-apostrophe contraction CM
#endif
#ifndef HP_HYPHEN_MOD
#define HP_HYPHEN_MOD 0            // hyphenated-compound hash CM
#endif
#ifndef HP_TOKENCLS_MOD
#define HP_TOKENCLS_MOD 0          // alpha/digit/mixed/punct/xml/wiki token CM
#endif
#ifndef HP_RUNLEN_MOD
#define HP_RUNLEN_MOD 0            // identical-byte run length CM
#endif
#ifndef HP_WPOS_MOD
#define HP_WPOS_MOD 0              // letter index in current word CM
#endif
#ifndef HP_BLANK_MOD
#define HP_BLANK_MOD 0             // consecutive newline / blank-line CM
#endif
#ifndef HP_SPRUN_MOD
#define HP_SPRUN_MOD 0             // space-run length CM
#endif
#ifndef HP_LINELEN_MOD
#define HP_LINELEN_MOD 0           // current line length CM
#endif
#ifndef HP_TAGDIST_MOD
#define HP_TAGDIST_MOD 0           // bytes since last '<' CM
#endif
#ifndef HP_MARKDIST_MOD
#define HP_MARKDIST_MOD 0          // bytes since last wiki markup char CM
#endif
#ifndef HP_UPPERGAP_MOD
#define HP_UPPERGAP_MOD 0          // bytes since last uppercase CM
#endif
#ifndef HP_MONTH_MOD
#define HP_MONTH_MOD 0             // last English month-name CM
#endif
#ifndef HP_GALLERY_MOD
#define HP_GALLERY_MOD 0           // <gallery> domain CM
#endif
#ifndef HP_SECKIND_MOD
#define HP_SECKIND_MOD 0           // classified heading kind CM
#endif
#ifndef HP_CITEKIND_MOD
#define HP_CITEKIND_MOD 0          // {{cite web/journal/book/news CM
#endif
#ifndef HP_TAGNAME_MOD
#define HP_TAGNAME_MOD 0           // current HTML/XML tag-name hash CM
#endif
#ifndef HP_COLSPAN_MOD
#define HP_COLSPAN_MOD 0           // colspan=/rowspan= value CM
#endif
#ifndef HP_STYLE_MOD
#define HP_STYLE_MOD 0             // style= CSS fragment CM
#endif
#ifndef HP_COORD_MOD
#define HP_COORD_MOD 0             // {{coord / {{Coord CM
#endif
#ifndef HP_DIGITGAP_MOD
#define HP_DIGITGAP_MOD 0          // bytes since last digit CM
#endif
#ifndef HP_DOTGAP_MOD
#define HP_DOTGAP_MOD 0            // bytes since last '.' CM
#endif
#ifndef HP_COMMAGAP_MOD
#define HP_COMMAGAP_MOD 0          // bytes since last ',' CM
#endif
#ifndef HP_WORDLEN_MOD
#define HP_WORDLEN_MOD 0           // last completed letter-word length CM
#endif
#ifndef HP_SENTLEN_MOD
#define HP_SENTLEN_MOD 0           // bytes since last .?! CM
#endif
#ifndef HP_LOWERGAP_MOD
#define HP_LOWERGAP_MOD 0          // bytes since last a-z CM
#endif
#ifndef HP_DIGITPOS_MOD
#define HP_DIGITPOS_MOD 0          // index in current digit run CM
#endif
#ifndef HP_SLASHGAP_MOD
#define HP_SLASHGAP_MOD 0          // bytes since last '/' CM
#endif
#ifndef HP_DIGLEN_MOD
#define HP_DIGLEN_MOD 0            // last completed digit-run length CM
#endif
#ifndef HP_PREVLINE_MOD
#define HP_PREVLINE_MOD 0          // last completed line length CM
#endif
#ifndef HP_PREVSENT_MOD
#define HP_PREVSENT_MOD 0          // last completed sentence length CM
#endif
#ifndef HP_LINKLEN_MOD
#define HP_LINKLEN_MOD 0           // last completed [[link]] length CM
#endif
#ifndef HP_TPLLEN_MOD
#define HP_TPLLEN_MOD 0            // last completed {{template}} length CM
#endif
#ifndef HP_PARALEN_MOD
#define HP_PARALEN_MOD 0           // last completed paragraph length CM
#endif
#ifndef HP_ALNUMLEN_MOD
#define HP_ALNUMLEN_MOD 0          // last completed alnum-token length CM
#endif
#ifndef HP_SPLEN_MOD
#define HP_SPLEN_MOD 0             // last completed space-run length CM
#endif
#ifndef HP_TITLEWORD_MOD
#define HP_TITLEWORD_MOD 0         // current word hits page-title token CM
#endif
#ifndef HP_HEADWORD_MOD
#define HP_HEADWORD_MOD 0          // current word hits last-heading token CM
#endif
#ifndef HP_INIT_MOD
#define HP_INIT_MOD 0              // letter-dot initials (U.S. / J.) CM
#endif
#ifndef HP_ORDINAL_MOD
#define HP_ORDINAL_MOD 0           // 1st/2nd/3rd/4th after digits CM
#endif
#ifndef HP_UNIT_MOD
#define HP_UNIT_MOD 0              // km/mi/kg unit after number CM
#endif
#ifndef HP_DECIMAL_MOD
#define HP_DECIMAL_MOD 0           // digit.digit decimal-number CM
#endif
#ifndef HP_REPEAT_MOD
#define HP_REPEAT_MOD 0            // current word repeats previous CM
#endif
#ifndef HP_CASEFLIP_MOD
#define HP_CASEFLIP_MOD 0          // bytes since lower-to-upper flip CM
#endif
#ifndef HP_LEAD_MOD
#define HP_LEAD_MOD 0              // before first heading (article lead) CM
#endif
#ifndef HP_INFOVAL_MOD
#define HP_INFOVAL_MOD 0           // template/infobox value class after = CM
#endif
#ifndef HP_LINKTRAIL_MOD
#define HP_LINKTRAIL_MOD 0         // letters immediately after ]] CM
#endif
#ifndef HP_CELLKIND_MOD
#define HP_CELLKIND_MOD 0          // wiki table caption/header/data/rowsep CM
#endif
#ifndef HP_TBLCOL_MOD
#define HP_TBLCOL_MOD 0            // column index in current table row CM
#endif
#ifndef HP_HEADIDX_MOD
#define HP_HEADIDX_MOD 0           // nth heading on the page CM
#endif
#ifndef HP_HTMLFMT_MOD
#define HP_HTMLFMT_MOD 0           // open inline HTML formatting bitmask CM
#endif
#ifndef HP_INFOBOX_MOD
#define HP_INFOBOX_MOD 0           // inside {{Infobox...}} region CM
#endif
#ifndef HP_SECLEVEL_MOD
#define HP_SECLEVEL_MOD 0          // sticky heading level of section body CM
#endif
#ifndef HP_BRACE3_MOD
#define HP_BRACE3_MOD 0            // {{{template-param}}} depth CM
#endif
#ifndef HP_NAMEDARG_MOD
#define HP_NAMEDARG_MOD 0          // named template arg body after = CM
#endif
#ifndef HP_INCLUDE_MOD
#define HP_INCLUDE_MOD 0           // includeonly/noinclude/onlyinclude CM
#endif
#ifndef HP_SIG_MOD
#define HP_SIG_MOD 0               // ~~~~ signature tilde-run CM
#endif
#ifndef HP_WIKIBOLD_MOD
#define HP_WIKIBOLD_MOD 0          // wiki '' / ''' / ''''' bold-italic CM
#endif
#ifndef HP_URLPART_MOD
#define HP_URLPART_MOD 0           // URL host/path/query/fragment CM
#endif
#ifndef HP_REFIDX_MOD
#define HP_REFIDX_MOD 0            // nth <ref> on the page CM
#endif
#ifndef HP_DUMP_XML
#define HP_DUMP_XML (HP_TITLE_MOD || HP_PAGEID_MOD || HP_USER_MOD || HP_TEXT_MOD \
    || HP_NS_MOD || HP_DUMPREDIR_MOD || HP_IP_MOD || HP_REVCOMMENT_MOD \
    || HP_MINOR_MOD || HP_WIKIMODEL_MOD)
#endif
