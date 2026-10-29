#!/bin/bash
# The report card (bill of work M1-M4; docs/reports/REPORT_CARD.md): one command runs CyphaLM's frozen suites for one
# configuration and writes one card, OUT/LABEL.card.md and OUT/LABEL.card.tsv.
#
#   scripts/cypha_report_card.sh [options] [-- cyphalm_trie flags...]
#
# Options (defaults in brackets):
#   --label L             the run's name [card_YYYYMMDD]
#   --out DIR             where the outputs go [build/report_card, which git ignores]
#   --flags "F"           flags for every cyphalm_trie part (also: anything after --) [none: main's defaults]
#   --parts qa,ctx,gen    which parts run [qa,ctx,gen]
#   --protocol c1l|a0     the qa protocol [c1l]
#   --fresh               also read SCRIBE on fresh prompts (and LCI's gamma in a comparison)
#   --read-test           read the T-QA frozen test (once per change); appends a line to REPORT_CARD.md's Reads table
#   --baseline-flags "F"  also run a baseline with these flags (label L_base) and compare
#   --baseline-label B    compare with an earlier run B in the same OUT
#   --gen-source auto|a4d|gena   generation pipeline [auto: gena when GEN-A's harness is ready, else a4d]
#   --reuse               skip a part whose output is already there and complete
#   --budget-mb N         the RAM budget for LCI's kappa [200]
#
# Environment (defaults in brackets):
#   CYPHALM_TRIE [native/build/cyphalm_trie]   SCRIBE [native/build/scribe]   REPORT_CARD [native/build/cypha_report_card]
#   ENWIK8 [/root/cdata/enwik8]   DOLLY [/root/cdata/dolly.jsonl]   SCRIBE_VEC [/root/work/scribe/vec32]
#   C1L inputs: QA_PARSE_DEV [/root/work/qa2/parse_dev_span.tsv]  QA_PARSE_ALL [/root/work/qa2/parse_all_span.tsv]
#               QA_DK [/root/work/answer/dk_mention.tsv]  QA_SHARD [/root/work/f2/shard_facts.tsv]
#               QA_BASE [/root/work/answer/a0_rows.tsv]
#   GENA_READY [/root/work/gena/HARNESS_READY]
#
# Every part runs under nice -n 5 in the foreground; peak RSS is VmHWM polled every second (the method of
# /root/work/r2/peak.sh). The card answers nothing and changes nothing in the model (rule 10).
#
# Author: Odin Loch
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRIE=${CYPHALM_TRIE:-$ROOT/native/build/cyphalm_trie}
SC=${SCRIBE:-$ROOT/native/build/scribe}
RC=${REPORT_CARD:-$ROOT/native/build/cypha_report_card}
E=${ENWIK8:-/root/cdata/enwik8}
D=${DOLLY:-/root/cdata/dolly.jsonl}
VEC=${SCRIBE_VEC:-/root/work/scribe/vec32}
PDEV=${QA_PARSE_DEV:-/root/work/qa2/parse_dev_span.tsv}
PALL=${QA_PARSE_ALL:-/root/work/qa2/parse_all_span.tsv}
DK=${QA_DK:-/root/work/answer/dk_mention.tsv}
SHARD=${QA_SHARD:-/root/work/f2/shard_facts.tsv}
BASE=${QA_BASE:-/root/work/answer/a0_rows.tsv}
GREADY=${GENA_READY:-/root/work/gena/HARNESS_READY}

LABEL=card_$(date -u +%Y%m%d); OUT=$ROOT/build/report_card; FLAGS=""; PARTS=qa,ctx,gen; PROTO=c1l
FRESH=0; TEST=0; BFLAGS=""; BLABEL=""; GSRC=auto; REUSE=0; BUDGET=200; HAVE_BFLAGS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL=$2; shift 2;;
        --out) OUT=$2; shift 2;;
        --flags) FLAGS=$2; shift 2;;
        --parts) PARTS=$2; shift 2;;
        --protocol) PROTO=$2; shift 2;;
        --fresh) FRESH=1; shift;;
        --read-test) TEST=1; shift;;
        --baseline-flags) BFLAGS=$2; HAVE_BFLAGS=1; shift 2;;
        --baseline-label) BLABEL=$2; shift 2;;
        --gen-source) GSRC=$2; shift 2;;
        --reuse) REUSE=1; shift;;
        --budget-mb) BUDGET=$2; shift 2;;
        --) shift; FLAGS="$FLAGS $*"; break;;
        -h|--help) sed -n '2,32p' "$0"; exit 0;;
        *) echo "cypha_report_card: unknown option $1 (see --help)" >&2; exit 2;;
    esac
done
case "$PROTO" in c1l|a0) ;; *) echo "protocol must be c1l or a0" >&2; exit 2;; esac
if [ "$GSRC" = auto ]; then
    if [ -e "$GREADY" ] && [ -x "$ROOT/scripts/gen_eval.sh" ]; then GSRC=gena; else GSRC=a4d; fi
fi
case "$GSRC" in a4d) ;; gena)
    [ -x "$ROOT/scripts/gen_eval.sh" ] || { echo "gen-source gena: scripts/gen_eval.sh is not in the tree" >&2; exit 2; };;
    *) echo "gen-source must be auto, a4d or gena" >&2; exit 2;; esac
if [ $HAVE_BFLAGS = 1 ] && [ -n "$BLABEL" ]; then echo "give --baseline-flags or --baseline-label, not both" >&2; exit 2; fi
for f in "$TRIE" "$RC"; do [ -x "$f" ] || { echo "missing binary $f" >&2; exit 2; }; done
case ",$PARTS," in *,gen,*) [ -x "$SC" ] || { echo "missing binary $SC" >&2; exit 2; };; esac
mkdir -p "$OUT"

# peak OUTFILE cmd args... : run cmd, poll VmHWM every second, write "peak_rss_kb N seconds S rc R".
peak() {
    local out=$1; shift
    local t0 t1 pid hwm=0 v rc=0
    t0=$(date +%s.%N)
    "$@" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
        if [ -n "$v" ] && [ "$v" -gt "$hwm" ]; then hwm=$v; fi
        sleep 1
    done
    wait "$pid" || rc=$?
    t1=$(date +%s.%N)
    echo "peak_rss_kb $hwm seconds $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b - a}') rc $rc" > "$out"
    return $rc
}
done_ok() { [ $REUSE = 1 ] && [ -s "$1" ] && grep -q ' rc 0$' "$1"; }

# run_one LABEL FLAGS READ_TEST : every asked part for one configuration.
run_one() {
    local L=$1 F=$2 TST=$3 P=$OUT/$1
    local commit dirty
    commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
    dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | head -1 || true)
    [ -n "$dirty" ] && commit="$commit+dirty"
    {
        printf 'date\t%s\n' "$(date -u +%Y-%m-%d)"
        printf 'commit\t%s\n' "$commit"
        printf 'protocol\t%s\n' "$PROTO"
        printf 'gen_source\t%s\n' "$GSRC"
        printf 'split\t%s\n' "$([ "$TST" = 1 ] && echo test || echo dev)"
        printf 'flags\t%s\n' "$(echo $F)"
        printf 'parts\t%s\n' "$PARTS"
        printf 'fresh\t%s\n' "$FRESH"
        printf 'binary\t%s\n' "$TRIE"
    } > "$P.meta.tsv"
    # shellcheck disable=SC2086
    if [[ ",$PARTS," == *,qa,* ]] && ! done_ok "$P.qa.peak"; then
        local Q="--dev $ROOT/bench/eval/t_qa_wiki_dev.tsv"
        if [ "$TST" = 1 ]; then Q="$Q --test $ROOT/bench/eval/t_qa_wiki.tsv"; else Q="$Q --leak $ROOT/bench/eval/t_qa_wiki.tsv"; fi
        if [ "$PROTO" = c1l ]; then
            local PARSE=$PDEV; [ "$TST" = 1 ] && PARSE=$PALL
            Q="$Q --parse $PARSE --dk $DK --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard $SHARD --base $BASE"
        fi
        echo "[card] $L qa" >&2
        peak "$P.qa.peak" nice -n 5 "$TRIE" qa "$E" "$D" $Q $F --out "$P.qa.rows.tsv" > "$P.qa.out" 2> "$P.qa.err" || true
    fi
    if [[ ",$PARTS," == *,ctx,* ]] && ! done_ok "$P.ctx.peak"; then
        echo "[card] $L context8k" >&2
        peak "$P.ctx.peak" nice -n 5 "$TRIE" context8k "$E" "$D" $F > "$P.ctx.out" 2> "$P.ctx.err" || true
    fi
    if [[ ",$PARTS," == *,gen,* ]]; then
        local LADDER="ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --vectors $VEC --profile v2"
        if [ "$GSRC" = a4d ]; then
            if ! done_ok "$P.gen.peak"; then
                echo "[card] $L sample" >&2
                peak "$P.gen.peak" nice -n 5 "$TRIE" sample "$E" "$P.gen_dev.txt" "$P.gen_fresh.txt" $F > "$P.gen.out" 2> "$P.gen.err" || true
            fi
            if ! done_ok "$P.scribe.peak"; then
                echo "[card] $L scribe (dev prompts)" >&2
                rm -rf "$P.scribe_export"
                peak "$P.scribe.peak" nice -n 5 "$SC" $LADDER --test "enwik:$E:56:64" --seed 20261027 --gen-trie "$P.gen_dev.txt" \
                    --label "$L" --export "$P.scribe_export" > "$P.scribe.md" 2> "$P.scribe.err" || true
            fi
            if [ $FRESH = 1 ] && ! done_ok "$P.fresh.peak"; then
                echo "[card] $L scribe (fresh prompts)" >&2
                peak "$P.fresh.peak" nice -n 5 "$SC" $LADDER --test "enwik:$E:64:72" --seed 31415926 --gen-trie "$P.gen_fresh.txt" \
                    --label "$L" > "$P.fresh.scribe.md" 2> "$P.fresh.scribe.err" || true
            fi
        else
            # GEN-A's harness: its frozen generation benchmark (dev split); the card reads its ladder table.
            if ! done_ok "$P.gen.peak"; then
                echo "[card] $L gen_eval.sh dev" >&2
                peak "$P.gen.peak" "$ROOT/scripts/gen_eval.sh" dev "$L" "$OUT/gena" $F > "$P.gena.log" 2>&1 || true
                cp -f "$OUT/gena/$L.scribe.md" "$P.scribe.md" 2>/dev/null || true
                cp -f "$OUT/gena/$L.gen.txt" "$P.gen_dev.txt" 2>/dev/null || true
            fi
            [ $FRESH = 1 ] && echo "[card] gena: fresh prompts are GEN-A's test split; read it through gen_eval.sh test, not the card" >&2
        fi
    fi
}

run_one "$LABEL" "$FLAGS" "$TEST"
if [ $HAVE_BFLAGS = 1 ]; then
    # The baseline never reads the frozen test: comparisons are dev only (rule 3).
    BLABEL=${LABEL}_base
    run_one "$BLABEL" "$BFLAGS" 0
    "$RC" summarize --dir "$OUT" --label "$BLABEL" --budget-mb "$BUDGET" > /dev/null
fi
if [ -n "$BLABEL" ]; then
    "$RC" summarize --dir "$OUT" --label "$LABEL" --baseline "$BLABEL" --budget-mb "$BUDGET"
else
    "$RC" summarize --dir "$OUT" --label "$LABEL" --budget-mb "$BUDGET"
fi

if [ $TEST = 1 ]; then
    # Rule 1: every frozen-test read through the card is logged.
    RCMD=$ROOT/docs/reports/REPORT_CARD.md
    printf '| %s | %s | %s | `%s` |\n' "$(date -u +'%Y-%m-%d %H:%M')" "$LABEL" \
        "$(awk -F'\t' '$1=="commit"{print $2}' "$OUT/$LABEL.meta.tsv")" "$(echo $FLAGS)" >> "$RCMD"
    echo "[card] frozen test read; logged in $RCMD (commit it with the change)" >&2
fi
