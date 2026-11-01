#!/bin/bash
# N0, the neural diagnostic (docs/GENERATION_PLAN.md §13.3), and Q0, the Qwen yardstick (§13.5), end to end:
# data -> tokenizer -> K5/K8 -> M5 -> TR -> sampling -> Q0 -> SCRIBE -> measures -> report draft.
#
#   bench/nn_diag/run_n0.sh [OUT]                 the full run (OUT defaults to /mnt/d/cypha_llm/nn_diag)
#   TINY=1 bench/nn_diag/run_n0.sh OUT            the smoke test: a 2 MiB slice, short training, 6 prompts
#
# Resumable: every step skips when its output exists. Progress: OUT/STATUS.txt. Logs: OUT/logs/.
# The neural models are instruments only: never part of CyphaLM, never a default, their text never training text.
# N0 reads the GEN-A dev prompts only (bench/eval/gen_prompts_dev.tsv).
#
# Environment (defaults in brackets):
#   BIN      cyphalm_tok, cyphalm_tok_gen, scribe built from this branch [~/cypha_build/nn_diag]
#   PY       python with torch (CPU) and numpy [~/nn_venv/bin/python]
#   ENWIK9   [/mnt/d/cypha_llm/enwik9]     ENWIK8  [/mnt/d/cypha_llm/enwik8]
#   VEC      SCRIBE's word vectors v32 [/mnt/d/cypha_llm/scribe/v32]
#   THREADS  torch threads [32]
#   Q0_MODELS  Ollama's model store holding qwen3.5:9b [/mnt/c/Users/odinl/.ollama/models]
#
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/mnt/d/cypha_llm/nn_diag}
BIN=${BIN:-$HOME/cypha_build/nn_diag}
PY=${PY:-$HOME/nn_venv/bin/python}
E9=${ENWIK9:-/mnt/d/cypha_llm/enwik9}
E8=${ENWIK8:-/mnt/d/cypha_llm/enwik8}
VEC=${VEC:-/mnt/d/cypha_llm/scribe/v32}
TH=${THREADS:-32}  # torch threads: one per physical core (32); 64 with hyperthreads trains 1.6-2.3x slower
Q0_MODELS=${Q0_MODELS:-/mnt/c/Users/odinl/.ollama/models}
DEV=$ROOT/bench/eval/gen_prompts_dev.tsv
NN="$PY -u $ROOT/bench/nn_diag/nn_diag.py"
mkdir -p "$OUT/logs" "$OUT/gen" "$OUT/scribe"
ST=$OUT/STATUS.txt
status() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
trap 'status "FAILED at line $LINENO (exit $?); see $OUT/logs"' ERR

if [ "${TINY:-0}" = 1 ]; then
    SRC_HI=110; TRAIN_BYTES=$((2 << 20)); VAL_BYTES=$((256 << 10)); LIMIT="--limit 6"; TINYF="--tiny"; QLIM="--limit 3"
else
    SRC_HI=953.67; TRAIN_BYTES=$((128 << 20)); VAL_BYTES=$((4 << 20)); LIMIT=""; TINYF=""; QLIM=""
fi
GENF=(--prompts-file "$DEV" --temps 0.8,1.0 --bytes 600 --nucleus 0.9 --no-repeat $LIMIT)
status "N0 start (TINY=${TINY:-0}) out=$OUT commit=$(git -C "$ROOT" rev-parse --short HEAD)"

# 0. SCRIBE's vectors (TRIE_1TB_PROFILE.md: ppmi_svd on enwik8 [16, 48) MiB)
if [ ! -f "$VEC.vectors.f32" ]; then
    status "vectors: building $VEC"
    g++ -std=c++20 -O2 -o "$BIN/ppmi_svd" "$ROOT/native/tools/ppmi_svd.cpp"
    W=$(dirname "$VEC"); mkdir -p "$W"
    dd if="$E8" of="$W/train32.xml" bs=1M skip=16 count=32 status=none
    (cd "$W" && "$BIN/ppmi_svd" train32.xml none none - "$(basename "$VEC")" --no-ri --no-ablation) > "$OUT/logs/vectors.log" 2>&1
fi

# 1. W1: enwik9 [100,000,000, 1,000,000,000) cleaned by SCRIBE, one page per line
if [ ! -s "$OUT/docs.jsonl" ]; then
    status "docs: scribe docs enwik9 [95.37, $SRC_HI) MiB"
    "$BIN/scribe" docs "enwik:$E9:95.37:$SRC_HI" --out "$OUT/docs.jsonl.tmp" 2> "$OUT/logs/docs.log"
    mv "$OUT/docs.jsonl.tmp" "$OUT/docs.jsonl"
fi
if [ ! -s "$OUT/data.json" ]; then
    status "data: drop pages holding a dev continuation span; train $TRAIN_BYTES, val $VAL_BYTES bytes"
    $NN data --docs "$OUT/docs.jsonl" --dev "$DEV" --out "$OUT" --train-bytes $TRAIN_BYTES --val-bytes $VAL_BYTES > "$OUT/logs/data.log" 2>&1
fi

# 2. One BPE tokenizer with word boundaries, 8,192, learned on train
TOK=$OUT/tok-bpe-8192.tok
if [ ! -s "$TOK" ]; then
    status "tokenizer: cyphalm_tok train (BPE 8192)"
    "$BIN/cyphalm_tok" train "$OUT/train.txt" --arm bpe --sizes 8192 --out "$OUT/tok" --bytes "$(stat -c%s "$OUT/train.txt")" > "$OUT/logs/tok.log" 2>&1
fi
for s in train val; do
    if [ ! -s "$OUT/$s.u16" ]; then
        status "encode $s"
        CHK=""; [ "$s" = val ] && CHK=--check
        "$BIN/cyphalm_tok_gen" n0-encode "$TOK" "$OUT/$s.txt" "$OUT/$s.u16.tmp" $CHK >> "$OUT/logs/encode.log" 2>&1
        mv "$OUT/$s.u16.tmp" "$OUT/$s.u16"
    fi
done

# 3. K5 and K8: bits per byte on val, then samples (T4c's decoder)
for K in 5 8; do
    A=K$K
    if [ ! -s "$OUT/$A.valbpb.json" ]; then
        status "$A: val bits per byte"
        "$BIN/cyphalm_tok_gen" n0-score "$OUT/train.u16" 8192 $K "$OUT/val.u16" > "$OUT/logs/$A.score.out" 2> "$OUT/logs/$A.score.err"
        "$PY" -c "import json,sys,os; f=open('$OUT/logs/$A.score.out').read().split(); b=float(f[f.index('bits')+1]); n=os.path.getsize('$OUT/val.txt'); json.dump(dict(arm='$A',bits=b,bytes=n,bpb=b/n,tokens=int(f[f.index('tokens')+1])),open('$OUT/$A.valbpb.json','w'),indent=1)"
    fi
    if [ ! -f "$OUT/gen/$A.gen.txt.done" ]; then
        status "$A: sampling"
        "$BIN/cyphalm_tok_gen" n0-gen "$TOK" "$OUT/gen/$A.gen.txt" --kn "$OUT/train.u16" --order $K "${GENF[@]}" 2> "$OUT/logs/$A.gen.err"
        touch "$OUT/gen/$A.gen.txt.done"
    fi
done

# 4. M5, then TR: two passes over train, AdamW, cosine, best checkpoint by val loss
for arm in m5 tr; do
    A=${arm^^}
    if [ ! -f "$OUT/$arm.trained" ]; then
        status "$A: training (progress in $OUT/$arm.progress)"
        $NN train --arm $arm --dir "$OUT" --threads $TH $TINYF >> "$OUT/logs/$arm.train.log" 2>&1
    fi
    if [ ! -s "$OUT/$A.valbpb.json" ]; then
        status "$A: val bits per byte"
        $NN valbpb --arm $arm --dir "$OUT" --threads $TH > "$OUT/logs/$arm.valbpb.log" 2>&1
    fi
done

# Q0 (§13.5): Qwen 3.5 9B on Ollama, after training, on the GPU beside step 5's CPU sampling. A private server on
# port 11535 reads Odin's model store; it is stopped when Q0 ends. Qwen's text is scored and used for nothing else.
q0_run() {
    status "Q0: starting a private ollama server (port 11535)"
    OLLAMA_HOST=127.0.0.1:11535 OLLAMA_MODELS="$Q0_MODELS" OLLAMA_NOPRUNE=1 nohup ollama serve > "$OUT/logs/q0.ollama.log" 2>&1 &
    local opid=$!
    for _ in $(seq 60); do curl -s -m 2 http://127.0.0.1:11535/api/tags > /dev/null && break; sleep 2; done
    status "Q0: sampling"
    if $NN q0 --dir "$OUT" --dev "$DEV" --url http://127.0.0.1:11535 $QLIM > "$OUT/logs/q0.log" 2>&1; then status "Q0: done"
    else status "Q0 failed (see logs/q0.log); the report goes on without it"; fi
    kill "$opid" 2> /dev/null || true
}
Q0PID=""
if [ ! -s "$OUT/gen/Q0.gen.txt" ] && [ "${SKIP_Q0:-0}" != 1 ]; then
    q0_run &
    Q0PID=$!
fi

# 5. M5 and TR samples: the same decoder (cyphalm_tok_gen n0-gen --pipe), 8 shards of the prompts
SH=8; PER=$((TH / SH))
for arm in m5 tr; do
    A=${arm^^}
    [ -f "$OUT/gen/$A.gen.txt.done" ] && continue
    status "$A: sampling ($SH shards x $PER threads)"
    pids=()
    for i in $(seq 0 $((SH - 1))); do
        F=$OUT/gen/$A.s$i.gen.txt
        if [ ! -f "$F.done" ]; then
            ( $NN serve --arm $arm --dir "$OUT" --threads $PER -- "$BIN/cyphalm_tok_gen" n0-gen "$TOK" "$F" --pipe --shard $i/$SH "${GENF[@]}" \
                > "$OUT/logs/$A.s$i.serve.log" 2> "$OUT/logs/$A.s$i.gen.err" && touch "$F.done" ) &
            pids+=($!)
        fi
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for i in $(seq 0 $((SH - 1))); do [ -f "$OUT/gen/$A.s$i.gen.txt.done" ] || { status "$A shard $i failed"; exit 1; }; done
    "$PY" - "$OUT/gen" "$A" "$SH" <<'EOF'
import sys
d, A, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
blocks, model, cont = {}, None, {}
for i in range(n):
    cur = None
    for line in open("%s/%s.s%d.gen.txt" % (d, A, i), encoding="utf-8", errors="surrogateescape"):
        if line.startswith("MODEL\t"):
            model = model or line
        elif line.startswith("PROMPT\t"):
            cur = int(line.split("\t")[1]); blocks[cur] = [line]
        else:
            blocks[cur].append(line)
    for line in list(open("%s/%s.s%d.gen.txt.cont.tsv" % (d, A, i)))[1:]:
        cont[int(line.split("\t")[0])] = line
with open("%s/%s.gen.txt" % (d, A), "w", encoding="utf-8", errors="surrogateescape") as o:
    o.write(model)
    for k in sorted(blocks):
        o.writelines(blocks[k])
with open("%s/%s.gen.txt.cont.tsv" % (d, A), "w") as o:
    o.write("row\tbytes\tbits\n")
    for k in sorted(cont):
        o.write(cont[k])
print("merged", A, len(blocks), "prompts")
EOF
    touch "$OUT/gen/$A.gen.txt.done"
done

# 6. Q0: wait for its background run (started after training, on the GPU)
if [ -n "${Q0PID:-}" ]; then wait "$Q0PID" || true; fi
ARMS=K5,K8,M5,TR
[ -s "$OUT/gen/Q0.gen.txt" ] && ARMS=$ARMS,Q0

# 7. SCRIBE v2 (gen_eval.sh's ladder) on every arm, in parallel
pids=()
for A in ${ARMS//,/ }; do
    [ -s "$OUT/scribe/$A.scribe.md" ] && continue
    status "$A: SCRIBE ladder"
    ( "$BIN/scribe" ladder --pack "enwik:$E8:16:48" --ref "enwik:$E8:48:56" --test "enwik:$E8:56:64" --vectors "$VEC" \
        --profile v2 --seed 20261027 --gen-trie "$OUT/gen/$A.gen.txt" --label "$A" --export "$OUT/scribe/$A.export" \
        > "$OUT/scribe/$A.scribe.md.tmp" 2> "$OUT/logs/$A.scribe.err" && mv "$OUT/scribe/$A.scribe.md.tmp" "$OUT/scribe/$A.scribe.md" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# 8. Measures and the report draft
status "measures"
$NN measures --dir "$OUT" --arms $ARMS > "$OUT/logs/measures.log" 2>&1
status "summary"
REPORT=$OUT/NN_DIAG_N0.draft.md
$NN summary --dir "$OUT" --arms $ARMS --out "$REPORT" --head "$ROOT/bench/nn_diag/report_head.md" > "$OUT/logs/summary.log" 2>&1
[ "${TINY:-0}" = 1 ] || cp "$REPORT" "$ROOT/docs/reports/NN_DIAG_N0.md"
status "N0 done: $REPORT"
