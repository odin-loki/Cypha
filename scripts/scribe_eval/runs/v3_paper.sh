#!/usr/bin/env bash
# The SCRIBE v3 paper's new reads (docs/reports/SCRIBE_V3_PAPER.md). Registered before any of them was run.
# Every score file is written with --profile v31, which also writes the v2 and v3 fields (scribe_v2, scribe_v3,
# surprise, match, min3, scribe_v31, surprise31, match31), so v2, v3 and v3.1 are read from the same run.
# Usage: scripts/scribe_eval/runs/v3_paper.sh STEP...   (steps below; "all" runs every step in order)
# Author: Odin Loch
set -euo pipefail
SP=${SCRIBE_SP:-/tmp/claude-0/-home-user-Cypha/c3804d0f-86ab-5ff6-9437-45a3d11e8017/scratchpad}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
B=$SP/bin/scribe; PY="python3 -I"; T=$REPO/scripts/scribe_eval/runs/v3_paper_tools; MT="python3 $T/memtime.py"
E=$SP/dl/enwik/enwik8; E9=$SP/dl/enwik/enwik9; V=$SP/work/vec/vec32; D=$SP/data; P=$SP/packs
O=$SP/run/v3p; mkdir -p "$O"
WIKI="--pack enwik:$E:16:48 --ref enwik:$E:48:56 --vectors $V"
SEED=20261104       # every new draw of this paper
J="--profile v31"

noprompt() { $PY -c "import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l); r.pop('prompt',None); print(json.dumps(r))" "$1" > "$2"; }

step_fresh() {   # R1: a new fresh read, enwik9 [900, 924) MiB, never read by any SCRIBE
  mkdir -p $O/fresh
  $B ladder $WIKI --test enwik:$E9:900:924 $J --seed $SEED --dedup --export $O/fresh/ladder --label fresh3 > $O/fresh/ladder.md 2> $O/fresh/ladder.err
  $B attacks $WIKI --test enwik:$E9:900:924 $J --seed $SEED --dedup --heldout --outside enwik:$E:0:16 --n 200 \
     --export $O/fresh/att --label att3 > $O/fresh/attacks.md 2> $O/fresh/attacks.err || true   # exit 3 = rule failed
}

step_genre() {   # R2: web and books ladders, each with its own pack and reference; web against the Wikipedia pack
  mkdir -p $O/genre
  $B ladder --pack $D/web/pack.jsonl --ref $D/web/ref.jsonl --test $D/web/test.jsonl --vectors $P/web $J --seed $SEED --export $O/genre/web --label web > $O/genre/web.md 2> $O/genre/web.err
  $B ladder --pack $D/books/pack.jsonl --ref $D/books/ref.jsonl --test $D/books/fresh.jsonl --vectors $P/books $J --seed $SEED --export $O/genre/books --label books > $O/genre/books.md 2> $O/genre/books.err
  $B ladder $WIKI --test $D/web/test.jsonl $J --seed $SEED --export $O/genre/web_wikiref --label web_wikiref > $O/genre/web_wikiref.md 2> $O/genre/web_wikiref.err
}

step_gpt2() {    # R3: the GPT-2 output dataset, 1,000 samples per system (cut as human samples are)
  mkdir -p $O/gpt2out
  bash $T/gpt2samples.sh $SEED $O/gpt2out/samples.jsonl
  $B score --pack $D/web/pack.jsonl --ref $D/web/ref.jsonl --vectors $P/web $J --in $O/gpt2out/samples.jsonl --out $O/gpt2out/samples.v31.jsonl 2> $O/gpt2out/score.err
}

step_dose() {    # R4: the nine held-out corruption families on 200 human samples of enwik8 [72, 95.37) MiB
  mkdir -p $O/dose
  $B samples enwik:$E:72:95.37 --n 200 --seed $SEED --set fresh --out $O/dose/human200.jsonl
  $PY $REPO/scripts/scribe_eval/corrupt.py --in $O/dose/human200.jsonl --vocab $V.vocab.tsv --out $O/dose/corrupt.jsonl
  $B score $WIKI $J --in $O/dose/corrupt.jsonl --out $O/dose/corrupt.v31.jsonl 2> $O/dose/score.err
}

step_human() {   # R5-R7: HANNA, OpenMEVA MANS and the MAUVE human evaluation, open (no prompt) and with the prompt
  mkdir -p $O/human
  for s in hanna/texts openmeva/mans_wp openmeva/mans_roc mauve_he/texts; do
    n=$(echo $s | tr / _); noprompt $D/$s.jsonl $O/human/$n.open.jsonl; done
  $PY -c "import json,re
with open('$O/human/roc_ref_cont.jsonl','w') as o:
    for l in open('$D/roc/ref.jsonl'):
        r=json.loads(l); s=re.split(r'(?<=[.!?]) ',r['text'].strip(),maxsplit=1)
        if len(s)==2: o.write(json.dumps({'id':r['id'],'text':s[1],'prompt':s[0],'system':'human'})+'\n')"
  for m in open prompt; do
    pick() { if [ $m = open ]; then echo $O/human/$1.open.jsonl; else echo $D/$2.jsonl; fi; }
    $B score $J --window --pack $D/wp/pack.jsonl --ref $D/wp/ref.jsonl --vectors $P/wp --in $(pick hanna_texts hanna/texts) --out $O/human/hanna.$m.jsonl 2>> $O/human/score.err
    $B score $J --window --pack $D/wp/pack.jsonl --ref $D/wp/ref.jsonl --vectors $P/wp --in $(pick openmeva_mans_wp openmeva/mans_wp) --out $O/human/mans_wp.$m.jsonl 2>> $O/human/score.err
    $B score $J --window --pack $D/web/pack.jsonl --ref $D/web/ref.jsonl --vectors $P/web --in $(pick mauve_he_texts mauve_he/texts) --out $O/human/mauve_he.$m.jsonl 2>> $O/human/score.err
    $B score $J --ref-whole --pack raw:$D/roc/pack.jsonl --ref raw:$O/human/roc_ref_cont.jsonl --vectors $P/roc --in $(pick openmeva_mans_roc openmeva/mans_roc) --out $O/human/mans_roc.$m.jsonl 2>> $O/human/score.err
  done
}

step_models() {  # R8: KN word n-grams of orders 1-5 (trained on enwik8 [8, 16) MiB) on 250 prompts of enwik8 [72, 95.37) MiB
  M=$SP/run/models; mkdir -p $O/models
  $PY -c "import json,sys,glob
rows=[]
for l in open('$M/prompts_fresh.jsonl'):
    r=json.loads(l); rows.append({'id':'ngram/human/'+r['id'],'set':'fresh','system':'human','text':r['text'],'prompt':r['prompt']})
for o in range(1,6):
    for l in open(f'$M/gen/ngram/ngram{o}.fresh.jsonl'):
        r=json.loads(l); rows.append({'id':r['id'],'set':'fresh','system':r['system'],'text':r['text'],'prompt':r['prompt']})
with open('$O/models/raw.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')"
  $B prepare --in $O/models/raw.jsonl --out $O/models/prepared.jsonl --mode continuation
  noprompt $O/models/prepared.jsonl $O/models/ngram.open.jsonl
  $B score $WIKI $J --in $O/models/ngram.open.jsonl --out $O/models/ngram.v31.jsonl 2> $O/models/score.err
}

step_qa() {      # R9: the Dolly question-answering ladder (qa_ladder.py, its default seed), with the prompt
  Q=$SP/run/qa; mkdir -p $O/qa
  $B score $J --pack raw:$Q/pack.jsonl --ref raw:$Q/ref.jsonl --vectors $P/qa --in $Q/test.jsonl --out $O/qa/test.v31.jsonl 2> $O/qa/score.err
  $B score --profile v2p --pack raw:$Q/pack.jsonl --ref raw:$Q/ref.jsonl --vectors $P/qa --ref-prompts $Q/ref.jsonl --in $Q/test.jsonl --out $O/qa/test.v2p.jsonl 2>> $O/qa/score.err
}

step_attacks() { # R10: v2's attack set (attacks.py) on 250 prompts of enwik8 [72, 95.37) MiB, scored with the prompt
  mkdir -p $O/attacks
  $B prompts enwik:$E:72:95.37 --n 250 --seed $SEED --set attacks --out $O/attacks/prompts.jsonl
  $B samples enwik:$E:16:48 --n 250 --seed $SEED --set pack --out $O/attacks/copy.jsonl
  $PY $REPO/scripts/scribe_eval/attacks.py --in $O/attacks/prompts.jsonl --copy $O/attacks/copy.jsonl --vocab $V.vocab.tsv --out $O/attacks/attacks.jsonl
  $B score $WIKI $J --in $O/attacks/attacks.jsonl --out $O/attacks/attacks.v31.jsonl 2> $O/attacks/score.err
}

step_stability() { # R11: seeds, reference size, pack size, length (v2 paper section 6.6, same commands, profile v31)
  S=$O/stab; mkdir -p $S
  L="$WIKI --test enwik:$E:72:95.37 $J"
  $B ladder $L --seed 1123581321 --export $S/base --label base > $S/base.md 2> $S/base.err
  for sd in 1 2 3 4 5; do $B ladder $L --seed $sd --label seed$sd > $S/seed$sd.md 2> $S/seed$sd.err; done
  for n in 50 100 200 500 1000; do for rs in 11 12 13; do
    $B ladder $L --seed 1123581321 --ref-n $n --ref-seed $rs --label ref${n}_$rs > $S/ref${n}_$rs.md 2> $S/ref${n}_$rs.err; done; done
  for mb in 8 16 32 64 128; do
    dd if=$E9 of=$S/pack$mb.xml bs=1048576 skip=128 count=$mb 2>/dev/null
    $MT $SP/bin/ppmi_svd $S/pack$mb.xml none none - $S/vec$mb --no-ri --no-ablation > $S/ppmi$mb.log 2> $S/ppmi$mb.time
    rm -f $S/pack$mb.xml
    $MT $B ladder --pack enwik:$E9:128:$((128+mb)) --ref enwik:$E:48:56 --test enwik:$E:72:95.37 --vectors $S/vec$mb $J --seed 1123581321 --label pack$mb > $S/pack$mb.md 2> $S/pack$mb.err
    rm -f $S/vec$mb.vectors.f32
  done
}

step_k5() {      # R12: K5's saved GEN-A dev continuations (bench/k5_scale), the GEN-A dev ladder, profile v31
  K=$O/k5; mkdir -p $K
  for f in $REPO/bench/k5_scale/results_pc_2026-11-04/reads/{E8,N7,N30,N100,N300,NALL}_s*.gen.txt \
           $REPO/bench/k5_scale/results_pc_2026-11-05/reads/{U8,C1}_s*.gen.txt; do
    n=$(basename $f .gen.txt)
    $B ladder $WIKI --test enwik:$E:56:64 $J --seed 20261027 --gen-trie $f --label $n > $K/$n.md 2> $K/$n.err
  done
}

step_cost() {    # R13: time and peak memory of scoring, one core, profiles v2, v3 and v31
  C=$O/cost; mkdir -p $C
  for prof in v2 v3 v31; do
    $MT $B score --timing $WIKI --profile $prof --in $O/stab/base/texts.jsonl --out /dev/null > $C/wiki.$prof.out 2> $C/wiki.$prof.err
    $MT $B score --timing --pack $D/web/pack.jsonl --ref $D/web/ref.jsonl --vectors $P/web --profile $prof --in $O/gpt2out/samples.jsonl --out /dev/null > $C/web.$prof.out 2> $C/web.$prof.err
  done
}

steps=("$@"); [ "${steps[0]:-}" = all ] && steps=(fresh genre gpt2 dose human models qa attacks stability k5 cost)
for s in "${steps[@]}"; do echo "== $s"; step_$s; done
