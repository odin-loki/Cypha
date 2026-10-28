set -u
B=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/bin/scribe; E=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/dl/enwik/enwik8; E9=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/dl/enwik/enwik9; V=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/work/v1/v32
L="--pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:72:95.37 --vectors $V --profile v2 --gen-trie /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/work/v1/gen_fresh.txt"
run() { name=$1; shift; nice $B ladder $L "$@" --label $name > abl_$name.md 2> abl_$name.err; echo "$name done"; }
# 1. ablations (fresh ladder, seed of the verdict)
run base --seed 1123581321
run noL --seed 1123581321 --v2 off=s1
run noS --seed 1123581321 --v2 off=s2g+s3g+s4
run noC --seed 1123581321 --v2 off=c0+c1+c1a+c2+c3
run noI --seed 1123581321 --v2 off=loop8+rep+d1+d2+d3+d4+i1
for st in s1 s2g s3g s4 c0 c1 c1a c2 c3 loop8 rep d1 d2 d3 d4 i1; do run off_$st --seed 1123581321 --v2 off=$st; done
run rawlm --seed 1123581321 --v2 off=s2g+s3g,on=s2+s3
run plusdiag --seed 1123581321 --v2 on=s3min+sl_mean+sl_max+wfreq
run linear --seed 1123581321 --v2 capS=1000000,kapS=1000000
run q95 --seed 1123581321 --v2 qS=0.95,qI=0.95
run dead05 --seed 1123581321 --v2 dead=0.5
run dead15 --seed 1123581321 --v2 dead=1.5
run wI1 --seed 1123581321 --v2 wI=1
# 2. seed variance
for sd in 1 2 3 4 5; do run seed$sd --seed $sd; done
# 3. reference size (3 reference draws each)
for n in 50 100 200 500 1000; do for rs in 11 12 13; do run ref${n}_$rs --seed 1123581321 --ref-n $n --ref-seed $rs; done; done
# 4. pack size (enwik9 from 128 MiB; vectors retrained on each pack)
for mb in 8 16 32 64 128; do
  dd if=$E9 of=pack$mb.xml bs=1048576 skip=128 count=$mb 2>/dev/null
  /usr/bin/time -v nice /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/bin/ppmi_svd pack$mb.xml none none - vec$mb --no-ri --no-ablation > ppmi$mb.log 2> ppmi$mb.time
  /usr/bin/time -v nice $B ladder --pack enwik:$E9:128:$((128+mb)) --ref enwik:$E:48:56 --test enwik:$E:72:95.37 --vectors vec$mb --profile v2 --seed 1123581321 --label pack$mb > abl_pack$mb.md 2> abl_pack$mb.err
  rm -f pack$mb.xml; echo "pack$mb done"
done
# 5. timing and RAM of scoring
/usr/bin/time -v $B score --timing --profile v2 --pack enwik:$E:16:48 --ref enwik:$E:48:56 --vectors $V --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/run/fresh/wiki_fresh/texts.jsonl --out /dev/null > timing_wiki.out 2> timing_wiki.err
/usr/bin/time -v $B score --timing --profile v2 --pack /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/web/pack.jsonl --ref /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/web/ref.jsonl --vectors /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/packs/web --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/run/gpt2out/samples.jsonl --out /dev/null > timing_web.out 2> timing_web.err
echo PROFILEDONE
