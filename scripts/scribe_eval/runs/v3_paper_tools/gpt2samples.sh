# usage: gpt2samples.sh SEED OUT   (1,000 samples per system, cut as human samples are)
SP=${SCRIBE_SP:-/tmp/claude-0/-home-user-Cypha/c3804d0f-86ab-5ff6-9437-45a3d11e8017/scratchpad}; D=$SP/data/web; s=$1; out=$2; : > $out
$SP/bin/scribe samples $D/test.jsonl --n 1000 --seed $s --set web/gpt2out --system human --out $out.t 2>/dev/null; cat $out.t >> $out
for c in xl-1542M large-762M medium-345M small-117M xl-1542M-k40 large-762M-k40 medium-345M-k40 small-117M-k40; do
  $SP/bin/scribe samples $D/gpt2/$c.jsonl --n 1000 --seed $s --set web/gpt2out --system $c --out $out.t 2>/dev/null; cat $out.t >> $out; done; rm -f $out.t
