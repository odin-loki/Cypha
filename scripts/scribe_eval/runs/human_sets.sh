set -e
B=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/bin/scribe
head -n 3000 /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/wp/ref.jsonl > wp_ref3000.jsonl
$B windows --in wp_ref3000.jsonl --out wp_ref_windows.jsonl
$B prompts /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/web/ref.jsonl --n 4000 --per-doc 2 --seed 61 --set web_ref --out web_ref_prompts.jsonl
/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/venv/bin/python -I -c "
import json,re
with open('roc_ref_cont.jsonl','w') as o:
    for l in open('/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/roc/ref.jsonl'):
        r=json.loads(l); s=re.split(r'(?<=[.!?]) ',r['text'].strip(),maxsplit=1)
        if len(s)==2: o.write(json.dumps({'id':r['id'],'text':s[1],'prompt':s[0],'system':'human'})+'\n')"
for prof in v2 v2p; do
  $B score --profile $prof --window --pack /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/wp/pack.jsonl --ref /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/wp/ref.jsonl --vectors /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/packs/wp --ref-prompts wp_ref_windows.jsonl --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/hanna/texts.jsonl --out hanna.$prof.jsonl
  $B score --profile $prof --window --pack /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/wp/pack.jsonl --ref /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/wp/ref.jsonl --vectors /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/packs/wp --ref-prompts wp_ref_windows.jsonl --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/openmeva/mans_wp.jsonl --out mans_wp.$prof.jsonl
  $B score --profile $prof --window --pack /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/web/pack.jsonl --ref /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/web/ref.jsonl --vectors /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/packs/web --ref-prompts web_ref_prompts.jsonl --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/mauve_he/texts.jsonl --out mauve_he.$prof.jsonl
  $B score --profile $prof --ref-whole --pack raw:/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/roc/pack.jsonl --ref raw:roc_ref_cont.jsonl --vectors /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/packs/roc --ref-prompts roc_ref_cont.jsonl --in /tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad/data/openmeva/mans_roc.jsonl --out mans_roc.$prof.jsonl
done
echo HUMANDONE
