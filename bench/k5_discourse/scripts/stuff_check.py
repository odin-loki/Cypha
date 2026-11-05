# Non-deciding stuffing check (K5_DISCOURSE.md amendment 2): content-word share of each arm's text, and K5's own bits
# per byte on it (K5 without the gate or rescaling reads PROMPT + GEN as a continuation; lower = more K5-fluent).
import sys, os, re, subprocess
sys.path.insert(0, '/home/odin/disc_wt/scripts')
OUT = '/mnt/d/cypha_llm/disc/out'; CH = '/mnt/d/cypha_llm/disc/stuff'; os.makedirs(CH, exist_ok=True)
STOP = set(re.findall(r'"([^"]+)"', open('/home/odin/disc_wt/native/include/cypha/lm/stopwords.hpp').read().split('determiners()')[0]))
TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:[.,]\d+)*|\S")
def unesc(s):
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
def content_share(texts):
    n = c = 0
    for t in texts:
        for w in TOK.findall(t):
            n += 1
            lw = w.lower()
            c += len(w) >= 3 and w.isalpha() and lw not in STOP
    return c / max(n, 1)
def read_gen(label):
    P, G = {}, {}
    for line in open(f'{OUT}/{label}.gen.txt', encoding='utf-8', errors='replace'):
        f = line.rstrip('\n').split('\t')
        if f[0] == 'PROMPT': cur = int(f[1]); P[cur] = unesc(f[2])
        elif f[0] == 'GEN': G[cur] = unesc(f[2])
    return P, G
def k5_bpb(label, P, G):
    tsv = f'{CH}/{label}.tsv'
    with open(tsv, 'w') as o:
        o.write('id\tdoc\tprompt\tcontinuation\n')
        for i in sorted(P): o.write(f'x/{i}\t0\t{P[i]}\t' + G[i].replace('\n', ' ').replace('\t', ' ').strip() + '\n')
    r = subprocess.run(['/home/odin/cypha_build/disc/cyphalm_trie', 'sample', '/mnt/d/cypha_llm/enwik8', f'{CH}/{label}.x.txt', '--prompts-file', tsv,
                        '--prompts', str(len(P)), '--k5', '/mnt/d/cypha_llm/fuzzy/train.txt', '--k5-gate', '1'], capture_output=True, text=True,
                       env=dict(os.environ, K5_THREADS='4'))
    m = re.search(r'k5_cont_bpb ([\d.]+)', r.stdout)
    return float(m.group(1)) if m else float('nan')
if sys.argv[1] == 'human':
    rows = [l.rstrip('\n').split('\t') for l in open('/home/odin/disc_wt/bench/eval/gen_prompts_dev.tsv')][1:]
    print('human', round(content_share([r[3] for r in rows]), 4), 'k5_bpb(human) = gen_eval k5_cont_bpb of K5 without the gate')
for label in sys.argv[1:]:
    if label == 'human': continue
    P, G = read_gen(label)
    print(label, round(content_share(G.values()), 4), round(k5_bpb(label, P, G), 4), flush=True)
