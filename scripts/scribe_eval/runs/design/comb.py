from lex import *
from lib import load, STATS
import numpy as np
for d in ['devA','devB']:
    T={};Z=load(d)
    for l in open(f'{SP}/work/v2dev/{d}/texts.jsonl'):
        r=json.loads(l); T.setdefault(r['system'],[]).append(r['text'])
    ks=['human','sent_shuffled','word_shuffled','random_lexicon']
    L={k:np.array([stats(t)['allp'] for t in T[k]]) for k in ks}
    h=L['human'][~np.isnan(L['human'])]; mu=np.median(h); sd=np.std(h)
    ic1=STATS.index('c1'); ic1a=STATS.index('c1a')
    for k in ks: L[k]=np.nan_to_num((L[k]-mu)/sd, nan=-3)
    for name,f in [('c1 z',lambda k: Z[k][:,ic1]),('lex z',lambda k: L[k]),('c1+lex',lambda k: Z[k][:,ic1]+L[k]),('min',lambda k: np.minimum(Z[k][:,ic1],L[k])),('c1+c1a+lex',lambda k: Z[k][:,ic1]+Z[k][:,ic1a]+L[k])]:
        print(d,'%-10s'%name,'human>sent %.3f'%acc(f('human'),f('sent_shuffled')),' word>lex %.3f'%acc(f('word_shuffled'),f('random_lexicon')))
