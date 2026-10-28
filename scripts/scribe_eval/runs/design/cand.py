from grid10 import make, IS, esum
from v2x import *
import itertools
syn=['s2g','s3g','s4']; cst=['c0','c1','c1a','c2','c3']
base=dict(qs=0.99,qi=0.99,cap=30,kap=4,wI=3,cdead=1)
def run(p,label):
    f=make(syn,['s1'],IS['I7'],cst,p['qs'],p['qi'],p['cap'],p['kap'],p['wI'],p['cdead'])
    a=accs(A,f); b=accs(B,f)
    print('%-28s A %s  B %s'%(label,[round(x,3) for x in a],[round(x,3) for x in b]))
    return f
f=run(base,'candidate')
for k,vals in [('qs',[0.98,0.995]),('qi',[0.98,0.995]),('cap',[20,50]),('kap',[2,8]),('wI',[1,10]),('cdead',[0.5,1.5])]:
    for v in vals:
        p=dict(base); p[k]=v; run(p,f'{k}={v}')
names=['human','sent_shuffled','word_shuffled','random_lexicon','random_letters','repeated_phrase','trie','trie_T0.6','trie_T0.8','trie_T1.0']
for by,lab in [(A,'devA'),(B,'devB')]:
    sc={k:np.exp(-f(by[k])/5) for k in names}
    print(lab,{k:round(float(sc[k].mean()),3) for k in names})
    print('  trie > lex %.3f  trie > letters %.3f  word>trie %.3f  T0.6>T1.0 %.3f'%(acc(sc['trie'],sc['random_lexicon']),acc(sc['trie'],sc['random_letters']),acc(sc['word_shuffled'],sc['trie']),acc(sc['trie_T0.6'],sc['trie_T1.0'])))
