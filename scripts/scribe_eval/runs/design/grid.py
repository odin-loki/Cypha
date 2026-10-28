from lib import *
import itertools
A=load('devA'); B=load('devB')
def accs(by,D_of):
    sc={k:-D_of(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
base=[s for s in STATS if AX[s]!='R']
subsets={'all':base,
 'no_i1_slmax':[s for s in base if s not in('i1','sl_max')],
 'core':['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4'],
}
res=[]
for (sn,st),dead,agg,cap,wS,wC,wI in itertools.product(subsets.items(),[1,1.5,2,2.5],['sum','mean','max','rss'],[0,4,16],[1,2,4],[1,2],[1,2,4]):
    f=lambda Z: deviance(Z,st,dead,cap,agg,{'S':wS,'C':wC,'I':wI})
    a=accs(A,f)
    res.append((min(a),sum(x>=0.95 for x in a),sn,dead,agg,cap,wS,wC,wI,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[1],-r[0]))
for r in res[:25]: print(r)
print('n configs',len(res))
best=res[:10]
print('--- check top 10 on devB')
for r in best:
    _,_,sn,dead,agg,cap,wS,wC,wI,_=r
    f=lambda Z: deviance(Z,subsets[sn],dead,cap,agg,{'S':wS,'C':wC,'I':wI})
    b=accs(B,f); print(sn,dead,agg,cap,wS,wC,wI,[round(x,3) for x in b], min(b))
