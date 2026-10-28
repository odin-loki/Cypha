import lib
from lib import *
import itertools
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c3':-1})
def fix_s4(Z,f=0.05/0.15):
    Z=Z.copy(); k=STATS.index('s4'); z=Z[:,k]; m=z< -2.0537
    z[m]=-2.0537-(-2.0537-z[m])*f; Z[:,k]=z; return Z
A={k:fix_s4(v) for k,v in load('devA').items()}; B={k:fix_s4(v) for k,v in load('devB').items()}
RA={b:fix_s4(v) for b,v in load_ref('devA').items()}
base=[s for s in STATS if AX[s]!='R']
core=['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4']
lean=['s1','s2','s3','s4','c0','c1','c1a','loop8','rep','d1','d2','d3','d4']
subs={'all':base,'core':core,'lean':lean}
def accs(by,f):
    sc={k:-f(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
res=[]
for (sn,st),dead,agg,cap,(wi,ws,wc) in itertools.product(subs.items(),[0.5,1,1.5,2],['sum','mean','max'],[0,9,16,25],[(1,1,1),(1,2,1),(2,2,1),(4,2,1),(2,3,1),(1,3,1),(4,4,1),(2,4,1),(1,1.5,1),(2,1.5,1)]):
    a=accs(A,lambda Z: deviance(Z,st,dead,cap,agg,{'I':wi,'S':ws,'C':wc}))
    res.append((sum(x>=0.95 for x in a),min(a),sn,dead,agg,cap,(wi,ws,wc),[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:25]: print(r)
print('--- top 25 on devB')
for r in res[:25]:
    _,_,sn,dead,agg,cap,w,_=r
    b=accs(B,lambda Z: deviance(Z,subs[sn],dead,cap,agg,{'I':w[0],'S':w[1],'C':w[2]})); print(sn,dead,agg,cap,w,[round(v,3) for v in b],round(min(b),3))
