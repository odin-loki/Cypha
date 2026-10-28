import lib
from lib import *
import itertools
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c3':-1})
def fix_s4(Z,f=0.05/0.15):
    Z=Z.copy(); k=STATS.index('s4'); z=Z[:,k]; m=z< -2.0537
    z[m]=-2.0537-(-2.0537-z[m])*f; Z[:,k]=z; return Z
A={k:fix_s4(v) for k,v in load('devA').items()}; B={k:fix_s4(v) for k,v in load('devB').items()}
def U(Z):
    out=np.zeros_like(Z)
    for k,s in enumerate(STATS):
        z=Z[:,k]; out[:,k]=np.maximum(0,-z) if SIDE[s]>0 else (np.maximum(0,z) if SIDE[s]<0 else np.abs(z))
    out[np.isnan(Z)]=0; return out
def D(Z,stats,dead,p,agg,W,cap=0):
    u=np.maximum(0,U(Z)-dead)**p
    if cap>0: u=cap*(1-np.exp(-u/cap))
    tot=np.zeros(len(Z))
    for a in 'SCI':
        idx=[STATS.index(s) for s in stats if AX[s]==a]
        P=u[:,idx]
        tot+=W[a]*(P.sum(1) if agg=='sum' else P.max(1) if agg=='max' else P.mean(1))
    return tot
def accs(by,f):
    sc={k:-f(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
subs={'core':['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4'],
      'lean':['s1','s2','s3','s4','c1','c1a','loop8','rep','d1','d2','d3','d4'],
      'lean2':['s1','s2','s3','s4','c0','c1','loop8','rep','d1','d2','d3','d4'],
      'min':['s1','s2','s3','c1','loop8','rep','d1','d4']}
res=[]
for (sn,st),dead,p,agg,(wi,ws,wc),cap in itertools.product(subs.items(),[0,0.5,1],[1,2],['sum','max','mean'],[(1,1,1),(1,2,1),(2,2,1),(2,3,1),(1,3,1),(4,4,1),(1,1.5,1),(2,1.5,1),(3,3,2)],[0,30]):
    a=accs(A,lambda Z: D(Z,st,dead,p,agg,{'I':wi,'S':ws,'C':wc},cap))
    res.append((sum(x>=0.95 for x in a),round(min(a),3),sn,dead,p,agg,(wi,ws,wc),cap,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:25]: print(r)
print('--- devB')
for r in res[:25]:
    _,_,sn,dead,p,agg,w,cap,_=r
    b=accs(B,lambda Z: D(Z,subs[sn],dead,p,agg,{'I':w[0],'S':w[1],'C':w[2]},cap)); print(sn,dead,p,agg,w,cap,[round(v,3) for v in b],round(min(b),3))
