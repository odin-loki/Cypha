import lib
from lib import *
import itertools
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c3':-1})
A=load('devA'); B=load('devB'); RA=load_ref('devA')
base=[s for s in STATS if AX[s]!='R']
core=['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4']
def accs(by,f):
    sc={k:-f(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
def plain(st,dead,agg,cap,W):
    return lambda Z: deviance(Z,st,dead,cap,agg,W)
def tier(st,dead,agg,ql,kappa,W,eps):
    R=axis_dev(RA[1],st,dead,0,agg); q={a:np.quantile(R[a],ql) for a in 'SCI'}
    def f(Z):
        X=axis_dev(Z,st,dead,0,agg); D=np.zeros(len(Z))
        for a in 'SCI': D+=W[a]*np.log1p(np.maximum(0,X[a]-q[a])/kappa)+eps*X[a]
        return D
    return f
res=[]
for (sn,st),dead,agg,cap,(wi,ws,wc) in itertools.product({'all':base,'core':core}.items(),[0.5,1,1.5,2],['sum','mean','max'],[0,4,16],[(1,1,1),(1,2,1),(2,2,1),(4,2,1),(2,4,1),(1,4,1),(4,4,1),(1,1,2)]):
    a=accs(A,plain(st,dead,agg,cap,{'I':wi,'S':ws,'C':wc})); res.append(('plain',sum(x>=0.95 for x in a),min(a),sn,dead,agg,cap,(wi,ws,wc),[round(x,3) for x in a]))
for (sn,st),dead,agg,ql,kappa,(wi,ws,wc),eps in itertools.product({'all':base,'core':core}.items(),[0.5,1,1.5],['sum','max'],[0.9,0.95,0.99],[0.5,1,4],[(2,2,1),(4,2,1),(3,2,1),(4,4,1)],[0,0.01,0.1]):
    a=accs(A,tier(st,dead,agg,ql,kappa,{'I':wi,'S':ws,'C':wc},eps)); res.append(('tier',sum(x>=0.95 for x in a),min(a),sn,dead,agg,(ql,kappa,eps),(wi,ws,wc),[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[1],-r[2]))
for r in res[:30]: print(r)
print('--- top 15 on devB')
for r in res[:15]:
    kind,_,_,sn,dead,agg,x,w,_=r; st={'all':base,'core':core}[sn]; W={'I':w[0],'S':w[1],'C':w[2]}
    f=plain(st,dead,agg,x,W) if kind=='plain' else tier(st,dead,agg,x[0],x[1],W,x[2])
    b=accs(B,f); print(kind,sn,dead,agg,x,w,[round(v,3) for v in b],round(min(b),3))
