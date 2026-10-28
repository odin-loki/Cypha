from lib import *
import itertools, sys
A=load('devA'); B=load('devB'); RA=load_ref('devA')
base=[s for s in STATS if AX[s]!='R']
core=['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4']
def make(stats,dead,agg,ql,kappa,W,eps):
    R=axis_dev(RA[1],stats,dead,0,agg)   # standard bucket reference (all ladder rungs are standard length)
    q={a:np.quantile(R[a],ql) for a in 'SCI'}
    def f(Z):
        X=axis_dev(Z,stats,dead,0,agg); D=np.zeros(len(Z))
        for a in 'SCI':
            e=np.maximum(0,X[a]-q[a]); D+=W[a]*np.log1p(e/kappa)+eps*X[a]
        return D
    return f,q
def accs(by,f):
    sc={k:-f(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
res=[]
for (sn,st),dead,agg,ql,kappa,(wi,ws,wc),eps in itertools.product({'all':base,'core':core}.items(),[0.5,1,1.5],['sum','max'],[0.9,0.95,0.98,0.99],[0.5,1,4],[(4,2,1),(8,4,1),(3,2,1),(16,4,1),(2,2,1)],[0,0.01,0.1]):
    f,q=make(st,dead,agg,ql,kappa,{'I':wi,'S':ws,'C':wc},eps)
    a=accs(A,f)
    res.append((sum(x>=0.95 for x in a),min(a),sn,dead,agg,ql,kappa,(wi,ws,wc),eps,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:20]: print(r)
print('--- top 12 on devB')
for r in res[:12]:
    _,_,sn,dead,agg,ql,kappa,w,eps,_=r
    f,q=make({'all':base,'core':core}[sn],dead,agg,ql,kappa,{'I':w[0],'S':w[1],'C':w[2]},eps)
    b=accs(B,f); print(sn,dead,agg,ql,kappa,w,eps,[round(x,3) for x in b],round(min(b),3))
