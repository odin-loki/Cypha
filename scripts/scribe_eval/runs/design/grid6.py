import lib
from lib import *
import itertools
exec(open('grid5.py').read().split("def make(")[0].split("A={k:fix_s4")[0])
A={k:fix_s4(v) for k,v in load('devA').items()}; B={k:fix_s4(v) for k,v in load('devB').items()}
RA={b:fix_s4(v) for b,v in load_ref('devA').items()}
def U(Z):
    out=np.zeros_like(Z)
    for k,s in enumerate(STATS):
        z=Z[:,k]; out[:,k]=np.maximum(0,-z) if SIDE[s]>0 else (np.maximum(0,z) if SIDE[s]<0 else np.abs(z))
    out[np.isnan(Z)]=0; return out
def ax(Z,stats,dead,p,agg):
    u=np.maximum(0,U(Z)-dead)**p; out={}
    for a in 'SCI':
        idx=[STATS.index(s) for s in stats if AX[s]==a]; P=u[:,idx]
        out[a]=np.zeros(len(Z)) if not idx else (P.sum(1) if agg=='sum' else P.max(1) if agg=='max' else P.mean(1))
    return out
def make(sst,ist,cst,qs,qi,capS,wI,cdead,cagg,iform):
    stats=sst+ist
    R=ax(RA[1],stats,1,1,'sum'); th={'S':np.quantile(R['S'],qs),'I':np.quantile(R['I'],qi)}
    def f(Z):
        X=ax(Z,stats,1,1,'sum'); Cx=ax(Z,cst,cdead,1,cagg)['C']
        es=np.maximum(0,X['S']-th['S']); ei=np.maximum(0,X['I']-th['I'])
        S_eff=capS*(1-np.exp(-es/capS))
        I_eff=wI*ei if iform=='lin' else wI*np.log1p(ei)
        return S_eff+I_eff+Cx
    return f
def accs(by,f,names=LADDER):
    sc={k:-f(by[k]) for k in names}
    return [acc(sc[a],sc[b]) for a,b in zip(names,names[1:])]
SS={'S4':['s1','s2','s3','s4'],'S6':['s1','s2','s3','s3min','s4','wfreq'],'S8':['s1','s2','s3','s3min','s4','sl_mean','sl_max','wfreq']}
IS={'I6':['loop8','rep','d1','d2','d3','d4']}
CS={'c1':['c1'],'c1c1a':['c1','c1a'],'c013':['c0','c1','c1a','c3']}
res=[]
for (sn,sst),(iname,ist),(cn,cst),qs,qi,capS,wI,cdead,cagg,iform in itertools.product(SS.items(),IS.items(),CS.items(),[0.95,0.98,0.99],[0.98,0.995],[3,5,10,20],[1,10],[0,0.5],['sum','mean'],['lin','log']):
    if cn=='c1' and cagg=='mean': continue
    a=accs(A,make(sst,ist,cst,qs,qi,capS,wI,cdead,cagg,iform))
    res.append((sum(x>=0.95 for x in a),round(min(a),3),sn,iname,cn,qs,qi,capS,wI,cdead,cagg,iform,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:25]: print(r)
print('--- devB')
for r in res[:25]:
    b=accs(B,make(SS[r[2]],IS[r[3]],CS[r[4]],*r[5:12])); print(r[2:12],[round(v,3) for v in b],round(min(b),3))
