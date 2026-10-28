import lib
from lib import *
import itertools
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c3':-1})
def fix_s4(Z,f=0.05/0.15):
    Z=Z.copy(); k=STATS.index('s4'); z=Z[:,k]; m=z< -2.0537
    z[m]=-2.0537-(-2.0537-z[m])*f; Z[:,k]=z; return Z
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
def make(sst,ist,cst,dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg):
    stats=sst+ist
    R=ax(RA[1],stats,dead,p,agg); th={'S':np.quantile(R['S'],qs),'I':np.quantile(R['I'],qi)}
    def f(Z):
        X=ax(Z,stats,dead,p,agg); Cx=ax(Z,cst,cdead,cp,cagg)['C']
        return wS*np.maximum(0,X['S']-th['S'])+wI*np.maximum(0,X['I']-th['I'])+Cx
    return f
def accs(by,f):
    sc={k:-f(by[k]) for k in LADDER}
    return [acc(sc[a],sc[b]) for a,b in zip(LADDER,LADDER[1:])]
SS={'S4':['s1','s2','s3','s4'],'S6':['s1','s2','s3','s3min','s4','wfreq'],'S8':['s1','s2','s3','s3min','s4','sl_mean','sl_max','wfreq']}
IS={'I6':['loop8','rep','d1','d2','d3','d4'],'I7':['loop8','rep','d1','d2','d3','d4','i1']}
CS={'c1':['c1'],'c1c1a':['c1','c1a'],'c013':['c0','c1','c1a','c3']}
res=[]
for (sn,sst),(iname,ist),(cn,cst),dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg in itertools.product(SS.items(),IS.items(),CS.items(),[1],[1,2],['sum','max'],[0.95,0.98,0.99,0.995],[0.98,0.995],[1,3,10],[10],[0,0.5],[1,2],['sum','mean']):
    if cn=='c1' and cagg=='mean': continue
    a=accs(A,make(sst,ist,cst,dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg))
    res.append((sum(x>=0.95 for x in a),round(min(a),3),sn,iname,cn,dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:25]: print(r)
print('--- devB')
for r in res[:25]:
    _,_,sn,iname,cn,dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg,_=r
    b=accs(B,make(SS[sn],IS[iname],CS[cn],dead,p,agg,qs,qi,wS,wI,cdead,cp,cagg)); print(r[2:15],[round(v,3) for v in b],round(min(b),3))
