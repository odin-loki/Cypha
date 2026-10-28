import lib
from lib import *
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c2':1,'c3':-1})
def fix_s4(Z,f=0.05/0.15):
    Z=Z.copy(); k=STATS.index('s4'); z=Z[:,k]; m=z< -2.0537
    z[m]=-2.0537-(-2.0537-z[m])*f; Z[:,k]=z; return Z
A=load('devA'); B=load('devB'); RA=load_ref('devA')
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
def accs(by,f,names=LADDER):
    sc={k:-f(by[k]) for k in names}
    return [acc(sc[a],sc[b]) for a,b in zip(names,names[1:])]
