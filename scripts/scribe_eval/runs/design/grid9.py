from v2x import *
import itertools
def make(sst,ist,cst,qs,qi,capS,kap,wI,cdead,dead=1):
    stats=sst+ist
    R=ax(RA[1],stats,dead,1,'sum'); th={'S':np.quantile(R['S'],qs),'I':np.quantile(R['I'],qi)}
    def f(Z):
        X=ax(Z,stats,dead,1,'sum'); Cx=ax(Z,cst,cdead,1,'sum')['C']
        es=np.maximum(0,X['S']-th['S']); ei=np.maximum(0,X['I']-th['I'])
        return capS*(1-np.exp(-es/kap))+wI*ei+Cx
    return f
SS={'g4':['s1','s2g','s3g','s4'],'g6':['s1','s2','s3','s2g','s3g','s4'],'g3':['s1','s2g','s3g']}
IS={'I7':['loop8','rep','d1','d2','d3','d4','i1']}
CS={'c5':['c0','c1','c1a','c2','c3'],'c4':['c0','c1','c1a','c2'],'c3':['c1','c1a','c2'],'c013':['c0','c1','c1a','c3']}
res=[]
for (sn,sst),(iname,ist),(cn,cst),qs,qi,capS,kap,cdead in itertools.product(SS.items(),IS.items(),CS.items(),[0.95,0.98,0.99],[0.98,0.99],[15,20,30,50],[1,2,4,8],[0.5,1]):
    a=accs(A,make(sst,ist,cst,qs,qi,capS,kap,1,cdead)); b=accs(B,make(sst,ist,cst,qs,qi,capS,kap,1,cdead))
    res.append((min(min(a),min(b)),sn,iname,cn,qs,qi,capS,kap,cdead,[round(x,3) for x in a],[round(x,3) for x in b]))
res.sort(key=lambda r:-r[0])
for r in res[:30]: print(r)
# robustness: how many configs pass both
print('configs passing both dev ladders:',sum(r[0]>=0.95 for r in res),'of',len(res))
