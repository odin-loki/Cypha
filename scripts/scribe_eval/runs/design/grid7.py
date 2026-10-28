from v2x import *
import itertools
def make(sst,ist,cst,qs,qi,capS,kap,wI,cdead,cagg,dead=1):
    stats=sst+ist
    R=ax(RA[1],stats,dead,1,'sum'); th={'S':np.quantile(R['S'],qs),'I':np.quantile(R['I'],qi)}
    def f(Z):
        X=ax(Z,stats,dead,1,'sum'); Cx=ax(Z,cst,cdead,1,cagg)['C']
        es=np.maximum(0,X['S']-th['S']); ei=np.maximum(0,X['I']-th['I'])
        return capS*(1-np.exp(-es/kap))+wI*ei+Cx
    return f
SS={'S4':['s1','s2','s3','s4'],'S6':['s1','s2','s3','s3min','s4','wfreq'],'S8':['s1','s2','s3','s3min','s4','sl_mean','sl_max','wfreq']}
IS={'I6':['loop8','rep','d1','d2','d3','d4']}
CS={'c1c1a':['c1','c1a'],'c013':['c0','c1','c1a','c3'],'c01':['c0','c1','c1a']}
res=[]
for (sn,sst),(iname,ist),(cn,cst),qs,qi,capS,kap,wI,cdead,cagg in itertools.product(SS.items(),IS.items(),CS.items(),[0.95,0.98,0.99],[0.98,0.995],[8,10,15,20,30],[0.5,1,2,4],[1],[0,0.5,1],['sum','mean']):
    a=accs(A,make(sst,ist,cst,qs,qi,capS,kap,wI,cdead,cagg))
    res.append((sum(x>=0.95 for x in a),round(min(a),3),sn,iname,cn,qs,qi,capS,kap,wI,cdead,cagg,[round(x,3) for x in a]))
res.sort(key=lambda r:(-r[0],-r[1]))
for r in res[:20]: print(r)
print('--- devB')
for r in res[:30]:
    b=accs(B,make(SS[r[2]],IS[r[3]],CS[r[4]],*r[5:12])); print(r[2:12],[round(v,3) for v in b],round(min(b),3), 'A',r[12])
