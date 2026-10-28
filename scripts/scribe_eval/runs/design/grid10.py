from v2x import *
import itertools
def esum(Z,stats,dead=1):
    u=np.maximum(0,U(Z)-dead); idx=[STATS.index(s) for s in stats]
    return u[:,idx].sum(1) if idx else np.zeros(len(Z))
def make(syn,lex,ist,cst,qs,qi,cap,kap,wI,cdead,ref=None):
    ref=RA[1] if ref is None else ref
    th={'syn':np.quantile(esum(ref,syn),qs),'lex':np.quantile(esum(ref,lex),qs),'I':np.quantile(esum(ref,ist),qi)}
    def f(Z):
        D=cap*(1-np.exp(-np.maximum(0,esum(Z,syn)-th['syn'])/kap))
        D+=cap*(1-np.exp(-np.maximum(0,esum(Z,lex)-th['lex'])/kap))
        D+=wI*np.maximum(0,esum(Z,ist)-th['I'])
        D+=esum(Z,cst,cdead)
        return D
    return f
SYN={'g3':['s2g','s3g','s4'],'g2':['s2g','s3g'],'g5':['s2','s3','s2g','s3g','s4']}
IS={'I7':['loop8','rep','d1','d2','d3','d4','i1']}
CS={'c5':['c0','c1','c1a','c2','c3'],'c4':['c0','c1','c1a','c2'],'c3':['c1','c1a','c2'],'c013':['c0','c1','c1a','c3']}
if __name__=="__main__":
  res=[]
  for (sn,syn),(cn,cst),qs,qi,cap,kap,wI,cdead in itertools.product(SYN.items(),CS.items(),[0.98,0.99,0.995],[0.98,0.99],[20,30,50],[1,2,4],[1,3],[0.5,1]):
      f=make(syn,['s1'],IS['I7'],cst,qs,qi,cap,kap,wI,cdead)
      a=accs(A,f); b=accs(B,f)
      res.append((min(min(a),min(b)),sn,cn,qs,qi,cap,kap,wI,cdead,[round(x,3) for x in a],[round(x,3) for x in b]))
  res.sort(key=lambda r:-r[0])
  for r in res[:30]: print(r)
  print('configs passing both dev ladders (all 5 pairs >= 0.95):',sum(r[0]>=0.95 for r in res),'of',len(res))
