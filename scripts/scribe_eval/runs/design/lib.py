import json, numpy as np
SP='/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad'
STATS=['s1','s2','s2g','s3','s3g','s3min','s4','sl_mean','sl_max','wfreq','c0','c1','c1a','c2','c3','loop8','rep','d1','d2','d3','d4','i1','r1','r4','r5','i4']
AX={'s1':'S','s2':'S','s2g':'S','s3g':'S','c2':'C','s3':'S','s3min':'S','s4':'S','sl_mean':'S','sl_max':'S','wfreq':'S','c0':'C','c1':'C','c1a':'C','c3':'C','loop8':'I','rep':'I','d1':'I','d2':'I','d3':'I','d4':'I','i1':'I','r1':'R','r4':'R','r5':'R','i4':'R'}
SIDE={'s1':1,'s2':1,'s2g':1,'s3g':1,'c2':1,'s3':1,'s3min':1,'s4':1,'sl_mean':0,'sl_max':-1,'wfreq':0,'c0':0,'c1':0,'c1a':0,'c3':0,'loop8':-1,'rep':-1,'d1':0,'d2':0,'d3':0,'d4':0,'i1':0,'r1':1,'r4':1,'r5':0,'i4':-1}
LADDER=['human','sent_shuffled','word_shuffled','random_lexicon','random_letters','repeated_phrase']
def load(d):
    rows=[json.loads(l) for l in open(f'{SP}/work/v2dev/{d}/scores.jsonl')]
    by={}
    for r in rows:
        z=r['v2']['z']
        v=np.array([z.get(s,np.nan) if z.get(s) is not None else np.nan for s in STATS],float)
        by.setdefault(r['system'],[]).append(v)
    return {k:np.array(v) for k,v in by.items()}
def acc(a,b):
    a=np.asarray(a);b=np.sort(np.asarray(b))
    lo=np.searchsorted(b,a,'left');hi=np.searchsorted(b,a,'right')
    return float((lo+0.5*(hi-lo)).sum()/(len(a)*len(b)))
def boot_ci(x,B=1000,seed=1):
    r=np.random.default_rng(seed);x=np.asarray(x)
    m=np.array([x[r.integers(0,len(x),len(x))].mean() for _ in range(B)])
    return np.quantile(m,0.025),np.quantile(m,0.975)
def bad(Z,dead=1.0,cap=0):
    U=np.zeros_like(Z)
    for k,s in enumerate(STATS):
        z=Z[:,k]
        u= np.maximum(0,-z) if SIDE[s]>0 else (np.maximum(0,z) if SIDE[s]<0 else np.abs(z))
        U[:,k]=u
    phi=np.maximum(0,U-dead)**2
    if cap>0: phi=cap*(1-np.exp(-phi/cap))
    phi[np.isnan(Z)]=0
    return phi
def deviance(Z,stats=None,dead=1.0,cap=0,agg='sum',w=None,tf=None):
    stats=stats or [s for s in STATS if AX[s]!='R']
    phi=bad(Z,dead,cap)
    if tf: phi=tf(phi)
    w=w or {'S':1,'C':1,'I':1}
    D=np.zeros(len(Z))
    for a in 'SCI':
        idx=[STATS.index(s) for s in stats if AX[s]==a]
        if not idx: continue
        P=phi[:,idx]
        if agg=='sum': Da=P.sum(1)
        elif agg=='mean': Da=P.mean(1)
        elif agg=='max': Da=P.max(1)
        elif agg=='rss': Da=np.sqrt((P).sum(1))
        D+=w.get(a,1)*Da
    return D
def ladder(by,D_of,tau=4,show=True,names=LADDER):
    sc={k:np.exp(-D_of(by[k])/tau) for k in names}
    res=[];npass=0
    for a,b in zip(names,names[1:]):
        ac=acc(sc[a],sc[b]); ca=boot_ci(sc[a]); cb=boot_ci(sc[b]); dj=ca[0]>cb[1]
        p=ac>=0.95 and dj; npass+=p
        res.append((a,b,round(ac,3),dj,p))
    if show:
        for r in res: print('  %-16s > %-16s %.3f %s %s'%(r[0],r[1],r[2],'disj' if r[3] else 'OVERLAP','PASS' if r[4] else 'FAIL'))
    return npass,res
def load_ref(d):
    rows=[json.loads(l) for l in open(f'{SP}/work/v2dev/{d}/reference.jsonl')]
    out={0:[],1:[]}
    for r in rows:
        b=0 if r['raw']['bytes']<300 else 1
        z=r['v2']['z']
        out[b].append([z.get(s,np.nan) if z.get(s) is not None else np.nan for s in STATS])
    return {b:np.array(v,float) for b,v in out.items()}
def axis_dev(Z,stats,dead=1.0,cap=0,agg='sum'):
    phi=bad(Z,dead,cap); out={}
    for a in 'SCI':
        idx=[STATS.index(s) for s in stats if AX[s]==a]
        P=phi[:,idx]
        out[a]= P.sum(1) if agg=='sum' else (P.mean(1) if agg=='mean' else P.max(1))
    return out
