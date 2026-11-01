# BKN4 (docs/reports/BYTE_KN.md): qa's answer gate re-read offline on dev from two rows files.
# Usage: python3 scripts/bkn4_dev_gate.py ON.rows.tsv OFF.rows.tsv  (BKN3's QONS and QOFFS)
import csv,sys
def load(p):
    R={}
    for r in csv.DictReader(open(p),delimiter='\t'):
        if r['set']=='dev' and r['form']=='chat' and r['lm']=='trie_ig': R[int(r['row'])]=r
    return R
on=load(sys.argv[1]); off=load(sys.argv[2])
def fv(r): return [float(r['mean_lp']),float(r['min_lp']),-float(r['entropy0']),float(r['ig_n0']),float(r['question_lp'])]
def choose(v):
    n=len(v); best=(-1,0,0,0,0,1e300)  # r,w,ans,f,tau
    bf,bt,br,bw,ba=-1,1e300,0,0,0
    for f in range(5):
        s=sorted([(x[0][f],x[1]) for x in v if x[1]!=1],key=lambda t:-t[0]); r=w=0
        for k,(val,g) in enumerate(s):
            if g==0: r+=1
            else: w+=1
            if k+1<len(s) and s[k+1][0]==val: continue
            if w>0.02*n: break
            a=k+1
            if r>br or (r==br and r>0 and (w<bw or (w==bw and a<ba))): br,bw,ba,bf,bt=r,w,a,f,val
    return bf,bt,br
def cv(items):
    t=[0,0,0]
    for fold in range(5):
        tr=[(x[1],x[2]) for x in items if x[0]%5!=fold]; f,tau,_=choose(tr)
        for row,feat,g in items:
            if row%5==fold:
                gg=g if (g!=1 and f>=0 and feat[f]>=tau) else 1
                t[0 if gg==0 else (1 if gg==1 else 2)]+=1
    f,tau,r=choose([(x[1],x[2]) for x in items])
    return 'right %d abst %d wrong %d | fulldev f=%d tau=%.4f right=%d'%(t[0],t[1],t[2],f,tau,r)
arms={}
arms['on']=[(k,fv(on[k]),int(on[k]['grade'])) for k in on]
arms['off']=[(k,fv(off[k]),int(off[k]['grade'])) for k in off]
arms['agree(on text, abstain if off text differs)']=[(k,fv(on[k]),int(on[k]['grade']) if on[k]['answer']==off[k]['answer'] else 1) for k in on]
same=sum(on[k]['answer']==off[k]['answer'] for k in on)
print('n',len(on),'same answer text',same)
for a,v in arms.items(): print(a,':',cv(v))
print('on right rows',[k for k in on if on[k]['grade']=='0'],'off right rows',[k for k in off if off[k]['grade']=='0'])
# both gates: on text emitted iff same text, on passes its CV gate and off passes its CV gate (each fit on the same 4 folds)
t=[0,0,0]
for fold in range(5):
    fo,to,_=choose([(fv(on[k]),int(on[k]['grade'])) for k in on if k%5!=fold])
    ff,tf,_=choose([(fv(off[k]),int(off[k]['grade'])) for k in off if k%5!=fold])
    for k in on:
        if k%5!=fold: continue
        g=int(on[k]['grade']); a,b=fv(on[k]),fv(off[k])
        ok=g!=1 and on[k]['answer']==off[k]['answer'] and fo>=0 and a[fo]>=to and ff>=0 and int(off[k]['grade'])!=1 and b[ff]>=tf
        gg=g if ok else 1; t[0 if gg==0 else (1 if gg==1 else 2)]+=1
print('both gates : right %d abst %d wrong %d'%tuple(t))
