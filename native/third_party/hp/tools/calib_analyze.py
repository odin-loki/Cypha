import json, csv, math, os, sys
H=os.path.dirname(os.path.abspath(__file__))
def rd(p):
    if not os.path.exists(p): return []
    return list(csv.DictReader(open(p)))
def spearman(a,b):
    n=len(a)
    if n<3: return float('nan')
    def rank(x):
        s=sorted(range(n), key=lambda i:x[i]); r=[0]*n
        i=0
        while i<n:
            j=i
            while j+1<n and x[s[j+1]]==x[s[i]]: j+=1
            for k in range(i,j+1): r[s[k]]=(i+j)/2.0
            i=j+1
        return r
    ra,rb=rank(a),rank(b)
    ma,mb=sum(ra)/n,sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da=math.sqrt(sum((x-ma)**2 for x in ra)); db=math.sqrt(sum((x-mb)**2 for x in rb))
    return num/(da*db) if da and db else float('nan')
def pearson(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((x-mb)**2 for x in b))
    return num/(da*db) if da and db else float('nan')

truth={r['flag']:r['d8mib'] for r in json.load(open(f'{H}/calib.json'))}
prox=rd(f'{H}/calib_proxy.csv'); m8=rd(f'{H}/calib_8mib.csv')
if prox and prox[0]['flag']=='BASE':
    bw=int(prox[0]['win12']); bh=int(prox[0]['head2mb'])
    rows=[(r['flag'], int(r['win12'])-bw, int(r['head2mb'])-bh, int(truth[r['flag']]))
          for r in prox[1:] if r['flag'] in truth and r['win12'].isdigit()]
    if rows:
        print(f"{'flag':22s} {'win12_d':>8s} {'head2mb_d':>10s} {'RECORD_8MiB_d':>14s}")
        for f,w,h,t in rows: print(f"{f:22s} {w:+8d} {h:+10d} {t:+14d}")
        W=[r[1] for r in rows]; Hd=[r[2] for r in rows]; T=[r[3] for r in rows]
        print(f"\nn={len(rows)}")
        print(f"win12   vs RECORD:  spearman={spearman(W,T):+.3f}  pearson={pearson(W,T):+.3f}")
        print(f"head2mb vs RECORD:  spearman={spearman(Hd,T):+.3f}  pearson={pearson(Hd,T):+.3f}")
        print(f"win12   vs head2mb: spearman={spearman(W,Hd):+.3f}")
        sg=sum(1 for w,t in zip(W,T) if (w>0)==(t>0))
        print(f"sign agreement win12 vs RECORD: {sg}/{len(rows)} = {100*sg/len(rows):.0f}%")
        sg2=sum(1 for h,t in zip(Hd,T) if (h>0)==(t>0))
        print(f"sign agreement head2mb vs RECORD: {sg2}/{len(rows)} = {100*sg2/len(rows):.0f}%")
        z=sum(1 for w in W if w==0); print(f"proxy-blind flags (win12 delta == 0): {z}/{len(rows)}")
if m8 and m8[0]['flag']=='BASE':
    b=int(m8[0]['bytes'])
    print(f"\n--- 8 MiB @ SLOT_MAX=24 (local) vs RECORD @ SLOT_MAX=35 ---")
    print(f"BASE local = {b:,}   RECORD v82 = 1,689,157   delta = {b-1689157:+,}")
    rr=[(r['flag'], int(r['bytes'])-b, int(truth[r['flag']])) for r in m8[1:] if r['flag'] in truth]
    for f,d,t in rr: print(f"{f:22s} local={d:+6d}  RECORD={t:+6d}  ratio={d/t if t else float('nan'):+.2f}")
    if len(rr)>=3:
        A=[x[1] for x in rr]; B=[x[2] for x in rr]
        print(f"local8 vs RECORD8: spearman={spearman(A,B):+.3f} pearson={pearson(A,B):+.3f}")
        sg=sum(1 for a,bb in zip(A,B) if (a>0)==(bb>0)); print(f"sign agreement: {sg}/{len(rr)}")
