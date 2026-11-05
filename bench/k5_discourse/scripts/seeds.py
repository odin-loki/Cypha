import sys, re, statistics as st
sys.argv=[sys.argv[0]]; exec(open('/mnt/d/cypha_llm/disc/collect.py').read().split('labels =')[0])
def f(x): return float(x.split()[0])
base = None
for arm in ['B0','T2','T2b','T2c','T6a','E1','X2','M3']:
    rs=[row(f'{arm}_s{s}') for s in (1000,2000,3000,4000)]
    if any('err' in r for r in rs): print(arm,'incomplete'); continue
    v31=[f(r['v31']) for r in rs]; v2=[f(r['v2']) for r in rs]; cp=[float(r['copy32']) for r in rs]; v13=[int(r['v13']) for r in rs]
    d=dict(arm=arm, v31=v31, v2=v2, m31=st.mean(v31), m2=st.mean(v2), cp=st.mean(cp), v13=sum(v13), lo=min(v31), hi=max(v31))
    if arm=='B0': base=d
    rule='' 
    if base and arm!='B0':
        ok1 = d['m31']>base['m31'] and d['lo']>base['hi']; ok2 = d['m2']>=base['m2']-0.02; ok3 = d['v13']==0 and d['cp']<=base['cp']+0.02
        rule=f"rule1 {'yes' if ok1 else 'no'} rule2 {'yes' if ok2 else 'no'} rule3 {'yes' if ok3 else 'no'} (copy +{d['cp']-base['cp']:.4f})"
    print(f"{arm}: v3.1 {' '.join('%.3f'%x for x in v31)} mean {d['m31']:.3f} range [{d['lo']:.3f}, {d['hi']:.3f}] | v2 {' '.join('%.3f'%x for x in v2)} mean {d['m2']:.3f} | copy32 {d['cp']:.4f} verbatim13 {d['v13']} | {rule}")
