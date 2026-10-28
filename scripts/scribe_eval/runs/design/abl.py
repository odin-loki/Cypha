import re,glob,os,sys
d=sys.argv[1]
rows=[]
for f in sorted(glob.glob(d+'/abl_*.md')):
    name=os.path.basename(f)[4:-3]
    txt=open(f).read()
    m=re.search(r'### control ladder \(no model rung\)(.*?)Verdict',txt,re.S)
    if not m: continue
    accs=re.findall(r'\| [^|]+ > [^|]+ \| ([0-9.]+) \| (yes|no) \| (PASS|FAIL)',m.group(1))
    hm=re.search(r'\| held-out human \| \d+ \| ([0-9.]+)',txt)
    tr=re.search(r'\| trie samples \(pooled\) \| \d+ \| ([0-9.]+)',txt)
    rows.append((name,[float(a[0]) for a in accs],sum(a[2]=='PASS' for a in accs),hm.group(1) if hm else '',tr.group(1) if tr else ''))
print('| variant | h>ss | ss>ws | ws>rl | rl>let | let>rep | passed | human | trie |')
print('|---|---|---|---|---|---|---|---|---|')
for n,a,p,h,t in rows: print('| %s | %s | %d | %s | %s |'%(n,' | '.join('%.3f'%x for x in a),p,h,t))
