import json,re,numpy as np,math
from lib import acc
SP='/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad'
cnt={}
for i,l in enumerate(open(f'{SP}/work/v1/v32.vocab.tsv')):
    w,c=l.rstrip('\n').split('\t'); cnt[w]=float(c)
N=sum(cnt.values())
rank={w:i for i,w in enumerate(sorted(cnt,key=lambda w:-cnt[w]))}
def toks(s): return re.findall(r"[a-z0-9']+",s.lower())
def sents(t): return [x for x in re.split(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z0-9"(])',t) if x.strip()]
def content(ws): return {w for w in ws if rank.get(w,10**9)>=150 and not w.isdigit() and len(w)>2}
def idf(w): return math.log(N/(cnt.get(w,1.0)))
def stats(t):
    ss=[content(toks(s)) for s in sents(t)]
    ss=[s for s in ss if s]
    if len(ss)<2: return dict(adj=np.nan,allp=np.nan,share=np.nan)
    adj=[]; allp=[]
    for i in range(len(ss)):
        for j in range(i+1,len(ss)):
            inter=ss[i]&ss[j]; wv=sum(idf(w) for w in inter); den=min(sum(idf(w) for w in ss[i]),sum(idf(w) for w in ss[j]))
            v=wv/den if den>0 else 0
            allp.append(v)
            if j==i+1: adj.append(v)
    # share of sentences sharing a content word with some other sentence
    share=np.mean([any(ss[i]&ss[j] for j in range(len(ss)) if j!=i) for i in range(len(ss))])
    return dict(adj=np.mean(adj),allp=np.mean(allp),share=share)
for d in ['devA','devB']:
    T={}
    for l in open(f'{SP}/work/v2dev/{d}/texts.jsonl'):
        r=json.loads(l); T.setdefault(r['system'],[]).append(r['text'])
    S={k:[stats(t) for t in T[k]] for k in ['human','sent_shuffled','word_shuffled','random_lexicon']}
    for key in ['adj','allp','share']:
        v={k:np.array([x[key] for x in S[k]]) for k in S}
        f=lambda a: a[~np.isnan(a)]
        print(d,key,'means',{k:round(float(np.nanmean(v[k])),3) for k in v},'human>sent %.3f'%acc(f(v['human']),f(v['sent_shuffled'])),'word>lex %.3f'%acc(f(v['word_shuffled']),f(v['random_lexicon'])))
