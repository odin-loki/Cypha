import lib
from lib import *
lib.SIDE.update({'c0':1,'c1':1,'c1a':1,'c3':-1})
A=load('devA'); R=load_ref('devA')
core=['s1','s2','s3','s3min','s4','wfreq','c0','c1','c1a','c3','loop8','rep','d1','d2','d3','d4']
q=[0.05,0.25,0.5,0.75,0.95]
for dead in [1.0]:
    for k in ['human','sent_shuffled','word_shuffled','random_lexicon','random_letters','repeated_phrase','trie']:
        X=axis_dev(A[k],core,dead,0,'sum')
        print('%-16s'%k,' '.join('%s:%s'%(a,np.round(np.quantile(X[a],q),1)) for a in 'SCI'))
    X=axis_dev(R[1],core,dead,0,'sum'); print('%-16s'%'REF std',' '.join('%s:%s'%(a,np.round(np.quantile(X[a],[0.5,0.9,0.95,0.99]),1)) for a in 'SCI'))
# per stat phi quantiles for word_shuffled and sent_shuffled
for k in ['word_shuffled','sent_shuffled']:
    P=bad(A[k],1.0)
    print(k, {s: np.round(np.quantile(P[:,STATS.index(s)],[0.1,0.5,0.9]),1).tolist() for s in core})
