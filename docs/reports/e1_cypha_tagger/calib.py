# Temperature-scaled confidence for a tagger's per-token scores (E1). Usage:
#   calib.py perc DEV_SCORES TEST_SCORES      (pos_tagger eval --scores rows; dictionary tokens get confidence 1)
#   calib.py cypha DEV_TSV TEST_TSV           (cypha_tagger --out rows)
# The temperature is fitted on dev (negative log-likelihood); reads on test: ECE (15 bins), accuracy at 90% / 80%
# coverage, AUROC of 1 - confidence for "the tag is wrong".
import sys, math, csv
def rows(path, kind):
    R = []
    with open(path) as f:
        rd = csv.reader(f, delimiter='\t'); hdr = next(rd)
        tags = hdr[4:] if kind == 'perc' else hdr[7:]
        for r in rd:
            gold, pred = r[0], r[1]
            if kind == 'perc':
                d = r[2] == '1'; sc = [float(x) for x in r[4:]]
            else:
                d = False; sc = [float(x) for x in r[7:]]
            R.append((gold, pred, d, sc, tags))
    return R
def probs(sc, T):
    m = max(sc); e = [math.exp((s - m) / T) for s in sc]; z = sum(e); return [x / z for x in e]
def fit_T(R):
    best = (1e300, 1.0)
    lt = -4.0
    while lt <= 6.0:
        T = math.exp(lt); nll = 0.0
        for g, p, d, sc, tags in R:
            if d: continue
            pr = probs(sc, T); nll -= math.log(max(1e-300, pr[tags.index(g)] if g in tags else 1e-300))
        best = min(best, (nll, T)); lt += 0.05
    return best[1]
def conf(R, T):
    C = []
    for g, p, d, sc, tags in R:
        C.append(1.0 if d else probs(sc, T)[tags.index(p)])
    return C
def ece(C, ok, B=15):
    s = [0.0]*B; a = [0.0]*B; n = [0]*B
    for c, o in zip(C, ok):
        b = min(B-1, int(c*B)); s[b] += c; a[b] += o; n[b] += 1
    return sum(abs(a[b]-s[b]) for b in range(B)) / len(C)
def acc_cov(C, ok, cov):
    idx = sorted(range(len(C)), key=lambda i: -C[i]); m = round(cov*len(C)); return sum(ok[i] for i in idx[:m]) / m
def auroc(s, pos):
    idx = sorted(range(len(s)), key=lambda i: s[i]); ranks = [0.0]*len(s); i = 0
    while i < len(idx):
        j = i
        while j < len(idx) and s[idx[j]] == s[idx[i]]: j += 1
        for k in range(i, j): ranks[idx[k]] = (i + j + 1) / 2.0
        i = j
    npos = sum(pos); nneg = len(pos) - npos
    return (sum(r for r, p in zip(ranks, pos) if p) - npos*(npos+1)/2) / (npos*nneg)
kind, dev, test = sys.argv[1], sys.argv[2], sys.argv[3]
D, X = rows(dev, kind), rows(test, kind)
T = fit_T(D)
C = conf(X, T); ok = [1 if g == p else 0 for g, p, *_ in X]
print(f"{kind}: T {T:.3f}  test acc {sum(ok)/len(ok)*100:.2f}%  ECE {ece(C, ok):.4f}  acc@90% {acc_cov(C, ok, 0.9)*100:.2f}%  acc@80% {acc_cov(C, ok, 0.8)*100:.2f}%  AUROC(1-conf, wrong) {auroc([1-c for c in C], [1-o for o in ok]):.4f}")
