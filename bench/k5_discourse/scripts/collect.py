import sys, re, os, glob
out = '/mnt/d/cypha_llm/disc/out'
def row(label):
    r = {'label': label}
    try:
        s31 = open(f'{out}/{label}.scribe31.md').read()
        m = re.search(r'^\| trie samples \(pooled\) \| (\d+) \| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\|', s31.split('## SCRIBE v3: the three judges')[1], re.M)
        r['v2'], r['S31'], r['M31'], r['v31'] = m.group(2).strip(), m.group(7).strip(), m.group(8).strip(), m.group(9).strip()
        ax = re.search(r'^\| trie samples \(pooled\) \| \d+ \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \| ([\d.]+) \|', s31.split('## Axis profile')[1], re.M)
        r['D'], r['D_S'], r['D_C'] = ax.groups()
    except Exception as e:
        r['err'] = str(e)
    try:
        so = open(f'{out}/{label}.sample.out').read()
        m = re.search(r'copy32 T1.0 ([\d.]+) verbatim13 [\d.]+ \((\d+) of', so); r['copy32'], r['v13'] = m.groups()
        m = re.search(r'k5_cont_bpb ([\d.]+)', so); r['bpb'] = m.group(1) if m else ''
        se = open(f'{out}/{label}.sample.err').read()
        m = re.search(r'generated \d+ bytes in ([\d.]+) s on (\d+) threads', se); r['gen_s'] = m.group(1) if m else ''
        m = re.search(r'LDA.*\(([\d.]+) s\)', se); r['lda_s'] = m.group(1) if m else ''
        m = re.search(r'MBR kept drafts of(.*)', se); r['pool'] = m.group(1).strip() if m else ''
        t = open(f'{out}/{label}.time').read()
        r['wall'] = re.search(r'Elapsed \(wall clock\) time.*: (\S+)', t).group(1)
        r['rss_gb'] = '%.2f' % (int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', t).group(1)) / 1e6)
    except Exception as e:
        r.setdefault('err', str(e))
    return r
labels = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in glob.glob(f'{out}/*.done'))
cols = ['label', 'v2', 'v31', 'S31', 'M31', 'D', 'D_S', 'D_C', 'copy32', 'v13', 'bpb', 'gen_s', 'wall', 'rss_gb', 'lda_s', 'pool']
print('| ' + ' | '.join(cols) + ' |')
for l in labels:
    r = row(l)
    print('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |' + (' ERR ' + r['err'] if 'err' in r else ''))
