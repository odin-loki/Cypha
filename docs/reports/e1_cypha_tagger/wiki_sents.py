# 300 sentences of Wikipedia prose from enwik8 (bytes 90,000,000 to 95,000,000), one token per line, blank line
# between sentences. Markup lines are skipped; [[a|b]] becomes b. Seeded order.
import re, sys, random
src, out = sys.argv[1], sys.argv[2]
with open(src, 'rb') as f:
    f.seek(90_000_000)
    raw = f.read(5_000_000).decode('utf-8', 'ignore')
sents = []
for line in raw.split('\n'):
    line = line.strip()
    if not line or line[0] in '[{|*=<!:#;&' or 'http' in line or '{{' in line or '}}' in line or '&lt;' in line:
        continue
    line = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", line)
    line = line.replace("'''", "").replace("''", "")
    if '[' in line or ']' in line or '|' in line:
        continue
    for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line):
        toks = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]", s)
        if 8 <= len(toks) <= 40:
            sents.append(toks)
random.Random(20261029).shuffle(sents)
with open(out, 'w') as f:
    for t in sents[:300]:
        f.write('\n'.join(t) + '\n\n')
print('sentences kept', min(300, len(sents)), 'of', len(sents))
