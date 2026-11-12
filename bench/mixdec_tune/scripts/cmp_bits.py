"""MIXDEC_TUNE part 1: each measurement's primed and neutral bits a byte against M0, paired bootstrap over prompts.
usage: cmp_bits.py DIR BASE LABEL...  (reads DIR/LABEL.txt's mixdec_eval_row lines)"""
import random
import re
import sys


def rows(path):
    r = {}
    for line in open(path, errors="replace"):
        m = re.match(r"mixdec_eval_row (\d+) (\d+) ([\d.]+) ([\d.]+)", line)
        if m:
            r[int(m.group(1))] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return r


def main():
    d, base, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
    B = rows(f"{d}/{base}.txt")
    ids = sorted(B)
    print("| Change | primed bits a byte | − M0 [95%] | neutral | gain (neutral − primed) |")
    print("|---|---:|---:|---:|---:|")
    for lab in [base] + labels:
        try:
            R = rows(f"{d}/{lab}.txt")
        except FileNotFoundError:
            continue
        if sorted(R) != ids:
            print(f"| {lab} | (incomplete: {len(R)} rows) | | | |")
            continue
        nb = [R[i][0] for i in ids]
        tot = sum(nb)
        pr = sum(R[i][0] * R[i][2] for i in ids) / tot
        ne = sum(R[i][0] * R[i][1] for i in ids) / tot
        dl = [R[i][0] * (R[i][2] - B[i][2]) for i in ids]
        dm = sum(dl) / tot
        rng = random.Random(20261113)
        bs = []
        for _ in range(2000):
            s = [rng.randrange(len(ids)) for _ in ids]
            bs.append(sum(dl[j] for j in s) / sum(nb[j] for j in s))
        bs.sort()
        ci = "—" if lab == base else f"{dm:+.4f} [{bs[50]:+.4f}, {bs[1949]:+.4f}]"
        print(f"| {lab} | {pr:.4f} | {ci} | {ne:.4f} | {ne - pr:.4f} |")


main()
