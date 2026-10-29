#!/usr/bin/env python3
"""FCAST: compare event models from forecast_vocab_bench outputs (docs/reports/FORECAST_EVENT_VOCAB.md).

Usage: forecast_vocab_compare.py DIR NAME [NAME ...] [--ref NAME] [--block 1000] [--resamples 2000] [--seed 20261029]

Reads DIR/NAME.loss (-log2 P per scored event, aligned across names) and DIR/NAME.sub.tsv. Prints, per name: bits
per event with a block-bootstrap 95% interval, top-1 accuracy, Brier, expected calibration error of the top-1
probability (15 equal-width bins), and the raw-draw invalid share; then each name minus --ref with its interval.
"""
import random
import sys


def read_loss(path):
    with open(path, encoding="utf-8") as f:
        return [float(x) for x in f if x.strip()]


def read_sub(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
    return rows


def block_means(x, block):
    return [sum(x[i:i + block]) / len(x[i:i + block]) for i in range(0, len(x), block)]


def boot(blocks, resamples, rng):
    n = len(blocks)
    out = []
    for _ in range(resamples):
        out.append(sum(blocks[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(0.025 * resamples)], out[int(0.975 * resamples) - 1]


def ece(rows, bins=15):
    tot = [0] * bins
    acc = [0.0] * bins
    conf = [0.0] * bins
    for r in rows:
        p = float(r["p_top1"])
        b = min(bins - 1, int(p * bins))
        tot[b] += 1
        acc[b] += int(r["correct"])
        conf[b] += p
    n = sum(tot)
    return sum(abs(acc[b] - conf[b]) for b in range(bins) if tot[b]) / n if n else float("nan")


def main(argv):
    args = argv[2:]
    names, ref, block, resamples, seed = [], None, 1000, 2000, 20261029
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--ref":
            ref = args[i + 1]; i += 2; continue
        if a == "--block":
            block = int(args[i + 1]); i += 2; continue
        if a == "--resamples":
            resamples = int(args[i + 1]); i += 2; continue
        if a == "--seed":
            seed = int(args[i + 1]); i += 2; continue
        names.append(a); i += 1
    d = argv[1]
    loss = {n: read_loss(f"{d}/{n}.loss") for n in names}
    sub = {n: read_sub(f"{d}/{n}.sub.tsv") for n in names}
    lens = {len(v) for v in loss.values()}
    if len(lens) != 1:
        print("loss files differ in length:", {n: len(v) for n, v in loss.items()})
        return 1
    print("| Model | Bits per event [95%] | Top-1 | Brier | ECE | Raw draws invalid |")
    print("|---|---|---|---|---|---|")
    for n in names:
        rng = random.Random(seed)
        bl = block_means(loss[n], block)
        lo, hi = boot(bl, resamples, rng)
        s = sub[n]
        top1 = sum(int(r["correct"]) for r in s) / len(s)
        brier = sum(float(r["brier"]) for r in s) / len(s)
        raw_bad = sum(1 - int(r["raw_valid"]) for r in s) / len(s)
        print(f"| {n} | {sum(loss[n]) / len(loss[n]):.4f} [{lo:.4f}, {hi:.4f}] | {top1:.4f} | {brier:.4f} | "
              f"{ece(s):.4f} | {raw_bad:.4f} |")
    if ref:
        print()
        print(f"| Model minus {ref} | Bits per event [95%] |")
        print("|---|---|")
        for n in names:
            if n == ref:
                continue
            diff = [a - b for a, b in zip(loss[n], loss[ref])]
            rng = random.Random(seed)
            lo, hi = boot(block_means(diff, block), resamples, rng)
            print(f"| {n} | {sum(diff) / len(diff):+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
