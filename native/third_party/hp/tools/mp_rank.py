#!/usr/bin/env python3
"""
mp_rank.py -- effective-rank analysis of the expert ensemble.

Marchenko-Pastur does NOT apply at gamma = p/n ~ 4e-5: the noise bulk
collapses and every eigenvalue reads as signal. This script still prints the
MP count (and flags the regime), but the numbers that matter are:

  participation ratio   (sum λ)^2 / sum λ^2
  entropy effective rank  exp(H(λ / sum λ))

Both live in the p << n regime. Use those before/after every model addition.
"""

import math
import sys
import numpy as np


# Names matching Predictor::predict() add order (bias is mixer-only, not dumped).
# dump_experts writes expert_p[0..n_exp) which skips bias (n_exp reset).
DEFAULT_NAMES = [
    "o1:ind", "o1:py", "o2:ind", "o2:py", "o3:ind", "o3:py",
    "o4:ind", "o4:py", "o6:ind", "o6:py",
    "word:ind", "word:py", "sp13:ind", "sp13:py", "sp24:ind", "sp24:py",
    "col:ind", "col:py", "tag:ind", "tag:py", "wbi:ind", "wbi:py",
    "wstr1:ind", "wstr1:py", "wstrsp:ind", "wstrsp:py",
    "brk:ind", "brk:py", "pat:ind", "pat:py",
    "m3", "m4", "m6", "m10", "m16",
    "wm1", "wm2", "wm3",
    "hebb", "ctw",
]


def load(path, n_experts):
    a = np.fromfile(path, dtype="<i2")
    w = n_experts + 1
    a = a[: (len(a) // w) * w].reshape(-1, w)
    return a[:, :n_experts].astype(np.float64), a[:, n_experts].astype(np.float64)


def mp_threshold(eigs, n, p):
    gamma = p / n
    s2 = np.median(eigs) / (1.0 + gamma)
    for _ in range(20):
        lam_p = s2 * (1 + np.sqrt(gamma)) ** 2
        bulk = eigs[eigs <= lam_p]
        if len(bulk) < 2:
            break
        s2_new = bulk.mean()
        if abs(s2_new - s2) < 1e-12:
            break
        s2 = s2_new
    return s2 * (1 + np.sqrt(gamma)) ** 2, s2


def participation_ratio(eigs):
    s = eigs.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / np.square(eigs).sum())


def entropy_rank(eigs):
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    return float(math.exp(h))


def names_for(p):
    names = list(DEFAULT_NAMES)
    # remaining slots are discovery-pool pairs
    i = 0
    while len(names) < p:
        names.append(f"disc{i}:ind")
        if len(names) < p:
            names.append(f"disc{i}:py")
        i += 1
    return names[:p]


def main():
    path = sys.argv[1]
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    X, y = load(path, p)
    n = X.shape[0]
    gamma = p / n
    print(f"samples n={n}  experts p={p}  gamma={gamma:.6f}")
    if gamma < 0.01:
        print("NOTE: Marchenko-Pastur does not apply at this gamma "
              "(noise bulk collapses). Use PR and entropy rank.\n")

    Xc = X - X.mean(0)
    sd = Xc.std(0)
    sd[sd == 0] = 1.0
    Z = Xc / sd

    C = (Z.T @ Z) / n
    eigs = np.linalg.eigvalsh(C)[::-1]
    lam_p, s2 = mp_threshold(eigs.copy(), n, p)
    n_sig = int((eigs > lam_p).sum())
    pr = participation_ratio(eigs)
    er = entropy_rank(eigs)

    print(f"MP noise edge lambda+ = {lam_p:.4f}  (sigma^2 = {s2:.4f})")
    print(f"MP effective rank      = {n_sig} of {p}   [unreliable at this gamma]")
    print(f"participation ratio    = {pr:.3f} of {p}")
    print(f"entropy effective rank = {er:.3f} of {p}")
    print(f"top-1 variance share   = {100.0 * eigs[0] / eigs.sum():.1f}%\n")

    print("eigenvalue spectrum:")
    for i, e in enumerate(eigs):
        tag = "SIGNAL" if e > lam_p else "noise"
        bar = "#" * min(60, int(e * 6))
        print(f"  {i:2d}  {e:8.4f}  {tag:6s} {bar}")

    names = names_for(p)
    print("\nper-expert |correlation| with every other (mean, max):")
    R = np.corrcoef(Z.T)
    np.fill_diagonal(R, 0.0)
    order = np.argsort(-np.abs(R).mean(1))
    for i in order:
        j = int(np.argmax(np.abs(R[i])))
        print(f"  {names[i]:12s} mean={np.abs(R[i]).mean():.3f}  "
              f"max={np.abs(R[i]).max():.3f} (with {names[j]})")

    print("\nindividual predictive value -- corr(expert, true bit):")
    yc = (y - y.mean()) / (y.std() if y.std() else 1)
    for i in np.argsort(-np.abs(Z.T @ yc / n)):
        print(f"  {names[i]:12s} {float(Z[:, i] @ yc / n):+.4f}")

    print("\nredundant pairs (|r| > 0.90):")
    found = False
    for i in range(p):
        for j in range(i + 1, p):
            if abs(R[i, j]) > 0.90:
                print(f"  {names[i]:12s} ~ {names[j]:12s}  r={R[i,j]:+.4f}")
                found = True
    if not found:
        print("  none")


if __name__ == "__main__":
    main()
