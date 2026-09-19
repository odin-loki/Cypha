#!/usr/bin/env python3
"""Diff Cypha hp compile profiles against vendored v78_flags.ps1 (v82 champ protocol)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "native/third_party/hp/include/hp/features.hpp"
V78_PS1 = ROOT / "native/third_party/hp/tools/v78_flags.ps1"


def parse_features_defaults(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"#ifndef (HP_[A-Z0-9_]+)\s*\n#define \1\s+([^\n/]+)", text):
        out[m.group(1)] = m.group(2).strip().split()[0]
    return out


def parse_v78_flags(text: str) -> dict[str, str]:
    return dict(re.findall(r"-D(HP_[A-Z0-9_]+)=([0-9]+)", text))


def main() -> int:
    defaults = parse_features_defaults(FEATURES.read_text())
    v78 = parse_v78_flags(V78_PS1.read_text())
    light = dict(defaults)
    light["HP_SLOT_MAX"] = "24"
    gate24 = dict(defaults)
    gate24.update(v78)
    gate24["HP_SLOT_MAX"] = "24"
    champ = dict(defaults)
    champ.update(v78)

    non_slot = [k for k in sorted(v78) if k != "HP_SLOT_MAX"]
    light_match = sum(1 for k in non_slot if light.get(k) == v78[k])
    gate24_match = sum(1 for k in non_slot if gate24.get(k) == v78[k])
    champ_match = sum(1 for k in non_slot if champ.get(k) == v78[k])

    print(f"v78_flags.ps1 HP -D count: {len(v78)}")
    print(f"Cypha light matches v82 (non-SLOT): {light_match}/{len(non_slot)}")
    print(f"Cypha gate24 matches v82 (non-SLOT): {gate24_match}/{len(non_slot)}")
    print(f"Cypha champ matches v82 (non-SLOT): {champ_match}/{len(non_slot)}")
    print("HP_SLOT_MAX: light=24 gate24=24 champ=35 v82=35")
    print()
    print("flag\tdefault\tlight\tgate24\tchamp\tlight_ok\tgate24_ok\tchamp_ok")
    for k in sorted(v78):
        d = defaults.get(k, "?")
        lv = light.get(k, "?")
        gv = gate24.get(k, v78[k])
        cv = champ.get(k, v78[k])
        v = v78[k]
        print(
            f"{k}\t{d}\t{lv}\t{gv}\t{cv}\t"
            f"{'Y' if lv == v else 'N'}\t{'Y' if gv == v else 'N'}\t{'Y' if cv == v else 'N'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
