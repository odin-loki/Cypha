#!/usr/bin/env python3
"""Validate Cypha's thought-process library and export it.

  python3 scripts/thought_check.py                       # validate docs/thought/processes.yaml
  python3 scripts/thought_check.py --index docs/thought/INDEX.md
  python3 scripts/thought_check.py --uml ach             # PlantUML state diagram (text) for one process
  python3 scripts/thought_check.py --mermaid ach         # the same as Mermaid text
  python3 scripts/thought_check.py --json docs/thought/processes.json   # validated library as JSON (for C++)

Checks: unique ids; known categories; required fields; start step exists; every
op is a primitive or "call:<existing process>"; every `next` target exists;
steps without `next` are `return`; every step is reachable from start; every
step can reach a final step. Model and maths: docs/THOUGHT.md.
Author: Odin Loch.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("needs PyYAML: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "docs", "thought", "processes.yaml")
REQUIRED = ["id", "name", "cat", "ref", "when", "start", "steps", "stop", "math", "feedback"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(lib):
    errors = []
    prims = set(lib.get("primitives", {}))
    cats = set(lib.get("categories", {}))
    procs = lib.get("processes", [])
    ids = [p.get("id") for p in procs]
    dup = [k for k, n in collections.Counter(ids).items() if n > 1]
    for d in dup:
        errors.append(f"duplicate id {d}")
    idset = set(ids)
    calls = collections.Counter()
    for p in procs:
        pid = p.get("id", "?")
        for k in REQUIRED:
            if k not in p:
                errors.append(f"{pid}: missing {k}")
        if p.get("cat") not in cats:
            errors.append(f"{pid}: unknown category {p.get('cat')}")
        steps = p.get("steps") or {}
        if p.get("start") not in steps:
            errors.append(f"{pid}: start '{p.get('start')}' is not a step")
            continue
        for sid, s in steps.items():
            if not isinstance(s, dict) or "op" not in s:
                errors.append(f"{pid}.{sid}: step needs an op")
                continue
            op = str(s["op"])
            if op.startswith("call:"):
                tgt = op[5:]
                if tgt not in idset:
                    errors.append(f"{pid}.{sid}: calls unknown process {tgt}")
                calls[tgt] += 1
            elif op not in prims:
                errors.append(f"{pid}.{sid}: unknown primitive {op}")
            nxt = s.get("next") or []
            for n in nxt:
                if n not in steps:
                    errors.append(f"{pid}.{sid}: next '{n}' is not a step")
            if not nxt and op != "return":
                errors.append(f"{pid}.{sid}: final step must be 'return' (or add next)")
            if nxt and op == "return":
                errors.append(f"{pid}.{sid}: 'return' step cannot have next")
        # reachability from start
        seen, stack = set(), [p["start"]]
        while stack:
            u = stack.pop()
            if u in seen or u not in steps:
                continue
            seen.add(u)
            stack.extend((steps[u] or {}).get("next") or [])
        for sid in steps:
            if sid not in seen:
                errors.append(f"{pid}.{sid}: unreachable from start")
        # every step can reach a final step
        finals = {sid for sid, s in steps.items() if not (s or {}).get("next")}
        rev = collections.defaultdict(list)
        for sid, s in steps.items():
            for n in (s or {}).get("next") or []:
                rev[n].append(sid)
        can, stack = set(), list(finals)
        while stack:
            u = stack.pop()
            if u in can:
                continue
            can.add(u)
            stack.extend(rev[u])
        for sid in steps:
            if sid not in can:
                errors.append(f"{pid}.{sid}: cannot reach a final step")
    return errors, calls


def write_json(lib, path):
    """Write the validated library as JSON (the C++ interpreter has no YAML parser).

    Key order follows the YAML. Refuses non-string ids or step targets (YAML 1.1 turns a bare
    yes/no/on/off into a boolean, which json would silently rewrite as true/false).
    """
    for p in lib["processes"]:
        for sid, s in p["steps"].items():
            ok = isinstance(sid, str) and all(isinstance(n, str) for n in (s.get("next") or []))
            if not ok or not isinstance(p["start"], str):
                sys.exit(f"{p['id']}.{sid}: step ids, next and start must be strings for JSON export")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=1)
        f.write("\n")


def index_md(lib, calls):
    cats = lib["categories"]
    by = collections.defaultdict(list)
    for p in lib["processes"]:
        by[p["cat"]].append(p)
    out = ["# Thought-process index", "",
           "Generated by `python3 scripts/thought_check.py --index docs/thought/INDEX.md` from "
           "[`processes.yaml`](processes.yaml). Model and maths: [`../THOUGHT.md`](../THOUGHT.md).", "",
           f"**{len(lib['processes'])} processes in {len(by)} categories, "
           f"{sum(len(p['steps']) for p in lib['processes'])} steps.**", ""]
    out.append("| Category | Processes |")
    out.append("|---|---|")
    for c in cats:
        if by.get(c):
            out.append(f"| `{c}`: {cats[c]} | {len(by[c])} |")
    out.append("")
    for c in cats:
        if not by.get(c):
            continue
        out.append(f"## {c}: {cats[c]}")
        out.append("")
        out.append("| id | Process | When | Core maths | Steps | Called by others |")
        out.append("|---|---|---|---|---:|---:|")
        for p in by[c]:
            math = str(p["math"]).replace("|", "\\|")
            when = str(p["when"]).replace("|", "\\|")
            out.append(f"| `{p['id']}` | {p['name']} ({p['ref']}) | {when} | {math} | {len(p['steps'])} | {calls.get(p['id'], 0)} |")
        out.append("")
    return "\n".join(out) + "\n"


def to_uml(p):
    lines = ["@startuml", f"title {p['name']} ({p['id']})", f"[*] --> {p['start']}"]
    for sid, s in p["steps"].items():
        label = s["op"] + (f": {s['do']}" if s.get("do") else "")
        lines.append(f"{sid} : {label}")
        nxt = s.get("next") or []
        for n in nxt:
            lines.append(f"{sid} --> {n}")
        if not nxt:
            lines.append(f"{sid} --> [*]")
    lines.append(f"note right of {p['start']} : stop: {p['stop']}")
    lines.append("@enduml")
    return "\n".join(lines)


def to_mermaid(p):
    lines = ["stateDiagram-v2", f"    [*] --> {p['start']}"]
    for sid, s in p["steps"].items():
        label = (s["op"] + (": " + s["do"] if s.get("do") else "")).replace(":", " -")
        lines.append(f"    {sid} : {label}")
        nxt = s.get("next") or []
        for n in nxt:
            lines.append(f"    {sid} --> {n}")
        if not nxt:
            lines.append(f"    {sid} --> [*]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=DEFAULT)
    ap.add_argument("--index", help="write the markdown index here")
    ap.add_argument("--json", metavar="OUT", help="write the validated library as JSON here")
    ap.add_argument("--uml", help="print a PlantUML state diagram for this process id")
    ap.add_argument("--mermaid", help="print a Mermaid state diagram for this process id")
    a = ap.parse_args()
    lib = load(a.lib)
    errors, calls = validate(lib)
    procs = {p["id"]: p for p in lib["processes"]}
    if a.uml or a.mermaid:
        pid = a.uml or a.mermaid
        if pid not in procs:
            sys.exit(f"no process {pid}")
        print(to_uml(procs[pid]) if a.uml else to_mermaid(procs[pid]))
        return
    for e in errors:
        print("ERROR", e)
    cats = collections.Counter(p["cat"] for p in lib["processes"])
    nsteps = sum(len(p["steps"]) for p in lib["processes"])
    print(f"{len(procs)} processes, {len(cats)} categories, {nsteps} steps, "
          f"{sum(calls.values())} sub-process calls, {len(errors)} errors")
    if a.index and not errors:
        with open(a.index, "w", encoding="utf-8") as f:
            f.write(index_md(lib, calls))
        print(f"index written to {a.index}")
    if a.json and not errors:
        write_json(lib, a.json)
        print(f"json written to {a.json}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
