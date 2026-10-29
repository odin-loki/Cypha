#!/usr/bin/env python3
"""DK2 (docs/reports/DK2_DONT_KNOW_TRAINING.md): rewrite the don't-know training text of a `cyphalm_trie dktext` file.

Each row's response is "I don't know. I have nothing about E in what I've read. <|unknown reason=R entity="E"
relation=L gap=G|>". Options (applied in this order):
  --quote        the message quotes its entity in curly quotes: "... nothing about “E” in what I've read."
                 (“ and ” are rare in the corpus, so the byte after a copied entity's ” points back to this text)
  --entity-last  the tag ends with its entity: "<|unknown reason=R relation=L gap=G entity="E"|>" (A1d's elast form)
Questions, entities, relations and reasons are unchanged. Every response must match the form, or the script stops.

usage: dk2_text.py IN.tsv OUT.tsv [--quote] [--entity-last]
"""
import re
import sys

MSG = re.compile(r"^(I don't know\. I have nothing about )(.*?)( in what I've read\. )(<\|unknown .*\|>)$")
TAG = re.compile(r'^<\|unknown reason=(\S+) entity="([^"]*)" relation=(\S+) gap=(G-[0-9A-F]+)\|>$')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2 or not opts <= {"--quote", "--entity-last"}:
        sys.exit(__doc__)
    src, dst = args
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as o:
        head = f.readline().rstrip("\n")
        cols = head.split("\t")
        ir = cols.index("response")
        o.write(head + "\n")
        for line in f:
            c = line.rstrip("\n").split("\t")
            m = MSG.match(c[ir])
            t = TAG.match(m.group(4)) if m else None
            if not m or not t:
                sys.exit(f"row {n + 2}: response not in dktext's form: {c[ir][:120]}")
            ent = m.group(2)
            if "--quote" in opts:
                ent = "“" + ent + "”"
            reason, tent, rel, gap = t.groups()
            tag = (f'<|unknown reason={reason} relation={rel} gap={gap} entity="{tent}"|>' if "--entity-last" in opts
                   else f'<|unknown reason={reason} entity="{tent}" relation={rel} gap={gap}|>')
            c[ir] = m.group(1) + ent + m.group(3) + tag
            o.write("\t".join(c) + "\n")
            n += 1
    print(f"dk2_text {src} -> {dst} rows {n} quote {int('--quote' in opts)} entity_last {int('--entity-last' in opts)}")


if __name__ == "__main__":
    main()
