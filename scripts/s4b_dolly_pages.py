#!/usr/bin/env python3
"""S4b's planted pool D (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b"): Dolly responses written as dump pages.

Every 10th Dolly record (file order) whose response has at least 256 bytes becomes one <page> in the enwik8 dump's
form: title "Dolly NNNNN" (the record's line number), the response as the body, escaped as the dump escapes text
(& < > "). pattern_learn --signals and cyphalm_trie surprise read the file with their own page parsers.

usage: s4b_dolly_pages.py DOLLY.jsonl OUT.xml
"""
import json
import sys


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    kept = qualifying = 0
    with open(sys.argv[1], encoding="utf-8") as f, open(sys.argv[2], "w", encoding="utf-8", newline="\n") as out:
        out.write("<mediawiki>\n")
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            resp = json.loads(line).get("response", "")
            if len(resp.encode("utf-8")) < 256:
                continue
            qualifying += 1
            if (qualifying - 1) % 10:
                continue
            kept += 1
            out.write("  <page>\n    <title>Dolly %05d</title>\n    <id>0</id>\n    <revision>\n      <id>0</id>\n"
                      "      <timestamp>2006-03-01T00:00:00Z</timestamp>\n      <contributor>\n        <username>dolly</username>\n"
                      "        <id>0</id>\n      </contributor>\n      <text xml:space=\"preserve\">%s</text>\n    </revision>\n"
                      "  </page>\n" % (line_no, esc(resp)))
        out.write("</mediawiki>\n")
    print("dolly pages %d of %d qualifying responses" % (kept, qualifying))
    return 0


if __name__ == "__main__":
    sys.exit(main())
