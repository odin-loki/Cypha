# JSONL documents -> <page><revision><text> XML for ppmi_svd (v2 code README). Author: Odin Loch
import json, sys, html
with open(sys.argv[2], "w") as o:
    for l in open(sys.argv[1]):
        t = json.loads(l)["text"]
        o.write("<page><revision><text>" + html.escape(t, quote=False) + "</text></revision></page>\n")
