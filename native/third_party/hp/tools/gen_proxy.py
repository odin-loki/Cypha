#!/usr/bin/env python3
"""Generate a MediaWiki-XML-like mixed-regime proxy for fast A.3 gates."""
import argparse
import random
import sys

ARTICLES = [
    "Alan_Turing", "Entropy", "Context_mixing", "Hutter_Prize",
    "Kolmogorov_complexity", "Arithmetic_coding", "Pitman-Yor",
    "Hebbian_theory", "Wiki_markup", "Infobox",
]
WORDS = (
    "the of and to in a is that for on with as by it from or at this "
    "which are was be has have not were can all their more also first "
    "compression model context mixer probability byte order match word "
    "table link citation template paragraph algorithm integer exact"
).split()


def wiki_article(rng, name, n_paras):
    out = [f'<page>\n<title>{name.replace("_", " ")}</title>\n<text>']
    out.append(f"{{{{Infobox person|name={name}|birth=1912|death=1954}}}}\n")
    for p in range(n_paras):
        if p % 4 == 0:
            out.append('{| class="wikitable"\n')
            for r in range(4):
                cells = "|".join(rng.choice(WORDS) for _ in range(5))
                out.append(f"|-\n| {cells}\n")
            out.append("|}\n")
        elif p % 4 == 1:
            for _ in range(6):
                tgt = rng.choice(ARTICLES)
                out.append(f"See also [[{tgt}|{tgt.replace('_', ' ')}]] and "
                           f"http://example.org/{tgt}. ")
            out.append("\n")
        elif p % 4 == 2:
            year = 1900 + rng.randint(0, 99)
            out.append(f"On 12:30, {year}-0{rng.randint(1,9)}-1{rng.randint(0,9)} "
                       f"the &amp;nbsp; source said <ref>Smith 2001</ref>. ")
            out.append("''Italics'' and '''bold''' and {{cite book|title=X}}.\n")
        else:
            sent = " ".join(rng.choice(WORDS) for _ in range(80))
            out.append(sent[0].upper() + sent[1:] + ".\n")
    out.append("</text>\n</page>\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--bytes", type=int, default=262144)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    chunks = ['<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">\n']
    i = 0
    while sum(len(c) for c in chunks) < args.bytes:
        chunks.append(wiki_article(rng, ARTICLES[i % len(ARTICLES)] + str(i), 8))
        i += 1
    chunks.append("</mediawiki>\n")
    data = "".join(chunks).encode("utf-8")[: args.bytes]
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} B to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
