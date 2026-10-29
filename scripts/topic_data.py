#!/usr/bin/env python3
"""C0/C7 topic data: 20 Newsgroups (bydate) and AG News as "label<TAB>text" files.

Data only; no model. Run with python3 -I. The inputs are untrusted downloads; only text is read.

  20 Newsgroups, bydate (http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz,
  sha256 8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610), unpacked:
    topic_data.py 20ng DIR_WITH_20news-bydate-{train,test} OUTDIR
  writes OUTDIR/20ng_{train,test}_{full,strip}.tsv. The label is the class index 0-19 in
  sorted folder order. "full" is the whole file (headers kept, as scikit-learn's default);
  "strip" removes the header block, quoting and the signature the way scikit-learn's
  remove=("headers", "footers", "quotes") does.

  AG News (Hugging Face fancyzhx/ag_news, revision eb185aade064a813bc0b7f42de02595523103ca4,
  data/{train,test}-00000-of-00001.parquet):
    topic_data.py ag PARQUET_DIR OUTDIR
  writes OUTDIR/ag_{train,test}.tsv with the label 0-3 (World, Sports, Business, Sci/Tech).

Text is put on one line: runs of white space become one space.
"""
import os
import re
import sys

_QUOTE_RE = re.compile(r"(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)")


def strip_header(text):
    _before, _blank, after = text.partition("\n\n")
    return after


def strip_quoting(text):
    return "\n".join(line for line in text.split("\n") if not _QUOTE_RE.search(line))


def strip_footer(text):
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        if lines[line_num].strip().strip("-") == "":
            break
    if line_num > 0:
        return "\n".join(lines[:line_num])
    return text


def one_line(text):
    return " ".join(text.split())


def do_20ng(src, out):
    for split in ("train", "test"):
        root = os.path.join(src, "20news-bydate-" + split)
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        assert len(classes) == 20, classes
        rows_full, rows_strip = [], []
        for ci, c in enumerate(classes):
            cdir = os.path.join(root, c)
            for name in sorted(os.listdir(cdir), key=lambda s: (len(s), s)):
                with open(os.path.join(cdir, name), "rb") as f:
                    text = f.read().decode("latin-1")
                rows_full.append((ci, one_line(text)))
                rows_strip.append((ci, one_line(strip_footer(strip_quoting(strip_header(text))))))
        for tag, rows in (("full", rows_full), ("strip", rows_strip)):
            path = os.path.join(out, "20ng_%s_%s.tsv" % (split, tag))
            with open(path, "w", encoding="utf-8") as f:
                for ci, t in rows:
                    f.write("%d\t%s\n" % (ci, t))
            print(path, len(rows))
    with open(os.path.join(out, "20ng_classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")


def do_ag(src, out):
    import pyarrow.parquet as pq
    for split in ("train", "test"):
        t = pq.read_table(os.path.join(src, split + ".parquet"), columns=["text", "label"])
        path = os.path.join(out, "ag_%s.tsv" % split)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for text, label in zip(t.column("text").to_pylist(), t.column("label").to_pylist()):
                f.write("%d\t%s\n" % (int(label), one_line(str(text))))
                n += 1
        print(path, n)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("20ng", "ag"):
        sys.exit(__doc__)
    os.makedirs(sys.argv[3], exist_ok=True)
    (do_20ng if sys.argv[1] == "20ng" else do_ag)(sys.argv[2], sys.argv[3])
