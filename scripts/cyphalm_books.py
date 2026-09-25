#!/usr/bin/env python3
"""PG-19 book data for the CyphaLM winner (docs/reports/CYPHALM_LM_QUALITY_REPORT.md, "Winner v4").

  cyphalm_books.py fetch OUT_DIR ENWIK8
      Download the books named in models/cyphalm_winner/pg19_*_ids.txt and the
      six test books from the public PG-19 bucket, check them against the
      digests below and write:
        books_a.txt          289 train books (110 MB), the book shard's data
        books_b.txt          715 train books (299 MB), a random sample
        wiki95_books.txt     enwik8[:95 MB] + books_a, the default ∞-gram datastore
        wiki95_books500.txt  enwik8[:95 MB] + books_a + books_b, the books tier's datastore
        mix_wiki_books.txt   100 MiB of interleaved 1 MiB wiki and book blocks + 64 KiB
                             validation, what the book-aware experts were fine-tuned on
        pg19test/*.txt       test books (evaluation only; never in any training data)

  cyphalm_books.py sample OUT_DIR CYPHALM_TRACE
      Re-derive the id lists as they were made: books A are the first books of the
      train listing (name order, books <= 3 MB) up to 110 MB; books B are a seeded
      random sample of train books <= 4 MB not in A, up to 300 MB; both without the
      Alice / Looking-Glass editions and without any book that shares a >= 200-byte
      span or > 0.1 % of its bytes (64-byte matches, cyphalm_trace) with the
      Canterbury alice29.txt and lcet10.txt used for evaluation (the original files
      in bench/data/canterbury/cantrbry.tar.gz).
"""
import hashlib, json, os, random, subprocess, sys, tarfile, time, urllib.request

BUCKET = "https://storage.googleapis.com/deepmind-gutenberg"
LISTING = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?maxResults=1000&prefix="
HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, "..", "models", "cyphalm_winner")
TEST = ["33756", "30981", "9931", "57791", "54624", "56410"]  # 33756 tunes, 9931 tests
ALICE = {"11", "12", "28885", "114", "19033", "19002"}
SHA = {
    "books_a.txt": "8840e658a3c1eb28729d1465132e5b5887c58b83af24ce10a946c686b45b6e26",
    "books_b.txt": "fc6f1883aa7d4f8a56599367a3cd2d2b9702f42f82675b0821ce482b6e6b9573",
    "wiki95_books.txt": "e3c3302c8fe4f921a1868dc8b675f7d906ccec1973218375b4c54ad509db3583",
    "mix_wiki_books.txt": "e511bb9104593be853353f12a755f865dc9426a2c09ae7e7192dc76e339c692b",
}


def retry(fn):
    for attempt in range(7):
        try:
            return fn()
        except Exception as e:  # connection resets happen; back off and retry
            if attempt == 6:
                raise
            print("retry:", e, file=sys.stderr)
            time.sleep(2 ** attempt)


def _drop_part(part):
    try:
        if os.path.exists(part):
            os.remove(part)
    except OSError:
        pass


def download(split, book_id, dst_dir):
    dst = os.path.join(dst_dir, book_id + ".txt")
    part = dst + ".part"
    if os.path.exists(dst):
        _drop_part(part)
        return dst
    os.makedirs(dst_dir, exist_ok=True)

    def grab():
        urllib.request.urlretrieve(f"{BUCKET}/{split}/{book_id}.txt", part)
        if os.path.exists(dst):
            _drop_part(part)
            return
        os.replace(part, dst)

    retry(grab)
    return dst


def read_ids(name):
    return [l.strip() for l in open(os.path.join(IDS, name)) if l.strip()]


def concat(out_path, parts):
    with open(out_path, "wb") as o:
        for p in parts:
            o.write(p if isinstance(p, bytes) else open(p, "rb").read())


def check(path):
    want = SHA.get(os.path.basename(path))
    if want is None:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != want:
        raise SystemExit(f"{path}: sha256 {h.hexdigest()} != {want}")


def fetch(out, enwik8):
    a = [download("train", i, os.path.join(out, "pg19")) for i in read_ids("pg19_train_ids.txt")]
    b = [download("train", i, os.path.join(out, "pg19")) for i in read_ids("pg19_sample_ids.txt")]
    for i in TEST:
        download("test", i, os.path.join(out, "pg19test"))
    wiki = open(enwik8, "rb").read(95_000_000)
    concat(os.path.join(out, "books_a.txt"), a)
    concat(os.path.join(out, "books_b.txt"), b)
    concat(os.path.join(out, "wiki95_books.txt"), [wiki] + a)
    concat(os.path.join(out, "wiki95_books500.txt"), [wiki] + a + b)
    # 50 x (1 MiB of wiki, 1 MiB of books) spread evenly over both, then 64 KiB
    # of held-out wiki (95.5 MB) as the validation tail byte_lm.py reads.
    books = open(os.path.join(out, "books_b.txt"), "rb").read() + open(os.path.join(out, "books_a.txt"), "rb").read()
    block, parts = 1 << 20, []
    for k in range(50):
        parts.append(wiki[k * (95_000_000 // 50):][:block])
        parts.append(books[k * (len(books) // 50):][:block])
    with open(enwik8, "rb") as f:
        f.seek(95_500_000)
        parts.append(f.read(65536))
    concat(os.path.join(out, "mix_wiki_books.txt"), parts)
    for name in SHA:
        check(os.path.join(out, name))
    print(f"fetched {len(a)} + {len(b)} train books and {len(TEST)} test books into {out}; digests match")


def listing(prefix):
    items, tok = [], None
    while True:
        url = LISTING + prefix + (f"&pageToken={tok}" if tok else "")
        d = retry(lambda: json.load(urllib.request.urlopen(url, timeout=60)))
        items += d.get("items", [])
        tok = d.get("nextPageToken")
        if not tok:
            return items


def contaminated(trace, probe, path):
    d = json.loads(subprocess.run([trace, "--corpus", probe, "--text", path, "--min-len", "64"],
                                  capture_output=True, text=True, check=True).stdout)
    return d["longest_match"] >= 200 or d["coverage"] > 0.001


def sample(out, trace):
    # The evaluation reads the original files (CRLF), so probe with those.
    tar = tarfile.open(os.path.join(HERE, "..", "bench", "data", "canterbury", "cantrbry.tar.gz"))
    probe = os.path.join(out, "probe.txt")
    os.makedirs(out, exist_ok=True)
    concat(probe, [tar.extractfile(n).read() for n in ("alice29.txt", "lcet10.txt")])
    train = listing("train/")
    pool = os.path.join(out, "pg19")
    name = lambda it: it["name"].split("/")[-1][:-4]
    a, total = [], 0
    for it in train[:3000]:  # the listing is in name order
        if total >= 110_000_000:
            break
        if int(it["size"]) > 3_000_000 or name(it) in ALICE:
            continue
        a.append(name(it))
        total += int(it["size"])
    rng = random.Random(19)
    rng.shuffle(train)
    b, total = [], 0
    for it in train:
        if name(it) in ALICE or name(it) in a or int(it["size"]) > 4_000_000:
            continue
        if total + int(it["size"]) > 300_000_000:
            break
        b.append(name(it))
        total += int(it["size"])
    keep = lambda ids: [i for i in ids if not contaminated(trace, probe, download("train", i, pool))]
    a, b = keep(a), sorted(keep(b))  # books_b is concatenated in name order
    open(os.path.join(out, "pg19_train_ids.txt"), "w").write("\n".join(a) + "\n")
    open(os.path.join(out, "pg19_sample_ids.txt"), "w").write("\n".join(b) + "\n")
    print(f"{len(a)} + {len(b)} ids written to {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "sample":
        sample(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
