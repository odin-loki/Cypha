#!/usr/bin/env python3
"""Evaluation data for the SCRIBE paper (docs/SCRIBE_PAPER_PLAN.md, sections 3 and 7).

Author: Odin Loch

Downloads each public dataset the paper uses and converts it into JSONL files
under a data root D. Raw downloads go under a download root DL, one directory
per source; they are treated as untrusted data (run this script with python -I).

    python -I scripts/scribe_eval/data.py --data D --dl DL all
    python -I scripts/scribe_eval/data.py --data D --dl DL web      (one dataset)

Commands: web books wp hanna openmeva roc mauve dolly ud manifest all.
--data and --dl default to $SCRIBE_DATA and $SCRIBE_DL. Downloads are skipped
when the file is already there, so a rerun only rebuilds the outputs.

Text schema (every text file): one JSON object per line,
    {"id", "set", "system", "text", "prompt"?, "ref"?, "meta"?}
"system" is "human" for text written by people. Texts are UTF-8 plain prose:
markup removed, line ends normalised, paragraphs separated by "\\n\\n", wording
unchanged. Lone surrogates become U+FFFD; U+2028, U+2029, VT, FF and NEL
become newlines (so no JSONL reader splits a line); other control characters
are dropped.

Outputs
-------
web/      GPT-2 output dataset (OpenAI). Whole documents; SCRIBE cuts samples.
  pack.jsonl          webtext.train documents in file order until 40 MiB of text
                      (the downloaded prefix is DL/gpt2out/webtext.train.head.jsonl);
                      pack and ref leave out any document equal to a test text
                      (webtext.test or a GPT-2 sample: 3 such in the pack)
  ref.jsonl           all of webtext.valid (5,000)
  test.jsonl          all of webtext.test (5,000)
  gpt2/<config>.jsonl the 8 model test files (5,000 each, unconditional samples);
                      system = config name, meta = {length, ended, src_id}
books/    Project Gutenberg. BOOKS below lists 120 English prose novels (1811-1925),
  none of them in CyphaLM's PG-19 training lists (models/cyphalm_winner/).
  Split by book with random.Random(BOOK_SEED): 80 pack, 20 ref, 10 dev, 10 fresh.
  {pack,ref,dev,fresh}.jsonl  documents of about 20 KB cut at paragraph ends;
                      meta = {gutenberg_id, title, author, part, parts}.
  Cleaning: PG header and footer removed (the "*** START OF" / "*** END OF"
  lines); front matter before the body's first chapter heading and back matter
  after "THE END" removed; headings, tables of contents, verse and other
  blocks of lines shorter than 0.8 x the book's wrap width, indented blocks
  (in books whose prose is not indented), illustration and footnote markers
  dropped; hard-wrapped lines joined; _italics_ underscores removed.
wp/       WritingPrompts (Fan et al. 2018), from the original fairseq archive
  (euclaise/writingprompts on Hugging Face was parsed from the same archive with
  the same rows, but its partial detokenisation is lossy, e.g. "see 'Mr ." ->
  "see'Mr.", so the raw tokenised text is used and detok() is applied to WP,
  HANNA and OpenMEVA alike).
  pack.jsonl          train stories in file order until 40 MiB of text
  ref.jsonl           all validation stories
  Stories whose prompt (normalised) is also a test-split prompt are left out of
  both, since HANNA and OpenMEVA-WP prompts come from the test split.
hanna/    HANNA (Chhun et al. 2022, 2024), git clone of dig-team/hanna-benchmark-asg.
  texts.jsonl         1,056 stories: id "hanna/<Story ID>", system = Model
                      ("human" for Human), prompt, text, ref = the human story;
                      meta = {story_id, prompt_id (= Story ID mod 96, checked),
                      model, human: {RE..CX means}, raters: {RE..CX: [3]},
                      workers: [3 worker ids]}; raters in file row order.
  shipped_metrics.jsonl  one line per story: id, story_id, model and every
                      automatic metric and LLM score of hanna_metric_scores_llm.csv
                      under its own column name (NaN -> null). The human columns
                      of that CSV (means, "Human k", leave-one-out "Human Avg k")
                      are left out; they are in texts.jsonl or follow from it.
  The CSV has one row per model and each cell is a list over that model's 96
  stories. check_hanna_order() proves the order is ascending Story ID: for all
  11 models the 6 mean-rating lists equal the annotation means in Story ID
  order, the "Human k <crit>" lists equal the k-th annotation row of each
  story, and the automatic metric "Text length" has Spearman > 0.6 with the
  story's token count in every model (a one-story shift gives about 0).
openmeva/ OpenMEVA (Guan et al. 2021), Hugging Face Jiann/OpenMEVA.
  mans_{roc,wp}.jsonl  MANS: 200 prompts x 5 systems (s2s, fusion, plan_write,
                      gpt, gpt_kg) per corpus; prompt = the beginning (ROC: first
                      sentence; WP: the prompt), ref = gold_response (human),
                      meta.human = {overall: mean of the 5 ratings, scores: [5]}.
  autos_<corpus>_<aspect>.jsonl  AUTOS (called ACTS in the GitHub README), the
                      perturbation tests. Raw lines are "src|||context|||story|||note".
     Discrimination aspects (lexical_rept, semantic_rept, character, commonsense,
     relatedness (= "coherence" in the paper), consistency, cause, time):
     <aspect>.txt holds human stories picked for the aspect (label 1, system
     "human"), <aspect>_perturb.txt holds stories perturbed in that aspect
     (label 0, system "perturbed"). src indexes the corpus's test set. Positives
     and negatives are not paired; a negative carries ref = its original story
     when that story is one of the corpus's positives (any aspect) and
     meta.paired = true when it is a positive of the same file.
     Invariance aspects (synonym_substitute, semantic_substitute, delete_punct,
     contraction, typos): rationality-preserving perturbations of (a) human
     stories (src = test-set index) and (b) the negative samples
     (src = line of <corpus>_negative_sample.txt). Each perturbed line
     (meta.role "perturbed", label 0) is paired with its original
     (meta.role "original", label 1, as in OpenMEVA's robust.py) through
     meta.pair; base = "human" or "negative". A metric should score both
     alike. Originals of human stories come from the positives files (ROC
     falls back to roc_test_*.txt); those not found have no original line
     (meta.pair = null).
  All meta also have corpus, test, aspect, label, src, note (OpenMEVA's
  annotation of the perturbation), line (line in the raw file). Lines with an
  empty context or story are skipped, as OpenMEVA's benchmark code does.
  ROC text is lower-cased with [MALE]/[FEMALE]/[NEUTRAL] name masks; WP AUTOS
  text is lower-cased and truncated by OpenMEVA. Both kept as given.
roc/      Human ROC stories for a pack and reference, from OpenMEVA's own ROC
  splits (autos_data/roc/roc_{train,dev}_{ipt,opt}.txt: first sentence +
  four more). pack = train (88,344), ref = dev (4,908); a story whose first
  sentence is also a test-set first sentence or a MANS-ROC prompt is left out.
mauve_he/ MAUVE human evaluation (Pillutla et al. 2021), mauve_human_eval.csv.
  texts.jsonl         one line per distinct completion: id
                      "mauve_he/<idx>/<system>/<k>" (system with "_" for the
                      space), system "gpt2-xl p0.95" or "human", prompt =
                      Input.ctx and text = the completion with HTML removed
                      (</p><p> -> paragraph break); meta = {idx, variant k,
                      variants, len}. k > 0 exists because 703 (idx, system)
                      pairs were shown with different samples in different
                      HITs (and idx 635 with two contexts); human texts have
                      one variant each.
  pairs.jsonl         {pair_id, idx, a, b, q1, q2, q3, worker}: raw answer codes,
                      2a = definitely A, 1a = slightly A or tie, 1b = slightly B,
                      2b = definitely B. q1 more interesting, q2 makes more sense,
                      q3 more likely human.
dolly/    databricks-dolly-15k: dolly.jsonl with instruction, context, response,
          category, plus prompt = instruction (+ "\\n\\n" + context), text = response.
ud/       UD English EWT 2.x: en_ewt-ud-{train,dev,test}.conllu, as downloaded.
MANIFEST.tsv  path, lines, bytes, sha256, source, licence for every output file
          of this script (other tools' directories under D are not listed).
"""
import argparse
import csv
import collections
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------------
# Sources (pinned where the host allows it)

GPT2_URL = "https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/"
GPT2_CONFIGS = ["small-117M", "small-117M-k40", "medium-345M", "medium-345M-k40",
                "large-762M", "large-762M-k40", "xl-1542M", "xl-1542M-k40"]
PG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
WP_URL = "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz"
WP_SHA256 = "4b8d0415c450fe1ceadec667bdf73c185268bc3d9e1f6ddf64edac0481cb71f4"
HANNA_GIT = "https://github.com/dig-team/hanna-benchmark-asg"
HANNA_COMMIT = "282f27536a5d05ad4ce14298abcd70c45668fed2"
OPENMEVA_REPO = "Jiann/OpenMEVA"
OPENMEVA_REV = "5ee4fe59caafce625670f91d3983f5c1ddab8c1d"
DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
MAUVE_GDRIVE_ID = "1doC-NjhUlt4YDnK1qm23tyrfbogVlZ29"
MAUVE_SHA256 = "ffdc55424520538b1a70337a88aefeb23078856e85df1ea897fbae207d73c21f"
UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{}.conllu"

PACK_BYTES = 40 * 2**20          # web and wp packs: 40 MiB of text
BOOK_DOC_BYTES = 20_000          # book documents: about 20 KB
BOOK_SEED = 20261028
BOOK_SPLITS = [("pack", 80), ("ref", 20), ("dev", 10), ("fresh", 10)]

# 120 English-language prose novels, 1811-1925: (Gutenberg id, author, title).
# Ids were checked against the PG catalogue (pg_catalog.csv, language en) and
# against models/cyphalm_winner/pg19_*_ids.txt (no overlap).
BOOKS = [
    (1342, "Jane Austen", "Pride and Prejudice"), (158, "Jane Austen", "Emma"),
    (161, "Jane Austen", "Sense and Sensibility"), (105, "Jane Austen", "Persuasion"),
    (121, "Jane Austen", "Northanger Abbey"), (141, "Jane Austen", "Mansfield Park"),
    (1260, "Charlotte Brontë", "Jane Eyre"), (9182, "Charlotte Brontë", "Villette"),
    (768, "Emily Brontë", "Wuthering Heights"),
    (969, "Anne Brontë", "The Tenant of Wildfell Hall"), (767, "Anne Brontë", "Agnes Grey"),
    (98, "Charles Dickens", "A Tale of Two Cities"), (1400, "Charles Dickens", "Great Expectations"),
    (730, "Charles Dickens", "Oliver Twist"), (766, "Charles Dickens", "David Copperfield"),
    (786, "Charles Dickens", "Hard Times"), (580, "Charles Dickens", "The Pickwick Papers"),
    (963, "Charles Dickens", "Little Dorrit"), (700, "Charles Dickens", "The Old Curiosity Shop"),
    (145, "George Eliot", "Middlemarch"), (6688, "George Eliot", "The Mill on the Floss"),
    (550, "George Eliot", "Silas Marner"), (507, "George Eliot", "Adam Bede"),
    (7469, "George Eliot", "Daniel Deronda"),
    (110, "Thomas Hardy", "Tess of the d'Urbervilles"), (107, "Thomas Hardy", "Far from the Madding Crowd"),
    (153, "Thomas Hardy", "Jude the Obscure"), (143, "Thomas Hardy", "The Mayor of Casterbridge"),
    (122, "Thomas Hardy", "The Return of the Native"), (482, "Thomas Hardy", "The Woodlanders"),
    (84, "Mary Shelley", "Frankenstein"), (345, "Bram Stoker", "Dracula"),
    (174, "Oscar Wilde", "The Picture of Dorian Gray"),
    (43, "Robert Louis Stevenson", "The Strange Case of Dr. Jekyll and Mr. Hyde"),
    (421, "Robert Louis Stevenson", "Kidnapped"), (864, "Robert Louis Stevenson", "The Master of Ballantrae"),
    (36, "H. G. Wells", "The War of the Worlds"), (35, "H. G. Wells", "The Time Machine"),
    (5230, "H. G. Wells", "The Invisible Man"), (159, "H. G. Wells", "The Island of Doctor Moreau"),
    (718, "H. G. Wells", "Tono-Bungay"),
    (2852, "Arthur Conan Doyle", "The Hound of the Baskervilles"), (244, "Arthur Conan Doyle", "A Study in Scarlet"),
    (2097, "Arthur Conan Doyle", "The Sign of the Four"), (3289, "Arthur Conan Doyle", "The Valley of Fear"),
    (139, "Arthur Conan Doyle", "The Lost World"),
    (76, "Mark Twain", "Adventures of Huckleberry Finn"), (74, "Mark Twain", "The Adventures of Tom Sawyer"),
    (86, "Mark Twain", "A Connecticut Yankee in King Arthur's Court"),
    (1837, "Mark Twain", "The Prince and the Pauper"),
    (2701, "Herman Melville", "Moby Dick"), (33, "Nathaniel Hawthorne", "The Scarlet Letter"),
    (77, "Nathaniel Hawthorne", "The House of the Seven Gables"),
    (514, "Louisa May Alcott", "Little Women"),
    (209, "Henry James", "The Turn of the Screw"), (432, "Henry James", "The Ambassadors"),
    (208, "Henry James", "Daisy Miller"), (2870, "Henry James", "Washington Square"),
    (177, "Henry James", "The American"),
    (4217, "James Joyce", "A Portrait of the Artist as a Young Man"),
    (541, "Edith Wharton", "The Age of Innocence"), (284, "Edith Wharton", "The House of Mirth"),
    (4517, "Edith Wharton", "Ethan Frome"), (11052, "Edith Wharton", "The Custom of the Country"),
    (64317, "F. Scott Fitzgerald", "The Great Gatsby"), (805, "F. Scott Fitzgerald", "This Side of Paradise"),
    (219, "Joseph Conrad", "Heart of Darkness"), (974, "Joseph Conrad", "The Secret Agent"),
    (5658, "Joseph Conrad", "Lord Jim"), (2021, "Joseph Conrad", "Nostromo"),
    (160, "Kate Chopin", "The Awakening"),
    (215, "Jack London", "The Call of the Wild"), (910, "Jack London", "White Fang"),
    (1056, "Jack London", "Martin Eden"), (1164, "Jack London", "The Iron Heel"),
    (55, "L. Frank Baum", "The Wonderful Wizard of Oz"), (16, "J. M. Barrie", "Peter Pan"),
    (113, "Frances Hodgson Burnett", "The Secret Garden"),
    (45, "L. M. Montgomery", "Anne of Green Gables"), (289, "Kenneth Grahame", "The Wind in the Willows"),
    (271, "Anna Sewell", "Black Beauty"), (498, "Kate Douglas Wiggin", "Rebecca of Sunnybrook Farm"),
    (1450, "Eleanor H. Porter", "Pollyanna"), (1874, "E. Nesbit", "The Railway Children"),
    (2226, "Rudyard Kipling", "Kim"), (60, "Baroness Orczy", "The Scarlet Pimpernel"),
    (140, "Upton Sinclair", "The Jungle"), (1695, "G. K. Chesterton", "The Man Who Was Thursday"),
    (2641, "E. M. Forster", "A Room with a View"), (2891, "E. M. Forster", "Howards End"),
    (394, "Elizabeth Gaskell", "Cranford"), (4276, "Elizabeth Gaskell", "North and South"),
    (2153, "Elizabeth Gaskell", "Mary Barton"),
    (599, "William Makepeace Thackeray", "Vanity Fair"),
    (155, "Wilkie Collins", "The Moonstone"), (583, "Wilkie Collins", "The Woman in White"),
    (558, "John Buchan", "The Thirty-Nine Steps"),
    (1155, "Agatha Christie", "The Secret Adversary"), (863, "Agatha Christie", "The Mysterious Affair at Styles"),
    (203, "Harriet Beecher Stowe", "Uncle Tom's Cabin"),
    (62, "Edgar Rice Burroughs", "A Princess of Mars"), (78, "Edgar Rice Burroughs", "Tarzan of the Apes"),
    (2166, "H. Rider Haggard", "King Solomon's Mines"), (3155, "H. Rider Haggard", "She"),
    (144, "Virginia Woolf", "The Voyage Out"), (5670, "Virginia Woolf", "Jacob's Room"),
    (217, "D. H. Lawrence", "Sons and Lovers"), (28948, "D. H. Lawrence", "The Rainbow"),
    (619, "Anthony Trollope", "The Warden"), (82, "Walter Scott", "Ivanhoe"),
    (73, "Stephen Crane", "The Red Badge of Courage"),
    (24, "Willa Cather", "O Pioneers!"), (242, "Willa Cather", "My Ántonia"),
    (543, "Sinclair Lewis", "Main Street"), (32, "Charlotte Perkins Gilman", "Herland"),
    (351, "W. Somerset Maugham", "Of Human Bondage"), (2775, "Ford Madox Ford", "The Good Soldier"),
    (308, "Jerome K. Jerome", "Three Men in a Boat"), (95, "Anthony Hope", "The Prisoner of Zenda"),
    (1300, "Zane Grey", "Riders of the Purple Sage"),
]

LICENCES = {
    "web": (GPT2_URL, "MIT (openai/gpt-2-output-dataset); WebText documents are third-party web pages"),
    "books": (PG_URL.replace("{id}", "<id>"), "Public domain in the USA (Project Gutenberg texts; PG header and licence footer removed)"),
    "wp": (WP_URL, "No licence stated for the data (Reddit r/WritingPrompts via fairseq, Fan et al. 2018); the HF mirror euclaise/writingprompts is labelled MIT"),
    "hanna": (f"{HANNA_GIT}@{HANNA_COMMIT}", "MIT (dig-team/hanna-benchmark-asg); stories from WritingPrompts and model outputs"),
    "openmeva": (f"https://huggingface.co/datasets/{OPENMEVA_REPO}@{OPENMEVA_REV}", "No licence stated (thu-coai/OpenMEVA has no LICENSE file); stories from ROCStories and WritingPrompts"),
    "roc": (f"https://huggingface.co/datasets/{OPENMEVA_REPO}@{OPENMEVA_REV} data/autos_data/roc", "No licence stated (OpenMEVA's copy of ROCStories, Mostafazadeh et al. 2016, whose own terms apply)"),
    "mauve_he": (f"https://drive.google.com/file/d/{MAUVE_GDRIVE_ID} (krishnap25/mauve-experiments human_evaluation.md)", "No licence stated (mauve-experiments has no LICENSE file); WebText contexts and GPT-2 outputs"),
    "dolly": (f"https://huggingface.co/datasets/{DOLLY_REPO}@{DOLLY_REV}", "CC BY-SA 3.0"),
    "ud": (UD_URL.format("{train,dev,test}"), "CC BY-SA 4.0"),
}

STATS = collections.Counter()    # text fixes, printed after each command


# ----------------------------------------------------------------------------
# Common helpers

def log(*a):
    print(*a, file=sys.stderr, flush=True)


def fetch(url, path, sleep=0.0, tries=4):
    """Download url to path unless path exists. Writes a .part file first."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scribe-eval-data/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(path)
            log(f"  fetched {url} ({path.stat().st_size:,} B)")
            if sleep:
                time.sleep(sleep)
            return path
        except Exception as e:  # network errors: retry with back-off
            log(f"  retry {k + 1}/{tries} {url}: {e}")
            time.sleep(2 + 5 * k)
    raise RuntimeError(f"download failed: {url}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


_CTRL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_SURR = re.compile(r"[\ud800-\udfff]")
_BREAKS = re.compile(r"[\u2028\u2029\x0b\x0c\x85]")
_BREAK_MAP = {0x2028: "\n", 0x2029: "\n\n", 0x0b: "\n", 0x0c: "\n", 0x85: "\n"}


def clean(s):
    """Normalise a text: line ends, lone surrogates, control characters,
    trailing spaces, runs of blank lines. Wording is not touched."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    n = len(_BREAKS.findall(s))
    if n:                          # other line breaks become plain newlines,
        STATS["unicode line breaks -> newline"] += n   # so no reader splits a JSONL line
        s = s.translate(_BREAK_MAP)
    n = len(_SURR.findall(s))
    if n:
        STATS["lone surrogates -> U+FFFD"] += n
        s = _SURR.sub("\ufffd", s)
    n = len(_CTRL.findall(s))
    if n:
        STATS["control chars dropped"] += n
        s = _CTRL.sub("", s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def detok(s):
    """Undo the WritingPrompts tokenisation (also used on HANNA, OpenMEVA and
    ROC). "<newline>" becomes a line break (two or more: a paragraph break),
    PTB quotes `` '' become straight double quotes attached to their word,
    spaces before closing punctuation and after opening brackets go, and split
    contractions (do n't, he 's, gon na) are rejoined. Safe on text that is
    already detokenised."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\s*<newline>\s*", "\n", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    s = re.sub(r"``[ \t]*", '"', s)                       # opening quote
    s = re.sub(r"[ \t]*''", '"', s)                       # closing quote
    s = re.sub(r"“[ \t]+", "“", s)
    s = re.sub(r"[ \t]+”", "”", s)
    s = re.sub(r"‘[ \t]+", "‘", s)
    s = re.sub(r"(\w) ’ ?(s|re|ve|m|ll|d|t)\b", r"\1’\2", s)
    s = re.sub(r"(\w) (n't|N'T)\b", r"\1\2", s)
    s = re.sub(r"([\w\]).*}]) ('(?:s|re|ve|m|ll|d|S|RE|VE|M|LL|D))\b", r"\1\2", s)
    s = re.sub(r"\b([Gg]on|[Ww]an|[Gg]ot) (na|ta)\b", r"\1\2", s)
    s = re.sub(r"[ \t]+([.,!?;:%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$])[ \t]+", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return clean(s)


def write_jsonl(path, rows, texts=True):
    """Write rows to path; ids must be unique. For text files (texts=True)
    prints counts of empty and duplicate texts. Returns the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    ids, seen = set(), collections.Counter()
    n = empty = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            key = r["id"] if "id" in r else r["pair_id"]
            if key in ids:
                raise ValueError(f"duplicate id {key} in {path}")
            ids.add(key)
            t = r.get("text", "")
            empty += not t
            seen[hashlib.md5(t.encode("utf-8")).digest()] += 1
            line = json.dumps(r, ensure_ascii=False, allow_nan=False)
            line.encode("utf-8")       # raises on anything that is not UTF-8
            f.write(line + "\n")
            n += 1
    tmp.replace(path)
    dup = sum(c - 1 for c in seen.values() if c > 1)
    extra = f" empty={empty} duplicate_texts={dup}" if texts else ""
    log(f"  wrote {path} lines={n:,} bytes={path.stat().st_size:,}{extra}")
    return n


def row(id_, set_, system, text, prompt=None, ref=None, meta=None, **extra):
    r = {"id": id_, "set": set_, "system": system, "text": text}
    if prompt is not None:
        r["prompt"] = prompt
    if ref is not None:
        r["ref"] = ref
    r.update(extra)
    if meta is not None:
        r["meta"] = meta
    return r


def hf_download(repo, revision, local_dir, patterns):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type="dataset", revision=revision,
                                  local_dir=str(local_dir), allow_patterns=patterns))


def git_clone(url, commit, dest):
    dest = Path(dest)
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        log(f"  note: {dest} is at {head}, the pinned commit is {commit}")
    return dest


# ----------------------------------------------------------------------------
# 1. Web: GPT-2 output dataset

def _stream_prefix(url, path, limit):
    """Save the first lines of a JSONL url until their cleaned text reaches limit bytes."""
    tmp = path.with_name(path.name + ".part")
    total = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        for line in r:
            f.write(line)
            total += len(clean(json.loads(line)["text"]).encode("utf-8"))
            if total >= limit:
                break
    tmp.replace(path)
    log(f"  saved prefix of {url} ({path.stat().st_size:,} B)")


def _web_rows(path, split, set_, system=None, limit=None, exclude=frozenset()):
    total = 0
    for rec in read_jsonl(path):
        text = clean(rec["text"])
        meta = {"length": rec.get("length"), "ended": rec.get("ended"), "src_id": rec["id"]}
        if system is None:
            if not text:            # an empty human document carries nothing
                STATS[f"web {split}: empty docs dropped"] += 1
                continue
            if text in exclude:     # the same document as a test text
                STATS[f"web {split}: doc equal to a test text dropped"] += 1
                continue
            yield row(f"web/{split}/{rec['id']}", set_, "human", text, meta=meta)
        else:
            yield row(f"web/gpt2/{system}/{rec['id']}", set_, system, text, meta=meta)
        total += len(text.encode("utf-8"))
        if limit and total >= limit:
            return


def do_web(a):
    dl = a.dl / "gpt2out"
    for name in ["webtext.valid", "webtext.test"] + [c + ".test" for c in GPT2_CONFIGS]:
        fetch(GPT2_URL + name + ".jsonl", dl / f"{name}.jsonl")
    head = dl / "webtext.train.head.jsonl"
    if not head.exists():
        _stream_prefix(GPT2_URL + "webtext.train.jsonl", head, PACK_BYTES)
    out = a.data / "web"
    # pack and reference must not hold a test text (webtext.test or any GPT-2
    # sample; some samples copy WebText training pages word for word)
    test = {clean(r["text"]) for name in ["webtext"] + GPT2_CONFIGS
            for r in read_jsonl(dl / f"{name}.test.jsonl")} - {""}
    write_jsonl(out / "pack.jsonl", _web_rows(head, "train", "web/pack", limit=PACK_BYTES, exclude=test))
    write_jsonl(out / "ref.jsonl", _web_rows(dl / "webtext.valid.jsonl", "valid", "web/ref", exclude=test))
    write_jsonl(out / "test.jsonl", _web_rows(dl / "webtext.test.jsonl", "test", "web/test"))
    for c in GPT2_CONFIGS:
        write_jsonl(out / "gpt2" / f"{c}.jsonl", _web_rows(dl / f"{c}.test.jsonl", "test", "web/gpt2", system=c))


# ----------------------------------------------------------------------------
# 2. Books: Project Gutenberg

_PG_START = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_PG_END = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_NUM = r"([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|the\s+\w+)"
_HEAD = re.compile(r"^(chapter|book|part|volume|vol\.|stave|letter|section|canto)\s+" + _NUM + r"\b", re.I)
_HEAD_WORD = re.compile(r"^(the\s+)?(preface|contents|table of contents|introduction|prelude|prologue|"
                        r"epilogue|appendix|postscript|conclusion|footnotes?|notes|the end|finis|"
                        r"illustrations?|list of illustrations|dedication|transcriber'?s? notes?)\b[\s.:]*$", re.I)
_FIRST = re.compile(r"^(chapter|book|part|volume|stave|letter)\s+(1|i|one|the first|first)\b|^(i|1)\.?$", re.I)
_ROMAN = re.compile(r"^[IVXLCDM]+(\.|$)|^[IVXLCDM]+\s+[^a-z]*$")
_END_PUNCT = re.compile(r"[.!?;:,\"'”’)\]—–-]$")


def _pg_body(raw):
    """Text between the START and END lines (whole file if a marker is missing)."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    m = _PG_START.search(raw)
    if m:
        raw = raw[m.end():]
    else:
        STATS["books: no START marker"] += 1
    m = _PG_END.search(raw)
    if m:
        raw = raw[:m.start()]
    else:
        STATS["books: no END marker"] += 1
    return raw


def _is_prose(lines, short_line=45):
    """False for headings, tables of contents, verse, separators and notes.
    short_line: lines shorter than this are not wrapped prose lines."""
    joined = " ".join(l.strip() for l in lines)
    letters = [c for c in joined if c.isalpha()]
    if not letters:
        return False                                   # "* * *", page numbers
    if joined.startswith("["):
        return False                                   # [Illustration], [Footnote]
    if joined.upper() == joined and len(joined) < 200:
        return False                                   # all-caps titles, THE END
    if (len(joined) < 400 and _HEAD.match(joined)) or (len(joined) < 100 and _ROMAN.match(joined)):
        return False                                   # CHAPTER IV. TITLE, BOOK THE FIRST, XII.
    if _HEAD_WORD.match(joined) or re.fullmatch(r"\d+\.?", joined):
        return False
    if len(lines) == 1 and len(joined) < 60 and not _END_PUNCT.search(joined):
        return False                                   # short line with no end mark: a title
    if len(lines) >= 2:                                # hard-wrapped prose: every line but
        body = lines[:-1]                              # the last runs to the wrap width
        short = sum(len(l.strip()) < short_line for l in body)
        if short * 3 > len(body):
            return False                               # verse, contents, lists
    return True


def gutenberg_paragraphs(raw):
    """Clean paragraphs of one Gutenberg book."""
    body = _pg_body(raw)
    blocks = [b.split("\n") for b in re.split(r"\n[ \t]*\n", body)]
    blocks = [[l for l in b if l.strip()] for b in blocks]
    blocks = [b for b in blocks if b]
    # Book-relative layout: W is the wrap width (95th percentile of line
    # lengths in multi-line blocks); a block line under 0.8 W is not wrapped
    # prose. If indented blocks are rare in this book, they are verse, letters
    # or notes set off from the text and are dropped too.
    lens = sorted(len(l.strip()) for b in blocks if len(b) > 1 for l in b if len(l.strip()) <= 100)
    wrap = min(100, max(50, lens[int(0.95 * (len(lens) - 1))])) if lens else 70
    indented = lambda b: all(re.match(r"\s{2,}\S", l) for l in b)
    n_ind = sum(indented(b) for b in blocks)
    drop_indented = n_ind < 0.2 * len(blocks)
    # Front matter: cut before the body's first chapter heading, i.e. the first
    # "Chapter I" / "Letter 1" / "I." heading within the first 12 % of the text
    # that has a prose paragraph soon after it (entries of a table of contents
    # are followed by more entries).
    size = sum(len(" ".join(b)) for b in blocks)
    pos, cut = 0, None
    for i, b in enumerate(blocks):
        if pos > 0.12 * size:
            break
        if len(b) <= 2 and _FIRST.match(b[0].strip()) and any(
                len(" ".join(n)) >= 200 and _is_prose(n) for n in blocks[i + 1:i + 8]):
            cut = i
            break
        pos += len(" ".join(b))
    if cut is not None:
        blocks = blocks[cut:]
    # Back matter: cut after the last "THE END" / "FINIS" in the last 10 %.
    pos = 0
    for i in range(len(blocks) - 1, -1, -1):
        if pos > 0.10 * size:
            break
        if re.fullmatch(r"(the\s+end|finis)[\s.!]*", " ".join(blocks[i]).strip(), re.I):
            blocks = blocks[:i]
            break
        pos += len(" ".join(blocks[i]))
    out = []
    for b in blocks:
        if not _is_prose(b, 0.8 * wrap) or (drop_indented and indented(b)):
            STATS["books: blocks dropped"] += 1
            continue
        p = " ".join(l.strip() for l in b)
        p = re.sub(r"\[(Illustration|Footnote|Transcriber)[^\]]*\]", "", p, flags=re.I)
        p = re.sub(r"\[\d+\]|\[[A-Z]\]", "", p)        # footnote markers
        p = re.sub(r"_([^_]+)_", r"\1", p).replace("_", "")
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def _check_pg_header(raw, gid, title):
    """The file's Title: line must agree with the expected title."""
    head = raw[:5000]
    m = re.search(r"^Title:\s*(.+)$", head, re.M)
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    got = norm(m.group(1)) if m else ""
    want = norm(title)
    lang = re.search(r"^Language:\s*(.+)$", head, re.M)
    if not got or not (got.startswith(want[:12]) or want.startswith(got[:12])):
        raise ValueError(f"pg{gid}: title {got!r} is not {want!r}")
    if lang and "english" not in lang.group(1).lower():
        raise ValueError(f"pg{gid}: language {lang.group(1)!r}")


def book_split():
    """{gutenberg id: split}, fixed by BOOK_SEED."""
    order = sorted(b[0] for b in BOOKS)
    random.Random(BOOK_SEED).shuffle(order)
    out, k = {}, 0
    for name, n in BOOK_SPLITS:
        for gid in order[k:k + n]:
            out[gid] = name
        k += n
    assert k == len(BOOKS) == len(set(order)), "BOOKS must hold 120 distinct ids"
    return out


def do_books(a):
    dl = a.dl / "gutenberg"
    split = book_split()
    docs = {name: [] for name, _ in BOOK_SPLITS}
    for gid, author, title in BOOKS:
        path = fetch(PG_URL.format(id=gid), dl / f"pg{gid}.txt", sleep=1.0)
        raw = path.read_bytes().decode("utf-8", errors="replace")
        STATS["books: undecodable bytes"] += raw.count("\ufffd")
        _check_pg_header(raw, gid, title)
        paras = gutenberg_paragraphs(raw)
        body = len(_pg_body(raw).encode("utf-8"))
        kept = sum(len(p.encode("utf-8")) for p in paras)
        if kept < 0.8 * body:
            log(f"  note: pg{gid} keeps {kept / body:.0%} of its body")
        # cut into documents of about 20 KB at paragraph ends
        chunks, cur, size = [], [], 0
        for p in paras:
            cur.append(clean(p))
            size += len(p.encode("utf-8")) + 2
            if size >= BOOK_DOC_BYTES:
                chunks.append(cur)
                cur, size = [], 0
        if cur:
            if chunks and size < BOOK_DOC_BYTES / 2:
                chunks[-1].extend(cur)     # a short tail joins the last document
            else:
                chunks.append(cur)
        s = split[gid]
        for k, c in enumerate(chunks):
            docs[s].append(row(f"books/{s}/{gid}/{k:03d}", f"books/{s}", "human", "\n\n".join(c),
                               meta={"gutenberg_id": gid, "title": title, "author": author,
                                     "part": k, "parts": len(chunks)}))
    for name, _ in BOOK_SPLITS:
        write_jsonl(a.data / "books" / f"{name}.jsonl", docs[name])


# ----------------------------------------------------------------------------
# 3. WritingPrompts

def norm_prompt(p):
    """Prompt key for overlap checks: no [ WP ] tag, lower case, letters and digits."""
    p = re.sub(r"^\s*(\[\s*\w+\s*\]\s*)+", "", detok(p).lower())
    return re.sub(r"[^a-z0-9]+", " ", p).strip()


def wp_files(a):
    d = a.dl / "wp_fairseq"
    tgz = fetch(WP_URL, d / "writingPrompts.tar.gz")
    if not (d / "writingPrompts" / "train.wp_source").exists():
        if sha256(tgz) != WP_SHA256:
            raise ValueError("writingPrompts.tar.gz checksum differs")
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    return d / "writingPrompts"


def _wp_pairs(root, split):
    with open(root / f"{split}.wp_source", encoding="utf-8") as fs, \
            open(root / f"{split}.wp_target", encoding="utf-8") as ft:
        for i, (p, s) in enumerate(zip(fs, ft)):
            yield i, p.rstrip("\n"), s.rstrip("\n")


def _story_test_prompts(a):
    """Normalised HANNA and MANS-WP prompts, when those downloads exist."""
    out = {}
    f = a.dl / "repos" / "hanna-benchmark-asg" / "hanna_stories_annotations.csv"
    if f.exists():
        with open(f, encoding="utf-8", newline="") as fh:
            out["hanna"] = {norm_prompt(r["Prompt"]) for r in csv.DictReader(fh)}
    f = a.dl / "openmeva" / "data" / "mans_data" / "mans_wp.json"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            out["mans_wp"] = {norm_prompt(d["prompt"]) for d in json.load(fh).values()}
    return out


def do_wp(a):
    root = wp_files(a)
    test_prompts = {norm_prompt(p) for _, p, _ in _wp_pairs(root, "test")}
    for name, ps in _story_test_prompts(a).items():
        log(f"  {name}: {len(ps)} prompts, {len(ps & test_prompts)} found in the WP test split")
        test_prompts |= ps

    def rows(split, set_, limit=None):
        total = 0
        for i, p, s in _wp_pairs(root, split):
            if norm_prompt(p) in test_prompts:
                STATS[f"wp {split}: prompt also in test, dropped"] += 1
                continue
            text = detok(s)
            if not text:
                STATS[f"wp {split}: empty story dropped"] += 1
                continue
            yield row(f"wp/{split}/{i:06d}", set_, "human", text, prompt=detok(p),
                      meta={"split": split, "line": i})
            total += len(text.encode("utf-8"))
            if limit and total >= limit:
                return

    write_jsonl(a.data / "wp" / "pack.jsonl", rows("train", "wp/pack", PACK_BYTES))
    write_jsonl(a.data / "wp" / "ref.jsonl", rows("valid", "wp/ref"))


# ----------------------------------------------------------------------------
# 4. HANNA

HANNA_CRIT = {"RE": "Relevance", "CH": "Coherence", "EM": "Empathy",
              "SU": "Surprise", "EG": "Engagement", "CX": "Complexity"}


def _cell(s):
    """A metric CSV cell: a list of numbers, written like a Python list."""
    v = json.loads(re.sub(r"\bnan\b", "NaN", s))
    return [None if isinstance(x, float) and math.isnan(x) else x for x in v]


def check_hanna_order(ann, metrics):
    """Each metric list must follow ascending Story ID within its model."""
    import numpy as np
    by_story = collections.defaultdict(list)
    for r in ann:
        by_story[int(r["Story ID"])].append(r)
    rhos = []
    for m in metrics:
        sids = sorted({int(r["Story ID"]) for r in ann if r["Model"] == m["Model"]})
        assert len(sids) == 96, m["Model"]
        for ab, crit in HANNA_CRIT.items():
            want = [np.mean([float(r[crit]) for r in by_story[s]]) for s in sids]
            assert np.allclose(_cell(m[crit]), want), (m["Model"], crit)
            for k in range(3):
                want = [float(by_story[s][k][crit]) for s in sids]
                assert np.allclose(_cell(m[f"Human {k + 1} {ab}"]), want), (m["Model"], ab, k)
        # an automatic metric too: "Text length" follows the story's token count
        tl = [c for c in m if c.startswith("Text length")][0]
        toks = [len(re.findall(r"\w+|[^\w\s]", by_story[s][0]["Story"])) for s in sids]
        rank = lambda v: np.argsort(np.argsort(v))
        r = np.corrcoef(rank(_cell(m[tl])), rank(toks))[0, 1]
        assert r > 0.6, (m["Model"], "text length", r)
        rhos.append(r)
    log(f"  HANNA order verified: {len(metrics)} models x 6 criteria (means and 3 raters) "
        f"equal the annotations in Story ID order; Spearman(Text length, token count) "
        f"min {min(rhos):.3f}, median {np.median(rhos):.3f}")


def do_hanna(a):
    repo = git_clone(HANNA_GIT, HANNA_COMMIT, a.dl / "repos" / "hanna-benchmark-asg")
    with open(repo / "hanna_stories_annotations.csv", encoding="utf-8", newline="") as f:
        ann = list(csv.DictReader(f))
    with open(repo / "hanna_metric_scores_llm.csv", encoding="utf-8", newline="") as f:
        metrics = list(csv.DictReader(f))
    check_hanna_order(ann, metrics)

    by_story = collections.defaultdict(list)
    for r in ann:
        by_story[int(r["Story ID"])].append(r)
    assert sorted(by_story) == list(range(1056)) and all(len(v) == 3 for v in by_story.values())
    texts = []
    for sid in sorted(by_story):
        rs = by_story[sid]
        first = rs[0]
        assert first["Prompt"] == by_story[sid % 96][0]["Prompt"]       # prompt id = sid mod 96
        assert first["Human"] == by_story[sid % 96][0]["Story"]         # ref = the human story
        model = first["Model"]
        raters = {ab: [int(r[c]) for r in rs] for ab, c in HANNA_CRIT.items()}
        meta = {"story_id": sid, "prompt_id": sid % 96, "model": model,
                "human": {ab: sum(v) / 3 for ab, v in raters.items()},
                "raters": raters, "workers": [r["Worker ID"] for r in rs]}
        texts.append(row(f"hanna/{sid}", "hanna", "human" if model == "Human" else model,
                         detok(first["Story"]), prompt=detok(first["Prompt"]),
                         ref=detok(first["Human"]), meta=meta))
    write_jsonl(a.data / "hanna" / "texts.jsonl", texts)

    human_cols = set(HANNA_CRIT.values()) | {c for c in metrics[0] if c.startswith("Human ")}
    cols = [c for c in metrics[0] if c != "Model" and c not in human_cols]
    lines = []
    for m in metrics:
        sids = sorted({int(r["Story ID"]) for r in ann if r["Model"] == m["Model"]})
        vals = {c: _cell(m[c]) for c in cols}
        for k, sid in enumerate(sids):
            d = {"id": f"hanna/{sid}", "story_id": sid, "model": m["Model"]}
            d.update({c: vals[c][k] for c in cols})
            lines.append(d)
    lines.sort(key=lambda d: d["story_id"])
    write_jsonl(a.data / "hanna" / "shipped_metrics.jsonl", lines, texts=False)
    log(f"  shipped metrics: {len(cols)} columns")


# ----------------------------------------------------------------------------
# 5. OpenMEVA (MANS and AUTOS)

OM_DISC = ["lexical_rept", "semantic_rept", "character", "commonsense",
           "relatedness", "consistency", "cause", "time"]
OM_INV = ["synonym_substitute", "semantic_substitute", "delete_punct", "contraction", "typos"]


def openmeva_files(a):
    return hf_download(OPENMEVA_REPO, OPENMEVA_REV, a.dl / "openmeva",
                       ["README.md", "data/README.md", "data/mans_data/*.json",
                        "data/autos_data/roc/*.txt", "data/autos_data/roc/*/*.txt",
                        "data/autos_data/wp/*/*.txt", "data/autos_data/wp/wp_test_*.txt"]) / "data"


def _om_lines(path, keep_empty=False):
    """Raw AUTOS lines as {line number: (src, context, story, note)}. Lines with
    an empty context or story are skipped, as OpenMEVA's benchmark code does."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("|||")
            if len(p) < 3:
                STATS["openmeva: malformed lines"] += 1
                continue
            src, ctx, story, note = int(p[0]), p[1].strip(), p[2].strip(), (p[3].strip() if len(p) > 3 else "")
            if not keep_empty and (not ctx or not story):
                STATS[f"openmeva {path.name}: empty context or story, skipped"] += 1
                continue
            out[i] = (src, ctx, story, note)
    return out


def do_openmeva(a):
    root = openmeva_files(a)
    out = a.data / "openmeva"
    # MANS
    for corpus in ["roc", "wp"]:
        with open(root / "mans_data" / f"mans_{corpus}.json", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for key in sorted(data, key=int):
            d = data[key]
            for system in sorted(d["gen"]):
                g = d["gen"][system]
                scores = [int(x) for x in g.get("score", [])]
                if not scores:
                    STATS[f"openmeva mans_{corpus}: no score"] += 1
                meta = {"key": key, "corpus": corpus,
                        "human": {"overall": sum(scores) / len(scores) if scores else None, "scores": scores}}
                rows.append(row(f"openmeva/mans_{corpus}/{key}/{system}", f"openmeva/mans_{corpus}", system,
                                detok(g["text"]), prompt=detok(d["prompt"]),
                                ref=detok(d.get("gold_response")) if d.get("gold_response") else None,
                                meta=meta))
        write_jsonl(out / f"mans_{corpus}.jsonl", rows)

    # AUTOS
    for corpus in ["roc", "wp"]:
        base = root / "autos_data" / corpus
        orig = {}                      # test-set index -> (context, story) of human stories
        for asp in OM_DISC:
            for src, ctx, story, _ in _om_lines(base / "discrimination_test" / f"{corpus}_{asp}.txt").values():
                orig.setdefault(src, (ctx, story))
        if corpus == "roc":            # fall back to the test files for the rest
            ti = (base / "roc_test_ipt.txt").read_text(encoding="utf-8").split("\n")
            to = (base / "roc_test_opt.txt").read_text(encoding="utf-8").split("\n")
            for k, (c, st) in enumerate(zip(ti, to)):
                if c.strip() and st.strip() and k not in orig:
                    orig[k] = (c.strip(), st.strip())
        common = lambda **kw: dict({"corpus": corpus}, **kw)

        for asp in OM_DISC:
            set_ = f"openmeva/autos_{corpus}_{asp}"
            rows, pos_src = [], set()
            for i, (src, ctx, story, note) in _om_lines(base / "discrimination_test" / f"{corpus}_{asp}.txt").items():
                pos_src.add(src)
                rows.append(row(f"{set_}/pos/{i}", set_, "human", detok(story), prompt=detok(ctx),
                                meta=common(test="discrimination", aspect=asp, label=1, src=src, note=note, line=i)))
            for i, (src, ctx, story, note) in _om_lines(base / "discrimination_test" / f"{corpus}_{asp}_perturb.txt").items():
                ref = detok(orig[src][1]) if src in orig else None
                rows.append(row(f"{set_}/neg/{i}", set_, "perturbed", detok(story), prompt=detok(ctx), ref=ref,
                                meta=common(test="discrimination", aspect=asp, label=0, src=src, note=note, line=i,
                                            paired=src in pos_src)))
            write_jsonl(out / f"autos_{corpus}_{asp}.jsonl", rows)

        # invariance: label 1 = original, 0 = perturbed (OpenMEVA's robust.py);
        # the metric should not tell them apart
        neg = _om_lines(base / "invariance_test" / f"{corpus}_negative_sample.txt")
        for asp in OM_INV:
            set_ = f"openmeva/autos_{corpus}_{asp}"
            rows, have = [], set()
            for basename, fname in [("human", f"{corpus}_{asp}_perturb.txt"),
                                    ("negative", f"{corpus}_negative_sample_{asp}_perturb.txt")]:
                for i, (src, ctx, story, note) in _om_lines(base / "invariance_test" / fname).items():
                    if basename == "human":
                        o, onote, osys = orig.get(src), "", "human"
                    else:
                        o = neg[src][1:3] if src in neg else None
                        onote, osys = (neg[src][3] if src in neg else ""), "perturbed"
                    oid = f"{set_}/{basename}/orig/{src}"
                    if o is None:
                        STATS[f"openmeva {corpus} {asp} {basename}: perturbed line without original"] += 1
                    elif oid not in have:
                        have.add(oid)
                        rows.append(row(oid, set_, osys, detok(o[1]), prompt=detok(o[0]),
                                        meta=common(test="invariance", aspect=asp, base=basename, role="original",
                                                    label=1, src=src, note=onote, line=None)))
                    rows.append(row(f"{set_}/{basename}/pert/{i}", set_, "perturbed", detok(story),
                                    prompt=detok(ctx),
                                    meta=common(test="invariance", aspect=asp, base=basename, role="perturbed",
                                                label=0, src=src, note=note, line=i,
                                                pair=oid if o is not None else None)))
            write_jsonl(out / f"autos_{corpus}_{asp}.jsonl", rows)


# ----------------------------------------------------------------------------
# 9. ROC human stories (OpenMEVA's ROC splits)

def do_roc(a):
    root = openmeva_files(a)
    base = root / "autos_data" / "roc"
    read = lambda n: (base / n).read_text(encoding="utf-8").rstrip("\n").split("\n")
    key = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    held = {key(s) for s in read("roc_test_ipt.txt")}
    with open(root / "mans_data" / "mans_roc.json", encoding="utf-8") as f:
        held |= {key(d["prompt"]) for d in json.load(f).values()}
    for split, name in [("train", "pack"), ("dev", "ref")]:
        rows = []
        for i, (c, s) in enumerate(zip(read(f"roc_{split}_ipt.txt"), read(f"roc_{split}_opt.txt"))):
            if key(c) in held:
                STATS[f"roc {split}: first sentence held out, dropped"] += 1
                continue
            rows.append(row(f"roc/{split}/{i:05d}", f"roc/{name}", "human", detok(c.strip() + " " + s.strip()),
                            meta={"split": split, "line": i}))
        write_jsonl(a.data / "roc" / f"{name}.jsonl", rows)


# ----------------------------------------------------------------------------
# 6. MAUVE human evaluation

def strip_html(s):
    """HTML fragment -> plain text; </p><p> and <br> become paragraph breaks."""
    s = re.sub(r"</p>\s*<p[^>]*>", "\n\n", s or "", flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return clean(html.unescape(s))


def _mauve_system(s):
    if s.strip() == "human":
        return "human"
    m = re.fullmatch(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", s.strip())
    if not m:
        raise ValueError(f"unknown model {s!r}")
    return f"{m.group(1)} {m.group(2)}"


def do_mauve(a):
    path = a.dl / "mauve_he" / "mauve_human_eval.csv"
    if not path.exists():
        import gdown
        path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=MAUVE_GDRIVE_ID, output=str(path), quiet=True)
    if sha256(path) != MAUVE_SHA256:
        log("  note: mauve_human_eval.csv checksum differs from the one used for the paper")
    with open(path, encoding="utf-8", newline="") as f:
        recs = list(csv.DictReader(f))
    # The same (idx, system) can carry different completions in different HITs
    # (703 of 5,350 keys: separate samples), and idx 635 has two contexts, so a
    # completion is keyed by (idx, system, context, text); k numbers the
    # variants of one (idx, system) in order of first appearance.
    texts, key_id, variants, pairs = {}, {}, collections.Counter(), []
    for r in recs:
        idx = int(r["Input.idx"])
        ctx = strip_html(r["Input.ctx"])
        ids = {}
        for side in "ab":
            system = _mauve_system(r[f"Input.model_{side}"])
            text = strip_html(r[f"Input.completion{side}"])
            key = (idx, system, ctx, text)
            if key not in key_id:
                k = variants[idx, system]
                variants[idx, system] += 1
                id_ = f"mauve_he/{idx}/{system.replace(' ', '_')}/{k}"
                key_id[key] = id_
                texts[id_] = row(id_, "mauve_he", system, text, prompt=ctx,
                                 meta={"idx": idx, "variant": k, "len": int(r[f"Input.len_{side}"])})
            ids[side] = key_id[key]
        pairs.append({"pair_id": f"mauve_he/pair/{r['HITId']}", "idx": idx, "a": ids["a"], "b": ids["b"],
                      "q1": r["Answer.q1"], "q2": r["Answer.q2"], "q3": r["Answer.q3"],
                      "worker": r["WorkerId"]})
    for d in texts.values():
        d["meta"]["variants"] = variants[d["meta"]["idx"], d["system"]]
    STATS["mauve: (idx, system) with more than one completion"] = sum(v > 1 for v in variants.values())
    write_jsonl(a.data / "mauve_he" / "texts.jsonl",
                sorted(texts.values(), key=lambda d: (d["meta"]["idx"], d["system"], d["meta"]["variant"])))
    write_jsonl(a.data / "mauve_he" / "pairs.jsonl", pairs, texts=False)


# ----------------------------------------------------------------------------
# 7. Dolly

def do_dolly(a):
    root = hf_download(DOLLY_REPO, DOLLY_REV, a.dl / "dolly", ["databricks-dolly-15k.jsonl"])
    rows = []
    for i, r in enumerate(read_jsonl(root / "databricks-dolly-15k.jsonl")):
        ins, ctx, resp = clean(r["instruction"]), clean(r.get("context", "")), clean(r["response"])
        prompt = ins + ("\n\n" + ctx if ctx else "")
        rows.append(row(f"dolly/{i:05d}", "dolly", "human", resp, prompt=prompt,
                        instruction=ins, context=ctx, response=resp, category=r["category"],
                        meta={"line": i, "category": r["category"]}))
    write_jsonl(a.data / "dolly" / "dolly.jsonl", rows)


# ----------------------------------------------------------------------------
# 8. Universal Dependencies English EWT

def do_ud(a):
    for split in ["train", "dev", "test"]:
        p = fetch(UD_URL.format(split), a.data / "ud" / f"en_ewt-ud-{split}.conllu")
        n = sum(1 for line in open(p, encoding="utf-8") if line.startswith("# sent_id"))
        log(f"  {p.name}: {n:,} sentences")


# ----------------------------------------------------------------------------
# Manifest

def do_manifest(a):
    rows = []
    for p in sorted(a.data.rglob("*")):
        rel = p.relative_to(a.data).as_posix()
        top = rel.split("/")[0]
        if not p.is_file() or top not in LICENCES or p.name.endswith(".part"):
            continue                   # only this script's outputs
        src, lic = LICENCES[top]
        if top == "web" and "/gpt2/" in rel:
            src = GPT2_URL + p.stem + ".test.jsonl"
        elif top == "web":
            src = GPT2_URL + {"pack": "webtext.train.jsonl (first documents)", "ref": "webtext.valid.jsonl",
                              "test": "webtext.test.jsonl"}.get(p.stem, "")
        elif top == "ud":
            src = UD_URL.format(p.stem.rsplit("-", 1)[-1])
        with open(p, "rb") as f:
            n = sum(1 for _ in f)
        rows.append([rel, str(n), str(p.stat().st_size), sha256(p), src, lic])
    out = a.data / "MANIFEST.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("path\tlines\tbytes\tsha256\tsource\tlicence\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    log(f"  wrote {out} ({len(rows)} files)")


COMMANDS = {"web": do_web, "books": do_books, "wp": do_wp, "hanna": do_hanna, "openmeva": do_openmeva,
            "roc": do_roc, "mauve": do_mauve, "dolly": do_dolly, "ud": do_ud, "manifest": do_manifest}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=os.environ.get("SCRIBE_DATA"), help="output root D")
    ap.add_argument("--dl", type=Path, default=os.environ.get("SCRIBE_DL"), help="download root DL")
    ap.add_argument("command", choices=list(COMMANDS) + ["all"])
    a = ap.parse_args(argv)
    if a.data is None or a.dl is None:
        ap.error("--data and --dl (or $SCRIBE_DATA and $SCRIBE_DL) are required")
    a.data, a.dl = a.data.resolve(), a.dl.resolve()
    a.data.mkdir(parents=True, exist_ok=True)
    names = [c for c in COMMANDS if c != "manifest"] + ["manifest"] if a.command == "all" else [a.command]
    for name in names:
        log(f"[{name}]")
        t = time.time()
        STATS.clear()
        COMMANDS[name](a)
        for k, v in sorted(STATS.items()):
            log(f"  {k}: {v:,}")
        log(f"  done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
