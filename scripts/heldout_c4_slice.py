"""Write a held-out C4 text slice that the 1 TB trie never trained on.

c4-train.01023-of-01024 was never downloaded (the articles cap stopped at 00738).
Non-ASCII is dropped because trie_stream_train skips \\u escapes.
"""
import json
import urllib.request
from pathlib import Path

URL = "https://data.together.xyz/redpajama-data-1T/v1.0.0/c4/c4-train.01023-of-01024.jsonl"
OUT = Path(r"D:\cypha_llm\profile\c4_heldout.txt")
req = urllib.request.Request(URL, headers={"User-Agent": "cypha-corpus-fetch/1.0", "Range": "bytes=0-25165823"})
raw = urllib.request.urlopen(req, timeout=300).read()
lines = raw.split(b"\n")[:-1]
parts = []
total = 0
for line in lines:
    text = json.loads(line)["text"]
    text = text.encode("ascii", "ignore")
    parts.append(text)
    total += len(text) + 2
    if total > 9_000_000:
        break
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(b"\n\n".join(parts))
print("docs", len(parts), "bytes", OUT.stat().st_size)
