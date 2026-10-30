# CyphaLM byte trie trained on the 1 TB corpus (packed)

The packed serving form of the order-4 byte trie trained on 918.6 GB of text (C4, GitHub, arXiv, Stack Exchange, 40
Gutenberg books). Profile, tests and every number: [`docs/reports/TRIE_1TB_PROFILE.md`](../../docs/reports/TRIE_1TB_PROFILE.md).

| File | What |
|------|------|
| `trie_1tb.pk.zst` | `trie_1tb.pk` compressed with `zstd -19` (Git LFS) |
| `SHA256SUMS` | SHA-256 of the compressed file, the packed file and the unpacked checkpoint |

```bash
zstd -d models/cyphalm_1tb/trie_1tb.pk.zst -o trie_1tb.pk
sha256sum -c models/cyphalm_1tb/SHA256SUMS --ignore-missing
cyphalm_trie sample /path/to/enwik8 out.txt --model trie_1tb.pk --no-sim --no-copy --no-bkn
cyphalm_trie context8k /path/to/enwik8 /path/to/dolly.jsonl --model trie_1tb.pk
trie_ckpt bpc trie_1tb.pk FILE[:FROM:LEN]
```

`trie_1tb.pk` is `packed_trie.hpp`'s `PackedModel` file ("CYPHAPK" version 1): 41,809,290 nodes, 389,733,888 followers
(count at least 2, one byte each), the unigram and order-1 tables, no 4-byte match, and the six mixture weights
(0.000095, 0.003963, 0.001806, 0.052814, 0.941226, 0.000095), fit on enwik9 bytes `[0, 7340032)` with the counts
frozen. 992,451,375 bytes; about 1.0 GB resident.

The unpacked model (`TR64` checkpoint, 12.4 GB, 64-bit counts) stays on the training machine at
`D:\cypha_llm\corpus\trie\checkpoint.bin`. It was packed with

```bash
trie_ckpt pack checkpoint.bin trie_1tb.pk --fit enwik9:0:7340032
```
