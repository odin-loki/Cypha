# data/cyphalm/

**Odin Loch** · 2026-10-28

Runtime data for `cyphalm_trie`. Report: [`docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md`](../../docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md) ("A4b").

| File | What it is |
|---|---|
| `simvec_ppmi32.bin` | The similar-context expert's word vectors (2,232,300 bytes, sha256 `c9fdf9ef01a1c853d26eb52203602ea09b3166a23660ca483b7d7c8e96a64335`). For each of PPMI-SVD's 50,000 words: its projection on the 32 leading principal components of the expert's two-word contexts over wiki train, int8 with one scale a row. The default of `cyphalm_trie qa`, `sample` and `context8k` reads it: from `$CYPHA_SIMVEC` if set, else from this file in the source tree (the path CMake compiles in), or from `data/cyphalm/simvec_ppmi32.bin` under the current directory when built without CMake. `--no-sim` turns the expert off |

**Format.** "CYSIMV1" and a zero byte; rows and dimensions (uint32); each row's word (a length byte, then its bytes); the rows' scales (float32); the rows (int8, rows × dimensions); the FNV-1a 64-bit hash of every byte before it. Little-endian. `SimilarContext::load_projected` refuses a file whose hash does not match.

**Rebuild** (from the repository root; `enwik8` is the 100 MB file, 85 s and about 0.6 GB for the vectors, 30 s for the file):

```bash
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
head -c 7340032 enwik8 > enwik8_noeval && tail -c +9437185 enwik8 >> enwik8_noeval   # no BPC holdout, no fresh text
./ppmi_svd enwik8_noeval none none - ppmi_noeval --no-ri --no-ablation
./cyphalm_trie simvec enwik8 --sim default,vec=ppmi_noeval,dim=32 --out data/cyphalm/simvec_ppmi32.bin
```
