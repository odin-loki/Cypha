# bench/eval/

Frozen evaluation sets. A file here is never edited: a fix makes a new version. Each set's report says how it was built and who may read it.

| File | What it is | Rows | sha256 |
|---|---|---:|---|
| `temporal_holdout_dev.tsv` | S15 temporal holdout, dev half: facts current Wikidata holds about enwik8's subjects, true and false, each labelled a (stated in the 2006 text), b (true by 2006, unstated), c (true only after 2006-03-04) or u (unknown when). Read freely while developing a generator ([report](../../docs/reports/S15_TEMPORAL_HOLDOUT.md)) | 38,162 | `9e1ae1ba9420b1b53e358ee815f431ecdeaad907d2a43eeb0235371a3e6ccc51` |
| `temporal_holdout_test.tsv` | S15 temporal holdout, test half (split by subject). Read only by a generator's final read, once per change, and logged in the S15 report's "Reads" table. Never training text, never a store, shard, teacher prompt or seed | 38,624 | `1535dcdd77a6d150604c7565e66566dd87f09a08a564941cca33baf855b32fb6` |
| `temporal_holdout_wikidata_cache.tar.gz` | S15: the Wikidata responses the two files were built from (unpack into the build's run directory); for audit and a byte-for-byte rebuild, never read by a generator | 2.35 MB | `1969d192f910a23407954037f8e924581e91c02fa0dbe6aaa033602473b5d948` |

Other files here (`t_qa_wiki*.tsv`, `s1_reason_audit_dev.tsv`) are described in the reports that use them (A0/A1, A1d, S1).
