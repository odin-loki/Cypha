# bench/eval/

Frozen evaluation sets. A file here is never edited: a fix makes a new version. Each set's report says how it was built and who may read it.

| File | What it is | Rows | sha256 |
|---|---|---:|---|
| `temporal_holdout_dev.tsv` | S15 temporal holdout, dev half: facts current Wikidata holds about enwik8's subjects, true and false, each labelled a (stated in the 2006 text), b (true by 2006, unstated), c (true only after 2006-03-04) or u (unknown when). Read freely while developing a generator ([report](../../docs/reports/S15_TEMPORAL_HOLDOUT.md)) | 38,162 | `9e1ae1ba9420b1b53e358ee815f431ecdeaad907d2a43eeb0235371a3e6ccc51` |
| `temporal_holdout_test.tsv` | S15 temporal holdout, test half (split by subject). Read only by a generator's final read, once per change, and logged in the S15 report's "Reads" table. Never training text, never a store, shard, teacher prompt or seed | 38,624 | `1535dcdd77a6d150604c7565e66566dd87f09a08a564941cca33baf855b32fb6` |
| `temporal_holdout_wikidata_cache.tar.gz` | S15: the Wikidata responses the two files were built from (unpack into the build's run directory); for audit and a byte-for-byte rebuild, never read by a generator | 2.35 MB | `1969d192f910a23407954037f8e924581e91c02fa0dbe6aaa033602473b5d948` |
| `s8e_slots_dev.tsv` | S8e fresh set, dev part: facts current Wikidata holds about enwik8's subjects on slots S15 never held (nationality P27, location P276, predecessor P1365/P155, successor P1366/P156, taxon family, order and class), true and false, in S15's columns and categories; split by subject. Read freely: S8e's calibration, its grader audit and the choice of variant ([report](../../docs/reports/S8E_GRADER_AND_FRESH_SET.md)) | 3,428 | `dd96be77f27c9914f0a7dc73e5e81effa94bdf3c8a1dd5fc127e01dc25b3f2bd` |
| `s8e_slots_test.tsv` | S8e fresh set, test part (split by subject). Read only by a gate read, once per change, and logged in the S8e report. Never training text, never a store, shard, teacher prompt or seed | 3,922 | `957532b20e91522f553af392af6251480e1eb6de0e82aac8a8c5ea1400ee864f` |
| `s8e_slots_wikidata_cache.tar.gz` | S8e: the Wikidata responses the two files were built from (fetched 2026-10-29; unpack into the build's run directory); for audit and a byte-for-byte rebuild, never read by a generator | 0.42 MB | `b86b16eac2636615cf1a0de88e3bbf16310203bcd2a50d100bb4d8cf1c9f936e` |
| `s8e_grader_tables.tar.gz` | S8e's grader tables (`names.tsv`, `anc.tsv`, `flags.tsv`; fetched from Wikidata 2026-10-29 by `scripts/fresh_slots.py grader`): every name, place ancestor and class flag S8e's grader reads. Unpack into a directory and pass it to `pattern_learn --sources … --s8e DIR …`. Grading only, never read by a generator | 0.77 MB | `5a99f36513826dbab00406a4c7d53d1dd6b7991471683b6cf0969e9e224ee620` |

Other files here (`t_qa_wiki*.tsv`, `s1_reason_audit_dev.tsv`) are described in the reports that use them (A0/A1, A1d, S1).
