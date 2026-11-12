# MIXDEC_T8: the strict reading, by prompt

Odin Loch · 2026-11-13 · dev prompts 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13; first 460 characters (cut at a word); the cut last sentence not counted.

W: whole sentences (a subject and a finite verb, not broken off). F: verbless fragments. B: broken (a verb, but the clause breaks off or runs on). K: the subject's word in three sentences running.

| dev | T8 W/F/B | T8MR W/F/B | T8MRJ W/F/B | K (T8, T8MR, T8MRJ) | Note |
|---|---|---|---|---|---|
| 0 | 4/0/0 | 4/0/0 | 4/0/0 | – | the same text in all three |
| 1 | 4/2/1 | 4/3/1 | 3/4/1 | – | |
| 2 | 1/1/1 | 2/2/2 | 1/2/3 | – | MR, MRJ: "the cars", "the band" (leave the subject) |
| 3 | 3/3/0 | 2/2/2 | 2/2/2 | – | MR, MRJ: "peroxide." as a word list |
| 4 | 3/3/2 | 3/4/1 | 3/6/1 | – | |
| 5 | 4/1/0 | 5/0/1 | 5/1/3 | – | MR better: "He was elected a member of the House of Commons." |
| 7 | 4/3/0 | 6/1/0 | 6/1/0 | – | MR, MRJ better: "The college was founded in 1965." |
| 9 | 3/2/0 | 2/3/1 | 3/6/0 | MRJ ("data") | |
| 10 | 2/0/2 | 2/3/2 | 2/4/2 | – | MR, MRJ leave the subject ("noise rock", "a hotel room") |
| 11 | 3/4/2 | 1/1/3 | 2/4/2 | – | MR worse |
| 12 | 0/1/3 | 3/1/1 | 6/3/2 | MR, MRJ ("Alia") | on the subject, but "Alia" in 4–5 sentences running |
| 13 | 1/3/1 | 1/7/0 | 2/7/0 | MR, MRJ ("Messianic") | MR, MRJ: a list of noun phrases |
| **all** | **32/23/12 of 67** | **35/27/14 of 76** | **39/40/16 of 95** | 0, 2, 3 | |
| **shares** | W 0.48, F 0.34 | W 0.46, F 0.36 | W 0.41, F 0.42 | | |
