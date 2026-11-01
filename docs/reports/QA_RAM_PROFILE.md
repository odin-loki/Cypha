# M18 step 1: where `qa`'s RAM peaks sit

PLAN Track P, row M18 (ledger 95, 96, 101). Run on Odin's PC (WSL, 64 threads) on 2026-11-01 with
`scripts/qa_ram_profile.sh` at commit `ac9ac07`. RSS was sampled every 0.25 s, along with the last stderr line, which names the phase.
No code was changed and nothing was trained beyond what `qa` itself builds.

## Result

| Run | Configuration | Peak RSS | Time | rc |
|---|---|---|---|---|
| R1 | built (C1L + defaults, `--save`) | 196,368 kB (191.8 MB) | 969 s | 0 |
| R2 | served (`--model`, R1's store) | 192,192 kB (187.7 MB) | 619 s | 0 |
| R3 | frozen-test configuration (BKN2) | 198,496 kB (193.8 MB) | 1047 s | 0 |

R1 and R2 rows are byte-identical. All three runs are under Odin's 200 MB limit, but none is 10 MB under it (M18's target).
These peaks are VmHWM from `/proc`, sampled. They sit 3.5–6 MB below the earlier `/usr/bin/time -v` figures (195.3 / 194.6 / 199.9 MB).

## Where the peak sits

In every configuration the peak is reached in the **last phase**, the doc cache (917 docs, vocabulary 59,654), on top of
the **similar-context expert**. They are the two largest steps:

| Phase (R1) | RSS at phase start | Step |
|---|---|---|
| wiki trained (404,601 nodes) | 69.4 MB | — |
| index built | 90.4 MB | +21 |
| chat trained | 136.4 MB | +46 |
| class table / byte Kneser–Ney built (23.95 MB of tables) | 152.8 MB | +16 |
| similar-context expert built | 183.6 MB | **+31** |
| doc cache | 196.4 MB (peak) | **+13** |

Served (R2) starts at 129 MB with no training or index build, but it still builds the similar-context expert (+37 MB) and the doc cache
at run time, so its peak is only 4 MB lower than R1's.

## Next (M18 step 2)

The cuts should target the similar-context expert (~31–37 MB) and the doc cache (~13 MB), for example by building them smaller or
storing them in the served model. Freeing the training-only tables before they are built would also help. Every cut must keep the rows byte-identical (R1 = R2 = the
current rows).

## Raw summary

```
M18 qa RAM profile, 2026-11-01T03:10:23Z, ac9ac07, OdinsPC
== R1: peak_rss_kb 196368 seconds 968.89 rc 0 (191.8 MB)
   first sample within 1 MB of the peak, and the phase (last stderr line) there:
   381.72 196368 196368 | similar-context expert built
   RSS at each new phase (first sample of each stderr line), largest 12:
   196368	944.40s	 doc cache docs 917 vocab 59654 trigger_pairs 0 topic_words 0 build_s 0.27
   183568	362.35s	 similar-context expert built
   152848	348.03s	 stored the qa model in /mnt/d/cypha_llm/profile/m18/store
   152848	347.23s	 byte Kneser-Ney built: 23950084 bytes
   152848	304.28s	 class table built
   136376	280.14s	 chat trained
   90384	284.93s	 index built
   69404	41.76s	 wiki trained, nodes 404601
   3584	0.01s	 
== R2: peak_rss_kb 192192 seconds 618.82 rc 0 (187.7 MB)
   first sample within 1 MB of the peak, and the phase (last stderr line) there:
   35.08 191168 191168 | similar-context expert built
   RSS at each new phase (first sample of each stderr line), largest 12:
   192192	593.01s	 doc cache docs 917 vocab 59654 trigger_pairs 0 topic_words 0 build_s 0.260689
   174784	16.18s	 similar-context expert built
   137680	1.91s	 served
   129308	1.62s	 served from /mnt/d/cypha_llm/profile/m18/store: no training, no index or class build
   512	0.01s	 
== R3: peak_rss_kb 198496 seconds 1046.54 rc 0 (193.8 MB)
   first sample within 1 MB of the peak, and the phase (last stderr line) there:
   989.80 198496 198496 | similar-context expert built
   RSS at each new phase (first sample of each stderr line), largest 12:
   198496	990.07s	 doc cache docs 917 vocab 59654 trigger_pairs 0 topic_words 0 build_s 0.28648
   181600	381.91s	 similar-context expert built
   154976	367.90s	 byte Kneser-Ney built: 23950084 bytes
   154976	325.59s	 class table built
   136120	301.08s	 chat trained
   90464	305.90s	 index built
   65824	46.58s	 wiki trained, nodes 404601
   512	0.01s	 
== rows: R1 vs R2 identical? yes
```

Notes: the run needed `CYPHA_ENWIK8` and `CYPHA_DOLLY` set, because `find1` failed under `set -o pipefail` when `find` could not
read `/root`. The commit field was blank because the source was a `git archive` copy. Both are fixed in the script.
