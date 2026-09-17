# Historical archive manifest

Byte-exact inventory of `docs/history/archive/`. Every entry below was verified
`sha256`-identical to the file in the source archive at staging time.

Source archive: `Cypha.zip` (6,108,467 bytes), 164 files. 30 `__pycache__/*.pyc` files were
excluded as build artifacts (they are also covered by the repository `.gitignore`).

`Directory mtime` is the only trustworthy timestamp from the zip: per-file mtimes were
all rewritten to the moment of zipping. See [`../TIMELINE.md`](../TIMELINE.md) for the
reconstructed chronology, which relies on internal evidence rather than these stamps.

## Integrity exception

**`big-data/download_datasets.py` is corrupt and contains no recoverable content.**
It is 8,755 bytes of `0x00` and nothing else — a single distinct byte value, zero printable
characters. It arrived that way: the copy inside the source zip is byte-identical, so this
is not a staging artifact. The file is preserved as delivered rather than deleted, because
its presence and size are themselves part of the record.

A scan of all 142 text files in the archive found **no other file with NUL bytes or missing
content**. Everything else is intact.

## Directories

| Archived as | Original name in zip | Directory mtime | Files | Bytes |
|---|---|---|---|---|
| `prototypes/` | `Prototypes` | 2025-02-26 | 9 | 360,877 |
| `cypha-v1/` | `Cypha V1` | 2025-03-04 | 4 | 148,269 |
| `cypha-v2/` | `Cypha v2` | 2026-02-21 | 5 | 435,953 |
| `cypha-v3/` | `Cypha v3` | 2026-02-21 | 4 | 136,203 |
| `cypha-v4/` | `Cypha v4` | 2026-02-21 | 4 | 95,392 |
| `cypha-v5/` | `Cypha v5` | 2026-02-21 | 10 | 365,497 |
| `cypha-v6/` | `Cypha v6` | 2026-02-27 | 27 | 4,089,525 |
| `cypa-v7-generation/` | `Cypa v7 Generation` | 2026-03-06 | 1 | 272,560 |
| `cypha-v8/` | `Cypha v8` | 2026-03-11 | 18 | 575,061 |
| `cypha-vchatgpt/` | `Cypha vChatGPT` | 2026-02-21 | 41 | 77,975 |
| `cypha-vpattern-matching/` | `Cypha vPattern Matching` | 2026-02-21 | 6 | 3,589,271 |
| `cypha-encoder/` | `Cypha Encoder` | 2026-02-21 | 1 | 36,340 |
| `big-data/` | `Big Data` | 2026-02-21 | 1 | 8,755 |
| `root-monolith/` | `(archive root)` | 2026-03-14 | 1 | 233,256 |
| **total (verbatim copies)** | | | **132** | **10,424,934** |

## Transformed artifacts

| Archived as | Origin | Bytes | Note |
|---|---|---|---|
| `datasets/cypha_classification_settings.jsonl.gz` | `Cypha Classification Settings.jsonl` | 419,579 | gzip -9 of the 20,243,385-byte original |
| `side-quests/retdec-upgrades5/patches5/apply_patches5.sh` | `retdec_upgrades5.zip (extracted)` | 10,102 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/if_structure_ext/if_structure_optimizer_ext.cpp` | `retdec_upgrades5.zip (extracted)` | 5,570 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/if_structure_ext/if_structure_optimizer_ext.h` | `retdec_upgrades5.zip (extracted)` | 489 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/strength_reduction/strength_reduction.cpp` | `retdec_upgrades5.zip (extracted)` | 6,812 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/strength_reduction/strength_reduction.h` | `retdec_upgrades5.zip (extracted)` | 563 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/intrinsic_conv_ext/llvm_intrinsic_converter_ext.cpp` | `retdec_upgrades5.zip (extracted)` | 10,520 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/intrinsic_conv_ext/llvm_intrinsic_converter_ext.h` | `retdec_upgrades5.zip (extracted)` | 452 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/pow2_arithm_ext/pow2_sub_optimizer.cpp` | `retdec_upgrades5.zip (extracted)` | 4,894 | expanded from the nested zip |
| `side-quests/retdec-upgrades5/patches5/pow2_arithm_ext/pow2_sub_optimizer.h` | `retdec_upgrades5.zip (extracted)` | 1,301 | expanded from the nested zip |

## Derived text extractions

31 `.docx` files were also converted to Markdown with a stdlib
(`zipfile` + `xml.etree`) extractor so the prose is greppable and diffable in-repo.
The binary `.docx` originals are retained beside them and remain authoritative.

| Extraction | Source document | Bytes |
|---|---|---|
| `cypha-v8/CyphaREADME.md` | `cypha-v8/CyphaREADME.docx` | 23,112 |
| `cypha-v8/Cypha_Research_Paper.md` | `cypha-v8/Cypha_Research_Paper.docx` | 32,395 |
| `cypha-v8/cypha Coding theory paper.md` | `cypha-v8/cypha Coding theory paper.docx` | 27,744 |
| `cypha-v8/cypha Control theory paper.md` | `cypha-v8/cypha Control theory paper.docx` | 30,153 |
| `cypha-v8/cypha Differential geometry (beyond Fisher–Rao) paper.md` | `cypha-v8/cypha Differential geometry (beyond Fisher–Rao) paper.docx` | 27,895 |
| `cypha-v8/cypha Dynamical systems (deeper — bifurcations, Arnold tongues, basin volumes) paper.md` | `cypha-v8/cypha Dynamical systems (deeper — bifurcations, Arnold tongues, basin volumes) paper.docx` | 31,827 |
| `cypha-v8/cypha Tropical geometry paper.md` | `cypha-v8/cypha Tropical geometry paper.docx` | 31,642 |
| `cypha-v8/cypha_Convex_analysis.md` | `cypha-v8/cypha_Convex_analysis.docx` | 30,292 |
| `cypha-v8/cypha_Harmonic_analysis.md` | `cypha-v8/cypha_Harmonic_analysis.docx` | 30,295 |
| `cypha-v8/cypha_Random_matrix_theory..md` | `cypha-v8/cypha_Random_matrix_theory..docx` | 26,378 |
| `cypha-v8/cypha_galois_paper.md` | `cypha-v8/cypha_galois_paper.docx` | 32,959 |
| `cypha-v8/cypha_markov.md` | `cypha-v8/cypha_markov.docx` | 30,936 |
| `cypha-v8/cypha_persistent_homology.md` | `cypha-v8/cypha_persistent_homology.docx` | 33,091 |
| `cypha-v8/cypha_stat_mech.md` | `cypha-v8/cypha_stat_mech.docx` | 30,486 |
| `cypha-v8/cypha_statpaper.md` | `cypha-v8/cypha_statpaper.docx` | 35,498 |
| `cypha-v8/cypha_synthesis.md` | `cypha-v8/cypha_synthesis.docx` | 50,255 |
| `cypha-v8/cypha_wasserstein.md` | `cypha-v8/cypha_wasserstein.docx` | 35,209 |
| `cypha-v2/Cypha.md` | `cypha-v2/Cypha.docx` | 27,712 |
| `cypha-v5/README_Cypha.md` | `cypha-v5/README_Cypha.docx` | 30,645 |
| `cypha-v5/README_benchmark.md` | `cypha-v5/README_benchmark.docx` | 13,879 |
| `cypha-v5/README_convert.md` | `cypha-v5/README_convert.docx` | 19,873 |
| `cypha-v5/README_download.md` | `cypha-v5/README_download.docx` | 18,751 |
| `cypha-v5/README_synthetic.md` | `cypha-v5/README_synthetic.docx` | 26,699 |
| `cypha-v6/Cypha Encoder Math Proving.md` | `cypha-v6/Cypha Encoder Math Proving.docx` | 45,847 |
| `cypha-v6/Cypha_README.md` | `cypha-v6/Cypha_README.docx` | 31,001 |
| `cypha-v6/README_benchmark.md` | `cypha-v6/README_benchmark.docx` | 13,879 |
| `cypha-v6/README_convert.md` | `cypha-v6/README_convert.docx` | 19,873 |
| `cypha-v6/README_download.md` | `cypha-v6/README_download.docx` | 18,751 |
| `cypha-v6/README_synthetic.md` | `cypha-v6/README_synthetic.docx` | 26,699 |
| `cypha-v6/game_benchmark_README.md` | `cypha-v6/game_benchmark_README.docx` | 32,943 |
| `cypha-v6/game_live_benchmark_README.md` | `cypha-v6/game_live_benchmark_README.docx` | 25,341 |

## Checksums

<details><summary>sha256 of every verbatim copy (click to expand)</summary>

```
c353048f90a846176510e3270ba32059f966961a6e57abe1454654f8930f52fc  prototypes/Brain Model Convo Log.txt
984dbb725880cb0ec3248fd9aec7e21487db14ed532fe94ebb8d681edc416600  prototypes/Brain Model Math Model.md
601cbd43c46bf4fca9173090074547834e389f5454b91fe22d388eb5701241e4  prototypes/Cell AI v2 Thinking Architecture.mermaid
b78786e008e034eda889ca0bcec22603f8185b4bf14dc71aa11d62a49fe9a138  prototypes/Cell AI v2 Thinking Convo Log.txt
80eb96193b92ee1ffe32349fc062d6f112c2da20660b8e5baddb222fa7f93d9f  prototypes/Cell AI v2 Thinking Intro.md
d746be56072b40383617eff977c1c3993a81f18708e4257bce367810b17b8652  prototypes/Cell AI v2 Thinking Math Model.md
0ea8eeec3dcd0cc9f418f2517d2ae2252704c46bbc12767e38007c28312598c9  prototypes/Cell AI v3 Architecture.md
def09f92755ef3e2670dc4e1390aab187efaced912fd21854a15d64bfaecabdd  prototypes/Cell AI v3 Convo Log.txt
a558f5b41fecc2aff0639e484c8c1a426478244fe43c80875533755199d742b5  prototypes/Cell AI v3 Math Model.md
e1fb4badf18a0aab6fb331ab7ea0621c015a6ce61b28172968e453ad1afb0e95  cypha-v1/Cypha Convo Log.txt
23d3ba3b5c96b131c41c9e1decb757a0372449c89af9af7edb6d7e66e9e0b440  cypha-v1/Cypha Speed Improvements.txt
a6d1259f5f48db83f4ea2f44d051043f0b9a3c6e90847366080f0b6652c4af6f  cypha-v1/enhanced-hrna.mermaid
9932cf1cf530ee6c8aa4344858785bb09c91cb93dbe5e35d202b848fb21e84d6  cypha-v1/event-driven-math.md
1b30f67b4295205c7159ba4dc6c90a91c478bc56544ab837410823c621bb025a  cypha-v2/Cypha Architecture Explanation.md
340b5715549d63c6b275ac472dfc464d7562fa1f4099844f08ece8f51bee36e6  cypha-v2/Cypha Architecture.mermaid
0c12a43919306bb05fd3b44597922c49529689f75c01984c0ae6037e05137657  cypha-v2/Cypha Math Framework.md
b7300e2627ad87bf3c0e2f9a7eb2c4375b17e0989eb91a5931de26a2fccd1168  cypha-v2/Cypha.docx
97e83c3a298cc1dd7704b7f5135bf49992fe9b73de27ca7c56e103921424df97  cypha-v2/Cypha.py
9aa0a2b7f502ee4c85ce8bd186caf6fdb53968d60071cbf4a528eee4d542f4f8  cypha-v3/Cypha.py
3a0082e3680e0eb4907483f8060f5de0e64b3c6650b513934b29014e3a083d99  cypha-v3/Cypha_Encoder.py
1a8524498a38a28261adb1f11ed292a587b0e16fc57d1ccbd59ec6410b304eb0  cypha-v3/data.txt
54a50ac8a4039419d4bf80641b0f82f0e815f141cee59674b9499270fc424a39  cypha-v3/generate_data.py
ec8e874c4b1746639e6738631e5ee3a4e2805237f3a8142bb243325e515e68d2  cypha-v4/Cypha.py
842489b8c2db059694f3e6a515866601c2f5969f29a35b3e12dbee9a81f435a9  cypha-v4/benchmark_suite.py
bf47f4980c405712dc2535f068d2acde0fb247543b90c94c3e98a20c78e0152d  cypha-v4/run_demo.py
ae9551a55840d9d3e57d7b77c4444a166990607390d372d139c705c3a997b4e1  cypha-v4/verify_thinking.py
91406b73e871c98b70c8dc54b435e9b50ae017f33b161f7c4d124bb78c45ec7c  cypha-v5/Cypha.py
0fc2cf3e2f534a3eba9d987f8951e090e59b505c74a646aad8b184716f47c67e  cypha-v5/README_Cypha.docx
d72b30167b7fe35869b287509c1409a533560f0549576106663b86bc445b23b9  cypha-v5/README_benchmark.docx
515987a63ae96dd4be301d44425ae83097104aa93a9fea9ba6a9796d52b521a2  cypha-v5/README_convert.docx
6e8b2cc7d1791a14a95efc37da4ebbe6b22df4bffe8f7269abff76288477c67c  cypha-v5/README_download.docx
3224f4be5252a415f2bb2b1df25c20ea0d1abaf256957d8120eed7cc77f06e04  cypha-v5/README_synthetic.docx
e53d1ba2ad8e7dd02a7afb5c6eeb62b2ec7755be6b672e6be4aa6ff9493f56b0  cypha-v5/benchmark.py
4b1c7d053b4fc7dd2fef17e6a50d87d96ce7a195f863e4ee690876a45c145eec  cypha-v5/convert.py
2a1f15101c9a43ca24aab5344ef38ba0e45d7f1b31ad4b31385cdfdcefc74779  cypha-v5/download.py
66fdac310a32a93ae47db3816303263a5d07972c4ec73a94d70aca4539bf2340  cypha-v5/synthetic_benchmark.py
982ca324b316fe66e317e40683822329fbe2b4e241f7811e552583f20b289ce0  cypha-v6/Cypha Encoder Math Proving.docx
522d9dd05e3f326949565bff136aa551083eaa20cd3eeb4a837bd5f0d4172368  cypha-v6/Cypha.py
842ec3e9cac1d627946be183724841c52aa98f767aeb136866bae4666b89035a  cypha-v6/Cypha_README.docx
7f2ec7c1cab374f41486202ce53e5b766387cbe07e4774985bc1e8283a4f6e6f  cypha-v6/Profiling/cypha_profiler1_transition.py
ff64ad6d79438e17d4042237b0a0f5809b766b5c285f3da4749c1a459621bb39  cypha-v6/Profiling/cypha_profiler2_shift.py
880ac152a4b09906bc66ff11e9622ef7b5ea050cfe38cdb02349be3981f2cd86  cypha-v6/Profiling/cypha_profiler3_alpha.py
f185f4ed5717c0f1a6c49a4319c20d05faf25e7b234babde0b5e5663c040dfeb  cypha-v6/Profiling/cypha_profiler4_archetype.py
9f1979acf6643c15c6dbd98d67bf87256669deece8815437ce3bcb476d134aa8  cypha-v6/Profiling/profiler1_transition_report.txt
a8ebe2b119469892a398c4d69f1eb6d3415f06e4a47e8ddca89c53995d1b7683  cypha-v6/Profiling/profiler1_transition_results.json
94deee87a03a21e8560d6f59474a06be2d57756518c82a516344a806e641517a  cypha-v6/Profiling/profiler2_shift_report.txt
304bd591a9fe951825581c0f5cd8260d8208a3a618da55beac0d45e896046af6  cypha-v6/Profiling/profiler2_shift_results.json
5ea36abcbbd58ccfbe94be18ba3abb1a16ff203a75b1829dd3595d3ebcf9a0a0  cypha-v6/Profiling/profiler3_alpha_report.txt
db2ef756091ddfa931cd0798eba27cc14cf4ec76e383d538c5307a88d2579a77  cypha-v6/Profiling/profiler3_alpha_results.json
70dd9ae9fd9f3b64fe6e1085c7394ac67d8dcde91f6adc722936d54ff48c69b2  cypha-v6/Profiling/profiler4_archetype_report.txt
3ea5fc0c676bb9adacb1e4968c97dbb33cf6e706f141e61dabffdb8053def456  cypha-v6/Profiling/profiler4_archetype_results.json
d72b30167b7fe35869b287509c1409a533560f0549576106663b86bc445b23b9  cypha-v6/README_benchmark.docx
515987a63ae96dd4be301d44425ae83097104aa93a9fea9ba6a9796d52b521a2  cypha-v6/README_convert.docx
6e8b2cc7d1791a14a95efc37da4ebbe6b22df4bffe8f7269abff76288477c67c  cypha-v6/README_download.docx
3224f4be5252a415f2bb2b1df25c20ea0d1abaf256957d8120eed7cc77f06e04  cypha-v6/README_synthetic.docx
e53d1ba2ad8e7dd02a7afb5c6eeb62b2ec7755be6b672e6be4aa6ff9493f56b0  cypha-v6/benchmark.py
4b1c7d053b4fc7dd2fef17e6a50d87d96ce7a195f863e4ee690876a45c145eec  cypha-v6/convert.py
191a74dc378019d929ab54e25d177bd109fa0fc26b42067419b3db11b44fc43e  cypha-v6/cypha_multimodal_profile.py
2a1f15101c9a43ca24aab5344ef38ba0e45d7f1b31ad4b31385cdfdcefc74779  cypha-v6/download.py
42e33bf70e92578c2dc5b84b1ef0c74bf56cbaa36e65a3a97528c79c301ecb81  cypha-v6/game_benchmark.py
dc7cf8e3f40e2d0f2076781ec7c8b0f4750a99e37b25c9d28d8c315bd00f4907  cypha-v6/game_benchmark_README.docx
fbca7bc55e615eeb3bfa3515b553cf911f70ba1f22273241896506fb74494d2c  cypha-v6/game_live_benchmark.py
52c9d27f298d54851f716adabeb056d552a4f3f18e8800e3ee7ef04d4b6df72c  cypha-v6/game_live_benchmark_README.docx
b5b7eea5d7e2fc1952d3ccb287176babfcf58796ce317fb20e9649c9cfc2cac3  cypa-v7-generation/Cypha.py
2075165b5f52829139ad281bdec50f45f24f7548f5ac6885a41616c0dfc78ce9  cypha-v8/Cypha.py
00e665a05bb62c827d18673235b9142de50b892175a717099dbd39a112a46b07  cypha-v8/CyphaREADME.docx
9fb03e1d9288883733a5eb523b9684977f4420c109f2d89d3e840ee9b4dedfd3  cypha-v8/Cypha_Research_Paper.docx
ffd2b8253b84cf274b6532ef061d9316d37a273a10bf7ff037cd84e19dd6cb8b  cypha-v8/cypha Coding theory paper.docx
aa5b8ed416d63801aa07fda28aba03ec6206a38c4945f1ddd1e0bff4cda90cbe  cypha-v8/cypha Control theory paper.docx
a064cd0301fb7cfb36ea0b8657dd56c70b4aa5c23ba960b742916228bc99f206  cypha-v8/cypha Differential geometry (beyond Fisher–Rao) paper.docx
262c77704d241ca127dcb965904a67c1f31636661e94c222ac3508865a3ee6e5  cypha-v8/cypha Dynamical systems (deeper — bifurcations, Arnold tongues, basin volumes) paper.docx
cdf628167dcf30b8653b593c8d5ff252bd436bd4c1eee066caf5a19e2c8922ca  cypha-v8/cypha Tropical geometry paper.docx
24c6a5cb4039e8b87d7284ded078f72716de30dff0f2d368447c460be765cd62  cypha-v8/cypha_Convex_analysis.docx
73e0a9a4b47f1605e554bf91212f368ce53bfa0e35417966dbcebf1ba9a7ccf0  cypha-v8/cypha_Harmonic_analysis.docx
74c87b88d537c9db3f9896d60927337c8c9f1509712e4f8ca2f076b226369427  cypha-v8/cypha_Random_matrix_theory..docx
ed216380951c01cfcd229da1ea850bfc16400ae3618ada22aa47a10564736905  cypha-v8/cypha_galois_paper.docx
d1a20a1139dbed8b340dd55f37ab52184ac4f30069b545ee5c91b5a3fc471703  cypha-v8/cypha_markov.docx
41457e6c32e8da64cfcf685e51e9e747318b977aab26d93f14c4aa5832078dda  cypha-v8/cypha_persistent_homology.docx
c43987ba86bb58df843a8295cd9e692f0d615b05c7dc0e4cb7d0f00d4924ab0f  cypha-v8/cypha_stat_mech.docx
2f6b889c6545ead61bc930210547443e327ba9a3d60a9bba6bfd72d665165e1a  cypha-v8/cypha_statpaper.docx
e7e086e2f746a02e066c56442baac54c43b38fe3f11cdfd9f654f4ec1bb8b11c  cypha-v8/cypha_synthesis.docx
5f7f90bd4ce8973cf7d3a5538a58b58026f8c4b86efc1a215529253aaba9a9dd  cypha-v8/cypha_wasserstein.docx
3653221875c193a104d56cb70a79f08b68f0b2300d9b4e3941cdaa04f6349434  cypha-vchatgpt/agi/code.py
7927161b1231967f9fd0210af533631da84ad0c3a697c98cdcdb16d88a88de7a  cypha-vchatgpt/agi/files.py
9f3e96d8693e93bd28c23e8d8532cd3436e18f8516ae695249603cafc4c9ccf4  cypha-vchatgpt/agi/memory.py
555389e1f44c4ca387599abd9d3eba073fe489c52030eefbb0e6fd0fccc37457  cypha-vchatgpt/agi/monitoring.py
49f7f4f24abe52ee1415e53b57213dee19b0d9656cd3095460dab716d4816347  cypha-vchatgpt/agi/nlp.py
8dc371c60c53eeaffacd570ba1c805126ede45c13e7c124864144541b0c74047  cypha-vchatgpt/agi/reasoning.py
beba421fb8efb40adf4208e465fd72ccf60427ea30ea46dc30041b8f212a2ebc  cypha-vchatgpt/agi/system.py
b7f51a5a6ef10714e69f773e381e0746a963fcb3354304c4d8cb5404a1069ca2  cypha-vchatgpt/config.yaml
3b3a7c872e59eea14d391204ab25a6251b1d040a4e5ebb87936ac2cec94b986a  cypha-vchatgpt/core/compression.py
4dc8414a86b2a949423de5b0754140c7be8ceb660039128d53bda44e00487d2b  cypha-vchatgpt/core/encoder.py
f412364835e838d537c77f1d39fb620aa43f30300575c8dab02465ef5bfaeae1  cypha-vchatgpt/core/events.py
86c3e4fbfd28d04a9539c18f91f977fd74aefdee1169189c9b0e8f3b7f34ae12  cypha-vchatgpt/core/feedback.py
57e5991c3a6cd253155bace63541eaec35a6102d6fed4ef38948c2637b1fc888  cypha-vchatgpt/core/levels.py
29b344601078b2f4a119187522392d419db272a769e87a1ba17cbad0ad69e4e5  cypha-vchatgpt/core/metalearning.py
c3fa73fc927b323226552075794f52db70a553ce4b89a88c3f03a1f801271944  cypha-vchatgpt/core/optimization.py
0b608b95d465a9ef8a82655f7722cd203d9e177642b18e1d5cbe7adedf75362a  cypha-vchatgpt/core/recursion.py
7706d7cf51f797ddc1c4cd985f30109bedd3640c620eda74efbefa32c698a0ff  cypha-vchatgpt/core/resonance.py
56558ae69040579c926b933928c6b31adfa16776ed37a65290f2bc012b75cc8b  cypha-vchatgpt/core/thought.py
74452f7287b61fc6109f5b467c45034d93906271fd9450f48e537605ea884e5f  cypha-vchatgpt/cypha.py
599444c55faa5628613b1953642c62a0a96550dd8e58fe51cf94fa7eb00af909  cypha-vchatgpt/data.py
8dcbbe64fe449fe723df7910c2f32a344659f1137deec5aa879085a16359f0f7  cypha-vchatgpt/data.txt
880cbd136794b2e35244ad0e00ce32794dc7f2436e4ce0455e0dc069ee85090b  cypha-vchatgpt/diagnose_layers.py
c6090da0694253069bfc855bd785a7f76e1089af078e7788b2d79dc4ed43b558  cypha-vchatgpt/gui/chat.py
de433a737c5d870193ca1000f5b1d2928f9542e0794bda9e82b45f295039eebc  cypha-vchatgpt/gui/main_window.py
edec5f151c5584611b743b3c05d276592d05a4c6cdb15959041b6003b3a7949f  cypha-vchatgpt/gui/monitor.py
4cc63814a89ba3fa5c56380758c4f9aef1fc9c520d02f48e59691a90201eeab2  cypha-vchatgpt/gui/settings.py
d937b3021cd2da792f006656673c645f13fd0cd67454f3987581b7baa431759b  cypha-vchatgpt/gui/training.py
7e508a5e820770793c0ca6a1312d88fad5defd6136bdd351dca5656788fc441b  cypha-vchatgpt/learning/pile_loader.py
e795875ed85abb5ece2da36a5163d3af56d9955ff10d19e18163efd77c867d3a  cypha-vchatgpt/learning/pipeline.py
8dd5dc3f7559e3c4f928eb6b5598fd5fb7fdccd9e4c777f64253cd60f16682be  cypha-vchatgpt/learning/scraper.py
f18d3287276231d2f9fa5a6a0b7c756fcf3779d19304d8ba6fff414e87c64768  cypha-vchatgpt/learning/trainer.py
758b3e792af4aa3ba94c14480356f28631b8d8c148b5bef44929f0deaf28eabd  cypha-vchatgpt/requirements.txt
94e4f2124d3543c66fb8619c5bc4c72a918b85c3463a0f9e6782e6f0620a3693  cypha-vchatgpt/security/audit.py
76ca99524d9ee8f2d3178dda19ecc30b94d1819841fc708d70e16f919794ec2a  cypha-vchatgpt/security/sandbox.py
a796af801b87e097e63f9fa31bd453176e7d8ff7ac6d288b9fbb2941d94adeaf  cypha-vchatgpt/security/validation.py
e4f7ed0e2cec3a8ce8f21f5b6ca65e865300fccd786dc47982c35406775d40d8  cypha-vchatgpt/test_inference.py
df60a1d87962a2ab1a0fc0ef894e8afbb6fef710f51e193775f68bcd529b41a3  cypha-vchatgpt/tests/test_agi/test_code.py
db26decc9d53794585e7c1c87582e54c1f91ae5443b6cd9aaf69bb183f4fc405  cypha-vchatgpt/tests/test_core/test_compression.py
c562b514724e62f5d12dc7a3320fdd7dbff6b27d84b57bcc373f7fec5c24300a  cypha-vchatgpt/tests/test_core/test_encoder.py
4bfc6433609e04c01520774e7718bb84358d587a72bf91e4a8b9b031621bed0d  cypha-vchatgpt/tests/test_core/test_resonance.py
dfafab37399b6b682b442436780922e90dcc1269ad08308b643b84e6f951c90d  cypha-vchatgpt/tests/test_learning/test_scraper.py
61eea1a94c271fd933d8cd6182077070199a8e6a0d45494093f8b174b1ee6b60  cypha-vpattern-matching/Cypha Demo.pdf
609d448db48ad6ef7f753f0d0e2aa95601cae0990dbe05c412abcbdd3e974015  cypha-vpattern-matching/QUICK_START.md
29ccefe9ee1470d8b99cda3fc0565081ad844a8ffb1b0de6aa3e7f34d844580b  cypha-vpattern-matching/README.md
47eb1ccf95a738e496cf2b76c9eba113108f62aab43940b2acb961d0f31f0f26  cypha-vpattern-matching/cypha_production.py
84825cb20f644e83e6346a73eae703ff982173d569a2ca29f749a274f02b0f68  cypha-vpattern-matching/generate_demo_data.py
7e5b7de329939ce48e52b5fac694af4382cea6c71f9942e8b647070844de1c44  cypha-vpattern-matching/showcase_demo.py
3a0082e3680e0eb4907483f8060f5de0e64b3c6650b513934b29014e3a083d99  cypha-encoder/Cypha_Encoder.py
b731e914374d8a0eca9d0dd8c46207909dc233043ba15b640b89594c004621d2  big-data/download_datasets.py
c94174964c56095a25e5c30fd1b9b45701cd221a36b833917636a6ee1141df56  root-monolith/Cypha.py
```
</details>

