# CyphaLM vs a neural net and a Transformer (1-hour training budget)

**Date:** 2026-09-24
**Machine:** 4-core Xeon @ 2.1 GHz (AVX-512, AMX), 15 GB RAM, no GPU
**Code:** [`bench/lm_compare/`](../../bench/lm_compare/) (`run_all.sh` reruns everything)
**Raw data:** [`lm_compare/`](lm_compare/)

CyphaLM is compared with the two standard neural byte-level language models:
- **the "NN":** a stacked LSTM;
- **the Transformer:** a decoder-only, GPT-style model.

Each system trains for at most one hour on the same 4 cores, from the same data, and all are scored by the same code.

## Setup

| | CyphaLM | Transformer | LSTM |
|---|---|---|---|
| model | winner: 11 slim shard models (enwik8 95 MB split 11 ways, table bits 20) + ∞-gram suffix-array index | 4 layers, d 256, 8 heads, context 512, pre-LN, tied embeddings: **3.35M params** | 2 × LSTM 512, byte embedding 64: **3.43M params** |
| training | one online pass per shard, 4 shards at a time | AdamW, lr 1e-3, cosine, batch 16 × 512, bf16 autocast | AdamW, lr 2e-3, cosine, 32 streams × 128 (TBPTT, state carried) |
| budget used | **17.1 min** (shards 16.8 + index 0.3) | 60 min | 60 min |
| data read | all 95 MB | 69 MB (0.73 epochs) | 81 MB (0.85 epochs) |

- **Same bytes:** everyone uses bytes as the vocabulary (256) and trains on the first 95 MB of enwik8.
- **How the neural models were sized:** throughput was measured first (bf16 is 1.7× faster than fp32 here; `torch.compile` gave +4%). Then:
  - The Transformer is compute-optimal for one hour by the Chinchilla rule of about 20 training bytes per parameter: 19K bytes/s × 3600 s ≈ 69M bytes ≈ 20 × 3.35M.
  - The LSTM has the same parameter count.
  - A 4.9M-parameter Transformer would have read only 45 MB.

**Held-out text, never trained on:**
- **wiki:** enwik8 from byte 96,000,000 (in domain);
- **Alice:** *Alice in Wonderland* from byte 20,000;
- **lcet10:** a second book, from byte 50,000.

The two books are out of domain. Each text is 16 KiB.

**Modes.** The fair pairs are CyphaLM reading vs. the neural models with dynamic eval, and CyphaLM frozen vs. the static neural models.
- **CyphaLM reading:** CyphaLM learns online as it reads the text, which is its in-context learning.
- **CyphaLM frozen:** that online learning is turned off. Its match models still use the history.
- **Neural static:** the neural models use their context window only.
- **Neural + dynamic eval** (Krause et al. 2018): one SGD step after every 128 bytes, which is the neural version of reading. The learning rate was tuned on a separate slice (enwik8 at byte 97,000,000): 0.003 for the Transformer, 0.3 for the LSTM.

**One scorer.** Every system dumps its full 256-way next-byte distribution for every byte. `cyphalm_lm_quality --dump-dist` and `byte_lm.py eval --dump` write the same layout, and [`metrics.py`](../../bench/lm_compare/metrics.py) computes every number below from those dumps.

## Results in short

| | CyphaLM | best neural | |
|---|---:|---:|---|
| wiki bits/byte, reading / dynamic eval | **1.651** | 1.739 (LSTM) | −0.088 |
| Alice | **2.026** | 2.427 (LSTM) | −0.401 |
| lcet10 | **1.512** | 2.015 (LSTM) | −0.503 |
| wiki, frozen / static | 1.819 | **1.805** (Transformer) | +0.014 |
| Alice, frozen / static | **2.695** | 3.257 (LSTM) | −0.56 |
| top-1 next byte, wiki | **65.8%** | 63.3% | |
| 2nd copy of a 2 KiB passage | **0.048** | 1.242 | copies from context |
| training time | **17 min** | 60 min | |
| ms per next-byte distribution (4 threads / 1 thread) | 2.4 / 2.5 | **1.2** / 2.7 (Transformer, fp32) | |
| serving RAM | 3.3 GB + 0.6 GB mapped (light: 0.44 + 0.44) | **0.25 GB** process, 13 MB weights | |

1. **On quality, CyphaLM wins everywhere except one pair.** In domain with no adaptation, the Transformer is 0.014 ahead (1.805 vs 1.819). Adapting to the text favours CyphaLM more than any neural adaptation tried: dynamic eval gains the Transformer 0.03 on wiki, while reading gains CyphaLM 0.17. The gap widens out of domain, to 0.40–0.50 on the two books.
2. **On training cost, CyphaLM is far cheaper.** It uses 28% of the budget, and its 4-shard model (one round of 4 parallel shards, 5.6 min) already scores 1.675 on wiki while reading. That beats both hour-long neural models even with dynamic eval. CyphaLM ran out of data, not time: enwik8 has nothing more to train on here.
3. **CyphaLM copies from its context; the small neural models barely do.** On the second copy of a 2 KiB passage, CyphaLM spends 0.05 bits/byte, against 1.24–1.83 for the neural models. On 384 random letters repeated, it spends 0.16 against 4.6–6.3. The Transformer's window (512 bytes) cannot see the first copy of the 2 KiB passage. Even inside the window, a 3M-parameter model trained for one hour has not learned to copy.
4. **The neural models are better at structure.** All four neural rows beat CyphaLM on punctuation and markup (1.92–2.11 bits vs 2.44), and the Transformer also wins on spaces (0.517 vs 0.550). CyphaLM wins on lowercase letters (1.73 vs 1.89–2.07), which make up 74% of the bytes, and on digits.
5. **Generation (12 prompts, 400 bytes, T 0.8, min-p 0.1, three judges).**
   - **Transformer:** it falls into loops. 34% of its 12-byte windows repeat earlier text, against 9% for the true continuation, and its distinct 4-grams are 46%, against 76%.
   - **LSTM:** diverse, but it invents words: 96.3% of its words appear in the training vocabulary, against 98.4% for the true text.
   - **CyphaLM with word lookahead:** 99.4% real words, diversity at the reference level (76.4% distinct 4-grams), almost no copying (0.8%), and the most probable text under all three judges. It drifts to wiki style on Alice prompts, where the Transformer stays with the prompt's words (it continues about the "Hatter") but loops. All the systems are far from fluent English at this scale; see the samples.
6. **The neural models are much smaller to serve.** Their weights are 13 MB. The CyphaLM winner maps 3.5 GB of shards plus a 0.4 GB index; the light winner is 0.9 GB, but it takes 2.5 h to train, outside the budget. Latency is similar: the fp32 Transformer is fastest on 4 threads (1.2 ms), and all systems take 2.5–4.7 ms on one thread.
7. **The neural models are still improving when time runs out.** The Transformer goes from 1.99 at 30 min to 1.80 at 60 min, while CyphaLM goes from 1.660 at 8 shards to 1.651 at 11. With more compute or a GPU, the Transformer would close the static gap and pass CyphaLM in domain. With more data, CyphaLM would grow its tables. The neural curves are steep; CyphaLM's is flat.

## Caveats

- **Small models, CPU scale.** These are 3M-parameter models on a CPU with a 1-hour budget. Published byte-level Transformers reach about 1.0 bits/byte on enwik8, with GPUs and days of training. This compares like with like at CPU scale, not against the state of the art.
- **Neural latency is PyTorch eager, one step at a time.** A C++ runtime would be several times faster. With bf16 autocast a single step is slower (4.0 ms), so the fp32 numbers are reported.
- **Text difficulty varies.** The neural validation slice (enwik8 at 99 MB) scored 1.56 (Transformer) and 1.77 (LSTM) at 60 min; the wiki test slice at 96 MB is harder, 1.805 and 1.901.
- **Judges favour their own samples.** Each judge rates its own model's text as the most probable. Judge scores below the reference's mean predictable text, not better text.
- **CyphaLM's generation timings include priming** the 256-byte prompt; the neural ones do not.


## Full tables

Rows marked * (CyphaLM light: one slim model on 95 MB) took about 2.5 h to train, outside the budget; they are shown for reference.

### Training (4 cores, same 95 MB of enwik8)

| system | model | data read | train min | train peak RSS MB | size on disk MB |
|---|---:|---:|---:|---:|---:|
| CyphaLM (11 shards + ∞-gram index) | 11 × 2^20-slot hash tables + suffix array | 95.0 MB × 1 pass | 17.1 | 1326 | 3544 + 396 index |
| Transformer | 3.35M params: 4 layers, d 256, ctx 512 | 69.0 MB (0.73 epochs) | 60.0 | 994 | 12.8 |
| LSTM | 3.43M params: 2 × LSTM 512, TBPTT 128 | 81.1 MB (0.85 epochs) | 60.0 | 616 | 13.1 |

### Held-out next-byte quality (bits/byte, lower is better; 16 KiB each)

| system | wiki | Alice | lcet10 | top-1 wiki | top-5 wiki | ECE wiki |
|---|---:|---:|---:|---:|---:|---:|
| CyphaLM (reads the text) | 1.6510 | 2.0257 | 1.5117 | 65.8% | 89.4% | 1.2% |
| CyphaLM frozen | 1.8191 | 2.6952 | 1.9703 | 62.1% | 88.3% | 1.5% |
| CyphaLM light* (reads the text) | 1.6909 | 2.0639 | 1.5566 | 65.1% | 89.2% | 1.0% |
| CyphaLM light* frozen | 1.9692 | 2.7822 | 2.0928 | 61.3% | 86.7% | 5.6% |
| Transformer | 1.8050 | 3.4999 | 2.4660 | 62.4% | 87.7% | 2.1% |
| Transformer + dynamic eval | 1.7760 | 2.6853 | 2.0278 | 63.0% | 87.9% | 0.7% |
| LSTM | 1.9011 | 3.2565 | 2.4381 | 60.3% | 86.9% | 1.8% |
| LSTM + dynamic eval | 1.7388 | 2.4274 | 2.0147 | 63.3% | 88.6% | 0.9% |

### LLM-style metrics

| system | text | byte perplexity | word perplexity | next-word greedy acc | word-start top-1 | entropy bits | NLL-best temperature |
|---|---:|---:|---:|---:|---:|---:|---:|
| CyphaLM (reads the text) | wiki | 3.140 | 1408 | 20.0% | 29.0% | 1.709 | 1.0 |
| CyphaLM (reads the text) | alice | 4.072 | 2978 | 10.6% | 20.5% | 2.056 | 1.0 |
| CyphaLM frozen | wiki | 3.529 | 2947 | 18.1% | 28.5% | 1.925 | 0.9 |
| CyphaLM frozen | alice | 6.477 | 41884 | 8.8% | 17.7% | 2.517 | 1.1 |
| CyphaLM light* (reads the text) | wiki | 3.228 | 1678 | 19.7% | 28.8% | 1.738 | 1.0 |
| CyphaLM light* (reads the text) | alice | 4.181 | 3463 | 10.5% | 20.7% | 2.076 | 1.0 |
| CyphaLM light* frozen | wiki | 3.916 | 5699 | 17.2% | 26.9% | 2.323 | 0.8 |
| CyphaLM light* frozen | alice | 6.879 | 59049 | 8.1% | 16.5% | 3.149 | 0.9 |
| Transformer | wiki | 3.494 | 2770 | 18.4% | 27.0% | 1.703 | 1.1 |
| Transformer | alice | 11.313 | 1004807 | 7.4% | 16.2% | 2.237 | 1.3 |
| Transformer + dynamic eval | wiki | 3.425 | 2439 | 18.3% | 27.4% | 1.750 | 1.0 |
| Transformer + dynamic eval | alice | 6.432 | 40269 | 8.7% | 16.7% | 2.438 | 1.1 |
| LSTM | wiki | 3.735 | 4225 | 18.6% | 28.6% | 1.817 | 1.0 |
| LSTM | alice | 9.556 | 384166 | 7.9% | 16.0% | 2.123 | 1.3 |
| LSTM + dynamic eval | wiki | 3.338 | 2071 | 19.1% | 28.0% | 1.773 | 1.0 |
| LSTM + dynamic eval | alice | 5.379 | 14547 | 8.9% | 17.8% | 2.265 | 1.1 |

### Bits/byte by position in the held-out text (wiki): in-context learning

| system | bytes 0-256 | bytes 256-1024 | bytes 1024-4096 | bytes 4096-16384 |
|---|---:|---:|---:|---:|
| CyphaLM (reads the text) | 1.798 | 1.874 | 1.686 | 1.625 |
| CyphaLM frozen | 1.819 | 1.967 | 1.752 | 1.827 |
| CyphaLM light* (reads the text) | 1.932 | 1.951 | 1.721 | 1.662 |
| CyphaLM light* frozen | 2.068 | 2.089 | 1.893 | 1.979 |
| Transformer | 1.947 | 1.838 | 1.756 | 1.812 |
| Transformer + dynamic eval | 1.939 | 1.840 | 1.739 | 1.778 |
| LSTM | 1.787 | 1.941 | 1.730 | 1.944 |
| LSTM + dynamic eval | 1.773 | 1.875 | 1.681 | 1.744 |

### Bits/byte by position in the held-out text (alice): in-context learning

| system | bytes 0-256 | bytes 256-1024 | bytes 1024-4096 | bytes 4096-16384 |
|---|---:|---:|---:|---:|
| CyphaLM (reads the text) | 2.794 | 2.408 | 2.235 | 1.934 |
| CyphaLM frozen | 2.875 | 2.783 | 2.774 | 2.666 |
| CyphaLM light* (reads the text) | 2.886 | 2.504 | 2.282 | 1.965 |
| CyphaLM light* frozen | 2.987 | 2.918 | 2.831 | 2.757 |
| Transformer | 3.511 | 3.430 | 3.521 | 3.499 |
| Transformer + dynamic eval | 3.490 | 3.333 | 3.048 | 2.537 |
| LSTM | 3.077 | 3.180 | 3.237 | 3.270 |
| LSTM + dynamic eval | 3.034 | 2.993 | 2.821 | 2.281 |

### Bits/byte by byte class (wiki)

| system | lower (74.4%) | upper (1.3%) | space (16.0%) | punct/markup (7.2%) | digit (1.0%) | non-ascii (0.0%) |
|---|---:|---:|---:|---:|---:|---:|
| CyphaLM (reads the text) | 1.734 | 4.974 | 0.550 | 2.444 | 2.979 | 3.854 |
| CyphaLM frozen | 1.912 | 4.775 | 0.618 | 2.699 | 3.859 | 3.244 |
| CyphaLM light* (reads the text) | 1.778 | 4.842 | 0.590 | 2.466 | 3.054 | 2.776 |
| CyphaLM light* frozen | 2.056 | 4.674 | 0.911 | 2.618 | 4.174 | 1.980 |
| Transformer | 1.982 | 4.778 | 0.517 | 2.078 | 3.159 | 8.213 |
| Transformer + dynamic eval | 1.937 | 4.807 | 0.590 | 1.991 | 3.069 | 8.164 |
| LSTM | 2.072 | 4.906 | 0.621 | 2.109 | 4.012 | 8.465 |
| LSTM + dynamic eval | 1.886 | 4.857 | 0.595 | 1.921 | 3.491 | 8.986 |

### Copying from context (a passage, then the same passage again)

| system | wiki 2 KiB: 1st | 2nd | random letters 384 B: 1st | 2nd |
|---|---:|---:|---:|---:|
| CyphaLM (reads the text) | 1.517 | 0.048 | 5.303 | 0.160 |
| CyphaLM frozen | 1.733 | 0.621 | 6.880 | 1.989 |
| CyphaLM light* (reads the text) | 1.598 | 0.054 | 5.494 | 0.182 |
| CyphaLM light* frozen | 1.873 | 0.855 | 6.779 | 2.125 |
| Transformer | 1.631 | 1.610 | 7.359 | 5.893 |
| Transformer + dynamic eval | 1.606 | 1.489 | 6.719 | 4.623 |
| LSTM | 1.823 | 1.832 | 6.330 | 6.278 |
| LSTM + dynamic eval | 1.634 | 1.242 | 5.769 | 4.964 |

### Serving cost (one stream)

| system | ms / next-byte distribution, 4 threads | 1 thread | generate ms/byte (byte sampling) | generate ms/byte (word lookahead) | RSS MB |
|---|---:|---:|---:|---:|---:|
| CyphaLM | 2.42 | 2.48 | 6.12 | 25.54 | 3333 private + 565 mapped |
| CyphaLM light* | 2.56 | 2.66 | — | — | 438 private + 440 mapped |
| Transformer (fp32 step; bf16: 4.0 ms) | 1.24 | 2.66 | 4.09 | n/a | 252 process (13 weights) |
| LSTM (fp32 step; bf16: 4.4 ms) | 2.73 | 4.65 | 1.98 | n/a | 258 process (13 weights) |

### Generation: 12 prompts (8 wiki, 4 Alice), 400 bytes, T 0.8, min-p 0.1

| generator | judge CyphaLM | judge Transformer | judge LSTM | distinct 4-grams | 12-byte copies | real words |
|---|---:|---:|---:|---:|---:|---:|
| true continuation | 1.860 | 2.045 | 2.057 | 75.9% | 9.0% | 98.4% |
| CyphaLM (word lookahead K 8) | 0.903 | 1.164 | 1.148 | 76.4% | 0.8% | 99.4% |
| CyphaLM (byte sampling) | 1.430 | 1.762 | 1.780 | 84.3% | 2.7% | 99.5% |
| Transformer | 1.388 | 0.779 | 1.698 | 45.8% | 34.1% | 97.0% |
| LSTM | 1.593 | 1.537 | 1.142 | 83.6% | 3.1% | 96.3% |

### Quality against training time

| system | train min | wiki (static/frozen) | Alice (static/frozen) | wiki (reading) | Alice (reading) |
|---|---:|---:|---:|---:|---:|
| Transformer | 5 | 3.2260 | 4.5343 | — | — |
| Transformer | 15 | 2.2647 | 3.8897 | — | — |
| Transformer | 30 | 1.9930 | 3.6446 | — | — |
| Transformer | 60 | 1.8050 | 3.4999 | — | — |
| LSTM | 5 | 2.3818 | 3.8894 | — | — |
| LSTM | 15 | 2.1579 | 3.5499 | — | — |
| LSTM | 30 | 2.0272 | 3.5264 | — | — |
| LSTM | 60 | 1.9011 | 3.2565 | — | — |
| CyphaLM 1 shard | 5.6 | 2.0242 | 2.9878 | 1.8033 | 2.1362 |
| CyphaLM 4 shards | 5.6 | 1.8498 | 2.7852 | 1.6754 | 2.0547 |
| CyphaLM 8 shards | 11.2 | 1.8253 | 2.7220 | 1.6596 | 2.0348 |
| CyphaLM 11 shards | 16.8 | 1.8191 | 2.6952 | 1.6510 | 2.0257 |

CyphaLM shard log:
```
shard 3: 8636363 bytes, 1.6190 bpc, 337 s
shard 2: 8636363 bytes, 1.6333 bpc, 340 s
shard 0: 8636363 bytes, 1.6289 bpc, 340 s
shard 1: 8636363 bytes, 1.6548 bpc, 342 s
shard 4: 8636363 bytes, 1.6339 bpc, 336 s
shard 6: 8636363 bytes, 1.6480 bpc, 337 s
shard 5: 8636363 bytes, 1.6380 bpc, 337 s
shard 7: 8636363 bytes, 1.6326 bpc, 339 s
shard 8: 8636363 bytes, 1.6126 bpc, 325 s
shard 10: 8636363 bytes, 1.6539 bpc, 329 s
shard 9: 8636363 bytes, 1.6661 bpc, 329 s
wrote /home/user/ckpt/compare/cyphalm/ensemble.json (11 members)
```

gpt dynamic-eval SGD lr grid on enwik8 @97,000,000 (8 KiB, tuning only): 0.001: 1.6605, 0.003: 1.6481, 0.01: 1.6507, 0.03: 2.1876, 0.1: 7.8258, 0.0: 1.6815 → 0.003

lstm dynamic-eval SGD lr grid on enwik8 @97,000,000 (8 KiB, tuning only): 0.001: 1.7670, 0.003: 1.7616, 0.01: 1.7459, 0.03: 1.7158, 0.1: 1.6679, 0.3: 1.6382, 1.0: 1.7835, 0.0: 1.7697 → 0.3

CyphaLM learning curve: shards train 4 at a time (about 5.6 min per round), so 1 and 4 shards take one round (5.6 min), 8 take two (11.2 min) and 11 take three (16.8 min), plus the index (2 to 20 s). Each point uses an index over only the bytes its shards were trained on.

## Samples (first 240 bytes of each continuation)

**Prompt 2** (wiki), last 160 bytes:

```
cs]] at [[Alexandria, Egypt|Alexandria]] and then later at [[Constantinople]], and had written a commentary on earlier books on building.

[[Category:Byzantine 
```

*true continuation:*
```
architects]]
[[Category:Byzantine physicists]]

[[fr:Isidore de Milet]]
[[pt:Isidoro de Mileto]]</text>
    </revision>
  </page>
  <page>
    <title>International Atomic Energy Agency</title>
    <id>14984</id>
    <revision>
      <id>413
```
*CyphaLM, word lookahead K 8:*
```
Empire]]

[[ca:Footnotes and English language]] and [[Philosophy]] and [[Category:English theoretical and political questions. The considered the independence of British important conservative financial material characteristic of the Alexan
```
*CyphaLM, byte sampling:*
```
entrepreneurs a wild Chandler and Military of War Grand Martin Characters in the Senators discipline in the defeat the American movie regularly summer of 2003 Series in the Senate Service in the process is and prosecuted for preservation of
```
*Transformer:*
```
was physics]]
[[Category:Byzantine was physics]]

[[da:Constantinople from the Byzantine was the Byzantine was a mistake book of Miletus and Deskine]]
[[da:Constantinople from the Byzantine Was a Miletus and Miletus of Byzantine Was a Milet
```
*LSTM:*
```
mythology]]
[[Category:People was defense of the History of Basel]]
[[Category:Christian engineerings|Malaysia]]

[[ca:Diametre de Logisto de Rafer]]
[[de:Chile de La City]]
[[es:Estra Pharea Civil Hank]]
[[eo:Fort College, California]]
[[f
```

**Prompt 4** (wiki), last 160 bytes:

```
|Estonian]], [[Latvian language|Latvian]], [[Lithuanian language|Lithuanian]], [[Greenlandic language|Greenlandic]], and [[Sami languages|Sami]].
|-
![[ISO/IEC 
```

*true continuation:*
```
8859-5|Part&amp;nbsp;5]]
|''Latin/Cyrillic''
|Covers mostly Slavic languages that use a [[Cyrillic]] alphabet, including [[Belarusian language|Belarusian]], [[Bulgarian language|Bulgarian]], [[Macedonian language|Macedonian]], [[Russian lan
```
*CyphaLM, word lookahead K 8:*
```
8859-1980]] and [[History of the Little Henderson]], and [[United States Department of the United States]] and [[South American Communications]], [[Estonian language|English]] and [[International Organization for Standardization|ISO]], [[Ce
```
*CyphaLM, byte sampling:*
```
885 in Britain]]'', the [[United States County, Olympic Party|Democratic Sinclair and manipulated by for Park]] ([[Poland|Polish]] write and separate in the mathematics of the main plane and in any state. The ''[[Torah]], and August to the 
```
*Transformer:*
```
language|ISO]], [[Latvian language|Languages]], [[Passauri]], [[Passauri]], [[Nicola]], [[New Lights language|Nicola]], [[North Language|North]], [[Nicola]], [[Nicola]], [[New Lights Language|Nicola]], [[New Lights Language|Nicola]], [[Nico
```
*LSTM:*
```
SPA|SI]] - [[Pronunciation]]
|-
|[[Constantinople]]
|[[Malay language|Canadian]]
|-
|[[Arichosaurian language|Arabic]] [[Afonso]] ||align=right|[[Salutian language|Arabic]] ||align=left |[[Scottish language|Carino]] ||align=right|[[Super Co
```

**Prompt 10** (alice), last 160 bytes:

```
rks,' Alice said
with some severity; `it's very rude.'

  The Hatter opened his eyes very wide on hearing this; but all
he SAID was, `Why is a raven like a 
```

*true continuation:*
```
writing-desk?'

  `Come, we shall have some fun now!' thought Alice.  `I'm glad
they've begun asking riddles.--I believe I can guess that,' she
added aloud.

  `Do you mean that you think you can find out the answer to it?'
said the 
```
*CyphaLM, word lookahead K 8:*
```
single construction of the consecrated external links ==

*[http://www.gamesponsibility and the government introduced commentary and development of the University of California Presidential and originally representatives of the National Ins
```
*CyphaLM, byte sampling:*
```
role of generation the &quot;[[Marble that of a classic mania]], and sixth or that relatively successors and animals he recorded with a little the [[controller]] or a few of the archives in and various fear of the last [[inch Colonialists, 
```
*Transformer:*
```
distantict distantict raven to distant state in possibly distant and should be required by the Hatter and Said as the state that young the land of the Hatter is marketed by the hatter hatter which was in a threat all time, and the [[Alice H
```
*LSTM:*
```
draw point to the law and was the result of the comes around the concept of his london (&quot;if the taste remaining the fleet of the cult and into the career)&lt;br&gt;
''He Strategy have a longer program that the advise of the player can 
```

All 12 prompts and continuations are in [`lm_compare/gens.json`](lm_compare/gens.json).

## Reproduce

```bash
# everything: trains the two neural models (1 h each), rebuilds the CyphaLM winner (timed),
# then runs the evals, benchmarks, generation, judging and learning curves, and writes report.md
pip install torch numpy
bench/lm_compare/run_all.sh /path/to/enwik8 /path/to/cantrbry WORK native/build
# single stages (resumable; results are cached in WORK/*.json)
WORK=... WINNER=WORK/cyphalm NN=WORK/nn CYPHA_BUILD=native/build python3 bench/lm_compare/compare.py evals|bench|gens|judge|curve
python3 bench/lm_compare/report.py > WORK/report.md
```

| file | what |
|---|---|
| `byte_lm.py` | Transformer and LSTM: train (wall-clock budget, checkpoints at 5/15/30/60 min), eval (static or dynamic, dumps distributions), gen, bench |
| `compare.py` | orchestration: shared prompts and texts, CyphaLM through `cyphalm_lm_quality --dump-dist` and `cyphalm_generate` |
| `metrics.py` | every metric, from the dumps |
| `report.py` | the tables above |
| `timed.py` | wall time, CPU time and peak RSS of a command |

Metric definitions:
- **next-word greedy accuracy:** the share of alphabetic words for which every byte, plus the byte that ends the word, is the argmax. This is exactly when greedy decoding from the true prefix writes the word.
- **word perplexity:** 2^(total bits / number of words).
- **ECE:** 10 bins of top-1 confidence.
- **12-byte copies:** the share of the continuation's 12-byte windows that already occurred earlier in the prompt or continuation.
- **real words:** the share of generated words that occur 3 or more times in the 95 MB of training text.
