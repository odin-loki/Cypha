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
