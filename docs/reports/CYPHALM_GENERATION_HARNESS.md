# CyphaLM generation harness (cold vs primed, qualitative)

- harness: `cyphalm_generation_harness`
- hp_table_bits: 16
- max_tokens: 16
- warmup_file: `bench/data/canterbury/alice29.txt`
- warmup_bytes: 4096
- primed defaults: {'ban_last_k': 3, 'repetition_penalty': 1.15, 'repetition_window': 24, 'text_like_prior': 0.35}
- runs: 20

Qualitative before/after only. No BLEU/perplexity/quality scores. Cold = no warmup and no serve-time penalties; primed = alice warmup + ban_last_k + repetition_penalty + text_like_prior (serve-time only).

## Samples (before / after per prompt)

### enwik_opening
- source: enwik8 XML header
- prompt (269 B): `<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xs`

#### cold_greedy — greedy (decode_ms=179.5)
```
\n               
```

#### primed_greedy — greedy (decode_ms=2861.3)
```
\n (and she tried
```

#### cold_top_p — top_p (decode_ms=2721.9)
```
\n   <title>\n    
```

#### primed_top_p — top_p (decode_ms=2823.8)
```
've down ther id
```


### alice_opening
- source: gutenberg/alice
- prompt (219 B): `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice s`

#### cold_greedy — greedy (decode_ms=148.8)
```
inning nothing n
```

#### primed_greedy — greedy (decode_ms=2725.7)
```
`and what is, bu
```

#### cold_top_p — top_p (decode_ms=2594.3)
```
rsat into titicd
```

#### primed_top_p — top_p (decode_ms=2709.5)
```
`and what is, ha
```


### code_c
- source: synthetic C
- prompt (87 B): `#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world\\n\");\n    return 0;\n}\n\n// `

#### cold_greedy — greedy (decode_ms=108.1)
```
                
```

#### primed_greedy — greedy (decode_ms=2811.5)
```
(and she tried t
```

#### cold_top_p — top_p (decode_ms=2654.8)
```
tf}\n, t         
```

#### primed_top_p — top_p (decode_ms=2760.8)
```
des before askem
```


### ascii_prose
- source: synthetic prose
- prompt (122 B): `The quick brown fox jumps over the lazy dog. In 2026, byte-level language models predict the next character from context`

#### cold_greedy — greedy (decode_ms=123.6)
```
dededededededede
```

#### primed_greedy — greedy (decode_ms=2715.8)
```
(as the she she 
```

#### cold_top_p — top_p (decode_ms=2623.1)
```
exexythe nexelrl
```

#### primed_top_p — top_p (decode_ms=2756.4)
```
it'listenichends
```


### xml_tag
- source: synthetic XML
- prompt (58 B): `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item id=\"1\">`

#### cold_greedy — greedy (decode_ms=114.9)
```
<?><?><?><?><?><
```

#### primed_greedy — greedy (decode_ms=2750.5)
```
\n (and the the t
```

#### cold_top_p — top_p (decode_ms=2621.6)
```
\u000b:\u000e<w\"=\"1\">\n:\u000fx2
```

#### primed_top_p — top_p (decode_ms=2774.0)
```
8(no\nsay it, nor
```


## Latency notes

- Compare `decode_ms` for `cold_*` vs `primed_*` per prompt/strategy.
- Warmup cost is included in primed `decode_ms` (one-shot serve_advance over warmup corpus).
- No automated quality metric — inspect completions for repetition, charset, and structure.
