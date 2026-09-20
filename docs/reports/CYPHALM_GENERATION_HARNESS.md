# CyphaLM generation harness (qualitative)

- harness: `cyphalm_generation_harness`
- hp_table_bits: 16
- max_tokens: 32
- runs: 15

Qualitative sampling only. No BLEU/perplexity/quality scores. Completions are from untrained gate24 hp (cold start) via generate_decode serve path.

## Samples

### enwik_opening — greedy (seed=42)
- source: enwik8 XML header
- prompt (269 B): `<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xs`
- decode_ms: 2664.0
- completion (32 B):
```
\n                               
```

### enwik_opening — temperature (seed=42)
- source: enwik8 XML header
- prompt (269 B): `<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xs`
- decode_ms: 9578.9
- completion (32 B):
```
\u001cat-n\"elSc\u000cUhtttti..oca-0-1/s/Y3
```

### enwik_opening — temperature (seed=137)
- source: enwik8 XML header
- prompt (269 B): `<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xs`
- decode_ms: 9421.1
- completion (32 B):
```
\n00.3\"export-0.3\"exma-\u001dang htiki
```

### alice_opening — greedy (seed=42)
- source: gutenberg/alice
- prompt (219 B): `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice s`
- decode_ms: 2433.9
- completion (32 B):
```
ing ing ing ing ing ing ing ing 
```

### alice_opening — temperature (seed=42)
- source: gutenberg/alice
- prompt (219 B): `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice s`
- decode_ms: 9958.6
- completion (32 B):
```
sisat, ilh liorxre bt lad lhh pe
```

### alice_opening — temperature (seed=137)
- source: gutenberg/alice
- prompt (219 B): `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice s`
- decode_ms: 9782.5
- completion (32 B):
```
but inpzinttttet pice anin or ip
```

### code_c — greedy (seed=42)
- source: synthetic C
- prompt (87 B): `#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world\\n\");\n    return 0;\n}\n\n// `
- decode_ms: 2597.1
- completion (32 B):
```
                                
```

### code_c — temperature (seed=42)
- source: synthetic C
- prompt (87 B): `#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world\\n\");\n    return 0;\n}\n\n// `
- decode_ms: 10587.7
- completion (32 B):
```
f^n v fM),\u00051_`p|pi\n\ni\n \n\n\n\n\n \n \n
```

### code_c — temperature (seed=137)
- source: synthetic C
- prompt (87 B): `#include <stdio.h>\n\nint main(void) {\n    printf(\"Hello, world\\n\");\n    return 0;\n}\n\n// `
- decode_ms: 9953.1
- completion (32 B):
```
\npp,  ;|<>twuv.t nc  \u000b\u0004:<<\u0003>l..e
```

### ascii_prose — greedy (seed=42)
- source: synthetic prose
- prompt (122 B): `The quick brown fox jumps over the lazy dog. In 2026, byte-level language models predict the next character from context`
- decode_ms: 2562.7
- completion (32 B):
```
dededededededededededededededede
```

### ascii_prose — temperature (seed=42)
- source: synthetic prose
- prompt (122 B): `The quick brown fox jumps over the lazy dog. In 2026, byte-level language models predict the next character from context`
- decode_ms: 11399.5
- completion (32 B):
```
oooCuHichc jthkumm %n b(  74j j3
```

### ascii_prose — temperature (seed=137)
- source: synthetic prose
- prompt (122 B): `The quick brown fox jumps over the lazy dog. In 2026, byte-level language models predict the next character from context`
- decode_ms: 11950.1
- completion (32 B):
```
 rn0002zprrttt8t/on)   2an 6K*+b
```

### xml_tag — greedy (seed=42)
- source: synthetic XML
- prompt (58 B): `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item id=\"1\">`
- decode_ms: 2676.8
- completion (32 B):
```
<?><?><?><?><?><?><?><?><?><?><?
```

### xml_tag — temperature (seed=42)
- source: synthetic XML
- prompt (58 B): `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item id=\"1\">`
- decode_ms: 10818.1
- completion (32 B):
```
>>> v!?=\"1\n<kUkvki  i g         
```

### xml_tag — temperature (seed=137)
- source: synthetic XML
- prompt (58 B): `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item id=\"1\">`
- decode_ms: 10406.1
- completion (32 B):
```
\n_eR;1P\u00ecTU\u00ba\u00b3\u00bf3\u00b5\"ih\"  \u0008U_`\u0006hn1._
```

## Qualitative notes (human-readable, not scores)

- **Cold start:** gate24 hp with no corpus warmup; expect repetitive or markup-like continuations.
- **Greedy vs temperature:** greedy should be deterministic per prompt; temperature runs differ by seed.
- **Structure:** XML/C/code prompts test whether continuations stay in-token class (tags, braces, semicolons).
- **No automated quality metric** — inspect completions above for coherence, repetition, and charset drift.
