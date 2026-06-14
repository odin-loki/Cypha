# WikiText-2 (D17 / D21 corpus)

Official WikiText-2 raw token files used by CyphaLM bench profiles **d17** and **d21**.

## Layout

```
bench/data/wikitext2/
  wikitext-2/
    wiki.train.tokens
    wiki.valid.tokens
    wiki.test.tokens
```

These files are **not** committed to git. Download them with one of the scripts below.

## Download

From the repo root:

**Windows (PowerShell 5+):**

```powershell
powershell -File scripts/download_wikitext2.ps1
```

**Linux / macOS / CI:**

```bash
bash scripts/download_wikitext2.sh
```

Source: [Salesforce WikiText-2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)
(canonical URL; scripts fall back to a community mirror if the hosted bucket is unavailable).

## Fallback

When WikiText-2 is absent, `load_bench_corpus("d17"|"d21", ...)` falls back to
`bench/data/gutenberg/*.txt` (Moby Dick preferred) with source tag `gutenberg_fallback`.
Install WikiText-2 for official train/valid splits and production baseline runs.
