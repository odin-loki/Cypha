# gen_facts: statements of fact in generated text, graded against CyphaLM's store

**Odin Loch** · 2026-11-12 · [GEN_FACTS](../../docs/reports/GEN_FACTS.md) (CT3.4)

Written by `scripts/gen_fact_check.py`. From the repository root:

```
python3 scripts/gen_fact_check.py run --out bench/gen_facts \
  D0=bench/gen_quality/runs/D0.gen.txt W1b=bench/gen_quality/runs/W1b.gen.txt \
  H1b=bench/gen_quality/runs/H1b.gen.txt G16=bench/tune_write/runs/G16.gen.txt \
  L1=bench/k5_coherence/runs/L1.gen.txt HUMAN=bench/eval/gen_prompts_dev.tsv
```

| File | What |
|---|---|
| `ARM.facts.tsv` | one row per statement: sample, sentence, relation, subject, store entity, value, verdict, the store's values, the pattern, the sentence |
| `summary.md`, `summary.json` | per arm: statements per sample, share contradicted, share supported, precision, by sentence position, with 95% intervals |
| `test/` | the same for the test reads already made |
| `full_store/` | the same with `--drop none` (beside the measure) |
| `hand_check.tsv` | the 40 hand-checked statements |
