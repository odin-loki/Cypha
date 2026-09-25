# CyphaLM vs neural byte LMs

This benchmark trains a byte-level Transformer and an LSTM (PyTorch, CPU) under the same wall-clock budget as CyphaLM, on the same data. It then scores all three from their full next-byte distributions with one shared metrics module.

The results and the method are in [`docs/reports/CYPHALM_VS_NEURAL_LM.md`](../../docs/reports/CYPHALM_VS_NEURAL_LM.md).

```bash
pip install torch numpy
bench/lm_compare/run_all.sh /path/to/enwik8 /path/to/cantrbry WORK native/build
```

## Mixing simulator (`mixsim.py`)

`mixsim.py` replays CyphaLM's mixing stages offline. You dump one harness run once, then screen mixing settings in seconds instead of re-running the models (about 0.1 ms per byte in numpy).

The replay can be exact because of how the stages depend on each other:

- Members read the true bytes on their own.
- ∞-gram counts depend only on the context.
- The neural experts' output-layer adaptation depends only on the truth.

So everything after the models' own distributions is a function of the dump.

```bash
B=native/build/cyphalm_lm_quality
$B --load v3_winner.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 \
   --prompt-bytes 0 --gen-bytes 0 --dump-components /tmp/d/wiki        # --dump-dtype f16|f32|f64 (f32)
python3 bench/lm_compare/mixsim.py check /tmp/d/wiki                   # reproduces eval.nll_bits_per_byte
python3 bench/lm_compare/mixsim.py run /tmp/d/wiki --set final_temperature=0.9 --save-nll t09.npy
python3 bench/lm_compare/mixsim.py compare /tmp/d/wiki /tmp/d/alice /tmp/d/lcet10 \
   --a neural_mix=switch ensemble_gate=1 --b                            # paired block bootstrap, 1 KiB blocks
python3 bench/lm_compare/mixsim.py check /tmp/d/wiki --set neural_mix=log --against /tmp/d/wiki_log
python3 bench/lm_compare/mixsim.py compare-dumps /tmp/d/wiki /tmp/d/wiki_other   # two harness runs, same bytes
```

**What a dump holds.** A dump directory has one row per scored byte. The layout is in `native/include/cypha/cyphalm/component_dump.hpp`:

- `truth.bin`: the true byte.
- `models.bin`: the log P of the primary, then each member, before the ensemble mix.
- `neural.bin`: each expert's log P from before it read the byte, including its adaptation.
- `ig_n.bin`, `ig_total.bin`, `ig_count.bin`: the ∞-gram longest match and reliable part, each as n, total and 256 counts.
- `served.bin`: the served log P.
- `meta.json`: every stage's mode and learning rate, whether the mixing weights learned, every mixing weight at the first byte, and the harness's `eval` block.

**Size.** A row takes (models + experts + 1) × 256 values plus 2 KiB of counts. For the v3 winner (4 models, 2 experts) that is about 9.3 KB per byte at f32, so a 16 KiB slice takes about 150 MB.

**What is replayed.** `mixsim.py` re-implements these parts of `hp_backend.cpp`, operation by operation:

- `mix_with_members_`: the geometric mix, or the context gate.
- `update_ensemble_weights_` and `update_gate_`.
- `infinigram_mix_`: `halving` and `longest16` buckets.
- `neural_mix_`: `linear`, `log` and `switch`.
- `final_sharpen_`: fixed or learned temperature.
- The per-bucket updates of each stage.

**Accuracy.** On a 2 KiB v3 run, `check` agrees with the harness to 1.2e-10 bits/byte from an f32 dump and 6e-15 from an f64 dump. A fixed `final_temperature=0.9` equals the harness's `nll_bits_by_temperature` "0.9" entry to 4e-15.

**Settings.** `--set`, `--a` and `--b` take the settings below. Anything left unset stays as dumped.

- `models=0,2,3`: the ensemble models to keep, where 0 is the primary. A subset starts at equal weights, or at `ensemble_start=w,...` if you give it.
- `neural=0,1`: the experts to keep.
- `infinigram=0|1`: drop or keep the ∞-gram stage.
- `ensemble_gate=0|1`
- `neural_mix=linear|log|switch`
- `learn_mix=0|1`
- `ensemble_lr`, `infinigram_lr`, `neural_lr`, `final_temperature`, `final_temperature_lr`

A stage whose mode, members or experts change restarts at the backend's start weights, as the C++ setters do. Unchanged stages start from the dumped state. CTest checks this: a dump replayed with another run's settings predicts that run's NLL (`check --set ... --against`).

**Limits.**

- `infinigram_mode` cannot change within a dump, because the reliable part was queried in the dumped mode. Dump again with `--infinigram-mode`.
- The session cache is not replayed, so the harness refuses to dump a model that has one.
- A changed `neural_adapt` needs a new dump.

**Bootstrap.** `block_bootstrap(a, b, block=1024)` resamples contiguous 1 KiB blocks of per-byte differences, which keeps the dependence between neighbouring bytes. It reports the byte-weighted mean difference, a 95 % percentile interval and a two-sided p. With several dumps, `compare` pools their blocks, and no block crosses from one text into another. A 16 KiB slice is only 16 blocks, so confirm small effects on several slices per domain.
