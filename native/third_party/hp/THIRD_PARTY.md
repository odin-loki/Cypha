# hp — vendored from odin-loki/CompressionAlgorithm

Source: https://github.com/odin-loki/CompressionAlgorithm (branch `master`, `hp/` tree)

**Local changes.** Reduced to the gate24 recipe: every ablation flag resolved to its
gate24 value and the rejected branches removed (bit-identical; see
`docs/reports/CYPHALM_HP_GATE24_STRIP.md`). Added on top: runtime lossy knobs on
`hp::Config`, demand-zero huge-page tables (as upstream), and undo-log fixes so
speculative bit-tree scoring leaves the live predictor untouched (DMC node splits
and word-match resets were not recorded). See
`docs/reports/CYPHALM_LOSSY_MIXER_REPORT.md`.

Cypha uses this integer-exact Hutter Prize context-mixing compressor as its LLM /
sequence algorithm. Encoder and decoder share the same model code path inside
`hp::Predictor`.

## License

See upstream repository license. xsimd is under BSD-3-Clause (`third_party/xsimd/LICENSE`).

## Integration

- Headers: `native/third_party/hp/include/hp/`
- Adapter: `native/include/cypha/cyphalm/hp_backend.hpp`
- Cypha facade: `CyphaLMModel` delegates to `HpSequenceBackend`

Do not mix Cypha BPC (next-byte log_probs over a token stream) with raw hp archive
sizes without explicit labeling — see `MODEL_CARD.md`.
