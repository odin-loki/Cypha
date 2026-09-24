# CyphaLM vs neural byte LMs

This benchmark trains a byte-level Transformer and an LSTM (PyTorch, CPU) under the same wall-clock budget as CyphaLM, on the same data. It then scores all three from their full next-byte distributions with one shared metrics module.

The results and the method are in [`docs/reports/CYPHALM_VS_NEURAL_LM.md`](../../docs/reports/CYPHALM_VS_NEURAL_LM.md).

```bash
pip install torch numpy
bench/lm_compare/run_all.sh /path/to/enwik8 /path/to/cantrbry WORK native/build
```
