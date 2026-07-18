# Demo Cypha sequence checkpoint

This directory ships a small Moby-Dick snippet checkpoint (`demo.json` + `demo.npz`) for native Cypha REST demos.

Load with **native `cypha_rest`** (from a `native/` build):

```bash
export CYPHA_SEQUENCE_CHECKPOINT=examples/demo_cyphalm/demo
./native/build/cypha_rest --listen 127.0.0.1:8099 \
  --cypha fixtures/reference.cypha \
  --sequence-checkpoint "$CYPHA_SEQUENCE_CHECKPOINT"
bash examples/curl_lm_generate_stream.sh
```

Or pass `--sequence-checkpoint examples/demo_cyphalm/demo` directly (same as `CYPHA_SEQUENCE_CHECKPOINT`; aliases `--cyphalm-checkpoint` / `CYPHALM_CHECKPOINT`).

The Qt shell can load the same checkpoint from the **Cypha** tab (**Browse…** + **Load checkpoint**) when built with sequence support (the **File** menu only has Settings).
