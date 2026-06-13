# Demo CyphaLM checkpoint

This directory ships a small Moby-Dick snippet checkpoint (`demo.json` + `demo.npz`) for native CyphaLM REST demos.

Load with **native `cypha_rest`** (from a `native/` build):

```bash
export CYPHALM_CHECKPOINT=examples/demo_cyphalm/demo
./native/build/cypha_rest --listen 127.0.0.1:8099 \
  --cypha fixtures/reference.cypha \
  --cyphalm-checkpoint "$CYPHALM_CHECKPOINT"
bash examples/curl_lm_generate_stream.sh
```

Or pass `--cyphalm-checkpoint examples/demo_cyphalm/demo` directly (same as `CYPHALM_CHECKPOINT`).

The Qt shell can load the same checkpoint via **File → Load CyphaLM…** when built with LM support.
