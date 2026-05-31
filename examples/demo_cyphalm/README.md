# Demo CyphaLM checkpoint

Generate the checkpoint (not committed — run after clone):

```bash
python scripts/generate_demo_lm_checkpoint.py
```

This writes `demo.json` + `demo.npz` here (~Moby-Dick snippet, 2000 train steps).

Load for REST:

```bash
export CYPHA_LM_CHECKPOINT=examples/demo_cyphalm/demo
uvicorn cypha_studio.server.api:app --port 7749
bash examples/curl_lm_generate_stream.sh
```

Or in Studio: **File → Load CyphaLM…** and select `demo.json`.
