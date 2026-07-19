# Studio Web UI — CyphaLM chat pane (2026-07-17)

**Scope:** Bill of Work / FUTURE.md §4 — add a focused CyphaLM chat UI to the existing Studio SPA. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight scripts.

## SPA location

| Path | Role |
|------|------|
| `native/tools/static/index.html` | Cypha Studio shell (tabs, dark theme) |
| `native/tools/static/app.js` | REST client + new chat logic |
| `native/tools/cypha_rest_static_ui.cpp` | Mounts `/` and `/ui/*` from static dir (or embedded via `CYPHA_EMBED_STATIC_UI`) |

No separate `web/` or `spa/` tree — the minimal SPA lives under `native/tools/static/`.

## Shipped

### CyphaLM tab — chat pane (primary)

- **Composer:** multi-line input, Enter to send (Shift+Enter for newline).
- **Message log:** user bubbles (accent border) and assistant bubbles (panel) matching existing dark palette.
- **Controls:** `strategy`, `max_tokens`, **stream (SSE)** toggle, `epistemic_halt`.
- **Endpoints:**
  - Streaming (default): `POST /generate/stream` → parse `data: {…}` SSE chunks per token; 404 → `POST /lm/generate/stream`.
  - Non-stream: `POST /generate` with 404 fallback to `POST /lm/generate`.
- **Token bridge:** client-side char encode/decode (same algorithm as Branch A `encode_prompt_chars` / `decode_generated_ids`, vocab 128) so users type plain text while the REST body stays `{ "prompt_ids": [...] }`.
- **Context:** conversation text accumulates across turns for encoding context; assistant decode appended after each reply.

### Advanced sub-panel

Previous raw `prompt_ids` JSON tester moved under `<details>` **Advanced — raw token API** (unchanged behaviour).

## API contract (unchanged server)

```json
POST /generate
POST /generate/stream
{
  "prompt_ids": [1, 2, 3],
  "max_tokens": 32,
  "temperature": 0.9,
  "strategy": "top_p",
  "top_k": 40,
  "top_p": 0.92,
  "epistemic_halt": false,
  "stream": true
}
```

SSE chunk shape (per token): `{ "index", "token_id", "loss", "epistemic_var", "aleatoric_var", "active_experts" }`; terminal `{ "done": true }` or `{ "index", "done": true, "halted_on_uncertainty" }`.

## Run

```powershell
# Build cypha_rest, copy static/ beside exe (or set CYPHA_REST_STATIC_DIR)
cypha_rest --listen 127.0.0.1:8099 --cyphalm-checkpoint examples/demo_cyphalm/checkpoint.cyphalm ...
# Open http://127.0.0.1:8099/ → CyphaLM tab → chat
```

## Tests

| CTest | Asserts |
|-------|---------|
| `native_rest_ui_smoke` | `GET /` HTML + `GET /ui/app.js` (unchanged; still serves SPA) |

Manual: load LM checkpoint, send chat message, confirm SSE token stream or batch JSON fallback; toggle **Advanced** for raw `prompt_ids`.

## Follow-up (not in this slice)

- Server-side `encode_text` / `decode_tokens` REST helpers (would remove client char heuristic).
- Branch A `/route/generate` integration in chat for OOD → Ollama routing.
- Persist chat sessions / export transcript.
