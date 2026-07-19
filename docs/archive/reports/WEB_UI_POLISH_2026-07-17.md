# Studio Web UI — chat empty state polish (2026-07-17)

**Scope:** Bill of Work §5 Web UI (Partial). One bounded polish slice for Studio Web CyphaLM chat. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight scripts.

## Path found

| Path | Role |
|------|------|
| `native/tools/static/index.html` | Cypha Studio SPA shell (CyphaLM tab) |
| `native/tools/static/app.js` | Chat logic (SSE stream, char token bridge) |
| `native/tools/cypha_rest_static_ui.cpp` | Serves `/` and `/ui/*` |

Prior chat pane shipped in `b706647` — see [`WEB_UI_GENERATE_2026-07-17.md`](WEB_UI_GENERATE_2026-07-17.md). No separate `web/` or `studio/` tree.

## Shipped (this slice)

**Clearer empty state + LM readiness label** (picked over error banner / model selector — highest value for first-open UX, smallest diff).

- **Empty placeholder** inside `#chat-log`: centered “No messages yet” + hint to load CyphaLM; hides when first message is appended; restored on Clear.
- **Readiness status** (`#chat-status`): on load and when opening the CyphaLM tab, `GET /metrics` → green “CyphaLM loaded — ready to chat.” or red warning when `lm_loaded` is false.
- Existing dark palette and bubble layout unchanged.

## Not in this slice

- Dismissible error banner (errors still appear in assistant bubble + status line).
- Multi-model selector in chat (predict/update already support `"model"`; LM chat is single-checkpoint).
- Streaming cursor / token animation tweaks.

## Manual check

```powershell
# cypha_rest with static UI + optional LM checkpoint
# Open http://127.0.0.1:8099/ → CyphaLM tab
# Expect empty-state placeholder; status green if lm_loaded else red hint
# Send message → placeholder hides; Clear → placeholder returns
```

## Tests

| CTest | Asserts |
|-------|---------|
| `native_rest_ui_smoke` | `GET /` HTML + `GET /ui/app.js` still serve SPA (unchanged) |
