# Branch A router: save and load round trip (RSAVE)

**Odin Loch** · 2026-10-29

## What was asked

CSAVE ([report](CYPHA_SAVE_LOAD_FIX.md)) fixed `Cypha::save` / `Cypha::load`. Its "Not fixed" list said
`BranchARouter::save_checkpoint` has the same kind of bug: it writes only the world and the classes, so the encoder,
temperature, `mahal_ema`, label context and field keep template values after a reload. RSAVE was asked to check that,
write a failing test first, and fix what the test finds.

## What the router does

- The router never learns. It has no `update`. It loads a checkpoint and routes.
- A checkpoint is `branch_a_router.json` plus a model file. The JSON holds the standardisation (`mean`, `std`), the
  hashing settings, an optional `projection`, the abstain threshold, two counters, and optionally `f_field_json`: the
  world field in a separate file. `docs/port/PORT_CONTRACT.md` names that layout.
- `load_checkpoint` reads the model file into a `CyphaInferModel` (`from_root`) and a `CyphaDifMemoryState`, and keeps
  the loaded root.
- `route` hashes the text, projects it, standardises it, and passes it to `infer_at_h` as the latent. It uses the
  world, the classes, the field and the temperature. It does not use the encoder.
- `save_checkpoint` clones the loaded root, rewrites the world and the classes from the memory state (the same merge
  `Cypha::save` uses), and writes a new JSON.

## What was measured before the fix

The new CTest builds five checkpoint layouts from a trained Cypha model (saved with `Cypha::save`, format 2):

| Layout | Before the fix |
|---|---|
| Field in the model file, with a projection | Round trip exact |
| No projection (hash truncation) | Round trip exact |
| Model file in save format 1 (no save-state block) | Round trip exact |
| Field in a separate `f_field_json` | **Does not load** |
| `f_field_json` and format 1 | **Does not load** |

- **CSAVE's note was wrong for the router.** Because save clones the loaded root, the encoder, temperature,
  `mahal_ema`, label context and field go back out exactly as they came in. With the field in the model file, a saved
  and reloaded router routes every probe bit for bit the same, and the second save is byte-identical.
- **Bug 1, load.** `infer_field_dim` took the field size from `world.F_field`, else a `field_dim` key, else 24. With the
  field in `f_field_json`, the model file has no `world.F_field`, so the router guessed 24 and `from_root` threw
  ("field_h length does not match"). Such checkpoints loaded only when the field size happened to be 24.
- **Bug 2, save.** Once bug 1 is fixed, such a checkpoint loads, but save drops its field. The memory merge rewrites
  only keys the root already has, and the saved JSON has no `f_field_json`. The saved checkpoint then fails to load
  ("world.F_field missing").

## The fix

- **Load:** with no embedded field, the size comes from `field_h`, as in `Cypha::load`. Then a `field_dim` key, then
  24, as before.
- **Save:** when the saved world has no `F_field`, the router embeds the field it holds (row-major, latent × field
  size, as the memory merge writes it). The saved model file is then complete, like a `Cypha::save` file, and the
  saved JSON needs no `f_field_json`.
- **Read-only views:** `infer_model()` and `memory_state()`, for the test. The rest of the public API is unchanged.

**Why not delegate to `Cypha::save` / `Cypha::load`.** The router does not hold a `Cypha`. It holds the inference model
and the memory state built from a root, and routes through `infer_at_h` on the standardised embedding. A `Cypha`
predicts through its encoder, label context, LLR scale and GH gate. Switching would change what the router answers,
which is outside a save fix. There is still one serialisation path: save writes the loaded root back through the same
memory merge that `Cypha::save` uses.

## Old checkpoints

- Every checkpoint that loaded before loads the same way. It routes the same and saves the same bytes as before,
  except that a saved file now always has `world.F_field`.
- A checkpoint with a separate `f_field_json` and no `field_dim` key now loads at any field size. Before, it threw
  unless the size was 24.
- A file saved by the new code should load in the old code too: its field is embedded, and the old loader reads
  `world.F_field` first (by reading its code; not run).

## Test

`native_branch_a_router_save_load_test` (`native/tools/branch_a_router_save_load_test.cpp`), five layouts. For each:

1. Router A loads the source checkpoint. The probes (60 texts, including an empty one) must reach at least two
   labels, else a lost class would pass unseen. The test takes the first spread of `std`, from a fixed list, at which
   they do.
2. A saves; router B loads A's save. B must route every probe at four thresholds exactly as A does: label, confidence
   and epistemic value bit for bit, abstain and action equal.
3. B's inference model and memory state must equal A's, field by field (58 fields: 40 of the inference model, 18 of the memory state), and their summaries must agree.
4. The saved JSON must keep every key the loader reads: mean, std (a zero is saved as the 1 the loader set),
   threshold, backend, hashing sizes and seed, sample count, train seconds, projection.
5. B saves again: both files must be byte-identical to A's save, and router C loaded from it must route like A.

The router has no learning step, so there is no "further learning" check.

**Planted bugs** (one field dropped at a time, the test rebuilt and run each time): 17 planted, 16 fail the test.

| Planted bug | Caught by |
|---|---|
| Save drops `mean` or `std` | The reload throws |
| Save drops the threshold, backend, `hash_n_features`, `hash_n_components` or `hash_seed` | Saved JSON check |
| Save drops `n_train_samples` or `train_seconds` | Summary check |
| Save drops the projection | Routes differ |
| Save skips embedding the field | The reload throws (2 layouts) |
| Load infers the field size the old way | The source does not load (2 layouts) |
| Load ignores `f_field_json` | The source does not load (2 layouts) |
| Load ignores the projection | The probes reach one label |
| Load ignores the threshold | Saved JSON check |
| Load keeps a zero `std` | Saved JSON check |
| Save skips the memory merge | **Not caught** |

The one not caught loses nothing: the router never changes its memory state, so on a loaded router the merge writes
back the same world and classes it read.

## Checks

- The new CTest passes on all five layouts.
- `native_cypha_save_load_test` and `native_cypha_ood_test` pass.
- CI set (`ctest -R native_ -LE cypha_slow`) on a full Release build of the tree merged with main `c756ee3`: 145 of 145
  pass, the new test among them.

## Callers checked

- `native/apps/branch_a_rest_routes.cpp`: `try_load_checkpoint` at start-up, `route` in `/route/text` and
  `/route/generate`, `save_checkpoint` in `/route/save`, `summary`.
- `native/apps/cypha_rest.cpp`: passes `--branch-a-json` or `CYPHA_BRANCH_A_CHECKPOINT` to the routes above.
- `native/tools/topic_bench.cpp`: copies the hashing featuriser only; it does not save or load a router.
- No router checkpoint is committed in the repository (none under `fixtures/`, `bench/` or `data/`), so nothing needs
  regenerating.

## Open (not changed here)

- **The router ignores format 2's save-state block when it routes.** A model saved by `Cypha::save` carries its label
  context, OOD threshold ring, LLR scale and GH session in `cypha_save_state`. The router keeps that block in its saves
  (the clone copies it), but `infer_at_h` does not use it, so the router answers differently from `Cypha::predict` on
  the same model. Whether Branch A should route like `Cypha::predict` is a separate question.
- The saved JSON keeps only the keys the router reads. Any other key in a source JSON is not written back.
