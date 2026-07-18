# Quality recipe Wave 1 (2026-07-18)

**Ledger:** [`ROADMAP_EXECUTION_LEDGER_2026-07-18.md`](ROADMAP_EXECUTION_LEDGER_2026-07-18.md)  
**Scope:** Opt-in CyphaLM training recipe (§1.1 BPTT, §1.2 Adam+clip, §1.5 classic init). **Defaults unchanged** — D17 pin **2.873** path is BPTT-1 / SGD / N(0,0.02).

## Flags

| Knob | CLI | Env | Default |
|------|-----|-----|---------|
| Truncated BPTT | `--bptt-lstm N` | `CYPHA_LSTM_BPTT` | `1` |
| Optimizer | `--optim adam\|sgd` | `CYPHA_LSTM_OPTIM` | `sgd` |
| Grad clip (L2) | `--grad-clip C` | `CYPHA_LSTM_GRAD_CLIP` | `0` (off) |
| Init | `--lstm-init classic\|default` | `CYPHA_LSTM_INIT` | `default` |

JSON profile keys: `lstm_bptt_steps`, `lstm_optim`, `lstm_grad_clip`, `lstm_init`.

## Smoke

CTest `native_quality_recipe_wave1_smoke` trains CharLSTM with `bptt=8`, Adam, clip=1.0, classic init on synthetic ids and asserts finite BPC in (0, 20].

## FAST before/after (not a new pin)

Wave 1 does **not** re-pin D17. A local FAST hybrid compare (n_train≤2k) is optional research follow-on; promote to 20k / 300k only after evidence. This report records the substrate landing only.

## Explicit non-goals

Full WikiText multi-epoch, LayerNorm, 25-cell re-sweep, MiniRocket — deferred per ledger.
