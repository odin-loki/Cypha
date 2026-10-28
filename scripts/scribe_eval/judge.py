"""LLM-as-judge baseline for the SCRIBE meta-evaluation (paper plan §6).

Author: Odin Loch

A small instruction model (default Qwen/Qwen2.5-1.5B-Instruct, float32, CPU)
rates each text's overall quality as written English (fluency, coherence, no
needless repetition, makes sense) on a 1-10 scale, G-Eval style (Liu et al.
2023): the score is the probability-weighted mean rating, not a sampled answer.
Nothing is generated; one forward pass reads the distribution of the first
answer token.

Rating tokens. Qwen2.5 writes every digit as its own token, so "10" is "1"
followed by "0". The first answer token gives P(d) for d = 1..9. P("1") is then
split into the ratings 1 and 10 by one more cached step that reads
P("0" | ..."1"):  P(10) = P("1") P("0"|"1"),  P(1) = P("1") (1 - P("0"|"1")).
The ten probabilities are renormalised; we report
  judge_score   sum_r r P(r) / sum_r P(r)      (1..10)
  judge_argmax  the most probable rating
  judge_mass    sum_r P(r) before renormalising (format compliance; near 1 is good)
  judge_p       the ten renormalised probabilities P(1..10)

The prompt shows the context (the "prompt" field) when there is one. The
assistant turn is prefilled with "Rating: " so that the next token is the number.
Text is cut to its first --max-text-tokens tokens and the context to its last
--max-context-tokens tokens.

Speed. The system prompt and the instructions are the same for every text, so
their keys and values are computed once and reused for every batch (texts are
left padded after the shared prefix, with matching position ids). This gives
the same numbers as running every full prompt (checked with --check-prefix),
at about half the cost for short texts.

Output: <results-root>/<name>/judge_<tag>.jsonl and .time.json, as baselines.py.
Use --estimate N to time N texts first and print texts/s and the projected time.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (DEFAULT_ROOT, finite, hf_revision, load_rows, run_family,  # noqa: E402
                       setup_torch, token_batches)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = "You are a strict and careful judge of English writing."
INSTRUCTION = (
    "Rate the overall quality of the text below as written English, on a scale from 1 to 10.\n\n"
    "Evaluation criteria:\n"
    "- Fluency: the sentences are grammatical and read naturally.\n"
    "- Coherence: the sentences connect and the text stays on its topic.\n"
    "- Repetition: no needless repetition of words, phrases or sentences.\n"
    "- Sense: the content makes sense.\n"
    "1 means unreadable or meaningless; 10 means excellent, natural and coherent writing.\n"
)
CONTEXT_BLOCK = (
    "\nThe text was written as a continuation of, or a response to, the context below. "
    "Rate only the text, but judge its coherence with the context.\n\n"
    "Context:\n\"\"\"\n{prompt}\n\"\"\"\n"
)
TEXT_BLOCK = "\nText:\n\"\"\"\n{text}\n\"\"\"\n\nReply with the rating only: one whole number from 1 to 10."
PREFILL = "Rating: "


def judge_ids(tok, text, prompt, max_text, max_ctx):
    """Token ids of the full judge prompt (chat template + prefill) and a truncation flag."""
    t_ids = tok(text, add_special_tokens=False).input_ids
    shown = tok.decode(t_ids[:max_text]) if len(t_ids) > max_text else text
    user = INSTRUCTION
    if (prompt or "").strip():
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ctx = tok.decode(p_ids[-max_ctx:]) if len(p_ids) > max_ctx else prompt
        user += CONTEXT_BLOCK.format(prompt=ctx.strip())
    user += TEXT_BLOCK.format(text=shown.strip())
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFILL
    return tok(s, add_special_tokens=False).input_ids, len(t_ids) > max_text


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def ratings_from(p_first, p_zero_after_one):
    """p_first: probabilities of the digit tokens "0".."9" as the first answer token."""
    p = np.zeros(11)
    p[2:10] = p_first[2:10]
    p[1] = p_first[1] * (1.0 - p_zero_after_one)
    p[10] = p_first[1] * p_zero_after_one
    mass = float(p.sum())
    if mass <= 0:
        return {"judge_score": None, "judge_argmax": None, "judge_mass": 0.0, "judge_p": None}
    q = p / mass
    return {"judge_score": finite(float((np.arange(11) * q).sum())),
            "judge_argmax": int(np.argmax(q)), "judge_mass": finite(mass),
            "judge_p": [round(float(x), 5) for x in q[1:]]}


class Judge:
    def __init__(self, model_id, threads, max_tokens, max_batch, max_text, max_ctx, use_prefix=True):
        torch = setup_torch(threads)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        self.digits = [self.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.max_tokens, self.max_batch = max_tokens, max_batch
        self.max_text, self.max_ctx = max_text, max_ctx
        self.prefixes = {}  # prefix ids tuple -> KV cache of the prefix (batch 1)
        if use_prefix:
            # Shared prefixes: the tokens before the text (no prompt) or before the context.
            for prompts in ((None, None), ("x", "y")):
                a = judge_ids(self.tok, "alpha", prompts[0], max_text, max_ctx)[0]
                b = judge_ids(self.tok, "omega", prompts[1], max_text, max_ctx)[0]
                pre = tuple(common_prefix(a, b))
                with torch.inference_mode():
                    out = self.model(input_ids=torch.tensor([pre]), use_cache=True)
                self.prefixes[pre] = out.past_key_values

    def build(self, rows):
        return [judge_ids(self.tok, r["text"], r.get("prompt"), self.max_text, self.max_ctx) for r in rows]

    def _run(self, suffixes, prefix, cache):
        """Rating probabilities for sequences prefix + suffix (left padded after the prefix)."""
        torch = self.torch
        B, P = len(suffixes), len(prefix)
        L = max(len(s) for s in suffixes)
        x = torch.full((B, L), self.pad, dtype=torch.long)
        m = torch.zeros((B, L), dtype=torch.long)
        for k, s in enumerate(suffixes):
            x[k, L - len(s):] = torch.tensor(s)
            m[k, L - len(s):] = 1
        pos = P + (m.cumsum(1) - 1).clamp(min=0)
        mask = torch.cat([torch.ones((B, P), dtype=torch.long), m], 1) if P else m
        past = None
        if P:
            past = copy.deepcopy(cache)
            for layer in past.layers:
                layer.batch_repeat_interleave(B)
        with torch.inference_mode():
            o = self.model(input_ids=x, attention_mask=mask, position_ids=pos, past_key_values=past,
                           use_cache=True, logits_to_keep=1)
            p1 = torch.softmax(o.logits[:, -1].float(), -1)[:, self.digits]
            # second step: feed "1" and read P("0"), which splits P("1") into ratings 1 and 10
            x2 = torch.full((B, 1), self.digits[1], dtype=torch.long)
            mask2 = torch.cat([mask, torch.ones((B, 1), dtype=torch.long)], 1)
            o2 = self.model(input_ids=x2, attention_mask=mask2, position_ids=pos[:, -1:] + 1,
                            past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
            p0 = torch.softmax(o2.logits[:, -1].float(), -1)[:, self.digits[0]]
        return [ratings_from(p1[k].numpy().astype(np.float64), float(p0[k])) for k in range(B)]

    def score_ids(self, seqs):
        # Assign each sequence to the longest cached prefix it starts with (or none).
        groups = {}
        for i, s in enumerate(seqs):
            best = ()
            for pre in self.prefixes:
                if len(pre) > len(best) and len(s) > len(pre) and tuple(s[:len(pre)]) == pre:
                    best = pre
            groups.setdefault(best, []).append(i)
        out = [None] * len(seqs)
        for pre, idx in groups.items():
            suf = [seqs[i][len(pre):] for i in idx]
            # token budget counts the prefix too (attention over it), batch by suffix length
            for b in token_batches([len(s) + len(pre) // 2 for s in suf], self.max_tokens, self.max_batch):
                for k, res in zip(b, self._run([suf[j] for j in b], list(pre), self.prefixes.get(pre))):
                    out[idx[k]] = res
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--name")
    ap.add_argument("--results-root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=2048)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-text-tokens", type=int, default=768)
    ap.add_argument("--max-context-tokens", type=int, default=256)
    ap.add_argument("--no-prefix-cache", action="store_true", help="run every full prompt (slower, same numbers)")
    ap.add_argument("--tag", help="output family suffix (default from the model id)")
    ap.add_argument("--estimate", type=int, default=0, help="time N texts, print texts/s, write nothing")
    ap.add_argument("--check-prefix", type=int, default=0, help="compare N texts with and without the prefix cache")
    args = ap.parse_args(argv)

    rows = load_rows(args)
    tag = args.tag or args.model.split("/")[-1].lower().replace(".", "").replace("-instruct", "")
    state = {}

    def setup():
        j = Judge(args.model, args.threads, args.max_batch_tokens, args.batch_size,
                  args.max_text_tokens, args.max_context_tokens, use_prefix=not args.no_prefix_cache)
        state["judge"] = j

        def score(batch):
            built = j.build(batch)
            res = j.score_ids([ids for ids, _ in built])
            for r, (ids, tr) in zip(res, built):
                r["judge_ntok"], r["judge_trunc"] = len(ids), tr
            return res

        def batches(rs):
            rs = sorted(rs, key=lambda r: -len(r["text"]))
            return [rs[i:i + 16] for i in range(0, len(rs), 16)]
        return score, batches

    if args.estimate or args.check_prefix:
        t0 = time.time()
        score, _ = setup()
        j = state["judge"]
        print(f"load {time.time() - t0:.1f}s; cached prefixes of {[len(p) for p in j.prefixes]} tokens")
        sample = rows[: max(args.estimate, args.check_prefix)]
        t0 = time.time()
        res = score(sample)
        dt = time.time() - t0
        built = j.build(sample)
        ntok = sum(len(ids) for ids, _ in built)
        pre = max((len(p) for p in j.prefixes), default=0)
        print(f"estimate: {len(sample)} texts ({ntok} prompt tokens, about {ntok - pre * len(sample)} beyond the "
              f"cached prefix) in {dt:.1f}s = {len(sample) / dt:.3f} texts/s")
        all_tok = sum(len(ids) - pre for ids, _ in j.build(rows))
        rate = (ntok - pre * len(sample)) / dt
        print(f"estimate: all {len(rows)} texts ~ {all_tok} uncached tokens ~ {all_tok / rate / 3600:.2f} h "
              f"(~{len(rows) / (all_tok / rate):.3f} texts/s)")
        if args.check_prefix:
            j.prefixes = {}
            t0 = time.time()
            res2 = j.score_ids([ids for ids, _ in built])
            dt2 = time.time() - t0
            d = max(abs(a["judge_score"] - b["judge_score"]) for a, b in zip(res, res2))
            print(f"check-prefix: without the cache {dt2:.1f}s; max |score difference| = {d:.2e}")
        for r, x in zip(sample, res):
            print(f"  {r['id']}: score={x['judge_score']:.2f} argmax={x['judge_argmax']} mass={x['judge_mass']:.3f}")
        return

    meta = {"model": args.model, "revision": hf_revision(args.model), "dtype": "float32",
            "system": SYSTEM, "instruction": INSTRUCTION, "context_block": CONTEXT_BLOCK,
            "text_block": TEXT_BLOCK, "prefill": PREFILL, "prefix_cache": not args.no_prefix_cache,
            "max_text_tokens": args.max_text_tokens, "max_context_tokens": args.max_context_tokens,
            "rating": "first token over digits 1-9; P(1) split into 1 and 10 by P('0'|'1'); "
                      "renormalised; score = expected rating"}
    run_family(args, f"judge_{tag}", rows, setup, meta)


if __name__ == "__main__":
    main()
