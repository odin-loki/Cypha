# The two side branches

`vChatGPT` and `vPattern Matching` are not steps on the main line. Both are family-A
experiments, both kept PyTorch, and neither continued. They are opposite answers to the same
question about v2.

Placement evidence is in [`../TIMELINE.md`](../TIMELINE.md#the-class-overlap-matrix-that-fixes-the-branch-points):
vChatGPT shares **30** class names with v2 against 8 with each of v3–v7; vPattern Matching's
`Resonator`, `AnchorMemory`, `MetaLearning` and `Cypha` are v3-generation names absent from v2.

---

## `cypha-vchatgpt` — v2 taken apart

**[`../archive/cypha-vchatgpt/`](../archive/cypha-vchatgpt/)** · 38 Python files, 1,906
lines, 57 classes · the only modular package, the only test suite, the only GUI ·
**authored 2026-02-13**

This is v2's ten-layer architecture decomposed into packages. Its `core/` re-implements the
named HRNA layers one per file:

```
core/levels.py        ResonatorLevel, AssemblyLevel, ModuleLevel, GlobalLevel
core/resonance.py     ResonanceField, FourierResonance, HarmonicCalculator
core/events.py        EventType, Event, EventQueue, LogSchedule, EventGenerator
core/recursion.py     HorizontalRecursion, VerticalRecursion, TemporalRecursion
core/feedback.py      ResonanceAmplifiedFeedback, CrossLevelFeedback, TemporalFeedback,
                      CriticalityEnhancedFeedback
core/compression.py   FundamentalExtractor, SymmetryEncoder, CrystalLatticeMapper, DNAFolder
core/encoder.py       UniversalEncoder, PrecisionPreservation
core/metalearning.py  RecursiveMetaLearning
core/optimization.py  AlternativeFastOperations, NaturalMathematicalShortcuts,
                      StrategicStochasticNoise, PrecisionControl
core/thought.py       MultiScaleThought, RecursiveEventCascades, ResonantEventChains,
                      SelfGeneratedEventStreams
```

Four of those — `AlternativeFastOperations`, `NaturalMathematicalShortcuts`,
`StrategicStochasticNoise`, `PrecisionControl` — are v2's Speed Enhancement Layer and exist
nowhere else on the main line. That is what pins this branch to v2.

Around the core it adds four packages the main line never had: `agi/`, `gui/` (PyQt6),
`learning/` (scraper, Pile loader, trainer, pipeline), `security/`, and a `tests/` tree.

### It is the most precisely dated version in the archive

The zip destroyed every file mtime, but vChatGPT shipped its `__pycache__`, and a CPython
`.pyc` header embeds the *source* file's mtime and size (PEP 552). Six directories can be
dated from evidence inside their files; this is the only one dated **per file**, thirty times
over. All 30 decode to a single session on **13 February 2026**, and 29 of the 30 embedded
sizes match the on-disk `.py` byte for byte:

```
05:29:13  core/thought.py          3417 B   <-- pyc says 3417, file is 3363
05:33:28  agi/nlp.py               1317 B
05:33:54  agi/code.py               879 B
05:34:23  agi/files.py              716 B
05:34:42  agi/system.py             519 B
05:35:02  agi/memory.py             601 B
05:35:20  agi/reasoning.py          360 B
05:35:43  agi/monitoring.py         247 B
05:36:43  learning/scraper.py      1134 B
05:37:27  learning/trainer.py       707 B
05:37:42  learning/pipeline.py      632 B
05:37:58  learning/pile_loader.py   568 B
05:39:07  gui/main_window.py       1043 B
05:40:23  gui/chat.py               832 B
05:40:36  gui/monitor.py            731 B
05:40:48  gui/training.py           446 B
05:41:02  gui/settings.py           592 B
05:41:57  security/sandbox.py       435 B
05:42:14  security/validation.py    341 B
05:42:27  security/audit.py         389 B
06:08:13  core/recursion.py        2765 B
06:08:44  core/feedback.py         2883 B
06:11:31  core/optimization.py     2434 B
06:26:06  core/encoder.py          2477 B
08:58:00  core/resonance.py        4388 B
08:59:52  core/compression.py      4165 B
09:00:35  core/events.py           3893 B
09:17:54  core/metalearning.py     3797 B
10:41:40  core/levels.py           7199 B
10:43:25  cypha.py                19052 B
```

Three things fall out of it.

**The whole package is 5 hours 14 minutes of work.** `agi/`, `learning/`, `gui/` and
`security/` — the four packages that make this look like an AGI system — were written between
05:33:28 and 05:42:27, **one file every 15 to 25 seconds**. That cadence is bulk generation,
and it is consistent with the directory's name.

**The session ended in debugging.** The last two files touched are `core/levels.py`
(10:41:40) and `cypha.py` (10:43:25) — the level hierarchy and the orchestrator that gave up
on it, below.

**One file was edited and never re-imported.** `core/thought.py` is the single size mismatch
— the `.pyc` records 3417 bytes against a 3363-byte file — which fits a module with zero
import sites.

This is the most precise authoring evidence anywhere in the archive, and the only reason it
survives is that someone zipped their `__pycache__`. It also independently confirms the DAG:
13 February is before v5's documentation (2026-02-21) and before v6 (2026-02-20 onward), and
the directory mtime of 2026-02-21 20:35 is a bulk archival copy eight days after the work.

The five `.pyc` files sitting in `cypha-v3/`, `cypha-v4/`, `cypha-v6/`, `cypha-v8/` and
`cypha-encoder/` are **not** part of this evidence and carry no historical information: they
are `cpython-311` artifacts generated while re-running the archived code during this analysis,
and their embedded source mtimes read 2026-09-17. They are untracked and gitignored. The
original zip contains exactly 30 `.pyc` files, all `cpython-312`, all under vChatGPT.


### The layers are bypassed in the source, and it says so

This is the archive's most explicit admission of the pattern traced in
[`../LINEAGE.md`](../LINEAGE.md#3-computed-and-discarded--the-pathology-that-ended-the-hrna-line).
In both the inference and the training path, `cypha.py` computes the level hierarchy and then
throws it away — with a comment:

```python
with self._time_block("assembly"):
    assem = self.assembly.update(reso)
with self._time_block("module"):
    module = self.module.update(assem)
with self._time_block("global"):
    # BYPASS broken layers - use Resonator directly
    globalv = reso[:64]          # Take first 64 elements from Resonator
    g = self._normalize(globalv) # Put normalization back
```
— `cypha.py:169-175`, repeated verbatim at `:215-221`

`AssemblyLevel` and `ModuleLevel` are still called, and still timed, and their outputs `assem`
and `module` are still returned in the result dict — but `g`, the value that goes on to the
meta-learner, is the **first 64 elements of the resonator output**, sliced. `GlobalLevel` is
constructed at `:79` and never invoked at all.

So of the ten layers vChatGPT decomposed out of v2, the working path is
encoder → resonance field → resonator → meta-learner. The rest is instrumented, timed, and
discarded.

Every other version in the archive does this silently, and the documentation describes the
discarded path as the mechanism. vChatGPT is the one place where someone wrote
`# BYPASS broken layers` and left it in. It is also, on the `.pyc` evidence above, the last
thing edited before the session ended.

### The ambition, and the implementation

`requirements.txt` lists **22 packages** — `torch`, `transformers`, `tokenizers`,
`datasets`, `PyQt6`, `beautifulsoup4`, `requests-html`, `python-docx`, `openpyxl`, `pypdf`,
`Pillow`, `watchdog`, `plotly`, `PyGObject` and more. It is by a wide margin the largest
dependency footprint in the archive, and the only appearance of the HuggingFace stack
anywhere in Cypha's history.

The `agi/` package that justifies the name is **120 lines across seven files**. In full,
`CyphaCode.generate` is:

```python
def generate(self, desc: str):
    if 'sort' in desc:
        return "def sort(xs):\n    return sorted(xs)\n"
    return f"# {desc}\npass\n"
```
— `agi/code.py`

`agi/reasoning.py` is 8 lines, `agi/monitoring.py` 9. These are placeholders, not
capabilities.

### Two things that do not work

**The trainer cannot train the model.** `CyphaTrainer` uses the standard `nn.Module` API:

```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
...
self.model.train();  out = self.model(x);  loss.backward();  self.optimizer.step()
```
— `learning/trainer.py`

But `CyphaHRNA` is a plain class — there is **no `nn.Module` anywhere in the package** — and
it defines no `parameters()`, no `train()` and no `eval()`. `CyphaTrainer` is also never
instantiated anywhere. The only gradient-descent code in Cypha's entire history is a trainer
that cannot attach to its model and is never called.

**The sandbox does not sandbox.** `security/sandbox.py` in full:

```python
def run(self, cmd: str):
    if cmd.split()[0] not in self.whitelist:
        return "Blocked by sandbox."
    try:
        return subprocess.check_output(cmd, shell=True, timeout=2).decode()
```

The whitelist is checked against the **first token only**, and the full string is then passed
to a shell. Anything beginning with an allowed word — `ls`, `pwd`, `head`, `tail` — carries
the rest of the line into `sh` unexamined. The module provides a name, not isolation.

This is dead archived code that was never on any product path, and it is recorded here
because a file called `sandbox.py` should not be assumed to be one.

The `tests/` tree is five files asserting tensor shapes.

### Its one inheritance

`config.yaml` contains:

```yaml
compression:
  target_ratio: 500000
```

The unsupported 500,000:1 figure — traced in [`00-prototypes.md`](00-prototypes.md#claims-that-do-not-hold-up)
from Cell AI v3 through V1 into v2 — has become a configuration parameter.

---

## `cypha-vpattern-matching` — v3 boiled down

**[`../archive/cypha-vpattern-matching/`](../archive/cypha-vpattern-matching/)** ·
3 Python files, 996 lines, 7 classes · the only branch labelled "production" · ships a
3.5 MB demo PDF

The opposite move. Where vChatGPT expanded v2 into 38 files, vPattern distils the HRNA core
into one:

```
TrainingMetrics · UniversalEncoder · ResonanceField · Resonator · AnchorMemory ·
MetaLearning · Cypha
```

and a linear pipeline: `Input → Encoder → Resonance Field → Resonator → Output`, with
`AnchorMemory` alongside.

### Its PDF dates it to 23 minutes after vChatGPT stopped

The 3.5 MB `Cypha Demo.pdf` is dead weight in every other respect, and it is the only reason
this branch can be dated at all. A PDF trailer carries its own creation metadata, and this one
survived the repack intact:

```
/CreationDate  D:20260213220627+11'00'      /Producer  Microsoft: Print To PDF
/ModDate       D:20260213220627+11'00'      /Author    Odin Loch
/Title         INTRODUCTION.md              18 pages, 3,542,797 bytes
```

`22:06:27 +11:00` is **2026-02-13 11:06:27 UTC**. The last file vChatGPT ever touched,
`cypha.py`, was saved at 10:43:25 UTC — [above](#it-is-the-most-precisely-dated-version-in-the-archive).
The two branches are **23 minutes and 2 seconds apart**, on the same evening.

That is a much stronger statement than the DAG alone can make. These are not two independent
experiments that happen to sit side by side in the archive: vPattern Matching was printed in
the same sitting in which vChatGPT was abandoned, minutes after someone wrote
`# BYPASS broken layers` and closed the file. Expanding v2 into 38 files and distilling v3
into one are the same evening's work, in that order.

Two smaller things fall out of the same block.

**The `+11'00'` offset is the archive's only explicit timezone.** It fixes the author's clock
at AEDT, which is what makes the UTC stamps elsewhere convertible — and it matches the
`+1000`/`+1100` offsets on this repository's own commits. Read locally, vChatGPT's "05:29 to
10:43" session is 16:29 to 21:43, and the PDF lands at 22:06.

**`INTRODUCTION.md` does not exist.** The PDF was printed from a Markdown file of that name,
and no file called `INTRODUCTION.md` appears anywhere in the archive — not in this directory,
not in any other. The branch ships an 18-page print of a document it does not contain. It joins
the [implementations the archive names but does not hold](../TIMELINE.md#three-implementations-named-by-the-archive-but-absent-from-it),
and it is the reason the PDF is worth keeping despite its size: the metadata is the evidence,
not the pages.

### It does not retrieve from the resonator either

`infer()` runs the full pipeline and then, for anything it has seen, ignores it:

```python
out = self.forward(x, raw_input=text)      ← resonance pipeline, computed

if text in self.target_mappings:           ← exact-string dict hit
    return self.target_mappings[text], 1.0

gs = out["global"].detach().cpu().numpy()  ← only reached for UNSEEN input
for word, anchor in self.vocab_anchors.items(): ...
```
— `cypha_production.py:487-500`

For any trained input the answer is an `O(1)` dict lookup on the raw string and `out` is
discarded. The resonance state is consulted only as the fallback for inputs not in the
mapping. `AnchorMemory.get` is defined at `:205` and the k-d tree only spaces anchors during
their own creation.

**Nothing is learned, either.** `amp_weights` and `phase_weights` are `torch.randn` draws
assigned once at `:39-40` and never reassigned; `MetaLearning` holds `recent_states` and no
parameters at all, so its `update()` computes losses that change nothing. "Training" builds
the `target_mappings` dict and allocates anchors.

That makes the demo's headline result circular: all five `test_cases` in `showcase_demo.py`
are among the 13 pairs written to `demo_data.txt`, so 5/5 is the dict answering questions it
was just given.

### Its claims, checked

| Claim | Reality |
|---|---|
| "Strong Separation: Average state distance > 0.9 (vs 0.0 for collapsed systems)" (`README.md:12`) | the null is wrong. Two independent random unit vectors in ℝ⁶⁴ are **1.4112** apart on average (≈ √2, measured over 200,000 pairs). A separation of 0.9 is *below* chance, not above a collapse baseline of 0. |
| "Semantic Clustering: Similar inputs produce similar states" | the mechanism is character-code proximity — the encoder is a fixed random projection of `text_to_tensor`, so nearby byte patterns give nearby states |
| "Fast Learning ... trains on thousands of examples in seconds" | dict construction and anchor allocation; no parameter is updated |
| "Anchor Memory (k-d tree lookup) → Output" (`README.md:30`) | retrieval for seen inputs is a Python dict hit; the k-d tree is not on the query path |

So vPattern does not escape the pattern its siblings are caught in — it adds an exact-match
shortcut *in front of* it. The distinction that matters is still real, though: it is the only
branch where the resonance state is on the query path **at all**, even if only for unseen
inputs, and the only one that kept the no-gradient commitment literally.

`README.md` gives the clearest statement of the HRNA thesis anywhere in the archive:

> **Harmonic Recursive Neural Architecture** — A resonance-based AGI system for learning
> input-output mappings through quantum-inspired field dynamics.
>
> Cypha is a novel AI architecture that learns mappings through resonance field evolution
> rather than traditional backpropagation.

### It means the no-backpropagation part literally

`cypha_production.py` imports `torch` but contains **zero** autograd: no `.backward()`, no
optimiser, no `requires_grad`, no `nn.Module`, no `nn.Linear`. The torch surface it actually
uses is an array library —

```
torch.is_complex ×7   torch.norm ×5   torch.randn ×4   torch.abs ×4   torch.linspace ×3
torch.zeros ×2  torch.sigmoid ×2  torch.mv ×2  torch.fft ×2  torch.exp ×2
torch.cosine_similarity ×2  torch.cfloat ×2
```

`amp_weights` and `phase_weights` are `torch.randn` matrices that are never optimised.
PyTorch is here for complex tensors and a possible GPU, not for learning.

That is a meaningful distinction from vChatGPT, which pulled in `transformers` and `datasets`
and wrote an Adam loop. vPattern kept the framework as a numerical backend while keeping the
project's actual commitment — no gradients — intact.

The separation metric is worth one note: it is the same anti-collapse diagnostic v3 shipped
as `AnchorMemory.separation_stats` with its "⚠ COLLAPSING" warning. Measuring
representational collapse was a standing concern across the whole line — the baseline it was
measured against is what went wrong here.

---

## Why neither continued

The main line's answer to both branches is visible in what it did next.

**Against vChatGPT:** v3 deleted `torch`, `ray` and `scipy` in one move and never took them
back. vChatGPT is the maximal version of the opposite bet — 22 dependencies, a GUI, a web
scraper, an `agi/` package — and its core capability modules are 8-to-23-line stubs while
its trainer cannot attach to its model. Breadth was added where depth was missing.

**Against vPattern Matching:** it is labelled production while answering from an exact-string
dict, learning nothing, and reporting a separation figure against a baseline of 0 when the
random-vector baseline is 1.41. There is nothing here for the main line to adopt. Its one
real distinction — keeping the resonance state on the query path, even if only for unseen
inputs — is the thing v3 had already abandoned deliberately, having measured what it costs:
`CyphaMicro`, the same approach at 64 dimensions, recalls 8 of its own 14 training pairs and
returns different answers to identical repeated queries (see
[`components.md`](components.md#cypha-encoder--the-standalone-encoder)).

Both branches kept a framework. Both stopped. The line that continued was the one that had
already thrown the framework away, and it is the only one that could be translated to C++
as arithmetic rather than reimplemented.

Both branches also carry the archive's recurring pathology in its purest forms — vChatGPT
annotating the bypass in a comment, vPattern short-circuiting past it with a dict. Whatever
else the main line got wrong, it never shipped a version whose demonstrated capability was
`dict.__getitem__`.

---

**See also:** [`../TIMELINE.md`](../TIMELINE.md) for the class-overlap matrix that places
these branches · [`../LINEAGE.md`](../LINEAGE.md#2-the-dependency-floor--why-a-c-port-was-available-at-all)
for the dependency thread.
