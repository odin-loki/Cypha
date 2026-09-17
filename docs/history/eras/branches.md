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

### It is the only version in the archive that can be dated to the minute

The zip destroyed every file mtime, but vChatGPT shipped its `__pycache__`, and a CPython
`.pyc` header embeds the *source* file's mtime and size (PEP 552). All 30 of them decode to a
single session on **13 February 2026**, and 29 of the 30 embedded sizes match the on-disk
`.py` byte for byte:

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
13 February is before v6 (2026-02-27), and the directory mtime of 2026-02-21 20:35 is a bulk
archival copy eight days after the work.


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
`AnchorMemory` alongside. It is the only version in the archive whose retrieval sits where
the architecture documents say it should — on the resonator output — while still being
presented as production-ready.

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

### Its claims

The README advertises "Fast Learning" (thousands of examples in seconds), "Strong
Separation" (average state distance > 0.9 versus 0.0 for collapsed systems), "Semantic
Clustering", `O(N log N)` complexity via FFT, and scaling across "math, language, logic,
sorting". The separation metric is the interesting one: it is the same anti-collapse
diagnostic that v3 shipped as `AnchorMemory.separation_stats` with its "⚠ COLLAPSING"
warning. Measuring representational collapse was a standing concern across the whole line.

---

## Why neither continued

The main line's answer to both branches is visible in what it did next.

**Against vChatGPT:** v3 deleted `torch`, `ray` and `scipy` in one move and never took them
back. vChatGPT is the maximal version of the opposite bet — 22 dependencies, a GUI, a web
scraper, an `agi/` package — and its core capability modules are 8-to-23-line stubs while
its trainer cannot attach to its model. Breadth was added where depth was missing.

**Against vPattern Matching:** it does the honest thing — retrieving from the dynamical
state — and the main line had already measured what that costs. `CyphaMicro`, the same
approach at 64 dimensions, is non-deterministic across repeated calls and scores 7/14 on its
own training set (see [`components.md`](components.md#cypha-encoder--the-standalone-encoder)).
v3 moved retrieval to encoder features precisely to escape that, and never went back.

Both branches kept a framework. Both stopped. The line that continued was the one that had
already thrown the framework away, and it is the only one that could be translated to C++
as arithmetic rather than reimplemented.

---

**See also:** [`../TIMELINE.md`](../TIMELINE.md) for the class-overlap matrix that places
these branches · [`../LINEAGE.md`](../LINEAGE.md#2-the-dependency-floor--why-a-c-port-was-available-at-all)
for the dependency thread.
