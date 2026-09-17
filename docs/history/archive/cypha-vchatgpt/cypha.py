import torch
import os
import pickle
import time
import numpy as np
from sklearn.neighbors import KDTree

from core.encoder import UniversalEncoder, PrecisionPreservation
from core.compression import FundamentalExtractor, SymmetryEncoder, CrystalLatticeMapper, DNAFolder
from core.resonance import ResonanceField, EnhancedResonance
from core.events import EventGenerator, EventQueue, EventType
from core.recursion import HorizontalRecursion, VerticalRecursion, TemporalRecursion
from core.feedback import ResonanceAmplifiedFeedback
from core.levels import ResonatorLevel, AssemblyLevel, ModuleLevel, GlobalLevel
from core.metalearning import RecursiveMetaLearning
from core.optimization import AlternativeFastOperations

class FastSeparationMemory:
    def __init__(self, dim=64, min_separation=0.5):
        self.anchors = {}
        self.anchor_list = []
        self.anchor_keys = []
        self.dim = dim
        self.min_separation = min_separation
        self.kdtree = None
    
    def _rebuild_index(self):
        if len(self.anchor_list) > 0:
            self.kdtree = KDTree(np.array(self.anchor_list))
    
    def set_anchor(self, input_key):
        if input_key in self.anchors:
            return
        best_anchor = None
        max_min_dist = 0
        for attempt in range(5):
            candidate = np.random.randn(self.dim)
            candidate /= np.linalg.norm(candidate)
            if len(self.anchor_list) == 0:
                best_anchor = candidate
                break
            dist, _ = self.kdtree.query([candidate], k=1)
            min_dist = float(dist[0, 0])
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_anchor = candidate
                if max_min_dist > self.min_separation:
                    break
        self.anchors[input_key] = best_anchor
        self.anchor_list.append(best_anchor)
        self.anchor_keys.append(input_key)
        if len(self.anchor_list) % 32 == 0:
            self._rebuild_index()
        elif self.kdtree is None:
            self._rebuild_index()
        print(f"Set anchor for '{input_key}' with separation={max_min_dist:.3f}")
        return best_anchor

class CyphaHRNA:
    def __init__(self, input_dim=32, resonance_dim=64, device="cpu"):
        self.device = device
        self.encoder = UniversalEncoder(input_dim, resonance_dim, device)
        self.precision = PrecisionPreservation(device)
        self.extractor = FundamentalExtractor(device=device)
        self.symmetry = SymmetryEncoder(device=device)
        self.lattice = CrystalLatticeMapper(device=device)
        self.dna = DNAFolder(device=device)
        self.resfield = ResonanceField(resonance_dim, device=device)
        self.enhanced_resonance = EnhancedResonance(temperature=2.0, device=device)
        self.eventgen = EventGenerator(device=device)
        self.events = EventQueue()
        self.hr = HorizontalRecursion(device=device)
        self.vr = VerticalRecursion(device=device)
        self.tr = TemporalRecursion(device=device)
        self.feedback = ResonanceAmplifiedFeedback(device=device)
        self.resonator = ResonatorLevel(n=resonance_dim, gamma_inhib=0.35, device=device)
        self.assembly = AssemblyLevel(n=32, nr=resonance_dim, device=device)
        self.module = ModuleLevel(n=16, na=32, device=device)
        self.global_level = GlobalLevel(d=64, nm=16, device=device)
        self.meta = RecursiveMetaLearning(state_dim=64, device=device)
        self.step = 0
        self.training_step = 0
        self.mem_trace = []
        self.forced_memory = FastSeparationMemory(dim=64, min_separation=0.6)
        self.target_answers = {}
        self.output_vocab = [
            "hello", "bark", "meow", "umbrella", "Paris", "goodbye", "off", "on", "blue", "moon",
            "down", "no", "one", "mars", "two", "snake", "growl", "quick",
            "short", "day", "west", "rain", "cat", "dog", "France", "python", "bear", "fox", "tall",
            "night", "east", "true", "false", "zero", "earth", "up", "yes", "sun"
        ]
        self.vocab_anchors = self._gen_vocab_anchors()
        self.event_chains = {}
        self.timings = {}

    def _time_block(self, name):
        class Timer:
            def __init__(self, timings, name):
                self.timings = timings
                self.name = name
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                if self.name not in self.timings:
                    self.timings[self.name] = []
                self.timings[self.name].append(elapsed)
        return Timer(self.timings, name)

    def text_to_tensor(self, text):
        vals = [ord(c)%256/255.0 for c in text[:self.encoder.input_dim]]
        t = torch.zeros(self.encoder.input_dim)
        for i, v in enumerate(vals): t[i]=v
        return t.to(self.device)

    def _normalize(self, x):
        if torch.is_complex(x): 
            x = x.real
        x = torch.clamp(x, -100, 100)
        norm = torch.norm(x) + 1e-8
        return x / norm

    def _pad_trunc(self, a, b):
        if a.shape[0] < b.shape[0]:
            a = torch.cat([a, torch.zeros(b.shape[0] - a.shape[0], device=a.device)])
        elif b.shape[0] < a.shape[0]:
            b = torch.cat([b, torch.zeros(a.shape[0] - b.shape[0], device=b.device)])
        return a, b

    def _gen_vocab_anchors(self):
        anchors = {}
        for word in self.output_vocab:
            for attempt in range(10):
                anchor = np.random.randn(64)
                anchor /= np.linalg.norm(anchor)
                if not anchors:
                    anchors[word] = anchor
                    break
                dists = [1 - np.dot(anchor, other) for other in anchors.values()]
                if min(dists) > 0.6:
                    anchors[word] = anchor
                    break
            else:
                anchors[word] = anchor
        return anchors

    def forward(self, inp, raw_input=None):
        with self._time_block("encode"):
            enc = self.encoder.encode(inp)
        with self._time_block("compression"):
            mant, exp = self.precision.preserve(enc)
            comp = self.extractor.extract(enc)
            sym = self.symmetry.encode(comp)
            lat = self.lattice.map(enc.real)
            dna = self.dna.fold(enc.real)
            with self._time_block("resonance"):
                self.resfield.add_event(enc.real)
                rfield = self.resfield.evolve(1)
        with self._time_block("events"):
            pattern_event = self.eventgen.pattern_event(enc.real, [enc.real], float(time.time()), source="pattern")
            for e in pattern_event: self.events.add(e)
        with self._time_block("recursion"):
            recur = self.hr.update("main", enc.real, {"comp": enc.real}, events=pattern_event)
        with self._time_block("feedback"):
            fed = self.feedback.compute(enc.real, recur)
        with self._time_block("resonator"):
            reso = self.resonator.update(events=pattern_event, external_drive=rfield.real)
        with self._time_block("assembly"):
            assem = self.assembly.update(reso)
        with self._time_block("module"):
            module = self.module.update(assem)
        with self._time_block("global"):
            # BYPASS broken layers - use Resonator directly
            globalv = reso[:64]  # Take first 64 elements from Resonator
            g = self._normalize(globalv)  # Put normalization back
        self.step += 1
        return {
            "encoding": enc,
            "compressed": comp,
            "res_field": rfield,
            "recurrent": recur,
            "feedback": fed,
            "resonator": reso,
            "assembly": assem,
            "module": module,
            "global": g,
            "meta_loss": 0.0,
        }

    def forward_supervised(self, inp, target, raw_input=None, raw_target=None, negatives=None):
        with self._time_block("encode"):
            enc = self.encoder.encode(inp)
            enc_target = self.encoder.encode(target)
        with self._time_block("compression"):
            mant, exp = self.precision.preserve(enc)
            comp = self.extractor.extract(enc)
            sym = self.symmetry.encode(comp)
            lat = self.lattice.map(enc.real)
            dna = self.dna.fold(enc.real)
            with self._time_block("resonance"):
                self.resfield.add_event(enc.real)
                rfield = self.resfield.evolve(1)
                self.resonator.R = self.resonator.R + 0.1 * rfield.real[:self.resonator.n]
        with self._time_block("events"):
            pattern_event = self.eventgen.pattern_event(enc.real, [enc.real], float(time.time()), source="pattern")
            for e in pattern_event: self.events.add(e)
        with self._time_block("recursion"):
            recur = self.hr.update("main", enc.real, {"comp": enc.real}, events=pattern_event)
        with self._time_block("feedback"):
            fed = self.feedback.compute(enc.real, recur)
        with self._time_block("resonator"):
            reso = self.resonator.update(events=pattern_event, external_drive=rfield.real)
        with self._time_block("assembly"):
            assem = self.assembly.update(reso)
        with self._time_block("module"):
            module = self.module.update(assem)
        with self._time_block("global"):
            # BYPASS broken layers - use Resonator directly
            globalv = reso[:64]  # Take first 64 elements from Resonator
            g = self._normalize(globalv)  # Put normalization back
        with self._time_block("meta"):
            t = enc_target.real if torch.is_complex(enc_target) else enc_target
            pred, loss = self.meta.update(g, t[:g.shape[0]], negatives=negatives)
        self.step += 1
        self.training_step += 1
        if self.training_step % 100 == 0:
            current_temp = self.enhanced_resonance.temperature
            new_temp = max(0.1, current_temp * 0.99)
            self.enhanced_resonance.temperature = new_temp
            if self.training_step % 1000 == 0:
                print(f"Temperature annealed to {new_temp:.3f}")
        if raw_input and raw_input not in self.forced_memory.anchors:
            self.forced_memory.set_anchor(raw_input)
            if raw_target:
                self.target_answers[raw_input] = raw_target
        return {
            "encoding": enc,
            "compressed": comp,
            "res_field": rfield,
            "recurrent": recur,
            "feedback": fed,
            "resonator": reso,
            "assembly": assem,
            "module": module,
            "global": g,
            "meta_loss": loss
        }

    def train_on_pairs(self, path, max_lines=None, epochs=10, batch_size=8):
        if not os.path.exists(path): 
            return 0
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and "|||" in l]
        count = 0
        for ep in range(epochs):
            print(f"Epoch {ep+1}/{epochs}")
            np.random.shuffle(lines)
            for batch_start in range(0, len(lines), batch_size):
                if max_lines and count >= max_lines:
                    break
                batch = lines[batch_start:batch_start + batch_size]
                batch_states = []
                for line in batch:
                    if "|||" not in line:
                        continue
                    parts = line.split("|||")
                    if len(parts) != 2:
                        continue
                    inp, tgt = [s.strip() for s in parts]
                    x = self.text_to_tensor(inp)
                    y = self.text_to_tensor(tgt)
                    enc = self.encoder.encode(x)
                    self.resfield.add_event(enc.real)
                    rfield = self.resfield.evolve(1)
                    reso = self.resonator.update()
                    assem = self.assembly.update(reso)
                    module = self.module.update(assem)
                    globalv = self.global_level.update(module)
                    g = self._normalize(globalv)
                    batch_states.append((g, y, inp, tgt))
                for i, (g, y, inp, tgt) in enumerate(batch_states):
                    negatives = [s for j, (s, _, _, _) in enumerate(batch_states) if j != i]
                    enc_target = self.encoder.encode(y)
                    t = enc_target.real if torch.is_complex(enc_target) else enc_target
                    pred, loss = self.meta.update(g, t[:g.shape[0]], negatives=negatives)
                    if inp not in self.forced_memory.anchors:
                        self.forced_memory.set_anchor(inp)
                        if tgt:
                            self.target_answers[inp] = tgt
                    count += 1
            if (ep + 1) % 1 == 0:
                self.test_separation()
        print(f"Done training {count} pairs for {epochs} epochs.")
        return count

    def compute_vocab_matches_with_temperature(self, global_state, temperature=None):
        if temperature is None:
            temperature = self.enhanced_resonance.temperature
        gs = global_state.detach().cpu().numpy()
        scores = {}
        for word, anchor in self.vocab_anchors.items():
            similarity = np.dot(gs, anchor) / (np.linalg.norm(gs) * np.linalg.norm(anchor) + 1e-9)
            scores[word] = np.exp(similarity / temperature)
        total = sum(scores.values())
        if total == 0: 
            return []
        scores = {k: v/total for k,v in scores.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def explicit_decoder(self, global_state, input_str=None):
        matches = self.compute_vocab_matches_with_temperature(global_state)
        if len(matches) < 2:
            return "[ambiguous/unknown]"
        (top_word, top_score), (runner_word, runner_score) = matches[0], matches[1]
        if top_score - runner_score < 0.15:
            if input_str and input_str in self.target_answers:
                return f"[anchor-memory:{self.target_answers[input_str]}]"
            return "[ambiguous/unknown]"
        return top_word

    def decode_output(self, global_state):
        arr = self._normalize(global_state).detach().cpu().numpy()
        arr = arr + np.random.randn(*arr.shape)*0.01
        chars = [chr(32 + int(np.clip(a,0,1) * 95)) for a in arr]
        return ''.join(chars)

    def test_separation(self):
        test_inputs = ["cat", "dog", "Paris", "python", "umbrella"]
        states = {}
        for inp in test_inputs:
            self.resonator.R = torch.zeros_like(self.resonator.R)
            self.assembly.A = torch.zeros_like(self.assembly.A)
            self.module.M = torch.zeros_like(self.module.M)
            self.global_level.G = torch.zeros_like(self.global_level.G)
            self.resfield.psi = torch.randn(self.resfield.dim, dtype=torch.cfloat, device=self.device)
            self.resfield.psi = self.resfield.psi / torch.norm(self.resfield.psi)
            out = self.forward(self.text_to_tensor(inp), raw_input=inp)
            g = out["global"]
            states[inp] = g.detach().cpu().numpy()
        print("\nPairwise State Distances:")
        print("-" * 50)
        for i, inp1 in enumerate(test_inputs):
            for inp2 in test_inputs[i+1:]:
                dist = np.linalg.norm(states[inp1] - states[inp2])
                sim = np.dot(states[inp1], states[inp2]) / (
                    (np.linalg.norm(states[inp1]) * np.linalg.norm(states[inp2]) + 1e-9)
                )
                print(f"{inp1:10} vs {inp2:10}: dist={dist:.4f}, sim={sim:.4f}")
        all_dists = [
            np.linalg.norm(states[inp1] - states[inp2])
            for i, inp1 in enumerate(test_inputs)
            for inp2 in test_inputs[i+1:]
        ]
        avg_dist = np.mean(all_dists)
        print(f"\nAverage distance: {avg_dist:.4f}")
        print("PASS" if avg_dist > 0.1 else "FAIL - States still collapsed!")

    def print_timings(self):
        print("\n" + "="*60)
        print("PERFORMANCE PROFILE")
        print("="*60)
        total = 0
        for name, times in sorted(self.timings.items(), 
                                   key=lambda x: sum(x[1]), 
                                   reverse=True):
            avg = sum(times) / len(times) * 1000
            tot = sum(times) * 1000
            pct = sum(times) / sum(sum(t) for t in self.timings.values()) * 100
            total += sum(times)
            print(f"{name:20s}: {avg:8.3f}ms avg, {tot:10.1f}ms total, {pct:5.1f}%")
        print("="*60)
        print(f"{'TOTAL':20s}: {total*1000:8.3f}ms")
        print("="*60)

def main_cli():
    model = CyphaHRNA(device="cpu")
    print("Cypha AGI HRNA System - Type 'batchtrain <file> [N]', 'trainpairs <file>', 'infer <text>', 'explicit <text>', 'testsep', 'profile', or 'exit'")
    while True:
        cmd = input("cypha> ").strip()
        if cmd.startswith("batchtrain "):
            tokens = cmd.split(" ", 2)
            path = tokens[1]
            n_epochs = int(tokens[2]) if len(tokens) > 2 else 10
            n = model.train_on_pairs(path, max_lines=None, epochs=n_epochs)
            print(f"Batch training finished. Trained on {n} pairs for {n_epochs} epochs.")
        elif cmd.startswith("trainpairs "):
            path = cmd[len("trainpairs "):]
            n = model.train_on_pairs(path, max_lines=5000)
            print("Paired training on", n, "lines.")
        elif cmd.startswith("infer "):
            t = cmd[len("infer "):]
            x = model.text_to_tensor(t)
            out = model.forward(x, raw_input=t)
            top_matches = model.compute_vocab_matches_with_temperature(out["global"])[:5]
            print("Top vocab matches:", top_matches)
            print("Global:", out["global"].detach().cpu().numpy())
            print("Meta loss:", out["meta_loss"])
            print("Cypha output:", model.decode_output(out["global"]))
            print("Explicit decode [WTA]:", model.explicit_decoder(out["global"], input_str=t))
            if t in model.target_answers:
                print("Memory anchor (if known):", model.target_answers[t])
        elif cmd.startswith("explicit "):
            t = cmd[len("explicit "):]
            x = model.text_to_tensor(t)
            out = model.forward(x, raw_input=t)
            print("Explicit decode:", model.explicit_decoder(out["global"], input_str=t))
        elif cmd == "testsep":
            model.test_separation()
        elif cmd == "profile":
            model.print_timings()
        elif cmd == "exit":
            break
        else:
            print("Commands: batchtrain <file> [N], trainpairs <file>, infer <text>, explicit <text>, testsep, profile, exit")

if __name__ == "__main__":
    main_cli()
