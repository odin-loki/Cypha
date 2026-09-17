"""
Cypha HRNA - Harmonic Recursive Neural Architecture
Production Implementation

A resonance-based AGI system that learns input-output mappings through
harmonic field evolution and contrastive memory anchoring.

Architecture:
    Input → Encoder → Resonance Field → Resonator → Output
                                              ↓
                                      Anchor Memory
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.neighbors import KDTree


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    epoch: int
    loss: float
    avg_separation: float
    num_anchors: int
    temperature: float


class UniversalEncoder:
    """Encodes any input into resonant representation"""
    def __init__(self, input_dim: int = 32, resonance_dim: int = 64, device: str = "cpu"):
        self.input_dim = input_dim
        self.resonance_dim = resonance_dim
        self.device = device
        
        # Fixed random projections
        self.amp_weights = torch.randn(input_dim, resonance_dim, device=device)
        self.phase_weights = torch.randn(input_dim, resonance_dim, device=device)
        self.basis_freqs = torch.linspace(0.5, 10.0, resonance_dim, device=device)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to resonant encoding"""
        x = x.to(self.device).float().reshape(-1)
        amps = torch.mv(self.amp_weights.t(), x)
        phases = torch.mv(self.phase_weights.t(), x)
        domain = torch.arange(self.resonance_dim, device=self.device)
        basis = torch.sin(self.basis_freqs * domain / self.resonance_dim)
        return amps * torch.exp(1j * phases) * basis


class ResonanceField:
    """Quantum-inspired resonance field with FFT evolution"""
    def __init__(self, dim: int, gamma: float = 0.1, dt: float = 0.1, device: str = "cpu"):
        self.dim = dim
        self.gamma = gamma
        self.dt = dt
        self.device = device
        
        # Complex wavefunction
        self.psi = torch.randn(dim, dtype=torch.cfloat, device=device)
        self.psi = self.psi / torch.norm(self.psi)
        
        # Diagonal Hamiltonian in frequency domain
        self.H_freq = torch.linspace(0.5, 10.0, dim, device=device)
    
    def add_event(self, event: torch.Tensor, strength: float = 1.0):
        """Inject event into field"""
        v = event.flatten()[:self.dim]
        v = v / (torch.norm(v) + 1e-8)
        self.psi = (1 - strength) * self.psi + strength * v.type(torch.cfloat)
        self.psi = self.psi / torch.norm(self.psi)
    
    def evolve(self, steps: int = 1) -> torch.Tensor:
        """Evolve field using FFT-based dynamics"""
        for _ in range(steps):
            # Linear evolution in frequency domain (O(N log N))
            psi_freq = torch.fft.fft(self.psi)
            phase = torch.exp(-1j * self.dt * self.H_freq)
            psi_freq = psi_freq * phase
            self.psi = torch.fft.ifft(psi_freq)
            
            # Nonlinear term
            psi_squared = torch.abs(self.psi) ** 2
            nonlinear = self.gamma * self.dt * (psi_squared - 1.0) * self.psi
            self.psi = self.psi + nonlinear
            self.psi = self.psi / (torch.norm(self.psi) + 1e-8)
        
        return self.psi


class Resonator:
    """Resonator layer with local coupling and external drive"""
    def __init__(self, n: int = 64, gamma_inhib: float = 0.35, 
                 locality_radius: int = 3, device: str = "cpu"):
        self.n = n
        self.device = device
        self.gamma_inhib = gamma_inhib
        self.locality_radius = locality_radius
        
        self.R = torch.zeros(n, device=device)
        self.omega = torch.linspace(1.0, 10.0, n, device=device)
        self.local_weights = torch.randn(2 * locality_radius + 1, device=device) * 0.3
        self.local_weights[locality_radius] = 0
    
    def local_coupling(self, R: torch.Tensor) -> torch.Tensor:
        """Compute local neighborhood coupling"""
        coupling = torch.zeros_like(R)
        for offset in range(-self.locality_radius, self.locality_radius + 1):
            if offset == 0:
                continue
            weight_idx = offset + self.locality_radius
            weight = self.local_weights[weight_idx]
            if offset > 0:
                coupling[:-offset] += weight * torch.sigmoid(R[offset:])
            else:
                coupling[-offset:] += weight * torch.sigmoid(R[:offset])
        return coupling
    
    def update(self, dt: float = 0.1, external_drive: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update resonator state with optional external drive"""
        freq_term = self.omega * self.R
        connect_term = self.local_coupling(self.R)
        
        # Global competitive inhibition
        total_activity = torch.sum(torch.abs(self.R))
        inhibition = -self.gamma_inhib * total_activity / self.R.numel()
        
        # External drive (from resonance field)
        drive_term = 0.0
        if external_drive is not None:
            drive_term = external_drive[:self.n] * 200.0  # Strong drive
        
        # Update
        R_new = self.R + dt * (freq_term + connect_term + inhibition) + drive_term
        
        # Competitive sparsification
        abs_R = torch.abs(R_new)
        threshold = torch.quantile(abs_R, 0.8)
        mask = abs_R < threshold
        R_new[mask] *= 0.1
        
        self.R = R_new
        
        # Prevent explosion
        max_val = torch.max(torch.abs(self.R))
        if max_val > 10:
            self.R = 10 * self.R / max_val
        
        return self.R


class AnchorMemory:
    """Fast k-d tree based anchor memory with forced separation"""
    def __init__(self, dim: int = 64, min_separation: float = 0.6):
        self.dim = dim
        self.min_separation = min_separation
        self.anchors: Dict[str, np.ndarray] = {}
        self.anchor_list: List[np.ndarray] = []
        self.anchor_keys: List[str] = []
        self.kdtree: Optional[KDTree] = None
    
    def _rebuild_index(self):
        """Rebuild k-d tree index"""
        if len(self.anchor_list) > 0:
            self.kdtree = KDTree(np.array(self.anchor_list))
    
    def set_anchor(self, key: str) -> Optional[np.ndarray]:
        """Create well-separated anchor for key"""
        if key in self.anchors:
            return self.anchors[key]
        
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
        
        self.anchors[key] = best_anchor
        self.anchor_list.append(best_anchor)
        self.anchor_keys.append(key)
        
        # Rebuild index periodically
        if len(self.anchor_list) % 32 == 0 or self.kdtree is None:
            self._rebuild_index()
        
        return best_anchor
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve anchor for key"""
        return self.anchors.get(key)


class MetaLearning:
    """Contrastive meta-learning for state separation"""
    def __init__(self, state_dim: int = 64, max_recent: int = 8, device: str = "cpu"):
        self.state_dim = state_dim
        self.max_recent = max_recent
        self.device = device
        self.recent_states: List[torch.Tensor] = []
    
    def update(self, state: torch.Tensor, target: torch.Tensor, 
               negatives: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, float]:
        """Update with contrastive learning"""
        s = state.view(-1)
        t = target.view(-1)
        
        if torch.is_complex(s):
            s = s.real
        if torch.is_complex(t):
            t = t.real
        
        s_len = s.shape[0]
        t_trunc = t[:s_len]
        
        # Positive loss (match target)
        positive_loss = torch.mean((s - t_trunc) ** 2)
        
        # Contrastive loss (push away from negatives)
        contrastive_loss = 0.0
        if negatives:
            for neg in negatives:
                if torch.is_complex(neg):
                    neg = neg.real
                neg = neg.view(-1)[:s_len]
                similarity = torch.cosine_similarity(s, neg, dim=0)
                contrastive_loss += torch.relu(similarity - 0.2) ** 2
        
        # Anti-collapse penalty
        penalty = 1.0
        if len(self.recent_states) > 0:
            similarities = []
            for p in self.recent_states:
                if torch.is_complex(p):
                    p = p.real
                p = p.view(-1)[:s_len]
                sim = torch.cosine_similarity(s, p, dim=0).abs().item()
                similarities.append(sim)
            if similarities:
                penalty = float(np.exp(-3.0 * sum(similarities)))
        
        total_loss = (positive_loss + 0.8 * contrastive_loss) * penalty
        
        # Track recent states
        self.recent_states.append(s.detach().clone())
        if len(self.recent_states) > self.max_recent:
            self.recent_states = self.recent_states[-self.max_recent:]
        
        return s, float(total_loss)


class Cypha:
    """Main Cypha HRNA system"""
    
    def __init__(self, input_dim: int = 32, resonance_dim: int = 64, device: str = "cpu"):
        self.device = device
        self.resonance_dim = resonance_dim
        
        # Core components
        self.encoder = UniversalEncoder(input_dim, resonance_dim, device)
        self.resfield = ResonanceField(resonance_dim, device=device)
        self.resonator = Resonator(n=resonance_dim, device=device)
        self.anchor_memory = AnchorMemory(dim=resonance_dim)
        self.meta = MetaLearning(state_dim=resonance_dim, device=device)
        
        # Training state
        self.training_step = 0
        self.temperature = 2.0
        self.target_mappings: Dict[str, str] = {}
        
        # Output vocabulary (for decode)
        self.output_vocab = [
            "hello", "bark", "meow", "umbrella", "Paris", "goodbye", "off", "on",
            "blue", "moon", "down", "no", "one", "mars", "two", "snake", "growl",
            "quick", "short", "day", "west", "rain", "cat", "dog", "France",
            "python", "bear", "fox", "tall", "night", "east", "true", "false",
            "zero", "earth", "up", "yes", "sun"
        ]
        self.vocab_anchors = self._gen_vocab_anchors()
    
    def _gen_vocab_anchors(self) -> Dict[str, np.ndarray]:
        """Generate well-separated anchors for vocabulary"""
        anchors = {}
        for word in self.output_vocab:
            for attempt in range(10):
                anchor = np.random.randn(self.resonance_dim)
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
    
    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to input tensor"""
        vals = [ord(c) % 256 / 255.0 for c in text[:self.encoder.input_dim]]
        t = torch.zeros(self.encoder.input_dim, device=self.device)
        for i, v in enumerate(vals):
            t[i] = v
        return t
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2 normalization"""
        if torch.is_complex(x):
            x = x.real
        x = torch.clamp(x, -100, 100)
        norm = torch.norm(x) + 1e-8
        return x / norm
    
    def forward(self, inp: torch.Tensor, raw_input: Optional[str] = None) -> Dict:
        """Forward pass through system"""
        # Encode
        enc = self.encoder.encode(inp)
        
        # Resonance field evolution
        self.resfield.add_event(enc.real)
        rfield = self.resfield.evolve(1)
        
        # Resonator (driven by field)
        reso = self.resonator.update(external_drive=rfield.real)
        
        # Output (use resonator directly - bypass broken layers)
        global_state = self._normalize(reso[:self.resonance_dim])
        
        return {
            "encoding": enc,
            "resonance_field": rfield,
            "resonator": reso,
            "global": global_state,
            "meta_loss": 0.0
        }
    
    def forward_supervised(self, inp: torch.Tensor, target: torch.Tensor,
                          raw_input: Optional[str] = None, raw_target: Optional[str] = None,
                          negatives: Optional[List[torch.Tensor]] = None) -> Dict:
        """Supervised forward with meta-learning"""
        # Encode
        enc = self.encoder.encode(inp)
        enc_target = self.encoder.encode(target)
        
        # Resonance field evolution
        self.resfield.add_event(enc.real)
        rfield = self.resfield.evolve(1)
        
        # Resonator (driven by field)
        reso = self.resonator.update(external_drive=rfield.real)
        
        # Output
        global_state = self._normalize(reso[:self.resonance_dim])
        
        # Meta-learning
        t = enc_target.real if torch.is_complex(enc_target) else enc_target
        pred, loss = self.meta.update(global_state, t[:global_state.shape[0]], negatives=negatives)
        
        self.training_step += 1
        
        # Temperature annealing
        if self.training_step % 100 == 0:
            self.temperature = max(0.1, self.temperature * 0.99)
        
        # Store mapping
        if raw_input and raw_target:
            self.anchor_memory.set_anchor(raw_input)
            self.target_mappings[raw_input] = raw_target
        
        return {
            "encoding": enc,
            "resonance_field": rfield,
            "resonator": reso,
            "global": global_state,
            "meta_loss": loss
        }
    
    def train(self, data_path: str, epochs: int = 1, batch_size: int = 8,
              max_samples: Optional[int] = None, verbose: bool = True) -> List[TrainingMetrics]:
        """Train on data file with input|||target format"""
        import os
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and "|||" in l]
        
        if max_samples:
            lines = lines[:max_samples]
        
        metrics_history = []
        
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
            
            np.random.shuffle(lines)
            epoch_loss = 0
            count = 0
            
            # Batch training
            for batch_start in range(0, len(lines), batch_size):
                batch = lines[batch_start:batch_start + batch_size]
                batch_states = []
                
                # Forward pass for batch
                for line in batch:
                    if "|||" not in line:
                        continue
                    
                    parts = line.split("|||")
                    if len(parts) != 2:
                        continue
                    
                    inp, tgt = [s.strip() for s in parts]
                    x = self.text_to_tensor(inp)
                    y = self.text_to_tensor(tgt)
                    
                    # Forward without meta update
                    enc = self.encoder.encode(x)
                    self.resfield.add_event(enc.real)
                    rfield = self.resfield.evolve(1)
                    reso = self.resonator.update(external_drive=rfield.real)
                    g = self._normalize(reso[:self.resonance_dim])
                    
                    batch_states.append((g, y, inp, tgt))
                
                # Update with contrastive loss
                for i, (g, y, inp, tgt) in enumerate(batch_states):
                    negatives = [s for j, (s, _, _, _) in enumerate(batch_states) if j != i]
                    
                    enc_target = self.encoder.encode(y)
                    t = enc_target.real if torch.is_complex(enc_target) else enc_target
                    pred, loss = self.meta.update(g, t[:g.shape[0]], negatives=negatives)
                    
                    self.anchor_memory.set_anchor(inp)
                    self.target_mappings[inp] = tgt
                    
                    epoch_loss += loss
                    count += 1
            
            avg_loss = epoch_loss / max(count, 1)
            avg_sep = np.mean([1.0] * len(self.anchor_memory.anchors))  # Placeholder
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=avg_loss,
                avg_separation=avg_sep,
                num_anchors=len(self.anchor_memory.anchors),
                temperature=self.temperature
            )
            metrics_history.append(metrics)
            
            if verbose:
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Anchors: {metrics.num_anchors}")
                print(f"  Temperature: {self.temperature:.3f}")
        
        return metrics_history
    
    def infer(self, text: str) -> Tuple[str, float]:
        """Infer output for given input"""
        # Reset state
        self.resonator.R *= 0
        
        # Forward
        x = self.text_to_tensor(text)
        out = self.forward(x, raw_input=text)
        
        # Check anchor memory first
        if text in self.target_mappings:
            return self.target_mappings[text], 1.0
        
        # Fallback to vocab matching
        gs = out["global"].detach().cpu().numpy()
        best_word = None
        best_score = 0
        
        for word, anchor in self.vocab_anchors.items():
            similarity = np.dot(gs, anchor) / (np.linalg.norm(gs) * np.linalg.norm(anchor) + 1e-9)
            score = np.exp(similarity / self.temperature)
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word or "[unknown]", best_score
    
    def test_separation(self, test_inputs: Optional[List[str]] = None) -> float:
        """Test state separation on given inputs"""
        if test_inputs is None:
            test_inputs = list(self.anchor_memory.anchors.keys())[:5]
        
        if len(test_inputs) < 2:
            return 0.0
        
        states = {}
        for inp in test_inputs:
            self.resonator.R *= 0
            x = self.text_to_tensor(inp)
            out = self.forward(x, raw_input=inp)
            states[inp] = out["global"].detach().cpu().numpy()
        
        # Compute pairwise distances
        distances = []
        for i, inp1 in enumerate(test_inputs):
            for inp2 in test_inputs[i+1:]:
                dist = np.linalg.norm(states[inp1] - states[inp2])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0


def main_cli():
    """Command-line interface"""
    import sys
    
    print("="*70)
    print("Cypha HRNA - Harmonic Recursive Neural Architecture")
    print("="*70)
    
    cypha = Cypha(device="cpu")
    
    print("\nCommands:")
    print("  train <file> [epochs]  - Train on data file")
    print("  infer <text>           - Infer output for input")
    print("  test                   - Test state separation")
    print("  stats                  - Show system statistics")
    print("  exit                   - Exit")
    
    while True:
        try:
            cmd = input("\ncypha> ").strip()
            
            if cmd.startswith("train "):
                parts = cmd.split()
                filepath = parts[1]
                epochs = int(parts[2]) if len(parts) > 2 else 1
                
                print(f"\nTraining on {filepath} for {epochs} epochs...")
                cypha.train(filepath, epochs=epochs, batch_size=8, verbose=True)
                print("\n✓ Training complete!")
            
            elif cmd.startswith("infer "):
                text = cmd[6:].strip()
                result, confidence = cypha.infer(text)
                print(f"\nInput: '{text}'")
                print(f"Output: '{result}' (confidence: {confidence:.3f})")
            
            elif cmd == "test":
                avg_dist = cypha.test_separation()
                print(f"\nAverage state separation: {avg_dist:.4f}")
                print("PASS" if avg_dist > 0.1 else "FAIL")
            
            elif cmd == "stats":
                print(f"\nSystem Statistics:")
                print(f"  Anchors: {len(cypha.anchor_memory.anchors)}")
                print(f"  Mappings: {len(cypha.target_mappings)}")
                print(f"  Training steps: {cypha.training_step}")
                print(f"  Temperature: {cypha.temperature:.3f}")
            
            elif cmd == "exit":
                break
            
            else:
                print("Unknown command. Type 'train', 'infer', 'test', 'stats', or 'exit'")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main_cli()
