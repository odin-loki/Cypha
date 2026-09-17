import torch

def orthogonalize(c, others):
    for o in others:
        proj = torch.dot(c, o) / (torch.dot(o, o) + 1e-8) * o
        c = c - proj
    return c

class FundamentalExtractor:
    def __init__(self, n_components=50, threshold=0.01, device="cpu"):
        self.n_components = n_components
        self.threshold = threshold
        self.device = device
        self.prev_compressions = []  # For orthogonalization

    def extract(self, x: torch.Tensor):
        fft = torch.fft.fft(x)
        amplitudes = torch.abs(fft)
        phases = torch.angle(fft)
        idxs = torch.argsort(amplitudes, descending=True)[:self.n_components]
        freqs = idxs.float() / x.shape[0]
        sel_amp = amplitudes[idxs]
        sel_phase = phases[idxs]
        # Compose compressed vector (real/imag for full identifiability)
        cvec = torch.cat([sel_amp, sel_phase], dim=0)
        # Orthogonalize vs. all previous
        if self.prev_compressions:
            cvec = orthogonalize(cvec, self.prev_compressions)
        # Save (Ring buffer, size 32)
        self.prev_compressions.append(cvec)
        if len(self.prev_compressions) > 32:
            self.prev_compressions = self.prev_compressions[-32:]
        return {"frequencies": freqs, "amplitudes": sel_amp, "phases": sel_phase, "comp_vec": cvec}

    def reconstruct(self, c, size):
        recon = torch.zeros(size, dtype=torch.cfloat, device=self.device)
        for i, f in enumerate(c["frequencies"]):
            idx = int(f * size)
            phase = c["phases"][i]
            amp = c["amplitudes"][i]
            recon[idx] = amp * torch.exp(1j * phase)
        result = torch.fft.ifft(recon).real
        return result

class SymmetryEncoder:
    def __init__(self, device="cpu"):
        self.device = device
    def encode(self, comp):
        return comp  # Placeholder

class CrystalLatticeMapper:
    def __init__(self, lattice_size=16, n_types=8, device="cpu"):
        self.lattice_size = lattice_size
        self.n_types = n_types
        self.device = device
        self.ref = self._get_lattice()
    def _get_lattice(self):
        x = torch.linspace(0, 2*torch.pi, self.lattice_size, device=self.device)
        return torch.sin(x)
    def map(self, signal):
        n = signal.shape[-1]
        if self.ref.shape[0] != n:
            ref_resized = torch.nn.functional.interpolate(
                self.ref.unsqueeze(0).unsqueeze(0), size=n, mode='linear', align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            ref_resized = self.ref
        diff = signal - ref_resized
        pos = torch.nonzero(diff.abs() > 0.1)
        vals = diff[pos].flatten()
        types = torch.clamp(torch.floor(vals * self.n_types / 2 + self.n_types / 2), 0, self.n_types-1).long()
        return {"lattice": ref_resized, "positions": pos, "types": types, "shape": signal.shape}
    def decompress(self, packed):
        recon = packed["lattice"].clone()
        for i, p in enumerate(packed["positions"]):
            val = (packed["types"][i].float() - (self.n_types/2)) / (self.n_types/2)
            recon[p] += val
        return recon[:packed["shape"][0]]

class DNAFolder:
    def __init__(self, levels=4, device="cpu"):
        self.levels = levels
        self.device = device
    def fold(self, x):
        orig = x.clone()
        data, meta = [orig], []
        for _ in range(self.levels):
            if data[-1].numel() <= 8: break
            if data[-1].numel() % 2: data[-1] = torch.cat([data[-1], torch.zeros(1, device=self.device)])
            pairwise = data[-1].reshape(-1,2)
            new = (pairwise[:,0]+pairwise[:,1])/2
            meta.append(pairwise)
            data.append(new)
        return {"folds": data, "meta": meta, "shape": orig.shape}
    def unfold(self, packed):
        x = packed["folds"][-1]
        for even, odd in reversed(packed["meta"]):
            y = torch.zeros(even.shape[0]+odd.shape[0], device=self.device)
            y[::2], y[1::2] = even, odd
            x = y
        return x[:packed["shape"][0]]
