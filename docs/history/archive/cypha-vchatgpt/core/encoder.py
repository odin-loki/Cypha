import torch
import numpy as np

class UniversalEncoder:
    def __init__(self, input_dim: int, resonance_dim: int, device="cpu"):
        self.input_dim = input_dim
        self.resonance_dim = resonance_dim
        self.device = device
        self.amp_weights = torch.randn(input_dim, resonance_dim, device=device)
        self.phase_weights = torch.randn(input_dim, resonance_dim, device=device)
        self.basis_freqs = torch.linspace(0.5, 10.0, resonance_dim).to(device)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float().reshape(-1)
        amps = torch.mv(self.amp_weights.t(), x)
        phases = torch.mv(self.phase_weights.t(), x)
        domain = torch.arange(self.resonance_dim, device=self.device)
        basef = torch.sin(self.basis_freqs * domain / self.resonance_dim)
        encoding = amps * torch.exp(1j * phases) * basef
        return encoding
    def multimodal_encode(self, data):
        if isinstance(data, str):
            vals = [ord(c)%256/255.0 for c in data[:self.input_dim]]
            t = torch.tensor(vals, device=self.device)
            return self.encode(t)
        elif isinstance(data, (bytes, bytearray)):
            vals = [b/255.0 for b in data[:self.input_dim]]
            t = torch.tensor(vals, device=self.device)
            return self.encode(t)
        elif isinstance(data, np.ndarray):
            return self.encode(torch.from_numpy(data))
        elif torch.is_tensor(data):
            return self.encode(data)
        else:
            t = torch.zeros(self.input_dim, device=self.device)
            return self.encode(t)

class PrecisionPreservation:
    def __init__(self, device="cpu"):
        self.device = device
    def preserve(self, x):
        absx = torch.abs(x)
        sign = torch.sgn(x)
        zero = absx < 1e-10
        exponent = torch.floor(torch.log2(absx + 1e-12))
        mantissa = sign * absx / (2.0 ** exponent)
        mantissa = torch.where(zero, torch.zeros_like(mantissa), mantissa)
        exponent = torch.where(zero, torch.zeros_like(exponent), exponent)
        return mantissa, exponent
    def reconstruct(self, mant, exp):
        return mant*(2.0**exp)
    def handle_overflow(self, mant, exp, thresh=1e7):
        ops = torch.abs(mant)>thresh
        adj = torch.log2(torch.abs(mant) + 1e-12)
        mant = mant/(2.0**adj)*ops
        exp = exp+adj*ops
        return mant, exp
