import torch
import math

class AlternativeFastOperations:
    def __init__(self, device='cpu'):
        self.device = device
    def matvec(self, M, v):
        return torch.matmul(M, v)
    def fft_convolve(self, x, k):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(k)).real
    def fast_dist(self, X):
        XX = (X ** 2).sum(1).view(-1, 1)
        return ((XX + XX.t() - 2 * X @ X.t()).clamp(min=0)).sqrt()

class NaturalMathematicalShortcuts:
    def __init__(self, device='cpu'):
        self.device = device
    def harmonics(self, base, max_n=7):
        base = torch.tensor(base, device=self.device)
        return {n: base * n for n in range(1, max_n+1)}
    def fast_fibonacci(self, n):
        a = torch.tensor([[1, 1], [1, 0]], dtype=torch.float)
        def mpow(a, n):
            res = torch.eye(2)
            while n:
                if n & 1: res = res @ a
                a = a @ a; n //= 2
            return res
        if n <= 1: return n
        return int(mpow(a, n - 1)[0, 0].item())

class StrategicStochasticNoise:
    def __init__(self, device='cpu'):
        self.device = device
    def add_noise(self, x, scale=0.1):
        return x + torch.randn_like(x) * scale
    def stochastic_resonance(self, s, thresh):
        std = s.std()
        delta = thresh - s.mean()
        if 0 < delta < 3 * std:
            sig = s + delta * torch.randn_like(s)
            return (sig > thresh).float() * s.abs().max()
        else:
            return (s > thresh).float() * s.abs().max()

class PrecisionControl:
    def __init__(self, device='cpu'):
        self.device = device
        self.levels = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64}
        self.cp = {}
        self.err = {'up': 1e-3, 'down': 1e-5}
    def set(self, cid, level):
        self.cp[cid] = self.levels[level]
    def adapt(self, cid, error, t):
        cur = self.cp.get(cid, torch.float32)
        if error > self.err['up']:
            return torch.float64 if cur == torch.float32 else torch.float32
        if error < self.err['down']:
            return torch.float16 if cur == torch.float32 else torch.float32
        return cur
    def convert(self, t, level):
        return t.to(self.levels[level])

def fused_op(a, b):
    c = a @ b
    d = torch.linalg.norm(c)
    return c, d, (a.mean() + b.mean()) / a.shape[-1]
