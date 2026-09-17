import torch

class ResonanceAmplifiedFeedback:
    def __init__(self, gamma_res: float = 0.5, device="cpu"):
        self.gamma_res = gamma_res
        self.device = device
    def compute(self, state: torch.Tensor, target: torch.Tensor):
        if state.shape != target.shape:
            target = torch.nn.functional.interpolate(target.unsqueeze(0).unsqueeze(0), size=state.shape, mode='linear').squeeze(0).squeeze(0)
        diff = target - state
        resonance = (state * target).sum() / (state.norm() * target.norm() + 1e-8)
        return diff * (1.0 + self.gamma_res * resonance)

class CrossLevelFeedback:
    def __init__(self, device="cpu"):
        self.device = device
        self.cross_weights = {}
    def set_weight(self, source: str, target: str, w: float):
        self.cross_weights[(source, target)] = w
    def compute(self, source: str, src_state: torch.Tensor, target: str, tgt_state: torch.Tensor):
        w = self.cross_weights.get((source, target), 0.1)
        if src_state.shape != tgt_state.shape:
            src_state = torch.nn.functional.interpolate(src_state.unsqueeze(0).unsqueeze(0), size=tgt_state.shape, mode='linear').squeeze(0).squeeze(0)
        resonance = (src_state * tgt_state).sum() / (src_state.norm() * tgt_state.norm() + 1e-8)
        return w * resonance * (src_state - tgt_state)

class TemporalFeedback:
    def __init__(self, window=1.0, decay=2.0, device="cpu"):
        self.window = window
        self.decay = decay
        self.device = device
        self.history = []
    def add_event(self, event):
        self.history.append((event.time, event))
        now = event.time
        self.history = [(t, e) for t, e in self.history if now - t <= self.window]
    def context(self, shape):
        now = torch.tensor(0.0)
        ctx = torch.zeros(shape, device=self.device)
        for t, e in self.history:
            k = torch.exp(-self.decay * ((now - t) / self.window))
            d = e.data['input'] if 'input' in e.data and isinstance(e.data['input'], torch.Tensor) else (e.data['feedback'] if 'feedback' in e.data else None)
            if d is None:
                continue
            if d.shape != shape:
                d = torch.nn.functional.interpolate(d.unsqueeze(0).unsqueeze(0), size=shape, mode='linear').squeeze(0).squeeze(0)
            ctx = ctx + k * getattr(e,'priority',1.0) * d
        return ctx

class CriticalityEnhancedFeedback:
    def __init__(self, delta_crit: float = 1.0, kc: float = 0.5, device="cpu"):
        self.delta_crit = delta_crit
        self.kc = kc
        self.device = device
        self.crit = kc
    def set_crit(self, v: float):
        self.crit = max(0.0, min(1.0, v))
    def enhance(self, base: torch.Tensor):
        kf = 1.0 + self.delta_crit * (self.crit - self.kc) ** 2
        return base * kf
