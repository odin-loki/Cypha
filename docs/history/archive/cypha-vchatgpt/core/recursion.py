import torch
from typing import Optional, Dict, List

class HorizontalRecursion:
    def __init__(self, alpha_h: float = 0.3, beta_e: float = 0.5, device="cpu"):
        self.alpha_h = alpha_h
        self.beta_e = beta_e
        self.device = device
        self.history: Dict[str, List[torch.Tensor]] = {}
    def update(self, component_id: str, state: torch.Tensor, inputs: Dict[str, torch.Tensor], events: Optional[List]=None):
        h = self.history.setdefault(component_id, [])
        h.append(state.clone())
        if len(h) > 10: h.pop(0)
        new_state = state * 0.9
        for v in inputs.values():
            v = v.to(self.device)
            nv = v if v.shape == state.shape else torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0), size=state.shape, mode='linear').squeeze(0).squeeze(0)
            new_state += 0.1 * nv
        rf = 1.0 + self.alpha_h
        ef = 1.0 + self.beta_e * sum(getattr(e, "priority", 1.0) for e in (events or []))
        return new_state * rf * ef

class VerticalRecursion:
    def __init__(self, alpha_v: float = 0.4, beta_e: float = 0.5, device="cpu"):
        self.alpha_v = alpha_v
        self.beta_e = beta_e
        self.device = device
    def update(self, idx: int, state: torch.Tensor, lower: Optional[torch.Tensor]=None, higher: Optional[torch.Tensor]=None, events: Optional[List]=None):
        ns = state * 0.8
        if lower is not None:
            ups = torch.nn.functional.interpolate(lower.unsqueeze(0).unsqueeze(0), size=state.shape, mode='linear').squeeze(0).squeeze(0)
            ns += 0.15 * ups
        if higher is not None:
            downs = torch.nn.functional.interpolate(higher.unsqueeze(0).unsqueeze(0), size=state.shape, mode='linear').squeeze(0).squeeze(0)
            ns += 0.15 * downs
        rf = 1.0 + self.alpha_v
        ef = 1.0 + self.beta_e * sum(getattr(e, "priority", 1.0) for e in (events or []))
        return ns * rf * ef

class TemporalRecursion:
    def __init__(self, alpha_t: float = 0.3, beta_e: float = 0.4, horizon: int = 3, device="cpu"):
        self.alpha_t = alpha_t
        self.beta_e = beta_e
        self.horizon = horizon
        self.device = device
        self.state_hist: Dict[str, List[torch.Tensor]] = {}
    def update(self, component_id: str, state: torch.Tensor, events: Optional[List]=None):
        sh = self.state_hist.setdefault(component_id, [])
        prev = sh[-1] if len(sh) > 0 else state
        sh.append(state.clone())
        if len(sh) > 10: sh.pop(0)
        new_state = state + 0.1 * (state - prev)
        rf = 1.0 + self.alpha_t
        ef = 1.0 + self.beta_e * sum(getattr(e, "priority", 1.0) for e in (events or []))
        return new_state * rf * ef
