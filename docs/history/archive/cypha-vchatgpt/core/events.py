import torch
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

class EventType(Enum):
    PATTERN = auto()
    SURPRISE = auto()
    RESONANCE = auto()
    EXTERNAL = auto()
    FEEDBACK = auto()
    META = auto()

@dataclass
class Event:
    type: EventType
    time: float
    data: Dict[str, Any]
    source: str
    target: Optional[str] = None
    priority: float = 1.0
    id: Optional[str] = None
    def __post_init__(self):
        if self.id is None:
            self.id = f"{self.source}-{self.type.name}-{self.time}"

class EventQueue:
    def __init__(self):
        self.events: List[Event] = []
    def add(self, event: Event):
        self.events.append(event)
    def pop(self) -> Optional[Event]:
        if not self.events:
            return None
        self.events.sort(key=lambda e: -e.priority)
        return self.events.pop(0)
    def due_by_time(self, t: float) -> List[Event]:
        es = [e for e in self.events if e.time <= t]
        self.events = [e for e in self.events if e.time > t]
        return es
    def __len__(self):
        return len(self.events)

class LogSchedule:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.scheduled = []
    def schedule(self, event: Event) -> float:
        t_next = event.time * (1 + self.alpha * event.priority) ** -1
        self.scheduled.append((t_next, event))
        self.scheduled.sort(key=lambda x: x[0])
        return t_next
    def get_due(self, now: float):
        due = []
        while self.scheduled and self.scheduled[0][0] <= now:
            _, e = self.scheduled.pop(0)
            due.append(e)
        return due

class EventGenerator:
    def __init__(self, pattern_thresh=0.7, surprise_thresh=0.3, resonance_thresh=0.5, device="cpu"):
        self.pattern_th = pattern_thresh
        self.surprise_th = surprise_thresh
        self.resonance_th = resonance_thresh
        self.device = device
        self.history = []

    def adaptive_competitive_threshold(self, scores):
        if len(scores) < 2:
            return scores.mean() if len(scores) > 0 else 0.0
        mu = scores.mean()
        sigma = scores.std()
        return mu + 2 * sigma  # Only the very strongest trigger

    def pattern_event(self, state: torch.Tensor, patterns: List[torch.Tensor], time: float, source="pattern"):
        es = []
        scores = torch.stack([(state*p).sum()/(state.norm()*p.norm()+1e-8) for p in patterns])
        theta = self.adaptive_competitive_threshold(scores)
        for i, score in enumerate(scores):
            if score > theta:
                es.append(Event(type=EventType.PATTERN, time=time, data={"pattern_id":i,"resonance":score.item()}, source=source, priority=score.item()))
        return es

    def resonance_event(self, state: torch.Tensor, R_enhanced: torch.Tensor, time: float, source="resonance"):
        es = []
        theta_res = R_enhanced.max() * 0.8
        for i, r in enumerate(R_enhanced):
            if r > theta_res:
                es.append(Event(type=EventType.RESONANCE, time=time, data={"index": int(i), "resonance": r.item()}, source=source, priority=r.item()))
        return es

    def surprise_event(self, state: torch.Tensor, pred: torch.Tensor, time: float, source="surprise"):
        error = (state - pred).abs().mean().item()
        self.history.append(state.clone())
        es = []
        if error > self.surprise_th:
            es.append(Event(type=EventType.SURPRISE, time=time, data={"error": error}, source=source, priority=error))
        return es

    def gen_external(self, inp: torch.Tensor, time: float, source="external"):
        return Event(type=EventType.EXTERNAL, time=time, data={"input": inp}, source=source, priority=1.0)
