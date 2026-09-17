import torch
import random
import time

class RecursiveEventCascades:
    def __init__(self, depth=3, branch=2, device='cpu'):
        self.depth = depth
        self.branch = branch
        self.device = device
        self.cascades = {}
    def create_cascade(self, seed_event):
        cid = f'cascade-{time.time()}-{random.randint(1,99999)}'
        self.cascades[cid] = {'seed': seed_event, 'events': [seed_event], 'depth': 0}
        return cid
    def generate_sub_events(self, cid):
        if cid not in self.cascades: return []
        cas = self.cascades[cid]
        if cas['depth'] >= self.depth: return []
        last_level = [e for e in cas['events'] if getattr(e.data, 'cascade_depth', 0)==cas['depth']]
        subs = []
        for p in last_level:
            n = random.randint(1, self.branch)
            for i in range(n):
                typ = p.type
                dat = dict(parent_id=getattr(p, 'id', None), cascade_depth=cas['depth']+1, sub_idx=i)
                sub = type(p)(type=typ, time=time.time()+0.1*(i+1), data=dat, source=p.source, target=p.target, priority=p.priority*0.9)
                subs.append(sub)
        cas['events'].extend(subs)
        cas['depth'] += 1
        return subs

class MultiScaleThought:
    def __init__(self, n_scales=4, dims=None, device='cpu'):
        self.n_scales = n_scales
        self.device = device
        self.dims = dims or [100, 30, 10, 5][:n_scales]
        self.scales = [torch.zeros(d, device=device) for d in self.dims]
    def update_scale(self, scale, dt=0.1):
        cs = self.scales[scale]
        lower = self.scales[scale-1] if scale>0 else torch.zeros_like(cs)
        upper = self.scales[scale+1] if scale<self.n_scales-1 else torch.zeros_like(cs)
        res = cs + 0.15*lower + 0.25*upper
        self.scales[scale] = torch.tanh(cs + dt*res)
        return self.scales[scale]
    def add_event_to_scale(self, scale, event):
        pass
    def distribute_event(self, event):
        pass

class SelfGeneratedEventStreams:
    def __init__(self, history=20, device='cpu'):
        self.history_len = history
        self.device = device
        self.event_history = []
        self.global_state = None
    def set_global_state(self, x):
        self.global_state = x.clone() if torch.is_tensor(x) else None
    def add_event(self, e):
        self.event_history.append(e)
        if len(self.event_history) > self.history_len:
            self.event_history.pop(0)
    def generate_event(self):
        if random.random() > 0.5 or not self.event_history:
            return None
        event = random.choice(self.event_history)
        return event

class ResonantEventChains:
    def __init__(self, maxlen=10, minres=0.3, device='cpu'):
        self.maxlen = maxlen
        self.minres = minres
        self.device = device
        self.chains = {}
    def create_chain(self, seed_event):
        cid = f'chain-{time.time()}-{random.randint(1,99999)}'
        self.chains[cid] = {'events': [seed_event], 'resonance': 1.0}
        return cid
    def add_to_chain(self, cid, event):
        if cid not in self.chains: return False
        chain = self.chains[cid]
        if len(chain['events']) >= self.maxlen: return False
        chain['events'].append(event)
        return True
