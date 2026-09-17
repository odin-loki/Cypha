import torch
import numpy as np

class RecursiveMetaLearning:
    def __init__(self, state_dim=64, device='cpu', max_recent=8):
        self.state_dim = state_dim
        self.device = device
        self.max_recent = max_recent
        self.alpha = torch.tensor(0.2, device=device)
        self.momentum = torch.tensor(0.9, device=device)
        self.sparsity = torch.tensor(0.2, device=device)
        self.meta_state = torch.zeros(state_dim, device=device)
        self.experience = {
            "states": [],
            "targets": [],
            "losses": []
        }
        self.recent_preds = []

    def update(self, state: torch.Tensor, target: torch.Tensor, negatives: list = None):
        s = state.view(-1)
        t = target.view(-1)
        if torch.is_complex(s): s = s.real
        if torch.is_complex(t): t = t.real
        
        s_len = s.shape[0]
        t_trunc = t[:s_len]
        
        positive_loss = torch.mean((s - t_trunc) ** 2)
        
        contrastive_loss = 0.0
        if negatives:
            for neg in negatives:
                if torch.is_complex(neg): neg = neg.real
                neg = neg.view(-1)[:s_len]
                similarity = torch.cosine_similarity(s, neg, dim=0)
                contrastive_loss += torch.relu(similarity - 0.2) ** 2
        
        penalty = 1.0
        if len(self.recent_preds) > 0:
            similarities = []
            for p in self.recent_preds:
                if torch.is_complex(p): p = p.real
                p = p.view(-1)[:s_len]
                sim = torch.cosine_similarity(s, p, dim=0).abs().item()
                similarities.append(sim)
            if similarities:
                penalty = float(np.exp(-3.0 * sum(similarities)))
        
        total_loss = (positive_loss + 0.8 * contrastive_loss) * penalty
        
        self.recent_preds.append(s.detach().clone())
        if len(self.recent_preds) > self.max_recent:
            self.recent_preds = self.recent_preds[-self.max_recent:]
        
        if torch.is_complex(total_loss): 
            total_loss = total_loss.real
        
        self.experience["states"].append(s.detach().clone())
        self.experience["targets"].append(t_trunc.detach().clone())
        self.experience["losses"].append(float(total_loss))
        
        if len(self.experience["states"]) > 200:
            self.experience["states"] = self.experience["states"][-200:]
            self.experience["targets"] = self.experience["targets"][-200:]
            self.experience["losses"] = self.experience["losses"][-200:]
        
        if len(self.experience["losses"]) >= 2:
            dloss = self.experience["losses"][-1] - self.experience["losses"][-2]
        else:
            dloss = 0.0
        
        if isinstance(dloss, complex): 
            dloss = dloss.real
        
        noise = torch.randn_like(self.meta_state) * 0.05
        meta_len = self.meta_state.shape[0]
        self.meta_state = 0.5 * self.meta_state + 0.5 * (t_trunc[:meta_len] - s[:meta_len]) + noise
        
        if dloss > 0: 
            self.alpha *= 0.95
        else: 
            self.alpha *= 1.05
        self.alpha = torch.clamp(self.alpha, 1e-4, 2.0)
        
        if len(self.experience["losses"]) > 1:
            if dloss * self.experience["losses"][-2] < 0:
                self.momentum *= 1.02
            else:
                self.momentum *= 0.98
        self.momentum = torch.clamp(self.momentum, 0.0, 0.999)
        
        if dloss > 0: 
            self.sparsity *= 1.01
        else: 
            self.sparsity *= 0.98
        self.sparsity = torch.clamp(self.sparsity, 0.0, 0.95)
        
        return s, float(total_loss)
