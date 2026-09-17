import torch

class CyphaTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history = []
    def train_batch(self, x, y):
        self.model.train()
        out = self.model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_history.append(loss.item())
        return loss.item()
    def eval(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    def get_loss_history(self):
        return self.loss_history
