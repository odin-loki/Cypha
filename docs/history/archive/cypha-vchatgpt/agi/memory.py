import torch
from core.compression import DNAFolder

class CyphaMemory:
    def __init__(self):
        self.episodic = []
        self.semantic = {}
        self.hl_folder = DNAFolder(device="cpu")
    def add_event(self, event):
        self.episodic.append(event)
        if len(self.episodic) > 1000:
            self.episodic.pop(0)
    def add_fact(self, key, value):
        self.semantic[key] = value
    def compress_longterm(self, data: torch.Tensor):
        return self.hl_folder.fold(data)
    def decompress_longterm(self, c):
        return self.hl_folder.unfold(c)

