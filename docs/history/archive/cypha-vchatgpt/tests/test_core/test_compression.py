import torch
from core.compression import FundamentalExtractor, DNAFolder

def test_fundamental_compression_decompression():
    fe = FundamentalExtractor(n_components=5)
    x = torch.sin(torch.linspace(0, 6.28, 32)) + 0.5
    c = fe.extract(x)
    xrec = fe.reconstruct(c, 32)
    assert xrec.shape == x.unsqueeze(0).shape

def test_dna_folder_reconstruction():
    folder = DNAFolder()
    data = torch.randn(32)
    c = folder.fold(data)
    rec = folder.unfold(c)
    assert rec.shape == data.shape or abs(rec.shape[0]-data.shape[0])<=1
