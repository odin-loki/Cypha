import torch
from core.encoder import UniversalEncoder, PrecisionPreservation

def test_universal_encode_decode_roundtrip():
    enc = UniversalEncoder(16, 8)
    data = torch.arange(16).float()
    code = enc.encode(data.unsqueeze(0))
    assert code.shape[-1] == 8

def test_precision_preservation():
    pp = PrecisionPreservation()
    x = torch.linspace(0.1, 1000, 20)
    mant, exp = pp.preserve(x)
    xr = pp.reconstruct(mant, exp)
    assert torch.allclose(x, xr, rtol=1e-5, atol=1e-5)
