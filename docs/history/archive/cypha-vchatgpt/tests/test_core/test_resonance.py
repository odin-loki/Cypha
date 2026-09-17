import torch
from core.resonance import ResonanceField, EnhancedResonance

def test_resonance_field_consistency():
    rf = ResonanceField(dim=5)
    v = torch.randn(5)
    pre = rf.R.clone()
    rf.add_event({'pattern':v}, t=1.0)
    post = rf.evolve(1)
    assert post.shape == pre.shape

def test_enhanced_resonance():
    er = EnhancedResonance()
    a = torch.randn(16)
    b = torch.randn(16)
    result = er.enhance(a, b)
    assert result.shape == torch.fft.fft(a).shape
