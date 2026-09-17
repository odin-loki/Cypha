import sys
sys.path.insert(0, '/mnt/c/Users/odinl/Downloads/Cypha')

from cypha import CyphaHRNA
import numpy as np
import torch

model = CyphaHRNA(device="cpu")

test_inputs = ["12+165", "69+50"]

print("=== LAYER-BY-LAYER COLLAPSE DIAGNOSIS ===\n")

for inp in test_inputs:
    print(f"\nInput: '{inp}'")
    x = model.text_to_tensor(inp)
    
    # Reset state
    model.resonator.R *= 0
    model.assembly.A *= 0
    model.module.M *= 0
    model.global_level.G *= 0
    
    # Step through each layer
    enc = model.encoder.encode(x)
    print(f"  Encoder output (first 5): {enc.real[:5].detach().numpy()}")
    print(f"  Encoder norm: {torch.norm(enc.real).item():.6f}")
    
    model.resfield.add_event(enc.real)
    rfield = model.resfield.evolve(1)
    print(f"  ResField output (first 5): {rfield.real[:5].detach().numpy()}")
    print(f"  ResField norm: {torch.norm(rfield.real).item():.6f}")
    
    reso = model.resonator.update(external_drive=rfield.real)
    print(f"  Resonator output (first 5): {reso[:5].detach().numpy()}")
    print(f"  Resonator norm: {torch.norm(reso).item():.6f}")
    
    assem = model.assembly.update(reso)
    print(f"  Assembly output (first 5): {assem[:5].detach().numpy()}")
    print(f"  Assembly norm: {torch.norm(assem).item():.6f}")
    
    module = model.module.update(assem)
    print(f"  Module output (first 5): {module[:5].detach().numpy()}")
    print(f"  Module norm: {torch.norm(module).item():.6f}")
    
    globalv = model.global_level.update(module)
    print(f"  Global output (first 5): {globalv[:5].detach().numpy()}")
    print(f"  Global norm: {torch.norm(globalv).item():.6f}")
    
    g = model._normalize(globalv)
    print(f"  Normalized (first 5): {g[:5].detach().numpy()}")
    print(f"  Normalized norm: {torch.norm(g).item():.6f}")

print("\n=== DISTANCE COMPARISON ===")
# Now compare two inputs
states = {}
for inp in test_inputs:
    model.resonator.R *= 0
    model.assembly.A *= 0  
    model.module.M *= 0
    model.global_level.G *= 0
    model.resfield.psi = torch.randn(model.resfield.dim, dtype=torch.cfloat, device=model.device)
    model.resfield.psi = model.resfield.psi / torch.norm(model.resfield.psi)
    
    x = model.text_to_tensor(inp)
    out = model.forward(x, raw_input=inp)
    states[inp] = out["global"].detach().cpu().numpy()

dist = np.linalg.norm(states[test_inputs[0]] - states[test_inputs[1]])
sim = np.dot(states[test_inputs[0]], states[test_inputs[1]]) / (
    np.linalg.norm(states[test_inputs[0]]) * np.linalg.norm(states[test_inputs[1]])
)

print(f"\nFinal distance: {dist:.6f}")
print(f"Final similarity: {sim:.6f}")

if dist < 0.001:
    print("\n✗ COLLAPSED - States are identical")
    print("Check which layer above shows identical outputs for both inputs")
else:
    print("\n✓ WORKING - States are different")
