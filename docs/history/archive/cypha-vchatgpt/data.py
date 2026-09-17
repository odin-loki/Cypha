import sys
sys.path.insert(0, '/mnt/c/Users/odinl/Downloads/Cypha')

from cypha import CyphaHRNA
import numpy as np

model = CyphaHRNA(device="cpu")

# Train
print("Training...")
model.train_on_pairs("data.txt", max_lines=1000, epochs=1, batch_size=8)

# Test with TRAINED inputs
test_inputs = list(model.forced_memory.anchors.keys())[:5]
print(f"\nTesting with trained inputs: {test_inputs}\n")

states = {}
for inp in test_inputs:
    model.resonator.R *= 0
    model.assembly.A *= 0
    model.module.M *= 0
    model.global_level.G *= 0
    out = model.forward(model.text_to_tensor(inp), raw_input=inp)
    states[inp] = out["global"].detach().cpu().numpy()

print("Pairwise State Distances:")
print("-" * 70)
for i, inp1 in enumerate(test_inputs):
    for inp2 in test_inputs[i+1:]:
        dist = np.linalg.norm(states[inp1] - states[inp2])
        sim = np.dot(states[inp1], states[inp2]) / (
            np.linalg.norm(states[inp1]) * np.linalg.norm(states[inp2]) + 1e-9
        )
        print(f"{inp1[:30]:30} vs {inp2[:30]:30}: dist={dist:.4f}, sim={sim:.4f}")

all_dists = [
    np.linalg.norm(states[inp1] - states[inp2])
    for i, inp1 in enumerate(test_inputs)
    for inp2 in test_inputs[i+1:]
]
avg_dist = np.mean(all_dists)
print(f"\nAverage distance: {avg_dist:.4f}")
print("PASS" if avg_dist > 0.1 else "FAIL - States still too similar")
