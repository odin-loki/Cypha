import sys
sys.path.insert(0, '/mnt/c/Users/odinl/Downloads/Cypha')

from cypha import CyphaHRNA

model = CyphaHRNA(device="cpu")

print("="*70)
print("TRAINING ON SAMPLE DATA")
print("="*70)

# Train on specific examples
training_pairs = [
    "12+165|||177",
    "cat sound|||meow",
    "capital of France|||Paris",
    "Sort: 5 2 9 1|||1 2 5 9",
    "dog sound|||bark"
]

with open("test_data.txt", "w") as f:
    for pair in training_pairs:
        f.write(pair + "\n")

model.train_on_pairs("test_data.txt", epochs=3, batch_size=4)

print("\n" + "="*70)
print("INFERENCE TEST - Does it remember what it learned?")
print("="*70)

test_cases = [
    ("12+165", "177"),
    ("cat sound", "meow"),
    ("capital of France", "Paris"),
    ("dog sound", "bark"),
]

for inp, expected in test_cases:
    x = model.text_to_tensor(inp)
    out = model.forward(x, raw_input=inp)
    
    # Get top vocab matches
    top_matches = model.compute_vocab_matches_with_temperature(out["global"])[:3]
    
    # Get explicit decode
    decoded = model.explicit_decoder(out["global"], input_str=inp)
    
    print(f"\nInput: '{inp}'")
    print(f"Expected: '{expected}'")
    print(f"Decoded: {decoded}")
    print(f"Top 3 vocab: {[(w, f'{s:.3f}') for w, s in top_matches]}")
    
    # Check if it learned the mapping
    if inp in model.target_answers:
        print(f"✓ Stored answer: {model.target_answers[inp]}")
    else:
        print("✗ No stored answer")

print("\n" + "="*70)
print("SEPARATION CHECK")
print("="*70)

import numpy as np
states = {}
for inp, _ in test_cases:
    model.resonator.R *= 0
    model.assembly.A *= 0
    model.module.M *= 0
    x = model.text_to_tensor(inp)
    out = model.forward(x, raw_input=inp)
    states[inp] = out["global"].detach().cpu().numpy()

for i, (inp1, _) in enumerate(test_cases):
    for inp2, _ in test_cases[i+1:]:
        dist = np.linalg.norm(states[inp1] - states[inp2])
        print(f"{inp1:20} vs {inp2:20}: dist={dist:.4f}")
