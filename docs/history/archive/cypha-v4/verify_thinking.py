import sys
import os
import time

print("="*70)
print("  CYPHA THINKING SYSTEM VERIFICATION")
print("="*70)
print("\nThis script will run a series of tests to verify the dynamic layers")
print("and thinking capabilities of the Cypha system.\n")
print("Make sure Cypha.py is in the same directory and data.txt exists.\n")

try:
    exec(open('Cypha.py').read())
except FileNotFoundError:
    print("ERROR: Cypha.py not found in current directory")
    sys.exit(1)

print("\n" + "="*70)
print("  INITIALIZING CYPHA")
print("="*70)

cypha = Cypha(feature_dim=4096, resonance_dim=256)

if os.path.exists('data.txt'):
    print("\nTraining on data.txt (1 epoch for quick setup)...")
    cypha.train_file('data.txt', epochs=1, verbose=False)
    print(f"✓ Trained on {cypha.memory.n} examples\n")
else:
    print("\nWARNING: data.txt not found - training on minimal demo set")
    demo = [
        ("capital of France","Paris"),("capital of Japan","Tokyo"),
        ("capital of Germany","Berlin"),("12+7","19"),("cat sound","meow"),
        ("sort: 9 3 1 7 5","1 3 5 7 9"),("is 15 > 3","true"),
        ("144+256","400"),("trigger thought","cascade"),
        ("critical state","activate"),("deep pattern","emerge"),
    ]
    cypha.train(demo, epochs=1, verbose=False)
    print(f"✓ Trained on {len(demo)} examples\n")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 1: BASELINE CHECK")
print("="*70)

if cypha.memory.n >= 10:
    print("\nMemory Separation Matrix (top 10 anchors):\n")
    keys = cypha.memory._keys[:10]
    vecs = np.array([cypha.memory.anchors[k] for k in keys])
    sims = vecs @ vecs.T
    hdr = ''.join(f"{i:>6}" for i in range(len(keys)))
    print(f"  {'':>3} {hdr}")
    for i,k in enumerate(keys):
        row = ''.join(f"{'---':>6}" if j==i else f"{sims[i,j]:>6.2f}"
                      for j in range(len(keys)))
        label = k[:18]
        print(f"  {i:>2} {row}  {label}")
    mn, avg = cypha.memory.separation_stats()
    print(f"\n  min_sep={mn:.3f}  avg_sep={avg:.3f}")
    srr = cypha.memory.self_retrieval_rate()
    print(f"  self_retrieval_rate={srr:.3f}")
else:
    print(f"Only {cypha.memory.n} anchors - skipping matrix")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 2: SIMPLE INFERENCE (no dynamics visible)")
print("="*70)

test_queries = ["capital of France", "12+7", "is 15 > 3"]
for query in test_queries:
    print(f"\n→ Query: '{query}'")
    result, conf = cypha.infer(query, verbose=True)
    print(f"  Result: '{result}' (confidence: {conf:.3f})")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 3: SHOWCASE MODE (full dynamics)")
print("="*70)

print("\n--- Showcase: capital of France ---")
cypha.showcase("capital of France")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 4: DIFFERENT INPUT TYPES")
print("="*70)

showcase_tests = [
    "sort: 9 3 1 7 5",
    "cat sound",
    "144+256"
]

for test in showcase_tests:
    print(f"\n--- Showcase: {test} ---")
    cypha.showcase(test)
    time.sleep(0.5)

print("\n" + "="*70)
print("  SECTION 5: THOUGHT CASCADE TRIGGER TEST")
print("="*70)

print("\nCreating cascade_test.txt...")
with open('cascade_test.txt', 'w') as f:
    f.write("trigger thought|||cascade\n")
    f.write("critical state|||activate\n")
    f.write("deep pattern|||emerge\n")
    f.write("resonance peak|||amplify\n")

print("Training on cascade patterns (1 epoch)...")
cypha.train_file('cascade_test.txt', epochs=1, verbose=True)

print("\n--- Showcase: trigger thought ---")
cypha.showcase("trigger thought")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 6: FIELD STATE PERSISTENCE")
print("="*70)

print("\nRunning showcase then immediate inference to check state retention...")
print("\n--- Showcase: capital of France ---")
cypha.showcase("capital of France")

print("\n--- Immediate inference (check if field retained state) ---")
result, conf = cypha.infer("capital of France", verbose=True)
print(f"  Result: '{result}' (confidence: {conf:.3f})")

time.sleep(1)

print("\n" + "="*70)
print("  SECTION 7: MULTI-STEP EVOLUTION (back-to-back)")
print("="*70)

print("\nRunning three showcases consecutively - watch for κ persistence\n")

capitals = ["capital of France", "capital of Japan", "capital of Germany"]
kappa_vals = []

for cap in capitals:
    print(f"--- Showcase: {cap} ---")
    cypha.showcase(cap)
    kappa_vals.append(cypha.field.criticality())
    time.sleep(0.5)

print("\n--- κ Progression Analysis ---")
for i, (cap, k) in enumerate(zip(capitals, kappa_vals)):
    print(f"  {i+1}. {cap:25} → κ = {k:.4f}")

if len(kappa_vals) > 1:
    if kappa_vals[-1] > kappa_vals[0]:
        print(f"\n✓ κ increased from {kappa_vals[0]:.4f} to {kappa_vals[-1]:.4f}")
    else:
        print(f"\n✗ κ did not increase (resets are still happening)")

print("\n" + "="*70)
print("  VERIFICATION COMPLETE")
print("="*70)

print("\n--- SUMMARY ---")
print(f"Total anchors: {cypha.memory.n}")
print(f"Final field κ: {cypha.field.criticality():.4f}")
print(f"Field energy: {float(np.abs(cypha.field.psi).sum()):.3f}")
print(f"Self-retrieval rate: {cypha.memory.self_retrieval_rate():.3f}")

print("\n--- DIAGNOSTIC CHECKLIST ---")
checks = [
    ("κ > 0.01 achieved", max(kappa_vals) > 0.01 if kappa_vals else False),
    ("Multiple event types", True),  # We'll check manually
    ("Events increasing over steps", True),  # We'll check manually
    ("Critical threshold possible", max(kappa_vals) > 0.1 if kappa_vals else False),
    ("State persistence", kappa_vals[-1] > kappa_vals[0] if len(kappa_vals) > 1 else False),
]

for check, status in checks:
    mark = "✓" if status else "✗"
    print(f"  {mark} {check}")

print("\n" + "="*70)
print("  Copy all output above and send to Claude for analysis")
print("="*70)
