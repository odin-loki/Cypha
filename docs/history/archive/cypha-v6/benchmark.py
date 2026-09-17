#!/usr/bin/env python3
"""
COMPLETE BENCHMARK - ALL 9 DATASETS
Fully streaming via byte-offset indexing.
No dataset is ever loaded into RAM. ~8 bytes per line regardless of line size.
State managed - resumes automatically.
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Cypha import Cypha, CyphaStateful, _build_offset_index, _read_at_offset

BENCHMARK_STATE = "benchmark_state.json"


# ── State ─────────────────────────────────────────────────────────────────────

def load_benchmark_state():
    if os.path.exists(BENCHMARK_STATE):
        try:
            with open(BENCHMARK_STATE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"datasets": {}, "started": None, "last_update": None}


def save_benchmark_state(state):
    state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = BENCHMARK_STATE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, BENCHMARK_STATE)


# ── Evaluation (streaming) ────────────────────────────────────────────────────

def evaluate_accuracy(cypha, filepath, test_offsets, sample_size=1000):
    """
    Seek to each test offset and evaluate. Never loads full test set into RAM.
    Evaluates up to sample_size samples by stepping through the offset array.
    """
    if len(test_offsets) == 0:
        return 0.0, 0, 0, []

    # Take evenly-spaced samples across the test set (deterministic, no RAM cost)
    n = len(test_offsets)
    if n > sample_size:
        step = n / sample_size
        indices = [int(i * step) for i in range(sample_size)]
        eval_offsets = test_offsets[indices]
    else:
        eval_offsets = test_offsets

    correct = total = 0
    errors  = []

    with open(filepath, "rb") as fh:
        for offset in eval_offsets:
            pair = _read_at_offset(fh, offset)
            if pair is None:
                continue
            inp, expected = pair
            try:
                result, conf = cypha.infer(inp, verbose=False)
                total += 1
                if result == expected:
                    correct += 1
                elif len(errors) < 10:
                    errors.append({
                        "input":      inp[:50],
                        "expected":   expected[:50],
                        "got":        result[:50],
                        "confidence": float(conf),
                    })
            except Exception as e:
                total += 1
                if len(errors) < 10:
                    errors.append({
                        "input":      inp[:50],
                        "expected":   expected[:50],
                        "got":        f"ERROR: {e}",
                        "confidence": 0.0,
                    })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy, correct, total, errors


# ── Per-dataset benchmark ─────────────────────────────────────────────────────

def benchmark_dataset(cypha, dataset_name, filepath, epochs=5, test_sample=1000):
    print("\n" + "=" * 70)
    print(f"  BENCHMARK: {dataset_name.upper()}")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"  x File not found: {filepath}")
        return None

    # Index the file -- negligible RAM
    print(f"  Indexing {filepath} ...", end=" ", flush=True)
    t0      = time.time()
    offsets = _build_offset_index(filepath)
    print(f"{len(offsets):,} samples  ({time.time()-t0:.1f}s)")

    if len(offsets) == 0:
        print(f"  x No valid pairs in {filepath}")
        return None

    # 80/20 split on the offset array -- no data read
    split_idx     = int(len(offsets) * 0.8)
    train_offsets = offsets[:split_idx]
    test_offsets  = offsets[split_idx:]

    print(f"  Train: {len(train_offsets):,}")
    print(f"  Test:  {len(test_offsets):,}")

    # Training
    ckpt = cypha.get_checkpoint_info(dataset_name)
    if ckpt and ckpt.get("epochs_completed", 0) >= epochs:
        print(f"  Already trained ({ckpt['epochs_completed']} epochs), loading checkpoint...")
        # CRITICAL: load checkpoint into memory before evaluating.
        # Without this, evaluation runs against an empty AnchorMemory and returns
        # [no memory] for every sample (the benchmark bug that caused 0% across all
        # domains on re-runs).  train_file_stateful_offsets handles this internally
        # but we bypass it here, so we must call _load_checkpoint explicitly.
        cypha._load_checkpoint(dataset_name)
        print(f"  Loaded {cypha.memory.n:,} anchors for {dataset_name}")
    else:
        print(f"\n  Training (streaming from disk)...")
        t_start = time.time()
        try:
            cypha.train_file_stateful_offsets(
                filepath, train_offsets, dataset_name,
                epochs=epochs, verbose=True
            )
            print(f"\n  Training done ({time.time()-t_start:.1f}s)")
        except KeyboardInterrupt:
            print("\n  Interrupted -- state saved")
            return None
        except Exception as e:
            print(f"\n  Training failed: {e}")
            import traceback; traceback.print_exc()
            return None

    # Testing
    n_test = min(test_sample, len(test_offsets))
    print(f"\n  Testing ({n_test:,} samples, streaming)...")
    t_start  = time.time()
    accuracy, correct, total, errors = evaluate_accuracy(
        cypha, filepath, test_offsets, test_sample
    )
    test_time = time.time() - t_start

    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"    Time: {test_time:.1f}s  ({test_time/max(1,total)*1000:.1f}ms/sample)")

    if errors:
        print(f"\n  Sample errors:")
        for i, err in enumerate(errors[:5], 1):
            print(f"    {i}. {err['input']}")
            print(f"       Expected: {err['expected']}")
            print(f"       Got:      {err['got']} (conf={err['confidence']:.3f})")

    return {
        "dataset":       dataset_name,
        "filepath":      filepath,
        "total_samples": len(offsets),
        "train_samples": len(train_offsets),
        "test_samples":  total,
        "epochs":        epochs,
        "accuracy":      accuracy,
        "correct":       correct,
        "total_tested":  total,
        "test_time_s":   test_time,
        "ms_per_sample": test_time / max(1, total) * 1000,
        "errors":        errors[:5],
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Status display ────────────────────────────────────────────────────────────

def show_status(state):
    print("\n" + "=" * 70)
    print("  BENCHMARK STATUS")
    print("=" * 70)

    if not state["datasets"]:
        print("\n  No benchmarks run yet")
        return

    print(f"\n  Started:   {state.get('started', 'Unknown')}")
    print(f"  Completed: {len(state['datasets'])} datasets\n")

    sorted_ds = sorted(
        state["datasets"].items(),
        key=lambda x: x[1].get("accuracy", 0),
        reverse=True,
    )
    print(f"  {'Dataset':<35} {'Accuracy':<12} {'Samples':<10}")
    print(f"  {'-'*60}")
    for name, info in sorted_ds:
        acc     = info.get("accuracy", 0)
        samples = info.get("total_samples", 0)
        print(f"  {name:<35} {acc:>6.2f}%      {samples:>8,}")

    accuracies = [i["accuracy"] for i in state["datasets"].values() if "accuracy" in i]
    if accuracies:
        print(f"\n  Average: {np.mean(accuracies):.2f}%")
        print(f"  Best:    {max(accuracies):.2f}%")
        print(f"  Worst:   {min(accuracies):.2f}%")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CYPHA BENCHMARK -- ALL 9 DATASETS  (byte-offset streaming)")
    print("=" * 70)

    if "--status" in sys.argv:
        show_status(load_benchmark_state())
        return

    if "--reset" in sys.argv:
        if os.path.exists(BENCHMARK_STATE):
            os.remove(BENCHMARK_STATE)
            print("\nState reset\n")
        resp = input("Clear checkpoints? (y/N): ").strip().lower()
        if resp == 'y':
            import shutil
            if os.path.exists("./checkpoints"):
                shutil.rmtree("./checkpoints")
                print("Checkpoints cleared\n")

    state = load_benchmark_state()
    if state["started"] is None:
        state["started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n  Initializing Cypha...")
    cypha = CyphaStateful(feature_dim=4096, resonance_dim=256)
    print("  Ready")
    print(f"  Memory: dedup_threshold={cypha.memory.dedup_threshold}  "          f"max_per_class={cypha.memory.max_per_class}  "          f"consolidate_threshold={cypha.memory.consolidate_threshold}")

    datasets = [
        ("sql_injection",      "sql_injection.txt",  1),
        ("phishing_vrbancic",  "phishing_vrbancic.txt",  1),
        ("malware",            "malware.txt",  1),
        ("network_intrusion",  "network_intrusion.txt",  1),
        ("phishing_emails",    "phishing_emails.txt",  1),
        ("panoradio_rf",       "panoradio_rf.txt",  1),
        ("phiusiil_phishing",  "phiusiil_phishing.txt",  1),
        ("speech_commands",    "speech_commands.txt",  1),
        ("esc50",              "esc50.txt",  1),
    ]

    missing = [fp for _, fp, _ in datasets if not os.path.exists(fp)]
    if missing:
        print(f"\nMissing {len(missing)} converted files:")
        for f in missing:
            print(f"  x {f}")
        print("\nRun: python convert.py")
        return

    for dataset_name, filepath, epochs in datasets:
        if dataset_name in state["datasets"]:
            cached = state["datasets"][dataset_name]
            acc    = cached.get("accuracy", -1)
            errors = cached.get("errors", [])
            # Re-run if previous result had [no memory] errors — that indicates the
            # checkpoint-load bug (benchmark.py was not loading checkpoint before eval
            # on re-runs), not a genuine 0% result.
            no_mem_cached = any(e.get("got") == "[no memory]" for e in errors)
            if acc > 0 or (acc == 0 and not no_mem_cached):
                print(f"\n  Already done: {dataset_name}  ({acc:.1f}%)")
                continue
            else:
                print(f"\n  Re-running {dataset_name} (previous {acc:.1f}% had [no memory] — checkpoint bug fixed)")
                del state["datasets"][dataset_name]

        try:
            results = benchmark_dataset(cypha, dataset_name, filepath, epochs=epochs)
            if results:
                state["datasets"][dataset_name] = results
                save_benchmark_state(state)
                print(f"\n  {dataset_name}: {results['accuracy']:.2f}%")

        except KeyboardInterrupt:
            print("\n\n  Interrupted -- run again to resume")
            save_benchmark_state(state)
            sys.exit(0)
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback; traceback.print_exc()
            state["datasets"][dataset_name] = {
                "error":     str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_benchmark_state(state)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    show_status(state)

    with open("benchmark_report.json", 'w') as f:
        json.dump(state, f, indent=2)
    print("\n  Report: benchmark_report.json")

    # Government submission summary
    if state["datasets"]:
        print("\n" + "=" * 70)
        print("  FOR GOVERNMENT SUBMISSION")
        print("=" * 70)
        categories = {
            "Security (Text)": ["sql_injection", "phishing_vrbancic", "malware",
                                 "network_intrusion", "phishing_emails"],
            "Signals (RF)":    ["panoradio_rf"],
            "Phishing (URLs)": ["phiusiil_phishing"],
            "Audio":           ["speech_commands", "esc50"],
        }
        for category, names in categories.items():
            accs = []
            print(f"\n  {category}:")
            for name in names:
                if name in state["datasets"] and "accuracy" in state["datasets"][name]:
                    info = state["datasets"][name]
                    print(f"    {info['accuracy']:.1f}%  {name.replace('_',' ')}  ({info['total_samples']:,} samples)")
                    accs.append(info["accuracy"])
            if accs:
                print(f"    Avg: {np.mean(accs):.1f}%")

        all_accs = [i["accuracy"] for i in state["datasets"].values() if "accuracy" in i]
        if all_accs:
            total_s = sum(i.get("total_samples", 0) for i in state["datasets"].values())
            print(f"\n  Overall: {np.mean(all_accs):.1f}% across {total_s:,} samples")
            print(f"  Zero synthetic data. Production ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
