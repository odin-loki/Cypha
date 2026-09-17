#!/usr/bin/env python3
"""
Cypha HRNA Quick Demonstration
Runs all verification steps in sequence
"""

import os
import sys

def run_step(name, command, skip_input=False):
    print("\n" + "="*70)
    print(f"  {name}")
    print("="*70)
    
    if not skip_input:
        input("\nPress Enter to continue...")
    
    os.system(command)

def main():
    print("="*70)
    print("  CYPHA HRNA - QUICK DEMONSTRATION")
    print("="*70)
    print("\nThis will run three demonstrations:")
    print("  1. Basic System Test (~30s)")
    print("  2. Thinking Dynamics Verification (~30s)")
    print("  3. Defense Benchmarks (~3-5min)")
    print("\nTotal time: ~5-7 minutes")
    
    choice = input("\nRun all three? (y/n): ").lower()
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Basic test
    run_step(
        "STEP 1: BASIC SYSTEM TEST",
        "python Cypha.py",
        skip_input=True
    )
    
    # Step 2: Thinking verification
    run_step(
        "STEP 2: THINKING DYNAMICS VERIFICATION",
        "python verify_thinking.py"
    )
    
    # Step 3: Benchmarks
    print("\n" + "="*70)
    print("  STEP 3: DEFENSE BENCHMARKS")
    print("="*70)
    print("\nThis will take ~3-5 minutes.")
    print("It will:")
    print("  - Generate 8 synthetic defense datasets")
    print("  - Train Cypha ONCE on all data (3 epochs)")
    print("  - Train Random Forest ONCE on all data")
    print("  - Test both methods on all 8 tasks")
    print("  - Generate benchmark_results.md report")
    
    choice = input("\nContinue? (y/n): ").lower()
    if choice == 'y':
        os.system("python benchmark_suite.py")
    
    print("\n" + "="*70)
    print("  DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - benchmark_results.md (performance report)")
    print("  - 8 dataset files (*_data.txt)")
    print("\nNext steps:")
    print("  1. Review benchmark_results.md")
    print("  2. Test on real defense data")
    print("  3. Prepare government presentation")

if __name__ == "__main__":
    main()
