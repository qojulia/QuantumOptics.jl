#!/usr/bin/env python3
"""Run all comparative benchmarks across frameworks.

Usage:
    python3 runall.py                    # Run all frameworks
    python3 runall.py --julia-only       # Run only Julia benchmarks
    python3 runall.py --qutip-only       # Run only QuTiP benchmarks
    python3 runall.py --quantumtoolbox-only  # Run only QuantumToolbox.jl benchmarks
"""

import os
import subprocess
import sys

juliabenchmarks_qo = "benchmarks-QuantumOptics.jl"
juliabenchmarks_qt = "benchmarks-QuantumToolbox.jl"
pythonbenchmarks = "benchmarks-QuTiP"


def run_julia_benchmarks(benchmark_dir, label):
    print(f"\n{'='*60}")
    print(f"Running {label} benchmarks")
    print(f"{'='*60}\n")
    os.chdir(benchmark_dir)
    filenames = sorted(os.listdir("."))
    for name in filenames:
        if "benchmarkutils" in name or not name.endswith(".jl"):
            continue
        print(f"\n--- {name} ---")
        result = subprocess.run(["julia", "--project=../../..", name], capture_output=False)
        if result.returncode != 0:
            print(f"Warning: {name} failed with return code {result.returncode}")
    os.chdir("..")


def run_python_benchmarks(benchmark_dir, label):
    print(f"\n{'='*60}")
    print(f"Running {label} benchmarks")
    print(f"{'='*60}\n")
    os.chdir(benchmark_dir)
    filenames = sorted(os.listdir("."))
    for name in filenames:
        if "benchmarkutils" in name or not name.endswith(".py") or name == "__init__.py":
            continue
        print(f"\n--- {name} ---")
        result = subprocess.run(["python3", name], capture_output=False)
        if result.returncode != 0:
            print(f"Warning: {name} failed with return code {result.returncode}")
    os.chdir("..")


def main():
    args = set(sys.argv[1:])
    run_all = len(args) == 0

    # Collect hardware specs
    try:
        subprocess.run(["python3", "hardware_specs.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not collect hardware specs")

    if run_all or "--julia-only" in args:
        run_julia_benchmarks(juliabenchmarks_qo, "QuantumOptics.jl")

    if run_all or "--qutip-only" in args:
        run_python_benchmarks(pythonbenchmarks, "QuTiP")

    if run_all or "--quantumtoolbox-only" in args:
        run_julia_benchmarks(juliabenchmarks_qt, "QuantumToolbox.jl")

    print(f"\n{'='*60}")
    print("All benchmarks complete. Results saved in results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
