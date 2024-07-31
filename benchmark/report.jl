using PkgBenchmark


import QuantumOptics

result = benchmarkpkg(QuantumOptics)
export_markdown("results.md", result)